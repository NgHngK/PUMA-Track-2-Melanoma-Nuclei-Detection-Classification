from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ..config import PipelineConfig
from ..constants import PUBLIC_MATCH_RADIUS_PIXELS
from ..data.preprocess import load_shared_artifacts
from ..evaluation.proposal_diagnostics import binary_detection_metrics
from ..utils.checkpointing import CheckpointManager
from ..utils.history import TrainingHistory
from ..utils.runtime import (
    configure_runtime,
    dataloader_worker_init,
    make_grad_scaler,
    release_unused_memory,
    shutdown_dataloader,
)
from .data import Stage1Dataset, collate_stage1
from .decode import decode_stage1_candidates, extract_stage1_candidates
from .losses import stage1_loss
from .model import build_stage1_model
from .registry import stage1_run_directory


def _stage1_training_signature(config: PipelineConfig) -> dict[str, Any]:
    """Critical settings that must not change when resuming a Stage-1 fold."""
    payload = config.as_dict()
    return {
        "data": {
            "number_of_folds": payload["data"]["number_of_folds"],
            "seed": payload["data"]["seed"],
            "heatmap_sigma_grid_units": payload["data"]["heatmap_sigma_grid_units"],
            "stage1_output_stride": payload["data"]["stage1_output_stride"],
        },
        "stage1": payload["stage1"],
        "runtime": {
            "use_bfloat16": payload["runtime"]["use_bfloat16"],
            "allow_tf32": payload["runtime"]["allow_tf32"],
            "channels_last": payload["runtime"]["channels_last"],
            "fused_adamw": payload["runtime"]["fused_adamw"],
        },
    }


def _assert_no_case_leakage(
    artifacts: dict[str, np.ndarray], train_indices: np.ndarray, validation_indices: np.ndarray, *, fold: int
) -> None:
    """Fail closed if any patient/case appears on both sides of a fold."""
    manifest = artifacts["manifest"]
    folds = np.asarray(artifacts["folds"], dtype=np.int64)
    case_ids = np.asarray(manifest["case_id"]).astype(str)
    train_cases = set(case_ids[np.asarray(train_indices, dtype=np.int64)].tolist())
    validation_cases = set(case_ids[np.asarray(validation_indices, dtype=np.int64)].tolist())
    overlap = sorted(train_cases & validation_cases)
    if overlap:
        raise RuntimeError(
            f"DATA LEAKAGE: Stage-1 fold {fold} has {len(overlap)} case_id values in both train and validation: "
            f"{overlap[:8]}"
        )
    for case_id in np.unique(case_ids):
        roi = np.flatnonzero(case_ids == case_id)
        assigned = np.unique(folds[roi])
        if len(assigned) != 1:
            raise RuntimeError(f"DATA LEAKAGE: case_id={case_id!r} spans folds {assigned.tolist()}")
    print(
        f"Stage1 fold {fold} leakage audit: PASS | "
        f"train_cases={len(train_cases)} held_cases={len(validation_cases)} overlap=0"
    )


def _transform_points_rot90_once(points: np.ndarray, grid_size: int, stride: int) -> np.ndarray:
    """Exact CCW 90-degree transform for sparse stride-grid targets.

    Pixel transforms use x' = y, y' = N-1-x. Because target coordinates are
    stored in grid units x/stride, the complement contributes
    (stride-1)/stride to the transformed sub-grid offset.
    """
    if len(points) == 0:
        return points.copy()
    out = points.copy()
    gx = points["grid_x"].astype(np.int64)
    gy = points["grid_y"].astype(np.int64)
    ox = points["offset_x"].astype(np.float32)
    oy = points["offset_y"].astype(np.float32)
    complement = float(stride - 1) / float(stride)
    out["grid_x"] = gy.astype(out.dtype["grid_x"])
    out["grid_y"] = (int(grid_size) - 1 - gx).astype(out.dtype["grid_y"])
    out["offset_x"] = oy
    out["offset_y"] = complement - ox
    return out


def _transform_points_hflip(points: np.ndarray, grid_size: int, stride: int) -> np.ndarray:
    if len(points) == 0:
        return points.copy()
    out = points.copy()
    gx = points["grid_x"].astype(np.int64)
    ox = points["offset_x"].astype(np.float32)
    complement = float(stride - 1) / float(stride)
    out["grid_x"] = (int(grid_size) - 1 - gx).astype(out.dtype["grid_x"])
    out["offset_x"] = complement - ox
    return out


def _d4_augmentation(
    images: torch.Tensor,
    heatmaps: torch.Tensor,
    sparse_points: list[np.ndarray],
    config: PipelineConfig,
) -> tuple[torch.Tensor, torch.Tensor, list[np.ndarray]]:
    """Apply per-ROI D4 transforms to image, dense heatmap and sparse offsets."""
    if not config.stage1.use_d4_augmentation or config.stage1.d4_augmentation_probability <= 0:
        return images, heatmaps, sparse_points
    if images.ndim != 4 or heatmaps.ndim != 3:
        raise ValueError("Stage-1 D4 augmentation expects BCHW images and BHW heatmaps")
    grid_size = int(heatmaps.shape[-1])
    stride = int(config.data.stage1_output_stride)
    transformed_images: list[torch.Tensor] = []
    transformed_heatmaps: list[torch.Tensor] = []
    transformed_points: list[np.ndarray] = []
    for batch_index in range(len(images)):
        image = images[batch_index]
        heatmap = heatmaps[batch_index]
        points = np.asarray(sparse_points[batch_index]).copy()
        if float(torch.rand(())) < float(config.stage1.d4_augmentation_probability):
            # Choose uniformly among the seven non-identity elements of D4.
            code = int(torch.randint(1, 8, ()).item())
            rotations = code % 4
            horizontal_reflection = code >= 4
            for _ in range(rotations):
                image = torch.rot90(image, 1, dims=(-2, -1))
                heatmap = torch.rot90(heatmap, 1, dims=(-2, -1))
                points = _transform_points_rot90_once(points, grid_size, stride)
            if horizontal_reflection:
                image = torch.flip(image, dims=(-1,))
                heatmap = torch.flip(heatmap, dims=(-1,))
                points = _transform_points_hflip(points, grid_size, stride)
        if len(points):
            max_offset = max(
                float(np.max(np.abs(points["offset_x"]))),
                float(np.max(np.abs(points["offset_y"]))),
            )
            if max_offset > float(config.stage1.offset_range_grid_units) + 1.0e-5:
                raise RuntimeError(
                    f"D4 target offset {max_offset:.4f} exceeds model range "
                    f"{config.stage1.offset_range_grid_units:.4f}"
                )
            packed = points["grid_y"].astype(np.int64) * grid_size + points["grid_x"].astype(np.int64)
            if len(np.unique(packed)) != len(points):
                raise RuntimeError("D4 augmentation created a duplicate Stage-1 target cell")
        transformed_images.append(image)
        transformed_heatmaps.append(heatmap)
        transformed_points.append(points)
    return (
        torch.stack(transformed_images, dim=0).contiguous(),
        torch.stack(transformed_heatmaps, dim=0).contiguous(),
        transformed_points,
    )


def _photometric_augmentation(images: torch.Tensor, config: PipelineConfig) -> torch.Tensor:
    if not config.stage1.photometric_augmentation:
        return images
    batch = len(images)
    device = images.device
    brightness = torch.empty(batch, 1, 1, 1, device=device).uniform_(*map(float, config.stage1.brightness_range))
    contrast = torch.empty(batch, 1, 1, 1, device=device).uniform_(*map(float, config.stage1.contrast_range))
    gamma = torch.empty(batch, 1, 1, 1, device=device).uniform_(*map(float, config.stage1.gamma_range))
    channel = torch.empty(batch, images.shape[1], 1, 1, device=device).uniform_(
        *map(float, config.stage1.channel_scale_range)
    )
    mean = images.mean(dim=(2, 3), keepdim=True)
    images = (images - mean) * contrast + mean
    images = (images * brightness * channel).clamp(0, 1).pow(gamma)
    return images.clamp(0, 1)


def _cosine_lambda(step: int, total_steps: int, warmup_steps: int, min_ratio: float) -> float:
    if step < warmup_steps:
        return max(float(min_ratio), (step + 1) / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress))


def _roi_gt_map(artifacts: dict[str, np.ndarray], roi_indices: np.ndarray) -> dict[int, np.ndarray]:
    nuclei = artifacts["nuclei"]
    offsets = artifacts["offsets"]
    return {int(roi): np.asarray(nuclei[int(offsets[roi]) : int(offsets[roi + 1])]) for roi in roi_indices}


def _validate_stage1_targets(artifacts: dict[str, np.ndarray], config: PipelineConfig) -> None:
    """Cheap preflight check for the exact supervision contract Stage-1 needs."""
    heatmaps = artifacts["heatmaps"]
    points = artifacts["point_targets"]
    offsets = np.asarray(artifacts["offsets"], dtype=np.int64)
    nuclei = artifacts["nuclei"]
    roi_count = len(artifacts["folds"])
    if len(offsets) != roi_count + 1 or int(offsets[-1]) != len(points) or len(points) != len(nuclei):
        raise RuntimeError("Stage-1 preprocessing arrays disagree on nucleus/target counts. Re-run notebook 00.")
    height, width = heatmaps.shape[1:]
    max_offset = float(config.stage1.offset_range_grid_units)
    for roi in range(roi_count):
        start, stop = int(offsets[roi]), int(offsets[roi + 1])
        rows = np.asarray(points[start:stop])
        if not len(rows):
            continue
        gx = rows["grid_x"].astype(np.int64)
        gy = rows["grid_y"].astype(np.int64)
        if np.any(gx < 0) or np.any(gx >= width) or np.any(gy < 0) or np.any(gy >= height):
            raise RuntimeError(f"Stage-1 target grid coordinate outside heatmap in ROI {roi}. Re-run notebook 00.")
        packed = gy * width + gx
        if len(np.unique(packed)) != len(rows):
            raise RuntimeError(f"Duplicate Stage-1 target cells in ROI {roi}. Re-run notebook 00.")
        ox = rows["offset_x"].astype(np.float32)
        oy = rows["offset_y"].astype(np.float32)
        if not np.isfinite(ox).all() or not np.isfinite(oy).all():
            raise RuntimeError(f"Non-finite Stage-1 offset target in ROI {roi}. Re-run notebook 00.")
        if np.any(np.abs(ox) > max_offset + 1.0e-5) or np.any(np.abs(oy) > max_offset + 1.0e-5):
            raise RuntimeError(f"Stage-1 offset target exceeds configured range in ROI {roi}. Re-run notebook 00.")
        positive_values = np.asarray(heatmaps[roi])[gy, gx]
        if not np.all(positive_values == 1.0):
            raise RuntimeError(
                f"Stage-1 heatmap is missing exact positive peaks in ROI {roi}. "
                "This target cannot train focal_heatmap_loss correctly; re-run notebook 00."
            )


def _localization_quality(metrics: dict[str, Any]) -> float:
    distance = metrics.get("mean_match_distance")
    if distance is None:
        return 0.0
    return float(np.clip(1.0 - float(distance) / float(PUBLIC_MATCH_RADIUS_PIXELS), 0.0, 1.0))


def _operating_point_key(
    metrics: dict[str, Any], *, threshold: float, suppression_radius_pixels: float, minimum_recall: float
) -> tuple[float, ...]:
    """Recall-constrained Stage-1 selection key.

    Stage 1 feeds Stage 2, so lost nuclei are unrecoverable. Once the recall floor
    is met, maximize F1 and precision; localization is a tertiary tie-breaker.
    If no operating point reaches the floor, maximize recall first.
    """
    recall = float(metrics["recall"])
    f1 = float(metrics["f1"])
    precision = float(metrics["precision"])
    localization = _localization_quality(metrics)
    accuracy = float(metrics["accuracy"])
    if recall >= float(minimum_recall):
        return (1.0, f1, precision, localization, recall, accuracy, -float(threshold), -float(suppression_radius_pixels))
    return (0.0, recall, f1, precision, localization, accuracy, -float(threshold), -float(suppression_radius_pixels))


def _evaluate_cached_operating_point(
    candidate_cache: dict[int, np.ndarray],
    gt_map: dict[int, np.ndarray],
    validation_indices: np.ndarray,
    *,
    threshold: float,
    suppression_radius_pixels: float,
    max_candidates: int,
    minimum_recall: float,
) -> dict[str, Any]:
    predictions = {
        int(roi): decode_stage1_candidates(
            candidate_cache[int(roi)],
            threshold=float(threshold),
            suppression_radius_pixels=float(suppression_radius_pixels),
            max_candidates=int(max_candidates),
        )
        for roi in validation_indices
    }
    metrics = binary_detection_metrics(
        gt_map, predictions, validation_indices, radius=PUBLIC_MATCH_RADIUS_PIXELS
    )
    key = _operating_point_key(
        metrics,
        threshold=float(threshold),
        suppression_radius_pixels=float(suppression_radius_pixels),
        minimum_recall=float(minimum_recall),
    )
    return {
        "threshold": float(threshold),
        "suppression_radius_pixels": float(suppression_radius_pixels),
        "meets_recall_floor": bool(float(metrics["recall"]) >= float(minimum_recall)),
        "selection_key": [float(value) for value in key],
        "localization_quality": float(_localization_quality(metrics)),
        "metrics": metrics,
        "predicted_rois": int(sum(len(rows) > 0 for rows in predictions.values())),
    }


def _validation_operating_point_sweep(
    model: torch.nn.Module,
    loader: DataLoader,
    gt_map: dict[int, np.ndarray],
    validation_indices: np.ndarray,
    config: PipelineConfig,
    device: torch.device,
    amp_dtype: torch.dtype,
    *,
    description: str = "Stage1 validation",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Forward validation once, then search fused threshold x NMS on cached peaks."""
    model.eval()
    thresholds = tuple(sorted({float(value) for value in config.stage1.validation_thresholds}))
    radii = tuple(sorted({float(value) for value in config.stage1.validation_suppression_radii_pixels}))
    base_radius = float(config.stage1.suppression_radius_pixels)
    minimum_score = min(thresholds)
    candidate_cache: dict[int, np.ndarray] = {}

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            if config.runtime.channels_last and device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                outputs = model(images)
            for local_index, roi in enumerate(batch["roi_index"].tolist()):
                single = {key: value[local_index : local_index + 1] for key, value in outputs.items()}
                candidate_cache[int(roi)] = extract_stage1_candidates(
                    single,
                    minimum_proposal_score=minimum_score,
                    local_max_radius=config.stage1.local_max_radius,
                    max_candidates=config.stage1.max_candidates_per_roi,
                    quality_power=config.stage1.proposal_quality_power,
                    uncertainty_penalty=config.stage1.proposal_uncertainty_penalty,
                )

    rows_by_setting: dict[tuple[float, float], dict[str, Any]] = {}

    def evaluate(threshold: float, radius: float) -> dict[str, Any]:
        setting = (float(threshold), float(radius))
        if setting not in rows_by_setting:
            rows_by_setting[setting] = _evaluate_cached_operating_point(
                candidate_cache,
                gt_map,
                validation_indices,
                threshold=setting[0],
                suppression_radius_pixels=setting[1],
                max_candidates=config.stage1.max_candidates_per_roi,
                minimum_recall=config.stage1.validation_min_recall,
            )
        return rows_by_setting[setting]

    # Exact decoder search: every configured fused-score threshold is crossed
    # with every configured NMS radius. Dense model work is still done only once.
    search_radii = tuple(sorted(set(radii + (base_radius,))))
    for threshold in thresholds:
        for radius in search_radii:
            evaluate(threshold, radius)

    sweep = sorted(rows_by_setting.values(), key=lambda row: (row["threshold"], row["suppression_radius_pixels"]))
    best = max(sweep, key=lambda row: tuple(float(v) for v in row["selection_key"]))
    return sweep, best


def _print_epoch_metrics(
    epoch: int,
    fold: int,
    loss_value: float,
    loss_parts: dict[str, float],
    metrics: dict[str, Any],
    ema_decay: float,
    *,
    best_threshold: float,
    best_suppression_radius: float,
    meets_recall_floor: bool,
) -> None:
    distance = metrics.get("mean_match_distance")
    distance_text = "NA" if distance is None else f"{float(distance):.3f}px"
    print(
        f"fold={fold} | epoch={epoch:03d} | loss={loss_value:.5f} "
        f"(heatmap={loss_parts['heatmap']:.4f}, offset={loss_parts['offset']:.4f}, "
        f"quality={loss_parts['quality']:.4f}, hardneg={loss_parts['hard_negative']:.4f}, "
        f"uncertainty={loss_parts['uncertainty']:.4f}) | "
        f"proposal_thr={best_threshold:.4f} | NMS={best_suppression_radius:.1f}px | "
        f"recall_floor={'PASS' if meets_recall_floor else 'MISS'} | "
        f"F1@15px={metrics['f1']:.6f} | Accuracy@15px={metrics['accuracy']:.6f} | "
        f"Precision@15px={metrics['precision']:.6f} | Recall@15px={metrics['recall']:.6f} | "
        f"TP={metrics['tp']} | FP={metrics['fp']} | FN={metrics['fn']} | "
        f"proposals={metrics['predictions']} | GT={metrics['ground_truth']} | "
        f"mean_match={distance_text} | EMA_decay={ema_decay:.6f}"
    )


def _ema_decay(target_decay: float, update: int) -> float:
    if update < 1:
        return 0.0
    return min(float(target_decay), (1.0 + float(update)) / (10.0 + float(update)))


@torch.no_grad()
def _ema_update(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    ema_parameters = dict(ema_model.named_parameters())
    model_parameters = dict(model.named_parameters())
    for name, ema_parameter in ema_parameters.items():
        ema_parameter.mul_(decay).add_(model_parameters[name].detach(), alpha=1.0 - decay)
    ema_buffers = dict(ema_model.named_buffers())
    for name, buffer in model.named_buffers():
        if name in ema_buffers:
            ema_buffers[name].copy_(buffer.detach())


def _hard_negative_weight(config: PipelineConfig, epoch: int) -> float:
    start = int(config.stage1.hard_negative_start_epoch)
    if epoch < start:
        return 0.0
    progress = min(1.0, float(epoch - start + 1) / float(max(1, config.stage1.hard_negative_ramp_epochs)))
    return float(config.stage1.hard_negative_loss_weight) * progress


def _build_stage1_optimizer(config: PipelineConfig, model: torch.nn.Module, device: torch.device) -> AdamW:
    backbone_parameters = list(model.encoder.parameters())
    head_parameters = list(model.fpn.parameters()) + list(model.head.parameters())
    groups = [
        {"params": backbone_parameters, "lr": config.stage1.learning_rate * config.stage1.backbone_learning_rate_scale},
        {"params": head_parameters, "lr": config.stage1.learning_rate},
    ]
    kwargs: dict[str, Any] = {"weight_decay": config.stage1.weight_decay}
    if config.runtime.fused_adamw and device.type == "cuda":
        kwargs["fused"] = True
    try:
        return AdamW(groups, **kwargs)
    except TypeError:
        kwargs.pop("fused", None)
        return AdamW(groups, **kwargs)


def train_stage1_fold(
    config: PipelineConfig,
    fold: int,
    *,
    epoch_callback: Callable[[int, dict[str, Any], Path], None] | None = None,
) -> dict[str, Any]:
    config.validate()
    if int(fold) < 0 or int(fold) >= config.data.number_of_folds:
        raise ValueError(f"Invalid Stage-1 fold: {fold}")
    runtime = configure_runtime(
        config.data.seed + int(fold),
        use_bfloat16=config.runtime.use_bfloat16,
        allow_tf32=config.runtime.allow_tf32,
        deterministic=config.runtime.deterministic,
    )
    artifacts = load_shared_artifacts(config)
    _validate_stage1_targets(artifacts, config)
    folds = np.asarray(artifacts["folds"])
    train_indices = np.flatnonzero(folds != fold)
    validation_indices = np.flatnonzero(folds == fold)
    if not len(train_indices) or not len(validation_indices):
        raise RuntimeError(f"Stage-1 fold {fold} has an empty train or validation split")
    _assert_no_case_leakage(artifacts, train_indices, validation_indices, fold=fold)

    workers = min(config.data.resolved_workers, 8)
    train_kwargs: dict[str, Any] = {
        "batch_size": config.stage1.micro_batch_size,
        "shuffle": True,
        "num_workers": workers,
        "pin_memory": config.runtime.pin_memory,
        "persistent_workers": config.runtime.persistent_workers and workers > 0,
        "collate_fn": collate_stage1,
        "worker_init_fn": dataloader_worker_init,
    }
    if workers > 0:
        train_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(Stage1Dataset(config, train_indices), **train_kwargs)
    validation_workers = workers
    validation_kwargs: dict[str, Any] = {
        "batch_size": min(4, config.stage1.micro_batch_size),
        "shuffle": False,
        "num_workers": validation_workers,
        "pin_memory": config.runtime.pin_memory,
        "persistent_workers": config.runtime.persistent_workers and validation_workers > 0,
        "collate_fn": collate_stage1,
        "worker_init_fn": dataloader_worker_init,
    }
    if validation_workers > 0:
        validation_kwargs["prefetch_factor"] = 4
    validation_loader = DataLoader(Stage1Dataset(config, validation_indices), **validation_kwargs)

    model = build_stage1_model(config).to(runtime.device)
    if config.runtime.channels_last and runtime.device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)

    optimizer = _build_stage1_optimizer(config, model, runtime.device)
    total_updates = max(1, math.ceil(len(train_loader) / config.stage1.accumulation_steps) * config.stage1.epochs)
    warmup_updates = math.ceil(len(train_loader) / config.stage1.accumulation_steps) * config.stage1.warmup_epochs
    minimum_ratio = config.stage1.minimum_learning_rate / config.stage1.learning_rate
    scheduler = LambdaLR(optimizer, lambda step: _cosine_lambda(step, total_updates, warmup_updates, minimum_ratio))
    scaler = make_grad_scaler(enabled=runtime.device.type == "cuda" and runtime.amp_dtype == torch.float16)
    run_directory = stage1_run_directory(config)
    fold_directory = run_directory / f"fold_{fold}"
    manager = CheckpointManager(fold_directory / "checkpoints")
    history = TrainingHistory(fold_directory)

    signature = _stage1_training_signature(config)
    if manager.latest_path.is_file():
        peek = torch.load(manager.latest_path, map_location="cpu", weights_only=False)
        previous_signature = dict(peek.get("extra") or {}).get("training_signature")
        if previous_signature != signature:
            raise RuntimeError(
                f"Stage-1 resume config mismatch at {manager.latest_path}. "
                "Do not mix the old run with the new augmentation/batch/decoder contract; "
                "archive or remove this fold's old checkpoints and start it fresh."
            )

    start_epoch, resume_extra = manager.resume(model, optimizer, scheduler, scaler, map_location=runtime.device)
    if start_epoch > config.stage1.epochs:
        raise RuntimeError(f"Stage-1 resume checkpoint epoch exceeds configured epochs: {manager.latest_path}")
    history.truncate_from("epoch", start_epoch)
    if resume_extra.get("ema_model") is not None:
        ema_model.load_state_dict(resume_extra["ema_model"], strict=True)
    else:
        ema_model.load_state_dict(model.state_dict(), strict=True)
    updates_per_epoch = math.ceil(len(train_loader) / config.stage1.accumulation_steps)
    ema_updates = int(resume_extra.get("ema_updates", start_epoch * updates_per_epoch))
    gt_map = _roi_gt_map(artifacts, validation_indices)

    saved_key = resume_extra.get("best_validation_key")
    best_validation_key: tuple[float, ...] | None = (
        tuple(float(value) for value in saved_key) if isinstance(saved_key, (list, tuple)) else None
    )
    best_validation_score = float(resume_extra.get("best_validation_score", -math.inf))
    best_validation_threshold = resume_extra.get("best_validation_threshold")
    best_validation_suppression_radius = resume_extra.get("best_validation_suppression_radius")
    best_validation_epoch = resume_extra.get("best_validation_epoch")
    best_validation_metrics = resume_extra.get("best_validation_metrics")

    for epoch in range(start_epoch, config.stage1.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        running_parts = {
            "heatmap": 0.0,
            "offset": 0.0,
            "quality": 0.0,
            "uncertainty": 0.0,
            "hard_negative": 0.0,
        }
        batches = 0
        weights = {
            "heatmap": config.stage1.heatmap_loss_weight,
            "offset": config.stage1.offset_loss_weight,
            "quality": config.stage1.quality_loss_weight,
            "uncertainty": config.stage1.uncertainty_loss_weight,
            "hard_negative": _hard_negative_weight(config, epoch),
        }
        if runtime.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(runtime.device)
        for step, batch in enumerate(train_loader):
            cpu_images, cpu_heatmaps, sparse_points = _d4_augmentation(
                batch["image"], batch["heatmap"], batch["points"], config
            )
            images = cpu_images.to(runtime.device, non_blocking=True)
            heatmaps = cpu_heatmaps.to(runtime.device, non_blocking=True)
            if config.runtime.channels_last and runtime.device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            images = _photometric_augmentation(images, config)
            with torch.autocast(device_type=runtime.device.type, dtype=runtime.amp_dtype, enabled=runtime.amp_enabled):
                outputs = model(images)
                loss, batch_loss_parts = stage1_loss(
                    outputs,
                    heatmaps,
                    sparse_points,
                    weights=weights,
                    quality_target_sigma_grid_units=config.stage1.quality_target_sigma_grid_units,
                    hard_negative_topk_per_image=config.stage1.hard_negative_topk_per_image,
                )
                group_start = (step // config.stage1.accumulation_steps) * config.stage1.accumulation_steps
                group_size = min(config.stage1.accumulation_steps, len(train_loader) - group_start)
                scaled_loss = loss / max(1, group_size)
            if not torch.isfinite(loss).all():
                raise FloatingPointError(
                    f"Non-finite Stage-1 loss at fold={fold}, epoch={epoch}, step={step}: {float(loss.detach())}"
                )
            scaler.scale(scaled_loss).backward()
            if (step + 1) % config.stage1.accumulation_steps == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.stage1.gradient_clip_norm, error_if_nonfinite=True
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema_updates += 1
                ema_decay = _ema_decay(config.stage1.ema_decay, ema_updates)
                _ema_update(ema_model, model, ema_decay)
            # stage1_loss already transferred this scalar with the other metrics.
            running += batch_loss_parts["total"]
            for name in running_parts:
                running_parts[name] += float(batch_loss_parts[name])
            batches += 1

        operating_point_sweep, selected = _validation_operating_point_sweep(
            ema_model,
            validation_loader,
            gt_map,
            validation_indices,
            config,
            runtime.device,
            runtime.amp_dtype,
            description=f"Stage1 fold {fold} validation epoch {epoch+1}/{config.stage1.epochs}",
        )
        selected_threshold = float(selected["threshold"])
        selected_radius = float(selected["suppression_radius_pixels"])
        selected_key = tuple(float(value) for value in selected["selection_key"])
        metrics = selected["metrics"]
        selected_score = float(metrics["f1"])
        prediction_count = int(metrics["predictions"])
        predicted_rois = int(selected["predicted_rois"])
        epoch_loss = running / max(1, batches)
        epoch_loss_parts = {name: value / max(1, batches) for name, value in running_parts.items()}
        peak_vram_gb = (
            float(torch.cuda.max_memory_reserved(runtime.device)) / (1024.0**3)
            if runtime.device.type == "cuda" else None
        )
        record = {
            "fold": int(fold),
            "epoch": int(epoch),
            "train_loss": float(epoch_loss),
            "train_loss_parts": epoch_loss_parts,
            "hard_negative_weight": float(weights["hard_negative"]),
            "learning_rate": float(optimizer.param_groups[1]["lr"]),
            "learning_rate_backbone": float(optimizer.param_groups[0]["lr"]),
            "learning_rate_heads": float(optimizer.param_groups[1]["lr"]),
            "binary_detection": metrics,
            "best_validation_threshold": selected_threshold,
            "best_validation_suppression_radius": selected_radius,
            "best_validation_score": selected_score,
            "best_validation_key": list(selected_key),
            "meets_recall_floor": bool(selected["meets_recall_floor"]),
            "operating_point_sweep": operating_point_sweep,
            "prediction_count": prediction_count,
            "predicted_rois": predicted_rois,
            "ema_updates": int(ema_updates),
            "ema_decay": float(_ema_decay(config.stage1.ema_decay, ema_updates)),
            "peak_reserved_vram_gb": peak_vram_gb,
        }
        history.append(record)
        if epoch_callback is not None:
            epoch_callback(int(fold), record, run_directory)
        _print_epoch_metrics(
            epoch, fold, epoch_loss, epoch_loss_parts, metrics,
            _ema_decay(config.stage1.ema_decay, ema_updates),
            best_threshold=selected_threshold,
            best_suppression_radius=selected_radius,
            meets_recall_floor=bool(selected["meets_recall_floor"]),
        )
        if peak_vram_gb is not None:
            print(f"  peak_reserved_vram={peak_vram_gb:.2f} GB | micro/effective={config.stage1.micro_batch_size}/{config.stage1.effective_batch_size}")

        if best_validation_key is None or selected_key > best_validation_key:
            best_validation_key = selected_key
            best_validation_score = selected_score
            best_validation_threshold = selected_threshold
            best_validation_suppression_radius = selected_radius
            best_validation_epoch = int(epoch)
            best_validation_metrics = dict(metrics)
            manager.save_best(
                model=ema_model,
                extra={
                    "fold": int(fold),
                    "epoch": int(epoch),
                    "weights": "ema",
                    "training_signature": signature,
                    "best_validation_threshold": best_validation_threshold,
                    "best_validation_suppression_radius": best_validation_suppression_radius,
                    "best_validation_score": best_validation_score,
                    "best_validation_key": list(best_validation_key),
                    "best_validation_metrics": best_validation_metrics,
                    "decode_config": {
                        "score_name": "heatmap_x_quality",
                        "proposal_quality_power": float(config.stage1.proposal_quality_power),
                        "proposal_uncertainty_penalty": float(config.stage1.proposal_uncertainty_penalty),
                        "local_max_radius": int(config.stage1.local_max_radius),
                        "suppression_radius_pixels": best_validation_suppression_radius,
                        "max_candidates_per_roi": int(config.stage1.max_candidates_per_roi),
                    },
                    "selection": (
                        f"recall>={config.stage1.validation_min_recall:.3f}; then F1, precision, "
                        "localization quality, recall, accuracy"
                    ),
                },
            )
            print(
                f"  -> NEW BEST fold={fold}: epoch={epoch:03d}, proposal_thr={best_validation_threshold:.4f}, "
                f"NMS={best_validation_suppression_radius:.1f}px, F1={metrics['f1']:.6f}, "
                f"P={metrics['precision']:.6f}, R={metrics['recall']:.6f}, Acc={metrics['accuracy']:.6f}"
            )
        if (epoch + 1) % config.stage1.checkpoint_every_epochs == 0 or epoch + 1 == config.stage1.epochs:
            manager.save_latest(
                epoch=epoch, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                extra={
                    "metrics": metrics,
                    "ema_model": ema_model.state_dict(),
                    "ema_updates": int(ema_updates),
                    "fold": int(fold),
                    "training_signature": signature,
                    "best_validation_score": float(best_validation_score),
                    "best_validation_key": None if best_validation_key is None else list(best_validation_key),
                    "best_validation_threshold": best_validation_threshold,
                    "best_validation_suppression_radius": best_validation_suppression_radius,
                    "best_validation_epoch": best_validation_epoch,
                    "best_validation_metrics": best_validation_metrics,
                },
            )

    if not manager.best_path.is_file():
        raise RuntimeError(f"Stage-1 fold {fold} finished without best.pt")
    # IMPORTANT: final.pt is the fixed-epoch EMA state and does NOT load best.pt.
    # best.pt is selected with this fold's held labels and is therefore diagnostic
    # only. Stage-2 OOF generation uses final.pt plus an operating point calibrated
    # from the *other* folds, keeping the target fold completely untouched.
    final_extra = {
        "epochs": config.stage1.epochs,
        "fold": int(fold),
        "weights": "ema_final_fixed_epoch",
        "training_signature": signature,
        "oof_safe": True,
        "held_diagnostic_best_validation_score": float(best_validation_score),
        "held_diagnostic_best_validation_key": None if best_validation_key is None else list(best_validation_key),
        "held_diagnostic_best_validation_threshold": best_validation_threshold,
        "held_diagnostic_best_validation_suppression_radius": best_validation_suppression_radius,
        "held_diagnostic_best_validation_epoch": best_validation_epoch,
        "diagnostic_best_checkpoint": str(manager.best_path),
    }
    manager.save_final(model=ema_model, extra=final_extra)
    result = {
        "fold": int(fold),
        "best_checkpoint": str(manager.best_path),
        "best_validation_threshold": best_validation_threshold,
        "best_validation_suppression_radius": best_validation_suppression_radius,
        "best_validation_score": float(best_validation_score),
        "final_checkpoint": str(manager.final_path),
    }
    # Both loaders keep workers persistent throughout the fold. Shut them down
    # explicitly before the next fold starts so no semaphore/file-descriptor
    # state can leak across fold boundaries.
    shutdown_dataloader(train_loader)
    shutdown_dataloader(validation_loader)
    return result


def train_all_stage1_folds(
    config: PipelineConfig,
    folds: list[int] | None = None,
    *,
    epoch_callback: Callable[[int, dict[str, Any], Path], None] | None = None,
) -> list[dict[str, Any]]:
    config.validate()
    if folds is None:
        folds = list(range(config.data.number_of_folds))
    else:
        folds = [int(fold) for fold in folds]
        if not folds:
            raise ValueError("folds cannot be empty; pass None explicitly to train all Stage-1 folds")
        if len(set(folds)) != len(folds):
            raise ValueError("folds cannot contain duplicate fold IDs")
        invalid = [fold for fold in folds if fold < 0 or fold >= config.data.number_of_folds]
        if invalid:
            raise ValueError(f"Invalid Stage-1 fold IDs: {invalid}")
    run_directory = stage1_run_directory(config)
    config.write_resolved(run_directory / "resolved_config.json")
    output: list[dict[str, Any]] = []
    for fold in folds:
        try:
            output.append(train_stage1_fold(config, int(fold), epoch_callback=epoch_callback))
        finally:
            # train_stage1_fold's frame has been released here, so cached CUDA
            # blocks that are no longer referenced can safely be returned.
            release_unused_memory()
    complete_path = run_directory / "training_complete.json"
    complete_tmp = complete_path.with_suffix(complete_path.suffix + ".tmp")
    complete_tmp.write_text(json.dumps(output, indent=2), encoding="utf-8")
    complete_tmp.replace(complete_path)
    return output
