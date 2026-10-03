from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import PipelineConfig
from ..constants import PUBLIC_MATCH_RADIUS_PIXELS, RAW_PROPOSAL_DTYPE
from ..data.preprocess import load_shared_artifacts
from ..evaluation.proposal_diagnostics import lexicographic_one_to_one_assignment
from ..utils.provenance import file_stat_identity, numpy_array_sha256, sha256_file
from ..utils.runtime import configure_runtime, dataloader_worker_count
from .data import Stage1Dataset, collate_stage1
from .decode import decode_stage1_outputs
from .model import build_stage1_model
from .registry import active_stage1_run


OOF_CHECKPOINT_POLICY = "fold-matched final.pt fixed-epoch EMA + fixed decoder operating point"


def _atomic_write_json(path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    temporary.replace(path)


def _atomic_save_npy(path, array: np.ndarray) -> None:
    """Commit an NPY artifact atomically so an interrupted Colab write is never reusable."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _expected_training_signature(config: PipelineConfig) -> dict[str, Any]:
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


def _load_oof_safe_model_and_operating_point(
    config: PipelineConfig,
    fold: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = active_stage1_run(config) / f"fold_{fold}" / "checkpoints" / "final.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing leakage-safe Stage-1 final checkpoint for fold {fold}: {checkpoint}. Complete that Stage-1 fold first."
        )
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    extra = dict(payload.get("extra") or {})
    if int(extra.get("fold", -1)) != int(fold):
        raise RuntimeError(f"Stage-1 final checkpoint fold mismatch: requested {fold}, checkpoint declares {extra.get('fold')}")
    if not bool(extra.get("oof_safe", False)) or str(extra.get("weights")) != "ema_final_fixed_epoch":
        raise RuntimeError(f"Stage-1 fold {fold} final.pt is not the fixed-epoch OOF-safe checkpoint")
    if extra.get("training_signature") != _expected_training_signature(config):
        raise RuntimeError(f"Stage-1 fold {fold} final.pt was trained with different critical settings")

    training_folds = [value for value in range(config.data.number_of_folds) if value != int(fold)]
    stat = checkpoint.stat()
    operating_point = {
        "fold": int(fold),
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": int(stat.st_size),
        "checkpoint_mtime_ns": int(stat.st_mtime_ns),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": int(config.stage1.epochs) - 1,
        "weights": "ema_final_fixed_epoch",
        "threshold": float(config.stage1.validation_decode_threshold),
        "suppression_radius_pixels": float(config.stage1.suppression_radius_pixels),
        "score_name": "heatmap_x_quality",
        "proposal_quality_power": float(config.stage1.proposal_quality_power),
        "proposal_uncertainty_penalty": float(config.stage1.proposal_uncertainty_penalty),
        "local_max_radius": int(config.stage1.local_max_radius),
        "max_candidates_per_roi": int(config.stage1.max_candidates_per_roi),
        "training_folds": training_folds,
        "decoder_label_dependency_folds": [],
        "decoder_selection": "fixed configuration; held labels never calibrate OOF decoder",
    }
    model = build_stage1_model(config).to(device)
    if config.runtime.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(payload["model"], strict=True)
    return model.eval(), operating_point


def _cached_oof_matches_current_sources(summary: dict[str, Any], config: PipelineConfig) -> bool:
    if summary.get("stage1_checkpoint_policy") != OOF_CHECKPOINT_POLICY:
        return False
    preprocessing_summary = config.paths.preprocessed / "preprocessing_summary.json"
    if not preprocessing_summary.is_file():
        return False
    if summary.get("shared_preprocessing_summary_sha256") != sha256_file(preprocessing_summary):
        return False
    output = config.paths.stage2_preprocessed / "stage1_oof"
    proposal_path = output / "proposals.npy"
    offsets_path = output / "roi_proposal_offsets.npy"
    if not proposal_path.is_file() or not offsets_path.is_file():
        return False
    identities = summary.get("artifact_identity")
    if isinstance(identities, dict):
        if identities.get("proposals.npy") != file_stat_identity(proposal_path):
            return False
        if identities.get("roi_proposal_offsets.npy") != file_stat_identity(offsets_path):
            return False
    else:
        # Existing OOF summaries already contain exact array SHA256 values.
        # Validate once, then bind cheap file identities for future Colab starts.
        cached_proposals = np.load(proposal_path, mmap_mode="r", allow_pickle=False)
        cached_offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
        if summary.get("proposal_sha256") != numpy_array_sha256(cached_proposals):
            return False
        if summary.get("roi_offsets_sha256") != numpy_array_sha256(cached_offsets):
            return False
        summary["artifact_identity"] = {
            "proposals.npy": file_stat_identity(proposal_path),
            "roi_proposal_offsets.npy": file_stat_identity(offsets_path),
        }
        _atomic_write_json(output / "summary.json", summary)
    points = summary.get("fold_operating_points")
    if not isinstance(points, list) or len(points) != config.data.number_of_folds:
        return False
    by_fold = {int(point.get("fold", -1)): point for point in points if isinstance(point, dict)}
    for fold in range(config.data.number_of_folds):
        point = by_fold.get(fold)
        if point is None:
            return False
        checkpoint = active_stage1_run(config) / f"fold_{fold}" / "checkpoints" / "final.pt"
        if not checkpoint.is_file():
            return False
        stat = checkpoint.stat()
        if int(point.get("checkpoint_size_bytes", -1)) != int(stat.st_size):
            return False
        if int(point.get("checkpoint_mtime_ns", -1)) != int(stat.st_mtime_ns):
            return False
        if point.get("checkpoint_sha256") != sha256_file(checkpoint):
            return False
        training_folds = [value for value in range(config.data.number_of_folds) if value != fold]
        if point.get("training_folds") != training_folds or fold in point.get("training_folds", []):
            return False
        if point.get("decoder_label_dependency_folds") != []:
            return False
        if float(point.get("threshold", -1)) != float(config.stage1.validation_decode_threshold):
            return False
        if float(point.get("suppression_radius_pixels", -1)) != float(config.stage1.suppression_radius_pixels):
            return False
    return True


def generate_stage1_oof_proposals(config: PipelineConfig, *, force: bool = False) -> dict[str, Any]:
    config.validate()
    output = config.paths.stage2_preprocessed / "stage1_oof"
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    proposal_path = output / "proposals.npy"
    offsets_path = output / "roi_proposal_offsets.npy"
    if not force and summary_path.is_file() and proposal_path.is_file() and offsets_path.is_file():
        cached_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if _cached_oof_matches_current_sources(cached_summary, config):
            print(f"Reusing leakage-safe Stage-1 OOF proposals: {output}")
            return cached_summary
        print("Stage-1 OOF cache is stale; regenerating it.")

    runtime = configure_runtime(
        config.data.seed,
        use_bfloat16=config.runtime.use_bfloat16,
        allow_tf32=config.runtime.allow_tf32,
        deterministic=config.runtime.deterministic,
    )
    artifacts = load_shared_artifacts(config)
    folds = np.asarray(artifacts["folds"])
    nuclei = artifacts["nuclei"]
    gt_offsets = artifacts["offsets"]
    all_rows: list[np.ndarray] = []
    fold_operating_points: list[dict[str, Any]] = []

    for fold in range(config.data.number_of_folds):
        model, operating_point = _load_oof_safe_model_and_operating_point(config, fold, runtime.device)
        fold_operating_points.append(operating_point)
        print(
            f"Stage1 OOF fold {fold}: final.pt epoch={operating_point['checkpoint_epoch']+1} "
            f"proposal_thr={operating_point['threshold']:.4f} NMS={operating_point['suppression_radius_pixels']:.1f}px "
            f"fixed_decoder=True training_folds={operating_point['training_folds']}"
        )
        roi_indices = np.flatnonzero(folds == fold)
        workers = min(8, dataloader_worker_count(config.data.cpu_workers))
        loader_kwargs: dict[str, Any] = {
            "batch_size": min(4, config.stage1.micro_batch_size),
            "shuffle": False,
            "num_workers": workers,
            "pin_memory": bool(config.runtime.pin_memory and runtime.device.type == "cuda"),
            "collate_fn": collate_stage1,
        }
        if workers > 0:
            loader_kwargs.update(persistent_workers=False, prefetch_factor=2)
        loader = DataLoader(Stage1Dataset(config, roi_indices), **loader_kwargs)
        for batch in loader:
            images = batch["image"].to(runtime.device, non_blocking=True)
            if config.runtime.channels_last and runtime.device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            with torch.inference_mode(), torch.autocast(
                device_type=runtime.device.type,
                dtype=runtime.amp_dtype,
                enabled=runtime.amp_enabled,
            ):
                outputs = model(images)
            for local_index, roi_value in enumerate(batch["roi_index"].tolist()):
                roi = int(roi_value)
                single = {key: value[local_index : local_index + 1] for key, value in outputs.items()}
                decoded = decode_stage1_outputs(
                    single,
                    threshold=operating_point["threshold"],
                    local_max_radius=operating_point["local_max_radius"],
                    suppression_radius_pixels=operating_point["suppression_radius_pixels"],
                    max_candidates=operating_point["max_candidates_per_roi"],
                    quality_power=operating_point["proposal_quality_power"],
                    uncertainty_penalty=operating_point["proposal_uncertainty_penalty"],
                )
                gt = np.asarray(nuclei[int(gt_offsets[roi]) : int(gt_offsets[roi + 1])])
                gt_xy = np.column_stack((gt["x"], gt["y"])).astype(np.float64) if len(gt) else np.empty((0, 2))
                pred_xy = np.column_stack((decoded["x"], decoded["y"])).astype(np.float64) if len(decoded) else np.empty((0, 2))
                gt_match, pred_match, distances = lexicographic_one_to_one_assignment(
                    gt_xy, pred_xy, PUBLIC_MATCH_RADIUS_PIXELS
                )
                matched_by_pred = {
                    int(pred): (int(gt_index), float(distance))
                    for gt_index, pred, distance in zip(gt_match, pred_match, distances, strict=True)
                }
                rows = np.empty(len(decoded), dtype=RAW_PROPOSAL_DTYPE)
                for index, item in enumerate(decoded):
                    match = matched_by_pred.get(index)
                    if match is None:
                        gt_global, class_id, reject = -1, -1, 1
                    else:
                        local_gt, _ = match
                        gt_row = gt[local_gt]
                        gt_global, class_id, reject = int(gt_row["nucleus_index"]), int(gt_row["class_id"]), 0
                    rows[index] = (
                        f"roi{roi}:stage1:{index}", roi, index,
                        float(item["x"]), float(item["y"]), float(item["heatmap_score"]),
                        float(item["quality"]), float(item["uncertainty"]), float(item["peak_sharpness"]),
                        gt_global, class_id, reject, fold,
                    )
                all_rows.append(rows)
        del model
        if runtime.device.type == "cuda":
            torch.cuda.empty_cache()

    proposals = np.concatenate(all_rows) if all_rows else np.empty(0, dtype=RAW_PROPOSAL_DTYPE)
    if len(proposals):
        proposals = proposals[np.lexsort((proposals["candidate_index"], proposals["roi_index"]))]
    counts = np.bincount(proposals["roi_index"], minlength=len(folds)) if len(proposals) else np.zeros(len(folds), dtype=np.int64)
    roi_offsets = np.zeros(len(folds) + 1, dtype=np.int64)
    roi_offsets[1:] = np.cumsum(counts)
    _atomic_save_npy(proposal_path, proposals)
    _atomic_save_npy(offsets_path, roi_offsets)
    zero_proposal_rois = np.flatnonzero(counts == 0).astype(int)
    summary = {
        "proposal_count": int(len(proposals)),
        "matched_count": int(np.sum(proposals["is_reject"] == 0)),
        "reject_count": int(np.sum(proposals["is_reject"] == 1)),
        "zero_proposal_roi_count": int(len(zero_proposal_rois)),
        "zero_proposal_roi_indices": zero_proposal_rois.tolist(),
        "min_proposals_per_roi": int(counts.min()) if len(counts) else 0,
        "max_proposals_per_roi": int(counts.max()) if len(counts) else 0,
        "stage1_checkpoint_policy": OOF_CHECKPOINT_POLICY,
        "proposal_sha256": numpy_array_sha256(proposals),
        "roi_offsets_sha256": numpy_array_sha256(roi_offsets),
        "artifact_identity": {
            "proposals.npy": file_stat_identity(proposal_path),
            "roi_proposal_offsets.npy": file_stat_identity(offsets_path),
        },
        "shared_preprocessing_summary_sha256": sha256_file(
            config.paths.preprocessed / "preprocessing_summary.json"
        ),
        "leakage_contract": (
            "each target fold uses a model trained on the other folds; the OOF decoder operating point is fixed "
            "in configuration and has no held-label calibration dependency"
        ),
        "fold_operating_points": fold_operating_points,
    }
    _atomic_write_json(summary_path, summary)
    return summary
