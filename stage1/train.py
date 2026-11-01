"""Stage 1: one best checkpoint + full train/validation CSV every epoch.

Completed matching runs are skipped. Previous outputs are preserved when a
fold restarts from epoch one; optimizer state is never resumed.
"""

from __future__ import annotations
import copy
import csv
import math
import random
import time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from common.run_history import SCHEMA, fingerprint, initialize, append_csv
from .config import default_pipeline_config, DETECTOR_REVISION
from .models.build import build_model
from .dataset import Cache, TileDataset, collate, tile_origins
from .splits import load_splits
from .losses import stage1_loss
from .evaluate import collect_candidates, choose_tile_batch_size, sweep_cached, match_counts, scores
from .decode import decode_stage1_candidates
from common.checkpoint import save_best


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _worker_seed(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _ema_update(ema, model, decay):
    with torch.no_grad():
        torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1.0 - decay)
        for dst, src in zip(ema.buffers(), model.buffers()):
            dst.copy_(src)


def _optimizer_step(parameters, optimizer, scaler, max_norm):
    """Return the gradient norm after an update, or None for AMP overflow."""
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    if not torch.isfinite(norm):
        if not scaler.is_enabled():
            raise FloatingPointError("Nonfinite Stage 1 gradient without loss scaling")
        # Do not update weights, EMA or scheduler. Reset scaler state and retry
        # the next accumulation group at a smaller scale.
        scaler.update(new_scale=scaler.get_scale() * scaler.get_backoff_factor())
        return None
    scaler.step(optimizer)
    scaler.update()
    return norm


def choose_batch_size(model, optimizer, device, tile_size, effective_batch, amp):
    if device.type != "cuda":
        return 1
    for batch_size in (16, 8, 4, 2, 1):
        if effective_batch % batch_size:
            continue
        images = pred = loss = reserve = None
        try:
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                images = torch.rand(batch_size, 3, tile_size, tile_size, device=device)
                pred = model(images)
                loss = diagnostic_loss(pred)
            loss.backward()
            model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            reserve = torch.empty(model_bytes * 2, dtype=torch.uint8, device=device)
            return batch_size
        except torch.OutOfMemoryError:
            pass
        finally:
            optimizer.zero_grad(set_to_none=True)
            del images, pred, loss, reserve
            torch.cuda.empty_cache()
    raise RuntimeError("Even micro_batch=1 does not fit. Reduce tile_size.")


def diagnostic_loss(outputs):
    """Exercise the real dense and sparse objectives for memory/gradient checks."""
    batch, _, height, width = outputs["heatmap_logits"].shape
    heatmap = outputs["heatmap_logits"].new_zeros((batch, height, width), dtype=torch.float32)
    dtype = np.dtype([("grid_x", "i4"), ("grid_y", "i4"), ("offset_x", "f4"), ("offset_y", "f4")])
    points = np.array(
        [(width // 3, height // 3, 0.25, 0.25), (2 * width // 3, 2 * height // 3, 0.25, 0.25)],
        dtype=dtype,
    )
    heatmap[:, points["grid_y"], points["grid_x"]] = 1
    loss, _ = stage1_loss(
        outputs,
        heatmap,
        [points] * batch,
        weights={
            "heatmap": 0.55,
            "offset": 0.18,
            "quality": 0.12,
            "uncertainty": 0.05,
            "hard_negative": 0.1,
        },
        return_metrics=False,
    )
    return loss


def _measure_at(candidates, cache, names, *, threshold, radius):
    total = {"tp": 0, "fp": 0, "fn": 0}
    roi_f1 = []
    for name in names:
        points = decode_stage1_candidates(
            candidates[name],
            threshold=threshold,
            suppression_radius_pixels=radius,
            max_candidates=4000,
        )
        pred = np.column_stack((points["x"], points["y"])) if len(points) else np.empty((0, 2))
        gt = cache.gt_for(name)
        truth = np.column_stack((gt["x"], gt["y"])) if len(gt) else np.empty((0, 2))
        c = match_counts(pred, truth)
        for key in total:
            total[key] += c[key]
        roi_f1.append(scores(**c)["f1"])
    return {**total, **scores(**total), "roi_macro_f1": float(np.mean(roi_f1)) if roi_f1 else 0.0}


COLS = [
    "epoch",
    "run_id",
    "train_loss",
    "val_loss",
    *[
        f"train_{x}"
        for x in ("accuracy", "precision", "recall", "f1", "roi_macro_f1", "tp", "fp", "fn")
    ],
    *[
        f"val_{x}"
        for x in ("accuracy", "precision", "recall", "f1", "roi_macro_f1", "tp", "fp", "fn")
    ],
    "train_selected_threshold",
    "train_selected_suppression_radius",
    "train_candidates",
    "val_candidates",
    "train_predictions",
    "val_predictions",
    "train_max_candidate_score",
    "val_max_candidate_score",
    "optimizer_updates",
    "mean_gradient_norm",
    "ema_decay_last",
    "learning_rate",
    "seconds",
    "micro_batch",
]


def fit_one(
    cache_dir: Path,
    output: Path,
    *,
    arch: str,
    plan: str,
    fold: int,
    split_root: Path | None = None,
    pretrained: bool = True,
    epochs: int = 120,
    tile_size: int = 512,
    overlap: int = 64,
    micro_batch: int = 0,
    effective_batch: int = 16,
    workers: int = 2,
    validate_every: int = 1,
    seed: int = 42,
    lr: float = 0.00035,
    ema_decay: float = 0.9995,
    amp: bool = True,
    grad_checkpointing: bool = False,
    device_str: str = "auto",
    train_thresholds: list[float] | None = None,
    train_radii: list[float] | None = None,
) -> dict:
    if fold not in range(4) or epochs < 1 or validate_every < 1:
        raise ValueError("Invalid fold/epochs/validate_every")
    if (
        micro_batch < 0
        or effective_batch < 1
        or (micro_batch and (micro_batch > effective_batch or effective_batch % micro_batch))
    ):
        raise ValueError("Invalid micro/effective batch")
    if validate_every != 1:
        raise ValueError("Full train/val metrics every epoch requires --validate-every 1")
    if not math.isfinite(lr) or lr <= 0 or workers < 0:
        raise ValueError("Require a finite positive learning rate and nonnegative workers")
    if not math.isfinite(ema_decay) or not 0 <= ema_decay <= 1:
        raise ValueError("EMA decay must be finite in [0, 1]")
    from .models.build import ARCHITECTURES

    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {arch}")
    cache = Cache(cache_dir)
    tile_origins(cache.size, tile_size, overlap)
    if cache.meta.get("target_version") != "puma_one_feature_one_nucleus_geometry_centroid_v2":
        raise ValueError(
            "Stage 1 cache is old. Re-run Notebook 01 to prepare the Stage 1 cache before training."
        )
    splits = load_splits(plan, split_root)
    data = splits[fold]
    if set(data["train"]) & set(data["val"]):
        raise RuntimeError("Train/val ROI leakage")
    missing = (set(data["train"]) | set(data["val"])) - set(cache.name_to_index)
    if missing:
        raise ValueError(f"Missing ROIs in cache: {sorted(missing)[:4]}")
    cfg = default_pipeline_config()

    # A score threshold below the initial heatmap*quality prior is essential.
    # Candidate generation MUST not discard points the train-only sweep could use.
    threshold_values = sorted(
        {
            float(v)
            for v in (
                train_thresholds if train_thresholds is not None else cfg.stage1.train_thresholds
            )
        }
    )
    if not threshold_values or not all(math.isfinite(v) and 0 < v <= 1 for v in threshold_values):
        raise ValueError("Training thresholds must be finite in (0, 1]")
    candidate_floor = threshold_values[0]
    radius_values = list(
        cfg.stage1.train_suppression_radii_pixels if train_radii is None else train_radii
    )
    if not radius_values or not all(math.isfinite(v) and v > 0 for v in radius_values):
        raise ValueError("Training radii must be finite and positive")

    applied_checkpointing = bool(grad_checkpointing)
    opts = {
        "arch": arch,
        "plan": plan,
        "fold": fold,
        "cache_id": cache.meta["id"],
        "detector_revision": DETECTOR_REVISION,
        "pipeline_config": asdict(cfg),
        "train": sorted(data["train"]),
        "val": sorted(data["val"]),
        "epochs": epochs,
        "tile_size": tile_size,
        "overlap": overlap,
        "micro_batch": micro_batch,
        "effective_batch": effective_batch,
        "workers": workers,
        "seed": seed,
        "lr": lr,
        "ema_decay": ema_decay,
        "amp": amp,
        "pretrained": bool(pretrained),
        "grad_checkpointing": applied_checkpointing,
        "train_thresholds": threshold_values,
        "train_radii": radius_values,
    }
    run_id = fingerprint(opts)
    folder = Path(output) / f"plan_{plan}" / arch / f"fold_{fold}"
    if initialize(folder, epochs, run_id):
        return {"output": str(folder), "epochs": epochs, "skipped": True}
    seed_everything(seed + fold)
    device = torch.device(
        "cuda"
        if device_str == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_str == "auto"
        else device_str
    )
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    print(
        f"BACKBONE {arch}: pretrained ImageNet weights={pretrained}; "
        "this is initialization, NOT loading a trained PUMA checkpoint.",
        flush=True,
    )
    model = build_model(arch, pretrained=pretrained).to(device)
    if applied_checkpointing:
        model.set_grad_checkpointing(True)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.stage1.weight_decay)
    if micro_batch == 0:
        micro_batch = choose_batch_size(model, optimizer, device, tile_size, effective_batch, amp)
    train_data = TileDataset(cache, data["train"], tile_size, train=True)
    loader = DataLoader(
        train_data,
        batch_size=micro_batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
        persistent_workers=workers > 0,
        worker_init_fn=_worker_seed,
    )
    accumulation = effective_batch // micro_batch
    steps_per_epoch = math.ceil(len(loader) / accumulation)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup = cfg.stage1.warmup_epochs * steps_per_epoch
    min_ratio = cfg.stage1.minimum_learning_rate / lr

    def factor(s):
        if s < warmup:
            return max(min_ratio, (s + 1) / max(1, warmup))
        frac = min(1.0, (s - warmup) / max(1, total_steps - warmup))
        return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * frac))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    best_f1 = -1.0
    ema_updates = 0
    validation_batch = None
    val_dataset = TileDataset(cache, data["val"], tile_size, train=False)
    val_loader = None

    for epoch in range(epochs):
        tic = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, batches = 0.0, 0
        gradient_norm_sum = 0.0
        gradient_norm_count = 0
        last_ema_decay = 0.0
        for bi, batch in enumerate(loader):
            image = batch["image"].to(device, non_blocking=True)
            heatmap = batch["heatmap"].to(device, non_blocking=True)
            ramp = min(
                1.0,
                max(
                    0.0,
                    (epoch - cfg.stage1.hard_negative_start_epoch + 1)
                    / cfg.stage1.hard_negative_ramp_epochs,
                ),
            )
            weights = {
                "heatmap": cfg.stage1.heatmap_loss_weight,
                "offset": cfg.stage1.offset_loss_weight,
                "quality": cfg.stage1.quality_loss_weight,
                "uncertainty": cfg.stage1.uncertainty_loss_weight,
                "hard_negative": cfg.stage1.hard_negative_loss_weight * ramp,
            }
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
                predicted = model(image)
                loss, _ = stage1_loss(
                    predicted,
                    heatmap,
                    batch["points"],
                    weights=weights,
                    quality_target_sigma_grid_units=cfg.stage1.quality_target_sigma_grid_units,
                    hard_negative_topk_per_image=cfg.stage1.hard_negative_topk_per_image,
                    return_metrics=False,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"NaN/Inf at epoch {epoch + 1}, batch {bi}")
            group_start = bi // accumulation * accumulation
            group_size = min(accumulation, len(loader) - group_start)
            scaler.scale(loss / group_size).backward()
            total_loss += float(loss.detach().item())
            batches += 1
            if (bi + 1) % accumulation == 0 or bi + 1 == len(loader):
                gradient_norm = _optimizer_step(
                    model.parameters(), optimizer, scaler, cfg.stage1.gradient_clip_norm
                )
                if gradient_norm is not None:
                    ema_updates += 1
                    gradient_norm_sum += float(gradient_norm.item())
                    gradient_norm_count += 1

                    # Fixed decay=0.9995 would keep ~94% of RANDOM initial EMA
                    # after 117 updates. Warm it up so validation sees learning.
                    last_ema_decay = min(
                        float(ema_decay),
                        (1.0 + ema_updates) / (10.0 + ema_updates),
                    )
                    _ema_update(ema, model, last_ema_decay)
                    scheduler.step()
                else:
                    print(
                        f"AMP overflow at epoch {epoch + 1}, batch {bi}; "
                        f"skipped update, loss scale reduced to {scaler.get_scale():g}",
                        flush=True,
                    )
                optimizer.zero_grad(set_to_none=True)
        if gradient_norm_count == 0:
            raise RuntimeError(
                f"Stage 1 completed epoch {epoch + 1} without optimizer updates. "
                "Check AMP overflow, gradients and loss."
            )
        if validation_batch is None:
            validation_batch = choose_tile_batch_size(ema, device, tile_size, amp=amp)
        # Calibrate threshold and NMS strictly on training ROIs.
        # The validation set is NEVER passed to the operating-point sweep.
        train_candidates = {
            name: collect_candidates(
                ema,
                cache,
                name,
                tile=tile_size,
                overlap=overlap,
                device=device,
                amp=amp,
                batch_size=validation_batch,
                minimum_proposal_score=candidate_floor,
            )
            for name in data["train"]
        }
        sweep = sweep_cached(
            train_candidates,
            cache,
            data["train"],
            thresholds=threshold_values,
            radii=radius_values,
            min_recall=cfg.stage1.train_min_recall,
        )
        chosen = sweep["best"]

        # Compute validation predictions independently at the TRAIN-selected
        # operating point, with no second threshold/NMS search.
        val_candidates = {
            name: collect_candidates(
                ema,
                cache,
                name,
                tile=tile_size,
                overlap=overlap,
                device=device,
                amp=amp,
                batch_size=validation_batch,
                minimum_proposal_score=candidate_floor,
            )
            for name in data["val"]
        }
        # Validation loss is computed without augmentation or gradient updates.
        # Threshold and NMS selection have already been fixed using TRAIN ROIs only.
        if val_loader is None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=validation_batch,
                shuffle=False,
                num_workers=workers,
                pin_memory=device.type == "cuda",
                collate_fn=collate,
                persistent_workers=workers > 0,
            )

        val_loss_sum = 0.0
        val_loss_batches = 0
        ema.eval()
        with torch.no_grad():
            for val_batch in val_loader:
                val_images = val_batch["image"].to(device, non_blocking=True)
                val_heatmap = val_batch["heatmap"].to(device, non_blocking=True)

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=amp and device.type == "cuda",
                ):
                    val_outputs = ema(val_images)
                    loss, _ = stage1_loss(
                        val_outputs,
                        val_heatmap,
                        val_batch["points"],
                        weights=weights,
                        quality_target_sigma_grid_units=cfg.stage1.quality_target_sigma_grid_units,
                        hard_negative_topk_per_image=cfg.stage1.hard_negative_topk_per_image,
                        return_metrics=False,
                    )

                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Validation NaN/Inf at epoch {epoch + 1}")
                val_loss_sum += float(loss.item())
                val_loss_batches += 1

        val_loss = val_loss_sum / max(1, val_loss_batches)
        training = _measure_at(
            train_candidates,
            cache,
            data["train"],
            threshold=chosen["threshold"],
            radius=chosen["suppression_radius"],
        )
        valid = _measure_at(
            val_candidates,
            cache,
            data["val"],
            threshold=chosen["threshold"],
            radius=chosen["suppression_radius"],
        )

        def peak_score(candidate_sets):
            return max(
                (float(c["proposal_score"].max()) for c in candidate_sets.values() if len(c)),
                default=0.0,
            )

        row = {
            "epoch": epoch + 1,
            "run_id": run_id,
            "train_loss": total_loss / max(1, batches),
            "val_loss": val_loss,
            **{f"train_{k}": v for k, v in training.items()},
            **{f"val_{k}": v for k, v in valid.items()},
            "train_selected_threshold": chosen["threshold"],
            "train_selected_suppression_radius": chosen["suppression_radius"],
            "train_candidates": sum(len(c) for c in train_candidates.values()),
            "val_candidates": sum(len(c) for c in val_candidates.values()),
            "train_predictions": training["tp"] + training["fp"],
            "val_predictions": valid["tp"] + valid["fp"],
            "train_max_candidate_score": peak_score(train_candidates),
            "val_max_candidate_score": peak_score(val_candidates),
            "optimizer_updates": ema_updates,
            "mean_gradient_norm": gradient_norm_sum / gradient_norm_count,
            "ema_decay_last": last_ema_decay,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": round(time.monotonic() - tic, 2),
            "micro_batch": micro_batch,
        }
        if valid["f1"] > best_f1:
            best_f1 = valid["f1"]
            save_best(
                folder / "best.pt",
                {
                    "model": ema.state_dict(),
                    "schema": SCHEMA,
                    "run_id": run_id,
                    "config": opts,
                    "epoch": epoch + 1,
                    "operating_point": {**chosen, "selection_set": "train"},
                },
            )
        append_csv(folder / "history.csv", COLS, row)
        print(
            f"Stage1 {plan}/{arch}/fold{fold} epoch {epoch + 1}/{epochs} "
            f"loss train/val={row['train_loss']:.4f}/{val_loss:.4f} "
            f"updates={ema_updates} grad={row['mean_gradient_norm']:.4f} "
            f"candidates train/val={row['train_candidates']}/{row['val_candidates']} "
            f"predictions train/val={row['train_predictions']}/{row['val_predictions']} "
            f"train P/R/F1={training['precision']:.3f}/{training['recall']:.3f}/{training['f1']:.3f} "
            f"val P/R/F1={valid['precision']:.3f}/{valid['recall']:.3f}/{valid['f1']:.3f} "
            f"train-selected threshold={chosen['threshold']:.6g} NMS={chosen['suppression_radius']:.1f} "
            f"best={best_f1:.4f}",
            flush=True,
        )
        if not row["val_candidates"]:
            print(
                "WARNING: zero Stage 1 validation candidates; inspect "
                "val_max_candidate_score and thresholds.",
                flush=True,
            )
    return {"output": str(folder), "best_f1": best_f1, "epochs": epochs, "skipped": False}


def summarize(output: Path, plan: str, architectures=None, require_complete=False):
    architectures = architectures or ["convnext_tiny", "convnext_base"]
    rows = []
    for arch in architectures:
        folds = []
        for fold in range(4):
            path = Path(output) / f"plan_{plan}" / arch / f"fold_{fold}" / "history.csv"
            if not path.exists():
                if require_complete:
                    raise FileNotFoundError(path)
                continue
            with path.open(newline="") as f:
                records = list(csv.DictReader(f))
            if not records:
                continue
            best = max(records, key=lambda r: float(r["val_f1"]))
            row = {
                "model": arch,
                "fold": fold,
                **{
                    k: float(best["val_" + k])
                    for k in ("accuracy", "precision", "recall", "f1", "roi_macro_f1")
                },
            }
            rows.append(row)
            folds.append(row)
        if folds:
            for key in ("accuracy", "precision", "recall", "f1", "roi_macro_f1"):
                a = np.asarray([r[key] for r in folds])
                rows.append(
                    {
                        "model": arch,
                        "fold": f"MEAN±STD ({len(a)} folds)",
                        "metric": key,
                        "mean": float(a.mean()),
                        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                    }
                )
    return rows
