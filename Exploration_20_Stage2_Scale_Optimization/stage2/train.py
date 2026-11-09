"""Stage 2 classification: five epochs, one best checkpoint and one history CSV.

Every epoch records complete train/validation metrics for ten nucleus classes.
Incomplete or incompatible runs restart from epoch one; optimizer state is
never resumed from metrics or a best-weight checkpoint.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common.checkpoint import save_best
from common.run_history import SCHEMA, append_csv, fingerprint, initialize, file_fingerprint
from .splits import load_splits

from .data import class_names, normalize_tier, read_features, split_rows
from .graph import gpu_bundle, validate_graph
from .metrics import classification_scores, roi_scores
from .model import NucleiFusion
from .settings import FEATURE_REVISION, MODEL_REVISIONS, FOVS, VARIANTS, graph_settings


BASE = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "macro_f1",
    "macro_precision",
    "macro_recall",
)

PER_CLASS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "support",
    "tp",
    "fp",
    "fn",
)

COLS = [
    "epoch",
    "run_id",
    "train_loss",
    "val_loss",
    *[f"{split}_{metric}" for split in ("train", "val") for metric in BASE],
    *[
        f"{split}_{name}_{metric}"
        for split in ("train", "val")
        for name in class_names
        for metric in PER_CLASS
    ],
    "seconds",
]


def _evaluate(model, tensors, labels, indices, batch_size=2048, return_predictions=False):
    model.eval()
    predicted = []
    loss_sum = 0.0

    with torch.inference_mode():
        for left in range(0, len(indices), batch_size):
            ids = torch.as_tensor(
                indices[left : left + batch_size],
                device=labels.device,
                dtype=torch.long,
            )
            logits = model(ids, tensors).float()
            predicted.append(logits.argmax(1).cpu().numpy())
            loss_sum += float(
                F.cross_entropy(
                    logits,
                    labels[ids],
                    reduction="sum",
                ).item()
            )

    truth = labels.detach().cpu().numpy()[indices]
    classes = np.concatenate(predicted) if predicted else np.empty(0, dtype=int)

    metrics, loss = classification_scores(truth, classes), loss_sum / max(1, len(indices))
    return (metrics, loss, classes) if return_predictions else (metrics, loss)


def _flatten_metrics(split, metrics):
    row = {f"{split}_{metric}": metrics[metric] for metric in BASE}

    for name in class_names:
        row.update(
            {f"{split}_{name}_{metric}": metrics["per_class"][name][metric] for metric in PER_CLASS}
        )

    return row


def _choose_batch_size(model, tensors, labels, train_ids, optimizer, device):
    """Probe GPU memory without modifying the optimizer or model parameters."""
    candidates = sorted(
        {
            min(size, len(train_ids))
            for size in (8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1)
        },
        reverse=True,
    )
    for candidate in candidates:
        ids = logits = residual = loss = reserve = None
        try:
            ids = torch.as_tensor(
                train_ids[:candidate],
                device=device,
                dtype=torch.long,
            )
            logits, residual = model(ids, tensors, return_delta=True)
            loss = F.cross_entropy(logits.float(), labels[ids])
            loss = loss + 0.001 * residual.float().square().mean()
            loss.backward()
            model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            reserve = torch.empty(model_bytes * 2, dtype=torch.uint8, device=device)
            return candidate

        except torch.OutOfMemoryError:
            if device.type != "cuda":
                raise

        finally:
            optimizer.zero_grad(set_to_none=True)
            del ids, logits, residual, loss, reserve
            if device.type == "cuda":
                torch.cuda.empty_cache()

    raise RuntimeError("Stage 2 cannot fit one training sample on this device")


def _best_roi_validation(feature_dir, fold_dir, plan, fold, mode, run_id,
                         values=None, tensors=None, labels=None, val_ids=None):
    """Evaluate the selected best checkpoint at GT coordinates, without Stage 1 proposals."""
    destination = fold_dir / "validation_roi.json"
    if destination.is_file():
        try:
            previous = json.loads(destination.read_text())
            if (
                previous.get("run_id") == run_id
                and previous.get("mode") == mode
                and previous.get("plan") == plan
                and previous.get("fold") == fold
                and previous.get("checkpoint_sha256") == file_fingerprint(fold_dir / "best.pt")
                and isinstance(previous.get("per_roi_f1"), dict)
            ):
                return previous
        except (ValueError, OSError, TypeError, KeyError):
            pass

    checkpoint = torch.load(fold_dir / "best.pt", map_location="cpu", weights_only=False)
    if checkpoint.get("run_id") != run_id or checkpoint.get("graph_mode") != mode:
        raise ValueError("Best checkpoint does not match expected graph mode/run ID")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if values is None:
        values = read_features(feature_dir)
    if val_ids is None:
        val_ids = split_rows(values, load_splits(plan)[fold]["val"])
    if tensors is None:
        tier = (np.asarray(values["tier_raw"]) - checkpoint["tier_mean"]) / checkpoint["tier_std"]
        arrays = {**values, "tier": tier.astype(np.float32)}
        with np.load(feature_dir / "graph.npz", allow_pickle=False) as graph_file:
            graph = {key: graph_file[key] for key in ("neighbors", "weights", "distance")}
        tensors = gpu_bundle(arrays, graph, device, mode=mode)
    if labels is None:
        labels = torch.as_tensor(values["label"].copy(), device=device, dtype=torch.long)

    model = NucleiFusion(mode=mode).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    _, _, predictions = _evaluate(model, tensors, labels, val_ids, return_predictions=True)
    names = np.asarray(values["roi"][val_ids])
    coordinates = np.asarray(values["xy"][val_ids])
    truth = np.asarray(values["label"][val_ids])
    scores = roi_scores(
        {"roi": names, "xy": coordinates, "label": truth},
        {"roi": names, "xy": coordinates, "label": predictions},
        roi_names=sorted(load_splits(plan)[fold]["val"]),
    )
    result = {
        "exploration": 20,
        "evaluation": "GT-centroid Stage 2 validation; NOT detector end-to-end",
        "metric_primary": "roi_f1",
        "mode": mode,
        "plan": plan,
        "fold": fold,
        "graph_mode": mode,
        "run_id": run_id,
        "checkpoint_sha256": file_fingerprint(fold_dir / "best.pt"),
        "best_epoch": int(checkpoint["epoch"]),
        "nuclei": int(len(val_ids)),
        "num_rois": int(scores["num_rois"]),
        "roi_f1": float(scores["roi_f1"]),
        "macro_f1_nuclei": float(scores["macro_f1_nuclei"]),
        "per_roi_f1": {roi: float(row["f1"]) for roi, row in scores["per_roi"].items()},
        "per_roi_class_f1": {
            roi: {name: float(score["f1"]) for name, score in by_class.items()}
            for roi, by_class in scores["per_roi_class"].items()
        },
        "observed_nuclei_classes": scores["observed_nuclei_classes"],
    }
    pending = destination.with_name(destination.name + ".pending")
    try:
        pending.write_text(json.dumps(result, indent=2))
        pending.replace(destination)
    finally:
        pending.unlink(missing_ok=True)
    print(
        f"E20 {mode} {plan}/fold{fold}: best_epoch={result['best_epoch']} "
        f"GT-centroid ROI_F1={result['roi_f1']:.5f} "
        f"macro_f1_nuclei={result['macro_f1_nuclei']:.5f}",
        flush=True,
    )
    return result


def train_one(feature_dir, output_dir, plan, fold, epochs=5, batch_size=0, seed=42, mode="G3"):
    """Train a single fold from epoch 1 unless a matching full history exists."""
    if mode not in VARIANTS:
        raise ValueError(f"Unknown Stage 2 graph mode: {mode}")
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if batch_size < 0:
        raise ValueError("batch_size must be >= 0")

    feature_dir = Path(feature_dir)
    fold_dir = Path(output_dir) / "stage2" / f"r224_{mode.lower()}" / f"plan_{plan}" / f"fold_{fold}"
    manifest = json.loads((feature_dir / "manifest.json").read_text())

    if (
        manifest.get("centroid") != "polygon_geometry_centroid"
        or manifest.get("feature_revision") != FEATURE_REVISION
        or tuple(manifest.get("class_order", ())) != class_names
        or tuple(manifest.get("fovs", ())) != FOVS
    ):
        raise ValueError("Stage 2 cache has an outdated centroid manifest; rerun Notebook 01")

    if not manifest.get("source_id"):
        raise ValueError("Stage 2 cache lacks input provenance; run Notebook 01 first")
    if manifest.get("graph_settings") != graph_settings():
        raise ValueError("Stage 2 graph must be rebuilt at radius 224; rerun Notebook 01")
    if manifest.get("graph_sha256") != file_fingerprint(feature_dir / "graph.npz"):
        raise ValueError("Stage 2 graph content differs from manifest; rerun Notebook 01")

    split = load_splits(plan)[fold]
    options = {
        "plan": plan,
        "fold": fold,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "manifest": manifest,
        "model_revision": MODEL_REVISIONS[mode],
        "graph_settings": graph_settings(),
        "cache_files": {
            path.name: file_fingerprint(path) for path in sorted(feature_dir.glob("*.npy"))
        },
        "graph": file_fingerprint(feature_dir / "graph.npz"),
        "train_rois": sorted(split["train"]),
        "val_rois": sorted(split["val"]),
    }
    run_id = fingerprint(options)

    random.seed(seed + fold)
    np.random.seed(seed + fold)
    torch.manual_seed(seed + fold)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    values = read_features(feature_dir)
    train_ids = split_rows(values, split["train"])
    val_ids = split_rows(values, split["val"])

    if len(train_ids) == 0 or len(val_ids) == 0:
        raise ValueError("Empty train or validation split")
    if set(values["roi"][train_ids]) & set(values["roi"][val_ids]):
        raise ValueError("ROI leakage between train and validation")
    if (
        set(manifest.get("names", ())) != set(split["train"]) | set(split["val"])
        or not set(values["roi"]).issubset(set(manifest.get("names", ())))
        or manifest.get("rois") != len(manifest.get("names", ()))
        or manifest.get("nuclei") != len(values["xy"])
    ):
        raise ValueError("Feature cache and fold ROI list do not match")

    mean, std = normalize_tier(values["tier_raw"], train_ids)
    tier = (np.asarray(values["tier_raw"]) - mean) / std
    arrays = {**values, "tier": tier.astype(np.float32)}

    with np.load(feature_dir / "graph.npz") as graph_file:
        graph = {key: graph_file[key] for key in ("neighbors", "weights", "distance")}

    validate_graph(graph, len(values["xy"]), values["roi"])
    if graph["neighbors"].shape[1] != graph_settings()["k"]:
        raise ValueError("Stage 2 graph neighbor count does not match settings")
    if initialize(fold_dir, epochs, run_id):
        _best_roi_validation(feature_dir, fold_dir, plan, fold, mode, run_id)
        return fold_dir

    tensors = gpu_bundle(arrays, graph, device, mode=mode)
    labels = torch.as_tensor(values["label"].copy(), device=device, dtype=torch.long)

    model = NucleiFusion(mode=mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.001)

    if batch_size == 0:
        batch_size = _choose_batch_size(
            model,
            tensors,
            labels,
            train_ids,
            optimizer,
            device,
        )

    best_macro_f1 = -1.0

    for epoch in range(epochs):
        tic = time.monotonic()
        model.train()
        order = np.random.default_rng(seed + fold + epoch).permutation(train_ids)

        for left in range(0, len(order), batch_size):
            ids = torch.as_tensor(
                order[left : left + batch_size],
                device=device,
                dtype=torch.long,
            )
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
            ):
                logits, residual = model(ids, tensors, return_delta=True)
                loss = F.cross_entropy(logits.float(), labels[ids])
                loss = loss + 0.001 * residual.float().square().mean()

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Stage 2 training NaN/Inf at epoch {epoch + 1}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()

        train_metrics, train_loss = _evaluate(
            model, tensors, labels, train_ids, min(batch_size, 2048)
        )
        val_metrics, val_loss = _evaluate(model, tensors, labels, val_ids, min(batch_size, 2048))

        row = {
            "epoch": epoch + 1,
            "run_id": run_id,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **_flatten_metrics("train", train_metrics),
            **_flatten_metrics("val", val_metrics),
            "seconds": round(time.monotonic() - tic, 2),
        }

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            save_best(
                fold_dir / "best.pt",
                {
                    "model": model.state_dict(),
                    "schema": SCHEMA,
                    "feature_source_id": manifest.get("source_id"),
                    "encoder_sources": manifest.get("encoder_sources"),
                    "image_sources": manifest.get("image_sources"),
                    "train_rois": options["train_rois"],
                    "val_rois": options["val_rois"],
                    "epoch": epoch + 1,
                    "run_id": run_id,
                    "training_epochs": epochs,
                    "plan": plan,
                    "fold": fold,
                    "tier_mean": mean,
                    "tier_std": std,
                    "class_names": class_names,
                    "model_revision": MODEL_REVISIONS[mode],
                    "feature_revision": FEATURE_REVISION,
                    "graph_mode": mode,
                    "graph_settings": graph_settings(),
                },
            )

        append_csv(fold_dir / "history.csv", COLS, row)

        print(
            f"Exploration 20 {mode} {plan}/fold{fold} epoch {epoch + 1}/{epochs} "
            f"train_macro_f1={train_metrics['macro_f1']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"best={best_macro_f1:.4f}",
            flush=True,
        )

    _best_roi_validation(feature_dir, fold_dir, plan, fold, mode, run_id,
                         values=values, tensors=tensors, labels=labels, val_ids=val_ids)
    return fold_dir
