from __future__ import annotations

import csv
import json
from pathlib import Path

from CODE.COMMON.constants import CLASSES


def save_history(path, history):
    Path(path).write_text(json.dumps(history, indent=2), encoding="utf-8")


def _flat_epoch_row(row: dict) -> dict:
    out = {
        "epoch": row.get("epoch"),
        "sampled_train_loss": row.get("sampled_train_loss", row.get("loss")),
        "grad_norm": row.get("grad_norm"),
        "lr": row.get("lr"),
        "train_seconds": row.get("train_seconds"),
        "execution_mode": row.get("execution_mode"),
    }
    split_keys = (
        "loss",
        "roi_precision",
        "roi_recall",
        "roi_f1",
        "f1",
        "precision",
        "accuracy",
        "balanced_accuracy_fixed10",
        "recall",
        "n",
        "roi_count",
    )
    class_keys = (
        "precision",
        "recall",
        "f1",
        "support",
        "predicted",
        "tp",
        "fp",
        "fn",
        "roi_precision",
        "roi_recall",
        "roi_f1",
        "positive_roi_precision",
        "positive_roi_recall",
        "positive_roi_f1",
        "positive_roi_count",
        "predicted_roi_count",
    )
    for split in ("train", "val"):
        metrics = row.get(split)
        if not metrics:
            continue
        for key in split_keys:
            out[f"{split}_{key}"] = metrics.get(key)
        for cls in CLASSES:
            class_metrics = metrics.get("per_class", {}).get(cls, {})
            for key in class_keys:
                out[f"{split}_{cls}_{key}"] = class_metrics.get(key)
    return out


def save_fit_history(directory, history):
    """Save complete nested JSON/JSONL plus analysis-friendly flattened CSV."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    with (directory / "history.jsonl").open("w", encoding="utf-8") as handle:
        for row in history:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    flat = [_flat_epoch_row(row) for row in history]
    fields = []
    for row in flat:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (directory / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
