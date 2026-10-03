from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ..constants import NUCLEUS_CLASSES


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "TP": int(tp), "FP": int(fp), "FN": int(fn),
        "precision": float(precision), "recall": float(recall), "f1_score": float(f1),
    }


def _average_roi_metrics(rows: list[dict[str, Any]], class_universe: list[str]) -> dict[str, Any]:
    n = len(rows)
    precision_by_class: dict[str, float] = {}
    recall_by_class: dict[str, float] = {}
    f1_by_class: dict[str, float] = {}
    for category in class_universe:
        precision_by_class[category] = (
            sum(float(metrics.get(category, {}).get("precision", 0.0)) for metrics in rows) / n if n else 0.0
        )
        recall_by_class[category] = (
            sum(float(metrics.get(category, {}).get("recall", 0.0)) for metrics in rows) / n if n else 0.0
        )
        f1_by_class[category] = (
            sum(float(metrics.get(category, {}).get("f1_score", 0.0)) for metrics in rows) / n if n else 0.0
        )
    if not n or not class_universe:
        macro_precision = macro_recall = macro_f1 = float("nan")
    else:
        macro_precision = float(np.mean([precision_by_class[name] for name in class_universe]))
        macro_recall = float(np.mean([recall_by_class[name] for name in class_universe]))
        macro_f1 = float(np.mean([f1_by_class[name] for name in class_universe]))
    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "precision_by_class": precision_by_class,
        "recall_by_class": recall_by_class,
        "f1_by_class": f1_by_class,
        "class_universe": list(class_universe),
    }


def aggregate_public_dynamic(roi_metrics: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(roi_metrics)
    class_universe: list[str] = []
    seen: set[str] = set()
    for metrics in rows:
        for category in metrics:
            if category in {"micro", "macro"} or category in seen:
                continue
            seen.add(category)
            class_universe.append(str(category))
    return _average_roi_metrics(rows, class_universe)


def aggregate_fixed10(roi_metrics: Iterable[dict[str, Any]]) -> dict[str, Any]:
    # This mirrors the PUMA challenge's ROI-level class-F1 aggregation, extended
    # in v16.4 to persist the identically aggregated class precision and recall.
    return _average_roi_metrics(list(roi_metrics), list(NUCLEUS_CLASSES))


def aggregate_summed(roi_metrics: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(roi_metrics)
    class_metrics: dict[str, dict[str, float | int]] = {}
    for category in NUCLEUS_CLASSES:
        tp = sum(int(metrics.get(category, {}).get("TP", 0)) for metrics in rows)
        fp = sum(int(metrics.get(category, {}).get("FP", 0)) for metrics in rows)
        fn = sum(int(metrics.get(category, {}).get("FN", 0)) for metrics in rows)
        class_metrics[category] = _prf(tp, fp, fn)
    macro_precision = float(np.mean([class_metrics[name]["precision"] for name in NUCLEUS_CLASSES])) if rows else float("nan")
    macro_recall = float(np.mean([class_metrics[name]["recall"] for name in NUCLEUS_CLASSES])) if rows else float("nan")
    macro = float(np.mean([class_metrics[name]["f1_score"] for name in NUCLEUS_CLASSES])) if rows else float("nan")
    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro,
        "precision_by_class": {name: float(class_metrics[name]["precision"]) for name in NUCLEUS_CLASSES},
        "recall_by_class": {name: float(class_metrics[name]["recall"]) for name in NUCLEUS_CLASSES},
        "f1_by_class": {name: float(class_metrics[name]["f1_score"]) for name in NUCLEUS_CLASSES},
        "class_metrics": class_metrics,
        "class_universe": list(NUCLEUS_CLASSES),
    }
