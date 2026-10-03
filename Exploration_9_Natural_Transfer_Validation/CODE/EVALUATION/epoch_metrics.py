from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from CODE.COMMON.constants import CLASSES, NUM_CLASSES
from CODE.EVALUATION.matcher import match_roi, prepare_roi_matcher, PreparedRoiMatcher
from CODE.EVALUATION.roi_metric import roi_averaged_fixed10_prf, roi_averaged_per_class
from CODE.EVALUATION.semantic import confusion_matrix, metrics_from_confusion


@dataclass(slots=True)
class EpochMetricContext:
    roi_ids: tuple[str, ...]
    indices: tuple[np.ndarray, ...]
    matchers: tuple[PreparedRoiMatcher, ...]
    n: int


def prepare_epoch_metric_context(y_true, rois, coords) -> EpochMetricContext:
    y_true = np.asarray(y_true, dtype=np.int64)
    rois = np.asarray(rois).astype(str)
    coords = np.asarray(coords, dtype=np.float64)
    if not (len(y_true) == len(rois) == len(coords)):
        raise ValueError("Epoch metric context arrays must have equal length")
    roi_ids = tuple(sorted(set(rois.tolist())))
    indices = tuple(np.flatnonzero(rois == roi) for roi in roi_ids)
    matchers = tuple(prepare_roi_matcher(coords[ix], y_true[ix]) for ix in indices)
    return EpochMetricContext(roi_ids=roi_ids, indices=indices, matchers=matchers, n=len(y_true))


def _accuracy(cm: np.ndarray) -> float:
    cm = np.asarray(cm, dtype=np.int64)
    total = int(cm.sum())
    return float(np.trace(cm) / total) if total else 0.0


def _counts_by_roi(y_true, y_pred, scores, rois, coords, context: EpochMetricContext | None = None):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    rois = np.asarray(rois)
    coords = np.asarray(coords, dtype=np.float64)
    if not (len(y_true) == len(y_pred) == len(scores) == len(rois) == len(coords)):
        raise ValueError("Epoch metric arrays must have equal length")

    if context is not None:
        if context.n != len(y_true):
            raise ValueError("Epoch metric context length mismatch")
        return {
            roi: matcher.match(y_pred[ix], scores[ix])
            for roi, ix, matcher in zip(context.roi_ids, context.indices, context.matchers)
        }

    counts = {}
    roi_s = rois.astype(str)
    for roi in sorted(set(roi_s.tolist())):
        ix = np.flatnonzero(roi_s == roi)
        counts[roi] = match_roi(coords[ix], y_true[ix], coords[ix], y_pred[ix], scores[ix])
    return counts


def classification_epoch_metrics(
    *,
    y_true,
    probs,
    rois,
    coords,
    loss: float | None = None,
    context: EpochMetricContext | None = None,
) -> dict:
    """Complete natural-prevalence diagnostics for one train/held split at one epoch."""
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != NUM_CLASSES or len(probs) != len(y_true):
        raise ValueError(f"Expected probabilities [N,{NUM_CLASSES}], got {probs.shape}")
    if not np.isfinite(probs).all():
        raise ValueError("Non-finite epoch probabilities")

    pred = probs.argmax(axis=1)
    score = probs.max(axis=1)
    cm = confusion_matrix(y_true, pred)
    sem = metrics_from_confusion(cm)
    counts = _counts_by_roi(y_true, pred, score, rois, coords, context=context)
    roi_macro = roi_averaged_fixed10_prf(counts)
    roi_class = roi_averaged_per_class(counts)

    per_class = {}
    for i in range(NUM_CLASSES):
        per_class[CLASSES[i]] = {
            "precision": float(sem["precision"][i]),
            "recall": float(sem["recall"][i]),
            "f1": float(sem["f1"][i]),
            "support": int(sem["support"][i]),
            "predicted": int(sem["predicted"][i]),
            "tp": int(sem["tp"][i]),
            "fp": int(sem["fp"][i]),
            "fn": int(sem["fn"][i]),
            "roi_precision": float(roi_class["precision"][i]),
            "roi_recall": float(roi_class["recall"][i]),
            "roi_f1": float(roi_class["f1"][i]),
            "positive_roi_precision": float(roi_class["positive_roi_precision"][i]),
            "positive_roi_recall": float(roi_class["positive_roi_recall"][i]),
            "positive_roi_f1": float(roi_class["positive_roi_f1"][i]),
            "positive_roi_count": int(roi_class["positive_roi_count"][i]),
            "predicted_roi_count": int(roi_class["predicted_roi_count"][i]),
        }

    result = {
        "roi_precision": float(roi_macro["precision"]),
        "roi_recall": float(roi_macro["recall"]),
        "roi_f1": float(roi_macro["f1"]),
        "f1": float(sem["macro_f1"]),
        "macro_f1": float(sem["macro_f1"]),
        "precision": float(np.mean(sem["precision"])),
        "macro_precision": float(np.mean(sem["precision"])),
        "recall": float(np.mean(sem["recall"])),
        "macro_recall": float(np.mean(sem["recall"])),
        "balanced_accuracy_fixed10": float(np.mean(sem["recall"])),
        "accuracy": _accuracy(cm),
        "per_class": per_class,
        "confusion": cm.tolist(),
        "n": int(len(y_true)),
        "roi_count": int(len(counts)),
    }
    if loss is not None:
        result["loss"] = float(loss)
    return result
