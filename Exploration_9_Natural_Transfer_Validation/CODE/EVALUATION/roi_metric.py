from __future__ import annotations

import numpy as np

from CODE.COMMON.constants import NUM_CLASSES


def _safe_div(num, den):
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    return np.divide(num, den, out=np.zeros_like(num, dtype=np.float64), where=den > 0)


def f1_from_counts(tp, fp, fn):
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)
    return _safe_div(2.0 * tp, 2.0 * tp + fp + fn)


def precision_from_counts(tp, fp):
    return _safe_div(tp, np.asarray(tp, dtype=np.float64) + np.asarray(fp, dtype=np.float64))


def recall_from_counts(tp, fn):
    return _safe_div(tp, np.asarray(tp, dtype=np.float64) + np.asarray(fn, dtype=np.float64))


def _stack_roi_counts(counts_by_roi):
    if not counts_by_roi:
        raise ValueError("No ROI counts")
    values = list(counts_by_roi.values())
    tp = np.stack([np.asarray(v[0], dtype=np.float64) for v in values], axis=0)
    fp = np.stack([np.asarray(v[1], dtype=np.float64) for v in values], axis=0)
    fn = np.stack([np.asarray(v[2], dtype=np.float64) for v in values], axis=0)
    if tp.ndim != 2 or tp.shape[1] != NUM_CLASSES or fp.shape != tp.shape or fn.shape != tp.shape:
        raise ValueError(f"Expected ROI count arrays [R,{NUM_CLASSES}]")
    return tp, fp, fn


def roi_averaged_per_class(counts_by_roi):
    """Per-class metrics averaged across ROI blocks.

    `precision`, `recall`, and `f1` average over every ROI, using 0 when the
    corresponding denominator is zero. This exactly matches the fixed-ten policy
    used by the overall E9 ROI metric.

    `positive_roi_*` averages only ROIs that contain GT support for that class.
    These diagnostics are especially useful for rare classes and are not used for
    model selection.
    """
    tp, fp, fn = _stack_roi_counts(counts_by_roi)
    p = precision_from_counts(tp, fp)
    r = recall_from_counts(tp, fn)
    f1 = f1_from_counts(tp, fp, fn)
    gt_positive = (tp + fn) > 0
    pred_positive = (tp + fp) > 0

    def positive_mean(values):
        out = np.zeros(NUM_CLASSES, dtype=np.float64)
        for c in range(NUM_CLASSES):
            mask = gt_positive[:, c]
            out[c] = float(values[mask, c].mean()) if np.any(mask) else 0.0
        return out

    return {
        "precision": p.mean(axis=0),
        "recall": r.mean(axis=0),
        "f1": f1.mean(axis=0),
        "positive_roi_precision": positive_mean(p),
        "positive_roi_recall": positive_mean(r),
        "positive_roi_f1": positive_mean(f1),
        "positive_roi_count": gt_positive.sum(axis=0).astype(np.int64),
        "predicted_roi_count": pred_positive.sum(axis=0).astype(np.int64),
    }


def roi_averaged_fixed10_prf(counts_by_roi):
    per_class = roi_averaged_per_class(counts_by_roi)
    return {
        "precision": float(np.mean(per_class["precision"])),
        "recall": float(np.mean(per_class["recall"])),
        "f1": float(np.mean(per_class["f1"])),
    }


def roi_averaged_fixed10(counts_by_roi):
    return roi_averaged_fixed10_prf(counts_by_roi)["f1"]
