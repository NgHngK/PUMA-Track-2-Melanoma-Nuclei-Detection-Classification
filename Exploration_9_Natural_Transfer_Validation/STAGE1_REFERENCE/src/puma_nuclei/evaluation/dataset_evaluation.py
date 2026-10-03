from __future__ import annotations

from typing import Any

import numpy as np

from ..constants import EVALUATION_CONTRACT_ID, NUCLEUS_CLASSES
from .aggregators import aggregate_fixed10, aggregate_public_dynamic, aggregate_summed
from .public_matcher import PublicMatcher, RoiMatchResult


def ground_truth_features(rows: np.ndarray) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        class_id = int(row["class_id"])
        output.append(
            {
                "uid": f"gt:{int(row['nucleus_index']) if 'nucleus_index' in rows.dtype.names else len(output)}",
                "category": NUCLEUS_CLASSES[class_id],
                "centroid": [float(row["x"]), float(row["y"])],
                "score": 1.0,
            }
        )
    return output


def prediction_features(rows: np.ndarray) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    names = set(rows.dtype.names or ())
    for index, row in enumerate(rows):
        class_id = int(row["class_id"] if "class_id" in names else row["predicted_class_id"])
        score = float(row["score"]) if "score" in names else 1.0
        uid = str(row["proposal_uid"]) if "proposal_uid" in names else f"pred:{index}"
        output.append(
            {
                "uid": uid,
                "category": NUCLEUS_CLASSES[class_id],
                "centroid": [float(row["x"]), float(row["y"])],
                "score": score,
            }
        )
    return output


def evaluate_roi_predictions(
    gt_rows: np.ndarray,
    pred_rows: np.ndarray,
    *,
    trace: bool = False,
    matcher: PublicMatcher | None = None,
) -> RoiMatchResult:
    matcher = matcher or PublicMatcher()
    return matcher.match(ground_truth_features(gt_rows), prediction_features(pred_rows), trace=trace)


def evaluate_dataset_predictions(
    gt_by_roi: dict[int, np.ndarray],
    pred_by_roi: dict[int, np.ndarray],
    roi_indices: list[int] | np.ndarray,
    *,
    trace: bool = False,
) -> dict[str, Any]:
    matcher = PublicMatcher()
    requested_rois = [int(value) for value in roi_indices]
    if len(set(requested_rois)) != len(requested_rois):
        raise ValueError("roi_indices contains duplicates; each ROI must contribute exactly once")
    roi_results: list[dict[str, Any]] = []
    traces: dict[int, list[Any]] = {}
    for roi_index in requested_rois:
        gt = gt_by_roi.get(roi_index, np.empty(0, dtype=[("x", "f4"), ("y", "f4"), ("class_id", "i2")]))
        pred = pred_by_roi.get(roi_index, np.empty(0, dtype=[("x", "f4"), ("y", "f4"), ("class_id", "i2")]))
        result = evaluate_roi_predictions(gt, pred, trace=trace, matcher=matcher)
        roi_results.append(result.metrics)
        if trace:
            traces[roi_index] = result.trace
    return {
        "evaluation_contract_id": EVALUATION_CONTRACT_ID,
        "public_dynamic": aggregate_public_dynamic(roi_results),
        "fixed10": aggregate_fixed10(roi_results),
        "summed": aggregate_summed(roi_results),
        "roi_metrics": roi_results,
        "traces": traces,
    }

