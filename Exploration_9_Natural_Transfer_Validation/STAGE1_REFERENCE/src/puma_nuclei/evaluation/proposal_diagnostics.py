from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist



def lexicographic_one_to_one_assignment(
    gt_xy: np.ndarray,
    proposal_xy: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt_xy = np.asarray(gt_xy, dtype=np.float64).reshape(-1, 2)
    proposal_xy = np.asarray(proposal_xy, dtype=np.float64).reshape(-1, 2)
    if not len(gt_xy) or not len(proposal_xy):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    distances = cdist(gt_xy, proposal_xy)
    n_gt, n_pred = distances.shape
    dummy_cost = 1.0e6
    invalid_cost = 1.0e9
    cost = np.full((n_gt, n_pred + n_gt), dummy_cost, dtype=np.float64)
    valid = distances < float(radius)
    cost[:, :n_pred] = np.where(valid, distances, invalid_cost)
    row_ind, col_ind = linear_sum_assignment(cost)
    keep = col_ind < n_pred
    row_ind = row_ind[keep]
    col_ind = col_ind[keep]
    keep_valid = distances[row_ind, col_ind] < float(radius)
    row_ind = row_ind[keep_valid]
    col_ind = col_ind[keep_valid]
    return row_ind.astype(np.int64), col_ind.astype(np.int64), distances[row_ind, col_ind].astype(np.float64)


def binary_detection_metrics(
    gt_by_roi: dict[int, np.ndarray],
    pred_by_roi: dict[int, np.ndarray],
    roi_indices: list[int] | np.ndarray,
    *,
    radius: float,
) -> dict[str, float | int | None]:
    """Class-agnostic one-to-one point detection metrics across ROIs."""
    tp = fp = fn = 0
    matched_distances: list[np.ndarray] = []
    requested = [int(value) for value in roi_indices]
    if len(set(requested)) != len(requested):
        raise ValueError("roi_indices contains duplicates")
    for roi in requested:
        gt = np.asarray(gt_by_roi.get(roi, np.empty(0)))
        pred = np.asarray(pred_by_roi.get(roi, np.empty(0)))
        gt_xy = (
            np.column_stack((gt["x"], gt["y"])).astype(np.float64)
            if len(gt) else np.empty((0, 2), dtype=np.float64)
        )
        pred_xy = (
            np.column_stack((pred["x"], pred["y"])).astype(np.float64)
            if len(pred) else np.empty((0, 2), dtype=np.float64)
        )
        _, _, distances = lexicographic_one_to_one_assignment(gt_xy, pred_xy, radius)
        matched = int(len(distances))
        tp += matched
        fp += int(len(pred_xy)) - matched
        fn += int(len(gt_xy)) - matched
        if matched:
            matched_distances.append(distances)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    # Point detection has no meaningful true-negative count.  We therefore use
    # the standard set/detection accuracy TP / (TP + FP + FN), equivalent to
    # the Jaccard/critical-success index for the matched detection set.
    accuracy = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    all_distances = np.concatenate(matched_distances) if matched_distances else np.empty(0, dtype=np.float64)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "predictions": int(tp + fp),
        "ground_truth": int(tp + fn),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_match_distance": float(all_distances.mean()) if len(all_distances) else None,
    }

