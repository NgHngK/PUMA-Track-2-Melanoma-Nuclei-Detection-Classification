from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..constants import PUBLIC_MATCH_RADIUS_PIXELS


@dataclass(slots=True)
class MatchEvent:
    gt_uid: str
    gt_category: str
    selected_uid: str
    removed_uid: str
    selected_uid_seen_before: bool
    centroid_collision: bool
    score: float
    distance: float
    eligible_uids: tuple[str, ...]


@dataclass(slots=True)
class RoiMatchResult:
    metrics: dict[str, Any]
    matches: list[dict[str, Any]]
    trace: list[MatchEvent]


def calculate_public_centroid(path_points: list[list[float]] | np.ndarray) -> np.ndarray:
    points = np.asarray(path_points, dtype=np.float64)
    if points.ndim != 2 or len(points) < 3:
        raise ValueError("A valid public-evaluator polygon needs at least three path points")
    return np.mean(points[:, :2], axis=0)


def feature_from_polygon(polygon: dict[str, Any], *, filename: str = "", uid: str = "") -> dict[str, Any] | None:
    path_points = polygon["path_points"]
    if len(path_points) < 3:
        return None
    exterior_coords = [coord[:2] for coord in path_points]
    centroid = calculate_public_centroid(exterior_coords)
    return {
        "filename": filename,
        "category": polygon["name"],
        "centroid": centroid.tolist(),
        "score": polygon.get("score", 1),
        "uid": uid,
    }


def features_from_multiple_polygons(payload: dict[str, Any], *, filename: str = "") -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    for index, polygon in enumerate(payload.get("polygons", [])):
        feature = feature_from_polygon(polygon, filename=filename, uid=str(polygon.get("uid", index)))
        if feature is not None:
            features.append(feature)
    return features


def _classification_metrics(
    matches: list[dict[str, Any]],
    gt_features: list[dict[str, Any]],
    pred_features: list[dict[str, Any]],
) -> dict[str, Any]:
    pred_tp = [match["pred_category"] for match in matches]
    ground_truth = [feature["category"] for feature in gt_features]
    pred_all = [feature["category"] for feature in pred_features]
    gt_dict = dict(zip(*np.unique(ground_truth, return_counts=True), strict=True)) if ground_truth else {}
    pred_dict = dict(zip(*np.unique(pred_all, return_counts=True), strict=True)) if pred_all else {}
    tp_dict = dict(zip(*np.unique(pred_tp, return_counts=True), strict=True)) if pred_tp else {}
    categories = np.unique(list(gt_dict.keys()) + list(pred_dict.keys())) if (gt_dict or pred_dict) else []
    micro_tp = micro_fp = micro_fn = 0
    results_metrics: dict[str, Any] = {}
    for category in categories:
        tp = int(tp_dict.get(category, 0))
        fp = int(pred_dict.get(category, 0)) - tp
        fn = int(gt_dict.get(category, 0)) - tp
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        results_metrics[str(category)] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }
    micro_precision = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp > 0 else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn > 0 else 0.0
    micro_f1 = (
        2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall > 0
        else 0.0
    )
    class_values = [value["f1_score"] for key, value in results_metrics.items() if key not in {"micro", "macro"}]
    results_metrics["micro"] = {
        "precision": float(micro_precision),
        "recall": float(micro_recall),
        "f1_score": float(micro_f1),
    }
    results_metrics["macro"] = {"f1_score": float(np.mean(class_values)) if class_values else float("nan")}
    return results_metrics


class PublicMatcher:
    """Behavioral clone of the public PUMA Track-2 nuclei matcher.

    It intentionally preserves centroid-equality deletion and stable input-order behavior.
    Validation belongs outside this class.
    """

    def __init__(self, radius: float = PUBLIC_MATCH_RADIUS_PIXELS) -> None:
        self.radius = float(radius)

    def match(
        self,
        gt_features: list[dict[str, Any]],
        pred_features: list[dict[str, Any]],
        *,
        trace: bool = False,
    ) -> RoiMatchResult:
        pred_structure: dict[str, list[dict[str, Any]]] = {}
        for index, pred_feature in enumerate(pred_features):
            feature = dict(pred_feature)
            feature.setdefault("uid", f"pred-{index}")
            match_key = feature["category"]
            pred_structure.setdefault(match_key, []).append(feature)

        matches: list[dict[str, Any]] = []
        events: list[MatchEvent] = []
        selected_seen: set[str] = set()
        for gt_index, gt_feature_raw in enumerate(gt_features):
            gt_feature = dict(gt_feature_raw)
            gt_feature.setdefault("uid", f"gt-{gt_index}")
            match_key = gt_feature["category"]
            eligible: list[dict[str, Any]] = []
            if match_key in pred_structure:
                for pred_feature in pred_structure[match_key]:
                    distance = float(
                        np.linalg.norm(
                            np.asarray(gt_feature["centroid"], dtype=np.float64)
                            - np.asarray(pred_feature["centroid"], dtype=np.float64)
                        )
                    )
                    if distance < self.radius:
                        eligible.append(
                            {
                                "pred_json": pred_feature.get("filename", ""),
                                "gt_category": gt_feature["category"],
                                "pred_category": pred_feature["category"],
                                "distance": distance,
                                "pred_score": pred_feature.get("score", 1),
                                "pred_feature": pred_feature,
                            }
                        )
            eligible.sort(key=lambda item: (-item["pred_score"], item["distance"]))
            if not eligible:
                continue
            best_match = eligible[0]
            matches.append(best_match)
            selected_uid = str(best_match["pred_feature"].get("uid", ""))
            removed_uid = ""
            collision = False
            for index, pred in enumerate(pred_structure[match_key]):
                if np.array_equal(pred["centroid"], best_match["pred_feature"]["centroid"]):
                    removed_uid = str(pred.get("uid", ""))
                    collision = removed_uid != selected_uid
                    del pred_structure[match_key][index]
                    break
            if trace:
                events.append(
                    MatchEvent(
                        gt_uid=str(gt_feature.get("uid", gt_index)),
                        gt_category=str(gt_feature["category"]),
                        selected_uid=selected_uid,
                        removed_uid=removed_uid,
                        selected_uid_seen_before=selected_uid in selected_seen,
                        centroid_collision=collision,
                        score=float(best_match["pred_score"]),
                        distance=float(best_match["distance"]),
                        eligible_uids=tuple(str(item["pred_feature"].get("uid", "")) for item in eligible),
                    )
                )
            selected_seen.add(selected_uid)
        metrics = _classification_metrics(matches, gt_features, pred_features)
        return RoiMatchResult(metrics=metrics, matches=matches, trace=events)
