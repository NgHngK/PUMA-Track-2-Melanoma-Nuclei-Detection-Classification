from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import numpy as np

from CODE.COMMON.constants import NUM_CLASSES, MATCH_RADIUS_PX


def _deduplicate_predictions(coords, classes, scores):
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    seen = set()
    keep = []
    for i in order:
        key = (int(classes[i]), float(coords[i, 0]), float(coords[i, 1]))
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return np.array(keep, dtype=int)


def _cell(x: float, y: float, radius: float) -> tuple[int, int]:
    return (math.floor(x / radius), math.floor(y / radius))


def match_roi(gt_xy, gt_cls, pred_xy, pred_cls, pred_score, radius: float = MATCH_RADIUS_PX):
    """Exact confidence-greedy class-first matcher with spatial-bin acceleration."""
    gt_xy = np.asarray(gt_xy, float)
    pred_xy = np.asarray(pred_xy, float)
    gt_cls = np.asarray(gt_cls, int)
    pred_cls = np.asarray(pred_cls, int)
    pred_score = np.asarray(pred_score, float)
    keep = _deduplicate_predictions(pred_xy, pred_cls, pred_score)
    pred_xy = pred_xy[keep]
    pred_cls = pred_cls[keep]
    pred_score = pred_score[keep]
    tp = np.zeros(NUM_CLASSES, dtype=int)
    fp = np.zeros(NUM_CLASSES, dtype=int)
    fn = np.zeros(NUM_CLASSES, dtype=int)
    r2 = float(radius) * float(radius)
    for c in range(NUM_CLASSES):
        gi = np.flatnonzero(gt_cls == c)
        pi = np.flatnonzero(pred_cls == c)
        if len(gi) == 0:
            fp[c] += len(pi)
            continue
        if len(pi) == 0:
            fn[c] += len(gi)
            continue
        bins = defaultdict(list)
        for g in map(int, gi.tolist()):
            bins[_cell(float(gt_xy[g, 0]), float(gt_xy[g, 1]), float(radius))].append(g)
        unmatched = set(map(int, gi.tolist()))
        p_order = sorted(map(int, pi.tolist()), key=lambda j: (-float(pred_score[j]), j))
        for j in p_order:
            cx, cy = _cell(float(pred_xy[j, 0]), float(pred_xy[j, 1]), float(radius))
            best = None
            px = float(pred_xy[j, 0])
            py = float(pred_xy[j, 1])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for g in bins.get((cx + dx, cy + dy), ()):
                        if g not in unmatched:
                            continue
                        ddx = px - float(gt_xy[g, 0])
                        ddy = py - float(gt_xy[g, 1])
                        d2 = ddx * ddx + ddy * ddy
                        if d2 < r2:
                            candidate = (d2, g)
                            if best is None or candidate < best:
                                best = candidate
            if best is None:
                fp[c] += 1
            else:
                unmatched.remove(best[1])
                tp[c] += 1
        fn[c] += len(unmatched)
    return tp, fp, fn


@dataclass(slots=True)
class PreparedRoiMatcher:
    """GT-side matcher structures reused across epochs for fixed GT coordinates."""

    coords: np.ndarray
    gt_cls: np.ndarray
    gt_indices_by_class: tuple[np.ndarray, ...]
    candidate_gt_by_prediction: tuple[tuple[tuple[int, ...], ...], ...]
    coord_keys: tuple[tuple[float, float], ...]

    def match(self, pred_cls, pred_score):
        pred_cls = np.asarray(pred_cls, dtype=np.int64)
        pred_score = np.asarray(pred_score, dtype=np.float64)
        if len(pred_cls) != len(self.coords) or len(pred_score) != len(self.coords):
            raise ValueError("Prepared matcher prediction length mismatch")

        order = sorted(range(len(pred_score)), key=lambda i: (-float(pred_score[i]), i))
        seen = set()
        keep = []
        for i in order:
            key = (int(pred_cls[i]), self.coord_keys[i][0], self.coord_keys[i][1])
            if key not in seen:
                seen.add(key)
                keep.append(i)

        tp = np.zeros(NUM_CLASSES, dtype=np.int64)
        fp = np.zeros(NUM_CLASSES, dtype=np.int64)
        fn = np.zeros(NUM_CLASSES, dtype=np.int64)
        keep_set = set(keep)
        for c in range(NUM_CLASSES):
            gi = self.gt_indices_by_class[c]
            pi = [j for j in keep if int(pred_cls[j]) == c]
            if len(gi) == 0:
                fp[c] = len(pi)
                continue
            if not pi:
                fn[c] = len(gi)
                continue
            unmatched = set(map(int, gi.tolist()))
            # keep is already global score-desc/stable order; filtering by class preserves it.
            for j in pi:
                matched = None
                for g in self.candidate_gt_by_prediction[j][c]:
                    if g in unmatched:
                        matched = g
                        break
                if matched is None:
                    fp[c] += 1
                else:
                    unmatched.remove(matched)
                    tp[c] += 1
            fn[c] = len(unmatched)
        return tp, fp, fn


def prepare_roi_matcher(gt_xy, gt_cls, radius: float = MATCH_RADIUS_PX) -> PreparedRoiMatcher:
    """Precompute all GT-neighbor ordering needed by repeated epoch matching.

    Prediction coordinates in E9 epoch diagnostics are the same fixed GT coordinates.
    Candidate GT lists are therefore invariant across epochs; only predicted class and
    confidence change. Reusing them removes repeated distance/bin construction.
    """
    coords = np.asarray(gt_xy, dtype=np.float64)
    classes = np.asarray(gt_cls, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) != len(classes):
        raise ValueError("Invalid prepared ROI matcher inputs")
    r2 = float(radius) * float(radius)

    bins = defaultdict(list)
    for g in range(len(coords)):
        bins[_cell(float(coords[g, 0]), float(coords[g, 1]), float(radius))].append(g)

    candidates = []
    for j in range(len(coords)):
        px = float(coords[j, 0])
        py = float(coords[j, 1])
        cx, cy = _cell(px, py, float(radius))
        per_class = [[] for _ in range(NUM_CLASSES)]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for g in bins.get((cx + dx, cy + dy), ()):
                    ddx = px - float(coords[g, 0])
                    ddy = py - float(coords[g, 1])
                    d2 = ddx * ddx + ddy * ddy
                    if d2 < r2:
                        per_class[int(classes[g])].append((d2, int(g)))
        candidates.append(tuple(tuple(g for _, g in sorted(items)) for items in per_class))

    gt_by_class = tuple(np.flatnonzero(classes == c).astype(np.int64) for c in range(NUM_CLASSES))
    coord_keys = tuple((float(x), float(y)) for x, y in coords)
    return PreparedRoiMatcher(
        coords=coords,
        gt_cls=classes,
        gt_indices_by_class=gt_by_class,
        candidate_gt_by_prediction=tuple(candidates),
        coord_keys=coord_keys,
    )
