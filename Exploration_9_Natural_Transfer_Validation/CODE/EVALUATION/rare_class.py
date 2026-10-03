from __future__ import annotations
import numpy as np
from CODE.COMMON.constants import NUM_CLASSES


def positive_roi_counts(labels, rois):
    labels = np.asarray(labels)
    rois = np.asarray(rois)
    return np.array([len(set(rois[labels == c])) for c in range(NUM_CLASSES)], dtype=int)


def kish_effective_roi_support(labels, rois):
    labels = np.asarray(labels)
    rois = np.asarray(rois)
    out = []
    for c in range(NUM_CLASSES):
        r = rois[labels == c]
        vals = np.array([np.sum(r == x) for x in set(r.tolist())], float)
        out.append(0.0 if not len(vals) else float(vals.sum() ** 2 / (vals**2).sum()))
    return np.array(out)
