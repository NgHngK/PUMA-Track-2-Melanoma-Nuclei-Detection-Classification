from __future__ import annotations
import numpy as np
from .roi_metric import f1_from_counts


def summed_fixed10(counts_by_roi):
    if not counts_by_roi:
        raise ValueError('No ROI counts')
    tp = sum((np.asarray(v[0]) for v in counts_by_roi.values()), start=np.zeros(10, dtype=int))
    fp = sum((np.asarray(v[1]) for v in counts_by_roi.values()), start=np.zeros(10, dtype=int))
    fn = sum((np.asarray(v[2]) for v in counts_by_roi.values()), start=np.zeros(10, dtype=int))
    return float(f1_from_counts(tp, fp, fn).mean())
