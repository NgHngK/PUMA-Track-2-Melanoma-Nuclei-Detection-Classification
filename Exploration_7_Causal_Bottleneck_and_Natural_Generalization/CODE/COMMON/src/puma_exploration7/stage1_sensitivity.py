from __future__ import annotations
import numpy as np, pandas as pd
from .io import softmax_np


def center_error_analysis(frame, labels, logits, bins=(0, 2, 4, 8, 16, 1e9)):
    if not {'gt_x', 'gt_y', 'x', 'y'}.issubset(frame.columns):
        raise ValueError('requires gt_x,gt_y,x,y')
    e = np.sqrt(
        (frame.x.to_numpy(float) - frame.gt_x.to_numpy(float)) ** 2
        + (frame.y.to_numpy(float) - frame.gt_y.to_numpy(float)) ** 2
    )
    pred = softmax_np(logits).argmax(1)
    ok = pred == np.asarray(labels)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        ix = (e >= lo) & (e < hi)
        if ix.any():
            rows.append(
                {
                    'lo': lo,
                    'hi': hi,
                    'n': int(ix.sum()),
                    'accuracy': float(ok[ix].mean()),
                    'mean_error_px': float(e[ix].mean()),
                }
            )
    return pd.DataFrame(rows)
