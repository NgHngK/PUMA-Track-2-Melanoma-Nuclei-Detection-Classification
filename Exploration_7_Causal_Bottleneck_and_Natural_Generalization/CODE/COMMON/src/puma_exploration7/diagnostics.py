from __future__ import annotations
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from .constants import CLASSES, NUM_CLASSES
from .io import softmax_np


def confusion_pair_table(labels, logits):
    y = np.asarray(labels, int)
    p = softmax_np(logits)
    pred = p.argmax(1)
    top2 = np.argsort(-p, axis=1)[:, :2]
    rows = []
    for a in range(NUM_CLASSES):
        for b in range(NUM_CLASSES):
            if a == b:
                continue
            ix = (y == a) & (pred == b)
            if ix.any():
                rows.append(
                    {
                        'true_id': a,
                        'true': CLASSES[a],
                        'pred_id': b,
                        'pred': CLASSES[b],
                        'n': int(ix.sum()),
                        'mean_conf': float(p[ix, b].mean()),
                        'median_margin': float(np.median(p[ix, top2[ix, 0]] - p[ix, top2[ix, 1]])),
                        'correct_is_second_rate': float((top2[ix, 1] == a).mean()),
                    }
                )
    return (
        pd.DataFrame(rows).sort_values('n', ascending=False)
        if rows
        else pd.DataFrame(columns=['true', 'pred', 'n'])
    )


def group_error_table(frame, labels, logits, group_col):
    p = softmax_np(logits)
    pred = p.argmax(1)
    z = frame[[group_col]].copy()
    z['correct'] = pred == np.asarray(labels)
    return (
        z.groupby(group_col)
        .agg(n=('correct', 'size'), accuracy=('correct', 'mean'))
        .reset_index()
        .sort_values(['accuracy', 'n'])
    )


def numeric_error_correlations(frame, labels, logits, exclude=('label', 'x', 'y', 'gt_x', 'gt_y')):
    pred = softmax_np(logits).argmax(1)
    err = (pred != np.asarray(labels)).astype(float)
    rows = []
    for c in frame.select_dtypes(include=[np.number]).columns:
        if c in exclude:
            continue
        x = frame[c].to_numpy(float)
        ok = np.isfinite(x)
        if ok.sum() < 20 or np.nanstd(x[ok]) == 0:
            continue
        r, p = spearmanr(x[ok], err[ok])
        rows.append({'feature': c, 'spearman_error': float(r), 'p_value': float(p)})
    return (
        pd.DataFrame(rows).sort_values('spearman_error', key=lambda s: s.abs(), ascending=False)
        if rows
        else pd.DataFrame()
    )


def top2_diagnostics(labels, logits):
    y = np.asarray(labels, int)
    p = softmax_np(logits)
    order = np.argsort(-p, axis=1)
    return {
        'top1_accuracy': float((order[:, 0] == y).mean()),
        'top2_accuracy': float(((order[:, :2] == y[:, None]).any(1)).mean()),
        'mean_true_rank': float((np.argsort(order, axis=1)[np.arange(len(y)), y] + 1).mean()),
        'mean_top1_margin': float(
            (p[np.arange(len(y)), order[:, 0]] - p[np.arange(len(y)), order[:, 1]]).mean()
        ),
    }
