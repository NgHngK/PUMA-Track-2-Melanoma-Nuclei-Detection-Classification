from __future__ import annotations
from collections import defaultdict
import numpy as np
from CODE.EVALUATION.matcher import match_roi
from CODE.EVALUATION.roi_metric import roi_averaged_fixed10
from CODE.EVALUATION.summed_metric import summed_fixed10
from CODE.EVALUATION.semantic import confusion_matrix, metrics_from_confusion
from CODE.EVALUATION.calibration import nll, ece15


def evaluate_oof_records(records):
    groups = defaultdict(list)
    for r in records:
        groups[(r['arm'], int(r['seed']))].append(r)
    out = {}
    for key, rows in groups.items():
        counts = {}
        by_roi = defaultdict(list)
        for r in rows:
            by_roi[r['roi_id']].append(r)
        for roi, rr in by_roi.items():
            gt_xy = np.array([[x['x'], x['y']] for x in rr], float)
            gt_cls = np.array([x['true_class'] for x in rr], int)
            pred_xy = gt_xy.copy()
            pred_cls = np.array([x['pred_class'] for x in rr], int)
            score = np.array([x['probs'][x['pred_class']] for x in rr], float)
            counts[roi] = match_roi(gt_xy, gt_cls, pred_xy, pred_cls, score)
        y = np.array([r['true_class'] for r in rows], int)
        p = np.array([r['pred_class'] for r in rows], int)
        logits = np.stack([r['logits'] for r in rows])
        sem = metrics_from_confusion(confusion_matrix(y, p))
        out[key] = {
            'roi_f1': roi_averaged_fixed10(counts),
            'summed_f1': summed_fixed10(counts),
            'semantic_f1': sem['macro_f1'],
            'precision': sem['precision'],
            'recall': sem['recall'],
            'f1': sem['f1'],
            'support': sem['support'],
            'confusion': confusion_matrix(y, p),
            'nll': nll(logits, y),
            'ece15': ece15(logits, y),
            'counts_by_roi': counts,
        }
    return out
