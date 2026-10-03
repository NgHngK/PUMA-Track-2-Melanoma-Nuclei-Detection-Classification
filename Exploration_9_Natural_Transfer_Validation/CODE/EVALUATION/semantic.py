from __future__ import annotations
import numpy as np
from CODE.COMMON.constants import NUM_CLASSES


def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def metrics_from_confusion(cm):
    cm = np.asarray(cm, dtype=np.int64)
    tp = np.diag(cm)
    support = cm.sum(1)
    predicted = cm.sum(0)
    fp = predicted - tp
    fn = support - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(NUM_CLASSES, float), where=denom > 0)
    precision = np.divide(tp, predicted, out=np.zeros(NUM_CLASSES, float), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(NUM_CLASSES, float), where=support > 0)
    return {
        'macro_f1': float(f1.mean()),
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'support': support,
        'predicted': predicted,
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }


def fixed10_macro_f1(y_true, y_pred):
    return metrics_from_confusion(confusion_matrix(y_true, y_pred))['macro_f1']
