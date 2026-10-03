from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def ece15(probs, y):
    p = np.asarray(probs)
    y = np.asarray(y)
    pred = p.argmax(1)
    conf = p.max(1)
    bins = np.minimum((conf * 15).astype(int), 14)
    e = 0.0
    for b in range(15):
        ix = bins == b
        if ix.any():
            e += ix.mean() * abs(conf[ix].mean() - (pred[ix] == y[ix]).mean())
    return float(e)


def classification_metrics(y, logits):
    y = np.asarray(y, int)
    z = np.asarray(logits, float)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    pred = p.argmax(1)
    cm = np.bincount(10 * y + pred, minlength=100).reshape(10, 10)
    tp = np.diag(cm)
    sup = cm.sum(1)
    pn = cm.sum(0)
    pr = np.divide(tp, pn, out=np.zeros(10, float), where=pn > 0)
    re = np.divide(tp, sup, out=np.zeros(10, float), where=sup > 0)
    f = np.divide(2 * tp, sup + pn, out=np.zeros(10, float), where=(sup + pn) > 0)
    auprc = []
    auroc = []
    for c in range(10):
        t = (y == c).astype(int)
        auprc.append(float(average_precision_score(t, p[:, c])) if len(np.unique(t)) == 2 else None)
        auroc.append(float(roc_auc_score(t, p[:, c])) if len(np.unique(t)) == 2 else None)
    nll = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())
    return {
        'macro_f1_fixed10': float(f.mean()),
        'macro_f1_support_aware': float(f[sup > 0].mean()) if (sup > 0).any() else 0.0,
        'balanced_accuracy': float(re[sup > 0].mean()) if (sup > 0).any() else 0.0,
        'accuracy': float((pred == y).mean()),
        'precision': pr.tolist(),
        'recall': re.tolist(),
        'f1': f.tolist(),
        'support': sup.tolist(),
        'predicted': pn.tolist(),
        'false_positive': (pn - tp).tolist(),
        'overprediction_ratio': np.divide(pn, sup, out=np.full(10, np.nan), where=sup > 0).tolist(),
        'tail_false_positive_sum': int((pn - tp)[2:].sum()),
        'auprc': auprc,
        'auroc': auroc,
        'nll': nll,
        'ece15': ece15(p, y),
        'confusion': cm.tolist(),
    }
