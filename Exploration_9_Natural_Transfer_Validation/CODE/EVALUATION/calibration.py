from __future__ import annotations
import numpy as np


def softmax(logits):
    z = np.asarray(logits, float)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(logits, labels):
    p = np.clip(softmax(logits), 1e-12, 1)
    return float(-np.log(p[np.arange(len(labels)), np.asarray(labels, int)]).mean())


def ece15(logits, labels, bins=15):
    p = softmax(logits)
    pred = p.argmax(1)
    conf = p.max(1)
    y = np.asarray(labels, int)
    total = len(y)
    ece = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1] if i < bins - 1 else conf <= edges[i + 1])
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)
