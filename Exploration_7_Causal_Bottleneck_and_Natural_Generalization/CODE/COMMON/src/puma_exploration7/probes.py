from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _probe_model():
    return make_pipeline(
        StandardScaler(copy=False),
        LogisticRegression(C=1.0, max_iter=600, solver='lbfgs'),
    )


def grouped_linear_probe(X, y, groups, n_splits=3, seed=17):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups).astype(str)
    support = {int(c): len(np.unique(groups[y == c])) for c in np.unique(y)}
    if min(support.values()) < n_splits:
        raise ValueError(
            f'need >= {n_splits} groups per class for grouped probe; support={support}'
        )
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, va) in enumerate(cv.split(X, y, groups)):
        model = _probe_model()
        model.fit(X[tr].copy(), y[tr])
        pred = model.predict(X[va])
        rows.append(
            {
                'fold': fold,
                'macro_f1': float(
                    f1_score(y[va], pred, average='macro', labels=np.unique(y), zero_division=0)
                ),
                'accuracy': float((pred == y[va]).mean()),
            }
        )
    return rows


def cross_group_knn_purity(X, y, groups, k=5, chunk_size=256):
    """Exact cosine kNN purity without materializing an NxN similarity matrix.

    This keeps the historical definition (neighbors from other groups only) but bounds
    memory to roughly chunk_size x N. The arithmetic remains an exact full search.
    """
    X = np.asarray(X, dtype=np.float32)
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    y = np.asarray(y)
    g = np.asarray(groups).astype(str)
    n = len(X)
    if n == 0:
        return float('nan')
    purities = []
    for start in range(0, n, int(chunk_size)):
        stop = min(start + int(chunk_size), n)
        sim = X[start:stop] @ X.T
        for local, i in enumerate(range(start, stop)):
            valid = g != g[i]
            nv = int(valid.sum())
            if nv == 0:
                continue
            kk = min(int(k), nv)
            scores = sim[local].copy()
            scores[~valid] = -np.inf
            if kk == nv:
                idx = np.flatnonzero(valid)
            else:
                cand = np.argpartition(scores, -kk)[-kk:]
                idx = cand[np.argsort(-scores[cand], kind='stable')]
            purities.append(float(np.mean(y[idx] == y[i])))
    return float(np.mean(purities)) if purities else float('nan')


def pairwise_probe(X, y, groups, a, b, n_splits=3, seed=17):
    mask = np.isin(y, [a, b])
    xx = np.asarray(X, dtype=np.float32)[mask]
    yy = (np.asarray(y)[mask] == b).astype(int)
    gg = np.asarray(groups)[mask]
    support = {int(c): len(np.unique(gg[yy == c])) for c in (0, 1)}
    if min(support.values()) < n_splits:
        raise ValueError(f'need >= {n_splits} groups per pair class; support={support}')
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = []
    for fold, (tr, va) in enumerate(cv.split(xx, yy, gg)):
        model = _probe_model()
        model.fit(xx[tr].copy(), yy[tr])
        p = model.predict_proba(xx[va])[:, 1]
        out.append({'fold': fold, 'auprc': float(average_precision_score(yy[va], p))})
    return out
