from __future__ import annotations
import numpy as np, torch
from torch.nn import functional as F
from .constants import NUM_CLASSES
from .io import softmax_np


def estimate_prior(labels, alpha=0.5):
    y = np.asarray(labels, int)
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(float) + float(alpha)
    return counts / counts.sum()


def prior_correct_logits(logits, source_prior, target_prior, eps=1e-12):
    z = np.asarray(logits, float)
    s = np.asarray(source_prior, float)
    t = np.asarray(target_prior, float)
    if s.shape != (NUM_CLASSES,) or t.shape != (NUM_CLASSES,) or (s <= 0).any() or (t <= 0).any():
        raise ValueError('priors must be positive 10-vectors')
    return z + np.log(np.clip(t, eps, None))[None, :] - np.log(np.clip(s, eps, None))[None, :]


def fit_temperature(logits, labels, max_iter=100):
    z = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    log_t = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.2, max_iter=max_iter, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / torch.exp(log_t), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).detach().clamp(0.05, 20.0))


def apply_temperature(logits, T):
    return np.asarray(logits, float) / float(T)


def fit_class_bias(logits, labels, l2=1e-4, steps=400, lr=0.05):
    z = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    b = torch.zeros(NUM_CLASSES, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([b], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(z + b, y) + l2 * b.square().mean()
        loss.backward()
        opt.step()
    b = b.detach().numpy()
    return b - b.mean()


def _best_binary_threshold(scores, positive):
    """Exact O(N log N) equivalent of scanning every unique >= threshold."""
    s = np.asarray(scores, float)
    y = np.asarray(positive, bool)
    order = np.argsort(-s, kind='stable')
    ss = s[order]
    yy = y[order].astype(np.int64)
    total_pos = int(yy.sum())
    # End index for each tied score group: predictions include all rows with score >= threshold.
    ends = np.flatnonzero(np.r_[ss[1:] != ss[:-1], True])
    cum_tp = np.cumsum(yy)
    n_pred = ends + 1
    tp = cum_tp[ends]
    fp = n_pred - tp
    fn = total_pos - tp
    den = 2 * tp + fp + fn
    f = np.divide(2 * tp, den, out=np.zeros_like(tp, dtype=float), where=den > 0)
    thresholds = ss[ends]
    # Old exhaustive implementation considered 0 and 1 in addition to all observed scores and chose the smallest threshold on ties.
    cand_t = np.r_[thresholds, 0.0, 1.0]
    all_pos = np.ones(len(y), bool)
    none_or_one = s >= 1.0

    def fscore(mask):
        t = int(np.logical_and(mask, y).sum())
        fp_ = int(np.logical_and(mask, ~y).sum())
        fn_ = int(np.logical_and(~mask, y).sum())
        d = 2 * t + fp_ + fn_
        return 2 * t / d if d else 0.0

    cand_f = np.r_[f, fscore(all_pos), fscore(none_or_one)]
    mx = float(cand_f.max())
    best_t = float(np.min(cand_t[np.isclose(cand_f, mx, rtol=0, atol=1e-15)]))
    return mx, best_t


def oracle_ovr_thresholds(labels, logits):
    y = np.asarray(labels, int)
    p = softmax_np(logits)
    rows = []
    for c in range(NUM_CLASSES):
        score, t = _best_binary_threshold(p[:, c], y == c)
        rows.append({'class_id': c, 'oracle_binary_f1': float(score), 'threshold': float(t)})
    return rows
