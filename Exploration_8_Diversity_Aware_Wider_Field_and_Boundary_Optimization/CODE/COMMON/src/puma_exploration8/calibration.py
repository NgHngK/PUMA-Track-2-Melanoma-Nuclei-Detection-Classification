from __future__ import annotations
import numpy as np
import torch
from .boundary import prior_offset, apply_h1, temperature
from .metrics import classification_metrics


def _selection_metrics(y, logits):
    """Confusion-only metrics for inner model selection; avoids unused AUC/AP/ECE work."""
    y = np.asarray(y, int)
    pred = np.asarray(logits).argmax(1)
    cm = np.bincount(10 * y + pred, minlength=100).reshape(10, 10)
    tp = np.diag(cm)
    support = cm.sum(1)
    predicted = cm.sum(0)
    recall = np.divide(tp, support, out=np.zeros(10, float), where=support > 0)
    f1 = np.divide(
        2 * tp, support + predicted, out=np.zeros(10, float), where=(support + predicted) > 0
    )
    return float(f1.mean()), recall, support


def _nll(y, logits):
    y = np.asarray(y, int)
    z = np.asarray(logits, float)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())


def empirical_prior(y, alpha=0.5):
    c = np.bincount(np.asarray(y, int), minlength=10).astype(float) + alpha
    return c / c.sum()


def _choose_lambda_inner(z, y, g, source, lambdas, max_recall_drop=0.10):
    uniq = np.unique(g)
    if len(uniq) < 2:
        return float(lambdas[0])
    agg = {float(l): [] for l in lambdas}
    base_rec = []
    for hold in uniq:
        va = g == hold
        tr = ~va
        if not tr.any() or not va.any():
            continue
        offset = prior_offset(source, empirical_prior(y[tr]))
        _, base_recall, _ = _selection_metrics(y[va], z[va])
        base_rec.append(base_recall)
        for lam in lambdas:
            agg[float(lam)].append(_selection_metrics(y[va], apply_h1(z[va], offset, lam)))
    candidates = []
    for lam, metrics in agg.items():
        if not metrics:
            continue
        score = float(np.mean([m[0] for m in metrics]))
        safe = True
        for m, base_r in zip(metrics, base_rec):
            recall = m[1]
            supported = m[2] > 0
            if supported.any() and np.min((recall - base_r)[supported]) < -max_recall_drop:
                safe = False
        candidates.append((safe, score, -abs(lam), lam))
    if not candidates:
        return float(lambdas[0])
    safe = [x for x in candidates if x[0]]
    return max(safe or candidates)[3]


def _outer_folds(groups, outer_folds=None):
    g = np.asarray(groups)
    if outer_folds is None:
        uniq = np.unique(g)
        mapping = {x: i for i, x in enumerate(uniq)}
        folds = np.asarray([mapping[x] for x in g], int)
    else:
        folds = np.asarray(outer_folds, int)
    if len(folds) != len(g) or (folds < 0).any() or len(np.unique(folds)) < 2:
        raise ValueError('invalid CAL outer folds')
    for hold in np.unique(folds):
        if set(g[folds == hold]) & set(g[folds != hold]):
            raise ValueError('CAL biological group leaked across outer folds')
    return folds


def crossfit_h1(logits, y, groups, source_prior, lambdas, max_recall_drop=0.10, outer_folds=None):
    z = np.asarray(logits, float)
    y = np.asarray(y, int)
    g = np.asarray(groups)
    folds = _outer_folds(g, outer_folds)
    oof = np.zeros_like(z)
    chosen = []
    for hold in np.unique(folds):
        va = folds == hold
        tr = ~va
        lam = _choose_lambda_inner(z[tr], y[tr], g[tr], source_prior, lambdas, max_recall_drop)
        target = empirical_prior(y[tr])
        offset = prior_offset(source_prior, target)
        oof[va] = apply_h1(z[va], offset, lam)
        chosen.append({'fold': int(hold), 'lambda': float(lam), 'target_prior': target.tolist()})
    final_lambda = _choose_lambda_inner(z, y, g, source_prior, lambdas, max_recall_drop)
    final_target = empirical_prior(y)
    return {
        'final_lambda': float(final_lambda),
        'final_target_prior': final_target.tolist(),
        'final_offset': prior_offset(source_prior, final_target).tolist(),
        'outer_chosen_lambdas': chosen,
        'oof_metrics': classification_metrics(y, oof),
        'oof_logits': oof,
    }


def fit_h2_bias(logits, y, shrinkage=100.0, steps=300, lr=0.03):
    z = torch.tensor(np.asarray(logits, float), dtype=torch.float32)
    yy = torch.tensor(np.asarray(y, int), dtype=torch.long)
    raw = torch.zeros(10, requires_grad=True)
    opt = torch.optim.LBFGS([raw], lr=lr, max_iter=steps)

    def closure():
        opt.zero_grad()
        delta = raw - raw.mean()
        loss = (
            torch.nn.functional.cross_entropy(z + delta, yy)
            + float(shrinkage) * delta.pow(2).mean()
        )
        loss.backward()
        return loss

    opt.step(closure)
    return (raw - raw.mean()).detach().numpy()


def _choose_h2_inner(z, y, g, shrinkages):
    scores = []
    for shrinkage in shrinkages:
        vals = []
        for hold in np.unique(g):
            va = g == hold
            tr = ~va
            if not tr.any() or not va.any():
                continue
            delta = fit_h2_bias(z[tr], y[tr], shrinkage)
            vals.append(_selection_metrics(y[va], z[va] + delta[None, :])[0])
        scores.append(
            (float(np.mean(vals)) if vals else -np.inf, -float(shrinkage), float(shrinkage))
        )
    return max(scores)[2]


def crossfit_h1_h2(
    logits,
    y,
    groups,
    source_prior,
    lambdas,
    shrinkages=(100.0,),
    max_recall_drop=0.10,
    outer_folds=None,
):
    """Nested H1→H2 cross-fit. The held outer group participates in neither H1 nor H2 fitting."""
    z = np.asarray(logits, float)
    y = np.asarray(y, int)
    g = np.asarray(groups)
    folds = _outer_folds(g, outer_folds)
    oof = np.zeros_like(z)
    choices = []
    for hold in np.unique(folds):
        va = folds == hold
        tr = ~va
        lam = _choose_lambda_inner(z[tr], y[tr], g[tr], source_prior, lambdas, max_recall_drop)
        target = empirical_prior(y[tr])
        offset = prior_offset(source_prior, target)
        h1_train = apply_h1(z[tr], offset, lam)
        h1_val = apply_h1(z[va], offset, lam)
        shrink = _choose_h2_inner(h1_train, y[tr], g[tr], shrinkages)
        delta = fit_h2_bias(h1_train, y[tr], shrink)
        oof[va] = h1_val + delta[None, :]
        choices.append(
            {
                'fold': int(hold),
                'lambda': float(lam),
                'shrinkage': float(shrink),
                'target_prior': target.tolist(),
            }
        )

    final_lam = _choose_lambda_inner(z, y, g, source_prior, lambdas, max_recall_drop)
    final_target = empirical_prior(y)
    final_offset = prior_offset(source_prior, final_target)
    h1_full = apply_h1(z, final_offset, final_lam)
    final_shrink = _choose_h2_inner(h1_full, y, g, shrinkages)
    final_delta = fit_h2_bias(h1_full, y, final_shrink)
    return {
        'outer_choices': choices,
        'final_lambda': float(final_lam),
        'final_target_prior': final_target.tolist(),
        'final_offset': final_offset.tolist(),
        'final_shrinkage': float(final_shrink),
        'final_delta_zero_sum': final_delta.tolist(),
        'oof_metrics': classification_metrics(y, oof),
        'oof_logits': oof,
    }


def fit_temperature_grid(logits, y, grid=None):
    grid = np.logspace(-1, 1, 161) if grid is None else np.asarray(grid, float)
    scores = [_nll(y, temperature(logits, t)) for t in grid]
    return float(grid[int(np.argmin(scores))])
