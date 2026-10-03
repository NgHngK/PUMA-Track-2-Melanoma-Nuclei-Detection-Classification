from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from .constants import CLASSES, NUM_CLASSES, CANONICAL_TO_V17, TAIL_DEFAULT
from .io import softmax_np
from .vendor.puma_v17_evaluator.constants import NUCLEUS_CLASSES
from .vendor.puma_v17_evaluator.evaluation.dataset_evaluation import evaluate_dataset_predictions


def semantic_metrics(labels, logits, groups=None):
    y = np.asarray(labels, int)
    p = softmax_np(logits)
    pred = p.argmax(1)
    cm = np.bincount(NUM_CLASSES * y + pred, minlength=NUM_CLASSES**2).reshape(
        NUM_CLASSES, NUM_CLASSES
    )
    tp = cm.diagonal()
    support = cm.sum(1)
    predicted = cm.sum(0)
    prec = np.divide(tp, predicted, out=np.zeros(NUM_CLASSES, float), where=predicted > 0)
    rec = np.divide(tp, support, out=np.zeros(NUM_CLASSES, float), where=support > 0)
    f1 = np.divide(
        2 * tp,
        support + predicted,
        out=np.zeros(NUM_CLASSES, float),
        where=(support + predicted) > 0,
    )
    supported = support > 0
    nll = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())
    conf = p.max(1)
    ok = pred == y
    bins = np.minimum((conf * 15).astype(int), 14)
    ece = sum(
        float((bins == b).mean() * abs(conf[bins == b].mean() - ok[bins == b].mean()))
        for b in range(15)
        if (bins == b).any()
    )
    auprc = []
    auroc = []
    for c in range(NUM_CLASSES):
        yy = (y == c).astype(int)
        valid = yy.sum() and yy.sum() < len(yy)
        auprc.append(float(average_precision_score(yy, p[:, c])) if valid else float('nan'))
        auroc.append(float(roc_auc_score(yy, p[:, c])) if valid else float('nan'))
    return {
        'macro_f1_fixed10': float(f1.mean()),
        'macro_f1_supported': float(f1[supported].mean()) if supported.any() else float('nan'),
        'balanced_accuracy_fixed10': float(rec.mean()),
        'balanced_accuracy_supported': (
            float(rec[supported].mean()) if supported.any() else float('nan')
        ),
        'accuracy': float(ok.mean()),
        'nll': nll,
        'ece15': float(ece),
        'support': support.tolist(),
        'predicted': predicted.tolist(),
        'false_positive_counts': (predicted - tp).tolist(),
        'tail_false_positive_sum': int((predicted - tp)[list(TAIL_DEFAULT)].sum()),
        'precision': prec.tolist(),
        'recall': rec.tolist(),
        'f1': f1.tolist(),
        'auprc': auprc,
        'auroc': auroc,
        'confusion': cm.tolist(),
        'missing_classes': [CLASSES[i] for i in range(NUM_CLASSES) if not supported[i]],
        'overprediction_ratio': [
            (
                float(predicted[i] / support[i])
                if support[i] > 0
                else (float('inf') if predicted[i] > 0 else 0.0)
            )
            for i in range(NUM_CLASSES)
        ],
    }


def puma_metrics(frame, labels, logits):
    p = softmax_np(logits)
    pred = p.argmax(1)
    labels = np.asarray(labels, int)
    roi_arr = frame['roi'].astype(str).to_numpy()
    names = np.unique(roi_arr)
    gd = np.dtype([('x', 'f8'), ('y', 'f8'), ('class_id', 'i2'), ('nucleus_index', 'i8')])
    pdtype = np.dtype(
        [('x', 'f8'), ('y', 'f8'), ('class_id', 'i2'), ('score', 'f8'), ('proposal_uid', 'U160')]
    )
    gt_by = {}
    pr_by = {}
    cols = set(frame.columns)
    xv = frame['x'].to_numpy(float)
    yv = frame['y'].to_numpy(float)
    uids = frame['uid'].astype(str).to_numpy()
    evalx = frame['eval_x'].to_numpy(float) if 'eval_x' in cols else None
    evaly = frame['eval_y'].to_numpy(float) if 'eval_y' in cols else None
    gtx = frame['gt_x'].to_numpy(float) if 'gt_x' in cols else None
    gty = frame['gt_y'].to_numpy(float) if 'gt_y' in cols else None
    for j, name in enumerate(names):
        ix = np.flatnonzero(roi_arr == name)
        gt = []
        for local, i in enumerate(ix):
            if evalx is not None and np.isfinite(evalx[i]) and np.isfinite(evaly[i]):
                centers = ((evalx[i], evaly[i]),)
            elif gtx is not None and np.isfinite(gtx[i]) and np.isfinite(gty[i]):
                centers = ((gtx[i], gty[i]),)
            else:
                centers = ((xv[i], yv[i]),)
            gt.extend(
                (x, y, int(CANONICAL_TO_V17[labels[i]]), local + k)
                for k, (x, y) in enumerate(centers)
            )
        gt_by[j] = np.asarray(gt, dtype=gd)
        pr_by[j] = np.asarray(
            [
                (xv[i], yv[i], int(CANONICAL_TO_V17[pred[i]]), float(p[i, pred[i]]), uids[i])
                for i in ix
            ],
            dtype=pdtype,
        )
    out = evaluate_dataset_predictions(gt_by, pr_by, list(range(len(names))), trace=False)
    out['per_roi_fixed10_macro_f1'] = [
        float(np.mean([float(rm.get(c, {}).get('f1_score', 0.0)) for c in NUCLEUS_CLASSES]))
        for rm in out['roi_metrics']
    ]
    out['roi_names'] = names.tolist()
    return out


def paired_roi_bootstrap(base, cand, n_boot=2000, seed=17):
    a = np.asarray(base, float)
    b = np.asarray(cand, float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError('paired arrays')
    if len(a) == 0:
        raise ValueError('empty paired arrays')
    d = b - a
    rng = np.random.default_rng(seed)
    n = len(d)
    # Vectorized in bounded blocks: same bootstrap estimator, far less Python overhead.
    block = max(1, min(n_boot, max(64, 2_000_000 // max(n, 1))))
    means = []
    for start in range(0, n_boot, block):
        m = min(block, n_boot - start)
        ix = rng.integers(0, n, size=(m, n))
        means.append(d[ix].mean(1))
    means = np.concatenate(means)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        'mean_delta': float(d.mean()),
        'ci95': [float(lo), float(hi)],
        'positive_fraction': float((d > 0).mean()),
        'n_roi': n,
    }
