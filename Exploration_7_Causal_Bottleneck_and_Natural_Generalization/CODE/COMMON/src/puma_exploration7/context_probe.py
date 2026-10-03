from __future__ import annotations

import numpy as np

from .probes import grouped_linear_probe, pairwise_probe


def _safe_pair(X, y, g, a, b, seed):
    try:
        return {'status': 'ok', 'folds': pairwise_probe(X, y, g, a, b, 3, seed)}
    except ValueError as exc:
        return {'status': 'unavailable', 'reason': str(exc), 'folds': []}


def compare_context(local_features, context_features, labels, groups, pairs=None, seed=17):
    """Capacity-matched information probe: local vs local+context vs shuffled context."""
    local = np.asarray(local_features, dtype=np.float32)
    context = np.asarray(context_features, dtype=np.float32)
    y = np.asarray(labels)
    g = np.asarray(groups)
    if local.shape != context.shape or local.ndim != 2:
        raise ValueError('local/context feature shape mismatch')
    if len(local) != len(y) or len(local) != len(g):
        raise ValueError('feature/label/group length mismatch')

    rng = np.random.default_rng(seed)
    shuffled = context[rng.permutation(len(context))]
    local_context = np.concatenate([local, context], axis=1)
    local_placebo = np.concatenate([local, shuffled], axis=1)

    local_result = grouped_linear_probe(local, y, g, 3, seed)
    context_result = grouped_linear_probe(local_context, y, g, 3, seed)
    placebo_result = grouped_linear_probe(local_placebo, y, g, 3, seed)
    out = {
        'local': local_result,
        'local_plus_context': context_result,
        'capacity_placebo_local_plus_shuffled_context': placebo_result,
        'fold_deltas_vs_local': [
            context_result[i]['macro_f1'] - local_result[i]['macro_f1']
            for i in range(len(local_result))
        ],
        'fold_deltas_vs_placebo': [
            context_result[i]['macro_f1'] - placebo_result[i]['macro_f1']
            for i in range(len(local_result))
        ],
        'pairs': {},
        'seed': int(seed),
    }
    for a, b in pairs or []:
        key = f'{a}_{b}'
        out['pairs'][key] = {
            'local': _safe_pair(local, y, g, a, b, seed),
            'local_plus_context': _safe_pair(local_context, y, g, a, b, seed),
            'capacity_placebo': _safe_pair(local_placebo, y, g, a, b, seed),
        }
    return out
