from __future__ import annotations
import numpy as np
from .metrics import paired_roi_bootstrap

PRIMARY_SEED = 17
REPLICATION_SEEDS = (29, 43)
ALL_SEEDS = (17, 29, 43)


def should_replicate(primary_result: dict, policy: str = 'on_primary_pass') -> bool:
    if policy == 'always':
        return True
    if policy != 'on_primary_pass':
        raise ValueError('replication_policy must be on_primary_pass or always')
    return bool(primary_result.get('promotion_gate', {}).get('passed', False))


def aggregate_seed_results(
    per_seed: dict[int, dict],
    candidate_key: str,
    baseline_key: str = 'baseline',
    n_boot: int = 5000,
    seed: int = 17029,
) -> dict:
    seeds = sorted(int(s) for s in per_seed)
    if len(seeds) < 2:
        return {'status': 'insufficient_replication', 'seeds': seeds}
    roi_deltas = []
    seed_means = []
    all_positive = True
    all_recall_ok = True
    all_fp_ok = True
    for s in seeds:
        r = per_seed[s]
        b = r[baseline_key]
        c = r[candidate_key]
        a = np.asarray(b['puma']['per_roi_fixed10_macro_f1'], float)
        d = np.asarray(c['puma']['per_roi_fixed10_macro_f1'], float) - a
        roi_deltas.append(d)
        seed_means.append(float(d.mean()))
        gate = r.get('promotion_gate', {})
        all_positive = all_positive and float(d.mean()) > 0
        all_recall_ok = all_recall_ok and bool(gate.get('recall_guard_ok', False))
        all_fp_ok = all_fp_ok and bool(gate.get('tail_fp_reduced', False))
    lengths = {len(x) for x in roi_deltas}
    if len(lengths) != 1:
        return {'status': 'roi_alignment_mismatch', 'seeds': seeds, 'seed_mean_deltas': seed_means}
    mean_roi_delta = np.stack(roi_deltas, axis=0).mean(0)
    agg = paired_roi_bootstrap(
        np.zeros_like(mean_roi_delta), mean_roi_delta, n_boot=n_boot, seed=seed
    )
    return {
        'status': 'complete',
        'seeds': seeds,
        'seed_mean_deltas': seed_means,
        'mean_seed_delta': float(np.mean(seed_means)),
        'min_seed_delta': float(np.min(seed_means)),
        'all_seed_mean_deltas_positive': bool(all_positive),
        'all_recall_guards_ok': bool(all_recall_ok),
        'all_tail_fp_reduced': bool(all_fp_ok),
        'roi_bootstrap_of_seed_mean_delta': agg,
        'replication_passed': bool(
            all_positive and all_recall_ok and all_fp_ok and agg['ci95'][0] > 0
        ),
    }
