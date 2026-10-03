from __future__ import annotations
import numpy as np
from CODE.EVALUATION.roi_metric import f1_from_counts
from CODE.EVALUATION.summed_metric import summed_fixed10
from CODE.EVALUATION.rare_class import positive_roi_counts, kish_effective_roi_support
from CODE.EVALUATION.transfer_gate import decide_transfer


def _roi_f1_map(counts):
    return {roi: float(f1_from_counts(*v).mean()) for roi, v in counts.items()}


def summarize_natural_transfer(
    seed_metrics: dict, labels, rois, *, bootstrap_reps=5000, bootstrap_seed=20260921
):
    seeds = sorted({int(k[0]) for k in seed_metrics})
    for seed in seeds:
        if (seed, 'ANCHOR') not in seed_metrics or (seed, 'ANCHOR_CONTEXT_C') not in seed_metrics:
            raise ValueError(f'Missing primary arm metrics for seed {seed}')
    roi_deltas = []
    summed_deltas = []
    recall_deltas = []
    per_seed = {}
    roi_effects_by_seed = {}
    all_rois = None
    for seed in seeds:
        a = seed_metrics[(seed, 'ANCHOR')]
        c = seed_metrics[(seed, 'ANCHOR_CONTEXT_C')]
        roi_delta = float(c['roi_f1'] - a['roi_f1'])
        summed_delta = float(c['summed_f1'] - a['summed_f1'])
        rd = np.asarray(c['recall'], float) - np.asarray(a['recall'], float)
        roi_deltas.append(roi_delta)
        summed_deltas.append(summed_delta)
        recall_deltas.append(rd)
        amap = _roi_f1_map(a['counts_by_roi'])
        cmap = _roi_f1_map(c['counts_by_roi'])
        if set(amap) != set(cmap):
            raise ValueError('ROI set mismatch between arms')
        if all_rois is None:
            all_rois = sorted(amap)
        elif set(all_rois) != set(amap):
            raise ValueError('ROI set mismatch across seeds')
        roi_effects_by_seed[seed] = {r: cmap[r] - amap[r] for r in all_rois}
        per_seed[str(seed)] = {
            'anchor_roi_f1': float(a['roi_f1']),
            'context_roi_f1': float(c['roi_f1']),
            'roi_delta': roi_delta,
            'anchor_summed_f1': float(a['summed_f1']),
            'context_summed_f1': float(c['summed_f1']),
            'summed_delta': summed_delta,
            'recall_delta': rd.tolist(),
            'anchor_nll': float(a['nll']),
            'context_nll': float(c['nll']),
            'anchor_ece15': float(a['ece15']),
            'context_ece15': float(c['ece15']),
        }
    mean_roi_delta = float(np.mean(roi_deltas))
    mean_summed_delta = float(np.mean(summed_deltas))
    mean_recall_delta = np.mean(np.stack(recall_deltas), axis=0)
    max_mean_recall_drop = float(max(0.0, -mean_recall_delta.min()))
    max_seed_recall_drop = float(max(0.0, -np.stack(recall_deltas).min()))
    positive_seed_count = int(np.sum(np.asarray(roi_deltas) > 0))

    # Paired ROI bootstrap: preserve the exact RNG draws, but vectorize the
    # expensive summed-count calculation instead of rebuilding Python dicts for
    # every replicate x seed. This is execution-only; bootstrap semantics are unchanged.
    mean_effect_by_roi = {
        r: float(np.mean([roi_effects_by_seed[s][r] for s in seeds])) for r in all_rois
    }
    roi_to_index = {r: i for i, r in enumerate(all_rois)}
    rng = np.random.default_rng(bootstrap_seed)
    sample_idx = np.empty((bootstrap_reps, len(all_rois)), dtype=np.int32)
    for b in range(bootstrap_reps):
        # choice(R) produces the same index stream as choice(all_rois) for this
        # replacement/uniform case, while giving us compact integer indices.
        sample_idx[b] = rng.choice(len(all_rois), size=len(all_rois), replace=True)

    effect_arr = np.asarray([mean_effect_by_roi[r] for r in all_rois], dtype=np.float64)
    boot_roi = effect_arr[sample_idx].mean(axis=1)
    boot_sum = np.zeros(bootstrap_reps, dtype=np.float64)

    def _count_array(metric, part):
        pos = {'tp': 0, 'fp': 1, 'fn': 2}[part]
        return np.stack(
            [np.asarray(metric['counts_by_roi'][r][pos], dtype=np.int64) for r in all_rois], axis=0
        )

    # Chunked vectorization keeps peak RAM modest (<~100 MB for the bootstrap work).
    chunk = 256
    for seed in seeds:
        a = seed_metrics[(seed, 'ANCHOR')]
        c = seed_metrics[(seed, 'ANCHOR_CONTEXT_C')]
        atp, afp, afn = (_count_array(a, k) for k in ('tp', 'fp', 'fn'))
        ctp, cfp, cfn = (_count_array(c, k) for k in ('tp', 'fp', 'fn'))
        seed_delta = np.empty(bootstrap_reps, dtype=np.float64)
        for lo in range(0, bootstrap_reps, chunk):
            hi = min(lo + chunk, bootstrap_reps)
            ix = sample_idx[lo:hi]

            def score(tp, fp, fn):
                stp = tp[ix].sum(axis=1)
                sfp = fp[ix].sum(axis=1)
                sfn = fn[ix].sum(axis=1)
                den = 2.0 * stp + sfp + sfn
                f = np.divide(
                    2.0 * stp, den, out=np.zeros_like(den, dtype=np.float64), where=den > 0
                )
                return f.mean(axis=1)

            seed_delta[lo:hi] = score(ctp, cfp, cfn) - score(atp, afp, afn)
        boot_sum += seed_delta
    boot_sum /= float(len(seeds))
    roi_ci = {
        'mean': float(boot_roi.mean()),
        'lower': float(np.quantile(boot_roi, 0.025)),
        'upper': float(np.quantile(boot_roi, 0.975)),
        'replicates': bootstrap_reps,
    }
    summed_ci = {
        'mean': float(boot_sum.mean()),
        'lower': float(np.quantile(boot_sum, 0.025)),
        'upper': float(np.quantile(boot_sum, 0.975)),
        'replicates': bootstrap_reps,
    }

    loo = []
    for omitted in all_rois:
        remain = [r for r in all_rois if r != omitted]
        loo.append(float(np.mean([mean_effect_by_roi[r] for r in remain])))
    loo_all_positive = bool(loo and min(loo) > 0)
    labels = np.asarray(labels)
    rois = np.asarray(rois)
    pos = positive_roi_counts(labels, rois)
    kish = kish_effective_roi_support(labels, rois)
    all_five = bool((pos >= 5).all())
    broad = bool(len(set(rois.tolist())) > 24)
    decision, reasons = decide_transfer(
        valid=True,
        broad_natural=broad,
        all_classes_five_rois=all_five,
        mean_roi_delta=mean_roi_delta,
        roi_ci_lower=roi_ci['lower'],
        positive_seed_count=positive_seed_count,
        summed_delta=mean_summed_delta,
        max_mean_recall_drop=max_mean_recall_drop,
        max_seed_recall_drop=max_seed_recall_drop,
        loo_all_positive=loo_all_positive,
        underpowered=False,
    )
    return {
        'decision': decision,
        'decision_reasons': reasons,
        'mean_roi_delta': mean_roi_delta,
        'roi_bootstrap': roi_ci,
        'mean_summed_delta': mean_summed_delta,
        'summed_bootstrap': summed_ci,
        'positive_seed_count': positive_seed_count,
        'mean_recall_delta': mean_recall_delta.tolist(),
        'max_mean_recall_drop': max_mean_recall_drop,
        'max_seed_recall_drop': max_seed_recall_drop,
        'positive_roi_counts_per_class': pos.tolist(),
        'kish_effective_roi_support': kish.tolist(),
        'all_classes_at_least_5_positive_rois': all_five,
        'roi_count': len(all_rois),
        'broad_natural': broad,
        'leave_one_roi_out_min_delta': float(min(loo)) if loo else None,
        'leave_one_roi_out_max_delta': float(max(loo)) if loo else None,
        'leave_one_roi_out_all_positive': loo_all_positive,
        'per_seed': per_seed,
    }
