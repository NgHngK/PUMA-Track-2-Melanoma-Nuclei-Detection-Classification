from __future__ import annotations
import numpy as np
from .calibration import (
    estimate_prior,
    prior_correct_logits,
    fit_temperature,
    apply_temperature,
    oracle_ovr_thresholds,
    fit_class_bias,
)
from .metrics import semantic_metrics, puma_metrics, paired_roi_bootstrap


def evaluate_frame(frame, logits):
    y = frame.label.to_numpy(int)
    sem = semantic_metrics(y, logits)
    puma = None
    if {'roi', 'x', 'y', 'uid'}.issubset(frame.columns):
        puma = puma_metrics(frame, y, logits)
    return {'semantic': sem, 'puma': puma}


def assess_candidate(base, cand, paired, recall_guard=0.10):
    if paired is None:
        return {'passed': False, 'reason': 'PUMA ROI pairing unavailable'}
    rb = np.asarray(base['semantic']['recall'], float)
    rc = np.asarray(cand['semantic']['recall'], float)
    support = np.asarray(base['semantic']['support']) > 0
    harms = rc - rb
    recall_ok = bool((harms[support] >= -recall_guard).all())
    ci_ok = bool(paired['ci95'][0] > 0)
    fp_ok = bool(
        cand['semantic']['tail_false_positive_sum'] < base['semantic']['tail_false_positive_sum']
    )
    return {
        'passed': bool(ci_ok and recall_ok and fp_ok),
        'ci_above_zero': ci_ok,
        'recall_guard_ok': recall_ok,
        'tail_fp_reduced': fp_ok,
        'min_supported_recall_delta': (
            float(harms[support].min()) if support.any() else float('nan')
        ),
    }


def run_exp1(
    frame_cal, logits_cal, frame_test, logits_test, source_prior, alpha=0.5, n_boot=2000, seed=17
):
    target = estimate_prior(frame_cal.label.to_numpy(int), alpha)
    corrected = prior_correct_logits(logits_test, source_prior, target)
    base = evaluate_frame(frame_test, logits_test)
    cand = evaluate_frame(frame_test, corrected)
    T = fit_temperature(logits_cal, frame_cal.label.to_numpy(int))
    temp = evaluate_frame(frame_test, apply_temperature(logits_test, T))
    bias = fit_class_bias(logits_cal, frame_cal.label.to_numpy(int))
    bias_eval = evaluate_frame(frame_test, logits_test + bias)
    oracle = oracle_ovr_thresholds(frame_cal.label.to_numpy(int), logits_cal)
    paired = None
    if base['puma'] and cand['puma']:
        paired = paired_roi_bootstrap(
            base['puma']['per_roi_fixed10_macro_f1'],
            cand['puma']['per_roi_fixed10_macro_f1'],
            n_boot,
            seed,
        )
    gate = assess_candidate(base, cand, paired)
    return {
        'source_prior': list(map(float, source_prior)),
        'target_prior': target.tolist(),
        'baseline': base,
        'prior_corrected': cand,
        'paired_roi_bootstrap': paired,
        'promotion_gate': gate,
        'temperature': T,
        'temperature_only': temp,
        'class_bias_diagnostic': bias.tolist(),
        'class_bias_test': bias_eval,
        'oracle_ovr_calibration_only': oracle,
        'temperature_argmax_invariant': bool(
            np.array_equal(
                np.asarray(logits_test).argmax(1), apply_temperature(logits_test, T).argmax(1)
            )
        ),
    }
