from __future__ import annotations
from CODE.COMMON.constants import (
    MIN_ROI_DELTA,
    MIN_POSITIVE_SEEDS,
    MAX_MEAN_RECALL_DROP,
    MAX_SEED_RECALL_DROP,
)


def decide_transfer(
    *,
    valid: bool,
    broad_natural: bool,
    all_classes_five_rois: bool,
    mean_roi_delta: float,
    roi_ci_lower: float,
    positive_seed_count: int,
    summed_delta: float,
    max_mean_recall_drop: float,
    max_seed_recall_drop: float,
    loo_all_positive: bool,
    underpowered: bool = False,
):
    if not valid:
        return 'TRANSFER_INVALID_PROVENANCE_OR_EVALUATOR', ['validity gate failed']
    if underpowered:
        return 'TRANSFER_UNDERPOWERED', ['group/support structure underpowered']
    if max_mean_recall_drop > MAX_MEAN_RECALL_DROP or max_seed_recall_drop > MAX_SEED_RECALL_DROP:
        return 'TRANSFER_FAIL_CLASS_SAFETY', [
            f'recall harm mean={max_mean_recall_drop:.4f}, seed={max_seed_recall_drop:.4f}'
        ]
    full = (
        broad_natural
        and all_classes_five_rois
        and mean_roi_delta >= MIN_ROI_DELTA
        and roi_ci_lower > 0
        and positive_seed_count >= MIN_POSITIVE_SEEDS
        and summed_delta >= 0
        and loo_all_positive
    )
    if full:
        return 'TRANSFER_PASS', ['all locked gates passed']
    if (
        mean_roi_delta >= MIN_ROI_DELTA
        and positive_seed_count >= MIN_POSITIVE_SEEDS
        and summed_delta >= 0
    ):
        return 'TRANSFER_PROMISING_BUT_INCONCLUSIVE', [
            'direction positive but one or more evidence-strength gates unresolved'
        ]
    return 'TRANSFER_FAIL_ROLLBACK_P6', ['primary transfer evidence insufficient']
