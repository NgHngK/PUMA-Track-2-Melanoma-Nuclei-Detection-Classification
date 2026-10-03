from __future__ import annotations

from CODE.MODELS.model_factory import make_model
from CODE.MODELS.parameter_audit import assert_parameter_count
from CODE.DATA.tier_a import TierANormalizer, verify_fit_scope
from CODE.COMMON.hashing import sha256_strings
from CODE.COMMON.seed import set_global_seed, derive_seed
from CODE.COMMON.constants import RECONSTRUCTED_CONTEXT_SCALE_INIT
from .loop import train_feature_head, predict_feature_head
from .resource_monitor import ResourceMonitor


def fit_predict_fold(
    arm,
    seed,
    fold,
    fit_idx,
    held_idx,
    uids,
    rois,
    coords,
    h_local,
    h_wide,
    tier_raw,
    labels,
    device='cpu',
    *,
    context_scale_init=RECONSTRUCTED_CONTEXT_SCALE_INIT,
    resume_dir=None,
    resume_metadata=None,
    execution_mode="fast",
):
    fit_uids = [uids[i] for i in fit_idx]
    held_uids = [uids[i] for i in held_idx]
    verify_fit_scope(fit_uids, held_uids)
    norm = TierANormalizer.fit(tier_raw[fit_idx], fit_uids, sha256_strings)
    bfit = norm.transform(tier_raw[fit_idx])
    bheld = norm.transform(tier_raw[held_idx])
    # Same seed/fold initialization base across arms. Because the context model constructs
    # its Anchor first, the shared anchor parameters are bit-identical between arms.
    set_global_seed(derive_seed(seed, 'model', fold))
    model = make_model(arm, context_scale_init=context_scale_init)
    assert_parameter_count(model, arm)
    with ResourceMonitor() as mon:
        if arm == 'ANCHOR':
            model, hist, logits = train_feature_head(
                model,
                h_local[fit_idx],
                bfit,
                labels[fit_idx],
                seed=seed,
                fold=fold,
                device=device,
                train_rois=rois[fit_idx],
                train_coords=coords[fit_idx],
                val_h_local=h_local[held_idx],
                val_tier_a=bheld,
                val_labels=labels[held_idx],
                val_rois=rois[held_idx],
                val_coords=coords[held_idx],
                run_tag=f'{arm} seed={seed} fold={fold}',
                resume_dir=resume_dir,
                resume_metadata=resume_metadata,
                execution_mode=execution_mode,
                return_last_val_logits=True,
            )
            if logits is None:
                logits = predict_feature_head(
                    model, h_local[held_idx], bheld, device=device, execution_mode=execution_mode
                )
        else:
            model, hist, logits = train_feature_head(
                model,
                h_local[fit_idx],
                bfit,
                labels[fit_idx],
                h_wide=h_wide[fit_idx],
                seed=seed,
                fold=fold,
                device=device,
                train_rois=rois[fit_idx],
                train_coords=coords[fit_idx],
                val_h_local=h_local[held_idx],
                val_tier_a=bheld,
                val_labels=labels[held_idx],
                val_h_wide=h_wide[held_idx],
                val_rois=rois[held_idx],
                val_coords=coords[held_idx],
                run_tag=f'{arm} seed={seed} fold={fold}',
                resume_dir=resume_dir,
                resume_metadata=resume_metadata,
                execution_mode=execution_mode,
                return_last_val_logits=True,
            )
            if logits is None:
                logits = predict_feature_head(
                    model,
                    h_local[held_idx],
                    bheld,
                    h_wide=h_wide[held_idx],
                    device=device,
                    execution_mode=execution_mode,
                )
    resource = mon.as_dict()
    return logits, hist, norm, resource
