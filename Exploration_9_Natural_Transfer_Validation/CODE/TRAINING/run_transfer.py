from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from CODE.COMMON.constants import SEEDS, TRAIN_EPOCHS
from CODE.COMMON.exceptions import FoldLeakageError, ResumeIntegrityError
from CODE.TRAINING.trainer import fit_predict_fold
from CODE.TRAINING.checkpointing import validate_resume_metadata
from CODE.TRAINING.histories import save_fit_history
from CODE.TRAINING.oof import validate_oof
from CODE.EVALUATION.calibration import softmax
from CODE.COMMON.hashing import canonical_json_hash, sha256_file


def _validate_arrays(uids, rois, labels, coords, h_local, h_wide, tier_raw, folds):
    n = len(uids)
    arrays = [rois, labels, coords, h_local, h_wide, tier_raw, folds]
    if any(len(x) != n for x in arrays):
        raise ValueError('Natural-transfer input length mismatch')
    if len(set(uids)) != n:
        raise ValueError('Duplicate UIDs')
    labels = np.asarray(labels)
    folds = np.asarray(folds)
    rois = np.asarray(rois)
    if (
        not np.isfinite(np.asarray(h_local)).all()
        or not np.isfinite(np.asarray(h_wide)).all()
        or not np.isfinite(np.asarray(tier_raw)).all()
    ):
        raise ValueError('Nonfinite transfer features')
    for k in sorted(set(folds.tolist())):
        fit_rois = set(rois[folds != k])
        held_rois = set(rois[folds == k])
        overlap = fit_rois & held_rois
        if overlap:
            raise FoldLeakageError(f'ROI leakage in fold {k}: {next(iter(overlap))}')


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    os.replace(tmp, path)


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(tmp, path)


def _load_completed_fit(fit_dir: Path, expected_metadata: dict, held_count: int):
    complete_path = fit_dir / 'COMPLETE.json'
    logits_path = fit_dir / 'held_logits.npy'
    history_path = fit_dir / 'history.json'
    resource_path = fit_dir / 'resource.json'
    if not complete_path.is_file():
        return None
    complete = json.loads(complete_path.read_text(encoding='utf-8'))
    validate_resume_metadata(complete.get('metadata', {}), expected_metadata)
    if int(complete.get('completed_epochs', -1)) != TRAIN_EPOCHS:
        raise ResumeIntegrityError(f'Invalid completed epoch count in {complete_path}')
    if not logits_path.is_file() or not history_path.is_file():
        raise ResumeIntegrityError(f'Completed fit is missing logits/history: {fit_dir}')
    logits = np.load(logits_path, allow_pickle=False)
    if logits.shape != (held_count, 10):
        raise ResumeIntegrityError(f'Held-logit shape mismatch in {fit_dir}: {logits.shape}')
    if sha256_file(logits_path) != complete.get('held_logits_sha256'):
        raise ResumeIntegrityError(f'Held-logit hash mismatch in {fit_dir}')
    history = json.loads(history_path.read_text(encoding='utf-8'))
    if len(history) != TRAIN_EPOCHS:
        raise ResumeIntegrityError(f'History length mismatch in {fit_dir}')
    resource = (
        json.loads(resource_path.read_text(encoding='utf-8')) if resource_path.is_file() else {}
    )
    return logits.astype(np.float32, copy=False), history, resource


def run_oof_transfer(
    *,
    uids,
    rois,
    labels,
    coords,
    h_local,
    h_wide,
    tier_raw,
    folds,
    seeds=SEEDS,
    arms=('ANCHOR', 'ANCHOR_CONTEXT_C'),
    device='cpu',
    config_hash=None,
    context_scale_init=0.0,
    resume_root=None,
    feature_hashes=None,
    execution_mode='fast',
):
    uids = list(map(str, uids))
    rois = np.asarray(rois)
    labels = np.asarray(labels, dtype=np.int64)
    coords = np.asarray(coords, dtype=np.float64)
    h_local = np.asarray(h_local, dtype=np.float32)
    h_wide = np.asarray(h_wide, dtype=np.float32)
    tier_raw = np.asarray(tier_raw, dtype=np.float32)
    folds = np.asarray(folds, dtype=np.int64)
    _validate_arrays(uids, rois, labels, coords, h_local, h_wide, tier_raw, folds)
    roster_hash = canonical_json_hash(
        [
            {'uid': u, 'roi': str(r), 'label': int(y), 'x': float(c[0]), 'y': float(c[1])}
            for u, r, y, c in zip(uids, rois, labels, coords)
        ]
    )
    fold_hash = canonical_json_hash([{'uid': u, 'fold': int(f)} for u, f in zip(uids, folds)])
    if config_hash is None:
        config_hash = canonical_json_hash(
            {'arms': list(arms), 'seeds': list(map(int, seeds)), 'scientific_recipe': 'E9_locked'}
        )
    feature_hashes = dict(feature_hashes or {})
    resume_root = None if resume_root is None else Path(resume_root)
    records = []
    histories = {}
    for seed in seeds:
        for fold in sorted(set(folds.tolist())):
            fit_idx = np.flatnonzero(folds != fold)
            held_idx = np.flatnonzero(folds == fold)
            for arm in arms:
                expected_metadata = {
                    'roster_hash': roster_hash,
                    'config_hash': config_hash,
                    'fold_hash': fold_hash,
                    'feature_hashes': feature_hashes,
                    'seed': int(seed),
                    'fold': int(fold),
                    'arm': str(arm),
                }
                fit_dir = (
                    None if resume_root is None else resume_root / f'{arm}_seed{seed}_fold{fold}'
                )
                completed = (
                    None
                    if fit_dir is None
                    else _load_completed_fit(fit_dir, expected_metadata, len(held_idx))
                )
                print(f'[FIT] {arm} seed={seed} fold={fold} mode={execution_mode}', flush=True)
                if completed is not None:
                    logits, hist, resource = completed
                    print(
                        f'[SKIP] {arm} seed={seed} fold={fold}: complete ({TRAIN_EPOCHS}/{TRAIN_EPOCHS} epochs)',
                        flush=True,
                    )
                else:
                    if fit_dir is not None:
                        fit_dir.mkdir(parents=True, exist_ok=True)
                    logits, hist, _, resource = fit_predict_fold(
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
                        device=device,
                        context_scale_init=context_scale_init,
                        resume_dir=fit_dir,
                        resume_metadata=expected_metadata,
                        execution_mode=execution_mode,
                    )
                    if fit_dir is not None:
                        save_fit_history(fit_dir, hist)
                        (fit_dir / 'resource.json').write_text(
                            json.dumps(resource, indent=2), encoding='utf-8'
                        )
                        _atomic_npy(
                            fit_dir / 'held_logits.npy', np.asarray(logits, dtype=np.float32)
                        )
                        _atomic_json(
                            fit_dir / 'COMPLETE.json',
                            {
                                'status': 'COMPLETE',
                                'completed_epochs': len(hist),
                                'metadata': expected_metadata,
                                'held_logits_sha256': sha256_file(fit_dir / 'held_logits.npy'),
                            },
                        )
                        print(f'[COMPLETE] {arm} seed={seed} fold={fold}', flush=True)
                probs = softmax(logits)
                pred = probs.argmax(1)
                histories[(int(seed), int(fold), arm)] = {'epochs': hist, 'resource': resource}
                for j, i in enumerate(held_idx):
                    records.append(
                        {
                            'uid': uids[i],
                            'roi_id': str(rois[i]),
                            'true_class': int(labels[i]),
                            'seed': int(seed),
                            'fold': int(fold),
                            'arm': arm,
                            'x': float(coords[i, 0]),
                            'y': float(coords[i, 1]),
                            'logits': logits[j].astype(np.float64),
                            'probs': probs[j].astype(np.float64),
                            'pred_class': int(pred[j]),
                            'config_hash': config_hash,
                            'roster_hash': roster_hash,
                            'fold_hash': fold_hash,
                        }
                    )
        for arm in arms:
            validate_oof(uids, records, int(seed), arm)
    return records, histories
