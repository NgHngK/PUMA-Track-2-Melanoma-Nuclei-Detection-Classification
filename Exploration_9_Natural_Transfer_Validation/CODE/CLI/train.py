from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from CODE.DATA.manifest import read_manifest
from CODE.DATA.folds import make_group_folds, choose_fold_count
from CODE.DATA.tier_a_extract import load_tier_a
from CODE.FEATURES.cache_reader import FeatureCache
from CODE.TRAINING.run_transfer import run_oof_transfer
from CODE.EVALUATION.evaluate_oof import evaluate_oof_records
from CODE.EVALUATION.transfer_summary import summarize_natural_transfer
from CODE.COMMON.constants import SEEDS, BOOTSTRAP_REPS, CONTEXT_INIT_PROVENANCE
from CODE.COMMON.hashing import canonical_json_hash, sha256_file
from CODE.COMMON.exceptions import ScientificContractError, ResumeIntegrityError


def _jsonable_metric(m):
    return {
        k: (
            v.tolist()
            if isinstance(v, np.ndarray)
            else (
                {r: [x.tolist() for x in vv] for r, vv in v.items()} if k == 'counts_by_roi' else v
            )
        )
        for k, v in m.items()
    }


def _save_oof(path: Path, records):
    rows = sorted(records, key=lambda r: r['uid'])
    np.savez_compressed(
        path,
        uid=np.asarray([r['uid'] for r in rows], dtype=str),
        roi_id=np.asarray([r['roi_id'] for r in rows], dtype=str),
        true_class=np.asarray([r['true_class'] for r in rows], dtype=np.int16),
        fold=np.asarray([r['fold'] for r in rows], dtype=np.int16),
        x=np.asarray([r['x'] for r in rows], dtype=np.float32),
        y=np.asarray([r['y'] for r in rows], dtype=np.float32),
        logits=np.stack([r['logits'] for r in rows]).astype(np.float32),
    )


def _load_preparation_status(manifest: Path):
    p = manifest.parent / 'dataset_summary.json'
    if not p.is_file():
        raise ScientificContractError(
            f'Missing preparation audit {p}; run CODE.CLI.prepare_dataset first'
        )
    obj = json.loads(p.read_text(encoding='utf-8'))
    return p, obj


def _aggregate_resources(histories_all):
    rows = []
    for group in histories_all.values():
        for key, payload in group.items():
            r = dict(payload.get('resource', {}))
            r['fit_key'] = str(key)
            rows.append(r)
    if not rows:
        return {'fit_count': 0}

    def mx(k):
        vals = [x[k] for x in rows if x.get(k) is not None]
        return max(vals) if vals else None

    return {
        'fit_count': len(rows),
        'total_fit_wall_seconds': float(sum(float(x.get('wall_seconds', 0.0)) for x in rows)),
        'max_fit_wall_seconds': mx('wall_seconds'),
        'peak_process_rss_kb': mx('peak_rss_kb'),
        'max_cuda_peak_allocated': mx('cuda_peak_allocated'),
        'max_cuda_peak_reserved': mx('cuda_peak_reserved'),
        'cpu_count': os.cpu_count(),
    }


def _check_or_write_run_contract(path: Path, payload: dict) -> None:
    if path.is_file():
        previous = json.loads(path.read_text(encoding='utf-8'))
        if previous.get('config_hash') != payload['config_hash']:
            raise ResumeIntegrityError(
                f'Existing training output uses a different run contract: {path}. '
                'Use a different output directory for a different experiment.'
            )
        # Runtime-only fields may change without changing the scientific recipe.
        previous.update({k: v for k, v in payload.items() if k.endswith('_last_run')})
        path.write_text(json.dumps(previous, indent=2, sort_keys=True), encoding='utf-8')
        return
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def main():
    # Keep the scientific path in true FP32. Throughput optimizations below do not
    # enable BF16/FP16/TF32, which would change numerical representation.
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
    p = argparse.ArgumentParser(description='Run the locked E9 natural-transfer OOF comparison')
    p.add_argument('--manifest', required=True)
    p.add_argument('--feature-dir', required=True)
    p.add_argument('--tier-a-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument(
        '--folds', type=int, default=None, help='Default: metadata-only feasible count, max 5'
    )
    p.add_argument('--bootstrap-reps', type=int, default=BOOTSTRAP_REPS)
    p.add_argument('--cpu-threads', type=int, default=0, help='0 uses all available CPU cores')
    p.add_argument('--context-scale-init', type=float, required=True)
    p.add_argument('--context-init-provenance', required=True)
    p.add_argument(
        '--allow-exploratory-provenance',
        action='store_true',
        help='Allow an all-official-training exploratory roster without audited historical E8 roles',
    )
    p.add_argument(
        '--execution-mode',
        choices=('fast', 'strict'),
        default='fast',
        help='FAST is default: same FP32 recipe with lower synchronization/setup overhead',
    )
    a = p.parse_args()

    threads = int(a.cpu_threads) if int(a.cpu_threads) > 0 else max(1, int(os.cpu_count() or 1))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(min(4, threads))
    except RuntimeError:
        pass
    print(
        f'[SETUP] execution_mode={a.execution_mode} cpu_threads={threads} device={a.device}',
        flush=True,
    )
    manifest = Path(a.manifest).resolve()
    print(f'[SETUP] reading manifest: {manifest}', flush=True)
    prep_path, prep = _load_preparation_status(manifest)
    strict_roles = prep.get('provenance_mode') == 'STRICT_AUDITED_ROI_ROLES'
    if not strict_roles and not a.allow_exploratory_provenance:
        raise ScientificContractError(
            f"Preparation provenance_mode={prep.get('provenance_mode')!r}. Strict natural transfer requires audited ROI roles; add --allow-exploratory-provenance only for explicitly exploratory official-training OOF."
        )
    context_scale_init = float(a.context_scale_init)
    context_init_provenance = str(a.context_init_provenance)

    rows = read_manifest(manifest, require_images=False, transfer_only=True)
    print(f'[SETUP] manifest rows={len(rows):,}', flush=True)
    uids = [r.uid for r in rows]
    rois = np.asarray([r.roi_id for r in rows])
    labels = np.asarray([r.label for r in rows], dtype=np.int64)
    coords = np.asarray([[r.x, r.y] for r in rows], dtype=np.float64)
    # Read cache metadata first. If the whole training run is already complete we can
    # skip without loading ~GB of feature arrays back into RAM.
    print('[SETUP] reading feature metadata', flush=True)
    local_meta = FeatureCache(a.feature_dir, 'gaussian96', verify_hash=False)
    wide_meta = FeatureCache(a.feature_dir, 'gaussian192', verify_hash=False)
    if (
        local_meta.meta['contract']['encoder_sha256']
        != wide_meta.meta['contract']['encoder_sha256']
    ):
        raise ScientificContractError('Local/wide caches use different encoder checkpoint hashes')
    tier_meta = json.loads((Path(a.tier_a_dir) / 'tier_a.json').read_text(encoding='utf-8'))
    feature_hashes = {
        'gaussian96': local_meta.meta['payload_sha256'],
        'gaussian192': wide_meta.meta['payload_sha256'],
        'tier_a': tier_meta['payload_sha256'],
    }

    k = a.folds if a.folds is not None else choose_fold_count(labels, rois)
    folds = make_group_folds(labels, rois, seed=17, n_splits=k)
    out = Path(a.output_dir)
    (out / 'OOF').mkdir(parents=True, exist_ok=True)
    (out / 'METRICS').mkdir(parents=True, exist_ok=True)
    (out / 'RESOURCES').mkdir(parents=True, exist_ok=True)
    (out / 'HISTORIES').mkdir(parents=True, exist_ok=True)

    folds_path = out / 'folds.npy'
    if folds_path.is_file():
        old = np.load(folds_path, allow_pickle=False)
        if not np.array_equal(old, folds):
            raise ResumeIntegrityError(
                'Existing folds.npy differs from the current deterministic fold assignment'
            )
    else:
        np.save(folds_path, folds)
        (out / 'folds.csv').write_text(
            'uid,roi_id,fold\n'
            + '\n'.join(f'{u},{r},{int(f)}' for u, r, f in zip(uids, rois, folds))
            + '\n',
            encoding='utf-8',
        )

    scientific_payload = {
        'scientific_recipe': 'E9_locked_natural_transfer',
        'seeds': list(SEEDS),
        'fold_count': int(k),
        'arms': ['ANCHOR', 'ANCHOR_CONTEXT_C'],
        'context_scale_init': context_scale_init,
        'context_init_provenance': context_init_provenance,
        'preparation_summary_sha256': sha256_file(prep_path),
        'provenance_mode': prep.get('provenance_mode'),
        'feature_hashes': feature_hashes,
    }
    config_hash = canonical_json_hash(scientific_payload)
    run_contract = {
        **scientific_payload,
        'config_hash': config_hash,
        'cpu_threads_last_run': threads,
        'execution_mode_last_run': a.execution_mode,
    }
    _check_or_write_run_contract(out / 'RUN_CONTRACT.json', run_contract)

    complete_marker = out / 'TRAINING_COMPLETE.json'
    if complete_marker.is_file():
        done = json.loads(complete_marker.read_text(encoding='utf-8'))
        if done.get('config_hash') != config_hash or done.get('feature_hashes') != feature_hashes:
            raise ResumeIntegrityError(
                'TRAINING_COMPLETE.json does not match the current run contract/features'
            )
        if (out / 'NATURAL_TRANSFER_SUMMARY.json').is_file():
            print('[SKIP] Training already COMPLETE. Nothing to resume.', flush=True)
            return

    # For an unfinished run, verify payload hashes once before loading features.
    print('[SETUP] verifying feature caches (one pass)', flush=True)
    local = FeatureCache(a.feature_dir, 'gaussian96', verify_hash=True)
    wide = FeatureCache(a.feature_dir, 'gaussian192', verify_hash=True)
    h_local = local.get(uids)
    h_wide = wide.get(uids)
    tier = load_tier_a(a.tier_a_dir, uids)
    print(
        f'[SETUP] features loaded: gaussian96={h_local.shape} gaussian192={h_wide.shape} tier_a={tier.shape}',
        flush=True,
    )

    seed_metrics = {}
    histories_all = {}
    if a.execution_mode == 'fast':
        # One orchestration pass avoids rescanning ~1.2 GB of arrays and rebuilding
        # roster/fold hashes six times (3 seeds x 2 arms). Each individual fit still
        # uses the same batch-64 optimizer trajectory and independent deterministic seed.
        print('[TRAIN] FAST one-pass OOF orchestration', flush=True)
        records, hist = run_oof_transfer(
            uids=uids,
            rois=rois,
            labels=labels,
            coords=coords,
            h_local=h_local,
            h_wide=h_wide,
            tier_raw=tier,
            folds=folds,
            seeds=SEEDS,
            arms=('ANCHOR', 'ANCHOR_CONTEXT_C'),
            device=a.device,
            config_hash=config_hash,
            context_scale_init=context_scale_init,
            resume_root=out / 'HISTORIES',
            feature_hashes=feature_hashes,
            execution_mode='fast',
        )
        all_metrics = evaluate_oof_records(records)
        for seed in SEEDS:
            for arm in ('ANCHOR', 'ANCHOR_CONTEXT_C'):
                met = all_metrics[(arm, seed)]
                seed_metrics[(seed, arm)] = met
                group = {k: v for k, v in hist.items() if k[0] == int(seed) and k[2] == arm}
                histories_all[f'{seed}:{arm}'] = group
                arm_records = [r for r in records if r['seed'] == int(seed) and r['arm'] == arm]
                _save_oof(out / 'OOF' / f'{arm}_seed{seed}.npz', arm_records)
                (out / 'METRICS' / f'{arm}_seed{seed}.json').write_text(
                    json.dumps(_jsonable_metric(met), indent=2), encoding='utf-8'
                )
        # Per-fit history/resource files are already persisted by run_oof_transfer.
        del records
    else:
        for seed in SEEDS:
            for arm in ('ANCHOR', 'ANCHOR_CONTEXT_C'):
                print(f'[TRAIN] STRICT seed={seed} arm={arm}', flush=True)
                records, hist = run_oof_transfer(
                    uids=uids,
                    rois=rois,
                    labels=labels,
                    coords=coords,
                    h_local=h_local,
                    h_wide=h_wide,
                    tier_raw=tier,
                    folds=folds,
                    seeds=(seed,),
                    arms=(arm,),
                    device=a.device,
                    config_hash=config_hash,
                    context_scale_init=context_scale_init,
                    resume_root=out / 'HISTORIES',
                    feature_hashes=feature_hashes,
                    execution_mode='strict',
                )
                met = evaluate_oof_records(records)[(arm, seed)]
                seed_metrics[(seed, arm)] = met
                histories_all[f'{seed}:{arm}'] = hist
                # Per-fit history/resource files are already persisted by run_oof_transfer.
                _save_oof(out / 'OOF' / f'{arm}_seed{seed}.npz', records)
                (out / 'METRICS' / f'{arm}_seed{seed}.json').write_text(
                    json.dumps(_jsonable_metric(met), indent=2), encoding='utf-8'
                )
                del records
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    summary = summarize_natural_transfer(
        seed_metrics, labels, rois, bootstrap_reps=a.bootstrap_reps
    )
    resources = _aggregate_resources(histories_all)
    scientific_status = (
        'STRICT_ROLE_AUDITED_BUT_CONTEXT_INIT_EXTERNAL'
        if strict_roles
        and context_init_provenance not in {CONTEXT_INIT_PROVENANCE, 'USER_ACKNOWLEDGED_UNVERIFIED'}
        else 'EXPLORATORY'
    )
    summary.update(
        {
            'fold_count': int(k),
            'config_hash': config_hash,
            'provenance_mode': prep.get('provenance_mode'),
            'context_scale_init': context_scale_init,
            'context_init_provenance': context_init_provenance,
            'scientific_status': scientific_status,
            'execution_mode': a.execution_mode,
            'resources': resources,
            'note': 'ROI-grouped official-training OOF. Patient-level independence is unverified.',
        }
    )
    (out / 'NATURAL_TRANSFER_SUMMARY.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8'
    )
    serial = {key: {str(k2): v2 for k2, v2 in val.items()} for key, val in histories_all.items()}
    (out / 'TRAINING_HISTORIES.json').write_text(json.dumps(serial, indent=2), encoding='utf-8')
    (out / 'RESOURCES' / 'FIT_RESOURCE_SUMMARY.json').write_text(
        json.dumps(resources, indent=2), encoding='utf-8'
    )
    complete_marker.write_text(
        json.dumps(
            {'status': 'COMPLETE', 'config_hash': config_hash, 'feature_hashes': feature_hashes},
            indent=2,
        ),
        encoding='utf-8',
    )
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
