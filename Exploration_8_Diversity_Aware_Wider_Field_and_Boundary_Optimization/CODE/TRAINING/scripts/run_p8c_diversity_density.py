from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'COMMON' / 'src'))

import argparse, json
import numpy as np
import torch
from puma_exploration8.config import load_json, load_protocol
from puma_exploration8.grouping import load_fold_payload, fold_vector
from puma_exploration8.diversity import plan_classes, paired_rosters, validate_pairs, roster_hash
from puma_exploration8.manifest import read_manifest
from puma_exploration8.config import sha256_json, sha256_file
from puma_exploration8.experiments import subset_arrays, train_model, build_stage2_model
from puma_exploration8.metrics import classification_metrics
from puma_exploration8.stageio import load_role_arrays


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--cache-dir')
    p.add_argument('--folds', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--strict-hash', action='store_true')
    p.add_argument(
        '--preflight-only',
        action='store_true',
        help='Generate and validate every roster without reading caches or fitting models',
    )
    a = p.parse_args()
    protocol = load_protocol(ROOT / 'CONFIGS' / '00_PROTOCOL.json')
    cfg = load_json(ROOT / 'CONFIGS' / '30_P8C_DIVERSITY_DENSITY.json')
    df = read_manifest(a.manifest, require_images=False)
    tr = df[df.role == 'TRAIN'].reset_index(drop=True)
    fold = load_fold_payload(a.folds, df, 'TRAIN')
    fv = fold_vector(df, fold, 'TRAIN')
    dev = a.device if a.device != 'cuda' or torch.cuda.is_available() else 'cpu'
    opt = protocol['optimizer']
    seeds = [int(x) for x in protocol['seeds']]
    roster_seeds = [int(x) for x in cfg['roster_seeds']]
    plan = plan_classes(tr, fv, cfg)
    binding = {
        'design_version': 2,
        'classes': plan,
        'config': cfg,
        'manifest_sha256': sha256_file(a.manifest),
        'frozen_fold_sha256': fold['sha256'],
        'protocol_sha256': sha256_json(protocol),
        'implementation_sha256': sha256_file(ROOT / 'COMMON/src/puma_exploration8/diversity.py'),
    }
    binding['sha256'] = sha256_json(binding)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / 'FEASIBILITY_PLAN.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != binding:
        raise ValueError(
            'output directory has a different frozen design; use a new output directory'
        )
    if (out / 'P8C_RESULTS.json').exists():
        raise FileExistsError('results already exist; use a new output directory')
    roster_dir = out / 'ROSTERS'
    roster_dir.mkdir(exist_ok=True)
    rosters, generation = {}, []
    for draw_seed in roster_seeds:
        for k in range(int(fold['k'])):
            train_df = tr.iloc[np.where(fv != k)[0]].copy()
            paired = paired_rosters(train_df, plan, draw_seed)
            validate_pairs(paired, train_df, plan)
            for arm, roster in paired.items():
                rh = roster_hash(roster)
                path = roster_dir / f'{arm}_draw{draw_seed}_fold{k}_{rh[:10]}.csv'
                rosters[(draw_seed, k, arm)] = roster
                generation.append(
                    {
                        'draw_seed': draw_seed,
                        'fold': k,
                        'arm': arm,
                        'status': 'OK',
                        'roster_hash': rh,
                        'roster_file': str(path.relative_to(out)),
                    }
                )
    # No model or cache is touched until the complete planned design passes.
    plan_path.write_text(json.dumps(binding, indent=2))
    for rec in generation:
        rosters[(rec['draw_seed'], rec['fold'], rec['arm'])].to_csv(
            out / rec['roster_file'], index=False
        )
    generation_payload = {
        'design_sha256': binding['sha256'],
        'frozen_fold_sha256': fold['sha256'],
        'draw_seeds': roster_seeds,
        'records': generation,
    }
    (out / 'ROSTER_GENERATION.json').write_text(json.dumps(generation_payload, indent=2))
    print(
        json.dumps({'preflight': 'PASS', 'rosters': len(generation), 'classes': plan}, indent=2),
        flush=True,
    )
    if a.preflight_only:
        return
    if not a.cache_dir:
        p.error('--cache-dir is required for training')
    _, _, _, arrays = load_role_arrays(a.manifest, a.cache_dir, 'TRAIN', strict_hash=a.strict_hash)

    pos = {u: i for i, u in enumerate(arrays.uids)}
    draws_per_epoch = sum(x['matched_nuclei'] for x in plan)
    results = []
    evaluated = {}
    for rec in generation:
        if rec['status'] != 'OK':
            results.append(rec)
            continue
        roster = rosters[(rec['draw_seed'], rec['fold'], rec['arm'])]
        train_ix = np.asarray([pos[str(u)] for u in roster.uid], int)
        val_ix = np.where(fv == rec['fold'])[0]
        for seed in seeds:
            key = (rec['fold'], rec['roster_hash'], seed)
            if key in evaluated:
                results.append({**rec, **evaluated[key], 'reused_identical_roster': True})
                continue
            model = build_stage2_model(seed, True, None)
            _, norm, z, _, audit = train_model(
                model,
                subset_arrays(arrays, train_ix),
                subset_arrays(arrays, val_ix),
                seed=seed,
                epochs=int(opt['epochs']),
                lr=float(opt['lr']),
                wd=float(opt['weight_decay']),
                batch=int(opt['batch']),
                draws=draws_per_epoch,
                exposure='inverse',
                device=dev,
                track_history=False,
            )
            met = classification_metrics(arrays.y[val_ix], z)
            evaluated[key] = {
                'model_seed': seed,
                'optimization_draws_per_epoch': draws_per_epoch,
                'macro_f1': met['macro_f1_fixed10'],
                'recall': met['recall'],
                'normalizer': norm.to_dict(),
                'parameter_audit': audit,
            }
            results.append({**rec, **evaluated[key], 'reused_identical_roster': False})
    contrasts = []
    for draw_seed in roster_seeds:
        for k in range(int(fold['k'])):
            for seed in seeds:
                matched = {
                    r['arm']: r
                    for r in results
                    if r['draw_seed'] == draw_seed and r['fold'] == k and r['model_seed'] == seed
                }
                for name, low, high in [
                    ('diversity', 'div_low', 'div_high'),
                    ('density', 'density_low', 'density_high'),
                ]:
                    contrasts.append(
                        {
                            'contrast': name,
                            'draw_seed': draw_seed,
                            'fold': k,
                            'model_seed': seed,
                            'delta_macro_f1': matched[high]['macro_f1'] - matched[low]['macro_f1'],
                            'delta_recall': [
                                h - l
                                for h, l in zip(matched[high]['recall'], matched[low]['recall'])
                            ],
                        }
                    )
    (out / 'P8C_RESULTS.json').write_text(
        json.dumps(
            {
                'design_sha256': binding['sha256'],
                'frozen_fold_sha256': fold['sha256'],
                'classes': plan,
                'unique_fits': len(evaluated),
                'paired_contrasts': contrasts,
                'results': results,
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                'generated': sum(x['status'] == 'OK' for x in generation),
                'evaluated': sum('model_seed' in x for x in results),
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
