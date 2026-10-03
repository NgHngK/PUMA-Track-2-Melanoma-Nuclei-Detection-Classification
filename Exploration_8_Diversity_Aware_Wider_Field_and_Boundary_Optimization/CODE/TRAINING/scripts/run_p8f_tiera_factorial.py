from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'COMMON' / 'src'))

import argparse, json
import numpy as np
import torch
from puma_exploration8.config import load_json, load_protocol
from puma_exploration8.grouping import load_fold_payload, fold_vector
from puma_exploration8.experiments import Arrays, subset_arrays, train_model, build_stage2_model
from puma_exploration8.metrics import classification_metrics
from puma_exploration8.stageio import load_role_arrays


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--cache-dir', required=True)
    p.add_argument('--folds', required=True)
    p.add_argument('--representation', choices=['gaussian192', 'annular192'], required=True)
    p.add_argument('--route', choices=['A', 'C'], default='A')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--exposure', choices=['inverse', 'group'], default='inverse')
    p.add_argument('--strict-hash', action='store_true')
    a = p.parse_args()
    protocol = load_protocol(ROOT / 'CONFIGS' / '00_PROTOCOL.json')
    stage = load_json(ROOT / 'CONFIGS' / '60_P8F_TIERA_FACTORIAL.json')
    opt = protocol['optimizer']
    if (
        stage.get('corners') != ['L', 'L+A', 'L+W', 'L+A+W']
        or len(stage.get('controls_after_W', [])) != 2
    ):
        raise ValueError('P8-F config/code contract mismatch')
    df, _, _, arrays = load_role_arrays(
        a.manifest, a.cache_dir, 'TRAIN', a.representation, a.strict_hash
    )
    fold = load_fold_payload(a.folds, df, 'TRAIN')
    fv = fold_vector(df, fold, 'TRAIN')
    dev = a.device if a.device != 'cuda' or torch.cuda.is_available() else 'cpu'
    results = []
    rng = np.random.default_rng(1701)
    W = rng.normal(0, 1 / np.sqrt(arrays.local.shape[1]), size=(arrays.local.shape[1], 16)).astype(
        np.float32
    )
    placebo = np.tanh(arrays.local @ W).astype(np.float32)
    for seed in map(int, protocol['seeds']):
        for k in range(int(fold['k'])):
            ta, va = np.where(fv != k)[0], np.where(fv == k)[0]
            for name, use_a, use_w in [
                ('L', False, False),
                ('L+A', True, False),
                ('L+W', False, True),
                ('L+A+W', True, True),
            ]:
                model = build_stage2_model(seed, use_a, a.route if use_w else None)
                _, norm, z, _, audit = train_model(
                    model,
                    subset_arrays(arrays, ta),
                    subset_arrays(arrays, va),
                    seed=seed,
                    epochs=int(opt['epochs']),
                    lr=float(opt['lr']),
                    wd=float(opt['weight_decay']),
                    batch=int(opt['batch']),
                    exposure=a.exposure,
                    device=dev,
                    track_history=False,
                )
                met = classification_metrics(arrays.y[va], z)
                results.append(
                    {
                        'seed': seed,
                        'fold': k,
                        'corner': name,
                        'control': 'real',
                        'macro_f1': met['macro_f1_fixed10'],
                        'recall': met['recall'],
                        'normalizer': norm.to_dict(),
                        'parameter_audit': audit,
                        'frozen_fold_sha256': fold['sha256'],
                        'exposure': a.exposure,
                    }
                )
            rr = np.random.default_rng(seed + 10000 * k)
            shuffled = arrays.bio.copy()
            shuffled[ta] = shuffled[ta][rr.permutation(len(ta))]
            for cname, bio in [
                ('L+A+W_SHUFFLED_TIERA', shuffled),
                ('L+A+W_APPEARANCE_PLACEBO', placebo),
            ]:
                control = Arrays(
                    arrays.uids, arrays.y, arrays.groups, arrays.local, bio, arrays.wider
                )
                model = build_stage2_model(seed, True, a.route)
                _, norm, z, _, audit = train_model(
                    model,
                    subset_arrays(control, ta),
                    subset_arrays(control, va),
                    seed=seed,
                    epochs=int(opt['epochs']),
                    lr=float(opt['lr']),
                    wd=float(opt['weight_decay']),
                    batch=int(opt['batch']),
                    exposure=a.exposure,
                    device=dev,
                    track_history=False,
                )
                met = classification_metrics(arrays.y[va], z)
                results.append(
                    {
                        'seed': seed,
                        'fold': k,
                        'corner': cname,
                        'control': (
                            'shuffle_train_only'
                            if 'SHUFFLED' in cname
                            else 'matched_capacity_label_free'
                        ),
                        'macro_f1': met['macro_f1_fixed10'],
                        'recall': met['recall'],
                        'normalizer': norm.to_dict(),
                        'parameter_audit': audit,
                        'frozen_fold_sha256': fold['sha256'],
                        'exposure': a.exposure,
                    }
                )
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'P8F_RESULTS.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
