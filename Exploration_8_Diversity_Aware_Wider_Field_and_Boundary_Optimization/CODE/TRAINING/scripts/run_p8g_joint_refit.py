from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'COMMON' / 'src'))

import argparse, json
import numpy as np
import torch
from puma_exploration8.config import load_json, load_protocol
from puma_exploration8.grouping import load_fold_payload, fold_vector
from puma_exploration8.experiments import subset_arrays, train_model, build_stage2_model
from puma_exploration8.metrics import classification_metrics
from puma_exploration8.stageio import load_role_arrays


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--cache-dir', required=True)
    p.add_argument('--folds', required=True)
    p.add_argument('--selected-wide', choices=['gaussian192', 'annular192'], required=True)
    p.add_argument('--route', choices=['A', 'C'], default='A')
    p.add_argument('--tiera', action='store_true')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--exposure', choices=['inverse', 'group'], default='inverse')
    p.add_argument('--strict-hash', action='store_true')
    a = p.parse_args()
    protocol = load_protocol(ROOT / 'CONFIGS' / '00_PROTOCOL.json')
    stage = load_json(ROOT / 'CONFIGS' / '70_P8G_JOINT_REFIT.json')
    opt = protocol['optimizer']
    if stage.get('comparators') != ['J0_local', 'J1_local_plus_selected_wider'] or not stage.get(
        'same_folds_optimizer_seed_budget'
    ):
        raise ValueError('P8-G config/code contract mismatch')
    df, _, _, arrays = load_role_arrays(
        a.manifest, a.cache_dir, 'TRAIN', a.selected_wide, a.strict_hash
    )
    fold = load_fold_payload(a.folds, df, 'TRAIN')
    fv = fold_vector(df, fold, 'TRAIN')
    dev = a.device if a.device != 'cuda' or torch.cuda.is_available() else 'cpu'
    results = []
    for seed in map(int, protocol['seeds']):
        for k in range(int(fold['k'])):
            ta, va = np.where(fv != k)[0], np.where(fv == k)[0]
            for name, use_w in [('J0', False), ('J1', True)]:
                model = build_stage2_model(seed, a.tiera, a.route if use_w else None)
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
                        'model': name,
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
    (out / 'P8G_RESULTS.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
