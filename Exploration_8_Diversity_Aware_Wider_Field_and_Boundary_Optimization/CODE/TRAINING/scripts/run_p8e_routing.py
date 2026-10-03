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
from puma_exploration8.checkpoint import save_checkpoint
from puma_exploration8.stageio import load_role_arrays


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--cache-dir', required=True)
    p.add_argument('--folds', required=True)
    p.add_argument('--representation', choices=['gaussian192', 'annular192'], required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--exposure', choices=['inverse', 'group'], default='inverse')
    p.add_argument('--strict-hash', action='store_true')
    a = p.parse_args()
    protocol = load_protocol(ROOT / 'CONFIGS' / '00_PROTOCOL.json')
    stage = load_json(ROOT / 'CONFIGS' / '50_P8E_ROUTING.json')
    opt = protocol['optimizer']
    if stage.get('candidates') != ['A_simple_residual', 'C_rank8_local_x_wider'] or not stage.get(
        'third_candidate_forbidden'
    ):
        raise ValueError('P8-E config/code contract mismatch')
    df, _, _, arrays = load_role_arrays(
        a.manifest, a.cache_dir, 'TRAIN', a.representation, a.strict_hash
    )
    fold = load_fold_payload(a.folds, df, 'TRAIN')
    fv = fold_vector(df, fold, 'TRAIN')
    dev = a.device if a.device != 'cuda' or torch.cuda.is_available() else 'cpu'
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in map(int, protocol['seeds']):
        for k in range(int(fold['k'])):
            ta, va = np.where(fv != k)[0], np.where(fv == k)[0]
            train, val = subset_arrays(arrays, ta), subset_arrays(arrays, va)
            base = build_stage2_model(seed, True, None)
            base, norm, base_z, _, base_audit = train_model(
                base,
                train,
                val,
                seed=seed,
                epochs=int(opt['epochs']),
                lr=float(opt['lr']),
                wd=float(opt['weight_decay']),
                batch=int(opt['batch']),
                exposure=a.exposure,
                device=dev,
                track_history=False,
            )
            base_ck = out / 'BASELINES' / f'seed{seed}_fold{k}.pt'
            base_ck.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                base_ck,
                base,
                {
                    'seed': seed,
                    'fold': k,
                    'normalizer': norm.to_dict(),
                    'architecture': 'exploration8_local_gaussian96_tiera_a5_rank8',
                    'fold_sha256': fold['sha256'],
                    'exposure': a.exposure,
                },
            )
            base_m = classification_metrics(val.y, base_z)
            for kind in ['A', 'C']:
                model = build_stage2_model(seed, True, kind)
                model.local.load_state_dict(base.local.state_dict())
                for q in model.local.parameters():
                    q.requires_grad_(False)
                model, norm2, z, _, audit = train_model(
                    model,
                    train,
                    val,
                    seed=seed,
                    epochs=int(opt['epochs']),
                    lr=float(opt['lr']),
                    wd=float(opt['weight_decay']),
                    batch=int(opt['batch']),
                    exposure=a.exposure,
                    device=dev,
                    track_history=False,
                    trainable_filter=lambda n: n.startswith('wider_route.'),
                )
                met = classification_metrics(val.y, z)
                results.append(
                    {
                        'seed': seed,
                        'fold': k,
                        'kind': kind,
                        'macro_f1': met['macro_f1_fixed10'],
                        'recall': met['recall'],
                        'baseline_macro_f1': base_m['macro_f1_fixed10'],
                        'delta_vs_base': met['macro_f1_fixed10'] - base_m['macro_f1_fixed10'],
                        'normalizer': norm2.to_dict(),
                        'parameter_audit': audit,
                        'baseline_checkpoint': str(base_ck.relative_to(out)),
                        'frozen_fold_sha256': fold['sha256'],
                        'exposure': a.exposure,
                    }
                )
    (out / 'P8E_RESULTS.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
