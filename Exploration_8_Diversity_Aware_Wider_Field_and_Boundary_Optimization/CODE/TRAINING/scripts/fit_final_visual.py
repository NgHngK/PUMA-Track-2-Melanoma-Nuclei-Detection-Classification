from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'COMMON' / 'src'))

import argparse, json
import torch
from puma_exploration8.checkpoint import save_checkpoint
from puma_exploration8.config import load_protocol, sha256_file, scientific_config_bundle_hash
from puma_exploration8.experiments import build_stage2_model, train_full_model
from puma_exploration8.protection import write_freeze, file_sha256
from puma_exploration8.stageio import load_role_arrays, manifest_sha256


def main():
    p = argparse.ArgumentParser(
        description='Final fixed-budget full-TRAIN refit after P8-G visual selection'
    )
    p.add_argument('--manifest', required=True)
    p.add_argument('--cache-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--selected-wide', choices=['gaussian192', 'annular192'])
    p.add_argument('--route', choices=['NONE', 'A', 'C'], required=True)
    p.add_argument('--tiera', action='store_true')
    p.add_argument('--exposure', choices=['inverse', 'group'], default='inverse')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--strict-hash', action='store_true')
    a = p.parse_args()
    protocol = load_protocol(ROOT / 'CONFIGS' / '00_PROTOCOL.json')
    opt = protocol['optimizer']
    allowed = [int(x) for x in protocol['seeds']]
    if a.seed not in allowed:
        raise ValueError(f'seed must be preregistered: {allowed}')
    if a.route != 'NONE' and not a.selected_wide:
        raise ValueError('selected-wide is required for route A/C')
    if a.route == 'NONE' and a.selected_wide:
        raise ValueError('selected-wide must be omitted for route NONE baseline')
    wide = a.selected_wide if a.route != 'NONE' else None
    df, _, _, arrays = load_role_arrays(a.manifest, a.cache_dir, 'TRAIN', wide, a.strict_hash)
    dev = a.device if a.device != 'cuda' or torch.cuda.is_available() else 'cpu'
    model = build_stage2_model(a.seed, a.tiera, None if a.route == 'NONE' else a.route)
    model, normalizer, audit = train_full_model(
        model,
        arrays,
        seed=a.seed,
        epochs=int(opt['epochs']),
        lr=float(opt['lr']),
        wd=float(opt['weight_decay']),
        batch=int(opt['batch']),
        exposure=a.exposure,
        device=dev,
    )
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / f'final_visual_seed{a.seed}.pt'
    protocol_sha = sha256_file(ROOT / 'CONFIGS' / '00_PROTOCOL.json')
    bundle_sha, config_hashes = scientific_config_bundle_hash(ROOT / 'CONFIGS')
    metadata = {
        'seed': a.seed,
        'normalizer': normalizer.to_dict(),
        'selected_wide': wide,
        'route': a.route,
        'use_tiera': bool(a.tiera),
        'exposure': a.exposure,
        'manifest_sha256': manifest_sha256(a.manifest),
        'protocol_sha256': protocol_sha,
        'config_bundle_sha256': bundle_sha,
        'effective_source_prior': audit['effective_source_prior'],
    }
    save_checkpoint(ckpt, model, metadata)
    visual_freeze = out / 'VISUAL_FREEZE.json'
    write_freeze(
        visual_freeze,
        {
            'kind': 'VISUAL_FREEZE',
            'protocol_sha256': protocol_sha,
            'config_bundle_sha256': bundle_sha,
            'config_hashes': config_hashes,
            'manifest_sha256': metadata['manifest_sha256'],
            'checkpoint_sha256': file_sha256(ckpt),
            'checkpoint_name': ckpt.name,
            'seed': a.seed,
            'selected_wide': wide,
            'route': a.route,
            'use_tiera': bool(a.tiera),
            'exposure': a.exposure,
            'epochs': int(opt['epochs']),
            'lr': float(opt['lr']),
            'weight_decay': float(opt['weight_decay']),
            'batch': int(opt['batch']),
        },
    )
    vh = file_sha256(visual_freeze)
    training_meta = out / 'TRAINING_METADATA.json'
    training_meta.write_text(
        json.dumps(
            {
                'visual_freeze_sha256': vh,
                'checkpoint_sha256': file_sha256(ckpt),
                'seed': a.seed,
                'effective_source_prior': audit['effective_source_prior'],
                'sampled_class_counts': audit['sampled_class_counts'],
                'feature_storage': audit['feature_storage'],
                'cuda_memory': audit.get('cuda_memory'),
                'normalizer_fold_id': audit['normalizer_fold_id'],
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                'status': 'OK',
                'checkpoint': str(ckpt),
                'visual_freeze': str(visual_freeze),
                'visual_freeze_sha256': vh,
                'training_metadata': str(training_meta),
                'cuda_memory': audit.get('cuda_memory'),
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
