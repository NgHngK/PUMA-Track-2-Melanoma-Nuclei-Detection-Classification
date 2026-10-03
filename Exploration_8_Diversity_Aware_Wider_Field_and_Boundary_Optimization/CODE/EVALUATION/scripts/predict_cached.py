from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'COMMON' / 'src'))

import argparse, json
import numpy as np
import torch
from puma_exploration8.boundary import apply_h1, temperature
from puma_exploration8.checkpoint import load_checkpoint
from puma_exploration8.experiments import build_stage2_model
from puma_exploration8.normalization import TierANormalizer
from puma_exploration8.protection import require_freeze, file_sha256
from puma_exploration8.config import sha256_file, scientific_config_bundle_hash
from puma_exploration8.runtime import can_reside_on_cuda, cuda_memory_stats, release_cuda
from puma_exploration8.stageio import load_role_arrays


def main():
    p = argparse.ArgumentParser(
        description='Predict one manifest role from compact Exploration-8 caches'
    )
    p.add_argument('--manifest', required=True)
    p.add_argument('--cache-dir', required=True)
    p.add_argument('--role', choices=['CAL', 'VALIDATE', 'TEST'], required=True)
    p.add_argument('--visual-freeze', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--boundary-freeze')
    p.add_argument('--complete-freeze')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--strict-hash', action='store_true')
    a = p.parse_args()
    visual = require_freeze(
        a.visual_freeze,
        'VISUAL_FREEZE',
        required_keys=[
            'checkpoint_sha256',
            'seed',
            'selected_wide',
            'route',
            'use_tiera',
            'manifest_sha256',
        ],
    )
    visual_hash = file_sha256(a.visual_freeze)
    if visual['protocol_sha256'] != sha256_file(ROOT / 'CONFIGS' / '00_PROTOCOL.json'):
        raise ValueError('current protocol differs from visual freeze')
    bundle, _ = scientific_config_bundle_hash(ROOT / 'CONFIGS')
    if visual.get('config_bundle_sha256') != bundle:
        raise ValueError('current config bundle differs from visual freeze')
    if file_sha256(a.checkpoint) != visual['checkpoint_sha256']:
        raise ValueError('checkpoint hash does not match visual freeze')
    route_kind = None if visual['route'] == 'NONE' else visual['route']
    df, sub, _, arrays = load_role_arrays(
        a.manifest, a.cache_dir, a.role, visual['selected_wide'], a.strict_hash
    )
    model = build_stage2_model(int(visual['seed']), bool(visual['use_tiera']), route_kind)
    meta = load_checkpoint(
        a.checkpoint,
        model,
        {
            'seed': int(visual['seed']),
            'manifest_sha256': visual['manifest_sha256'],
            'selected_wide': visual['selected_wide'],
            'route': visual['route'],
            'use_tiera': bool(visual['use_tiera']),
        },
    )
    norm = TierANormalizer(**meta['normalizer'])
    bio = norm.transform(arrays.bio)
    dev = torch.device(a.device if a.device != 'cuda' or torch.cuda.is_available() else 'cpu')
    model = model.to(dev).eval()
    required = (
        arrays.local.nbytes + bio.nbytes + (0 if arrays.wider is None else arrays.wider.nbytes)
    )
    resident = dev.type == 'cuda' and can_reside_on_cuda(required * 2)
    pin = dev.type == 'cuda' and not resident

    def cpu_t(x):
        t = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        return t.pin_memory() if pin else t

    local = cpu_t(arrays.local)
    b = cpu_t(bio)
    wide = None if arrays.wider is None else cpu_t(arrays.wider)
    if resident:
        local = local.to(dev)
        b = b.to(dev)
        wide = None if wide is None else wide.to(dev)
    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(dev)
    out_logits = []
    with torch.inference_mode():
        for start in range(0, len(sub), int(a.batch_size)):
            sl = slice(start, min(start + int(a.batch_size), len(sub)))
            if resident:
                ll, bb, ww = local[sl], b[sl], None if wide is None else wide[sl]
            else:
                ll = local[sl].to(dev, non_blocking=True)
                bb = b[sl].to(dev, non_blocking=True)
                ww = None if wide is None else wide[sl].to(dev, non_blocking=True)
            z = model(ll, bb, ww).float().cpu().numpy()
            out_logits.append(z)
            if not resident:
                del ll, bb, ww
    logits = np.concatenate(out_logits, axis=0)
    binding_hash = visual_hash
    if a.boundary_freeze:
        boundary = require_freeze(
            a.boundary_freeze, 'BOUNDARY_FREEZE', required_keys=['visual_freeze_sha256', 'recipe']
        )
        if boundary['visual_freeze_sha256'] != visual_hash:
            raise ValueError('boundary freeze does not belong to visual freeze')
        recipe = boundary['recipe']
        logits = (
            apply_h1(logits, np.asarray(recipe['offset'], float), float(recipe['lambda']))
            + np.asarray(recipe['delta_zero_sum'], float)[None, :]
        )
        logits = temperature(logits, float(recipe.get('temperature', 1.0)))
        binding_hash = file_sha256(a.boundary_freeze)
    if a.complete_freeze:
        complete = require_freeze(
            a.complete_freeze,
            'COMPLETE_PIPELINE_FREEZE',
            required_keys=['visual_freeze_sha256', 'boundary_freeze_sha256', 'seed'],
        )
        if complete['visual_freeze_sha256'] != visual_hash:
            raise ValueError('complete freeze visual mismatch')
        if not a.boundary_freeze or complete['boundary_freeze_sha256'] != file_sha256(
            a.boundary_freeze
        ):
            raise ValueError('complete freeze boundary mismatch')
        binding_hash = file_sha256(a.complete_freeze)
    mem = cuda_memory_stats() if dev.type == 'cuda' else {'cuda': False}
    model.to('cpu')
    del model, local, b, wide
    release_cuda()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out,
        uids=np.asarray(arrays.uids, dtype=str),
        labels=np.asarray(arrays.y, int),
        logits=np.asarray(logits, np.float32),
        pipeline_freeze_sha256=np.asarray(binding_hash),
        role=np.asarray(a.role),
        seed=np.asarray(int(visual['seed'])),
    )
    print(
        json.dumps(
            {
                'status': 'OK',
                'role': a.role,
                'rows': len(sub),
                'freeze_binding': binding_hash,
                'feature_storage': (
                    'gpu_resident' if resident else ('pinned_cpu_streaming' if pin else 'cpu')
                ),
                'cuda_memory': mem,
                'gpu_released': True,
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
