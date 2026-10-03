from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'COMMON' / 'src'))

import argparse, gc, hashlib, json
import numpy as np
import torch
from PIL import Image
from puma_exploration8.manifest import read_manifest
from puma_exploration8.models.uni2 import load_uni2_h, token_grid
from puma_exploration8.models.pooling import gaussian_pool, annular_pool
from puma_exploration8.replication import local_wide_scale_tensors, tier_a_features
from puma_exploration8.config import sha256_file
from puma_exploration8.cache_integrity import write_cache
from puma_exploration8.runtime import cuda_memory_stats, release_cuda


def auto_batch(device: torch.device, requested: int) -> int:
    if requested > 0:
        return int(requested)
    if device.type != 'cuda':
        return 2
    free, total = torch.cuda.mem_get_info(device)
    gb = min(free, total) / 1024**3
    if gb >= 70:
        return 64
    if gb >= 40:
        return 32
    if gb >= 24:
        return 16
    if gb >= 14:
        return 8
    if gb >= 9:
        return 4
    return 2


def _to_device_batch(tensors, device):
    x = torch.stack(tensors)
    if device.type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
    return x


def extract_batch(model, device, image, rows):
    local_t, wide_t, scale_t, c96, c192, cs = [], [], [], [], [], []
    for r in rows:
        lt, wt, st, p96, p192, ps = local_wide_scale_tensors(image, float(r.x), float(r.y))
        local_t.append(lt)
        wide_t.append(wt)
        scale_t.append(st)
        c96.append(p96)
        c192.append(p192)
        cs.append(ps)
    u96 = np.asarray([q[0] for q in c96], np.float32)
    v96 = np.asarray([q[1] for q in c96], np.float32)
    u192 = np.asarray([q[0] for q in c192], np.float32)
    v192 = np.asarray([q[1] for q in c192], np.float32)
    us = np.asarray([q[0] for q in cs], np.float32)
    vs = np.asarray([q[1] for q in cs], np.float32)

    # Run one forward pass at a time to limit VRAM use. Move the pooled features to CPU before releasing the grid.
    lo = _to_device_batch(local_t, device)
    lg = token_grid(model, lo)
    local_out = gaussian_pool(lg, u96, v96, 1.5).cpu().numpy()
    del lo, lg

    wi = _to_device_batch(wide_t, device)
    wg = token_grid(model, wi)
    gaussian_out = gaussian_pool(wg, u192, v192, 1.5).cpu().numpy()
    annular_out = annular_pool(wg, u192, v192, 0.5).cpu().numpy()
    del wi, wg

    sc = _to_device_batch(scale_t, device)
    sg = token_grid(model, sc)
    scale_out = gaussian_pool(sg, us, vs, 1.5).cpu().numpy()
    del sc, sg
    return local_out, gaussian_out, annular_out, scale_out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument(
        '--batch-size',
        type=int,
        default=0,
        help='0 = deterministic VRAM-based initial size with OOM halving',
    )
    args = p.parse_args()

    df = read_manifest(args.manifest)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device if args.device != 'cuda' or torch.cuda.is_available() else 'cpu')
    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(dev)
    model = load_uni2_h(args.weights, dev)
    weights_hash = sha256_file(args.weights)
    manifest_hash = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    uids, local, gaussian, annular, scale, bio = [], [], [], [], [], []
    chosen_bs = auto_batch(dev, args.batch_size)
    oom_backoffs = 0

    for image_path, group in df.groupby('image', sort=False):
        with Image.open(image_path) as im:
            image = im.convert('RGB')
        rows = list(group.itertuples())
        points = [(float(r.x), float(r.y)) for r in rows]
        bio.extend(tier_a_features(image, points))
        uids.extend([str(r.uid) for r in rows])
        start = 0
        while start < len(rows):
            bs = min(chosen_bs, len(rows) - start)
            while True:
                try:
                    l, g, r, s = extract_batch(model, dev, image, rows[start : start + bs])
                    local.extend(l)
                    gaussian.extend(g)
                    annular.extend(r)
                    scale.extend(s)
                    del l, g, r, s
                    start += bs
                    break
                except RuntimeError as e:
                    if 'out of memory' not in str(e).lower() or bs <= 1:
                        raise
                    oom_backoffs += 1
                    bs = max(1, bs // 2)
                    chosen_bs = min(chosen_bs, bs)  # shrink globally only after a real OOM
                    gc.collect()
                    if dev.type == 'cuda':
                        torch.cuda.empty_cache()
        del image

    mem = cuda_memory_stats() if dev.type == 'cuda' else {'cuda': False}
    model.to('cpu')
    del model
    release_cuda()

    meta = {
        'manifest_sha256': manifest_hash,
        'encoder_sha256': weights_hash,
        'feature_dim': 1536,
        'precision': 'fp32',
        'code_contract': 'exploration8-v2',
    }
    payloads = [
        (
            'local_gaussian96',
            local,
            {
                'fov': 96,
                'pool': 'coordinate_aware_gaussian_sigma1.5',
                'coordinate_rule': 'u=(x-left)/(fov/16); no half-token shift',
            },
        ),
        (
            'gaussian192',
            gaussian,
            {
                'fov': 192,
                'pool': 'coordinate_aware_gaussian_sigma1.5',
                'coordinate_rule': 'u=(x-left)/(fov/16); no half-token shift',
            },
        ),
        (
            'annular192',
            annular,
            {
                'fov': 192,
                'pool': 'coordinate_aware_center_excluded_fov96_footprint',
                'local_fraction': 0.5,
            },
        ),
        (
            'scale_only192',
            scale,
            {
                'source_pixels': 'FOV96_only',
                'pool': 'coordinate_aware_gaussian_sigma1.5',
                'canvas': 'central_112_of_224_white',
            },
        ),
        ('tiera16', bio, {'feature_order': 'canonical_16', 'feature_dim': 16}),
    ]
    for name, data, extra in payloads:
        write_cache(out / name, np.asarray(data, np.float32), uids, meta | extra)
    print(
        json.dumps(
            {
                'status': 'OK',
                'rows': len(uids),
                'encoder_loaded_instances': 1,
                'selected_batch_size': chosen_bs,
                'oom_backoffs': oom_backoffs,
                'cuda_memory': mem,
                'model_released_from_gpu': True,
                'out': str(out),
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
