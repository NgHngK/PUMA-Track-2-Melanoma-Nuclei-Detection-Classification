from __future__ import annotations
from pathlib import Path
from typing import Mapping
import json
import numpy as np
import torch
from .constants import CLASSES, EMBED_DIM, TIER_A_DIM
from .models.a5 import A5Head

EXPLORATION6_SEEDS = (17, 29, 43)


def _load_checkpoint_compat(path):
    """Load tensor checkpoints with mmap when supported, otherwise fall back safely."""
    p = Path(path)
    try:
        return torch.load(p, map_location='cpu', weights_only=True, mmap=True)
    except RuntimeError as e:
        if 'mmap' not in str(e).lower():
            raise
        return torch.load(p, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(p, map_location='cpu')


class TierANormalizer:
    def __init__(self, state):
        self.mean = np.asarray(state['mean'], np.float32)
        self.std = np.asarray(state['std'], np.float32)
        if (
            self.mean.shape != (TIER_A_DIM,)
            or self.std.shape != (TIER_A_DIM,)
            or (self.std <= 0).any()
        ):
            raise ValueError('invalid Exploration-6 Tier-A normalizer')

    def transform(self, x):
        x = np.asarray(x, np.float32)
        z = (x - self.mean) / self.std
        if not np.isfinite(z).all():
            raise FloatingPointError('nonfinite normalized Tier-A')
        return z.astype(np.float32, copy=False)


def _validate_anchor_config(ck: dict, expected_seed: int | None = None) -> dict:
    if tuple(ck.get('classes', ())) != CLASSES:
        raise ValueError('class ontology mismatch')
    cfg = ck.get('config', {})
    mc = cfg.get('model', {})
    observed = {
        'seed': int(cfg.get('seed', -1)),
        'experiment_id': str(cfg.get('experiment_id', '')),
        'fov': int(cfg.get('fov', -1)),
        'representation': mc.get('representation'),
        'gaussian_sigma': float(mc.get('gaussian_sigma', float('nan'))),
        'interaction_rank': int(mc.get('interaction_rank', -1)),
        'use_tier_a': bool(mc.get('use_tier_a', False)),
        'epoch': int(ck.get('epoch', -1)),
        'score': float(ck.get('score', float('nan'))),
    }
    if expected_seed is not None and observed['seed'] != int(expected_seed):
        raise ValueError(
            f'checkpoint seed mismatch: expected {expected_seed}, got {observed["seed"]}'
        )
    if (
        observed['fov'] != 96
        or observed['representation'] != 'gaussian'
        or abs(observed['gaussian_sigma'] - 1.5) > 1e-9
    ):
        raise ValueError(f'not the frozen Exploration-6 Gaussian FOV96 anchor: {observed}')
    if observed['interaction_rank'] != 8 or not observed['use_tier_a']:
        raise ValueError(f'not the retained rank-8 Tier-A A5 anchor: {observed}')
    return observed


def load_exploration6_head(checkpoint_path, device='cpu', expected_seed: int | None = None):
    ck = _load_checkpoint_compat(checkpoint_path)
    meta = _validate_anchor_config(ck, expected_seed)
    mc = ck['config']['model']
    head = A5Head(int(mc.get('interaction_rank', 8)), bool(mc.get('use_tier_a', True)))
    state = {k.removeprefix('head.'): v for k, v in ck['model'].items() if k.startswith('head.')}
    head.load_state_dict(state, strict=True)
    head.to(device).eval()
    ck['_prompt7_anchor_metadata'] = meta
    return head, ck, TierANormalizer(ck['tier_a_normalizer'])


def load_checkpoint_map(path_or_mapping, require_all: bool = True) -> dict[int, Path]:
    if isinstance(path_or_mapping, (str, Path)):
        payload = json.loads(Path(path_or_mapping).read_text(encoding='utf-8'))
    elif isinstance(path_or_mapping, Mapping):
        payload = dict(path_or_mapping)
    else:
        raise TypeError('checkpoint map must be a JSON path or mapping')
    if 'checkpoints' in payload:
        payload = payload['checkpoints']
    out = {int(k): Path(v) for k, v in payload.items()}
    seeds = set(EXPLORATION6_SEEDS) if require_all else set(out)
    missing = [s for s in seeds if s not in out or not out[s].is_file()]
    if missing:
        raise FileNotFoundError(f'checkpoint map missing seeds/files: {missing}')
    for seed in sorted(seeds):
        load_exploration6_head(out[seed], 'cpu', expected_seed=seed)
    return {s: out[s] for s in sorted(out)}


def write_checkpoint_map(path: str | Path, mapping: Mapping[int, str | Path]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        'primary_seed': 17,
        'replication_seeds': [29, 43],
        'checkpoints': {str(k): str(v) for k, v in sorted(mapping.items())},
    }
    p.write_text(json.dumps(data, indent=2), encoding='utf-8')


def _auto_infer_batch(n: int, device) -> int:
    dev = torch.device(device)
    if dev.type == 'cuda' and torch.cuda.is_available():
        return min(max(1024, n), 16384)
    return min(max(512, n), 8192)


def infer_head(head, features, tier_a, batch_size: int | None = None, device='cpu'):
    features = np.asarray(features, np.float32)
    tier_a = np.asarray(tier_a, np.float32)
    if features.shape != (len(tier_a), EMBED_DIM) or tier_a.shape[1:] != (TIER_A_DIM,):
        raise ValueError('feature/Tier-A shape mismatch')
    n = len(features)
    if n == 0:
        return np.empty((0, len(CLASSES)), np.float32)
    bs = _auto_infer_batch(n, device) if not batch_size or batch_size <= 0 else int(batch_size)
    out = np.empty((n, len(CLASSES)), np.float32)
    head.eval()
    dev = torch.device(device)
    with torch.inference_mode():
        for i in range(0, n, bs):
            j = min(i + bs, n)
            h = torch.from_numpy(
                np.array(features[i:j], dtype=np.float32, copy=True, order='C')
            ).to(dev, non_blocking=True)
            b = torch.from_numpy(np.array(tier_a[i:j], dtype=np.float32, copy=True, order='C')).to(
                dev, non_blocking=True
            )
            out[i:j] = head(h, b).float().cpu().numpy()
    return out
