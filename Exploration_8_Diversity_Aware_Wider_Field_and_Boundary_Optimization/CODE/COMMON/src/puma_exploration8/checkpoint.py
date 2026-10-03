from __future__ import annotations
from pathlib import Path
import torch
from .constants import CLASSES


def save_checkpoint(path, model, metadata):
    m = dict(metadata)
    m['classes'] = list(CLASSES)
    torch.save({'metadata': m, 'state_dict': model.state_dict()}, Path(path))


def load_checkpoint(path, model, expected=None):
    x = torch.load(Path(path), map_location='cpu', weights_only=True)
    m = x['metadata']
    if m.get('classes') != list(CLASSES):
        raise ValueError('checkpoint class order mismatch')
    for k, v in (expected or {}).items():
        if m.get(k) != v:
            raise ValueError(f'checkpoint metadata mismatch: {k}')
    model.load_state_dict(x['state_dict'], strict=True)
    return m
