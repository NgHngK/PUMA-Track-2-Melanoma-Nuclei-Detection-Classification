from __future__ import annotations
import hashlib
from pathlib import Path
import torch
from torch import nn
from ..constants import EMBED_DIM, UNI2_TOKEN_COUNT


def checkpoint_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def _load_weights_compat(path):
    """Load tensor-only checkpoints with mmap when possible and fall back for legacy serialization."""
    p = Path(path)
    try:
        return torch.load(p, map_location='cpu', weights_only=True, mmap=True)
    except RuntimeError as e:
        if 'mmap' not in str(e).lower():
            raise
        return torch.load(p, map_location='cpu', weights_only=True)
    except TypeError:
        # Compatibility with older PyTorch builds that do not expose weights_only/mmap.
        return torch.load(p, map_location='cpu')


def load_uni2_h(path):
    try:
        import timm
    except ImportError as e:
        raise RuntimeError('install timm==1.0.20') from e
    if getattr(timm, '__version__', None) != '1.0.20':
        raise RuntimeError(f'expected timm==1.0.20, got {getattr(timm,"__version__",None)}')
    state = _load_weights_compat(path)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    for k, shape in {
        'cls_token': (1, 1, 1536),
        'reg_token': (1, 8, 1536),
        'patch_embed.proj.weight': (1536, 3, 14, 14),
    }.items():
        if k not in state or tuple(state[k].shape) != shape:
            raise ValueError(f'not UNI2-h: {k}')
    kw = dict(
        img_size=224,
        patch_size=14,
        depth=24,
        num_heads=24,
        init_values=1e-5,
        embed_dim=1536,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=True,
    )
    enc = timm.create_model('vit_giant_patch14_224', pretrained=False, weight_init='skip', **kw)
    try:
        enc.load_state_dict(state, strict=True, assign=True)
    except TypeError:
        enc.load_state_dict(state, strict=True)
    return enc


class UNI2Backbone(nn.Module):
    """Exploration-7 frozen UNI2-h wrapper. Encoder adaptation is intentionally out of scope."""

    def __init__(self, weights):
        super().__init__()
        self.encoder = load_uni2_h(weights)
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward_tokens(self, x):
        with torch.inference_mode():
            t = self.encoder.forward_features(x)
        if t.shape[1:] != (UNI2_TOKEN_COUNT, EMBED_DIM):
            raise RuntimeError(tuple(t.shape))
        return t.float()
