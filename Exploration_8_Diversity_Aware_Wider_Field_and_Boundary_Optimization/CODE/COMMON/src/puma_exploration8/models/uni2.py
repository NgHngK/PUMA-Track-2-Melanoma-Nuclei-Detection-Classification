from __future__ import annotations
from pathlib import Path
import gc
import torch
from torch import nn


def load_uni2_h(path: str | Path, device='cpu'):
    try:
        import timm
    except ImportError as e:
        raise RuntimeError('timm==1.0.20 is required for real UNI2-h extraction') from e
    state = torch.load(Path(path), map_location='cpu', weights_only=True, mmap=True)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    expected = {
        'cls_token': (1, 1, 1536),
        'reg_token': (1, 8, 1536),
        'patch_embed.proj.weight': (1536, 3, 14, 14),
    }
    for k, shape in expected.items():
        if k not in state or tuple(state[k].shape) != shape:
            raise ValueError(f'not compatible UNI2-h: {k}')
    if not hasattr(timm.layers, 'SwiGLUPacked'):
        raise RuntimeError('timm==1.0.20 with SwiGLUPacked is required')
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
    model = timm.create_model('vit_giant_patch14_224', pretrained=False, weight_init='skip', **kw)
    model.load_state_dict(state, strict=True, assign=True)
    del state
    gc.collect()
    model.requires_grad_(False).eval().to(device)
    return model


@torch.inference_mode()
def token_grid(model, x: torch.Tensor) -> torch.Tensor:
    t = model.forward_features(x)
    if t.ndim != 3 or tuple(t.shape[1:]) != (265, 1536):
        raise ValueError(f'unexpected UNI2 token shape {tuple(t.shape)}; expected Bx265x1536')
    if int(getattr(model, 'num_prefix_tokens', 9)) != 9:
        raise ValueError('expected 9 UNI2 prefix tokens')
    patches = t[:, 9:]
    if patches.shape[1] != 256:
        raise ValueError('expected 256 spatial tokens')
    # Explicit FP32 pooled-feature contract. Caller should pool then release this grid immediately.
    return patches.reshape(len(x), 16, 16, 1536).float()
