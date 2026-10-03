from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
from CODE.COMMON.hashing import sha256_file
from CODE.COMMON.constants import APPEARANCE_DIM
from CODE.COMMON.exceptions import CheckpointIdentityError, TokenGeometryError

EXPECTED_ENCODER_SHA256 = "32cc070acd809d087debdbe483f388561cf03de825cc7c2935d1732dfb427ffe"


def load_uni2_h(path: str | Path, enforce_known_hash: bool = True):
    import timm

    path = Path(path)
    digest = sha256_file(path)
    if enforce_known_hash and digest != EXPECTED_ENCODER_SHA256:
        raise CheckpointIdentityError(f"UNI2-h SHA256 mismatch: {digest}")
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if 'state_dict' in state:
        state = state['state_dict']
    expected = {
        'cls_token': (1, 1, 1536),
        'reg_token': (1, 8, 1536),
        'patch_embed.proj.weight': (1536, 3, 14, 14),
    }
    for key, shape in expected.items():
        if key not in state or tuple(state[key].shape) != shape:
            raise CheckpointIdentityError(f"Not UNI2-h: {key} expected {shape}")
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
    model.eval().requires_grad_(False)
    return model, digest


def forward_all_tokens(model, x: torch.Tensor) -> torch.Tensor:
    # timm ViT forward_features returns final-norm token sequence for this model family.
    with torch.inference_mode():
        t = model.forward_features(x)
    if t.ndim != 3 or t.shape[-1] != APPEARANCE_DIM:
        raise TokenGeometryError(f"Unexpected UNI2 token output {tuple(t.shape)}")
    return t
