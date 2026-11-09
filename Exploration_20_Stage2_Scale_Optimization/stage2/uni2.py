"""Original E10 frozen UNI2-h, white-padded FOV112/224 and Gaussian pooling."""

import numpy as np
import torch
from PIL import Image
from torch import nn
from .settings import POOL_SIGMA

MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]


def load_uni2(checkpoint, device):
    import timm

    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if "state_dict" in state:
        state = state["state_dict"]
    options = dict(
        img_size=224,
        patch_size=14,
        depth=24,
        num_heads=24,
        init_values=1e-05,
        embed_dim=1536,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=True,
    )
    model = timm.create_model(
        "vit_giant_patch14_224", pretrained=False, weight_init="skip", **options
    )
    model.load_state_dict(state, strict=True, assign=True)
    return model.to(device).eval().requires_grad_(False)


def crop_at(image, x, y, fov):
    if not (np.isfinite(x) and np.isfinite(y)) or fov <= 0:
        raise ValueError("Crop coordinates must be finite and FOV positive")
    left = int(np.floor(x - fov / 2 + 0.5))
    top = int(np.floor(y - fov / 2 + 0.5))
    width, height = image.size
    region = (max(left, 0), max(top, 0), min(left + fov, width), min(top + fov, height))
    crop = Image.new("RGB", (fov, fov), (255, 255, 255))
    crop.paste(image.crop(region), (region[0] - left, region[1] - top))
    # PIL resize maps output pixel q to source (q+.5)*FOV/224-.5.
    # Patch j covers output pixels 14*j..14*j+13, center 14*j+6.5.
    # Inverting that mapping gives the fractional patch-center index below.
    return (crop, (x - left + 0.5) / (fov / 16) - 0.5, (y - top + 0.5) / (fov / 16) - 0.5)


def pool(tokens, x, y):
    if tokens.ndim != 3 or tokens.shape[1:] != (265, 1536):
        raise ValueError(
            "UNI2-h requires one CLS, eight register and 256 spatial tokens of width 1536"
        )
    spatial = tokens[:, 9:, :].reshape(len(tokens), 16, 16, 1536)
    xs = torch.arange(16, device=tokens.device, dtype=tokens.dtype)[None, None, :]
    ys = torch.arange(16, device=tokens.device, dtype=tokens.dtype)[None, :, None]
    x = torch.as_tensor(x, device=tokens.device, dtype=tokens.dtype)[:, None, None]
    y = torch.as_tensor(y, device=tokens.device, dtype=tokens.dtype)[:, None, None]
    weights = torch.exp(-((xs - x).square() + (ys - y).square()) / (2 * POOL_SIGMA**2))
    weights /= weights.sum(dim=(1, 2), keepdim=True)
    return (spatial * weights.unsqueeze(-1)).sum(dim=(1, 2)).float()
