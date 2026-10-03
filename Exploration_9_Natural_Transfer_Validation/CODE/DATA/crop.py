from __future__ import annotations
import math
from PIL import Image
import numpy as np
import torch
from CODE.COMMON.constants import INPUT_SIZE
from CODE.COMMON.exceptions import CoordinateError

MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]


def crop_origin(center: float, fov: int) -> int:
    # Historical contract: floor(center - fov/2 + 0.5)
    return int(math.floor(center - fov / 2 + 0.5))


def centered_crop(
    image: Image.Image, x: float, y: float, fov: int, pad_value=(255, 255, 255)
) -> tuple[Image.Image, float, float]:
    image = image.convert('RGB')
    w, h = image.size
    if not (0 <= x < w and 0 <= y < h):
        raise CoordinateError(f"Center {(x,y)} outside {(w,h)}")
    left, top = crop_origin(x, fov), crop_origin(y, fov)
    right, bottom = left + fov, top + fov
    src = (max(left, 0), max(top, 0), min(right, w), min(bottom, h))
    out = Image.new('RGB', (fov, fov), pad_value)
    if src[2] > src[0] and src[3] > src[1]:
        out.paste(image.crop(src), (src[0] - left, src[1] - top))
    crop_x = x - left
    crop_y = y - top
    if not (0 <= crop_x < fov + 1 and 0 <= crop_y < fov + 1):
        raise CoordinateError('Internal crop coordinate contract violated')
    return out, crop_x, crop_y


def preprocess_crop(crop: Image.Image, output_size: int = INPUT_SIZE) -> torch.Tensor:
    crop = crop.resize((output_size, output_size), Image.Resampling.BICUBIC)
    arr = np.asarray(crop, dtype=np.float32) / 255.0
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB, got {arr.shape}")
    t = torch.from_numpy(arr.copy()).permute(2, 0, 1)
    return (t - MEAN) / STD
