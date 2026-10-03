from __future__ import annotations
import numpy as np, torch
from PIL import Image
from .constants import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, UNI2_GRID_SIZE

MEAN = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[:, None, None]
STD = torch.tensor(IMAGENET_STD, dtype=torch.float32)[:, None, None]


def centered_crop_with_token_coordinate(image: Image.Image, cx: float, cy: float, fov: int):
    if fov < 16 or fov % 2:
        raise ValueError('fov must be even >=16')
    w, h = image.size
    if not (0 <= cx < w and 0 <= cy < h):
        raise ValueError('center outside ROI')
    left = int(np.floor(cx - fov / 2 + 0.5))
    top = int(np.floor(cy - fov / 2 + 0.5))
    box = (max(left, 0), max(top, 0), min(left + fov, w), min(top + fov, h))
    out = Image.new('RGB', (fov, fov), (255, 255, 255))
    out.paste(image.crop(box), (box[0] - left, box[1] - top))
    s = fov / UNI2_GRID_SIZE
    return out, float((cx - left) / s), float((cy - top) / s)


def image_to_tensor(crop):
    arr = (
        np.asarray(
            crop.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC), dtype=np.float32
        )
        / 255.0
    )
    t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
    return (t - MEAN) / STD
