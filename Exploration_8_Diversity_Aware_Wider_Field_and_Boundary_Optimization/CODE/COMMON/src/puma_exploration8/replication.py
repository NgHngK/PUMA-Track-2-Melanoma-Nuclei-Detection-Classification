from __future__ import annotations
import random
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import laplace

MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def crop_origin(x: float, y: float, fov: int) -> tuple[int, int]:
    return int(np.floor(float(x) - fov / 2 + 0.5)), int(np.floor(float(y) - fov / 2 + 0.5))


def target_grid_coordinate(
    x: float, y: float, fov: int, grid_side: int = 16
) -> tuple[float, float]:
    """Historical continuous UNI2 token-grid coordinate; no half-token shift."""
    left, top = crop_origin(x, y, fov)
    px_per_token = float(fov) / float(grid_side)
    return (float(x) - left) / px_per_token, (float(y) - top) / px_per_token


def scale_only_grid_coordinate(x: float, y: float) -> tuple[float, float]:
    """Target coordinate after FOV96 is shrunk to central 112 px of a 224 canvas."""
    left, top = crop_origin(x, y, 96)
    # 56 px canvas offset = 4 patches; 96 source px -> 112 canvas px = 12 source px/patch.
    return 4.0 + (float(x) - left) / 12.0, 4.0 + (float(y) - top) / 12.0


def centered_crop(
    image: Image.Image, x: float, y: float, fov: int, fill=(255, 255, 255)
) -> Image.Image:
    if fov < 2:
        raise ValueError('fov must be >=2')
    w, h = image.size
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError(f'center outside image: {(x,y)} not in {(w,h)}')
    left, top = crop_origin(x, y, fov)
    src = (max(0, left), max(0, top), min(w, left + fov), min(h, top + fov))
    out = Image.new('RGB', (fov, fov), fill)
    out.paste(image.crop(src), (src[0] - left, src[1] - top))
    return out


def to_encoder_tensor(crop: Image.Image, size: int = 224) -> torch.Tensor:
    a = np.asarray(crop.resize((size, size), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    t = torch.from_numpy(a.copy()).permute(2, 0, 1)
    return (t - MEAN) / STD


def local_wide_scale_tensors(image: Image.Image, x: float, y: float):
    local = centered_crop(image, x, y, 96)
    wide = centered_crop(image, x, y, 192)
    lt = to_encoder_tensor(local)
    wt = to_encoder_tensor(wide)
    local112 = np.asarray(local.resize((112, 112), Image.Resampling.BICUBIC), dtype=np.uint8)
    canvas = np.full((224, 224, 3), 255, dtype=np.uint8)
    canvas[56:168, 56:168] = local112
    st = to_encoder_tensor(Image.fromarray(canvas), 224)
    u96, v96 = target_grid_coordinate(x, y, 96)
    u192, v192 = target_grid_coordinate(x, y, 192)
    us, vs = scale_only_grid_coordinate(x, y)
    return lt, wt, st, (u96, v96), (u192, v192), (us, vs)


def h_proxy(rgb: np.ndarray) -> np.ndarray:
    x = np.clip(rgb.astype(np.float32) / 255.0, 1 / 255, 1.0)
    od = -np.log(x)
    return np.clip((od * np.array([0.650, 0.704, 0.286], dtype=np.float32)).sum(-1) / 2.0, 0, 2)


def tier_a_roi_channels(image: Image.Image):
    rgb = np.asarray(image.convert('RGB'))
    H = h_proxy(rgb)
    gray = (rgb.astype(np.float32) / 255.0).mean(-1)
    gy, gx = np.gradient(H)
    grad = np.sqrt(gx * gx + gy * gy + 1e-8)
    # Match historical scipy.ndimage.laplace(..., mode='reflect') exactly.
    return rgb, H, gray, grad, np.abs(laplace(H, mode='reflect'))


def tier_a_features(image: Image.Image, points: list[tuple[float, float]]) -> np.ndarray:
    rgb, H, gray, grad, alap = tier_a_roi_channels(image)
    h, w = H.shape
    med = float(np.median(H))
    q25, q75 = np.percentile(H, [25, 75])
    iqr = max(float(q75 - q25), 1e-4)
    sorted_h = np.sort(H.ravel())
    yy, xx = np.mgrid[-24:25, -24:25]
    d2 = xx * xx + yy * yy
    center = d2 <= 12**2
    ring = (d2 >= 16**2) & (d2 <= 24**2)
    out = []
    nominal_c = center.sum()
    nominal_r = ring.sum()
    for x, y in points:
        cx = int(np.floor(float(x) + 0.5))
        cy = int(np.floor(float(y) + 0.5))
        xs = np.arange(cx - 24, cx + 25)
        ys = np.arange(cy - 24, cy + 25)
        validx = (xs >= 0) & (xs < w)
        validy = (ys >= 0) & (ys < h)
        valid = validy[:, None] & validx[None, :]
        X = np.clip(xs, 0, w - 1)
        Y = np.clip(ys, 0, h - 1)
        Hp = H[np.ix_(Y, X)]
        Gp = grad[np.ix_(Y, X)]
        Lp = alap[np.ix_(Y, X)]
        grayp = gray[np.ix_(Y, X)]
        cm = center & valid
        rm = ring & valid
        if not cm.any() or not rm.any():
            raise ValueError('Tier-A center/ring has no valid support')
        hc = Hp[cm]
        hr = Hp[rm]
        gc = grayp[cm]
        gr = grayp[rm]
        hist = np.histogram(hc, bins=32, range=(0, 2))[0].astype(float)
        hist /= hist.sum()
        prob = hist[hist > 0]
        ent = float(-(prob * np.log(prob)).sum()) if len(prob) else 0.0
        hmean = float(hc.mean())
        out.append(
            [
                hmean,
                float(hc.std()),
                *map(float, np.percentile(hc, [10, 50, 90])),
                float(Gp[cm].mean()),
                float(Gp[cm].std()),
                float(Lp[cm].mean()),
                ent,
                hmean - float(hr.mean()),
                float(gc.mean() - gr.mean()),
                float(hr.std()),
                float(cm.sum() / nominal_c),
                float(rm.sum() / nominal_r),
                (hmean - med) / iqr,
                float(np.searchsorted(sorted_h, hmean, side='right') / len(sorted_h)),
            ]
        )
    a = np.asarray(out, dtype=np.float32)
    if a.shape != (len(points), 16) or not np.isfinite(a).all():
        raise RuntimeError('Tier-A contract failure')
    return a
