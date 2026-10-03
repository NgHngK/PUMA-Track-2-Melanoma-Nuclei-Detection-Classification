from __future__ import annotations
import numpy as np
from PIL import Image
from scipy.ndimage import laplace

SCHEMA = {
    'stain': ['H_mean', 'H_std', 'H_p10', 'H_p50', 'H_p90'],
    'gradient_texture': [
        'H_gradient_mean',
        'H_gradient_std',
        'H_abs_laplacian_mean',
        'H_entropy32',
    ],
    'ring': [
        'H_center_minus_ring',
        'gray_center_minus_ring',
        'H_ring_std',
        'central_valid_fraction',
        'ring_valid_fraction',
    ],
    'roi_relative': ['H_robust_z', 'H_ROI_percentile'],
}
NAMES = tuple(sum(SCHEMA.values(), []))
_YY, _XX = np.mgrid[-24:25, -24:25]
_D2 = _XX * _XX + _YY * _YY
_CENTER = _D2 <= 12**2
_RING = (_D2 >= 16**2) & (_D2 <= 24**2)
_HEM = np.array([0.650, 0.704, 0.286], dtype=np.float32)


def roi_channels(rgb: Image.Image | np.ndarray) -> dict:
    x = np.asarray(rgb, dtype=np.float32) / 255.0
    if x.ndim != 3 or x.shape[2] < 3:
        raise ValueError('RGB image expected')
    x = x[:, :, :3]
    h = np.clip((-np.log(np.clip(x, 1 / 255, 1)) @ _HEM) / 2, 0, 2)
    gy, gx = np.gradient(h)
    grad = np.sqrt(gx * gx + gy * gy + 1e-8)
    return {
        'h': h,
        'g': grad,
        'lap': np.abs(laplace(h, mode='reflect')),
        'gray': x.mean(2),
        'median': float(np.median(h)),
        'iqr': float(np.percentile(h, 75) - np.percentile(h, 25)),
        'sorted_h': np.sort(h.ravel()),
    }


def point_features(ch: dict, x: float, y: float) -> np.ndarray:
    ix = _XX + int(np.floor(x + 0.5))
    iy = _YY + int(np.floor(y + 0.5))
    height, width = ch['h'].shape
    valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
    c = _CENTER & valid
    r = _RING & valid
    if not c.any() or not r.any():
        raise ValueError('insufficient image support for Tier-A')

    def vals(key, mask):
        return ch[key][iy[mask], ix[mask]]

    h = vals('h', c)
    hr = vals('h', r)
    g = vals('g', c)
    mean = float(h.mean())
    hist = np.histogram(h, bins=32, range=(0, 2))[0].astype(float)
    if hist.sum() <= 0:
        raise ValueError('empty Tier-A histogram')
    hist /= hist.sum()
    hist = hist[hist > 0]
    v = [
        mean,
        h.std(),
        *np.percentile(h, [10, 50, 90]),
        g.mean(),
        g.std(),
        vals('lap', c).mean(),
        -(hist * np.log(hist)).sum(),
        mean - hr.mean(),
        vals('gray', c).mean() - vals('gray', r).mean(),
        hr.std(),
        c.sum() / _CENTER.sum(),
        r.sum() / _RING.sum(),
        (mean - ch['median']) / max(ch['iqr'], 1e-4),
        np.searchsorted(ch['sorted_h'], mean, side='right') / len(ch['sorted_h']),
    ]
    out = np.asarray(v, dtype=np.float32)
    if out.shape != (16,) or not np.isfinite(out).all():
        raise ValueError('invalid Tier-A features')
    return out


def compute_tier_a(rows: list[dict]) -> np.ndarray:
    """Compute Tier-A once per ROI image rather than repeatedly rebuilding ROI channels.

    Rows are grouped by image internally and written back to their original positions, so
    output is bitwise/order compatible with the manifest while eliminating redundant TIFF
    decoding, H-channel construction, gradients, Laplacians and full-ROI sorting.
    """
    result = np.empty((len(rows), 16), dtype=np.float32)
    by_image = {}
    for i, row in enumerate(rows):
        by_image.setdefault(str(row['image']), []).append((i, row))
    for path, items in by_image.items():
        with Image.open(path) as im:
            rgb = im.convert('RGB').copy()
        ch = roi_channels(rgb)
        for i, row in items:
            result[i] = point_features(ch, float(row['x']), float(row['y']))
    return result


class TierANormalizer:
    def __init__(self, mean=None, std=None):
        self.mean = None if mean is None else np.asarray(mean, np.float32)
        self.std = None if std is None else np.asarray(std, np.float32)

    def fit(self, x):
        x = np.asarray(x, np.float32)
        if x.ndim != 2 or x.shape[1] != 16 or not np.isfinite(x).all():
            raise ValueError('Tier-A fit expects finite Nx16')
        self.mean = x.mean(0)
        self.std = np.maximum(x.std(0), 1e-6)
        return self

    def transform(self, x):
        if self.mean is None or self.std is None:
            raise RuntimeError('TierANormalizer not fitted')
        z = (np.asarray(x, np.float32) - self.mean) / self.std
        if not np.isfinite(z).all():
            raise FloatingPointError('nonfinite normalized Tier-A')
        return z.astype(np.float32, copy=False)

    def state_dict(self):
        if self.mean is None or self.std is None:
            raise RuntimeError('TierANormalizer not fitted')
        return {'mean': self.mean.tolist(), 'std': self.std.tolist(), 'names': list(NAMES)}
