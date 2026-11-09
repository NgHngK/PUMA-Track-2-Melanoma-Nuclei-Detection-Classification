"""Tier-A16 features from local histology image statistics."""

import math
import numpy as np
from scipy.ndimage import laplace

_H_VECTOR = np.asarray([0.65, 0.704, 0.286], dtype=np.float32)
_OFFSETS = np.arange(-24, 25, dtype=np.int32)
_DY, _DX = np.meshgrid(_OFFSETS, _OFFSETS, indexing="ij")
_R2 = _DX**2 + _DY**2
_CENTRAL_MASK = _R2 <= 12**2
_RING_MASK = (_R2 >= 16**2) & (_R2 <= 24**2)
_CENTRAL_NOMINAL = int(_CENTRAL_MASK.sum())
_RING_NOMINAL = int(_RING_MASK.sum())


def h_proxy_from_rgb(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(rgb, dtype=np.float32) / 255.0
    od = -np.log(np.clip(x, 1.0 / 255.0, 1.0))
    h = np.clip(od @ _H_VECTOR / 2.0, 0.0, 2.0).astype(np.float32)
    gray = x.mean(axis=2, dtype=np.float32)
    return (h, gray)


def _entropy32(values: np.ndarray) -> float:
    counts, _ = np.histogram(values, bins=32, range=(0.0, 2.0))
    total = counts.sum()
    p = counts[counts > 0].astype(np.float64) / float(total)
    return float(-(p * np.log(p)).sum())


def _roi_maps(rgb: np.ndarray):
    h, gray = h_proxy_from_rgb(rgb)
    gy, gx = np.gradient(h)
    grad = np.sqrt(gx * gx + gy * gy + 1e-08).astype(np.float32)
    abs_lap = np.abs(laplace(h, mode="reflect")).astype(np.float32)
    med = float(np.median(h))
    q25, q75 = np.percentile(h, [25, 75])
    iqr = max(float(q75 - q25), 0.0001)
    sorted_h = np.sort(h.reshape(-1))
    return (h, gray, grad, abs_lap, med, iqr, sorted_h)


def point_features(maps, x: float, y: float) -> np.ndarray:
    h, gray, grad, abs_lap, roi_med, roi_iqr, sorted_h = maps
    height, width = h.shape
    cx = int(math.floor(float(x) + 0.5))
    cy = int(math.floor(float(y) + 0.5))
    xs = cx + _DX
    ys = cy + _DY
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    central = valid & _CENTRAL_MASK
    ring = valid & _RING_MASK
    cys, cxs = (ys[central], xs[central])
    rys, rxs = (ys[ring], xs[ring])
    hc = h[cys, cxs]
    hr = h[rys, rxs]
    gc = gray[cys, cxs]
    gr = gray[rys, rxs]
    gradc = grad[cys, cxs]
    lapc = abs_lap[cys, cxs]
    hmean = float(hc.mean())
    percentile = float(np.searchsorted(sorted_h, hmean, side="right") / len(sorted_h))
    out = np.asarray(
        [
            hmean,
            float(hc.std(ddof=0)),
            *[float(v) for v in np.percentile(hc, [10, 50, 90])],
            float(gradc.mean()),
            float(gradc.std(ddof=0)),
            float(lapc.mean()),
            _entropy32(hc),
            hmean - float(hr.mean()),
            float(gc.mean()) - float(gr.mean()),
            float(hr.std(ddof=0)),
            float(central.sum()) / _CENTRAL_NOMINAL,
            float(ring.sum()) / _RING_NOMINAL,
            (hmean - roi_med) / roi_iqr,
            percentile,
        ],
        dtype=np.float32,
    )
    return out
