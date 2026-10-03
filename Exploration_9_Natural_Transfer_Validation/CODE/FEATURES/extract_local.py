from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Executor
from typing import Iterable

from PIL import Image
import torch

from CODE.DATA.crop import centered_crop, preprocess_crop
from CODE.FEATURES.token_geometry import extract_spatial_tokens, source_to_token_coordinate
from CODE.FEATURES.gaussian_pool import gaussian_pool


class ROIImageCache:
    """Small LRU cache so each 1024 ROI is opened only when needed."""

    def __init__(self, max_images: int = 4):
        self.max_images = max(1, int(max_images))
        self._cache = OrderedDict()

    def get(self, path: str):
        if path not in self._cache:
            with Image.open(path) as im:
                self._cache[path] = im.convert("RGB").copy()
            while len(self._cache) > self.max_images:
                self._cache.popitem(last=False)
        self._cache.move_to_end(path)
        return self._cache[path]

    def clear(self):
        self._cache.clear()


def _prepare_one(row, images: dict[str, Image.Image], fovs: tuple[int, ...]):
    image = images[row.image_path]
    tensors = {}
    coords = {}
    for fov in fovs:
        crop, cx, cy = centered_crop(image, row.x, row.y, int(fov))
        tensors[int(fov)] = preprocess_crop(crop)
        coords[int(fov)] = (
            source_to_token_coordinate(cx, int(fov)),
            source_to_token_coordinate(cy, int(fov)),
        )
    return tensors, coords


def prepare_multiview_batch(
    rows,
    fovs: Iterable[int],
    *,
    image_cache: ROIImageCache,
    executor: Executor | None = None,
):
    """Prepare one or more FOVs for the same nucleus rows on CPU.

    Images are loaded once through the ROI cache. Per-nucleus crop/resize work is
    parallelized across the supplied executor, while result order remains identical
    to the manifest order.
    """
    fovs = tuple(dict.fromkeys(int(v) for v in fovs))
    if not fovs:
        raise ValueError("At least one FOV is required")
    rows = list(rows)
    if not rows:
        return {fov: (torch.empty((0, 3, 224, 224), dtype=torch.float32), [], []) for fov in fovs}

    images = {path: image_cache.get(path) for path in dict.fromkeys(r.image_path for r in rows)}
    if executor is None or len(rows) == 1:
        prepared = [_prepare_one(r, images, fovs) for r in rows]
    else:
        prepared = list(executor.map(lambda r: _prepare_one(r, images, fovs), rows))

    out = {}
    for fov in fovs:
        tensors = [item[0][fov] for item in prepared]
        tx = [item[1][fov][0] for item in prepared]
        ty = [item[1][fov][1] for item in prepared]
        out[fov] = (torch.stack(tensors, dim=0), tx, ty)
    return out


def run_prepared_multiview(model, prepared: dict[int, tuple], *, device="cpu") -> dict[int, object]:
    """Run all prepared FOVs in one frozen UNI2 forward when possible."""
    fovs = list(prepared)
    if not fovs:
        return {}
    sizes = [len(prepared[fov][0]) for fov in fovs]
    x = torch.cat([prepared[fov][0] for fov in fovs], dim=0)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        x = x.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)

    with torch.inference_mode():
        tokens = model.forward_features(x)
        spatial = extract_spatial_tokens(tokens)

    out = {}
    start = 0
    for fov, size in zip(fovs, sizes):
        stop = start + size
        _, tx, ty = prepared[fov]
        h = gaussian_pool(spatial[start:stop], tx, ty)
        out[fov] = h.float().cpu().numpy()
        start = stop
    return out


def extract_batch(model, rows, fov: int, device="cpu", image_cache: ROIImageCache | None = None):
    """Compatibility single-view extractor."""
    cache = image_cache or ROIImageCache(4)
    prepared = prepare_multiview_batch(rows, (int(fov),), image_cache=cache)
    return run_prepared_multiview(model, prepared, device=device)[int(fov)]
