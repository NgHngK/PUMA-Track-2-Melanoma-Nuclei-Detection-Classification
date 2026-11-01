"""Read-only mmap training and validation dataset with geometry-correct D4 transforms."""

from __future__ import annotations
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from common.run_history import file_fingerprint
from .target_core import _assign_unique_grid_cells, _gaussian_patch


class Cache:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.meta = json.loads((self.folder / "manifest.json").read_text())
        self.name_to_index = {s: i for i, s in enumerate(self.meta["names"])}
        self.points = np.load(self.folder / "points.npy", allow_pickle=False)
        self.gt = np.load(self.folder / "gt.npy", allow_pickle=False)
        self.offsets = np.load(self.folder / "offsets.npy", allow_pickle=False)
        self.size = int(self.meta["size"])
        self._images = None
        self._heatmaps = None
        self._validate()

    def _validate(self):
        """Reject caches whose sparse supervision no longer matches their GT."""
        files = {"images.npy", "heatmaps.npy", "points.npy", "gt.npy", "offsets.npy"}
        for name, digest in self.meta.get("files_sha256", {}).items():
            if name not in files or file_fingerprint(self.folder / name) != digest:
                raise ValueError(f"Stage 1 cache file checksum mismatch: {name}")
        n = len(self.meta["names"])
        if len(self.name_to_index) != n or self.size < 64 or self.size % 32:
            raise ValueError("Invalid Stage 1 cache names or ROI size")
        images = np.load(self.folder / "images.npy", mmap_mode="r", allow_pickle=False)
        heatmaps = np.load(self.folder / "heatmaps.npy", mmap_mode="r", allow_pickle=False)
        if images.shape != (n, self.size, self.size, 3) or images.dtype != np.uint8:
            raise ValueError("Stage 1 image cache must contain uint8 RGB ROIs")
        if heatmaps.shape != (n, self.size // 2, self.size // 2) or heatmaps.dtype.kind != "f":
            raise ValueError("Stage 1 heatmap cache has incorrect shape or dtype")
        if (
            self.points.ndim != 1
            or self.gt.ndim != 1
            or not {"grid_x", "grid_y", "offset_x", "offset_y"}
            <= set(self.points.dtype.names or ())
            or not {"x", "y", "class_id"} <= set(self.gt.dtype.names or ())
        ):
            raise ValueError("Stage 1 sparse targets have invalid fields")
        if (
            len(self.points) != len(self.gt)
            or self.meta.get("num_nuclei", len(self.gt)) != len(self.gt)
            or self.offsets.shape != (n + 1,)
            or self.offsets.dtype.kind not in "iu"
            or self.offsets[0] != 0
            or self.offsets[-1] != len(self.gt)
            or np.any(self.offsets[1:] < self.offsets[:-1])
        ):
            raise ValueError("Stage 1 cached nucleus indices are incomplete")
        for field in ("grid_x", "grid_y"):
            value = self.points[field]
            if value.dtype.kind not in "iu" or np.any(value < 0) or np.any(value >= self.size // 2):
                raise ValueError("Stage 1 cached grid coordinates are outside the heatmap")
        for field in ("offset_x", "offset_y"):
            value = self.points[field]
            if not np.isfinite(value).all() or np.any(np.abs(value) > 2.5):
                raise ValueError("Stage 1 cached offsets exceed the detector range")
        for coordinate, grid, offset in [("x", "grid_x", "offset_x"), ("y", "grid_y", "offset_y")]:
            values = self.gt[coordinate]
            if not np.isfinite(values).all() or not np.allclose(
                2 * (self.points[grid].astype(np.float64) + self.points[offset]),
                values,
                rtol=0,
                atol=1e-4,
            ):
                raise ValueError("Stage 1 cached target coordinates do not reconstruct GT")
        labels = self.gt["class_id"]
        if labels.dtype.kind not in "iu" or np.any(labels < 0) or np.any(labels >= 10):
            raise ValueError("Stage 1 cache contains invalid nucleus classes")
        if "nucleus_index" in self.points.dtype.names:
            if not np.array_equal(self.points["nucleus_index"], np.arange(len(self.points))):
                raise ValueError("Stage 1 cached nucleus IDs are inconsistent")
        for i in range(n):
            points = self.points[self.offsets[i] : self.offsets[i + 1]]
            if "roi_index" in self.points.dtype.names and np.any(points["roi_index"] != i):
                raise ValueError("Stage 1 cached ROI boundaries are inconsistent")
            cells = points["grid_y"].astype(np.int64) * (self.size // 2) + points["grid_x"]
            if len(np.unique(cells)) != len(cells):
                raise ValueError("Stage 1 cache has colliding nucleus grid cells")
            heatmap = heatmaps[i]
            if not np.isfinite(heatmap).all() or np.any(heatmap < 0) or np.any(heatmap > 1):
                raise ValueError("Stage 1 cache contains invalid heatmap values")
            if not np.all(heatmap[points["grid_y"], points["grid_x"]] == 1):
                raise ValueError("Stage 1 heatmap peaks disagree with sparse target cells")

    @property
    def images(self):
        if self._images is None:
            self._images = np.load(self.folder / "images.npy", mmap_mode="r")
        return self._images

    @property
    def heatmaps(self):
        if self._heatmaps is None:
            self._heatmaps = np.load(self.folder / "heatmaps.npy", mmap_mode="r")
        return self._heatmaps

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_images"] = None
        d["_heatmaps"] = None
        return d

    def gt_for(self, name: str) -> np.ndarray:
        idx = self.name_to_index[name]
        return self.gt[self.offsets[idx] : self.offsets[idx + 1]]


def tile_origins(size: int, tile: int, overlap: int = 0) -> list[int]:
    if tile > size or tile % 32 or tile < 64 or (overlap < 0) or (overlap >= tile):
        raise ValueError("Require 64 <= tile <= size, tile divisible by 32, 0 <= overlap < tile")
    result = list(range(0, size - tile + 1, tile - overlap))
    if result[-1] != size - tile:
        result.append(size - tile)
    return result


def _transform_points_rot90(points: np.ndarray, grid_size: int):
    out = points.copy()
    gx = points["grid_x"].astype(np.int64)
    gy = points["grid_y"].astype(np.int64)
    ox = points["offset_x"].copy()
    oy = points["offset_y"].copy()
    out["grid_x"] = gy
    out["grid_y"] = grid_size - 1 - gx
    out["offset_x"] = oy
    out["offset_y"] = 0.5 - ox
    return out


def _transform_points_hflip(points: np.ndarray, grid_size: int):
    out = points.copy()
    out["grid_x"] = grid_size - 1 - points["grid_x"].astype(np.int64)
    out["offset_x"] = 0.5 - points["offset_x"]
    return out


def apply_d4(
    image: torch.Tensor, hm: torch.Tensor, points: np.ndarray, code: int, *, sigma: float = 1.7
):
    """Same seven non-identity D4 transformations used by original trainer."""
    original = image, hm, points
    n = hm.shape[-1]
    for _ in range(code % 4):
        image = torch.rot90(image, 1, (-2, -1))
        hm = torch.rot90(hm, 1, (-2, -1))
        points = _transform_points_rot90(points, n)
    if code >= 4:
        image = torch.flip(image, (-1,))
        hm = torch.flip(hm, (-1,))
        points = _transform_points_hflip(points, n)
    # A pixel reflection maps o to 0.5-o on this stride-2 grid. The
    # symmetric [-2.5, 2.5] head range is therefore not D4-invariant.
    if np.any(np.abs(points["offset_x"]) > 2.5) or np.any(np.abs(points["offset_y"]) > 2.5):
        centers = list(
            zip(
                2 * (points["grid_x"].astype(np.float64) + points["offset_x"]),
                2 * (points["grid_y"].astype(np.float64) + points["offset_y"]),
            )
        )
        try:
            assignments = _assign_unique_grid_cells(
                centers, heatmap_size=n, stride=2, max_offset=2.5
            )
        except RuntimeError:
            # Extreme density or nuclei assigned across a crop boundary can
            # make this transformed view unrepresentable. Keep every original
            # nucleus and use the valid original view for this sample.
            return original
        points = points.copy()
        rebuilt = np.zeros((n, n), dtype=np.float32)
        for i, (gx, gy, ox, oy) in enumerate(assignments):
            points["grid_x"][i], points["grid_y"][i] = gx, gy
            points["offset_x"][i], points["offset_y"][i] = ox, oy
            _gaussian_patch(rebuilt, gx, gy, sigma)
        hm = torch.as_tensor(rebuilt, dtype=hm.dtype, device=hm.device)
    return (image, hm, points)


def photometric_jitter(image: torch.Tensor) -> torch.Tensor:
    bright = torch.empty(()).uniform_(0.92, 1.08)
    contrast = torch.empty(()).uniform_(0.9, 1.1)
    gamma = torch.empty(()).uniform_(0.9, 1.1)
    channel = torch.empty(3, 1, 1).uniform_(0.96, 1.04)
    mean = image.mean(dim=(1, 2), keepdim=True)
    return ((image - mean) * contrast + mean).mul(bright * channel).clamp_(0, 1).pow_(gamma)


class TileDataset(Dataset):
    def __init__(self, cache: Cache, names: list[str], tile_size: int, train: bool = False):
        self.cache = cache
        self.tile_size = tile_size
        self.train = train
        locations = tile_origins(cache.size, tile_size)
        self.samples = [
            (cache.name_to_index[n], x, y) for n in names for y in locations for x in locations
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        idx, x, y = self.samples[index]
        s = self.tile_size
        gx = x // 2
        gy = y // 2
        g = s // 2
        image = np.array(self.cache.images[idx, y : y + s, x : x + s, :], copy=True)
        hm = np.array(
            self.cache.heatmaps[idx, gy : gy + g, gx : gx + g], dtype=np.float32, copy=True
        )
        points = self.cache.points[self.cache.offsets[idx] : self.cache.offsets[idx + 1]]
        in_tile = (
            (points["grid_x"] >= gx)
            & (points["grid_x"] < gx + g)
            & (points["grid_y"] >= gy)
            & (points["grid_y"] < gy + g)
        )
        local = points[in_tile].copy()
        local["grid_x"] -= gx
        local["grid_y"] -= gy
        image_t = torch.from_numpy(image).permute(2, 0, 1).float().div_(255)
        hm_t = torch.from_numpy(hm)
        if self.train:
            if torch.rand(()) < 0.875:
                code = int(torch.randint(1, 8, ()).item())
                image_t, hm_t, local = apply_d4(
                    image_t, hm_t, local, code, sigma=float(self.cache.meta.get("sigma", 1.7))
                )
            image_t = photometric_jitter(image_t)
        return {
            "image": image_t.contiguous(),
            "heatmap": hm_t.contiguous(),
            "points": local,
            "roi": idx,
            "x": x,
            "y": y,
        }


def collate(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "heatmap": torch.stack([b["heatmap"] for b in batch]),
        "points": [b["points"] for b in batch],
        "roi": [b["roi"] for b in batch],
        "x": [b["x"] for b in batch],
        "y": [b["y"] for b in batch],
    }
