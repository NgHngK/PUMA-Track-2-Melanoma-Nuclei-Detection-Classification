"""One-time disk-backed preprocessing of raw TIFF/GeoJSON using supplied ROI names."""

from __future__ import annotations
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from numpy.lib.format import open_memmap

from common.resources import cpu_workers
from common.run_history import file_fingerprint
from .annotations import load_nuclei_instances, normalized_stem
from .splits import expected_names
from .target_core import _read_rgb, _gaussian_patch, _assign_unique_grid_cells

POINT_DTYPE = np.dtype(
    [
        ("roi_index", "i4"),
        ("nucleus_index", "i4"),
        ("grid_x", "i2"),
        ("grid_y", "i2"),
        ("offset_x", "f4"),
        ("offset_y", "f4"),
    ]
)
GT_DTYPE = np.dtype([("x", "f4"), ("y", "f4"), ("class_id", "i2")])


def _index_files(root: Path, suffixes: set[str]) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    files: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        key = normalized_stem(path)
        if key in files:
            raise ValueError(f"Ambiguous duplicate ROI basename {key}: {files[key]}, {path}")
        files[key] = path
    return files


def prepare(
    images_dir: Path,
    annotations_dir: Path,
    output: Path,
    *,
    plan: str = "B",
    size: int = 1024,
    sigma: float = 1.7,
    allow_existing: bool = True,
    split_root: Path | None = None,
    workers: int = 0,
) -> dict:
    """Write image/heatmap .npy memory maps plus compact sparse centroid targets.

    Fail fast if *any* split filename lacks a corresponding input image or annotation.
    Never write an apparently complete cache from a subset of the fold names.
    """
    images_dir, annotations_dir, output = map(Path, (images_dir, annotations_dir, output))
    if size < 64 or size % 32:
        raise ValueError("ROI size must be divisible by 32 and >=64")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Gaussian sigma must be finite and positive")
    names = expected_names(plan, split_root)
    image_paths = _index_files(images_dir, {".tif", ".tiff"})
    ann_paths = _index_files(annotations_dir, {".geojson", ".json"})
    missing_images = sorted(set(names) - set(image_paths))
    missing_anns = sorted(set(names) - set(ann_paths))
    if missing_images or missing_anns:
        raise FileNotFoundError(
            f"Missing {len(missing_images)} TIFFs and {len(missing_anns)} annotations. TIFF examples={missing_images[:5]}, GeoJSON examples={missing_anns[:5]}"
        )
    source_facts = []
    for name in names:
        for path in (image_paths[name], ann_paths[name]):
            stat = path.stat()
            source_facts.append(
                (name, str(path.resolve()), stat.st_size, stat.st_mtime_ns, file_fingerprint(path))
            )
    signature = {
        "source_file_metadata": source_facts,
        "names": names,
        "images_dir": str(images_dir.resolve()),
        "annotations_dir": str(annotations_dir.resolve()),
        "size": size,
        "sigma": sigma,
        "stride": 2,
        "target_version": "puma_one_feature_one_nucleus_geometry_centroid_v2",
    }
    signature["id"] = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    meta_file = output / "manifest.json"
    if allow_existing and meta_file.exists():
        old = json.loads(meta_file.read_text())
        if old.get("id") != signature["id"]:
            raise ValueError(
                "Existing cache belongs to different dataset/config. Use a NEW --cache directory."
            )
        required = ["images.npy", "heatmaps.npy", "points.npy", "gt.npy", "offsets.npy"]
        if all((output / f).is_file() for f in required):
            from .dataset import Cache

            cache = Cache(output)
            if cache.meta["names"] != names or cache.size != size:
                raise ValueError("Stage 1 cache manifest has inconsistent names or size")
            return old
        raise RuntimeError("Cache is incomplete; remove it and rerun preprocessing")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty to prevent partial cache overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    n = len(names)
    images = open_memmap(output / "images.npy", mode="w+", dtype="uint8", shape=(n, size, size, 3))
    heatmaps = open_memmap(
        output / "heatmaps.npy", mode="w+", dtype="float16", shape=(n, size // 2, size // 2)
    )
    points = []
    gt = []
    offsets = [0]

    def prepare_roi(name):
        image = _read_rgb(image_paths[name], size=size)
        instances = load_nuclei_instances(ann_paths[name])
        centers = [tuple(map(float, centroid)) for _, centroid in instances]
        if any(not (0 <= x < size and 0 <= y < size) for x, y in centers):
            raise ValueError(f"Nucleus centroid outside the {size}x{size} ROI: {name}")
        classes = [int(class_id) for class_id, _ in instances]
        assignments = _assign_unique_grid_cells(
            centers, heatmap_size=size // 2, stride=2, max_offset=2.5
        )
        heatmap = np.zeros((size // 2, size // 2), dtype=np.float32)
        for gx, gy, _, _ in assignments:
            _gaussian_patch(heatmap, gx, gy, sigma)
        return image, centers, classes, assignments, heatmap

    with ThreadPoolExecutor(max_workers=workers or cpu_workers()) as pool:
        for roi_idx, (name, prepared) in enumerate(zip(names, pool.map(prepare_roi, names))):
            image, centers, classes, assignments, heatmap = prepared
            for (x, y), class_id, (gx, gy, ox, oy) in zip(centers, classes, assignments):
                points.append((roi_idx, len(gt), gx, gy, ox, oy))
                gt.append((x, y, class_id))
            images[roi_idx] = image
            heatmaps[roi_idx] = heatmap.astype(np.float16)
            offsets.append(len(points))
            print(f"Stage 1 preprocessing {roi_idx + 1}/{n}: {name}", flush=True)
    images.flush()
    heatmaps.flush()
    del images, heatmaps
    np.save(output / "points.npy", np.asarray(points, dtype=POINT_DTYPE), allow_pickle=False)
    np.save(output / "gt.npy", np.asarray(gt, dtype=GT_DTYPE), allow_pickle=False)
    np.save(output / "offsets.npy", np.asarray(offsets, dtype=np.int64), allow_pickle=False)
    signature["num_nuclei"] = len(gt)
    signature["class_counts"] = np.bincount(
        np.asarray(gt, dtype=GT_DTYPE)["class_id"].astype(int), minlength=10
    ).tolist()
    signature["files_sha256"] = {
        name: file_fingerprint(output / name)
        for name in ("images.npy", "heatmaps.npy", "points.npy", "gt.npy", "offsets.npy")
    }
    meta_file.write_text(json.dumps(signature, indent=2) + "\n")
    return signature
