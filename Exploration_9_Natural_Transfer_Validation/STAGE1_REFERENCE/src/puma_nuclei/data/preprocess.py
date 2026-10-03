from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from ..config import PipelineConfig
from ..constants import NUCLEUS_DTYPE, NUMBER_OF_CLASSES, ROI_MANIFEST_DTYPE, ROI_SIZE, STAGE1_OUTPUT_STRIDE
from ..utils.provenance import canonical_fingerprint, file_stat_identity
from .annotations import annotation_centroid, infer_case_id, infer_sample_type, load_nuclei_polygons, normalized_stem
from .folds import grouped_balanced_folds

POINT_TARGET_DTYPE = np.dtype(
    [
        ("roi_index", "i4"),
        ("nucleus_index", "i4"),
        ("grid_x", "i2"),
        ("grid_y", "i2"),
        ("offset_x", "f4"),
        ("offset_y", "f4"),
    ]
)


def _read_rgb(path: Path) -> np.ndarray:
    image = np.asarray(tifffile.imread(path))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim == 3 and image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"Expected RGB TIFF, got {image.shape}: {path}")
    image = image[..., :3]
    if image.shape[:2] != (ROI_SIZE, ROI_SIZE):
        raise ValueError(f"Expected {ROI_SIZE}x{ROI_SIZE} ROI, got {image.shape}: {path}")
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.integer):
            integer = image.astype(np.int64, copy=False)
            minimum = int(integer.min())
            maximum = int(integer.max())
            if minimum < 0 or maximum > 255:
                raise ValueError(
                    f"Integer TIFF dtype {image.dtype} has RGB range [{minimum}, {maximum}] outside 0..255: {path}. "
                    "Refusing ambiguous dtype-based intensity rescaling; convert the ROI explicitly to 8-bit RGB first."
                )
            image = integer.astype(np.uint8)
        elif np.issubdtype(image.dtype, np.floating):
            floating = image.astype(np.float32, copy=False)
            if not np.isfinite(floating).all():
                raise ValueError(f"Floating TIFF contains NaN/Inf: {path}")
            minimum = float(floating.min())
            maximum = float(floating.max())
            if minimum < 0.0:
                raise ValueError(f"Floating TIFF has negative intensity values: {path}")
            if maximum <= 1.5:
                floating = floating * 255.0
            elif maximum > 255.0:
                raise ValueError(
                    f"Floating TIFF range [{minimum:.3f}, {maximum:.3f}] is not an understood 0..1 or 0..255 RGB range: {path}"
                )
            image = np.rint(floating).clip(0, 255).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported TIFF dtype {image.dtype}: {path}")
    return np.ascontiguousarray(image)


def _gaussian_patch(heatmap: np.ndarray, grid_x: int, grid_y: int, sigma: float) -> None:
    # The sparse offset target stores the sub-grid displacement. The heatmap
    # center itself therefore belongs on the integer output cell so that every
    # nucleus has an exact target value of 1.0. Centering the Gaussian at the
    # fractional coordinate makes almost every peak < 1, while focal_heatmap_loss
    # defines positives with target == 1.0; that silently removes positive
    # heatmap supervision.
    gx = int(grid_x)
    gy = int(grid_y)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x0 = max(0, gx - radius)
    x1 = min(heatmap.shape[1], gx + radius + 1)
    y0 = max(0, gy - radius)
    y1 = min(heatmap.shape[0], gy + radius + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    values = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2.0 * sigma**2))
    np.maximum(heatmap[y0:y1, x0:x1], values, out=heatmap[y0:y1, x0:x1])


def _assign_unique_grid_cells(
    centers: list[tuple[float, float]],
    *,
    heatmap_size: int,
    stride: int,
    max_offset: float,
) -> list[tuple[int, int, float, float]]:
    """Globally assign one representable output cell per nucleus.

    This is a sparse minimum-cost bipartite matching problem. A greedy
    nucleus-by-nucleus assignment can fail even when a valid complete matching
    exists (an early flexible nucleus can steal the only cell available to a
    later constrained nucleus). The sparse solver minimizes total squared
    localization offset while enforcing one unique grid cell per nucleus.
    """
    if not centers:
        return []
    if heatmap_size <= 0 or stride <= 0 or max_offset <= 0:
        raise ValueError("heatmap_size, stride and max_offset must be positive")

    candidate_cells: list[list[tuple[int, int, float, float, float]]] = []
    all_cells: set[tuple[int, int]] = set()
    for nucleus_index, (x, y) in enumerate(centers):
        cx = float(x) / float(stride)
        cy = float(y) / float(stride)
        gx0 = max(0, int(math.ceil(cx - max_offset)))
        gx1 = min(heatmap_size - 1, int(math.floor(cx + max_offset)))
        gy0 = max(0, int(math.ceil(cy - max_offset)))
        gy1 = min(heatmap_size - 1, int(math.floor(cy + max_offset)))
        candidates: list[tuple[int, int, float, float, float]] = []
        for gy in range(gy0, gy1 + 1):
            for gx in range(gx0, gx1 + 1):
                offset_x = cx - gx
                offset_y = cy - gy
                if abs(offset_x) <= max_offset + 1e-9 and abs(offset_y) <= max_offset + 1e-9:
                    distance2 = offset_x * offset_x + offset_y * offset_y
                    candidates.append((gx, gy, float(offset_x), float(offset_y), float(distance2)))
                    all_cells.add((gx, gy))
        if not candidates:
            raise RuntimeError(
                "Stage-1 target cannot be represented within the configured offset range "
                f"for nucleus {nucleus_index} at ({x:.3f}, {y:.3f})."
            )
        candidate_cells.append(candidates)

    if len(all_cells) < len(centers):
        raise RuntimeError(
            "Stage-1 target collision cannot be represented within the configured offset range: "
            f"{len(centers)} nuclei share only {len(all_cells)} candidate grid cells. "
            "Increase stage1.offset_range_grid_units or inspect duplicate/dense annotations."
        )

    ordered_cells = sorted(all_cells, key=lambda cell: (cell[1], cell[0]))
    cell_to_column = {cell: index for index, cell in enumerate(ordered_cells)}
    row_indices: list[int] = []
    column_indices: list[int] = []
    costs: list[float] = []
    candidate_lookup: list[dict[tuple[int, int], tuple[float, float]]] = []
    tie_scale = 1.0e-10 / max(1, len(ordered_cells))
    for nucleus_index, candidates in enumerate(candidate_cells):
        lookup: dict[tuple[int, int], tuple[float, float]] = {}
        for gx, gy, offset_x, offset_y, distance2 in candidates:
            column = cell_to_column[(gx, gy)]
            row_indices.append(nucleus_index)
            column_indices.append(column)
            # Sparse matching treats exact zeros as absent; the tiny positive
            # deterministic term also resolves equal-distance cell choices.
            costs.append(float(distance2) + 1.0e-12 + tie_scale * column)
            lookup[(gx, gy)] = (offset_x, offset_y)
        candidate_lookup.append(lookup)

    graph = coo_matrix(
        (np.asarray(costs, dtype=np.float64), (row_indices, column_indices)),
        shape=(len(centers), len(ordered_cells)),
    ).tocsr()
    try:
        matched_rows, matched_columns = min_weight_full_bipartite_matching(graph)
    except ValueError as exc:
        raise RuntimeError(
            "Stage-1 target collision cannot be represented within the configured offset range. "
            "Increase stage1.offset_range_grid_units or inspect duplicate/dense annotations."
        ) from exc
    if len(matched_rows) != len(centers) or set(matched_rows.tolist()) != set(range(len(centers))):
        raise RuntimeError("Stage-1 target matching did not cover every nucleus")

    assignments: list[tuple[int, int, float, float] | None] = [None] * len(centers)
    for nucleus_index, column in zip(matched_rows.tolist(), matched_columns.tolist(), strict=True):
        gx, gy = ordered_cells[int(column)]
        offset_x, offset_y = candidate_lookup[int(nucleus_index)][(gx, gy)]
        assignments[int(nucleus_index)] = (gx, gy, float(offset_x), float(offset_y))
    if any(value is None for value in assignments):
        raise RuntimeError("Stage-1 target matching left an unassigned nucleus")
    return [value for value in assignments if value is not None]


def _process_one(args: tuple[int, Path, Path]) -> dict[str, Any]:
    index, image_path, nuclei_path = args
    return {
        "index": index,
        "image": _read_rgb(image_path),
        "nuclei_path": nuclei_path,
        "nuclei": load_nuclei_polygons(nuclei_path),
        "case_id": infer_case_id(image_path),
        "sample_type": infer_sample_type(image_path),
    }


def _annotation_entries(directory: Path, suffix: str = ".geojson") -> list[tuple[str, Path]]:
    """Index annotation filenames once instead of re-scanning the directory per ROI."""
    if not directory.is_dir():
        return []
    return [(normalized_stem(path), path) for path in sorted(directory.rglob(f"*{suffix}"))]


def _resolve_indexed_annotation(image_path: Path, entries: list[tuple[str, Path]]) -> Path | None:
    target = normalized_stem(image_path)
    exact = [path for stem, path in entries if stem == target]
    candidates = exact
    if not candidates:
        candidates = [path for stem, path in entries if stem.endswith(target) or target.endswith(stem)]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None
    raise RuntimeError(f"Ambiguous annotation for {image_path.name}: {[str(p) for p in candidates]}")


def _relative_source_identity(path: Path, root: Path) -> dict[str, Any]:
    identity = file_stat_identity(path)
    try:
        identity["path"] = str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        identity["path"] = path.name
    return identity


def _shared_preprocess_contract(
    config: PipelineConfig,
    images: list[Path],
    nuclei_paths: list[Path],
) -> dict[str, Any]:
    """Fast stale-cache guard for the shared preprocessing directory.

    Raw TIFFs can be many gigabytes. Reading every byte only to decide whether a
    cache may be reused is counterproductive on Google Drive, so source identity
    uses relative path + byte size + nanosecond mtime. The semantic preprocessing
    code/config is independently content-fingerprinted.
    """
    payload = {
        "schema": "puma-shared-preprocess-v2",
        "data": {
            "number_of_folds": int(config.data.number_of_folds),
            "seed": int(config.data.seed),
            "fold_search_restarts": int(config.data.fold_search_restarts),
            "heatmap_sigma_grid_units": float(config.data.heatmap_sigma_grid_units),
            "stage1_output_stride": int(config.data.stage1_output_stride),
        },
        "stage1_target": {
            "offset_range_grid_units": float(config.stage1.offset_range_grid_units),
        },
        "images": [_relative_source_identity(path, config.paths.images) for path in images],
        "nuclei_annotations": [
            _relative_source_identity(path, config.paths.nuclei_annotations) for path in nuclei_paths
        ],
    }
    here = Path(__file__).resolve().parent
    payload["contract_fingerprint"] = canonical_fingerprint(
        payload,
        (
            Path(__file__),
            here / "annotations.py",
            here / "folds.py",
            Path(__file__).resolve().parents[1] / "constants.py",
        ),
    )
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def preprocess_shared_data(config: PipelineConfig, *, force: bool = False) -> dict[str, Any]:
    config.validate()
    output = config.paths.preprocessed
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "preprocessing_summary.json"
    required = (
        "images.npy", "roi_manifest.npy", "nuclei.npy", "polygon_points.npy",
        "stage1_heatmaps.npy", "stage1_point_targets.npy", "roi_nucleus_offsets.npy",
        "folds.npy", "class_counts_by_roi.npy",
    )
    images = sorted({*config.paths.images.rglob("*.tif"), *config.paths.images.rglob("*.tiff")})
    if not images:
        raise FileNotFoundError(f"No TIFF images found in {config.paths.images}")

    nuclei_entries = _annotation_entries(config.paths.nuclei_annotations)
    nuclei_paths: list[Path] = []
    for image_path in images:
        nuclei_path = _resolve_indexed_annotation(image_path, nuclei_entries)
        if nuclei_path is None:
            raise FileNotFoundError(f"No nuclei annotation for {image_path}")
        nuclei_paths.append(nuclei_path)

    contract = _shared_preprocess_contract(config, images, nuclei_paths)
    if not force and summary_path.is_file() and all((output / name).is_file() for name in required):
        cached = json.loads(summary_path.read_text(encoding="utf-8"))
        cached_contract = cached.get("cache_contract")
        cached_identities = cached.get("artifact_identity")
        if cached_contract is None:
            # Backward-compatible adoption keeps already-trained Stage-1 runs
            # usable. Bind the exact current output file identities as well, so
            # any later truncation/replacement is detected without hashing GiBs.
            cached["cache_contract"] = contract
            cached["cache_contract_status"] = "legacy_adopted"
            cached["artifact_identity"] = {name: file_stat_identity(output / name) for name in required}
            _atomic_write_json(summary_path, cached)
            print(f"Validated file set and adopted legacy preprocessing cache: {output}")
            return cached
        identities_current = (
            isinstance(cached_identities, dict)
            and all(cached_identities.get(name) == file_stat_identity(output / name) for name in required)
        )
        if cached_contract == contract and identities_current:
            print(f"Reusing preprocessing: {output}")
            return cached
        if cached_contract == contract and not identities_current:
            print("Shared preprocessing artifact identity changed; rebuilding cache.")
        else:
            print("Shared preprocessing source/config/code contract changed; rebuilding cache.")

    workers = config.data.resolved_workers
    arguments = [(index, path, nuclei_paths[index]) for index, path in enumerate(images)]
    if workers == 0:
        records = [_process_one(arg) for arg in arguments]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(_process_one, arguments))
    records.sort(key=lambda item: item["index"])

    case_ids = [str(record["case_id"]) for record in records]
    if len(set(case_ids)) < config.data.number_of_folds:
        raise ValueError("Not enough independent groups for the requested number of folds")

    image_store = np.lib.format.open_memmap(output / "images.npy", mode="w+", dtype=np.uint8, shape=(len(images), ROI_SIZE, ROI_SIZE, 3))

    manifest = np.empty(len(images), dtype=ROI_MANIFEST_DTYPE)
    heatmap_size = ROI_SIZE // STAGE1_OUTPUT_STRIDE
    heatmaps = np.lib.format.open_memmap(output / "stage1_heatmaps.npy", mode="w+", dtype=np.float16, shape=(len(images), heatmap_size, heatmap_size))
    heatmaps[:] = 0
    nucleus_rows: list[tuple[Any, ...]] = []
    point_rows: list[tuple[Any, ...]] = []
    polygon_points: list[np.ndarray] = []
    offsets = [0]
    class_counts = np.zeros((len(images), NUMBER_OF_CLASSES), dtype=np.int64)
    nucleus_index_global = 0
    polygon_cursor = 0

    for record, image_path in zip(records, images, strict=True):
        roi_index = int(record["index"])
        image_store[roi_index] = record["image"]
        manifest[roi_index] = (
            image_path.stem,
            str(image_path),
            str(record["nuclei_path"]),
            str(record["case_id"]),
            str(record["sample_type"]),
        )

        heatmap = np.zeros((heatmap_size, heatmap_size), dtype=np.float32)
        valid_nuclei: list[tuple[int, np.ndarray, float, float]] = []
        for class_id, points in record["nuclei"]:
            x, y = map(float, annotation_centroid(points))
            if not (0 <= x < ROI_SIZE and 0 <= y < ROI_SIZE):
                raise ValueError(f"Nucleus centroid outside ROI in {image_path}: ({x:.3f}, {y:.3f})")
            valid_nuclei.append((int(class_id), points, x, y))

        assignments = _assign_unique_grid_cells(
            [(x, y) for _, _, x, y in valid_nuclei],
            heatmap_size=heatmap_size,
            stride=STAGE1_OUTPUT_STRIDE,
            max_offset=float(config.stage1.offset_range_grid_units),
        )
        for (class_id, points, x, y), (grid_x, grid_y, offset_x, offset_y) in zip(valid_nuclei, assignments, strict=True):
            length = len(points)
            polygon_points.append(points.astype(np.float32, copy=False))
            nucleus_rows.append((roi_index, nucleus_index_global, x, y, class_id, polygon_cursor, length))
            point_rows.append((roi_index, nucleus_index_global, grid_x, grid_y, offset_x, offset_y))
            class_counts[roi_index, class_id] += 1
            _gaussian_patch(heatmap, grid_x, grid_y, config.data.heatmap_sigma_grid_units)
            polygon_cursor += length
            nucleus_index_global += 1
        heatmaps[roi_index] = heatmap.astype(np.float16)
        offsets.append(len(nucleus_rows))

    del image_store, heatmaps
    np.save(output / "roi_manifest.npy", manifest, allow_pickle=False)
    np.save(output / "nuclei.npy", np.asarray(nucleus_rows, dtype=NUCLEUS_DTYPE), allow_pickle=False)
    all_points = np.concatenate(polygon_points, axis=0) if polygon_points else np.empty((0, 2), dtype=np.float32)
    np.save(output / "polygon_points.npy", all_points, allow_pickle=False)
    np.save(output / "stage1_point_targets.npy", np.asarray(point_rows, dtype=POINT_TARGET_DTYPE), allow_pickle=False)
    np.save(output / "roi_nucleus_offsets.npy", np.asarray(offsets, dtype=np.int64), allow_pickle=False)
    np.save(output / "class_counts_by_roi.npy", class_counts, allow_pickle=False)

    folds = grouped_balanced_folds(class_counts, case_ids, config.data.number_of_folds, config.data.seed, config.data.fold_search_restarts)
    np.save(output / "folds.npy", folds, allow_pickle=False)

    positive_roi_counts = (class_counts > 0).sum(axis=0).astype(int)
    payload = {
        "roi_count": int(len(images)),
        "nuclei_count": int(class_counts.sum()),
        "class_counts": class_counts.sum(axis=0).astype(int).tolist(),
        "positive_roi_counts": positive_roi_counts.tolist(),
        "fold_roi_counts": np.bincount(folds.astype(int), minlength=config.data.number_of_folds).astype(int).tolist(),
        "cpu_workers": workers,
        "cache_contract": contract,
        "cache_contract_status": "current",
        "artifact_identity": {
            name: file_stat_identity(output / name) for name in required
        },
    }
    _atomic_write_json(summary_path, payload)
    return payload


def load_shared_artifacts(config: PipelineConfig, mmap_mode: str = "r") -> dict[str, np.ndarray | None]:
    root = config.paths.preprocessed
    return {
        "images": np.load(root / "images.npy", mmap_mode=mmap_mode, allow_pickle=False),
        "manifest": np.load(root / "roi_manifest.npy", mmap_mode=mmap_mode, allow_pickle=False),
        "nuclei": np.load(root / "nuclei.npy", mmap_mode=mmap_mode, allow_pickle=False),
        "polygon_points": np.load(root / "polygon_points.npy", mmap_mode=mmap_mode, allow_pickle=False),
        "heatmaps": np.load(root / "stage1_heatmaps.npy", mmap_mode=mmap_mode, allow_pickle=False),
        "point_targets": np.load(root / "stage1_point_targets.npy", mmap_mode=mmap_mode, allow_pickle=False),
        "offsets": np.load(root / "roi_nucleus_offsets.npy", mmap_mode=mmap_mode, allow_pickle=False),
        "folds": np.load(root / "folds.npy", mmap_mode=mmap_mode, allow_pickle=False),
        "class_counts": np.load(root / "class_counts_by_roi.npy", mmap_mode=mmap_mode, allow_pickle=False),
    }

