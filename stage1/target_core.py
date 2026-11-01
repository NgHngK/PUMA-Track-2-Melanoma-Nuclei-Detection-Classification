"""Exact reference Stage-1 RGB loading, Gaussian target and collision-safe cell assignment.
Selected source functions copied without algorithmic changes from the supplied Google Drive project.
"""

from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import tifffile
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from .constants import ROI_SIZE


def _read_rgb(path: Path, size: int = ROI_SIZE) -> np.ndarray:
    image = np.asarray(tifffile.imread(path))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim == 3 and image.shape[0] in (3, 4) and (image.shape[-1] not in (3, 4)):
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"Expected RGB TIFF, got {image.shape}: {path}")
    image = image[..., :3]
    if image.shape[:2] != (size, size):
        raise ValueError(f"Expected {size}x{size} ROI, got {image.shape}: {path}")
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.integer):
            integer = image.astype(np.int64, copy=False)
            minimum = int(integer.min())
            maximum = int(integer.max())
            if minimum < 0 or maximum > 255:
                raise ValueError(
                    f"Integer TIFF dtype {image.dtype} has RGB range [{minimum}, {maximum}] outside 0..255: {path}. Refusing ambiguous dtype-based intensity rescaling; convert the ROI explicitly to 8-bit RGB first."
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
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Gaussian sigma must be finite and positive")
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
    centers: list[tuple[float, float]], *, heatmap_size: int, stride: int, max_offset: float
) -> list[tuple[int, int, float, float]]:
    """Globally assign one representable output cell per nucleus.

    This is a sparse minimum-cost bipartite matching problem. A greedy
    nucleus-by-nucleus assignment can fail even when a valid complete matching
    exists (an early flexible nucleus can steal the only cell available to a
    later constrained nucleus). The sparse solver minimizes total squared
    localization offset while enforcing one unique grid cell per nucleus.
    """
    if heatmap_size <= 0 or stride <= 0 or not math.isfinite(max_offset) or max_offset <= 0:
        raise ValueError("heatmap_size, stride and max_offset must be positive")
    if not centers:
        return []
    candidate_cells: list[list[tuple[int, int, float, float, float]]] = []
    all_cells: set[tuple[int, int]] = set()
    for nucleus_index, (x, y) in enumerate(centers):
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"Nonfinite nucleus coordinates at index {nucleus_index}")
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
                if abs(offset_x) <= max_offset + 1e-09 and abs(offset_y) <= max_offset + 1e-09:
                    distance2 = offset_x * offset_x + offset_y * offset_y
                    candidates.append((gx, gy, float(offset_x), float(offset_y), float(distance2)))
                    all_cells.add((gx, gy))
        if not candidates:
            raise RuntimeError(
                f"Stage-1 target cannot be represented within the configured offset range for nucleus {nucleus_index} at ({x:.3f}, {y:.3f})."
            )
        candidate_cells.append(candidates)
    if len(all_cells) < len(centers):
        raise RuntimeError(
            f"Stage-1 target collision cannot be represented within the configured offset range: {len(centers)} nuclei share only {len(all_cells)} candidate grid cells. Increase stage1.offset_range_grid_units or inspect duplicate/dense annotations."
        )
    ordered_cells = sorted(all_cells, key=lambda cell: (cell[1], cell[0]))
    cell_to_column = {cell: index for index, cell in enumerate(ordered_cells)}
    row_indices: list[int] = []
    column_indices: list[int] = []
    costs: list[float] = []
    candidate_lookup: list[dict[tuple[int, int], tuple[float, float]]] = []
    tie_scale = 1e-10 / max(1, len(ordered_cells))
    for nucleus_index, candidates in enumerate(candidate_cells):
        lookup: dict[tuple[int, int], tuple[float, float]] = {}
        for gx, gy, offset_x, offset_y, distance2 in candidates:
            column = cell_to_column[gx, gy]
            row_indices.append(nucleus_index)
            column_indices.append(column)
            costs.append(float(distance2) + 1e-12 + tie_scale * column)
            lookup[gx, gy] = (offset_x, offset_y)
        candidate_lookup.append(lookup)
    graph = coo_matrix(
        (np.asarray(costs, dtype=np.float64), (row_indices, column_indices)),
        shape=(len(centers), len(ordered_cells)),
    ).tocsr()
    try:
        matched_rows, matched_columns = min_weight_full_bipartite_matching(graph)
    except ValueError as exc:
        raise RuntimeError(
            "Stage-1 target collision cannot be represented within the configured offset range. Increase stage1.offset_range_grid_units or inspect duplicate/dense annotations."
        ) from exc
    if len(matched_rows) != len(centers) or set(matched_rows.tolist()) != set(range(len(centers))):
        raise RuntimeError("Stage-1 target matching did not cover every nucleus")
    assignments: list[tuple[int, int, float, float] | None] = [None] * len(centers)
    for nucleus_index, column in zip(matched_rows.tolist(), matched_columns.tolist(), strict=True):
        gx, gy = ordered_cells[int(column)]
        offset_x, offset_y = candidate_lookup[int(nucleus_index)][gx, gy]
        assignments[int(nucleus_index)] = (gx, gy, float(offset_x), float(offset_y))
    if any((value is None for value in assignments)):
        raise RuntimeError("Stage-1 target matching left an unassigned nucleus")
    return [value for value in assignments if value is not None]
