"""Class-agnostic detection evaluation: strict distance <15 pixels, one-to-one matching."""

from __future__ import annotations
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from .dataset import Cache, tile_origins
from .decode import extract_stage1_candidates, select_stage1_candidate_indices, CANDIDATE_DTYPE


def match_counts(
    predictions: np.ndarray, truth: np.ndarray, radius: float = 15.0
) -> dict[str, int]:
    """Max-cardinality matching avoids counting the same nucleus twice.

    Does not use class labels because Stage 1 is a class-agnostic detector.
    """
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and nonnegative")
    # Match the KD-tree's float64 geometry at the strict radius boundary.
    predictions = np.asarray(predictions, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    n, m = (len(predictions), len(truth))
    if not n or not m:
        return {"tp": 0, "fp": n, "fn": m}
    pairs = cKDTree(truth[:, :2]).query_ball_point(predictions[:, :2], r=radius)
    rows = []
    cols = []
    for i, near in enumerate(pairs):
        for j in near:
            if np.sum((predictions[i, :2] - truth[j, :2]) ** 2) < radius**2:
                rows.append(i)
                cols.append(j)
    if not rows:
        return {"tp": 0, "fp": n, "fn": m}
    graph = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, m))
    matched = maximum_bipartite_matching(graph, perm_type="column")
    tp = int(np.count_nonzero(matched >= 0))
    return {"tp": tp, "fp": n - tp, "fn": m - tp}


def scores(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "accuracy": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
    }


def core_bounds(origins: list[int], tile: int, size: int, index: int) -> tuple[int, int]:
    low = 0 if index == 0 else (origins[index - 1] + origins[index] + tile) // 2
    high = size if index == len(origins) - 1 else (origins[index] + origins[index + 1] + tile) // 2
    return (low, high)


@torch.inference_mode()
def choose_tile_batch_size(model, device, tile_size, amp: bool = True):
    """Find the largest fitting evaluation batch on the selected GPU."""
    if device.type != "cuda":
        return 1

    for size in (8, 4, 2, 1):
        sample = output = None
        try:
            sample = torch.zeros(size, 3, tile_size, tile_size, device=device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                output = model(sample)
            torch.cuda.synchronize()
            return size
        except torch.OutOfMemoryError:
            pass
        finally:
            del sample, output
            torch.cuda.empty_cache()

    raise RuntimeError("Not enough CUDA memory to evaluate one tile")


@torch.inference_mode()
def collect_candidates(
    model: torch.nn.Module,
    cache: Cache,
    name: str,
    *,
    tile: int,
    overlap: int,
    device: torch.device,
    amp: bool = True,
    max_candidates: int = 4000,
    batch_size: int = 1,
    minimum_proposal_score: float = 1e-6,
) -> np.ndarray:
    """Evaluate overlapping tiles in batches, then keep non-overlapping cores."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    idx = cache.name_to_index[name]
    size = cache.size
    origins = tile_origins(size, tile, overlap)
    locations = []

    for yi, y in enumerate(origins):
        ymin, ymax = core_bounds(origins, tile, size, yi)
        for xi, x in enumerate(origins):
            xmin, xmax = core_bounds(origins, tile, size, xi)
            locations.append((x, y, xmin, xmax, ymin, ymax))

    collected = []
    model.eval()

    for left in range(0, len(locations), batch_size):
        group = locations[left : left + batch_size]
        images = np.stack(
            [
                np.array(cache.images[idx, y : y + tile, x : x + tile], copy=True)
                for x, y, _, _, _, _ in group
            ]
        )
        tensor = torch.from_numpy(images).permute(0, 3, 1, 2)
        tensor = tensor.to(device=device, dtype=torch.float32).div_(255.0)

        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"
        ):
            outputs = model(tensor)

        for row, (x, y, xmin, xmax, ymin, ymax) in enumerate(group):
            tile_output = {key: value[row : row + 1] for key, value in outputs.items()}
            candidates = extract_stage1_candidates(
                tile_output,
                minimum_proposal_score=minimum_proposal_score,
                local_max_radius=2,
                max_candidates=max_candidates,
                quality_power=1.0,
                uncertainty_penalty=0.0,
            )

            if len(candidates):
                candidates["x"] += x
                candidates["y"] += y
                inside = (
                    (candidates["x"] >= xmin)
                    & (candidates["x"] < xmax)
                    & (candidates["y"] >= ymin)
                    & (candidates["y"] < ymax)
                )
                collected.append(candidates[inside])

    return np.concatenate(collected) if collected else np.empty(0, dtype=CANDIDATE_DTYPE)


def sweep_cached(
    cache_candidates: dict[str, np.ndarray],
    cache: Cache,
    names: list[str],
    *,
    thresholds: list[float],
    radii: list[float],
    min_recall: float = 0.93,
    max_candidates: int = 4000,
) -> dict:
    """Model forward only once/ROI; many operating points evaluated on cached candidates."""
    if not names or not thresholds or not radii:
        raise ValueError("names, thresholds and radii must be nonempty")
    if not np.isfinite(min_recall) or not 0 <= min_recall <= 1:
        raise ValueError("min_recall must be finite and in [0,1]")
    if not all(np.isfinite(th) and 0 <= th <= 1 for th in thresholds):
        raise ValueError("thresholds must be finite and in [0,1]")
    # Greedy NMS processes descending scores: removing lower-score candidates
    # cannot affect earlier decisions. One run at the minimum threshold is
    # therefore exact for every higher threshold, including the final limit.
    minimum = min(thresholds)
    measurements = {}
    for rad in radii:
        for name in names:
            candidates = cache_candidates[name]
            keep = select_stage1_candidate_indices(
                candidates,
                threshold=minimum,
                suppression_radius_pixels=rad,
                max_candidates=max_candidates,
            )
            kept_scores = candidates["proposal_score"][keep]
            xy = np.column_stack((candidates["x"][keep], candidates["y"][keep]))
            gt = cache.gt_for(name)
            gt_xy = np.column_stack((gt["x"], gt["y"]))
            by_count = {}
            for th in thresholds:
                count = int(np.count_nonzero(kept_scores >= th))
                if count not in by_count:
                    by_count[count] = match_counts(xy[:count], gt_xy)
                measurements[name, rad, th] = by_count[count]
    trial = []
    for th in thresholds:
        for rad in radii:
            total = {"tp": 0, "fp": 0, "fn": 0}
            roi_scores = []
            for name in names:
                c = measurements[name, rad, th]
                for key in total:
                    total[key] += c[key]
                roi_scores.append(scores(**c)["f1"])
            s = scores(**total)
            s.update(total)
            s["roi_macro_f1"] = float(np.mean(roi_scores))
            s["threshold"] = float(th)
            s["suppression_radius"] = float(rad)
            trial.append(s)
    selected = max(
        trial,
        key=lambda r: (
            int(r["recall"] >= min_recall),
            r["f1"] if r["recall"] >= min_recall else r["recall"],
            r["precision"] if r["recall"] >= min_recall else r["f1"],
            -r["threshold"],
            -r["suppression_radius"],
        ),
    )
    return {
        "best": selected,
        "sweep": trial,
        "matching": "class_agnostic_one_to_one_strict_lt_15px",
        "selection_note": "train-only threshold and NMS calibration; no validation GT used",
    }
