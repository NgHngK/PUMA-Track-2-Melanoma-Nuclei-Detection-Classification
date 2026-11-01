from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from .constants import STAGE1_OUTPUT_STRIDE

PREDICTION_DTYPE = np.dtype(
    [
        ("x", "f4"),
        ("y", "f4"),
        ("heatmap_score", "f4"),
        ("quality", "f4"),
        ("uncertainty", "f4"),
        ("peak_sharpness", "f4"),
    ]
)
CANDIDATE_DTYPE = np.dtype(
    [
        ("x", "f4"),
        ("y", "f4"),
        ("proposal_score", "f4"),
        ("heatmap_score", "f4"),
        ("quality", "f4"),
        ("uncertainty", "f4"),
        ("peak_sharpness", "f4"),
    ]
)


def _greedy_spatial_suppression(
    points: np.ndarray, scores: np.ndarray, radius: float, limit: int
) -> np.ndarray:
    """Exact greedy NMS with a spatial hash (same ordering/radius as reference).

    The reference implementation rescans all kept points for every candidate.
    Quantizing coordinates into radius-sized cells means only the 3x3
    neighborhood can contain a suppression neighbor. Strict inequality (<r²)
    and stable descending score order are preserved.
    """
    if limit < 1:
        raise ValueError("limit must be >= 1")
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and nonnegative")
    if not len(points):
        return np.empty(0, dtype=np.int64)
    order = np.argsort(-scores, kind="stable")
    if radius == 0:
        return order[:limit].astype(np.int64, copy=False)
    keep: list[int] = []
    cell_size = float(radius)
    radius2 = cell_size * cell_size
    occupied: dict[tuple[int, int], list[int]] = {}
    for index in order:
        cx = int(np.floor(float(points[index, 0]) / cell_size))
        cy = int(np.floor(float(points[index, 1]) / cell_size))
        blocked = False
        for xx in (cx - 1, cx, cx + 1):
            for yy in (cy - 1, cy, cy + 1):
                neighbors = occupied.get((xx, yy), ())
                for prev in neighbors:
                    dx = float(points[prev, 0] - points[index, 0])
                    dy = float(points[prev, 1] - points[index, 1])
                    if dx * dx + dy * dy < radius2:
                        blocked = True
                        break
                if blocked:
                    break
            if blocked:
                break
        if blocked:
            continue
        keep.append(int(index))
        occupied.setdefault((cx, cy), []).append(int(index))
        if len(keep) >= limit:
            break
    return np.asarray(keep, dtype=np.int64)


def _proposal_score(
    heatmap_score: torch.Tensor,
    quality: torch.Tensor,
    uncertainty: torch.Tensor,
    *,
    quality_power: float,
    uncertainty_penalty: float,
) -> torch.Tensor:
    score = heatmap_score
    if float(quality_power) != 0.0:
        score = score * quality.clamp_min(1e-06).pow(float(quality_power))
    if float(uncertainty_penalty) != 0.0:
        score = score * torch.exp(-float(uncertainty_penalty) * uncertainty.clamp_min(0.0))
    return score.clamp(0.0, 1.0)


@torch.inference_mode()
def extract_stage1_candidates(
    outputs: dict[str, torch.Tensor],
    *,
    minimum_proposal_score: float,
    local_max_radius: int,
    max_candidates: int,
    quality_power: float = 1.0,
    uncertainty_penalty: float = 0.0,
) -> np.ndarray:
    """Extract one tile's complete local-max pool for decoder sweeps.

    ``max_candidates`` is the downstream prediction limit. Truncating before
    spatial suppression (or tile-core filtering) can lose valid final peaks.
    """
    if not 0.0 <= float(minimum_proposal_score) <= 1.0:
        raise ValueError("minimum_proposal_score must be in [0,1]")
    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")
    if (
        isinstance(local_max_radius, bool)
        or not np.isfinite(local_max_radius)
        or int(local_max_radius) != local_max_radius
        or local_max_radius < 0
    ):
        raise ValueError("local_max_radius must be a nonnegative integer")
    for name, value in (
        ("quality_power", quality_power),
        ("uncertainty_penalty", uncertainty_penalty),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    channels = {"heatmap_logits": 1, "offset": 2, "quality_logits": 1, "log_variance": 1}
    reference = outputs["heatmap_logits"]
    for name, count in channels.items():
        value = outputs[name]
        if (
            value.ndim != 4
            or value.shape[:2] != (1, count)
            or value.shape[-2:] != reference.shape[-2:]
        ):
            raise ValueError("Decoder expects one tile with aligned NCHW prediction maps")
    heatmap = reference.float().sigmoid()[0, 0]
    kernel = 2 * int(local_max_radius) + 1
    maxima = (
        heatmap
        >= F.max_pool2d(heatmap[None, None], kernel, stride=1, padding=local_max_radius)[0, 0]
    )
    mask = maxima & (heatmap >= float(minimum_proposal_score))
    yy, xx = torch.where(mask)
    if len(xx) == 0:
        return np.empty(0, dtype=CANDIDATE_DTYPE)
    heatmap_score = heatmap[yy, xx]
    offsets = outputs["offset"][0, :, yy, xx].T.float()
    quality = outputs["quality_logits"][0, 0, yy, xx].float().sigmoid()
    uncertainty = outputs["log_variance"][0, 0, yy, xx].float().exp().sqrt()
    peak_sharpness = heatmap_score - F.avg_pool2d(heatmap[None, None], 3, 1, 1)[0, 0, yy, xx]
    score = _proposal_score(
        heatmap_score,
        quality,
        uncertainty,
        quality_power=quality_power,
        uncertainty_penalty=uncertainty_penalty,
    )
    keep_score = score >= float(minimum_proposal_score)
    if not bool(keep_score.any()):
        return np.empty(0, dtype=CANDIDATE_DTYPE)
    yy, xx = (yy[keep_score], xx[keep_score])
    offsets = offsets[keep_score]
    heatmap_score = heatmap_score[keep_score]
    quality = quality[keep_score]
    uncertainty = uncertainty[keep_score]
    peak_sharpness = peak_sharpness[keep_score]
    score = score[keep_score]
    height, width = heatmap.shape
    x = ((xx.float() + offsets[:, 0]) * STAGE1_OUTPUT_STRIDE).clamp(
        0.0, float(width * STAGE1_OUTPUT_STRIDE - 1)
    )
    y = ((yy.float() + offsets[:, 1]) * STAGE1_OUTPUT_STRIDE).clamp(
        0.0, float(height * STAGE1_OUTPUT_STRIDE - 1)
    )
    output = np.empty(len(score), dtype=CANDIDATE_DTYPE)
    output["x"] = x.cpu().numpy().astype(np.float32, copy=False)
    output["y"] = y.cpu().numpy().astype(np.float32, copy=False)
    output["proposal_score"] = score.cpu().numpy().astype(np.float32, copy=False)
    output["heatmap_score"] = heatmap_score.cpu().numpy().astype(np.float32, copy=False)
    output["quality"] = quality.cpu().numpy().astype(np.float32, copy=False)
    output["uncertainty"] = uncertainty.cpu().numpy().astype(np.float32, copy=False)
    output["peak_sharpness"] = peak_sharpness.cpu().numpy().astype(np.float32, copy=False)
    return output


def select_stage1_candidate_indices(
    candidates: np.ndarray,
    *,
    threshold: float,
    suppression_radius_pixels: float,
    max_candidates: int,
) -> np.ndarray:
    """Select indices in stable descending proposal-score order."""
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0,1]")
    selected = np.flatnonzero(candidates["proposal_score"] >= float(threshold))
    xy = np.column_stack((candidates["x"][selected], candidates["y"][selected]))
    keep = _greedy_spatial_suppression(
        xy,
        candidates["proposal_score"][selected],
        float(suppression_radius_pixels),
        int(max_candidates),
    )
    return selected[keep]


def decode_stage1_candidates(
    candidates: np.ndarray,
    *,
    threshold: float,
    suppression_radius_pixels: float,
    max_candidates: int,
) -> np.ndarray:
    """Cheap CPU decode from a cached local-max candidate pool."""
    keep = select_stage1_candidate_indices(
        candidates,
        threshold=threshold,
        suppression_radius_pixels=suppression_radius_pixels,
        max_candidates=max_candidates,
    )
    selected = candidates
    output = np.empty(len(keep), dtype=PREDICTION_DTYPE)
    output["x"] = selected["x"][keep]
    output["y"] = selected["y"][keep]
    output["heatmap_score"] = selected["heatmap_score"][keep]
    output["quality"] = selected["quality"][keep]
    output["uncertainty"] = selected["uncertainty"][keep]
    output["peak_sharpness"] = selected["peak_sharpness"][keep]
    return output


@torch.inference_mode()
def decode_stage1_outputs_multi_threshold(
    outputs: dict[str, torch.Tensor],
    *,
    thresholds: tuple[float, ...] | list[float],
    local_max_radius: int,
    suppression_radius_pixels: float,
    max_candidates: int,
    quality_power: float = 1.0,
    uncertainty_penalty: float = 0.0,
) -> dict[float, np.ndarray]:
    """Decode one ROI at many fused-score thresholds after one model forward."""
    threshold_values = tuple(sorted({float(value) for value in thresholds}))
    if not threshold_values:
        raise ValueError("thresholds cannot be empty")
    if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in threshold_values):
        raise ValueError("thresholds must be in [0,1]")
    candidates = extract_stage1_candidates(
        outputs,
        minimum_proposal_score=threshold_values[0],
        local_max_radius=local_max_radius,
        max_candidates=max_candidates,
        quality_power=quality_power,
        uncertainty_penalty=uncertainty_penalty,
    )
    return {
        threshold: decode_stage1_candidates(
            candidates,
            threshold=threshold,
            suppression_radius_pixels=suppression_radius_pixels,
            max_candidates=max_candidates,
        )
        for threshold in threshold_values
    }


@torch.inference_mode()
def decode_stage1_outputs(
    outputs: dict[str, torch.Tensor],
    *,
    threshold: float,
    local_max_radius: int,
    suppression_radius_pixels: float,
    max_candidates: int,
    quality_power: float = 1.0,
    uncertainty_penalty: float = 0.0,
) -> np.ndarray:
    return decode_stage1_outputs_multi_threshold(
        outputs,
        thresholds=(float(threshold),),
        local_max_radius=local_max_radius,
        suppression_radius_pixels=suppression_radius_pixels,
        max_candidates=max_candidates,
        quality_power=quality_power,
        uncertainty_penalty=uncertainty_penalty,
    )[float(threshold)]
