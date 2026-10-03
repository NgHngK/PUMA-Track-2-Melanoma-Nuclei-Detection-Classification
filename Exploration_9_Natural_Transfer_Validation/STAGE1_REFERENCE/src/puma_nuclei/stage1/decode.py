from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..constants import ROI_SIZE, STAGE1_OUTPUT_STRIDE

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

# Internal validation cache. proposal_score is intentionally not part of the
# Stage-2 public proposal dtype; Stage 2 keeps the raw detector evidence fields.
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


def _greedy_spatial_suppression(points: np.ndarray, scores: np.ndarray, radius: float, limit: int) -> np.ndarray:
    if not len(points):
        return np.empty(0, dtype=np.int64)
    order = np.argsort(-scores, kind="stable")
    if radius <= 0:
        return order[:limit].astype(np.int64, copy=False)
    keep: list[int] = []
    radius2 = float(radius) ** 2
    for index in order:
        if keep:
            delta = points[np.asarray(keep)] - points[index]
            if np.any(np.sum(delta * delta, axis=1) < radius2):
                continue
        keep.append(int(index))
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
        score = score * quality.clamp_min(1.0e-6).pow(float(quality_power))
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
    """Extract one ROI's local-max candidate pool once for decoder sweeps."""
    if not 0.0 <= float(minimum_proposal_score) <= 1.0:
        raise ValueError("minimum_proposal_score must be in [0,1]")
    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")

    heatmap = outputs["heatmap_logits"].float().sigmoid()[0, 0]
    kernel = 2 * int(local_max_radius) + 1
    maxima = heatmap >= F.max_pool2d(
        heatmap[None, None], kernel, stride=1, padding=local_max_radius
    )[0, 0]

    # proposal_score <= heatmap for the supported non-negative score terms, so
    # heatmap >= minimum score is a safe superset filter before reading heads.
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
        heatmap_score, quality, uncertainty,
        quality_power=quality_power,
        uncertainty_penalty=uncertainty_penalty,
    )

    keep_score = score >= float(minimum_proposal_score)
    if not bool(keep_score.any()):
        return np.empty(0, dtype=CANDIDATE_DTYPE)
    yy, xx = yy[keep_score], xx[keep_score]
    offsets = offsets[keep_score]
    heatmap_score = heatmap_score[keep_score]
    quality = quality[keep_score]
    uncertainty = uncertainty[keep_score]
    peak_sharpness = peak_sharpness[keep_score]
    score = score[keep_score]

    # Keep a generous pre-NMS pool, ranked by the same fused proposal score that
    # will be used by every downstream operating point.
    pool_limit = int(max_candidates) * 3
    if len(score) > pool_limit:
        top = torch.topk(score, k=pool_limit, sorted=False).indices
        yy, xx = yy[top], xx[top]
        offsets = offsets[top]
        heatmap_score = heatmap_score[top]
        quality = quality[top]
        uncertainty = uncertainty[top]
        peak_sharpness = peak_sharpness[top]
        score = score[top]

    x = ((xx.float() + offsets[:, 0]) * STAGE1_OUTPUT_STRIDE).clamp(0.0, float(ROI_SIZE - 1))
    y = ((yy.float() + offsets[:, 1]) * STAGE1_OUTPUT_STRIDE).clamp(0.0, float(ROI_SIZE - 1))

    output = np.empty(len(score), dtype=CANDIDATE_DTYPE)
    output["x"] = x.cpu().numpy().astype(np.float32, copy=False)
    output["y"] = y.cpu().numpy().astype(np.float32, copy=False)
    output["proposal_score"] = score.cpu().numpy().astype(np.float32, copy=False)
    output["heatmap_score"] = heatmap_score.cpu().numpy().astype(np.float32, copy=False)
    output["quality"] = quality.cpu().numpy().astype(np.float32, copy=False)
    output["uncertainty"] = uncertainty.cpu().numpy().astype(np.float32, copy=False)
    output["peak_sharpness"] = peak_sharpness.cpu().numpy().astype(np.float32, copy=False)
    return output


def decode_stage1_candidates(
    candidates: np.ndarray,
    *,
    threshold: float,
    suppression_radius_pixels: float,
    max_candidates: int,
) -> np.ndarray:
    """Cheap CPU decode from a cached local-max candidate pool."""
    if len(candidates) == 0:
        return np.empty(0, dtype=PREDICTION_DTYPE)
    selected = candidates[candidates["proposal_score"] >= float(threshold)]
    if len(selected) == 0:
        return np.empty(0, dtype=PREDICTION_DTYPE)
    xy = np.column_stack((selected["x"], selected["y"])).astype(np.float32, copy=False)
    keep = _greedy_spatial_suppression(
        xy,
        selected["proposal_score"].astype(np.float32, copy=False),
        float(suppression_radius_pixels),
        int(max_candidates),
    )
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
    if threshold_values[0] < 0.0 or threshold_values[-1] > 1.0:
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

