from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def focal_heatmap_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target[:, None].float()
    probability = logits.sigmoid().clamp(1.0e-5, 1.0 - 1.0e-5)
    positive = target.eq(1.0).float()
    negative = target.lt(1.0).float()
    negative_weight = (1.0 - target).pow(4)
    positive_loss = -(probability.log()) * (1.0 - probability).pow(2) * positive
    negative_loss = -((1.0 - probability).log()) * probability.pow(2) * negative_weight * negative
    number_positive = positive.sum().clamp_min(1.0)
    return (positive_loss.sum() + negative_loss.sum()) / number_positive


def hard_negative_heatmap_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    topk_per_image: int,
    negative_target_ceiling: float = 0.05,
) -> torch.Tensor:
    """Online hard-negative mining for the detector confidence map.

    The standard focal heatmap loss sees the entire dense background.  This
    auxiliary term focuses additional gradient on the highest-scoring cells
    that are safely outside the Gaussian positive neighbourhood.  Those cells
    are the ones most likely to become false proposal peaks at validation time.
    """
    if topk_per_image < 1:
        raise ValueError("topk_per_image must be >= 1")
    dense_target = target[:, None].float()
    probability = logits.sigmoid()
    negative_mask = dense_target <= float(negative_target_ceiling)
    # BCE(target=0) = softplus(logit). Focal modulation prioritizes confident FP.
    per_cell = F.softplus(logits) * probability.pow(2.0)
    terms: list[torch.Tensor] = []
    for batch_index in range(len(logits)):
        values = per_cell[batch_index][negative_mask[batch_index]]
        if values.numel() == 0:
            continue
        k = min(int(topk_per_image), int(values.numel()))
        terms.append(torch.topk(values, k=k, sorted=False).values.mean())
    if not terms:
        return logits.new_zeros(())
    return torch.stack(terms).mean()


def dense_quality_loss(
    logits: torch.Tensor,
    sparse_points: list,
    *,
    sigma_grid_units: float,
) -> torch.Tensor:
    """Dense proposal-quality supervision with meaningful negatives.

    Positive cells receive a localization-quality target based on how far the
    nucleus lies from the assigned output cell; every other grid location is a
    supervised negative. Focal modulation prevents the dense background from
    overwhelming the sparse positives.
    """
    target = torch.zeros_like(logits[:, 0], dtype=torch.float32)
    for batch_index, points in enumerate(sparse_points):
        if len(points) == 0:
            continue
        yy = torch.as_tensor(points["grid_y"].astype("int64"), device=logits.device)
        xx = torch.as_tensor(points["grid_x"].astype("int64"), device=logits.device)
        offset = torch.as_tensor(
            np.column_stack((points["offset_x"], points["offset_y"])),
            device=logits.device,
            dtype=torch.float32,
        )
        distance2 = torch.square(offset).sum(dim=1)
        quality = torch.exp(-distance2 / (2.0 * float(sigma_grid_units) ** 2)).clamp(0.05, 1.0)
        target[batch_index, yy, xx] = quality
    probability = logits[:, 0].sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits[:, 0], target, reduction="none")
    focal = torch.abs(target - probability).pow(2.0)
    positive_count = (target > 0).sum().clamp_min(1)
    return (bce * focal).sum() / positive_count


def stage1_loss(
    outputs: dict[str, torch.Tensor],
    heatmap_target: torch.Tensor,
    sparse_points: list,
    *,
    weights: dict[str, float],
    quality_target_sigma_grid_units: float = 0.75,
    hard_negative_topk_per_image: int = 256,
) -> tuple[torch.Tensor, dict[str, float]]:
    heatmap_loss = focal_heatmap_loss(outputs["heatmap_logits"], heatmap_target)
    if float(weights.get("hard_negative", 0.0)) > 0.0:
        hard_negative_loss = hard_negative_heatmap_loss(
            outputs["heatmap_logits"], heatmap_target,
            topk_per_image=hard_negative_topk_per_image,
        )
    else:
        hard_negative_loss = heatmap_loss.new_zeros(())

    offset_terms: list[torch.Tensor] = []
    uncertainty_terms: list[torch.Tensor] = []
    for batch_index, points in enumerate(sparse_points):
        if len(points) == 0:
            continue
        yy = torch.as_tensor(points["grid_y"].astype("int64"), device=outputs["offset"].device)
        xx = torch.as_tensor(points["grid_x"].astype("int64"), device=outputs["offset"].device)
        target_offset = torch.as_tensor(
            np.column_stack((points["offset_x"], points["offset_y"])),
            device=outputs["offset"].device,
            dtype=torch.float32,
        )
        predicted_offset = outputs["offset"][batch_index, :, yy, xx].T
        log_variance = outputs["log_variance"][batch_index, 0, yy, xx]
        base_offset = F.smooth_l1_loss(predicted_offset, target_offset, reduction="none").sum(dim=1)
        offset_terms.append(base_offset.mean())
        uncertainty_terms.append((torch.exp(-log_variance) * base_offset.detach() + 0.5 * log_variance).mean())
    zero = heatmap_loss.new_zeros(())
    offset_loss = torch.stack(offset_terms).mean() if offset_terms else zero
    quality_loss = dense_quality_loss(
        outputs["quality_logits"], sparse_points, sigma_grid_units=quality_target_sigma_grid_units
    )
    uncertainty_loss = torch.stack(uncertainty_terms).mean() if uncertainty_terms else zero
    total = (
        weights["heatmap"] * heatmap_loss
        + weights["offset"] * offset_loss
        + weights["quality"] * quality_loss
        + weights["uncertainty"] * uncertainty_loss
        + float(weights.get("hard_negative", 0.0)) * hard_negative_loss
    )
    # Move all logging scalars to the host in a single transfer.  Calling
    # float(cuda_tensor) for every term forces one CUDA synchronization per
    # scalar and can noticeably stall the training loop.  This changes only
    # metric collection; the loss tensor and gradient graph above are untouched.
    metric_values = torch.stack((
        heatmap_loss.detach(),
        offset_loss.detach(),
        quality_loss.detach(),
        uncertainty_loss.detach(),
        hard_negative_loss.detach(),
        total.detach(),
    )).float().cpu().tolist()
    return total, dict(zip(
        ("heatmap", "offset", "quality", "uncertainty", "hard_negative", "total"),
        (float(value) for value in metric_values),
        strict=True,
    ))

