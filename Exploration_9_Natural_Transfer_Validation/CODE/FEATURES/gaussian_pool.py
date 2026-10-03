from __future__ import annotations
import torch
from CODE.COMMON.constants import TOKEN_GRID, APPEARANCE_DIM, GAUSSIAN_SIGMA
from CODE.COMMON.validation import require_finite


def gaussian_weights(
    cx: torch.Tensor,
    cy: torch.Tensor,
    sigma: float = GAUSSIAN_SIGMA,
    device=None,
    dtype=torch.float32,
):
    cx = torch.as_tensor(cx, device=device, dtype=dtype).reshape(-1, 1, 1)
    cy = torch.as_tensor(cy, device=device, dtype=dtype).reshape(-1, 1, 1)
    ys = torch.arange(TOKEN_GRID, device=device, dtype=dtype).reshape(1, TOKEN_GRID, 1)
    xs = torch.arange(TOKEN_GRID, device=device, dtype=dtype).reshape(1, 1, TOKEN_GRID)
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    w = torch.exp(-d2 / (2.0 * float(sigma) ** 2))
    w = w / w.sum(dim=(1, 2), keepdim=True)
    require_finite(w, 'gaussian_weights')
    return w


def gaussian_pool(
    spatial_tokens: torch.Tensor, cx, cy, sigma: float = GAUSSIAN_SIGMA
) -> torch.Tensor:
    if spatial_tokens.ndim != 4 or tuple(spatial_tokens.shape[1:]) != (
        TOKEN_GRID,
        TOKEN_GRID,
        APPEARANCE_DIM,
    ):
        raise ValueError(f"Bad spatial-token shape {tuple(spatial_tokens.shape)}")
    w = gaussian_weights(cx, cy, sigma, device=spatial_tokens.device, dtype=spatial_tokens.dtype)
    h = (spatial_tokens * w.unsqueeze(-1)).sum(dim=(1, 2))
    require_finite(h, 'gaussian_pool_output')
    return h
