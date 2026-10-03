from __future__ import annotations
import torch


def _centers(value, batch: int, device, dtype, name: str) -> torch.Tensor:
    t = torch.as_tensor(value, device=device, dtype=dtype)
    if t.ndim == 0:
        t = t.repeat(batch)
    if t.ndim != 1 or len(t) != batch:
        raise ValueError(f'{name} must be scalar or length-B')
    if not torch.isfinite(t).all():
        raise ValueError(f'{name} contains non-finite values')
    return t


def gaussian_weights(
    side: int, u, v, sigma: float = 1.5, device=None, dtype=torch.float32
) -> torch.Tensor:
    """Coordinate-aware P6/P8 Gaussian over spatial-token index coordinates.

    The historical convention uses integer token coordinates j/i directly and does not
    insert an unannounced +0.5/-0.5 token-center correction.
    Returns B x side x side normalized weights.
    """
    if side <= 0 or sigma <= 0:
        raise ValueError('invalid Gaussian geometry')
    uu = torch.as_tensor(u, device=device, dtype=dtype)
    vv = torch.as_tensor(v, device=device, dtype=dtype)
    if uu.ndim == 0:
        uu = uu[None]
    if vv.ndim == 0:
        vv = vv[None]
    if uu.shape != vv.shape or uu.ndim != 1:
        raise ValueError('u/v must be matching scalars or vectors')
    yy, xx = torch.meshgrid(
        torch.arange(side, device=device, dtype=dtype),
        torch.arange(side, device=device, dtype=dtype),
        indexing='ij',
    )
    w = torch.exp(
        -((xx[None] - uu[:, None, None]) ** 2 + (yy[None] - vv[:, None, None]) ** 2)
        / (2 * float(sigma) ** 2)
    )
    denom = w.sum((1, 2), keepdim=True)
    if (denom <= 0).any() or not torch.isfinite(denom).all():
        raise ValueError('invalid Gaussian weight mass')
    return w / denom


def gaussian_pool(grid: torch.Tensor, u, v, sigma: float = 1.5) -> torch.Tensor:
    if grid.ndim != 4 or grid.shape[1] != grid.shape[2]:
        raise ValueError('grid must BxSxSxD')
    b, s, _, _ = grid.shape
    uu = _centers(u, b, grid.device, grid.dtype, 'u')
    vv = _centers(v, b, grid.device, grid.dtype, 'v')
    w = gaussian_weights(s, uu, vv, sigma, grid.device, grid.dtype)
    return torch.einsum('bij,bijd->bd', w, grid)


def annular_pool(grid: torch.Tensor, u, v, local_fraction: float = 0.5) -> torch.Tensor:
    """Uniform center-excluded FOV192 representation around each continuous target.

    With local_fraction=.5, the excluded square is the geometric FOV96 footprint
    inside FOV192: half-width 4 token-index units on a 16x16 grid.
    """
    if grid.ndim != 4 or grid.shape[1] != grid.shape[2]:
        raise ValueError('grid must BxSxSxD')
    if not (0 < local_fraction < 1):
        raise ValueError('local_fraction must be in (0,1)')
    b, s, _, _ = grid.shape
    uu = _centers(u, b, grid.device, grid.dtype, 'u')
    vv = _centers(v, b, grid.device, grid.dtype, 'v')
    yy, xx = torch.meshgrid(
        torch.arange(s, device=grid.device, dtype=grid.dtype),
        torch.arange(s, device=grid.device, dtype=grid.dtype),
        indexing='ij',
    )
    half = float(s) * float(local_fraction) / 2.0
    central = ((xx[None] - uu[:, None, None]).abs() < half) & (
        (yy[None] - vv[:, None, None]).abs() < half
    )
    outer = (~central).to(grid.dtype)
    mass = outer.sum((1, 2), keepdim=True)
    if (mass <= 0).any():
        raise ValueError('annular mask has zero mass')
    w = outer / mass
    return torch.einsum('bij,bijd->bd', w, grid)
