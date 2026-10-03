from __future__ import annotations
import numpy as np
import torch
from .exceptions import NonFiniteTrainingError


def require_shape(t, shape, name: str) -> None:
    actual = tuple(t.shape)
    if len(actual) != len(shape) or any(e is not None and a != e for a, e in zip(actual, shape)):
        raise ValueError(f"{name} shape {actual}, expected {shape}")


def require_finite(x, name: str) -> None:
    if isinstance(x, torch.Tensor):
        ok = torch.isfinite(x).all().item()
    else:
        ok = np.isfinite(np.asarray(x)).all()
    if not ok:
        raise NonFiniteTrainingError(f"Nonfinite values in {name}")
