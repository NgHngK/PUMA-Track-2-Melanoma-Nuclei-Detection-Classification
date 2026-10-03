from __future__ import annotations
import numpy as np
import torch
from CODE.COMMON.constants import NUM_CLASSES
from CODE.COMMON.seed import derive_seed


def inverse_class_weights(labels) -> torch.Tensor:
    labels = torch.as_tensor(labels, dtype=torch.long)
    counts = torch.bincount(labels, minlength=NUM_CLASSES).double()
    if (counts == 0).any():
        missing = torch.where(counts == 0)[0].tolist()
        raise ValueError(f"Training fold missing classes {missing}")
    return (1.0 / counts[labels]).double()


def epoch_sample_indices(
    labels, base_seed: int, fold: int, epoch: int, num_samples: int | None = None
) -> np.ndarray:
    labels = torch.as_tensor(labels, dtype=torch.long)
    weights = inverse_class_weights(labels)
    n = len(labels) if num_samples is None else int(num_samples)
    g = torch.Generator().manual_seed(derive_seed(base_seed, 'sampler', fold, epoch))
    idx = torch.multinomial(weights, n, replacement=True, generator=g)
    return idx.numpy()
