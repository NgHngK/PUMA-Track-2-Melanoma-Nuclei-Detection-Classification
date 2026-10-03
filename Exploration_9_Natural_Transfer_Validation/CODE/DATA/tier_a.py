from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from CODE.COMMON.constants import TIER_A_DIM, EPS_STD
from CODE.COMMON.exceptions import NormalizationLeakageError


@dataclass(frozen=True)
class TierANormalizer:
    mean: np.ndarray
    std: np.ndarray
    fit_uid_hash: str

    @classmethod
    def fit(cls, x: np.ndarray, fit_uids: list[str], uid_hash_fn) -> 'TierANormalizer':
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != TIER_A_DIM:
            raise ValueError(f"Tier-A must be [N,{TIER_A_DIM}]")
        if len(fit_uids) != len(x):
            raise ValueError('Tier-A/UID length mismatch')
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite Tier-A')
        mean = x.mean(0)
        std = x.std(0, ddof=0)
        std = np.maximum(std, EPS_STD)
        return cls(mean.astype(np.float32), std.astype(np.float32), uid_hash_fn(fit_uids))

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != TIER_A_DIM:
            raise ValueError('Bad Tier-A shape')
        z = (x - self.mean) / self.std
        if not np.isfinite(z).all():
            raise ValueError('Nonfinite normalized Tier-A')
        return z.astype(np.float32)


def verify_fit_scope(fit_uids: list[str], held_uids: list[str]) -> None:
    overlap = set(fit_uids) & set(held_uids)
    if overlap:
        raise NormalizationLeakageError(f"Tier-A fit/held UID overlap, e.g. {next(iter(overlap))}")
