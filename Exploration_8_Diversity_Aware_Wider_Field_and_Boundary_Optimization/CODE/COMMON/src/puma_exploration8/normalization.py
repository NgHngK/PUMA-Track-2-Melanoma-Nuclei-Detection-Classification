from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from .constants import TIERA_COLUMNS


@dataclass(frozen=True)
class TierANormalizer:
    mean: list[float]
    std: list[float]
    fit_uids: list[str]
    fold_id: str
    feature_order: list[str]

    def transform(self, x):
        a = np.asarray(x, np.float32)
        m = np.asarray(self.mean, np.float32)
        s = np.asarray(self.std, np.float32)
        if a.shape[-1] != 16:
            raise ValueError('Tier-A must be 16D')
        return (a - m) / s

    def to_dict(self):
        return asdict(self)


def fit_tiera_normalizer(x, uids, fold_id):
    a = np.asarray(x, np.float32)
    u = list(map(str, uids))
    if a.ndim != 2 or a.shape[1] != 16 or len(a) != len(u):
        raise ValueError('normalizer input mismatch')
    m = a.mean(0)
    s = a.std(0)
    s = np.maximum(s, 1e-6)
    return TierANormalizer(m.tolist(), s.tolist(), u, str(fold_id), list(TIERA_COLUMNS))
