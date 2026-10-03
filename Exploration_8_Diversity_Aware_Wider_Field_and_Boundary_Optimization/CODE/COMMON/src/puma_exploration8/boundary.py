from __future__ import annotations
import numpy as np


def prior_offset(source_prior, target_prior):
    s = np.asarray(source_prior, float)
    t = np.asarray(target_prior, float)
    if s.shape != (10,) or t.shape != (10,) or (s <= 0).any() or (t <= 0).any():
        raise ValueError('priors must be positive 10-vectors')
    s = s / s.sum()
    t = t / t.sum()
    return np.log(t) - np.log(s)


def apply_h1(logits, d, lam):
    return np.asarray(logits, float) + float(lam) * np.asarray(d, float)[None, :]


def temperature(logits, T):
    if T <= 0:
        raise ValueError('temperature >0')
    return np.asarray(logits, float) / float(T)
