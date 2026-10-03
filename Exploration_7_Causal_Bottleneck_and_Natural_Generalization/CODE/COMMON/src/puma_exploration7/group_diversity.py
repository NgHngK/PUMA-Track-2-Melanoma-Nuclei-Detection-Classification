from __future__ import annotations
import numpy as np, pandas as pd
from .constants import CLASSES


def _sample_covering_groups(
    x: pd.DataFrame, group_col: str, groups, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample exactly n rows while guaranteeing every requested group contributes.

    This matters scientifically: merely choosing a high-diversity candidate pool does
    not guarantee a random row sample actually contains every selected biological
    group.  The helper takes one row per group first, then fills the remaining quota
    without replacement from the same pool.
    """
    groups = list(groups)
    if n < len(groups):
        raise ValueError(f'row quota {n} is smaller than requested group count {len(groups)}')
    chosen = []
    for g in groups:
        ix = x.index[x[group_col].astype(str) == str(g)].to_numpy()
        if len(ix) == 0:
            raise ValueError(f'empty selected group {g!r}')
        chosen.append(int(rng.choice(ix)))
    pool = x[x[group_col].astype(str).isin([str(g) for g in groups])].index.to_numpy()
    remaining = np.setdiff1d(pool, np.asarray(chosen, dtype=int), assume_unique=False)
    need = n - len(chosen)
    if len(remaining) < need:
        raise ValueError('selected groups do not contain enough rows for quota')
    if need:
        chosen.extend(rng.choice(remaining, size=need, replace=False).astype(int).tolist())
    out = np.asarray(chosen, dtype=int)
    rng.shuffle(out)
    return out


def build_matched_diversity_indices(
    df, group_col, low_groups=2, high_groups=8, rows_per_class=40, seed=17, max_attempts=200
):
    """Create matched-N arms differing in realized biological-group diversity.

    Low-diversity groups are a subset of the corresponding high-diversity group pool.
    Both arms contain exactly ``rows_per_class`` rows and every selected group is
    guaranteed to contribute at least one row.  Construction is outcome-blind.
    """
    if not (1 <= low_groups < high_groups):
        raise ValueError('require 1 <= low_groups < high_groups')
    if rows_per_class < high_groups:
        raise ValueError('rows_per_class must be >= high_groups so every group can contribute')
    rng = np.random.default_rng(seed)
    arms = {'low': [], 'high': []}
    audit = []
    for c in sorted(df.label.unique()):
        if c < 0:
            continue
        x = df[df.label == c]
        groups = np.array(list(x.groupby(group_col).groups), dtype=object)
        if len(groups) < high_groups:
            audit.append(
                {
                    'class_id': int(c),
                    'status': 'skipped',
                    'available_groups': len(groups),
                    'reason': 'insufficient_groups',
                }
            )
            continue
        feasible = None
        for _ in range(max_attempts):
            high = rng.choice(groups, size=high_groups, replace=False)
            low = rng.choice(high, size=low_groups, replace=False)
            pool_low = x[x[group_col].isin(low)]
            pool_high = x[x[group_col].isin(high)]
            if len(pool_low) >= rows_per_class and len(pool_high) >= rows_per_class:
                feasible = (low, high)
                break
        if feasible is None:
            audit.append(
                {
                    'class_id': int(c),
                    'status': 'skipped',
                    'available_groups': len(groups),
                    'reason': 'quota_not_feasible',
                }
            )
            continue
        low, high = feasible
        low_ix = _sample_covering_groups(x, group_col, low, rows_per_class, rng)
        high_ix = _sample_covering_groups(x, group_col, high, rows_per_class, rng)
        arms['low'].extend(low_ix.tolist())
        arms['high'].extend(high_ix.tolist())
        audit.append(
            {
                'class_id': int(c),
                'class_name': CLASSES[int(c)],
                'status': 'included',
                'available_groups': len(groups),
                'rows_per_arm': rows_per_class,
                'low_groups': len(low),
                'high_groups': len(high),
                'realized_low_groups': int(df.loc[low_ix, group_col].nunique()),
                'realized_high_groups': int(df.loc[high_ix, group_col].nunique()),
            }
        )
    if not arms['low'] or not arms['high']:
        raise ValueError('no classes eligible for diversity experiment')
    return {k: np.array(v, dtype=int) for k, v in arms.items()}, pd.DataFrame(audit)


def build_matched_density_indices(
    df,
    group_col,
    groups_per_class=8,
    small_rows_per_class=20,
    large_rows_per_class=40,
    seed=17,
    max_attempts=200,
):
    """Hold realized source groups fixed while increasing nuclei from those groups."""
    if small_rows_per_class >= large_rows_per_class:
        raise ValueError('small quota must be < large quota')
    if small_rows_per_class < groups_per_class:
        raise ValueError('small quota must be >= groups_per_class so all fixed groups contribute')
    rng = np.random.default_rng(seed)
    arms = {'small': [], 'large': []}
    audit = []
    for c in sorted(df.label.unique()):
        if c < 0:
            continue
        x = df[df.label == c]
        groups = np.array(list(x.groupby(group_col).groups), dtype=object)
        if len(groups) < groups_per_class:
            audit.append({'class_id': int(c), 'status': 'skipped', 'reason': 'insufficient_groups'})
            continue
        selected = None
        for _ in range(max_attempts):
            gs = rng.choice(groups, size=groups_per_class, replace=False)
            pool = x[x[group_col].isin(gs)]
            if len(pool) >= large_rows_per_class:
                selected = gs
                break
        if selected is None:
            audit.append({'class_id': int(c), 'status': 'skipped', 'reason': 'quota_not_feasible'})
            continue
        gs = selected
        large = _sample_covering_groups(x, group_col, gs, large_rows_per_class, rng)
        # Keep the small arm inside the large arm while retaining every group.
        large_df = df.loc[large]
        small = _sample_covering_groups(large_df, group_col, gs, small_rows_per_class, rng)
        arms['small'].extend(small.tolist())
        arms['large'].extend(large.tolist())
        audit.append(
            {
                'class_id': int(c),
                'status': 'included',
                'groups': groups_per_class,
                'small_rows': small_rows_per_class,
                'large_rows': large_rows_per_class,
                'realized_small_groups': int(df.loc[small, group_col].nunique()),
                'realized_large_groups': int(df.loc[large, group_col].nunique()),
            }
        )
    if not arms['small']:
        raise ValueError('no classes eligible for density experiment')
    return {k: np.asarray(v, int) for k, v in arms.items()}, pd.DataFrame(audit)
