from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from .manifest import strongest_group_column


def draw_roster(
    df: pd.DataFrame, groups_per_class: int, nuclei_per_class: int, seed: int
) -> pd.DataFrame:
    """Compatibility entry point using the current capacity-aware sampler."""
    if nuclei_per_class < groups_per_class:
        raise ValueError(
            'nuclei_per_class must be >= groups_per_class to represent every selected group'
        )
    group_col = strongest_group_column(df)
    train = df[df.role == 'TRAIN'].sort_values('uid').reset_index(drop=True).copy()
    train[group_col] = train[group_col].astype(str)
    rng = np.random.default_rng(seed)
    pieces = []
    for c in range(10):
        z = train[train.label == c]
        chosen_groups = feasible_groups(z, group_col, groups_per_class, nuclei_per_class, rng)
        part = sample_rows(z, group_col, chosen_groups, nuclei_per_class, rng)
        pieces.append(part)
    return pd.concat(pieces).sort_values('uid').reset_index(drop=True)


def roster_hash(df):
    return hashlib.sha256('\n'.join(df.uid.astype(str)).encode()).hexdigest()


def plan_classes(train, fold_vector, cfg):
    """Freeze common quotas using TRAIN annotation support across all training portions."""
    g = strongest_group_column(train)
    plan = []
    for c in range(10):
        counts = [
            train[(train.label == c) & (fold_vector != k)]
            .groupby(g)
            .size()
            .sort_values(ascending=False)
            for k in sorted(set(fold_vector))
        ]
        minimum_groups = min(len(x) for x in counts)
        if minimum_groups == 0:
            raise ValueError(f'class {c} absent from a training portion')
        low_g = int(cfg['low_groups_per_class'])
        high_g = min(int(cfg['high_groups_per_class']), minimum_groups)
        n = min(int(cfg['nuclei_per_class']), min(int(x.head(low_g).sum()) for x in counts))
        enabled = high_g > low_g and n >= max(high_g, int(cfg['minimum_contrast_nuclei']))
        if not enabled:
            n = min(int(cfg['nuclei_per_class']), min(int(x.sum()) for x in counts))
            high_g = min(minimum_groups, n)
            low_g = high_g
            n = min(n, min(int(x.head(high_g).sum()) for x in counts))
        low_n = max(high_g, min(int(cfg['low_density_nuclei_per_class']), n // 2))
        # Classes excluded from diversity also remain fixed in density.
        density = enabled and low_n < n
        plan.append(
            dict(
                label=c,
                class_name=str(train.loc[train.label == c, 'class_name'].iloc[0]),
                diversity_enabled=enabled,
                density_enabled=density,
                low_groups=low_g,
                high_groups=high_g,
                matched_nuclei=n,
                low_nuclei=low_n if density else n,
                minimum_available_groups=minimum_groups,
                minimum_total_nuclei=min(int(x.sum()) for x in counts),
                reason=(
                    'feasible contrast'
                    if enabled
                    else 'fixed: insufficient support for minimum contrast'
                ),
            )
        )
    if not any(x['diversity_enabled'] for x in plan) or not any(x['density_enabled'] for x in plan):
        raise ValueError('no estimable diversity or density contrast')
    return plan


def feasible_groups(z, g, k, n, rng):
    """Random sequential selection, restricted to choices allowing completion.

    This is capacity-conditioned sampling, not uniform sampling over group sets.
    """
    counts = z.groupby(g).size().to_dict()
    chosen = []
    capacity = 0
    for step in range(k):
        slots = k - step - 1
        eligible = []
        for candidate in sorted(counts):
            others = sorted((v for key, v in counts.items() if key != candidate), reverse=True)
            if len(others) >= slots and capacity + counts[candidate] + sum(others[:slots]) >= n:
                eligible.append(candidate)
        if not eligible:
            raise ValueError('no capacity-feasible group completion')
        pick = eligible[int(rng.integers(len(eligible)))]
        chosen.append(pick)
        capacity += counts.pop(pick)
    return chosen


def sample_rows(z, g, groups, n, rng):
    picks = [int(rng.choice(z.index[z[g] == group])) for group in groups]
    pool = z.index[z[g].isin(groups)].difference(picks).to_numpy()
    if n < len(picks) or n > len(pool) + len(picks):
        raise ValueError('invalid unique-nucleus quota')
    picks.extend(map(int, rng.choice(pool, n - len(picks), replace=False)))
    return z.loc[picks].copy()


def paired_rosters(train, plan, seed):
    train = train.sort_values('uid').reset_index(drop=True).copy()
    g = strongest_group_column(train)
    train[g] = train[g].astype(str)
    arms = {name: [] for name in ('div_low', 'div_high', 'density_low', 'density_high')}
    for spec in plan:
        c = spec['label']
        z = train[train.label == c]
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), c]))
        n = spec['matched_nuclei']
        low_groups = feasible_groups(z, g, spec['low_groups'], n, rng)
        low = sample_rows(z, g, low_groups, n, rng)
        if spec['diversity_enabled']:
            remaining = sorted(set(z[g]) - set(low_groups))
            extra = rng.choice(
                remaining, spec['high_groups'] - len(low_groups), replace=False
            ).tolist()
            high_groups = low_groups + extra
            high = sample_rows(z, g, high_groups, n, rng)
        else:
            high_groups = low_groups
            high = low.copy()
        density = (
            sample_rows(high, g, high_groups, spec['low_nuclei'], rng)
            if spec['density_enabled']
            else high.copy()
        )
        for name, part in [
            ('div_low', low),
            ('div_high', high),
            ('density_low', density),
            ('density_high', high),
        ]:
            arms[name].append(part)
    return {
        name: pd.concat(parts).sort_values('uid').reset_index(drop=True)
        for name, parts in arms.items()
    }


def validate_pairs(arms, train, plan):
    g = strongest_group_column(train)
    allowed = set(train.uid)
    for name, roster in arms.items():
        if roster.uid.duplicated().any() or not set(roster.uid).issubset(allowed):
            raise ValueError(f'{name}: duplicate or non-training UID')
        for spec in plan:
            part = roster[roster.label == spec['label']]
            n = spec['low_nuclei'] if name == 'density_low' else spec['matched_nuclei']
            k = spec['low_groups'] if name == 'div_low' else spec['high_groups']
            if len(part) != n or part[g].nunique() != k:
                raise ValueError(f'{name}: quota/group mismatch for class {spec["label"]}')
    if not set(arms['density_low'].uid).issubset(set(arms['density_high'].uid)):
        raise ValueError('density samples are not nested')
    for spec in plan:
        if not spec['diversity_enabled']:
            sets = [set(x.loc[x.label == spec['label'], 'uid']) for x in arms.values()]
            if any(x != sets[0] for x in sets):
                raise ValueError('fixed class changed between arms')
