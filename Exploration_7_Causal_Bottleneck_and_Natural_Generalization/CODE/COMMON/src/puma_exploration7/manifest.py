from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from .constants import NUM_CLASSES, CLASSES

REQUIRED = ('uid', 'roi', 'label')


def read_manifest(path: str | Path, require_images: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        raise ValueError(f'manifest missing {miss}')
    if df['uid'].astype(str).duplicated().any():
        raise ValueError('uid must be unique')
    df = df.copy()
    df['uid'] = df['uid'].astype(str)
    df['roi'] = df['roi'].astype(str)
    df['label'] = df['label'].astype(int)
    bad = ~df['label'].isin(list(range(NUM_CLASSES)) + [-1])
    if bad.any():
        raise ValueError('labels must be -1 or 0..9')
    if 'group' not in df.columns:
        df['group'] = df['roi']
    for c in ['group', 'patient', 'case', 'slide', 'stain_batch', 'split']:
        if c in df.columns:
            df[c] = df[c].fillna('').astype(str)
    if require_images:
        if 'image' not in df.columns:
            raise ValueError('image column required')
        missing = [p for p in df['image'].astype(str).unique() if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(f'{len(missing)} image files missing; first={missing[0]}')
    return df


def strongest_group_column(df: pd.DataFrame, requested: str = 'auto') -> str:
    if requested != 'auto':
        if requested not in df.columns:
            raise ValueError(f'group column {requested} unavailable')
        return requested
    for c in ('patient', 'case', 'slide', 'group', 'roi'):
        if c in df.columns and df[c].astype(str).str.len().gt(0).all() and df[c].nunique() > 1:
            return c
    return 'roi'


def assert_group_disjoint(df: pd.DataFrame, split_col='split', group_col='auto') -> None:
    if split_col not in df.columns:
        return
    g = strongest_group_column(df, group_col)
    z = df[[g, split_col]].drop_duplicates()
    counts = z.groupby(g)[split_col].nunique()
    if (counts > 1).any():
        raise ValueError(f'group leakage across {split_col}: {counts[counts>1].index[:5].tolist()}')


def join_labels(pred: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    # Read labels and group metadata from the manifest, not the prediction file.
    drop = [
        c
        for c in ['label', 'label_gt', 'group', 'patient', 'case', 'slide', 'stain_batch']
        if c in pred.columns
    ]
    pred = pred.drop(columns=drop, errors='ignore').copy()
    cols = ['uid', 'label', 'roi'] + [
        c
        for c in [
            'split',
            'group',
            'patient',
            'case',
            'slide',
            'stain_batch',
            'image',
            'x',
            'y',
            'gt_x',
            'gt_y',
            'eval_x',
            'eval_y',
        ]
        if c in manifest.columns
    ]
    m = manifest[cols].drop_duplicates('uid')
    out = pred.merge(m, on='uid', how='inner', suffixes=('', '_gt'), validate='one_to_one')
    if len(out) != len(pred):
        raise ValueError(
            f'prediction/manifest uid mismatch: predictions={len(pred)} joined={len(out)}'
        )
    # Use prediction coordinates for proposal centres when available; keep manifest coordinates as *_gt.
    return out


def assert_external_group_disjoint(
    train: pd.DataFrame, test: pd.DataFrame, requested='auto'
) -> str:
    g = strongest_group_column(pd.concat([train, test], ignore_index=True), requested)
    overlap = set(train[g].astype(str)) & set(test[g].astype(str))
    if overlap:
        raise ValueError(f'train/test group leakage in {g}; examples={sorted(overlap)[:5]}')
    return g


def class_group_support(df: pd.DataFrame, group_col='auto') -> pd.DataFrame:
    g = strongest_group_column(df, group_col)
    rows = []
    for c in range(NUM_CLASSES):
        x = df[df.label == c]
        counts = x.groupby(g).size().to_numpy(float)
        eff = (
            float(counts.sum() ** 2 / (counts @ counts))
            if len(counts) and (counts @ counts) > 0
            else 0.0
        )
        rows.append(
            {
                'class_id': c,
                'class_name': CLASSES[c],
                'n': len(x),
                'groups': x[g].nunique(),
                'effective_groups_kish': eff,
                'group_col': g,
            }
        )
    return pd.DataFrame(rows)


EXPLORATION7_CONFIRMATORY_COLUMNS = ("uid", "roi", "group", "image", "x", "y", "label", "split")


def assert_confirmatory_manifest(
    df: pd.DataFrame, role: str, *, require_images: bool = False
) -> str:
    """Validate the stricter Exploration-7 confirmatory data contract.

    ``read_manifest`` intentionally remains backward-compatible with historical
    Exploration-6 artifacts.  Confirmatory Exploration-7 scripts call this function to fail
    closed on missing coordinates/grouping/split roles.
    """
    role = str(role)
    allowed = {"train", "calibration", "test"}
    if role not in allowed:
        raise ValueError(f"unknown confirmatory role: {role}")
    missing = [c for c in EXPLORATION7_CONFIRMATORY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Exploration-7 {role} manifest missing required columns: {missing}")
    if (df["label"].to_numpy(int) < 0).any():
        raise ValueError(f"Exploration-7 {role} manifest cannot contain unlabeled rows")
    if (
        not np.isfinite(pd.to_numeric(df["x"], errors="coerce")).all()
        or not np.isfinite(pd.to_numeric(df["y"], errors="coerce")).all()
    ):
        raise ValueError(f"Exploration-7 {role} manifest contains non-finite x/y")
    observed = set(df["split"].astype(str))
    if observed != {role}:
        raise ValueError(
            f"Exploration-7 {role} manifest must contain split={role!r} only, got {sorted(observed)}"
        )
    if df["group"].astype(str).str.len().eq(0).any():
        raise ValueError(f"Exploration-7 {role} manifest has empty group identifiers")
    if require_images:
        missing_images = [p for p in df["image"].astype(str).unique() if not Path(p).is_file()]
        if missing_images:
            raise FileNotFoundError(
                f"Exploration-7 {role} images missing; first={missing_images[0]}"
            )
    return strongest_group_column(df)
