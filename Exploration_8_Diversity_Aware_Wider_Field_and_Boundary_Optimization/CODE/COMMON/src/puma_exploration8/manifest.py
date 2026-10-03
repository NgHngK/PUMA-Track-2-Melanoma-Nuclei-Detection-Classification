from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from .constants import CLASSES, ROLES

REQUIRED = ("uid", "role", "roi", "image", "x", "y", "label", "class_name", "coordinate_source")
GROUP_HIERARCHY = ("patient_id", "case_id", "slide_id", "roi")


def read_manifest(
    path: str | Path, require_images: bool = True, check_bounds: bool = False
) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        raise ValueError(f'manifest missing columns: {miss}')
    if df.empty or df.uid.astype(str).duplicated().any():
        raise ValueError('empty manifest or duplicate uid')
    if not set(df.role).issubset(set(ROLES)):
        raise ValueError(f'invalid roles: {sorted(set(df.role)-set(ROLES))}')
    image_sizes = {}
    for i, r in df.iterrows():
        lab = int(r.label)
        if lab not in range(10):
            raise ValueError(f'label outside 0..9 at {r.uid}')
        if str(r.class_name) != CLASSES[lab]:
            raise ValueError(f'class map mismatch at {r.uid}')
        if not np.isfinite([float(r.x), float(r.y)]).all():
            raise ValueError(f'nonfinite coordinates at {r.uid}')
        p = Path(str(r.image))
        p = (p if p.is_absolute() else path.parent / p).resolve()
        df.at[i, 'image'] = str(p)
        if require_images and not p.is_file():
            raise FileNotFoundError(p)
        if check_bounds and p.is_file():
            size = image_sizes.get(p)
            if size is None:
                with Image.open(p) as im:
                    size = im.size
                image_sizes[p] = size
            if not (0 <= float(r.x) < size[0] and 0 <= float(r.y) < size[1]):
                raise ValueError(f'coordinate outside image at {r.uid}')
    verify_role_isolation(df)
    return df


def strongest_group_column(df: pd.DataFrame) -> str:
    for c in GROUP_HIERARCHY:
        if c in df.columns and df[c].notna().all() and (df[c].astype(str).str.len() > 0).all():
            return c
    raise ValueError('no verified group column')


def verify_role_isolation(df: pd.DataFrame) -> str:
    col = strongest_group_column(df)
    cross = df.groupby(col).role.nunique()
    if (cross > 1).any():
        raise ValueError(
            f'group leakage across roles at {col}: {cross[cross>1].index.tolist()[:8]}'
        )
    # UID unique already guarantees row disjointness.
    return col


def manifest_summary(df: pd.DataFrame) -> dict:
    g = strongest_group_column(df)
    return {
        'rows': len(df),
        'group_column': g,
        'patient_level_verified': g == 'patient_id',
        'roles': {
            r: {'rows': int((df.role == r).sum()), 'groups': int(df.loc[df.role == r, g].nunique())}
            for r in ROLES
        },
    }
