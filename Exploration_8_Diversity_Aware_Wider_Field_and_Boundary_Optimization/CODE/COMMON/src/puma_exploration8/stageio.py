from __future__ import annotations
from pathlib import Path
from .cache_integrity import aligned_cache
from .manifest import read_manifest, strongest_group_column
from .experiments import Arrays
from .config import sha256_file


def manifest_sha256(path: str | Path) -> str:
    return sha256_file(path)


def cache_expected(manifest_hash: str, name: str) -> dict:
    base = {'manifest_sha256': manifest_hash, 'code_contract': 'exploration8-v2'}
    contracts = {
        'local_gaussian96': {
            'feature_dim': 1536,
            'fov': 96,
            'pool': 'coordinate_aware_gaussian_sigma1.5',
        },
        'gaussian192': {
            'feature_dim': 1536,
            'fov': 192,
            'pool': 'coordinate_aware_gaussian_sigma1.5',
        },
        'annular192': {
            'feature_dim': 1536,
            'fov': 192,
            'pool': 'coordinate_aware_center_excluded_fov96_footprint',
        },
        'scale_only192': {'feature_dim': 1536, 'source_pixels': 'FOV96_only'},
        'tiera16': {'feature_dim': 16, 'feature_order': 'canonical_16'},
    }
    if name not in contracts:
        raise ValueError(f'unknown cache contract: {name}')
    return base | contracts[name]


def load_role_arrays(manifest_path, cache_dir, role='TRAIN', wider_name=None, strict_hash=False):
    df = read_manifest(manifest_path, require_images=False)
    sub = df[df.role == role].reset_index(drop=True)
    if sub.empty:
        raise ValueError(f'{role} role empty')
    uids = sub.uid.astype(str).tolist()
    mh = manifest_sha256(manifest_path)
    cd = Path(cache_dir)
    local = aligned_cache(
        cd / 'local_gaussian96', uids, cache_expected(mh, 'local_gaussian96'), strict_hash
    )
    bio = aligned_cache(cd / 'tiera16', uids, cache_expected(mh, 'tiera16'), strict_hash)
    wide = (
        None
        if wider_name is None
        else aligned_cache(cd / wider_name, uids, cache_expected(mh, wider_name), strict_hash)
    )
    group_col = strongest_group_column(df)
    arrays = Arrays(
        uids, sub.label.to_numpy(int), sub[group_col].astype(str).to_numpy(), local, bio, wide
    )
    return df, sub, group_col, arrays
