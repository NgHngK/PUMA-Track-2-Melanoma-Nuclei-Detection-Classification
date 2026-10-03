from __future__ import annotations
from pathlib import Path
import hashlib, json, os
import numpy as np, pandas as pd
from .io import atomic_json_dump, sha256_file

IDENTITY_FIELDS = (
    'uid',
    'roi',
    'group',
    'patient',
    'case',
    'slide',
    'image',
    'x',
    'y',
    'label',
    'split',
)
# CSV staging can perturb binary float coordinates at ~1e-12 while preserving the same nucleus.
# Nine decimal places is far below any meaningful pixel-scale difference and keeps cache identity stable.
IDENTITY_COORD_DECIMALS = 9


def row_identity_sha256(df: pd.DataFrame) -> str:
    cols = [c for c in IDENTITY_FIELDS if c in df.columns]
    rows = []
    use_source_image = 'source_image' in df.columns
    for row in df.itertuples(index=False):
        raw = row._asdict()
        item = {}
        for c in cols:
            v = raw.get('source_image') if c == 'image' and use_source_image else raw.get(c)
            if pd.isna(v):
                v = None
            elif c in {'x', 'y'}:
                v = round(float(v), IDENTITY_COORD_DECIMALS)
            elif c == 'label':
                v = int(v)
            elif isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = float(v)
            else:
                v = str(v)
            item[c] = v
        rows.append(item)
    blob = json.dumps(
        rows, sort_keys=True, separators=(',', ':'), ensure_ascii=False, default=str
    ).encode()
    return hashlib.sha256(blob).hexdigest()


def sidecar_path(array_path):
    return Path(array_path).with_suffix('.json')


def write_cache_sidecar(array_path, manifest_path, df, kind, **extra):
    p = Path(array_path)
    st = p.stat()
    meta = {
        'kind': str(kind),
        'array_file': p.name,
        'array_sha256': sha256_file(p),
        'array_size_bytes': st.st_size,
        'array_mtime_ns': st.st_mtime_ns,
        'manifest_file': Path(manifest_path).name,
        'manifest_sha256': sha256_file(manifest_path),
        'row_identity_sha256': row_identity_sha256(df),
        'n': int(len(df)),
    }
    meta.update(extra)
    atomic_json_dump(sidecar_path(p), meta)
    return meta


def verify_cache(
    array_path, df, kind=None, expected_dim=None, expected_meta=None, strict_hash=None
):
    p = Path(array_path)
    mp = sidecar_path(p)
    if not p.is_file():
        raise FileNotFoundError(p)
    if not mp.is_file():
        raise FileNotFoundError(f'cache metadata missing: {mp}')
    meta = json.loads(mp.read_text(encoding='utf-8'))
    if kind is not None and meta.get('kind') != kind:
        raise ValueError(f'cache kind mismatch: expected {kind}, got {meta.get("kind")}')
    if meta.get('n') != len(df):
        raise ValueError(f'cache row count mismatch: metadata={meta.get("n")} manifest={len(df)}')
    if meta.get('row_identity_sha256') != row_identity_sha256(df):
        raise ValueError('cache/manifest row identity mismatch')
    st = p.stat()
    strict = (
        (os.environ.get('PUMA_P7_STRICT_CACHE_HASH', '0') == '1')
        if strict_hash is None
        else bool(strict_hash)
    )
    unchanged = (
        meta.get('array_size_bytes') == st.st_size and meta.get('array_mtime_ns') == st.st_mtime_ns
    )
    if strict or not unchanged:
        if meta.get('array_sha256') and meta.get('array_sha256') != sha256_file(p):
            raise ValueError('cache file SHA256 mismatch')
    for key, value in (expected_meta or {}).items():
        got = meta.get(key)
        ok = (
            (abs(float(got) - value) <= 1e-9)
            if isinstance(value, float) and got is not None
            else (got == value)
        )
        if not ok:
            raise ValueError(f'cache metadata mismatch for {key}: expected {value!r}, got {got!r}')
    arr = np.load(p, mmap_mode='r')
    if len(arr) != len(df):
        raise ValueError(f'cache length mismatch: array={len(arr)} manifest={len(df)}')
    if expected_dim is not None and (arr.ndim != 2 or arr.shape[1] != expected_dim):
        raise ValueError(f'cache shape {arr.shape}; expected Nx{expected_dim}')
    return meta
