from __future__ import annotations
from pathlib import Path
import hashlib, json
import numpy as np
from .config import sha256_file


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_cache(prefix: str | Path, features: np.ndarray, uids, meta: dict) -> None:
    p = Path(prefix)
    p.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(features)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError('cache features must be finite N x D')
    u = np.asarray(list(map(str, uids)), dtype=str)
    if len(u) != len(a) or len(set(u.tolist())) != len(u):
        raise ValueError('uid/cache length or uniqueness mismatch')
    data_path = p.with_suffix('.npy')
    uid_path = p.with_name(p.name + '_uids.npy')
    np.save(data_path, a)
    np.save(uid_path, u)
    ds, us = data_path.stat(), uid_path.stat()
    m = dict(meta)
    m.update(
        shape=list(a.shape),
        dtype=str(a.dtype),
        uid_sha256=sha256_bytes('\n'.join(u).encode()),
        data_file_sha256=sha256_file(data_path),
        uid_file_sha256=sha256_file(uid_path),
        data_file_size=ds.st_size,
        uid_file_size=us.st_size,
        data_file_mtime_ns=ds.st_mtime_ns,
        uid_file_mtime_ns=us.st_mtime_ns,
    )
    p.with_suffix('.json').write_text(json.dumps(m, indent=2, sort_keys=True))


def read_cache(prefix: str | Path, expected_meta: dict | None = None, strict_hash: bool = False):
    p = Path(prefix)
    data_path = p.with_suffix('.npy')
    uid_path = p.with_name(p.name + '_uids.npy')
    sidecar = p.with_suffix('.json')
    if not data_path.is_file() or not uid_path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f'incomplete cache: {p}')
    m = json.loads(sidecar.read_text())
    ds, us = data_path.stat(), uid_path.stat()
    for key, actual in [('data_file_size', ds.st_size), ('uid_file_size', us.st_size)]:
        if key in m and int(m[key]) != int(actual):
            raise ValueError(f'cache file-size mismatch: {key}')
    # mtime is a cheap warm-cache corruption signal. Strict mode additionally rehashes bytes.
    for key, actual in [
        ('data_file_mtime_ns', ds.st_mtime_ns),
        ('uid_file_mtime_ns', us.st_mtime_ns),
    ]:
        if key in m and int(m[key]) != int(actual):
            raise ValueError(f'cache file metadata changed: {key}; use strict verification/rebuild')
    if strict_hash:
        if m.get('data_file_sha256') != sha256_file(data_path):
            raise ValueError('cache data file hash mismatch')
        if m.get('uid_file_sha256') != sha256_file(uid_path):
            raise ValueError('cache UID file hash mismatch')
    a = np.load(data_path, mmap_mode='r')
    u = np.load(uid_path)
    if list(a.shape) != m.get('shape') or str(a.dtype) != m.get('dtype'):
        raise ValueError('cache shape/dtype sidecar mismatch')
    if sha256_bytes('\n'.join(map(str, u)).encode()) != m.get('uid_sha256'):
        raise ValueError('uid hash mismatch')
    if expected_meta:
        for k, v in expected_meta.items():
            if m.get(k) != v:
                raise ValueError(f'stale/incompatible cache: {k}: expected {v!r}, got {m.get(k)!r}')
    return a, u, m


def aligned_cache(
    prefix: str | Path, reference_uids, expected_meta: dict | None = None, strict_hash: bool = False
) -> np.ndarray:
    a, u, _ = read_cache(prefix, expected_meta, strict_hash)
    ref = list(map(str, reference_uids))
    pos = {str(uid): i for i, uid in enumerate(u)}
    if len(pos) != len(u) or not set(ref).issubset(pos):
        raise ValueError(f'cache UID set does not cover requested rows: {prefix}')
    return np.asarray(a)[[pos[x] for x in ref]]
