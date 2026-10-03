from __future__ import annotations
from pathlib import Path
import hashlib, json


def sha256_file(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_hash(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode(
        'utf-8'
    )
    return sha256_bytes(payload)


def sha256_strings(items) -> str:
    h = hashlib.sha256()
    for item in items:
        b = str(item).encode('utf-8')
        h.update(len(b).to_bytes(8, 'little'))
        h.update(b)
    return h.hexdigest()
