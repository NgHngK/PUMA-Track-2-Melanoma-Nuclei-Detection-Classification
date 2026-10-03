from __future__ import annotations
from pathlib import Path
import datetime, hashlib, json
from .constants import CLASSES
from .config import sha256_file as file_sha256


def _payload_hash(payload: dict) -> str:
    clean = {k: v for k, v in payload.items() if k != 'payload_sha256'}
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()


def write_freeze(path: str | Path, payload: dict) -> dict:
    data = dict(payload)
    data.setdefault('classes', list(CLASSES))
    data['payload_sha256'] = _payload_hash(data)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True))
    return data


def require_freeze(path: str | Path, kind: str | None = None, required_keys=()) -> dict:
    p = Path(path)
    if not p.is_file():
        raise RuntimeError(f'protected stage blocked: missing freeze artifact {p}')
    data = json.loads(p.read_text())
    if not isinstance(data, dict):
        raise RuntimeError('protected stage blocked: malformed freeze artifact')
    if data.get('payload_sha256') != _payload_hash(data):
        raise RuntimeError('protected stage blocked: freeze payload hash mismatch')
    if kind is not None and data.get('kind') != kind:
        raise RuntimeError(f'protected stage blocked: expected {kind} freeze')
    if data.get('classes') != list(CLASSES):
        raise RuntimeError('protected stage blocked: class-order mismatch')
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise RuntimeError(f'protected stage blocked: incomplete freeze artifact: {missing}')
    return data


def open_once(state_path: str | Path, kind: str, freeze_hash: str) -> None:
    p = Path(state_path)
    if p.exists():
        raise RuntimeError(f'{kind} already opened; one-shot gate refuses silent rerun')
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                'kind': kind,
                'freeze_hash': freeze_hash,
                'opened_utc': datetime.datetime.now(datetime.UTC).isoformat(),
            },
            indent=2,
        )
    )
