from __future__ import annotations
from pathlib import Path
import hashlib, json


def load_json(path: str | Path) -> dict:
    p = Path(path)
    data = json.loads(p.read_text())
    if not isinstance(data, dict):
        raise ValueError(f'config must be a JSON object: {p}')
    return data


def sha256_file(path: str | Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def sha256_json(data: dict) -> str:
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()


def config_bundle_hash(
    config_dir: str | Path, names: list[str] | None = None
) -> tuple[str, dict[str, str]]:
    root = Path(config_dir)
    paths = [root / n for n in names] if names else sorted(root.rglob('*.json'))
    hashes = {
        str(p.relative_to(root)).replace('\\', '/'): sha256_file(p) for p in paths if p.is_file()
    }
    return sha256_json(hashes), hashes


def scientific_config_bundle_hash(config_dir: str | Path) -> tuple[str, dict[str, str]]:
    """Hash only behavior-driving scientific configs.

    Runtime tuning, design-gate certificates and amendment lineage are provenance,
    not inputs to the learned function, so they are excluded deliberately.
    """
    root = Path(config_dir)
    paths = [
        p
        for p in sorted(root.rglob('*.json'))
        if p.name not in {'99_RUNTIME_PROFILE.json', 'DESIGN_GATE.json'}
        and 'AMENDMENTS' not in p.relative_to(root).parts
    ]
    hashes = {str(p.relative_to(root)).replace('\\', '/'): sha256_file(p) for p in paths}
    return sha256_json(hashes), hashes


def load_protocol(path: str | Path) -> dict:
    p = load_json(path)
    checks = {
        ('local', 'a5_rank'): 8,
        ('local', 'dim'): 1536,
        ('local', 'fov'): 96,
        ('local', 'sigma'): 1.5,
        ('local', 'tierA_dim'): 16,
        ('encoder', 'frozen'): True,
        ('encoder', 'no_lora'): True,
        ('routing', 'third_candidate_forbidden'): True,
    }
    for keys, expected in checks.items():
        value = p
        for key in keys:
            value = value[key]
        if value != expected:
            raise ValueError(
                f'fixed protocol/code contract mismatch at {".".join(keys)}: {value!r} != {expected!r}'
            )
    scale = p['routing']['global_scale']
    if (
        float(scale['initial_fraction']) != 0.25
        or not bool(scale['bounded'])
        or not bool(scale['class_shared'])
    ):
        raise ValueError('routing global-scale contract differs from implementation')
    return p
