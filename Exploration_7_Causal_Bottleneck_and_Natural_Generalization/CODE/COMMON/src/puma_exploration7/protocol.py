from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json
from .io import atomic_json_dump, sha256_file, stable_json_hash


def freeze_protocol(config_path, out_path):
    cfg = json.loads(Path(config_path).read_text())
    payload = {
        'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'config': cfg,
        'config_sha256': stable_json_hash(cfg),
        'config_file_sha256': sha256_file(config_path),
        'status': 'FROZEN_BEFORE_RESULTS',
    }
    atomic_json_dump(out_path, payload)
    return payload
