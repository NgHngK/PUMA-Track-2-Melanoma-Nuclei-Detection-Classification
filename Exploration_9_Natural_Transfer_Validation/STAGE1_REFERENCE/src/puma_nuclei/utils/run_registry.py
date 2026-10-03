from __future__ import annotations

import re
from pathlib import Path


def create_experiment_directory(root: Path, hypothesis_name: str) -> tuple[str, Path]:
    name = hypothesis_name.strip()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None:
        raise ValueError("hypothesis_name may contain only letters, digits, '.', '_' and '-'")
    directory = Path(root) / "runs" / name
    directory.mkdir(parents=True, exist_ok=True)
    return name, directory

