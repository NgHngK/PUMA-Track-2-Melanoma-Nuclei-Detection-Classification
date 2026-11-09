"""Atomic best-model checkpoint only; optimizer/EMA snapshots are never persisted."""

from pathlib import Path
import os
import shutil
import tempfile
import torch


def save_best(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix="puma_best_", suffix=".pt", dir="/content" if Path("/content").is_dir() else None
    )
    os.close(fd)
    tmp = Path(tmp)
    stage_fd, stage_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(stage_fd)
    staged = Path(stage_name)
    try:
        torch.save(state, tmp)
        original_size = tmp.stat().st_size
        shutil.copyfile(tmp, staged)
        if staged.stat().st_size != original_size:
            raise IOError(f"Checkpoint copy incomplete: {path}")
        staged.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
        staged.unlink(missing_ok=True)
