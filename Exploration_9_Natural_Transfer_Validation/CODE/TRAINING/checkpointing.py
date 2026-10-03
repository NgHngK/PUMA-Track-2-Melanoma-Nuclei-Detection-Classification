from __future__ import annotations

import os
from pathlib import Path
import random

import numpy as np
import torch

from CODE.COMMON.exceptions import ResumeIntegrityError

REQUIRED_RESUME_KEYS = (
    "roster_hash",
    "config_hash",
    "fold_hash",
    "feature_hashes",
    "seed",
    "fold",
    "arm",
)


def validate_resume_metadata(saved: dict, expected: dict) -> None:
    for key in REQUIRED_RESUME_KEYS:
        if key not in saved or key not in expected:
            raise ResumeIntegrityError(f"Missing resume key {key}")
        if saved[key] != expected[key]:
            raise ResumeIntegrityError(
                f"Resume mismatch for {key}: {saved[key]!r} != {expected[key]!r}"
            )


def _capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    completed_epochs: int,
    history: list[dict],
    metadata: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "completed_epochs": int(completed_epochs),
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "metadata": metadata,
            "rng": _capture_rng_state(),
        },
        tmp,
    )
    os.replace(tmp, path)


def load_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_metadata: dict,
    device: str,
) -> tuple[int, list[dict]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    validate_resume_metadata(payload.get("metadata", {}), expected_metadata)
    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    _restore_rng_state(payload.get("rng"))
    completed = int(payload.get("completed_epochs", 0))
    history = list(payload.get("history", []))
    if completed != len(history):
        raise ResumeIntegrityError(
            f"Checkpoint says {completed} completed epochs but stores {len(history)} history rows"
        )
    return completed, history
