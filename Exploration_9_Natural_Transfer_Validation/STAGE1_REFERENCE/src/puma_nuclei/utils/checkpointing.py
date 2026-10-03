from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if state is None:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


class CheckpointManager:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.directory / "latest.pt"
        self.best_path = self.directory / "best.pt"
        self.final_path = self.directory / "final.pt"
        # Interrupted atomic saves may leave harmless .tmp files. Remove them so
        # a later Colab session cannot confuse manual inspection or disk checks.
        for target in (self.latest_path, self.best_path, self.final_path):
            target.with_suffix(target.suffix + ".tmp").unlink(missing_ok=True)

    def save_latest(
        self,
        *,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        _atomic_torch_save(
            {
                "epoch": int(epoch),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "rng": capture_rng_state(),
                "extra": extra or {},
            },
            self.latest_path,
        )

    def save_best(self, *, model: torch.nn.Module, extra: dict[str, Any] | None = None) -> None:
        _atomic_torch_save({"model": model.state_dict(), "extra": extra or {}}, self.best_path)

    def save_final(self, *, model: torch.nn.Module, extra: dict[str, Any] | None = None) -> None:
        _atomic_torch_save({"model": model.state_dict(), "extra": extra or {}}, self.final_path)

    def resume(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        map_location: str | torch.device = "cpu",
    ) -> tuple[int, dict[str, Any]]:
        if not self.latest_path.is_file():
            return 0, {}
        payload = torch.load(self.latest_path, map_location=map_location, weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        if optimizer is not None:
            state = payload.get("optimizer")
            if state is None:
                raise RuntimeError(f"Checkpoint {self.latest_path} has no optimizer state")
            optimizer.load_state_dict(state)
        if scheduler is not None:
            state = payload.get("scheduler")
            if state is None:
                raise RuntimeError(f"Checkpoint {self.latest_path} has no scheduler state")
            scheduler.load_state_dict(state)
        if scaler is not None:
            state = payload.get("scaler")
            if state is None:
                raise RuntimeError(f"Checkpoint {self.latest_path} has no scaler state")
            scaler.load_state_dict(state)
        restore_rng_state(payload.get("rng"))
        return int(payload["epoch"]) + 1, payload.get("extra", {})

