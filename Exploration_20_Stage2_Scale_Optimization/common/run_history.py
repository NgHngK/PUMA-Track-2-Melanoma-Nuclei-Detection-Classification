"""Two-file fold lifecycle: skip complete runs; restart incomplete ones from epoch 1.

No optimizer-resume state is read from a best.pt checkpoint or history CSV.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import pickle
from pathlib import Path

SCHEMA = "puma-audited-v6-20261028"


def file_fingerprint(path: Path) -> str:
    """Content identity; equal size and modification time do not imply equal bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(config: dict) -> str:
    payload = json.dumps({"schema": SCHEMA, **config}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _valid_rows(history: Path, epochs: int, run_id: str):
    with history.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)

    if len(rows) != epochs or not rows:
        return None

    if [int(row["epoch"]) for row in rows] != list(range(1, epochs + 1)):
        return None

    if any(row.get("run_id") != run_id for row in rows):
        return None

    metric = "val_macro_f1" if "val_macro_f1" in rows[0] else "val_f1"
    if any(metric not in row or not math.isfinite(float(row[metric])) for row in rows):
        return None

    return rows, metric


def status(folder: Path, epochs: int, run_id: str) -> str:
    """Return new/skip/restart, validating all epochs AND the best weight."""
    folder = Path(folder)
    if not folder.exists() or (folder.is_dir() and not any(folder.iterdir())):
        return "new"

    history = folder / "history.csv"
    checkpoint = folder / "best.pt"
    if not history.is_file() or not checkpoint.is_file():
        return "restart"

    try:
        inspected = _valid_rows(history, epochs, run_id)
        if inspected is None:
            return "restart"

        rows, metric = inspected
        import torch

        weights = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(weights, dict) or weights.get("run_id") != run_id:
            return "restart"
        if not isinstance(weights.get("model"), dict) or not weights["model"]:
            return "restart"
        if weights.get("schema") != SCHEMA:
            return "restart"
        if any(
            not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
            for value in weights["model"].values()
        ):
            return "restart"

        chosen_epoch = int(weights["epoch"])
        if chosen_epoch < 1 or chosen_epoch > epochs:
            return "restart"

        selected = max(rows, key=lambda item: float(item[metric]))
        if int(selected["epoch"]) != chosen_epoch:
            return "restart"

        return "skip"
    except (
        ValueError,
        KeyError,
        OSError,
        TypeError,
        RuntimeError,
        EOFError,
        pickle.UnpicklingError,
        OverflowError,
        IndexError,
    ):
        return "restart"


def initialize(folder: Path, epochs: int, run_id: str) -> bool:
    """Skip a verified run, otherwise preserve old output and start a new run."""
    folder = Path(folder)
    result = status(folder, epochs, run_id)

    if result == "skip":
        print(f"SKIP verified {epochs}-epoch fold: {folder}", flush=True)
        return True

    if result == "restart":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        previous = folder.with_name(folder.name + ".previous." + stamp)
        folder.rename(previous)
        print(f"Previous output preserved at {previous}; restarting from epoch 1", flush=True)

    folder.mkdir(parents=True, exist_ok=True)
    return False


def append_csv(path: Path, columns, row: dict) -> None:
    """Persist a complete CSV row immediately after each epoch."""
    path = Path(path)
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="raise")
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
