"""Read supplied filename-only ROI folds. Do not regenerate or re-shuffle them."""

from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_fold_list(path: Path) -> dict[int, list[str]]:
    folds: dict[int, list[str]] = {}
    current = None
    for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        match = re.match("#\\s*FOLD\\s+(\\d+)\\b", line, re.IGNORECASE)
        if match:
            current = int(match.group(1))
            if current in folds:
                raise ValueError(f"Duplicate fold heading {current} in {path}")
            folds[current] = []
        elif line and (not line.startswith("#")):
            if current is None:
                raise ValueError(f"Filename before fold heading in {path}: {line}")
            if "/" in line or "\\" in line:
                raise ValueError(f"Expected an ROI basename, not a path: {line}")
            folds[current].append(Path(line).stem.lower())
    if sorted(folds) != [0, 1, 2, 3] or any((not rows for rows in folds.values())):
        raise ValueError(f"Expected four nonempty folds numbered 0..3: {path}")
    for k, rows in folds.items():
        if len(set(rows)) != len(rows):
            raise ValueError(f"Duplicate filename within fold {k}: {path}")
    return folds


def load_splits(plan: str, root: Path | None = None) -> dict[int, dict[str, list[str]]]:
    if plan not in ("A", "B"):
        raise ValueError("plan must be A or B")
    if root is None:
        from .fold_definitions import FOLD_DEFINITIONS

        trains = {i: list(FOLD_DEFINITIONS[plan][i]["train"]) for i in range(4)}
        vals = {i: list(FOLD_DEFINITIONS[plan][i]["val"]) for i in range(4)}
    else:
        root = Path(root)
        trains = read_fold_list(root / f"train_plan{plan}_4fold.txt")
        vals = read_fold_list(root / f"val_plan{plan}_4fold.txt")
    all_names = set(trains[0]) | set(vals[0])
    for fold in range(4):
        a, b = (set(trains[fold]), set(vals[fold]))
        if a & b:
            raise ValueError(f"Data leakage in fold {fold}: train/val overlap {a & b}")
        if a | b != all_names:
            raise ValueError(f"Fold {fold} changes the ROI universe")
    counts = {name: sum((name in vals[k] for k in range(4))) for name in all_names}
    anchor = "training_set_metastatic_roi_027"
    if plan == "B" and any((n != 1 for n in counts.values())):
        raise ValueError("Plan B must validate every ROI exactly once")
    if plan == "A" and (
        counts.get(anchor) != 0 or any((v != 1 for k, v in counts.items() if k != anchor))
    ):
        raise ValueError("Plan A requires a training-only anchor, all others validated once")
    return {i: {"train": trains[i], "val": vals[i]} for i in range(4)}


def expected_names(plan: str, root: Path | None = None) -> list[str]:
    folds = load_splits(plan, root)
    return sorted(set(folds[0]["train"]) | set(folds[0]["val"]))
