"""Read-only summary for the evaluation/history.csv produced by Notebook 04."""

import csv
from pathlib import Path


def summarize(results_dir):
    path = Path(results_dir) / "evaluation" / "history.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Run Notebook 04 first: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r["fold"] == "OOF":
            print(
                f"{r['plan']} | {r['architecture']} | official nuclei OOF macro F1="
                f"{float(r['macro_f1_nuclei']):.6f} | ROIs={r['num_rois']}"
            )
    return rows
