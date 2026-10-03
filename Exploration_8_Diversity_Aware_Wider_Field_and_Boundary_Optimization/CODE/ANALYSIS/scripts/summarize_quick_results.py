from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def add(records: dict[tuple[str, str], list[float]], stage: str, model: str, value) -> None:
    if value is not None:
        records[(stage, model)].append(float(value))


def load_if_present(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize same-subset Exploration-8 quick benchmark results."
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.output_root.resolve()
    destination = (args.out or root / "QUICK_COMPARISON.csv").resolve()
    records: dict[tuple[str, str], list[float]] = defaultdict(list)

    p8d = load_if_present(root / "P8D" / "P8D.json")
    if p8d:
        for model, metrics in p8d.items():
            if not model.startswith("_"):
                add(records, "P8D", model, metrics.get("macro_f1_fixed10"))

    p8c = load_if_present(root / "P8C_SUPPORT_AWARE" / "P8C_RESULTS.json")
    if p8c:
        for row in p8c.get("results", []):
            add(records, "P8C_DIVERSITY_DENSITY", row.get("arm", "unknown"), row.get("macro_f1"))

    exposure = load_if_present(root / "P8C_EXPOSURE" / "P8C_EXPOSURE_RESULTS.json")
    if exposure:
        for row in exposure:
            add(records, "P8C_EXPOSURE", row.get("exposure", "unknown"), row.get("macro_f1"))

    p8e = load_if_present(root / "P8E" / "P8E_RESULTS.json")
    if p8e:
        for row in p8e:
            key = f"seed{row.get('seed')}_fold{row.get('fold')}"
            add(records, "P8E", row.get("kind", "unknown"), row.get("macro_f1"))
            # One baseline was evaluated for each seed/fold and repeated in A/C rows.
            if row.get("kind") == "A":
                add(records, "P8E", "local_baseline", row.get("baseline_macro_f1"))

    p8f = load_if_present(root / "P8F" / "P8F_RESULTS.json")
    if p8f:
        for row in p8f:
            add(records, "P8F", row.get("corner", "unknown"), row.get("macro_f1"))

    p8g = load_if_present(root / "P8G" / "P8G_RESULTS.json")
    if p8g:
        for row in p8g:
            add(records, "P8G", row.get("model", "unknown"), row.get("macro_f1"))

    rows = []
    for (stage, model), values in sorted(records.items()):
        rows.append(
            {
                "stage": stage,
                "model_or_arm": model,
                "metric": "grouped-CV fixed-10 semantic Macro-F1",
                "mean": statistics.fmean(values),
                "sample_sd": statistics.stdev(values) if len(values) > 1 else "",
                "n": len(values),
                "comparison_status": "same_quick_subset",
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stage",
                "model_or_arm",
                "metric",
                "mean",
                "sample_sd",
                "n",
                "comparison_status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "output": str(destination),
                "rows": len(rows),
                "stages": sorted({row["stage"] for row in rows}),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
