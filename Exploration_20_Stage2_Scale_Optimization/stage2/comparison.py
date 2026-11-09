"""Out-of-fold Stage 2 ROI F1 comparison for both Exploration 20 graph variants."""

import csv
import json
from pathlib import Path

import numpy as np

from .data import class_names
from .settings import VARIANTS
from .splits import load_splits
from common.run_history import file_fingerprint


def compare(output_dir, plans=("A", "B"), variants=VARIANTS):
    """Require every fold, average held-out ROIs per plan, then average plans equally."""
    root = Path(output_dir) / "stage2"
    rows = []
    fold_rows = []
    for plan in plans:
        splits = load_splits(plan)
        expected = set().union(*(set(splits[fold]["val"]) for fold in range(4)))
        plan_membership = None
        for mode in variants:
            per_roi = {}
            per_roi_class = {}
            observed = set()
            nuclei_count = 0
            for fold in range(4):
                folder = root / f"r224_{mode.lower()}" / f"plan_{plan}" / f"fold_{fold}"
                path = folder / "validation_roi.json"
                if not path.is_file() or not (folder / "best.pt").is_file():
                    raise FileNotFoundError(f"Missing completed E20 {mode} fold {plan}/{fold}: {path}")
                record = json.loads(path.read_text())
                if (
                    record.get("mode") != mode
                    or record.get("plan") != plan
                    or record.get("fold") != fold
                    or record.get("metric_primary") != "roi_f1"
                    or not record.get("run_id")
                    or record.get("checkpoint_sha256") != file_fingerprint(folder / "best.pt")
                ):
                    raise ValueError(f"Incompatible ROI evaluation: {path}")
                rois = record["per_roi_f1"]
                labels = record["per_roi_class_f1"]
                if set(rois) != set(splits[fold]["val"]) or set(labels) != set(rois):
                    raise ValueError(f"Fold ROI membership mismatch: {path}")
                if set(rois) & set(per_roi):
                    raise ValueError(f"Duplicate validation ROI across folds: {path}")
                for name in rois:
                    if set(labels[name]) != set(class_names):
                        raise ValueError(f"Classwise ROI F1 must cover all ten classes: {path}")
                    value = float(rois[name])
                    if not np.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError(f"Invalid ROI F1 value: {path}")
                    if any(not np.isfinite(float(v)) or not 0 <= float(v) <= 1 for v in labels[name].values()):
                        raise ValueError(f"Invalid classwise ROI F1: {path}")
                fold_rows.append({
                    "plan": plan,
                    "fold": fold,
                    "mode": mode,
                    "rois": len(rois),
                    "nuclei": int(record["nuclei"]),
                    "best_epoch": int(record["best_epoch"]),
                    "roi_f1": float(record["roi_f1"]),
                    "macro_f1_nuclei": float(record["macro_f1_nuclei"]),
                })
                per_roi.update(rois)
                per_roi_class.update(labels)
                observed.update(record["observed_nuclei_classes"])
                nuclei_count += int(record["nuclei"])
            if set(per_roi) != expected:
                raise ValueError(f"E20 {mode} Plan {plan} has incomplete out-of-fold ROI coverage")
            if plan_membership is None:
                plan_membership = set(per_roi)
            elif set(per_roi) != plan_membership:
                raise ValueError(f"Two graph variants have different evaluated ROIs for Plan {plan}")
            roi_f1 = float(np.mean([float(per_roi[name]) for name in sorted(per_roi)]))
            macro = float(np.mean([
                np.mean([float(per_roi_class[roi][cls]) for roi in sorted(per_roi_class)])
                for cls in sorted(observed)
            ])) if observed else 0.0
            rows.append({
                "plan": plan,
                "mode": mode,
                "rois": len(per_roi),
                "nuclei": nuclei_count,
                "roi_f1": roi_f1,
                "macro_f1_nuclei": macro,
            })

    overview = {}
    for mode in variants:
        selected = [row for row in rows if row["mode"] == mode]
        overview[mode] = {
            "mean_plan_roi_f1": float(np.mean([row["roi_f1"] for row in selected])),
            "mean_plan_macro_f1_nuclei": float(np.mean([row["macro_f1_nuclei"] for row in selected])),
            "plans": {row["plan"]: row for row in selected},
        }
    ranked = sorted(variants, key=lambda key: overview[key]["mean_plan_roi_f1"], reverse=True)
    scores = [overview[mode]["mean_plan_roi_f1"] for mode in ranked]
    winner = ranked[0] if len(scores) < 2 or not np.isclose(scores[0], scores[1], rtol=0, atol=1e-12) else "TIE"
    result = {
        "exploration": 20,
        "evaluation": "GT-centroid Stage 2 ROI F1 (NOT Stage 1 + Stage 2 detection F1)",
        "primary_metric": "mean of per-plan out-of-fold roi_f1",
        "checkpoint_selection": "original val_macro_f1, NOT val ROI F1",
        "variants": overview,
        "winner": winner,
        "difference_gat_minus_g3": overview["GAT"]["mean_plan_roi_f1"] - overview["G3"]["mean_plan_roi_f1"],
    }
    root.mkdir(parents=True, exist_ok=True)
    path = root / "exploration20_comparison.json"
    path.write_text(json.dumps(result, indent=2))
    with (root / "exploration20_comparison.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("plan", "mode", "rois", "nuclei", "roi_f1", "macro_f1_nuclei"))
        writer.writeheader()
        writer.writerows(rows)

    with (root / "exploration20_folds.csv").open("w", newline="", encoding="utf8") as stream:
        columns = ("plan", "fold", "mode", "rois", "nuclei", "best_epoch", "roi_f1", "macro_f1_nuclei")
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(fold_rows)

    print("\n" + "=" * 76, flush=True)
    print("EXPLORATION 20 | SHARED FOV112/224 | GRAPH R224 | ROI F1 (OOF)", flush=True)
    print("Evaluation at GT nucleus centers: not end-to-end detector performance.", flush=True)
    print("Model checkpoints selected by original validation nucleus Macro-F1.", flush=True)
    print("-" * 76, flush=True)
    print(f"{'Variant':<14} {'Plan A ROI F1':>15} {'Plan B ROI F1':>15} {'MEAN':>14}", flush=True)
    for mode in variants:
        x = overview[mode]
        print(f"{mode:<14} {x['plans']['A']['roi_f1']:>15.6f} "
              f"{x['plans']['B']['roi_f1']:>15.6f} {x['mean_plan_roi_f1']:>14.6f}", flush=True)
    print("-" * 76, flush=True)
    print(f"WINNER (higher mean of Plan A/B ROI F1): {winner}", flush=True)
    print(f"GAT - G3 = {result['difference_gat_minus_g3']:+.6f}", flush=True)
    print(f"Saved: {path}", flush=True)
    return result
