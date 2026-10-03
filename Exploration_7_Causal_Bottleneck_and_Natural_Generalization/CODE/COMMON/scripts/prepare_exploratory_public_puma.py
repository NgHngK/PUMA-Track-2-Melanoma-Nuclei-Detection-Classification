from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath

import pandas as pd

from puma_exploration7.constants import NUM_CLASSES
from puma_exploration7.group_diversity import (
    build_matched_density_indices,
    build_matched_diversity_indices,
)
from puma_exploration7.manifest import (
    assert_confirmatory_manifest,
    assert_external_group_disjoint,
    read_manifest,
)

ROLES = ("train", "calibration", "test")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def select_role(
    counts: pd.DataFrame,
    available: set[str],
    minimum_groups: int,
    rich_groups: int,
) -> list[str]:
    """Deterministically select compact class-covering ROI sets for a smoke run."""
    chosen: set[str] = set()
    pool = counts.loc[sorted(available)]
    for class_id in range(NUM_CLASSES):
        positive = pool[class_id][pool[class_id] > 0]
        if len(positive) < minimum_groups:
            raise ValueError(
                f"class {class_id} has only {len(positive)} available positive ROIs; "
                f"need {minimum_groups}"
            )
        chosen.update(positive.sort_values(ascending=False, kind="stable").head(rich_groups).index)

    while True:
        support = (
            (pool.loc[sorted(chosen)] > 0).sum() if chosen else pd.Series(0, index=pool.columns)
        )
        deficits = (minimum_groups - support).clip(lower=0)
        if int(deficits.sum()) == 0:
            break
        candidates = sorted(available - chosen)
        scored = []
        for roi in candidates:
            present = (pool.loc[roi] > 0).astype(int)
            coverage = int((present * deficits).sum())
            rare_mass = float(
                sum(
                    min(float(pool.loc[roi, class_id]), 40.0) / 40.0
                    for class_id in range(NUM_CLASSES)
                    if deficits[class_id] > 0
                )
            )
            scored.append((coverage, rare_mass, roi))
        coverage, _, roi = max(scored, key=lambda item: (item[0], item[1], item[2]))
        if coverage <= 0:
            raise ValueError(f"cannot satisfy class-group deficits: {deficits.to_dict()}")
        chosen.add(roi)
    return sorted(chosen)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a compact, explicitly exploratory split from exposed public PUMA data."
    )
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--local-image-dir", required=True, type=Path)
    parser.add_argument("--colab-image-root", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--bundle-images",
        action="store_true",
        help="Copy only selected ROI TIFFs into OUT_DIR/images for a portable Colab upload.",
    )
    parser.add_argument("--train-groups-per-class", type=int, default=8)
    parser.add_argument("--evaluation-groups-per-class", type=int, default=2)
    args = parser.parse_args()

    source = read_manifest(args.source_manifest)
    if set(source["label"].unique()) != set(range(NUM_CLASSES)):
        raise ValueError("source manifest must contain all ten classes")
    counts = (
        source.assign(one=1)
        .pivot_table(index="roi", columns="label", values="one", aggfunc="sum", fill_value=0)
        .reindex(columns=range(NUM_CLASSES), fill_value=0)
        .astype(int)
    )
    available = set(counts.index.astype(str))
    selected = {}
    selected["train"] = select_role(counts, available, args.train_groups_per_class, rich_groups=2)
    available -= set(selected["train"])
    selected["calibration"] = select_role(
        counts, available, args.evaluation_groups_per_class, rich_groups=1
    )
    available -= set(selected["calibration"])
    selected["test"] = select_role(
        counts, available, args.evaluation_groups_per_class, rich_groups=1
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {role: args.out_dir / f"prompt7_natural_{role}_manifest.csv" for role in ROLES}
    audit_path = args.out_dir / "EXPLORATORY_PUBLIC_PUMA_AUDIT.json"
    collisions = [path for path in [*outputs.values(), audit_path] if path.exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite outputs: {collisions}")
    bundled_dir = args.out_dir / "images"
    if args.bundle_images:
        bundled_dir.mkdir(parents=True, exist_ok=True)

    manifests = {}
    for role in ROLES:
        frame = source[source["roi"].isin(selected[role])].copy()
        frame["split"] = role
        frame["image"] = frame["image"].map(
            lambda value: str(PurePosixPath(args.colab_image_root) / Path(value).name)
        )
        local_missing = [
            name
            for name in frame["image"].map(lambda value: PurePosixPath(value).name).unique()
            if not (args.local_image_dir / name).is_file()
        ]
        if local_missing:
            raise FileNotFoundError(f"missing local source images; first={local_missing[0]}")
        if args.bundle_images:
            for name in frame["image"].map(lambda value: PurePosixPath(value).name).unique():
                destination = bundled_dir / name
                if not destination.exists():
                    shutil.copy2(args.local_image_dir / name, destination)
        frame.to_csv(outputs[role], index=False)
        checked = read_manifest(outputs[role])
        assert_confirmatory_manifest(checked, role)
        manifests[role] = {
            "path": str(outputs[role].resolve()),
            "sha256": sha256(outputs[role]),
            "rows": len(checked),
            "rois": checked["roi"].nunique(),
            "groups": checked["group"].nunique(),
            "class_counts": checked["label"].value_counts().sort_index().to_dict(),
            "roster": selected[role],
        }

    checked = {role: read_manifest(path) for role, path in outputs.items()}
    assert_external_group_disjoint(checked["train"], checked["calibration"])
    assert_external_group_disjoint(checked["train"], checked["test"])
    assert_external_group_disjoint(checked["calibration"], checked["test"])
    _, diversity_audit = build_matched_diversity_indices(checked["train"], "group")
    _, density_audit = build_matched_density_indices(checked["train"], "group")
    diversity_classes = int((diversity_audit["status"] == "included").sum())
    density_classes = int((density_audit["status"] == "included").sum())
    if diversity_classes != NUM_CLASSES or density_classes != NUM_CLASSES:
        raise ValueError(
            "selected train cohort cannot exercise every Exp3 class: "
            f"diversity={diversity_classes}, density={density_classes}"
        )

    audit = {
        "status": "EXPLORATORY_ONLY_NOT_CONFIRMATORY",
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256(args.source_manifest),
        "source_exposure": "Public PUMA ROIs already used or opened during Exploration 6.",
        "selection": (
            "Deterministic label-informed compact ROI selection for code-path coverage; "
            "all nuclei in each selected ROI are retained."
        ),
        "colab_image_root": args.colab_image_root,
        "bundled_images": args.bundle_images,
        "bundled_image_count": len(list(bundled_dir.glob("*.tif*"))) if args.bundle_images else 0,
        "exp3_eligible_classes": {
            "diversity": diversity_classes,
            "density": density_classes,
        },
        "manifests": manifests,
        "warning": (
            "These files are only for pipeline debugging. Their label-informed ROI selection and "
            "historical exposure invalidate prospective or confirmatory interpretation."
        ),
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "audit": str(audit_path), "manifests": manifests}, indent=2))


if __name__ == "__main__":
    main()
