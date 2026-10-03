from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = (
    "tumor",
    "lymphocyte",
    "plasma_cell",
    "histiocyte",
    "melanophage",
    "neutrophil",
    "stroma",
    "epithelium",
    "endothelium",
    "apoptosis",
)
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
ROLES = ("TRAIN", "CAL", "VALIDATE", "TEST")
DEFAULT_ROLE_COUNTS = {"TRAIN": 24, "CAL": 4, "VALIDATE": 4, "TEST": 4}


def annotation_class(feature: dict) -> str:
    raw = str(feature.get("properties", {}).get("classification", {}).get("name", ""))
    name = raw.removeprefix("nuclei_")
    if name not in CLASS_TO_ID:
        raise ValueError(f"unknown nucleus class {raw!r}")
    return name


def roi_name(geojson_path: Path) -> str:
    suffix = "_nuclei.geojson"
    if not geojson_path.name.endswith(suffix):
        raise ValueError(f"unexpected annotation filename: {geojson_path.name}")
    return geojson_path.name[: -len(suffix)]


def ring_centroid(ring: list) -> tuple[float, float, float]:
    points = [(float(p[0]), float(p[1])) for p in ring]
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 3:
        return sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points), 0.0
    twice_area = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]):
        cross = x0 * y1 - x1 * y0
        twice_area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(twice_area) < 1e-12:
        return sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points), 0.0
    return cx / (3.0 * twice_area), cy / (3.0 * twice_area), abs(twice_area) / 2.0


def geometry_centroid(geometry: dict) -> tuple[float, float]:
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if kind == "Polygon":
        x, y, _ = ring_centroid(coordinates[0])
        return x, y
    if kind == "MultiPolygon":
        parts = [ring_centroid(polygon[0]) for polygon in coordinates]
        weight = sum(part[2] for part in parts)
        if weight <= 0:
            return sum(part[0] for part in parts) / len(parts), sum(
                part[1] for part in parts
            ) / len(parts)
        return (
            sum(part[0] * part[2] for part in parts) / weight,
            sum(part[1] * part[2] for part in parts) / weight,
        )
    raise ValueError(f"unsupported geometry type {kind!r}")


def read_features(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("features")
    if not isinstance(features, list):
        raise ValueError(f"invalid FeatureCollection: {path}")
    return features


def census(
    annotation_dir: Path, image_dir: Path
) -> tuple[dict[str, Counter], dict[str, Path], dict[str, Path]]:
    counts: dict[str, Counter] = {}
    annotations: dict[str, Path] = {}
    images: dict[str, Path] = {}
    paths = sorted(annotation_dir.glob("*_nuclei.geojson"))
    if not paths:
        raise FileNotFoundError(f"no *_nuclei.geojson files in {annotation_dir}")
    for i, path in enumerate(paths, 1):
        roi = roi_name(path)
        image = image_dir / f"{roi}.tif"
        if not image.is_file():
            raise FileNotFoundError(image)
        counts[roi] = Counter(annotation_class(feature) for feature in read_features(path))
        annotations[roi] = path
        images[roi] = image
        if i % 25 == 0 or i == len(paths):
            print(f"census: {i}/{len(paths)} ROI files", flush=True)
    return counts, annotations, images


def role_value(
    selected: list[str], counts: dict[str, Counter], role: str, rarity: dict[str, float]
) -> float:
    # P8-C's high-diversity arm asks for eight groups per class after one fold
    # is held out. Twelve selected TRAIN groups per class leaves useful margin.
    group_target = 12 if role == "TRAIN" else 1
    presence = Counter()
    nuclei = Counter()
    subtypes = set()
    for roi in selected:
        subtypes.add("metastatic" if "_metastatic_" in roi else "primary")
        for name, n in counts[roi].items():
            if n:
                presence[name] += 1
                nuclei[name] += n
    score = 0.0
    for name in CLASSES:
        weight = rarity[name]
        score += 1000.0 * weight * min(presence[name], group_target) / group_target
        score += 0.05 * weight * math.log1p(nuclei[name])
    if role == "TRAIN":
        # Max-min coverage prevents a greedy set-cover solution from neglecting
        # one of the two rarest classes while filling the other classes.
        score += 10000.0 * min(presence[name] for name in CLASSES)
    if len(selected) >= 2:
        score += 2.0 * len(subtypes)
    return score


def choose_roles(
    counts: dict[str, Counter], role_counts: dict[str, int], seed: int, restarts: int
) -> dict[str, list[str]]:
    if sum(role_counts.values()) > len(counts):
        raise ValueError("requested more ROIs than the dataset contains")
    rois = sorted(counts)
    all_group_counts = Counter(
        name for roi_counts in counts.values() for name, n in roi_counts.items() if n
    )
    rarity = {name: 1.0 / math.sqrt(max(1, all_group_counts[name])) for name in CLASSES}
    best_score = -math.inf
    best: dict[str, list[str]] | None = None
    for restart in range(restarts):
        rng = random.Random(seed + 104729 * restart)
        available = set(rois)
        assignment = {role: [] for role in ROLES}
        role_order = list(ROLES)
        if restart:
            rng.shuffle(role_order)
        for role in role_order:
            for _ in range(role_counts[role]):
                base = role_value(assignment[role], counts, role, rarity)
                scored = []
                for roi in available:
                    gain = role_value(assignment[role] + [roi], counts, role, rarity) - base
                    scored.append((gain + rng.random() * 1e-7, roi))
                _, chosen = max(scored)
                assignment[role].append(chosen)
                available.remove(chosen)
        total = sum(role_value(assignment[role], counts, role, rarity) for role in ROLES)
        if total > best_score:
            best_score, best = total, assignment
    assert best is not None
    return {role: sorted(best[role]) for role in ROLES}


def stable_sample(
    features: list[dict], roi: str, cap: int, seed: int
) -> list[tuple[int, dict, str]]:
    by_class: dict[str, list[tuple[str, int, dict]]] = defaultdict(list)
    for index, feature in enumerate(features):
        name = annotation_class(feature)
        identity = str(feature.get("id") or index)
        key = hashlib.sha256(f"{seed}:{roi}:{identity}".encode()).hexdigest()
        by_class[name].append((key, index, feature))
    sampled = []
    for name in CLASSES:
        for _, index, feature in sorted(by_class[name])[:cap]:
            sampled.append((index, feature, name))
    return sorted(sampled)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_inputs(
    selected: set[str], annotations: dict[str, Path], images: dict[str, Path], out: Path
) -> None:
    out_images = out / "Dataset" / "01_training_dataset_tif_ROIs"
    out_annotations = out / "Dataset" / "01_training_dataset_geojson_nuclei"
    out_images.mkdir(parents=True, exist_ok=True)
    out_annotations.mkdir(parents=True, exist_ok=True)
    for i, roi in enumerate(sorted(selected), 1):
        shutil.copy2(images[roi], out_images / images[roi].name)
        shutil.copy2(annotations[roi], out_annotations / annotations[roi].name)
        print(f"copy: {i}/{len(selected)} {roi}", flush=True)


def parse_role_counts(value: str) -> dict[str, int]:
    result = dict(DEFAULT_ROLE_COUNTS)
    if value:
        result = {}
        for item in value.split(","):
            role, count = item.split("=", 1)
            role = role.strip().upper()
            if role not in ROLES:
                raise ValueError(f"unknown role {role!r}")
            result[role] = int(count)
    if set(result) != set(ROLES) or any(result[r] < 1 for r in ROLES):
        raise ValueError("role counts must contain positive TRAIN,CAL,VALIDATE,TEST values")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a portable, group-disjoint PUMA Exploration-8 quick benchmark."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--role-counts", default="TRAIN=24,CAL=4,VALIDATE=4,TEST=4")
    parser.add_argument("--max-per-class-per-roi", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--selection-restarts", type=int, default=250)
    parser.add_argument("--uni2-weights", type=Path)
    parser.add_argument(
        "--colab-project-root",
        default="/content/drive/MyDrive/Research/PUMA/Exploration_8_Diversity_Aware_Wider_Field_and_Boundary_Optimization",
    )
    parser.add_argument(
        "--skip-copy",
        action="store_true",
        help="Keep source TIFF paths; output will not be portable to Colab.",
    )
    args = parser.parse_args()
    if args.max_per_class_per_roi < 1:
        raise ValueError("--max-per-class-per-roi must be positive")

    dataset_root = args.dataset_root.resolve()
    out = args.out_dir.resolve()
    annotation_dir = dataset_root / "01_training_dataset_geojson_nuclei"
    image_dir = dataset_root / "01_training_dataset_tif_ROIs"
    role_counts = parse_role_counts(args.role_counts)
    counts, annotations, images = census(annotation_dir, image_dir)
    roles = choose_roles(counts, role_counts, args.seed, args.selection_restarts)
    role_by_roi = {roi: role for role, rois in roles.items() for roi in rois}
    selected = set(role_by_roi)
    train_group_support = {
        name: sum(counts[roi][name] > 0 for roi in roles["TRAIN"]) for name in CLASSES
    }
    if min(train_group_support.values()) < 12:
        raise RuntimeError(
            "selection could not provide the 12 TRAIN groups/class needed for the P8-C high-diversity arm; "
            f"support={train_group_support}. Increase TRAIN role count or selection restarts."
        )

    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if not args.skip_copy:
        copy_inputs(selected, annotations, images, out)

    manifest_rows = []
    selection_rows = []
    for roi in sorted(selected):
        role = role_by_roi[roi]
        features = read_features(annotations[roi])
        sample = stable_sample(features, roi, args.max_per_class_per_roi, args.seed)
        sampled_counts = Counter(name for _, _, name in sample)
        for index, feature, name in sample:
            x, y = geometry_centroid(feature["geometry"])
            image_value = (
                f"Dataset/01_training_dataset_tif_ROIs/{images[roi].name}"
                if not args.skip_copy
                else str(images[roi].resolve())
            )
            manifest_rows.append(
                {
                    "uid": f"{roi}:{index}",
                    "role": role,
                    "patient_id": "",
                    "case_id": "",
                    "slide_id": "",
                    "roi": roi,
                    "image": image_value,
                    "x": f"{x:.6f}",
                    "y": f"{y:.6f}",
                    "label": CLASS_TO_ID[name],
                    "class_name": name,
                    "coordinate_source": "gt",
                }
            )
        row = {
            "roi": roi,
            "role": role,
            "source": "metastatic" if "_metastatic_" in roi else "primary",
        }
        row.update({f"raw_{name}": counts[roi][name] for name in CLASSES})
        row.update({f"sampled_{name}": sampled_counts[name] for name in CLASSES})
        selection_rows.append(row)

    manifest_fields = [
        "uid",
        "role",
        "patient_id",
        "case_id",
        "slide_id",
        "roi",
        "image",
        "x",
        "y",
        "label",
        "class_name",
        "coordinate_source",
    ]
    write_csv(out / "EXPLORATION8_QUICK_MANIFEST.csv", manifest_fields, manifest_rows)
    selection_fields = (
        ["roi", "role", "source"]
        + [f"raw_{x}" for x in CLASSES]
        + [f"sampled_{x}" for x in CLASSES]
    )
    write_csv(out / "ROI_SELECTION.csv", selection_fields, selection_rows)

    support_rows = []
    for role in ROLES:
        role_manifest = [row for row in manifest_rows if row["role"] == role]
        for name in CLASSES:
            class_rows = [row for row in role_manifest if row["class_name"] == name]
            support_rows.append(
                {
                    "role": role,
                    "label": CLASS_TO_ID[name],
                    "class_name": name,
                    "sampled_nuclei": len(class_rows),
                    "independent_rois": len({row["roi"] for row in class_rows}),
                }
            )
    write_csv(
        out / "CLASS_SUPPORT.csv",
        ["role", "label", "class_name", "sampled_nuclei", "independent_rois"],
        support_rows,
    )

    local_weights = str(args.uni2_weights.resolve()) if args.uni2_weights else "EDIT_ME"
    local_paths = {
        "manifest": str((out / "EXPLORATION8_QUICK_MANIFEST.csv").resolve()),
        "output_root": str((out / "OUTPUTS").resolve()),
        "runtime_local": str((out / "RUNTIME").resolve()),
        "stage1_proposals": "",
        "uni2_weights": local_weights,
    }
    (out / "PATHS_LOCAL.json").write_text(json.dumps(local_paths, indent=2), encoding="utf-8")
    colab_input = f"{args.colab_project_root.rstrip('/')}/INPUTS/{out.name}"
    colab_paths = {
        "manifest": f"{colab_input}/EXPLORATION8_QUICK_MANIFEST.csv",
        "output_root": f"{args.colab_project_root.rstrip('/')}/RESULTS/{out.name}",
        "runtime_local": "/content/PUMA_P8_RUNTIME",
        "stage1_proposals": "",
        "uni2_weights": "/content/drive/MyDrive/Research/PUMA/PUMA_pretrained_checkpoints/UNI2-h/uni2_h_model.bin",
    }
    (out / "PATHS_COLAB.json").write_text(json.dumps(colab_paths, indent=2), encoding="utf-8")

    summary = {
        "purpose": "rapid exploratory benchmark; not a confirmatory full-dataset result",
        "seed": args.seed,
        "max_per_class_per_roi": args.max_per_class_per_roi,
        "source_roi_count": len(counts),
        "selected_roi_count": len(selected),
        "sampled_nuclei": len(manifest_rows),
        "role_counts": {role: len(roles[role]) for role in ROLES},
        "patient_level_verified": False,
        "group_column": "roi",
        "portable_copy": not args.skip_copy,
    }
    (out / "BUILD_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
