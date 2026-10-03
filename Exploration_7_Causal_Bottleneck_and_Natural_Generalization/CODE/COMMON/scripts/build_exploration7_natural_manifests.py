from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

import pandas as pd
from PIL import Image

from puma_exploration7.constants import CLASS_TO_ID
from puma_exploration7.manifest import (
    assert_confirmatory_manifest,
    assert_external_group_disjoint,
    read_manifest,
)

ROLES = ("train", "calibration", "test")
ID_COLUMNS = ("patient", "case", "slide", "group")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ring_area_centroid(ring: list[list[float]]) -> tuple[float, float, float]:
    points = [(float(p[0]), float(p[1])) for p in ring]
    if len(points) < 3:
        raise ValueError("polygon ring has fewer than three points")
    if points[0] != points[-1]:
        points.append(points[0])
    twice_area = 0.0
    cx = 0.0
    cy = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        cross = x0 * y1 - x1 * y0
        twice_area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(twice_area) < 1e-12:
        raise ValueError("polygon ring has zero area")
    return abs(twice_area) / 2.0, cx / (3.0 * twice_area), cy / (3.0 * twice_area)


def polygon_area_centroid(rings: list[list[list[float]]]) -> tuple[float, float, float]:
    if not rings:
        raise ValueError("polygon has no rings")
    weighted_x = 0.0
    weighted_y = 0.0
    net_area = 0.0
    for index, ring in enumerate(rings):
        area, cx, cy = ring_area_centroid(ring)
        weight = area if index == 0 else -area
        net_area += weight
        weighted_x += weight * cx
        weighted_y += weight * cy
    if net_area <= 0:
        raise ValueError("polygon has non-positive area after holes")
    return net_area, weighted_x / net_area, weighted_y / net_area


def geometry_centroid(geometry: dict) -> tuple[float, float]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "Polygon":
        polygons = [coordinates]
    elif geometry_type == "MultiPolygon":
        polygons = coordinates
    else:
        raise ValueError(f"unsupported geometry type: {geometry_type!r}")
    parts = [polygon_area_centroid(rings) for rings in polygons]
    area = sum(part[0] for part in parts)
    if area <= 0:
        raise ValueError("geometry has non-positive area")
    x = sum(part[0] * part[1] for part in parts) / area
    y = sum(part[0] * part[2] for part in parts) / area
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("geometry centroid is non-finite")
    return x, y


def read_roi_column(path: Path) -> set[str]:
    frame = pd.read_csv(path, usecols=["roi"])
    return set(frame["roi"].astype(str))


def output_image_path(image: Path, image_root: str | None) -> str:
    if image_root is None:
        return str(image.resolve())
    return str(PurePosixPath(image_root) / image.name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the three Exploration-7 manifests from a new PUMA-format cohort."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--split-map", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--excluded-rois-csv",
        type=Path,
        help="CSV with an roi column; fail if any candidate ROI was previously exposed.",
    )
    parser.add_argument(
        "--colab-image-root",
        help="POSIX image directory written into manifests instead of local absolute paths.",
    )
    args = parser.parse_args()

    geo_dir = args.dataset_root / "01_training_dataset_geojson_nuclei"
    image_dir = args.dataset_root / "01_training_dataset_tif_ROIs"
    geo_files = sorted(geo_dir.glob("*.geojson"))
    if not geo_files:
        raise FileNotFoundError(f"no nuclei GeoJSON files found in {geo_dir}")

    split_map = pd.read_csv(args.split_map, dtype=str).fillna("")
    required = {"roi", "split"}
    missing = sorted(required - set(split_map.columns))
    if missing:
        raise ValueError(f"split map missing columns: {missing}")
    if split_map["roi"].duplicated().any():
        raise ValueError("split map roi values must be unique")
    if set(split_map["split"]) != set(ROLES):
        raise ValueError(f"split map must contain exactly {ROLES}")
    for column in ID_COLUMNS:
        if column not in split_map.columns:
            continue
        populated = split_map[column].ne("")
        if populated.any() and not populated.all():
            raise ValueError(f"split map has incomplete {column} identifiers")
        if not populated.any():
            split_map = split_map.drop(columns=column)
    if "group" not in split_map.columns:
        split_map["group"] = split_map["roi"]

    geo_rois = {path.stem.removesuffix("_nuclei") for path in geo_files}
    mapped_rois = set(split_map["roi"])
    if geo_rois != mapped_rois:
        raise ValueError(
            "split-map/dataset ROI mismatch: "
            f"unmapped={sorted(geo_rois - mapped_rois)[:5]}, "
            f"missing={sorted(mapped_rois - geo_rois)[:5]}"
        )
    if args.excluded_rois_csv:
        overlap = geo_rois & read_roi_column(args.excluded_rois_csv)
        if overlap:
            raise ValueError(
                f"candidate cohort reuses {len(overlap)} excluded ROIs; "
                f"examples={sorted(overlap)[:5]}"
            )

    assignment = split_map.set_index("roi").to_dict("index")
    rows: list[dict] = []
    annotation_hashes: dict[str, str] = {}
    for geojson_path in geo_files:
        roi = geojson_path.stem.removesuffix("_nuclei")
        candidates = [image_dir / f"{roi}.tif", image_dir / f"{roi}.tiff"]
        images = [path for path in candidates if path.is_file()]
        if len(images) != 1:
            raise FileNotFoundError(f"expected one TIFF for {roi}; found {images}")
        image = images[0]
        with Image.open(image) as opened:
            width, height = opened.size
        collection = json.loads(geojson_path.read_text(encoding="utf-8"))
        if collection.get("type") != "FeatureCollection" or not isinstance(
            collection.get("features"), list
        ):
            raise ValueError(f"unsupported GeoJSON collection: {geojson_path}")
        meta = assignment[roi]
        annotation_hashes[roi] = sha256(geojson_path)
        for index, feature in enumerate(collection["features"]):
            try:
                class_name = feature["properties"]["classification"]["name"]
            except (KeyError, TypeError) as error:
                raise ValueError(f"missing classification at {roi}/{index}") from error
            class_name = str(class_name).removeprefix("nuclei_")
            if class_name not in CLASS_TO_ID:
                raise ValueError(f"unknown class {class_name!r} at {roi}/{index}")
            x, y = geometry_centroid(feature.get("geometry", {}))
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(
                    f"centroid outside image at {roi}/{index}: {(x, y)} vs {(width, height)}"
                )
            row = {
                "uid": f"{roi}:{index}",
                "roi": roi,
                "group": meta["group"],
                "image": output_image_path(image, args.colab_image_root),
                "x": x,
                "y": y,
                "label": CLASS_TO_ID[class_name],
                "split": meta["split"],
            }
            for column in ("patient", "case", "slide", "stain_batch"):
                if column in split_map.columns:
                    row[column] = meta[column]
            rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = {
        role: args.out_dir / f"prompt7_natural_{role}_manifest.csv" for role in ROLES
    }
    audit_path = args.out_dir / "PROMPT7_NATURAL_MANIFEST_AUDIT.json"
    collisions = [path for path in [*expected_outputs.values(), audit_path] if path.exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite outputs: {collisions}")

    frame = pd.DataFrame(rows)
    manifests = {}
    for role, path in expected_outputs.items():
        cohort = frame[frame["split"] == role].copy()
        cohort.to_csv(path, index=False)
        checked = read_manifest(path)
        group_column = assert_confirmatory_manifest(checked, role)
        manifests[role] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "rows": len(checked),
            "rois": checked["roi"].nunique(),
            "groups": checked[group_column].nunique(),
            "group_column": group_column,
            "class_counts": checked["label"].value_counts().sort_index().to_dict(),
        }
    cohorts = {role: read_manifest(path) for role, path in expected_outputs.items()}
    assert_external_group_disjoint(cohorts["train"], cohorts["calibration"])
    assert_external_group_disjoint(cohorts["train"], cohorts["test"])
    assert_external_group_disjoint(cohorts["calibration"], cohorts["test"])

    audit = {
        "status": "prepared_locally",
        "dataset_root": str(args.dataset_root.resolve()),
        "split_map": str(args.split_map.resolve()),
        "split_map_sha256": sha256(args.split_map),
        "excluded_rois_csv": (
            str(args.excluded_rois_csv.resolve()) if args.excluded_rois_csv else None
        ),
        "colab_image_root": args.colab_image_root,
        "annotation_sha256_by_roi": annotation_hashes,
        "manifests": manifests,
        "warning": (
            "This audit proves format, pairing, exclusion-list non-overlap, and group disjointness. "
            "It cannot independently prove that the source cohort was never inspected; retain the "
            "data-provider provenance and preregistration with this file."
        ),
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "audit": str(audit_path), "manifests": manifests}, indent=2))


if __name__ == "__main__":
    main()
