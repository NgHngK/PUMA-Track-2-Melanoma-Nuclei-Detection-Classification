from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv
import json
import re
from typing import Iterable

import numpy as np
from PIL import Image
from shapely.geometry import shape

from CODE.COMMON.constants import CLASSES, CLASS_TO_ID
from CODE.COMMON.exceptions import ManifestError, ClassMapError, CoordinateError, ImageContractError
from CODE.COMMON.hashing import canonical_json_hash, sha256_strings

LOCAL_DIR = "01_training_dataset_tif_ROIs"
CONTEXT_DIR = "01_training_dataset_tif_context_ROIs"
NUCLEI_DIR = "01_training_dataset_geojson_nuclei"
TISSUE_DIR = "01_training_dataset_geojson_tissue"

EXPECTED_CLASS_COUNTS = {
    "tumor": 57234,
    "lymphocyte": 21643,
    "plasma_cell": 520,
    "histiocyte": 7168,
    "melanophage": 695,
    "neutrophil": 366,
    "stroma": 3856,
    "epithelium": 2211,
    "endothelium": 1696,
    "apoptosis": 1804,
}
EXPECTED_NUCLEI = 97193

ROI_RE = re.compile(r"^training_set_(primary|metastatic)_roi_(\d{3})\.tif$", re.IGNORECASE)


def normalize_roi_id(value: str) -> str:
    s = str(value).strip()
    s = s.removeprefix("training_set_")
    s = s.removesuffix("_nuclei.geojson")
    s = s.removesuffix("_tissue.geojson")
    s = s.removesuffix("_context.tif")
    s = s.removesuffix(".tif")
    if not re.match(r"^(primary|metastatic)_roi_\d{3}$", s):
        raise ManifestError(f"Unrecognized ROI identifier {value!r}")
    return s


def _expected_paths(root: Path, roi_id: str) -> dict[str, Path]:
    stem = f"training_set_{roi_id}"
    return {
        "image": root / LOCAL_DIR / f"{stem}.tif",
        "context": root / CONTEXT_DIR / f"{stem}_context.tif",
        "nuclei": root / NUCLEI_DIR / f"{stem}_nuclei.geojson",
        "tissue": root / TISSUE_DIR / f"{stem}_tissue.geojson",
    }


def discover_rois(root: str | Path, require_all_four: bool = True) -> list[dict[str, str]]:
    root = Path(root).expanduser().resolve()
    local = root / LOCAL_DIR
    if not local.is_dir():
        raise ManifestError(f"Missing dataset folder: {local}")
    rois = []
    for p in sorted(local.glob("training_set_*_roi_*.tif")):
        m = ROI_RE.match(p.name)
        if not m:
            continue
        roi_id = f"{m.group(1).lower()}_roi_{m.group(2)}"
        paths = _expected_paths(root, roi_id)
        missing = [k for k, q in paths.items() if not q.is_file()]
        if require_all_four and missing:
            raise ManifestError(f"ROI {roi_id} missing paired files: {missing}")
        rois.append({"roi_id": roi_id, **{k: str(v) for k, v in paths.items()}})
    if not rois:
        raise ManifestError(f"No ROI TIFFs found under {local}")
    ids = [r["roi_id"] for r in rois]
    if len(ids) != len(set(ids)):
        raise ManifestError("Duplicate ROI IDs discovered")
    return rois


def load_excluded_rois(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    p = Path(path)
    if not p.is_file():
        raise ManifestError(f"Exclude-ROI file does not exist: {p}")
    out = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.add(normalize_roi_id(line))
    return out


def _feature_class_name(feature: dict) -> str:
    props = feature.get("properties") or {}
    cls = props.get("classification") or {}
    name = cls.get("name")
    if not isinstance(name, str):
        raise ClassMapError("GeoJSON feature missing properties.classification.name")
    name = name.removeprefix("nuclei_")
    if name not in CLASS_TO_ID:
        raise ClassMapError(f"Unknown nuclei class {name!r}")
    return name


def _centroid(feature: dict, roi_id: str, ordinal: int) -> tuple[float, float]:
    geom = shape(feature.get("geometry"))
    if geom.is_empty or (not geom.is_valid) or geom.area <= 0:
        raise CoordinateError(f"Invalid nucleus geometry {roi_id}/{ordinal}")
    c = geom.centroid
    x, y = float(c.x), float(c.y)
    if not np.isfinite([x, y]).all():
        raise CoordinateError(f"Nonfinite centroid {roi_id}/{ordinal}")
    return x, y


def build_manifest_rows(
    root: str | Path,
    *,
    excluded_rois: Iterable[str] = (),
    role: str = "train",
    role_by_roi: dict[str, str] | None = None,
    exposure_by_roi: dict[str, str] | None = None,
    require_expected_205: bool = True,
) -> tuple[list[dict], dict]:
    root = Path(root).expanduser().resolve()
    excluded = {normalize_roi_id(x) for x in excluded_rois}
    discovered = discover_rois(root, require_all_four=True)
    if require_expected_205 and len(discovered) != 205:
        raise ManifestError(f"Expected 205 paired ROIs, found {len(discovered)}")
    rows: list[dict] = []
    class_counts = Counter()
    source_class_counts = Counter()
    kept_roi_counts = Counter()
    source_feature_count = 0
    for item in discovered:
        roi_id = item["roi_id"]
        row_role = (role_by_roi or {}).get(roi_id, role)
        row_exposure = (exposure_by_roi or {}).get(roi_id, "UNKNOWN_NOT_IN_DATASET_METADATA")
        obj = json.loads(Path(item["nuclei"]).read_text(encoding="utf-8"))
        features = obj.get("features") if isinstance(obj, dict) else None
        if not isinstance(features, list):
            raise ManifestError(f"Bad FeatureCollection in {item['nuclei']}")
        source_feature_count += len(features)
        parsed_names = []
        for feature in features:
            nm = _feature_class_name(feature)
            source_class_counts[nm] += 1
            parsed_names.append(nm)
        if roi_id in excluded:
            continue
        with Image.open(item["image"]) as im:
            width, height = im.size
            if (width, height) != (1024, 1024):
                raise ImageContractError(
                    f"Expected 1024x1024 ROI, got {(width,height)} for {roi_id}"
                )
        for ordinal, feature in enumerate(features):
            name = parsed_names[ordinal]
            label = CLASS_TO_ID[name]
            x, y = _centroid(feature, roi_id, ordinal)
            if not (0 <= x < width and 0 <= y < height):
                raise CoordinateError(f"Centroid outside 1024 ROI: {roi_id}/{ordinal} {(x,y)}")
            feature_id = feature.get("id")
            if feature_id is None:
                feature_id = f"ordinal_{ordinal:06d}"
            uid = f"{roi_id}:{feature_id}"
            rows.append(
                {
                    "uid": uid,
                    "roi_id": roi_id,
                    "image_path": str(Path(item["image"]).resolve()),
                    "context_image_path": str(Path(item["context"]).resolve()),
                    "nuclei_geojson_path": str(Path(item["nuclei"]).resolve()),
                    "tissue_geojson_path": str(Path(item["tissue"]).resolve()),
                    "x": x,
                    "y": y,
                    "label": label,
                    "class_name": name,
                    "role": row_role,
                    "coordinate_source": "gt",
                    "historical_exposure": row_exposure,
                }
            )
            class_counts[name] += 1
            kept_roi_counts[roi_id] += 1
    if require_expected_205:
        prim = sum(r["roi_id"].startswith("primary_") for r in discovered)
        meta = sum(r["roi_id"].startswith("metastatic_") for r in discovered)
        if (prim, meta) != (103, 102):
            raise ManifestError(f"Expected 103 primary + 102 metastatic ROIs, found {(prim,meta)}")
        if source_feature_count != EXPECTED_NUCLEI:
            raise ManifestError(f"Expected {EXPECTED_NUCLEI} nuclei, found {source_feature_count}")
        got = {c: int(source_class_counts[c]) for c in CLASSES}
        if got != EXPECTED_CLASS_COUNTS:
            raise ClassMapError(f"Full-dataset class census mismatch: {got}")
    uids = [r["uid"] for r in rows]
    if len(uids) != len(set(uids)):
        raise ManifestError("Generated UIDs are not unique")
    rows.sort(key=lambda r: (r["roi_id"], r["uid"]))
    summary = {
        "dataset_root": str(root),
        "discovered_roi_count": len(discovered),
        "kept_roi_count": len(kept_roi_counts),
        "excluded_roi_count": len(excluded),
        "excluded_rois": sorted(excluded),
        "source_semantic_feature_count": source_feature_count,
        "kept_nucleus_count": len(rows),
        "class_counts": {c: int(class_counts[c]) for c in CLASSES},
        "source_class_counts": {c: int(source_class_counts[c]) for c in CLASSES},
        "class_order": list(CLASSES),
        "uid_order_sha256": sha256_strings([r["uid"] for r in rows]),
        "manifest_semantics": "one row per semantic GeoJSON feature; area centroid; ROI is highest verified grouping level",
        "wide_feature_source": "same 1024x1024 ROI TIFF, FOV192; context TIFF is paired/audited but not used by default",
        "role_counts": dict(Counter(r["role"] for r in rows)),
        "provenance_warning": "historical E8 role/exposure is external to the dataset folder; strict transfer requires an audited ROI provenance CSV",
    }
    summary["summary_sha256"] = canonical_json_hash(summary)
    return rows, summary


def write_manifest(rows: list[dict], path: str | Path) -> None:
    if not rows:
        raise ManifestError("Cannot write empty manifest")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def audit_context_center_pair(local_path: str | Path, context_path: str | Path) -> dict:
    with Image.open(local_path) as a, Image.open(context_path) as b:
        a = np.asarray(a.convert("RGB"))
        b = np.asarray(b.convert("RGB"))
    if a.shape[:2] != (1024, 1024):
        raise ImageContractError(f"Local ROI shape {a.shape}")
    if b.shape[:2] != (5120, 5120):
        raise ImageContractError(f"Context ROI shape {b.shape}")
    offset = (b.shape[0] - a.shape[0]) // 2
    center = b[offset : offset + 1024, offset : offset + 1024]
    exact = bool(np.array_equal(a, center))
    return {
        "exact_center_match": exact,
        "context_offset_x": offset,
        "context_offset_y": offset,
        "mean_abs_diff": float(np.abs(a.astype(np.int16) - center.astype(np.int16)).mean()),
    }
