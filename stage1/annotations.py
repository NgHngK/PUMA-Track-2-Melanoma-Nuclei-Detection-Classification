from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any
import numpy as np
from .constants import CLASS_TO_ID


def _token(value: Any) -> str:
    return re.sub("_+", "_", re.sub("[^a-z0-9]+", "_", str(value or "").lower())).strip("_")


NUCLEUS_ALIASES = {
    "tumor": "nuclei_tumor",
    "tumour": "nuclei_tumor",
    "nuclei_tumor": "nuclei_tumor",
    "lymphocyte": "nuclei_lymphocyte",
    "nuclei_lymphocyte": "nuclei_lymphocyte",
    "plasma": "nuclei_plasma_cell",
    "plasma_cell": "nuclei_plasma_cell",
    "nuclei_plasma_cell": "nuclei_plasma_cell",
    "histiocyte": "nuclei_histiocyte",
    "histocyte": "nuclei_histiocyte",
    "nuclei_histiocyte": "nuclei_histiocyte",
    "melanophage": "nuclei_melanophage",
    "melaphonage": "nuclei_melanophage",
    "nuclei_melanophage": "nuclei_melanophage",
    "neutrophil": "nuclei_neutrophil",
    "nuclei_neutrophil": "nuclei_neutrophil",
    "stroma": "nuclei_stroma",
    "stromal": "nuclei_stroma",
    "nuclei_stroma": "nuclei_stroma",
    "endothelium": "nuclei_endothelium",
    "endothelial": "nuclei_endothelium",
    "nuclei_endothelium": "nuclei_endothelium",
    "epithelium": "nuclei_epithelium",
    "epithelial": "nuclei_epithelium",
    "nuclei_epithelium": "nuclei_epithelium",
    "apoptosis": "nuclei_apoptosis",
    "apoptotic": "nuclei_apoptosis",
    "nuclei_apoptosis": "nuclei_apoptosis",
}


def _label_from_mapping(mapping: dict[str, Any], aliases: dict[str, Any]) -> Any | None:
    values: list[Any] = []
    for key in ("name", "label", "class", "classification", "type", "sub_type"):
        value = mapping.get(key)
        if isinstance(value, dict):
            values.extend(value.values())
        else:
            values.append(value)
    for value in values:
        token = _token(value)
        if token in aliases:
            return aliases[token]
        generic = {
            "nuclei",
            "nucleus",
            "cell",
            "cells",
            "annotation",
            "annotations",
            "tissue",
            "region",
            "roi",
        }
        for alias, canonical in aliases.items():
            if token.startswith(alias + "_"):
                remainder = token[len(alias) + 1 :].split("_")
                if remainder and all((part in generic for part in remainder)):
                    return canonical
            if token.endswith("_" + alias):
                remainder = token[: -(len(alias) + 1)].split("_")
                if remainder and all((part in generic for part in remainder)):
                    return canonical
    return None


def annotation_centroid(points: np.ndarray) -> np.ndarray:
    """Official centroid for competition path-point polygon annotations."""
    points = np.asarray(points, dtype=np.float32)

    if points.ndim != 2 or len(points) < 3 or points.shape[1] < 2:
        raise ValueError("Nucleus polygon needs at least three XY path points")
    if not np.isfinite(points[:, :2]).all():
        raise ValueError("Nucleus polygon contains nonfinite XY path points")

    return points[:, :2].mean(axis=0, dtype=np.float64).astype(np.float32)


def load_nuclei_instances(path: Path) -> list[tuple[int, np.ndarray]]:
    """One annotated FEATURE = one nucleus, including a MultiPolygon feature.

    The official PUMA dataset contains 97,429 annotated nuclei in 206 ROIs.
    Use area-weighted geometric centroid for GeoJSON and mean vertices for
    official competition `path_points`, preserving instance cardinality.
    This is the SAME parser used by both training stages.
    """
    from shapely.geometry import shape

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    result = []
    if payload.get("type") == "FeatureCollection":
        for i, feature in enumerate(payload.get("features", [])):
            label = _label_from_mapping(feature.get("properties") or {}, NUCLEUS_ALIASES)
            if label is None:
                label = _label_from_mapping(feature, NUCLEUS_ALIASES)
            geometry = feature.get("geometry") or {}
            if label is None:
                raise ValueError(f"Unknown nucleus label in {path}, feature {i}")
            if geometry.get("type") not in ("Polygon", "MultiPolygon"):
                raise ValueError(f"Invalid geometry in {path}, feature {i}: {geometry.get('type')}")
            poly = shape(geometry)
            if poly.is_empty or not np.isfinite(poly.bounds).all():
                raise ValueError(f"Empty/nonfinite geometry in {path}, feature {i}")
            center = poly.centroid
            if not (np.isfinite(center.x) and np.isfinite(center.y)):
                raise ValueError(f"Invalid centroid in {path}, feature {i}")
            result.append((CLASS_TO_ID[label], np.asarray([center.x, center.y], dtype=np.float32)))
        return result
    if payload.get("type") == "Multiple polygons":
        for i, item in enumerate(payload.get("polygons", [])):
            label = _label_from_mapping(item, NUCLEUS_ALIASES)
            if label is None:
                raise ValueError(f"Unknown nucleus label in {path}, polygon {i}")
            points = np.asarray(item.get("path_points", []), dtype=np.float32)
            result.append((CLASS_TO_ID[label], annotation_centroid(points)))
        return result
    raise ValueError(f"Unsupported annotation format: {path}")


def normalized_stem(path: Path | str) -> str:
    stem = Path(path).stem.lower()
    stem = re.sub("(?i)([_-]?(nuclei|nucleus|tissue|annotations?|annotation))+$", "", stem)
    return re.sub("[^a-z0-9]+", "_", stem).strip("_")
