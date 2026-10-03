from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from ..constants import CLASS_TO_ID


def _token(value: Any) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(value or "").lower())).strip("_")


NUCLEUS_ALIASES = {
    "tumor": "nuclei_tumor", "tumour": "nuclei_tumor", "nuclei_tumor": "nuclei_tumor",
    "lymphocyte": "nuclei_lymphocyte", "nuclei_lymphocyte": "nuclei_lymphocyte",
    "plasma": "nuclei_plasma_cell", "plasma_cell": "nuclei_plasma_cell", "nuclei_plasma_cell": "nuclei_plasma_cell",
    "histiocyte": "nuclei_histiocyte", "histocyte": "nuclei_histiocyte", "nuclei_histiocyte": "nuclei_histiocyte",
    "melanophage": "nuclei_melanophage", "melaphonage": "nuclei_melanophage", "nuclei_melanophage": "nuclei_melanophage",
    "neutrophil": "nuclei_neutrophil", "nuclei_neutrophil": "nuclei_neutrophil",
    "stroma": "nuclei_stroma", "stromal": "nuclei_stroma", "nuclei_stroma": "nuclei_stroma",
    "endothelium": "nuclei_endothelium", "endothelial": "nuclei_endothelium", "nuclei_endothelium": "nuclei_endothelium",
    "epithelium": "nuclei_epithelium", "epithelial": "nuclei_epithelium", "nuclei_epithelium": "nuclei_epithelium",
    "apoptosis": "nuclei_apoptosis", "apoptotic": "nuclei_apoptosis", "nuclei_apoptosis": "nuclei_apoptosis",
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
        generic = {"nuclei", "nucleus", "cell", "cells", "annotation", "annotations", "tissue", "region", "roi"}
        for alias, canonical in aliases.items():
            if token.startswith(alias + "_"):
                remainder = token[len(alias) + 1 :].split("_")
                if remainder and all(part in generic for part in remainder):
                    return canonical
            if token.endswith("_" + alias):
                remainder = token[: -(len(alias) + 1)].split("_")
                if remainder and all(part in generic for part in remainder):
                    return canonical
    return None


def _rings(geometry: dict[str, Any]) -> list[np.ndarray]:
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if kind == "Polygon" and coordinates:
        polygons = [coordinates]
    elif kind == "MultiPolygon" and coordinates:
        polygons = coordinates
    else:
        return []

    rings: list[np.ndarray] = []
    for polygon in polygons:
        if not polygon:
            continue
        exterior = np.asarray(polygon[0], dtype=np.float32)
        if exterior.ndim != 2 or exterior.shape[1] < 2 or len(exterior) < 3:
            raise ValueError("Nucleus polygon must contain at least three XY points")
        if not np.isfinite(exterior[:, :2]).all():
            raise ValueError("Nucleus polygon contains NaN/Inf")
        rings.append(exterior[:, :2])
    return rings


def annotation_centroid(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or len(points) < 3:
        raise ValueError("Nucleus polygon needs at least three path points")
    # Match the public evaluator: every serialized path point contributes to the arithmetic mean.
    return points[:, :2].mean(axis=0, dtype=np.float64).astype(np.float32)


def load_nuclei_polygons(path: Path) -> list[tuple[int, np.ndarray]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    output: list[tuple[int, np.ndarray]] = []
    if payload.get("type") == "FeatureCollection":
        for feature_index, feature in enumerate(payload.get("features", [])):
            label = _label_from_mapping(feature.get("properties") or {}, NUCLEUS_ALIASES)
            if label is None:
                label = _label_from_mapping(feature, NUCLEUS_ALIASES)
            geometry = feature.get("geometry") or {}
            rings = _rings(geometry)
            if label is None and rings:
                raise ValueError(f"Unknown nucleus label in {path} feature {feature_index}")
            if label is None:
                continue
            if not rings:
                raise ValueError(
                    f"Nucleus feature {feature_index} in {path} has a recognized label but no valid Polygon/MultiPolygon geometry"
                )
            for ring in rings:
                if len(ring) >= 3:
                    output.append((CLASS_TO_ID[label], ring[:, :2].astype(np.float32, copy=False)))
        return output
    if payload.get("type") == "Multiple polygons":
        for polygon_index, polygon in enumerate(payload.get("polygons", [])):
            label = _label_from_mapping(polygon, NUCLEUS_ALIASES)
            points = np.asarray(polygon.get("path_points", []), dtype=np.float32)
            if points.size and (points.ndim != 2 or points.shape[1] < 2 or not np.isfinite(points[:, :2]).all()):
                raise ValueError(f"Invalid nucleus path_points in {path} polygon {polygon_index}")
            if points.ndim == 2 and len(points) >= 3 and label is None:
                raise ValueError(f"Unknown nucleus label in {path} polygon {polygon_index}")
            if label is not None:
                if points.ndim != 2 or len(points) < 3:
                    raise ValueError(f"Nucleus polygon {polygon_index} in {path} has a recognized label but fewer than three valid points")
                output.append((CLASS_TO_ID[label], points[:, :2]))
        return output
    raise ValueError(f"Unsupported nuclei annotation format: {path}")


def normalized_stem(path: Path | str) -> str:
    stem = Path(path).stem.lower()
    stem = re.sub(r"(?i)([_-]?(nuclei|nucleus|tissue|annotations?|annotation))+$", "", stem)
    return re.sub(r"[^a-z0-9]+", "_", stem).strip("_")


def infer_case_id(path: Path | str) -> str:
    """Return the independent split-group identifier for a public PUMA ROI.

    Public PUMA filenames are ROI identifiers such as
    ``training_set_primary_roi_001`` and do not expose a reliable patient/case
    mapping.  Therefore each ROI must remain its own CV group.

    IMPORTANT: do not strip the ``_roi_###`` suffix here.  Doing so collapses
    the complete public training set into only two groups
    (primary/metastatic), which makes 5-fold CV impossible.
    """
    return normalized_stem(path)


def infer_sample_type(path: Path | str) -> str:
    token = normalized_stem(path)
    if "metastatic" in token or "metastasis" in token:
        return "metastatic"
    if "primary" in token:
        return "primary"
    return "unknown"



