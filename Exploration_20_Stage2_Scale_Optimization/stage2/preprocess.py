"""Prepare ground-truth Stage 2 features once for both cross-validation plans."""

import json
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from .annotations import load_nuclei_instances, normalized_stem
from .splits import expected_names
from .data import class_names, read_features
from .features import Extractor
from .nito import Nito
from .graph import build as build_graph, validate_graph
from .settings import FEATURE_REVISION, FOVS, graph_settings
from common.run_history import file_fingerprint, fingerprint


def _index_files(root: Path, suffixes: set[str]) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    files: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        key = normalized_stem(path)
        if key in files:
            raise ValueError(f"Ambiguous duplicate ROI basename {key}: {files[key]}, {path}")
        files[key] = path
    return files


def verify_alignment(annotations, names, stage2_dir):
    """Ensure cached ground-truth labels and centroids match raw annotations."""
    values = {
        key: np.load(Path(stage2_dir) / f"{key}.npy", mmap_mode="r", allow_pickle=False)
        for key in ("roi", "xy", "label")
    }
    start = 0
    for name in names:
        instances = load_nuclei_instances(annotations[name])
        end = start + len(instances)
        labels = np.asarray([int(label) for label, _ in instances], dtype=np.int64)
        xy = np.asarray([center for _, center in instances], dtype=np.float32).reshape(-1, 2)
        if not np.all(values["roi"][start:end] == name):
            raise ValueError(f"Stage 2 ROI order mismatch: {name}")
        if not np.array_equal(values["label"][start:end], labels):
            raise ValueError(f"Stage 2 class labels mismatch: {name}")
        if not np.array_equal(values["xy"][start:end], xy):
            raise ValueError(f"Stage 2 centroid coordinates mismatch: {name}")
        start = end
    if start != len(values["xy"]):
        raise ValueError("Stage 2 cache contains extra nuclei")
    return True


def _write_graph(output_dir):
    """Rebuild graph independently from expensive image feature extraction."""
    settings = graph_settings()
    graph = build_graph(
        np.load(output_dir / "xy.npy", allow_pickle=False),
        np.load(output_dir / "roi.npy", allow_pickle=False),
        radius=settings["radius"], k=settings["k"], sigma=settings["sigma"],
    )
    validate_graph(graph, len(graph["neighbors"]))
    pending = output_dir / "graph.pending.npz"
    try:
        np.savez(pending, **graph)
        pending.replace(output_dir / "graph.npz")
    finally:
        pending.unlink(missing_ok=True)
    return file_fingerprint(output_dir / "graph.npz")


def _ensure_graph(output_dir, manifest, values):
    """A 192px graph is not compatible; rebuild only graph on a valid 112/224 cache."""
    actual = output_dir / "graph.npz"
    correct = manifest.get("graph_settings") == graph_settings()
    correct = correct and actual.is_file()
    if correct:
        correct = manifest.get("graph_sha256") == file_fingerprint(actual)
    if correct:
        with np.load(actual, allow_pickle=False) as data:
            validate_graph(data, len(values["xy"]), values["roi"])
            correct = (
                data["neighbors"].shape[1] == graph_settings()["k"]
                and np.all(data["distance"][data["neighbors"] >= 0] <= graph_settings()["radius"] + 1e-4)
            )
    if not correct:
        print("Rebuilding Stage 2 graph at radius=224 without re-extracting UNI2/NiTo", flush=True)
        manifest["graph_sha256"] = _write_graph(output_dir)
        manifest["graph_settings"] = graph_settings()
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return actual


def prepare(images_dir, annotations_dir, output_dir, uni2_path, nito_dir, device="cuda"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = expected_names("B")
    images = _index_files(Path(images_dir), {".tif", ".tiff"})
    annotations = _index_files(Path(annotations_dir), {".geojson", ".json"})
    missing_images = sorted(set(names) - set(images))
    missing_anns = sorted(set(names) - set(annotations))
    if missing_images or missing_anns:
        raise FileNotFoundError(
            f"Missing {len(missing_images)} TIFFs and {len(missing_anns)} annotations. "
            f"TIFF examples={missing_images[:5]}, GeoJSON examples={missing_anns[:5]}"
        )

    image_sources = {name: file_fingerprint(images[name]) for name in names}
    annotation_sources = {name: file_fingerprint(annotations[name]) for name in names}
    source = {
        "images": image_sources,
        "annotations": annotation_sources,
        "uni2": file_fingerprint(Path(uni2_path)),
        "nito": {
            f"{group}{fold}": file_fingerprint(path) if path.is_file() else None
            for group in ("classifier", "Primary", "Metastatic")
            for fold in range(5)
            for path in [Path(nito_dir) / f"{group}{fold}" / "checkpoint_epoch1.pth"]
        },
        "fovs": FOVS,
        "revision": FEATURE_REVISION,
    }
    source_id = fingerprint(source)
    existing = output_dir / "manifest.json"
    if existing.exists():
        manifest = json.loads(existing.read_text())
        if (
            manifest.get("source_id") != source_id
            or manifest.get("feature_revision") != FEATURE_REVISION
            or tuple(manifest.get("class_order", ())) != class_names
            or tuple(manifest.get("fovs", ())) != FOVS
            or manifest.get("names") != names
            or manifest.get("rois") != len(names)
        ):
            raise ValueError("Stage 2 feature cache source/revision mismatch; create a fresh cache")
        values = read_features(output_dir)
        if manifest.get("nuclei") != len(values["xy"]):
            raise ValueError("Stage 2 feature cache nucleus count mismatch")
        progress = np.load(output_dir / "progress.npy", allow_pickle=False)
        if progress.shape != (len(names),) or progress.dtype.kind != "b" or not progress.all():
            raise ValueError("Stage 2 feature cache is incomplete")
        verify_alignment(annotations, names, output_dir)
        _ensure_graph(output_dir, manifest, values)
        print(f"Verified existing Stage 2 cache ({len(values['xy']):,} instances)", flush=True)
        return output_dir

    preparation = output_dir / "preparation.json"
    if preparation.exists():
        if json.loads(preparation.read_text()).get("source_id") != source_id:
            raise ValueError(
                "Cannot resume Stage 2 extraction after changing source images or encoder weights"
            )
    elif (output_dir / "progress.npy").exists():
        raise ValueError("Cannot verify provenance of partial Stage 2 cache; create a fresh cache")
    else:
        if any(output_dir.iterdir()):
            raise FileExistsError("Stage 2 output must be empty before a new preprocessing run")
        preparation.write_text(json.dumps({"source_id": source_id, "source": source}, indent=2))

    samples = []
    counts = {}
    for name in names:
        instances = load_nuclei_instances(annotations[name])
        counts[name] = len(instances)
        for class_id, center in instances:
            samples.append((name, float(center[0]), float(center[1]), int(class_id)))

    count = len(samples)
    if len(names) == 206 and count != 97429:
        raise ValueError(f"Expected 97429 PUMA nuclei; parsed {count} from {len(names)} ROIs")
    np.save(output_dir / "roi.npy", np.asarray([row[0] for row in samples], dtype="U64"))
    np.save(output_dir / "xy.npy", np.asarray([(row[1], row[2]) for row in samples], np.float32).reshape(-1, 2))
    np.save(output_dir / "label.npy", np.asarray([row[3] for row in samples], np.int64))
    verify_alignment(annotations, names, output_dir)

    dimensions = {f"h{FOVS[0]}": 1536, f"h{FOVS[1]}": 1536, "tier_raw": 16, "tissue": 20}
    progress_file = output_dir / "progress.npy"
    continuing = progress_file.exists()
    completed = (
        np.load(progress_file, allow_pickle=False)
        if continuing
        else np.zeros(len(names), dtype=bool)
    )
    if completed.shape != (len(names),) or completed.dtype.kind != "b":
        raise ValueError("Invalid Stage 2 extraction progress")

    outputs = {
        name: open_memmap(
            output_dir / f"{name}.npy",
            mode="r+" if continuing else "w+",
            dtype=np.float32,
            shape=(count, width),
        )
        for name, width in dimensions.items()
    }
    for name, output in outputs.items():
        if output.shape != (count, dimensions[name]) or output.dtype != np.float32:
            raise ValueError(f"Incompatible partial Stage 2 feature file: {name}")

    probability_dir = output_dir.parent / "nito_maps"
    Nito(nito_dir, device).prepare_maps({name: images[name] for name in names}, probability_dir)

    extractor = Extractor(uni2_path, nito_dir, device=device, probability_dir=probability_dir)
    left = 0
    for index, name in enumerate(names):
        end = left + counts[name]
        if not completed[index]:
            xy = np.asarray([(row[1], row[2]) for row in samples[left:end]], dtype=np.float32)
            values = extractor.extract(images[name], xy)
            for key, output in outputs.items():
                output[left:end] = values[key]
                output.flush()
            completed[index] = True
            np.save(progress_file, completed)
            print(f"UNI2 features {index + 1}/{len(names)}: {name} | {len(xy)} nuclei", flush=True)
        left = end
    extractor.close()

    graph_sha256 = _write_graph(output_dir)
    verify_alignment(annotations, names, output_dir)
    read_features(output_dir)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "rois": len(names),
                "names": names,
                "source_id": source_id,
                "encoder_sources": {"uni2": source["uni2"], "nito": source["nito"]},
                "image_sources": image_sources,
                "annotation_sources": annotation_sources,
                "feature_revision": FEATURE_REVISION,
                "fovs": FOVS,
                "graph_settings": graph_settings(),
                "graph_sha256": graph_sha256,
                "uni2_input_size": 224,
                "nuclei": count,
                "class_order": class_names,
                "centroid": "polygon_geometry_centroid",
                "feature_source": "E16 with UNI2 GPU mixed precision",
            },
            indent=2,
        )
    )
    print("Prepared Stage 2 GT features:", count, "nuclei", flush=True)
    return output_dir
