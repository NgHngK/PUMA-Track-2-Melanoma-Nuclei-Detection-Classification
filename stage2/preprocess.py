"""Prepare ground-truth features once for both cross-validation plans."""

import json
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from stage1.annotations import load_nuclei_instances
from stage1.constants import NUCLEUS_CLASSES

from stage1.dataset import Cache
from stage1.preprocess import _index_files
from .data import class_names, read_features
from .features import Extractor
from .nito import Nito
from .graph import build as build_graph, validate_graph
from .settings import FEATURE_REVISION
from common.run_history import file_fingerprint, fingerprint


def verify_images_against_stage1(cache, image_index):
    """Return image digests only when raw images agree with the detector cache.

    ``cache`` is a Stage1 Cache; ``image_index`` maps canonical ROI names to
    source paths. Legacy metadata is checked against the cached RGB pixels.
    """
    from stage1.target_core import _read_rgb

    expected = {
        row[0]: row[4]
        for row in cache.meta.get("source_file_metadata", [])
        if len(row) >= 5 and Path(row[1]).suffix.lower() in (".tif", ".tiff")
    }
    digests = {}
    for index, name in enumerate(cache.meta["names"]):
        if name not in image_index:
            raise FileNotFoundError(f"Missing Stage 1 source image: {name}")
        path = Path(image_index[name])
        digest = file_fingerprint(path)
        if name in expected:
            matches = digest == expected[name]
        else:
            matches = np.array_equal(_read_rgb(path, size=cache.size), cache.images[index])
        if not matches:
            raise ValueError(
                f"Raw image differs from Stage 1 cached source: {name}; create a new Stage 1 cache"
            )
        digests[name] = digest
    return digests


def verify_alignment(stage1_cache, stage2_dir):
    """Cross-check the cached Stage 2 nucleus instances against Stage 1 GT.

    Class IDs have different ordering for epithelium and endothelium, so
    compare canonical class names rather than raw integer IDs.
    """
    cache = Cache(stage1_cache)
    values = {
        key: np.load(Path(stage2_dir) / f"{key}.npy", mmap_mode="r", allow_pickle=False)
        for key in ("roi", "xy", "label")
    }

    if len(values["xy"]) != len(cache.gt):
        raise ValueError("Stage 1 and Stage 2 cache nuclei counts do not match")

    start = 0
    for name in cache.meta["names"]:
        stage1_gt = cache.gt_for(name)
        end = start + len(stage1_gt)
        expected_labels = np.array(
            [
                class_names.index(NUCLEUS_CLASSES[int(i)].removeprefix("nuclei_"))
                for i in stage1_gt["class_id"]
            ],
            dtype=np.int64,
        )
        xy = np.column_stack((stage1_gt["x"], stage1_gt["y"]))

        if not np.all(values["roi"][start:end] == name):
            raise ValueError(f"Stage 2 ROI order mismatch: {name}")
        if not np.array_equal(values["label"][start:end], expected_labels):
            raise ValueError(f"Stage 2 class labels mismatch: {name}")
        if not np.array_equal(values["xy"][start:end], xy):
            raise ValueError(f"Stage 2 centroid coordinates mismatch: {name}")
        start = end

    if start != len(values["xy"]):
        raise ValueError("Stage 2 cache contains extra nuclei")

    return True


def prepare(cache_dir, images_dir, annotations_dir, output_dir, uni2_path, nito_dir, device="cuda"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = Cache(cache_dir)
    names = cache.meta["names"]
    images = _index_files(Path(images_dir), {".tif", ".tiff"})
    annotations = _index_files(Path(annotations_dir), {".geojson", ".json"})
    image_sources = verify_images_against_stage1(cache, images)
    source = {
        "stage1_id": cache.meta["id"],
        "images": image_sources,
        "annotations": {name: file_fingerprint(annotations[name]) for name in names},
        "uni2": file_fingerprint(Path(uni2_path)),
        "nito": {
            f"{group}{fold}": file_fingerprint(path) if path.is_file() else None
            for group in ("classifier", "Primary", "Metastatic")
            for fold in range(5)
            for path in [Path(nito_dir) / f"{group}{fold}" / "checkpoint_epoch1.pth"]
        },
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
            or manifest.get("names") != names
            or manifest.get("rois") != len(names)
            or manifest.get("nuclei") != len(cache.gt)
        ):
            raise ValueError("Stage 2 feature cache source/revision mismatch; create a fresh cache")
        values = read_features(output_dir)
        if len(values["xy"]) != len(cache.gt):
            raise ValueError("Stage 2 feature cache nucleus count mismatch")
        with np.load(output_dir / "graph.npz", allow_pickle=False) as graph:
            validate_graph(graph, len(values["xy"]), values["roi"])
        progress = np.load(output_dir / "progress.npy", allow_pickle=False)
        if progress.shape != (len(names),) or progress.dtype.kind != "b" or not progress.all():
            raise ValueError("Stage 2 feature cache is incomplete")
        verify_alignment(cache_dir, output_dir)
        print(
            f"Verified and aligned existing Stage 2 cache ({len(cache.gt):,} instances)", flush=True
        )
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
        preparation.write_text(json.dumps({"source_id": source_id, "source": source}, indent=2))
    labels = {name: i for i, name in enumerate(class_names)}

    samples = []
    counts = {}

    for name in names:
        instances = load_nuclei_instances(annotations[name])
        counts[name] = len(instances)
        for class_id, center in instances:
            label = NUCLEUS_CLASSES[int(class_id)].removeprefix("nuclei_")
            samples.append((name, float(center[0]), float(center[1]), labels[label]))

    count = len(samples)
    if len(names) == 206 and count != 97429:
        raise ValueError(f"Expected 97429 PUMA nuclei; parsed {count} from {len(names)} ROIs")
    np.save(output_dir / "roi.npy", np.asarray([row[0] for row in samples], dtype="U64"))
    np.save(
        output_dir / "xy.npy",
        np.asarray([(row[1], row[2]) for row in samples], np.float32).reshape(-1, 2),
    )
    np.save(output_dir / "label.npy", np.asarray([row[3] for row in samples], np.int64))

    verify_alignment(cache_dir, output_dir)

    dimensions = {"h96": 1536, "h192": 1536, "tier_raw": 16, "tissue": 20}
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

    probability_dir = Path(cache_dir).parent / "nito_maps"
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

    graph = build_graph(np.load(output_dir / "xy.npy"), np.load(output_dir / "roi.npy"))
    np.savez(output_dir / "graph.npz", **graph)

    verify_alignment(cache_dir, output_dir)
    read_features(output_dir)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "rois": len(names),
                "names": names,
                "source_id": source_id,
                "encoder_sources": {"uni2": source["uni2"], "nito": source["nito"]},
                "image_sources": image_sources,
                "feature_revision": FEATURE_REVISION,
                "nuclei": count,
                "class_order": class_names,
                "centroid": "polygon_geometry_centroid",
                "feature_source": "E16 with UNI2 GPU mixed precision",
                "source_stage1_id": cache.meta["id"],
            },
            indent=2,
        )
    )

    verify_alignment(cache_dir, output_dir)
    print("Prepared Stage 2 GT features:", count, "nuclei", flush=True)
    return output_dir
