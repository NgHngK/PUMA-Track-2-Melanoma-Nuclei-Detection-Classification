"""Clean OOF evaluation. No proposals, features, predictions or JSON saved.

Produces ONE file: results/evaluation/history.csv (16 fold rows + OOF summaries).
Uses class-aware official PUMA per-ROI class F1 aggregation, not tissue Dice.
"""

from __future__ import annotations
import argparse
import csv
import gc
from pathlib import Path
import numpy as np
import torch
from common.run_history import SCHEMA
from stage1.dataset import Cache
from stage1.splits import load_splits
from stage1.models.build import build_model
from stage1.config import DETECTOR_REVISION
from stage1.evaluate import collect_candidates, choose_tile_batch_size
from stage1.decode import decode_stage1_candidates
from stage1.preprocess import _index_files
from stage2.features import Extractor
from stage2.inference import Classifier
from stage2.data import class_names
from stage2.metrics import roi_scores
from stage2.settings import FEATURE_REVISION, MODEL_REVISION
from stage2.nito import Nito
from stage2.preprocess import verify_images_against_stage1
from common.run_history import file_fingerprint

architectures = ("convnext_tiny", "convnext_base")
plans = ("A", "B")
METRICS = ("accuracy", "precision", "recall", "f1", "macro_f1_nuclei", "roi_f1")
COLUMNS = [
    "plan",
    "architecture",
    "fold",
    "num_rois",
    "tp",
    "fp",
    "fn",
    *METRICS,
    *[f"{c}_{m}" for c in class_names for m in ("accuracy", "precision", "recall", "f1")],
    *[f"{c}_official_roi_f1" for c in class_names],
]


def row_metrics(metrics):
    row = {key: metrics[key] for key in ("tp", "fp", "fn", *METRICS)}
    for name in class_names:
        c = metrics["per_class"][name]
        row[f"{name}_accuracy"] = c["tp"] / max(
            1, c["tp"] + c["fp"] + c["fn"]
        )  # detection Jaccard/IoU
        row.update({f"{name}_{m}": c[m] for m in ("precision", "recall", "f1")})
        row[f"{name}_official_roi_f1"] = metrics["f1_nuclei"]["nuclei_" + name]
    return row


def _evaluate_one(cache, images, results, uni2, nito, plan, arch, fold, device, probability_dir):
    det_path = results / "stage1" / f"plan_{plan}" / arch / f"fold_{fold}" / "best.pt"
    cl_path = results / "stage2" / f"plan_{plan}" / f"fold_{fold}" / "best.pt"
    det = torch.load(det_path, map_location="cpu", weights_only=False)
    if det.get("config", {}).get("cache_id") != cache.meta["id"]:
        raise ValueError(f"Stage 1 checkpoint uses old preprocessing: {det_path}")
    config = det.get("config", {})
    if (
        det.get("schema") != SCHEMA
        or config.get("detector_revision") != DETECTOR_REVISION
        or config.get("arch") != arch
        or config.get("plan") != plan
        or config.get("fold") != fold
    ):
        raise ValueError(f"Incompatible Stage 1 checkpoint identity/geometry: {det_path}")
    split = load_splits(plan)[fold]
    if set(config.get("train", ())) != set(split["train"]) or set(config.get("val", ())) != set(
        split["val"]
    ):
        raise ValueError(f"Stage 1 checkpoint split differs from evaluation split: {det_path}")

    setting = det["operating_point"]
    if setting.get("selection_set") != "train":
        raise ValueError(
            f"Stage 1 operating point was not calibrated exclusively on train: {det_path}"
        )
    if "threshold" not in setting or "suppression_radius" not in setting:
        raise ValueError(f"No Stage 1 NMS parameters inside best.pt: {det_path}")
    model = build_model(arch, pretrained=False).to(device).eval()
    model.load_state_dict(det["model"], strict=True)
    model.pretrained = bool(det["config"]["pretrained"])
    tile = int(config["tile_size"])
    overlap = int(config["overlap"])
    amp = bool(config.get("amp", True))
    batch = choose_tile_batch_size(model, device, tile, amp=amp)
    names = load_splits(plan)[fold]["val"]
    proposals = {}
    for i, name in enumerate(names, 1):
        thresholds = det["config"]["train_thresholds"]
        candidates = collect_candidates(
            model,
            cache,
            name,
            tile=tile,
            overlap=overlap,
            device=device,
            amp=amp,
            batch_size=batch,
            minimum_proposal_score=min(thresholds),
        )
        predicted = decode_stage1_candidates(
            candidates,
            threshold=setting["threshold"],
            suppression_radius_pixels=setting["suppression_radius"],
            max_candidates=4000,
        )
        xy = (
            np.column_stack((predicted["x"], predicted["y"])).astype(np.float32)
            if len(predicted)
            else np.empty((0, 2), np.float32)
        )
        proposals[name] = (xy, predicted["heatmap_score"].copy())
        if i % 10 == 0:
            print(f"Detect {plan}/{arch}/fold{fold}: {i}/{len(names)}", flush=True)
    del model, det
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    extractor = Extractor(uni2, nito, device=device, probability_dir=probability_dir)
    classifier = Classifier(cl_path, device=device)
    gt = {"roi": [], "xy": [], "label": []}
    pred = {"roi": [], "xy": [], "label": [], "score": []}
    for i, name in enumerate(names, 1):
        xy, score = proposals[name]
        values = extractor.extract(images[name], xy)
        proba = classifier.predict(values)
        labels = proba.argmax(1) if len(proba) else np.empty(0, dtype=int)
        confidence = (proba.max(1) * score).clip(0, 1) if len(proba) else np.empty(0, dtype=float)
        truth = cache.gt_for(name)
        mapped = truth["class_id"].copy()
        # Stage 1 class ids swap endothelium/epithelium vs Stage 2 class order.
        mapped[mapped == 7] = 11
        mapped[mapped == 8] = 7
        mapped[mapped == 11] = 8
        gt["roi"].extend([name] * len(truth))
        gt["xy"].extend(np.column_stack((truth["x"], truth["y"])))
        gt["label"].extend(mapped)
        pred["roi"].extend([name] * len(labels))
        pred["xy"].extend(xy)
        pred["label"].extend(labels)
        pred["score"].extend(confidence)
        if i % 10 == 0:
            print(f"Classify {plan}/{arch}/fold{fold}: {i}/{len(names)}", flush=True)
    extractor.close()
    del extractor, classifier
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    for key in gt:
        gt[key] = np.asarray(gt[key]).reshape(-1, 2) if key == "xy" else np.asarray(gt[key])
    for key in pred:
        pred[key] = np.asarray(pred[key]).reshape(-1, 2) if key == "xy" else np.asarray(pred[key])
    return roi_scores(gt, pred, roi_names=names)


def _summarize(rows, reports, plan, arch):
    items = [reports[(plan, arch, f)] for f in range(4)]
    detail = {}
    for item in items:
        for roi, classes in item["per_roi_class"].items():
            if roi in detail:
                raise ValueError(f"Duplicate OOF validation ROI: {roi}")
            detail[roi] = classes
    expected = set().union(*(set(x["val"]) for x in load_splits(plan).values()))
    if set(detail) != expected:
        raise ValueError(
            f"OOF ROI coverage mismatch for {plan}/{arch}: {len(detail)} vs {len(expected)}"
        )
    f1s = {
        name: float(np.mean([classes[name]["f1"] for classes in detail.values()]))
        for name in class_names
    }
    observed = set().union(
        *(set(item.get("observed_nuclei_classes", class_names)) for item in items)
    )
    macro = float(np.mean([f1s[name] for name in observed])) if observed else 0.0
    result = {
        "plan": plan,
        "architecture": arch,
        "fold": "OOF",
        "num_rois": len(detail),
        "macro_f1_nuclei": macro,
    }
    result.update({f"{name}_official_roi_f1": value for name, value in f1s.items()})
    rows.append(result)
    for kind, agg in [
        ("mean", lambda x: float(np.mean(x))),
        ("std", lambda x: float(np.std(x, ddof=1))),
    ]:
        x = {
            "plan": plan,
            "architecture": arch,
            "fold": kind,
            "num_rois": sum(item["num_rois"] for item in items),
        }
        for key in METRICS:
            x[key] = agg([item[key] for item in items])
        rows.append(x)
    print(f"OOF {plan}/{arch}: macro F1 {macro:.6f}, ROIs={len(detail)}", flush=True)


def evaluate_all(cache_dir, images_dir, results_dir, uni2_path, nito_dir, probability_dir=None):
    cache = Cache(cache_dir)
    if cache.meta.get("target_version") != "puma_one_feature_one_nucleus_geometry_centroid_v2":
        raise ValueError("Refusing evaluation using stale 97,614-instance Stage 1 cache")
    results = Path(results_dir)
    missing = []
    for plan in plans:
        for fold in range(4):
            path = results / "stage2" / f"plan_{plan}" / f"fold_{fold}" / "best.pt"
            if not path.is_file():
                missing.append(str(path))
            for arch in architectures:
                path = results / "stage1" / f"plan_{plan}" / arch / f"fold_{fold}" / "best.pt"
                if not path.is_file():
                    missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            f"Cannot evaluate: {len(missing)} missing best.pt. First: {missing[:5]}"
        )
    for plan in plans:
        for fold in range(4):
            for stage, arch in [("stage2", None), *[("stage1", a) for a in architectures]]:
                folder = (
                    results / stage / f"plan_{plan}" / f"fold_{fold}"
                    if arch is None
                    else results / stage / f"plan_{plan}" / arch / f"fold_{fold}"
                )
                # Evaluate the available best checkpoint regardless of epoch count.
                # No history.csv / completed-training requirement.
                state = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
                if state.get("schema") != SCHEMA:
                    raise ValueError(
                        f"Checkpoint was not trained with current clean release: {folder}"
                    )

                if (
                    stage == "stage1"
                    and state.get("operating_point", {}).get("selection_set") != "train"
                ):
                    raise ValueError(f"Stage 1 threshold/NMS was not selected on train: {folder}")

                if stage == "stage2":
                    if (
                        state.get("plan") != plan
                        or state.get("fold") != fold
                        or state.get("source_stage1_id") != cache.meta["id"]
                    ):
                        raise ValueError(f"Stage 2 fold/cache mismatch: {folder}")
                    if (
                        state.get("feature_revision") != FEATURE_REVISION
                        or state.get("model_revision") != MODEL_REVISION
                        or tuple(state.get("class_names", ())) != class_names
                    ):
                        raise ValueError(f"Stage 2 feature/model/class revision mismatch: {folder}")
                    split = load_splits(plan)[fold]
                    if set(state.get("train_rois", ())) != set(split["train"]) or set(
                        state.get("val_rois", ())
                    ) != set(split["val"]):
                        raise ValueError(
                            f"Stage 2 checkpoint split differs from evaluation split: {folder}"
                        )

                del state
    images = _index_files(Path(images_dir), {".tif", ".tiff"})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probability_dir = (
        Path(probability_dir) if probability_dir else Path(cache_dir).parent / "nito_maps"
    )
    selected_images = {name: images[name] for name in cache.meta["names"]}
    image_sources = verify_images_against_stage1(cache, selected_images)
    encoder_sources = {
        "uni2": file_fingerprint(Path(uni2_path)),
        "nito": {
            f"{group}{fold}": file_fingerprint(path)
            for group in ("classifier", "Primary", "Metastatic")
            for fold in range(5)
            for path in [Path(nito_dir) / f"{group}{fold}" / "checkpoint_epoch1.pth"]
        },
    }
    for plan in plans:
        for fold in range(4):
            classifier_state = torch.load(
                results / "stage2" / f"plan_{plan}" / f"fold_{fold}" / "best.pt",
                map_location="cpu",
                weights_only=False,
            )
            if classifier_state.get("encoder_sources") != encoder_sources:
                raise ValueError(
                    f"Stage 2 encoder sources changed after training: {plan}/fold{fold}"
                )
            if classifier_state.get("image_sources") != image_sources:
                raise ValueError(f"Stage 2 source images changed after training: {plan}/fold{fold}")
    del classifier_state
    Nito(nito_dir, device).prepare_maps(selected_images, probability_dir)
    rows = []
    reports = {}
    for plan in plans:
        for fold in range(4):
            for arch in architectures:
                print(f"EVAL {plan}/{arch}/fold{fold}", flush=True)
                r = _evaluate_one(
                    cache,
                    images,
                    results,
                    uni2_path,
                    nito_dir,
                    plan,
                    arch,
                    fold,
                    device,
                    probability_dir,
                )
                reports[(plan, arch, fold)] = r
                rows.append(
                    {
                        "plan": plan,
                        "architecture": arch,
                        "fold": fold,
                        "num_rois": r["num_rois"],
                        **row_metrics(r),
                    }
                )
    for plan in plans:
        for arch in architectures:
            _summarize(rows, reports, plan, arch)
    out = results / "evaluation" / "history.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Evaluation saved ONE file: {out}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    for key in ("cache", "images", "results", "uni2", "nito"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--probability-dir")
    a = parser.parse_args()
    evaluate_all(a.cache, a.images, a.results, a.uni2, a.nito, probability_dir=a.probability_dir)


if __name__ == "__main__":
    main()
