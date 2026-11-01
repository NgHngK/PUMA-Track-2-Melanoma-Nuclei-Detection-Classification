import numpy as np
from scipy.spatial import cKDTree
from .data import class_names


def classification_scores(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    if actual.ndim != 1 or predicted.shape != actual.shape:
        raise ValueError("Classification labels must have matching one-dimensional shapes")
    for labels in (actual, predicted):
        if len(labels) and (
            labels.dtype.kind not in "iu" or np.any((labels < 0) | (labels >= len(class_names)))
        ):
            raise ValueError("Classification labels must be integers in 0..9")
    correct = int(np.sum(actual == predicted))
    per_class = {}
    for i, name in enumerate(class_names):
        tp = int(np.sum((actual == i) & (predicted == i)))
        fp = int(np.sum((actual != i) & (predicted == i)))
        fn = int(np.sum((actual == i) & (predicted != i)))
        per_class[name] = _score(tp, fp, fn)
        tn = len(actual) - tp - fp - fn
        per_class[name]["accuracy"] = (tp + tn) / len(actual) if len(actual) else 0.0
        per_class[name]["support"] = int(np.count_nonzero(actual == i))
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))
    result = _score(correct, len(predicted) - correct, len(actual) - correct)
    result["accuracy"] = correct / len(actual) if len(actual) else 0.0
    result["macro_f1"] = macro_f1
    result["macro_precision"] = float(np.mean([v["precision"] for v in per_class.values()]))
    result["macro_recall"] = float(np.mean([v["recall"] for v in per_class.values()]))
    result["per_class"] = per_class
    return result


def _score(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), precision=precision, recall=recall, f1=f1)


def roi_scores(gt, predicted, radius=15.0, roi_names=None):
    """Class-aware, GT-order greedy PUMA matching, accelerated by spatial search.

    Reference: tueimage/PUMA-challenge-eval-track2, eval_nuclei.py and evaluate.py.
    Coincident predictions are consumed by selected index; the reference instead
    removes the first equal centroid, an edge case excluded by Stage 1 NMS.
    """
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Matching radius must be finite and positive")
    gt = {key: np.asarray(value) for key, value in gt.items()}
    predicted = {key: np.asarray(value) for key, value in predicted.items()}
    # KDTree and final strict-radius check must use the same float64 geometry.
    gt["xy"] = np.asarray(gt["xy"], dtype=np.float64)
    predicted["xy"] = np.asarray(predicted["xy"], dtype=np.float64)
    for values in (gt, predicted):
        n = len(values["roi"])
        if (
            values["roi"].shape != (n,)
            or values["label"].shape != (n,)
            or values["xy"].shape != (n, 2)
            or not np.isfinite(values["xy"]).all()
        ):
            raise ValueError(
                "ROI metrics require aligned ROI/label arrays and finite N x 2 coordinates"
            )
        labels = values["label"]
        if len(labels) and (
            labels.dtype.kind not in "iu" or np.any((labels < 0) | (labels >= len(class_names)))
        ):
            raise ValueError("ROI labels must be integers in 0..9")
    if "score" not in predicted:
        predicted["score"] = np.ones(len(predicted["roi"]), dtype=np.float32)
    if (
        predicted["score"].shape != predicted["roi"].shape
        or not np.isfinite(predicted["score"]).all()
    ):
        raise ValueError("Prediction scores must be aligned and finite")
    truth_rois = np.asarray(gt["roi"]).astype(str)
    pred_rois = np.asarray(predicted["roi"]).astype(str)
    rois = sorted(set(truth_rois) | set(pred_rois) | set([] if roi_names is None else roi_names))

    truth_groups = {roi: np.flatnonzero(truth_rois == roi) for roi in rois}
    pred_groups = {roi: np.flatnonzero(pred_rois == roi) for roi in rois}
    totals = [0, 0, 0]
    counts = np.zeros((10, 3), dtype=np.int64)
    each_roi = {}
    each_roi_class = {}  # Full 10-class metrics per ROI, including absent classes as F1=0

    for roi in rois:
        truth_idx = truth_groups[roi]
        pred_idx = pred_groups[roi]
        matched = set()
        tp = 0
        roi_counts = np.zeros((len(class_names), 3), dtype=np.int64)
        trees = {}

        for label in range(10):
            class_idx = pred_idx[np.asarray(predicted["label"])[pred_idx] == label]
            if len(class_idx):
                trees[label] = (class_idx, cKDTree(predicted["xy"][class_idx]))

        for gi in truth_idx:
            label = int(gt["label"][gi])
            candidates = []

            if label in trees:
                indices, tree = trees[label]
                for local_idx in tree.query_ball_point(gt["xy"][gi], r=radius):
                    pi = int(indices[local_idx])
                    if pi in matched:
                        continue
                    distance = float(np.linalg.norm(gt["xy"][gi] - predicted["xy"][pi]))
                    if distance < radius:
                        candidates.append((-float(predicted["score"][pi]), distance, pi))

            if candidates:
                _, _, pi = min(candidates)
                matched.add(pi)
                tp += 1
                counts[label, 0] += 1
                roi_counts[label, 0] += 1
            else:
                counts[label, 2] += 1
                roi_counts[label, 2] += 1

        fp = len(pred_idx) - tp
        fn = len(truth_idx) - tp
        for pi in pred_idx:
            if int(pi) not in matched:
                counts[int(predicted["label"][pi]), 1] += 1
                roi_counts[int(predicted["label"][pi]), 1] += 1

        totals[0] += tp
        totals[1] += fp
        totals[2] += fn
        each_roi[roi] = _score(tp, fp, fn)
        each_roi_class[roi] = {
            name: _score(*map(int, roi_counts[i])) for i, name in enumerate(class_names)
        }

    result = _score(*totals)
    result["accuracy"] = result["tp"] / max(1, sum(totals))
    result["roi_f1"] = float(np.mean([r["f1"] for r in each_roi.values()])) if each_roi else 0.0
    result["per_class"] = {name: _score(*map(int, counts[i])) for i, name in enumerate(class_names)}
    result["per_roi"] = each_roi
    result["per_roi_class"] = each_roi_class
    # evaluate.py averages each observed class over ALL ROIs (including zero
    # contributions where it is absent), then averages those observed classes.
    # Keep ten-class detail even for synthetic subsets missing some classes.
    n_roi = len(each_roi_class)
    result["f1_nuclei"] = {
        "nuclei_" + name: float(
            sum(each_roi_class[roi][name]["f1"] for roi in each_roi_class) / n_roi
        )
        if n_roi
        else 0.0
        for name in class_names
    }
    observed = sorted(set(gt["label"]) | set(predicted["label"]))
    result["observed_nuclei_classes"] = [class_names[int(i)] for i in observed]
    result["macro_f1_nuclei"] = (
        float(np.mean([result["f1_nuclei"]["nuclei_" + class_names[int(i)]] for i in observed]))
        if observed
        else 0.0
    )
    result["macro_f1_nuclei_average"] = result["macro_f1_nuclei"]
    # Retain clearly named research diagnostics; these are NOT leaderboard macro F1.
    result["summed_macro_f1_nuclei"] = float(
        np.mean([v["f1"] for v in result["per_class"].values()])
    )
    result["macro_f1"] = result["macro_f1_nuclei"]
    result["num_rois"] = n_roi
    return result
