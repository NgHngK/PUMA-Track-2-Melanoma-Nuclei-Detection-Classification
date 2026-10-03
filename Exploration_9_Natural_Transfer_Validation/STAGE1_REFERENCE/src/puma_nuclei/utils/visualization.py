from __future__ import annotations

from math import ceil
from typing import Sequence

import numpy as np

from ..config import PipelineConfig
from ..constants import CLASS_SHORT_NAMES, SEMANTIC_MODE_HARD, SEMANTIC_MODE_RESOLVED, SEMANTIC_MODE_SAME_CLASS
from ..data.preprocess import load_shared_artifacts


DEFAULT_DASHBOARD_DPI = 300
DEFAULT_DASHBOARD_COLUMNS = 4
DEFAULT_CELL_SIZE_INCHES = 4.0


def _select_evenly(indices: np.ndarray, count: int, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    count = max(0, min(int(count), len(indices)))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    if count == len(indices):
        return np.sort(indices.copy())
    chosen = rng.choice(indices, size=count, replace=False)
    return np.sort(chosen.astype(np.int64, copy=False))


def _grid_shape(count: int, max_columns: int = DEFAULT_DASHBOARD_COLUMNS) -> tuple[int, int]:
    columns = max(1, min(int(max_columns), int(count)))
    rows = max(1, int(ceil(count / columns)))
    return rows, columns


def _semantic_mode_name(value: int) -> str:
    mapping = {
        SEMANTIC_MODE_HARD: "hard",
        SEMANTIC_MODE_SAME_CLASS: "same-class",
        SEMANTIC_MODE_RESOLVED: "resolved",
    }
    return mapping.get(int(value), "drop")


def _render_dashboard(
    *,
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    overlays: Sequence[dict[str, np.ndarray] | None],
    figure_title: str,
    max_columns: int,
    cell_size_inches: float,
    dpi: int,
    interpolation: str = "nearest",
) -> None:
    import matplotlib.pyplot as plt

    count = len(images)
    rows, columns = _grid_shape(count, max_columns=max_columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(cell_size_inches * columns, cell_size_inches * rows),
        squeeze=False,
        dpi=int(dpi),
    )
    for axis in axes.flat:
        axis.axis("off")
    # axes can contain trailing unused cells when the dashboard is not square;
    # truncating to the shortest display input is intentional.
    for axis, image, title, overlay in zip(axes.flat, images, titles, overlays):  # noqa: B905
        axis.imshow(np.asarray(image), interpolation=interpolation)
        if overlay:
            if "scatter_x" in overlay and "scatter_y" in overlay:
                scatter_kwargs = {
                    "s": overlay.get("scatter_size", 12.0),
                    "marker": str(overlay.get("scatter_marker", "o")),
                    "linewidths": float(overlay.get("linewidths", 0.7)),
                }
                if "facecolors" in overlay:
                    scatter_kwargs["facecolors"] = overlay["facecolors"]
                if "edgecolors" in overlay:
                    scatter_kwargs["edgecolors"] = overlay["edgecolors"]
                if "c" in overlay:
                    scatter_kwargs["c"] = overlay["c"]
                axis.scatter(overlay["scatter_x"], overlay["scatter_y"], **scatter_kwargs)
            if "cross_x" in overlay and "cross_y" in overlay:
                axis.scatter(
                    [float(overlay["cross_x"])],
                    [float(overlay["cross_y"])],
                    s=float(overlay.get("cross_size", 36.0)),
                    marker=str(overlay.get("cross_marker", "x")),
                    linewidths=float(overlay.get("cross_linewidths", 1.2)),
                    c=overlay.get("cross_color", None),
                )
        axis.set_title(title, fontsize=8, pad=3)
        axis.axis("off")
    fig.suptitle(figure_title, fontsize=14, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97), pad=0.6, w_pad=0.25, h_pad=0.8)
    plt.show()
    plt.close(fig)


def visualize_stage1_train_validation(
    config: PipelineConfig,
    *,
    fold: int,
    train_count: int = 16,
    validation_count: int = 16,
    seed: int | None = None,
    max_columns: int = DEFAULT_DASHBOARD_COLUMNS,
    dashboard_dpi: int = DEFAULT_DASHBOARD_DPI,
    cell_size_inches: float = DEFAULT_CELL_SIZE_INCHES,
) -> None:
    """Display sharp train/validation Stage-1 dashboards.

    Each split is rendered as one dashboard figure (typically a 4x4 grid for 16
    samples) rather than many separate figures. The function reads only shared
    preprocessed arrays and never changes training state.
    """

    artifacts = load_shared_artifacts(config)
    images = artifacts["images"]
    folds = np.asarray(artifacts["folds"], dtype=np.int64)
    offsets = np.asarray(artifacts["offsets"], dtype=np.int64)
    points = artifacts["point_targets"]
    stride = float(config.data.stage1_output_stride)
    random_seed = int(config.data.seed if seed is None else seed)

    train_indices = np.flatnonzero(folds != int(fold))
    validation_indices = np.flatnonzero(folds == int(fold))
    selected = {
        "Train": _select_evenly(train_indices, train_count, random_seed + 101 * int(fold)),
        "Validation": _select_evenly(validation_indices, validation_count, random_seed + 1009 + 101 * int(fold)),
    }

    for split_name, roi_indices in selected.items():
        if len(roi_indices) == 0:
            print(f"Stage1 visualization skipped: no {split_name.lower()} ROI for fold {fold}.")
            continue
        dashboard_images: list[np.ndarray] = []
        dashboard_titles: list[str] = []
        dashboard_overlays: list[dict[str, np.ndarray] | None] = []
        for roi in roi_indices.tolist():
            image = np.asarray(images[int(roi)], dtype=np.uint8)
            start, stop = int(offsets[int(roi)]), int(offsets[int(roi) + 1])
            roi_points = np.asarray(points[start:stop])
            overlay: dict[str, np.ndarray] | None = None
            if len(roi_points):
                x = (roi_points["grid_x"].astype(np.float32) + roi_points["offset_x"].astype(np.float32)) * stride
                y = (roi_points["grid_y"].astype(np.float32) + roi_points["offset_y"].astype(np.float32)) * stride
                overlay = {
                    "scatter_x": x,
                    "scatter_y": y,
                    "scatter_size": 8.0,
                    "scatter_marker": "o",
                    "facecolors": "none",
                    "linewidths": 0.7,
                }
            dashboard_images.append(image)
            dashboard_titles.append(f"{split_name} | ROI {roi} | nuclei={len(roi_points)}")
            dashboard_overlays.append(overlay)
        _render_dashboard(
            images=dashboard_images,
            titles=dashboard_titles,
            overlays=dashboard_overlays,
            figure_title=f"Stage 1 fold {fold} — {split_name} dashboard ({len(roi_indices)} samples)",
            max_columns=max_columns,
            cell_size_inches=cell_size_inches,
            dpi=dashboard_dpi,
            interpolation="nearest",
        )


def visualize_stage2_train_validation(
    config: PipelineConfig,
    *,
    fold: int,
    train_count: int = 16,
    validation_count: int = 16,
    crop_size: int | None = None,
    seed: int | None = None,
    max_columns: int = DEFAULT_DASHBOARD_COLUMNS,
    dashboard_dpi: int = DEFAULT_DASHBOARD_DPI,
    cell_size_inches: float = DEFAULT_CELL_SIZE_INCHES,
) -> None:
    """Display sharp proposal-centered Stage-2 dashboards.

    Each split is shown as a single dashboard figure with proposal markers. The
    titles may include GT-derived metadata for notebook inspection only; none of
    it enters Stage-2 model inputs.
    """

    # Lazy Stage-2 imports keep the archived Stage-1 visualization path runnable
    # without bundling the unrelated historical Stage-2 implementation.
    from ..stage2.crops import CropTransform
    from ..stage2.preprocess import load_stage2_feature_bank

    artifacts = load_shared_artifacts(config)
    images = artifacts["images"]
    bank = load_stage2_feature_bank(config)
    rows_bank = np.asarray(bank["rows"])
    size = int(config.stage2_preprocess.crop_size if crop_size is None else crop_size)
    random_seed = int(config.data.seed if seed is None else seed)

    train_indices = np.flatnonzero(rows_bank["fold"].astype(np.int64) != int(fold))
    validation_indices = np.flatnonzero(rows_bank["fold"].astype(np.int64) == int(fold))
    selected = {
        "Train": _select_evenly(train_indices, train_count, random_seed + 211 * int(fold)),
        "Validation": _select_evenly(validation_indices, validation_count, random_seed + 2003 + 211 * int(fold)),
    }

    for split_name, proposal_indices in selected.items():
        if len(proposal_indices) == 0:
            print(f"Stage2 visualization skipped: no {split_name.lower()} proposal for fold {fold}.")
            continue
        dashboard_images: list[np.ndarray] = []
        dashboard_titles: list[str] = []
        dashboard_overlays: list[dict[str, np.ndarray] | None] = []
        for proposal_index in proposal_indices.tolist():
            row = rows_bank[int(proposal_index)]
            roi = int(row["roi_index"])
            transform = CropTransform.from_center(float(row["x"]), float(row["y"]), size, np.asarray(images[roi]).shape)
            crop = transform.extract_uint8(np.asarray(images[roi]))
            local_prompt = transform.roi_to_local(np.asarray([[float(row["x"]), float(row["y"])]], dtype=np.float32))[0]
            class_id = int(row["semantic_class_target"])
            class_name = CLASS_SHORT_NAMES[class_id] if 0 <= class_id < len(CLASS_SHORT_NAMES) else "unresolved"
            dashboard_images.append(crop)
            dashboard_overlays.append(
                {
                    "cross_x": float(local_prompt[0]),
                    "cross_y": float(local_prompt[1]),
                    "cross_size": 32.0,
                    "cross_marker": "x",
                    "cross_linewidths": 1.2,
                }
            )
            dashboard_titles.append(
                f"{split_name} | ROI {roi} | {class_name} | {_semantic_mode_name(int(row['semantic_mode_target']))}\n"
                f"presence={int(row['nucleus_presence_target'])} viable={int(row['geometry_viability_target'])}"
            )
        _render_dashboard(
            images=dashboard_images,
            titles=dashboard_titles,
            overlays=dashboard_overlays,
            figure_title=f"Stage 2 fold {fold} — {split_name} dashboard ({len(proposal_indices)} samples)",
            max_columns=max_columns,
            cell_size_inches=cell_size_inches,
            dpi=dashboard_dpi,
            interpolation="nearest",
        )
