from pathlib import Path
import numpy as np

class_names = (
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


def read_rgb_image(path):
    """Use the detector's TIFF intensity/channel convention in both stages."""
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        from stage1.target_core import _read_rgb

        with tifffile.TiffFile(path) as archive:
            shape = archive.series[0].shape
        size = (
            shape[1]
            if len(shape) == 3 and shape[0] in (3, 4) and shape[-1] not in (3, 4)
            else shape[0]
        )
        return _read_rgb(path, size=size)
    from PIL import Image

    with Image.open(path) as image:
        return np.array(image.convert("RGB"), copy=True)


def read_features(folder):
    folder = Path(folder)
    values = {
        name: np.load(folder / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in ("roi", "xy", "label", "h96", "h192", "tier_raw", "tissue")
    }
    n = len(values["roi"])
    shapes = {
        "roi": (n,),
        "label": (n,),
        "xy": (n, 2),
        "h96": (n, 1536),
        "h192": (n, 1536),
        "tier_raw": (n, 16),
        "tissue": (n, 20),
    }
    for key, shape in shapes.items():
        if values[key].shape != shape:
            raise ValueError(
                f"Invalid Stage 2 {key} shape: expected {shape}, got {values[key].shape}"
            )
    if values["roi"].dtype.kind not in "US" or np.any(values["roi"] == ""):
        raise ValueError("Stage 2 ROI names must be nonempty strings")
    labels = values["label"]
    if labels.dtype.kind not in "iu" or np.any((labels < 0) | (labels >= len(class_names))):
        raise ValueError("Stage 2 label values must be integers in 0..9")
    for key in ("xy", "h96", "h192", "tier_raw", "tissue"):
        for left in range(0, n, 8192):
            if not np.isfinite(values[key][left : left + 8192]).all():
                raise ValueError(f"Stage 2 {key} must contain finite values")
    return values


def normalize_tier(tier, training_rows):
    subset = np.asarray(tier[training_rows], dtype=np.float64)
    if len(subset) == 0 or not np.isfinite(subset).all():
        raise ValueError("Tier normalization needs nonempty, finite training rows")
    mean = subset.mean(0).astype(np.float32)
    std = np.maximum(subset.std(0).astype(np.float32), 1e-06)
    return (mean, std)


def split_rows(data, names):
    return np.flatnonzero(np.isin(data["roi"], list(names)))
