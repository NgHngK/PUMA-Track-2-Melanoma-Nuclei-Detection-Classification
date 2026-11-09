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
    """Preserve the original TIFF intensity/channel convention without Stage 1."""
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        image = np.asarray(tifffile.imread(path))
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if image.ndim == 3 and image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
            image = np.moveaxis(image, 0, -1)
        if image.ndim != 3 or image.shape[-1] < 3:
            raise ValueError(f"Expected RGB TIFF, got {image.shape}: {path}")
        image = image[..., :3]
        if image.shape[0] != image.shape[1]:
            raise ValueError(f"Expected a square RGB ROI, got {image.shape}: {path}")
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.integer):
                integer = image.astype(np.int64, copy=False)
                minimum, maximum = int(integer.min()), int(integer.max())
                if minimum < 0 or maximum > 255:
                    raise ValueError(
                        f"Integer TIFF dtype {image.dtype} has RGB range [{minimum}, {maximum}] outside 0..255: {path}. Refusing ambiguous dtype-based intensity rescaling; convert the ROI explicitly to 8-bit RGB first."
                    )
                image = integer.astype(np.uint8)
            elif np.issubdtype(image.dtype, np.floating):
                floating = image.astype(np.float32, copy=False)
                if not np.isfinite(floating).all():
                    raise ValueError(f"Floating TIFF contains NaN/Inf: {path}")
                minimum, maximum = float(floating.min()), float(floating.max())
                if minimum < 0.0:
                    raise ValueError(f"Floating TIFF has negative intensity values: {path}")
                if maximum <= 1.5:
                    floating = floating * 255.0
                elif maximum > 255.0:
                    raise ValueError(
                        f"Floating TIFF range [{minimum:.3f}, {maximum:.3f}] is not an understood 0..1 or 0..255 RGB range: {path}"
                    )
                image = np.rint(floating).clip(0, 255).astype(np.uint8)
            else:
                raise ValueError(f"Unsupported TIFF dtype {image.dtype}: {path}")
        return np.ascontiguousarray(image)
    from PIL import Image

    with Image.open(path) as image:
        return np.array(image.convert("RGB"), copy=True)


def read_features(folder):
    folder = Path(folder)
    values = {
        name: np.load(folder / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in ("roi", "xy", "label", "h112", "h224", "tier_raw", "tissue")
    }
    n = len(values["roi"])
    shapes = {
        "roi": (n,),
        "label": (n,),
        "xy": (n, 2),
        "h112": (n, 1536),
        "h224": (n, 1536),
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
    for key in ("xy", "h112", "h224", "tier_raw", "tissue"):
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
