from __future__ import annotations

CLASSES = (
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
CLASS_TO_ID = {x: i for i, x in enumerate(CLASSES)}
SEEDS = (17, 29, 43)
ROLES = ("TRAIN", "CAL", "VALIDATE", "TEST")
FEATURE_DIM = 1536
BIO_DIM = 16
A5_RANK = 8
GAUSSIAN_SIGMA = 1.5
FOV_LOCAL = 96
FOV_WIDE = 192
V17_RADIUS = 15.0
TIERA_COLUMNS = (
    "H_mean",
    "H_std",
    "H_p10",
    "H_p50",
    "H_p90",
    "H_gradient_mean",
    "H_gradient_std",
    "H_abs_laplacian_mean",
    "H_entropy32",
    "H_center_minus_ring",
    "gray_center_minus_ring",
    "H_ring_std",
    "central_valid_fraction",
    "ring_valid_fraction",
    "H_robust_z",
    "H_ROI_percentile",
)
