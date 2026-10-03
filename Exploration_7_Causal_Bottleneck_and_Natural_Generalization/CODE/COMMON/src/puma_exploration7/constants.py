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
NUM_CLASSES = len(CLASSES)
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
PROB_COLUMNS = tuple(f"p_{c}" for c in CLASSES)
EMBED_DIM = 1536
TIER_A_DIM = 16
UNI2_GRID_SIZE = 16
UNI2_SPATIAL_START = 9
UNI2_TOKEN_COUNT = 265
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# PUMA public evaluator order differs only for endothelium/epithelium.
V17_CLASSES = (
    "tumor",
    "lymphocyte",
    "plasma_cell",
    "histiocyte",
    "melanophage",
    "neutrophil",
    "stroma",
    "endothelium",
    "epithelium",
    "apoptosis",
)
CANONICAL_TO_V17 = tuple(V17_CLASSES.index(c) for c in CLASSES)
TAIL_DEFAULT = (2, 4, 5, 7, 8, 9)
