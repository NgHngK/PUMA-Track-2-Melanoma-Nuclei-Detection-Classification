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
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
NUM_CLASSES = 10
APPEARANCE_DIM = 1536
TIER_A_DIM = 16
INTERACTION_RANK = 8
LOCAL_FOV = 96
WIDE_FOV = 192
INPUT_SIZE = 224
PATCH_SIZE = 14
TOKEN_GRID = 16
NUM_PATCH_TOKENS = 256
NUM_REGISTER_TOKENS = 8
NUM_PREFIX_TOKENS = 1 + NUM_REGISTER_TOKENS
GAUSSIAN_SIGMA = 1.5
MATCH_RADIUS_PX = 15.0
SEEDS = (17, 29, 43)
TRAIN_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
GRAD_CLIP = 1.0
EPS_STD = 1e-6
ALLOWED_TRANSFER_ROLE = "train"
PROTECTED_ROLES = frozenset({"cal", "calibration", "validate", "validation", "test"})
# Locked natural-transfer decision/evaluation constants.
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20260921
MIN_ROI_DELTA = 0.003
MIN_POSITIVE_SEEDS = 2
MAX_MEAN_RECALL_DROP = 0.10
MAX_SEED_RECALL_DROP = 0.30
MIN_POSITIVE_ROIS_PER_CLASS = 5
PREFERRED_FOLDS = 5
MIN_FOLDS = 3
# Historical reports establish the scalar gate form but do not expose its exact initial value.
# 0.0 is retained only as an explicit exploratory reconstruction; strict runs must acknowledge it.
RECONSTRUCTED_CONTEXT_SCALE_INIT = 0.0
CONTEXT_INIT_PROVENANCE = "RECONSTRUCTED_UNVERIFIED_FROM_AVAILABLE_REPORTS"
