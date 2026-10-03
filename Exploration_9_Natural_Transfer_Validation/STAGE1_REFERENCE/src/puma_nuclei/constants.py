from __future__ import annotations

import numpy as np

NUCLEUS_CLASSES: tuple[str, ...] = (
    "nuclei_tumor",
    "nuclei_lymphocyte",
    "nuclei_plasma_cell",
    "nuclei_histiocyte",
    "nuclei_melanophage",
    "nuclei_neutrophil",
    "nuclei_stroma",
    "nuclei_endothelium",
    "nuclei_epithelium",
    "nuclei_apoptosis",
)
CLASS_SHORT_NAMES: tuple[str, ...] = (
    "tumor", "lymphocyte", "plasma", "histiocyte", "melanophage",
    "neutrophil", "stroma", "endothelium", "epithelium", "apoptosis",
)
CLASS_TO_ID = {name: index for index, name in enumerate(NUCLEUS_CLASSES)}
NUMBER_OF_CLASSES = len(NUCLEUS_CLASSES)

ROI_SIZE = 1024
STAGE1_OUTPUT_STRIDE = 2
PUBLIC_MATCH_RADIUS_PIXELS = 15.0
BIOLOGICAL_SUPERVISION_RADIUS_PIXELS = 20.0
EVALUATION_CONTRACT_ID = "puma_public_track2_nuclei_class_first_greedy_dynamic_avg_2026"

UNI2_REPO_ID = "MahmoodLab/UNI2-h"
UNI2_HF_FILENAME = "pytorch_model.bin"
UNI2_EMBED_DIM = 1536
UNI2_VIEW_SIZES: tuple[int, ...] = (64, 128, 256)
UNI2_POOL_PARTS = 3
UNI2_POOLED_DIM = UNI2_EMBED_DIM * UNI2_POOL_PARTS
IDENTITY_POOL_COUNT = 5
IDENTITY_POOL_DIM = UNI2_EMBED_DIM

BIOLOGY_DIM = 20
RELIABILITY_DIM = 10
SPATIAL_DIM = 12
GEOMETRY_FEATURE_DIM = 11

SEMANTIC_MODE_NONE = 0
SEMANTIC_MODE_HARD = 1
SEMANTIC_MODE_SAME_CLASS = 2
SEMANTIC_MODE_RESOLVED = 3

INSTANCE_MODE_NONE = 0
INSTANCE_MODE_CLEAR = 1
INSTANCE_MODE_AMBIGUOUS = 2
INSTANCE_MODE_RESOLVED = 3

NEGATIVE_BACKGROUND = 0
NEGATIVE_NEAR_MISS = 1
NEGATIVE_HARD = 2
POSITIVE_REGION = -1

MASK_SUPERVISION_INSTANCE = 1
MASK_SUPERVISION_EMPTY = 2
MASK_SUPERVISION_AMBIGUOUS = 3

BIOMASK_NEGATIVE_NEAR = 0
BIOMASK_NEGATIVE_CROWDED = 1
BIOMASK_NEGATIVE_HIGH_SCORE = 2
BIOMASK_NEGATIVE_EASY = 3
BIOMASK_POSITIVE = -1
BIOMASK_NEGATIVE_STRATUM_COUNT = 4

ROI_MANIFEST_DTYPE = np.dtype(
    [
        ("roi_id", "U160"),
        ("image_file", "U512"),
        ("nuclei_file", "U512"),
        ("case_id", "U160"),
        ("sample_type", "U32"),
    ]
)

NUCLEUS_DTYPE = np.dtype(
    [
        ("roi_index", "i4"),
        ("nucleus_index", "i4"),
        ("x", "f4"),
        ("y", "f4"),
        ("class_id", "i2"),
        ("polygon_start", "i8"),
        ("polygon_length", "i4"),
    ]
)

RAW_PROPOSAL_DTYPE = np.dtype(
    [
        ("proposal_uid", "U96"),
        ("roi_index", "i4"),
        ("candidate_index", "i4"),
        ("x", "f4"),
        ("y", "f4"),
        ("heatmap_score", "f4"),
        ("quality", "f4"),
        ("uncertainty", "f4"),
        ("peak_sharpness", "f4"),
        ("matched_gt_index", "i4"),
        ("class_id", "i2"),
        ("is_reject", "u1"),
        ("fold", "i1"),
    ]
)

STAGE2_ROW_DTYPE = np.dtype(
    [
        ("proposal_uid", "U96"),
        ("roi_index", "i4"),
        ("x", "f4"),
        ("y", "f4"),
        ("fold", "i1"),
        ("semantic_class_target", "i2"),
        ("semantic_weight_target", "f4"),
        ("semantic_mode_target", "i1"),
        ("nucleus_presence_target", "u1"),
        ("geometry_viability_target", "u1"),
        ("resolved_gt_index_meta", "i4"),
        ("resolved_group_meta", "i4"),
        ("negative_stratum_target", "i1"),
        ("stage1_score_pred", "f4"),
        ("stage1_quality_pred", "f4"),
        ("stage1_uncertainty_pred", "f4"),
        ("peak_sharpness_pred", "f4"),
        ("presence_probability_pred", "f4"),
        ("mask_quality_pred", "f4"),
        ("center_offset_x_pred", "f4"),
        ("center_offset_y_pred", "f4"),
        ("border_touch_pred", "f4"),
        ("prompt_inside_mask_pred", "f4"),
        ("valid_mask_fraction_pred", "f4"),
        ("mask_area_pred", "f4"),
        ("center_agreement_pred", "f4"),
    ]
)

SUBMISSION_PREDICTION_DTYPE = np.dtype(
    [
        ("proposal_uid", "U96"),
        ("roi_index", "i4"),
        ("x", "f8"),
        ("y", "f8"),
        ("class_id", "i2"),
        ("score", "f8"),
        ("tie_break_rank", "i8"),
    ]
)

