from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .constants import STAGE1_OUTPUT_STRIDE, UNI2_VIEW_SIZES


@dataclass(slots=True)
class ProjectPaths:
    # Notebooks set this explicitly. The default keeps plain Python usage simple.
    project_root: str = "."
    image_directory: str = "Dataset/01_training_dataset_tif_ROIs"
    nuclei_annotation_directory: str = "Dataset/01_training_dataset_geojson_nuclei"
    uni2_checkpoint: str = "PUMA_pretrained_checkpoints/uni2_h_model.bin"
    shared_preprocessing_directory: str = "PUMA_preprocessed"
    stage1_output_directory: str = "PUMA_stage1_training_outputs"
    stage2_preprocessing_directory: str = "PUMA_stage2_preprocessed"
    stage2_output_directory: str = "PUMA_stage2_training_outputs"

    @property
    def root(self) -> Path:
        return Path(self.project_root).expanduser().resolve()

    def resolve(self, value: str | os.PathLike[str]) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    @property
    def images(self) -> Path:
        return self.resolve(self.image_directory)

    @property
    def nuclei_annotations(self) -> Path:
        return self.resolve(self.nuclei_annotation_directory)

    @property
    def uni2(self) -> Path:
        return self.resolve(self.uni2_checkpoint)

    @property
    def preprocessed(self) -> Path:
        return self.resolve(self.shared_preprocessing_directory)

    @property
    def stage1_outputs(self) -> Path:
        return self.resolve(self.stage1_output_directory)

    @property
    def stage2_preprocessed(self) -> Path:
        return self.resolve(self.stage2_preprocessing_directory)

    @property
    def stage2_outputs(self) -> Path:
        return self.resolve(self.stage2_output_directory)


@dataclass(slots=True)
class DataConfig:
    number_of_folds: int = 5
    seed: int = 42
    cpu_workers: int = -1
    fold_search_restarts: int = 96
    heatmap_sigma_grid_units: float = 1.7
    stage1_output_stride: int = STAGE1_OUTPUT_STRIDE

    @property
    def resolved_workers(self) -> int:
        cores = os.cpu_count() or 2
        return cores if self.cpu_workers < 0 else min(self.cpu_workers, cores)


@dataclass(slots=True)
class Stage1Config:
    epochs: int = 100
    # 24-GB target: two physical micro-batches of 8 make one optimizer batch of 16.
    micro_batch_size: int = 8
    effective_batch_size: int = 16
    base_channels: int = 48
    fpn_channels: int = 96
    # Conservative batch-scaled LR: higher than the original batch-4 run without
    # applying the overly aggressive full linear-scaling jump to 1e-3.
    learning_rate: float = 3.5e-4
    backbone_learning_rate_scale: float = 1.0
    minimum_learning_rate: float = 2.0e-6
    weight_decay: float = 1.0e-4
    warmup_epochs: int = 5
    gradient_clip_norm: float = 1.0
    ema_decay: float = 0.9995

    # Loss mix: localization is already strong in the baseline, so more capacity
    # is allocated to proposal confidence and hard false-positive suppression.
    heatmap_loss_weight: float = 0.55
    offset_loss_weight: float = 0.18
    quality_loss_weight: float = 0.12
    uncertainty_loss_weight: float = 0.05
    hard_negative_loss_weight: float = 0.10
    hard_negative_topk_per_image: int = 256
    hard_negative_start_epoch: int = 5
    hard_negative_ramp_epochs: int = 10

    # D4 geometry is safe for histology orientation. With stride=2, exact flips
    # can move the sparse offset by +0.5 grid units, hence the 2.5 output range.
    use_d4_augmentation: bool = True
    d4_augmentation_probability: float = 0.875
    photometric_augmentation: bool = True
    brightness_range: tuple[float, float] = (0.92, 1.08)
    contrast_range: tuple[float, float] = (0.90, 1.10)
    gamma_range: tuple[float, float] = (0.90, 1.10)
    channel_scale_range: tuple[float, float] = (0.96, 1.04)

    # Decoder score = heatmap * quality**quality_power. Uncertainty is retained as
    # a feature but not trusted for rejection until it is empirically calibrated.
    proposal_quality_power: float = 1.0
    proposal_uncertainty_penalty: float = 0.0
    validation_decode_threshold: float = 0.03  # legacy single-point fallback only
    validation_thresholds: tuple[float, ...] = (
        0.005, 0.010, 0.020, 0.030, 0.040, 0.050, 0.075, 0.100,
        0.125, 0.150, 0.200, 0.250, 0.300, 0.400, 0.500, 0.650, 0.800, 0.900, 0.950,
    )
    validation_suppression_radii_pixels: tuple[float, ...] = (3.0, 4.0, 5.0, 6.0, 8.0)
    validation_min_recall: float = 0.93
    local_max_radius: int = 2
    suppression_radius_pixels: float = 3.0
    max_candidates_per_roi: int = 4000
    offset_range_grid_units: float = 2.5
    quality_target_sigma_grid_units: float = 0.75
    checkpoint_every_epochs: int = 1

    @property
    def accumulation_steps(self) -> int:
        if self.effective_batch_size % self.micro_batch_size:
            raise ValueError("effective_batch_size must be divisible by micro_batch_size")
        return self.effective_batch_size // self.micro_batch_size


@dataclass(slots=True)
class Uni2LoRAConfig:
    """LoRA architecture injected directly into UNI2-h by Stage-2 training.

    v16.4 no longer has a standalone UNI2 fine-tuning stage.  These fields only
    describe the low-rank modules created inside :class:`IntegratedStage2Model`.
    The UNI2-h base stays frozen; LoRA is frozen in Phase 1A, trainable in
    Phases 1B/2, and frozen again in Phase 3.
    """

    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    last_blocks: int = 4
    target_modules: tuple[str, ...] = ("attn.qkv", "attn.proj")


@dataclass(slots=True)
class Stage2PreprocessConfig:
    # Stage-2 preprocessing builds fixed OOF evidence only. UNI2-h is never run
    # here; all 64/128/256 representations are materialized online in Stage 2.
    selected_folds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    association_radius_pixels: float = 15.0
    biological_supervision_radius_pixels: float = 20.0
    crop_size: int = 96
    identity_view_size: int = 64
    uni2_view_sizes: tuple[int, ...] = UNI2_VIEW_SIZES
    # Zero enables a one-time CUDA probe that selects the largest safe batch.
    # A positive value remains available for reproducible/manual overrides.
    uni2_chunk_size: int = 2048
    uni2_token_map_batch_size: int = 256
    feature_dtype: str = "float16"
    biomask_epochs: int = 30
    biomask_batch_size: int = 128
    biomask_learning_rate: float = 3.0e-4
    biomask_minimum_learning_rate: float = 2.0e-6
    biomask_weight_decay: float = 1.0e-4
    biomask_base_channels: int = 32
    biomask_crop_shift_pixels: int = 8
    biomask_d4_probability: float = 0.875
    biomask_prompt_sigma_pixels: float = 2.5
    biomask_presence_sigma_pixels: float = 9.0
    biomask_presence_support_radius_pixels: float = 22.0
    biomask_instance_margin_pixels: float = 3.0
    biomask_boundary_ramp_pixels: float = 2.0
    biomask_boundary_min_weight: float = 0.25
    biomask_empty_mask_loss_weight: float = 0.25
    biomask_positive_visibility_retained_min: float = 0.80
    biomask_ambiguous_visibility_retained_min: float = 0.55
    biomask_crop_shift_attempts: int = 6
    biomask_ambiguous_shift_scale: float = 0.50
    biomask_negative_near_outer_radius_pixels: float = 28.0
    biomask_negative_crowded_min_nuclei: int = 2
    biomask_negative_high_score_quantile: float = 0.80
    biomask_negative_sampling_fractions: tuple[float, float, float, float] = (0.30, 0.30, 0.25, 0.15)
    biomask_max_importance_weight: float = 4.0
    # Fixed after development-fold resolver audits. These values construct labels
    # identically for every OOF fold and never adapt to the held fold.
    resolver_quality_threshold: float = 0.65
    resolver_margin_threshold: float = 0.20
    resolver_geometry_weight: float = 0.25
    resolver_minimum_mask_support: float = 6.0
    # -1 uses every logical CPU exposed to the runtime.
    cpu_crop_workers: int = 8
    require_existing_complete_biomask: bool = False


@dataclass(slots=True)
class Stage2Config:
    selected_folds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])

    # v16.4 integrated Stage-2 curriculum: 5 + 65 + 50 + 20 = 140 epochs.
    # Keep the historical field names semantic_epochs / balanced_refinement_epochs /
    # utility_epochs so existing notebooks and downstream tooling do not break.
    semantic_epochs: int = 5
    joint_identity_epochs: int = 65
    balanced_refinement_epochs: int = 50
    utility_epochs: int = 20

    # semantic_batch_size remains the EFFECTIVE optimizer batch from v16.3.2.
    # Physical UNI2 micro-batches are auto-probed and accumulated to preserve the
    # already-proven Stage-2 optimization scale while fitting 24-GB GPUs.
    semantic_batch_size: int = 512
    semantic_micro_batch_size: int = 0
    maximum_semantic_micro_batch_size: int = 128
    context_micro_batch_size: int = 0
    maximum_context_micro_batch_size: int = 128
    utility_batch_size: int = 1024
    inference_batch_size: int = 128

    hidden_dim: int = 256
    biology_hidden_dim: int = 128
    gate_hidden_dim: int = 96
    dropout: float = 0.08

    # Phase-specific learning rates.  Longer schedules intentionally use lower
    # cumulative optimization pressure than a naive 1.4x extension.
    semantic_learning_rate: float = 2.0e-4
    joint_identity_learning_rate: float = 1.4e-4
    joint_identity_minimum_learning_rate: float = 2.0e-6
    joint_identity_lora_learning_rate: float = 1.25e-5
    joint_identity_lora_minimum_learning_rate: float = 6.0e-7
    joint_identity_lora_warmup_epochs: int = 3
    balanced_learning_rate: float = 4.0e-5
    context_learning_rate: float = 5.0e-5
    balanced_minimum_learning_rate: float = 2.0e-6
    balanced_lora_learning_rate: float = 5.0e-6
    balanced_lora_minimum_learning_rate: float = 2.5e-7
    utility_learning_rate: float = 1.5e-4
    minimum_learning_rate: float = 2.0e-6

    weight_decay: float = 1.0e-4
    lora_weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    lora_gradient_clip_norm: float = 1.0

    identity_loss_weight: float = 0.50
    residual_loss_weight: float = 2.0e-3
    mask_residual_limit: float = 1.0
    biology_residual_limit: float = 1.0
    local_logit_residual_limit: float = 1.25
    tissue_logit_residual_limit: float = 1.25
    spatial_logit_residual_limit: float = 1.0
    mask_gate_initial_probability: float = 0.50
    biology_gate_initial_probability: float = 0.50
    context_gate_initial_probability: float = 0.12
    context_dropout_start: float = 0.30
    context_dropout_end: float = 0.12

    # Preserve biological GT-group sampling from v16.3.2.  The 65-epoch joint
    # identity phase uses only mild tempering; the context phase uses the proven
    # stronger ROI-aware refinement settings.
    phase1_secondary_per_group: int = 1
    phase1a_roi_sampling_alpha: float = 1.0
    phase1a_class_tempering_gamma: float = 0.0
    phase1b_roi_sampling_alpha: float = 0.75
    phase1b_class_tempering_gamma: float = 0.25
    phase2_secondary_per_group: int = 2
    roi_sampling_alpha: float = 0.5
    class_tempering_gamma: float = 0.5

    utility_positive_fraction: float = 0.60
    utility_hard_negative_fraction: float = 0.70

    identity_anchor_mode: str = "center"
    use_mask_identity: bool = True
    use_morphology_identity: bool = True
    use_local_context: bool = True
    use_tissue_context: bool = True
    use_spatial_context: bool = True
    utility_use_semantic_features: bool = False

    # Integrated UNI2-h + fresh LoRA. The base remains frozen.
    integrated_uni2: bool = True
    identity_only_encoder_gradient: bool = False
    use_bfloat16: bool = True
    use_fused_adamw: bool = True

    # Proposal-level synchronized augmentation.  One D4 transform is shared by
    # 64/128/256 and by the BioMask token maps; photometric jitter touches RGB only.
    use_d4_augmentation: bool = True
    d4_augmentation_probability: float = 0.875
    photometric_augmentation: bool = True
    contrast_range: tuple[float, float] = (0.88, 1.12)
    brightness_range: tuple[float, float] = (-0.05, 0.05)
    gamma_range: tuple[float, float] = (0.90, 1.10)
    channel_scale_range: tuple[float, float] = (0.95, 1.05)

    validation_thresholds: tuple[float, ...] = (
        0.000, 0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200, 0.225,
        0.250, 0.275, 0.300, 0.325, 0.350, 0.375, 0.400, 0.425, 0.450, 0.475,
        0.500, 0.525, 0.550, 0.575, 0.600, 0.625, 0.650, 0.675, 0.700, 0.725,
        0.750, 0.775, 0.800, 0.825, 0.850, 0.875, 0.900, 0.925, 0.950,
    )
    checkpoint_every_epochs: int = 1
    memory_cleanup_each_epoch: bool = True


@dataclass(slots=True)
class AdapterConfig:
    meta_cv_folds: int = 3
    score_families: tuple[str, ...] = (
        "semantic",
        "semantic_utility",
        "semantic_utility_localization",
    )
    nms_radii: tuple[float, ...] = (0.0, 2.0, 3.0, 4.0, 5.0)
    global_thresholds: tuple[float, ...] = (
        0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
    )
    class_offset_grid: tuple[float, ...] = (-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15)
    class_threshold_shrinkage_strength: float = 8.0
    minimum_positive_rois_for_class_offset: int = 5
    minimum_positive_cases_for_class_offset: int = 3
    use_probability_calibration: bool = False
    allow_multi_emission: bool = False


@dataclass(slots=True)
class RuntimeConfig:
    use_bfloat16: bool = True
    allow_tf32: bool = True
    deterministic: bool = False
    pin_memory: bool = True
    persistent_workers: bool = True
    channels_last: bool = True
    fused_adamw: bool = True


@dataclass(slots=True)
class PipelineConfig:
    paths: ProjectPaths = field(default_factory=ProjectPaths)
    data: DataConfig = field(default_factory=DataConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    uni2_lora: Uni2LoRAConfig = field(default_factory=Uni2LoRAConfig)
    stage2_preprocess: Stage2PreprocessConfig = field(default_factory=Stage2PreprocessConfig)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        # Keep validation focused on settings that can make the pipeline invalid.
        if self.data.stage1_output_stride != STAGE1_OUTPUT_STRIDE:
            raise ValueError(f"stage1_output_stride must be {STAGE1_OUTPUT_STRIDE}")
        if self.data.number_of_folds < 2:
            raise ValueError("number_of_folds must be >= 2")
        if self.data.cpu_workers < -1:
            raise ValueError("cpu_workers must be -1 or >= 0")
        if self.data.fold_search_restarts < 1 or self.data.heatmap_sigma_grid_units <= 0:
            raise ValueError("invalid fold/heatmap preprocessing settings")

        s1 = self.stage1
        if min(s1.epochs, s1.micro_batch_size, s1.effective_batch_size, s1.base_channels, s1.fpn_channels) <= 0:
            raise ValueError("Stage-1 epochs, batch sizes and channel counts must be > 0")
        if s1.effective_batch_size < s1.micro_batch_size:
            raise ValueError("effective_batch_size must be >= micro_batch_size")
        _ = s1.accumulation_steps
        if s1.fpn_channels % 8:
            raise ValueError("stage1.fpn_channels must be divisible by 8")
        if not (0 <= s1.warmup_epochs <= s1.epochs):
            raise ValueError("stage1.warmup_epochs must be between 0 and epochs")
        if not (0 <= s1.validation_decode_threshold <= 1):
            raise ValueError("stage1.validation_decode_threshold must be in [0,1]")
        if not s1.validation_thresholds:
            raise ValueError("stage1.validation_thresholds cannot be empty")
        if any(not (0 <= float(value) <= 1) for value in s1.validation_thresholds):
            raise ValueError("every stage1.validation_thresholds value must be in [0,1]")
        if len(set(float(value) for value in s1.validation_thresholds)) != len(s1.validation_thresholds):
            raise ValueError("stage1.validation_thresholds cannot contain duplicates")
        if not s1.validation_suppression_radii_pixels or any(float(v) < 0 for v in s1.validation_suppression_radii_pixels):
            raise ValueError("stage1.validation_suppression_radii_pixels must contain non-negative radii")
        if not (0.0 < s1.validation_min_recall <= 1.0):
            raise ValueError("stage1.validation_min_recall must be in (0,1]")
        if s1.proposal_quality_power < 0 or s1.proposal_uncertainty_penalty < 0:
            raise ValueError("Stage-1 proposal score exponents/penalties must be >= 0")
        if s1.hard_negative_loss_weight < 0 or s1.hard_negative_topk_per_image < 1:
            raise ValueError("invalid Stage-1 hard-negative settings")
        if s1.hard_negative_start_epoch < 0 or s1.hard_negative_ramp_epochs < 1:
            raise ValueError("invalid Stage-1 hard-negative schedule")
        if not (0.0 <= s1.d4_augmentation_probability <= 1.0):
            raise ValueError("stage1.d4_augmentation_probability must be in [0,1]")
        for name, bounds in (
            ("brightness_range", s1.brightness_range),
            ("contrast_range", s1.contrast_range),
            ("gamma_range", s1.gamma_range),
            ("channel_scale_range", s1.channel_scale_range),
        ):
            if len(bounds) != 2 or float(bounds[0]) <= 0 or float(bounds[1]) < float(bounds[0]):
                raise ValueError(f"stage1.{name} must be a positive ordered pair")
        if s1.use_d4_augmentation and self.data.stage1_output_stride == 2 and s1.offset_range_grid_units < 2.5:
            raise ValueError("Exact stride-2 D4 augmentation requires stage1.offset_range_grid_units >= 2.5")
        if s1.learning_rate <= 0 or s1.minimum_learning_rate < 0 or s1.minimum_learning_rate > s1.learning_rate:
            raise ValueError("invalid Stage-1 learning rates")
        if s1.offset_range_grid_units < 1 or s1.max_candidates_per_roi < 1:
            raise ValueError("invalid Stage-1 decoding/target settings")
        if s1.local_max_radius < 0 or s1.suppression_radius_pixels < 0:
            raise ValueError("Stage-1 suppression settings must be >= 0")
        if s1.gradient_clip_norm <= 0 or s1.quality_target_sigma_grid_units <= 0:
            raise ValueError("Stage-1 gradient_clip_norm and quality sigma must be > 0")
        if s1.backbone_learning_rate_scale <= 0:
            raise ValueError("stage1.backbone_learning_rate_scale must be > 0")
        if s1.heatmap_loss_weight <= 0 or min(s1.offset_loss_weight, s1.quality_loss_weight, s1.uncertainty_loss_weight) < 0:
            raise ValueError("Stage-1 heatmap loss weight must be > 0 and auxiliary loss weights must be >= 0")
        if s1.checkpoint_every_epochs < 1:
            raise ValueError("stage1.checkpoint_every_epochs must be >= 1")
        if not (0 <= s1.ema_decay < 1):
            raise ValueError("stage1.ema_decay must be in [0,1)")

        lora = self.uni2_lora
        if min(lora.rank, lora.last_blocks) < 1:
            raise ValueError("UNI2 LoRA rank and last_blocks must be >= 1")
        if lora.alpha <= 0 or not (0.0 <= lora.dropout < 1.0):
            raise ValueError("invalid UNI2 LoRA alpha/dropout")
        if not lora.target_modules or any(not str(value).strip() for value in lora.target_modules):
            raise ValueError("uni2_lora.target_modules cannot be empty")

        p = self.stage2_preprocess
        if not p.selected_folds or len(set(int(fold) for fold in p.selected_folds)) != len(p.selected_folds):
            raise ValueError("stage2_preprocess.selected_folds must be non-empty and unique")
        if any(int(fold) < 0 or int(fold) >= self.data.number_of_folds for fold in p.selected_folds):
            raise ValueError("stage2_preprocess.selected_folds contains an invalid fold")
        if p.association_radius_pixels <= 0 or p.crop_size <= 0:
            raise ValueError("invalid Stage-2 preprocessing geometry settings")
        if not p.uni2_view_sizes or min(p.uni2_view_sizes) <= 0:
            raise ValueError("invalid UNI2 view sizes")
        if min(p.uni2_chunk_size, p.uni2_token_map_batch_size) < 1:
            raise ValueError("token-map chunk and resize batch sizes must be > 0")
        if p.feature_dtype not in {"float16", "float32"}:
            raise ValueError("feature_dtype must be 'float16' or 'float32'")
        if min(p.biomask_epochs, p.biomask_batch_size, p.biomask_base_channels) <= 0:
            raise ValueError("BioMask epochs, batch size and base channels must be > 0")
        if p.biomask_learning_rate <= 0 or p.biomask_minimum_learning_rate < 0 or p.biomask_minimum_learning_rate > p.biomask_learning_rate:
            raise ValueError("invalid BioMask learning rates")
        if p.biomask_weight_decay < 0:
            raise ValueError("biomask_weight_decay must be >= 0")
        if p.association_radius_pixels <= 0 or p.biological_supervision_radius_pixels < p.association_radius_pixels:
            raise ValueError("biological_supervision_radius_pixels must be >= association_radius_pixels > 0")
        if p.identity_view_size <= 0 or p.identity_view_size not in p.uni2_view_sizes:
            raise ValueError("identity_view_size must be a positive member of uni2_view_sizes")
        if tuple(int(value) for value in p.uni2_view_sizes) != tuple(UNI2_VIEW_SIZES):
            raise ValueError(f"uni2_view_sizes must preserve the feature contract {UNI2_VIEW_SIZES}")
        if p.crop_size < p.identity_view_size or (p.crop_size - p.identity_view_size) % 2:
            raise ValueError("crop_size and identity_view_size must be nested and center-aligned")
        largest_view = max(int(value) for value in p.uni2_view_sizes)
        if any((largest_view - int(value)) % 2 for value in p.uni2_view_sizes):
            raise ValueError("all UNI2 view sizes must be center-aligned within the largest view")
        if p.biomask_crop_shift_pixels < 0 or not (0.0 <= p.biomask_d4_probability <= 1.0):
            raise ValueError("invalid BioMask crop-shift/D4 settings")
        if min(p.biomask_prompt_sigma_pixels, p.biomask_presence_sigma_pixels, p.biomask_presence_support_radius_pixels) <= 0 or p.biomask_instance_margin_pixels < 0:
            raise ValueError("invalid BioMask prompt/presence/instance-margin settings")
        if p.biomask_presence_support_radius_pixels < p.biomask_presence_sigma_pixels:
            raise ValueError("biomask_presence_support_radius_pixels must be >= biomask_presence_sigma_pixels")
        if p.biomask_boundary_ramp_pixels <= 0 or not (0.0 < p.biomask_boundary_min_weight <= 1.0):
            raise ValueError("invalid BioMask boundary-ramp settings")
        if not (0.0 <= p.biomask_empty_mask_loss_weight <= 1.0):
            raise ValueError("biomask_empty_mask_loss_weight must be in [0,1]")
        for name, value in (("positive_visibility_retained_min", p.biomask_positive_visibility_retained_min), ("ambiguous_visibility_retained_min", p.biomask_ambiguous_visibility_retained_min), ("ambiguous_shift_scale", p.biomask_ambiguous_shift_scale)):
            if not (0.0 < float(value) <= 1.0):
                raise ValueError(f"biomask_{name} must be in (0,1]")
        if p.biomask_crop_shift_attempts < 1 or p.biomask_negative_crowded_min_nuclei < 1:
            raise ValueError("invalid BioMask crop-shift attempts/crowding settings")
        if p.biomask_negative_near_outer_radius_pixels <= p.biological_supervision_radius_pixels:
            raise ValueError("BioMask near-negative outer radius must exceed biological radius")
        if not (0.0 < p.biomask_negative_high_score_quantile < 1.0):
            raise ValueError("biomask_negative_high_score_quantile must be in (0,1)")
        fractions = tuple(float(v) for v in p.biomask_negative_sampling_fractions)
        if len(fractions) != 4 or any(v < 0 for v in fractions) or sum(fractions) <= 0:
            raise ValueError("biomask_negative_sampling_fractions must contain four non-negative values")
        if p.biomask_max_importance_weight < 1.0:
            raise ValueError("biomask_max_importance_weight must be >= 1")
        if not (0.0 <= p.resolver_quality_threshold <= 2.0) or p.resolver_margin_threshold < 0:
            raise ValueError("invalid fixed resolver quality/margin thresholds")
        if p.cpu_crop_workers < -1:
            raise ValueError("cpu_crop_workers must be -1 or >= 0")

        s2 = self.stage2
        if min(s2.semantic_epochs, s2.joint_identity_epochs, s2.utility_epochs) < 1 or s2.balanced_refinement_epochs < 1:
            raise ValueError("invalid Stage-2 epoch counts")
        if min(s2.semantic_batch_size, s2.utility_batch_size, s2.inference_batch_size) <= 0:
            raise ValueError("Stage-2 batch sizes must be > 0")
        if s2.semantic_micro_batch_size < 0 or s2.context_micro_batch_size < 0:
            raise ValueError("Stage-2 micro-batches must be 0(auto) or positive")
        if min(s2.maximum_semantic_micro_batch_size, s2.maximum_context_micro_batch_size) < 1:
            raise ValueError("Stage-2 maximum micro-batches must be > 0")
        for name, value in (("semantic_micro_batch_size", s2.semantic_micro_batch_size), ("context_micro_batch_size", s2.context_micro_batch_size)):
            if value > 0 and s2.semantic_batch_size % value:
                raise ValueError(f"stage2.{name} must divide semantic_batch_size")
        if min(s2.hidden_dim, s2.biology_hidden_dim, s2.gate_hidden_dim) <= 0:
            raise ValueError("Stage-2 hidden dimensions must be > 0")
        if not (0 <= s2.dropout < 1):
            raise ValueError("stage2.dropout must be in [0,1)")
        if not (0 < s2.utility_positive_fraction < 1) or not (0 <= s2.utility_hard_negative_fraction <= 1):
            raise ValueError("invalid utility sampling fractions")
        if not (0.0 < s2.identity_loss_weight <= 2.0) or s2.residual_loss_weight < 0:
            raise ValueError("invalid identity/residual loss weights")
        if min(s2.mask_residual_limit, s2.biology_residual_limit, s2.local_logit_residual_limit, s2.tissue_logit_residual_limit, s2.spatial_logit_residual_limit) <= 0:
            raise ValueError("all Stage-2 residual limits must be > 0")
        for name, value in (("mask_gate_initial_probability", s2.mask_gate_initial_probability), ("biology_gate_initial_probability", s2.biology_gate_initial_probability), ("context_gate_initial_probability", s2.context_gate_initial_probability), ("context_dropout_start", s2.context_dropout_start), ("context_dropout_end", s2.context_dropout_end), ("d4_augmentation_probability", s2.d4_augmentation_probability)):
            if not (0.0 <= float(value) < 1.0 if name != "d4_augmentation_probability" else 0.0 <= float(value) <= 1.0):
                raise ValueError(f"stage2.{name} is outside its valid probability range")
        if s2.context_dropout_end > s2.context_dropout_start:
            raise ValueError("context_dropout_end cannot exceed context_dropout_start")
        if min(s2.phase1_secondary_per_group, s2.phase2_secondary_per_group) < 0:
            raise ValueError("group secondary counts must be >= 0")
        if s2.identity_anchor_mode not in {"center", "pooled64"}:
            raise ValueError("stage2.identity_anchor_mode must be center or pooled64")
        if not s2.validation_thresholds:
            raise ValueError("stage2.validation_thresholds cannot be empty")
        if any(not (0 <= float(value) < 1) for value in s2.validation_thresholds):
            raise ValueError("every stage2.validation_thresholds value must be in [0,1)")
        if len(set(float(value) for value in s2.validation_thresholds)) != len(s2.validation_thresholds):
            raise ValueError("stage2.validation_thresholds cannot contain duplicates")
        positive_lrs = (
            s2.semantic_learning_rate, s2.joint_identity_learning_rate, s2.joint_identity_lora_learning_rate,
            s2.balanced_learning_rate, s2.context_learning_rate, s2.balanced_lora_learning_rate,
            s2.utility_learning_rate,
        )
        if min(positive_lrs) <= 0:
            raise ValueError("Stage-2 learning rates must be > 0")
        if min(s2.minimum_learning_rate, s2.joint_identity_minimum_learning_rate, s2.joint_identity_lora_minimum_learning_rate, s2.balanced_minimum_learning_rate, s2.balanced_lora_minimum_learning_rate) < 0:
            raise ValueError("Stage-2 minimum learning rates must be >= 0")
        if s2.joint_identity_minimum_learning_rate > s2.joint_identity_learning_rate or s2.joint_identity_lora_minimum_learning_rate > s2.joint_identity_lora_learning_rate:
            raise ValueError("invalid Stage-2 joint-identity LR minima")
        if s2.balanced_minimum_learning_rate > min(s2.balanced_learning_rate, s2.context_learning_rate) or s2.balanced_lora_minimum_learning_rate > s2.balanced_lora_learning_rate:
            raise ValueError("invalid Stage-2 balanced LR minima")
        if not (0 <= s2.joint_identity_lora_warmup_epochs <= s2.joint_identity_epochs):
            raise ValueError("joint_identity_lora_warmup_epochs must be within joint_identity_epochs")
        if s2.weight_decay < 0 or s2.lora_weight_decay < 0 or min(s2.gradient_clip_norm, s2.lora_gradient_clip_norm) <= 0:
            raise ValueError("invalid Stage-2 weight decay / gradient clip settings")
        for name, value in (("phase1a_roi_sampling_alpha", s2.phase1a_roi_sampling_alpha), ("phase1a_class_tempering_gamma", s2.phase1a_class_tempering_gamma), ("phase1b_roi_sampling_alpha", s2.phase1b_roi_sampling_alpha), ("phase1b_class_tempering_gamma", s2.phase1b_class_tempering_gamma), ("roi_sampling_alpha", s2.roi_sampling_alpha), ("class_tempering_gamma", s2.class_tempering_gamma)):
            if not (0.0 <= float(value) <= 1.0):
                raise ValueError(f"stage2.{name} must be in [0,1]")
        for name, bounds, allow_negative in (("contrast_range", s2.contrast_range, False), ("brightness_range", s2.brightness_range, True), ("gamma_range", s2.gamma_range, False), ("channel_scale_range", s2.channel_scale_range, False)):
            if len(bounds) != 2 or float(bounds[1]) < float(bounds[0]):
                raise ValueError(f"stage2.{name} must be an ordered pair")
            if not allow_negative and float(bounds[0]) <= 0:
                raise ValueError(f"stage2.{name} must be positive")
        if not s2.selected_folds or len(set(s2.selected_folds)) != len(s2.selected_folds):
            raise ValueError("stage2.selected_folds must be non-empty and unique")
        if any(fold < 0 or fold >= self.data.number_of_folds for fold in s2.selected_folds):
            raise ValueError("stage2.selected_folds contains an invalid fold")

        a = self.adapter
        if a.meta_cv_folds < 2:
            raise ValueError("adapter.meta_cv_folds must be >= 2")
        if not a.score_families or not a.nms_radii or not a.global_thresholds or not a.class_offset_grid:
            raise ValueError("adapter search grids cannot be empty")
        if any(v < 0 for v in a.nms_radii) or any(v < 0 or v > 1 for v in a.global_thresholds):
            raise ValueError("invalid adapter search grid")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_resolved(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        temporary.replace(path)
