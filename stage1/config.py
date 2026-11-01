"""Shared Stage 1 training configuration for ConvNeXt-Tiny and ConvNeXt-Base."""

from dataclasses import dataclass

DETECTOR_REVISION = "stage1-v10-audited-geometry-loss-decode"


@dataclass
class Stage1Config:
    minimum_learning_rate: float = 2e-06
    warmup_epochs: int = 5
    weight_decay: float = 0.0001
    gradient_clip_norm: float = 1.0
    heatmap_loss_weight: float = 0.55
    offset_loss_weight: float = 0.18
    quality_loss_weight: float = 0.12
    uncertainty_loss_weight: float = 0.05
    hard_negative_loss_weight: float = 0.1
    hard_negative_topk_per_image: int = 256
    hard_negative_start_epoch: int = 5
    hard_negative_ramp_epochs: int = 10
    quality_target_sigma_grid_units: float = 0.75
    train_min_recall: float = 0.93
    train_thresholds: tuple[float, ...] = (
        0.000001,
        0.000003,
        0.00001,
        0.00003,
        0.0001,
        0.0003,
        0.001,
        0.003,
        0.005,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.075,
        0.1,
        0.125,
        0.15,
        0.2,
        0.25,
        0.3,
        0.4,
        0.5,
        0.65,
        0.8,
        0.9,
        0.95,
    )
    train_suppression_radii_pixels: tuple[float, ...] = (3.0, 4.0, 5.0, 6.0, 8.0)


@dataclass
class PipelineConfig:
    stage1: Stage1Config


def default_pipeline_config() -> PipelineConfig:
    return PipelineConfig(Stage1Config())
