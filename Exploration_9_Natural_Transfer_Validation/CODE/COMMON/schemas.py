from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass(frozen=True)
class ManifestRow:
    uid: str
    roi_id: str
    image_path: str
    x: float
    y: float
    label: int
    class_name: str
    role: str
    coordinate_source: str = "gt"


@dataclass(frozen=True)
class FeatureContract:
    feature_name: str
    fov: int
    sigma: float
    dim: int
    encoder_sha256: str
    input_contract_hash: str
    class_map_hash: str


@dataclass
class ModelOutput:
    logits: torch.Tensor
    anchor_logits: torch.Tensor
    appearance_logits: torch.Tensor
    a5_logits: torch.Tensor
    context_logits: Optional[torch.Tensor] = None
    context_gate: Optional[torch.Tensor] = None


@dataclass
class FeatureBatch:
    uids: list[str]
    roi_ids: list[str]
    labels: torch.Tensor
    h_local: torch.Tensor
    tier_a: torch.Tensor
    h_wide: Optional[torch.Tensor] = None
