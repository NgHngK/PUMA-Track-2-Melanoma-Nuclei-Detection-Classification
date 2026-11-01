"""Architecture selection. Only the feature encoder and its connection to FPN differ."""

from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .components import DensePredictionHead, FeaturePyramid, GlobalContextMixer

ARCHITECTURES = ("convnext_tiny", "convnext_base")
TIMM_MODELS = {"convnext_tiny": "convnext_tiny.fb_in1k", "convnext_base": "convnext_base.fb_in1k"}


class OfficialConvNeXtDetector(nn.Module):
    """Official timm ConvNeXt encoder + the reference Transformer/FPN/detection heads.

    timm features are at output strides 4/8/16/32, versus 2/4/8/16/32
    in the old custom encoder. Explicit 2x upsampling restores stride 2.
    """

    def __init__(self, name: str, pretrained: bool = True, fpn_channels: int = 96):
        super().__init__()
        if name not in ("convnext_tiny", "convnext_base"):
            raise ValueError(f"Unknown official ConvNeXt variant: {name}")
        import timm

        self.timm_model_name = TIMM_MODELS[name]
        self.encoder = timm.create_model(
            self.timm_model_name,
            pretrained=pretrained,
            features_only=True,
            # Use timm's official Facebook URL/cache without probing Colab HF secrets.
            pretrained_cfg_overlay={"hf_hub_id": None},
            out_indices=(0, 1, 2, 3),
        )
        channels = tuple(self.encoder.feature_info.channels())
        strides = tuple(self.encoder.feature_info.reduction())
        if strides != (4, 8, 16, 32):
            raise RuntimeError(f"Unexpected {name} feature strides {strides}; need (4,8,16,32)")
        self.context = GlobalContextMixer(channels[-1])
        self.fpn = FeaturePyramid(channels, fpn_channels)
        self.head = DensePredictionHead(fpn_channels, offset_range_grid_units=2.5)
        self.output_stride = 2
        self.grad_checkpointing = False
        self.pretrained = pretrained
        cfg = getattr(self.encoder, "pretrained_cfg", None) or {}
        mean = cfg.get("mean", (0.485, 0.456, 0.406))
        std = cfg.get("std", (0.229, 0.224, 0.225))
        self.register_buffer("input_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("input_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    def set_grad_checkpointing(self, enabled: bool = True) -> None:
        self.grad_checkpointing = bool(enabled)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.pretrained:
            image = (image - self.input_mean) / self.input_std
        if self.grad_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint

            maps = list(checkpoint(lambda x: tuple(self.encoder(x)), image, use_reentrant=False))
        else:
            maps = list(self.encoder(image))
        maps[-1] = self.context(maps[-1])
        fused_stride4 = self.fpn(maps)
        fused_stride2 = F.interpolate(
            fused_stride4, scale_factor=2.0, mode="bilinear", align_corners=False
        )
        return self.head(fused_stride2)


def build_model(name: str, *, pretrained: bool = True) -> nn.Module:
    if name in ("convnext_tiny", "convnext_base"):
        return OfficialConvNeXtDetector(name, pretrained=pretrained)
    raise ValueError(f"Unknown architecture {name!r}. Available: {ARCHITECTURES}")
