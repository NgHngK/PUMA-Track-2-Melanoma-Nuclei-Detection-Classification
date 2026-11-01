"""Shared context mixer, feature pyramid and dense detection heads for timm ConvNeXt.

No custom/original backbone is included in this release.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class GlobalContextMixer(nn.Module):
    def __init__(self, channels: int, layers: int = 2) -> None:
        super().__init__()
        heads = 12 if channels % 12 == 0 else 8
        if channels % heads:
            heads = 4
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=channels,
                nhead=heads,
                dim_feedforward=2 * channels,
                dropout=0.08,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(layers)
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class FeaturePyramid(nn.Module):
    def __init__(self, channels: tuple[int, ...], out_channels: int) -> None:
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in channels])
        self.refine = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False
                    ),
                    nn.Conv2d(out_channels, out_channels, 1, bias=False),
                    nn.GroupNorm(8, out_channels),
                    nn.GELU(),
                )
                for _ in channels
            ]
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        x = self.refine[-1](self.lateral[-1](features[-1]))
        for index in range(len(features) - 2, -1, -1):
            x = F.interpolate(
                x, size=features[index].shape[-2:], mode="bilinear", align_corners=False
            )
            x = self.refine[index](x + self.lateral[index](features[index]))
        return x


class DensePredictionHead(nn.Module):
    def __init__(self, channels: int, offset_range_grid_units: float) -> None:
        super().__init__()
        self.offset_range_grid_units = float(offset_range_grid_units)
        self.shared = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )
        self.heatmap = nn.Conv2d(channels, 1, 1)
        self.offset = nn.Conv2d(channels, 2, 1)
        self.quality = nn.Conv2d(channels, 1, 1)
        self.log_variance = nn.Conv2d(channels, 1, 1)
        prior = 0.01
        nn.init.constant_(self.heatmap.bias, math.log(prior / (1.0 - prior)))
        nn.init.zeros_(self.offset.bias)
        nn.init.constant_(self.quality.bias, math.log(prior / (1.0 - prior)))
        nn.init.zeros_(self.log_variance.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.shared(x)
        return {
            "heatmap_logits": self.heatmap(x),
            "offset": self.offset_range_grid_units * torch.tanh(self.offset(x)),
            "quality_logits": self.quality(x),
            "log_variance": self.log_variance(x).clamp(-5.0, 5.0),
        }
