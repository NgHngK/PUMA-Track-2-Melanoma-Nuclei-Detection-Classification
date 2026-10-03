from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import PipelineConfig


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (x.shape[-1],), self.weight, self.bias)
        return x.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, drop_path: float = 0.0) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = LayerNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, 4 * channels, 1)
        self.pw2 = nn.Conv2d(4 * channels, channels, 1)
        self.scale = nn.Parameter(torch.full((channels,), 1.0e-6))
        self.drop_path = float(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw(x)
        x = self.norm(x)
        x = F.gelu(self.pw1(x))
        x = self.pw2(x) * self.scale[None, :, None, None]
        if self.training and self.drop_path > 0:
            keep = 1.0 - self.drop_path
            mask = x.new_empty((len(x), 1, 1, 1)).bernoulli_(keep)
            x = x * mask / keep
        return residual + x


class EncoderStage(nn.Module):
    def __init__(self, channels: int, depth: int, drop_paths: list[float]) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[ConvNeXtBlock(channels, drop_paths[i]) for i in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)




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


class FullRoiEncoder(nn.Module):
    def __init__(self, base: int) -> None:
        super().__init__()
        channels = (base, base * 2, base * 4, base * 8, base * 12)
        depths = (2, 2, 5, 3, 2)
        drop_paths = torch.linspace(0.0, 0.12, sum(depths)).tolist()
        self.channels = channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 4, stride=2, padding=1, bias=False),
            LayerNorm2d(channels[0]),
        )
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.context = GlobalContextMixer(channels[-1])
        cursor = 0
        for index, (channel, depth) in enumerate(zip(channels, depths, strict=True)):
            self.stages.append(EncoderStage(channel, depth, drop_paths[cursor : cursor + depth]))
            cursor += depth
            if index < len(channels) - 1:
                self.downsamples.append(
                    nn.Sequential(LayerNorm2d(channel), nn.Conv2d(channel, channels[index + 1], 2, stride=2))
                )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        outputs: list[torch.Tensor] = []
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index == len(self.stages) - 1:
                x = self.context(x)
            outputs.append(x)
            if index < len(self.downsamples):
                x = self.downsamples[index](x)
        return outputs


class FeaturePyramid(nn.Module):
    def __init__(self, channels: tuple[int, ...], out_channels: int) -> None:
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in channels])
        self.refine = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
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
            x = F.interpolate(x, size=features[index].shape[-2:], mode="bilinear", align_corners=False)
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


class FullRoiPointDetector(nn.Module):
    def __init__(self, config: PipelineConfig) -> None:
        super().__init__()
        self.output_stride = int(config.data.stage1_output_stride)
        if self.output_stride != 2:
            raise ValueError("Current architecture uses a stride-2 prediction grid")
        self.encoder = FullRoiEncoder(config.stage1.base_channels)
        self.fpn = FeaturePyramid(self.encoder.channels, config.stage1.fpn_channels)
        self.head = DensePredictionHead(config.stage1.fpn_channels, config.stage1.offset_range_grid_units)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head(self.fpn(self.encoder(image)))


def build_stage1_model(config: PipelineConfig) -> FullRoiPointDetector:
    return FullRoiPointDetector(config)

