from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import PipelineConfig
from ..data.preprocess import load_shared_artifacts


class Stage1Dataset(Dataset):
    def __init__(self, config: PipelineConfig, roi_indices: np.ndarray) -> None:
        artifacts = load_shared_artifacts(config)
        self.images = artifacts["images"]
        self.heatmaps = artifacts["heatmaps"]
        self.point_targets = artifacts["point_targets"]
        self.offsets = artifacts["offsets"]
        self.roi_indices = np.asarray(roi_indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.roi_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        roi = int(self.roi_indices[index])
        image = np.asarray(self.images[roi], dtype=np.uint8)
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0)
        heatmap = torch.from_numpy(np.asarray(self.heatmaps[roi], dtype=np.float32).copy())
        start, stop = int(self.offsets[roi]), int(self.offsets[roi + 1])
        points = np.asarray(self.point_targets[start:stop]).copy()
        return {"roi_index": roi, "image": tensor, "heatmap": heatmap, "points": points}


def collate_stage1(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "roi_index": torch.as_tensor([item["roi_index"] for item in batch], dtype=torch.long),
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "heatmap": torch.stack([item["heatmap"] for item in batch], dim=0),
        "points": [item["points"] for item in batch],
    }

