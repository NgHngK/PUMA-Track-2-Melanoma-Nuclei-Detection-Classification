"""Classify features extracted at Stage 1 predicted locations."""

import numpy as np
import torch
from .model import NucleiFusion
from .graph import build, gpu_bundle
from common.resources import gpu_batch_size
from .data import class_names
from .settings import MODEL_REVISION, FEATURE_REVISION


class Classifier:
    def __init__(self, checkpoint, device="cuda"):
        self.device = device
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if tuple(state.get("class_names", ())) != class_names:
            raise ValueError("Stage 2 checkpoint class order does not match this classifier")
        if (
            state.get("model_revision") != MODEL_REVISION
            or state.get("feature_revision") != FEATURE_REVISION
        ):
            raise ValueError(
                "Stage 2 checkpoint model/feature revision mismatch; retrain this fold"
            )
        self.mean = np.asarray(state["tier_mean"], dtype=np.float32)
        self.std = np.asarray(state["tier_std"], dtype=np.float32)
        if (
            self.mean.shape != (16,)
            or self.std.shape != (16,)
            or not np.isfinite(self.mean).all()
            or not np.isfinite(self.std).all()
            or np.any(self.std <= 0)
        ):
            raise ValueError("Stage 2 checkpoint has invalid Tier normalization statistics")
        self.model = NucleiFusion().to(device).eval()
        self.model.load_state_dict(state["model"], strict=True)
        self.batch_size = 0

    @torch.inference_mode()
    def predict(self, features, batch_size=0):
        if batch_size < 0:
            raise ValueError("batch_size must be >= 0")
        xy = features["xy"]
        if len(xy) == 0:
            return np.empty((0, 10), dtype=np.float32)
        tier = ((features["tier_raw"] - self.mean) / self.std).astype(np.float32)
        rois = np.zeros(len(xy), dtype=np.int32)
        graph = build(xy, rois)
        data = gpu_bundle({**features, "tier": tier}, graph, self.device)
        if batch_size == 0:
            if not self.batch_size or len(xy) > self.batch_size:

                def run(size):
                    ids = torch.arange(size, device=self.device) % len(xy)
                    output = self.model(ids, data)
                    del ids, output

                self.batch_size = gpu_batch_size(
                    torch.device(self.device), run, limit=min(2048, len(xy))
                )
                print(f"Stage 2 inference batch size: {self.batch_size}", flush=True)

            batch_size = self.batch_size

        logits = []
        for left in range(0, len(xy), batch_size):
            ids = torch.arange(left, min(left + batch_size, len(xy)), device=self.device)
            logits.append(self.model(ids, data).float().cpu())
        return torch.cat(logits).softmax(1).numpy()
