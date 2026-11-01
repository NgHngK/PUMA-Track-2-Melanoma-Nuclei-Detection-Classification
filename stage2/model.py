"""E16 G3 nucleus classifier, with unchanged trainable layers."""

import torch
import math
import torch.nn.functional as f
from torch import nn
from .context import AnchorContextC


class NucleiFusion(nn.Module):
    def __init__(self, mode="G3", hidden=64, dropout=0.2, residual_bound=1.0):
        super().__init__()
        if mode != "G3":
            raise ValueError("Only G3 fusion is supported")
        if not math.isfinite(residual_bound) or residual_bound <= 0:
            raise ValueError("residual_bound must be finite and positive")
        self.base = AnchorContextC()
        self.residual_bound = residual_bound
        self.tissue_head = nn.Sequential(
            nn.LayerNorm(20), nn.Linear(20, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 10)
        )
        nn.init.zeros_(self.tissue_head[-1].weight)
        nn.init.zeros_(self.tissue_head[-1].bias)
        self.self_proj = nn.Sequential(nn.Linear(3108, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.nei_proj = nn.Sequential(nn.Linear(3108, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.graph_out = nn.Sequential(
            nn.Linear(hidden * 2 + 1, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 10)
        )
        nn.init.zeros_(self.graph_out[-1].weight)
        nn.init.zeros_(self.graph_out[-1].bias)

    def assemble(self, data, indices):
        if "node_features" in data:
            return data["node_features"][indices]
        h96 = f.layer_norm(data["h96"][indices], (1536,))
        h192 = f.layer_norm(data["h192"][indices], (1536,))
        return torch.cat((h96, h192, data["tier"][indices], data["tissue"][indices]), 1)

    def forward(self, ids, data, return_delta=False):
        h96, h192, tier = (data[key][ids] for key in ("h96", "h192", "tier"))
        logits = self.base(h96, h192, tier)
        correction = self.tissue_head(data["tissue"][ids])
        neighbors = data["neighbors"][ids]
        weights = data["weights"][ids]
        present = ((neighbors >= 0) & (weights > 0)).any(dim=1).float().unsqueeze(1)
        root = self.self_proj(self.assemble(data, ids))
        if "neighbor_mean" in data:
            message = self.nei_proj(data["neighbor_mean"][ids])
        else:
            safe = neighbors.clamp_min(0)
            features = self.assemble(data, safe.reshape(-1))
            features = features.reshape(neighbors.shape[0], neighbors.shape[1], -1)
            message = self.nei_proj((features * weights.unsqueeze(-1)).sum(1))
        graph = self.graph_out(torch.cat((root, message, present), dim=1))
        bounded = self.residual_bound * torch.tanh((correction + graph) / self.residual_bound)
        logits = logits + bounded
        return (logits, bounded) if return_delta else logits


def n_parameters():
    return sum((param.numel() for param in NucleiFusion().parameters()))
