"""E16 nucleus classifier: original G3 or four-head distance-aware GAT."""

import torch
import math
import torch.nn.functional as f
from torch import nn
from .context import AnchorContextC


class NucleiFusion(nn.Module):
    def __init__(self, mode="G3", hidden=64, dropout=0.2, residual_bound=1.0):
        super().__init__()
        if mode not in ("G3", "GAT"):
            raise ValueError("Graph mode must be G3 or GAT")
        if hidden % 4:
            raise ValueError("GAT requires hidden divisible by four")
        self.mode = mode
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
        if mode == "GAT":
            self.attention_heads = 4
            self.attention_q = nn.Linear(hidden, hidden, bias=False)
            self.attention_k = nn.Linear(hidden, hidden, bias=False)
            self.attention_v = nn.Linear(hidden, hidden, bias=False)

    def assemble(self, data, indices):
        if "node_features" in data:
            return data["node_features"][indices]
        h112 = f.layer_norm(data["h112"][indices], (1536,))
        h224 = f.layer_norm(data["h224"][indices], (1536,))
        return torch.cat((h112, h224, data["tier"][indices], data["tissue"][indices]), 1)

    def forward(self, ids, data, return_delta=False):
        h112, h224, tier = (data[key][ids] for key in ("h112", "h224", "tier"))
        logits = self.base(h112, h224, tier)
        correction = self.tissue_head(data["tissue"][ids])
        neighbors = data["neighbors"][ids]
        weights = data["weights"][ids]
        present = ((neighbors >= 0) & (weights > 0)).any(dim=1).float().unsqueeze(1)
        root = self.self_proj(self.assemble(data, ids))
        if self.mode == "G3":
            if "neighbor_mean" in data:
                message = self.nei_proj(data["neighbor_mean"][ids])
            else:
                safe = neighbors.clamp_min(0)
                features = self.assemble(data, safe.reshape(-1))
                features = features.reshape(neighbors.shape[0], neighbors.shape[1], -1)
                message = self.nei_proj((features * weights.unsqueeze(-1)).sum(1))
        else:
            # Compress each unique neighbor first: do NOT materialize B x K x 3108.
            safe = neighbors.clamp_min(0)
            unique, inverse = torch.unique(safe.reshape(-1), sorted=True, return_inverse=True)
            projected = self.nei_proj(data["node_features"][unique])
            keys = projected[inverse].reshape(neighbors.shape[0], neighbors.shape[1], -1)
            heads = self.attention_heads
            width = root.shape[1] // heads
            query = self.attention_q(root).reshape(len(ids), heads, width)
            key = self.attention_k(keys).reshape(len(ids), neighbors.shape[1], heads, width)
            value = self.attention_v(keys).reshape(len(ids), neighbors.shape[1], heads, width)
            scores = (query[:, None] * key).sum(-1) / math.sqrt(width)
            # Same Gaussian distance prior as G3, augmented by learned attention.
            scores = scores + weights.clamp_min(1e-12).log().unsqueeze(-1)
            valid = neighbors >= 0
            scores = scores.masked_fill(~valid.unsqueeze(-1), -1e4)
            attention = torch.softmax(scores, dim=1) * valid.unsqueeze(-1)
            attention = attention / attention.sum(1, keepdim=True).clamp_min(1e-12)
            message = (attention.unsqueeze(-1) * value).sum(1).reshape(len(ids), -1)
        graph = self.graph_out(torch.cat((root, message, present), dim=1))
        bounded = self.residual_bound * torch.tanh((correction + graph) / self.residual_bound)
        logits = logits + bounded
        return (logits, bounded) if return_delta else logits


def n_parameters(mode="G3"):
    return sum((param.numel() for param in NucleiFusion(mode=mode).parameters()))
