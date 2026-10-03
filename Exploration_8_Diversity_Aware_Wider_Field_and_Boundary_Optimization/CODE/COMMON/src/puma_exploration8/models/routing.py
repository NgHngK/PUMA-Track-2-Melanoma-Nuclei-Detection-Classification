from __future__ import annotations
import math, torch
from torch import nn
import torch.nn.functional as F


class GlobalResidualScale(nn.Module):
    def __init__(self, max_scale: float = 1.0, initial_fraction: float = 0.25):
        super().__init__()
        self.max_scale = float(max_scale)
        f = max(-0.95, min(0.95, initial_fraction))
        self.raw = nn.Parameter(torch.tensor(math.atanh(f), dtype=torch.float32))

    def forward(self):
        return self.max_scale * torch.tanh(self.raw)


class WiderResidualA(nn.Module):
    def __init__(self, d=1536, n=10, max_scale=1.0):
        super().__init__()
        self.proj = nn.Linear(d, n, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.scale = GlobalResidualScale(max_scale)

    def forward(self, h_local, h_wide):
        return self.scale() * self.proj(F.layer_norm(h_wide, (h_wide.shape[-1],)))


class WiderInteractionC(nn.Module):
    def __init__(self, d=1536, rank=8, n=10, max_scale=1.0):
        super().__init__()
        self.ql = nn.Linear(d, rank, bias=False)
        self.qw = nn.Linear(d, rank, bias=False)
        self.out = nn.Linear(rank, n, bias=False)
        nn.init.zeros_(self.out.weight)
        self.scale = GlobalResidualScale(max_scale)

    def forward(self, h_local, h_wide):
        l = F.layer_norm(h_local, (h_local.shape[-1],))
        w = F.layer_norm(h_wide, (h_wide.shape[-1],))
        return self.scale() * self.out(self.ql(l) * self.qw(w))
