from __future__ import annotations
from torch import nn
import torch.nn.functional as F


class A5Head(nn.Module):
    def __init__(self, appearance_dim=1536, bio_dim=16, rank=8, n_classes=10):
        super().__init__()
        self.appearance = nn.Linear(appearance_dim, n_classes)
        self.proj_h = nn.Linear(appearance_dim, rank, bias=False)
        self.proj_b = nn.Linear(bio_dim, rank, bias=False)
        self.out = nn.Linear(rank, n_classes, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, h, b):
        h = F.layer_norm(h, (h.shape[-1],))
        return self.appearance(h) + self.out(self.proj_h(h) * self.proj_b(b))

    @property
    def expected_trainable_parameters(self):
        return 27866


class AppearanceHead(nn.Module):
    def __init__(self, d=1536, n=10):
        super().__init__()
        self.linear = nn.Linear(d, n)

    def forward(self, h):
        return self.linear(F.layer_norm(h, (h.shape[-1],)))
