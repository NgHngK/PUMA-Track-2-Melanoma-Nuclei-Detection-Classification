from __future__ import annotations
from torch import nn
from .pooling import TokenPooler
from .a5 import A5Head


class Stage2FromTokens(nn.Module):
    def __init__(self, sigma=1.5, rank=8):
        super().__init__()
        self.pooler = TokenPooler('gaussian', sigma)
        self.head = A5Head(rank, True)

    def forward(self, tokens, bio, u, v):
        return self.head(self.pooler(tokens, u, v), bio)
