from __future__ import annotations
from torch import nn
from .a5 import A5Head, AppearanceHead


class Stage2CachedModel(nn.Module):
    def __init__(self, use_tiera=True, wider_route: nn.Module | None = None):
        super().__init__()
        self.use_tiera = use_tiera
        self.local = A5Head() if use_tiera else AppearanceHead()
        self.wider_route = wider_route

    def forward(self, h_local, bio=None, h_wide=None):
        z = self.local(h_local, bio) if self.use_tiera else self.local(h_local)
        if self.wider_route is not None:
            if h_wide is None:
                raise ValueError('wider feature required')
            z = z + self.wider_route(h_local, h_wide)
        if z.ndim != 2 or z.shape[1] != 10:
            raise RuntimeError('logits must be N x 10')
        return z
