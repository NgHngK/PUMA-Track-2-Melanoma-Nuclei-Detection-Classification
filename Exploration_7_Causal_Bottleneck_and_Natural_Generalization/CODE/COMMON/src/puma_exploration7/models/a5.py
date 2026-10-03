from __future__ import annotations
from torch import nn
from ..constants import EMBED_DIM, NUM_CLASSES, TIER_A_DIM


class A5Head(nn.Module):
    """Exact Exploration-3/6 rank-r appearance x Tier-A interaction head."""

    def __init__(self, interaction_rank: int = 8, use_tier_a: bool = True):
        super().__init__()
        self.rank = int(interaction_rank)
        self.use_tier_a = bool(use_tier_a)
        if self.rank < 1:
            raise ValueError('rank must be positive')
        self.head = nn.Linear(EMBED_DIM, NUM_CLASSES)
        if self.use_tier_a:
            self.ph = nn.Linear(EMBED_DIM, self.rank, bias=False)
            self.pb = nn.Linear(TIER_A_DIM, self.rank, bias=False)
            self.branch = nn.Linear(self.rank, NUM_CLASSES, bias=False)
            nn.init.zeros_(self.branch.weight)
        else:
            self.ph = self.pb = self.branch = None

    def forward(self, h, bio=None):
        if h.ndim != 2 or h.shape[1] != EMBED_DIM:
            raise ValueError('h must be Bx1536')
        app = self.head(h)
        if not self.use_tier_a:
            return app
        if bio is None or bio.shape != (len(h), TIER_A_DIM):
            raise ValueError('Tier-A must be Bx16')
        return app + self.branch(self.ph(h) * self.pb(bio))

    def reset_classifier_outputs(self):
        self.head.reset_parameters()
        if self.branch is not None:
            nn.init.zeros_(self.branch.weight)

    def freeze_feature_transform(self):
        if self.ph is not None:
            self.ph.requires_grad_(False)
        if self.pb is not None:
            self.pb.requires_grad_(False)

    def classifier_parameters(self):
        yield from self.head.parameters()
        if self.branch is not None:
            yield from self.branch.parameters()
