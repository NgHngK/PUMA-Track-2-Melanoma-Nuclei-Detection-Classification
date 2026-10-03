from __future__ import annotations
import torch
from torch import nn
from CODE.COMMON.constants import APPEARANCE_DIM, TIER_A_DIM, INTERACTION_RANK, NUM_CLASSES
from CODE.COMMON.schemas import ModelOutput
from CODE.COMMON.validation import require_shape, require_finite


class Anchor(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(APPEARANCE_DIM, elementwise_affine=False)
        self.appearance = nn.Linear(APPEARANCE_DIM, NUM_CLASSES, bias=True)
        self.p_h = nn.Linear(APPEARANCE_DIM, INTERACTION_RANK, bias=False)
        self.p_b = nn.Linear(TIER_A_DIM, INTERACTION_RANK, bias=False)
        self.residual = nn.Linear(INTERACTION_RANK, NUM_CLASSES, bias=False)
        nn.init.zeros_(self.residual.weight)

    def _logits_components(self, h_local: torch.Tensor, tier_a: torch.Tensor):
        h = self.ln(h_local)
        z_app = self.appearance(h)
        z_a5 = self.residual(self.p_h(h) * self.p_b(tier_a))
        return z_app + z_a5, z_app, z_a5

    def forward_logits_fast(self, h_local: torch.Tensor, tier_a: torch.Tensor) -> torch.Tensor:
        """Same FP32 model equation as forward(), without hot-path debug syncs."""
        z, _, _ = self._logits_components(h_local, tier_a)
        return z

    def forward(self, h_local: torch.Tensor, tier_a: torch.Tensor) -> ModelOutput:
        require_shape(h_local, (None, APPEARANCE_DIM), 'h_local')
        require_shape(tier_a, (h_local.shape[0], TIER_A_DIM), 'tier_a')
        z, z_app, z_a5 = self._logits_components(h_local, tier_a)
        require_finite(z, 'anchor logits')
        return ModelOutput(z, z, z_app, z_a5, None, None)
