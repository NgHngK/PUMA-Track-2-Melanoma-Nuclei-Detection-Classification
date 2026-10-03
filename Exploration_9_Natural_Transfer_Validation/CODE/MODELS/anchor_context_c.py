from __future__ import annotations
import torch
from torch import nn
from CODE.COMMON.constants import APPEARANCE_DIM, INTERACTION_RANK, NUM_CLASSES
from CODE.COMMON.schemas import ModelOutput
from CODE.COMMON.validation import require_shape, require_finite
from .anchor import Anchor


class AnchorContextC(nn.Module):
    def __init__(self, context_scale_init: float = 0.0):
        super().__init__()
        self.anchor = Anchor()
        self.ln_local = nn.LayerNorm(APPEARANCE_DIM, elementwise_affine=False)
        self.ln_wide = nn.LayerNorm(APPEARANCE_DIM, elementwise_affine=False)
        self.q_local = nn.Linear(APPEARANCE_DIM, INTERACTION_RANK, bias=False)
        self.q_wide = nn.Linear(APPEARANCE_DIM, INTERACTION_RANK, bias=False)
        self.context_out = nn.Linear(INTERACTION_RANK, NUM_CLASSES, bias=False)
        self.context_scale = nn.Parameter(torch.tensor(float(context_scale_init)))

    def _context_components(self, h_local: torch.Tensor, h_wide: torch.Tensor):
        ql = self.q_local(self.ln_local(h_local))
        qw = self.q_wide(self.ln_wide(h_wide))
        residual = self.context_out(ql * qw)
        gate = torch.tanh(self.context_scale)
        return gate * residual, gate

    def forward_logits_fast(
        self, h_local: torch.Tensor, h_wide: torch.Tensor, tier_a: torch.Tensor
    ) -> torch.Tensor:
        """Same FP32 model equation as forward(), without hot-path debug syncs."""
        anchor_logits = self.anchor.forward_logits_fast(h_local, tier_a)
        z_context, _ = self._context_components(h_local, h_wide)
        return anchor_logits + z_context

    def forward(
        self, h_local: torch.Tensor, h_wide: torch.Tensor, tier_a: torch.Tensor
    ) -> ModelOutput:
        require_shape(h_wide, (h_local.shape[0], APPEARANCE_DIM), 'h_wide')
        a = self.anchor(h_local, tier_a)
        z_context, gate = self._context_components(h_local, h_wide)
        z = a.logits + z_context
        require_finite(z, 'context logits')
        return ModelOutput(z, a.logits, a.appearance_logits, a.a5_logits, z_context, gate)
