import torch
from torch import nn


class Anchor(nn.Module):
    """Local appearance branch plus the low-rank Tier-A interaction."""

    def __init__(self, d=1536, b=16, r=8, c=10):
        super().__init__()
        self.ln = nn.LayerNorm(d, elementwise_affine=False, eps=1e-05)
        self.appearance = nn.Linear(d, c)
        self.p_h = nn.Linear(d, r, bias=False)
        self.p_b = nn.Linear(b, r, bias=False)
        self.residual = nn.Linear(r, c, bias=False)
        nn.init.zeros_(self.residual.weight)

    def forward(self, h, tier, normalized=None):
        x = self.ln(h) if normalized is None else normalized
        interaction = self.p_h(x) * self.p_b(tier)
        return self.appearance(x) + self.residual(interaction)


class AnchorContextC(nn.Module):
    """Local and wide context: local appearance + Tier-A + local/wide context."""

    def __init__(self, d=1536, b=16, r=8, c=10):
        super().__init__()
        self.anchor = Anchor(d, b, r, c)
        self.ln_local = nn.LayerNorm(d, elementwise_affine=False, eps=1e-05)
        self.ln_wide = nn.LayerNorm(d, elementwise_affine=False, eps=1e-05)
        self.q_local = nn.Linear(d, r, bias=False)
        self.q_wide = nn.Linear(d, r, bias=False)
        self.context_out = nn.Linear(r, c, bias=False)
        self.context_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, h_local, h_wide, tier):
        local = self.ln_local(h_local)
        logits = self.anchor(h_local, tier, normalized=local)
        local_context = self.q_local(local)
        wide_context = self.q_wide(self.ln_wide(h_wide))
        context = self.context_out(local_context * wide_context)
        return logits + torch.tanh(self.context_scale) * context
