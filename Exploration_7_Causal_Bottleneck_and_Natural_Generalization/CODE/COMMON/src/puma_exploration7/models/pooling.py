from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from ..constants import EMBED_DIM, UNI2_SPATIAL_START, UNI2_TOKEN_COUNT


def _check(tokens):
    if tokens.ndim != 3 or tokens.shape[1:] != (UNI2_TOKEN_COUNT, EMBED_DIM):
        raise ValueError('tokens must be Bx265x1536')


def gaussian_pool(tokens, u, v, sigma=1.5):
    _check(tokens)
    if sigma <= 0:
        raise ValueError('sigma')
    b = len(tokens)
    u = torch.as_tensor(u, device=tokens.device, dtype=tokens.dtype).reshape(-1)
    v = torch.as_tensor(v, device=tokens.device, dtype=tokens.dtype).reshape(-1)
    yy, xx = torch.meshgrid(
        torch.arange(16, device=tokens.device, dtype=tokens.dtype),
        torch.arange(16, device=tokens.device, dtype=tokens.dtype),
        indexing='ij',
    )
    d2 = (xx.reshape(1, -1) - u[:, None]).square() + (yy.reshape(1, -1) - v[:, None]).square()
    ww = torch.exp(-d2 / (2 * sigma * sigma))
    ww = ww / ww.sum(1, keepdim=True)
    return torch.einsum('bn,bnd->bd', ww, tokens[:, UNI2_SPATIAL_START:])


class TokenPooler(nn.Module):
    def __init__(self, kind='gaussian', sigma=1.5):
        super().__init__()
        self.kind = kind
        self.sigma = float(sigma)

    def forward(self, tokens, u, v):
        if self.kind == 'gaussian':
            raw = gaussian_pool(tokens, u, v, self.sigma)
        elif self.kind == 'cls':
            raw = tokens[:, 0]
        else:
            raise ValueError(self.kind)
        return F.layer_norm(raw.float(), (EMBED_DIM,)).to(tokens.dtype)
