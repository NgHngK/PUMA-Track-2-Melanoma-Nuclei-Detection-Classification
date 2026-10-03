from __future__ import annotations
import torch
from CODE.COMMON.constants import TOKEN_GRID, NUM_PATCH_TOKENS, NUM_PREFIX_TOKENS, APPEARANCE_DIM
from CODE.COMMON.exceptions import TokenGeometryError


def extract_spatial_tokens(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3:
        raise TokenGeometryError(f"Expected [B,T,D], got {tuple(tokens.shape)}")
    if tokens.shape[-1] != APPEARANCE_DIM:
        raise TokenGeometryError(f"Expected dim {APPEARANCE_DIM}")
    needed = NUM_PREFIX_TOKENS + NUM_PATCH_TOKENS
    if tokens.shape[1] != needed:
        raise TokenGeometryError(
            f"Expected exactly {needed} tokens (CLS+8 regs+256 patches), got {tokens.shape[1]}"
        )
    spatial = tokens[:, NUM_PREFIX_TOKENS:, :]
    return spatial.reshape(tokens.shape[0], TOKEN_GRID, TOKEN_GRID, APPEARANCE_DIM)


def source_to_token_coordinate(crop_coordinate: float, fov: int) -> float:
    # Historical E4/E6 contract: u=(x-left)/(fov/16), v=(y-top)/(fov/16).
    # For integer-centred FOV96 crops this maps crop coordinate 48 to u=8.
    patch_source = fov / TOKEN_GRID
    return crop_coordinate / patch_source
