from __future__ import annotations
import hashlib, random
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derive_seed(base_seed: int, *parts) -> int:
    text = '|'.join(map(str, (base_seed,) + parts)).encode('utf-8')
    return int.from_bytes(hashlib.sha256(text).digest()[:8], 'little') % (2**31 - 1)
