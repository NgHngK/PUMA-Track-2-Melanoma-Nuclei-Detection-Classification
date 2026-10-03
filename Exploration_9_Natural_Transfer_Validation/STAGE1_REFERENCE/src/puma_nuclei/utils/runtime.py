from __future__ import annotations

import gc
import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(slots=True)
class RuntimeState:
    device: torch.device
    amp_dtype: torch.dtype
    amp_enabled: bool


def configure_cpu_parallelism(requested_workers: int = -1) -> int:
    """Use the requested logical CPUs without oversubscribing PyTorch pools."""
    workers = dataloader_worker_count(requested_workers)
    threads = max(1, workers)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(min(4, threads))
    except RuntimeError:
        # PyTorch only permits changing this pool before parallel work starts.
        pass
    return workers


def choose_amp_dtype(use_bfloat16: bool = True) -> torch.dtype:
    if torch.cuda.is_available() and use_bfloat16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def configure_runtime(
    seed: int,
    *,
    use_bfloat16: bool = True,
    allow_tf32: bool = True,
    deterministic: bool = False,
) -> RuntimeState:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # PYTHONHASHSEED is read when the interpreter starts. Setting it here would
    # misleadingly suggest it affects the current process, so only validate the
    # external setting when strict determinism was requested.
    if deterministic:
        hash_seed = os.environ.get("PYTHONHASHSEED")
        if hash_seed != str(seed):
            actual = "unset" if hash_seed is None else hash_seed
            raise RuntimeError(
                f"Strict deterministic mode requires PYTHONHASHSEED={seed} before Python starts; currently {actual}. "
                "Restart Python with the matching environment variable."
            )
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
        # Prefer fused scaled-dot-product attention kernels whenever this
        # PyTorch/CUDA build and GPU support them. Unsupported backends simply
        # fall back to the math implementation.
        cuda_backends = getattr(torch.backends, "cuda", None)
        for name in ("enable_flash_sdp", "enable_mem_efficient_sdp"):
            setter = getattr(cuda_backends, name, None)
            if callable(setter):
                setter(True)
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_amp_dtype(use_bfloat16)
    return RuntimeState(device=device, amp_dtype=dtype, amp_enabled=device.type == "cuda")



def make_grad_scaler(*, enabled: bool) -> object:
    """Construct a CUDA GradScaler across the supported PyTorch 2.2+ range."""
    amp = getattr(torch, "amp", None)
    scaler_cls = getattr(amp, "GradScaler", None) if amp is not None else None
    if scaler_cls is not None:
        try:
            return scaler_cls("cuda", enabled=bool(enabled))
        except TypeError:
            # Some transitional releases exposed GradScaler without device arg.
            try:
                return scaler_cls(enabled=bool(enabled))
            except TypeError:
                pass
    cuda_amp = getattr(getattr(torch, "cuda", None), "amp", None)
    legacy_cls = getattr(cuda_amp, "GradScaler", None) if cuda_amp is not None else None
    if legacy_cls is None:
        if enabled:
            raise RuntimeError("This PyTorch build does not provide a CUDA GradScaler")
        # Disabled scaling is still needed for a uniform training loop. Current
        # supported PyTorch CUDA builds always provide one of the APIs above.
        raise RuntimeError("This PyTorch build does not provide any GradScaler implementation")
    return legacy_cls(enabled=bool(enabled))

def dataloader_worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def dataloader_worker_count(requested: int) -> int:
    cores = os.cpu_count() or 2
    if requested < 0:
        return cores
    return max(0, min(int(requested), cores))

def shutdown_dataloader(loader: object | None) -> None:
    """Deterministically stop persistent DataLoader workers when a loader is finished."""
    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        try:
            shutdown()
        except (AssertionError, OSError, ValueError):
            # The loader is already tearing down. Avoid masking the actual
            # training result with a multiprocessing cleanup race.
            pass
    try:
        loader._iterator = None  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        pass


def release_unused_memory() -> None:
    """Release unreachable Python objects and unused cached CUDA memory at safe boundaries."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
