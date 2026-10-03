from __future__ import annotations
import gc, os, platform, shutil
from pathlib import Path
import torch


def environment_report(path='.') -> dict:
    du = shutil.disk_usage(Path(path))
    report = {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda,
        'cpu_count': os.cpu_count(),
        'free_disk_bytes': du.free,
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        report.update(
            gpu=torch.cuda.get_device_name(0),
            vram_total_bytes=int(total),
            vram_free_bytes=int(free),
            vram_allocated_bytes=int(torch.cuda.memory_allocated()),
            vram_reserved_bytes=int(torch.cuda.memory_reserved()),
        )
    else:
        report.update(
            gpu=None,
            vram_total_bytes=0,
            vram_free_bytes=0,
            vram_allocated_bytes=0,
            vram_reserved_bytes=0,
        )
    return report


def release_cuda(*objects, empty_cache: bool = True) -> None:
    # Caller should delete its own strong references; this helper handles allocator cleanup.
    del objects
    gc.collect()
    if torch.cuda.is_available():
        if empty_cache:
            torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def cuda_memory_stats() -> dict:
    if not torch.cuda.is_available():
        return {'cuda': False}
    free, total = torch.cuda.mem_get_info()
    return {
        'cuda': True,
        'free_bytes': int(free),
        'total_bytes': int(total),
        'allocated_bytes': int(torch.cuda.memory_allocated()),
        'reserved_bytes': int(torch.cuda.memory_reserved()),
        'peak_allocated_bytes': int(torch.cuda.max_memory_allocated()),
        'peak_reserved_bytes': int(torch.cuda.max_memory_reserved()),
    }


def can_reside_on_cuda(
    nbytes: int, reserve_fraction: float = 0.25, reserve_bytes: int = 2 * 1024**3
) -> bool:
    if not torch.cuda.is_available():
        return False
    free, total = torch.cuda.mem_get_info()
    reserve = max(int(total * reserve_fraction), int(reserve_bytes))
    return int(nbytes) < max(0, int(free) - reserve)
