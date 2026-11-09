"""Resource selection for preprocessing and inference."""

import os

import torch


batch_sizes = (2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1)


def cpu_workers():
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))

    return os.cpu_count() or 1


def batch_candidates(limit=2048):
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("Batch limit must be a positive integer")
    return [size for size in batch_sizes if size <= limit]


def gpu_batch_size(device, run, limit=2048):
    """Select the largest fitting inference batch, up to 2048."""
    candidates = batch_candidates(limit)
    if device.type != "cuda":
        return min(limit, 4)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    before = torch.cuda.memory_allocated(device)
    run(1)
    torch.cuda.synchronize(device)
    per_sample = max(1, torch.cuda.max_memory_allocated(device) - before)
    torch.cuda.empty_cache()

    for size in candidates:
        free, _ = torch.cuda.mem_get_info(device)
        if size > 1 and size * per_sample > free * 0.8:
            continue

        try:
            run(size)
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            return size
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()

    raise RuntimeError("No CUDA inference batch fits in memory")
