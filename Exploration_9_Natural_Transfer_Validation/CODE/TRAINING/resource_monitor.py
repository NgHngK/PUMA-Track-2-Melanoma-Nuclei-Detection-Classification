from __future__ import annotations
import os
import time
import torch

try:
    import resource
except ImportError:  # Windows has no Unix resource module.
    resource = None


class ResourceMonitor:
    def __enter__(self):
        self.start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.wall_seconds = time.perf_counter() - self.start
        self.peak_rss_kb = (
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if resource is not None else None
        )
        self.cuda_peak_allocated = (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        )
        self.cuda_peak_reserved = (
            int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None
        )
        self.cpu_count = os.cpu_count()

    def as_dict(self):
        return {
            'wall_seconds': float(self.wall_seconds),
            'peak_rss_kb': self.peak_rss_kb,
            'cuda_peak_allocated': self.cuda_peak_allocated,
            'cuda_peak_reserved': self.cuda_peak_reserved,
            'cpu_count': self.cpu_count,
        }
