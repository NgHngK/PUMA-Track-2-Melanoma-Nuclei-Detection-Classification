from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import os

import torch

from CODE.FEATURES.cache_writer import FeatureCacheWriter
from CODE.FEATURES.extract_local import (
    ROIImageCache,
    prepare_multiview_batch,
    run_prepared_multiview,
)
from CODE.COMMON.exceptions import ResourceError


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        "cuda" in text and "out of memory" in text
    )


def _clear_after_oom():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cpu_workers(value: int) -> int:
    return max(1, int(value) if int(value) > 0 else int(os.cpu_count() or 1))


def autotune_microbatch(
    *,
    model,
    rows,
    fovs,
    device="cuda",
    max_microbatch: int | None = None,
    cpu_workers: int = 0,
    safety_fraction: float = 0.05,
    safety_bytes: int = 512 << 20,
) -> dict:
    """Find the largest safe *row* microbatch for the requested view set.

    There is no fixed upper cap by default. Search doubles 1,2,4,... until CUDA
    memory/headroom fails or every remaining row fits, then binary-searches the exact
    largest safe integer. With two FOVs, a row microbatch of N sends 2N images through
    UNI2 in the fused forward.
    """
    fovs = tuple(dict.fromkeys(int(v) for v in fovs))
    if not rows:
        raise ValueError("No rows for microbatch autotune")
    ceiling = (
        len(rows)
        if max_microbatch is None or int(max_microbatch) <= 0
        else min(len(rows), int(max_microbatch))
    )
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return {
            "microbatch": min(8, ceiling),
            "mode": "cpu_default",
            "views": list(fovs),
            "probes": [],
        }

    dev = torch.device(device)
    props = torch.cuda.get_device_properties(dev)
    total = int(props.total_memory)
    required_headroom = max(int(total * float(safety_fraction)), int(safety_bytes))
    cache = ROIImageCache(4)
    probes = []
    workers = _cpu_workers(cpu_workers)

    with ThreadPoolExecutor(max_workers=workers) as executor:

        def probe(n: int) -> bool:
            n = max(1, min(int(n), ceiling))
            prepared = None
            result = None
            try:
                prepared = prepare_multiview_batch(
                    rows[:n], fovs, image_cache=cache, executor=executor
                )
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(dev)
                result = run_prepared_multiview(model, prepared, device=device)
                torch.cuda.synchronize(dev)
                peak_alloc = int(torch.cuda.max_memory_allocated(dev))
                peak_reserved = int(torch.cuda.max_memory_reserved(dev))
                headroom = total - peak_reserved
                ok = headroom >= required_headroom
                probes.append(
                    {
                        "microbatch": n,
                        "images_per_forward": n * len(fovs),
                        "status": "PASS" if ok else "HEADROOM_FAIL",
                        "peak_allocated": peak_alloc,
                        "peak_reserved": peak_reserved,
                        "headroom": headroom,
                    }
                )
                return ok
            except RuntimeError as exc:
                if not is_cuda_oom(exc):
                    raise
                probes.append(
                    {"microbatch": n, "images_per_forward": n * len(fovs), "status": "OOM"}
                )
                return False
            finally:
                del prepared, result
                _clear_after_oom()

        good = 0
        bad = None
        n = 1
        while n <= ceiling:
            if probe(n):
                good = n
                if n == ceiling:
                    break
                n = min(ceiling, n * 2)
                if n == good:
                    break
            else:
                bad = n
                break
        if good == 0:
            cache.clear()
            raise ResourceError("No safe CUDA extraction microbatch, even batch=1")

        if bad is not None and bad - good > 1:
            lo, hi = good, bad
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if probe(mid):
                    lo = mid
                else:
                    hi = mid
            good = lo

    cache.clear()
    _clear_after_oom()
    return {
        "microbatch": int(good),
        "images_per_forward": int(good * len(fovs)),
        "mode": (
            "cuda_autotune_unbounded"
            if max_microbatch is None or int(max_microbatch) <= 0
            else "cuda_autotune_capped"
        ),
        "views": list(fovs),
        "cpu_workers": workers,
        "total_vram": total,
        "required_headroom": required_headroom,
        "search_ceiling": int(ceiling),
        "probes": probes,
    }


def _active_fovs(writers: dict[int, FeatureCacheWriter], n: int):
    pending = {fov: w for fov, w in writers.items() if int(w.next_index) < n}
    if not pending:
        return None
    pos = min(int(w.next_index) for w in pending.values())
    active = tuple(fov for fov, w in pending.items() if int(w.next_index) == pos)
    ahead = [int(w.next_index) for w in pending.values() if int(w.next_index) > pos]
    barrier = min(ahead) if ahead else n
    return pos, active, barrier


def extract_multiview_to_caches(
    *,
    model,
    rows,
    writers: dict[int, FeatureCacheWriter],
    initial_microbatch: int,
    device="cuda",
    cpu_workers: int = 0,
):
    """Resume-safe fused multiview extraction with CPU/GPU overlap.

    All CPU cores can prepare crops in parallel. After the first batch, the next batch
    is prepared while the GPU processes the current one. When both 96 and 192 views are
    pending at the same row, they share one ROI traversal and one concatenated UNI2 forward.
    """
    if initial_microbatch < 1:
        raise ValueError("microbatch must be >=1")
    n = len(rows)
    workers = _cpu_workers(cpu_workers)
    batch = int(initial_microbatch)
    oom_events = []
    image_cache = ROIImageCache(4)

    for fov, writer in writers.items():
        if writer.next_index:
            print(
                f"[RESUME] {writer.name}: {writer.next_index}/{n} rows already extracted",
                flush=True,
            )

    row_executor = ThreadPoolExecutor(max_workers=workers)
    batch_executor = ThreadPoolExecutor(max_workers=1)
    pending_future = None
    pending_spec = None

    def schedule(spec):
        pos, active, end = spec
        return batch_executor.submit(
            prepare_multiview_batch,
            rows[pos:end],
            active,
            image_cache=image_cache,
            executor=row_executor,
        )

    try:
        while True:
            state = _active_fovs(writers, n)
            if state is None:
                break
            pos, active, barrier = state
            end = min(pos + batch, barrier, n)
            spec = (pos, active, end)

            if pending_future is not None and pending_spec == spec:
                prepared = pending_future.result()
                pending_future = None
                pending_spec = None
            else:
                if pending_future is not None:
                    if not pending_future.cancel():
                        pending_future.result()
                    pending_future = None
                    pending_spec = None
                prepared = prepare_multiview_batch(
                    rows[pos:end], active, image_cache=image_cache, executor=row_executor
                )

            # Schedule the next aligned chunk before the current GPU forward so CPU crop
            # preparation overlaps GPU compute. OOM is exceptional after autotuning; if it
            # happens, the speculative batch is discarded and rebuilt at the smaller size.
            if end < n and all(int(writers[f].next_index) == pos for f in active):
                next_end = min(end + batch, barrier, n)
                if next_end > end:
                    pending_spec = (end, active, next_end)
                    pending_future = schedule(pending_spec)

            try:
                features = run_prepared_multiview(model, prepared, device=device)
                for fov in active:
                    writer = writers[fov]
                    writer.write(pos, features[fov], [r.uid for r in rows[pos:end]])
            except RuntimeError as exc:
                if not is_cuda_oom(exc):
                    raise
                oom_events.append(
                    {
                        "start": pos,
                        "failed_microbatch": end - pos,
                        "views": list(active),
                        "images_per_forward": (end - pos) * len(active),
                    }
                )
                if pending_future is not None:
                    if not pending_future.cancel():
                        pending_future.result()
                    pending_future = None
                    pending_spec = None
                _clear_after_oom()
                if batch <= 1:
                    raise ResourceError(f"CUDA OOM at microbatch=1, row {pos}") from exc
                batch = max(1, batch // 2)
                print(f"[OOM] reducing extraction row microbatch to {batch}", flush=True)
                continue
            finally:
                del prepared

        metas = {}
        for fov, writer in writers.items():
            meta = writer.finalize()
            meta["oom_events"] = list(oom_events)
            meta["final_microbatch"] = int(batch)
            meta["cpu_workers"] = int(workers)
            metas[fov] = meta
        return metas
    finally:
        if pending_future is not None:
            if not pending_future.cancel():
                pending_future.result()
        batch_executor.shutdown(wait=True, cancel_futures=True)
        row_executor.shutdown(wait=True, cancel_futures=True)
        image_cache.clear()
        _clear_after_oom()


def extract_to_cache(
    *,
    model,
    rows,
    fov,
    writer: FeatureCacheWriter,
    initial_microbatch: int = 64,
    device="cuda",
    cpu_workers: int = 0,
):
    """Compatibility single-view wrapper."""
    return extract_multiview_to_caches(
        model=model,
        rows=rows,
        writers={int(fov): writer},
        initial_microbatch=initial_microbatch,
        device=device,
        cpu_workers=cpu_workers,
    )[int(fov)]
