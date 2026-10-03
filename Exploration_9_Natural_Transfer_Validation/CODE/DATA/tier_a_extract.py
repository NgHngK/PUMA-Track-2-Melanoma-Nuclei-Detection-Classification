from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from scipy.ndimage import laplace

from CODE.COMMON.constants import TIER_A_DIM
from CODE.COMMON.hashing import sha256_file, sha256_strings, canonical_json_hash

TIER_A_COLUMNS = (
    "H_mean",
    "H_std",
    "H_p10",
    "H_p50",
    "H_p90",
    "H_gradient_mean",
    "H_gradient_std",
    "H_abs_laplacian_mean",
    "H_entropy32",
    "H_center_minus_ring",
    "gray_center_minus_ring",
    "H_ring_std",
    "central_valid_fraction",
    "ring_valid_fraction",
    "H_robust_z",
    "H_ROI_percentile",
)
assert len(TIER_A_COLUMNS) == TIER_A_DIM

_H_VECTOR = np.asarray([0.650, 0.704, 0.286], dtype=np.float32)
_OFF = np.arange(-24, 25, dtype=np.int32)
_DY, _DX = np.meshgrid(_OFF, _OFF, indexing="ij")
_R2 = _DX * _DX + _DY * _DY
_CENTRAL_MASK = _R2 <= 12 * 12
_RING_MASK = (_R2 >= 16 * 16) & (_R2 <= 24 * 24)
_CENTRAL_NOMINAL = int(_CENTRAL_MASK.sum())
_RING_NOMINAL = int(_RING_MASK.sum())


def h_proxy_from_rgb(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(rgb, dtype=np.float32) / 255.0
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got {x.shape}")
    od = -np.log(np.clip(x, 1.0 / 255.0, 1.0))
    h = np.clip((od @ _H_VECTOR) / 2.0, 0.0, 2.0).astype(np.float32)
    gray = x.mean(axis=2, dtype=np.float32)
    return h, gray


def _entropy32(values: np.ndarray) -> float:
    counts, _ = np.histogram(values, bins=32, range=(0.0, 2.0))
    total = counts.sum()
    if total <= 0:
        raise ValueError("Entropy received empty values")
    p = counts[counts > 0].astype(np.float64) / float(total)
    return float(-(p * np.log(p)).sum())


def _roi_maps(rgb: np.ndarray):
    h, gray = h_proxy_from_rgb(rgb)
    gy, gx = np.gradient(h)
    grad = np.sqrt(gx * gx + gy * gy + 1e-8).astype(np.float32)
    abs_lap = np.abs(laplace(h, mode="reflect")).astype(np.float32)
    med = float(np.median(h))
    q25, q75 = np.percentile(h, [25, 75])
    iqr = max(float(q75 - q25), 1e-4)
    sorted_h = np.sort(h.reshape(-1))
    return h, gray, grad, abs_lap, med, iqr, sorted_h


def point_features(maps, x: float, y: float) -> np.ndarray:
    h, gray, grad, abs_lap, roi_med, roi_iqr, sorted_h = maps
    height, width = h.shape
    cx = int(math.floor(float(x) + 0.5))
    cy = int(math.floor(float(y) + 0.5))
    xs = cx + _DX
    ys = cy + _DY
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    central = valid & _CENTRAL_MASK
    ring = valid & _RING_MASK
    if not central.any() or not ring.any():
        raise ValueError(f"No valid Tier-A central/ring support at {(x,y)}")
    cys, cxs = ys[central], xs[central]
    rys, rxs = ys[ring], xs[ring]
    hc = h[cys, cxs]
    hr = h[rys, rxs]
    gc = gray[cys, cxs]
    gr = gray[rys, rxs]
    gradc = grad[cys, cxs]
    lapc = abs_lap[cys, cxs]
    hmean = float(hc.mean())
    percentile = float(np.searchsorted(sorted_h, hmean, side="right") / len(sorted_h))
    out = np.asarray(
        [
            hmean,
            float(hc.std(ddof=0)),
            *[float(v) for v in np.percentile(hc, [10, 50, 90])],
            float(gradc.mean()),
            float(gradc.std(ddof=0)),
            float(lapc.mean()),
            _entropy32(hc),
            hmean - float(hr.mean()),
            float(gc.mean()) - float(gr.mean()),
            float(hr.std(ddof=0)),
            float(central.sum()) / _CENTRAL_NOMINAL,
            float(ring.sum()) / _RING_NOMINAL,
            (hmean - roi_med) / roi_iqr,
            percentile,
        ],
        dtype=np.float32,
    )
    if out.shape != (TIER_A_DIM,) or not np.isfinite(out).all():
        raise ValueError("Invalid Tier-A vector")
    return out


def _extract_one_roi(task):
    path, entries = task
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
    maps = _roi_maps(rgb)
    result = []
    for i, x, y in entries:
        result.append((i, point_features(maps, x, y)))
    return result


def extract_tier_a(rows, *, progress_every_rois: int = 10, workers: int = 0) -> np.ndarray:
    if not rows:
        raise ValueError("No rows")
    by_image = OrderedDict()
    for i, r in enumerate(rows):
        path = str(r.image_path if hasattr(r, "image_path") else r["image_path"])
        x = float(r.x if hasattr(r, "x") else r["x"])
        y = float(r.y if hasattr(r, "y") else r["y"])
        by_image.setdefault(path, []).append((i, x, y))
    out = np.empty((len(rows), TIER_A_DIM), dtype=np.float32)
    tasks = list(by_image.items())
    max_workers = int(workers) if int(workers) > 0 else max(1, int(os.cpu_count() or 1))
    # ROI images are independent. Threads keep memory bounded and let NumPy/SciPy kernels use
    # native parallelism without duplicating the full Python process/model state.
    if max_workers == 1 or len(tasks) == 1:
        iterator = map(_extract_one_roi, tasks)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=min(max_workers, len(tasks)))
        iterator = executor.map(_extract_one_roi, tasks)
    try:
        for n, result in enumerate(iterator, 1):
            for i, vec in result:
                out[i] = vec
            if progress_every_rois and (n % progress_every_rois == 0 or n == len(tasks)):
                print(
                    f"[Tier-A] processed {n}/{len(tasks)} ROI images with {min(max_workers,len(tasks))} worker(s)",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite Tier-A output")
    return out


def write_tier_a(directory: str | Path, rows, arr: np.ndarray) -> dict:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr, dtype=np.float32)
    uids = [str(r.uid if hasattr(r, "uid") else r["uid"]) for r in rows]
    if arr.shape != (len(uids), TIER_A_DIM):
        raise ValueError("Tier-A shape/row mismatch")
    np.save(d / "tier_a.npy", arr)
    (d / "tier_a.uids.txt").write_text("\n".join(uids) + "\n", encoding="utf-8")
    meta = {
        "status": "COMPLETE",
        "shape": list(arr.shape),
        "dtype": "float32",
        "columns": list(TIER_A_COLUMNS),
        "uid_order_sha256": sha256_strings(uids),
        "payload_sha256": sha256_file(d / "tier_a.npy"),
        "schema_sha256": canonical_json_hash(list(TIER_A_COLUMNS)),
        "definition": "historical E2/E3 Tier-A 16-D RGB/point schema; raw inference-safe per-ROI features; no cross-row standardization applied here",
    }
    (d / "tier_a.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return meta


def load_tier_a(
    directory: str | Path, target_uids: list[str], verify_hash: bool = True
) -> np.ndarray:
    d = Path(directory)
    meta = json.loads((d / "tier_a.json").read_text(encoding="utf-8"))
    if meta.get("status") != "COMPLETE":
        raise ValueError("Tier-A cache incomplete")
    uids = (d / "tier_a.uids.txt").read_text(encoding="utf-8").splitlines()
    if sha256_strings(uids) != meta["uid_order_sha256"]:
        raise ValueError("Tier-A UID hash mismatch")
    if verify_hash and sha256_file(d / "tier_a.npy") != meta["payload_sha256"]:
        raise ValueError("Tier-A payload hash mismatch")
    arr = np.load(d / "tier_a.npy", mmap_mode="r")
    if arr.shape != tuple(meta["shape"]) or arr.dtype != np.float32:
        raise ValueError("Tier-A shape/dtype mismatch")
    pos = {u: i for i, u in enumerate(uids)}
    missing = [u for u in target_uids if u not in pos]
    if missing:
        raise ValueError(f"Tier-A missing UID {missing[0]}")
    return np.asarray(arr[[pos[u] for u in target_uids]], dtype=np.float32)


def extract_tier_a_resumable(
    directory: str | Path, rows, *, progress_every_rois: int = 10, workers: int = 0
) -> dict:
    """Extract Tier-A with ROI-level resume checkpoints written directly to the final cache."""
    if not rows:
        raise ValueError("No rows")
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    data_path = d / 'tier_a.npy'
    uid_path = d / 'tier_a.uids.txt'
    meta_path = d / 'tier_a.json'
    uids = [str(r.uid if hasattr(r, 'uid') else r['uid']) for r in rows]
    uid_hash = sha256_strings(uids)
    schema_hash = canonical_json_hash(list(TIER_A_COLUMNS))

    by_roi = OrderedDict()
    for i, r in enumerate(rows):
        roi_id = str(r.roi_id if hasattr(r, 'roi_id') else r['roi_id'])
        path = str(r.image_path if hasattr(r, 'image_path') else r['image_path'])
        x = float(r.x if hasattr(r, 'x') else r['x'])
        y = float(r.y if hasattr(r, 'y') else r['y'])
        item = by_roi.setdefault(roi_id, {'path': path, 'entries': []})
        if item['path'] != path:
            raise ValueError(f'ROI {roi_id} maps to multiple image paths')
        item['entries'].append((i, x, y))

    completed = []
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta.get('status') == 'COMPLETE':
            load_tier_a(d, uids, verify_hash=True)
            print('[SKIP] Tier-A cache already COMPLETE', flush=True)
            return meta
        if meta.get('uid_order_sha256') != uid_hash or meta.get('schema_sha256') != schema_hash:
            raise ValueError('Tier-A incomplete cache does not match current rows/schema')
        if list(meta.get('shape', [])) != [len(rows), TIER_A_DIM]:
            raise ValueError('Tier-A incomplete cache shape mismatch')
        if not data_path.is_file():
            raise ValueError('Tier-A incomplete metadata exists but tier_a.npy is missing')
        out = np.load(data_path, mmap_mode='r+')
        completed = list(meta.get('completed_rois', []))
        print(f'[RESUME] Tier-A: {len(completed)}/{len(by_roi)} ROIs complete', flush=True)
    else:
        out = np.lib.format.open_memmap(
            data_path, mode='w+', dtype=np.float32, shape=(len(rows), TIER_A_DIM)
        )

    def write_progress():
        progress = {
            'status': 'INCOMPLETE',
            'shape': [len(rows), TIER_A_DIM],
            'dtype': 'float32',
            'columns': list(TIER_A_COLUMNS),
            'uid_order_sha256': uid_hash,
            'schema_sha256': schema_hash,
            'completed_rois': completed,
            'definition': 'historical E2/E3 Tier-A 16-D RGB/point schema; raw inference-safe per-ROI features; no cross-row standardization applied here',
        }
        tmp = meta_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(progress, indent=2, sort_keys=True), encoding='utf-8')
        os.replace(tmp, meta_path)

    if not meta_path.is_file():
        write_progress()

    completed_set = set(completed)
    pending = [(roi_id, info) for roi_id, info in by_roi.items() if roi_id not in completed_set]
    tasks = [(info['path'], info['entries']) for _, info in pending]
    max_workers = int(workers) if int(workers) > 0 else max(1, int(os.cpu_count() or 1))
    if max_workers == 1 or len(tasks) <= 1:
        iterator = map(_extract_one_roi, tasks)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=min(max_workers, len(tasks)))
        iterator = executor.map(_extract_one_roi, tasks)
    try:
        for n, ((roi_id, _), result) in enumerate(zip(pending, iterator), 1):
            for i, vec in result:
                out[i] = vec
            out.flush()
            completed.append(roi_id)
            write_progress()
            done = len(completed)
            if progress_every_rois and (done % progress_every_rois == 0 or done == len(by_roi)):
                print(f'[Tier-A] processed {done}/{len(by_roi)} ROI images', flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if len(completed) != len(by_roi):
        raise ValueError('Tier-A extraction ended before all ROIs completed')
    if not np.isfinite(np.asarray(out)).all():
        raise ValueError('Nonfinite Tier-A output')
    out.flush()
    uid_path.write_text('\n'.join(uids) + '\n', encoding='utf-8')
    meta = {
        'status': 'COMPLETE',
        'shape': [len(rows), TIER_A_DIM],
        'dtype': 'float32',
        'columns': list(TIER_A_COLUMNS),
        'uid_order_sha256': uid_hash,
        'payload_sha256': sha256_file(data_path),
        'schema_sha256': schema_hash,
        'definition': 'historical E2/E3 Tier-A 16-D RGB/point schema; raw inference-safe per-ROI features; no cross-row standardization applied here',
    }
    tmp = meta_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding='utf-8')
    os.replace(tmp, meta_path)
    print('[COMPLETE] Tier-A cache', flush=True)
    return meta
