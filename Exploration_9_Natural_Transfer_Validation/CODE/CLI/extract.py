from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from CODE.DATA.manifest import read_manifest
from CODE.FEATURES.uni2_loader import load_uni2_h, EXPECTED_ENCODER_SHA256
from CODE.FEATURES.feature_contract import build_contract
from CODE.FEATURES.cache_writer import FeatureCacheWriter
from CODE.FEATURES.cache_reader import FeatureCache
from CODE.FEATURES.extract_driver import extract_multiview_to_caches, autotune_microbatch
from CODE.COMMON.constants import APPEARANCE_DIM, LOCAL_FOV, WIDE_FOV
from CODE.COMMON.hashing import canonical_json_hash, sha256_file, sha256_strings
from CODE.COMMON.exceptions import CheckpointIdentityError


def _contract_hash_payload(fov: int):
    return canonical_json_hash(
        {
            "image_source": "1024x1024 ROI TIFF",
            "fov": int(fov),
            "padding": "white",
            "crop_origin": "floor(center-fov/2+0.5)",
            "resize": "PIL bicubic 224x224",
            "rgb_range": "float32 [0,1]",
            "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "encoder_output": "final normalized token sequence",
            "spatial_tokens": "indices 9..264 reshaped row-major 16x16",
            "coordinate": "crop_coordinate/(fov/16)",
            "pool": "Gaussian sigma=1.5 token units normalized to unit mass",
        }
    )


def _cache_files(out: Path, name: str):
    return [
        out / f"{name}.npy",
        out / f"{name}.uids.txt",
        out / f"{name}.json",
        out / f"{name}.extraction.json",
        out / f"{name}.json.tmp",
    ]


def _complete_cache_matches(out: Path, name: str, digest: str, expected_uids: list[str]) -> bool:
    meta_path = out / f"{name}.json"
    if not meta_path.is_file():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("status") != "COMPLETE":
        return False
    cache = FeatureCache(out, name, verify_hash=True)
    if cache.meta["contract"]["encoder_sha256"] != digest:
        raise ValueError(f"{name} was extracted with a different UNI2 checkpoint")
    if cache.meta["uid_order_sha256"] != sha256_strings(expected_uids):
        raise ValueError(f"{name} UID order differs from the current manifest")
    return True


def main():
    # Keep the scientific path in true FP32. Throughput optimizations below do not
    # enable BF16/FP16/TF32, which would change numerical representation.
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
    p = argparse.ArgumentParser(description="Extract E9 Gaussian96/Gaussian192 UNI2-h caches")
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--microbatch",
        default="auto",
        help="integer or auto; auto finds the largest safe row batch",
    )
    p.add_argument(
        "--max-microbatch",
        type=int,
        default=0,
        help="optional hard ceiling; 0 means no artificial cap (search up to all remaining rows)",
    )
    p.add_argument(
        "--cpu-workers", type=int, default=0, help="0 uses all available logical CPU cores"
    )
    p.add_argument("--features", nargs="+", choices=["local", "wide"], default=["local", "wide"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--allow-unknown-checkpoint-hash",
        action="store_true",
        help="NOT recommended; changes encoder provenance",
    )
    a = p.parse_args()

    rows = read_manifest(a.manifest, require_images=True, transfer_only=True)
    expected_uids = [r.uid for r in rows]
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    specs = []
    if "local" in a.features:
        specs.append(("gaussian96", LOCAL_FOV))
    if "wide" in a.features:
        specs.append(("gaussian192", WIDE_FOV))

    if a.overwrite:
        for name, _ in specs:
            for path in _cache_files(out, name):
                path.unlink(missing_ok=True)

    digest = sha256_file(a.checkpoint)
    if not a.allow_unknown_checkpoint_hash and digest != EXPECTED_ENCODER_SHA256:
        raise CheckpointIdentityError(f"UNI2-h SHA256 mismatch: {digest}")

    complete = {
        name: _complete_cache_matches(out, name, digest, expected_uids) for name, _ in specs
    }
    for name, is_complete in complete.items():
        if is_complete:
            print(f"[SKIP] {name}: cache already COMPLETE", flush=True)
    if all(complete.values()):
        print("[DONE] All requested UNI2 feature caches are already complete.", flush=True)
        return

    model, loaded_digest = load_uni2_h(
        a.checkpoint, enforce_known_hash=not a.allow_unknown_checkpoint_hash
    )
    if loaded_digest != digest:
        raise RuntimeError("Checkpoint digest changed while loading")
    model = model.to(a.device)

    pending_specs = [(name, fov) for name, fov in specs if not complete[name]]
    pending_fovs = tuple(fov for _, fov in pending_specs)

    contracts = {
        fov: build_contract(name, fov, digest, _contract_hash_payload(fov))
        for name, fov in pending_specs
    }
    writers = {
        fov: FeatureCacheWriter(
            out,
            name,
            len(rows),
            APPEARANCE_DIM,
            contracts[fov],
            expected_uids=expected_uids,
        )
        for name, fov in pending_specs
    }

    if str(a.microbatch).lower() == "auto":
        tune = autotune_microbatch(
            model=model,
            rows=rows,
            fovs=pending_fovs,
            device=a.device,
            max_microbatch=None if int(a.max_microbatch) <= 0 else int(a.max_microbatch),
            cpu_workers=a.cpu_workers,
        )
        microbatch = int(tune["microbatch"])
    else:
        microbatch = int(a.microbatch)
        tune = {
            "microbatch": microbatch,
            "images_per_forward": microbatch * len(pending_fovs),
            "mode": "manual",
            "views": list(pending_fovs),
            "probes": [],
        }

    print(
        f"[AUTOTUNE] row_microbatch={microbatch} | views={len(pending_fovs)} | "
        f"images/UNI2-forward={microbatch * len(pending_fovs)} | mode={tune['mode']}",
        flush=True,
    )

    metas = extract_multiview_to_caches(
        model=model,
        rows=rows,
        writers=writers,
        initial_microbatch=microbatch,
        device=a.device,
        cpu_workers=a.cpu_workers,
    )

    results = {}
    for name, fov in pending_specs:
        meta = metas[fov]
        engineering = {
            "device": a.device,
            "requested_microbatch": a.microbatch,
            "autotune": tune,
            "final_microbatch": meta.get("final_microbatch"),
            "cpu_workers": meta.get("cpu_workers"),
            "oom_events": meta.get("oom_events", []),
            "encoder_sha256": digest,
            "compute_precision": "fp32",
            "tf32": False,
            "fused_view_extraction": len(pending_fovs) > 1,
        }
        (out / f"{name}.extraction.json").write_text(
            json.dumps(engineering, indent=2), encoding="utf-8"
        )
        results[name] = {
            "n": len(rows),
            "fov": fov,
            "payload_sha256": meta["payload_sha256"],
            "microbatch": microbatch,
            "final_microbatch": meta.get("final_microbatch"),
        }
        print(f"[COMPLETE] {name}: {len(rows)}/{len(rows)} rows", flush=True)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
