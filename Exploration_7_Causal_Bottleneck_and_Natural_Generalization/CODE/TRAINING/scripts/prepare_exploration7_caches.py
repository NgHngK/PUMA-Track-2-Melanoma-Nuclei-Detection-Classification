from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'COMMON/src'))
import argparse, json, numpy as np
from puma_exploration7.manifest import read_manifest, assert_confirmatory_manifest
from puma_exploration7.tier_a import compute_tier_a, NAMES
from puma_exploration7.cache_integrity import sidecar_path, write_cache_sidecar, verify_cache
from puma_exploration7.features import FrozenUNI2Extractor

p = argparse.ArgumentParser(
    description='Prepare all reusable Exploration-7 Tier-A/Gaussian caches with one resident UNI2-h load.'
)
p.add_argument('--train-manifest', required=True)
p.add_argument('--calibration-manifest', required=True)
p.add_argument('--test-manifest', required=True)
p.add_argument('--weights', required=True)
p.add_argument('--weights-sha256', default='')
p.add_argument('--out-dir', required=True)
p.add_argument('--with-context', action='store_true')
p.add_argument('--batch-size', type=int, default=0)
p.add_argument('--num-workers', type=int, default=-1)
p.add_argument('--device', default='cuda')
a = p.parse_args()
out = Path(a.out_dir)
out.mkdir(parents=True, exist_ok=True)


def cache_is_valid(target, df, kind, dim, meta=None):
    """Return True only for a cache that matches the current manifest and metadata.

    This is a cache *preparation* script, so stale/corrupt cache artifacts are inputs to
    rebuild rather than fatal pipeline errors.  Remove both the array and sidecar
    atomically-enough for the subsequent writer to recreate a consistent pair.
    """
    target = Path(target)
    if not target.exists():
        return False
    try:
        verify_cache(target, df, kind, dim, meta)
        return True
    except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError) as exc:
        print(f'[cache] stale -> rebuilding {target.name}: {type(exc).__name__}: {exc}', flush=True)
        for stale in (target, sidecar_path(target)):
            try:
                stale.unlink()
            except FileNotFoundError:
                pass
        return False


roles = {
    'train': Path(a.train_manifest),
    'calibration': Path(a.calibration_manifest),
    'test': Path(a.test_manifest),
}
frames = {}
for role, mp in roles.items():
    df = read_manifest(mp, require_images=True)
    assert_confirmatory_manifest(df, role, require_images=True)
    frames[role] = df
# Build Tier-A features on the CPU first so each ROI is decoded once per cohort.
for role, mp in roles.items():
    target = out / f'{role}_tierA.npy'
    df = frames[role]
    if not cache_is_valid(target, df, 'tier_a', 16):
        print(f'[cache] rebuilding {role} Tier-A ({len(df)} rows)...', flush=True)
        x = compute_tier_a(df.to_dict('records'))
        np.save(target, x)
        write_cache_sidecar(
            target, mp, df, 'tier_a', shape=list(x.shape), feature_names=list(NAMES)
        )
        print(f'[cache] ready {target.name} {tuple(x.shape)}', flush=True)
    else:
        print(f'[cache] reuse {target.name}', flush=True)
# Find the missing feature tasks before loading UNI2.
tasks = []
for role, mp in roles.items():
    target = out / f'{role}_gaussian_f96.npy'
    df = frames[role]
    expected = (
        {'fov': 96, 'sigma': 1.5, 'weights_sha256': a.weights_sha256}
        if a.weights_sha256
        else {'fov': 96, 'sigma': 1.5}
    )
    if not cache_is_valid(target, df, 'gaussian_features', 1536, expected):
        tasks.append((role, mp, target, 96))
if a.with_context:
    target = out / 'train_gaussian_f192.npy'
    df = frames['train']
    expected = (
        {'fov': 192, 'sigma': 1.5, 'weights_sha256': a.weights_sha256}
        if a.weights_sha256
        else {'fov': 192, 'sigma': 1.5}
    )
    if not cache_is_valid(target, df, 'gaussian_features', 1536, expected):
        tasks.append(('train_context', roles['train'], target, 192))
if tasks:
    print(
        f'[cache] {len(tasks)} UNI2 feature task(s) pending; loading frozen UNI2-h...', flush=True
    )
    extractor = FrozenUNI2Extractor(
        a.weights, a.device, a.batch_size, a.num_workers, a.weights_sha256 or None
    )
    try:
        for task_name, mp, target, fov in tasks:
            print(f'[cache] extracting {task_name}: fov={fov} -> {target.name}', flush=True)
            extractor.extract(mp, target, fov=fov, sigma=1.5)
            print(f'[cache] ready {target.name}', flush=True)
    finally:
        extractor.close()

# Run the same final checks used by Exp1-Exp5 so later cells can trust these caches.
for role, mp in roles.items():
    df = frames[role]
    verify_cache(out / f'{role}_tierA.npy', df, 'tier_a', 16)
    expected = (
        {'fov': 96, 'sigma': 1.5, 'weights_sha256': a.weights_sha256}
        if a.weights_sha256
        else {'fov': 96, 'sigma': 1.5}
    )
    verify_cache(out / f'{role}_gaussian_f96.npy', df, 'gaussian_features', 1536, expected)
if a.with_context:
    df = frames['train']
    expected = (
        {'fov': 192, 'sigma': 1.5, 'weights_sha256': a.weights_sha256}
        if a.weights_sha256
        else {'fov': 192, 'sigma': 1.5}
    )
    verify_cache(out / 'train_gaussian_f192.npy', df, 'gaussian_features', 1536, expected)
print('[cache] final downstream-contract verification passed', flush=True)
print(
    json.dumps(
        {
            'status': 'ok',
            'prepared': [str(x) for x in sorted(out.glob('*.npy'))],
            'feature_tasks_executed': len(tasks),
            'single_uni2_load': bool(tasks),
        },
        indent=2,
    )
)
