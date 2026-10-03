from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import argparse, hashlib, json, os, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from puma_exploration7.checkpoint import load_checkpoint_map, write_checkpoint_map
from puma_exploration7.manifest import read_manifest, assert_confirmatory_manifest


def _copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and dst.stat().st_size == src.stat().st_size:
        return
    tmp = dst.with_suffix(dst.suffix + '.part')
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def _copy_hash(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    tmp = dst.with_suffix(dst.suffix + '.part')
    with src.open('rb') as f, tmp.open('wb') as g:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(block)
            g.write(block)
    os.replace(tmp, dst)
    return h.hexdigest()


def _staged_name(src: Path):
    return hashlib.sha1(str(src).encode()).hexdigest()[:16] + '_' + src.name


p = argparse.ArgumentParser(
    description='Stage immutable Exploration-7 runtime inputs to fast Colab local storage without changing scientific content.'
)
p.add_argument('--train-manifest', required=True)
p.add_argument('--calibration-manifest', required=True)
p.add_argument('--test-manifest', required=True)
p.add_argument('--weights', required=True)
p.add_argument('--checkpoint-map', required=True)
p.add_argument('--runtime-root', default='/content/PUMA_P7_RUNTIME')
p.add_argument('--out', required=True)
p.add_argument('--image-workers', type=int, default=4)
p.add_argument('--images', choices=['auto', 'always', 'never'], default='auto')
a = p.parse_args()
rt = Path(a.runtime_root)
rt.mkdir(parents=True, exist_ok=True)
roles = {'train': a.train_manifest, 'calibration': a.calibration_manifest, 'test': a.test_manifest}
frames = {}
for role, path in roles.items():
    df = read_manifest(path, require_images=True)
    assert_confirmatory_manifest(df, role, require_images=True)
    frames[role] = df
# Stage checkpoints (tiny) and validate both source and staged copies.
cmap = load_checkpoint_map(a.checkpoint_map, require_all=True)
local_ck = {}
for seed, src in cmap.items():
    dst = rt / 'checkpoints' / f'prompt6_seed{seed}.pt'
    _copy_file(Path(src), dst)
    local_ck[seed] = dst
local_map = rt / 'EXPLORATION6_CHECKPOINTS.json'
write_checkpoint_map(local_map, local_ck)
load_checkpoint_map(local_map, require_all=True)
# One sequential read of the 2.7GB weight file both stages and hashes it.
wsrc = Path(a.weights)
wdst = rt / 'weights' / wsrc.name
if wdst.is_file() and wdst.stat().st_size == wsrc.stat().st_size:
    h = hashlib.sha256()
    with wdst.open('rb') as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(block)
    weight_sha = h.hexdigest()
else:
    weight_sha = _copy_hash(wsrc, wdst)
# Decide whether raw images fit comfortably in ephemeral storage.
unique = sorted({str(x) for df in frames.values() for x in df['image'].astype(str).unique()})
image_bytes = sum(Path(x).stat().st_size for x in unique)
free = shutil.disk_usage(rt).free
stage_images = (a.images == 'always') or (a.images == 'auto' and image_bytes < free * 0.70)
image_map = {}
if stage_images:
    image_dir = rt / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)

    def job(src_s):
        src = Path(src_s)
        dst = image_dir / _staged_name(src)
        _copy_file(src, dst)
        return src_s, str(dst)

    with ThreadPoolExecutor(max_workers=max(1, int(a.image_workers))) as ex:
        futs = [ex.submit(job, x) for x in unique]
        for fut in as_completed(futs):
            k, v = fut.result()
            image_map[k] = v
# Write deterministic execution manifests; source_image keeps cache identity relocation-safe.
manifest_paths = {}
for role, df in frames.items():
    z = df.copy()
    if 'source_image' not in z.columns:
        z['source_image'] = z['image'].astype(str)
    if stage_images:
        z['image'] = z['source_image'].map(image_map)
    dst = rt / 'manifests' / f'{role}.csv'
    dst.parent.mkdir(parents=True, exist_ok=True)
    z.to_csv(dst, index=False)
    manifest_paths[role] = str(dst)
payload = {
    'runtime_root': str(rt),
    'manifests': manifest_paths,
    'uni2_weights': str(wdst),
    'uni2_weights_sha256': weight_sha,
    'checkpoint_map': str(local_map),
    'images_staged': stage_images,
    'unique_images': len(unique),
    'image_bytes': image_bytes,
    'free_bytes_before_images': free,
}
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(payload, indent=2), encoding='utf-8')
print(json.dumps(payload, indent=2))
