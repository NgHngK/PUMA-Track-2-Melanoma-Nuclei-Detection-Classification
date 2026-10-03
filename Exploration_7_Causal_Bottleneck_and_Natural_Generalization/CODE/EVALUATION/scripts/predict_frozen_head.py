from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'COMMON/src'
sys.path.insert(0, str(SRC))
import argparse, numpy as np
from puma_exploration7.manifest import read_manifest
from puma_exploration7.checkpoint import load_exploration6_head, infer_head
from puma_exploration7.io import save_prediction_csv
from puma_exploration7.cache_integrity import verify_cache

p = argparse.ArgumentParser()
p.add_argument('--manifest', required=True)
p.add_argument('--features', required=True)
p.add_argument('--tier-a', required=True)
p.add_argument('--checkpoint', required=True)
p.add_argument('--out', required=True)
p.add_argument('--device', default='cpu')
a = p.parse_args()
df = read_manifest(a.manifest)
verify_cache(a.features, df, 'gaussian_features', 1536, {'fov': 96, 'sigma': 1.5})
verify_cache(a.tier_a, df, 'tier_a', 16)
X = np.load(a.features, mmap_mode='r')
T = np.load(a.tier_a, mmap_mode='r')
head, ck, norm = load_exploration6_head(a.checkpoint, a.device)
z = infer_head(head, X, norm.transform(T), device=a.device)
meta = df[
    [
        c
        for c in ['uid', 'roi', 'group', 'patient', 'case', 'slide', 'image', 'x', 'y']
        if c in df.columns
    ]
].copy()
save_prediction_csv(a.out, meta, z)
print(a.out)
