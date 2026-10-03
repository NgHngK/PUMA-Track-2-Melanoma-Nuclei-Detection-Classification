from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'COMMON/src'
sys.path.insert(0, str(SRC))
import argparse, numpy as np
from puma_exploration7.manifest import read_manifest
from puma_exploration7.tier_a import compute_tier_a, NAMES
from puma_exploration7.cache_integrity import write_cache_sidecar

p = argparse.ArgumentParser()
p.add_argument('--manifest', required=True)
p.add_argument('--out', required=True)
a = p.parse_args()
df = read_manifest(a.manifest, require_images=True)
x = compute_tier_a(df.to_dict('records'))
np.save(a.out, x)
write_cache_sidecar(a.out, a.manifest, df, 'tier_a', shape=list(x.shape), feature_names=list(NAMES))
print(x.shape)
