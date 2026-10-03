from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'COMMON/src'
sys.path.insert(0, str(SRC))

import argparse, json, numpy as np
from puma_exploration7.manifest import read_manifest, strongest_group_column
from puma_exploration7.probes import cross_group_knn_purity, grouped_linear_probe
from puma_exploration7.io import atomic_json_dump
from puma_exploration7.cache_integrity import verify_cache

p = argparse.ArgumentParser()
p.add_argument('--manifest', required=True)
p.add_argument('--features', required=True)
p.add_argument('--out', required=True)
a = p.parse_args()
df = read_manifest(a.manifest)
verify_cache(a.features, df, 'gaussian_features', 1536)
X = np.load(a.features, mmap_mode='r')
g = strongest_group_column(df)
res = {
    'cross_group_knn5_purity': cross_group_knn_purity(X, df.label, df[g], 5),
    'grouped_linear_probe': grouped_linear_probe(X, df.label, df[g], 3, 17),
    'group_column': g,
}
atomic_json_dump(a.out, res)
print(json.dumps(res, indent=2))
