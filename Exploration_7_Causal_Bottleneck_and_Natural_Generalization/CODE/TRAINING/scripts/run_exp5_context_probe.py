from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'COMMON/src'))

import argparse
import json
import numpy as np

from puma_exploration7.cache_integrity import verify_cache
from puma_exploration7.context_probe import compare_context
from puma_exploration7.io import atomic_json_dump
from puma_exploration7.manifest import (
    assert_confirmatory_manifest,
    read_manifest,
    strongest_group_column,
)

p = argparse.ArgumentParser()
p.add_argument('--manifest', required=True)
p.add_argument('--local-features', required=True)
p.add_argument('--context-features', required=True)
p.add_argument('--out', required=True)
p.add_argument('--group-col', default='auto')
p.add_argument('--seeds', default='17,29,43')
a = p.parse_args()

df = read_manifest(a.manifest)
assert_confirmatory_manifest(df, 'train')
group_col = strongest_group_column(df, a.group_col)
verify_cache(a.local_features, df, 'gaussian_features', 1536, {'fov': 96, 'sigma': 1.5})
verify_cache(a.context_features, df, 'gaussian_features', 1536, {'fov': 192, 'sigma': 1.5})

pairs = [(7, 0), (0, 1), (0, 3), (5, 9)]
local = np.load(a.local_features, mmap_mode='r')
context = np.load(a.context_features, mmap_mode='r')
labels = df.label.to_numpy(int)
groups = df[group_col].to_numpy()
seeds = [int(x.strip()) for x in a.seeds.split(',') if x.strip()]
if not seeds:
    raise ValueError('at least one probe seed is required')

per_seed = {
    str(seed): compare_context(local, context, labels, groups, pairs, seed) for seed in seeds
}
all_local_deltas = [
    delta for result in per_seed.values() for delta in result['fold_deltas_vs_local']
]
all_placebo_deltas = [
    delta for result in per_seed.values() for delta in result['fold_deltas_vs_placebo']
]
seed_mean_deltas = [float(np.mean(result['fold_deltas_vs_local'])) for result in per_seed.values()]
res = {
    'group_column': group_col,
    'seeds': seeds,
    'per_seed': per_seed,
    'mean_delta_vs_local': float(np.mean(all_local_deltas)),
    'seed_mean_deltas_vs_local': seed_mean_deltas,
    'min_seed_mean_delta_vs_local': float(min(seed_mean_deltas)),
    'all_seed_mean_deltas_positive': bool(all(x > 0 for x in seed_mean_deltas)),
    'mean_delta_vs_placebo': float(np.mean(all_placebo_deltas)),
}
atomic_json_dump(a.out, res)
print(
    json.dumps(
        {
            'status': 'ok',
            'seeds': seeds,
            'mean_delta_vs_local': res['mean_delta_vs_local'],
            'mean_delta_vs_placebo': res['mean_delta_vs_placebo'],
        },
        indent=2,
    )
)
