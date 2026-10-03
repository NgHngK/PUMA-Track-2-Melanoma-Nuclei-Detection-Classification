from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'COMMON/src'))
import argparse, json, numpy as np
from puma_exploration7.manifest import (
    read_manifest,
    assert_external_group_disjoint,
    strongest_group_column,
    assert_confirmatory_manifest,
)
from puma_exploration7.checkpoint import load_checkpoint_map, load_exploration6_head
from puma_exploration7.group_diversity import (
    build_matched_diversity_indices,
    build_matched_density_indices,
)
from puma_exploration7.training import train_crt_output, infer_a5
from puma_exploration7.metrics import semantic_metrics, puma_metrics
from puma_exploration7.io import atomic_json_dump
from puma_exploration7.cache_integrity import verify_cache

p = argparse.ArgumentParser()
p.add_argument('--train-manifest', required=True)
p.add_argument('--features', required=True)
p.add_argument('--tier-a', required=True)
p.add_argument('--test-manifest', required=True)
p.add_argument('--test-features', required=True)
p.add_argument('--test-tier-a', required=True)
p.add_argument('--checkpoint-map', required=True)
p.add_argument('--out-dir', required=True)
p.add_argument('--group-col', default='auto')
p.add_argument('--rows-per-class', type=int, default=40)
p.add_argument('--low-groups', type=int, default=2)
p.add_argument('--high-groups', type=int, default=8)
p.add_argument('--density-small', type=int, default=20)
p.add_argument('--epochs', type=int, default=10)
p.add_argument('--seeds', default='17,29,43')
p.add_argument('--device', default='cuda')
a = p.parse_args()
tr = read_manifest(a.train_manifest)
te = read_manifest(a.test_manifest)
assert_confirmatory_manifest(tr, 'train')
assert_confirmatory_manifest(te, 'test')
assert_external_group_disjoint(tr, te)
g = strongest_group_column(tr, a.group_col)
for path, df, kind, dim, meta in [
    (a.features, tr, 'gaussian_features', 1536, {'fov': 96, 'sigma': 1.5}),
    (a.tier_a, tr, 'tier_a', 16, None),
    (a.test_features, te, 'gaussian_features', 1536, {'fov': 96, 'sigma': 1.5}),
    (a.test_tier_a, te, 'tier_a', 16, None),
]:
    verify_cache(path, df, kind, dim, meta)
X = np.load(a.features, mmap_mode='r')
rawT = np.load(a.tier_a, mmap_mode='r')
Xt = np.load(a.test_features, mmap_mode='r')
rawTt = np.load(a.test_tier_a, mmap_mode='r')
cmap = load_checkpoint_map(a.checkpoint_map)
out = Path(a.out_dir)
out.mkdir(parents=True, exist_ok=True)
div, audit = build_matched_diversity_indices(
    tr, g, a.low_groups, a.high_groups, a.rows_per_class, 17
)
den, daudit = build_matched_density_indices(
    tr, g, a.high_groups, a.density_small, a.rows_per_class, 17
)
audit.to_csv(out / 'diversity_subset_audit.csv', index=False)
daudit.to_csv(out / 'density_subset_audit.csv', index=False)
res = {'group_column': g, 'diversity': {}, 'density': {}}
labels = tr.label.to_numpy(int)
seeds = [int(x) for x in a.seeds.split(',') if x.strip()]
if set(seeds) - set(cmap):
    raise ValueError('checkpoint map does not cover requested seeds')
# Normalizers are seed-specific, so normalize once per seed and reuse across every arm.
normed = {}
for seed in seeds:
    _, _, norm = load_exploration6_head(cmap[seed], 'cpu', expected_seed=seed)
    normed[seed] = (norm.transform(rawT), norm.transform(rawTt))
for design, arms in [('diversity', div), ('density', den)]:
    for name, ix in arms.items():
        runs = []
        for seed in seeds:
            T, Tt = normed[seed]
            m, h = train_crt_output(
                X[ix],
                T[ix],
                labels[ix],
                cmap[seed],
                out / f'{design}_{name}_s{seed}.pt',
                seed=seed,
                epochs=a.epochs,
                device=a.device,
            )
            z = infer_a5(m, Xt, Tt, device=a.device)
            runs.append(
                {
                    'seed': seed,
                    'checkpoint': str(cmap[seed]),
                    'n': len(ix),
                    'groups': int(tr.iloc[ix][g].nunique()),
                    'semantic': semantic_metrics(te.label, z),
                    'puma': puma_metrics(te, te.label, z),
                    'history': h,
                }
            )
        res[design][name] = runs
atomic_json_dump(out / 'result.json', res)
print(json.dumps({'status': 'ok', 'group_column': g, 'seeds': seeds, 'epochs': a.epochs}, indent=2))
