from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'COMMON/src'))
import argparse, json, numpy as np
from puma_exploration7.manifest import (
    read_manifest,
    assert_external_group_disjoint,
    assert_confirmatory_manifest,
)
from puma_exploration7.checkpoint import load_checkpoint_map, load_exploration6_head, infer_head
from puma_exploration7.cache_integrity import verify_cache
from puma_exploration7.experiments import run_exp1
from puma_exploration7.replication import should_replicate, aggregate_seed_results
from puma_exploration7.io import atomic_json_dump, save_prediction_csv

p = argparse.ArgumentParser()
p.add_argument('--calibration-manifest', required=True)
p.add_argument('--calibration-features', required=True)
p.add_argument('--calibration-tier-a', required=True)
p.add_argument('--test-manifest', required=True)
p.add_argument('--test-features', required=True)
p.add_argument('--test-tier-a', required=True)
p.add_argument('--checkpoint-map', required=True)
p.add_argument('--out-dir', required=True)
p.add_argument('--device', default='cuda')
p.add_argument('--source-prior', default='uniform')
p.add_argument(
    '--replication-policy', choices=['on_primary_pass', 'always'], default='on_primary_pass'
)
p.add_argument('--save-primary-predictions', action='store_true')
a = p.parse_args()
cal = read_manifest(a.calibration_manifest)
te = read_manifest(a.test_manifest)
assert_confirmatory_manifest(cal, 'calibration')
assert_confirmatory_manifest(te, 'test')
assert_external_group_disjoint(cal, te)
for path, df, kind, dim, meta in [
    (a.calibration_features, cal, 'gaussian_features', 1536, {'fov': 96, 'sigma': 1.5}),
    (a.calibration_tier_a, cal, 'tier_a', 16, None),
    (a.test_features, te, 'gaussian_features', 1536, {'fov': 96, 'sigma': 1.5}),
    (a.test_tier_a, te, 'tier_a', 16, None),
]:
    verify_cache(path, df, kind, dim, meta)
Xc = np.load(a.calibration_features, mmap_mode='r')
Xt = np.load(a.test_features, mmap_mode='r')
Tc = np.load(a.calibration_tier_a, mmap_mode='r')
Tt = np.load(a.test_tier_a, mmap_mode='r')
cmap = load_checkpoint_map(a.checkpoint_map)
src = (
    np.ones(10) / 10
    if a.source_prior == 'uniform'
    else np.asarray(json.loads(Path(a.source_prior).read_text())['prior'], float)
)
out = Path(a.out_dir)
out.mkdir(parents=True, exist_ok=True)
per = {}


def run_seed(seed):
    head, ck, norm = load_exploration6_head(cmap[seed], a.device, expected_seed=seed)
    zc = infer_head(head, Xc, norm.transform(Tc), device=a.device)
    zt = infer_head(head, Xt, norm.transform(Tt), device=a.device)
    res = run_exp1(cal, zc, te, zt, src, seed=seed)
    res['checkpoint'] = str(cmap[seed])
    res['checkpoint_seed'] = seed
    atomic_json_dump(out / f'seed{seed}.json', res)
    if seed == 17 and a.save_primary_predictions:
        cols = [
            c
            for c in ['uid', 'roi', 'group', 'patient', 'case', 'slide', 'image', 'x', 'y']
            if c in cal.columns
        ]
        save_prediction_csv(out / 'calibration_seed17.csv', cal[cols], zc)
        cols = [
            c
            for c in ['uid', 'roi', 'group', 'patient', 'case', 'slide', 'image', 'x', 'y']
            if c in te.columns
        ]
        save_prediction_csv(out / 'test_seed17.csv', te[cols], zt)
    return res


per[17] = run_seed(17)
replicated = should_replicate(per[17], a.replication_policy)
if replicated:
    for seed in (29, 43):
        per[seed] = run_seed(seed)
agg = (
    aggregate_seed_results(per, 'prior_corrected')
    if len(per) == 3
    else {
        'status': 'skipped',
        'reason': 'primary seed17 promotion gate did not pass',
        'seeds': sorted(per),
    }
)
final = {
    'primary_seed': 17,
    'replication_policy': a.replication_policy,
    'replication_triggered': replicated,
    'per_seed': {str(k): v for k, v in per.items()},
    'replication_summary': agg,
}
atomic_json_dump(out / 'result.json', final)
print(
    json.dumps(
        {
            'status': 'ok',
            'seed17_passed': per[17]['promotion_gate']['passed'],
            'replication_triggered': replicated,
            'replication_passed': agg.get('replication_passed'),
        },
        indent=2,
    )
)
