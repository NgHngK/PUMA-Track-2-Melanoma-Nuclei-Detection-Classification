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
from puma_exploration7.training import train_crt_output, infer_a5
from puma_exploration7.metrics import semantic_metrics, puma_metrics, paired_roi_bootstrap
from puma_exploration7.io import atomic_json_dump
from puma_exploration7.experiments import assess_candidate
from puma_exploration7.replication import should_replicate, aggregate_seed_results
from puma_exploration7.cache_integrity import verify_cache

p = argparse.ArgumentParser()
p.add_argument('--train-manifest', required=True)
p.add_argument('--train-features', required=True)
p.add_argument('--train-tier-a', required=True)
p.add_argument('--test-manifest', required=True)
p.add_argument('--test-features', required=True)
p.add_argument('--test-tier-a', required=True)
p.add_argument('--checkpoint-map', required=True)
p.add_argument('--out-dir', required=True)
p.add_argument('--device', default='cuda')
p.add_argument('--epochs', type=int, default=10)
p.add_argument('--batch-size', type=int, default=64)
p.add_argument(
    '--replication-policy', choices=['on_primary_pass', 'always'], default='on_primary_pass'
)
a = p.parse_args()
tr = read_manifest(a.train_manifest)
te = read_manifest(a.test_manifest)
assert_confirmatory_manifest(tr, 'train')
assert_confirmatory_manifest(te, 'test')
assert_external_group_disjoint(tr, te)
for path, df, kind, dim, meta in [
    (a.train_features, tr, 'gaussian_features', 1536, {'fov': 96, 'sigma': 1.5}),
    (a.train_tier_a, tr, 'tier_a', 16, None),
    (a.test_features, te, 'gaussian_features', 1536, {'fov': 96, 'sigma': 1.5}),
    (a.test_tier_a, te, 'tier_a', 16, None),
]:
    verify_cache(path, df, kind, dim, meta)
Xtr = np.load(a.train_features, mmap_mode='r')
Xte = np.load(a.test_features, mmap_mode='r')
rawTtr = np.load(a.train_tier_a, mmap_mode='r')
rawTte = np.load(a.test_tier_a, mmap_mode='r')
labels = tr.label.to_numpy(int)
cmap = load_checkpoint_map(a.checkpoint_map)
out = Path(a.out_dir)
out.mkdir(parents=True, exist_ok=True)
per = {}


def run_seed(seed):
    sd = out / f'seed{seed}'
    sd.mkdir(exist_ok=True)
    base, ck, norm = load_exploration6_head(cmap[seed], a.device, expected_seed=seed)
    Ttr = norm.transform(rawTtr)
    Tte = norm.transform(rawTte)
    z0 = infer_head(base, Xte, Tte, device=a.device)
    model, hist = train_crt_output(
        Xtr,
        Ttr,
        labels,
        cmap[seed],
        sd / 'crt.pt',
        seed=seed,
        epochs=a.epochs,
        batch_size=a.batch_size,
        device=a.device,
    )
    z1 = infer_a5(model, Xte, Tte, device=a.device)
    b = {'semantic': semantic_metrics(te.label, z0), 'puma': puma_metrics(te, te.label, z0)}
    c = {'semantic': semantic_metrics(te.label, z1), 'puma': puma_metrics(te, te.label, z1)}
    paired = paired_roi_bootstrap(
        b['puma']['per_roi_fixed10_macro_f1'], c['puma']['per_roi_fixed10_macro_f1'], seed=seed
    )
    res = {
        'seed': seed,
        'checkpoint': str(cmap[seed]),
        'baseline': b,
        'natural_crt': c,
        'paired_roi_bootstrap': paired,
        'promotion_gate': assess_candidate(b, c, paired),
        'history': hist,
    }
    atomic_json_dump(sd / 'result.json', res)
    return res


per[17] = run_seed(17)
replicated = should_replicate(per[17], a.replication_policy)
if replicated:
    for seed in (29, 43):
        per[seed] = run_seed(seed)
agg = (
    aggregate_seed_results(per, 'natural_crt')
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
