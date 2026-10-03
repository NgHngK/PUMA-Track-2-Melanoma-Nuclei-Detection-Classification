"""CPU regression checks for real frozen folds and paired roster invariants."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import json
import numpy as np
from puma_exploration8.manifest import read_manifest
from puma_exploration8.grouping import load_fold_payload, fold_vector
from puma_exploration8.diversity import plan_classes, paired_rosters, validate_pairs, roster_hash


def main():
    root = Path(__file__).resolve().parents[3]
    q = root / 'INPUTS/QUICK_BENCHMARK'
    df = read_manifest(q / 'EXPLORATION8_QUICK_MANIFEST.csv', require_images=False)
    tr = df[df.role == 'TRAIN'].reset_index(drop=True)
    folds = load_fold_payload(q / 'TRAIN_FOLDS.json', df, 'TRAIN')
    fv = fold_vector(df, folds, 'TRAIN')
    cfg = json.loads((root / 'CODE/CONFIGS/30_P8C_DIVERSITY_DENSITY.json').read_text())
    plan = plan_classes(tr, fv, cfg)
    count = 0
    for seed in cfg['roster_seeds']:
        for k in range(folds['k']):
            ta = tr.iloc[np.where(fv != k)[0]]
            arms = paired_rosters(ta, plan, seed)
            validate_pairs(arms, ta, plan)
            repeat = paired_rosters(ta.sample(frac=1, random_state=3), plan, seed)
            assert all(roster_hash(x) == roster_hash(repeat[n]) for n, x in arms.items())
            assert roster_hash(arms['div_high']) == roster_hash(arms['density_high'])
            for x in arms.values():
                assert set(x.roi).isdisjoint(set(tr.loc[fv == k, 'roi']))
            count += len(arms)
    bad = {name: x.copy() for name, x in arms.items()}
    bad['div_low'].loc[0, 'uid'] = 'unseen-held-out-uid'
    try:
        validate_pairs(bad, ta, plan)
    except ValueError:
        pass
    else:
        raise AssertionError('foreign UID accepted')
    missing = tr[tr.label != 2].copy()
    try:
        plan_classes(missing, fv[tr.label != 2], cfg)
    except ValueError:
        pass
    else:
        raise AssertionError('missing class accepted')
    print(
        f'PASS: {count} real-data rosters; deterministic under row reorder; no fold leakage; invalid inputs rejected'
    )


if __name__ == '__main__':
    main()
