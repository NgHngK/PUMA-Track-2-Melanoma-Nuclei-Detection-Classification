from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'COMMON' / 'src'))

import argparse, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from puma_exploration8.cache_integrity import aligned_cache
from puma_exploration8.config import load_json
from puma_exploration8.grouping import load_fold_payload, fold_vector
from puma_exploration8.manifest import read_manifest
from puma_exploration8.metrics import classification_metrics
from puma_exploration8.stageio import manifest_sha256, cache_expected


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--cache-dir', required=True)
    p.add_argument('--folds', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--strict-hash', action='store_true')
    a = p.parse_args()
    cfg = load_json(ROOT / 'CONFIGS' / '40_P8D_WIDER_FIELD_AUDIT.json')
    df = read_manifest(a.manifest, require_images=False)
    tr = df[df.role == 'TRAIN'].reset_index(drop=True)
    uids = tr.uid.astype(str).tolist()
    rois = tr.roi.astype(str).to_numpy()
    y = tr.label.to_numpy(int)
    mh = manifest_sha256(a.manifest)
    cd = Path(a.cache_dir)
    names = ['local_gaussian96', 'gaussian192', 'annular192', 'scale_only192']
    arr = {n: aligned_cache(cd / n, uids, cache_expected(mh, n), a.strict_hash) for n in names}
    folds = load_fold_payload(a.folds, df, 'TRAIN')
    fv = fold_vector(df, folds, 'TRAIN')
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(arr['local_gaussian96'].shape[1])
    signs = rng.choice([-1.0, 1.0], size=len(perm))
    arr['local_placebo'] = arr['local_gaussian96'][:, perm] * signs[None, :]
    pred = {
        n: np.zeros((len(y), 10), float)
        for n in [
            'local_only',
            'gaussian192',
            'annular192',
            'scale_only192',
            'shuffle',
            'local_placebo',
            'within_roi_swap',
        ]
    }

    def fit_predict(xtr, ytr, xva):
        sc = StandardScaler()
        xt = sc.fit_transform(xtr)
        xv = sc.transform(xva)
        clf = LogisticRegression(max_iter=600, C=1.0, solver='lbfgs')
        clf.fit(xt, ytr)
        out = np.full((len(xva), 10), 1e-12, float)
        out[:, clf.classes_.astype(int)] = clf.predict_proba(xv)
        out /= out.sum(1, keepdims=True)
        return out

    def isolated_shuffle(extra, idx, seed):
        rr = np.random.default_rng(seed)
        return extra[idx][rr.permutation(len(idx))]

    def within_roi_swap(extra, idx, seed):
        rr = np.random.default_rng(seed)
        out = extra[idx].copy()
        subset_rois = rois[idx]
        for roi in np.unique(subset_rois):
            loc = np.where(subset_rois == roi)[0]
            if len(loc) > 1:
                out[loc] = out[loc][rr.permutation(len(loc))]
        return out

    local = arr['local_gaussian96']
    wide = arr['gaussian192']
    for fi in range(int(folds['k'])):
        va, ta = np.where(fv == fi)[0], np.where(fv != fi)[0]
        pred['local_only'][va] = fit_predict(local[ta], y[ta], local[va])
        for name in ['gaussian192', 'annular192', 'scale_only192', 'local_placebo']:
            extra = arr[name]
            pred[name][va] = fit_predict(
                np.concatenate([local[ta], extra[ta]], 1),
                y[ta],
                np.concatenate([local[va], extra[va]], 1),
            )
        gtr = isolated_shuffle(wide, ta, a.seed + 1000 * fi + 1)
        gva = isolated_shuffle(wide, va, a.seed + 1000 * fi + 2)
        pred['shuffle'][va] = fit_predict(
            np.concatenate([local[ta], gtr], 1), y[ta], np.concatenate([local[va], gva], 1)
        )
        rtr = within_roi_swap(wide, ta, a.seed + 2000 * fi + 1)
        rva = within_roi_swap(wide, va, a.seed + 2000 * fi + 2)
        pred['within_roi_swap'][va] = fit_predict(
            np.concatenate([local[ta], rtr], 1), y[ta], np.concatenate([local[va], rva], 1)
        )
    results = {
        name: classification_metrics(y, np.log(np.clip(prob, 1e-12, 1)))
        for name, prob in pred.items()
    }
    results['_provenance'] = {
        'frozen_fold_sha256': folds['sha256'],
        'probe': cfg['probe'],
        'seed': a.seed,
        'cache_contract': 'exploration8-v2',
    }
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'P8D.json').write_text(json.dumps(results, indent=2))
    print(
        json.dumps(
            {k: v['macro_f1_fixed10'] for k, v in results.items() if not k.startswith('_')},
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
