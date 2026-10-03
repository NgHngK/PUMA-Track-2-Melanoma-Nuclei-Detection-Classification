from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'COMMON/src'
sys.path.insert(0, str(SRC))

import argparse, json, pandas as pd, numpy as np
from sklearn.metrics import cohen_kappa_score
from puma_exploration7.constants import CLASS_TO_ID
from puma_exploration7.io import atomic_json_dump

p = argparse.ArgumentParser(
    description='Analyze one or more blinded annotation-review CSVs against the hidden key.'
)
p.add_argument(
    '--review',
    required=True,
    nargs='+',
    help='One or more completed review_blinded.csv files, one per reviewer.',
)
p.add_argument('--key', required=True)
p.add_argument('--out', required=True)
a = p.parse_args()
key = pd.read_csv(a.key)


def parse(v):
    if pd.isna(v) or str(v).strip() == '':
        return None
    s = str(v).strip().lower().replace(' ', '_')
    if s.isdigit():
        i = int(s)
        return i if 0 <= i < 10 else None
    aliases = {
        'plasma': 'plasma_cell',
        'endothelial': 'endothelium',
        'epithelial': 'epithelium',
        'apoptotic': 'apoptosis',
    }
    s = aliases.get(s, s)
    return CLASS_TO_ID.get(s)


def is_uncertain(series):
    return (
        series.fillna('')
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(['1', 'true', 'yes', 'y', 'uncertain'])
    )


reviewers = []
per = []
for idx, path in enumerate(a.review, 1):
    r = pd.read_csv(path)
    if 'review_id' not in r or 'reviewed_label' not in r:
        raise ValueError(f'{path}: review_id and reviewed_label required')
    z = r.merge(key, on='review_id', validate='one_to_one')
    label = z.reviewed_label.map(parse)
    unc = is_uncertain(z['uncertain']) if 'uncertain' in z else pd.Series(False, index=z.index)
    valid = label.notna()
    certain = valid & ~unc
    name = Path(path).stem or f'reviewer_{idx}'
    per.append(
        {
            'reviewer': name,
            'n_rows': len(z),
            'n_reviewed': int(valid.sum()),
            'n_certain': int(certain.sum()),
            'uncertain_rate': float(unc[valid].mean()) if valid.any() else None,
            'original_label_confirmation_rate': (
                float(
                    (
                        label[certain].astype(int).to_numpy()
                        == z.loc[certain, 'original_label'].astype(int).to_numpy()
                    ).mean()
                )
                if certain.any()
                else None
            ),
            'model_agreement_with_review': (
                float(
                    (
                        label[certain].astype(int).to_numpy()
                        == z.loc[certain, 'model_pred'].astype(int).to_numpy()
                    ).mean()
                )
                if certain.any()
                else None
            ),
        }
    )
    reviewers.append(
        pd.DataFrame(
            {'review_id': z.review_id.astype(str), 'label': label, 'certain': certain}
        ).set_index('review_id')
    )

pairwise = []
for i in range(len(reviewers)):
    for j in range(i + 1, len(reviewers)):
        a1 = reviewers[i]
        a2 = reviewers[j]
        common = a1.index.intersection(a2.index)
        if len(common) == 0:
            continue
        x = a1.loc[common]
        y = a2.loc[common]
        ok = x.certain.to_numpy(bool) & y.certain.to_numpy(bool)
        if ok.any():
            l1 = x.label.to_numpy()[ok].astype(int)
            l2 = y.label.to_numpy()[ok].astype(int)
            agreement = float((l1 == l2).mean())
            kappa = float(cohen_kappa_score(l1, l2)) if len(np.unique(np.r_[l1, l2])) > 1 else None
        else:
            agreement = kappa = None
        pairwise.append(
            {
                'reviewer_a': per[i]['reviewer'],
                'reviewer_b': per[j]['reviewer'],
                'n_common_certain': int(ok.sum()),
                'raw_agreement': agreement,
                'cohen_kappa': kappa,
            }
        )

res = {'n_reviewers': len(reviewers), 'per_reviewer': per, 'pairwise_inter_reviewer': pairwise}
if per:
    vals = [
        x['original_label_confirmation_rate']
        for x in per
        if x['original_label_confirmation_rate'] is not None
    ]
    u = [x['uncertain_rate'] for x in per if x['uncertain_rate'] is not None]
    res['mean_original_label_confirmation_rate'] = float(np.mean(vals)) if vals else None
    res['mean_uncertain_rate'] = float(np.mean(u)) if u else None
atomic_json_dump(a.out, res)
print(json.dumps(res, indent=2))
