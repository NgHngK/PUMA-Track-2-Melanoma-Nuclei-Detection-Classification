from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'COMMON' / 'src'))

import argparse, json
import numpy as np
from puma_exploration8.config import load_json
from puma_exploration8.stageio import manifest_sha256
from puma_exploration8.manifest import read_manifest, strongest_group_column
from puma_exploration8.prediction import (
    load_prediction_npz,
    align_predictions,
    protected_metrics,
    group_primary_scores,
)
from puma_exploration8.protection import require_freeze, open_once, file_sha256


def paired_bootstrap(base: dict[str, float], candidate: dict[str, float], reps=5000, seed=17):
    keys = sorted(set(base) & set(candidate))
    if len(keys) < 2:
        raise ValueError('paired protected uncertainty requires >=2 biological groups')
    delta = np.asarray([candidate[k] - base[k] for k in keys], float)
    rng = np.random.default_rng(seed)
    vals = rng.choice(delta, size=(int(reps), len(delta)), replace=True).mean(axis=1)
    return {
        'n_groups': len(keys),
        'mean_delta': float(delta.mean()),
        'lo': float(np.percentile(vals, 2.5)),
        'hi': float(np.percentile(vals, 97.5)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--complete-freeze', required=True)
    p.add_argument('--state', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--predictions', required=True)
    p.add_argument('--baseline', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    cfg = load_json(ROOT / 'CONFIGS' / '90_VALIDATION_AND_TEST.json')
    freeze_hash = file_sha256(a.complete_freeze)
    complete = require_freeze(
        a.complete_freeze,
        'COMPLETE_PIPELINE_FREEZE',
        required_keys=[
            'protocol_sha256',
            'manifest_sha256',
            'visual_checkpoint_sha256',
            'boundary_freeze_sha256',
            'evaluator_contract',
            'seed',
        ],
    )
    if complete['manifest_sha256'] != manifest_sha256(a.manifest):
        raise ValueError('protected manifest differs from complete freeze')
    df = read_manifest(a.manifest, require_images=False)
    role = df[df.role == 'VALIDATE'].copy()
    if role.empty:
        raise ValueError('VALIDATE role empty')
    candidate_raw = load_prediction_npz(a.predictions, expected_freeze_sha256=freeze_hash)
    baseline_raw = load_prediction_npz(a.baseline)
    role_c, y, z = align_predictions(candidate_raw, role)
    role_b, yb, zb = align_predictions(baseline_raw, role)
    if (
        not np.array_equal(y, yb)
        or role_c.uid.astype(str).tolist() != role_b.uid.astype(str).tolist()
    ):
        raise ValueError('candidate/baseline protected rows do not match')
    group_col = strongest_group_column(df)
    cm = protected_metrics(role_c, y, z)
    bm = protected_metrics(role_b, yb, zb)
    if cm['primary_name'] != bm['primary_name']:
        raise ValueError('candidate/baseline primary metric contract differs')
    boot = paired_bootstrap(
        group_primary_scores(role_b, zb, group_col), group_primary_scores(role_c, z, group_col)
    )
    recall_delta = np.asarray(cm['semantic']['recall']) - np.asarray(bm['semantic']['recall'])
    group_support = np.asarray(
        [role_c.loc[role_c.label == c, group_col].astype(str).nunique() for c in range(10)], int
    )
    safety_estimable = group_support >= int(cfg['min_groups_for_hard_class_safety'])
    catastrophic = bool(
        np.any(recall_delta[safety_estimable] < -float(cfg['catastrophic_recall_drop']))
    )
    primary = bool(boot['mean_delta'] >= float(cfg['material_group_delta']) and boot['lo'] > 0)
    decision = (
        'PROMOTE_TO_TEST'
        if primary and not catastrophic
        else 'EXPLORATION_8_VALIDATION_FAILED_OR_INCONCLUSIVE'
    )
    result = {
        'decision': decision,
        'primary_metric': cm['primary_name'],
        'candidate': cm,
        'baseline': bm,
        'paired_group_effect': boot,
        'recall_delta': recall_delta.tolist(),
        'class_positive_groups': group_support.tolist(),
        'class_safety_estimable': safety_estimable.tolist(),
        'primary_efficacy_pass': primary,
        'supported_class_safety_pass': not catastrophic,
        'rare_class_safety_note': 'RARE-CLASS SAFETY NOT ESTABLISHED where class_safety_estimable=false',
        'freeze_sha256': freeze_hash,
    }
    open_once(a.state, 'VALIDATE', freeze_hash)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                'decision': decision,
                'primary_metric': cm['primary_name'],
                'paired_group_effect': boot,
                'safety_estimable': safety_estimable.tolist(),
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
