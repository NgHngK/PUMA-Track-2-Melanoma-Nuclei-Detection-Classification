from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'COMMON' / 'src'))

import argparse, json
import numpy as np
from puma_exploration8.calibration import crossfit_h1, crossfit_h1_h2, fit_temperature_grid
from puma_exploration8.boundary import apply_h1
from puma_exploration8.config import load_json, sha256_file
from puma_exploration8.manifest import read_manifest, strongest_group_column
from puma_exploration8.grouping import load_fold_payload
from puma_exploration8.metrics import classification_metrics
from puma_exploration8.prediction import load_prediction_npz, align_predictions
from puma_exploration8.protection import require_freeze, write_freeze, file_sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz', required=True, help='CAL visual logits produced by predict_cached.py')
    p.add_argument('--manifest', required=True)
    p.add_argument('--folds', required=True)
    p.add_argument('--visual-freeze', required=True)
    p.add_argument('--training-metadata', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--boundary-freeze', required=True)
    p.add_argument('--run-h2', action='store_true')
    p.add_argument('--fit-temperature', action='store_true')
    a = p.parse_args()

    cfg = load_json(ROOT / 'CONFIGS' / '80_P8H1_SCALAR_PRIOR.json')
    h2cfg = load_json(ROOT / 'CONFIGS' / 'conditional' / '81_P8H2_SHRUNK_BIAS.json')
    visual = require_freeze(
        a.visual_freeze,
        'VISUAL_FREEZE',
        required_keys=[
            'protocol_sha256',
            'manifest_sha256',
            'checkpoint_sha256',
            'seed',
            'selected_wide',
            'route',
            'use_tiera',
        ],
    )
    visual_hash = file_sha256(a.visual_freeze)
    if visual['protocol_sha256'] != sha256_file(ROOT / 'CONFIGS' / '00_PROTOCOL.json'):
        raise ValueError('current protocol differs from visual freeze')
    train_meta = json.loads(Path(a.training_metadata).read_text())
    if train_meta.get('visual_freeze_sha256') != visual_hash:
        raise ValueError('training metadata does not belong to visual freeze')
    source = np.asarray(train_meta.get('effective_source_prior'), float)
    if source.shape != (10,) or not np.isfinite(source).all() or (source <= 0).any():
        raise ValueError('measured effective_source_prior is required for P8-H')
    source = source / source.sum()

    df = read_manifest(a.manifest, require_images=False)
    cal = df[df.role == 'CAL'].copy()
    pred = load_prediction_npz(a.npz, expected_freeze_sha256=visual_hash)
    cal, y, z = align_predictions(pred, cal)
    gcol = strongest_group_column(df)
    groups = cal[gcol].astype(str).to_numpy()
    fold = load_fold_payload(a.folds, df, 'CAL')
    fmap = {str(q['uid']): int(q['fold']) for q in fold['assignments']}
    fv = np.asarray([fmap[str(u)] for u in cal.uid], int)

    lambdas = [float(x) for x in cfg['lambda_grid']]
    max_drop = float(cfg['max_supported_class_recall_drop'])
    h1 = crossfit_h1(z, y, groups, source, lambdas, max_drop, fv)
    h1_oof = h1.pop('oof_logits')
    baseline = classification_metrics(y, z)
    h1m = h1['oof_metrics']
    delta = np.asarray(h1m['recall']) - np.asarray(baseline['recall'])
    pos_groups = np.asarray([len(np.unique(groups[y == c])) for c in range(10)], int)
    min_groups = int(h2cfg['min_positive_groups_per_class'])
    support_ok = pos_groups >= min_groups
    aggregate_gain = float(h1m['macro_f1_fixed10'] - baseline['macro_f1_fixed10'])
    harm = np.where(support_ok & (delta < -max_drop))[0].tolist()
    unlock = bool(aggregate_gain > 0 and harm and support_ok.all())

    selected = 'H0_NO_BOUNDARY_CHANGE'
    selected_recipe = {
        'lambda': 0.0,
        'target_prior': h1['final_target_prior'],
        'offset': h1['final_offset'],
        'delta_zero_sum': [0.0] * 10,
        'shrinkage': None,
    }
    if aggregate_gain > 0 and not harm:
        selected = 'H1'
        selected_recipe = {
            'lambda': float(h1['final_lambda']),
            'target_prior': h1['final_target_prior'],
            'offset': h1['final_offset'],
            'delta_zero_sum': [0.0] * 10,
            'shrinkage': None,
        }
    h2_result = {'status': 'LOCKED_BY_DEFAULT'}
    if a.run_h2:
        if not unlock:
            h2_result = {'status': 'LOCKED_OR_NOT_ESTIMABLE'}
        else:
            shrinkages = tuple(float(x) for x in h2cfg['shrinkage_family'])
            h2 = crossfit_h1_h2(z, y, groups, source, lambdas, shrinkages, max_drop, fv)
            h2_oof = h2.pop('oof_logits')
            h2_result = h2
            # Keep H2 only if it beats H1 without causing more harm to the supported classes.
            h2m = h2['oof_metrics']
            h2_delta = np.asarray(h2m['recall']) - np.asarray(baseline['recall'])
            h2_harm = np.where(support_ok & (h2_delta < -max_drop))[0].tolist()
            if h2m['macro_f1_fixed10'] > baseline['macro_f1_fixed10'] and not h2_harm:
                selected = 'H2'
                selected_recipe = {
                    'lambda': float(h2['final_lambda']),
                    'target_prior': h2['final_target_prior'],
                    'offset': h2['final_offset'],
                    'delta_zero_sum': h2['final_delta_zero_sum'],
                    'shrinkage': float(h2['final_shrinkage']),
                }

    full_boundary_logits = (
        apply_h1(z, np.asarray(selected_recipe['offset']), selected_recipe['lambda'])
        + np.asarray(selected_recipe['delta_zero_sum'])[None, :]
    )
    temp = fit_temperature_grid(full_boundary_logits, y) if a.fit_temperature else 1.0
    selected_recipe['temperature'] = float(temp)
    result = {
        'frozen_cal_fold_sha256': fold['sha256'],
        'visual_freeze_sha256': visual_hash,
        'baseline_metrics': baseline,
        'H1': h1,
        'aggregate_gain': aggregate_gain,
        'recall_delta': delta.tolist(),
        'positive_groups_per_class': pos_groups.tolist(),
        'H2_unlock': unlock,
        'H2_unlock_reasons': {
            'aggregate_benefit': aggregate_gain > 0,
            'localized_recall_harm': bool(harm),
            'cal_support_all_classes': bool(support_ok.all()),
            'harm_classes': harm,
        },
        'H2': h2_result,
        'selected_boundary': selected,
        'selected_recipe': selected_recipe,
        'temperature_argmax_unchanged': bool(
            np.array_equal(full_boundary_logits.argmax(1), (full_boundary_logits / temp).argmax(1))
        ),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    write_freeze(
        a.boundary_freeze,
        {
            'kind': 'BOUNDARY_FREEZE',
            'visual_freeze_sha256': visual_hash,
            'protocol_sha256': visual['protocol_sha256'],
            'manifest_sha256': visual['manifest_sha256'],
            'seed': int(visual['seed']),
            'selected_boundary': selected,
            'recipe': selected_recipe,
            'cal_fold_sha256': fold['sha256'],
            'training_metadata_sha256': file_sha256(a.training_metadata),
        },
    )
    print(
        json.dumps(
            {
                'selected_boundary': selected,
                'aggregate_gain': aggregate_gain,
                'H2_unlock': unlock,
                'temperature': temp,
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
