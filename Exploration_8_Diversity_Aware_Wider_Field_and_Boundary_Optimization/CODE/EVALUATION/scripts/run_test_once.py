from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'COMMON' / 'src'))

import argparse, json
from puma_exploration8.manifest import read_manifest
from puma_exploration8.stageio import manifest_sha256
from puma_exploration8.prediction import load_prediction_npz, align_predictions, protected_metrics
from puma_exploration8.protection import require_freeze, open_once, file_sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--promotion', required=True)
    p.add_argument('--complete-freeze', required=True)
    p.add_argument('--state', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--predictions', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
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
    freeze_hash = file_sha256(a.complete_freeze)
    if complete['manifest_sha256'] != manifest_sha256(a.manifest):
        raise ValueError('protected manifest differs from complete freeze')
    promotion = json.loads(Path(a.promotion).read_text())
    if promotion.get('decision') != 'PROMOTE_TO_TEST':
        raise RuntimeError('TEST blocked: VALIDATE did not promote')
    if promotion.get('freeze_sha256') != freeze_hash:
        raise RuntimeError('TEST blocked: promotion belongs to a different frozen pipeline')
    df = read_manifest(a.manifest, require_images=False)
    role = df[df.role == 'TEST'].copy()
    if role.empty:
        raise ValueError('TEST role empty')
    pred = load_prediction_npz(a.predictions, expected_freeze_sha256=freeze_hash)
    role, y, z = align_predictions(pred, role)
    metrics = protected_metrics(role, y, z)
    result = {
        'status': 'TEST_OPENED_ONCE',
        'metrics': metrics,
        'freeze_sha256': freeze_hash,
        'seed': int(complete['seed']),
        'claim_scope': 'INTERNAL HELD-GROUP IMPROVEMENT unless SOTA_BENCHMARK_CONTRACT establishes 1:1 comparability',
    }
    open_once(a.state, 'TEST', freeze_hash)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                'status': result['status'],
                'primary_metric': metrics['primary_name'],
                'primary_value': metrics['primary_value'],
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
