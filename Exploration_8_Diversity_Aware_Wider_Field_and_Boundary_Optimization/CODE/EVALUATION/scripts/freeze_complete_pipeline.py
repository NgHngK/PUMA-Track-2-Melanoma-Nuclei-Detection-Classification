from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'COMMON' / 'src'))

import argparse, json
from puma_exploration8.protection import require_freeze, write_freeze, file_sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--visual-freeze', required=True)
    p.add_argument('--boundary-freeze', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    visual = require_freeze(
        a.visual_freeze,
        'VISUAL_FREEZE',
        required_keys=['protocol_sha256', 'manifest_sha256', 'checkpoint_sha256', 'seed'],
    )
    boundary = require_freeze(
        a.boundary_freeze,
        'BOUNDARY_FREEZE',
        required_keys=['visual_freeze_sha256', 'recipe', 'seed'],
    )
    vh = file_sha256(a.visual_freeze)
    bh = file_sha256(a.boundary_freeze)
    if boundary['visual_freeze_sha256'] != vh or int(boundary['seed']) != int(visual['seed']):
        raise ValueError('visual/boundary freeze mismatch')
    data = write_freeze(
        a.out,
        {
            'kind': 'COMPLETE_PIPELINE_FREEZE',
            'protocol_sha256': visual['protocol_sha256'],
            'config_bundle_sha256': visual.get('config_bundle_sha256'),
            'manifest_sha256': visual['manifest_sha256'],
            'visual_checkpoint_sha256': visual['checkpoint_sha256'],
            'visual_freeze_sha256': vh,
            'boundary_freeze_sha256': bh,
            'seed': int(visual['seed']),
            'selected_wide': visual.get('selected_wide'),
            'route': visual.get('route'),
            'use_tiera': visual.get('use_tiera'),
            'boundary_recipe': boundary['recipe'],
            'evaluator_contract': 'PUMA_V17_STRICT_LT15_SAME_CLASS_WHEN_GEOMETRY_ESTIMABLE; otherwise semantic fixed10 fallback',
        },
    )
    print(
        json.dumps(
            {
                'status': 'OK',
                'complete_freeze': str(Path(a.out)),
                'complete_freeze_sha256': file_sha256(a.out),
                'payload_sha256': data['payload_sha256'],
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
