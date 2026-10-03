from __future__ import annotations
import argparse, json, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from .replication import centered_crop, to_encoder_tensor, tier_a_features
from .models.pooling import gaussian_pool, annular_pool
from .models.a5 import A5Head
from .models.routing import WiderResidualA, WiderInteractionC
from .models.stage2 import Stage2CachedModel
from .cache_integrity import write_cache, read_cache
from .checkpoint import save_checkpoint, load_checkpoint
from .normalization import fit_tiera_normalizer
from .calibration import crossfit_h1, crossfit_h1_h2, fit_temperature_grid
from .experiments import build_stage2_model
from .diversity import draw_roster
from .protection import write_freeze, require_freeze
from .vendor.puma_v17_evaluator import evaluate_rois


def run(tmp: Path):
    tmp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(17)
    img = Image.fromarray(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8))
    crop = centered_crop(img, 64, 64, 96)
    t = to_encoder_tensor(crop)
    assert t.shape == (3, 224, 224)
    bio = tier_a_features(img, [(64, 64), (40, 40)])
    assert bio.shape == (2, 16) and np.isfinite(bio).all()
    norm = fit_tiera_normalizer(bio, ['a', 'b'], 'smoke')
    assert norm.transform(bio).shape == (2, 16)
    grid = torch.randn(4, 16, 16, 1536)
    gl = gaussian_pool(grid, [8.0] * 4, [8.0] * 4, 1.5)
    an = annular_pool(grid, [8.0] * 4, [8.0] * 4)
    assert gl.shape == an.shape == (4, 1536)

    uids = [f'u{i}' for i in range(4)]
    write_cache(tmp / 'g', gl.numpy(), uids, {'representation': 'gaussian96'})
    _, u, _ = read_cache(tmp / 'g', {'representation': 'gaussian96'}, strict_hash=True)
    assert list(u) == uids
    try:
        read_cache(tmp / 'g', {'representation': 'annular192'})
        raise AssertionError('stale cache accepted')
    except ValueError:
        pass
    # A same-size edit should fail the metadata check; strict mode also catches it from the hash.
    data_path = tmp / 'g.npy'
    original = data_path.read_bytes()
    mutated = bytearray(original)
    mutated[-1] ^= 1
    data_path.write_bytes(mutated)
    try:
        read_cache(tmp / 'g', strict_hash=True)
        raise AssertionError('mutated cache accepted')
    except ValueError:
        pass
    data_path.write_bytes(original)

    b = torch.randn(4, 16)
    y = torch.tensor([0, 1, 2, 3])
    for route in [WiderResidualA(), WiderInteractionC()]:
        model = Stage2CachedModel(True, route)
        z = model(gl, b, an)
        loss = torch.nn.functional.cross_entropy(z, y)
        loss.backward()
        assert z.shape == (4, 10) and torch.isfinite(z).all()
        assert any(p.grad is not None for p in route.parameters())
    base = A5Head()
    assert sum(p.numel() for p in base.parameters()) == 27866
    save_checkpoint(tmp / 'x.pt', base, {'seed': 17, 'fold': 0})
    base2 = A5Head()
    load_checkpoint(tmp / 'x.pt', base2, {'seed': 17, 'fold': 0})

    m1 = build_stage2_model(17, True, 'A')
    m2 = build_stage2_model(17, True, 'A')
    assert all(
        torch.equal(a, b) for a, b in zip(m1.state_dict().values(), m2.state_dict().values())
    )
    m3 = build_stage2_model(29, True, 'A')
    assert any(
        not torch.equal(a, b) for a, b in zip(m1.state_dict().values(), m3.state_dict().values())
    )

    logits = rng.normal(size=(60, 10))
    yy = np.tile(np.arange(10), 6)
    groups = np.repeat(np.arange(6), 10)
    folds = np.repeat(np.arange(3), 20)
    h1 = crossfit_h1(logits, yy, groups, np.ones(10) / 10, [0, 0.25, 0.5, 1])
    assert h1['oof_logits'].shape == (60, 10)
    # Run the nested H1/H2 path here instead of reusing a saved H1 OOF stack.
    h2 = crossfit_h1_h2(logits, yy, groups, np.ones(10) / 10, [0, 0.5, 1], (100.0,))
    assert h2['oof_logits'].shape == (60, 10)
    T = fit_temperature_grid(logits, yy)
    assert np.array_equal(logits.argmax(1), (logits / T).argmax(1))

    # Check that the roster includes every selected biological group, not just rows from the combined pool.
    rows = []
    for c in range(10):
        for g in range(8):
            for n in range(3):
                rows.append(
                    {
                        'uid': f'{c}-{g}-{n}',
                        'role': 'TRAIN',
                        'label': c,
                        'roi': f'r{c}-{g}',
                        'patient_id': f'p{c}-{g}',
                    }
                )
    roster = draw_roster(pd.DataFrame(rows), 8, 16, 17)
    assert all(roster[roster.label == c].patient_id.nunique() == 8 for c in range(10))

    freeze = tmp / 'freeze.json'
    write_freeze(
        freeze,
        {
            'kind': 'COMPLETE_PIPELINE_FREEZE',
            'protocol_sha256': 'p',
            'manifest_sha256': 'm',
            'visual_checkpoint_sha256': 'v',
            'boundary_freeze_sha256': 'b',
            'evaluator_contract': 'e',
            'seed': 17,
        },
    )
    require_freeze(freeze, 'COMPLETE_PIPELINE_FREEZE', ['protocol_sha256'])
    fake = tmp / 'fake.json'
    fake.write_text('{}')
    try:
        require_freeze(fake, 'COMPLETE_PIPELINE_FREEZE')
        raise AssertionError('empty freeze accepted')
    except RuntimeError:
        pass

    gt = {'r': [{'x': 0, 'y': 0, 'class_id': 0}]}
    pred = {'r': [{'x': 1, 'y': 0, 'class_id': 0, 'score': 0.9}]}
    ev = evaluate_rois(gt, pred, ['r'])
    assert ev['tp'][0] == 1
    return {
        'status': 'PASS',
        'a5_params': 27866,
        'temperature': T,
        'evaluator_roi_f1': ev['roi_fixed10_macro_f1'],
        'seed_binding': 'PASS',
        'cache_corruption_rejected': 'PASS',
        'nested_h2': 'PASS',
        'freeze_binding': 'PASS',
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tmp')
    a = p.parse_args()
    tmp = Path(a.tmp) if a.tmp else Path(tempfile.mkdtemp(prefix='p8_selfcheck_'))
    print(json.dumps(run(tmp), indent=2))


if __name__ == '__main__':
    main()
