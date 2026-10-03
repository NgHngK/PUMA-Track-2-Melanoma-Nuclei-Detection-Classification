from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'COMMON/src'
sys.path.insert(0, str(SRC))

import argparse
import pandas as pd
from puma_exploration7.io import load_exploration6_result, prediction_frame_to_logits
from puma_exploration7.manifest import read_manifest, join_labels
from puma_exploration7.review import build_blinded_review

p = argparse.ArgumentParser(description='Build a blinded Exploration-7 annotation-review package.')
src = p.add_mutually_exclusive_group(required=True)
src.add_argument(
    '--result',
    help='Historical Exploration-6 result JSON containing predictions (diagnostic only).',
)
src.add_argument(
    '--predictions', help='Exploration-7 prediction CSV with p_<class> or logit_<class> columns.'
)
p.add_argument('--manifest', required=True)
p.add_argument('--out-dir', required=True)
p.add_argument('--per-pair', type=int, default=30)
a = p.parse_args()

pred = load_exploration6_result(a.result) if a.result else pd.read_csv(a.predictions)
f = join_labels(pred, read_manifest(a.manifest))
z = prediction_frame_to_logits(f)
pairs = [(7, 0), (0, 1), (0, 3), (5, 9)]
build_blinded_review(f, f.label.to_numpy(int), z, a.out_dir, pairs, per_pair=a.per_pair)
print('review package written:', a.out_dir)
