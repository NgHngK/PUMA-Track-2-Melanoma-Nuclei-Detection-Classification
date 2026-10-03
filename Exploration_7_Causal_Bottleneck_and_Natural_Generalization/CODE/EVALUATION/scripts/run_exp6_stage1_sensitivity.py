from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'COMMON/src'
sys.path.insert(0, str(SRC))

import argparse, pandas as pd
from puma_exploration7.io import prediction_frame_to_logits
from puma_exploration7.manifest import read_manifest, join_labels
from puma_exploration7.stage1_sensitivity import center_error_analysis

p = argparse.ArgumentParser()
p.add_argument('--manifest', required=True)
p.add_argument('--predictions', required=True)
p.add_argument('--out', required=True)
a = p.parse_args()
m = read_manifest(a.manifest)
q = pd.read_csv(a.predictions)
f = join_labels(q, m)
center_error_analysis(f, f.label.to_numpy(int), prediction_frame_to_logits(f)).to_csv(
    a.out, index=False
)
print(a.out)
