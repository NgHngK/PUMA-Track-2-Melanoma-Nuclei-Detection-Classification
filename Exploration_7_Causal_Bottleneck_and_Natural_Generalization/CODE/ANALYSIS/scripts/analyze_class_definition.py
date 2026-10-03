from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'COMMON/src'
sys.path.insert(0, str(SRC))

import argparse, numpy as np, pandas as pd
from puma_exploration7.io import load_exploration6_result, prediction_frame_to_logits, softmax_np
from puma_exploration7.manifest import read_manifest, join_labels
from puma_exploration7.constants import CLASSES

p = argparse.ArgumentParser(
    description='Quantify symmetric/asymmetric pair confusion for flat-ten-class diagnosis.'
)
src = p.add_mutually_exclusive_group(required=True)
src.add_argument('--result', help='Historical Exploration-6 result JSON (diagnostic only).')
src.add_argument('--predictions', help='Exploration-7 prediction CSV.')
p.add_argument('--manifest', required=True)
p.add_argument('--out', required=True)
a = p.parse_args()
pred = load_exploration6_result(a.result) if a.result else pd.read_csv(a.predictions)
f = join_labels(pred, read_manifest(a.manifest))
z = prediction_frame_to_logits(f)
y = f.label.to_numpy(int)
pr = softmax_np(z).argmax(1)
cm = np.bincount(10 * y + pr, minlength=100).reshape(10, 10)
pairs = []
for i in range(10):
    for j in range(i + 1, 10):
        mutual = int(cm[i, j] + cm[j, i])
        den = int(cm[i].sum() + cm[j].sum())
        pairs.append(
            {
                'a': CLASSES[i],
                'b': CLASSES[j],
                'a_to_b': int(cm[i, j]),
                'b_to_a': int(cm[j, i]),
                'mutual_errors': mutual,
                'pair_error_fraction_of_true_pair': float(mutual / den) if den else 0.0,
                'asymmetry': float(abs(cm[i, j] - cm[j, i]) / mutual) if mutual else 0.0,
            }
        )
pd.DataFrame(pairs).sort_values('mutual_errors', ascending=False).to_csv(a.out, index=False)
print(a.out)
