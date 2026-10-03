from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = (
    ROOT / 'COMMON' / 'src'
    if (ROOT / 'COMMON' / 'src').exists()
    else Path(__file__).resolve().parents[1] / 'src'
)
sys.path.insert(0, str(SRC))
import argparse, json, numpy as np
from puma_exploration8.metrics import classification_metrics

p = argparse.ArgumentParser()
p.add_argument('--npz', required=True)
a = p.parse_args()
x = np.load(a.npz)
print(json.dumps(classification_metrics(x['labels'], x['logits']), indent=2))
