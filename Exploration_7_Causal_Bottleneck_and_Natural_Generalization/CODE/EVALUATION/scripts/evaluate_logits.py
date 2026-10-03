from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'COMMON/src'
sys.path.insert(0, str(SRC))

import argparse, json, pandas as pd
from puma_exploration7.io import prediction_frame_to_logits, atomic_json_dump
from puma_exploration7.manifest import read_manifest
from puma_exploration7.experiments import evaluate_frame

p = argparse.ArgumentParser()
p.add_argument('--manifest', required=True)
p.add_argument('--predictions', required=True)
p.add_argument('--out', required=True)
a = p.parse_args()
m = read_manifest(a.manifest)
q = pd.read_csv(a.predictions)
f = q.merge(m, on='uid', how='inner', suffixes=('', '_m'), validate='one_to_one')
res = evaluate_frame(f, prediction_frame_to_logits(f))
atomic_json_dump(a.out, res)
print(json.dumps({'status': 'ok', 'out': a.out}, indent=2))
