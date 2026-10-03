from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'COMMON/src'
sys.path.insert(0, str(SRC))

import argparse, json
from puma_exploration7.io import (
    load_exploration6_result,
    prediction_frame_to_logits,
    atomic_json_dump,
)
from puma_exploration7.manifest import read_manifest, join_labels, strongest_group_column
from puma_exploration7.metrics import semantic_metrics, puma_metrics
from puma_exploration7.diagnostics import (
    confusion_pair_table,
    top2_diagnostics,
    group_error_table,
    numeric_error_correlations,
)

p = argparse.ArgumentParser()
p.add_argument('--result', required=True)
p.add_argument('--manifest', required=True)
p.add_argument('--out-dir', required=True)
a = p.parse_args()
pred = load_exploration6_result(a.result)
frame = join_labels(pred, read_manifest(a.manifest))
z = prediction_frame_to_logits(frame)
out = __import__('pathlib').Path(a.out_dir)
out.mkdir(parents=True, exist_ok=True)
m = semantic_metrics(frame.label, z)
m['puma'] = puma_metrics(frame, frame.label.to_numpy(int), z)
atomic_json_dump(out / 'metrics.json', m)
confusion_pair_table(frame.label, z).to_csv(out / 'confusion_pairs.csv', index=False)
g = strongest_group_column(frame)
group_error_table(frame, frame.label, z, g).to_csv(out / 'group_errors.csv', index=False)
numeric_error_correlations(frame, frame.label, z).to_csv(
    out / 'error_correlations.csv', index=False
)
atomic_json_dump(out / 'top2.json', top2_diagnostics(frame.label, z))
print(json.dumps({'status': 'ok', 'n': len(frame), 'out': str(out)}, indent=2))
