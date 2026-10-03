from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = (
    ROOT / 'COMMON' / 'src'
    if (ROOT / 'COMMON' / 'src').exists()
    else Path(__file__).resolve().parents[1] / 'src'
)
sys.path.insert(0, str(SRC))
import argparse
from puma_exploration8.manifest import read_manifest
from puma_exploration8.grouping import build_train_folds, build_cal_folds, save_fold_payload

p = argparse.ArgumentParser()
p.add_argument('--manifest', required=True)
p.add_argument('--out', required=True)
p.add_argument('--cal', action='store_true')
a = p.parse_args()
df = read_manifest(a.manifest, require_images=False)
x = build_cal_folds(df) if a.cal else build_train_folds(df)
save_fold_payload(x, a.out)
print(a.out)
