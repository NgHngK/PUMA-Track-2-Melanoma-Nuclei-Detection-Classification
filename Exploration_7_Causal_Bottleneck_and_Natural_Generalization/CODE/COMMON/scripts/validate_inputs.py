from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'COMMON/src'
sys.path.insert(0, str(SRC))

import argparse, json
from puma_exploration7.manifest import read_manifest, class_group_support, strongest_group_column
from puma_exploration7.io import atomic_json_dump

p = argparse.ArgumentParser()
p.add_argument('--manifest', required=True)
p.add_argument('--out', required=True)
a = p.parse_args()
df = read_manifest(a.manifest)
g = strongest_group_column(df)
res = {
    'n': len(df),
    'group_column': g,
    'groups': int(df[g].nunique()),
    'class_group_support': class_group_support(df, g).to_dict('records'),
}
atomic_json_dump(a.out, res)
print(json.dumps(res, indent=2))
