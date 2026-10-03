from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'COMMON/src'
sys.path.insert(0, str(SRC))

import argparse, json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--results', required=True)
a = p.parse_args()
root = Path(a.results)
files = sorted(root.rglob('*.json'))
print(json.dumps({'json_artifacts': [str(x.relative_to(root)) for x in files]}, indent=2))
