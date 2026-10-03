from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = (
    ROOT / 'COMMON' / 'src'
    if (ROOT / 'COMMON' / 'src').exists()
    else Path(__file__).resolve().parents[1] / 'src'
)
sys.path.insert(0, str(SRC))
import argparse, json
from puma_exploration8.manifest import read_manifest, manifest_summary

p = argparse.ArgumentParser()
p.add_argument('--manifest', required=True)
p.add_argument('--check-bounds', action='store_true')
a = p.parse_args()
df = read_manifest(a.manifest, check_bounds=a.check_bounds)
print(json.dumps(manifest_summary(df), indent=2))
