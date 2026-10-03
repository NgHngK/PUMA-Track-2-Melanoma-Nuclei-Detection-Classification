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

p = argparse.ArgumentParser(
    description='Validate a predeclared four-role manifest; role creation is intentionally not outcome-driven.'
)
p.add_argument('--manifest', required=True)
p.add_argument('--out', required=True)
a = p.parse_args()
df = read_manifest(a.manifest, require_images=False)
df.to_csv(a.out, index=False)
print(a.out)
