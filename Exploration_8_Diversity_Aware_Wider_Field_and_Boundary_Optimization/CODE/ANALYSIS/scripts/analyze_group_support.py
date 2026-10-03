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
from puma_exploration8.feasibility import support_table

p = argparse.ArgumentParser()
p.add_argument('--manifest', required=True)
p.add_argument('--role', default='TRAIN')
a = p.parse_args()
print(support_table(read_manifest(a.manifest, require_images=False), a.role).to_csv(index=False))
