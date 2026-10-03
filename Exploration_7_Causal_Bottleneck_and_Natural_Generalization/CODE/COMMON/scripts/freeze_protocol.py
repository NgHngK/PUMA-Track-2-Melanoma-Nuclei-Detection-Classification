from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'COMMON/src'
sys.path.insert(0, str(SRC))

import argparse
from puma_exploration7.protocol import freeze_protocol

p = argparse.ArgumentParser()
p.add_argument('--config', required=True)
p.add_argument('--out', required=True)
a = p.parse_args()
print(freeze_protocol(a.config, a.out))
