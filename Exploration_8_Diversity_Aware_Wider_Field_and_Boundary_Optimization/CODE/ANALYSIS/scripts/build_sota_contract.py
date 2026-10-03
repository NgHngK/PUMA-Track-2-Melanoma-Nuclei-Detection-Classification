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

p = argparse.ArgumentParser()
p.add_argument('--contract', default=str(ROOT / 'CONFIGS' / '01_SOTA_BENCHMARK_CONTRACT.json'))
a = p.parse_args()
print(Path(a.contract).read_text())
