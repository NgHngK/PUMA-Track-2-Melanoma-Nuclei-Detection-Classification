from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = (
    ROOT / 'COMMON' / 'src'
    if (ROOT / 'COMMON' / 'src').exists()
    else Path(__file__).resolve().parents[1] / 'src'
)
sys.path.insert(0, str(SRC))
import argparse, shutil, json

p = argparse.ArgumentParser()
p.add_argument('--source', required=True)
p.add_argument('--dest', required=True)
a = p.parse_args()
s = Path(a.source)
d = Path(a.dest)
if d.exists():
    print(json.dumps({'status': 'REUSED', 'dest': str(d)}))
else:
    shutil.copytree(s, d) if s.is_dir() else shutil.copy2(s, d)
    print(json.dumps({'status': 'COPIED', 'dest': str(d)}))
