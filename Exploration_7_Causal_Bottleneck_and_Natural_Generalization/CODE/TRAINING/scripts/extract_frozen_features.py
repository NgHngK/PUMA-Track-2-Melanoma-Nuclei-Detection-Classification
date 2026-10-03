from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'COMMON/src'))
import argparse
from puma_exploration7.features import extract_gaussian_features

p = argparse.ArgumentParser()
p.add_argument('--manifest', required=True)
p.add_argument('--weights', required=True)
p.add_argument('--out', required=True)
p.add_argument('--fov', type=int, default=96)
p.add_argument('--sigma', type=float, default=1.5)
p.add_argument('--batch-size', type=int, default=0, help='0=auto from GPU memory')
p.add_argument('--num-workers', type=int, default=-1, help='-1=auto')
p.add_argument('--device', default='cuda')
a = p.parse_args()
extract_gaussian_features(
    a.manifest, a.weights, a.out, a.fov, a.sigma, a.batch_size, a.device, a.num_workers
)
print(a.out)
