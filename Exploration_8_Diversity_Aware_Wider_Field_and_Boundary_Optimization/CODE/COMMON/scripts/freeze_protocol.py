from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'COMMON' / 'src'))

import argparse, datetime, json
from puma_exploration8.config import config_bundle_hash, sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config')
    p.add_argument('--config-dir')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    if bool(a.config) == bool(a.config_dir):
        raise ValueError('provide exactly one of --config or --config-dir')
    if a.config:
        path = Path(a.config)
        payload = {'config': path.name, 'sha256': sha256_file(path)}
    else:
        bundle, hashes = config_bundle_hash(a.config_dir)
        payload = {'config_dir': str(Path(a.config_dir).name), 'sha256': bundle, 'files': hashes}
    payload['frozen_utc'] = datetime.datetime.now(datetime.UTC).isoformat()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
