from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from CODE.COMMON.hashing import sha256_file, sha256_strings, canonical_json_hash
from CODE.COMMON.schemas import FeatureContract


class FeatureCacheWriter:
    """Sequential feature-cache writer with epoch-like row checkpointing.

    The cache metadata stores `next_index` after every flushed chunk. If Colab stops,
    reopening the same writer continues at that row. The COMPLETE metadata is written last.
    """

    def __init__(
        self,
        directory: str | Path,
        name: str,
        n: int,
        dim: int,
        contract: FeatureContract,
        *,
        expected_uids: list[str],
    ):
        if int(contract.dim) != int(dim):
            raise ValueError(f'Contract dim {contract.dim} != cache dim {dim}')
        if len(expected_uids) != int(n):
            raise ValueError('expected_uids length mismatch')
        if len(set(expected_uids)) != len(expected_uids):
            raise ValueError('Duplicate expected cache UID')

        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.n = int(n)
        self.dim = int(dim)
        self.contract = contract
        self.expected_uids = list(map(str, expected_uids))
        self.uid_hash = sha256_strings(self.expected_uids)
        self.data_path = self.dir / f'{name}.npy'
        self.uid_path = self.dir / f'{name}.uids.txt'
        self.meta_path = self.dir / f'{name}.json'
        self.contract_hash = canonical_json_hash(asdict(contract))
        self.next_index = 0

        if self.meta_path.is_file():
            meta = json.loads(self.meta_path.read_text(encoding='utf-8'))
            if meta.get('status') == 'COMPLETE':
                raise ValueError(f'{name} cache is already COMPLETE')
            if int(meta.get('n', -1)) != self.n or int(meta.get('dim', -1)) != self.dim:
                raise ValueError(f'{name} incomplete cache shape contract changed')
            if canonical_json_hash(meta.get('contract')) != self.contract_hash:
                raise ValueError(f'{name} incomplete cache feature contract changed')
            if meta.get('uid_order_sha256') != self.uid_hash:
                raise ValueError(f'{name} incomplete cache UID order changed')
            if not self.data_path.is_file():
                raise ValueError(f'{name} metadata exists but data file is missing')
            self._arr = np.load(self.data_path, mmap_mode='r+')
            if self._arr.shape != (self.n, self.dim) or self._arr.dtype != np.float32:
                raise ValueError(f'{name} incomplete cache array shape/dtype mismatch')
            self.next_index = int(meta.get('next_index', 0))
            if not (0 <= self.next_index <= self.n):
                raise ValueError(f'{name} invalid resume index {self.next_index}')
        else:
            self._arr = np.lib.format.open_memmap(
                self.data_path, mode='w+', dtype=np.float32, shape=(self.n, self.dim)
            )
            self._write_progress_meta()

    def _write_progress_meta(self) -> None:
        meta = {
            'status': 'INCOMPLETE',
            'n': self.n,
            'dim': self.dim,
            'dtype': 'float32',
            'contract': asdict(self.contract),
            'contract_sha256': self.contract_hash,
            'uid_order_sha256': self.uid_hash,
            'next_index': int(self.next_index),
        }
        tmp = self.meta_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding='utf-8')
        os.replace(tmp, self.meta_path)

    def write(self, start: int, features: np.ndarray, uids: list[str]):
        features = np.asarray(features, dtype=np.float32)
        uids = list(map(str, uids))
        end = int(start) + len(features)
        if features.shape != (len(uids), self.dim):
            raise ValueError('Cache chunk shape mismatch')
        if int(start) != self.next_index:
            raise ValueError(
                f'Cache write must start at resume index {self.next_index}, got {start}'
            )
        if start < 0 or end > self.n:
            raise ValueError('Cache write outside allocated array')
        if not np.isfinite(features).all():
            raise ValueError('Nonfinite cache features')
        if uids != self.expected_uids[start:end]:
            raise ValueError('Cache chunk UID order mismatch')
        self._arr[start:end] = features
        self._arr.flush()
        self.next_index = end
        self._write_progress_meta()

    def finalize(self):
        if self.next_index != self.n:
            raise ValueError(f'Cannot finalize incomplete cache: {self.next_index}/{self.n}')
        self._arr.flush()
        self.uid_path.write_text('\n'.join(self.expected_uids) + '\n', encoding='utf-8')
        meta = {
            'status': 'COMPLETE',
            'n': self.n,
            'dim': self.dim,
            'dtype': 'float32',
            'contract': asdict(self.contract),
            'payload_sha256': sha256_file(self.data_path),
            'uid_order_sha256': self.uid_hash,
            'contract_sha256': self.contract_hash,
            'next_index': self.n,
        }
        tmp = self.meta_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding='utf-8')
        os.replace(tmp, self.meta_path)
        return meta
