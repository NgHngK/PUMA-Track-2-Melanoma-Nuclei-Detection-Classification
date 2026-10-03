from __future__ import annotations
from pathlib import Path
import json, numpy as np
from CODE.COMMON.hashing import sha256_file, sha256_strings, canonical_json_hash
from CODE.COMMON.exceptions import FeatureContractError, CacheAlignmentError


class FeatureCache:
    def __init__(self, directory: str | Path, name: str, verify_hash: bool = True):
        d = Path(directory)
        self.data_path = d / f'{name}.npy'
        self.uid_path = d / f'{name}.uids.txt'
        self.meta_path = d / f'{name}.json'
        meta = json.loads(self.meta_path.read_text())
        if meta.get('status') != 'COMPLETE':
            raise FeatureContractError(f'{name} cache incomplete')
        if canonical_json_hash(meta.get('contract')) != meta.get('contract_sha256'):
            raise FeatureContractError('Contract hash mismatch')
        if int(meta['contract']['dim']) != int(meta['dim']):
            raise FeatureContractError('Contract/cache dimension mismatch')
        self.uids = self.uid_path.read_text().splitlines()
        self.meta = meta
        if len(self.uids) != meta['n']:
            raise FeatureContractError('UID count mismatch')
        if sha256_strings(self.uids) != meta['uid_order_sha256']:
            raise FeatureContractError('UID-order hash mismatch')
        if verify_hash and sha256_file(self.data_path) != meta['payload_sha256']:
            raise FeatureContractError('Payload hash mismatch')
        self.array = np.load(self.data_path, mmap_mode='r')
        if self.array.shape != (meta['n'], meta['dim']) or self.array.dtype != np.float32:
            raise FeatureContractError('Cache shape/dtype mismatch')

    def aligned_indices(self, target_uids: list[str]):
        if len(set(self.uids)) != len(self.uids):
            raise CacheAlignmentError('Duplicate cache UIDs')
        pos = {u: i for i, u in enumerate(self.uids)}
        missing = [u for u in target_uids if u not in pos]
        if missing:
            raise CacheAlignmentError(f'Missing UIDs, e.g. {missing[0]}')
        return np.array([pos[u] for u in target_uids], dtype=np.int64)

    def get(self, target_uids: list[str]):
        return np.asarray(self.array[self.aligned_indices(target_uids)], dtype=np.float32)
