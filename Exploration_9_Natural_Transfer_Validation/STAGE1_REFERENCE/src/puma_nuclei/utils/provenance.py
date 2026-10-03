from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@lru_cache(maxsize=256)
def _sha256_file_cached(resolved_path: str, size: int, modified_ns: int) -> str:
    digest = hashlib.sha256()
    with Path(resolved_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return _sha256_file_cached(str(resolved), int(stat.st_size), int(stat.st_mtime_ns))


class StableArtifactIdentity(dict[str, Any]):
    """Dict-compatible artifact identity with Drive-remount-stable equality.

    Google Drive/FUSE may normalize ``st_mtime_ns`` after a Colab remount even
    when the underlying immutable file is unchanged.  Keep the timestamp for
    diagnostics/serialization, but compare cached identities by filename + byte
    size.  Missing/truncated/replaced-size artifacts still fail closed, while
    ndarray schema and OOF provenance are validated by the Stage-2 loader.
    """

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict) and "name" in self and "size_bytes" in self:
            if "name" in other and "size_bytes" in other:
                return (
                    str(self.get("name")) == str(other.get("name"))
                    and int(self.get("size_bytes", -1)) == int(other.get("size_bytes", -2))
                )
        return dict.__eq__(self, other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)


def file_stat_identity(path: str | Path) -> dict[str, Any]:
    """Cheap identity for large immutable artifacts.

    Content SHA256 remains the strongest identity and should be used for model
    checkpoints and compact manifests. For multi-gigabyte memmaps that are read
    on every Colab restart, size + nanosecond mtime avoids an expensive full-file
    scan while still detecting ordinary replacement/truncation/edit operations.
    """
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return StableArtifactIdentity({
        "name": resolved.name,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    })


def numpy_array_sha256(array: np.ndarray) -> str:
    """Stable SHA256 for an ndarray/structured array without dtype conversion."""
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(values.dtype.str.encode("utf-8"))
    if values.dtype.fields:
        digest.update(repr(values.dtype.descr).encode("utf-8"))
    digest.update(repr(tuple(int(v) for v in values.shape)).encode("ascii"))
    digest.update(values.view(np.uint8))
    return digest.hexdigest()


def canonical_fingerprint(payload: Any, source_files: Iterable[str | Path] = ()) -> str:
    """Stable fingerprint for a configuration plus the code that defines its meaning."""
    digest = hashlib.sha256()
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    digest.update(normalized)
    for path in sorted((Path(value).resolve() for value in source_files), key=str):
        digest.update(str(path.name).encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def assert_fold_independent(*, sample_fold: int, source: dict[str, Any], source_name: str) -> None:
    if source.get("source") == "external_pretrained_frozen":
        return
    if "training_folds" not in source:
        raise RuntimeError(f"OOF provenance failure: {source_name} does not declare training_folds")
    values = source.get("training_folds")
    if not isinstance(values, (list, tuple, set)) or not values:
        raise RuntimeError(f"OOF provenance failure: {source_name} has empty/invalid training_folds")
    training_folds = {int(value) for value in values}
    if int(sample_fold) in training_folds:
        raise RuntimeError(
            f"OOF provenance failure: {source_name} for sample fold {sample_fold} "
            f"was trained on folds {sorted(training_folds)}"
        )
