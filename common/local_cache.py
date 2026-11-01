"""Copy verified preprocessing cache once to Colab local disk when space permits."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from .run_history import file_fingerprint


def _inventory(root: Path) -> dict[str, int]:
    return {
        str(path.relative_to(root)): path.stat().st_size
        for path in root.rglob("*")
        if path.is_file()
    }


def _cache_matches(source: Path, destination: Path) -> bool:
    if not destination.is_dir():
        return False

    a = source / "manifest.json"
    b = destination / "manifest.json"
    return (
        a.is_file()
        and b.is_file()
        and a.read_bytes() == b.read_bytes()
        and _inventory(source) == _inventory(destination)
        and all(
            file_fingerprint(source / name) == file_fingerprint(destination / name)
            for name in _inventory(source)
        )
    )


def local_copy(source, destination):
    """Mirror a cache transactionally; never mix stale and current files."""
    source = Path(source)
    destination = Path(destination)
    temp = destination.with_name(destination.name + ".copying")

    if not (source / "manifest.json").is_file():
        raise FileNotFoundError(f"Missing source manifest: {source}")

    if source.resolve() == destination.resolve():
        return source
    src = source.resolve()
    for target in (destination.resolve(), temp.resolve()):
        if target == src or src in target.parents or target in src.parents:
            raise ValueError("Source, destination and staging cache trees must not overlap")

    if _cache_matches(source, destination):
        return destination

    if temp.exists():
        shutil.rmtree(temp)

    try:
        shutil.copytree(source, temp, copy_function=shutil.copy2)
        if not _cache_matches(source, temp):
            raise OSError(f"Copied cache missing/incomplete: {destination}")

        if destination.exists():
            shutil.rmtree(destination)

        os.rename(temp, destination)
    finally:
        if temp.exists():
            shutil.rmtree(temp)

    return destination


def local_copy_if_space(source, destination, buffer_gib=4):
    """Avoid a failed multi-GiB copy when Colab's ephemeral disk is full."""
    source = Path(source)
    destination = Path(destination)
    if not (source / "manifest.json").is_file():
        raise FileNotFoundError(f"Missing source manifest: {source}")
    if buffer_gib < 0:
        raise ValueError("buffer_gib must be nonnegative")

    if _cache_matches(source, destination):
        print("Using verified Colab-local cache:", destination, flush=True)
        return destination

    required = sum(_inventory(source).values())
    base = destination.parent
    base.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(base).free

    if free <= required + int(buffer_gib * 1024**3):
        print("Insufficient Colab disk for local cache; using Drive cache:", source, flush=True)
        return source

    target = local_copy(source, destination)
    print("Copied preprocessing cache to Colab:", target, flush=True)
    return target
