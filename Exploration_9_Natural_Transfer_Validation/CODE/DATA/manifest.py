from __future__ import annotations
from pathlib import Path
import csv, math
from PIL import Image
from CODE.COMMON.constants import CLASSES, PROTECTED_ROLES
from CODE.COMMON.exceptions import ManifestError, ClassMapError, ProtectedRoleError, CoordinateError
from CODE.COMMON.schemas import ManifestRow

REQUIRED = {"uid", "roi_id", "image_path", "x", "y", "label", "class_name", "role"}


def normalize_role(role: str) -> str:
    return role.strip().lower()


def assert_role_allowed_for_e9_transfer(role: str) -> None:
    role = normalize_role(role)
    if role in PROTECTED_ROLES or role != 'train':
        raise ProtectedRoleError(
            f"Role {role!r} is not allowed in E9 natural-transfer training/model selection"
        )


def read_manifest(
    path: str | Path, require_images: bool = True, transfer_only: bool = False
) -> list[ManifestRow]:
    path = Path(path)
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ManifestError('Empty manifest')
    if not REQUIRED.issubset(rows[0]):
        raise ManifestError(f"Manifest missing columns: {sorted(REQUIRED-set(rows[0]))}")
    seen = set()
    out = []
    image_sizes = {}
    for raw in rows:
        uid = raw['uid'].strip()
        if not uid or uid in seen:
            raise ManifestError(f"Duplicate/empty uid: {uid!r}")
        seen.add(uid)
        label = int(raw['label'])
        if not 0 <= label < len(CLASSES):
            raise ClassMapError(f"Invalid label {label} for {uid}")
        name = raw['class_name'].strip()
        if name != CLASSES[label]:
            raise ClassMapError(f"Class mismatch for {uid}: {label} -> {name}")
        role = normalize_role(raw['role'])
        if transfer_only:
            assert_role_allowed_for_e9_transfer(role)
        p = Path(raw['image_path'])
        if not p.is_absolute():
            p = (path.parent / p).resolve()
        if require_images and not p.is_file():
            raise ManifestError(f"Missing image: {p}")
        x, y = float(raw['x']), float(raw['y'])
        if not math.isfinite(x) or not math.isfinite(y):
            raise CoordinateError(f"Nonfinite coordinate {uid}")
        if require_images:
            key = str(p)
            if key not in image_sizes:
                with Image.open(p) as im:
                    image_sizes[key] = im.size
            w, h = image_sizes[key]
            if not (0 <= x < w and 0 <= y < h):
                raise CoordinateError(f"Coordinate outside image for {uid}: {(x,y)} vs {(w,h)}")
        out.append(
            ManifestRow(
                uid,
                raw['roi_id'].strip(),
                str(p),
                x,
                y,
                label,
                name,
                role,
                raw.get('coordinate_source', 'gt').strip() or 'gt',
            )
        )
    return out
