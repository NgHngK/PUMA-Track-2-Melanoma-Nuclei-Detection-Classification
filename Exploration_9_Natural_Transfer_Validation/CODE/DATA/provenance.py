from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv
from CODE.COMMON.constants import PROTECTED_ROLES
from CODE.COMMON.exceptions import ManifestError, ProtectedRoleError
from CODE.DATA.puma_dataset import normalize_roi_id

_ALLOWED = {'train', 'cal', 'validate', 'test'}
_ROLE_ALIASES = {
    'training': 'train',
    'validation': 'validate',
    'val': 'validate',
    'calibration': 'cal',
    'external': 'test',
}


@dataclass(frozen=True)
class RoiProvenance:
    roi_id: str
    role: str
    historical_exposure: str


def normalize_research_role(value: str) -> str:
    r = str(value).strip().lower()
    r = _ROLE_ALIASES.get(r, r)
    if r not in _ALLOWED:
        raise ManifestError(
            f'Unsupported research role {value!r}; expected one of {sorted(_ALLOWED)}'
        )
    return r


def load_roi_provenance(path: str | Path) -> dict[str, RoiProvenance]:
    p = Path(path)
    if not p.is_file():
        raise ManifestError(f'Provenance CSV does not exist: {p}')
    with p.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ManifestError('Empty provenance CSV')
    required = {'roi_id', 'role'}
    if not required.issubset(rows[0]):
        raise ManifestError(f'Provenance CSV missing columns {sorted(required-set(rows[0]))}')
    out = {}
    for raw in rows:
        roi = normalize_roi_id(raw['roi_id'])
        role = normalize_research_role(raw['role'])
        if roi in out:
            raise ManifestError(f'Duplicate ROI in provenance CSV: {roi}')
        exposure = (raw.get('historical_exposure') or 'UNKNOWN').strip() or 'UNKNOWN'
        out[roi] = RoiProvenance(roi, role, exposure)
    return out


def assert_complete_provenance(discovered_roi_ids, provenance: dict[str, RoiProvenance]):
    discovered = set(discovered_roi_ids)
    supplied = set(provenance)
    missing = sorted(discovered - supplied)
    extra = sorted(supplied - discovered)
    if missing:
        raise ManifestError(f'Provenance missing {len(missing)} ROI(s), e.g. {missing[0]}')
    if extra:
        raise ManifestError(f'Provenance contains unknown ROI(s), e.g. {extra[0]}')


def eligible_train_rois(provenance: dict[str, RoiProvenance]) -> set[str]:
    return {roi for roi, p in provenance.items() if p.role == 'train'}


def assert_no_protected_selected(selected_rois, provenance: dict[str, RoiProvenance]):
    for roi in selected_rois:
        p = provenance[roi]
        if p.role in PROTECTED_ROLES:
            raise ProtectedRoleError(f'Protected ROI selected for transfer: {roi} ({p.role})')
