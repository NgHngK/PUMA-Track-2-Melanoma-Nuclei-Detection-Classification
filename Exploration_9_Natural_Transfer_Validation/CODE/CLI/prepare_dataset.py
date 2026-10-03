from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from CODE.DATA.puma_dataset import (
    build_manifest_rows,
    load_excluded_rois,
    write_manifest,
    audit_context_center_pair,
    discover_rois,
)
from CODE.DATA.provenance import load_roi_provenance, assert_complete_provenance
from CODE.DATA.manifest import read_manifest
from CODE.DATA.tier_a_extract import extract_tier_a_resumable, load_tier_a
from CODE.COMMON.hashing import sha256_file, canonical_json_hash
from CODE.COMMON.exceptions import ScientificContractError


def _write_exposure_ledger(rows, path: Path):
    fields = ['roi_id', 'role', 'historical_exposure', 'eligible_e9_natural_transfer']
    seen = {}
    for r in rows:
        seen[r['roi_id']] = {
            'roi_id': r['roi_id'],
            'role': r['role'],
            'historical_exposure': r.get('historical_exposure', 'UNKNOWN'),
            'eligible_e9_natural_transfer': str(r['role'] == 'train').lower(),
        }
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(seen[k] for k in sorted(seen))


def _already_complete(out: Path, *, need_tier_a: bool) -> bool:
    manifest = out / 'e9_natural_train_roster.csv'
    summary_path = out / 'dataset_summary.json'
    ledger = out / 'E9_EXPOSURE_LEDGER.csv'
    if not (manifest.is_file() and summary_path.is_file() and ledger.is_file()):
        return False
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if sha256_file(manifest) != summary.get('manifest_sha256'):
        return False
    rows = read_manifest(manifest, require_images=True, transfer_only=True)
    if len(rows) != int(summary.get('natural_train_nucleus_count', -1)):
        return False
    if need_tier_a:
        load_tier_a(out, [r.uid for r in rows], verify_hash=True)
    return True


def main():
    p = argparse.ArgumentParser(
        description='Prepare the supplied PUMA training folder for Exploration 9'
    )
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument(
        '--provenance-csv',
        default=None,
        help='Strict mode: complete 205-ROI CSV with roi_id,role[,historical_exposure]',
    )
    p.add_argument(
        '--exploratory-all-official-train',
        action='store_true',
        help='Explicitly treat all supplied official training ROIs as exploratory TRAIN; not fresh confirmation',
    )
    p.add_argument(
        '--exclude-rois', default=None, help='Optional ROI exclusions; applied after provenance'
    )
    p.add_argument('--skip-tier-a', action='store_true')
    p.add_argument(
        '--workers', type=int, default=0, help='Tier-A workers; 0 uses all available CPU cores'
    )
    p.add_argument('--allow-non205', action='store_true', help='Only for tests/smoke fixtures')
    p.add_argument(
        '--overwrite',
        action='store_true',
        help='Rebuild prepared artifacts even if a complete preparation exists',
    )
    a = p.parse_args()
    if bool(a.provenance_csv) == bool(a.exploratory_all_official_train) and not a.allow_non205:
        raise ScientificContractError(
            'Choose exactly one: --provenance-csv for strict transfer OR --exploratory-all-official-train for explicitly exploratory full-source OOF'
        )
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not a.overwrite and _already_complete(out, need_tier_a=not a.skip_tier_a):
        print(f'[SKIP] Preparation already COMPLETE: {out}', flush=True)
        return

    excluded = load_excluded_rois(a.exclude_rois)
    role_by_roi = None
    exposure_by_roi = None
    provenance_mode = 'TEST_FIXTURE' if a.allow_non205 else None
    if a.provenance_csv:
        prov = load_roi_provenance(a.provenance_csv)
        discovered = discover_rois(a.dataset_root, require_all_four=True)
        assert_complete_provenance([x['roi_id'] for x in discovered], prov)
        role_by_roi = {k: v.role for k, v in prov.items()}
        exposure_by_roi = {k: v.historical_exposure for k, v in prov.items()}
        provenance_mode = 'STRICT_AUDITED_ROI_ROLES'
    elif a.exploratory_all_official_train:
        provenance_mode = 'EXPLORATORY_ALL_OFFICIAL_TRAIN'
    rows, summary = build_manifest_rows(
        a.dataset_root,
        excluded_rois=excluded,
        role_by_roi=role_by_roi,
        exposure_by_roi=exposure_by_roi,
        require_expected_205=not a.allow_non205,
    )
    _write_exposure_ledger(rows, out / 'E9_EXPOSURE_LEDGER.csv')
    transfer_rows = [r for r in rows if r['role'] == 'train']
    if not transfer_rows:
        raise ScientificContractError('No TRAIN ROIs remain after provenance/exclusions')
    manifest = out / 'e9_natural_train_roster.csv'
    write_manifest(transfer_rows, manifest)
    parsed = read_manifest(manifest, require_images=True, transfer_only=True)
    if len(parsed) != len(transfer_rows):
        raise RuntimeError('Manifest round-trip row count mismatch')
    first = transfer_rows[0]
    summary['context_center_audit_first_train_roi'] = audit_context_center_pair(
        first['image_path'], first['context_image_path']
    )
    summary['provenance_mode'] = provenance_mode
    summary['natural_train_roi_count'] = len({r['roi_id'] for r in transfer_rows})
    summary['natural_train_nucleus_count'] = len(transfer_rows)
    summary['manifest_sha256'] = sha256_file(manifest)
    summary['scientific_status'] = (
        'EXPLORATORY' if provenance_mode != 'STRICT_AUDITED_ROI_ROLES' else 'STRICT_ROLE_AUDITED'
    )
    summary['summary_sha256'] = canonical_json_hash(
        {k: v for k, v in summary.items() if k != 'summary_sha256'}
    )
    (out / 'dataset_summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8'
    )
    if not a.skip_tier_a:
        print('[RUN] Tier-A extraction', flush=True)
        extract_tier_a_resumable(out, parsed, workers=a.workers)
    print(
        json.dumps(
            {
                'manifest': str(manifest),
                'rows': len(transfer_rows),
                'rois': summary['natural_train_roi_count'],
                'tier_a': not a.skip_tier_a,
                'provenance_mode': provenance_mode,
                'summary': str(out / 'dataset_summary.json'),
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
