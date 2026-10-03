from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from .cache_integrity import read_cache
from .config import sha256_file
from .grouping import (
    build_cal_folds,
    build_train_folds,
    load_fold_payload,
    save_fold_payload,
)
from .manifest import manifest_summary, read_manifest
from .runtime import environment_report
from .stageio import cache_expected, manifest_sha256

STAGES = {
    'smoke',
    'prepare',
    'p8c',
    'p8c_exposure',
    'p8d',
    'p8e',
    'p8f',
    'p8g',
    'final_refit',
    'boundary',
    'validate',
    'test',
}
CONFIRMATORY_STAGES = {'final_refit', 'boundary', 'validate', 'test'}
CACHE_NAMES = ('local_gaussian96', 'gaussian192', 'annular192', 'scale_only192', 'tiera16')
SEEDS = {17, 29, 43}


def _project_root(root: str | Path | None = None) -> Path:
    p = Path(root).expanduser().resolve() if root else Path(__file__).resolve().parents[4]
    if not (p / 'CODE/COMMON/src/puma_exploration8').is_dir():
        raise FileNotFoundError(f'not an Exploration-8 project root: {p}')
    return p


def _resolve_relative(base: Path, value: str) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def resolve_paths(
    root: str | Path | None = None, paths_json: str | Path | None = None
) -> dict[str, str]:
    root = _project_root(root)
    paths_file = (
        Path(paths_json).expanduser()
        if paths_json
        else root / 'INPUTS/QUICK_BENCHMARK/PATHS_COLAB.json'
    )
    if not paths_file.is_absolute():
        paths_file = (root / paths_file).resolve()
    if not paths_file.is_file():
        raise FileNotFoundError(f'PATHS_JSON not found: {paths_file}')

    paths = json.loads(paths_file.read_text())
    if not isinstance(paths, dict):
        raise ValueError(f'PATHS_JSON must contain an object: {paths_file}')

    bundled_manifest = root / 'INPUTS/QUICK_BENCHMARK/EXPLORATION8_QUICK_MANIFEST.csv'
    raw_manifest = str(paths.get('manifest', '')).strip()
    manifest = (
        _resolve_relative(paths_file.parent, raw_manifest) if raw_manifest else bundled_manifest
    )
    if (
        'Exploration_8_Diversity_Aware_Wider_Field_and_Boundary_Optimization' in raw_manifest
        or not manifest.is_file()
    ):
        if not bundled_manifest.is_file():
            raise FileNotFoundError(f'manifest not found: {manifest}')
        manifest = bundled_manifest

    raw_output = str(paths.get('output_root', '')).strip()
    output = (
        _resolve_relative(paths_file.parent, raw_output)
        if raw_output
        else root / 'RESULTS/QUICK_BENCHMARK'
    )
    if 'Exploration_8_Diversity_Aware_Wider_Field_and_Boundary_Optimization' in raw_output:
        output = root / 'RESULTS/QUICK_BENCHMARK'

    paths['manifest'] = str(manifest.resolve())
    paths['output_root'] = str(output.resolve())
    for key in ('uni2_weights', 'stage1_proposals', 'runtime_local'):
        value = str(paths.get(key, '')).strip()
        if value:
            paths[key] = str(_resolve_relative(paths_file.parent, value))
    return paths


def _device(value: str) -> str:
    if value == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if value == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use device="auto" or "cpu"')
    if value not in {'cuda', 'cpu'}:
        raise ValueError('device must be "auto", "cuda", or "cpu"')
    return value


def _script(root: Path, group: str, name: str) -> Path:
    p = root / 'CODE' / group / 'scripts' / name
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def _run(command: list[str]) -> None:
    print('$', ' '.join(map(str, command)))
    subprocess.run(command, check=True)


def _with_strict(command: list[str], strict: bool) -> list[str]:
    if strict:
        command.append('--strict-hash')
    return command


def _require(paths: list[Path], stage: str) -> None:
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise RuntimeError(f'{stage} prerequisites missing: ' + ', '.join(map(str, missing)))


def _input_validation(root: Path, paths: dict[str, str], out: Path):
    manifest_path = Path(paths['manifest'])
    manifest_hash = sha256_file(manifest_path)
    # Read/resolve rows without opening every image, then bind the validation stamp to
    # cheap image metadata. Replaced/moved images therefore invalidate the fast path.
    quick = read_manifest(manifest_path, check_bounds=False)
    image_stats = []
    for image in sorted(set(quick.image.astype(str))):
        stat = Path(image).stat()
        image_stats.append((image, int(stat.st_size), int(stat.st_mtime_ns)))
    fingerprint = hashlib.sha256(
        (manifest_hash + json.dumps(image_stats, separators=(',', ':'))).encode()
    ).hexdigest()
    stamp = out / 'INPUT_VALIDATION.json'
    if stamp.is_file():
        try:
            saved = json.loads(stamp.read_text())
            if saved.get('fingerprint') == fingerprint:
                return quick
        except (json.JSONDecodeError, OSError):
            pass
    df = read_manifest(manifest_path, check_bounds=True)
    stamp.write_text(
        json.dumps(
            {
                'manifest_sha256': manifest_hash,
                'fingerprint': fingerprint,
            },
            indent=2,
        )
    )
    return df


def _materialize_folds(target: Path, bundled: Path, role: str, df, builder):
    if target.is_file():
        payload = load_fold_payload(target, df, role)
    elif bundled.is_file():
        load_fold_payload(bundled, df, role)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundled, target)
        payload = load_fold_payload(target, df, role)
    else:
        payload = builder(df)
        save_fold_payload(payload, target)
        payload = load_fold_payload(target, df, role)
    return payload


def _ensure_train_folds(paths: dict[str, str], out: Path, df) -> Path:
    target = out / 'TRAIN_FOLDS.json'
    bundled = Path(paths['manifest']).parent / 'TRAIN_FOLDS.json'
    _materialize_folds(target, bundled, 'TRAIN', df, build_train_folds)
    return target


def _ensure_cal_folds(paths: dict[str, str], out: Path, df) -> Path:
    target = out / 'CAL_FOLDS.json'
    bundled = Path(paths['manifest']).parent / 'CAL_FOLDS.json'
    _materialize_folds(target, bundled, 'CAL', df, build_cal_folds)
    return target


def _ensure_caches(root: Path, paths: dict[str, str], out: Path, device: str) -> Path:
    cache = out / 'caches'
    mh = manifest_sha256(paths['manifest'])
    try:
        for name in CACHE_NAMES:
            # Warm-cache verification stays cheap. Confirmatory scripts perform their own strict rehash.
            read_cache(cache / name, cache_expected(mh, name), strict_hash=False)
        return cache
    except (FileNotFoundError, ValueError) as exc:
        print('cache rebuild required:', exc)

    weights = Path(str(paths.get('uni2_weights', ''))).expanduser()
    if not weights.is_file():
        raise FileNotFoundError(f'UNI2-h checkpoint unavailable: {weights}')
    _run(
        [
            sys.executable,
            str(_script(root, 'TRAINING', 'prepare_exploration8_caches.py')),
            '--manifest',
            paths['manifest'],
            '--weights',
            str(weights),
            '--out-dir',
            str(cache),
            '--device',
            device,
        ]
    )
    for name in CACHE_NAMES:
        read_cache(cache / name, cache_expected(mh, name), strict_hash=False)
    return cache


def _decision_file(out: Path) -> Path:
    return out / 'WORKFLOW_DECISIONS.json'


def load_decisions(output_root: str | Path) -> dict:
    path = _decision_file(Path(output_root))
    raw = json.loads(path.read_text()) if path.is_file() else {}
    values = {'exposure': 'inverse'}
    for key, item in raw.items():
        values[key] = item['value'] if isinstance(item, dict) and 'value' in item else item
    return values


def freeze_decision(
    root: str | Path,
    key: str,
    value,
    *,
    source: str,
    paths_json: str | Path | None = None,
) -> Path:
    validators = {
        'exposure': lambda x: x in {'inverse', 'group'},
        'selected_wider': lambda x: x in {'gaussian192', 'annular192'},
        'selected_route': lambda x: x in {'A', 'C', 'NONE'},
        'tiera': lambda x: isinstance(x, bool),
    }
    if key not in validators or not validators[key](value):
        raise ValueError(f'invalid {key} decision: {value!r}')

    paths = resolve_paths(root, paths_json)
    out = Path(paths['output_root'])
    out.mkdir(parents=True, exist_ok=True)
    file = _decision_file(out)
    data = json.loads(file.read_text()) if file.is_file() else {}
    previous = data.get(key)
    previous_value = previous.get('value') if isinstance(previous, dict) else previous
    if previous is not None and previous_value != value:
        raise RuntimeError(
            f'{key} is already frozen as {previous_value!r}. '
            'Use a new output_root for a different scientific branch.'
        )
    data[key] = {'value': value, 'source': str(source)}
    file.write_text(json.dumps(data, indent=2, sort_keys=True))
    print(f'frozen: {key}={value!r} [{source}] -> {file}')
    return file


def _decision(decisions: dict, key: str, allowed, stage: str):
    value = decisions.get(key)
    if value not in allowed:
        raise RuntimeError(f'{stage} requires frozen decision {key}; current value: {value!r}')
    return value


def status(root: str | Path, *, paths_json: str | Path | None = None, seed: int = 17) -> dict:
    paths = resolve_paths(root, paths_json)
    out = Path(paths['output_root'])
    candidate = out / f'FINAL_CANDIDATE_seed{seed}'
    state = {
        'output_root': str(out),
        'decisions': load_decisions(out),
        'train_folds': (out / 'TRAIN_FOLDS.json').is_file(),
        'caches': all(
            (out / 'caches' / name).with_suffix('.json').is_file() for name in CACHE_NAMES
        ),
        'p8c': (out / 'P8C_SUPPORT_AWARE/P8C_RESULTS.json').is_file(),
        'p8c_exposure': (out / 'P8C_EXPOSURE/P8C_EXPOSURE_RESULTS.json').is_file(),
        'p8d': (out / 'P8D/P8D.json').is_file(),
        'p8e': (out / 'P8E/P8E_RESULTS.json').is_file(),
        'p8f': (out / 'P8F/P8F_RESULTS.json').is_file(),
        'p8g': (out / 'P8G/P8G_RESULTS.json').is_file(),
        'visual_freeze': (candidate / 'VISUAL_FREEZE.json').is_file(),
        'boundary_freeze': (candidate / 'BOUNDARY_FREEZE.json').is_file(),
        'complete_freeze': (candidate / 'COMPLETE_PIPELINE_FREEZE.json').is_file(),
        'validate_result': (candidate / 'VALIDATE_RESULT.json').is_file(),
        'test_result': (candidate / 'TEST_RESULT.json').is_file(),
    }
    print(json.dumps(state, indent=2))
    return state


def _stage_result(out: Path, stage: str, seed: int) -> Path | None:
    candidate = out / f'FINAL_CANDIDATE_seed{seed}'
    return {
        'p8c': out / 'P8C_SUPPORT_AWARE/P8C_RESULTS.json',
        'p8c_exposure': out / 'P8C_EXPOSURE/P8C_EXPOSURE_RESULTS.json',
        'p8d': out / 'P8D/P8D.json',
        'p8e': out / 'P8E/P8E_RESULTS.json',
        'p8f': out / 'P8F/P8F_RESULTS.json',
        'p8g': out / 'P8G/P8G_RESULTS.json',
        'validate': candidate / 'VALIDATE_RESULT.json',
        'test': candidate / 'TEST_RESULT.json',
    }.get(stage)


def _reuse(path: Path | None, stage: str, force: bool) -> bool:
    if path is not None and path.is_file() and not force:
        print(f'{stage}: already complete; reusing {path}')
        return True
    return False


def run_stage(
    root: str | Path,
    stage: str,
    *,
    paths_json: str | Path | None = None,
    device: str = 'auto',
    seed: int = 17,
    strict_hash: bool | None = None,
    run_h2: bool = False,
    fit_temperature: bool = False,
    force: bool = False,
) -> dict:
    if stage not in STAGES:
        raise ValueError(f'unknown stage {stage!r}; choose one of {sorted(STAGES)}')
    if seed not in SEEDS:
        raise ValueError(f'seed must be one of {sorted(SEEDS)}')

    root = _project_root(root)
    device = _device(device)
    strict = stage in CONFIRMATORY_STAGES if strict_hash is None else bool(strict_hash)

    if stage == 'smoke':
        from .selfcheck import run

        result = run(Path(tempfile.mkdtemp(prefix='p8_smoke_')))
        print(result)
        return {'stage': stage, 'result': result}

    paths = resolve_paths(root, paths_json)
    out = Path(paths['output_root'])
    out.mkdir(parents=True, exist_ok=True)
    # Protected completed results are immutable. Do not rehash caches or risk reopening
    # a one-shot role just to discover that the result already exists.
    if stage in {'validate', 'test'} and _reuse(_stage_result(out, stage, seed), stage, force):
        return {'stage': stage, 'output_root': str(out), 'reused': True}
    df = _input_validation(root, paths, out)
    train_folds = _ensure_train_folds(paths, out, df)
    cache = _ensure_caches(root, paths, out, device)
    decisions = load_decisions(out)

    print(json.dumps(manifest_summary(df), indent=2))
    print(json.dumps(environment_report(root), indent=2))
    print('stage:', stage, '| device:', device, '| strict_hash:', strict)
    if stage == 'prepare':
        return {
            'stage': stage,
            'output_root': str(out),
            'cache': str(cache),
            'train_folds': str(train_folds),
        }
    if _reuse(_stage_result(out, stage, seed), stage, force):
        return {'stage': stage, 'output_root': str(out), 'decisions': decisions, 'reused': True}

    manifest = paths['manifest']
    py = sys.executable
    training = lambda name: str(_script(root, 'TRAINING', name))
    evaluation = lambda name: str(_script(root, 'EVALUATION', name))

    if stage == 'p8c':
        _run(
            _with_strict(
                [
                    py,
                    training('run_p8c_diversity_density.py'),
                    '--manifest',
                    manifest,
                    '--cache-dir',
                    str(cache),
                    '--folds',
                    str(train_folds),
                    '--out-dir',
                    str(out / 'P8C_SUPPORT_AWARE'),
                    '--device',
                    device,
                ],
                strict,
            )
        )
    elif stage == 'p8c_exposure':
        _run(
            _with_strict(
                [
                    py,
                    training('run_p8c_group_exposure.py'),
                    '--manifest',
                    manifest,
                    '--cache-dir',
                    str(cache),
                    '--folds',
                    str(train_folds),
                    '--out-dir',
                    str(out / 'P8C_EXPOSURE'),
                    '--device',
                    device,
                ],
                strict,
            )
        )
    elif stage == 'p8d':
        _run(
            _with_strict(
                [
                    py,
                    training('run_p8d_representation_audit.py'),
                    '--manifest',
                    manifest,
                    '--cache-dir',
                    str(cache),
                    '--folds',
                    str(train_folds),
                    '--out-dir',
                    str(out / 'P8D'),
                ],
                strict,
            )
        )
    elif stage == 'p8e':
        wide = _decision(decisions, 'selected_wider', {'gaussian192', 'annular192'}, stage)
        _run(
            _with_strict(
                [
                    py,
                    training('run_p8e_routing.py'),
                    '--manifest',
                    manifest,
                    '--cache-dir',
                    str(cache),
                    '--folds',
                    str(train_folds),
                    '--representation',
                    wide,
                    '--out-dir',
                    str(out / 'P8E'),
                    '--exposure',
                    decisions['exposure'],
                    '--device',
                    device,
                ],
                strict,
            )
        )
    elif stage == 'p8f':
        wide = _decision(decisions, 'selected_wider', {'gaussian192', 'annular192'}, stage)
        route = _decision(decisions, 'selected_route', {'A', 'C'}, stage)
        _run(
            _with_strict(
                [
                    py,
                    training('run_p8f_tiera_factorial.py'),
                    '--manifest',
                    manifest,
                    '--cache-dir',
                    str(cache),
                    '--folds',
                    str(train_folds),
                    '--representation',
                    wide,
                    '--route',
                    route,
                    '--out-dir',
                    str(out / 'P8F'),
                    '--exposure',
                    decisions['exposure'],
                    '--device',
                    device,
                ],
                strict,
            )
        )
    elif stage == 'p8g':
        wide = _decision(decisions, 'selected_wider', {'gaussian192', 'annular192'}, stage)
        route = _decision(decisions, 'selected_route', {'A', 'C'}, stage)
        tiera = _decision(decisions, 'tiera', {True, False}, stage)
        cmd = [
            py,
            training('run_p8g_joint_refit.py'),
            '--manifest',
            manifest,
            '--cache-dir',
            str(cache),
            '--folds',
            str(train_folds),
            '--selected-wide',
            wide,
            '--route',
            route,
            '--out-dir',
            str(out / 'P8G'),
            '--exposure',
            decisions['exposure'],
            '--device',
            device,
        ]
        if tiera:
            cmd.append('--tiera')
        _run(_with_strict(cmd, strict))
    else:
        _run_protected_or_final(
            root,
            stage,
            paths,
            out,
            cache,
            df,
            decisions,
            device,
            seed,
            strict,
            run_h2,
            fit_temperature,
            force,
        )

    return {'stage': stage, 'output_root': str(out), 'decisions': decisions}


def _run_protected_or_final(
    root,
    stage,
    paths,
    out,
    cache,
    df,
    decisions,
    device,
    seed,
    strict,
    run_h2,
    fit_temperature,
    force,
):
    py = sys.executable
    training = lambda name: str(_script(root, 'TRAINING', name))
    evaluation = lambda name: str(_script(root, 'EVALUATION', name))
    manifest = paths['manifest']
    exposure = decisions['exposure']
    candidate = out / f'FINAL_CANDIDATE_seed{seed}'
    baseline = out / f'FINAL_BASELINE_seed{seed}'

    if stage == 'final_refit':
        route = _decision(decisions, 'selected_route', {'NONE', 'A', 'C'}, stage)
        tiera = _decision(decisions, 'tiera', {True, False}, stage)
        candidate_done = candidate / 'VISUAL_FREEZE.json'
        baseline_done = baseline / 'VISUAL_FREEZE.json'
        if force or not candidate_done.is_file():
            cmd = [
                py,
                training('fit_final_visual.py'),
                '--manifest',
                manifest,
                '--cache-dir',
                str(cache),
                '--out-dir',
                str(candidate),
                '--route',
                route,
                '--exposure',
                exposure,
                '--seed',
                str(seed),
                '--device',
                device,
            ]
            if route in {'A', 'C'}:
                cmd += [
                    '--selected-wide',
                    _decision(decisions, 'selected_wider', {'gaussian192', 'annular192'}, stage),
                ]
            if tiera:
                cmd.append('--tiera')
            _run(_with_strict(cmd, strict))
        else:
            print(f'final candidate: reuse {candidate_done}')

        if force or not baseline_done.is_file():
            baseline_cmd = [
                py,
                training('fit_final_visual.py'),
                '--manifest',
                manifest,
                '--cache-dir',
                str(cache),
                '--out-dir',
                str(baseline),
                '--route',
                'NONE',
                '--tiera',
                '--exposure',
                exposure,
                '--seed',
                str(seed),
                '--device',
                device,
            ]
            _run(_with_strict(baseline_cmd, strict))
        else:
            print(f'final baseline: reuse {baseline_done}')
        return

    visual_freeze = candidate / 'VISUAL_FREEZE.json'
    checkpoint = candidate / f'final_visual_seed{seed}.pt'
    training_meta = candidate / 'TRAINING_METADATA.json'
    boundary_freeze = candidate / 'BOUNDARY_FREEZE.json'
    complete_freeze = candidate / 'COMPLETE_PIPELINE_FREEZE.json'

    if stage == 'boundary':
        if not force and boundary_freeze.is_file() and complete_freeze.is_file():
            print(f'boundary: already complete; reuse {complete_freeze}')
            return
        cal_folds = _ensure_cal_folds(paths, out, df)
        _require([visual_freeze, checkpoint, training_meta, cal_folds], stage)
        cal_pred = candidate / 'CAL_VISUAL.npz'
        _run(
            _with_strict(
                [
                    py,
                    evaluation('predict_cached.py'),
                    '--manifest',
                    manifest,
                    '--cache-dir',
                    str(cache),
                    '--role',
                    'CAL',
                    '--visual-freeze',
                    str(visual_freeze),
                    '--checkpoint',
                    str(checkpoint),
                    '--out',
                    str(cal_pred),
                    '--device',
                    device,
                ],
                strict,
            )
        )
        cmd = [
            py,
            evaluation('run_p8h_boundary.py'),
            '--npz',
            str(cal_pred),
            '--manifest',
            manifest,
            '--folds',
            str(cal_folds),
            '--visual-freeze',
            str(visual_freeze),
            '--training-metadata',
            str(training_meta),
            '--out',
            str(candidate / 'P8H_BOUNDARY.json'),
            '--boundary-freeze',
            str(boundary_freeze),
        ]
        if run_h2:
            cmd.append('--run-h2')
        if fit_temperature:
            cmd.append('--fit-temperature')
        _run(cmd)
        _run(
            [
                py,
                evaluation('freeze_complete_pipeline.py'),
                '--visual-freeze',
                str(visual_freeze),
                '--boundary-freeze',
                str(boundary_freeze),
                '--out',
                str(complete_freeze),
            ]
        )
        return

    if stage == 'validate':
        baseline_freeze = baseline / 'VISUAL_FREEZE.json'
        baseline_ck = baseline / f'final_visual_seed{seed}.pt'
        _require(
            [
                visual_freeze,
                checkpoint,
                boundary_freeze,
                complete_freeze,
                baseline_freeze,
                baseline_ck,
            ],
            stage,
        )
        candidate_pred = candidate / 'VALIDATE_COMPLETE.npz'
        baseline_pred = baseline / 'VALIDATE_BASELINE.npz'
        _run(
            _with_strict(
                [
                    py,
                    evaluation('predict_cached.py'),
                    '--manifest',
                    manifest,
                    '--cache-dir',
                    str(cache),
                    '--role',
                    'VALIDATE',
                    '--visual-freeze',
                    str(visual_freeze),
                    '--checkpoint',
                    str(checkpoint),
                    '--boundary-freeze',
                    str(boundary_freeze),
                    '--complete-freeze',
                    str(complete_freeze),
                    '--out',
                    str(candidate_pred),
                    '--device',
                    device,
                ],
                strict,
            )
        )
        _run(
            _with_strict(
                [
                    py,
                    evaluation('predict_cached.py'),
                    '--manifest',
                    manifest,
                    '--cache-dir',
                    str(cache),
                    '--role',
                    'VALIDATE',
                    '--visual-freeze',
                    str(baseline_freeze),
                    '--checkpoint',
                    str(baseline_ck),
                    '--out',
                    str(baseline_pred),
                    '--device',
                    device,
                ],
                strict,
            )
        )
        _run(
            [
                py,
                evaluation('run_validate_once.py'),
                '--complete-freeze',
                str(complete_freeze),
                '--state',
                str(candidate / 'PROTECTED/VALIDATE_OPENED.json'),
                '--manifest',
                manifest,
                '--predictions',
                str(candidate_pred),
                '--baseline',
                str(baseline_pred),
                '--out',
                str(candidate / 'VALIDATE_RESULT.json'),
            ]
        )
        return

    if stage == 'test':
        promotion = candidate / 'VALIDATE_RESULT.json'
        _require([visual_freeze, checkpoint, boundary_freeze, complete_freeze, promotion], stage)
        test_pred = candidate / 'TEST_COMPLETE.npz'
        _run(
            _with_strict(
                [
                    py,
                    evaluation('predict_cached.py'),
                    '--manifest',
                    manifest,
                    '--cache-dir',
                    str(cache),
                    '--role',
                    'TEST',
                    '--visual-freeze',
                    str(visual_freeze),
                    '--checkpoint',
                    str(checkpoint),
                    '--boundary-freeze',
                    str(boundary_freeze),
                    '--complete-freeze',
                    str(complete_freeze),
                    '--out',
                    str(test_pred),
                    '--device',
                    device,
                ],
                strict,
            )
        )
        _run(
            [
                py,
                evaluation('run_test_once.py'),
                '--promotion',
                str(promotion),
                '--complete-freeze',
                str(complete_freeze),
                '--state',
                str(candidate / 'PROTECTED/TEST_OPENED.json'),
                '--manifest',
                manifest,
                '--predictions',
                str(test_pred),
                '--out',
                str(candidate / 'TEST_RESULT.json'),
            ]
        )
        return

    raise AssertionError(stage)
