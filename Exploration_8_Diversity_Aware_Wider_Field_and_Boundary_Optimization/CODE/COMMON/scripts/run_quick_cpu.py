from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(command: list[str]) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.check_call(command)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Exploration-8 quick-benchmark CPU stages after Colab cache extraction."
    )
    parser.add_argument(
        "--paths", required=True, type=Path, help="Usually INPUTS/QUICK_BENCHMARK/PATHS_LOCAL.json"
    )
    parser.add_argument(
        "--stage",
        choices=["preflight", "p8d", "p8c", "exposure", "p8e", "p8f", "p8g"],
        default="preflight",
    )
    parser.add_argument(
        "--selected-wide", choices=["gaussian192", "annular192"], default="gaussian192"
    )
    parser.add_argument("--route", choices=["A", "C"], default="A")
    parser.add_argument("--exposure", choices=["inverse", "group"], default="inverse")
    parser.add_argument("--tiera", action="store_true")
    args = parser.parse_args()

    paths_file = args.paths.resolve()
    paths = json.loads(paths_file.read_text(encoding="utf-8"))
    manifest = Path(paths["manifest"])
    output = Path(paths["output_root"])
    cache = output / "caches"
    folds = paths_file.parent / "TRAIN_FOLDS.json"
    cal_folds = paths_file.parent / "CAL_FOLDS.json"
    python = sys.executable

    validate = ROOT / "COMMON" / "scripts" / "validate_inputs.py"
    build_folds = ROOT / "COMMON" / "scripts" / "build_group_folds.py"
    if args.stage == "preflight":
        run([python, str(validate), "--manifest", str(manifest), "--check-bounds"])
        if not folds.exists():
            run([python, str(build_folds), "--manifest", str(manifest), "--out", str(folds)])
        if not cal_folds.exists():
            run(
                [
                    python,
                    str(build_folds),
                    "--manifest",
                    str(manifest),
                    "--out",
                    str(cal_folds),
                    "--cal",
                ]
            )
        print(f"Preflight passed. Frozen folds: {folds} and {cal_folds}")
        return

    required_cache_files = [
        cache / f"{name}.npy"
        for name in ("local_gaussian96", "gaussian192", "annular192", "scale_only192", "tiera16")
    ]
    missing = [str(path) for path in required_cache_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Colab-generated caches are required for CPU experiment stages. Copy the complete cache directory "
            f"to {cache}. Missing examples: {missing[:3]}"
        )
    if not folds.is_file():
        raise FileNotFoundError(f"run --stage preflight first: {folds}")

    common = ["--manifest", str(manifest), "--cache-dir", str(cache), "--folds", str(folds)]
    scripts = ROOT / "TRAINING" / "scripts"
    if args.stage == "p8d":
        command = [
            python,
            str(scripts / "run_p8d_representation_audit.py"),
            *common,
            "--out-dir",
            str(output / "P8D"),
        ]
    elif args.stage == "p8c":
        command = [
            python,
            str(scripts / "run_p8c_diversity_density.py"),
            *common,
            "--out-dir",
            str(output / "P8C_SUPPORT_AWARE"),
            "--device",
            "cpu",
        ]
    elif args.stage == "exposure":
        command = [
            python,
            str(scripts / "run_p8c_group_exposure.py"),
            *common,
            "--out-dir",
            str(output / "P8C_EXPOSURE"),
            "--device",
            "cpu",
        ]
    elif args.stage == "p8e":
        command = [
            python,
            str(scripts / "run_p8e_routing.py"),
            *common,
            "--representation",
            args.selected_wide,
            "--out-dir",
            str(output / "P8E"),
            "--exposure",
            args.exposure,
            "--device",
            "cpu",
        ]
    elif args.stage == "p8f":
        command = [
            python,
            str(scripts / "run_p8f_tiera_factorial.py"),
            *common,
            "--representation",
            args.selected_wide,
            "--route",
            args.route,
            "--out-dir",
            str(output / "P8F"),
            "--exposure",
            args.exposure,
            "--device",
            "cpu",
        ]
    else:
        command = [
            python,
            str(scripts / "run_p8g_joint_refit.py"),
            *common,
            "--selected-wide",
            args.selected_wide,
            "--route",
            args.route,
            "--out-dir",
            str(output / "P8G"),
            "--exposure",
            args.exposure,
            "--device",
            "cpu",
        ]
        if args.tiera:
            command.append("--tiera")
    run(command)


if __name__ == "__main__":
    main()
