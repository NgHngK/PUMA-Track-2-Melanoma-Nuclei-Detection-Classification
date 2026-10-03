from __future__ import annotations

from pathlib import Path

from ..config import PipelineConfig


def stage1_run_directory(config: PipelineConfig, *, create: bool = True) -> Path:
    directory = config.paths.stage1_outputs / "run"
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def active_stage1_run(config: PipelineConfig) -> Path:
    directory = stage1_run_directory(config, create=False)
    if not directory.is_dir():
        raise FileNotFoundError("Stage-1 outputs not found. Run notebook 01_Train_Stage1.ipynb first.")
    return directory

