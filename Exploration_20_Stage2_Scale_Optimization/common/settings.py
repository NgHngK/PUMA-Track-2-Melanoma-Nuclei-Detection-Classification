"""Paths used by the two Stage 2 Colab notebooks."""

from pathlib import Path
import os

project_dir = Path(os.environ.get("PUMA_PROJECT_DIR", Path(__file__).resolve().parents[1]))
dataset_dir = Path(os.environ.get("PUMA_DATASET_DIR", project_dir.parent / "Dataset"))
weights_dir = Path(
    os.environ.get("PUMA_WEIGHTS_DIR", project_dir.parent / "PUMA_pretrained_checkpoints")
)

plans = ("A", "B")


def paths():
    return {
        "dataset": dataset_dir,
        "images": dataset_dir / "01_training_dataset_tif_ROIs",
        "annotations": dataset_dir / "01_training_dataset_geojson_nuclei",
        "stage2_cache": project_dir / "preprocess/stage2",
        "results": Path(os.environ.get("PUMA_RESULTS_DIR", project_dir / "results")),
        "uni2": weights_dir / "UNI2-h/uni2_h_model.bin",
        "nito": weights_dir / "Tissue (NiTo)/Model_weights_track2/Model_weights",
    }
