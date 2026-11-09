# Exploration 20 — FOV112-224 · R224-G3 vs R224-GAT

**Short name:** `E20-FOV112224-DualGraph`  
**Scope:** Stage 2 feature preprocessing, Stage 2 training and fold-wise comparison only.
The original `stage2/`, `common/`, `preprocess/stage2/`, `results/stage2/` and notebook structure is retained. No Stage 1 code, Stage 1 cache or Stage 1 checkpoints are included or imported.

## Experiments in ONE package

| Component | Scenario 1: R224-G3 | Scenario 2: R224-GAT |
|---|---|---|
| Raw nucleus-centered crop FOVs | 112 / 224 px | 112 / 224 px |
| Frozen UNI2-h model input | 224 x 224 px | 224 x 224 px |
| Embedding dimensions | 1536 + 1536 | 1536 + 1536 |
| Tier-A / NiTo | 16 / 20 dims, unchanged | same |
| Graph radius / max neighbors / sigma | 224 px / 20 / 64 px | 224 px / 20 / 64 px |
| Neighbor aggregation | Original Gaussian-weighted G3 | Learned 4-head graph attention with Gaussian distance prior |
| Main G3 residual head | Unchanged | Unchanged |
| Train/validation folds | **Original Plan A/B, 4 folds/plan** | **Exactly the same splits** |
| Epochs, seed, optimizer, loss, checkpoint selection | Original: notebook EPOCHS=20, seed=42, AdamW, original loss, select by val nucleus Macro-F1 | Same |
| Final primary comparison | **OOF per-ROI F1** | **Same definition** |

**Fold membership unchanged:** original `stage2/fold_definitions.py` and `stage2/splits.py` are literal copies of the original definitions. Plan A: 205 validation ROIs across 4 folds, with ROI 027 kept as training-only anchor. Plan B: 206 validation ROIs across 4 folds. Each nucleus is evaluated using only the matching plan/fold's selected best checkpoint. The two models see identical training and validation ROIs within each plan/fold. There are **16 training runs total** (2 scenarios x 2 plans x 4 folds), not 16 new fold definitions.

**Important evaluation distinction:** this Stage 2-only experiment evaluates classifications at **ground-truth nuclei centroids** using the existing `stage2.metrics.roi_scores` function with its unchanged matching radius. It is NOT detector-proposal/end-to-end Stage 1+2 ROI F1. Since this ZIP excludes Stage 1, the latter cannot be computed here. Checkpoints remain selected by the original **validation nucleus Macro-F1**; ROI F1 is computed using each fold's selected best checkpoint.

## Paths and setup

Place this project beside the original project, under the same `Research/PUMA/` folder:

```text
Research/PUMA/
├── Dataset/                              # external, unchanged
├── PUMA_pretrained_checkpoints/         # external, unchanged
└── Exploration_20_Crop112-224_Graph224_G3_vs_GAT/
    ├── common/
    ├── stage2/
    ├── notebooks/
    │   ├── 01_preprocessing.ipynb
    │   └── 03_train_stage2.ipynb
    ├── preprocess/
    │   ├── stage2/                     # generated 112/224 features + R224 graph
    │   └── nito_maps/                  # cached NiTo ROI maps (when available)
    └── results/
        └── stage2/
            ├── r224_g3/plan_A/fold_0/... # per-fold best.pt / history.csv / validation_roi.json
            ├── r224_g3/plan_B/fold_0/...
            ├── r224_gat/plan_A/fold_0/...
            ├── r224_gat/plan_B/fold_0/...
            ├── exploration20_comparison.csv
            └── exploration20_comparison.json
```

The existing `common.settings.paths()` conventions are preserved. Optional environment overrides: `PUMA_PROJECT_DIR`, `PUMA_DATASET_DIR`, `PUMA_WEIGHTS_DIR`, `PUMA_RESULTS_DIR`.

## How to run (Colab, full 16-run experiment)

1. Run **`notebooks/01_preprocessing.ipynb`**. It builds `h112.npy` and `h224.npy` at new source FOVs and builds a **radius 224** `graph.npz`. It preserves the original nucleus annotations and coordinate conventions. It validates source images and encoder provenance. If a verified 112/224 cache already exists with an old radius-192 graph, it rebuilds **only** the graph, not the costly image features.
2. Run **`notebooks/03_train_stage2.ipynb`**, *all cells in order*. It automatically loops **Plan A/B x fold 0..3 x graph mode G3/GAT**, with `EPOCHS = 20`, all original optimizer and validation settings. It keeps separate checkpoints and epoch-history files for every scenario/fold. Completed matching folds can be safely skipped according to the original run-history checks.
3. The last cell calls `stage2.comparison.compare()`, shows **Plan A OOF ROI F1, Plan B OOF ROI F1 and their equal-weight mean** for each scenario, and prints the winner or tie. Its summary is stored in `results/stage2/exploration20_comparison.csv` and `.json`.

Command-line single fold (the notebook handles all 16 runs automatically):

```bash
python -m stage2 --features preprocess/stage2 --results results --plan A --fold 0 --epochs 20 --batch-size 0 --mode G3
python -m stage2 --features preprocess/stage2 --results results --plan A --fold 0 --epochs 20 --batch-size 0 --mode GAT
```

## Preprocessing and graph cache requirements

- **Do not use old `h96.npy`/`h192.npy` feature arrays** as if they were the new FOVs. `h112.npy` and `h224.npy` must be extracted from source TIFFs through the frozen UNI2-h model (each then resized to 224x224 for UNI2 input).
- The new graph has radius 224, k=20 and Gaussian sigma=64, built once from the same original nucleus centroids and shared by both scenarios. Graph attention changes **learned aggregation**, not which neighbors are eligible.
- Source TIFFs, GeoJSON annotations, UNI2-h weights and NiTo weights stay external. Do not place large datasets or model weights inside this source ZIP.
- Versioned feature manifests and graph hashes prevent silently using stale graph caches or incompatible FOV tensors. Use separate results folders for each scenario.
- The G3 model architecture and training algorithm are unchanged apart from the graph radius and 112/224 input features. GAT replaces only neighbor aggregation with a 4-head distance-aware attention step and adds parameters for Q/K/V; the remaining model, optimizer, loss and folds are unchanged.

## Verification and limits

Static compilation, synthetic forward/backward passes for both graph modes, and **one-epoch synthetic Stage 2 training + ROI F1 evaluation + verified checkpoint skip** have been tested. Real PUMA model quality cannot be claimed until all 16 runs are executed with the dataset and external pretrained weights.
