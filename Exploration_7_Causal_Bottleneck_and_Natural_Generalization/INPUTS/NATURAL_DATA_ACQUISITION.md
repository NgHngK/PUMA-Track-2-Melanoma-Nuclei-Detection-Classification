# Exploration-7 natural cohort acquisition

## What can satisfy the confirmatory contract

Use either:

1. the hidden PUMA preliminary/final cohort, obtained with explicit organizer permission for this research; or
2. newly collected melanoma H&E ROIs annotated with the same ten PUMA nucleus classes.

The public PUMA training archive already used in Explorations 1–6 cannot become a new confirmatory cohort by resplitting it. The local archive contains 205 ROIs and all 205 were exposed by Exploration 6. PUMA v5 restores one corrected public ROI (`training_set_metastatic_roi_103`), but one ROI cannot provide independent train, calibration, and test cohorts.

The PUMA paper states that the 10 preliminary and 94 final-test ROIs remain hidden until 10 October 2029 and that earlier educational/collaborative access may be requested from the corresponding author:

- Mark Schuiveling: `m.schuiveling@umcutrecht.nl`
- Paper/data-availability statement: <https://pmc.ncbi.nlm.nih.gov/articles/PMC11837757/>
- Official dataset page: <https://puma.grand-challenge.org/dataset/>
- Public v5 archive (not sufficient by itself): <https://zenodo.org/records/15050523>

Request the 1024×1024 TIFF ROIs, nuclei GeoJSON annotations, 5120×5120 context ROIs for Exp5, and the strongest de-identified patient/case/slide grouping map permitted. Also request explicit permission to partition these data into Exploration-7 train, calibration, and untouched test cohorts.

Suggested request:

> I am conducting a preregistered follow-up study of frozen-representation nuclei classification on PUMA. May I obtain early educational/collaborative access to unused labeled PUMA ROIs, including 1024×1024 TIFFs, nuclei GeoJSONs, context ROIs, and de-identified grouping identifiers? I need permission to partition them once into group-disjoint training, calibration, and untouched test cohorts. I will not use the challenge leaderboard or inspect test performance before protocol freeze and will follow any data-use restrictions you specify.

## Local preprocessing after data access

Place the received cohort in this PUMA layout:

```text
NEW_DATASET_ROOT/
  01_training_dataset_tif_ROIs/
  01_training_dataset_geojson_nuclei/
```

Create a frozen split map from `INPUTS/TEMPLATES/SPLIT_MAP_TEMPLATE.csv`. Every ROI must occur once, each group must occur in only one split, and the map must include the strongest identifiers supplied by the data provider.

Run locally before uploading images/manifests to Drive:

```powershell
$env:PYTHONPATH = "<PROMPT7_ROOT>\CODE\COMMON\src"
python "<PROMPT7_ROOT>\CODE\COMMON\scripts\build_prompt7_natural_manifests.py" `
  --dataset-root "<NEW_DATASET_ROOT>" `
  --split-map "<FROZEN_SPLIT_MAP.csv>" `
  --excluded-rois-csv "<PROMPT6_ROOT>\INPUTS\DATA_PRESELECTION\SELECTION\ROI_CENSUS.csv" `
  --colab-image-root "/content/drive/MyDrive/Research/PUMA/PROMPT7_NATURAL_DATA/01_training_dataset_tif_ROIs" `
  --out-dir "<PROMPT7_ROOT>\INPUTS\NATURAL_MANIFESTS"
```

The builder validates GeoJSON/TIFF pairing, class names, centroids, the Exploration-6 exclusion roster, exact split roles, and pairwise strongest-group disjointness. It writes the three required manifest CSVs and a provenance/audit JSON. It does not prove historical untouchedness by itself, so retain the provider correspondence and preregistration alongside the audit.

Do not change `run_mode` to `confirmatory` until the data provenance is resolved. The bundled internal split of already exposed public PUMA data is suitable only for a pipeline smoke test or explicitly labeled exploratory analysis.
