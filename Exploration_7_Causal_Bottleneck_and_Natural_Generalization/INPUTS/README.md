# Exploration-7 inputs
The archive includes the exploratory input subset described below and the retained seed-17 Exploration-6 head. Seed-29/43 head checkpoints were pruned; restore or regenerate them to replay all three seeds. The notebook points to external UNI2-h weights and historical diagnostic artifacts. It does not bundle the foundation encoder weights.

For **confirmatory** Exploration-7 results, supply a NEW naturally distributed cohort split into group-disjoint `train`, `calibration`, and `test` (prefer patient > case > slide > ROI). Exploration-6 locked holdouts may be replayed only under `historical_diagnostic` mode and must not be used to tune priors, thresholds, losses, or model selection.

Minimum manifest columns: `uid,roi,group,image,x,y,label,split`. Optional stronger identifiers: `patient,case,slide,stain_batch`. Stage-1 sensitivity additionally accepts `gt_x,gt_y`.

`EXPLORATORY_PUBLIC_PUMA_READY/` is a portable technical-test cohort: 49 ROI TIFFs and three ROI-disjoint manifests containing 26,518 nuclei. Its selection used labels to guarantee rare-class and Exp3 code-path coverage, and every ROI was historically exposed during Exploration 6. It must never be described as prospective or confirmatory.
