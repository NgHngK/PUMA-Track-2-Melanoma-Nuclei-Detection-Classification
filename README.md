# PUMA Stage 2 Nuclei Classification

## About this project

This project studies ten-class nuclei classification on the PUMA pathology dataset. The goal is to classify nuclei at fixed coordinates while keeping the existing Stage 1 detector unchanged.

The final retained architecture uses a **frozen UNI2-h encoder** with a single 96-pixel RGB crop around each nucleus. The 1536-dimensional UNI2-h CLS representation is combined with 16 RGB/point features through a small rank-eight multiplicative interaction head.

The ten classes are:

```text
tumor
lymphocyte
plasma_cell
histiocyte
melanophage
neutrophil
stroma
epithelium
endothelium
apoptosis
```

## Data

The local dataset contains **205 ROI images** and **97,193 annotated nuclei**. The class distribution is highly imbalanced: tumor and lymphocyte make up most of the nuclei, while classes such as neutrophil and plasma cell are much rarer.

| Class | Nuclei |
|---|---:|
| tumor | 57,234 |
| lymphocyte | 21,643 |
| histiocyte | 7,168 |
| stroma | 3,856 |
| epithelium | 2,211 |
| apoptosis | 1,804 |
| endothelium | 1,696 |
| melanophage | 695 |
| plasma_cell | 520 |
| neutrophil | 366 |

This imbalance is important because a model can obtain good overall performance while still failing on rare classes. For this reason, the project focuses on **Macro-F1, balanced accuracy, per-class recall, and class-level error analysis**, rather than accuracy alone.

## Final architecture

For each nucleus, we extract one **96 × 96 source-pixel RGB crop**, keep the target centered, white-pad image boundaries when necessary, and resize the crop to 224 × 224 before passing it through UNI2-h.

The encoder is frozen. Its CLS output is normalized and combined with 16 standardized RGB/point features.

```text
96 px RGB crop
      ↓
Frozen UNI2-h
      ↓
CLS representation (1536)
      ↓
Non-affine LayerNorm
      ├─────────────── Linear appearance logits
      │
      └── rank-8 interaction with 16 Tier-A features
                          ↓
                    10-class logits
```

The final logits are:

```text
z = W h + a + R[(P_h h) ⊙ (P_b b)]
```

where `h` is the normalized UNI2-h CLS representation and `b` is the 16-dimensional Tier-A feature vector.

The final head contains **27,866 trainable parameters**. UNI2-h remains fully frozen. The retained model does not use LoRA, BioMask, tissue branches, multiscale input, or an additional learned gate.

## Experimental process

The project was developed through four stages of experiments. Early work first verified the dataset, encoder identity, class mapping, sampling strategy, and evaluator behavior. Later experiments compared different losses, crop sizes, RGB/biology features, interaction heads, local patch-token representations, mask pooling, and restricted LoRA.

A major architecture search in Exploration 3 tested **26 predefined real and control configurations** using three seeds and three ROI-grouped folds. This produced **234 cross-validation fits**, followed by three confirmation runs for the selected model.

The resulting A5 architecture passed the predefined cross-validation and confirmation checks and became the candidate for the next full-scale experiment.

## Key results

### Frozen UNI2 baseline and loss experiments

Early frozen-UNI2 experiments showed that UNI2 representations contain useful information for nuclei classification, but rare-class performance remained difficult.

Across three seeds, the best-checkpoint Macro-F1 results were:

| Training setup | Mean Macro-F1 ± SD |
|---|---:|
| Cross-entropy | 0.4262 ± 0.0243 |
| Logit-adjusted CE | **0.4535 ± 0.0056** |
| Balanced sampling + CE | 0.4162 ± 0.0234 |

Logit adjustment produced the highest mean development Macro-F1 in this small experiment. However, several rare classes still had unstable recall, showing that a better average score does not automatically solve the long-tail problem.

### Architecture search

Exploration 3 moved from a simple linear Tier-A correction to a rank-eight interaction between image appearance and the 16 Tier-A features.

The selected **A5** model uses:

```text
Frozen UNI2-h CLS
+
16 Tier-A RGB/point features
+
rank-8 multiplicative interaction
```

This architecture passed the predefined promotion tests and the final epoch comparison against the previous A1 model. It was therefore selected as the single candidate for further full-scale training.

### Exploration 4

Exploration 4 tested whether the UNI2 representation could be improved by using spatial patch tokens instead of only the CLS token.

A 3 × 3 target-neighborhood representation improved ROI Macro-F1 during internal cross-validation, so it was taken to confirmation. On the fixed 300/150 train-validation confirmation setup, its mean ROI-F1 improvement over A5 was approximately:

```text
+0.0160
```

However, the improvement came with large recall losses for several classes. Mean recall changed by about:

```text
tumor       -0.137
neutrophil  -0.111
epithelium  -0.333
```

These declines exceeded the predefined class-recall guardrail. The neighborhood model was therefore rejected even though its average ROI score improved.

This is an important result: **a higher average metric is not enough when the improvement damages specific classes, especially rare or clinically relevant classes.**

## Final decision

The final retained architecture is **Exploration 3 A5**:

```text
Frozen UNI2-h
FOV = 96 source pixels
CLS representation = 1536 dimensions
16 standardized Tier-A RGB/point features
rank-8 multiplicative interaction
27,866 trainable head parameters
```

Exploration 4 did not provide enough evidence to replace it. Neighborhood pooling improved internal average performance but failed the class-recall confirmation rule. Mask-based pooling, larger representation variants, and restricted LoRA also did not provide enough consistent evidence for promotion.

## What we learned

The experiments show that pretrained pathology features are useful for nuclei classification, but the main challenge is not simply fitting the dominant classes. The dataset is strongly imbalanced, and rare-class recall can remain unstable even when overall Macro-F1 improves.

The results also show why controlled ablations are important. More complex representations do not automatically produce a better final model. Several additions improved one metric or one internal split but failed when checked against class-level behavior or an independent confirmation rule.

The final architecture remains relatively small because every additional component had to show a clear and reproducible benefit before being kept.

## Contribution

This project provides a reproducible experimental framework for Stage 2 nuclei classification with a pathology foundation model. It includes:

- strict UNI2-h checkpoint verification
- fixed nucleus-centered image preprocessing
- ten-class ontology protection
- long-tail sampling and loss experiments
- ROI-grouped cross-validation
- multi-seed architecture comparison
- per-class recall and Macro-F1 analysis
- exact local evaluator parity tests
- BioMask and representation controls
- restricted LoRA experiments
- final confirmation guardrails
- a compact retained architecture for future full-scale training

The main contribution is not a claim of state-of-the-art or clinical performance. It is a carefully controlled architecture-selection process that avoids promoting a model only because one average metric improved.

## Training setup

The main frozen-head experiments use:

```text
Optimizer: AdamW
Learning rate: 1e-3
Weight decay: 0.01
Batch size: 64
Gradient clipping: 1
Epochs: 10
Seeds: 17, 29, 43
Loss: Cross-entropy
Sampling: inverse-class replacement sampling
Precision: FP32
```

Tier-A normalization statistics are fitted using training data only.

## Evaluation

The main development metric is fixed ten-class **Macro-F1**. The project also tracks balanced accuracy, per-class precision, recall and F1, confusion matrices, NLL, calibration error, and ROI-level PUMA metrics.

Evaluation is deliberately class-aware. A model is not promoted simply because its mean score improves if the gain causes unacceptable recall loss in individual classes.

## Current status

The selected A5 architecture is the **single retained architecture for the next full-scale experiment**.

The current experiments do not establish external validation, clinical readiness, or state-of-the-art performance. Full-scale patient-level evaluation, calibration, deployable Stage-1-coordinate testing, and external benchmarking remain future work.
