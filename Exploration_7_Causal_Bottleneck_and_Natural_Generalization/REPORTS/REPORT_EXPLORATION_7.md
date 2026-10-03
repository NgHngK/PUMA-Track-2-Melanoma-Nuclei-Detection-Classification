# Exploration 7: causal bottleneck and natural generalization: exploratory performance report

## Executive conclusion

I designed Exploration 7 to break the broad failure in Exploration 6 into controlled tests of possible bottlenecks. The completed run is exploratory. I used a deterministic, label-informed, ROI-disjoint subset of public PUMA data that had already been seen during Exploration 6, and the package records `confirmatory_claims_allowed=false`. I therefore use this study to decide what to test next, rather than to claim confirmatory performance or deployability.

The results led me to consider three limitations together. Each showed some scope for improvement, and they may interact:

1. **The inverse-sampled decision boundary is mismatched to natural prevalence.** A fixed prior correction raised seed-17 ROI fixed-10 Macro-F1 from **0.273766 to 0.299341** and reduced the designated tail false-positive burden from **1,269 to 264** (−79.2%). However, it reduced recall by more than 0.10 for six supported classes, with a worst loss of **−0.250**, so the preregistered promotion gate failed.
2. **Classifier-only natural-risk relearning also helps some aggregate metrics but does not solve the class trade-off.** Natural cRT raised ROI F1 to **0.291752**, semantic Macro-F1 from **0.424361 to 0.488562**, and accuracy from **0.576283 to 0.714382**, while reducing tail false positives to **344**. Its paired ROI interval crossed zero, **[−0.010249, +0.044299]**, and its worst supported-class recall loss was **−0.411765**, so this intervention also failed its gate.
3. **Independent ROI diversity and wider context both carry signal that nucleus count or local appearance alone does not capture.** At a matched 400 nuclei, the high-diversity arm exceeded the low-diversity arm in every seed, with mean ROI F1 **0.213311 versus 0.183022**. Separately, adding FOV192 context to the grouped linear probe improved all nine seed-fold Macro-F1 comparisons, by a mean **+0.055881** over local FOV96 and **+0.048861** over a shuffled-context capacity placebo.

At this point, I would keep the frozen Gaussian representation as the anchor and build a confirmatory study around natural-prior boundary estimation, genuinely independent biological diversity and one targeted source of contextual information. I did not find a reason to resume a broad head search, a generic optimizer sweep or immediate encoder adaptation. Two other questions remained open: human label review had no completed entries, and Stage-1 sensitivity could not be evaluated without certified proposal artifacts.

I promoted no Exploration-7 model. After the seed-17 gates failed, I followed the stopping rule and skipped seeds 29 and 43 for Exp1 and Exp2. I constructed no ensemble and adopted no post-result threshold. These results cannot establish patient-level generalization, prospective external validity, end-to-end detection performance, clinical readiness or state of the art.

## 1. Scope, question, and relationship to Exploration 6

Exploration 6 selected a frozen Stage-2 classifier consisting of UNI2-h, nucleus-centred Gaussian token pooling with sigma 1.5, FOV96, 16 Tier-A features, and the rank-8 appearance × Tier-A interaction head. Gaussian pooling produced a replicated mean paired ROI-F1 gain of **+0.028923** over CLS on the D900 development design. The frozen seed-17 checkpoint scored **0.183281** on the one-shot confirmation set but only **0.157786** on the locked natural population. The natural result showed severe rare-class overprediction and only **51.0%** tumor recall despite **98.9%** tumor precision.

I therefore paused the broad architecture search and asked which of the following could explain the poor natural generalization:

- source-versus-target prior mismatch;
- a classifier boundary or objective mismatch;
- too few independent biological groups despite many nuclei;
- annotation or ontology ambiguity;
- missing information outside the nucleus-local field;
- an additional Stage-1 localization ceiling; or
- a genuinely inadequate frozen representation after the other explanations were tested.

I held the encoder, Gaussian pooling rule, FOV96 local representation, Tier-A definition and Exploration-6 interaction transforms fixed. FOV192 was reserved for the preregistered information probe. I excluded encoder adaptation and LoRA from this exploration so that the other explanations could be tested first.

## 2. Evidentiary status and data integrity

### 2.1 Exploratory-only cohort

I ran the study in `exploratory_public_puma` mode. The cohort audit identifies the source as public PUMA ROIs already used or opened during Exploration 6. Its deterministic, label-informed compact ROI selection was designed to cover the code paths. I kept three limitations explicit when interpreting this cohort:

- the run is not a new prospective or external evaluation;
- label-informed ROI selection can alter class prevalence and difficulty;
- historical exposure prevents treating the test split as a pristine holdout for the overall research program.

Within this exploratory cohort, the manifests were ROI-disjoint:

| Split | Nuclei | ROIs | Recorded groups | Class coverage |
|---|---:|---:|---:|---:|
| Train | 14,762 | 27 | 27 | 10/10 |
| Calibration | 5,811 | 11 | 11 | 10/10 |
| Test | 5,945 | 11 | 11 | 10/10 |

The `group` field equals the ROI identifier. Patient, case, and slide identifiers are not present in these manifests. Consequently, “group-disjoint” here means **ROI-disjoint**, not patient-disjoint or case-disjoint. The effective number of class-positive groups is also much smaller than the nucleus count suggests. For example, the training set contains 264 plasma cells across 11 ROIs but a Kish effective group count of only **2.48**; 294 neutrophils across 8 ROIs yield an effective group count of **2.84**; and 225 melanophages across 15 ROIs yield **3.61**. Calibration is still thinner for several tail classes, including only 11 neutrophils from 2 ROIs.

This exploratory test set includes all ten classes and was deliberately assembled to cover the experiments. It differs from the Exploration-6 locked natural population, so I do not interpret its baseline ROI F1 of 0.273766 as an improvement over Exploration 6's natural 0.157786. The evaluation populations changed.

### 2.2 Protocol timing and frozen artifacts

Machine-readable protocols for H7.1, H7.2, H7.3, and H7.5 were frozen on  before the saved results dated . The three Exploration-6 seed checkpoints were staged separately. Cached arrays record row-identity hashes, manifest hashes, array hashes, the UNI2 weights hash, shape, FOV, and sigma. The principal cache shapes are:

- train FOV96 Gaussian: 14,762 × 1,536;
- calibration FOV96 Gaussian: 5,811 × 1,536;
- test FOV96 Gaussian: 5,945 × 1,536;
- train FOV192 Gaussian: 14,762 × 1,536;
- Tier-A: 16 features per nucleus for all three splits.

These records let me trace the local and context probes to specific cached arrays and establish that the exploratory protocol preceded its recorded outcomes. They improve execution provenance, but the cohort remains historically exposed and label-informed. I do not treat that provenance as a substitute for confirmatory data.

### 2.3 Metric contract and a small reconstruction discrepancy

The primary metric is the vendored PUMA V17 class-first greedy ROI fixed-10 Macro-F1. Semantic fixed-10 Macro-F1, supported-class Macro-F1, balanced accuracy, accuracy, NLL, ECE15, per-class precision/recall/F1, AUPRC, predicted counts, and false positives are diagnostics.

Exploration 7 reconstructed the historical Exploration-6 natural result from saved prediction probabilities and the historical manifest. Its semantic metrics exactly reproduce Exploration 6, including semantic fixed-10 Macro-F1 **0.209392**, accuracy **0.531180**, NLL **1.545238**, and ECE15 **0.141573**. Its reconstructed ROI F1 is **0.157845**, whereas the authoritative Exploration-6 result is **0.157786**, a difference of approximately **+0.000060**. Several per-ROI values differ only at very small magnitude. This does not affect any scientific conclusion, but the two values should not be silently conflated. Exploration 6’s frozen result remains authoritative; Exploration 7’s value is a diagnostic replay.

## 3. Historical Exploration-6 natural diagnosis

The diagnostic replay reinforces the failure pattern that motivated Exploration 7. On 10,712 nuclei from 24 ROIs, top-1 semantic accuracy was **0.531180**, while top-2 accuracy was **0.725915** and mean true-class rank was **2.1135**. Thus, the correct class was often represented near the top even when the final boundary chose incorrectly. However, the mean top-1 margin was **0.49825**, so many errors were not merely near-ties; the inverse-sampled model was often confidently wrong.

Group accuracy ranged from **0.1040 to 0.8213** across the 24 ROIs. This extreme heterogeneity is consistent with substantial ROI-specific shift, but ROI identity may combine biological composition, stain, acquisition, annotation, and other nuisance factors. Without patient/case/slide identifiers, it cannot be assigned a purely biological cause.

The largest directed errors were overwhelmingly tumor nuclei mapped into rare or intermediate classes: tumor→histiocyte (1,137), tumor→apoptosis (780), tumor→lymphocyte (644), tumor→epithelium (482), tumor→stroma (438), tumor→endothelium (366), tumor→plasma cell (284), and tumor→melanophage (168). These errors explain how the model could retain moderate tail recall while suffering severe tail precision collapse and reduced tumor recall.

I did not interpret the high numeric `class_id` correlation in the saved error table as a meaningful continuous association. Class IDs are arbitrary categorical codes, so their integer order carries no such meaning. I instead examined which classes were confused and how their predicted probabilities behaved.

## 4. Exp1: natural-prior/logit correction

### 4.1 Design

Exp1 added a fixed class offset to the frozen Exploration-6 logits:

`corrected logit = original logit + log(target prior) − log(source prior)`

The source prior was set to a declared uniform approximation, consistent with idealized inverse-frequency replacement sampling when measured exposure is unavailable. The target prior was estimated only from the 5,811-nucleus calibration split with Dirichlet alpha 0.5 smoothing. No weights or representations were retrained. Seed 17 was the primary checkpoint; seeds 29 and 43 were conditional replications only after a full seed-17 pass.

I used this design to test whether an intercept or prior mismatch explained a substantial part of the natural-population failure. One limitation remained: finite sampling and discriminative training may leave the effective source decision prior different from the declared uniform approximation.

### 4.2 Aggregate results

| Metric | Frozen baseline | Prior corrected | Change |
|---|---:|---:|---:|
| ROI fixed-10 Macro-F1 | 0.273766 | 0.299341 | +0.025575 |
| Semantic fixed-10 Macro-F1 | 0.424361 | 0.502606 | +0.078245 |
| Balanced accuracy | 0.580588 | 0.490605 | −0.089983 |
| Accuracy | 0.576283 | 0.683263 | +0.106981 |
| NLL | 1.361503 | 1.017752 | −0.343751 |
| ECE15 | 0.125678 | 0.110419 | −0.015259 |
| Tail false positives | 1,269 | 264 | −1,005 (−79.2%) |

The paired test-ROI mean delta was **+0.025575**, with bootstrap interval **[+0.005511, +0.045714]**; 9 of 11 ROIs had a positive delta. On the primary metric and tail false-positive criterion alone, this looks successful.

### 4.3 Class trade-offs and gate decision

| Class | Support | Baseline predicted | Corrected predicted | Baseline recall | Corrected recall | Recall change | Baseline F1 | Corrected F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Tumor | 2,505 | 1,718 | 2,521 | 0.5832 | 0.7796 | +0.1964 | 0.6919 | 0.7772 |
| Lymphocyte | 1,874 | 1,492 | 2,021 | 0.6462 | 0.7855 | +0.1393 | 0.7195 | 0.7558 |
| Plasma cell | 40 | 328 | 43 | 0.7250 | 0.4750 | −0.2500 | 0.1576 | 0.4578 |
| Histiocyte | 702 | 853 | 716 | 0.3946 | 0.4074 | +0.0128 | 0.3563 | 0.4034 |
| Melanophage | 60 | 112 | 39 | 0.6500 | 0.4167 | −0.2333 | 0.4535 | 0.5051 |
| Neutrophil | 17 | 65 | 22 | 0.5882 | 0.3529 | −0.2353 | 0.2439 | 0.3077 |
| Stroma | 223 | 245 | 154 | 0.4888 | 0.3677 | −0.1211 | 0.4658 | 0.4350 |
| Epithelium | 185 | 190 | 93 | 0.6162 | 0.4378 | −0.1784 | 0.6080 | 0.5827 |
| Endothelium | 142 | 510 | 220 | 0.7887 | 0.6549 | −0.1338 | 0.3436 | 0.5138 |
| Apoptosis | 197 | 432 | 116 | 0.3249 | 0.2284 | −0.0964 | 0.2035 | 0.2875 |

The correction recovered much of the tumor and lymphocyte mass and improved tail precision, but it also suppressed several tail decisions. Recall losses exceeded the −0.10 guard for plasma cell, melanophage, neutrophil, stroma, epithelium and endothelium. Balanced accuracy fell by about 0.09 while overall accuracy rose by 0.107. I read this as an improvement in population-weighted behavior with a substantial class trade-off, not as a uniformly safer ten-class classifier.

I stopped this intervention at the preregistered gate. Its ROI interval was above zero and tail false positives fell, but the recall guard failed. That meant I did not proceed to replication on seeds 29 and 43.

### 4.4 Calibration and boundary diagnostics

A fitted scalar temperature of **1.5464** preserved argmax predictions as expected. It reduced NLL from 1.361503 to **1.267464** and ECE15 from 0.125678 to **0.039633**, showing that confidence calibration is independently improvable. Semantic classification metrics were unchanged. The reconstructed ROI metric changed by approximately −0.000048 despite argmax invariance, which is small enough to treat as evaluator/numerical reconstruction noise rather than a substantive effect.

The exploratory free class-bias vector produced semantic Macro-F1 **0.510854** and accuracy **0.691337**, slightly above principled prior correction on those semantic metrics, but its ROI F1 was lower at **0.288313**. It therefore does not strongly establish that a fully free intercept solution dominates prior correction. Calibration-only one-vs-rest thresholds varied widely (from 0.039 for tumor to 0.943 for neutrophil), confirming heterogeneous class score scales, but these oracle thresholds are not prospective test results and must not be adopted from this exploratory analysis.

### 4.5 Interpretation

I concluded that H7.1 had mechanistic support without meeting the requirements for promotion. Prior mismatch contributed substantially to rare-class overprediction, yet this single global correction could not restore natural head prevalence while preserving tail recall. It shifted the operating point without creating new class separation. The thin calibration support and concentration in a few ROIs for several rare classes likely made the estimates less stable as well.

## 5. Exp2: natural-risk classifier relearning

### 5.1 Design and convergence

Exp2 froze UNI2, Gaussian features, FOV96, Tier-A, and the Exploration-6 `ph`/`pb` transforms, then reset and retrained only the classifier output terms with natural sampling and ordinary cross-entropy. The recipe was fixed at 10 epochs, learning rate 0.001, weight decay 0.01, and batch size 64. Training loss declined monotonically from approximately **0.710** to **0.386**, with no recorded numerical instability.

### 5.2 Aggregate results

| Metric | Frozen baseline | Natural cRT | Change |
|---|---:|---:|---:|
| ROI fixed-10 Macro-F1 | 0.273766 | 0.291752 | +0.017987 |
| Semantic fixed-10 Macro-F1 | 0.424361 | 0.488562 | +0.064201 |
| Balanced accuracy | 0.580588 | 0.509504 | −0.071084 |
| Accuracy | 0.576283 | 0.714382 | +0.138099 |
| NLL | 1.361503 | 0.987116 | −0.374387 |
| ECE15 | 0.125678 | 0.082763 | −0.042915 |
| Tail false positives | 1,269 | 344 | −925 (−72.9%) |

The paired ROI delta was **+0.017987**, but the bootstrap interval **[−0.010249, +0.044299]** crossed zero, and only 7 of 11 ROIs improved.

The relearned boundary shifted predictions strongly toward common classes. Tumor recall rose from **0.5832 to 0.8926**, lymphocyte recall rose from **0.6462 to 0.7487**, and overall accuracy improved materially. At the same time, plasma-cell recall fell by **0.3500**, histiocyte by **0.1838**, melanophage by **0.1500**, neutrophil by **0.4118**, and endothelium by **0.3099**. Stroma and epithelium recall improved. The worst supported-class recall loss, −0.411765 for neutrophil, was far beyond the guard.

I also stopped Exp2: it failed both the statistical-evidence requirement and the recall-safety guard, so I skipped seeds 29 and 43. Boundary-only retraining improved accuracy, NLL, ECE, semantic F1 and tail false-positive burden while leaving the encoder frozen. That tells me the representation contains useful natural-distribution information. It does not make this fixed cRT recipe a balanced ten-class solution.

## 6. Exp3: independent ROI diversity versus nucleus density

### 6.1 Design

The diversity contrast held the sample count at 40 nuclei per class, 400 total, while selecting 2 source ROIs per class for the low-diversity arm and 8 per class for the high-diversity arm. Because ROIs contain multiple classes, the realized number of unique source ROIs was 14 versus 27. Every class met the intended per-class group count.

The density contrast used the same 8 source ROIs per class in both arms and changed the count from 20 to 40 nuclei per class, 200 versus 400 total. The realized number of unique source ROIs was 26 in both arms. Both contrasts were trained with the same frozen-representation classifier protocol for seeds 17, 29, and 43 and evaluated on the same 11 exploratory test ROIs.

### 6.2 Diversity results at matched N

| Arm | Seed | N | Unique ROIs | ROI F1 | Semantic F1 | Accuracy | NLL | ECE15 | Tail FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Low diversity | 17 | 400 | 14 | 0.185136 | 0.288634 | 0.333726 | 2.471127 | 0.305501 | 1,805 |
| Low diversity | 29 | 400 | 14 | 0.191689 | 0.286636 | 0.349706 | 2.354777 | 0.269966 | 1,962 |
| Low diversity | 43 | 400 | 14 | 0.172242 | 0.265058 | 0.305635 | 2.571720 | 0.316424 | 2,102 |
| High diversity | 17 | 400 | 27 | 0.222187 | 0.345863 | 0.460892 | 1.743998 | 0.139533 | 1,667 |
| High diversity | 29 | 400 | 27 | 0.215944 | 0.320128 | 0.425736 | 1.849226 | 0.166978 | 2,004 |
| High diversity | 43 | 400 | 27 | 0.201800 | 0.319062 | 0.398823 | 1.940597 | 0.184254 | 2,117 |

Across seeds, high diversity improved ROI F1 by **+0.037051, +0.024254, and +0.029558**. The mean rose from **0.183022 to 0.213311**, a mean gain of **+0.030288**. Mean semantic F1 rose by **+0.048242**, mean accuracy by **+0.098795**, mean NLL improved by approximately **−0.621**, and mean ECE15 improved by approximately **−0.134**.

I checked the false-positive counts alongside the broad metrics. Tail false positives fell by 138 in seed 17, but rose by 42 and 15 in seeds 29 and 43. The diversity benefit was consistent on the broad performance measures, while its effect on class balance was mixed.

### 6.3 Density results at matched ROI count

| Arm | Seed | N | Unique ROIs | ROI F1 | Semantic F1 | Accuracy | NLL | ECE15 | Tail FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20/class | 17 | 200 | 26 | 0.238278 | 0.370066 | 0.459378 | 1.631938 | 0.135746 | 1,768 |
| 20/class | 29 | 200 | 26 | 0.239588 | 0.366214 | 0.433137 | 1.682945 | 0.155264 | 1,995 |
| 20/class | 43 | 200 | 26 | 0.245638 | 0.370510 | 0.464592 | 1.599972 | 0.122334 | 1,683 |
| 40/class | 17 | 400 | 26 | 0.262513 | 0.404503 | 0.545164 | 1.415042 | 0.108022 | 1,207 |
| 40/class | 29 | 400 | 26 | 0.256951 | 0.388741 | 0.504626 | 1.517551 | 0.125866 | 1,641 |
| 40/class | 43 | 400 | 26 | 0.249902 | 0.385348 | 0.510177 | 1.498973 | 0.110683 | 1,401 |

Doubling nuclei at fixed source-ROI coverage also helped, but less on the primary metric. Mean ROI F1 rose from **0.241168 to 0.256455**, a gain of **+0.015287**; the seed-specific gains were approximately +0.0242, +0.0174, and +0.0043. Mean semantic F1 rose by **+0.023934**, mean accuracy by about **+0.0676**, and mean tail false positives fell by **399**.

### 6.4 Causal interpretation and limitations

The matched-N comparison supported H7.3: in this design, broader ROI coverage gave about twice the mean ROI F1 gain obtained by increasing nucleus density. I still need to keep the causal claim narrow. I manipulated ROI rosters, not patient or case diversity; the low- and high-diversity arms necessarily used different ROIs; and the exploratory split and subset construction were reused across seeds. Those seeds check optimization and initialization robustness, not independent biological replication. I would need a prospective patient-grouped repeat with repeated matched subset draws to estimate a more general diversity effect.

## 7. Exp4: blinded label/confusion adjudication

I generated a blinded review package with **204** unique items and 204 corresponding image crops. The review CSV contains image, coordinates, ROI and UID; a separate key holds the original label, model prediction, confidence and margin. The sampling covered four predefined biological pairs as well as high-confidence-wrong and low-margin-wrong cases.

The `reviewed_label`, `uncertain`, and `notes` fields are all empty. I therefore have no completed reviewer file, inter-reviewer comparison, original-label confirmation rate or model-versus-review agreement result to analyze. H7.4 remains unevaluated. The image set is ready for blinded pathology review, but model predictions cannot be used to fill in or revise those labels automatically.

## 8. Exp5: context information probe

### 8.1 Design

Exp5 compared the same grouped linear probe on:

- local Gaussian FOV96 features;
- concatenated local FOV96 plus wider FOV192 Gaussian features; and
- local features plus shuffled context, a capacity-matched placebo.

I used three predefined probe seeds with three grouped folds per seed to ask whether the context features contained additional information. I did not treat this probe as a production multiscale architecture or use it to promote a model on the test set.

### 8.2 Global grouped-probe results

| Seed | Fold | Local F1 | Local + context F1 | Shuffled-context placebo F1 | Gain vs local | Gain vs placebo |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 0 | 0.449304 | 0.510637 | 0.451904 | +0.061332 | +0.058733 |
| 17 | 1 | 0.404735 | 0.476516 | 0.430598 | +0.071781 | +0.045919 |
| 17 | 2 | 0.406435 | 0.459215 | 0.410113 | +0.052781 | +0.049102 |
| 29 | 0 | 0.393099 | 0.427349 | 0.394810 | +0.034250 | +0.032539 |
| 29 | 1 | 0.457347 | 0.486969 | 0.454641 | +0.029621 | +0.032328 |
| 29 | 2 | 0.304791 | 0.337559 | 0.316870 | +0.032769 | +0.020689 |
| 43 | 0 | 0.460905 | 0.547566 | 0.469553 | +0.086661 | +0.078013 |
| 43 | 1 | 0.373160 | 0.433924 | 0.361831 | +0.060764 | +0.072094 |
| 43 | 2 | 0.422427 | 0.495392 | 0.445056 | +0.072965 | +0.050336 |

All nine local-plus-context comparisons were positive. Mean improvement was **+0.055881** versus local alone. The seed-mean gains were **+0.061965, +0.032213, and +0.073464**, all positive. Mean improvement over shuffled context was **+0.048861**, strongly arguing that the result is not explained only by doubling feature dimension.

### 8.3 Targeted pairwise AUPRC

Mean AUPRC across folds improved for the two most informative available pairs:

- tumor versus lymphocyte: approximately 0.925→0.965, 0.935→0.967, and 0.917→0.952 across seeds;
- tumor versus histiocyte: approximately 0.800→0.899, 0.801→0.889, and 0.748→0.887.

Epithelium versus tumor was already near ceiling and improved only slightly in mean AUPRC: about 0.972→0.982, 0.978→0.986, and 0.980→0.982. Neutrophil versus apoptosis was unavailable for seeds 17 and 29 because at least one fold contained only one of the two classes. It was available for seed 43, where mean AUPRC was essentially unchanged, 0.894946 local versus 0.894248 with context.

I found strong exploratory support for H7.5 in grouped Macro-F1 and in tumor/lymphocyte and tumor/histiocyte separation, including the comparison with shuffled context. I could not declare the complete pairwise success criterion satisfied: the neutrophil/apoptosis comparison was mostly unidentifiable and did not improve in the only seed where it was available. My proposed follow-up was one targeted, preregistered context architecture on new grouped data, rather than another broad multiscale search.

## 9. Exp6: Stage-1 sensitivity

I could not run Exp6 because its required certified Stage-1 manifest and matched predictions were absent. That test needed ground truth, proposal centres and the exact V17 evaluator contract. With no Stage-1 sensitivity result, Exploration 7 remains a GT-centred, conditional Stage-2 classification study. I cannot quantify the extra end-to-end loss from missed, duplicated or displaced proposals.

## 10. Cross-experiment causal synthesis

Taken together, these experiments gave me reasons to investigate several causes of the error rather than attribute it to a single component.

### 10.1 What is supported

- **Boundary/prior mismatch is real.** Both fixed prior correction and natural cRT sharply reduce rare-class false positives and improve population-weighted accuracy, NLL, ECE, and semantic F1 without changing the frozen encoder.
- **Boundary repair alone is insufficient.** Both interventions sacrifice recall for several supported tail classes, and cRT lacks a positive paired ROI interval. The existing representation contains useful signal, but its class overlap does not permit a cost-free global boundary shift under these recipes.
- **ROI diversity matters independently of nucleus count.** At the same 400-nucleus budget, broader source-ROI coverage improves every seed and multiple metrics. More nuclei from the same ROI roster also help, but the primary-metric gain is smaller.
- **Wider context contains nonredundant information.** The context probe improves every grouped fold and defeats a shuffled-context capacity control, especially for tumor/histiocyte and tumor/lymphocyte discrimination.
- **The remaining error is heterogeneous.** Historical Exploration-6 accuracy varies dramatically by ROI, and many tumor errors are high-margin. This is consistent with interacting prevalence, group shift, missing context, and possible labeling effects rather than one calibration defect.

### 10.2 What is not established

- No intervention passed its full model-promotion gate.
- No Exploration-7 Exp1/2 effect was replicated across checkpoints 17, 29, and 43 because the preregistered stop rule correctly prevented those runs.
- The diversity experiment does not establish patient-level biological diversity because the only available group is ROI.
- The label/ontology hypothesis remains unanswered until blinded human reviews are completed.
- The context probe shows information availability, not the generalization of a deployable context model.
- The Stage-1 ceiling remains unknown.
- The exploratory public cohort cannot support prospective external, clinical, or SOTA claims.

## 11. Hypothesis disposition

| Hypothesis | Result | Status |
|---|---|---|
| H7.1 prior mismatch | ROI and tail-FP improvement, but broad recall harm | Exploratory support; promotion failed |
| H7.2 boundary/objective mismatch | Aggregate improvement, but CI crosses zero and severe recall harm | Exploratory partial support; promotion failed |
| H7.3 independent diversity | High-diversity wins in all seeds at matched N | Exploratory support at ROI level |
| H7.4 label/ontology ambiguity | Review package generated; no reviews completed | Unevaluated |
| H7.5 missing wider context | All global folds improve; placebo defeated; one pair unidentifiable | Strong exploratory support, incomplete pairwise gate |
| H7.6 Stage-1 localization ceiling | Required certified artifacts absent | Not executable |

My proposed next study would keep the frozen Gaussian anchor, obtain stronger independent data, and concentrate on the boundary and context mechanisms for which these probes provided clear information.

## 12. Recommended confirmatory sequence

1. Acquire genuinely new train, calibration, and untouched test cohorts with patient/case/slide identifiers and enforce disjointness at the strongest available level.
2. Ensure all ten classes have adequate calibration and test support across multiple independent groups; report effective group counts, not only nuclei.
3. Keep the Exploration-6 Gaussian/Tier-A/A5 representation frozen as the baseline anchor.
4. Predeclare one boundary study that compares unchanged logits, principled prior correction using measured sampler exposure, and one natural-risk classifier relearning recipe. Preserve the ROI interval, class-recall, and false-positive guards.
5. Repeat the diversity contrast with multiple independently drawn matched subsets or a hierarchical group-subsampling curve. The experimental unit should be patient or case when available.
6. Test exactly one targeted context model motivated by Exp5, with the shuffled-context or another capacity-matched control retained. Avoid a new broad multiscale search.
7. Complete the blinded pathology review before changing the ontology or constructing hierarchical labels.
8. Recover and certify complete Stage-1 proposals before making end-to-end claims.

## 13. Final retained decision

I kept the seed-17 frozen Gaussian/Tier-A/A5 checkpoint from Exploration 6 as the historical anchor. Exploration 7 was exploratory, and neither Exp1 nor Exp2 met the promotion rules. The results made another open-ended head or optimizer search a lower priority. My proposed follow-up was a controlled study of natural-prior boundaries, independent biological-group coverage and one targeted context pathway, with label review and Stage-1 evaluation handled separately.

## 14. Main evidence

The main Exploration-7 artifacts used for this report are:

- `RUN_MODE.json`;
- `INPUT_AUDIT_{TRAIN,CALIBRATION,TEST}.json`;
- `PROTOCOL_FREEZE_H7_{1,2,3,5}.json`;
- `result.json`;
- `result.json`;
- `result.json`;
- `EXP4_BLINDED`;
- `EXP5_CONTEXT_PROBE.json`;
- `HISTORICAL_EXPLORATION6_NATURAL`;
- `EXPLORATORY_PUBLIC_PUMA_AUDIT.json`;
- `PREREGISTERED_PROTOCOL.md`; and
- `DETAILED_PSEUDOCODE_AND_DEBATE.md`.

I leave the missing evidence visible: there is no completed Exp4 reviewer analysis, Exp6 result, prospective Exploration-7 cohort or patient-level grouping. Those gaps limit the conclusions that can be carried forward from this report.

## Architecture matrix
| Component | Exploration-7 default | Scientific role | Trainable in default experiments? |
|---|---|---|---|
| UNI2-h | 24-layer ViT-G/14-style pathology FM, 1536-D, 8 register tokens | appearance representation | frozen |
| Crop | nucleus-centred FOV96 -> 224x224 | target appearance | fixed |
| Pooling | 16x16 spatial tokens, Gaussian sigma1.5 | emphasize nucleus-centred local tokens | frozen |
| Tier-A | 16 H/stain/texture/ring/ROI-relative point features | complementary local handcrafted signal | fixed definition; normalized with Exploration 6 TRAIN stats for frozen checkpoint |
| A5 | appearance logits + rank-8 `R[(Ph h) * (Pb bio)]` | low-rank appearance×Tier-A interaction | frozen in Exp1; output W/R retrained only in Exp2/3 |
| Prior correction | classwise additive logit intercept | target/source prior mismatch | no model retraining |
| Context probe | FOV96 local + FOV192 frozen Gaussian features | asks if missing larger context contains information | probe only; not promoted automatically |

## Exploration 7 detailed pseudocode, adversarial review, and design debate

## Bottleneck diagnosis
I began with an apparent tension in Exploration 6: nucleus-centred Gaussian UNI2 tokens repeatedly improved the controlled development results, but absolute ROI F1 remained low and rare classes were heavily overpredicted on the natural holdout. Exploration 5 had also shown a large training-to-held gap. Before testing more encoder capacity, I wanted Exploration 7 to distinguish boundary mismatch, insufficient independent data, label or ontology ambiguity, and missing information.

## Competing designs considered and rejected
1. **Another head/HPO sweep**: rejected. High train fit plus failed larger heads/HPO means this has low information gain and high metric-shopping risk.
2. **Temperature scaling as the main fix**: rejected. A positive scalar temperature preserves argmax; it is retained only for NLL/ECE diagnosis.
3. **Tune class thresholds directly on Exploration-6 natural holdout**: rejected. That holdout is already opened and would become a development set. Oracle thresholds are allowed only as exploratory upper-bound diagnostics.
4. **Rerun LoRA / encoder adaptation**: rejected for Exploration 7. Previous adaptation did not earn promotion, while the strongest current evidence points first to data diversity and decision-boundary mismatch. UNI2-h therefore stays frozen throughout this exploration.
5. **Train a hierarchical network now**: rejected until confusion structure and human label ambiguity demonstrate a biological hierarchy.
6. **Use the large raw nucleus count as sample size**: rejected. The causal unit for generalization is the strongest independent biological group available, not the number of nuclei.

## Locked experiment order
### Exp 0: evaluation/representation audit (no training)
- Validate manifests, group IDs, class support, cache/checkpoint ontology.
- Record fixed 10 + support-aware metrics, per-class PR/recall/predicted counts, AUPRC, NLL/ECE, full confusion matrix, top-2 ranks/margins.
- Measure cross-group kNN purity and grouped linear-probe performance on frozen Gaussian embeddings.
- Output confusion-pair and group-error tables.

### Exp 1: prior/logit correction
Pseudocode:
1. Load frozen Exploration-6 predictions for a **new calibration** cohort and untouched new natural test.
2. Recover logits from saved logits or `log(probability)` (equivalent up to a sample-wise constant for class offsets).
3. Define source effective prior from measured sampler exposure if available. If unavailable, explicitly declare uniform as the approximation implied by inverse-frequency replacement sampling.
4. Estimate target prior **only on calibration** with fixed alpha=0.5 smoothing.
5. Compute `z' = z + log(pi_target) - log(pi_source)`.
6. Evaluate frozen baseline and corrected logits on test.
7. Pair exact ROI fixed 10 scores by ROI; cluster/bootstrap ROI deltas.
8. Check class recall harm and false-positive count ratios.
9. Separately fit temperature on calibration; verify test argmax is unchanged; report NLL/ECE only.
10. Fit a free 10-class intercept vector as an exploratory diagnostic. If it strongly beats principled prior correction, pure prior shift is incomplete.
11. Compute one-vs-rest oracle thresholds on calibration only as a representation/boundary upper bound; do not present them as prospective test performance.

### Exp 2: classifier boundary relearning
1. Freeze UNI2, Gaussian sigma1.5, FOV96, Tier-A definition, Exploration-6 ph/pb interaction transforms.
2. Reset only classifier output W and interaction output R (cRT-style).
3. Train natural sampling + ordinary CE, fixed 10 epochs, same lr/wd/batch as historical control.
4. Never use inverse replacement sampling.
5. Evaluate once on untouched natural test and pair ROI scores against the original checkpoint.
6. If this succeeds, representation is more useful than original natural F1 suggests; the boundary/objective was a substantial cause.
7. If it fails, raise posterior weight on representation.

### Exp 3: independent-diversity causal test
1. Restrict to TRAIN groups; keep external test fixed and group-disjoint.
2. For each class with enough support, create two matched-N subsets:
   - low diversity: 2 groups/class;
   - high diversity: 8 groups/class;
   - 40 rows/class where feasible.
3. Train the identical frozen-representation classifier protocol on each arm.
4. Compare external grouped test performance. Repeat with seeds 29/43 only if the first result supports a diversity effect.
5. A consistent high-diversity win at matched N falsifies the idea that adding more same-ROI nuclei is equivalent to adding biological groups.

### Exp 4: label/confusion adjudication
1. Rank directed confusion pairs and compute model confidence, true-class rank, and top-2 status.
2. Stratify samples into high-confidence wrong, low-margin wrong, and predeclared biological pairs.
3. Generate a **blinded** review CSV with image/coordinate only; keep true/prediction/probabilities in a separate key.
4. Human reviewers assign label + uncertainty without seeing model prediction.
5. Analyze agreement and original-label confirmation. Do not let model predictions rewrite labels automatically.
6. If human ambiguity is high, estimate an annotation ceiling before model redesign.

### Exp 5: context information probe
1. Extract frozen Gaussian features at FOV96 and exactly one larger context FOV192.
2. Use the same grouped linear probe, same folds, same regularization.
3. Compare local-only to `[local, context]` concatenation. This is an information probe, not a production architecture.
4. Focus pairwise AUPRC on tumor/epithelium, tumor/lymphocyte, tumor/histiocyte, neutrophil/apoptosis.
5. If context consistently helps across grouped folds, a targeted context architecture is justified. If not, stop multiscale expansion.

### Exp 6: Stage-1 sensitivity (conditional)
- If certified Stage-1 matched proposals include GT and proposal centres, quantify classification versus centre error bins and run the exact V17 evaluator. Stage 1 cannot explain the existing GT-centred Stage-2 weakness, but it may impose an additional end-to-end ceiling.

## Missing-information review
I could not establish patient-level independence from this package because patient/case/slide IDs were not guaranteed. I therefore used the strongest available grouping and described ROI-only conclusions at ROI level. I also could not create a new natural holdout by changing the split. Confirmatory Exp1/2 would require new data; I retained the existing Exploration-6 locked-natural set only for historical diagnostic replay.

## What would change the plan?
- Strong Exp1/2 gain -> deepen boundary/calibration; do not adapt encoder.
- Strong Exp3 diversity effect -> prioritize data acquisition over GPU search.
- High Exp4 disagreement -> repair/adjudicate ontology before model complexity.
- Strong Exp5 context effect -> test one targeted context representation.
- None of the above + adequate independent data + clean labels -> conclude that a future exploration may need a dedicated representation study; do not reopen encoder adaptation inside Exploration 7.

## Amendment: three-seed replication and runtime-efficiency pass

Exploration 6 retained three Gaussian runs (17/29/43), so a Exploration-7 conclusion based only on checkpoint 17 would leave avoidable stochastic uncertainty. The amended design keeps seed17 as the sole discovery anchor and reserves seed29/43 for replication. This is preferable to training all three before reading the primary gate because it preserves the preregistered discovery/replication asymmetry and avoids unnecessary repeated classifier training after a clear primary failure.

For Exp1 and Exp2:

```text
run seed17 with frozen settings
    -> evaluate preregistered promotion gate
        -> fail: stop seed replication; diagnose the negative result
        -> pass: run identical intervention on seed29 and seed43
                 -> require same effect direction and harm guards
                 -> aggregate per-ROI effect across seeds, then bootstrap ROIs
```

I treated seed differently for Exp3 because it is part of the predefined causal robustness design: every diversity/density arm runs with 17/29/43 and starts from the corresponding Exploration-6 Gaussian checkpoint. Exp5 also repeats its grouped probe split with 17/29/43, at low cost because the same FOV96/FOV192 cache can be reused.

In the runtime review, I ruled out changes to the scientific model, including confirmatory batch-size changes, an ensemble, a larger context by default or mixed-precision frozen feature extraction. I focused instead on repeated work that could be removed: shared encoder caches, ROI-grouped image I/O, one-time seed normalizers, device-resident cached classifier data, inference mode, bounded-vectorized bootstrap, exact O(N log N) threshold analysis and memory-bounded exact kNN. I also removed the previously nonproductive LoRA branch from this workflow.

## Exploration 7 preregistered protocol

The JSON files in `CONFIGS` are the machine-readable protocol. Before a confirmatory experiment, freeze its JSON with `freeze_protocol.py`. Any scientific change requires an explicit amended protocol rather than silently editing a completed freeze.

## Primary endpoint
For Exp1/2 the primary endpoint is exact PUMA V17 ROI fixed-10 Macro-F1 on a new untouched natural test cohort. Secondary endpoints are fixed-10/support-aware semantic Macro-F1, balanced accuracy, class precision/recall/F1, predicted-vs-true counts, false positives, AUPRC, NLL, ECE15 and confusion structure. Promotion requires a positive paired ROI bootstrap interval, no supported-class recall drop greater than 0.10 and reduced tail false-positive burden.

## Seed contract
Seed 17 is the primary discovery checkpoint. Exploration-6 Gaussian seed29 and seed43 checkpoints are preregistered replication anchors. Exp1 and Exp2 run seed17 first; 29/43 are executed with identical settings only if the primary gate passes, unless a run is explicitly labeled exploratory with `replication_policy=always`. Exp3 and Exp5 are predefined three-seed mechanistic studies. No seed-specific tuning and no confirmatory ensemble are allowed.

A replicated intervention is considered robust only when the effect direction is positive in all three seeds, the seed-specific harm guards pass, and the bootstrap interval of the per-ROI effect averaged over the three seeds remains above zero.

## Frozen representation
UNI2-h remains frozen throughout Exploration 7. Gaussian sigma1.5, FOV96, Tier-A and the A5 rank-8 interaction are fixed. FOV192 is used only in the predefined Exp5 information probe. Encoder adaptation/LoRA is out of scope for this exploration.

## Holdout protection
Exploration-6 confirmation and locked-natural holdouts are historical diagnostics only. They cannot determine Exploration-7 target priors, thresholds, model selection, stopping rules or new success criteria. New group-disjoint calibration and test data are required.

## Runtime optimizations
Caching, batching, vectorization, fused optimizer execution, ROI-grouped I/O and three-seed cache reuse are execution optimizations only. They may not change the scientific batch size, objective, representation, checkpoint selection or seed-specific settings.

## Data flow
`manifest -> (optional raw image + UNI2) -> Gaussian 1536-D feature -> Exploration 6 Tier-A normalizer -> A5 -> logits -> diagnostics/calibration/evaluation`.

Exp1 consumes only frozen logits and labels. Exp2/3 consume frozen Gaussian features + raw Tier-A transformed by the checkpoint normalizer. Exp5 consumes two frozen feature matrices aligned by UID. Exp4 consumes coordinates only for blinded review. All caches must remain UID-aligned; scripts merge by UID wherever prediction tables are used.

## Runtime acceleration path

On Colab, immutable source data may be staged into `PUMA_P7_RUNTIME`. The source manifests are never modified; execution copies add `source_image` and point `image` to the local copy. Cache identity uses `source_image` when present, so relocation does not change sample identity. One resident frozen UNI2-h model then extracts FOV96 features for train/calibration/test and, when requested, FOV192 train features. Those caches are shared unchanged by Exploration-6 seeds 17/29/43.

## Metrics and gates
- Primary: exact vendored PUMA V17 ROI fixed-10 macro-F1 when coordinates are available.
- Semantic fixed 10 Macro-F1 remains a diagnostic; support-aware Macro-F1 is always shown beside it.
- Tail false positives: report predicted/support ratio and raw predicted counts, including zero-support classes.
- AUPRC is preferred to AUROC for rare one-vs-rest discrimination.
- Temperature scaling is calibration-only because scalar positive temperature preserves argmax.
- Exp1/2 gate: paired ROI bootstrap CI for candidate-baseline delta must lie above zero; no supported-class recall loss >0.10; false-positive pattern must improve rather than merely trade head recall for tail predictions.
- Exp3 is causal evidence about group diversity, not a model promotion gate.
- Exp4 requires human adjudication; software alone cannot infer annotation correctness.

## Primary literature used to justify Exploration-7 interventions

1. Menon AK, Jayasumana S, Rawat AS, Jain H, Veit A, Kumar S. **Long-tail learning via logit adjustment.** ICLR. [Source](https://research.google/pubs/long-tail-learning-via-logit-adjustment/)  
   Relevance: gives a principled basis for class-prior logit adjustment rather than arbitrary thresholds.

2. Kang B, Xie S, Rohrbach M, Yan Z, Gordo A, Feng J, Kalantidis Y. **Decoupling representation and classifier for long-tailed recognition.** ICLR. [Source](https://ai.meta.com/research/publications/decoupling-representation-and-classifier-for-long-tailed-recognition/)  
   Relevance: motivates testing classifier-only relearning independently from the frozen representation.

3. Hörst F, Rempe M, Becker H, Heine L, Keyl J, Kleesiek J. **CellViT++: Energy-efficient and adaptive cell segmentation and classification using foundation models.** Comput Methods Programs Biomed. 277:109206. doi:10.1016/j.cmpb.2025.109206. [Source](https://pubmed.ncbi.nlm.nih.gov/41576779/)  
   Relevance: precedent for extracting cell-level transformer embeddings and adapting lightweight classifiers without automatically fine-tuning the full foundation model.

4. PUMA dataset paper. **A novel dataset for nuclei and tissue segmentation in melanoma with baseline nuclei segmentation and tissue segmentation benchmarks.** GigaScience. [Source](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giaf011/8024182)  
   Relevance: dataset/task provenance and evidence that tissue context can matter for uncommon nuclei classes.

I used these sources to justify the questions and interventions, not to predict that they would succeed. Prospective evidence in the local PUMA pipeline would still be needed for Exploration 7.

## Complete all-class evidence for executed causal probes

I keep the historical natural-population diagnosis separate from the new exploratory public-PUMA probes in the tables below. Combining them would produce a misleading validation score. Repeated baseline copies and duplicate seed-specific JSON exports appear once. A temperature-only change preserves argmax and therefore class P/R/F1, although it can change NLL and ECE. Every class value comes from a saved result; I did not retrain a model or reconstruct predictions to fill these tables.

### EXP1_PRIOR_CORRECTION per_seed / 17 / baseline / semantic

[Results](../RESULTS/EXPLORATORY_PUBLIC_PUMA/EXPERIMENTS/EXP1_PRIOR_CORRECTION/result.json). . Semantic fixed-ten macro-F1 0.424361, accuracy 0.576283, NLL 1.361503, ECE15 0.125678.


| Class | GT support | Predicted | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| tumor | 2505 | 1718 | 0.850407 | 0.583234 | 0.691925 |
| lymphocyte | 1874 | 1492 | 0.811662 | 0.646211 | 0.719548 |
| plasma_cell | 40 | 328 | 0.088415 | 0.725000 | 0.157609 |
| histiocyte | 702 | 853 | 0.324736 | 0.394587 | 0.356270 |
| melanophage | 60 | 112 | 0.348214 | 0.650000 | 0.453488 |
| neutrophil | 17 | 65 | 0.153846 | 0.588235 | 0.243902 |
| stroma | 223 | 245 | 0.444898 | 0.488789 | 0.465812 |
| epithelium | 185 | 190 | 0.600000 | 0.616216 | 0.608000 |
| endothelium | 142 | 510 | 0.219608 | 0.788732 | 0.343558 |
| apoptosis | 197 | 432 | 0.148148 | 0.324873 | 0.203498 |

### EXP1_PRIOR_CORRECTION per_seed / 17 / prior_corrected / semantic

[Results](../RESULTS/EXPLORATORY_PUBLIC_PUMA/EXPERIMENTS/EXP1_PRIOR_CORRECTION/result.json). . Semantic fixed-ten macro-F1 0.502606, accuracy 0.683263, NLL 1.017752, ECE15 0.110419.


| Class | GT support | Predicted | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| tumor | 2505 | 2521 | 0.774693 | 0.779641 | 0.777159 |
| lymphocyte | 1874 | 2021 | 0.728352 | 0.785486 | 0.755841 |
| plasma_cell | 40 | 43 | 0.441860 | 0.475000 | 0.457831 |
| histiocyte | 702 | 716 | 0.399441 | 0.407407 | 0.403385 |
| melanophage | 60 | 39 | 0.641026 | 0.416667 | 0.505051 |
| neutrophil | 17 | 22 | 0.272727 | 0.352941 | 0.307692 |
| stroma | 223 | 154 | 0.532468 | 0.367713 | 0.435013 |
| epithelium | 185 | 93 | 0.870968 | 0.437838 | 0.582734 |
| endothelium | 142 | 220 | 0.422727 | 0.654930 | 0.513812 |
| apoptosis | 197 | 116 | 0.387931 | 0.228426 | 0.287540 |

### EXP1_PRIOR_CORRECTION per_seed / 17 / class_bias_test / semantic

[Results](../RESULTS/EXPLORATORY_PUBLIC_PUMA/EXPERIMENTS/EXP1_PRIOR_CORRECTION/result.json). . Semantic fixed-ten macro-F1 0.510854, accuracy 0.691337, NLL 1.066507, ECE15 0.113082.


| Class | GT support | Predicted | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| tumor | 2505 | 3205 | 0.695788 | 0.890220 | 0.781086 |
| lymphocyte | 1874 | 1599 | 0.820513 | 0.700107 | 0.755543 |
| plasma_cell | 40 | 58 | 0.396552 | 0.575000 | 0.469388 |
| histiocyte | 702 | 410 | 0.460976 | 0.269231 | 0.339928 |
| melanophage | 60 | 40 | 0.650000 | 0.433333 | 0.520000 |
| neutrophil | 17 | 16 | 0.375000 | 0.352941 | 0.363636 |
| stroma | 223 | 196 | 0.515306 | 0.452915 | 0.482100 |
| epithelium | 185 | 135 | 0.688889 | 0.502703 | 0.581250 |
| endothelium | 142 | 192 | 0.458333 | 0.619718 | 0.526946 |
| apoptosis | 197 | 94 | 0.446809 | 0.213198 | 0.288660 |

### EXP2_NATURAL_CRT per_seed / 17 / natural_crt / semantic

[Results](../RESULTS/EXPLORATORY_PUBLIC_PUMA/EXPERIMENTS/EXP2_NATURAL_CRT/result.json). seed=17. Semantic fixed-ten macro-F1 0.488562, accuracy 0.714382, NLL 0.987116, ECE15 0.082763.


| Class | GT support | Predicted | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| tumor | 2505 | 2896 | 0.772099 | 0.892615 | 0.827995 |
| lymphocyte | 1874 | 1665 | 0.842643 | 0.748666 | 0.792879 |
| plasma_cell | 40 | 36 | 0.416667 | 0.375000 | 0.394737 |
| histiocyte | 702 | 351 | 0.421652 | 0.210826 | 0.281102 |
| melanophage | 60 | 61 | 0.491803 | 0.500000 | 0.495868 |
| neutrophil | 17 | 23 | 0.130435 | 0.176471 | 0.150000 |
| stroma | 223 | 367 | 0.376022 | 0.618834 | 0.467797 |
| epithelium | 185 | 280 | 0.521429 | 0.789189 | 0.627957 |
| endothelium | 142 | 184 | 0.369565 | 0.478873 | 0.417178 |
| apoptosis | 197 | 82 | 0.731707 | 0.304569 | 0.430108 |

### EXP3_GROUP_DIVERSITY diversity / low / 0 / semantic

[Results](../RESULTS/EXPLORATORY_PUBLIC_PUMA/EXPERIMENTS/EXP3_GROUP_DIVERSITY/result.json). seed=17, n=400, groups=14. Semantic fixed-ten macro-F1 0.288634, accuracy 0.333726, NLL 2.471127, ECE15 0.305501.


| Class | GT support | Predicted | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| tumor | 2505 | 756 | 0.887566 | 0.267864 | 0.411530 |
| lymphocyte | 1874 | 765 | 0.717647 | 0.292956 | 0.416067 |
| plasma_cell | 40 | 200 | 0.160000 | 0.800000 | 0.266667 |
| histiocyte | 702 | 984 | 0.257114 | 0.360399 | 0.300119 |
| melanophage | 60 | 201 | 0.199005 | 0.666667 | 0.306513 |
| neutrophil | 17 | 116 | 0.120690 | 0.823529 | 0.210526 |
| stroma | 223 | 1226 | 0.083197 | 0.457399 | 0.140787 |
| epithelium | 185 | 704 | 0.250000 | 0.951351 | 0.395951 |
| endothelium | 142 | 400 | 0.142500 | 0.401408 | 0.210332 |
| apoptosis | 197 | 593 | 0.151771 | 0.456853 | 0.227848 |

### EXP3_GROUP_DIVERSITY diversity / low / 1 / semantic

[Results](../RESULTS/EXPLORATORY_PUBLIC_PUMA/EXPERIMENTS/EXP3_GROUP_DIVERSITY/result.json). seed=29, n=400, groups=14. Semantic fixed-ten macro-F1 0.286636, accuracy 0.349706, NLL 2.354777, ECE15 0.269966.


| Class | GT support | Predicted | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| tumor | 2505 | 770 | 0.905195 | 0.278244 | 0.425649 |
| lymphocyte | 1874 | 979 | 0.674157 | 0.352188 | 0.462671 |
| plasma_cell | 40 | 227 | 0.140969 | 0.800000 | 0.239700 |
| histiocyte | 702 | 765 | 0.286275 | 0.311966 | 0.298569 |
| melanophage | 60 | 313 | 0.137380 | 0.716667 | 0.230563 |
| neutrophil | 17 | 113 | 0.123894 | 0.823529 | 0.215385 |
| stroma | 223 | 1057 | 0.086093 | 0.408072 | 0.142187 |
| epithelium | 185 | 597 | 0.291457 | 0.940541 | 0.445013 |
| endothelium | 142 | 553 | 0.122966 | 0.478873 | 0.195683 |
| apoptosis | 197 | 571 | 0.141856 | 0.411168 | 0.210938 |

### EXP3_GROUP_DIVERSITY diversity / low / 2 / semantic

[Results](../RESULTS/EXPLORATORY_PUBLIC_PUMA/EXPERIMENTS/EXP3_GROUP_DIVERSITY/result.json). seed=43, n=400, groups=14. Semantic fixed-ten macro-F1 0.265058, accuracy 0.305635, NLL 2.571720, ECE15 0.316424.


| Class | GT support | Predicted | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| tumor | 2505 | 590 | 0.867797 | 0.204391 | 0.330856 |
| lymphocyte | 1874 | 788 | 0.728426 | 0.306297 | 0.431255 |
| plasma_cell | 40 | 243 | 0.135802 | 0.825000 | 0.233216 |
| histiocyte | 702 | 796 | 0.255025 | 0.289174 | 0.271028 |
| melanophage | 60 | 467 | 0.100642 | 0.783333 | 0.178368 |
| neutrophil | 17 | 109 | 0.128440 | 0.823529 | 0.222222 |
| stroma | 223 | 1254 | 0.090112 | 0.506726 | 0.153013 |
| epithelium | 185 | 734 | 0.239782 | 0.951351 | 0.383025 |
| endothelium | 142 | 410 | 0.156098 | 0.450704 | 0.231884 |
| apoptosis | 197 | 554 | 0.146209 | 0.411168 | 0.215712 |

### EXP3_GROUP_DIVERSITY diversity / high / 0 / semantic

[Results](../RESULTS/EXPLORATORY_PUBLIC_PUMA/EXPERIMENTS/EXP3_GROUP_DIVERSITY/result.json). seed=17, n=400, groups=27. Semantic fixed-ten macro-F1 0.345863, accuracy 0.460892, NLL 1.743998, ECE15 0.139533.


| Class | GT support | Predicted | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| tumor | 2505 | 1500 | 0.808000 | 0.483832 | 0.605243 |
| lymphocyte | 1874 | 1013 | 0.813425 | 0.439701 | 0.570835 |
| plasma_cell | 40 | 390 | 0.079487 | 0.775000 | 0.144186 |
| histiocyte | 702 | 754 | 0.246684 | 0.264957 | 0.255495 |
| melanophage | 60 | 129 | 0.317829 | 0.683333 | 0.433862 |
| neutrophil | 17 | 115 | 0.121739 | 0.823529 | 0.212121 |
| stroma | 223 | 597 | 0.174204 | 0.466368 | 0.253659 |
| epithelium | 185 | 687 | 0.235808 | 0.875676 | 0.371560 |
| endothelium | 142 | 540 | 0.183333 | 0.697183 | 0.290323 |
| apoptosis | 197 | 220 | 0.304545 | 0.340102 | 0.321343 |

### EXP3_GROUP_DIVERSITY diversity / high / 1 / semantic

[Results](../RESULTS/EXPLORATORY_PUBLIC_PUMA/EXPERIMENTS/EXP3_GROUP_DIVERSITY/result.json). seed=29, n=400, groups=27. Semantic fixed-ten macro-F1 0.320128, accuracy 0.425736, NLL 1.849226, ECE15 0.166978.


| Class | GT support | Predicted | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| tumor | 2505 | 1424 | 0.830758 | 0.472255 | 0.602189 |
| lymphocyte | 1874 | 821 | 0.833130 | 0.364995 | 0.507607 |
| plasma_cell | 40 | 484 | 0.066116 | 0.800000 | 0.122137 |
| histiocyte | 702 | 728 | 0.211538 | 0.219373 | 0.215385 |
| melanophage | 60 | 188 | 0.223404 | 0.700000 | 0.338710 |
| neutrophil | 17 | 95 | 0.157895 | 0.882353 | 0.267857 |
| stroma | 223 | 553 | 0.171790 | 0.426009 | 0.244845 |
| epithelium | 185 | 720 | 0.212500 | 0.827027 | 0.338122 |
| endothelium | 142 | 680 | 0.150000 | 0.718310 | 0.248175 |
| apoptosis | 197 | 252 | 0.281746 | 0.360406 | 0.316258 |

### EXP3_GROUP_DIVERSITY diversity / high / 2 / semantic

[Results](../RESULTS/EXPLORATORY_PUBLIC_PUMA/EXPERIMENTS/EXP3_GROUP_DIVERSITY/result.json). seed=43, n=400, groups=27. Semantic fixed-ten macro-F1 0.319062, accuracy 0.398823, NLL 1.940597, ECE15 0.184254.


| Class | GT support | Predicted | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| tumor | 2505 | 1372 | 0.769679 | 0.421557 | 0.544751 |
| lymphocyte | 1874 | 795 | 0.792453 | 0.336179 | 0.472087 |
| plasma_cell | 40 | 389 | 0.079692 | 0.775000 | 0.144522 |
| histiocyte | 702 | 724 | 0.211326 | 0.217949 | 0.214586 |
| melanophage | 60 | 150 | 0.280000 | 0.700000 | 0.400000 |
| neutrophil | 17 | 88 | 0.159091 | 0.823529 | 0.266667 |
| stroma | 223 | 501 | 0.191617 | 0.430493 | 0.265193 |
| epithelium | 185 | 895 | 0.188827 | 0.913514 | 0.312963 |
| endothelium | 142 | 785 | 0.131210 | 0.725352 | 0.222222 |
| apoptosis | 197 | 246 | 0.313008 | 0.390863 | 0.347630 |

### EXP3_GROUP_DIVERSITY density / small / 0 / semantic

[Results](../RESULTS/EXPLORATORY_PUBLIC_PUMA/EXPERIMENTS/EXP3_GROUP_DIVERSITY/result.json). seed=17, n=200, groups=26. Semantic fixed-ten macro-F1 0.370066, accuracy 0.459378, NLL 1.631938, ECE15 0.135746.


| Class | GT support | Predicted | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| tumor | 2505 | 1040 | 0.891346 | 0.370060 | 0.522990 |
| lymphocyte | 1874 | 1215 | 0.794239 | 0.514941 | 0.624798 |
| plasma_cell | 40 | 577 | 0.057192 | 0.825000 | 0.106969 |
| histiocyte | 702 | 1146 | 0.268761 | 0.438746 | 0.333333 |
| melanophage | 60 | 116 | 0.293103 | 0.566667 | 0.386364 |
| neutrophil | 17 | 90 | 0.166667 | 0.882353 | 0.280374 |
| stroma | 223 | 340 | 0.279412 | 0.426009 | 0.337478 |
| epithelium | 185 | 789 | 0.221800 | 0.945946 | 0.359343 |
| endothelium | 142 | 471 | 0.229299 | 0.760563 | 0.352365 |
| apoptosis | 197 | 161 | 0.440994 | 0.360406 | 0.396648 |

### EXP3_GROUP_DIVERSITY density / small / 1 / semantic

[Results](../RESULTS/EXPLORATORY_PUBLIC_PUMA/EXPERIMENTS/EXP3_GROUP_DIVERSITY/result.json). seed=29, n=200, groups=26. Semantic fixed-ten macro-F1 0.366214, accuracy 0.433137, NLL 1.682945, ECE15 0.155264.


| Class | GT support | Predicted | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| tumor | 2505 | 1235 | 0.887449 | 0.437525 | 0.586096 |
| lymphocyte | 1874 | 822 | 0.766423 | 0.336179 | 0.467359 |
| plasma_cell | 40 | 1120 | 0.027679 | 0.775000 | 0.053448 |
| histiocyte | 702 | 1225 | 0.280816 | 0.490028 | 0.357032 |
| melanophage | 60 | 191 | 0.193717 | 0.616667 | 0.294821 |
| neutrophil | 17 | 87 | 0.149425 | 0.764706 | 0.250000 |
| stroma | 223 | 246 | 0.337398 | 0.372197 | 0.353945 |
| epithelium | 185 | 521 | 0.320537 | 0.902703 | 0.473088 |
| endothelium | 142 | 342 | 0.304094 | 0.732394 | 0.429752 |
| apoptosis | 197 | 156 | 0.448718 | 0.355330 | 0.396601 |

### EXP3_GROUP_DIVERSITY density / small / 2 / semantic

[Results](../RESULTS/EXPLORATORY_PUBLIC_PUMA/EXPERIMENTS/EXP3_GROUP_DIVERSITY/result.json). seed=43, n=200, groups=26. Semantic fixed-ten macro-F1 0.370510, accuracy 0.464592, NLL 1.599972, ECE15 0.122334.


| Class | GT support | Predicted | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| tumor | 2505 | 1127 | 0.838509 | 0.377246 | 0.520374 |
| lymphocyte | 1874 | 1350 | 0.737778 | 0.531483 | 0.617866 |
| plasma_cell | 40 | 594 | 0.052189 | 0.775000 | 0.097792 |
| histiocyte | 702 | 1075 | 0.273488 | 0.418803 | 0.330895 |
| melanophage | 60 | 212 | 0.179245 | 0.633333 | 0.279412 |
| neutrophil | 17 | 73 | 0.178082 | 0.764706 | 0.288889 |
| stroma | 223 | 275 | 0.334545 | 0.412556 | 0.369478 |
| epithelium | 185 | 647 | 0.265842 | 0.929730 | 0.413462 |
| endothelium | 142 | 457 | 0.247265 | 0.795775 | 0.377295 |
| apoptosis | 197 | 135 | 0.503704 | 0.345178 | 0.409639 |

### EXP3_GROUP_DIVERSITY density / large / 0 / semantic

[Results](../RESULTS/EXPLORATORY_PUBLIC_PUMA/EXPERIMENTS/EXP3_GROUP_DIVERSITY/result.json). seed=17, n=400, groups=26. Semantic fixed-ten macro-F1 0.404503, accuracy 0.545164, NLL 1.415042, ECE15 0.108022.


| Class | GT support | Predicted | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| tumor | 2505 | 1718 | 0.870780 | 0.597206 | 0.708501 |
| lymphocyte | 1874 | 1174 | 0.793015 | 0.496798 | 0.610892 |
| plasma_cell | 40 | 486 | 0.069959 | 0.850000 | 0.129278 |
| histiocyte | 702 | 1120 | 0.289286 | 0.461538 | 0.355653 |
| melanophage | 60 | 130 | 0.246154 | 0.533333 | 0.336842 |
| neutrophil | 17 | 76 | 0.144737 | 0.647059 | 0.236559 |
| stroma | 223 | 324 | 0.271605 | 0.394619 | 0.321755 |
| epithelium | 185 | 384 | 0.382812 | 0.794595 | 0.516696 |
| endothelium | 142 | 380 | 0.263158 | 0.704225 | 0.383142 |
| apoptosis | 197 | 153 | 0.509804 | 0.395939 | 0.445714 |

### EXP3_GROUP_DIVERSITY density / large / 1 / semantic

[Results](../RESULTS/EXPLORATORY_PUBLIC_PUMA/EXPERIMENTS/EXP3_GROUP_DIVERSITY/result.json). seed=29, n=400, groups=26. Semantic fixed-ten macro-F1 0.388741, accuracy 0.504626, NLL 1.517551, ECE15 0.125866.


| Class | GT support | Predicted | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| tumor | 2505 | 1531 | 0.906597 | 0.554092 | 0.687810 |
| lymphocyte | 1874 | 1014 | 0.781065 | 0.422625 | 0.548476 |
| plasma_cell | 40 | 768 | 0.046875 | 0.900000 | 0.089109 |
| histiocyte | 702 | 972 | 0.302469 | 0.418803 | 0.351254 |
| melanophage | 60 | 201 | 0.179104 | 0.600000 | 0.275862 |
| neutrophil | 17 | 78 | 0.153846 | 0.705882 | 0.252632 |
| stroma | 223 | 358 | 0.270950 | 0.434978 | 0.333907 |
| epithelium | 185 | 443 | 0.349887 | 0.837838 | 0.493631 |
| endothelium | 142 | 433 | 0.247113 | 0.753521 | 0.372174 |
| apoptosis | 197 | 147 | 0.564626 | 0.421320 | 0.482558 |

### EXP3_GROUP_DIVERSITY density / large / 2 / semantic

[Results](../RESULTS/EXPLORATORY_PUBLIC_PUMA/EXPERIMENTS/EXP3_GROUP_DIVERSITY/result.json). seed=43, n=400, groups=26. Semantic fixed-ten macro-F1 0.385348, accuracy 0.510177, NLL 1.498973, ECE15 0.110683.


| Class | GT support | Predicted | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| tumor | 2505 | 1516 | 0.871372 | 0.527345 | 0.657050 |
| lymphocyte | 1874 | 1175 | 0.747234 | 0.468517 | 0.575927 |
| plasma_cell | 40 | 445 | 0.071910 | 0.800000 | 0.131959 |
| histiocyte | 702 | 1138 | 0.285589 | 0.462963 | 0.353261 |
| melanophage | 60 | 217 | 0.161290 | 0.583333 | 0.252708 |
| neutrophil | 17 | 74 | 0.162162 | 0.705882 | 0.263736 |
| stroma | 223 | 302 | 0.317881 | 0.430493 | 0.365714 |
| epithelium | 185 | 472 | 0.326271 | 0.832432 | 0.468798 |
| endothelium | 142 | 461 | 0.229935 | 0.746479 | 0.351575 |
| apoptosis | 197 | 145 | 0.510345 | 0.375635 | 0.432749 |

### HISTORICAL_EXPLORATION6_NATURAL 

[Results](../RESULTS/METRICS/HISTORICAL_EXPLORATION6_NATURAL/metrics.json). . Semantic fixed-ten macro-F1 0.209392, accuracy 0.531180, NLL 1.545238, ECE15 0.141573.


| Class | GT support | Predicted | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| tumor | 8817 | 4550 | 0.989011 | 0.510378 | 0.673300 |
| lymphocyte | 1177 | 1540 | 0.529870 | 0.693288 | 0.600662 |
| plasma_cell | 8 | 378 | 0.010582 | 0.500000 | 0.020725 |
| histiocyte | 231 | 1436 | 0.094011 | 0.584416 | 0.161968 |
| melanophage | 11 | 200 | 0.035000 | 0.636364 | 0.066351 |
| neutrophil | 0 | 47 | 0.000000 | 0.000000 | 0.000000 |
| stroma | 278 | 615 | 0.183740 | 0.406475 | 0.253080 |
| epithelium | 0 | 517 | 0.000000 | 0.000000 | 0.000000 |
| endothelium | 129 | 501 | 0.147705 | 0.573643 | 0.234921 |
| apoptosis | 61 | 928 | 0.044181 | 0.672131 | 0.082912 |

### Context probe: fold-level outcomes and the missing class detail

[Results](../RESULTS/EXPLORATORY_PUBLIC_PUMA/METRICS/EXP5_CONTEXT_PROBE.json). This artifact records fold-level macro-F1/accuracy and selected binary-pair AUPRC. It does not contain a ten-class P/R/F1/support vector for every context arm, so those absent fields cannot be reconstructed from its aggregate scores. The complete fold comparison is:

| Seed | Arm | Fold | Semantic macro-F1 | Accuracy |
| --- | --- | --- | --- | --- |
| 17 | local | 0 | 0.449304 | 0.572967 |
| 17 | local | 1 | 0.404735 | 0.628416 |
| 17 | local | 2 | 0.406435 | 0.643692 |
| 17 | local_plus_context | 0 | 0.510637 | 0.652236 |
| 17 | local_plus_context | 1 | 0.476516 | 0.631963 |
| 17 | local_plus_context | 2 | 0.459215 | 0.711428 |
| 17 | capacity_placebo_local_plus_shuffled_context | 0 | 0.451904 | 0.582927 |
| 17 | capacity_placebo_local_plus_shuffled_context | 1 | 0.430598 | 0.611308 |
| 17 | capacity_placebo_local_plus_shuffled_context | 2 | 0.410113 | 0.653397 |
| 29 | local | 0 | 0.393099 | 0.519651 |
| 29 | local | 1 | 0.457347 | 0.633663 |
| 29 | local | 2 | 0.304791 | 0.606538 |
| 29 | local_plus_context | 0 | 0.427349 | 0.598452 |
| 29 | local_plus_context | 1 | 0.486969 | 0.646578 |
| 29 | local_plus_context | 2 | 0.337559 | 0.612643 |
| 29 | capacity_placebo_local_plus_shuffled_context | 0 | 0.394810 | 0.558158 |
| 29 | capacity_placebo_local_plus_shuffled_context | 1 | 0.454641 | 0.624623 |
| 29 | capacity_placebo_local_plus_shuffled_context | 2 | 0.316870 | 0.597479 |
| 43 | local | 0 | 0.460905 | 0.709349 |
| 43 | local | 1 | 0.373160 | 0.561076 |
| 43 | local | 2 | 0.422427 | 0.563400 |
| 43 | local_plus_context | 0 | 0.547566 | 0.764931 |
| 43 | local_plus_context | 1 | 0.433924 | 0.591049 |
| 43 | local_plus_context | 2 | 0.495392 | 0.629014 |
| 43 | capacity_placebo_local_plus_shuffled_context | 0 | 0.469553 | 0.707437 |
| 43 | capacity_placebo_local_plus_shuffled_context | 1 | 0.361831 | 0.542804 |
| 43 | capacity_placebo_local_plus_shuffled_context | 2 | 0.445056 | 0.577243 |

### What the interventions can and cannot establish

The prior-correction result helped me separate boundary changes from representation changes. It improved an aggregate score with the representation fixed, but the large class-recall losses prevented replacement of the baseline. Temperature scaling preserved argmax and could only address confidence calibration, not those confusions. Natural-risk relearning changed the classifier more substantially and still had to satisfy the same class-harm and population-validity requirements. Once the primary seed failed, I closed conditional replication instead of looking for a more favorable seed.

Within the admitted sampling design, I compared adding groups with adding nuclei inside existing groups. Reused groups, folds and seeds remain dependent observations, so I interpret the positive means as directional support for a data-diversity hypothesis rather than a patient-level causal claim. The wider-context gain over shuffled-context capacity control also supports useful contextual alignment. It cannot identify the responsible tissue cue or establish a deployable architecture.

I could not resolve the label-ambiguity explanation because the blinded pathology sheet contains no completed labels. Likewise, missing fixed Stage-1 provenance prevents a complete end-to-end performance claim. I retained the prior Gaussian checkpoint and used the completed probes to narrow the follow-up design.
