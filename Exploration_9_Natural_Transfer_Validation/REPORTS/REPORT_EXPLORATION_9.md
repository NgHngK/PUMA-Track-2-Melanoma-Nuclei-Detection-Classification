# Exploration 9: natural transfer validation

## Purpose and result

I wanted to check whether Context-C still improved on the local anchor across the broader natural PUMA training population. It did: the saved decision is **TRANSFER_PASS**, and all locked gates passed. I interpret this as an exploratory, ROI-grouped out-of-fold comparison on official training data. It is not an independent external test.

## Data and representation

To isolate Stage 2, the evaluation used ten-class classification at ground-truth nucleus centroids. Detection and tissue segmentation were outside this comparison. The official training population contained 97,193 nuclei in 205 ROIs: 103 primary and 102 metastatic ROIs. ROI was the highest grouping level I could verify; patient-level independence remains unestablished.

The fixed class order was tumor, lymphocyte, plasma cell, histiocyte, melanophage, neutrophil, stroma, epithelium, endothelium, and apoptosis. Both FOV96 and FOV192 crops came from the same 1024-pixel ROI, without using the separate 5120-pixel context image. White padding handled crop boundaries, and PIL bicubic interpolation resized the crops to 224 pixels. RGB normalization used means [0.485, 0.456, 0.406] and standard deviations [0.229, 0.224, 0.225].

UNI2-h stayed frozen during this comparison. Its output contained 1 CLS token, 8 register tokens, and 256 spatial tokens of dimension 1536. Removing the nine prefix tokens left a 16 × 16 spatial grid. Gaussian pooling around the actual nucleus position, with sigma 1.5 token units, produced a 1536-dimensional vector for each field of view. Stable nucleus UIDs aligned these vectors with the labels and 16-dimensional Tier-A features. Tier-A normalization was fitted only on the training rows in each comparison.

## Classifier

The anchor combined nonaffine LayerNorm with a linear 1536-to-10 appearance classifier. Its rank-eight multiplicative interaction between local appearance and Tier-A added a ten-logit residual, whose output projection started at zero. The resulting anchor had 27,866 trainable parameters.

Context-C extended this anchor with a second rank-eight interaction between normalized local and wider-field features. A learned scalar passed through tanh scaled its ten-logit output and started at zero. The complete classifier had 52,523 trainable parameters. The encoder remained frozen during cached-head training, and the class mapping, feature definitions, and evaluator contract stayed identical across the core comparisons.

## Experimental setup

I compared ANCHOR with ANCHOR_CONTEXT_C using seeds 17, 29, and 43 across five ROI-grouped folds. For each of the 30 fits, I used ten epochs, batch size 64, AdamW, inverse-class-frequency sampling, and ordinary cross-entropy. I recorded the configuration and feature hashes in [Run contract](../TRAINING_RESULTS/RUN_CONTRACT.json). One provenance limitation remains: context-scale initialization was reconstructed as the zero-residual default, as the saved record states.

FP32 execution preserves the feature and training contract. The fast execution path reduces synchronization, reuses final held-fold logits, and defers redundant history exports without changing optimizer updates. The preprocessing and training notebooks are the two entry points. Feature extraction requires the original images and encoder weights; cached-head training uses the saved aligned features.

## Quantitative results

| Seed | Anchor ROI-F1 | Context-C ROI-F1 | Paired gain |
| --- | --- | --- | --- |
| 17 | 0.2580759787907793 | 0.28248320359265044 | 0.02440722480187113 |
| 29 | 0.25522993330307103 | 0.27603392794723686 | 0.020803994644165824 |
| 43 | 0.2616883732022749 | 0.27516759276424796 | 0.013479219561973066 |

Mean paired ROI-F1 gain is **0.019563479669336675**. The 5,000-replicate ROI bootstrap interval is **[0.016548355719802475, 0.022665436440666983]**. All three seeds improve. Mean summed-F1 gain is **0.0621266852631512**. Leaving out any single ROI keeps the mean ROI gain positive.

The largest mean class-recall decrease is 0.07482014388489207, and the largest single-seed decrease is 0.08057553956834529. The complete class changes, calibration metrics, ROI support, and bootstrap results are in [Natural transfer summary](../TRAINING_RESULTS/NATURAL_TRANSFER_SUMMARY.json).

## Interpretation and limitations

Wider context contributes useful information under the tested training recipe. Passing the relative transfer gates does not resolve rare-class generalization: Kish effective ROI support remains only about 4.35 for neutrophils and 7.20 for plasma cells. Repeated sampling of a cached vector does not create new biological examples. Calibration also needs separate attention; the per-seed ECE values are not uniformly improved by Context-C.

The next study retains Context-C and tests how sampling, source diversity, and prior-aware objectives affect the fixed-split result. Histories, checkpoints, out-of-fold predictions, and the final transfer summary support reproduction of this comparison.

## Why a natural-population transfer test was necessary

The low-rank local-by-context interaction had worked during earlier development, but I had selected it on a restricted roster. I still needed to test it across the broader source distribution. In Exploration 9, I kept the two architectures fixed and used five ROI-grouped folds with three optimizer seeds. Each held-fold prediction came from a fit that had not trained on that ROI. Pooling those predictions covered the 97,193-nucleus population once per seed, with both arms evaluated on the same nuclei.

By "natural", I mean the population and held-ROI evaluation. I still trained with inverse-class-frequency replacement sampling, so the Context-C improvement applies to that historical recipe; I had not established that its balancing was optimal. The three seeds help me assess optimization robustness on the same data and folds. They do not provide three biological cohorts.

I used a fixed ten-epoch endpoint to avoid giving each arm its own best epoch. Earlier validation peaks remain useful for understanding training, but I do not use them to replace the endpoint. I also distinguish the OOF aggregate from individual fold scores: their weights differ, so averaging five fold scores equally need not reproduce the saved ROI-level result.

## What the transfer result resolves

The improvement appeared in all three paired seeds, the bootstrap interval excluded zero, and it stayed positive when I removed any one ROI. That gives me more confidence that the gain is not carried by one seed or one exceptional source. My conclusion is limited to this experiment: the context interaction contributes useful classification information.

Looking at the classes changed how I read the overall gain. Melanophage recall fell in every seed, while tumor, lymphocyte, and epithelium recall improved consistently. I therefore keep the class-level results alongside the transfer decision. ECE15 was higher for Context-C in all three seeds, and the smaller NLL changes were not uniformly favorable either. The better F1 did not bring better calibration.

One explanation I would investigate is that wider neighborhoods help distinguish tissue compartments but suppress minority phenotypes when local evidence conflicts with context. The extra branch might also shift decision boundaries under the balanced training prior. I cannot separate these explanations using aggregate metrics. That is why Exploration 10 examines the training recipe and source exposure before increasing architecture complexity again.

## Architecture and information flow in detail

For B nuclei, local and wide appearance matrices each have shape B x 1536 and standardized Tier-A has shape B x 16. Local appearance passes through nonaffine LayerNorm and a 1536-to-10 classifier. A second path projects local appearance and Tier-A independently to eight dimensions, multiplies these vectors elementwise, and projects the product to ten residual logits. This is a conditional interaction: a Tier-A cue can have a different influence for different local appearance coordinates.

Context-C independently projects normalized local and wider appearance to eight dimensions. Their elementwise product produces ten context logits. The final logits are the appearance classifier plus the Tier-A residual plus tanh(s) times the context residual. The anchor has 27,866 trainable parameters; Context-C has 52,523. The frozen UNI2-h encoder is not updated by cached-head optimization. A small head can still memorize source-associated directions when rare examples are reused.

The Tier-A output projection starts at zero; its upstream projections initially receive no gradient through that residual until the output weights learn. The context scalar starts at zero, so its predictive contribution is initially zero and the scalar can receive a gradient before the context branch receives its full predictive gradient. This starts prediction from the simpler anchor but does not guarantee that the learned residual stays small or improves generalization.

I use the wider field from the same ROI; neither another source nor the separate large context image enters this branch. Neighborhood arrangement and source-specific texture may both affect the prediction, and this comparison measures their combined effect. I rely on training-only standardization and UID alignment to keep the comparison valid: mismatched labels, coordinates, appearance features, or Tier-A rows would invalidate it.

## Detailed measurement and interpretation guide

I include every completed arm and seed in the tables below. Scores are proportions shown to six decimals, while the linked artifacts retain full precision. Where a value is missing I write "not saved", rather than replacing it with zero.

Semantic precision is TP/(TP+FP), recall is TP/(TP+FN), and F1 is their harmonic mean. These nucleus-pooled quantities differ from the equally weighted mean over ROIs. A common class in a cellular ROI can dominate pooled counts but contributes only one ROI to the primary average. A rare class with good pooled F1 can have a low all-ROI class F1 because most ROIs lack the class and absent fixed classes contribute zero. "Positive-ROI R" averages recall over ROIs with ground-truth support. "Pred/GT" is predicted count divided by ground-truth count; values above one indicate overprediction and below one indicate underprediction, without identifying the cause.

The train-validation gap uses natural full-training evaluation where available, not a comparison of sampled minibatch loss with natural validation loss. Those losses may use different distributions and weights. Improving training scores with deteriorating validation is consistent with overfitting, but does not identify whether source appearance, class morphology, annotation variability, or optimization is responsible. Mechanistic explanations are hypotheses, not measured causal findings.

## Complete out-of-fold class measurements by arm and seed

For each table, I use all OOF nuclei at epoch ten. I reconstruct the ROI quantities from the saved per-ROI TP/FP/FN arrays and take the semantic quantities from the same endpoint file.

### ANCHOR_CONTEXT_C_seed17

[Source](../TRAINING_RESULTS/METRICS/ANCHOR_CONTEXT_C_seed17.json); ROI-F1 0.282483, semantic Macro-F1 0.537309, pooled matching F1 0.537562, NLL 1.109191, ECE15 0.147489.

| Class | GT nuclei | Semantic P | Semantic R | Semantic F1 | All-ROI class F1 | GT-positive ROIs | Positive-ROI R | Pred/GT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tumor | 57234 | 0.954110 | 0.784289 | 0.860905 | 0.834792 | 205 | 0.773878 | 0.822011 |
| lymphocyte | 21643 | 0.781411 | 0.798272 | 0.789752 | 0.593681 | 193 | 0.685336 | 1.021577 |
| plasma_cell | 520 | 0.461005 | 0.511538 | 0.484959 | 0.073614 | 56 | 0.329665 | 1.109615 |
| histiocyte | 7168 | 0.425362 | 0.610212 | 0.501289 | 0.255461 | 115 | 0.574925 | 1.434570 |
| melanophage | 695 | 0.416484 | 0.545324 | 0.472274 | 0.161822 | 77 | 0.498945 | 1.309353 |
| neutrophil | 366 | 0.450602 | 0.510929 | 0.478873 | 0.065418 | 30 | 0.458418 | 1.133880 |
| stroma | 3856 | 0.330331 | 0.523081 | 0.404939 | 0.269996 | 160 | 0.507715 | 1.583506 |
| epithelium | 2211 | 0.435303 | 0.756219 | 0.552545 | 0.090405 | 29 | 0.711725 | 1.737223 |
| endothelium | 1696 | 0.401674 | 0.566038 | 0.469897 | 0.283640 | 126 | 0.573948 | 1.409198 |
| apoptosis | 1804 | 0.270632 | 0.527162 | 0.357653 | 0.196004 | 126 | 0.555701 | 1.947894 |

The strongest semantic F1 is tumor (0.860905); the weakest is apoptosis (0.357653). This is a within-run comparison, not a claim that the ranking is stable across populations.


### ANCHOR_CONTEXT_C_seed29

[Source](../TRAINING_RESULTS/METRICS/ANCHOR_CONTEXT_C_seed29.json); ROI-F1 0.276034, semantic Macro-F1 0.526974, pooled matching F1 0.527072, NLL 1.110240, ECE15 0.147089.

| Class | GT nuclei | Semantic P | Semantic R | Semantic F1 | All-ROI class F1 | GT-positive ROIs | Positive-ROI R | Pred/GT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tumor | 57234 | 0.944533 | 0.791121 | 0.861047 | 0.832990 | 205 | 0.779596 | 0.837579 |
| lymphocyte | 21643 | 0.808783 | 0.744583 | 0.775356 | 0.578217 | 193 | 0.632883 | 0.920621 |
| plasma_cell | 520 | 0.410811 | 0.438462 | 0.424186 | 0.069307 | 56 | 0.325018 | 1.067308 |
| histiocyte | 7168 | 0.399114 | 0.628348 | 0.488159 | 0.250863 | 115 | 0.596437 | 1.574358 |
| melanophage | 695 | 0.431990 | 0.493525 | 0.460712 | 0.140038 | 77 | 0.414619 | 1.142446 |
| neutrophil | 366 | 0.482587 | 0.530055 | 0.505208 | 0.059425 | 30 | 0.427808 | 1.098361 |
| stroma | 3856 | 0.299373 | 0.557054 | 0.389448 | 0.260090 | 160 | 0.522318 | 1.860737 |
| epithelium | 2211 | 0.445102 | 0.680235 | 0.538104 | 0.089635 | 29 | 0.657855 | 1.528268 |
| endothelium | 1696 | 0.417042 | 0.545401 | 0.472662 | 0.274351 | 126 | 0.526286 | 1.307783 |
| apoptosis | 1804 | 0.268313 | 0.523836 | 0.354863 | 0.205424 | 126 | 0.559873 | 1.952328 |

The strongest semantic F1 is tumor (0.861047); the weakest is apoptosis (0.354863). This is a within-run comparison, not a claim that the ranking is stable across populations.


### ANCHOR_CONTEXT_C_seed43

[Source](../TRAINING_RESULTS/METRICS/ANCHOR_CONTEXT_C_seed43.json); ROI-F1 0.275168, semantic Macro-F1 0.528581, pooled matching F1 0.528786, NLL 1.082696, ECE15 0.142190.

| Class | GT nuclei | Semantic P | Semantic R | Semantic F1 | All-ROI class F1 | GT-positive ROIs | Positive-ROI R | Pred/GT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tumor | 57234 | 0.951236 | 0.794475 | 0.865817 | 0.835760 | 205 | 0.777471 | 0.835203 |
| lymphocyte | 21643 | 0.757696 | 0.788153 | 0.772624 | 0.569145 | 193 | 0.674587 | 1.040198 |
| plasma_cell | 520 | 0.353325 | 0.521154 | 0.421134 | 0.070672 | 56 | 0.370314 | 1.475000 |
| histiocyte | 7168 | 0.437793 | 0.546875 | 0.486292 | 0.238556 | 115 | 0.500323 | 1.249163 |
| melanophage | 695 | 0.402273 | 0.509353 | 0.449524 | 0.143038 | 77 | 0.444690 | 1.266187 |
| neutrophil | 366 | 0.444700 | 0.527322 | 0.482500 | 0.065743 | 30 | 0.448017 | 1.185792 |
| stroma | 3856 | 0.314268 | 0.538641 | 0.396942 | 0.271533 | 160 | 0.531928 | 1.713952 |
| epithelium | 2211 | 0.444931 | 0.781999 | 0.567164 | 0.091477 | 29 | 0.718185 | 1.757576 |
| endothelium | 1696 | 0.378486 | 0.560142 | 0.451736 | 0.274218 | 126 | 0.550318 | 1.479953 |
| apoptosis | 1804 | 0.320648 | 0.504435 | 0.392072 | 0.191532 | 126 | 0.502580 | 1.573171 |

The strongest semantic F1 is tumor (0.865817); the weakest is apoptosis (0.392072). This is a within-run comparison, not a claim that the ranking is stable across populations.


### ANCHOR_seed17

[Source](../TRAINING_RESULTS/METRICS/ANCHOR_seed17.json); ROI-F1 0.258076, semantic Macro-F1 0.464407, pooled matching F1 0.464723, NLL 1.110773, ECE15 0.137503.

| Class | GT nuclei | Semantic P | Semantic R | Semantic F1 | All-ROI class F1 | GT-positive ROIs | Positive-ROI R | Pred/GT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tumor | 57234 | 0.938869 | 0.699305 | 0.801570 | 0.769887 | 205 | 0.692013 | 0.744837 |
| lymphocyte | 21643 | 0.768383 | 0.754147 | 0.761199 | 0.563331 | 193 | 0.646217 | 0.981472 |
| plasma_cell | 520 | 0.234525 | 0.517308 | 0.322735 | 0.057272 | 56 | 0.377919 | 2.205769 |
| histiocyte | 7168 | 0.352158 | 0.553292 | 0.430385 | 0.232481 | 115 | 0.525889 | 1.571150 |
| melanophage | 695 | 0.307638 | 0.625899 | 0.412518 | 0.152734 | 77 | 0.582317 | 2.034532 |
| neutrophil | 366 | 0.454094 | 0.500000 | 0.475943 | 0.055882 | 30 | 0.411155 | 1.101093 |
| stroma | 3856 | 0.303727 | 0.534751 | 0.387412 | 0.266220 | 160 | 0.530047 | 1.760633 |
| epithelium | 2211 | 0.295553 | 0.646314 | 0.405620 | 0.080797 | 29 | 0.635593 | 2.186793 |
| endothelium | 1696 | 0.266766 | 0.525354 | 0.353852 | 0.235388 | 126 | 0.525205 | 1.969340 |
| apoptosis | 1804 | 0.210361 | 0.481707 | 0.292839 | 0.166767 | 126 | 0.530146 | 2.289911 |

The strongest semantic F1 is tumor (0.801570); the weakest is apoptosis (0.292839). This is a within-run comparison, not a claim that the ranking is stable across populations.


### ANCHOR_seed29

[Source](../TRAINING_RESULTS/METRICS/ANCHOR_seed29.json); ROI-F1 0.255230, semantic Macro-F1 0.472143, pooled matching F1 0.472585, NLL 1.120367, ECE15 0.136412.

| Class | GT nuclei | Semantic P | Semantic R | Semantic F1 | All-ROI class F1 | GT-positive ROIs | Positive-ROI R | Pred/GT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tumor | 57234 | 0.937769 | 0.696928 | 0.799607 | 0.766963 | 205 | 0.689071 | 0.743177 |
| lymphocyte | 21643 | 0.793625 | 0.720140 | 0.755099 | 0.543337 | 193 | 0.594836 | 0.907407 |
| plasma_cell | 520 | 0.356936 | 0.475000 | 0.407591 | 0.068033 | 56 | 0.335879 | 1.330769 |
| histiocyte | 7168 | 0.324865 | 0.602958 | 0.422235 | 0.231777 | 115 | 0.571249 | 1.856027 |
| melanophage | 695 | 0.304381 | 0.569784 | 0.396794 | 0.141713 | 77 | 0.504069 | 1.871942 |
| neutrophil | 366 | 0.426339 | 0.521858 | 0.469287 | 0.058772 | 30 | 0.470054 | 1.224044 |
| stroma | 3856 | 0.275740 | 0.567427 | 0.371131 | 0.258774 | 160 | 0.557023 | 2.057832 |
| epithelium | 2211 | 0.372334 | 0.568521 | 0.449973 | 0.079398 | 29 | 0.575770 | 1.526911 |
| endothelium | 1696 | 0.259238 | 0.537736 | 0.349827 | 0.237664 | 126 | 0.531669 | 2.074292 |
| apoptosis | 1804 | 0.210799 | 0.519401 | 0.299888 | 0.165869 | 126 | 0.539183 | 2.463969 |

The strongest semantic F1 is tumor (0.799607); the weakest is apoptosis (0.299888). This is a within-run comparison, not a claim that the ranking is stable across populations.


### ANCHOR_seed43

[Source](../TRAINING_RESULTS/METRICS/ANCHOR_seed43.json); ROI-F1 0.261688, semantic Macro-F1 0.469481, pooled matching F1 0.469732, NLL 1.080409, ECE15 0.132587.

| Class | GT nuclei | Semantic P | Semantic R | Semantic F1 | All-ROI class F1 | GT-positive ROIs | Positive-ROI R | Pred/GT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tumor | 57234 | 0.936788 | 0.712321 | 0.809278 | 0.779645 | 205 | 0.707750 | 0.760387 |
| lymphocyte | 21643 | 0.758768 | 0.752668 | 0.755706 | 0.549548 | 193 | 0.634802 | 0.991960 |
| plasma_cell | 520 | 0.271293 | 0.496154 | 0.350782 | 0.072423 | 56 | 0.391817 | 1.828846 |
| histiocyte | 7168 | 0.381737 | 0.546456 | 0.449481 | 0.232389 | 115 | 0.508785 | 1.431501 |
| melanophage | 695 | 0.309653 | 0.576978 | 0.403015 | 0.159306 | 77 | 0.556386 | 1.863309 |
| neutrophil | 366 | 0.436195 | 0.513661 | 0.471769 | 0.060257 | 30 | 0.431753 | 1.177596 |
| stroma | 3856 | 0.297031 | 0.516338 | 0.377119 | 0.259572 | 160 | 0.506019 | 1.738330 |
| epithelium | 2211 | 0.297447 | 0.632293 | 0.404572 | 0.082655 | 29 | 0.649149 | 2.125735 |
| endothelium | 1696 | 0.261687 | 0.574292 | 0.359542 | 0.249233 | 126 | 0.581558 | 2.194575 |
| apoptosis | 1804 | 0.225066 | 0.516630 | 0.313541 | 0.171856 | 126 | 0.554482 | 2.295455 |

The strongest semantic F1 is tumor (0.809278); the weakest is apoptosis (0.313541). This is a within-run comparison, not a claim that the ranking is stable across populations.


## All 30 fold fits: training and held-ROI dynamics

I show epochs one, five, and ten for every completed fit so I can compare the early behavior with the fixed epoch-ten endpoint. Natural-training evaluation remains separate from sampled minibatch CE.

| Fit | Epoch | Train ROI-F1 | Held ROI-F1 | Train Macro-F1 | Held Macro-F1 | Sampled CE | Held NLL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ANCHOR_CONTEXT_C_seed17_fold0 | 1 | 0.337460 | 0.267832 | 0.641017 | 0.525726 | 0.561721 | 0.820992 |
| ANCHOR_CONTEXT_C_seed17_fold0 | 5 | 0.376216 | 0.262691 | 0.720675 | 0.514177 | 0.274743 | 0.966086 |
| ANCHOR_CONTEXT_C_seed17_fold0 | 10 | 0.397567 | 0.268389 | 0.777915 | 0.523058 | 0.217945 | 1.133827 |
| ANCHOR_CONTEXT_C_seed17_fold1 | 1 | 0.317072 | 0.281965 | 0.615100 | 0.484019 | 0.566059 | 1.020234 |
| ANCHOR_CONTEXT_C_seed17_fold1 | 5 | 0.368266 | 0.278994 | 0.723186 | 0.482909 | 0.271036 | 1.166713 |
| ANCHOR_CONTEXT_C_seed17_fold1 | 10 | 0.405821 | 0.293533 | 0.798845 | 0.511962 | 0.210165 | 1.183998 |
| ANCHOR_CONTEXT_C_seed17_fold2 | 1 | 0.330555 | 0.258703 | 0.647274 | 0.467815 | 0.588739 | 1.025125 |
| ANCHOR_CONTEXT_C_seed17_fold2 | 5 | 0.393840 | 0.267676 | 0.772348 | 0.476214 | 0.275666 | 0.927550 |
| ANCHOR_CONTEXT_C_seed17_fold2 | 10 | 0.399271 | 0.265879 | 0.770853 | 0.478637 | 0.211501 | 1.287295 |
| ANCHOR_CONTEXT_C_seed17_fold3 | 1 | 0.321857 | 0.255241 | 0.626382 | 0.515018 | 0.566545 | 1.060656 |
| ANCHOR_CONTEXT_C_seed17_fold3 | 5 | 0.356794 | 0.257709 | 0.704167 | 0.522349 | 0.273722 | 0.961241 |
| ANCHOR_CONTEXT_C_seed17_fold3 | 10 | 0.420506 | 0.279741 | 0.817102 | 0.542440 | 0.215831 | 0.970968 |
| ANCHOR_CONTEXT_C_seed17_fold4 | 1 | 0.306402 | 0.284930 | 0.606752 | 0.498627 | 0.570440 | 0.971774 |
| ANCHOR_CONTEXT_C_seed17_fold4 | 5 | 0.358362 | 0.282766 | 0.695453 | 0.505776 | 0.274849 | 1.035104 |
| ANCHOR_CONTEXT_C_seed17_fold4 | 10 | 0.404419 | 0.306824 | 0.794081 | 0.549871 | 0.211487 | 0.956572 |
| ANCHOR_CONTEXT_C_seed29_fold0 | 1 | 0.335794 | 0.265148 | 0.635536 | 0.524926 | 0.565156 | 0.782596 |
| ANCHOR_CONTEXT_C_seed29_fold0 | 5 | 0.373639 | 0.262586 | 0.717709 | 0.513712 | 0.275153 | 0.954686 |
| ANCHOR_CONTEXT_C_seed29_fold0 | 10 | 0.400691 | 0.268457 | 0.762979 | 0.519468 | 0.223912 | 1.177160 |
| ANCHOR_CONTEXT_C_seed29_fold1 | 1 | 0.311263 | 0.283171 | 0.627153 | 0.481669 | 0.568417 | 1.169769 |
| ANCHOR_CONTEXT_C_seed29_fold1 | 5 | 0.379540 | 0.291596 | 0.761854 | 0.516308 | 0.267368 | 1.039382 |
| ANCHOR_CONTEXT_C_seed29_fold1 | 10 | 0.390919 | 0.283170 | 0.796809 | 0.509183 | 0.214085 | 1.199026 |
| ANCHOR_CONTEXT_C_seed29_fold2 | 1 | 0.327780 | 0.254899 | 0.647928 | 0.471832 | 0.585969 | 0.953567 |
| ANCHOR_CONTEXT_C_seed29_fold2 | 5 | 0.364880 | 0.263922 | 0.720051 | 0.475728 | 0.274064 | 0.949732 |
| ANCHOR_CONTEXT_C_seed29_fold2 | 10 | 0.412382 | 0.262150 | 0.810727 | 0.489972 | 0.217913 | 1.201356 |
| ANCHOR_CONTEXT_C_seed29_fold3 | 1 | 0.319680 | 0.257407 | 0.640617 | 0.521348 | 0.559882 | 0.857227 |
| ANCHOR_CONTEXT_C_seed29_fold3 | 5 | 0.384212 | 0.268682 | 0.749035 | 0.532002 | 0.276579 | 0.848290 |
| ANCHOR_CONTEXT_C_seed29_fold3 | 10 | 0.402182 | 0.268659 | 0.789242 | 0.513360 | 0.216335 | 0.952178 |
| ANCHOR_CONTEXT_C_seed29_fold4 | 1 | 0.333980 | 0.307709 | 0.654437 | 0.524370 | 0.573668 | 0.669761 |
| ANCHOR_CONTEXT_C_seed29_fold4 | 5 | 0.365149 | 0.290248 | 0.741758 | 0.523756 | 0.279701 | 0.888905 |
| ANCHOR_CONTEXT_C_seed29_fold4 | 10 | 0.397899 | 0.299153 | 0.800493 | 0.541612 | 0.218354 | 1.022429 |
| ANCHOR_CONTEXT_C_seed43_fold0 | 1 | 0.341410 | 0.266076 | 0.674609 | 0.547363 | 0.566458 | 0.633705 |
| ANCHOR_CONTEXT_C_seed43_fold0 | 5 | 0.401481 | 0.278408 | 0.775792 | 0.571999 | 0.283715 | 0.738920 |
| ANCHOR_CONTEXT_C_seed43_fold0 | 10 | 0.412370 | 0.273088 | 0.791587 | 0.551194 | 0.219751 | 0.875639 |
| ANCHOR_CONTEXT_C_seed43_fold1 | 1 | 0.317429 | 0.279749 | 0.621010 | 0.490760 | 0.560387 | 0.950098 |
| ANCHOR_CONTEXT_C_seed43_fold1 | 5 | 0.377604 | 0.284838 | 0.755274 | 0.504812 | 0.272713 | 1.032289 |
| ANCHOR_CONTEXT_C_seed43_fold1 | 10 | 0.409582 | 0.279972 | 0.792464 | 0.487412 | 0.210737 | 1.272251 |
| ANCHOR_CONTEXT_C_seed43_fold2 | 1 | 0.321723 | 0.266998 | 0.608798 | 0.453761 | 0.580576 | 1.036325 |
| ANCHOR_CONTEXT_C_seed43_fold2 | 5 | 0.356316 | 0.254975 | 0.707782 | 0.452453 | 0.271957 | 1.197963 |
| ANCHOR_CONTEXT_C_seed43_fold2 | 10 | 0.414916 | 0.267449 | 0.816566 | 0.495910 | 0.216312 | 1.120485 |
| ANCHOR_CONTEXT_C_seed43_fold3 | 1 | 0.315639 | 0.263739 | 0.618201 | 0.510151 | 0.563807 | 0.803550 |
| ANCHOR_CONTEXT_C_seed43_fold3 | 5 | 0.366417 | 0.259519 | 0.717471 | 0.506992 | 0.277364 | 0.936643 |
| ANCHOR_CONTEXT_C_seed43_fold3 | 10 | 0.392664 | 0.260867 | 0.768530 | 0.515881 | 0.220303 | 1.029154 |
| ANCHOR_CONTEXT_C_seed43_fold4 | 1 | 0.299363 | 0.275118 | 0.582168 | 0.484102 | 0.574611 | 0.986358 |
| ANCHOR_CONTEXT_C_seed43_fold4 | 5 | 0.374107 | 0.307986 | 0.756115 | 0.555370 | 0.272372 | 0.935166 |
| ANCHOR_CONTEXT_C_seed43_fold4 | 10 | 0.379367 | 0.295144 | 0.747470 | 0.517388 | 0.217212 | 1.113818 |
| ANCHOR_seed17_fold0 | 1 | 0.291748 | 0.247237 | 0.532380 | 0.441895 | 0.675837 | 1.114303 |
| ANCHOR_seed17_fold0 | 5 | 0.337511 | 0.254362 | 0.659157 | 0.454955 | 0.398217 | 0.985874 |
| ANCHOR_seed17_fold0 | 10 | 0.344757 | 0.245990 | 0.666509 | 0.445229 | 0.335258 | 1.151108 |
| ANCHOR_seed17_fold1 | 1 | 0.289840 | 0.261350 | 0.551310 | 0.417890 | 0.681129 | 1.116984 |
| ANCHOR_seed17_fold1 | 5 | 0.325542 | 0.268923 | 0.643723 | 0.442262 | 0.396778 | 1.112200 |
| ANCHOR_seed17_fold1 | 10 | 0.350539 | 0.264990 | 0.676074 | 0.424747 | 0.323928 | 1.310393 |
| ANCHOR_seed17_fold2 | 1 | 0.300646 | 0.249682 | 0.567509 | 0.414806 | 0.700585 | 1.072922 |
| ANCHOR_seed17_fold2 | 5 | 0.344138 | 0.256350 | 0.663967 | 0.441551 | 0.405268 | 1.012001 |
| ANCHOR_seed17_fold2 | 10 | 0.345406 | 0.248763 | 0.673352 | 0.429303 | 0.336401 | 1.143939 |
| ANCHOR_seed17_fold3 | 1 | 0.297461 | 0.247588 | 0.566858 | 0.467713 | 0.670600 | 1.012672 |
| ANCHOR_seed17_fold3 | 5 | 0.311682 | 0.235721 | 0.592065 | 0.439055 | 0.395191 | 1.106226 |
| ANCHOR_seed17_fold3 | 10 | 0.361418 | 0.255094 | 0.709156 | 0.466693 | 0.332898 | 0.954591 |
| ANCHOR_seed17_fold4 | 1 | 0.282793 | 0.274482 | 0.555137 | 0.458306 | 0.688541 | 1.027856 |
| ANCHOR_seed17_fold4 | 5 | 0.319910 | 0.278160 | 0.616274 | 0.469392 | 0.403132 | 0.984660 |
| ANCHOR_seed17_fold4 | 10 | 0.355230 | 0.276846 | 0.696055 | 0.489066 | 0.328685 | 0.992279 |
| ANCHOR_seed29_fold0 | 1 | 0.293797 | 0.236816 | 0.543066 | 0.431403 | 0.675980 | 1.085797 |
| ANCHOR_seed29_fold0 | 5 | 0.318410 | 0.243577 | 0.606353 | 0.433838 | 0.398128 | 1.174933 |
| ANCHOR_seed29_fold0 | 10 | 0.345429 | 0.243188 | 0.671042 | 0.449338 | 0.338189 | 1.284980 |
| ANCHOR_seed29_fold1 | 1 | 0.273013 | 0.262200 | 0.523955 | 0.417634 | 0.684250 | 1.281277 |
| ANCHOR_seed29_fold1 | 5 | 0.339561 | 0.276273 | 0.673878 | 0.450125 | 0.391704 | 1.081420 |
| ANCHOR_seed29_fold1 | 10 | 0.349296 | 0.269331 | 0.704243 | 0.446999 | 0.324367 | 1.148142 |
| ANCHOR_seed29_fold2 | 1 | 0.284799 | 0.240948 | 0.535644 | 0.404981 | 0.697394 | 1.272415 |
| ANCHOR_seed29_fold2 | 5 | 0.318360 | 0.237276 | 0.603944 | 0.402349 | 0.401059 | 1.212747 |
| ANCHOR_seed29_fold2 | 10 | 0.364944 | 0.249244 | 0.717392 | 0.454011 | 0.337001 | 1.028276 |
| ANCHOR_seed29_fold3 | 1 | 0.305245 | 0.254331 | 0.597219 | 0.482096 | 0.664195 | 0.851859 |
| ANCHOR_seed29_fold3 | 5 | 0.346002 | 0.252956 | 0.670148 | 0.458158 | 0.397494 | 0.933328 |
| ANCHOR_seed29_fold3 | 10 | 0.338084 | 0.238175 | 0.679739 | 0.446541 | 0.333490 | 1.098832 |
| ANCHOR_seed29_fold4 | 1 | 0.304257 | 0.289949 | 0.583902 | 0.477681 | 0.688756 | 0.818867 |
| ANCHOR_seed29_fold4 | 5 | 0.329465 | 0.280633 | 0.654746 | 0.471858 | 0.401155 | 0.946097 |
| ANCHOR_seed29_fold4 | 10 | 0.353326 | 0.277263 | 0.701436 | 0.494916 | 0.329540 | 1.054852 |
| ANCHOR_seed43_fold0 | 1 | 0.311933 | 0.252606 | 0.598218 | 0.481129 | 0.674740 | 0.850600 |
| ANCHOR_seed43_fold0 | 5 | 0.342909 | 0.253520 | 0.660366 | 0.477404 | 0.400571 | 0.918900 |
| ANCHOR_seed43_fold0 | 10 | 0.359094 | 0.257809 | 0.696866 | 0.472324 | 0.330097 | 1.006312 |
| ANCHOR_seed43_fold1 | 1 | 0.282168 | 0.256699 | 0.532267 | 0.417097 | 0.677989 | 1.132577 |
| ANCHOR_seed43_fold1 | 5 | 0.329848 | 0.271078 | 0.627656 | 0.431212 | 0.395523 | 1.092388 |
| ANCHOR_seed43_fold1 | 10 | 0.356314 | 0.268965 | 0.692936 | 0.427074 | 0.323132 | 1.242220 |
| ANCHOR_seed43_fold2 | 1 | 0.304474 | 0.255636 | 0.550475 | 0.415091 | 0.695579 | 1.096740 |
| ANCHOR_seed43_fold2 | 5 | 0.326108 | 0.239512 | 0.633561 | 0.425880 | 0.402242 | 1.311114 |
| ANCHOR_seed43_fold2 | 10 | 0.363947 | 0.254417 | 0.714057 | 0.452836 | 0.334815 | 1.062361 |
| ANCHOR_seed43_fold3 | 1 | 0.285824 | 0.245185 | 0.545247 | 0.447834 | 0.672905 | 0.958262 |
| ANCHOR_seed43_fold3 | 5 | 0.324091 | 0.240652 | 0.622322 | 0.432253 | 0.399658 | 1.093416 |
| ANCHOR_seed43_fold3 | 10 | 0.354834 | 0.250574 | 0.698323 | 0.461023 | 0.334809 | 0.988199 |
| ANCHOR_seed43_fold4 | 1 | 0.273346 | 0.259298 | 0.512916 | 0.426159 | 0.685322 | 1.097716 |
| ANCHOR_seed43_fold4 | 5 | 0.326039 | 0.275355 | 0.642659 | 0.469018 | 0.402228 | 1.065881 |
| ANCHOR_seed43_fold4 | 10 | 0.343022 | 0.277416 | 0.684392 | 0.479527 | 0.337086 | 1.105468 |


## Source support and paired class changes

I use Kish support to describe how strongly class nuclei concentrate within ROIs, not to estimate independent patients. The recall deltas below are Context-C minus anchor, taken from the paired summary.

| Class | Positive ROIs | Kish ROI support | Recall delta seed17 | Recall delta seed29 | Recall delta seed43 |
| --- | --- | --- | --- | --- | --- |
| tumor | 205 | 152.857059 | 0.084984 | 0.094192 | 0.082154 |
| lymphocyte | 193 | 81.370980 | 0.044125 | 0.024442 | 0.035485 |
| plasma_cell | 56 | 7.195317 | -0.005769 | -0.036538 | 0.025000 |
| histiocyte | 115 | 65.208957 | 0.056920 | 0.025391 | 0.000419 |
| melanophage | 77 | 22.682555 | -0.080576 | -0.076259 | -0.067626 |
| neutrophil | 30 | 4.346116 | 0.010929 | 0.008197 | 0.013661 |
| stroma | 160 | 75.594774 | -0.011670 | -0.010373 | 0.022303 |
| epithelium | 29 | 12.494590 | 0.109905 | 0.111714 | 0.149706 |
| endothelium | 126 | 68.512195 | 0.040684 | 0.007665 | -0.014151 |
| apoptosis | 126 | 26.431590 | 0.045455 | 0.004435 | -0.012195 |


## Limitations and the next justified question

I evaluated Stage-2 classification at ground-truth centroids. Detector false positives, missed nuclei, localization shifts, segmentation errors, and background rejection are therefore outside these results. I would need separate alignment and end-to-end evaluation before using Stage-1 proposals. I verified ROI grouping, but have not established patient, case, or slide independence.

I keep validation reuse and checkpoint selection in mind when interpreting these scores. The paired ROI bootstrap quantifies uncertainty for the observed ROIs and grouping assumption; it cannot correct earlier model selection, dependence beyond ROI, or an unseen domain shift. The seeds describe optimization variation on the same dataset. Likewise, the rare-class safety thresholds are decision rules with finite support, not guarantees for future cases.

I use negative results to narrow the choice of objective, weighting, geometry, and duration for the next experiment. They do not establish that an entire method family is ineffective. I count only completed evidence, excluding optional refits, uncompleted seeds, skipped diagnostics, and missing artifacts. Broader performance claims would still need independent evaluation.
