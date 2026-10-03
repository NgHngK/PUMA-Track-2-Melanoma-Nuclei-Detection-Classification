# Exploration 9

Context-C passed the exploratory natural-transfer gates across three seeds and five ROI folds. Mean paired ROI-F1 gain was 0.019563479669336675, with interval [0.016548355719802475, 0.022665436440666983].

Read the [Detailed report](REPORTS/REPORT_EXPLORATION_9.md).

Run [Preprocessing](00_Preprocess_Stage2.ipynb) before [Training](01_Train_Stage2.ipynb). Configure input and output locations in the notebooks for your environment.

## Local readiness

The two notebooks target Colab and require the external dataset and local UNI2-h weights. `STAGE1_REFERENCE/` is historical reference code and is not imported by Stage-2 training. Resource monitoring can also import on Windows; unavailable Unix peak-RSS statistics are reported as `null`. Saved training metadata can refer to checkpoints that were pruned under the archive retention policy.
