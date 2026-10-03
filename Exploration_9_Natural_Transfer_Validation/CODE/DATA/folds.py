from __future__ import annotations
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from CODE.COMMON.constants import NUM_CLASSES, PREFERRED_FOLDS, MIN_FOLDS
from CODE.COMMON.exceptions import FoldLeakageError


def positive_group_counts(labels, groups):
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    return np.array([len(set(groups[labels == c].tolist())) for c in range(NUM_CLASSES)], dtype=int)


def choose_fold_count(
    labels, groups, preferred: int = PREFERRED_FOLDS, minimum: int = MIN_FOLDS
) -> int:
    counts = positive_group_counts(labels, groups)
    positive = counts[counts > 0]
    if len(positive) < NUM_CLASSES:
        return 0
    k = min(preferred, int(positive.min()), len(set(groups)))
    return k if k >= minimum else 0


def make_group_folds(labels, groups, seed: int = 17, n_splits: int | None = None):
    labels = np.asarray(labels, dtype=int)
    groups = np.asarray(groups)
    if n_splits is None:
        n_splits = choose_fold_count(labels, groups)
    if n_splits < 3:
        raise FoldLeakageError('Fewer than 3 feasible grouped folds')
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.full(len(labels), -1, dtype=int)
    for k, (_, held) in enumerate(sgkf.split(np.zeros(len(labels)), labels, groups)):
        fold[held] = k
    if (fold < 0).any():
        raise FoldLeakageError('Some rows not assigned')
    for k in range(n_splits):
        fit_groups = set(groups[fold != k])
        held_groups = set(groups[fold == k])
        if fit_groups & held_groups:
            raise FoldLeakageError(f'Group leakage fold {k}')
    return fold
