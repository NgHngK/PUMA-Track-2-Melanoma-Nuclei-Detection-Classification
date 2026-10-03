from __future__ import annotations

from collections import defaultdict

import numpy as np


def grouped_balanced_folds(
    class_counts: np.ndarray,
    case_ids: list[str],
    number_of_folds: int,
    seed: int,
    restarts: int = 96,
) -> np.ndarray:
    class_counts = np.asarray(class_counts)
    if class_counts.ndim != 2:
        raise ValueError("class_counts must be [N,C]")
    if class_counts.dtype.kind not in "iu" or np.any(class_counts < 0):
        raise ValueError("class_counts must contain non-negative integers")
    class_counts = class_counts.astype(np.int64, copy=False)
    if len(case_ids) != len(class_counts):
        raise ValueError("case_ids length mismatch")
    if len(class_counts) == 0 or class_counts.shape[1] == 0:
        raise ValueError("class_counts must contain at least one sample and one class")
    if not isinstance(number_of_folds, int) or isinstance(number_of_folds, bool) or number_of_folds < 2:
        raise ValueError("number_of_folds must be an integer >= 2")
    if not isinstance(restarts, int) or isinstance(restarts, bool) or restarts < 1:
        raise ValueError("restarts must be an integer >= 1")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be an integer")
    if any(not str(case_id).strip() for case_id in case_ids):
        raise ValueError("case_ids cannot contain empty identifiers")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, case_id in enumerate(case_ids):
        groups[str(case_id)].append(index)
    group_keys = sorted(groups)
    group_counts = np.stack([class_counts[groups[key]].sum(axis=0) for key in group_keys], axis=0)
    group_sizes = np.asarray([len(groups[key]) for key in group_keys], dtype=np.int64)
    if len(group_keys) < number_of_folds:
        raise ValueError("Fewer independent cases than folds")

    totals = group_counts.sum(axis=0).astype(np.float64)
    target_class = totals / number_of_folds
    target_size = len(class_counts) / number_of_folds
    rarity = np.where(totals > 0, 1.0 / np.sqrt(np.maximum(totals, 1.0)), 0.0)
    priority = (group_counts > 0).astype(np.float64) @ rarity + 0.01 * np.log1p(group_counts.sum(axis=1))

    best_assignment = None
    best_objective = float("inf")
    for restart in range(int(restarts)):
        rng = np.random.default_rng(seed + 10007 * restart)
        order = np.argsort(-(priority + rng.uniform(0.0, 1.0e-8, len(priority))))
        assignment = np.full(len(group_keys), -1, dtype=np.int16)
        fold_counts = np.zeros((number_of_folds, class_counts.shape[1]), dtype=np.float64)
        fold_sizes = np.zeros(number_of_folds, dtype=np.float64)
        fold_presence = np.zeros_like(fold_counts, dtype=np.int64)
        for rank, group_index in enumerate(order):
            if rank < number_of_folds:
                fold = rank
            else:
                costs = []
                for fold in range(number_of_folds):
                    after_counts = fold_counts[fold] + group_counts[group_index]
                    class_error = np.mean(((after_counts - target_class) / np.maximum(target_class, 1.0)) ** 2)
                    after_size = fold_sizes[fold] + group_sizes[group_index]
                    size_error = ((after_size - target_size) / max(target_size, 1.0)) ** 2
                    missing_before = fold_presence[fold] == 0
                    newly_covered = missing_before & (group_counts[group_index] > 0)
                    coverage_reward = float(np.sum(rarity[newly_covered]))
                    costs.append(class_error + 0.6 * size_error - 0.25 * coverage_reward)
                minimum = min(costs)
                choices = np.flatnonzero(np.isclose(costs, minimum, atol=1e-12, rtol=0))
                fold = int(rng.choice(choices))
            assignment[group_index] = fold
            fold_counts[fold] += group_counts[group_index]
            fold_sizes[fold] += group_sizes[group_index]
            fold_presence[fold] += group_counts[group_index] > 0
        objective = float(
            np.mean(((fold_counts - target_class) / np.maximum(target_class, 1.0)) ** 2)
            + 0.6 * np.mean(((fold_sizes - target_size) / max(target_size, 1.0)) ** 2)
        )
        if objective < best_objective:
            best_objective = objective
            best_assignment = assignment.copy()
    if best_assignment is None:
        raise RuntimeError("Could not build fold assignment")
    roi_folds = np.empty(len(class_counts), dtype=np.int16)
    for group_index, key in enumerate(group_keys):
        roi_folds[groups[key]] = best_assignment[group_index]
    return roi_folds

