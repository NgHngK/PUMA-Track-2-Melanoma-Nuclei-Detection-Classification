from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics import classification_metrics
from .vendor.puma_v17_evaluator import evaluate_rois


def load_prediction_npz(path: str | Path, expected_freeze_sha256: str | None = None) -> dict:
    x = np.load(path, allow_pickle=False)
    required = {'uids', 'labels', 'logits'}
    if not required.issubset(x.files):
        raise ValueError(f'prediction NPZ missing {sorted(required - set(x.files))}')
    uids = np.asarray(x['uids']).astype(str)
    labels = np.asarray(x['labels'], int)
    logits = np.asarray(x['logits'], float)
    if (
        len(uids) != len(labels)
        or logits.shape != (len(labels), 10)
        or len(set(uids.tolist())) != len(uids)
    ):
        raise ValueError('prediction NPZ shape/UID contract failure')
    freeze = (
        str(np.asarray(x['pipeline_freeze_sha256']).item())
        if 'pipeline_freeze_sha256' in x.files
        else None
    )
    if expected_freeze_sha256 is not None and freeze != expected_freeze_sha256:
        raise ValueError('prediction file is not bound to the requested complete pipeline freeze')
    return {'uids': uids, 'labels': labels, 'logits': logits, 'pipeline_freeze_sha256': freeze}


def align_predictions(
    pred: dict, role_df: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    role = role_df.copy()
    ref = role.uid.astype(str).tolist()
    pos = {u: i for i, u in enumerate(pred['uids'])}
    if set(pos) != set(ref):
        raise ValueError('prediction UID set does not exactly match protected role manifest')
    order = np.asarray([pos[u] for u in ref], int)
    y = pred['labels'][order]
    z = pred['logits'][order]
    if not np.array_equal(y, role.label.to_numpy(int)):
        raise ValueError('prediction labels do not align with protected manifest labels')
    return role.reset_index(drop=True), y, z


def _softmax(z):
    z = np.asarray(z, float)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def conditional_v17_metrics(role_df: pd.DataFrame, logits: np.ndarray) -> dict | None:
    """Exact V17 matcher on conditional rows when GT coordinates are available.

    GT-centred rows use x/y as GT and prediction coordinates. Stage-1-centred rows
    require explicit gt_x/gt_y columns; otherwise V17 geometry is not estimable here.
    """
    if len(role_df) != len(logits):
        raise ValueError('V17 row/logit length mismatch')
    if {'gt_x', 'gt_y'}.issubset(role_df.columns):
        gx, gy = role_df.gt_x.to_numpy(float), role_df.gt_y.to_numpy(float)
        scope = 'CONDITIONAL_STAGE1_MATCHED_ROWS'
    elif (role_df.coordinate_source.astype(str) == 'gt').all():
        gx, gy = role_df.x.to_numpy(float), role_df.y.to_numpy(float)
        scope = 'GT_CENTERED_CONDITIONAL'
    else:
        return None
    probs = _softmax(logits)
    pred_cls = probs.argmax(1)
    conf = probs.max(1)
    gt_by_roi, pred_by_roi = {}, {}
    for i, r in role_df.reset_index(drop=True).iterrows():
        roi = str(r.roi)
        gt_by_roi.setdefault(roi, []).append(
            {'x': float(gx[i]), 'y': float(gy[i]), 'class_id': int(r.label)}
        )
        pred_by_roi.setdefault(roi, []).append(
            {
                'x': float(r.x),
                'y': float(r.y),
                'class_id': int(pred_cls[i]),
                'score': float(conf[i]),
            }
        )
    out = evaluate_rois(
        gt_by_roi, pred_by_roi, roi_order=list(dict.fromkeys(role_df.roi.astype(str)))
    )
    out['scope'] = scope
    return out


def protected_metrics(role_df: pd.DataFrame, y: np.ndarray, z: np.ndarray) -> dict:
    semantic = classification_metrics(y, z)
    v17 = conditional_v17_metrics(role_df, z)
    if v17 is not None:
        primary_name = 'puma_v17_roi_fixed10_macro_f1'
        primary_value = float(v17['roi_fixed10_macro_f1'])
    else:
        primary_name = 'semantic_fixed10_macro_f1_fallback'
        primary_value = float(semantic['macro_f1_fixed10'])
    return {
        'primary_name': primary_name,
        'primary_value': primary_value,
        'semantic': semantic,
        'v17': v17,
    }


def group_primary_scores(
    role_df: pd.DataFrame, logits: np.ndarray, group_col: str
) -> dict[str, float]:
    scores = {}
    for group in role_df[group_col].astype(str).unique():
        mask = role_df[group_col].astype(str).to_numpy() == group
        sub = role_df.loc[mask].reset_index(drop=True)
        y = sub.label.to_numpy(int)
        scores[str(group)] = protected_metrics(sub, y, np.asarray(logits)[mask])['primary_value']
    return scores
