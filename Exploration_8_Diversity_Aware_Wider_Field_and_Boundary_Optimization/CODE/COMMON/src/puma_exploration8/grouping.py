from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from .manifest import strongest_group_column
from .config import sha256_json


def _payload_hash(payload: dict) -> str:
    return sha256_json({k: v for k, v in payload.items() if k != 'sha256'})


def choose_grouped_k(df: pd.DataFrame, max_k: int = 5, min_k: int = 2) -> int:
    g = strongest_group_column(df)
    train = df[df.role == 'TRAIN']
    if train.empty:
        raise ValueError('TRAIN empty')
    class_groups = train.groupby('label')[g].nunique()
    if class_groups.empty:
        raise ValueError('no classes')
    possible = min(max_k, int(class_groups.min()), int(train[g].nunique()))
    return max(min_k, possible) if possible >= min_k else 0


def build_train_folds(df: pd.DataFrame, seed: int = 17, max_k: int = 5) -> dict:
    g = strongest_group_column(df)
    tr = df[df.role == 'TRAIN'].copy()
    k = choose_grouped_k(df, max_k)
    if k < 2:
        raise ValueError('insufficient independent groups for grouped CV')
    y = tr.label.to_numpy()
    groups = tr[g].astype(str).to_numpy()
    fold = np.full(len(tr), -1)
    try:
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        for j, (_, va) in enumerate(splitter.split(np.zeros(len(tr)), y, groups)):
            fold[va] = j
    except ValueError:
        splitter = GroupKFold(n_splits=k)
        for j, (_, va) in enumerate(splitter.split(np.zeros(len(tr)), y, groups)):
            fold[va] = j
    if (fold < 0).any():
        raise RuntimeError('unassigned fold')
    payload = {
        'role': 'TRAIN',
        'group_column': g,
        'seed': int(seed),
        'k': int(k),
        'assignments': [{'uid': u, 'fold': int(f)} for u, f in zip(tr.uid.astype(str), fold)],
    }
    payload['sha256'] = _payload_hash(payload)
    return payload


def build_cal_folds(df: pd.DataFrame, max_k: int = 5) -> dict:
    g = strongest_group_column(df)
    cal = df[df.role == 'CAL'].copy()
    ng = cal[g].nunique()
    k = min(max_k, ng)
    if k < 2:
        raise ValueError('CAL needs >=2 groups for cross-fitting')
    splitter = GroupKFold(n_splits=k)
    fold = np.full(len(cal), -1)
    for j, (_, va) in enumerate(splitter.split(np.zeros(len(cal)), cal.label, cal[g])):
        fold[va] = j
    payload = {
        'role': 'CAL',
        'group_column': g,
        'k': int(k),
        'assignments': [{'uid': u, 'fold': int(f)} for u, f in zip(cal.uid.astype(str), fold)],
    }
    payload['sha256'] = _payload_hash(payload)
    return payload


def save_fold_payload(payload: dict, path: str | Path) -> None:
    if payload.get('sha256') != _payload_hash(payload):
        raise ValueError('fold payload hash invalid before save')
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_fold_payload(path: str | Path, df: pd.DataFrame, role: str) -> dict:
    """Load a frozen fold artifact and fail closed on any manifest/group drift."""
    payload = json.loads(Path(path).read_text())
    if payload.get('sha256') != _payload_hash(payload):
        raise ValueError('frozen fold payload hash mismatch')
    if payload.get('role') != role:
        raise ValueError(f'fold role mismatch: expected {role}')
    g = strongest_group_column(df)
    if payload.get('group_column') != g:
        raise ValueError('fold grouping level differs from current strongest verified grouping')
    sub = df[df.role == role].copy()
    expected = set(sub.uid.astype(str))
    ass = payload.get('assignments', [])
    got = [str(x['uid']) for x in ass]
    if len(got) != len(set(got)) or set(got) != expected:
        raise ValueError('fold UID set does not exactly match role manifest')
    fmap = {str(x['uid']): int(x['fold']) for x in ass}
    k = int(payload.get('k', 0))
    if k < 2 or any(v < 0 or v >= k for v in fmap.values()):
        raise ValueError('invalid fold indices')
    # Put each biological group in one held-out fold only.
    tmp = sub[['uid', g]].copy()
    tmp['fold'] = tmp.uid.astype(str).map(fmap)
    if (tmp.groupby(g).fold.nunique() != 1).any():
        raise ValueError('biological group split across frozen folds')
    return payload


def fold_vector(df: pd.DataFrame, payload: dict, role: str = 'TRAIN') -> np.ndarray:
    sub = df[df.role == role]
    fmap = {str(x['uid']): int(x['fold']) for x in payload['assignments']}
    return np.asarray([fmap[str(u)] for u in sub.uid], dtype=int)
