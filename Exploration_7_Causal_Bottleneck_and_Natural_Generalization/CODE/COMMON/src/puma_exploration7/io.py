from __future__ import annotations
from pathlib import Path
import hashlib, json, os, tempfile
import numpy as np
import pandas as pd
from .constants import CLASSES, PROB_COLUMNS, NUM_CLASSES


def sha256_file(path: str | Path, block=8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(block), b''):
            h.update(b)
    return h.hexdigest()


def stable_json_hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(',', ':'), default=str).encode()
    ).hexdigest()


def atomic_json_dump(path: str | Path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=p.name + '.', dir=p.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x).__name__)


def read_json(path):
    with Path(path).open(encoding='utf-8') as f:
        return json.load(f)


def softmax_np(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def probabilities_to_logits(p, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != NUM_CLASSES:
        raise ValueError('probabilities must be Nx10')
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('invalid probabilities')
    s = p.sum(1, keepdims=True)
    if (s <= 0).any():
        raise ValueError('zero probability row')
    p = p / s
    return np.log(np.clip(p, eps, 1.0))


def load_exploration6_result(path: str | Path) -> pd.DataFrame:
    payload = read_json(path)
    preds = payload.get('predictions')
    if not isinstance(preds, list) or not preds:
        raise ValueError('Exploration-6 result lacks predictions')
    df = pd.DataFrame(preds)
    missing = [c for c in PROB_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f'missing probability columns: {missing}')
    probs = df.loc[:, PROB_COLUMNS].to_numpy(float)
    df['__logits__'] = list(probabilities_to_logits(probs))
    return df


def prediction_frame_to_logits(df: pd.DataFrame) -> np.ndarray:
    if all(c in df.columns for c in PROB_COLUMNS):
        return probabilities_to_logits(df.loc[:, PROB_COLUMNS].to_numpy(float))
    logit_cols = [f'logit_{c}' for c in CLASSES]
    if all(c in df.columns for c in logit_cols):
        return df.loc[:, logit_cols].to_numpy(float)
    raise ValueError('prediction frame needs p_<class> or logit_<class> columns')


def save_prediction_csv(path, meta: pd.DataFrame, logits):
    p = softmax_np(logits)
    out = meta.copy()
    pred = p.argmax(1)
    out['pred_class_id'] = pred
    out['pred_class_name'] = [CLASSES[i] for i in pred]
    out['score'] = p.max(1)
    for i, c in enumerate(CLASSES):
        out[f'p_{c}'] = p[:, i]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
