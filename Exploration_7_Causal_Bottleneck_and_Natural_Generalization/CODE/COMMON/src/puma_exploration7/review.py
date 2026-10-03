from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
from .io import softmax_np


def build_blinded_review(frame, labels, logits, out_dir, pairs=None, per_pair=30, seed=17):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = softmax_np(logits)
    pred = p.argmax(1)
    rng = np.random.default_rng(seed)
    selected = []
    pairs = pairs or []
    for a, b in pairs:
        pool = np.where(((labels == a) & (pred == b)) | ((labels == b) & (pred == a)))[0]
        if len(pool):
            selected.extend(rng.choice(pool, size=min(per_pair, len(pool)), replace=False).tolist())
    # add high-confidence wrong and low-margin wrong strata
    wrong = np.where(pred != labels)[0]
    conf = p.max(1)
    order = np.argsort(-p, axis=1)
    margin = p[np.arange(len(p)), order[:, 0]] - p[np.arange(len(p)), order[:, 1]]
    for pool in [
        wrong[np.argsort(-conf[wrong])[: min(50, len(wrong))]],
        wrong[np.argsort(margin[wrong])[: min(50, len(wrong))]],
    ]:
        selected.extend(pool.tolist())
    selected = sorted(set(selected))
    base = frame.iloc[selected].copy()
    base['review_id'] = [f'R{i:05d}' for i in range(len(base))]
    blind_cols = ['review_id'] + [c for c in ['image', 'x', 'y', 'roi', 'uid'] if c in base.columns]
    blind = base[blind_cols].copy()
    blind['reviewed_label'] = ''
    blind['uncertain'] = ''
    blind['notes'] = ''
    key = base[['review_id', 'uid']].copy()
    key['original_label'] = np.asarray(labels)[selected]
    key['model_pred'] = pred[selected]
    key['model_conf'] = conf[selected]
    key['model_margin'] = margin[selected]
    crops = out / 'crops'
    crops.mkdir(exist_ok=True)
    if {'image', 'x', 'y'}.issubset(base.columns):
        for _, r in base.iterrows():
            try:
                with Image.open(str(r.image)) as im:
                    im = im.convert('RGB')
                    cx = float(r.x)
                    cy = float(r.y)
                    half = 64
                    box = (
                        max(0, int(cx - half)),
                        max(0, int(cy - half)),
                        min(im.width, int(cx + half)),
                        min(im.height, int(cy + half)),
                    )
                    im.crop(box).save(crops / f"{r.review_id}.png")
            except Exception:
                pass
    blind.to_csv(out / 'review_blinded.csv', index=False)
    key.to_csv(out / 'review_key.csv', index=False)
    return blind, key
