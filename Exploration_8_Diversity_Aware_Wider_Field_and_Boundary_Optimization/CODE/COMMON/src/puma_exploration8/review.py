from __future__ import annotations
import pandas as pd

REQUIRED = ('uid', 'reviewer_label', 'uncertain', 'notes')


def read_review(path):
    df = pd.read_csv(path)
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        raise ValueError(f'review file missing {miss}')
    return df
