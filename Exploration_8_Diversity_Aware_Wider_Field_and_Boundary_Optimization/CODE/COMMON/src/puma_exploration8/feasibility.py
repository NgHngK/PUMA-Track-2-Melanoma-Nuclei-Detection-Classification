from __future__ import annotations
import numpy as np, pandas as pd
from .constants import CLASSES
from .manifest import strongest_group_column


def kish_effective_count(counts) -> float:
    a = np.asarray(counts, dtype=float)
    s = a.sum()
    return float(s * s / (a @ a)) if s > 0 else 0.0


def support_table(df: pd.DataFrame, role: str) -> pd.DataFrame:
    g = strongest_group_column(df)
    x = df[df.role == role]
    out = []
    for c, name in enumerate(CLASSES):
        z = x[x.label == c]
        counts = z.groupby(g).size().to_numpy()
        ng = len(counts)
        n = len(z)
        if ng == 0:
            status = 'NOT_ESTIMABLE'
        elif ng < 3:
            status = 'UNDERPOWERED_FOR_CLASS_SPECIFIC_CLAIM'
        elif ng < 6:
            status = 'ESTIMABLE_WITH_LIMITED_SAFETY_POWER'
        else:
            status = 'ESTIMABLE'
        out.append(
            dict(
                class_id=c,
                class_name=name,
                raw_n=n,
                unique_groups=ng,
                kish_effective_groups=kish_effective_count(counts),
                largest_group_share=float(counts.max() / n) if n else 0.0,
                median_n_per_group=float(np.median(counts)) if ng else 0.0,
                max_n_per_group=int(counts.max()) if ng else 0,
                status=status,
            )
        )
    return pd.DataFrame(out)
