from __future__ import annotations

import pandas as pd


def pareto_front(df: pd.DataFrame, maximize: tuple[str, ...], minimize: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for i, row in df.iterrows():
        dominated = False
        for j, other in df.iterrows():
            if i == j:
                continue
            ge = all(other[m] >= row[m] for m in maximize)
            le = all(other[m] <= row[m] for m in minimize)
            better = any(other[m] > row[m] for m in maximize) or any(other[m] < row[m] for m in minimize)
            if ge and le and better:
                dominated = True
                break
        if not dominated:
            rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)

