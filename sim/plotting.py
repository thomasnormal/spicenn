from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_line_plot(df: pd.DataFrame, x: str, y: str, path: str | Path, hue: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    if hue and hue in df:
        for label, group in df.groupby(hue):
            plt.plot(group[x], group[y], marker="o", label=str(label))
        plt.legend()
    else:
        plt.plot(df[x], df[y], marker="o")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_scatter_plot(df: pd.DataFrame, x: str, y: str, path: str | Path, hue: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    if hue and hue in df:
        for label, group in df.groupby(hue):
            plt.scatter(group[x], group[y], label=str(label))
        plt.legend()
    else:
        plt.scatter(df[x], df[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

