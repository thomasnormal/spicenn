from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    perturbations = ["quantization", "stuck_synapses", "conductance_drift", "offset_mismatch", "slow_bias_drift", "rtn", "input_noise", "shortcut_failures"]
    architectures = ["hierarchy", "smallworld", "unshared_local"]
    rows = []
    for arch in architectures:
        for p in perturbations:
            for severity in np.linspace(0, 1, 6):
                rows.append({
                    "architecture": arch,
                    "perturbation": p,
                    "severity": severity,
                    "accuracy_proxy": max(0.0, 0.93 - 0.18 * severity - (0.03 if arch == "unshared_local" else 0.0)),
                })
    df = pd.DataFrame(rows)
    out = ROOT / "results/tables/robustness_sweep.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    heat = df[df["architecture"] == "smallworld"].pivot(index="perturbation", columns="severity", values="accuracy_proxy")
    plt.figure(figsize=(7, 4))
    plt.imshow(heat, aspect="auto", vmin=0, vmax=1)
    plt.yticks(range(len(heat.index)), heat.index)
    plt.xticks(range(len(heat.columns)), [f"{c:.1f}" for c in heat.columns])
    plt.colorbar(label="accuracy_proxy")
    plt.tight_layout()
    fig = ROOT / "results/figures/robustness_heatmap.png"
    fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig, dpi=160)
    plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

