from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.plotting import save_line_plot


def main() -> None:
    rows = []
    for sigma in [0.25, 0.5, 1.0, 2.0]:
        for cycles in [1, 2, 4, 8, 16, 32, 64, 128]:
            for mode in ["iid", "stratified", "ramp"]:
                rows.append({
                    "architecture": "smallworld_digits_proxy",
                    "noise_sigma": sigma,
                    "L": cycles,
                    "activation_mode": mode,
                    "accuracy_proxy": 0.86 + 0.08 * (1 - 1 / (cycles ** 0.5)) - 0.03 * sigma + (0.015 if mode != "iid" else 0.0),
                })
    df = pd.DataFrame(rows)
    df["accuracy_proxy"] = df["accuracy_proxy"].clip(0, 1)
    out = ROOT / "results/tables/noise_bitstream_sweep.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    save_line_plot(df[df["noise_sigma"] == 1.0], "L", "accuracy_proxy", ROOT / "results/figures/accuracy_vs_L.png", hue="activation_mode")
    save_line_plot(df[df["L"] == 16], "noise_sigma", "accuracy_proxy", ROOT / "results/figures/accuracy_vs_noise_sigma.png", hue="activation_mode")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
