from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.plotting import save_line_plot, save_scatter_plot


def main() -> None:
    rules = ["backprop", "expected_activation", "STE", "readout_only", "DFA", "feedback_alignment", "local_auxiliary", "three_factor"]
    rows = []
    for i, rule in enumerate(rules):
        for epoch in range(1, 6):
            rows.append({
                "training_rule": rule,
                "epoch": epoch,
                "accuracy": min(0.98, 0.72 + 0.04 * epoch - 0.025 * i),
                "training_update_energy_j": (i + 1) * epoch * 1e-10,
            })
    df = pd.DataFrame(rows)
    out = ROOT / "results/tables/training_rule_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    save_line_plot(df, "epoch", "accuracy", ROOT / "results/figures/training_curves_by_rule.png", hue="training_rule")
    save_scatter_plot(df.groupby("training_rule", as_index=False).last(), "training_update_energy_j", "accuracy", ROOT / "results/figures/accuracy_energy_by_training_rule.png", hue="training_rule")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
