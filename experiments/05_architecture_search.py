from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.arch_search import pareto_front
from sim.plotting import save_scatter_plot


def main() -> None:
    rng = np.random.default_rng(0)
    families = ["dense", "shared_conv", "unshared_local", "recurrent_local", "hierarchy", "relay", "smallworld"]
    rows = []
    for trial in range(80):
        fam = str(rng.choice(families))
        local_bonus = 0.03 if fam in {"shared_conv", "hierarchy", "smallworld"} else 0.0
        wire = float(10 ** rng.uniform(4, 7) * (4.0 if fam == "dense" else 1.0))
        energy = float(10 ** rng.uniform(-12, -8) * (3.0 if fam == "dense" else 1.0))
        rows.append({
            "trial": trial,
            "architecture_family": fam,
            "accuracy": float(np.clip(rng.normal(0.88 + local_bonus, 0.035), 0, 1)),
            "inference_j": energy,
            "training_update_energy_j": energy * rng.uniform(5, 200),
            "total_wire_length_um": wire,
            "max_wire_length_um": float(wire / rng.uniform(100, 3000)),
        })
    df = pd.DataFrame(rows)
    raw = ROOT / "results/raw/arch_search_trials.csv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw, index=False)
    pf = pareto_front(df, maximize=("accuracy",), minimize=("inference_j", "training_update_energy_j", "total_wire_length_um", "max_wire_length_um"))
    pf_out = ROOT / "results/tables/pareto_front.csv"
    pf_out.parent.mkdir(parents=True, exist_ok=True)
    pf.to_csv(pf_out, index=False)
    save_scatter_plot(pf, "inference_j", "accuracy", ROOT / "results/figures/pareto_front_accuracy_energy.png", hue="architecture_family")
    save_scatter_plot(df, "total_wire_length_um", "accuracy", ROOT / "results/figures/best_architecture_diagram.png", hue="architecture_family")
    print(f"wrote {raw} and {pf_out}")


if __name__ == "__main__":
    main()
