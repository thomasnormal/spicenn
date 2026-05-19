from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.datasets import digits_loaders, set_seed
from sim.energy_model import estimate_inference_energy
from sim.local_layers import DenseStochasticMLP, SmallWorldMLP, StochasticActivation, UnsharedLocalNet
from sim.plotting import save_line_plot
from sim.topology import grid_coordinates, local_grid_edges, small_world_edges, wire_metrics
from sim.train_backprop import train_classifier


def run(epochs: int = 3, seed: int = 0) -> pd.DataFrame:
    set_seed(seed)
    train_loader, test_loader = digits_loaders(seed=seed)
    models = {
        "dense_mlp": DenseStochasticMLP(64, hidden=(64,), activation=StochasticActivation(sigma=0.75)),
        "unshared_local": UnsharedLocalNet(image_size=8, channels=(4, 8)),
        "smallworld": SmallWorldMLP(64, hidden=96, shortcut_fraction=0.02),
    }
    rows = []
    for name, model in models.items():
        result = train_classifier(model, train_loader, test_loader, epochs=epochs, lr=1e-3)
        if name == "dense_mlp":
            coords = grid_coordinates(16, 8)
            wire = wire_metrics(coords, [(i, 64 + j) for i in range(64) for j in range(64)])
        elif name == "smallworld":
            coords = grid_coordinates(8, 8)
            wire = wire_metrics(coords, small_world_edges(8, 8, shortcut_fraction=0.02, seed=seed))
        else:
            coords = grid_coordinates(8, 8)
            wire = wire_metrics(coords, local_grid_edges(8, 8, radius=1))
        energy = estimate_inference_energy(
            synapse_count=wire.synapse_count,
            active_synapse_count=wire.synapse_count,
            total_wire_length_um=wire.total_wire_length_um,
            neuron_count=sum(p.numel() for p in model.parameters()) // 10,
            stochastic_cycles=8,
        )
        for epoch, (loss, acc) in enumerate(zip(result.train_loss, result.test_accuracy), start=1):
            rows.append(
                {
                    "seed": seed,
                    "architecture": name,
                    "epoch": epoch,
                    "train_loss": loss,
                    "accuracy": acc,
                    "synapse_count": wire.synapse_count,
                    "average_wire_length_um": wire.average_wire_length_um,
                    "max_wire_length_um": wire.max_wire_length_um,
                    "total_wire_length_um": wire.total_wire_length_um,
                    "inference_j": energy.inference_j,
                }
            )
    df = pd.DataFrame(rows)
    out_csv = ROOT / "results/tables/smoke_digits.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    save_line_plot(df, "epoch", "accuracy", ROOT / "results/figures/smoke_digits_accuracy.png", hue="architecture")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    df = run(args.epochs, args.seed)
    print(df.groupby("architecture")["accuracy"].last().to_string())


if __name__ == "__main__":
    main()
