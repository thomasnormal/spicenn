from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.arch_search import pareto_front
from sim.datasets import digits_loaders, mnist_loaders, set_seed
from sim.local_layers import DenseStochasticMLP, HierarchicalLocalNet, RelayChainNet, SharedConvNet, SmallWorldMLP, UnsharedLocalNet
from sim.plotting import save_scatter_plot
from sim.topology import dense_edges, grid_coordinates, local_grid_edges, small_world_edges, wire_metrics
from sim.train_backprop import train_classifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Use sklearn digits instead of MNIST.")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    if args.quick:
        train_loader, test_loader = digits_loaders(seed=args.seed)
        input_dim, image_size = 64, 8
    else:
        train_loader, test_loader = mnist_loaders(download=args.download)
        input_dim, image_size = 784, 28
    models = {
        "dense_mlp": DenseStochasticMLP(input_dim, hidden=(128,)),
        "shared_cnn": SharedConvNet(),
        "unshared_local": UnsharedLocalNet(image_size=image_size),
        "hierarchical_local": HierarchicalLocalNet(image_size=image_size),
        "smallworld": SmallWorldMLP(input_dim),
        "relay_chain": RelayChainNet(input_dim, chain_length=3),
    }
    rows = []
    for name, model in models.items():
        res = train_classifier(model, train_loader, test_loader, epochs=args.epochs)
        coords = grid_coordinates(image_size, image_size)
        if name == "dense_mlp":
            coords2 = grid_coordinates(max(image_size, 32), max(image_size, 32))
            wire = wire_metrics(coords2, dense_edges(min(input_dim, len(coords2) // 2), min(128, len(coords2) // 2), offset_dst=min(input_dim, len(coords2) // 2)))
        elif name == "smallworld":
            wire = wire_metrics(coords, small_world_edges(image_size, image_size, shortcut_fraction=0.02, seed=args.seed))
        else:
            wire = wire_metrics(coords, local_grid_edges(image_size, image_size, radius=1))
        rows.append({
            "architecture": name,
            "accuracy": res.test_accuracy[-1],
            "synapse_count": wire.synapse_count,
            "total_wire_length_um": wire.total_wire_length_um,
            "max_wire_length_um": wire.max_wire_length_um,
            "inference_energy_proxy": wire.total_wire_length_um * 1e-16 + wire.synapse_count * 1e-15,
        })
    df = pd.DataFrame(rows)
    (ROOT / "results/tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / "results/tables/mnist_architecture_comparison.csv", index=False)
    save_scatter_plot(df, "inference_energy_proxy", "accuracy", ROOT / "results/figures/accuracy_vs_energy.png", hue="architecture")
    save_scatter_plot(df, "total_wire_length_um", "accuracy", ROOT / "results/figures/accuracy_vs_total_wire_length.png", hue="architecture")
    pf = pareto_front(df, maximize=("accuracy",), minimize=("inference_energy_proxy", "total_wire_length_um"))
    save_scatter_plot(pf, "inference_energy_proxy", "accuracy", ROOT / "results/figures/pareto_accuracy_energy.png", hue="architecture")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
