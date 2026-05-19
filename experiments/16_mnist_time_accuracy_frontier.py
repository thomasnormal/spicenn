from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.arch_search import pareto_front
from sim.datasets import mnist_loaders, set_seed
from sim.hardware_metrics import local_evidence_net_layers, summarize_layers
from sim.local_layers import HardwareLocalEvidenceNet, SharedConvNet
from sim.plotting import save_scatter_plot
from sim.train_backprop import train_classifier


def default_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def local_energy(channels: tuple[int, int, int], comparator_cycles: int = 4) -> dict:
    layers = local_evidence_net_layers(channels=channels)
    return summarize_layers(
        layers,
        stochastic_cycles=1,
        wire_read_cycles=1,
        comparator_cycles=comparator_cycles,
    )


def shared_conv_proxy(channels: tuple[int, int]) -> dict:
    # Same local wire geometry as two local convolution stages plus 1x1 readout,
    # but fewer channels and shared weights. This is a proxy, not layout.
    layers = local_evidence_net_layers(channels=(channels[0], channels[1], channels[1]))
    rec = summarize_layers(layers[:2] + layers[-1:], stochastic_cycles=1, wire_read_cycles=1, comparator_cycles=1)
    rec["hardware_parameter_count"] = channels[0] * 1 * 3 * 3 + channels[1] * channels[0] * 3 * 3 + 10 * channels[1]
    return rec


def make_model(family: str, channels: str):
    nums = tuple(int(v) for v in channels.split(","))
    if family == "hardware_local":
        if len(nums) != 3:
            raise ValueError("hardware_local channels must be a,b,c")
        return HardwareLocalEvidenceNet(
            channels=nums,
            activation_mode="quantized_relu6",
            activation_bits=4,
            activation_sigma=0.5,
            coord_channels=True,
        ), local_energy(nums), nums
    if family == "shared_conv":
        if len(nums) != 2:
            raise ValueError("shared_conv channels must be a,b")
        return SharedConvNet(channels=nums), shared_conv_proxy(nums), nums
    raise ValueError(f"unknown family {family}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-limit", type=int, default=10000)
    ap.add_argument("--test-limit", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--configs",
        default=(
            "hardware_local:16,32,48:1;"
            "hardware_local:16,32,48:2;"
            "hardware_local:24,48,64:2;"
            "hardware_local:32,64,96:2;"
            "hardware_local:32,64,96:3;"
            "hardware_local:48,96,128:2;"
            "shared_conv:16,32:2;"
            "shared_conv:32,64:2;"
            "shared_conv:32,64:3"
        ),
    )
    args = ap.parse_args()
    set_seed(args.seed)
    train_loader, test_loader = mnist_loaders(
        batch_size=args.batch_size,
        flatten=False,
        normalize=True,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        seed=args.seed,
    )

    rows = []
    for spec in [s for s in args.configs.split(";") if s.strip()]:
        family, channels, epochs_s = spec.split(":")
        epochs = int(epochs_s)
        set_seed(args.seed)
        model, hw, parsed_channels = make_model(family, channels)
        t0 = time.perf_counter()
        result = train_classifier(
            model,
            train_loader,
            test_loader,
            epochs=epochs,
            lr=2e-3,
            weight_decay=1e-4,
            device=args.device,
        )
        wall = time.perf_counter() - t0
        row = {
            "family": family,
            "channels": channels,
            "epochs": epochs,
            "train_limit": args.train_limit,
            "test_limit": args.test_limit,
            "device": args.device,
            "wall_time_s": wall,
            "accuracy": result.test_accuracy[-1],
            "best_epoch_accuracy": result.best_accuracy,
            "train_loss_final": result.train_loss[-1],
            "pytorch_parameter_count": count_params(model),
            "accuracy_per_second": result.test_accuracy[-1] / max(wall, 1e-9),
            **hw,
        }
        rows.append(row)
        print(json.dumps(row, indent=2))

    df = pd.DataFrame(rows)
    raw = ROOT / "results/raw/mnist_time_accuracy_frontier_trials.csv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw, index=False)
    pf = pareto_front(df, maximize=("accuracy",), minimize=("wall_time_s", "inference_j", "total_wire_length_um"))
    pf_out = ROOT / "results/tables/mnist_time_accuracy_pareto.csv"
    pf_out.parent.mkdir(parents=True, exist_ok=True)
    pf.to_csv(pf_out, index=False)
    save_scatter_plot(df, "wall_time_s", "accuracy", ROOT / "results/figures/mnist_accuracy_vs_wall_time.png", hue="family")
    save_scatter_plot(df, "inference_j", "accuracy", ROOT / "results/figures/mnist_accuracy_vs_inference_energy_frontier.png", hue="family")
    save_scatter_plot(pf, "wall_time_s", "accuracy", ROOT / "results/figures/mnist_pareto_time_accuracy.png", hue="family")
    summary = {
        "raw_trials": str(raw),
        "pareto": str(pf_out),
        "best_accuracy": df.sort_values("accuracy", ascending=False).iloc[0].to_dict(),
        "best_accuracy_per_second": df.sort_values("accuracy_per_second", ascending=False).iloc[0].to_dict(),
        "pareto_count": int(len(pf)),
    }
    summary_out = ROOT / "results/tables/mnist_time_accuracy_frontier_summary.json"
    summary_out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

