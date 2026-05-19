from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.datasets import mnist_loaders, set_seed
from sim.hardware_metrics import local_evidence_net_layers, summarize_layers
from sim.local_layers import HardwareLocalEvidenceNet, set_activation_mode, set_stochastic_mode
from sim.plotting import save_scatter_plot
from sim.train_backprop import accuracy, train_classifier


CONFIGS = [
    {"name": "clean_silu_upper", "train_mode": "clean_silu_or_tanh", "eval_mode": "clean_silu_or_tanh", "sigma": 0.5, "bits": 8, "cycles": 1},
    {"name": "binary_expected_probit", "train_mode": "expected_probit_from_sigma", "eval_mode": "expected_probit_from_sigma", "sigma": 0.5, "bits": 1, "cycles": 1},
    {"name": "binary_bipolar_probit", "train_mode": "expected_probit_bipolar", "eval_mode": "expected_probit_bipolar", "sigma": 0.5, "bits": 1, "cycles": 1},
    {"name": "thermometer_3bit", "train_mode": "thermometer_probit", "eval_mode": "thermometer_probit", "sigma": 0.5, "bits": 3, "cycles": 1},
    {"name": "thermometer_3bit_bipolar", "train_mode": "thermometer_probit_bipolar", "eval_mode": "thermometer_probit_bipolar", "sigma": 0.5, "bits": 3, "cycles": 1},
    {"name": "quantized_charge_4bit", "train_mode": "quantized_charge", "eval_mode": "quantized_charge", "sigma": 0.5, "bits": 4, "cycles": 1},
    {"name": "quantized_relu6_4bit", "train_mode": "quantized_relu6", "eval_mode": "quantized_relu6", "sigma": 0.5, "bits": 4, "cycles": 1},
    {"name": "quantized_tanh_4bit", "train_mode": "quantized_tanh", "eval_mode": "quantized_tanh", "sigma": 0.5, "bits": 4, "cycles": 1},
]


def default_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def comparator_factor(mode: str, bits: int, cycles: int) -> int:
    if "thermometer" in mode or "multi_comparator" in mode:
        return max(1, 2**bits - 1)
    if "quantized" in mode or "multi_bit" in mode:
        return max(1, bits)
    return max(1, cycles)


def wire_read_factor(mode: str, cycles: int) -> int:
    if "thermometer" in mode or "multi_comparator" in mode or "quantized" in mode or "multi_bit" in mode:
        return 1
    return max(1, cycles)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--train-limit", type=int, default=10000)
    ap.add_argument("--test-limit", type=int, default=2000)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--channels", default="32,64,96")
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-configs", type=int, default=None)
    args = ap.parse_args()

    channels = tuple(int(v) for v in args.channels.split(","))
    set_seed(args.seed)
    train_loader, test_loader = mnist_loaders(
        batch_size=args.batch_size,
        download=args.download,
        flatten=False,
        normalize=True,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        seed=args.seed,
    )
    layers = local_evidence_net_layers(channels=channels)
    rows = []
    configs = CONFIGS[: args.max_configs] if args.max_configs else CONFIGS
    for cfg in configs:
        set_seed(args.seed)
        model = HardwareLocalEvidenceNet(
            channels=channels,
            activation_sigma=cfg["sigma"],
            activation_bits=cfg["bits"],
            activation_mode=cfg["train_mode"],
            coord_channels=True,
        )
        result = train_classifier(
            model,
            train_loader,
            test_loader,
            epochs=args.epochs,
            lr=2e-3,
            weight_decay=1e-4,
            device=args.device,
        )
        set_activation_mode(model, cfg["eval_mode"], sigma=cfg["sigma"], bits=cfg["bits"])
        set_stochastic_mode(model, cycles=cfg["cycles"], sample_mode="expected")
        acc = accuracy(model, test_loader, device=args.device)
        factor = comparator_factor(cfg["eval_mode"], cfg["bits"], cfg["cycles"])
        wire_factor = wire_read_factor(cfg["eval_mode"], cfg["cycles"])
        hw = summarize_layers(
            layers,
            stochastic_cycles=cfg["cycles"],
            wire_read_cycles=wire_factor,
            comparator_cycles=factor,
        )
        rows.append(
            {
                "activation_design": cfg["name"],
                "train_mode": cfg["train_mode"],
                "eval_mode": cfg["eval_mode"],
                "sigma": cfg["sigma"],
                "bits": cfg["bits"],
                "epochs": args.epochs,
                "train_limit": args.train_limit,
                "test_limit": args.test_limit,
                "accuracy": acc,
                "best_epoch_accuracy": result.best_accuracy,
                "value_decision_factor": factor,
                "wire_read_cycles": wire_factor,
                "comparator_cycles": factor,
                **hw,
            }
        )
        print(json.dumps(rows[-1], indent=2))

    df = pd.DataFrame(rows)
    out = ROOT / "results/tables/value_activation_sweep.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    save_scatter_plot(df, "inference_j", "accuracy", ROOT / "results/figures/value_activation_accuracy_vs_energy.png", hue="activation_design")
    best = df.sort_values(["accuracy", "inference_j"], ascending=[False, True]).iloc[0].to_dict()
    hardware_df = df[df["activation_design"] != "clean_silu_upper"]
    best_hardware = hardware_df.sort_values(["accuracy", "inference_j"], ascending=[False, True]).iloc[0].to_dict()
    summary = ROOT / "results/tables/value_activation_sweep_best.json"
    summary.write_text(json.dumps({"best_overall": best, "best_hardware_plausible": best_hardware}, indent=2) + "\n")
    print(f"wrote {out}")
    print(json.dumps({"best_overall": best, "best_hardware_plausible": best_hardware}, indent=2))


if __name__ == "__main__":
    main()
