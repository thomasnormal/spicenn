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
from sim.hardware_metrics import layer_table, local_evidence_net_layers, summarize_layers
from sim.local_layers import HardwareLocalEvidenceNet, set_activation_mode, set_stochastic_mode
from sim.plotting import save_line_plot
from sim.train_backprop import accuracy, train_classifier


def activation_decision_factor(mode: str, bits: int, cycles: int) -> tuple[int, int]:
    """Return wire/read/integrator cycles and comparator decision cycles."""
    if "thermometer" in mode or "multi_comparator" in mode:
        return 1, max(1, 2**bits - 1)
    if "quantized" in mode or "multi_bit" in mode:
        return 1, max(1, bits)
    return max(1, cycles), max(1, cycles)


def default_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--test-limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--channels", default="24,48,64")
    ap.add_argument("--activation-sigma", type=float, default=0.75)
    ap.add_argument("--activation-bits", type=int, default=4)
    ap.add_argument("--train-activation-mode", default=None)
    ap.add_argument("--clean-train", action="store_true", help="Train with SiLU as an upper-bound surrogate.")
    ap.add_argument("--eval-activation-mode", default="expected_probit_from_sigma")
    ap.add_argument("--no-coord-channels", action="store_true")
    ap.add_argument("--weight-bits", type=int, default=None, help="Fake-quantize conv/readout weights during training and eval.")
    args = ap.parse_args()

    channels = tuple(int(v) for v in args.channels.split(","))
    if len(channels) != 3:
        raise ValueError("--channels must have three comma-separated integers")
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

    model = HardwareLocalEvidenceNet(
        channels=channels,
        activation_sigma=args.activation_sigma,
        activation_bits=args.activation_bits,
        clean_train=args.clean_train,
        activation_mode=args.train_activation_mode,
        coord_channels=not args.no_coord_channels,
        weight_bits=args.weight_bits,
    )
    result = train_classifier(
        model,
        train_loader,
        test_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
    )

    layers = local_evidence_net_layers(channels=channels, coord_channels=not args.no_coord_channels)
    layer_df = layer_table(layers)
    layer_out = ROOT / "results/tables/hardware_mnist_candidate_layers.csv"
    layer_out.parent.mkdir(parents=True, exist_ok=True)
    layer_df.to_csv(layer_out, index=False)

    sweep_rows = []
    for sample_mode in ["expected", "iid", "stratified", "ramp"]:
        for cycles in [1, 2, 4, 8, 16, 32, 64]:
            set_activation_mode(model, args.eval_activation_mode, sigma=args.activation_sigma, bits=args.activation_bits)
            set_stochastic_mode(model, cycles=cycles, sample_mode=sample_mode)
            acc = accuracy(model, test_loader, device=args.device)
            wire_cycles, comparator_cycles = activation_decision_factor(args.eval_activation_mode, args.activation_bits, cycles)
            rec = summarize_layers(
                layers,
                stochastic_cycles=cycles,
                wire_read_cycles=wire_cycles,
                comparator_cycles=comparator_cycles,
            )
            sweep_rows.append({
                "architecture": "hardware_local_evidence",
                "seed": args.seed,
                "epochs": args.epochs,
                "channels": ",".join(map(str, channels)),
                "training_activation": args.train_activation_mode or ("clean_silu_or_tanh" if args.clean_train else "expected_probit_from_sigma"),
                "eval_activation": args.eval_activation_mode,
                "coord_channels": not args.no_coord_channels,
                "activation_sigma_normalized": args.activation_sigma,
                "activation_bits": args.activation_bits,
                "weight_bits": args.weight_bits,
                "sample_mode": sample_mode,
                "L": cycles,
                "accuracy": acc,
                "wire_read_cycles": wire_cycles,
                "comparator_cycles": comparator_cycles,
                **rec,
            })
    sweep = pd.DataFrame(sweep_rows)
    sweep_out = ROOT / "results/tables/hardware_mnist_stochastic_sweep.csv"
    sweep.to_csv(sweep_out, index=False)
    save_line_plot(sweep, "L", "accuracy", ROOT / "results/figures/hardware_mnist_accuracy_vs_L.png", hue="sample_mode")

    train_df = pd.DataFrame({
        "epoch": list(range(1, args.epochs + 1)),
        "train_loss": result.train_loss,
        "accuracy": result.test_accuracy,
    })
    train_out = ROOT / "results/tables/hardware_mnist_training_curve.csv"
    train_df.to_csv(train_out, index=False)
    save_line_plot(train_df, "epoch", "accuracy", ROOT / "results/figures/hardware_mnist_training_curve.png")

    best = sweep.sort_values(["accuracy", "inference_j"], ascending=[False, True]).iloc[0].to_dict()
    summary = {
        "device": args.device,
        "best_training_accuracy": result.best_accuracy,
        "best_inference_point": best,
        "outputs": {
            "layers": str(layer_out),
            "training_curve": str(train_out),
            "stochastic_sweep": str(sweep_out),
        },
    }
    summary_out = ROOT / "results/tables/hardware_mnist_candidate_summary.json"
    summary_out.write_text(json.dumps(summary, indent=2) + "\n")
    ckpt = ROOT / "results/raw/hardware_mnist_candidate.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args), "summary": summary}, ckpt)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
