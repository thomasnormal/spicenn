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
from sim.energy_model import estimate_inference_energy
from sim.local_layers import LayerlessRecurrentSheetNet
from sim.plotting import save_line_plot
from sim.train_backprop import train_classifier


def default_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def hardware_proxy(channels: int, ticks: int, image_size: int = 28, pitch_um: float = 10.0) -> dict:
    cells = image_size * image_size * channels
    input_syn = image_size * image_size * channels * 3
    recurrent_syn = image_size * image_size * channels * channels * 9
    readout_syn = image_size * image_size * channels * 10
    synapses = input_syn + recurrent_syn + readout_syn
    total_wire = image_size * image_size * channels * channels * 9 * (12.0 * pitch_um / 9.0)
    total_wire += image_size * image_size * channels * (3 + 10) * pitch_um
    energy = estimate_inference_energy(
        synapse_count=synapses,
        active_synapse_count=int(0.25 * synapses),
        total_wire_length_um=total_wire,
        neuron_count=cells,
        stochastic_cycles=ticks,
        wire_cycles=ticks,
        read_cycles=ticks,
        integrator_cycles=ticks,
        comparator_cycles=4 * ticks,
        activity=0.25,
    )
    return {
        "synapse_instances": synapses,
        "hardware_parameter_count": 3 * channels + channels * channels * 9 + channels * 10,
        "neuron_count": cells,
        "total_wire_length_um": total_wire,
        "max_wire_length_um": 20.0,
        **energy.to_dict(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--train-limit", type=int, default=10000)
    ap.add_argument("--test-limit", type=int, default=2000)
    ap.add_argument("--ticks", default="2,4,6,8")
    ap.add_argument("--channels", type=int, default=48)
    ap.add_argument("--activation-mode", default="quantized_relu6")
    ap.add_argument("--activation-bits", type=int, default=4)
    ap.add_argument("--activation-sigma", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--seed", type=int, default=0)
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
    for ticks in [int(v) for v in args.ticks.split(",")]:
        set_seed(args.seed)
        model = LayerlessRecurrentSheetNet(
            channels=args.channels,
            ticks=ticks,
            activation_mode=args.activation_mode,
            activation_bits=args.activation_bits,
            activation_sigma=args.activation_sigma,
        )
        result = train_classifier(
            model,
            train_loader,
            test_loader,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=1e-4,
            device=args.device,
        )
        rows.append(
            {
                "architecture": "layerless_recurrent_sheet",
                "ticks": ticks,
                "channels": args.channels,
                "activation_mode": args.activation_mode,
                "activation_bits": args.activation_bits,
                "activation_sigma": args.activation_sigma,
                "epochs": args.epochs,
                "train_limit": args.train_limit,
                "test_limit": args.test_limit,
                "accuracy": result.test_accuracy[-1],
                "best_epoch_accuracy": result.best_accuracy,
                **hardware_proxy(args.channels, ticks),
            }
        )
        print(json.dumps(rows[-1], indent=2))
    df = pd.DataFrame(rows)
    out = ROOT / "results/tables/layerless_recurrent_sheet.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    save_line_plot(df, "ticks", "accuracy", ROOT / "results/figures/layerless_recurrent_accuracy_vs_ticks.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
