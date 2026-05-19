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

from sim.datasets import mnist_loaders, set_seed
from sim.hardware_metrics import local_evidence_net_layers, summarize_layers
from sim.local_layers import HardwareLocalEvidenceNet
from sim.neuron_models import SpiceChargeADCLUT
from sim.plotting import save_line_plot, save_scatter_plot
from sim.train_backprop import train_classifier


def default_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def parse_channels(text: str) -> tuple[int, int, int]:
    vals = tuple(int(v) for v in text.split(","))
    if len(vals) != 3:
        raise ValueError("--channels must be a,b,c")
    return vals


def lut_stats(csv_path: Path, bits: int, cint_f: float, sigma_v: float) -> dict:
    df = pd.read_csv(csv_path)
    df = df[(df["bits"] == bits) & df["cint_f"].sub(cint_f).abs().le(cint_f * 1e-6)]
    df = df[df["sigma_v"].sub(sigma_v).abs().le(max(sigma_v, 1e-12) * 1e-6)]
    if df.empty:
        return {}
    return {
        "spice_vin_min": float(df["vin"].min()),
        "spice_vin_max": float(df["vin"].max()),
        "spice_code_min": float(df["mean_code_0_1"].min()),
        "spice_code_max": float(df["mean_code_0_1"].max()),
        "spice_code_span": float(df["mean_code_0_1"].max() - df["mean_code_0_1"].min()),
        "mean_ecap_j": float(df["ecap_j_mean"].mean()),
        "spice_simulator": str(df["simulator"].iloc[0]),
    }


def make_spice_config(spec: str) -> dict:
    # Format: name:bits:cint_f:input_scale[:output_mode]
    # Example: adc4_c10f_s006_rect:4:1e-14:0.06:rectified
    parts = spec.split(":")
    if len(parts) not in {4, 5}:
        raise ValueError("config must be name:bits:cint_f:input_scale[:output_mode]")
    name, bits, cint_f, scale = parts[:4]
    return {
        "name": name,
        "activation_mode": "spice_charge_adc_lut",
        "bits": int(bits),
        "cint_f": float(cint_f),
        "input_scale": float(scale),
        "output_mode": parts[4] if len(parts) == 5 else "unipolar",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--train-limit", type=int, default=10000)
    ap.add_argument("--test-limit", type=int, default=2000)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--channels", default="16,32,48")
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma-v", type=float, default=1.5e-3)
    ap.add_argument("--lut-csv", default=str(ROOT / "spice/results/charge_adc_sweep.csv"))
    ap.add_argument(
        "--configs",
        default=(
            "adc4_c3f_s003_rect:4:3e-15:0.03:rectified;"
            "adc4_c3f_s006_rect:4:3e-15:0.06:rectified;"
            "adc4_c10f_s006_rect:4:1e-14:0.06:rectified;"
            "adc4_c10f_s010_rect:4:1e-14:0.10:rectified;"
            "adc3_c10f_s006_rect:3:1e-14:0.06:rectified"
        ),
    )
    ap.add_argument("--include-ideal-baseline", action="store_true")
    ap.add_argument("--max-configs", type=int, default=None)
    args = ap.parse_args()

    channels = parse_channels(args.channels)
    lut_csv = Path(args.lut_csv)
    if not lut_csv.exists():
        raise FileNotFoundError(f"Missing SPICE charge ADC LUT: {lut_csv}")

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
    rows: list[dict] = []
    configs = [make_spice_config(s) for s in args.configs.split(";") if s.strip()]
    if args.max_configs is not None:
        configs = configs[: args.max_configs]

    if args.include_ideal_baseline:
        configs.insert(
            0,
            {
                "name": "ideal_quantized_relu6_4b",
                "activation_mode": "quantized_relu6",
                "bits": 4,
                "cint_f": float("nan"),
                "input_scale": float("nan"),
                "output_mode": "rectified",
            },
        )

    for cfg in configs:
        set_seed(args.seed)
        lut = None
        stats = {}
        comparator_cycles = int(cfg["bits"])
        if cfg["activation_mode"] == "spice_charge_adc_lut":
            lut = SpiceChargeADCLUT(
                lut_csv,
                bits=int(cfg["bits"]),
                cint_f=float(cfg["cint_f"]),
                sigma_v=args.sigma_v,
                input_scale=float(cfg["input_scale"]),
                output_mode=str(cfg["output_mode"]),
            )
            stats = lut_stats(lut_csv, int(cfg["bits"]), float(cfg["cint_f"]), args.sigma_v)
            comparator_cycles = int(cfg["bits"])

        model = HardwareLocalEvidenceNet(
            channels=channels,
            activation_sigma=0.5,
            activation_bits=int(cfg["bits"]),
            activation_mode=cfg["activation_mode"],
            coord_channels=True,
            activation_lut=lut,
        )
        t0 = time.perf_counter()
        result = train_classifier(
            model,
            train_loader,
            test_loader,
            epochs=args.epochs,
            lr=2e-3,
            weight_decay=1e-4,
            device=args.device,
        )
        wall = time.perf_counter() - t0
        hw = summarize_layers(
            layers,
            stochastic_cycles=1,
            wire_read_cycles=1,
            comparator_cycles=comparator_cycles,
        )
        row = {
            "activation_design": cfg["name"],
            "activation_mode": cfg["activation_mode"],
            "bits": int(cfg["bits"]),
            "cint_f": cfg["cint_f"],
            "input_scale": cfg["input_scale"],
            "output_mode": cfg["output_mode"],
            "sigma_v": args.sigma_v,
            "channels": args.channels,
            "epochs": args.epochs,
            "train_limit": args.train_limit,
            "test_limit": args.test_limit,
            "device": args.device,
            "wall_time_s": wall,
            "accuracy": result.test_accuracy[-1],
            "best_epoch_accuracy": result.best_accuracy,
            "train_loss_final": result.train_loss[-1],
            "pytorch_parameter_count": count_params(model),
            "source_lut": str(lut_csv) if lut is not None else "ideal_pytorch",
            "equivalence_note": (
                "PyTorch forward uses the measured SPICE charge-ADC transfer LUT with STE gradients"
                if lut is not None
                else "Ideal PyTorch quantized ReLU6 baseline; not SPICE-derived"
            ),
            **stats,
            **hw,
        }
        rows.append(row)
        print(json.dumps(row, indent=2))

    df = pd.DataFrame(rows)
    out = ROOT / "results/tables/spice_lut_mnist_calibration.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    save_scatter_plot(
        df,
        "inference_j",
        "accuracy",
        ROOT / "results/figures/spice_lut_accuracy_vs_energy.png",
        hue="activation_design",
    )
    spice_df = df[df["activation_mode"] == "spice_charge_adc_lut"].copy()
    if not spice_df.empty:
        spice_df = spice_df.sort_values("input_scale")
        save_line_plot(
            spice_df,
            "input_scale",
            "accuracy",
            ROOT / "results/figures/spice_lut_accuracy_vs_input_scale.png",
            hue="cint_f",
        )
    best = df.sort_values(["accuracy", "inference_j"], ascending=[False, True]).iloc[0].to_dict()
    best_spice = None
    if not spice_df.empty:
        best_spice = spice_df.sort_values(["accuracy", "inference_j"], ascending=[False, True]).iloc[0].to_dict()
    summary = {
        "result_table": str(out),
        "best": best,
        "best_spice_lut": best_spice,
        "statement": (
            "This is closer to the SPICE architecture than the 98.92% quantized-ReLU6 model, "
            "because the activation nonlinearity is imported from ngspice charge-ADC sweeps. "
            "It is still not a full SPICE MNIST implementation: convolution, BatchNorm, pooling, "
            "and weight updates are executed by PyTorch."
        ),
    }
    summary_out = ROOT / "results/tables/spice_lut_mnist_calibration_summary.json"
    summary_out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
