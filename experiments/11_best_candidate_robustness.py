from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.datasets import mnist_loaders, set_seed
from sim.local_layers import HardwareLocalEvidenceNet, set_activation_mode
from sim.train_backprop import accuracy


def default_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_candidate(path: Path, device: str) -> HardwareLocalEvidenceNet:
    ckpt = torch.load(path, map_location="cpu")
    args = ckpt["args"]
    channels = tuple(int(v) for v in args["channels"].split(","))
    model = HardwareLocalEvidenceNet(
        channels=channels,
        activation_sigma=args["activation_sigma"],
        activation_bits=args["activation_bits"],
        activation_mode=args["eval_activation_mode"],
        coord_channels=not args["no_coord_channels"],
        weight_bits=args.get("weight_bits"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    set_activation_mode(
        model,
        args["eval_activation_mode"],
        sigma=args["activation_sigma"],
        bits=args["activation_bits"],
    )
    model.to(device)
    return model


def quantize_tensor_symmetric(t: torch.Tensor, bits: int) -> torch.Tensor:
    if bits <= 0 or t.numel() == 0:
        return t
    max_abs = t.detach().abs().max()
    if float(max_abs) == 0.0:
        return t.clone()
    levels = 2 ** (bits - 1) - 1
    return torch.round(t / max_abs * levels).clamp(-levels, levels) / levels * max_abs


def perturb_state(
    state: dict[str, torch.Tensor],
    perturbation: str,
    severity: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    out = {k: v.clone() for k, v in state.items()}
    for name, tensor in list(out.items()):
        if not torch.is_floating_point(tensor):
            continue
        is_weight_matrix = tensor.ndim >= 2 and ("weight" in name)
        if perturbation == "weight_quantization" and is_weight_matrix:
            bits = max(2, int(round(severity)))
            out[name] = quantize_tensor_symmetric(tensor, bits)
        elif perturbation == "stuck_zero_weights" and is_weight_matrix:
            mask = torch.rand(tensor.shape, generator=g, dtype=tensor.dtype) < severity
            out[name] = tensor.masked_fill(mask, 0.0)
        elif perturbation == "conductance_drift" and is_weight_matrix:
            drift = torch.randn(tensor.shape, generator=g, dtype=tensor.dtype) * severity
            out[name] = tensor * (1.0 + drift)
        elif perturbation == "activation_offset_mismatch" and name.endswith(".bias") and any(bn in name for bn in ["bn1", "bn2", "bn3"]):
            out[name] = tensor + torch.randn(tensor.shape, generator=g, dtype=tensor.dtype) * severity
    return out


def noisy_accuracy(model, loader, sigma: float, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if sigma > 0:
                x = x + sigma * torch.randn_like(x)
            pred = model(x).argmax(1)
            correct += int((pred == y).sum())
            total += int(y.numel())
    return correct / max(total, 1)


def draw_heatmap(df: pd.DataFrame, out: Path) -> None:
    piv = df.pivot(index="perturbation", columns="severity_label", values="accuracy")
    plt.figure(figsize=(8, 4.5))
    plt.imshow(piv, aspect="auto", vmin=max(0.0, piv.min().min() - 0.03), vmax=1.0)
    plt.xticks(range(len(piv.columns)), piv.columns, rotation=30, ha="right")
    plt.yticks(range(len(piv.index)), piv.index)
    plt.colorbar(label="accuracy")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "results/raw/hardware_mnist_candidate.pt"))
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--test-limit", type=int, default=None)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    _, test_loader = mnist_loaders(
        batch_size=args.batch_size,
        flatten=False,
        normalize=True,
        test_limit=args.test_limit,
        seed=args.seed,
    )
    base = load_candidate(Path(args.checkpoint), args.device)
    base_state = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    rows = []
    base_acc = accuracy(base, test_loader, device=args.device)
    rows.append({"perturbation": "baseline", "severity": 0.0, "severity_label": "0", "accuracy": base_acc})

    specs = [
        ("weight_quantization", [8, 6, 4, 3, 2]),
        ("stuck_zero_weights", [0.001, 0.003, 0.01, 0.03, 0.10]),
        ("conductance_drift", [0.01, 0.03, 0.05, 0.10, 0.20]),
        ("activation_offset_mismatch", [0.01, 0.03, 0.05, 0.10, 0.20]),
    ]
    for perturbation, severities in specs:
        for severity in severities:
            model = load_candidate(Path(args.checkpoint), args.device)
            perturbed = perturb_state(base_state, perturbation, float(severity), args.seed + int(float(severity) * 10000))
            model.load_state_dict(perturbed)
            acc = accuracy(model, test_loader, device=args.device)
            rows.append(
                {
                    "perturbation": perturbation,
                    "severity": float(severity),
                    "severity_label": str(severity),
                    "accuracy": acc,
                }
            )

    for sigma in [0.01, 0.03, 0.05, 0.10, 0.20]:
        acc = noisy_accuracy(base, test_loader, sigma=sigma, device=args.device)
        rows.append({"perturbation": "input_noise_sigma", "severity": sigma, "severity_label": str(sigma), "accuracy": acc})

    df = pd.DataFrame(rows)
    out_csv = ROOT / "results/tables/best_candidate_robustness.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    draw_heatmap(df[df["perturbation"] != "baseline"], ROOT / "results/figures/best_candidate_robustness_heatmap.png")
    summary = {
        "baseline_accuracy": base_acc,
        "worst_case_accuracy": float(df["accuracy"].min()),
        "csv": str(out_csv),
        "figure": str(ROOT / "results/figures/best_candidate_robustness_heatmap.png"),
    }
    out_json = ROOT / "results/tables/best_candidate_robustness_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
