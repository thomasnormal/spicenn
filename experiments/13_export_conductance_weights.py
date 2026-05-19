from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fold_conv_bn(state: dict[str, torch.Tensor], conv_name: str, bn_name: str) -> tuple[np.ndarray, np.ndarray]:
    w = state[f"{conv_name}.weight"].detach().cpu()
    gamma = state[f"{bn_name}.weight"].detach().cpu()
    beta = state[f"{bn_name}.bias"].detach().cpu()
    mean = state[f"{bn_name}.running_mean"].detach().cpu()
    var = state[f"{bn_name}.running_var"].detach().cpu()
    eps = torch.tensor(1e-5)
    scale = gamma / torch.sqrt(var + eps)
    folded_w = w * scale.reshape((-1, 1, 1, 1))
    folded_b = beta - scale * mean
    return folded_w.numpy(), folded_b.numpy()


def conv_only(state: dict[str, torch.Tensor], conv_name: str) -> tuple[np.ndarray, np.ndarray]:
    w = state[f"{conv_name}.weight"].detach().cpu().numpy()
    b_key = f"{conv_name}.bias"
    if b_key in state and state[b_key] is not None:
        b = state[b_key].detach().cpu().numpy()
    else:
        b = np.zeros(w.shape[0], dtype=np.float32)
    return w, b


def quantize_symmetric(w: np.ndarray, bits: int | None) -> np.ndarray:
    if bits is None or bits <= 0:
        return w
    levels = 2 ** (bits - 1) - 1
    max_abs = float(np.max(np.abs(w)))
    if max_abs == 0.0:
        return w.copy()
    return np.clip(np.round(w / max_abs * levels), -levels, levels) / levels * max_abs


def to_differential_conductance(
    w: np.ndarray,
    g_min: float,
    g_max: float,
    weight_bits: int | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    wq = quantize_symmetric(w, weight_bits)
    scale = float(np.max(np.abs(wq)))
    if scale == 0.0:
        normalized = np.zeros_like(wq)
    else:
        normalized = np.clip(wq / scale, -1.0, 1.0)
    span = g_max - g_min
    g_pos = g_min + np.maximum(normalized, 0.0) * span
    g_neg = g_min + np.maximum(-normalized, 0.0) * span
    return g_pos.astype(np.float32), g_neg.astype(np.float32), scale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "results/raw/hardware_mnist_candidate.pt"))
    ap.add_argument("--g-min", type=float, default=10e-9)
    ap.add_argument("--g-max", type=float, default=1e-6)
    ap.add_argument("--weight-bits", type=int, default=None, help="Optional export-time symmetric weight quantization.")
    ap.add_argument("--out-prefix", default=str(ROOT / "results/raw/conductance_weights"))
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model_state_dict"]
    layers = {
        "conv1": fold_conv_bn(state, "conv1", "bn1"),
        "conv2": fold_conv_bn(state, "conv2", "bn2"),
        "conv3": fold_conv_bn(state, "conv3", "bn3"),
        "readout": conv_only(state, "readout"),
    }

    arrays: dict[str, np.ndarray] = {}
    rows = []
    total_cells = 0
    for name, (w, b) in layers.items():
        g_pos, g_neg, scale = to_differential_conductance(w, args.g_min, args.g_max, args.weight_bits)
        arrays[f"{name}_g_pos"] = g_pos
        arrays[f"{name}_g_neg"] = g_neg
        arrays[f"{name}_bias"] = b.astype(np.float32)
        total_cells += 2 * w.size
        rows.append(
            {
                "layer": name,
                "shape": "x".join(map(str, w.shape)),
                "stored_weights": int(w.size),
                "differential_conductance_cells": int(2 * w.size),
                "bias_terms": int(b.size),
                "weight_scale_before_conductance": scale,
                "g_min_siemens": args.g_min,
                "g_max_siemens": args.g_max,
                "g_pos_mean_siemens": float(g_pos.mean()),
                "g_neg_mean_siemens": float(g_neg.mean()),
                "fraction_positive": float(np.mean(w > 0)),
                "fraction_negative": float(np.mean(w < 0)),
                "export_weight_bits": args.weight_bits,
            }
        )

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = out_prefix.with_suffix(".npz")
    np.savez_compressed(npz_path, **arrays)
    table = pd.DataFrame(rows)
    suffix = "" if args.weight_bits is None else f"_{args.weight_bits}bit"
    csv_path = ROOT / f"results/tables/conductance_weight_export{suffix}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)

    manifest = {
        "checkpoint": args.checkpoint,
        "npz": str(npz_path),
        "table": str(csv_path),
        "mapping": "signed folded weights represented by differential conductance pair G_pos-G_neg",
        "g_min_siemens": args.g_min,
        "g_max_siemens": args.g_max,
        "export_weight_bits": args.weight_bits,
        "total_stored_weights": int(sum(row["stored_weights"] for row in rows)),
        "total_differential_conductance_cells": int(total_cells),
        "layers": rows,
    }
    json_path = ROOT / f"results/tables/conductance_weight_export{suffix}.json"
    json_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
