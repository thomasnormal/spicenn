from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_spice_mnist_batch_op_train import read_wrdata_row
from run_spice_mnist_train import load_mnist_sequence
from run_spice_sweep import ROOT, detect_spice, run_tiny_test


Feature = tuple[str, int, float]


def make_patches(image_size: int, kernel_size: int, stride: int, coord_channels: bool) -> list[list[Feature]]:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if stride <= 0:
        raise ValueError("stride must be positive")
    pad = kernel_size // 2
    coords = np.linspace(-1.0, 1.0, image_size)
    patches: list[list[Feature]] = []
    for cy in range(0, image_size, stride):
        for cx in range(0, image_size, stride):
            feats: list[Feature] = []
            for dy in range(-pad, pad + 1):
                for dx in range(-pad, pad + 1):
                    y = cy + dy
                    x = cx + dx
                    if 0 <= y < image_size and 0 <= x < image_size:
                        feats.append(("input", y * image_size + x, 0.0))
                        if coord_channels:
                            feats.append(("const", -1, float(coords[x])))
                            feats.append(("const", -1, float(coords[y])))
                    else:
                        feats.append(("const", -1, 0.0))
                        if coord_channels:
                            feats.append(("const", -1, 0.0))
                            feats.append(("const", -1, 0.0))
            patches.append(feats)
    return patches


def feature_expr(sample: int, feature: Feature) -> str:
    kind, idx, value = feature
    if kind == "input":
        return f"V(x{sample}_{idx})"
    return f"{value:.12g}"


def make_train_netlist(
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    weights: np.ndarray,
    hidden_bias: np.ndarray,
    gains: np.ndarray,
    output_bias: np.ndarray,
    patches: list[list[Feature]],
    lr: float,
    out_path: Path,
    train_gains: bool,
) -> str:
    batch, n_in = x_batch.shape
    n_classes, channels, patch_len = weights.shape
    n_pos = len(patches)
    norm = 1.0 / max(n_pos, 1)
    lines = [
        "* Shared local evidence batch operating-point SPICE training.",
        "* Class-specific local kernels are reused across the sheet; ngspice computes forward, backward, and update equations.",
        "* The hidden state is analog/multilevel tanh voltage evidence, not a stochastic 0/1 bit.",
        f".param LR={lr:.12g}",
        f".param BS={batch}",
        f".param NORM={norm:.12g}",
        "",
    ]
    for s in range(batch):
        for i in range(n_in):
            lines.append(f"Vx{s}_{i} x{s}_{i} 0 DC {float(x_batch[s, i]):.12g}")
        target = -np.ones(n_classes)
        target[int(y_batch[s])] = 1.0
        for k in range(n_classes):
            lines.append(f"Vt{s}_{k} t{s}_{k} 0 DC {target[k]:.12g}")
    lines.append("")
    for k in range(n_classes):
        for c in range(channels):
            for p in range(patch_len):
                lines.append(f"Vw{k}_{c}_{p} w{k}_{c}_{p} 0 DC {weights[k, c, p]:.12g}")
            lines.append(f"Vhb{k}_{c} hb{k}_{c} 0 DC {hidden_bias[k, c]:.12g}")
            lines.append(f"Vg{k}_{c} g{k}_{c} 0 DC {gains[k, c]:.12g}")
        lines.append(f"Vob{k} ob{k} 0 DC {output_bias[k]:.12g}")
    lines.append("")
    for s in range(batch):
        for k in range(n_classes):
            out_terms = []
            for c in range(channels):
                for j, patch in enumerate(patches):
                    terms = [f"V(w{k}_{c}_{p})*{feature_expr(s, patch[p])}" for p in range(patch_len)]
                    terms.append(f"V(hb{k}_{c})")
                    lines.append(f"Bh{s}_{k}_{c}_{j} h{s}_{k}_{c}_{j} 0 V = 2/(1+exp(-2*({' + '.join(terms)})))-1")
                    out_terms.append(f"{{NORM}}*V(g{k}_{c})*V(h{s}_{k}_{c}_{j})")
            out_terms.append(f"V(ob{k})")
            lines.append(f"By{s}_{k} y{s}_{k} 0 V = 2/(1+exp(-2*({' + '.join(out_terms)})))-1")
            lines.append(f"Be{s}_{k} e{s}_{k} 0 V = V(t{s}_{k})-V(y{s}_{k})")
            lines.append(f"Bd{s}_{k} d{s}_{k} 0 V = V(e{s}_{k})*(1-V(y{s}_{k})*V(y{s}_{k}))")
            for c in range(channels):
                for j in range(n_pos):
                    lines.append(
                        f"Bdh{s}_{k}_{c}_{j} dh{s}_{k}_{c}_{j} 0 V = "
                        f"V(d{s}_{k})*{{NORM}}*V(g{k}_{c})*(1-V(h{s}_{k}_{c}_{j})*V(h{s}_{k}_{c}_{j}))"
                    )
    lines.append("")
    for k in range(n_classes):
        for c in range(channels):
            for p in range(patch_len):
                grad = " + ".join(
                    f"V(dh{s}_{k}_{c}_{j})*{feature_expr(s, patch[p])}"
                    for s in range(batch)
                    for j, patch in enumerate(patches)
                )
                lines.append(f"Bnw{k}_{c}_{p} nw{k}_{c}_{p} 0 V = V(w{k}_{c}_{p}) + {{LR}}*(({grad})/{{BS}})")
            grad_b = " + ".join(f"V(dh{s}_{k}_{c}_{j})" for s in range(batch) for j in range(n_pos))
            lines.append(f"Bnhb{k}_{c} nhb{k}_{c} 0 V = V(hb{k}_{c}) + {{LR}}*(({grad_b})/{{BS}})")
            if train_gains:
                grad_g = " + ".join(
                    f"V(d{s}_{k})*{{NORM}}*V(h{s}_{k}_{c}_{j})" for s in range(batch) for j in range(n_pos)
                )
                lines.append(f"Bng{k}_{c} ng{k}_{c} 0 V = V(g{k}_{c}) + {{LR}}*(({grad_g})/{{BS}})")
        grad_o = " + ".join(f"V(d{s}_{k})" for s in range(batch))
        lines.append(f"Bnob{k} nob{k} 0 V = V(ob{k}) + {{LR}}*(({grad_o})/{{BS}})")
    vectors = [f"V(nw{k}_{c}_{p})" for k in range(n_classes) for c in range(channels) for p in range(patch_len)]
    vectors += [f"V(nhb{k}_{c})" for k in range(n_classes) for c in range(channels)]
    if train_gains:
        vectors += [f"V(ng{k}_{c})" for k in range(n_classes) for c in range(channels)]
    vectors += [f"V(nob{k})" for k in range(n_classes)]
    lines += ["", ".control", "op", f"wrdata {out_path} " + " ".join(vectors), ".endc", ".end", ""]
    return "\n".join(lines)


def make_eval_netlist(
    x_batch: np.ndarray,
    weights: np.ndarray,
    hidden_bias: np.ndarray,
    gains: np.ndarray,
    output_bias: np.ndarray,
    patches: list[list[Feature]],
    out_path: Path,
) -> str:
    batch, n_in = x_batch.shape
    n_classes, channels, patch_len = weights.shape
    n_pos = len(patches)
    norm = 1.0 / max(n_pos, 1)
    lines = [
        "* Shared local evidence batch operating-point SPICE inference.",
        f".param NORM={norm:.12g}",
        "",
    ]
    for s in range(batch):
        for i in range(n_in):
            lines.append(f"Vx{s}_{i} x{s}_{i} 0 DC {float(x_batch[s, i]):.12g}")
    lines.append("")
    for k in range(n_classes):
        for c in range(channels):
            for p in range(patch_len):
                lines.append(f"Vw{k}_{c}_{p} w{k}_{c}_{p} 0 DC {weights[k, c, p]:.12g}")
            lines.append(f"Vhb{k}_{c} hb{k}_{c} 0 DC {hidden_bias[k, c]:.12g}")
            lines.append(f"Vg{k}_{c} g{k}_{c} 0 DC {gains[k, c]:.12g}")
        lines.append(f"Vob{k} ob{k} 0 DC {output_bias[k]:.12g}")
    lines.append("")
    for s in range(batch):
        for k in range(n_classes):
            out_terms = []
            for c in range(channels):
                for j, patch in enumerate(patches):
                    terms = [f"V(w{k}_{c}_{p})*{feature_expr(s, patch[p])}" for p in range(patch_len)]
                    terms.append(f"V(hb{k}_{c})")
                    lines.append(f"Bh{s}_{k}_{c}_{j} h{s}_{k}_{c}_{j} 0 V = 2/(1+exp(-2*({' + '.join(terms)})))-1")
                    out_terms.append(f"{{NORM}}*V(g{k}_{c})*V(h{s}_{k}_{c}_{j})")
            out_terms.append(f"V(ob{k})")
            lines.append(f"By{s}_{k} y{s}_{k} 0 V = 2/(1+exp(-2*({' + '.join(out_terms)})))-1")
    vectors = [f"V(y{s}_{k})" for s in range(batch) for k in range(n_classes)]
    lines += ["", ".control", "op", f"wrdata {out_path} " + " ".join(vectors), ".endc", ".end", ""]
    return "\n".join(lines)


def run_train_batch(
    spice_bin,
    netlist_path,
    data_path,
    x,
    y,
    weights,
    hidden_bias,
    gains,
    output_bias,
    patches,
    lr,
    timeout,
    train_gains,
):
    netlist_path.write_text(
        make_train_netlist(x, y, weights, hidden_bias, gains, output_bias, patches, lr, data_path, train_gains)
    )
    proc = subprocess.run([spice_bin, "-b", str(netlist_path)], text=True, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-3000:] or proc.stdout[-3000:])
    n_classes, channels, patch_len = weights.shape
    n = n_classes * channels * patch_len + n_classes * channels + (n_classes * channels if train_gains else 0) + n_classes
    vals = read_wrdata_row(data_path, n)
    offset = 0
    nw = vals[offset : offset + n_classes * channels * patch_len].reshape(weights.shape)
    offset += n_classes * channels * patch_len
    nhb = vals[offset : offset + n_classes * channels].reshape(hidden_bias.shape)
    offset += n_classes * channels
    if train_gains:
        ng = vals[offset : offset + n_classes * channels].reshape(gains.shape)
        offset += n_classes * channels
    else:
        ng = gains
    nob = vals[offset : offset + n_classes]
    return nw, nhb, ng, nob


def run_eval(spice_bin, netlist_path, data_path, x_eval, y_eval, weights, hidden_bias, gains, output_bias, patches, batch_size, timeout):
    correct = 0
    for start in range(0, len(y_eval), batch_size):
        x = x_eval[start : start + batch_size]
        y = y_eval[start : start + batch_size]
        netlist_path.write_text(make_eval_netlist(x, weights, hidden_bias, gains, output_bias, patches, data_path))
        proc = subprocess.run([spice_bin, "-b", str(netlist_path)], text=True, capture_output=True, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-3000:] or proc.stdout[-3000:])
        vals = read_wrdata_row(data_path, len(y) * weights.shape[0]).reshape(len(y), weights.shape[0])
        correct += int((np.argmax(vals, axis=1) == y).sum())
    return correct / max(len(y_eval), 1)


def plot_curve(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(df["epoch"], df["heldout_accuracy"], marker="o")
    plt.xlabel("epoch")
    plt.ylabel("held-out accuracy")
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-samples", type=int, default=200)
    ap.add_argument("--test-samples", type=int, default=200)
    ap.add_argument("--image-size", type=int, default=8)
    ap.add_argument("--kernel-size", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--init-scale", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--coord-channels", action="store_true")
    ap.add_argument("--train-gains", action="store_true")
    ap.add_argument("--tag", default="shared_local_evidence")
    args = ap.parse_args()

    patches = make_patches(args.image_size, args.kernel_size, args.stride, args.coord_channels)
    x_train, y_train, x_test, y_test = load_mnist_sequence(args.train_samples, args.test_samples, args.image_size, args.seed)
    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    generated.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)
    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag)
    stem = f"spice_mnist_shared_local_{safe_tag}"
    netlist_path = generated / f"{stem}_step.cir"
    eval_netlist = generated / f"{stem}_eval.cir"
    data_path = ROOT / f"spice/results/{stem}_step.dat"
    rng = np.random.default_rng(args.seed)
    patch_len = len(patches[0])
    weights = rng.normal(0.0, args.init_scale, size=(10, args.channels, patch_len))
    hidden_bias = np.zeros((10, args.channels))
    gains = np.full((10, args.channels), 1.0 / max(args.channels, 1))
    output_bias = np.zeros(10)
    rows = []
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        order = np.arange(len(y_train))
        rng.shuffle(order)
        epoch_start = time.perf_counter()
        for n, start in enumerate(range(0, len(order), args.batch_size)):
            idx = order[start : start + args.batch_size]
            weights, hidden_bias, gains, output_bias = run_train_batch(
                spice_bin,
                netlist_path,
                data_path,
                x_train[idx],
                y_train[idx],
                weights,
                hidden_bias,
                gains,
                output_bias,
                patches,
                args.lr,
                args.timeout,
                args.train_gains,
            )
            if (n + 1) % 5 == 0:
                print(f"epoch {epoch + 1} batch {n + 1}", flush=True)
        heldout = run_eval(
            spice_bin,
            eval_netlist,
            data_path,
            x_test,
            y_test,
            weights,
            hidden_bias,
            gains,
            output_bias,
            patches,
            args.batch_size,
            args.timeout,
        )
        row = {"epoch": epoch + 1, "heldout_accuracy": heldout, "epoch_wall_time_s": time.perf_counter() - epoch_start}
        rows.append(row)
        print(json.dumps(row), flush=True)
    curve = pd.DataFrame(rows)
    curve_path = ROOT / f"spice/results/{stem}_learning_curve.csv"
    curve.to_csv(curve_path, index=False)
    fig = ROOT / f"spice/results/{stem}_learning_curve.png"
    plot_curve(curve, fig)
    weights_path = ROOT / f"spice/results/{stem}_final_weights.npz"
    np.savez_compressed(weights_path, weights=weights, hidden_bias=hidden_bias, gains=gains, output_bias=output_bias)
    summary = {
        "simulator": version,
        "dataset": "MNIST train/test split, downsampled",
        "architecture": "shared_local_class_evidence",
        "activation": "analog_tanh_voltage_state",
        "image_size": args.image_size,
        "kernel_size": args.kernel_size,
        "stride": args.stride,
        "positions": len(patches),
        "channels_per_class": args.channels,
        "coord_channels": bool(args.coord_channels),
        "local": True,
        "weight_sharing": True,
        "train_gains": bool(args.train_gains),
        "inputs": int(x_train.shape[1]),
        "classes": 10,
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "init_scale": args.init_scale,
        "wall_time_s": time.perf_counter() - t0,
        "learning_curve": str(curve_path),
        "figure": str(fig),
        "final_weights": str(weights_path),
        "heldout_test_accuracy": float(curve.iloc[-1]["heldout_accuracy"]),
        "best_heldout_accuracy": float(curve["heldout_accuracy"].max()),
        "note": (
            "Shared local class-evidence batch-op all-SPICE training: ngspice computes analog local kernel evidence "
            "at every scanned position, class errors, and shared programmable weight updates."
        ),
    }
    out = ROOT / f"spice/results/{stem}_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
