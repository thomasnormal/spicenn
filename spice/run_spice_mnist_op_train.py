from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_spice_mnist_train import load_mnist_sequence
from run_spice_sweep import ROOT, detect_spice, run_tiny_test


def make_train_op_netlist(
    x: np.ndarray,
    label: int,
    weights: np.ndarray,
    bias: np.ndarray,
    lr: float,
    out_path: Path,
) -> str:
    n_classes, n_in = weights.shape
    target = -np.ones(n_classes, dtype=float)
    target[int(label)] = 1.0
    lines = [
        "* Per-sample SPICE operating-point training step.",
        "* Forward class evidence, error, derivative, and updated programmable state are evaluated in ngspice.",
        f".param LR={lr:.12g}",
        "",
    ]
    for i, val in enumerate(x):
        lines.append(f"Vx{i} x{i} 0 DC {float(val):.12g}")
    lines.append("")
    for k in range(n_classes):
        for i in range(n_in):
            lines.append(f"Vw{k}_{i} w{k}_{i} 0 DC {weights[k, i]:.12g}")
        lines.append(f"Vb{k} b{k} 0 DC {bias[k]:.12g}")
        lines.append(f"Vt{k} t{k} 0 DC {target[k]:.12g}")
    lines.append("")
    for k in range(n_classes):
        terms = [f"V(w{k}_{i})*V(x{i})" for i in range(n_in)] + [f"V(b{k})"]
        summed = " + ".join(terms)
        lines.append(f"By{k} y{k} 0 V = 2/(1+exp(-2*({summed})))-1")
        lines.append(f"Be{k} e{k} 0 V = V(t{k})-V(y{k})")
        lines.append(f"Bd{k} d{k} 0 V = V(e{k})*(1-V(y{k})*V(y{k}))")
    lines.append("")
    for k in range(n_classes):
        for i in range(n_in):
            lines.append(f"Bnw{k}_{i} nw{k}_{i} 0 V = V(w{k}_{i}) + {{LR}}*V(d{k})*V(x{i})")
        lines.append(f"Bnb{k} nb{k} 0 V = V(b{k}) + {{LR}}*V(d{k})")
    vectors = [f"V(y{k})" for k in range(n_classes)]
    vectors += [f"V(nw{k}_{i})" for k in range(n_classes) for i in range(n_in)]
    vectors += [f"V(nb{k})" for k in range(n_classes)]
    lines += [
        "",
        ".control",
        "op",
        f"wrdata {out_path} " + " ".join(vectors),
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def make_eval_op_netlist(x: np.ndarray, weights: np.ndarray, bias: np.ndarray, out_path: Path) -> str:
    n_classes, n_in = weights.shape
    lines = ["* Per-sample SPICE operating-point inference.", ""]
    for i, val in enumerate(x):
        lines.append(f"Vx{i} x{i} 0 DC {float(val):.12g}")
    lines.append("")
    for k in range(n_classes):
        for i in range(n_in):
            lines.append(f"Vw{k}_{i} w{k}_{i} 0 DC {weights[k, i]:.12g}")
        lines.append(f"Vb{k} b{k} 0 DC {bias[k]:.12g}")
    lines.append("")
    for k in range(n_classes):
        terms = [f"V(w{k}_{i})*V(x{i})" for i in range(n_in)] + [f"V(b{k})"]
        lines.append(f"By{k} y{k} 0 V = 2/(1+exp(-2*({' + '.join(terms)})))-1")
    lines += [
        "",
        ".control",
        "op",
        f"wrdata {out_path} " + " ".join(f"V(y{k})" for k in range(n_classes)),
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def read_wrdata_row(path: Path, n_vec: int) -> np.ndarray:
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    row = arr[-1]
    return np.asarray([row[2 * i + 1] for i in range(n_vec)], dtype=float)


def run_step(
    spice_bin: str,
    generated: Path,
    tmp_data: Path,
    x: np.ndarray,
    label: int,
    weights: np.ndarray,
    bias: np.ndarray,
    lr: float,
    sample_idx: int,
    timeout: float,
) -> tuple[int, np.ndarray, np.ndarray]:
    netlist = generated / "spice_mnist_op_train_step.cir"
    netlist.write_text(make_train_op_netlist(x, label, weights, bias, lr, tmp_data))
    proc = subprocess.run([spice_bin, "-b", str(netlist)], text=True, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-3000:] or proc.stdout[-3000:])
    n_classes, n_in = weights.shape
    vals = read_wrdata_row(tmp_data, n_classes + n_classes * n_in + n_classes)
    pred = int(np.argmax(vals[:n_classes]))
    offset = n_classes
    next_weights = vals[offset : offset + n_classes * n_in].reshape(n_classes, n_in)
    next_bias = vals[offset + n_classes * n_in :]
    return pred, next_weights, next_bias


def run_eval(
    spice_bin: str,
    generated: Path,
    tmp_data: Path,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    timeout: float,
) -> float:
    correct = 0
    netlist = generated / "spice_mnist_op_eval_step.cir"
    for idx, (x, label) in enumerate(zip(x_eval, y_eval)):
        netlist.write_text(make_eval_op_netlist(x, weights, bias, tmp_data))
        proc = subprocess.run([spice_bin, "-b", str(netlist)], text=True, capture_output=True, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-3000:] or proc.stdout[-3000:])
        y = read_wrdata_row(tmp_data, weights.shape[0])
        correct += int(np.argmax(y) == int(label))
        if (idx + 1) % 100 == 0:
            print(f"eval {idx + 1}/{len(y_eval)}", flush=True)
    return correct / max(len(y_eval), 1)


def plot_curve(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(df["epoch"], df["train_accuracy"], marker="o", label="train")
    plt.plot(df["epoch"], df["heldout_accuracy"], marker="o", label="held-out")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-samples", type=int, default=200)
    ap.add_argument("--test-samples", type=int, default=200)
    ap.add_argument("--image-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=20)
    ap.add_argument("--tag", default="op")
    args = ap.parse_args()

    x_train, y_train, x_test, y_test = load_mnist_sequence(args.train_samples, args.test_samples, args.image_size, args.seed)
    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    generated.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)

    rng = np.random.default_rng(args.seed)
    weights = rng.normal(0.0, 0.02, size=(10, x_train.shape[1]))
    bias = np.zeros(10)
    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag)
    stem = f"spice_mnist_op_{safe_tag}"
    tmp_data = ROOT / f"spice/results/{stem}_step.dat"
    rows = []
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        order = np.arange(len(y_train))
        rng.shuffle(order)
        correct = 0
        epoch_start = time.perf_counter()
        for n, idx in enumerate(order):
            pred, weights, bias = run_step(
                spice_bin,
                generated,
                tmp_data,
                x_train[idx],
                int(y_train[idx]),
                weights,
                bias,
                args.lr,
                int(idx),
                args.timeout,
            )
            correct += int(pred == int(y_train[idx]))
            if (n + 1) % 100 == 0:
                print(f"train epoch {epoch + 1} sample {n + 1}/{len(order)}", flush=True)
        heldout = run_eval(spice_bin, generated, tmp_data, x_test, y_test, weights, bias, args.timeout)
        row = {
            "epoch": epoch + 1,
            "train_accuracy": correct / len(order),
            "heldout_accuracy": heldout,
            "epoch_wall_time_s": time.perf_counter() - epoch_start,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    curve = pd.DataFrame(rows)
    curve_path = ROOT / f"spice/results/{stem}_learning_curve.csv"
    curve.to_csv(curve_path, index=False)
    fig = ROOT / f"spice/results/{stem}_learning_curve.png"
    plot_curve(curve, fig)
    weights_path = ROOT / f"spice/results/{stem}_final_weights.npz"
    np.savez_compressed(weights_path, weights=weights, bias=bias)
    summary = {
        "simulator": version,
        "dataset": "MNIST train/test split, downsampled",
        "image_size": args.image_size,
        "inputs": int(x_train.shape[1]),
        "classes": 10,
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "epochs": args.epochs,
        "lr": args.lr,
        "wall_time_s": time.perf_counter() - t0,
        "learning_curve": str(curve_path),
        "figure": str(fig),
        "final_weights": str(weights_path),
        "heldout_test_accuracy": float(curve.iloc[-1]["heldout_accuracy"]),
        "best_heldout_accuracy": float(curve["heldout_accuracy"].max()),
        "note": (
            "Per-sample operating-point all-SPICE training: forward, error, derivative, and updated "
            "programmable state are computed by ngspice for each sample; Python carries measured state."
        ),
    }
    out = ROOT / f"spice/results/{stem}_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
