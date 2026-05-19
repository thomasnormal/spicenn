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


def read_wrdata_row(path: Path, n_vec: int) -> np.ndarray:
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    row = arr[-1]
    return np.asarray([row[2 * i + 1] for i in range(n_vec)], dtype=float)


def make_batch_train_netlist(
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    lr: float,
    out_path: Path,
) -> str:
    batch, n_in = x_batch.shape
    n_classes = weights.shape[0]
    lines = [
        "* Batch operating-point SPICE training step.",
        "* ngspice computes forward outputs, batch errors, and updated programmable state.",
        f".param LR={lr:.12g}",
        f".param BS={batch}",
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
        for i in range(n_in):
            lines.append(f"Vw{k}_{i} w{k}_{i} 0 DC {weights[k, i]:.12g}")
        lines.append(f"Vb{k} b{k} 0 DC {bias[k]:.12g}")
    lines.append("")
    for s in range(batch):
        for k in range(n_classes):
            terms = [f"V(w{k}_{i})*V(x{s}_{i})" for i in range(n_in)] + [f"V(b{k})"]
            summed = " + ".join(terms)
            lines.append(f"By{s}_{k} y{s}_{k} 0 V = 2/(1+exp(-2*({summed})))-1")
            lines.append(f"Be{s}_{k} e{s}_{k} 0 V = V(t{s}_{k})-V(y{s}_{k})")
            lines.append(f"Bd{s}_{k} d{s}_{k} 0 V = V(e{s}_{k})*(1-V(y{s}_{k})*V(y{s}_{k}))")
    lines.append("")
    for k in range(n_classes):
        for i in range(n_in):
            grad = " + ".join(f"V(d{s}_{k})*V(x{s}_{i})" for s in range(batch))
            lines.append(f"Bnw{k}_{i} nw{k}_{i} 0 V = V(w{k}_{i}) + {{LR}}*(({grad})/{{BS}})")
        grad_b = " + ".join(f"V(d{s}_{k})" for s in range(batch))
        lines.append(f"Bnb{k} nb{k} 0 V = V(b{k}) + {{LR}}*(({grad_b})/{{BS}})")
    vectors = [f"V(nw{k}_{i})" for k in range(n_classes) for i in range(n_in)]
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


def make_batch_eval_netlist(x_batch: np.ndarray, weights: np.ndarray, bias: np.ndarray, out_path: Path) -> str:
    batch, n_in = x_batch.shape
    n_classes = weights.shape[0]
    lines = ["* Batch operating-point SPICE inference.", ""]
    for s in range(batch):
        for i in range(n_in):
            lines.append(f"Vx{s}_{i} x{s}_{i} 0 DC {float(x_batch[s, i]):.12g}")
    lines.append("")
    for k in range(n_classes):
        for i in range(n_in):
            lines.append(f"Vw{k}_{i} w{k}_{i} 0 DC {weights[k, i]:.12g}")
        lines.append(f"Vb{k} b{k} 0 DC {bias[k]:.12g}")
    lines.append("")
    for s in range(batch):
        for k in range(n_classes):
            terms = [f"V(w{k}_{i})*V(x{s}_{i})" for i in range(n_in)] + [f"V(b{k})"]
            lines.append(f"By{s}_{k} y{s}_{k} 0 V = 2/(1+exp(-2*({' + '.join(terms)})))-1")
    vectors = [f"V(y{s}_{k})" for s in range(batch) for k in range(n_classes)]
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


def run_train_batch(
    spice_bin: str,
    netlist_path: Path,
    data_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    lr: float,
    timeout: float,
) -> tuple[np.ndarray, np.ndarray]:
    netlist_path.write_text(make_batch_train_netlist(x, y, weights, bias, lr, data_path))
    proc = subprocess.run([spice_bin, "-b", str(netlist_path)], text=True, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-3000:] or proc.stdout[-3000:])
    n_classes, n_in = weights.shape
    vals = read_wrdata_row(data_path, n_classes * n_in + n_classes)
    next_weights = vals[: n_classes * n_in].reshape(n_classes, n_in)
    next_bias = vals[n_classes * n_in :]
    return next_weights, next_bias


def run_eval(
    spice_bin: str,
    netlist_path: Path,
    data_path: Path,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    batch_size: int,
    timeout: float,
) -> float:
    correct = 0
    for start in range(0, len(y_eval), batch_size):
        x = x_eval[start : start + batch_size]
        y = y_eval[start : start + batch_size]
        netlist_path.write_text(make_batch_eval_netlist(x, weights, bias, data_path))
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
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--tag", default="batch_op")
    args = ap.parse_args()

    x_train, y_train, x_test, y_test = load_mnist_sequence(args.train_samples, args.test_samples, args.image_size, args.seed)
    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    generated.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)
    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag)
    stem = f"spice_mnist_batch_op_{safe_tag}"
    netlist_path = generated / f"{stem}_step.cir"
    data_path = ROOT / f"spice/results/{stem}_step.dat"
    rng = np.random.default_rng(args.seed)
    weights = rng.normal(0.0, 0.02, size=(10, x_train.shape[1]))
    bias = np.zeros(10)
    rows = []
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        order = np.arange(len(y_train))
        rng.shuffle(order)
        epoch_start = time.perf_counter()
        for n, start in enumerate(range(0, len(order), args.batch_size)):
            idx = order[start : start + args.batch_size]
            weights, bias = run_train_batch(
                spice_bin,
                netlist_path,
                data_path,
                x_train[idx],
                y_train[idx],
                weights,
                bias,
                args.lr,
                args.timeout,
            )
            if (n + 1) % 5 == 0:
                print(f"epoch {epoch + 1} batch {n + 1}", flush=True)
        heldout = run_eval(
            spice_bin,
            generated / f"{stem}_eval.cir",
            data_path,
            x_test,
            y_test,
            weights,
            bias,
            args.batch_size,
            args.timeout,
        )
        row = {
            "epoch": epoch + 1,
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
        "batch_size": args.batch_size,
        "lr": args.lr,
        "wall_time_s": time.perf_counter() - t0,
        "learning_curve": str(curve_path),
        "figure": str(fig),
        "final_weights": str(weights_path),
        "heldout_test_accuracy": float(curve.iloc[-1]["heldout_accuracy"]),
        "best_heldout_accuracy": float(curve["heldout_accuracy"].max()),
        "note": (
            "Batch operating-point all-SPICE training: ngspice computes batch forward outputs, errors, "
            "and programmable-state updates; Python carries measured state between batches."
        ),
    }
    out = ROOT / f"spice/results/{stem}_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
