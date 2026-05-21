from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_spice_mnist_train import (
    load_mnist_sequence,
    make_eval_netlist,
    make_netlist,
    parse_final_weights,
    read_wrdata,
    sample_accuracy,
)
from run_spice_sweep import ROOT, detect_spice, prepare_netlist_for_simulator, run_tiny_test, run_simulator_netlist


def evaluate(
    spice_bin: str,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    sample_period: float,
    stem: str,
    generated: Path,
) -> tuple[float, Path]:
    trace_path = ROOT / f"spice/results/{stem}_eval_trace.dat"
    netlist_path = generated / f"{stem}_eval.cir"
    netlist_path.write_text(
        prepare_netlist_for_simulator(make_eval_netlist(x_eval, y_eval, weights, bias, sample_period, trace_path), spice_bin)
    )
    proc = run_simulator_netlist(spice_bin, netlist_path, timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-3000:] or proc.stdout[-3000:])
    trace = read_wrdata(trace_path, 20)
    trace_csv = ROOT / f"spice/results/{stem}_eval_trace.csv"
    trace.to_csv(trace_csv, index=False)
    return sample_accuracy(trace, y_eval, sample_period), trace_csv


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
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--chunk-size", type=int, default=40)
    ap.add_argument("--lr", type=float, default=8e4)
    ap.add_argument("--sample-period", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--tag", default="stream")
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
    stem = f"spice_mnist_stream_{safe_tag}"
    rows = []
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        order = np.arange(len(y_train))
        rng.shuffle(order)
        epoch_start = time.perf_counter()
        for chunk_idx, start in enumerate(range(0, len(order), args.chunk_size)):
            idx = order[start : start + args.chunk_size]
            chunk_stem = f"{stem}_e{epoch + 1:02d}_c{chunk_idx + 1:03d}"
            trace_path = ROOT / f"spice/results/{chunk_stem}_trace.dat"
            netlist_path = generated / f"{chunk_stem}.cir"
            netlist_path.write_text(
                prepare_netlist_for_simulator(
                    make_netlist(
                        x_train[idx],
                        y_train[idx],
                        epochs=1,
                        lr=args.lr,
                        sample_period=args.sample_period,
                        trace_path=trace_path,
                        seed=args.seed + epoch * 1000 + chunk_idx,
                        initial_weights=weights,
                        initial_bias=bias,
                    ),
                    spice_bin,
                )
            )
            proc = run_simulator_netlist(spice_bin, netlist_path, timeout=args.timeout)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr[-3000:] or proc.stdout[-3000:])
            weights, bias = parse_final_weights(proc.stdout + "\n" + proc.stderr, x_train.shape[1], 10)
        heldout_acc, eval_trace = evaluate(
            spice_bin,
            x_test,
            y_test,
            weights,
            bias,
            args.sample_period,
            f"{stem}_epoch{epoch + 1:02d}",
            generated,
        )
        row = {
            "epoch": epoch + 1,
            "heldout_accuracy": heldout_acc,
            "epoch_wall_time_s": time.perf_counter() - epoch_start,
            "eval_trace": str(eval_trace),
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
        "chunk_size": args.chunk_size,
        "lr": args.lr,
        "sample_period_s": args.sample_period,
        "wall_time_s": time.perf_counter() - t0,
        "learning_curve": str(curve_path),
        "figure": str(fig),
        "final_weights": str(weights_path),
        "heldout_test_accuracy": float(curve.iloc[-1]["heldout_accuracy"]),
        "best_heldout_accuracy": float(curve["heldout_accuracy"].max()),
        "note": (
            "Chunked all-SPICE training: each chunk runs forward, class-error, and weight-update currents "
            "inside ngspice; Python only transfers final capacitor-state measurements to the next chunk."
        ),
    }
    out = ROOT / f"spice/results/{stem}_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
