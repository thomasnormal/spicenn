from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_spice_mnist_train import load_mnist_sequence, pwl, read_wrdata, sample_accuracy, update_gate
from run_spice_sweep import ROOT, detect_spice, run_tiny_test


def make_netlist(x_train, y_train, hidden: int, epochs: int, lr: float, sample_period: float, trace_path: Path, seed: int) -> str:
    rng = np.random.default_rng(seed)
    n_samples, n_in = x_train.shape
    num_classes = 10
    x_seq = np.tile(x_train, (epochs, 1))
    y_seq = np.tile(y_train, epochs)
    n_steps = len(y_seq)
    targets = -np.ones((n_steps, num_classes), dtype=float)
    targets[np.arange(n_steps), y_seq] = 1.0
    tstop = n_steps * sample_period

    lines = [
        "* Entirely-in-SPICE MNIST hidden-layer MLP: pixels -> hidden tanh -> 10 tanh outputs.",
        f".param LR={lr:.12g}",
        ".param CW=1",
        f".param TSTOP={tstop:.12g}",
        "",
    ]
    for i in range(n_in):
        lines.append(f"Vx{i} x{i} 0 {pwl(x_seq[:, i], sample_period)}")
    for k in range(num_classes):
        lines.append(f"Vt{k} t{k} 0 {pwl(targets[:, k], sample_period)}")
    lines.append(f"Vu upd 0 {update_gate(n_steps, sample_period)}")
    lines.append("")

    wh = rng.normal(0.0, 0.08, size=(hidden, n_in))
    vo = rng.normal(0.0, 0.08, size=(num_classes, hidden))
    for j in range(hidden):
        for i in range(n_in):
            lines.append(f"Cwh{j}_{i} wh{j}_{i} 0 {{CW}} IC={wh[j, i]:.8g}")
            lines.append(f"Rwh{j}_{i} wh{j}_{i} 0 1e15")
        lines.append(f"Cbh{j} bh{j} 0 {{CW}} IC=0")
        lines.append(f"Rbh{j} bh{j} 0 1e15")
    for k in range(num_classes):
        for j in range(hidden):
            lines.append(f"Cv{k}_{j} v{k}_{j} 0 {{CW}} IC={vo[k, j]:.8g}")
            lines.append(f"Rv{k}_{j} v{k}_{j} 0 1e15")
        lines.append(f"Cbo{k} bo{k} 0 {{CW}} IC=0")
        lines.append(f"Rbo{k} bo{k} 0 1e15")
    lines.append("")

    for j in range(hidden):
        terms = [f"V(wh{j}_{i})*V(x{i})" for i in range(n_in)] + [f"V(bh{j})"]
        lines.append(f"Bh{j} h{j} 0 V = 2/(1+exp(-2*({' + '.join(terms)})))-1")
    for k in range(num_classes):
        terms = [f"V(v{k}_{j})*V(h{j})" for j in range(hidden)] + [f"V(bo{k})"]
        lines.append(f"By{k} y{k} 0 V = 2/(1+exp(-2*({' + '.join(terms)})))-1")
        lines.append(f"Be{k} e{k} 0 V = V(t{k})-V(y{k})")
        lines.append(f"Bd{k} d{k} 0 V = V(e{k})*(1-V(y{k})*V(y{k}))")
    for j in range(hidden):
        fb = " + ".join([f"V(v{k}_{j})*V(d{k})" for k in range(num_classes)])
        lines.append(f"Bdh{j} dh{j} 0 V = (1-V(h{j})*V(h{j}))*({fb})")
    lines.append("")

    for k in range(num_classes):
        for j in range(hidden):
            lines.append(f"Buv{k}_{j} 0 v{k}_{j} I = {{LR}}*V(upd)*V(d{k})*V(h{j})")
        lines.append(f"Bubo{k} 0 bo{k} I = {{LR}}*V(upd)*V(d{k})")
    for j in range(hidden):
        for i in range(n_in):
            lines.append(f"Buwh{j}_{i} 0 wh{j}_{i} I = {{LR}}*V(upd)*V(dh{j})*V(x{i})")
        lines.append(f"Bubh{j} 0 bh{j} I = {{LR}}*V(upd)*V(dh{j})")

    measure_time = tstop - sample_period / 80
    measures = []
    for j in range(hidden):
        for i in range(n_in):
            measures.append(f"meas tran fwh{j}x{i} FIND V(wh{j}_{i}) AT={measure_time:.12g}")
        measures.append(f"meas tran fbh{j} FIND V(bh{j}) AT={measure_time:.12g}")
    for k in range(num_classes):
        for j in range(hidden):
            measures.append(f"meas tran fv{k}h{j} FIND V(v{k}_{j}) AT={measure_time:.12g}")
        measures.append(f"meas tran fbo{k} FIND V(bo{k}) AT={measure_time:.12g}")

    lines += [
        "",
        f".tran {sample_period/80:.12g} {{TSTOP}} uic",
        ".control",
        "run",
        *measures,
    ]
    vectors = [f"V(y{k})" for k in range(num_classes)] + [f"V(t{k})" for k in range(num_classes)]
    lines.append(f"wrdata {trace_path} " + " ".join(vectors))
    lines += [".endc", ".end", ""]
    return "\n".join(lines)


def parse_weights(text: str, hidden: int, n_in: int) -> dict[str, np.ndarray]:
    wh = np.zeros((hidden, n_in))
    bh = np.zeros(hidden)
    vo = np.zeros((10, hidden))
    bo = np.zeros(10)
    def get(name):
        m = re.search(rf"(?im)^\s*{name}\s*=\s*([-+0-9.eE]+)", text)
        if not m:
            raise ValueError(f"missing {name}")
        return float(m.group(1))
    for j in range(hidden):
        for i in range(n_in):
            wh[j, i] = get(f"fwh{j}x{i}")
        bh[j] = get(f"fbh{j}")
    for k in range(10):
        for j in range(hidden):
            vo[k, j] = get(f"fv{k}h{j}")
        bo[k] = get(f"fbo{k}")
    return {"wh": wh, "bh": bh, "vo": vo, "bo": bo}


def make_eval_netlist(x_eval, y_eval, weights: dict[str, np.ndarray], sample_period: float, trace_path: Path) -> str:
    hidden, n_in = weights["wh"].shape
    n_samples = len(y_eval)
    targets = -np.ones((n_samples, 10), dtype=float)
    targets[np.arange(n_samples), y_eval] = 1.0
    lines = [f".param TSTOP={n_samples * sample_period:.12g}", ""]
    for i in range(n_in):
        lines.append(f"Vx{i} x{i} 0 {pwl(x_eval[:, i], sample_period)}")
    for k in range(10):
        lines.append(f"Vt{k} t{k} 0 {pwl(targets[:, k], sample_period)}")
    for j in range(hidden):
        for i in range(n_in):
            lines.append(f"Vwh{j}_{i} wh{j}_{i} 0 DC {weights['wh'][j, i]:.12g}")
        lines.append(f"Vbh{j} bh{j} 0 DC {weights['bh'][j]:.12g}")
    for k in range(10):
        for j in range(hidden):
            lines.append(f"Vv{k}_{j} v{k}_{j} 0 DC {weights['vo'][k, j]:.12g}")
        lines.append(f"Vbo{k} bo{k} 0 DC {weights['bo'][k]:.12g}")
    for j in range(hidden):
        terms = [f"V(wh{j}_{i})*V(x{i})" for i in range(n_in)] + [f"V(bh{j})"]
        lines.append(f"Bh{j} h{j} 0 V = 2/(1+exp(-2*({' + '.join(terms)})))-1")
    for k in range(10):
        terms = [f"V(v{k}_{j})*V(h{j})" for j in range(hidden)] + [f"V(bo{k})"]
        lines.append(f"By{k} y{k} 0 V = 2/(1+exp(-2*({' + '.join(terms)})))-1")
    lines += [f".tran {sample_period/80:.12g} {n_samples * sample_period:.12g}", ".control", "run"]
    vectors = [f"V(y{k})" for k in range(10)] + [f"V(t{k})" for k in range(10)]
    lines.append(f"wrdata {trace_path} " + " ".join(vectors))
    lines += [".endc", ".end", ""]
    return "\n".join(lines)


def train_curve(trace: pd.DataFrame, labels: np.ndarray, epochs: int, sample_period: float) -> pd.DataFrame:
    rows = []
    n = len(labels)
    for epoch in range(epochs):
        acc = sample_accuracy(trace.iloc[:], np.tile(labels, epochs)[epoch * n:(epoch + 1) * n], sample_period)
        # sample_accuracy assumes trace starts at sample 0, so use explicit loop for epoch offset.
        correct = 0
        losses = []
        for j in range(n):
            idx = epoch * n + j
            t = (idx + 0.9) * sample_period
            row = trace.iloc[(trace["time"] - t).abs().argmin()]
            y = np.asarray([row[f"v{k}"] for k in range(10)])
            target = -np.ones(10)
            target[labels[j]] = 1.0
            correct += int(np.argmax(y) == labels[j])
            losses.append(float(np.mean((target - y) ** 2)))
        rows.append({"epoch": epoch + 1, "train_accuracy": correct / n, "mse": float(np.mean(losses))})
    return pd.DataFrame(rows)


def plot_curve(curve: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(curve["epoch"], curve["train_accuracy"], marker="o")
    plt.xlabel("epoch")
    plt.ylabel("SPICE train accuracy")
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-samples", type=int, default=80)
    ap.add_argument("--test-samples", type=int, default=80)
    ap.add_argument("--image-size", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e4)
    ap.add_argument("--sample-period", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="mlp")
    args = ap.parse_args()

    x_train, y_train, x_test, y_test = load_mnist_sequence(args.train_samples, args.test_samples, args.image_size, args.seed)
    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    generated.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)
    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag)
    stem = f"spice_mnist_mlp_{safe_tag}"
    trace_path = ROOT / f"spice/results/{stem}_trace.dat"
    netlist = generated / f"{stem}.cir"
    netlist.write_text(make_netlist(x_train, y_train, args.hidden, args.epochs, args.lr, args.sample_period, trace_path, args.seed))
    proc = subprocess.run([spice_bin, "-b", str(netlist)], text=True, capture_output=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-3000:] or proc.stdout[-3000:])
    trace = read_wrdata(trace_path, 20)
    trace_csv = ROOT / f"spice/results/{stem}_trace.csv"
    trace.to_csv(trace_csv, index=False)
    curve = train_curve(trace, y_train, args.epochs, args.sample_period)
    curve_csv = ROOT / f"spice/results/{stem}_learning_curve.csv"
    curve.to_csv(curve_csv, index=False)
    fig = ROOT / f"spice/results/{stem}_learning_curve.png"
    plot_curve(curve, fig)
    weights = parse_weights(proc.stdout + "\n" + proc.stderr, args.hidden, x_train.shape[1])
    weights_path = ROOT / f"spice/results/{stem}_final_weights.npz"
    np.savez_compressed(weights_path, **weights)

    eval_trace_path = ROOT / f"spice/results/{stem}_eval_trace.dat"
    eval_netlist = generated / f"{stem}_eval.cir"
    eval_netlist.write_text(make_eval_netlist(x_test, y_test, weights, args.sample_period, eval_trace_path))
    eval_proc = subprocess.run([spice_bin, "-b", str(eval_netlist)], text=True, capture_output=True, timeout=120)
    if eval_proc.returncode != 0:
        raise RuntimeError(eval_proc.stderr[-3000:] or eval_proc.stdout[-3000:])
    eval_trace = read_wrdata(eval_trace_path, 20)
    eval_csv = ROOT / f"spice/results/{stem}_eval_trace.csv"
    eval_trace.to_csv(eval_csv, index=False)
    test_acc = sample_accuracy(eval_trace, y_test, args.sample_period)

    summary = {
        "simulator": version,
        "netlist": str(netlist),
        "eval_netlist": str(eval_netlist),
        "trace": str(trace_csv),
        "eval_trace": str(eval_csv),
        "learning_curve": str(curve_csv),
        "figure": str(fig),
        "final_weights": str(weights_path),
        "dataset": "MNIST train/test split, downsampled",
        "image_size": args.image_size,
        "inputs": int(x_train.shape[1]),
        "hidden": args.hidden,
        "classes": 10,
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "epochs": args.epochs,
        "lr": args.lr,
        "final": curve.iloc[-1].to_dict(),
        "heldout_test_accuracy": test_acc,
        "note": "Hidden-layer MNIST MLP trained and evaluated with forward, backward, and updates in ngspice.",
    }
    out = ROOT / f"spice/results/{stem}_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
