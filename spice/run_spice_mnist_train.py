from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from run_spice_sweep import ROOT, detect_spice, prepare_netlist_for_simulator, run_tiny_test, run_simulator_netlist


def pwl(values: np.ndarray, sample_period: float, edge: float = 1e-9) -> str:
    vals = [float(v) for v in values]
    pts: list[tuple[float, float]] = [(0.0, vals[0])]
    for i, val in enumerate(vals):
        t0 = i * sample_period
        pts.append((t0 + edge, val))
        pts.append(((i + 1) * sample_period - edge, val))
        if i + 1 < len(vals):
            pts.append(((i + 1) * sample_period, vals[i + 1]))
    return "PWL(" + " ".join(f"{t:.12g} {v:.12g}" for t, v in pts) + ")"


def update_gate(n_samples: int, sample_period: float, settle_frac: float = 0.45, off_frac: float = 0.95) -> str:
    pts: list[tuple[float, float]] = [(0.0, 0.0)]
    for i in range(n_samples):
        t0 = i * sample_period
        pts += [
            (t0, 0.0),
            (t0 + settle_frac * sample_period, 0.0),
            (t0 + settle_frac * sample_period + 1e-9, 1.0),
            (t0 + off_frac * sample_period, 1.0),
            (t0 + off_frac * sample_period + 1e-9, 0.0),
            ((i + 1) * sample_period, 0.0),
        ]
    return "PWL(" + " ".join(f"{t:.12g} {v:.12g}" for t, v in pts) + ")"


def mnist_index_splits(n_train: int, n_test: int, train_count: int, test_count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Choose train/test samples from independent RNG streams.

    Keeping streams independent makes eval slices comparable when only the
    online training horizon changes. Taking prefixes from full permutations
    also makes shorter training horizons match the prefix of longer runs.
    """
    if n_train < 0 or n_test < 0:
        raise ValueError("sample counts must be non-negative")
    if n_train > train_count or n_test > test_count:
        raise ValueError("requested MNIST sample count exceeds dataset size")
    seed_sequence = np.random.SeedSequence(seed)
    train_seed, test_seed = seed_sequence.spawn(2)
    train_idx = np.random.default_rng(train_seed).permutation(train_count)[:n_train]
    test_idx = np.random.default_rng(test_seed).permutation(test_count)[:n_test]
    return train_idx, test_idx


def load_mnist_sequence(n_train: int, n_test: int, image_size: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from torchvision import datasets, transforms

    ds_train = datasets.MNIST(root=str(ROOT / "data"), train=True, download=True, transform=transforms.ToTensor())
    ds_test = datasets.MNIST(root=str(ROOT / "data"), train=False, download=True, transform=transforms.ToTensor())
    train_idx, test_idx = mnist_index_splits(n_train, n_test, len(ds_train), len(ds_test), seed)

    def extract(ds, idx):
        xs, ys = [], []
        if len(idx) == 0:
            return np.zeros((0, image_size * image_size), dtype=np.float64), np.zeros((0,), dtype=int)
        for i in idx:
            x, y = ds[int(i)]
            x = F.interpolate(x.unsqueeze(0), size=(image_size, image_size), mode="area").squeeze()
            x = (x.reshape(-1).numpy().astype(np.float64) - 0.1307) / 0.3081
            x = np.clip(x / 3.0, -1.0, 1.0)
            xs.append(x)
            ys.append(int(y))
        return np.stack(xs), np.asarray(ys, dtype=int)

    return *extract(ds_train, train_idx), *extract(ds_test, test_idx)


def make_netlist(
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    lr: float,
    sample_period: float,
    trace_path: Path,
    seed: int,
    initial_weights: np.ndarray | None = None,
    initial_bias: np.ndarray | None = None,
) -> str:
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
        "* Entirely-in-SPICE MNIST training demo: downsampled pixels -> 10 tanh outputs.",
        "* Data cycling, forward pass, backward/error voltages, and conductance-state updates happen in ngspice.",
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

    init = rng.normal(0.0, 0.02, size=(num_classes, n_in)) if initial_weights is None else np.asarray(initial_weights)
    init_bias = np.zeros(num_classes) if initial_bias is None else np.asarray(initial_bias)
    for k in range(num_classes):
        for i in range(n_in):
            lines.append(f"Cw{k}_{i} w{k}_{i} 0 {{CW}} IC={init[k, i]:.8g}")
            lines.append(f"Rw{k}_{i} w{k}_{i} 0 1e15")
        lines.append(f"Cb{k} b{k} 0 {{CW}} IC={init_bias[k]:.8g}")
        lines.append(f"Rb{k} b{k} 0 1e15")
    lines.append("")

    for k in range(num_classes):
        terms = [f"V(w{k}_{i})*V(x{i})" for i in range(n_in)] + [f"V(b{k})"]
        summed = " + ".join(terms)
        lines.append(f"By{k} y{k} 0 V = 2/(1+exp(-2*({summed})))-1")
        lines.append(f"Be{k} e{k} 0 V = V(t{k})-V(y{k})")
        lines.append(f"Bd{k} d{k} 0 V = V(e{k})*(1-V(y{k})*V(y{k}))")
    lines.append("")

    for k in range(num_classes):
        for i in range(n_in):
            lines.append(f"Buw{k}_{i} 0 w{k}_{i} I = {{LR}}*V(upd)*V(d{k})*V(x{i})")
        lines.append(f"Bub{k} 0 b{k} I = {{LR}}*V(upd)*V(d{k})")
    measure_lines = []
    measure_time = tstop - sample_period / 80
    for k in range(num_classes):
        for i in range(n_in):
            measure_lines.append(f"meas tran fw{k}x{i} FIND V(w{k}_{i}) AT={measure_time:.12g}")
        measure_lines.append(f"meas tran fb{k} FIND V(b{k}) AT={measure_time:.12g}")
    lines += [
        "",
        f".tran {sample_period/80:.12g} {{TSTOP}} uic",
    ]
    lines += [
        ".control",
        "run",
        *measure_lines,
    ]
    vectors = [f"V(y{k})" for k in range(num_classes)] + [f"V(t{k})" for k in range(num_classes)]
    lines.append(f"wrdata {trace_path} " + " ".join(vectors))
    lines += [".endc", ".end", ""]
    return "\n".join(lines)


def parse_final_weights(stdout: str, n_in: int, num_classes: int = 10) -> tuple[np.ndarray, np.ndarray]:
    weights = np.zeros((num_classes, n_in), dtype=float)
    bias = np.zeros(num_classes, dtype=float)
    for k in range(num_classes):
        for i in range(n_in):
            name = f"fw{k}x{i}"
            m = re.search(rf"(?im)^\s*{name}\s*=\s*([-+0-9.eE]+)", stdout)
            if not m:
                raise ValueError(f"missing final weight measurement {name}")
            weights[k, i] = float(m.group(1))
        name = f"fb{k}"
        m = re.search(rf"(?im)^\s*{name}\s*=\s*([-+0-9.eE]+)", stdout)
        if not m:
            raise ValueError(f"missing final bias measurement {name}")
        bias[k] = float(m.group(1))
    return weights, bias


def make_eval_netlist(
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    sample_period: float,
    trace_path: Path,
) -> str:
    n_samples, n_in = x_eval.shape
    num_classes = 10
    targets = -np.ones((n_samples, num_classes), dtype=float)
    targets[np.arange(n_samples), y_eval] = 1.0
    tstop = n_samples * sample_period
    lines = [
        "* SPICE MNIST evaluation netlist using weights learned by training netlist.",
        f".param TSTOP={tstop:.12g}",
        "",
    ]
    for i in range(n_in):
        lines.append(f"Vx{i} x{i} 0 {pwl(x_eval[:, i], sample_period)}")
    for k in range(num_classes):
        lines.append(f"Vt{k} t{k} 0 {pwl(targets[:, k], sample_period)}")
    lines.append("")
    for k in range(num_classes):
        for i in range(n_in):
            lines.append(f"Vw{k}_{i} w{k}_{i} 0 DC {weights[k, i]:.12g}")
        lines.append(f"Vb{k} b{k} 0 DC {bias[k]:.12g}")
    lines.append("")
    for k in range(num_classes):
        terms = [f"V(w{k}_{i})*V(x{i})" for i in range(n_in)] + [f"V(b{k})"]
        lines.append(f"By{k} y{k} 0 V = 2/(1+exp(-2*({' + '.join(terms)})))-1")
    lines += [
        f".tran {sample_period/80:.12g} {{TSTOP}}",
        ".control",
        "run",
    ]
    vectors = [f"V(y{k})" for k in range(num_classes)] + [f"V(t{k})" for k in range(num_classes)]
    lines.append(f"wrdata {trace_path} " + " ".join(vectors))
    lines += [".endc", ".end", ""]
    return "\n".join(lines)


def read_wrdata(path: Path, n_vec: int) -> pd.DataFrame:
    arr = np.loadtxt(path)
    data = {"time": arr[:, 0]}
    for i in range(n_vec):
        data[f"v{i}"] = arr[:, 2 * i + 1]
    return pd.DataFrame(data)


def sample_train_curve(trace: pd.DataFrame, y_train: np.ndarray, epochs: int, sample_period: float) -> pd.DataFrame:
    rows = []
    n_train = len(y_train)
    for epoch in range(epochs):
        correct = 0
        losses = []
        for j in range(n_train):
            idx = epoch * n_train + j
            t = (idx + 0.9) * sample_period
            row = trace.iloc[(trace["time"] - t).abs().argmin()]
            y = np.asarray([row[f"v{k}"] for k in range(10)])
            target = -np.ones(10)
            target[y_train[j]] = 1.0
            correct += int(np.argmax(y) == y_train[j])
            losses.append(float(np.mean((target - y) ** 2)))
        rows.append({"epoch": epoch + 1, "train_accuracy": correct / n_train, "mse": float(np.mean(losses))})
    return pd.DataFrame(rows)


def sample_accuracy(trace: pd.DataFrame, labels: np.ndarray, sample_period: float) -> float:
    correct = 0
    for j, label in enumerate(labels):
        t = (j + 0.9) * sample_period
        row = trace.iloc[(trace["time"] - t).abs().argmin()]
        y = np.asarray([row[f"v{k}"] for k in range(10)])
        correct += int(np.argmax(y) == label)
    return correct / max(len(labels), 1)


def plot_curve(curve: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(curve["epoch"], curve["train_accuracy"], marker="o")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("SPICE train accuracy")
    ax1.set_ylim(-0.05, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(curve["epoch"], curve["mse"], color="tab:red", alpha=0.7)
    ax2.set_ylabel("MSE")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-samples", type=int, default=80)
    ap.add_argument("--test-samples", type=int, default=80)
    ap.add_argument("--image-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=8e4)
    ap.add_argument("--sample-period", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="default")
    args = ap.parse_args()

    x_train, y_train, _x_test, _y_test = load_mnist_sequence(args.train_samples, args.test_samples, args.image_size, args.seed)
    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    generated.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)

    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag)
    stem = "spice_mnist_train" if safe_tag == "default" else f"spice_mnist_train_{safe_tag}"
    trace_path = ROOT / f"spice/results/{stem}_trace.dat"
    netlist_path = generated / f"{stem}.cir"
    netlist_path.write_text(
        prepare_netlist_for_simulator(
            make_netlist(x_train, y_train, args.epochs, args.lr, args.sample_period, trace_path, args.seed),
            spice_bin,
        )
    )
    proc = run_simulator_netlist(spice_bin, netlist_path, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-3000:] or proc.stdout[-3000:])

    trace = read_wrdata(trace_path, 20)
    trace_csv = ROOT / f"spice/results/{stem}_trace.csv"
    trace.to_csv(trace_csv, index=False)
    curve = sample_train_curve(trace, y_train, args.epochs, args.sample_period)
    curve_csv = ROOT / f"spice/results/{stem}_learning_curve.csv"
    curve.to_csv(curve_csv, index=False)
    fig = ROOT / f"spice/results/{stem}_learning_curve.png"
    plot_curve(curve, fig)
    weights, bias = parse_final_weights(proc.stdout + "\n" + proc.stderr, x_train.shape[1], 10)
    weights_path = ROOT / f"spice/results/{stem}_final_weights.npz"
    np.savez_compressed(weights_path, weights=weights, bias=bias)

    test_accuracy = None
    eval_netlist_path = None
    eval_trace_csv = None
    if len(_y_test) > 0:
        eval_trace_path = ROOT / f"spice/results/{stem}_eval_trace.dat"
        eval_netlist_path = generated / f"{stem}_eval.cir"
        eval_netlist_path.write_text(
            prepare_netlist_for_simulator(
                make_eval_netlist(_x_test, _y_test, weights, bias, args.sample_period, eval_trace_path),
                spice_bin,
            )
        )
        eval_proc = run_simulator_netlist(spice_bin, eval_netlist_path, timeout=120)
        if eval_proc.returncode != 0:
            raise RuntimeError(eval_proc.stderr[-3000:] or eval_proc.stdout[-3000:])
        eval_trace = read_wrdata(eval_trace_path, 20)
        eval_trace_csv = ROOT / f"spice/results/{stem}_eval_trace.csv"
        eval_trace.to_csv(eval_trace_csv, index=False)
        test_accuracy = sample_accuracy(eval_trace, _y_test, args.sample_period)
    summary = {
        "simulator": version,
        "netlist": str(netlist_path),
        "trace": str(trace_csv),
        "learning_curve": str(curve_csv),
        "figure": str(fig),
        "final_weights": str(weights_path),
        "eval_netlist": str(eval_netlist_path) if eval_netlist_path else None,
        "eval_trace": str(eval_trace_csv) if eval_trace_csv else None,
        "dataset": "MNIST train split, downsampled",
        "image_size": args.image_size,
        "inputs": int(x_train.shape[1]),
        "classes": 10,
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "epochs": args.epochs,
        "sample_period_s": args.sample_period,
        "lr": args.lr,
        "final": curve.iloc[-1].to_dict(),
        "heldout_test_accuracy": test_accuracy,
        "note": "MNIST sample cycling, forward outputs, class errors, and weight update currents were simulated inside ngspice.",
    }
    summary_path = ROOT / f"spice/results/{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
