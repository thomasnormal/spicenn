from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier

from run_spice_sweep import ROOT, detect_spice, run_tiny_test


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


def load_mnist(train_samples: int, test_samples: int, image_size: int, seed: int):
    from torchvision import datasets, transforms

    train = datasets.MNIST(root=str(ROOT / "data"), train=True, download=False, transform=transforms.ToTensor())
    test = datasets.MNIST(root=str(ROOT / "data"), train=False, download=False, transform=transforms.ToTensor())
    rng = np.random.default_rng(seed)
    train_idx = rng.choice(len(train), size=train_samples, replace=False)
    test_idx = rng.choice(len(test), size=test_samples, replace=False)

    def extract(ds, idx):
        xs, ys = [], []
        for i in idx:
            x, y = ds[int(i)]
            x = F.interpolate(x.unsqueeze(0), size=(image_size, image_size), mode="area").reshape(-1)
            xs.append(x.numpy().astype(np.float64))
            ys.append(int(y))
        return np.stack(xs), np.asarray(ys, dtype=int)

    return *extract(train, train_idx), *extract(test, test_idx)


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    hidden: int,
    epochs: int,
    seed: int,
) -> MLPClassifier:
    model = MLPClassifier(
        hidden_layer_sizes=(hidden,),
        activation="tanh",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=epochs,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    return model


def make_netlist(
    x_eval: np.ndarray,
    w0: np.ndarray,
    b0: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    sample_period: float,
    trace_path: Path,
) -> str:
    n_samples, n_in = x_eval.shape
    hidden = w0.shape[1]
    n_classes = w1.shape[1]
    tstop = n_samples * sample_period
    lines = [
        "* MNIST SPICE inference benchmark: 14x14 pixels -> tanh hidden layer -> 10 class evidence nodes.",
        "* Weights are trained offline, then evaluated by ngspice behavioral weighted sums.",
        "* This is SPICE forward inference, not SPICE training.",
        f".param TSTOP={tstop:.12g}",
        "",
    ]
    for i in range(n_in):
        lines.append(f"Vx{i} x{i} 0 {pwl(x_eval[:, i], sample_period)}")
    lines.append("")
    for j in range(hidden):
        terms = [f"({w0[i, j]:.12g})*V(x{i})" for i in range(n_in)]
        terms.append(f"({b0[j]:.12g})")
        lines.append(f"Bh{j} h{j} 0 V = 2/(1+exp(-2*(" + " + ".join(terms) + ")))-1")
    lines.append("")
    for k in range(n_classes):
        terms = [f"({w1[j, k]:.12g})*V(h{j})" for j in range(hidden)]
        terms.append(f"({b1[k]:.12g})")
        lines.append(f"By{k} y{k} 0 V = " + " + ".join(terms))
    lines += [
        "",
        f".tran {sample_period/20:.12g} {{TSTOP}}",
        ".control",
        "run",
        "wrdata " + str(trace_path) + " " + " ".join(f"V(y{k})" for k in range(n_classes)),
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def make_op_netlist(
    x: np.ndarray,
    w0: np.ndarray,
    b0: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    trace_path: Path,
) -> str:
    n_in = x.shape[0]
    hidden = w0.shape[1]
    n_classes = w1.shape[1]
    lines = [
        "* MNIST SPICE per-sample operating-point inference.",
        "* Offline-trained weights, ngspice-evaluated forward pass.",
        "",
    ]
    for i in range(n_in):
        lines.append(f"Vx{i} x{i} 0 DC {float(x[i]):.12g}")
    lines.append("")
    for j in range(hidden):
        terms = [f"({w0[i, j]:.12g})*V(x{i})" for i in range(n_in)]
        terms.append(f"({b0[j]:.12g})")
        lines.append(f"Bh{j} h{j} 0 V = 2/(1+exp(-2*(" + " + ".join(terms) + ")))-1")
    lines.append("")
    for k in range(n_classes):
        terms = [f"({w1[j, k]:.12g})*V(h{j})" for j in range(hidden)]
        terms.append(f"({b1[k]:.12g})")
        lines.append(f"By{k} y{k} 0 V = " + " + ".join(terms))
    lines += [
        "",
        ".control",
        "op",
        "wrdata " + str(trace_path) + " " + " ".join(f"V(y{k})" for k in range(n_classes)),
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def read_wrdata(path: Path, n_vec: int) -> pd.DataFrame:
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    data = {"time": arr[:, 0]}
    for i in range(n_vec):
        data[f"v{i}"] = arr[:, 2 * i + 1]
    return pd.DataFrame(data)


def sample_predictions(trace: pd.DataFrame, n_samples: int, sample_period: float, n_classes: int) -> np.ndarray:
    pred = np.zeros(n_samples, dtype=int)
    times = trace["time"].to_numpy()
    values = trace[[f"v{k}" for k in range(n_classes)]].to_numpy()
    for j in range(n_samples):
        t = (j + 0.8) * sample_period
        idx = int(np.abs(times - t).argmin())
        pred[j] = int(np.argmax(values[idx]))
    return pred


def run_op_predictions(
    spice_bin: str,
    x_test: np.ndarray,
    w0: np.ndarray,
    b0: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    netlist_path: Path,
    trace_path: Path,
) -> np.ndarray:
    preds = np.zeros(len(x_test), dtype=int)
    for idx, x in enumerate(x_test):
        netlist_path.write_text(make_op_netlist(x, w0, b0, w1, b1, trace_path))
        proc = subprocess.run([spice_bin, "-b", str(netlist_path)], text=True, capture_output=True, timeout=20)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-3000:] or proc.stdout[-3000:])
        row = read_wrdata(trace_path, 10).iloc[-1]
        vals = np.asarray([row[f"v{k}"] for k in range(10)])
        preds[idx] = int(np.argmax(vals))
        if (idx + 1) % 100 == 0:
            print(f"evaluated {idx + 1}/{len(x_test)} samples in ngspice")
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-samples", type=int, default=10000)
    ap.add_argument("--test-samples", type=int, default=2000)
    ap.add_argument("--image-size", type=int, default=14)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-period", type=float, default=50e-9)
    ap.add_argument("--eval-mode", choices=("per-sample", "transient"), default="per-sample")
    ap.add_argument("--tag", default="mlp14x14_h64")
    args = ap.parse_args()

    x_train, y_train, x_test, y_test = load_mnist(args.train_samples, args.test_samples, args.image_size, args.seed)
    model = train_mlp(x_train, y_train, args.hidden, args.epochs, args.seed)
    py_pred = model.predict(x_test)
    py_accuracy = float(accuracy_score(y_test, py_pred))

    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    generated.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)

    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag)
    stem = f"spice_mnist_{safe_tag}_inference"
    trace_path = ROOT / f"spice/results/{stem}_trace.dat"
    netlist_path = generated / f"{stem}.cir"
    w0 = model.coefs_[0].astype(float)
    b0 = model.intercepts_[0].astype(float)
    w1 = model.coefs_[1].astype(float)
    b1 = model.intercepts_[1].astype(float)
    if args.eval_mode == "transient":
        netlist_path.write_text(
            make_netlist(
                x_test,
                w0,
                b0,
                w1,
                b1,
                args.sample_period,
                trace_path,
            )
        )
        proc = subprocess.run([spice_bin, "-b", str(netlist_path)], text=True, capture_output=True, timeout=240)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-3000:] or proc.stdout[-3000:])
        trace = read_wrdata(trace_path, 10)
        spice_pred = sample_predictions(trace, len(y_test), args.sample_period, 10)
    else:
        spice_pred = run_op_predictions(spice_bin, x_test, w0, b0, w1, b1, netlist_path, trace_path)
        trace = read_wrdata(trace_path, 10)
    trace_csv = ROOT / f"spice/results/{stem}_trace.csv"
    trace.to_csv(trace_csv, index=False)
    spice_accuracy = float(accuracy_score(y_test, spice_pred))
    pred_csv = ROOT / f"spice/results/{stem}_predictions.csv"
    pd.DataFrame({"label": y_test, "python_pred": py_pred, "spice_pred": spice_pred}).to_csv(pred_csv, index=False)
    cm_csv = ROOT / f"spice/results/{stem}_confusion.csv"
    pd.DataFrame(confusion_matrix(y_test, spice_pred)).to_csv(cm_csv, index=False)
    weights_path = ROOT / f"spice/results/{stem}_weights.npz"
    np.savez_compressed(weights_path, w0=model.coefs_[0], b0=model.intercepts_[0], w1=model.coefs_[1], b1=model.intercepts_[1])

    summary = {
        "simulator": version,
        "dataset": "MNIST, area-downsampled for tractable ngspice netlist size",
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "image_size": args.image_size,
        "features": int(x_train.shape[1]),
        "hidden": args.hidden,
        "classes": 10,
        "model": "offline-trained one-hidden-layer tanh MLP, SPICE-evaluated forward inference",
        "eval_mode": args.eval_mode,
        "python_accuracy": py_accuracy,
        "spice_accuracy": spice_accuracy,
        "python_spice_prediction_agreement": float(np.mean(py_pred == spice_pred)),
        "netlist": str(netlist_path),
        "trace": str(trace_csv),
        "predictions": str(pred_csv),
        "confusion": str(cm_csv),
        "weights": str(weights_path),
        "sample_period_s": args.sample_period,
        "note": "Forward inference is evaluated inside ngspice. Training is offline, and MNIST is downsampled to keep the transient netlist tractable.",
    }
    summary_path = ROOT / f"spice/results/{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
