from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

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


def load_split(seed: int, test_fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y = load_digits(return_X_y=True)
    x = x.astype(np.float64) / 16.0
    return train_test_split(x, y.astype(int), test_size=test_fraction, random_state=seed, stratify=y)


def train_logreg(x_train: np.ndarray, y_train: np.ndarray, c_value: float) -> LogisticRegression:
    model = LogisticRegression(max_iter=2000, C=c_value, solver="lbfgs")
    model.fit(x_train, y_train)
    return model


def make_netlist(
    x_eval: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    sample_period: float,
    trace_path: Path,
) -> str:
    n_samples, n_in = x_eval.shape
    n_classes = weights.shape[0]
    tstop = n_samples * sample_period
    lines = [
        "* sklearn digits SPICE inference benchmark.",
        "* Offline-trained programmable weights are evaluated by ngspice as behavioral conductance-weighted sums.",
        "* This is SPICE forward inference, not SPICE training.",
        f".param TSTOP={tstop:.12g}",
        "",
    ]
    for i in range(n_in):
        lines.append(f"Vx{i} x{i} 0 {pwl(x_eval[:, i], sample_period)}")
    lines.append("")
    for k in range(n_classes):
        terms = [f"({weights[k, i]:.12g})*V(x{i})" for i in range(n_in)]
        terms.append(f"({bias[k]:.12g})")
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-fraction", type=float, default=0.25)
    ap.add_argument("--test-samples", type=int, default=0, help="0 means use the whole held-out split")
    ap.add_argument("--c-value", type=float, default=10.0)
    ap.add_argument("--sample-period", type=float, default=50e-9)
    ap.add_argument("--tag", default="logreg")
    args = ap.parse_args()

    x_train, x_test, y_train, y_test = load_split(args.seed, args.test_fraction)
    model = train_logreg(x_train, y_train, args.c_value)
    if args.test_samples > 0:
        x_test = x_test[: args.test_samples]
        y_test = y_test[: args.test_samples]

    py_pred = model.predict(x_test)
    py_accuracy = float(accuracy_score(y_test, py_pred))

    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    generated.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)

    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag)
    stem = f"spice_digits_{safe_tag}_inference"
    trace_path = ROOT / f"spice/results/{stem}_trace.dat"
    netlist_path = generated / f"{stem}.cir"
    netlist_path.write_text(
        make_netlist(x_test, model.coef_.astype(float), model.intercept_.astype(float), args.sample_period, trace_path)
    )
    proc = subprocess.run([spice_bin, "-b", str(netlist_path)], text=True, capture_output=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-3000:] or proc.stdout[-3000:])

    trace = read_wrdata(trace_path, 10)
    trace_csv = ROOT / f"spice/results/{stem}_trace.csv"
    trace.to_csv(trace_csv, index=False)
    spice_pred = sample_predictions(trace, len(y_test), args.sample_period, 10)
    spice_accuracy = float(accuracy_score(y_test, spice_pred))
    pred_csv = ROOT / f"spice/results/{stem}_predictions.csv"
    pd.DataFrame({"label": y_test, "python_pred": py_pred, "spice_pred": spice_pred}).to_csv(pred_csv, index=False)
    cm_csv = ROOT / f"spice/results/{stem}_confusion.csv"
    pd.DataFrame(confusion_matrix(y_test, spice_pred)).to_csv(cm_csv, index=False)
    weights_path = ROOT / f"spice/results/{stem}_weights.npz"
    np.savez_compressed(weights_path, weights=model.coef_, bias=model.intercept_)

    summary = {
        "simulator": version,
        "dataset": "sklearn digits 8x8",
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "classes": 10,
        "features": int(x_train.shape[1]),
        "model": "offline-trained multinomial logistic regression, SPICE-evaluated linear class evidence",
        "python_accuracy": py_accuracy,
        "spice_accuracy": spice_accuracy,
        "python_spice_prediction_agreement": float(np.mean(py_pred == spice_pred)),
        "netlist": str(netlist_path),
        "trace": str(trace_csv),
        "predictions": str(pred_csv),
        "confusion": str(cm_csv),
        "weights": str(weights_path),
        "sample_period_s": args.sample_period,
        "note": "Forward inference is evaluated inside ngspice. Training is offline, and this is the sklearn digits benchmark, not full MNIST.",
    }
    summary_path = ROOT / f"spice/results/{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
