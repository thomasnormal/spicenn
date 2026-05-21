from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_spice_sweep import ROOT, detect_spice, prepare_netlist_for_simulator, run_tiny_test, run_simulator_netlist


def pwl(values: list[float], sample_period: float, edge: float = 1e-9) -> str:
    pts: list[tuple[float, float]] = [(0.0, values[0])]
    for i, val in enumerate(values):
        t0 = i * sample_period
        pts.append((t0 + edge, val))
        pts.append(((i + 1) * sample_period - edge, val))
        if i + 1 < len(values):
            pts.append(((i + 1) * sample_period, values[i + 1]))
    return "PWL(" + " ".join(f"{t:.12g} {v:.12g}" for t, v in pts) + ")"


def update_gate(n_samples: int, sample_period: float, settle_frac: float = 0.35, off_frac: float = 0.95) -> str:
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


def generate_xor_sequence(epochs: int) -> tuple[list[float], list[float], list[float]]:
    samples = [(-1.0, -1.0, -1.0), (-1.0, 1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, -1.0)]
    x1, x2, target = [], [], []
    for _ in range(epochs):
        for a, b, y in samples:
            x1.append(a)
            x2.append(b)
            target.append(y)
    return x1, x2, target


def netlist_text(epochs: int, lr: float, sample_period: float, trace_path: Path) -> str:
    x1, x2, target = generate_xor_sequence(epochs)
    n = len(target)
    tstop = n * sample_period
    return f"""* Entirely-in-SPICE backprop training demo: 2-2-1 XOR MLP.
* Weight voltages are programmable conductance-state proxies.
* Forward pass, backward pass, and update currents are all behavioral SPICE.
.param LR={lr:.12g}
.param CW=1
.param TSTOP={tstop:.12g}

Vx1 x1 0 {pwl(x1, sample_period)}
Vx2 x2 0 {pwl(x2, sample_period)}
Vt target 0 {pwl(target, sample_period)}
Vu upd 0 {update_gate(n, sample_period)}

* Weight and bias state capacitors. V(weight_node) is the signed conductance proxy.
Cw11 w11 0 {{CW}} IC=0.61
Cw12 w12 0 {{CW}} IC=-0.47
Cw21 w21 0 {{CW}} IC=-0.32
Cw22 w22 0 {{CW}} IC=0.58
Cbh1 bh1 0 {{CW}} IC=0.03
Cbh2 bh2 0 {{CW}} IC=-0.04
Cv1 v1 0 {{CW}} IC=0.52
Cv2 v2 0 {{CW}} IC=-0.44
Cbo bo 0 {{CW}} IC=0.02

* Huge leaks give DC paths without materially changing training.
Rw11 w11 0 1e15
Rw12 w12 0 1e15
Rw21 w21 0 1e15
Rw22 w22 0 1e15
Rbh1 bh1 0 1e15
Rbh2 bh2 0 1e15
Rv1 v1 0 1e15
Rv2 v2 0 1e15
Rbo bo 0 1e15

* Forward pass: hidden and output activations.
Bh1 h1 0 V = 2/(1+exp(-2*(V(w11)*V(x1)+V(w12)*V(x2)+V(bh1))))-1
Bh2 h2 0 V = 2/(1+exp(-2*(V(w21)*V(x1)+V(w22)*V(x2)+V(bh2))))-1
By y 0 V = 2/(1+exp(-2*(V(v1)*V(h1)+V(v2)*V(h2)+V(bo))))-1

* Backward pass: output delta and hidden deltas.
Berr err 0 V = V(target)-V(y)
Bdo do 0 V = V(err)*(1-V(y)*V(y))
Bdh1 dh1 0 V = (1-V(h1)*V(h1))*V(v1)*V(do)
Bdh2 dh2 0 V = (1-V(h2)*V(h2))*V(v2)*V(do)

* Programming/update currents into weight-state capacitors.
* Current direction 0 -> node increases V(node), so C*dV/dt = LR*gradient.
Buw11 0 w11 I = {{LR}}*V(upd)*V(dh1)*V(x1)
Buw12 0 w12 I = {{LR}}*V(upd)*V(dh1)*V(x2)
Buw21 0 w21 I = {{LR}}*V(upd)*V(dh2)*V(x1)
Buw22 0 w22 I = {{LR}}*V(upd)*V(dh2)*V(x2)
Bubh1 0 bh1 I = {{LR}}*V(upd)*V(dh1)
Bubh2 0 bh2 I = {{LR}}*V(upd)*V(dh2)
Buv1 0 v1 I = {{LR}}*V(upd)*V(do)*V(h1)
Buv2 0 v2 I = {{LR}}*V(upd)*V(do)*V(h2)
Bubo 0 bo I = {{LR}}*V(upd)*V(do)

.tran {sample_period/100:.12g} {{TSTOP}} uic
.control
run
wrdata {trace_path} V(x1) V(x2) V(target) V(upd) V(h1) V(h2) V(y) V(err) V(w11) V(w12) V(w21) V(w22) V(v1) V(v2)
.endc
.end
"""


def read_wrdata(path: Path) -> pd.DataFrame:
    arr = np.loadtxt(path)
    # ngspice wrdata writes time,value pairs for each vector.
    names = ["x1", "x2", "target", "upd", "h1", "h2", "y", "err", "w11", "w12", "w21", "w22", "v1", "v2"]
    data = {"time": arr[:, 0]}
    for i, name in enumerate(names):
        data[name] = arr[:, 2 * i + 1]
    return pd.DataFrame(data)


def sample_learning_curve(trace: pd.DataFrame, epochs: int, sample_period: float) -> pd.DataFrame:
    rows = []
    for epoch in range(epochs):
        preds = []
        targets = []
        losses = []
        for j in range(4):
            idx = epoch * 4 + j
            t = (idx + 0.9) * sample_period
            row = trace.iloc[(trace["time"] - t).abs().argmin()]
            preds.append(1.0 if row["y"] >= 0 else -1.0)
            targets.append(row["target"])
            losses.append(0.5 * (row["target"] - row["y"]) ** 2)
        acc = float(np.mean(np.asarray(preds) == np.sign(np.asarray(targets))))
        rows.append({"epoch": epoch + 1, "xor_accuracy": acc, "mse": float(np.mean(losses))})
    return pd.DataFrame(rows)


def plot_curve(curve: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(curve["epoch"], curve["xor_accuracy"], marker="o", label="accuracy")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("XOR accuracy")
    ax1.set_ylim(-0.05, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(curve["epoch"], curve["mse"], color="tab:red", alpha=0.7, label="MSE")
    ax2.set_ylabel("MSE")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=2e5)
    ap.add_argument("--sample-period", type=float, default=1e-6)
    args = ap.parse_args()

    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    generated.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)
    trace_path = ROOT / "spice/results/spice_backprop_xor_trace.dat"
    netlist_path = generated / "spice_backprop_xor.cir"
    netlist = netlist_text(args.epochs, args.lr, args.sample_period, trace_path)
    netlist_path.write_text(prepare_netlist_for_simulator(netlist, spice_bin))
    proc = run_simulator_netlist(spice_bin, netlist_path, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:] or proc.stdout[-2000:])
    trace = read_wrdata(trace_path)
    trace_csv = ROOT / "spice/results/spice_backprop_xor_trace.csv"
    trace.to_csv(trace_csv, index=False)
    curve = sample_learning_curve(trace, args.epochs, args.sample_period)
    curve_csv = ROOT / "spice/results/spice_backprop_xor_learning_curve.csv"
    curve.to_csv(curve_csv, index=False)
    fig = ROOT / "spice/results/spice_backprop_xor_learning_curve.png"
    plot_curve(curve, fig)
    final = curve.iloc[-1].to_dict()
    summary = {
        "simulator": version,
        "netlist": str(netlist_path),
        "trace": str(trace_csv),
        "learning_curve": str(curve_csv),
        "figure": str(fig),
        "epochs": args.epochs,
        "sample_period_s": args.sample_period,
        "lr": args.lr,
        "final": final,
        "note": "Forward pass, backward deltas, and weight update currents were simulated inside ngspice.",
    }
    summary_path = ROOT / "spice/results/spice_backprop_xor_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
