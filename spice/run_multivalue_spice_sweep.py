from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from parse_ngspice import parse_measure
from run_spice_sweep import ROOT, detect_spice, prepare_netlist_for_simulator, run_tiny_test, run_simulator_netlist


TEMPLATE = ROOT / "spice/templates/multivalue_thermometer_behavioral.cir"


def therm_expr(levels: int, span: float) -> str:
    thresholds = np.linspace(-span, span, levels - 1)
    terms = [f"(V(sig)+V(n) > {thr:.12g})" for thr in thresholds]
    return "(" + " + ".join(terms) + f")/{levels - 1}"


def render(signal: float, sigma: float, seed: int, bits: int, threshold_span: float, path: Path, spice_bin: str) -> None:
    levels = 2 ** bits
    text = TEMPLATE.read_text()
    text = text.replace("THERM_EXPR", therm_expr(levels, threshold_span))
    text = text.replace("{S}", f"{signal:.12g}")
    text = text.replace("{SIGMA}", f"{sigma:.12g}")
    text = text.replace("{SEED}", str(seed))
    path.write_text(prepare_netlist_for_simulator(text, spice_bin))


def run_sweep(bits: int, trials: int, points: int, sigma: float, signal_span: float, threshold_span: float, seed: int) -> pd.DataFrame:
    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    generated.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)
    rng = np.random.default_rng(seed)
    rows = []
    for signal in np.linspace(-signal_span, signal_span, points):
        values = []
        for _ in range(trials):
            trial_seed = int(rng.integers(1, 2**30))
            netlist = generated / f"therm_b{bits}_s{signal:+.6e}_seed{trial_seed}.cir"
            render(float(signal), sigma, trial_seed, bits, threshold_span, netlist, spice_bin)
            proc = run_simulator_netlist(spice_bin, netlist, timeout=30)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr[-1000:] or proc.stdout[-1000:])
            y = parse_measure(proc.stdout + "\n" + proc.stderr, "y") / 0.8
            values.append(float(y))
        rows.append(
            {
                "signal": float(signal),
                "bits": bits,
                "levels": 2**bits,
                "trials": trials,
                "mean_code_0_1": float(np.mean(values)),
                "std_code_0_1": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "mean_code_bipolar": float(2 * np.mean(values) - 1),
                "sigma": sigma,
                "threshold_span": threshold_span,
                "source": "spice",
                "simulator": version,
            }
        )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    for bits, group in df.groupby("bits"):
        plt.errorbar(group["signal"], group["mean_code_0_1"], yerr=group["std_code_0_1"], marker="o", label=f"{bits} bit")
    plt.xlabel("signal")
    plt.ylabel("mean normalized code")
    plt.ylim(-0.03, 1.03)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", default="2,3,4")
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--points", type=int, default=13)
    ap.add_argument("--sigma", type=float, default=1.5e-3)
    ap.add_argument("--signal-span", type=float, default=6e-3)
    ap.add_argument("--threshold-span", type=float, default=4.5e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=str(ROOT / "spice/results/multivalue_activation_curve.csv"))
    ap.add_argument("--plot", default=str(ROOT / "spice/results/multivalue_activation_curve.png"))
    args = ap.parse_args()
    frames = []
    for bits in [int(v) for v in args.bits.split(",")]:
        frames.append(run_sweep(bits, args.trials, args.points, args.sigma, args.signal_span, args.threshold_span, args.seed + bits))
    df = pd.concat(frames, ignore_index=True)
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    plot(df, Path(args.plot))
    print(json.dumps({"csv": str(csv_path), "plot": args.plot, "rows": len(df), "source": "spice"}, indent=2))


if __name__ == "__main__":
    main()
