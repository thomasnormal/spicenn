from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from parse_ngspice import parse_measure
from run_spice_sweep import ROOT, detect_spice, run_tiny_test


TEMPLATE = ROOT / "spice/templates/charge_adc_behavioral.cir"


def adc_expr(levels: int, vmin: float, vmax: float) -> str:
    thresholds = np.linspace(vmin, vmax, levels - 1)
    terms = [f"(V(int)+V(n) > {thr:.12g})" for thr in thresholds]
    return "(" + " + ".join(terms) + f")/{levels - 1}"


def render(
    path: Path,
    vin: float,
    gsum: float,
    cint: float,
    tau: float,
    sigma: float,
    seed: int,
    bits: int,
    vmin: float,
    vmax: float,
) -> None:
    levels = 2**bits
    text = TEMPLATE.read_text()
    replacements = {
        "ADC_EXPR": adc_expr(levels, vmin, vmax),
        "{VIN}": f"{vin:.12g}",
        "{GSUM}": f"{gsum:.12g}",
        "{CINT}": f"{cint:.12g}",
        "{TAU}": f"{tau:.12g}",
        "{SIGMA}": f"{sigma:.12g}",
        "{SEED}": str(seed),
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    path.write_text(text)


def run_one(spice_bin: str, netlist: Path) -> tuple[float, float, float]:
    proc = subprocess.run([spice_bin, "-b", str(netlist)], text=True, capture_output=True, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-1200:] or proc.stdout[-1200:])
    text = proc.stdout + "\n" + proc.stderr
    return parse_measure(text, "vint"), parse_measure(text, "y") / 0.8, parse_measure(text, "ecap")


def run_sweep(
    bits_list: list[int],
    vin_values: np.ndarray,
    trials: int,
    c_values: list[float],
    tau_values: list[float],
    gsum_values: list[float],
    sigma_values: list[float],
    adc_vmin: float,
    adc_vmax: float,
    seed: int,
) -> pd.DataFrame:
    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    generated.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)
    rng = np.random.default_rng(seed)
    rows = []
    for bits in bits_list:
        for cint in c_values:
            for tau in tau_values:
                for gsum in gsum_values:
                    for sigma in sigma_values:
                        for vin in vin_values:
                            vint_values = []
                            code_values = []
                            ecap_values = []
                            for _ in range(trials):
                                trial_seed = int(rng.integers(1, 2**30))
                                netlist = generated / (
                                    f"charge_adc_b{bits}_vin{vin:+.3e}_c{cint:.1e}_"
                                    f"tau{tau:.1e}_g{gsum:.1e}_sig{sigma:.1e}_seed{trial_seed}.cir"
                                )
                                render(netlist, vin, gsum, cint, tau, sigma, trial_seed, bits, adc_vmin, adc_vmax)
                                vint, code, ecap = run_one(spice_bin, netlist)
                                vint_values.append(vint)
                                code_values.append(code)
                                ecap_values.append(ecap)
                            analytic_vint = tau * gsum * vin / cint
                            rows.append(
                                {
                                    "bits": bits,
                                    "levels": 2**bits,
                                    "vin": float(vin),
                                    "cint_f": cint,
                                    "tau_s": tau,
                                    "gsum_siemens": gsum,
                                    "sigma_v": sigma,
                                    "adc_vmin": adc_vmin,
                                    "adc_vmax": adc_vmax,
                                    "trials": trials,
                                    "vint_mean": float(np.mean(vint_values)),
                                    "vint_std": float(np.std(vint_values, ddof=1)) if trials > 1 else 0.0,
                                    "vint_analytic": float(analytic_vint),
                                    "mean_code_0_1": float(np.mean(code_values)),
                                    "std_code_0_1": float(np.std(code_values, ddof=1)) if trials > 1 else 0.0,
                                    "mean_code_bipolar": float(2.0 * np.mean(code_values) - 1.0),
                                    "ecap_j_mean": float(np.mean(ecap_values)),
                                    "source": "spice",
                                    "simulator": version,
                                }
                            )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.5, 4.5))
    for (bits, cint), group in df.groupby(["bits", "cint_f"]):
        label = f"{bits} bit, C={cint * 1e15:.0f} fF"
        g = group.groupby("vin", as_index=False)["mean_code_0_1"].mean()
        plt.plot(g["vin"], g["mean_code_0_1"], marker="o", label=label)
    plt.xlabel("input voltage / pulse amplitude (V)")
    plt.ylabel("mean normalized ADC code")
    plt.ylim(-0.03, 1.03)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def parse_float_list(text: str) -> list[float]:
    return [float(v) for v in text.split(",") if v.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", default="2,3,4")
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--vin-min", type=float, default=-0.12)
    ap.add_argument("--vin-max", type=float, default=0.12)
    ap.add_argument("--points", type=int, default=9)
    ap.add_argument("--c-values", default="3e-15,10e-15,30e-15")
    ap.add_argument("--tau-values", default="3e-9")
    ap.add_argument("--gsum-values", default="100e-9")
    ap.add_argument("--sigma-values", default="1.5e-3")
    ap.add_argument("--adc-vmin", type=float, default=-0.0045)
    ap.add_argument("--adc-vmax", type=float, default=0.0045)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=str(ROOT / "spice/results/charge_adc_sweep.csv"))
    ap.add_argument("--plot", default=str(ROOT / "spice/results/charge_adc_transfer.png"))
    args = ap.parse_args()
    df = run_sweep(
        bits_list=[int(v) for v in args.bits.split(",")],
        vin_values=np.linspace(args.vin_min, args.vin_max, args.points),
        trials=args.trials,
        c_values=parse_float_list(args.c_values),
        tau_values=parse_float_list(args.tau_values),
        gsum_values=parse_float_list(args.gsum_values),
        sigma_values=parse_float_list(args.sigma_values),
        adc_vmin=args.adc_vmin,
        adc_vmax=args.adc_vmax,
        seed=args.seed,
    )
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    plot(df, Path(args.plot))
    summary = {
        "csv": str(csv_path),
        "plot": args.plot,
        "rows": len(df),
        "bits": sorted(df["bits"].unique().tolist()),
        "source": "spice",
    }
    summary_path = ROOT / "spice/results/charge_adc_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

