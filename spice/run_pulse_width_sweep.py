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


TEMPLATE = ROOT / "spice/templates/pulse_width_neuron.cir"


def render(path: Path, vin: float, gsum: float, cint: float, rleak: float, vth: float, sigma: float, seed: int) -> None:
    text = TEMPLATE.read_text()
    for old, new in {
        "{VIN}": f"{vin:.12g}",
        "{GSUM}": f"{gsum:.12g}",
        "{CINT}": f"{cint:.12g}",
        "{RLEAK}": f"{rleak:.12g}",
        "{VTH}": f"{vth:.12g}",
        "{SIGMA}": f"{sigma:.12g}",
        "{SEED}": str(seed),
    }.items():
        text = text.replace(old, new)
    path.write_text(text)


def parse_optional(text: str, name: str) -> float | None:
    try:
        return parse_measure(text, name)
    except ValueError:
        return None


def run_sweep(vin_values, trials, c_values, r_values, vth_values, sigma_values, seed) -> pd.DataFrame:
    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    generated.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)
    rng = np.random.default_rng(seed)
    rows = []
    for cint in c_values:
        for rleak in r_values:
            for vth in vth_values:
                for sigma in sigma_values:
                    for vin in vin_values:
                        widths, vmaxes, ecaps = [], [], []
                        for _ in range(trials):
                            trial_seed = int(rng.integers(1, 2**30))
                            netlist = generated / (
                                f"pulse_width_vin{vin:+.3e}_c{cint:.1e}_r{rleak:.1e}_"
                                f"vth{vth:.1e}_sig{sigma:.1e}_seed{trial_seed}.cir"
                            )
                            render(netlist, vin, gsum=100e-9, cint=cint, rleak=rleak, vth=vth, sigma=sigma, seed=trial_seed)
                            proc = subprocess.run([spice_bin, "-b", str(netlist)], text=True, capture_output=True, timeout=30)
                            if proc.returncode != 0:
                                raise RuntimeError(proc.stderr[-1200:] or proc.stdout[-1200:])
                            text = proc.stdout + "\n" + proc.stderr
                            width = parse_optional(text, "width")
                            widths.append(0.0 if width is None or width < 0 else width)
                            vmaxes.append(parse_measure(text, "vmax"))
                            ecaps.append(parse_measure(text, "ecap_peak"))
                        rows.append(
                            {
                                "vin": float(vin),
                                "cint_f": cint,
                                "rleak_ohm": rleak,
                                "vth_v": vth,
                                "sigma_v": sigma,
                                "trials": trials,
                                "pulse_width_mean_s": float(np.mean(widths)),
                                "pulse_width_std_s": float(np.std(widths, ddof=1)) if trials > 1 else 0.0,
                                "pulse_width_code_0_1": float(np.clip(np.mean(widths) / 60e-9, 0.0, 1.0)),
                                "vmax_mean": float(np.mean(vmaxes)),
                                "ecap_peak_j_mean": float(np.mean(ecaps)),
                                "source": "spice",
                                "simulator": version,
                            }
                        )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.5, 4.5))
    for (cint, rleak), group in df.groupby(["cint_f", "rleak_ohm"]):
        g = group.groupby("vin", as_index=False)["pulse_width_code_0_1"].mean()
        plt.plot(g["vin"], g["pulse_width_code_0_1"], marker="o", label=f"C={cint*1e15:.0f} fF, R={rleak/1e6:.1f} M")
    plt.xlabel("input voltage / pulse amplitude (V)")
    plt.ylabel("pulse-width code")
    plt.ylim(-0.03, 1.03)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def parse_float_list(text: str) -> list[float]:
    return [float(v) for v in text.split(",") if v.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--vin-min", type=float, default=0.0)
    ap.add_argument("--vin-max", type=float, default=0.15)
    ap.add_argument("--points", type=int, default=7)
    ap.add_argument("--c-values", default="3e-15,10e-15,30e-15")
    ap.add_argument("--r-values", default="2e6,5e6,10e6")
    ap.add_argument("--vth-values", default="4e-3")
    ap.add_argument("--sigma-values", default="0,1e-3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=str(ROOT / "spice/results/pulse_width_sweep.csv"))
    ap.add_argument("--plot", default=str(ROOT / "spice/results/pulse_width_transfer.png"))
    args = ap.parse_args()
    df = run_sweep(
        vin_values=np.linspace(args.vin_min, args.vin_max, args.points),
        trials=args.trials,
        c_values=parse_float_list(args.c_values),
        r_values=parse_float_list(args.r_values),
        vth_values=parse_float_list(args.vth_values),
        sigma_values=parse_float_list(args.sigma_values),
        seed=args.seed,
    )
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    plot(df, Path(args.plot))
    span = df.groupby(["cint_f", "rleak_ohm"])["pulse_width_code_0_1"].agg(lambda s: s.max() - s.min()).max()
    summary = {"csv": str(csv_path), "plot": args.plot, "rows": len(df), "source": "spice", "best_code_span": float(span)}
    (ROOT / "spice/results/pulse_width_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

