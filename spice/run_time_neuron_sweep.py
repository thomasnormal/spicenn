from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from parse_ngspice import parse_measure
from run_spice_sweep import ROOT, detect_spice, prepare_netlist_for_simulator, run_tiny_test, run_simulator_netlist


TEMPLATE = ROOT / "spice/templates/time_to_threshold_neuron.cir"


def render(path: Path, spice_bin: str, vin: float, gsum: float, cint: float, vth: float, sigma: float, seed: int) -> None:
    text = TEMPLATE.read_text()
    for old, new in {
        "{VIN}": f"{vin:.12g}",
        "{GSUM}": f"{gsum:.12g}",
        "{CINT}": f"{cint:.12g}",
        "{VTH}": f"{vth:.12g}",
        "{SIGMA}": f"{sigma:.12g}",
        "{SEED}": str(seed),
    }.items():
        text = text.replace(old, new)
    path.write_text(prepare_netlist_for_simulator(text, spice_bin))


def parse_optional_measure(text: str, name: str) -> float | None:
    try:
        return parse_measure(text, name)
    except ValueError:
        if re.search(rf"\b{name}\b.*failed", text, flags=re.IGNORECASE):
            return None
        return None


def run_sweep(
    vin_values: np.ndarray,
    trials: int,
    c_values: list[float],
    gsum_values: list[float],
    vth_values: list[float],
    sigma_values: list[float],
    seed: int,
) -> pd.DataFrame:
    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    generated.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)
    rng = np.random.default_rng(seed)
    rows = []
    for cint in c_values:
        for gsum in gsum_values:
            for vth in vth_values:
                for sigma in sigma_values:
                    for vin in vin_values:
                        tfires = []
                        vint_end = []
                        ecap_end = []
                        for _ in range(trials):
                            trial_seed = int(rng.integers(1, 2**30))
                            netlist = generated / (
                                f"time_neuron_vin{vin:+.3e}_c{cint:.1e}_g{gsum:.1e}_"
                                f"vth{vth:.1e}_sig{sigma:.1e}_seed{trial_seed}.cir"
                            )
                            render(netlist, spice_bin, vin, gsum, cint, vth, sigma, trial_seed)
                            proc = run_simulator_netlist(spice_bin, netlist, timeout=30)
                            if proc.returncode != 0:
                                raise RuntimeError(proc.stderr[-1200:] or proc.stdout[-1200:])
                            text = proc.stdout + "\n" + proc.stderr
                            tfires.append(parse_optional_measure(text, "tfire"))
                            vint_end.append(parse_measure(text, "vint_end"))
                            ecap_end.append(parse_measure(text, "ecap_end"))
                        finite = [t for t in tfires if t is not None]
                        fire_prob = len(finite) / trials
                        mean_tfire = float(np.mean(finite)) if finite else np.nan
                        # Larger code means stronger signal/faster firing. No-spike maps to 0.
                        code = 0.0 if not finite else float(np.mean([(40e-9 - t) / (40e-9 - 0.5e-9) for t in finite]))
                        rows.append(
                            {
                                "vin": float(vin),
                                "cint_f": cint,
                                "gsum_siemens": gsum,
                                "vth_v": vth,
                                "sigma_v": sigma,
                                "trials": trials,
                                "fire_probability": fire_prob,
                                "tfire_mean_s": mean_tfire,
                                "time_code_0_1": max(0.0, min(1.0, code)),
                                "vint_end_mean": float(np.mean(vint_end)),
                                "ecap_end_j_mean": float(np.mean(ecap_end)),
                                "source": "spice",
                                "simulator": version,
                            }
                        )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.5, 4.5))
    for (cint, vth), group in df.groupby(["cint_f", "vth_v"]):
        g = group.groupby("vin", as_index=False)["time_code_0_1"].mean()
        plt.plot(g["vin"], g["time_code_0_1"], marker="o", label=f"C={cint*1e15:.0f} fF, Vth={vth*1e3:.1f} mV")
    plt.xlabel("input voltage / pulse amplitude (V)")
    plt.ylabel("time code (fast spike = high)")
    plt.ylim(-0.03, 1.03)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def parse_float_list(text: str) -> list[float]:
    return [float(v) for v in text.split(",") if v.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--vin-min", type=float, default=0.0)
    ap.add_argument("--vin-max", type=float, default=0.15)
    ap.add_argument("--points", type=int, default=9)
    ap.add_argument("--c-values", default="3e-15,10e-15,30e-15")
    ap.add_argument("--gsum-values", default="100e-9")
    ap.add_argument("--vth-values", default="2e-3,4e-3,8e-3")
    ap.add_argument("--sigma-values", default="0,1e-3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=str(ROOT / "spice/results/time_neuron_sweep.csv"))
    ap.add_argument("--plot", default=str(ROOT / "spice/results/time_neuron_transfer.png"))
    args = ap.parse_args()
    df = run_sweep(
        vin_values=np.linspace(args.vin_min, args.vin_max, args.points),
        trials=args.trials,
        c_values=parse_float_list(args.c_values),
        gsum_values=parse_float_list(args.gsum_values),
        vth_values=parse_float_list(args.vth_values),
        sigma_values=parse_float_list(args.sigma_values),
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
        "source": "spice",
        "best_code_span": float(df.groupby(["cint_f", "vth_v"])["time_code_0_1"].agg(lambda s: s.max() - s.min()).max()),
    }
    (ROOT / "spice/results/time_neuron_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
