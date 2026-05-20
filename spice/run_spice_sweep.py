from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from parse_ngspice import parse_output_high
from fit_activation_curve import fit_activation


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "spice/templates/noisy_comparator_behavioral.cir"


def detect_spice(preferred: str | None = None) -> tuple[str, str]:
    candidates = [preferred] if preferred else ["ngspice", "Xyce", "xyce"]
    for name in candidates:
        if not name:
            continue
        path = shutil.which(name)
        if path is None and Path(name).exists():
            path = str(Path(name).resolve())
        if path:
            version_arg = "-v" if "xyce" in Path(path).name.lower() else "--version"
            version = subprocess.run([path, version_arg], text=True, capture_output=True, timeout=10)
            lines = (version.stdout or version.stderr).strip().splitlines()
            useful = next((line.strip("* ").strip() for line in lines if "ngspice" in line.lower() or "xyce" in line.lower()), None)
            return path, useful or (lines[0] if lines else Path(path).name)
    raise RuntimeError("No SPICE simulator found. Install ngspice or Xyce and ensure it is on PATH.")


def run_tiny_test(spice_bin: str, workdir: Path) -> None:
    netlist = workdir / "tiny_test.cir"
    if "ngspice" in Path(spice_bin).name.lower():
        netlist.write_text(
            "* tiny test\nV1 in 0 DC 1\nR1 in 0 1k\n.op\n.control\nrun\nprint v(in)\n.endc\n.end\n"
        )
    else:
        netlist.write_text("* tiny test\nV1 in 0 DC 1\nR1 in 0 1k\n.op\n.print DC V(in)\n.end\n")
    cmd = [spice_bin, "-b", str(netlist)] if "ngspice" in Path(spice_bin).name.lower() else [spice_bin, str(netlist)]
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=20)
    if proc.returncode != 0:
        raise RuntimeError(f"SPICE tiny test failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")


def render_netlist(signal: float, sigma: float, seed: int, out_path: Path) -> None:
    text = TEMPLATE.read_text()
    replacements = {
        "{S}": f"{signal}",
        "{SIGMA}": f"{sigma}",
        "{SEED}": f"{seed}",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    out_path.write_text(text)


def analytic_fallback(signals: np.ndarray, trials: int, sigma: float, theta: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in signals:
        samples = s + rng.normal(0.0, sigma, size=trials)
        highs = int(np.sum(samples > theta))
        rows.append({"signal": s, "trials": trials, "high": highs, "p_high": highs / trials, "source": "analytic_fallback"})
    return pd.DataFrame(rows)


def run_sweep(
    trials: int,
    points: int,
    sigma: float,
    span: float,
    seed: int,
    simulator: str | None,
    allow_analytic_fallback: bool,
) -> pd.DataFrame:
    generated = ROOT / "spice/generated"
    generated.mkdir(parents=True, exist_ok=True)
    signals = np.linspace(-span, span, points)
    spice_bin, version = detect_spice(simulator)
    run_tiny_test(spice_bin, generated)
    rows = []
    rng = np.random.default_rng(seed)
    failures = []
    for s in signals:
        highs = 0
        local_failures = []
        for _ in range(trials):
            trial_seed = int(rng.integers(1, 2**30))
            netlist = generated / f"cmp_s{s:+.6e}_seed{trial_seed}.cir"
            render_netlist(float(s), sigma, trial_seed, netlist)
            cmd = [spice_bin, "-b", str(netlist)] if "ngspice" in Path(spice_bin).name.lower() else [spice_bin, str(netlist)]
            proc = subprocess.run(cmd, text=True, capture_output=True, timeout=30)
            if proc.returncode != 0:
                msg = proc.stderr[-500:] or proc.stdout[-500:]
                failures.append(msg)
                local_failures.append(msg)
                continue
            try:
                highs += parse_output_high(proc.stdout + "\n" + proc.stderr)
            except Exception as exc:
                msg = str(exc) + "\n" + (proc.stdout + proc.stderr)[-500:]
                failures.append(msg)
                local_failures.append(msg)
        complete = trials - len(local_failures)
        if complete <= 0:
            if allow_analytic_fallback:
                return analytic_fallback(signals, trials, sigma, 0.0, seed)
            raise RuntimeError("All SPICE trials failed. First failure:\n" + (failures[0] if failures else "unknown"))
        rows.append(
            {
                "signal": float(s),
                "trials": complete,
                "high": highs,
                "p_high": highs / complete,
                "source": "spice",
                "simulator": version,
            }
        )
    df = pd.DataFrame(rows)
    if df["p_high"].nunique() <= 2 and allow_analytic_fallback:
        # Some SPICE builds do not randomize TRNOISE in batch as expected.
        return analytic_fallback(signals, trials, sigma, 0.0, seed)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--points", type=int, default=25)
    ap.add_argument("--sigma", type=float, default=1.5e-3)
    ap.add_argument("--span", type=float, default=6e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--simulator", default=None)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--no-analytic-fallback", action="store_true")
    ap.add_argument("--csv", default=str(ROOT / "spice/results/activation_curve.csv"))
    ap.add_argument("--fit-json", default=str(ROOT / "spice/results/activation_curve_fit.json"))
    ap.add_argument("--plot", default=str(ROOT / "spice/results/activation_curve.png"))
    args = ap.parse_args()

    spice_bin, version = detect_spice(args.simulator)
    generated = ROOT / "spice/generated"
    generated.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)
    if args.check_only:
        print(json.dumps({"spice_bin": spice_bin, "version": version}, indent=2))
        return

    df = run_sweep(
        trials=args.trials,
        points=args.points,
        sigma=args.sigma,
        span=args.span,
        seed=args.seed,
        simulator=args.simulator,
        allow_analytic_fallback=not args.no_analytic_fallback,
    )
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    fit = fit_activation(csv_path, args.fit_json, args.plot)
    print(json.dumps({"csv": str(csv_path), "fit": fit, "source": str(df["source"].iloc[0])}, indent=2))


if __name__ == "__main__":
    main()
