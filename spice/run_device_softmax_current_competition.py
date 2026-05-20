from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import pandas as pd

from run_device_multicell_classifier import mos_models
from run_spice_sweep import ROOT, detect_spice, run_tiny_test


MEAS_RE = re.compile(r"(?im)^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([-+0-9.eE]+)")


def parse_measures(text: str) -> dict[str, float]:
    return {name.lower(): float(value) for name, value in MEAS_RE.findall(text)}


def netlist(
    scores: tuple[float, float, float],
    branch_model: str,
    branch_width_u: float,
    tail_width_u: float,
    tail_gate_v: float,
) -> str:
    if branch_model not in {"NMOS", "NREL", "NSENSE"}:
        raise ValueError(f"unknown branch model: {branch_model}")
    score_sources = "\n".join(f"Vscore{k} score{k} 0 {score:.12g}" for k, score in enumerate(scores))
    branches = "\n".join(
        [
            f"Vsense{k} vdd drain{k} 0",
            f"Mbranch{k} drain{k} score{k} src 0 {branch_model} W={branch_width_u:.12g}u L=180n",
        ][line]
        for k in range(3)
        for line in range(2)
    )
    measures = "\n".join(f".meas tran i{k} FIND I(Vsense{k}) AT=0.05n" for k in range(3))
    prints = "\n".join(f"print i{k}" for k in range(3))
    return f"""
* Three-class source-coupled MOS current competition.
* Branch currents are the hardware-native softmax-like class probabilities.
.param VDD=1.2
{mos_models()}
Vdd vdd 0 {{VDD}}
{score_sources}
Vtail tail 0 {tail_gate_v:.12g}

{branches}
Mtail src tail 0 0 NMOS W={tail_width_u:.12g}u L=180n
Rsrc src 0 1e12

.tran 1p 0.1n
{measures}
.control
run
{prints}
.endc
.end
""".lstrip()


def run_netlist(spice_bin: str, path: Path, text: str, timeout: float) -> dict[str, float]:
    path.write_text(text)
    cmd = [spice_bin, "-b", str(path)] if "ngspice" in Path(spice_bin).name.lower() else [spice_bin, str(path)]
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-3000:])
    parsed = parse_measures(proc.stdout + "\n" + proc.stderr)
    if not parsed:
        raise RuntimeError("ngspice produced no parseable measurements:\n" + (proc.stdout + proc.stderr)[-3000:])
    return parsed


def default_cases() -> list[tuple[float, float, float]]:
    return [
        (0.35, 0.35, 0.35),
        (0.38, 0.35, 0.35),
        (0.42, 0.38, 0.35),
        (0.46, 0.42, 0.35),
        (0.35, 0.38, 0.35),
        (0.35, 0.35, 0.38),
        (0.45, 0.40, 0.38),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="device_softmax_current_competition")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--branch-model", choices=["NMOS", "NREL", "NSENSE"], default="NREL")
    ap.add_argument("--branch-width-u", type=float, default=12.0)
    ap.add_argument("--tail-width-u", type=float, default=16.0)
    ap.add_argument("--tail-gate-v", type=float, default=0.55)
    args = ap.parse_args()

    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    results = ROOT / "spice/results"
    tables = ROOT / "results/tables"
    for directory in [generated, results, tables]:
        directory.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)

    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag)
    rows: list[dict[str, float | int | bool | str]] = []
    t0 = time.perf_counter()
    for idx, scores in enumerate(default_cases()):
        parsed = run_netlist(
            spice_bin,
            generated / f"{safe_tag}_{idx}.cir",
            netlist(scores, args.branch_model, args.branch_width_u, args.tail_width_u, args.tail_gate_v),
            args.timeout,
        )
        currents = [abs(parsed[f"i{k}"]) for k in range(3)]
        total = sum(currents)
        shares = [current / total if total > 0 else 0.0 for current in currents]
        rows.append(
            {
                "case": idx,
                "score0_v": scores[0],
                "score1_v": scores[1],
                "score2_v": scores[2],
                "i0_a": currents[0],
                "i1_a": currents[1],
                "i2_a": currents[2],
                "total_i_a": total,
                "share0": shares[0],
                "share1": shares[1],
                "share2": shares[2],
                "score_argmax": int(max(range(3), key=lambda k: scores[k])),
                "share_argmax": int(max(range(3), key=lambda k: shares[k])),
                "argmax_matches": int(max(range(3), key=lambda k: scores[k]))
                == int(max(range(3), key=lambda k: shares[k])),
            }
        )

    df = pd.DataFrame(rows)
    curve_path = results / f"{safe_tag}.csv"
    table_path = tables / f"{safe_tag}.csv"
    df.to_csv(curve_path, index=False)
    df.to_csv(table_path, index=False)

    equal = df.iloc[0]
    total_values = df["total_i_a"].to_numpy()
    summary = {
        "simulator": version,
        "architecture": "three_class_source_coupled_mos_current_competition",
        "status": "softmax_like_current_normalizer_primitive",
        "model_level": "ngspice built-in LEVEL=1 MOS models; not a subthreshold/PDK model.",
        "branch_model": args.branch_model,
        "branch_width_u": args.branch_width_u,
        "tail_width_u": args.tail_width_u,
        "tail_gate_v": args.tail_gate_v,
        "no_behavioral_exp_or_divide_in_signal_path": True,
        "probabilities_measured_offline_as_current_shares": True,
        "all_argmax_matches": bool(df["argmax_matches"].all()),
        "equal_input_max_share_error": float(
            max(abs(float(equal[f"share{k}"]) - 1.0 / 3.0) for k in range(3))
        ),
        "total_current_min_a": float(total_values.min()),
        "total_current_max_a": float(total_values.max()),
        "total_current_relative_span": float(
            (total_values.max() - total_values.min()) / total_values.mean()
            if total_values.mean() > 0
            else 0.0
        ),
        "curve": str(curve_path),
        "table_curve": str(table_path),
        "wall_time_s": time.perf_counter() - t0,
        "interpretation": (
            "This is not exact softmax in the current LEVEL=1 model. It is a hardware-native normalized "
            "positive-current competition. In subthreshold CMOS, the same topology is the standard path "
            "toward exponential softmax-like branch currents."
        ),
    }
    summary_path = results / f"{safe_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
