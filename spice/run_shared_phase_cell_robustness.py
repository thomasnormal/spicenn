from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from run_device_mnist01_scalar_training import sanitize_tag
from run_device_sequential_training import run_netlist
from run_spice_sweep import ROOT, detect_spice


TEMPLATE = ROOT / "spice/templates/shared_phase_local_feature_cell_full_smoke.cir"
VDD_NOMINAL = 1.2


@dataclass(frozen=True)
class CellStress:
    dvdd: float = 0.0
    dx: float = 0.0
    j_fwd: float = 0.0
    j_err: float = 0.0
    j_bwd: float = 0.0
    j_edp: float = 0.0
    dwhp: float = 0.0
    dwhn: float = 0.0
    dbhp: float = 0.0
    dbhn: float = 0.0
    dvwp: float = 0.0
    dvwn: float = 0.0
    dvto_rvwp_n: float = 0.0
    dvto_rvwn_n: float = 0.0
    dkp_rvwp_n: float = 0.0
    dkp_rvwn_n: float = 0.0
    cfeed: float = 0.0
    cpar_dyn: float = 0.0
    temp_c: float = 27.0


@dataclass(frozen=True)
class RobustnessThresholds:
    score_margin_min: float = 50e-3
    rvw_margin_min: float = 200e-3
    hdp_min: float = 50e-3
    hdn_max: float = 25e-3


def pulse(delay: float, width: float, period: float = 24e-9) -> str:
    delay = max(0.0, delay)
    width = max(1e-12, width)
    return f"PULSE(0 {VDD_NOMINAL:.12g} {delay:.12g} 10p 10p {width:.12g} {period:.12g})"


def generate_netlist(stress: CellStress = CellStress()) -> str:
    text = TEMPLATE.read_text()
    text = text.replace(
        "Runnable shared-phase local feature cell full smoke",
        "* Runnable shared-phase local feature cell full smoke",
        1,
    )
    vdd = VDD_NOMINAL + stress.dvdd
    x = min(VDD_NOMINAL, max(0.0, 0.85 + stress.dx))
    replacements = {
        "Vdd vdd 0 1.2": f"Vdd vdd 0 {vdd:.12g}",
        "Vx x 0 0.85": f"Vx x 0 {x:.12g}",
        "Vfwd fwd 0 PULSE(0 1.2 1n 10p 10p 4n 24n)": f"Vfwd fwd 0 {pulse(1e-9 + stress.j_fwd, 4e-9)}",
        "Verr err 0 PULSE(0 1.2 6n 10p 10p 3n 24n)": f"Verr err 0 {pulse(6e-9 + stress.j_err, 3e-9)}",
        "Vbwd bwd 0 PULSE(0 1.2 9n 10p 10p 3n 24n)": f"Vbwd bwd 0 {pulse(9e-9 + stress.j_bwd, 3e-9)}",
        "Vedp edp 0 PULSE(0 1.2 6n 10p 10p 6n 24n)": f"Vedp edp 0 {pulse(6e-9 + stress.j_edp, 6e-9)}",
        "Cwhp whp 0 20f IC=0.7": f"Cwhp whp 0 20f IC={0.7 + stress.dwhp:.12g}",
        "Cwhn whn 0 20f IC=0.2": f"Cwhn whn 0 20f IC={0.2 + stress.dwhn:.12g}",
        "Cbhp bhp 0 20f IC=0.5": f"Cbhp bhp 0 20f IC={0.5 + stress.dbhp:.12g}",
        "Cbhn bhn 0 20f IC=0.2": f"Cbhn bhn 0 20f IC={0.2 + stress.dbhn:.12g}",
        "Cvwp vwp 0 20f IC=0.52": f"Cvwp vwp 0 20f IC={0.52 + stress.dvwp:.12g}",
        "Cvwn vwn 0 20f IC=0.25": f"Cvwn vwn 0 20f IC={0.25 + stress.dvwn:.12g}",
        "Mrvwp_n rvwp vwn rvw_src 0 NSENSE W=6u L=180n": "Mrvwp_n rvwp vwn rvw_src 0 NS_RVWP W=6u L=180n",
        "Mrvwn_n rvwn vwp rvw_src 0 NSENSE W=6u L=180n": "Mrvwn_n rvwn vwp rvw_src 0 NS_RVWN W=6u L=180n",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    mismatch_models = "\n".join(
        [
            f".model NS_RVWP NMOS LEVEL=1 VTO={0.03 + stress.dvto_rvwp_n:.12g} KP={260e-6 * (1.0 + stress.dkp_rvwp_n):.12g} LAMBDA=0.03 GAMMA=0.05 PHI=0.60",
            f".model NS_RVWN NMOS LEVEL=1 VTO={0.03 + stress.dvto_rvwn_n:.12g} KP={260e-6 * (1.0 + stress.dkp_rvwn_n):.12g} LAMBDA=0.03 GAMMA=0.05 PHI=0.60",
        ]
    )
    text = text.replace(".model NSENSE NMOS LEVEL=1 VTO=0.03 KP=260u LAMBDA=0.03 GAMMA=0.05 PHI=0.60", ".model NSENSE NMOS LEVEL=1 VTO=0.03 KP=260u LAMBDA=0.03 GAMMA=0.05 PHI=0.60\n" + mismatch_models)
    text = text.replace(".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6", f".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6\n.temp {stress.temp_c:.12g}")

    parasitic_lines = ["* Robustness stress: clock feedthrough and extra dynamic capacitance."]
    if stress.cfeed > 0.0:
        parasitic_lines += [
            f"Cfeed_fwd_pre fwd pre {stress.cfeed:.12g}",
            f"Cfeed_fwd_score fwd score {stress.cfeed:.12g}",
            f"Cfeed_fwd_scoren fwd scoren {stress.cfeed:.12g}",
            f"Cfeed_err_rvwp err rvwp {stress.cfeed:.12g}",
            f"Cfeed_err_rvwn err rvwn {stress.cfeed:.12g}",
            f"Cfeed_bwd_hdp bwd hdp {stress.cfeed:.12g}",
        ]
    if stress.cpar_dyn > 0.0:
        parasitic_lines += [
            f"Cpar_pre pre 0 {stress.cpar_dyn:.12g}",
            f"Cpar_act act 0 {stress.cpar_dyn:.12g}",
            f"Cpar_score score 0 {stress.cpar_dyn:.12g}",
            f"Cpar_scoren scoren 0 {stress.cpar_dyn:.12g}",
            f"Cpar_hdp hdp 0 {stress.cpar_dyn:.12g}",
        ]
    parasitics = "\n".join(parasitic_lines)
    measurements = "\n".join(
        [
            ".meas tran score_margin PARAM='score_forward-scoren_forward'",
            ".meas tran rvw_margin PARAM='rvwp_after_err-rvwn_after_err'",
            ".meas tran hidden_credit_margin PARAM='hdp_after_bwd-hdn_after_bwd'",
        ]
    )
    text = text.replace(".tran 10p 24n uic", parasitics + "\n\n.tran 10p 24n uic")
    text = text.replace(".end", measurements + "\n.end")
    return text


def sample_stresses(
    count: int,
    *,
    seed: int,
    vdd_sigma: float,
    input_sigma: float,
    phase_jitter_sigma: float,
    state_sigma: float,
    latch_vto_sigma: float,
    latch_kp_sigma: float,
    cfeed: float,
    cpar_dyn: float,
    temp_c: float,
) -> list[CellStress]:
    rng = np.random.default_rng(seed)
    return [
        CellStress(
            dvdd=float(rng.normal(0.0, vdd_sigma)),
            dx=float(rng.normal(0.0, input_sigma)),
            j_fwd=float(rng.normal(0.0, phase_jitter_sigma)),
            j_err=float(rng.normal(0.0, phase_jitter_sigma)),
            j_bwd=float(rng.normal(0.0, phase_jitter_sigma)),
            j_edp=float(rng.normal(0.0, phase_jitter_sigma)),
            dwhp=float(rng.normal(0.0, state_sigma)),
            dwhn=float(rng.normal(0.0, state_sigma)),
            dbhp=float(rng.normal(0.0, state_sigma)),
            dbhn=float(rng.normal(0.0, state_sigma)),
            dvwp=float(rng.normal(0.0, state_sigma)),
            dvwn=float(rng.normal(0.0, state_sigma)),
            dvto_rvwp_n=float(rng.normal(0.0, latch_vto_sigma)),
            dvto_rvwn_n=float(rng.normal(0.0, latch_vto_sigma)),
            dkp_rvwp_n=float(rng.normal(0.0, latch_kp_sigma)),
            dkp_rvwn_n=float(rng.normal(0.0, latch_kp_sigma)),
            cfeed=cfeed,
            cpar_dyn=cpar_dyn,
            temp_c=temp_c,
        )
        for _ in range(count)
    ]


def classify(measures: dict[str, float], thresholds: RobustnessThresholds) -> dict[str, Any]:
    failures: list[str] = []
    if measures.get("score_margin", float("-inf")) < thresholds.score_margin_min:
        failures.append("score_margin")
    if measures.get("rvw_margin", float("-inf")) < thresholds.rvw_margin_min:
        failures.append("rvw_margin")
    if measures.get("hdp_after_bwd", float("-inf")) < thresholds.hdp_min:
        failures.append("hdp")
    if measures.get("hdn_after_bwd", float("inf")) > thresholds.hdn_max:
        failures.append("hdn_leak")
    return {"passed": not failures, "failures": failures}


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    generated = ROOT / "spice/generated"
    tables = ROOT / "results/tables"
    generated.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    tag = sanitize_tag(args.tag)
    spice_bin, version = detect_spice(args.spice_bin)
    thresholds = RobustnessThresholds(
        score_margin_min=args.score_margin_min,
        rvw_margin_min=args.rvw_margin_min,
        hdp_min=args.hdp_min,
        hdn_max=args.hdn_max,
    )
    stresses = sample_stresses(
        args.runs,
        seed=args.seed,
        vdd_sigma=args.vdd_sigma,
        input_sigma=args.input_sigma,
        phase_jitter_sigma=args.phase_jitter_sigma_ps * 1e-12,
        state_sigma=args.state_sigma,
        latch_vto_sigma=args.latch_vto_sigma,
        latch_kp_sigma=args.latch_kp_sigma,
        cfeed=args.cfeed,
        cpar_dyn=args.cpar_dyn,
        temp_c=args.temp_c,
    )
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for idx, stress in enumerate(stresses):
        path = generated / f"{tag}_{idx:04d}.cir"
        measures = run_netlist(spice_bin, path, generate_netlist(stress), timeout=args.timeout)
        verdict = classify(measures, thresholds)
        rows.append(
            {
                "run": idx,
                **asdict(stress),
                **measures,
                "passed": verdict["passed"],
                "failures": ",".join(verdict["failures"]),
            }
        )
    csv_path = tables / f"{tag}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()) if rows else ["run"])
        writer.writeheader()
        writer.writerows(rows)
    passed = sum(1 for row in rows if row["passed"])
    failure_counts: dict[str, int] = {}
    for row in rows:
        for failure in str(row["failures"]).split(","):
            if failure:
                failure_counts[failure] = failure_counts.get(failure, 0) + 1
    margin_stats = {}
    for name in ["score_margin", "rvw_margin", "hdp_after_bwd", "hdn_after_bwd", "hidden_credit_margin"]:
        values = [float(row[name]) for row in rows if name in row]
        if values:
            margin_stats[name] = {
                "min": min(values),
                "mean": sum(values) / len(values),
                "max": max(values),
            }
    summary = {
        "simulator": version,
        "architecture": "shared_phase_local_feature_cell_robustness",
        "model_level": "ngspice built-in LEVEL=1 MOS models; not a foundry PDK.",
        "runs": args.runs,
        "seed": args.seed,
        "yield": passed / args.runs if args.runs else 0.0,
        "failed": args.runs - passed,
        "failure_counts": failure_counts,
        "margin_stats": margin_stats,
        "thresholds": asdict(thresholds),
        "stress": {
            "vdd_sigma": args.vdd_sigma,
            "input_sigma": args.input_sigma,
            "phase_jitter_sigma_ps": args.phase_jitter_sigma_ps,
            "state_sigma": args.state_sigma,
            "latch_vto_sigma": args.latch_vto_sigma,
            "latch_kp_sigma": args.latch_kp_sigma,
            "cfeed": args.cfeed,
            "cpar_dyn": args.cpar_dyn,
            "temp_c": args.temp_c,
        },
        "csv": str(csv_path),
        "wall_time_s": time.perf_counter() - start,
    }
    summary_path = tables / f"{tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spice-bin", default=None)
    ap.add_argument("--tag", default="shared_phase_cell_robustness")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--vdd-sigma", type=float, default=0.0)
    ap.add_argument("--input-sigma", type=float, default=0.0)
    ap.add_argument("--phase-jitter-sigma-ps", type=float, default=0.0)
    ap.add_argument("--state-sigma", type=float, default=0.0)
    ap.add_argument("--latch-vto-sigma", type=float, default=0.0)
    ap.add_argument("--latch-kp-sigma", type=float, default=0.0)
    ap.add_argument("--cfeed", type=float, default=0.0)
    ap.add_argument("--cpar-dyn", type=float, default=0.0)
    ap.add_argument("--temp-c", type=float, default=27.0)
    ap.add_argument("--score-margin-min", type=float, default=50e-3)
    ap.add_argument("--rvw-margin-min", type=float, default=200e-3)
    ap.add_argument("--hdp-min", type=float, default=50e-3)
    ap.add_argument("--hdn-max", type=float, default=25e-3)
    return ap


def validate_args(args: argparse.Namespace) -> None:
    if args.runs <= 0:
        raise ValueError("runs must be positive")
    for name in [
        "vdd_sigma",
        "input_sigma",
        "phase_jitter_sigma_ps",
        "state_sigma",
        "latch_vto_sigma",
        "latch_kp_sigma",
        "cfeed",
        "cpar_dyn",
    ]:
        if getattr(args, name) < 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be nonnegative")


def main_for_test(argv: list[str]) -> argparse.Namespace:
    args = build_arg_parser().parse_args(argv)
    validate_args(args)
    return args


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)
    print(json.dumps(run_campaign(args), indent=2))


if __name__ == "__main__":
    main()
