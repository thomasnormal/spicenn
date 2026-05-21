#!/usr/bin/env python3
"""Characterize the MOS read kernel of one readout branch.

The full learner treats a readout synapse as if its forward contribution is
roughly monotone in both the stored weight-cap voltage and the pre-activation.
This harness measures that assumption directly for the production readout
branch topologies.  It does not use behavioral multiplication; the score
capacitor is charged through the same MOS stacks used by the device generator.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from spicenn import ReadoutBranch
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from spicenn import ReadoutBranch

import run_device_xor2_random_hidden as direct_flow
from run_device_multicell_classifier import mos_models
from run_spice_sweep import ROOT, detect_spice, run_text_netlist, run_tiny_test
from _util import MEAS_RE, parse_measures


SURFACE_MODES = {
    "floating_delta",
    "clamped_current",
    "diode_voltage",
    "diode_current",
    "diode_mirror_voltage",
}
DIODE_SURFACE_MODES = {"diode_voltage", "diode_current", "diode_mirror_voltage"}


def parse_float_list(text: str) -> list[float]:
    values = [float(part) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return values


def safe_tag(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)


def width_for_branch(design: direct_flow.SynapseDesign, branch: str) -> float:
    if branch == "pos":
        return design.output_forward_pos_width_u
    if branch == "neg":
        return design.output_forward_neg_width_u
    raise ValueError(f"unknown branch: {branch}")


def branch_devices(design: direct_flow.SynapseDesign, branch: str, *, width_scale: float) -> str:
    width = width_for_branch(design, branch) * width_scale
    return ReadoutBranch(
        "branch",
        style=design.output_forward_style,
        branch=branch,
        activation_node="act",
        weight_node="weight",
        score_node="score",
        width_u=width,
        internal_prefix="rb",
        buffered_activation_name="actbuf",
    ).render_spice()


def readout_branch_netlist(
    *,
    design_name: str,
    branch: str,
    act_v: float,
    weight_v: float,
    width_scale: float = 1.0,
    score_reset_v: float = 0.0,
    cap_f: float = 10.0,
    tran_step_ps: float = 1.0,
    sample_ns: float = 2.8,
    stop_ns: float = 3.2,
    surface_mode: str = "floating_delta",
    diode_width_u: float = 256.0,
    mirror_cap_f: float = 20.0,
) -> str:
    if surface_mode not in SURFACE_MODES:
        raise ValueError(f"unknown surface mode: {surface_mode}")
    design = direct_flow.SYNAPSE_DESIGNS[design_name]
    devices = branch_devices(design, branch, width_scale=width_scale)
    if surface_mode == "floating_delta":
        score_cell = f"""
Vscore_reset score_reset 0 {score_reset_v:.12g}
Cscore score 0 {cap_f:.12g}f IC={score_reset_v:.12g}
Rscore score 0 1e13
Mscore_rst score rstf score_reset 0 NREL W=4u L=180n
""".strip()
        score_measures = f"""
.meas tran score_before FIND V(score) AT=0.69n
.meas tran score_after FIND V(score) AT={sample_ns:.12g}n
.meas tran score_delta PARAM='score_after-score_before'
""".strip()
        score_print = "print score_before score_after score_delta"
    elif surface_mode == "clamped_current":
        score_cell = f"Vscore_clamp score 0 {score_reset_v:.12g}"
        score_measures = f"""
.meas tran score_before FIND I(Vscore_clamp) AT=0.69n
.meas tran score_after FIND I(Vscore_clamp) AT={sample_ns:.12g}n
.meas tran score_delta FIND I(Vscore_clamp) AT={sample_ns:.12g}n
""".strip()
        score_print = "print score_before score_after score_delta"
    elif surface_mode == "diode_voltage":
        score_cell = f"""
Cscore score 0 {cap_f:.12g}f IC=0
Rscore score 0 1e12
Mscore_diode score score 0 0 NSENSE W={diode_width_u:.12g}u L=180n
Mscore_rst score rstf 0 0 NMOS W=4u L=180n
""".strip()
        score_measures = f"""
.meas tran score_before FIND V(score) AT=0.69n
.meas tran score_after FIND V(score) AT={sample_ns:.12g}n
.meas tran score_delta PARAM='score_after-score_before'
""".strip()
        score_print = "print score_before score_after score_delta"
    elif surface_mode == "diode_current":
        score_cell = f"""
Cscore score 0 {cap_f:.12g}f IC=0
Rscore score 0 1e12
Vscore_sense score_sense 0 0
Mscore_diode score score score_sense 0 NSENSE W={diode_width_u:.12g}u L=180n
Mscore_rst score rstf 0 0 NMOS W=4u L=180n
""".strip()
        score_measures = f"""
.meas tran score_before FIND I(Vscore_sense) AT=0.69n
.meas tran score_after FIND I(Vscore_sense) AT={sample_ns:.12g}n
.meas tran score_delta FIND I(Vscore_sense) AT={sample_ns:.12g}n
""".strip()
        score_print = "print score_before score_after score_delta"
    else:
        score_cell = f"""
Cscore score 0 {cap_f:.12g}f IC=0
Rscore score 0 1e12
Mscore_diode score score 0 0 NSENSE W={diode_width_u:.12g}u L=180n
Mscore_rst score rstf 0 0 NMOS W=4u L=180n
Cmirror mirror 0 {mirror_cap_f:.12g}f IC=1.2
Rmirror mirror 0 1e12
Mmirror_rst vdd rstf mirror 0 NSENSE W=16u L=180n
Mmirror_sink mirror score 0 0 NSENSE W={diode_width_u:.12g}u L=180n
""".strip()
        score_measures = f"""
.meas tran mirror_before FIND V(mirror) AT=0.69n
.meas tran mirror_after FIND V(mirror) AT={sample_ns:.12g}n
.meas tran score_before PARAM='1.2-mirror_before'
.meas tran score_after PARAM='1.2-mirror_after'
.meas tran score_delta PARAM='mirror_before-mirror_after'
""".strip()
        score_print = "print mirror_before mirror_after score_before score_after score_delta"
    return f"""
* Single readout-branch MOS transfer characterization.
* design={design_name} branch={branch} style={design.output_forward_style} surface_mode={surface_mode}
.param VDD=1.2
{mos_models()}
Vdd vdd 0 {{VDD}}
Vact act 0 {act_v:.12g}
Vweight weight 0 {weight_v:.12g}
Vrstf rstf 0 PWL(0n 1.2 0.45n 1.2 0.50n 0 3.2n 0)
Vfwd fwd 0 PWL(0n 0 0.70n 0 0.75n 1.2 3.00n 1.2 3.05n 0 3.2n 0)

{score_cell}

{devices}

.tran {tran_step_ps:.12g}p {stop_ns:.12g}n uic
{score_measures}
.control
run
{score_print}
.endc
.end
""".lstrip()


def run_netlist(spice_bin: str, path: Path, netlist: str, timeout: float) -> dict[str, float]:
    proc = run_text_netlist(spice_bin, path, netlist, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-3000:])
    parsed = parse_measures(proc.stdout + "\n" + proc.stderr)
    if not parsed:
        raise RuntimeError("SPICE produced no parseable measurements:\n" + (proc.stdout + proc.stderr)[-3000:])
    return parsed


def monotone_fraction(df: pd.DataFrame, *, fixed: str, swept: str, value: str, tolerance: float) -> float:
    groups = 0
    ok = 0
    for _key, group in df.sort_values(swept).groupby(fixed):
        vals = group.sort_values(swept)[value].to_numpy(dtype=float)
        if len(vals) < 2:
            continue
        groups += 1
        ok += int(bool(np.all(np.diff(vals) >= -tolerance)))
    return ok / groups if groups else 0.0


def bilinear_fit(df: pd.DataFrame) -> dict[str, Any]:
    act = df["act_v"].to_numpy(dtype=float)
    weight = df["weight_v"].to_numpy(dtype=float)
    y = df["score_delta_v"].to_numpy(dtype=float)
    x = np.column_stack([np.ones_like(act), act, weight, act * weight])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    return {
        "features": ["1", "act_v", "weight_v", "act_v*weight_v"],
        "coef": [float(c) for c in coef],
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
        "rmse_v": float(np.sqrt(np.mean((y - pred) ** 2))) if len(y) else 0.0,
    }


def summarize(df: pd.DataFrame, *, tolerance: float) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    for (design, branch), group in df.groupby(["design", "branch"]):
        min_act = float(group["act_v"].min())
        min_weight = float(group["weight_v"].min())
        zero_act = group[np.isclose(group["act_v"], min_act)]["score_delta_v"].abs()
        min_weight_delta = group[np.isclose(group["weight_v"], min_weight)]["score_delta_v"].abs()
        groups.append(
            {
                "design": design,
                "branch": branch,
                "rows": int(len(group)),
                "style": str(group["style"].iloc[0]),
                "score_delta_min_v": float(group["score_delta_v"].min()),
                "score_delta_max_v": float(group["score_delta_v"].max()),
                "score_delta_range_v": float(group["score_delta_v"].max() - group["score_delta_v"].min()),
                "zero_act_abs_delta_max_v": float(zero_act.max()) if len(zero_act) else 0.0,
                "min_weight_abs_delta_max_v": float(min_weight_delta.max()) if len(min_weight_delta) else 0.0,
                "monotone_vs_weight_fraction": monotone_fraction(
                    group, fixed="act_v", swept="weight_v", value="score_delta_v", tolerance=tolerance
                ),
                "monotone_vs_act_fraction": monotone_fraction(
                    group, fixed="weight_v", swept="act_v", value="score_delta_v", tolerance=tolerance
                ),
                "bilinear_fit": bilinear_fit(group),
            }
        )
    return {"groups": groups}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="device_readout_branch_surface")
    ap.add_argument("--simulator", default=None)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--designs", default="split_signed_v1,split_signed_passact_v1,split_signed_passact_buffered_v1")
    ap.add_argument("--branches", default="pos,neg")
    ap.add_argument("--act-values", type=parse_float_list, default=parse_float_list("0,0.2,0.4,0.6,0.8,1.0,1.2"))
    ap.add_argument("--weight-values", type=parse_float_list, default=parse_float_list("0.3,0.45,0.6,0.75,0.9,1.05"))
    ap.add_argument("--width-scale", type=float, default=1.0)
    ap.add_argument("--score-reset-v", type=float, default=0.0)
    ap.add_argument("--cap-f", type=float, default=10.0)
    ap.add_argument("--tran-step-ps", type=float, default=1.0)
    ap.add_argument("--sample-ns", type=float, default=2.8)
    ap.add_argument("--stop-ns", type=float, default=3.2)
    ap.add_argument(
        "--surface-mode",
        choices=sorted(SURFACE_MODES),
        default="floating_delta",
    )
    ap.add_argument("--diode-width-u", type=float, default=256.0)
    ap.add_argument("--mirror-cap-f", type=float, default=20.0)
    ap.add_argument("--monotone-tolerance-v", type=float, default=1e-5)
    args = ap.parse_args()

    designs = [part.strip() for part in args.designs.split(",") if part.strip()]
    branches = [part.strip() for part in args.branches.split(",") if part.strip()]
    for design_name in designs:
        if design_name not in direct_flow.SYNAPSE_DESIGNS:
            raise SystemExit(f"unknown design: {design_name}")
    for branch in branches:
        if branch not in {"pos", "neg"}:
            raise SystemExit(f"unknown branch: {branch}")
    if args.diode_width_u <= 0:
        raise SystemExit("--diode-width-u must be positive.")
    if args.mirror_cap_f <= 0:
        raise SystemExit("--mirror-cap-f must be positive.")

    spice_bin, version = detect_spice(args.simulator)
    generated = ROOT / "spice/generated"
    results = ROOT / "spice/results"
    tables = ROOT / "results/tables"
    for path in (generated, results, tables):
        path.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)

    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    tag = safe_tag(args.tag)
    for design_name in designs:
        design = direct_flow.SYNAPSE_DESIGNS[design_name]
        for branch in branches:
            for act_v in args.act_values:
                for weight_v in args.weight_values:
                    deck = readout_branch_netlist(
                        design_name=design_name,
                        branch=branch,
                        act_v=act_v,
                        weight_v=weight_v,
                        width_scale=args.width_scale,
                        score_reset_v=args.score_reset_v,
                        cap_f=args.cap_f,
                        tran_step_ps=args.tran_step_ps,
                        sample_ns=args.sample_ns,
                        stop_ns=args.stop_ns,
                        surface_mode=args.surface_mode,
                        diode_width_u=args.diode_width_u,
                        mirror_cap_f=args.mirror_cap_f,
                    )
                    point_tag = f"{tag}_{design_name}_{branch}_a{act_v:.3f}_w{weight_v:.3f}"
                    parsed = run_netlist(spice_bin, generated / f"{safe_tag(point_tag)}.cir", deck, args.timeout)
                    rows.append(
                        {
                            "design": design_name,
                            "style": design.output_forward_style,
                            "branch": branch,
                            "act_v": float(act_v),
                            "weight_v": float(weight_v),
                            "score_before_v": parsed["score_before"],
                            "score_after_v": parsed["score_after"],
                            "score_delta_v": parsed["score_delta"],
                        }
                    )

    df = pd.DataFrame(rows)
    summary = {
        "tag": tag,
        "simulator": version,
        "elapsed_s": time.perf_counter() - t0,
        "designs": designs,
        "branches": branches,
        "act_values": [float(v) for v in args.act_values],
        "weight_values": [float(v) for v in args.weight_values],
        "width_scale": args.width_scale,
        "score_reset_v": args.score_reset_v,
        "cap_f": args.cap_f,
        "tran_step_ps": args.tran_step_ps,
        "sample_ns": args.sample_ns,
        "surface_mode": args.surface_mode,
        "diode_width_u": args.diode_width_u if args.surface_mode in DIODE_SURFACE_MODES else None,
        "mirror_cap_f": args.mirror_cap_f if args.surface_mode == "diode_mirror_voltage" else None,
        **summarize(df, tolerance=args.monotone_tolerance_v),
    }

    csv_path = results / f"{tag}.csv"
    table_csv_path = tables / f"{tag}.csv"
    summary_path = results / f"{tag}_summary.json"
    table_summary_path = tables / f"{tag}_summary.json"
    df.to_csv(csv_path, index=False)
    df.to_csv(table_csv_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    table_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
