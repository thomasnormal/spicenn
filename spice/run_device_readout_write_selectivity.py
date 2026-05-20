#!/usr/bin/env python3
"""Isolate the multiclass readout write contract in SPICE.

The full 3-class runs can fail either because the output/error circuit produces
bad dp/dn rails or because the local readout write primitive does not move the
target row up and the non-target rows down.  This harness bypasses the output
head and drives explicit dp/dn rails into the actual `readout_flow_updates`
netlist fragment used by the full device runner.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

import run_device_xor2_random_hidden as direct_flow
from run_device_multicell_classifier import mos_models, pwl
from run_spice_sweep import ROOT, detect_spice, run_tiny_test


def constant_until(stop_ns: float, value: float) -> str:
    return pwl([(0.0, value), (stop_ns, value)])


def pulse(start_ns: float, end_ns: float, high_v: float, stop_ns: float) -> str:
    return pwl(
        [
            (0.0, 0.0),
            (start_ns, 0.0),
            (start_ns + 0.02, high_v),
            (end_ns, high_v),
            (end_ns + 0.02, 0.0),
            (stop_ns, 0.0),
        ]
    )


def readout_capacitors(outputs: int, hidden: int, center_v: float, cap_f: float) -> str:
    lines: list[str] = []
    for out in range(outputs):
        for h in range(hidden):
            for suffix in ("p", "n"):
                node = f"vw{out}{h}{suffix}"
                lines += [
                    f"C{node} {node} 0 {cap_f:.12g}f IC={center_v:.12g}",
                    f"R{node} {node} 0 1e15",
                ]
        for suffix in ("p", "n"):
            node = f"vbo{out}{suffix}"
            lines += [
                f"C{node} {node} 0 {cap_f:.12g}f IC={center_v:.12g}",
                f"R{node} {node} 0 1e15",
            ]
    return "\n".join(lines)


def error_sources(
    label: int,
    *,
    outputs: int,
    high_v: float,
    low_v: float,
    stop_ns: float,
) -> str:
    lines: list[str] = []
    for out in range(outputs):
        dp_v = high_v if out == label else low_v
        dn_v = low_v if out == label else high_v
        lines += [
            f"Vdp{out} dp{out} 0 {constant_until(stop_ns, dp_v)}",
            f"Vdn{out} dn{out} 0 {constant_until(stop_ns, dn_v)}",
        ]
    return "\n".join(lines)


def measurement_lines(outputs: int, hidden: int, before_ns: float, after_ns: float) -> str:
    lines: list[str] = []
    for out in range(outputs):
        row_terms: list[str] = []
        for h in range(hidden):
            for suffix in ("p", "n"):
                node = f"vw{out}{h}{suffix}"
                lines += [
                    f".meas tran {node}_before FIND V({node}) AT={before_ns:.12g}n",
                    f".meas tran {node}_after FIND V({node}) AT={after_ns:.12g}n",
                ]
            term = f"d_w{out}{h}_signed"
            row_terms.append(term)
            lines.append(
                f".meas tran {term} PARAM='(vw{out}{h}p_after-vw{out}{h}n_after)-(vw{out}{h}p_before-vw{out}{h}n_before)'"
            )
        lines.append(f".meas tran row{out}_signed_delta PARAM='" + "+".join(row_terms) + "'")
    return "\n".join(lines)


def print_line(outputs: int) -> str:
    names = [f"row{out}_signed_delta" for out in range(outputs)]
    return "print " + " ".join(names)


def build_readout_write_selectivity_netlist(
    *,
    label: int,
    hidden_values: list[float],
    outputs: int = 3,
    center_v: float = 0.64,
    weight_cap_f: float = 4.0,
    readout_update_width_u: float = 0.00002,
    readout_pos_update_width_u: float | None = None,
    readout_neg_update_width_u: float | None = None,
    readout_charge_update_width_u: float | None = None,
    readout_discharge_update_width_u: float | None = None,
    readout_flow_polarity: str = "normal",
    readout_write_high_v: float = 1.0,
    readout_write_low_v: float = 0.16,
    error_high_v: float = 1.1,
    error_low_v: float = 0.0,
    readout_flow_write_mode: str = "bounded_charge_discharge",
    write_error_exclusion: str = "none",
    write_error_exclusion_width_u: float = 8.0,
    bwd_start_ns: float = 1.0,
    bwd_end_ns: float = 5.0,
    stop_ns: float = 8.0,
) -> str:
    if not 0 <= label < outputs:
        raise ValueError("label must be a valid output index.")
    if not hidden_values:
        raise ValueError("hidden_values must be nonempty.")
    if not readout_write_low_v < center_v < readout_write_high_v:
        raise ValueError("bounded write rails must straddle the initial center voltage.")
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(len(hidden_values))
        direct_flow.set_output_count(outputs)
        updates = direct_flow.readout_flow_updates(
            readout_update_width_u=readout_update_width_u,
            output_bias_update_width_u=0.0,
            flow_pre_store="shared_node",
            readout_flow_polarity=readout_flow_polarity,
            readout_flow_write_mode=readout_flow_write_mode,
            readout_pos_update_width_u=readout_pos_update_width_u,
            readout_neg_update_width_u=readout_neg_update_width_u,
            readout_charge_update_width_u=readout_charge_update_width_u,
            readout_discharge_update_width_u=readout_discharge_update_width_u,
            write_error_exclusion=write_error_exclusion,
            write_error_exclusion_width_u=write_error_exclusion_width_u,
        )
        caps = readout_capacitors(outputs, len(hidden_values), center_v, weight_cap_f)
        measures = measurement_lines(outputs, len(hidden_values), before_ns=bwd_start_ns - 0.1, after_ns=bwd_end_ns + 0.5)
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    act_sources = "\n".join(
        f"Vact{h} act{h} 0 {constant_until(stop_ns, value)}" for h, value in enumerate(hidden_values)
    )
    return f"""
* Isolated multiclass readout write-selectivity deck.
.param VDD=1.2
{mos_models()}
Vdd vdd 0 {{VDD}}
Vbwd bwd 0 {pulse(bwd_start_ns, bwd_end_ns, 1.2, stop_ns)}
Vapply apply 0 0
Vwhigh whigh 0 {readout_write_high_v:.12g}
Vwlow wlow 0 {readout_write_low_v:.12g}
Vwcenter wcenter 0 {center_v:.12g}
{act_sources}
{error_sources(label, outputs=outputs, high_v=error_high_v, low_v=error_low_v, stop_ns=stop_ns)}

{caps}

{updates}

.options method=gear maxord=2 rshunt=1e12 gmin=1e-12
.tran 5p {stop_ns:.12g}n uic
{measures}
.control
run
{print_line(outputs)}
.endc
.end
""".lstrip()


def expected_sign_ok(row_deltas: dict[int, float], label: int) -> bool:
    return row_deltas[label] > 0.0 and all(delta < 0.0 for out, delta in row_deltas.items() if out != label)


def run_case(spice_bin: str, path: Path, label: int, args: argparse.Namespace) -> dict[str, Any]:
    hidden_values = [float(part) for part in args.hidden_values.split(",") if part.strip()]
    netlist = build_readout_write_selectivity_netlist(
        label=label,
        hidden_values=hidden_values,
        outputs=args.outputs,
        center_v=args.center_v,
        weight_cap_f=args.weight_cap_f,
        readout_update_width_u=args.readout_update_width_u,
        readout_pos_update_width_u=args.readout_pos_update_width_u,
        readout_neg_update_width_u=args.readout_neg_update_width_u,
        readout_charge_update_width_u=args.readout_charge_update_width_u,
        readout_discharge_update_width_u=args.readout_discharge_update_width_u,
        readout_flow_polarity=args.readout_flow_polarity,
        readout_write_high_v=args.readout_write_high_v,
        readout_write_low_v=args.readout_write_low_v,
        error_high_v=args.error_high_v,
        error_low_v=args.error_low_v,
        readout_flow_write_mode=args.readout_flow_write_mode,
        write_error_exclusion=args.write_error_exclusion,
        write_error_exclusion_width_u=args.write_error_exclusion_width_u,
    )
    parsed = direct_flow.run_netlist(spice_bin, path, netlist, args.timeout)
    row_deltas = {out: float(parsed[f"row{out}_signed_delta"]) for out in range(args.outputs)}
    return {
        "label": label,
        "expected_sign_ok": expected_sign_ok(row_deltas, label),
        **{f"row{out}_signed_delta": delta for out, delta in row_deltas.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="device_readout_write_selectivity")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--outputs", type=int, default=3)
    ap.add_argument("--hidden-values", default="0.8,0.45")
    ap.add_argument("--center-v", type=float, default=0.64)
    ap.add_argument("--weight-cap-f", type=float, default=4.0)
    ap.add_argument("--readout-update-width-u", type=float, default=0.00002)
    ap.add_argument("--readout-pos-update-width-u", type=float, default=None)
    ap.add_argument("--readout-neg-update-width-u", type=float, default=None)
    ap.add_argument("--readout-charge-update-width-u", type=float, default=None)
    ap.add_argument("--readout-discharge-update-width-u", type=float, default=None)
    ap.add_argument("--readout-flow-polarity", choices=direct_flow.READOUT_FLOW_POLARITIES, default="normal")
    ap.add_argument("--readout-write-high-v", type=float, default=1.0)
    ap.add_argument("--readout-write-low-v", type=float, default=0.16)
    ap.add_argument("--error-high-v", type=float, default=1.1)
    ap.add_argument("--error-low-v", type=float, default=0.0)
    ap.add_argument("--readout-flow-write-mode", choices=direct_flow.READOUT_FLOW_WRITE_MODES, default="bounded_charge_discharge")
    ap.add_argument("--write-error-exclusion", choices=direct_flow.WRITE_ERROR_EXCLUSION_MODES, default="none")
    ap.add_argument("--write-error-exclusion-width-u", type=float, default=8.0)
    ap.add_argument("--simulator")
    args = ap.parse_args()

    spice_bin, version = detect_spice(args.simulator)
    generated = ROOT / "spice/generated"
    results = ROOT / "spice/results"
    tables = ROOT / "results/tables"
    for directory in [generated, results, tables]:
        directory.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)

    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag)
    t0 = time.perf_counter()
    rows = [
        run_case(spice_bin, generated / f"{safe_tag}_label{label}.cir", label, args)
        for label in range(args.outputs)
    ]
    df = pd.DataFrame(rows)
    curve_path = results / f"{safe_tag}.csv"
    table_path = tables / f"{safe_tag}.csv"
    df.to_csv(curve_path, index=False)
    df.to_csv(table_path, index=False)
    summary = {
        "simulator": version,
        "architecture": "isolated_multiclass_readout_write_selectivity",
        "model_level": "ngspice/Xyce MOS devices through the production readout_flow_updates fragment.",
        "curve": str(curve_path),
        "table_curve": str(table_path),
        "outputs": args.outputs,
        "hidden_values": args.hidden_values,
        "readout_flow_polarity": args.readout_flow_polarity,
        "readout_flow_write_mode": args.readout_flow_write_mode,
        "readout_write_high_v": args.readout_write_high_v,
        "readout_write_low_v": args.readout_write_low_v,
        "center_v": args.center_v,
        "all_labels_expected_sign": bool(df["expected_sign_ok"].all()),
        "wall_time_s": time.perf_counter() - t0,
    }
    summary_path = results / f"{safe_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
