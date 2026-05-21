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
        row_pos_terms: list[str] = []
        row_neg_terms: list[str] = []
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
            pos_term = f"d_w{out}{h}_pos"
            neg_term = f"d_w{out}{h}_neg"
            row_pos_terms.append(pos_term)
            row_neg_terms.append(neg_term)
            lines += [
                f".meas tran {pos_term} PARAM='vw{out}{h}p_after-vw{out}{h}p_before'",
                f".meas tran {neg_term} PARAM='vw{out}{h}n_after-vw{out}{h}n_before'",
            ]
        lines.append(f".meas tran row{out}_signed_delta PARAM='" + "+".join(row_terms) + "'")
        lines.append(f".meas tran row{out}_pos_delta PARAM='" + "+".join(row_pos_terms) + "'")
        lines.append(f".meas tran row{out}_neg_delta PARAM='" + "+".join(row_neg_terms) + "'")
        lines.append(f".meas tran row{out}_common_delta PARAM='row{out}_pos_delta+row{out}_neg_delta'")
        for suffix in ("p", "n"):
            node = f"vbo{out}{suffix}"
            lines += [
                f".meas tran {node}_before FIND V({node}) AT={before_ns:.12g}n",
                f".meas tran {node}_after FIND V({node}) AT={after_ns:.12g}n",
            ]
        lines += [
            f".meas tran bias{out}_signed_delta PARAM='(vbo{out}p_after-vbo{out}n_after)-(vbo{out}p_before-vbo{out}n_before)'",
            f".meas tran bias{out}_common_delta PARAM='(vbo{out}p_after-vbo{out}p_before)+(vbo{out}n_after-vbo{out}n_before)'",
        ]
    return "\n".join(lines)


def print_line(outputs: int) -> str:
    names = [
        name
        for out in range(outputs)
        for name in (
            f"row{out}_signed_delta",
            f"row{out}_pos_delta",
            f"row{out}_neg_delta",
            f"row{out}_common_delta",
            f"bias{out}_signed_delta",
            f"bias{out}_common_delta",
        )
    ]
    return "print " + " ".join(names)


def pre_trace_measurement_lines(outputs: int, hidden: int, mode: str, at_ns: float) -> tuple[str, list[str]]:
    if mode != "synapse_spike":
        return "", []
    lines: list[str] = []
    names: list[str] = []
    for out in range(outputs):
        for h in range(hidden):
            for node in (f"fprg{out}{h}", f"fprbar{out}{h}"):
                name = f"{node}_write"
                names.append(name)
                lines.append(f".meas tran {name} FIND V({node}) AT={at_ns:.12g}n")
    return "\n".join(lines), names


def print_line_with_extra(outputs: int, extra_names: list[str]) -> str:
    base = print_line(outputs)
    if not extra_names:
        return base
    return base + " " + " ".join(extra_names)


def build_readout_write_selectivity_netlist(
    *,
    label: int,
    hidden_values: list[float],
    outputs: int = 3,
    center_v: float = 0.64,
    weight_cap_f: float = 4.0,
    readout_update_width_u: float = 0.00002,
    output_bias_update_width_u: float = 0.0,
    readout_pos_update_width_u: float | None = None,
    readout_neg_update_width_u: float | None = None,
    readout_charge_update_width_u: float | None = None,
    readout_discharge_update_width_u: float | None = None,
    readout_center_pull_width_u: float = 0.0,
    output_bias_center_pull_width_u: float = 0.0,
    readout_center_pull_mode: str = "always",
    readout_write_state_gate_mode: str = "none",
    output_bias_write_pre_gate: str = "none",
    output_bias_flow_polarity: str = "follow_readout",
    readout_flow_polarity: str = "normal",
    readout_write_high_v: float = 1.0,
    readout_write_low_v: float = 0.16,
    readout_write_gate_device: str = "NSENSE",
    flow_pre_store: str = "shared_node",
    boosted_pre_offset_v: float = 0.75,
    spike_pre_threshold_v: float = 0.2,
    dynamic_pre_store: bool = False,
    flow_pre_cap_f: float = 2.0,
    flow_pre_consume_width_u: float = 0.05,
    flow_pre_boost_width_u: float = 4.0,
    flow_pre_spike_ref_v: float = 0.30,
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
    if flow_pre_store not in direct_flow.FLOW_PRE_STORES:
        raise ValueError(f"unknown flow pre-store mode: {flow_pre_store}")
    if dynamic_pre_store and flow_pre_store == "shared_node":
        raise ValueError("dynamic pre-store mode requires a per-synapse flow-pre-store.")
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(len(hidden_values))
        direct_flow.set_output_count(outputs)
        updates = direct_flow.readout_flow_updates(
            readout_update_width_u=readout_update_width_u,
            output_bias_update_width_u=output_bias_update_width_u,
            flow_pre_store=flow_pre_store,
            readout_flow_polarity=readout_flow_polarity,
            readout_flow_write_mode=readout_flow_write_mode,
            readout_pos_update_width_u=readout_pos_update_width_u,
            readout_neg_update_width_u=readout_neg_update_width_u,
            readout_charge_update_width_u=readout_charge_update_width_u,
            readout_discharge_update_width_u=readout_discharge_update_width_u,
            readout_center_pull_width_u=readout_center_pull_width_u,
            output_bias_center_pull_width_u=output_bias_center_pull_width_u,
            readout_center_pull_mode=readout_center_pull_mode,
            readout_write_state_gate_mode=readout_write_state_gate_mode,
            readout_write_gate_device=readout_write_gate_device,
            output_bias_write_pre_gate=output_bias_write_pre_gate,
            output_bias_flow_polarity=output_bias_flow_polarity,
            write_error_exclusion=write_error_exclusion,
            write_error_exclusion_width_u=write_error_exclusion_width_u,
        )
        caps = readout_capacitors(outputs, len(hidden_values), center_v, weight_cap_f)
        measures = measurement_lines(outputs, len(hidden_values), before_ns=bwd_start_ns - 0.1, after_ns=bwd_end_ns + 0.5)
        pre_measures, pre_measure_names = pre_trace_measurement_lines(
            outputs,
            len(hidden_values),
            flow_pre_store if dynamic_pre_store else "shared_node",
            at_ns=bwd_start_ns + 0.1,
        )
        if pre_measures:
            measures = measures + "\n" + pre_measures
        dynamic_pre_text = (
            direct_flow.flow_pre_activation_stores(
                flow_pre_store,
                flow_pre_cap_f,
                flow_pre_consume_width_u,
                flow_pre_boost_width_u,
                "spikeref",
            )
            if dynamic_pre_store
            else ""
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    act_sources = "\n".join(
        f"Vact{h} act{h} 0 {constant_until(stop_ns, value)}" for h, value in enumerate(hidden_values)
    )
    pre_sources: list[str] = []
    if dynamic_pre_store:
        pre_sources = []
    elif flow_pre_store in {"synapse_gate", "synapse_consume"}:
        for out in range(outputs):
            for h, value in enumerate(hidden_values):
                pre_sources.append(f"Vfpro{out}{h} fpro{out}{h} 0 {constant_until(stop_ns, value)}")
    elif flow_pre_store == "synapse_boost":
        for out in range(outputs):
            for h, value in enumerate(hidden_values):
                boosted = max(0.0, min(1.2, value + boosted_pre_offset_v))
                pre_sources.append(f"Vfprb{out}{h} fprb{out}{h} 0 {constant_until(stop_ns, boosted)}")
    elif flow_pre_store == "synapse_spike":
        for out in range(outputs):
            for h, value in enumerate(hidden_values):
                gate = 1.2 if value > spike_pre_threshold_v else 0.0
                pre_sources.append(f"Vfprg{out}{h} fprg{out}{h} 0 {constant_until(stop_ns, gate)}")
    pre_source_text = "\n".join(pre_sources)
    return f"""
* Isolated multiclass readout write-selectivity deck.
.param VDD=1.2
{mos_models()}
Vdd vdd 0 {{VDD}}
Vrstf rstf 0 {pulse(0.0, 0.5, 1.2, stop_ns) if dynamic_pre_store else "0"}
Vrste rste 0 {pulse(0.0, 0.5, 1.2, stop_ns) if dynamic_pre_store else "0"}
Vfwd fwd 0 {pulse(0.75, max(0.76, bwd_start_ns - 0.25), 1.2, stop_ns) if dynamic_pre_store else "0"}
Vspikeref spikeref 0 {flow_pre_spike_ref_v:.12g}
Vbwd bwd 0 {pulse(bwd_start_ns, bwd_end_ns, 1.2, stop_ns)}
Vapply apply 0 0
Vwhigh whigh 0 {readout_write_high_v:.12g}
Vwlow wlow 0 {readout_write_low_v:.12g}
Vwcenter wcenter 0 {center_v:.12g}
{act_sources}
{pre_source_text}
{error_sources(label, outputs=outputs, high_v=error_high_v, low_v=error_low_v, stop_ns=stop_ns)}

{caps}
{dynamic_pre_text}

{updates}

.options method=gear maxord=2 rshunt=1e12 gmin=1e-12
.tran 5p {stop_ns:.12g}n uic
{measures}
.control
run
{print_line_with_extra(outputs, pre_measure_names)}
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
        output_bias_update_width_u=args.output_bias_update_width_u,
        readout_pos_update_width_u=args.readout_pos_update_width_u,
        readout_neg_update_width_u=args.readout_neg_update_width_u,
        readout_charge_update_width_u=args.readout_charge_update_width_u,
        readout_discharge_update_width_u=args.readout_discharge_update_width_u,
        readout_center_pull_width_u=args.readout_center_pull_width_u,
        output_bias_center_pull_width_u=args.output_bias_center_pull_width_u,
        readout_center_pull_mode=args.readout_center_pull_mode,
        readout_write_state_gate_mode=args.readout_write_state_gate_mode,
        output_bias_write_pre_gate=args.output_bias_write_pre_gate,
        output_bias_flow_polarity=args.output_bias_flow_polarity,
        readout_flow_polarity=args.readout_flow_polarity,
        readout_write_high_v=args.readout_write_high_v,
        readout_write_low_v=args.readout_write_low_v,
        readout_write_gate_device=args.readout_write_gate_device,
        flow_pre_store=args.flow_pre_store,
        boosted_pre_offset_v=args.boosted_pre_offset_v,
        spike_pre_threshold_v=args.spike_pre_threshold_v,
        dynamic_pre_store=args.dynamic_pre_store,
        flow_pre_cap_f=args.flow_pre_cap_f,
        flow_pre_consume_width_u=args.flow_pre_consume_width_u,
        flow_pre_boost_width_u=args.flow_pre_boost_width_u,
        flow_pre_spike_ref_v=args.flow_pre_spike_ref_v,
        error_high_v=args.error_high_v,
        error_low_v=args.error_low_v,
        readout_flow_write_mode=args.readout_flow_write_mode,
        write_error_exclusion=args.write_error_exclusion,
        write_error_exclusion_width_u=args.write_error_exclusion_width_u,
        bwd_start_ns=args.bwd_start_ns,
        bwd_end_ns=args.bwd_end_ns,
        stop_ns=args.stop_ns,
    )
    parsed = direct_flow.run_netlist(spice_bin, path, netlist, args.timeout)
    row_deltas = {out: float(parsed[f"row{out}_signed_delta"]) for out in range(args.outputs)}
    row_pos = {out: float(parsed[f"row{out}_pos_delta"]) for out in range(args.outputs)}
    row_neg = {out: float(parsed[f"row{out}_neg_delta"]) for out in range(args.outputs)}
    row_common = {out: float(parsed[f"row{out}_common_delta"]) for out in range(args.outputs)}
    bias_deltas = {out: float(parsed[f"bias{out}_signed_delta"]) for out in range(args.outputs)}
    bias_common = {out: float(parsed[f"bias{out}_common_delta"]) for out in range(args.outputs)}
    pre_gate_values = [
        value
        for name, value in parsed.items()
        if name.startswith("fprg") and name.endswith("_write")
    ]
    pre_bar_values = [
        value
        for name, value in parsed.items()
        if name.startswith("fprbar") and name.endswith("_write")
    ]
    return {
        "label": label,
        "expected_sign_ok": expected_sign_ok(row_deltas, label),
        "bias_expected_sign_ok": args.output_bias_update_width_u == 0.0 or expected_sign_ok(bias_deltas, label),
        "pre_gate_on_fraction": (
            float(sum(value > 0.6 for value in pre_gate_values) / len(pre_gate_values)) if pre_gate_values else None
        ),
        "pre_gate_min_v": min(pre_gate_values) if pre_gate_values else None,
        "pre_gate_max_v": max(pre_gate_values) if pre_gate_values else None,
        "pre_bar_min_v": min(pre_bar_values) if pre_bar_values else None,
        "pre_bar_max_v": max(pre_bar_values) if pre_bar_values else None,
        **{f"row{out}_signed_delta": delta for out, delta in row_deltas.items()},
        **{f"row{out}_pos_delta": delta for out, delta in row_pos.items()},
        **{f"row{out}_neg_delta": delta for out, delta in row_neg.items()},
        **{f"row{out}_common_delta": delta for out, delta in row_common.items()},
        **{f"bias{out}_signed_delta": delta for out, delta in bias_deltas.items()},
        **{f"bias{out}_common_delta": delta for out, delta in bias_common.items()},
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
    ap.add_argument("--output-bias-update-width-u", type=float, default=0.0)
    ap.add_argument("--readout-pos-update-width-u", type=float, default=None)
    ap.add_argument("--readout-neg-update-width-u", type=float, default=None)
    ap.add_argument("--readout-charge-update-width-u", type=float, default=None)
    ap.add_argument("--readout-discharge-update-width-u", type=float, default=None)
    ap.add_argument("--readout-center-pull-width-u", type=float, default=0.0)
    ap.add_argument("--output-bias-center-pull-width-u", type=float, default=0.0)
    ap.add_argument("--readout-center-pull-mode", choices=direct_flow.READOUT_CENTER_PULL_MODES, default="always")
    ap.add_argument("--readout-write-state-gate-mode", choices=direct_flow.READOUT_WRITE_STATE_GATE_MODES, default="none")
    ap.add_argument("--output-bias-write-pre-gate", choices=direct_flow.OUTPUT_BIAS_WRITE_PRE_GATES, default="none")
    ap.add_argument("--output-bias-flow-polarity", choices=direct_flow.OUTPUT_BIAS_FLOW_POLARITIES, default="follow_readout")
    ap.add_argument("--readout-flow-polarity", choices=direct_flow.READOUT_FLOW_POLARITIES, default="normal")
    ap.add_argument("--readout-write-high-v", type=float, default=1.0)
    ap.add_argument("--readout-write-low-v", type=float, default=0.16)
    ap.add_argument("--readout-write-gate-device", choices=direct_flow.WRITE_GATE_DEVICES, default="NSENSE")
    ap.add_argument("--flow-pre-store", choices=direct_flow.FLOW_PRE_STORES, default="shared_node")
    ap.add_argument("--boosted-pre-offset-v", type=float, default=0.75)
    ap.add_argument("--spike-pre-threshold-v", type=float, default=0.2)
    ap.add_argument(
        "--dynamic-pre-store",
        action="store_true",
        help="Use the production MOS forward-store trace cell instead of directly driving fpro/fprb/fprg.",
    )
    ap.add_argument("--flow-pre-cap-f", type=float, default=2.0)
    ap.add_argument("--flow-pre-consume-width-u", type=float, default=0.05)
    ap.add_argument("--flow-pre-boost-width-u", type=float, default=4.0)
    ap.add_argument("--flow-pre-spike-ref-v", type=float, default=0.30)
    ap.add_argument("--error-high-v", type=float, default=1.1)
    ap.add_argument("--error-low-v", type=float, default=0.0)
    ap.add_argument("--readout-flow-write-mode", choices=direct_flow.READOUT_FLOW_WRITE_MODES, default="bounded_charge_discharge")
    ap.add_argument("--write-error-exclusion", choices=direct_flow.WRITE_ERROR_EXCLUSION_MODES, default="none")
    ap.add_argument("--write-error-exclusion-width-u", type=float, default=8.0)
    ap.add_argument("--bwd-start-ns", type=float, default=1.0)
    ap.add_argument("--bwd-end-ns", type=float, default=5.0)
    ap.add_argument("--stop-ns", type=float, default=8.0)
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
        "readout_write_gate_device": args.readout_write_gate_device,
        "flow_pre_store": args.flow_pre_store,
        "boosted_pre_offset_v": args.boosted_pre_offset_v,
        "spike_pre_threshold_v": args.spike_pre_threshold_v,
        "dynamic_pre_store": args.dynamic_pre_store,
        "flow_pre_cap_f": args.flow_pre_cap_f if args.dynamic_pre_store else None,
        "flow_pre_consume_width_u": args.flow_pre_consume_width_u if args.dynamic_pre_store else None,
        "flow_pre_boost_width_u": args.flow_pre_boost_width_u if args.dynamic_pre_store else None,
        "flow_pre_spike_ref_v": args.flow_pre_spike_ref_v if args.dynamic_pre_store else None,
        "center_v": args.center_v,
        "output_bias_update_width_u": args.output_bias_update_width_u,
        "readout_center_pull_width_u": args.readout_center_pull_width_u,
        "output_bias_center_pull_width_u": args.output_bias_center_pull_width_u,
        "readout_center_pull_mode": args.readout_center_pull_mode,
        "readout_write_state_gate_mode": args.readout_write_state_gate_mode,
        "output_bias_write_pre_gate": args.output_bias_write_pre_gate,
        "output_bias_flow_polarity": args.output_bias_flow_polarity,
        "bwd_start_ns": args.bwd_start_ns,
        "bwd_end_ns": args.bwd_end_ns,
        "stop_ns": args.stop_ns,
        "all_labels_expected_sign": bool(df["expected_sign_ok"].all()),
        "all_labels_bias_expected_sign": bool(df["bias_expected_sign_ok"].all()),
        "wall_time_s": time.perf_counter() - t0,
    }
    summary_path = results / f"{safe_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
