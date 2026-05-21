#!/usr/bin/env python3
"""Exercise the composed multiclass readout/error/write loop in SPICE.

This harness sits between the fully isolated readout-write selectivity test and
the full network runner.  Hidden activations are externally held, but the deck
uses the production output-forward fragment, production error cells, and
production direct-flow readout writes.  It answers whether the composed
output/error/write loop moves the target row and optional output bias in the
right direction before hidden-layer effects enter the experiment.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

import parameter_theory as theory
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


def parse_floats(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one numeric value.")
    return values


def parse_matrix(text: str) -> list[list[float]]:
    rows = [parse_floats(row) for row in text.split(";") if row.strip()]
    if not rows:
        raise ValueError("expected a semicolon-separated matrix.")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("all matrix rows must have the same width.")
    return rows


def cap_voltage(center_v: float, signed_delta_v: float, branch: str) -> float:
    if branch == "p":
        value = center_v + max(0.0, signed_delta_v)
    elif branch == "n":
        value = center_v + max(0.0, -signed_delta_v)
    else:
        raise ValueError("branch must be 'p' or 'n'.")
    return min(1.18, max(0.02, value))


def readout_capacitors(
    weights: list[list[float]],
    biases: list[float],
    *,
    center_v: float,
    cap_f: float,
) -> str:
    lines: list[str] = []
    for out, row in enumerate(weights):
        for h, signed_delta in enumerate(row):
            for suffix in ("p", "n"):
                node = f"vw{out}{h}{suffix}"
                lines += [
                    f"C{node} {node} 0 {cap_f:.12g}f IC={cap_voltage(center_v, signed_delta, suffix):.12g}",
                    f"R{node} {node} 0 1e15",
                ]
        for suffix in ("p", "n"):
            node = f"vbo{out}{suffix}"
            lines += [
                f"C{node} {node} 0 {cap_f:.12g}f IC={cap_voltage(center_v, biases[out], suffix):.12g}",
                f"R{node} {node} 0 1e15",
            ]
    return "\n".join(lines)


def target_sources(label: int, outputs: int, target_high_v: float, target_low_v: float, stop_ns: float) -> str:
    lines: list[str] = []
    for out in range(outputs):
        target = target_high_v if out == label else target_low_v
        complement = target_low_v if out == label else target_high_v
        lines += [
            f"Vt{out} t{out} 0 {constant_until(stop_ns, target)}",
            f"Vnt{out} nt{out} 0 {constant_until(stop_ns, complement)}",
        ]
    return "\n".join(lines)


def measurement_lines(
    outputs: int,
    hidden: int,
    initial_ns: float,
    before_ns: float,
    fwd_ns: float,
    err_ns: float,
    after_ns: float,
) -> str:
    lines: list[str] = []
    for out in range(outputs):
        row_terms: list[str] = []
        row_pos_terms: list[str] = []
        row_neg_terms: list[str] = []
        for h in range(hidden):
            for suffix in ("p", "n"):
                node = f"vw{out}{h}{suffix}"
                lines += [
                    f".meas tran {node}_initial FIND V({node}) AT={initial_ns:.12g}n",
                    f".meas tran {node}_before FIND V({node}) AT={before_ns:.12g}n",
                    f".meas tran {node}_after FIND V({node}) AT={after_ns:.12g}n",
                ]
            term = f"d_w{out}{h}_signed"
            pos_term = f"d_w{out}{h}_pos"
            neg_term = f"d_w{out}{h}_neg"
            fwd_term = f"d_w{out}{h}_fwd_signed"
            fwd_pos_term = f"d_w{out}{h}_fwd_pos"
            fwd_neg_term = f"d_w{out}{h}_fwd_neg"
            row_terms.append(term)
            row_pos_terms.append(pos_term)
            row_neg_terms.append(neg_term)
            lines.append(
                f".meas tran {term} PARAM='(vw{out}{h}p_after-vw{out}{h}n_after)-(vw{out}{h}p_before-vw{out}{h}n_before)'"
            )
            lines += [
                f".meas tran {pos_term} PARAM='vw{out}{h}p_after-vw{out}{h}p_before'",
                f".meas tran {neg_term} PARAM='vw{out}{h}n_after-vw{out}{h}n_before'",
                f".meas tran {fwd_term} PARAM='(vw{out}{h}p_before-vw{out}{h}n_before)-(vw{out}{h}p_initial-vw{out}{h}n_initial)'",
                f".meas tran {fwd_pos_term} PARAM='vw{out}{h}p_before-vw{out}{h}p_initial'",
                f".meas tran {fwd_neg_term} PARAM='vw{out}{h}n_before-vw{out}{h}n_initial'",
            ]
        lines.append(f".meas tran row{out}_signed_delta PARAM='" + "+".join(row_terms) + "'")
        lines.append(f".meas tran row{out}_pos_delta PARAM='" + "+".join(row_pos_terms) + "'")
        lines.append(f".meas tran row{out}_neg_delta PARAM='" + "+".join(row_neg_terms) + "'")
        lines.append(f".meas tran row{out}_common_delta PARAM='row{out}_pos_delta+row{out}_neg_delta'")
        lines.append(
            f".meas tran row{out}_fwd_signed_delta PARAM='"
            + "+".join(f"d_w{out}{h}_fwd_signed" for h in range(hidden))
            + "'"
        )
        lines.append(
            f".meas tran row{out}_fwd_pos_delta PARAM='"
            + "+".join(f"d_w{out}{h}_fwd_pos" for h in range(hidden))
            + "'"
        )
        lines.append(
            f".meas tran row{out}_fwd_neg_delta PARAM='"
            + "+".join(f"d_w{out}{h}_fwd_neg" for h in range(hidden))
            + "'"
        )
        lines.append(f".meas tran row{out}_fwd_common_delta PARAM='row{out}_fwd_pos_delta+row{out}_fwd_neg_delta'")
        for suffix in ("p", "n"):
            node = f"vbo{out}{suffix}"
            lines += [
                f".meas tran {node}_before FIND V({node}) AT={before_ns:.12g}n",
                f".meas tran {node}_after FIND V({node}) AT={after_ns:.12g}n",
            ]
        lines += [
            f".meas tran bias{out}_signed_delta PARAM='(vbo{out}p_after-vbo{out}n_after)-(vbo{out}p_before-vbo{out}n_before)'",
            f".meas tran bias{out}_common_delta PARAM='(vbo{out}p_after-vbo{out}p_before)+(vbo{out}n_after-vbo{out}n_before)'",
            f".meas tran scorep{out}_fwd FIND V(scorep{out}) AT={fwd_ns:.12g}n",
            f".meas tran scoren{out}_fwd FIND V(scoren{out}) AT={fwd_ns:.12g}n",
            f".meas tran score{out}_fwd PARAM='scorep{out}_fwd-scoren{out}_fwd'",
            f".meas tran out{out}_fwd FIND V(out{out}) AT={fwd_ns:.12g}n",
            f".meas tran dp{out}_err FIND V(dp{out}) AT={err_ns:.12g}n",
            f".meas tran dn{out}_err FIND V(dn{out}) AT={err_ns:.12g}n",
            f".meas tran errdiff{out}_err PARAM='dp{out}_err-dn{out}_err'",
        ]
    return "\n".join(lines)


def print_line(outputs: int) -> str:
    names: list[str] = []
    for out in range(outputs):
        names += [
            f"score{out}_fwd",
            f"out{out}_fwd",
            f"dp{out}_err",
            f"dn{out}_err",
            f"row{out}_signed_delta",
            f"row{out}_common_delta",
            f"bias{out}_signed_delta",
            f"bias{out}_common_delta",
        ]
    return "print " + " ".join(names)


def build_readout_error_loop_netlist(
    *,
    label: int,
    hidden_values: list[float],
    weights: list[list[float]],
    biases: list[float],
    center_v: float = 0.64,
    weight_cap_f: float = 4.0,
    synapse_design: str = "split_signed_v1",
    output_head: str = "split_score_caps",
    error_rule: str = "onehot",
    flow_pre_store: str = "shared_node",
    flow_pre_cap_f: float = 2.0,
    flow_pre_consume_width_u: float = 4.0,
    flow_pre_boost_width_u: float = 4.0,
    flow_pre_spike_ref_v: float = 0.3,
    readout_update_width_u: float = 0.00002,
    readout_pos_update_width_u: float | None = None,
    readout_neg_update_width_u: float | None = None,
    output_bias_update_width_u: float | None = None,
    readout_charge_update_width_u: float | None = None,
    readout_discharge_update_width_u: float | None = None,
    readout_dp_gate_update_width_u: float | None = None,
    readout_dn_gate_update_width_u: float | None = None,
    readout_dp_discharge_gate_update_width_u: float | None = None,
    readout_dp_charge_gate_update_width_u: float | None = None,
    readout_dn_discharge_gate_update_width_u: float | None = None,
    readout_dn_charge_gate_update_width_u: float | None = None,
    readout_flow_polarity: str = "normal",
    readout_flow_write_mode: str = "bounded_charge_discharge",
    readout_write_high_v: float = 1.0,
    readout_write_low_v: float = 0.16,
    readout_pos_write_high_v: float | None = None,
    readout_pos_write_low_v: float | None = None,
    readout_neg_write_high_v: float | None = None,
    readout_neg_write_low_v: float | None = None,
    readout_write_error_exclusion: str = "none",
    readout_write_error_exclusion_width_u: float = 8.0,
    readout_center_pull_width_u: float = 0.0,
    output_bias_center_pull_width_u: float = 0.0,
    readout_center_pull_v: float = 0.64,
    readout_pos_center_pull_v: float | None = None,
    readout_neg_center_pull_v: float | None = None,
    output_bias_pos_center_pull_v: float | None = None,
    output_bias_neg_center_pull_v: float | None = None,
    readout_center_pull_gate: str = "bwd",
    readout_center_pull_mode: str = "always",
    residual_target_width_u: float = 96.0,
    residual_output_width_u: float = 64.0,
    target_high_v: float = 1.1,
    target_low_v: float = 0.0,
    score_reset_v: float = 0.0,
    fwd_start_ns: float = 0.5,
    fwd_end_ns: float = 3.0,
    bwd_start_ns: float = 4.0,
    bwd_end_ns: float = 8.0,
    stop_ns: float = 10.0,
) -> str:
    outputs = len(weights)
    hidden = len(hidden_values)
    if not 0 <= label < outputs:
        raise ValueError("label must be a valid output index.")
    if any(len(row) != hidden for row in weights):
        raise ValueError("each readout weight row must match hidden_values length.")
    if len(biases) != outputs:
        raise ValueError("bias count must match output count.")
    pos_high_v = readout_write_high_v if readout_pos_write_high_v is None else readout_pos_write_high_v
    pos_low_v = readout_write_low_v if readout_pos_write_low_v is None else readout_pos_write_low_v
    neg_high_v = readout_write_high_v if readout_neg_write_high_v is None else readout_neg_write_high_v
    neg_low_v = readout_write_low_v if readout_neg_write_low_v is None else readout_neg_write_low_v
    pos_center_v = readout_center_pull_v if readout_pos_center_pull_v is None else readout_pos_center_pull_v
    neg_center_v = readout_center_pull_v if readout_neg_center_pull_v is None else readout_neg_center_pull_v
    bias_pos_center_v = (
        readout_center_pull_v if output_bias_pos_center_pull_v is None else output_bias_pos_center_pull_v
    )
    bias_neg_center_v = (
        readout_center_pull_v if output_bias_neg_center_pull_v is None else output_bias_neg_center_pull_v
    )
    if not readout_write_low_v < center_v < readout_write_high_v:
        raise ValueError("bounded write rails must straddle the initial center voltage.")
    if not pos_low_v < center_v < pos_high_v:
        raise ValueError("positive bounded write rails must straddle the initial center voltage.")
    if not neg_low_v < center_v < neg_high_v:
        raise ValueError("negative bounded write rails must straddle the initial center voltage.")
    if (
        error_rule
        in {
            "ce_split_score",
            "ce_split_diffgate",
            "ce_split_dpair",
            "ce_split_compete",
            "ce_split_current",
            "ce_split_hybrid",
            "ce_split_limited",
            "ce_mirror_limited",
            "ce_mirror_winner_limited",
            "ce_mirror_compete_limited",
        }
        and output_head not in direct_flow.SPLIT_SCORE_OUTPUT_HEADS
    ):
        raise ValueError(f"{error_rule} requires a split-score output_head.")
    if error_rule in {
        "ce_mirror_limited",
        "ce_mirror_winner_limited",
        "ce_mirror_compete_limited",
    } and output_head not in direct_flow.DIODE_MIRROR_OUTPUT_HEADS:
        raise ValueError(f"{error_rule} requires a diode-mirror split-score output_head.")
    output_bias_width = readout_update_width_u if output_bias_update_width_u is None else output_bias_update_width_u
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(hidden)
        direct_flow.set_output_count(outputs)
        design = direct_flow.scaled_synapse_design(
            synapse_design,
            hidden_delta_width_scale=1.0,
            hidden_gradient_width_scale=1.0,
            readout_gradient_width_scale=1.0,
            output_forward_width_scale=1.0,
            output_forward_pos_width_scale=1.0,
            output_forward_neg_width_scale=1.0,
            output_bias_forward_width_scale=1.0,
            output_relu_width_scale=1.0,
        )
        caps = direct_flow.temporary_caps(
            gradient_cap_f=4.0,
            hidden_gradient_cap_f=4.0,
            hidden_delta_cap_f=4.0,
            lead_cap_f=1.0,
            include_gradient_caps=False,
            score_reset_v=score_reset_v,
            output_head=output_head,
        )
        forward = direct_flow.output_forward(design, output_head)
        errors = direct_flow.error_cells(
            error_rule,
            latch_boost_width_u=0.0,
            residual_target_width_u=residual_target_width_u,
            residual_output_width_u=residual_output_width_u,
            lead_mode="score_direct",
        )
        updates = direct_flow.readout_flow_updates(
            readout_update_width_u=readout_update_width_u,
            output_bias_update_width_u=output_bias_width,
            flow_pre_store=flow_pre_store,
            readout_flow_polarity=readout_flow_polarity,
            readout_flow_write_mode=readout_flow_write_mode,
            readout_pos_update_width_u=readout_pos_update_width_u,
            readout_neg_update_width_u=readout_neg_update_width_u,
            readout_charge_update_width_u=readout_charge_update_width_u,
            readout_discharge_update_width_u=readout_discharge_update_width_u,
            readout_dp_gate_update_width_u=readout_dp_gate_update_width_u,
            readout_dn_gate_update_width_u=readout_dn_gate_update_width_u,
            readout_dp_discharge_gate_update_width_u=readout_dp_discharge_gate_update_width_u,
            readout_dp_charge_gate_update_width_u=readout_dp_charge_gate_update_width_u,
            readout_dn_discharge_gate_update_width_u=readout_dn_discharge_gate_update_width_u,
            readout_dn_charge_gate_update_width_u=readout_dn_charge_gate_update_width_u,
            readout_center_pull_width_u=readout_center_pull_width_u,
            output_bias_center_pull_width_u=output_bias_center_pull_width_u,
            readout_center_pull_gate=readout_center_pull_gate,
            readout_center_pull_mode=readout_center_pull_mode,
            readout_pos_write_high_node="wphigh",
            readout_pos_write_low_node="wplow",
            readout_neg_write_high_node="wnhigh",
            readout_neg_write_low_node="wnlow",
            readout_pos_center_pull_node="wcenterp",
            readout_neg_center_pull_node="wcentern",
            output_bias_pos_center_pull_node="wbocenterp",
            output_bias_neg_center_pull_node="wbocentern",
            write_error_exclusion=readout_write_error_exclusion,
            write_error_exclusion_width_u=readout_write_error_exclusion_width_u,
        )
        flow_stores = direct_flow.flow_pre_activation_stores(
            flow_pre_store,
            cap_f=flow_pre_cap_f,
            consume_width_u=flow_pre_consume_width_u,
            boost_width_u=flow_pre_boost_width_u,
            spike_ref_node="spikeref",
        )
        readout_caps = readout_capacitors(weights, biases, center_v=center_v, cap_f=weight_cap_f)
        measures = measurement_lines(
            outputs,
            hidden,
            initial_ns=max(0.0, fwd_start_ns - 0.1),
            before_ns=bwd_start_ns - 0.1,
            fwd_ns=fwd_end_ns - 0.1,
            err_ns=(bwd_start_ns + bwd_end_ns) / 2,
            after_ns=bwd_end_ns + 0.5,
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    act_sources = "\n".join(
        f"Vact{h} act{h} 0 {constant_until(stop_ns, value)}" for h, value in enumerate(hidden_values)
    )
    return f"""
* Composed multiclass readout/error/write loop deck.
.param VDD=1.2
.option method=gear maxord=2 rshunt=1e12 gmin=1e-12
{mos_models()}
Vdd vdd 0 1.2
Vbias bias 0 1.1
Vspikeref spikeref 0 {flow_pre_spike_ref_v:.12g}
Vscorecm scorecm 0 {score_reset_v:.12g}
Vwhigh whigh 0 {readout_write_high_v:.12g}
Vwlow wlow 0 {readout_write_low_v:.12g}
Vwphigh wphigh 0 {pos_high_v:.12g}
Vwplow wplow 0 {pos_low_v:.12g}
Vwnhigh wnhigh 0 {neg_high_v:.12g}
Vwnlow wnlow 0 {neg_low_v:.12g}
Vwcenter wcenter 0 {readout_center_pull_v:.12g}
Vwcenterp wcenterp 0 {pos_center_v:.12g}
Vwcentern wcentern 0 {neg_center_v:.12g}
Vwbocenterp wbocenterp 0 {bias_pos_center_v:.12g}
Vwbocentern wbocentern 0 {bias_neg_center_v:.12g}
Vfwd fwd 0 {pulse(fwd_start_ns, fwd_end_ns, 1.1, stop_ns)}
Voutg outg 0 {pulse(max(fwd_start_ns, fwd_end_ns - 0.30), fwd_end_ns, 1.1, stop_ns)}
Vbwd bwd 0 {pulse(bwd_start_ns, bwd_end_ns, 1.1, stop_ns)}
Verr err 0 {pulse(bwd_start_ns, bwd_end_ns, 1.1, stop_ns)}
    Vrstf rstf 0 {pulse(0.0, min(0.45, fwd_start_ns - 0.05), 1.1, stop_ns)}
    Vrste rste 0 {pulse(0.0, min(0.45, fwd_start_ns - 0.05), 1.1, stop_ns)}
Vrstg rstg 0 0
{target_sources(label, outputs, target_high_v, target_low_v, stop_ns)}
{act_sources}
{caps}
{readout_caps}
{flow_stores}
{forward}
{errors}
{updates}
{measures}
.tran 10p {stop_ns:.12g}n uic
.control
run
{print_line(outputs)}
.endc
.end
""".lstrip()


def run_case(args: argparse.Namespace, label: int, weights: list[list[float]], biases: list[float]) -> dict[str, Any]:
    hidden_values = parse_floats(args.hidden_values)
    netlist = build_readout_error_loop_netlist(
        label=label,
        hidden_values=hidden_values,
        weights=weights,
        biases=biases,
        center_v=args.center_v,
        weight_cap_f=args.weight_cap_f,
        synapse_design=args.synapse_design,
        output_head=args.output_head,
        error_rule=args.error_rule,
        flow_pre_store=args.flow_pre_store,
        flow_pre_cap_f=args.flow_pre_cap_f,
        flow_pre_consume_width_u=args.flow_pre_consume_width_u,
        flow_pre_boost_width_u=args.flow_pre_boost_width_u,
        flow_pre_spike_ref_v=args.flow_pre_spike_ref_v,
        readout_update_width_u=args.readout_update_width_u,
        output_bias_update_width_u=args.output_bias_update_width_u,
        readout_charge_update_width_u=args.readout_charge_update_width_u,
        readout_discharge_update_width_u=args.readout_discharge_update_width_u,
        readout_flow_polarity=args.readout_flow_polarity,
        readout_flow_write_mode=args.readout_flow_write_mode,
        readout_pos_update_width_u=args.readout_pos_update_width_u,
        readout_neg_update_width_u=args.readout_neg_update_width_u,
        readout_dp_gate_update_width_u=args.readout_dp_gate_update_width_u,
        readout_dn_gate_update_width_u=args.readout_dn_gate_update_width_u,
        readout_dp_discharge_gate_update_width_u=args.readout_dp_discharge_gate_update_width_u,
        readout_dp_charge_gate_update_width_u=args.readout_dp_charge_gate_update_width_u,
        readout_dn_discharge_gate_update_width_u=args.readout_dn_discharge_gate_update_width_u,
        readout_dn_charge_gate_update_width_u=args.readout_dn_charge_gate_update_width_u,
        readout_write_high_v=args.readout_write_high_v,
        readout_write_low_v=args.readout_write_low_v,
        readout_pos_write_high_v=args.readout_pos_write_high_v,
        readout_pos_write_low_v=args.readout_pos_write_low_v,
        readout_neg_write_high_v=args.readout_neg_write_high_v,
        readout_neg_write_low_v=args.readout_neg_write_low_v,
        readout_write_error_exclusion=args.readout_write_error_exclusion,
        readout_write_error_exclusion_width_u=args.readout_write_error_exclusion_width_u,
        readout_center_pull_width_u=args.readout_center_pull_width_u,
        output_bias_center_pull_width_u=args.output_bias_center_pull_width_u,
        readout_center_pull_v=args.readout_center_pull_v,
        readout_pos_center_pull_v=args.readout_pos_center_pull_v,
        readout_neg_center_pull_v=args.readout_neg_center_pull_v,
        output_bias_pos_center_pull_v=args.output_bias_pos_center_pull_v,
        output_bias_neg_center_pull_v=args.output_bias_neg_center_pull_v,
        readout_center_pull_gate=args.readout_center_pull_gate,
        readout_center_pull_mode=args.readout_center_pull_mode,
        residual_target_width_u=args.residual_target_width_u,
        residual_output_width_u=args.residual_output_width_u,
        target_high_v=args.target_high_v,
        target_low_v=args.target_low_v,
        score_reset_v=args.score_reset_v,
    )
    generated = ROOT / "spice/generated"
    generated.mkdir(parents=True, exist_ok=True)
    spice_bin, version = detect_spice(args.simulator)
    run_tiny_test(spice_bin, generated)
    parsed = direct_flow.run_netlist(spice_bin, generated / f"{args.tag}_label{label}.cir", netlist, args.timeout)
    row_deltas = {out: float(parsed[f"row{out}_signed_delta"]) for out in range(len(weights))}
    row_common = {out: float(parsed[f"row{out}_common_delta"]) for out in range(len(weights))}
    row_fwd_deltas = {out: float(parsed[f"row{out}_fwd_signed_delta"]) for out in range(len(weights))}
    row_fwd_common = {out: float(parsed[f"row{out}_fwd_common_delta"]) for out in range(len(weights))}
    bias_deltas = {out: float(parsed[f"bias{out}_signed_delta"]) for out in range(len(weights))}
    bias_common = {out: float(parsed[f"bias{out}_common_delta"]) for out in range(len(weights))}
    scores = {out: float(parsed[f"score{out}_fwd"]) for out in range(len(weights))}
    outs = {out: float(parsed[f"out{out}_fwd"]) for out in range(len(weights))}
    dp = {out: float(parsed[f"dp{out}_err"]) for out in range(len(weights))}
    dn = {out: float(parsed[f"dn{out}_err"]) for out in range(len(weights))}
    return {
        "label": label,
        "simulator": version,
        "pred_score": max(scores, key=scores.get),
        "pred_out": max(outs, key=outs.get),
        "target_row_delta": row_deltas[label],
        "target_row_common_delta": row_common[label],
        "target_row_fwd_delta": row_fwd_deltas[label],
        "target_row_fwd_common_delta": row_fwd_common[label],
        "target_bias_delta": bias_deltas[label],
        "target_bias_common_delta": bias_common[label],
        "nontarget_row_delta_mean": sum(delta for out, delta in row_deltas.items() if out != label)
        / max(1, len(row_deltas) - 1),
        "nontarget_row_common_delta_mean": sum(delta for out, delta in row_common.items() if out != label)
        / max(1, len(row_common) - 1),
        "nontarget_row_fwd_delta_mean": sum(delta for out, delta in row_fwd_deltas.items() if out != label)
        / max(1, len(row_fwd_deltas) - 1),
        "nontarget_row_fwd_common_delta_mean": sum(
            delta for out, delta in row_fwd_common.items() if out != label
        )
        / max(1, len(row_fwd_common) - 1),
        "nontarget_bias_delta_mean": sum(delta for out, delta in bias_deltas.items() if out != label)
        / max(1, len(bias_deltas) - 1),
        "nontarget_bias_common_delta_mean": sum(delta for out, delta in bias_common.items() if out != label)
        / max(1, len(bias_common) - 1),
        **{f"row{out}_signed_delta": delta for out, delta in row_deltas.items()},
        **{f"row{out}_common_delta": delta for out, delta in row_common.items()},
        **{f"row{out}_fwd_signed_delta": delta for out, delta in row_fwd_deltas.items()},
        **{f"row{out}_fwd_common_delta": delta for out, delta in row_fwd_common.items()},
        **{f"bias{out}_signed_delta": delta for out, delta in bias_deltas.items()},
        **{f"bias{out}_common_delta": delta for out, delta in bias_common.items()},
        **{f"score{out}_fwd": value for out, value in scores.items()},
        **{f"out{out}_fwd": value for out, value in outs.items()},
        **{f"dp{out}_err": value for out, value in dp.items()},
        **{f"dn{out}_err": value for out, value in dn.items()},
        **{f"errdiff{out}_err": float(parsed[f"errdiff{out}_err"]) for out in range(len(weights))},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="device_readout_error_loop")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--simulator", default=None)
    ap.add_argument("--labels", default="0,1,2")
    ap.add_argument("--hidden-values", default="0.8,0.45")
    ap.add_argument("--weights", default="0.08,-0.04;-0.04,0.08;0.02,0.02")
    ap.add_argument("--biases", default="0.0,0.0,0.0")
    ap.add_argument("--center-v", type=float, default=0.64)
    ap.add_argument("--weight-cap-f", type=float, default=4.0)
    ap.add_argument("--synapse-design", choices=direct_flow.SYNAPSE_DESIGNS.keys(), default="split_signed_v1")
    ap.add_argument("--output-head", choices=direct_flow.OUTPUT_HEAD_MODES, default="split_score_caps")
    ap.add_argument(
        "--error-rule",
        choices=[
            "score",
            "onehot",
            "onehot_limited",
            "onehot_out",
            "ce_out",
            "ce_split_score",
            "ce_split_diffgate",
            "ce_split_dpair",
            "ce_split_compete",
            "ce_split_current",
            "ce_split_hybrid",
            "ce_split_limited",
            "ce_mirror_limited",
            "ce_mirror_winner_limited",
            "ce_mirror_compete_limited",
        ],
        default="onehot",
    )
    ap.add_argument("--flow-pre-store", choices=direct_flow.FLOW_PRE_STORES, default="shared_node")
    ap.add_argument("--flow-pre-cap-f", type=float, default=2.0)
    ap.add_argument("--flow-pre-consume-width-u", type=float, default=4.0)
    ap.add_argument("--flow-pre-boost-width-u", type=float, default=4.0)
    ap.add_argument("--flow-pre-spike-ref-v", type=float, default=0.3)
    ap.add_argument("--readout-update-width-u", type=float, default=0.00002)
    ap.add_argument("--readout-pos-update-width-u", type=float, default=None)
    ap.add_argument("--readout-neg-update-width-u", type=float, default=None)
    ap.add_argument("--readout-dp-gate-update-width-u", type=float, default=None)
    ap.add_argument("--readout-dn-gate-update-width-u", type=float, default=None)
    ap.add_argument("--readout-dp-discharge-gate-update-width-u", type=float, default=None)
    ap.add_argument("--readout-dp-charge-gate-update-width-u", type=float, default=None)
    ap.add_argument("--readout-dn-discharge-gate-update-width-u", type=float, default=None)
    ap.add_argument("--readout-dn-charge-gate-update-width-u", type=float, default=None)
    ap.add_argument("--output-bias-update-width-u", type=float, default=None)
    ap.add_argument("--readout-charge-update-width-u", type=float, default=None)
    ap.add_argument("--readout-discharge-update-width-u", type=float, default=None)
    ap.add_argument("--readout-flow-polarity", choices=direct_flow.READOUT_FLOW_POLARITIES, default="normal")
    ap.add_argument(
        "--readout-flow-write-mode",
        choices=direct_flow.READOUT_FLOW_WRITE_MODES,
        default="bounded_charge_discharge",
    )
    ap.add_argument("--readout-write-high-v", type=float, default=1.0)
    ap.add_argument("--readout-write-low-v", type=float, default=0.16)
    ap.add_argument("--readout-pos-write-high-v", type=float, default=None)
    ap.add_argument("--readout-pos-write-low-v", type=float, default=None)
    ap.add_argument("--readout-neg-write-high-v", type=float, default=None)
    ap.add_argument("--readout-neg-write-low-v", type=float, default=None)
    ap.add_argument("--readout-write-error-exclusion", choices=direct_flow.WRITE_ERROR_EXCLUSION_MODES, default="none")
    ap.add_argument("--readout-write-error-exclusion-width-u", type=float, default=8.0)
    ap.add_argument("--readout-center-pull-width-u", type=float, default=0.0)
    ap.add_argument("--output-bias-center-pull-width-u", type=float, default=0.0)
    ap.add_argument("--readout-center-pull-v", type=float, default=0.64)
    ap.add_argument("--readout-pos-center-pull-v", type=float, default=None)
    ap.add_argument("--readout-neg-center-pull-v", type=float, default=None)
    ap.add_argument("--output-bias-pos-center-pull-v", type=float, default=None)
    ap.add_argument("--output-bias-neg-center-pull-v", type=float, default=None)
    ap.add_argument("--readout-center-pull-gate", choices=direct_flow.READOUT_CENTER_PULL_GATES, default="bwd")
    ap.add_argument("--readout-center-pull-mode", choices=direct_flow.READOUT_CENTER_PULL_MODES, default="always")
    ap.add_argument("--residual-target-width-u", type=float, default=96.0)
    ap.add_argument("--residual-output-width-u", type=float, default=64.0)
    ap.add_argument("--target-high-v", type=float, default=1.1)
    ap.add_argument("--target-low-v", type=float, default=0.0)
    ap.add_argument("--score-reset-v", type=float, default=0.0)
    args = ap.parse_args()

    labels = [int(value) for value in parse_floats(args.labels)]
    weights = parse_matrix(args.weights)
    biases = parse_floats(args.biases)
    outputs = len(weights)
    if len(biases) != outputs:
        raise SystemExit("--biases length must match --weights row count.")
    if any(not 0 <= label < outputs for label in labels):
        raise SystemExit("--labels must refer to valid output rows.")

    t0 = time.perf_counter()
    rows = [run_case(args, label, weights, biases) for label in labels]
    df = pd.DataFrame(rows)
    class_count = len(weights)
    mean_target_row_delta = float(df["target_row_delta"].mean())
    mean_nontarget_row_delta = float(df["nontarget_row_delta_mean"].mean())
    mean_target_row_common_delta = float(df["target_row_common_delta"].mean())
    mean_nontarget_row_common_delta = float(df["nontarget_row_common_delta_mean"].mean())
    mean_target_row_fwd_delta = float(df["target_row_fwd_delta"].mean())
    mean_nontarget_row_fwd_delta = float(df["nontarget_row_fwd_delta_mean"].mean())
    if mean_target_row_delta > 0.0 and mean_nontarget_row_delta < 0.0:
        measured_one_vs_rest_balance_ratio = theory.one_vs_rest_signed_update_balance_ratio(
            class_count,
            mean_target_row_delta,
            mean_nontarget_row_delta,
        )
    else:
        measured_one_vs_rest_balance_ratio = None
    measured_one_vs_rest_epoch_delta = theory.one_vs_rest_signed_epoch_delta(
        class_count,
        mean_target_row_delta,
        mean_nontarget_row_delta,
    )
    measured_one_vs_rest_common_epoch_delta = theory.one_vs_rest_common_epoch_delta(
        class_count,
        mean_target_row_common_delta,
        mean_nontarget_row_common_delta,
    )
    measured_common_drift_to_signed_step_ratio = theory.common_drift_to_signed_step_ratio(
        class_count,
        mean_target_row_delta,
        mean_nontarget_row_delta,
        mean_target_row_common_delta,
        mean_nontarget_row_common_delta,
    )
    results = ROOT / "results/tables"
    results.mkdir(parents=True, exist_ok=True)
    path = results / f"{args.tag}.csv"
    df.to_csv(path, index=False)
    summary = {
        "tag": args.tag,
        "architecture": "composed_multiclass_readout_error_write_loop",
        "rows": len(rows),
        "table": str(path),
        "error_rule": args.error_rule,
        "readout_flow_polarity": args.readout_flow_polarity,
        "readout_flow_write_mode": args.readout_flow_write_mode,
        "flow_pre_store": args.flow_pre_store,
        "flow_pre_cap_f": args.flow_pre_cap_f,
        "flow_pre_consume_width_u": args.flow_pre_consume_width_u,
        "flow_pre_boost_width_u": args.flow_pre_boost_width_u,
        "flow_pre_spike_ref_v": args.flow_pre_spike_ref_v,
        "readout_dp_gate_update_width_u": args.readout_dp_gate_update_width_u,
        "readout_dn_gate_update_width_u": args.readout_dn_gate_update_width_u,
        "readout_dp_discharge_gate_update_width_u": args.readout_dp_discharge_gate_update_width_u,
        "readout_dp_charge_gate_update_width_u": args.readout_dp_charge_gate_update_width_u,
        "readout_dn_discharge_gate_update_width_u": args.readout_dn_discharge_gate_update_width_u,
        "readout_dn_charge_gate_update_width_u": args.readout_dn_charge_gate_update_width_u,
        "readout_write_error_exclusion": args.readout_write_error_exclusion,
        "readout_write_error_exclusion_width_u": args.readout_write_error_exclusion_width_u,
        "readout_center_pull_width_u": args.readout_center_pull_width_u,
        "output_bias_center_pull_width_u": args.output_bias_center_pull_width_u,
        "readout_center_pull_v": args.readout_center_pull_v,
        "readout_center_pull_gate": args.readout_center_pull_gate,
        "readout_center_pull_mode": args.readout_center_pull_mode,
        "mean_target_row_delta": mean_target_row_delta,
        "mean_nontarget_row_delta": mean_nontarget_row_delta,
        "mean_target_row_common_delta": mean_target_row_common_delta,
        "mean_nontarget_row_common_delta": mean_nontarget_row_common_delta,
        "mean_target_row_fwd_delta": mean_target_row_fwd_delta,
        "mean_nontarget_row_fwd_delta": mean_nontarget_row_fwd_delta,
        "measured_one_vs_rest_balance_ratio": measured_one_vs_rest_balance_ratio,
        "measured_one_vs_rest_epoch_delta": measured_one_vs_rest_epoch_delta,
        "measured_one_vs_rest_common_epoch_delta": measured_one_vs_rest_common_epoch_delta,
        "measured_common_drift_to_signed_step_ratio": measured_common_drift_to_signed_step_ratio,
        "target_row_delta_positive_count": int((df["target_row_delta"] > 0).sum()),
        "nontarget_row_delta_negative_count": int((df["nontarget_row_delta_mean"] < 0).sum()),
        "target_bias_delta_positive_count": int((df["target_bias_delta"] > 0).sum()),
        "nontarget_bias_delta_negative_count": int((df["nontarget_bias_delta_mean"] < 0).sum()),
        "max_abs_row_signed_delta": float(
            max(abs(float(row[f"row{out}_signed_delta"])) for row in rows for out in range(outputs))
        ),
        "max_abs_row_common_delta": float(
            max(abs(float(row[f"row{out}_common_delta"])) for row in rows for out in range(outputs))
        ),
        "max_abs_row_fwd_signed_delta": float(
            max(abs(float(row[f"row{out}_fwd_signed_delta"])) for row in rows for out in range(outputs))
        ),
        "max_abs_row_fwd_common_delta": float(
            max(abs(float(row[f"row{out}_fwd_common_delta"])) for row in rows for out in range(outputs))
        ),
        "row_common_to_signed_delta_ratio": float(
            max(abs(float(row[f"row{out}_common_delta"])) for row in rows for out in range(outputs))
            / max(max(abs(float(row[f"row{out}_signed_delta"])) for row in rows for out in range(outputs)), 1e-12)
        ),
        "row_fwd_common_to_signed_delta_ratio": float(
            max(abs(float(row[f"row{out}_fwd_common_delta"])) for row in rows for out in range(outputs))
            / max(
                max(abs(float(row[f"row{out}_fwd_signed_delta"])) for row in rows for out in range(outputs)),
                1e-12,
            )
        ),
        "wall_time_s": time.perf_counter() - t0,
    }
    (results / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(df.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
