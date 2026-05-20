from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_device_xor2_random_hidden import error_cells
from run_device_multicell_classifier import mos_models
from run_spice_sweep import ROOT, detect_spice, run_tiny_test


MEAS_RE = re.compile(r"(?im)^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([-+0-9.eE]+)")
VDD = 1.2
WRITE_MODES = [
    "discharge",
    "bounded_discharge",
    "charge_only",
    "bounded_charge_only",
    "charge_discharge",
    "bounded_charge_discharge",
]
WRITE_STATE_GATE_MODES = ["none", "state_high_discharge", "state_window"]
SIGNED_ACTIONS = ["increase", "decrease"]
ERROR_RULE_ACTIONS = ["label0_mistake", "label1_mistake"]
ERROR_RULE_ACTION_PRE_GATE_MODES = ["raw", "boosted"]


def parse_float_list(text: str) -> list[float]:
    values = [float(part) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return values


def parse_measures(text: str) -> dict[str, float]:
    return {name.lower(): float(value) for name, value in MEAS_RE.findall(text)}


def common_header() -> str:
    return f"""
* Readout MOS mobility characterization.
* Uses the same LEVEL=1 MOS models and read/write stack topology as the
* direct-flow random-hidden runner.
.param VDD={VDD:.12g}
{mos_models()}
Vdd vdd 0 {{VDD}}
.options method=gear maxord=2
""".strip()


def read_branch_netlist(
    theta: float,
    act: float,
    branch: str,
    score_ic: float = 0.30,
    pos_width_u: float = 56.0,
    neg_width_u: float = 48.0,
    score_cap_f: float = 10.0,
) -> str:
    if branch not in {"positive", "negative"}:
        raise ValueError(f"unknown read branch: {branch}")
    if branch == "positive":
        devices = [
            f"Mpos_a vdd act pos0 0 NSENSE W={pos_width_u:.12g}u L=180n",
            f"Mpos_w pos0 w pos1 0 NREL W={pos_width_u:.12g}u L=180n",
            f"Mpos_f pos1 fwd score 0 NREL W={pos_width_u:.12g}u L=180n",
        ]
        score_init = 0.0
        response_expr = "score_final"
    else:
        devices = [
            f"Mneg_f score fwd neg0 0 NREL W={neg_width_u:.12g}u L=180n",
            f"Mneg_a neg0 act neg1 0 NSENSE W={neg_width_u:.12g}u L=180n",
            f"Mneg_w neg1 w 0 0 NREL W={neg_width_u:.12g}u L=180n",
        ]
        score_init = score_ic
        response_expr = "score_initial-score_final"
    return f"""
{common_header()}
Vact act 0 PULSE(0 {act:.12g} 0.75n 20p 20p 3n 8n)
Vfwd fwd 0 PULSE(0 {{VDD}} 0.75n 20p 20p 3n 8n)
Cw w 0 20f IC={theta:.12g}
Rw w 0 1e15
Cscore score 0 {score_cap_f:.12g}f IC={score_init:.12g}
Rscore score 0 1G
{chr(10).join(devices)}
.tran 5p 4.5n uic
.meas tran score_initial FIND V(score) AT=0.60n
.meas tran score_final FIND V(score) AT=3.75n
.meas tran read_response PARAM='{response_expr}'
.meas tran w_final FIND V(w) AT=3.75n
.control
run
print score_initial score_final read_response w_final
.endc
.end
""".lstrip()


def write_actions(write_mode: str) -> tuple[bool, bool]:
    if write_mode not in WRITE_MODES:
        raise ValueError(f"unknown write mode: {write_mode}")
    discharge_enabled = write_mode in {
        "discharge",
        "bounded_discharge",
        "charge_discharge",
        "bounded_charge_discharge",
    }
    charge_enabled = write_mode in {
        "charge_only",
        "bounded_charge_only",
        "charge_discharge",
        "bounded_charge_discharge",
    }
    return discharge_enabled, charge_enabled


def write_mobility_netlist(
    theta: float,
    pre: float,
    delta: float,
    width_u: float,
    action: str,
    write_mode: str,
    write_high_v: float,
    write_low_v: float,
    write_state_gate_mode: str = "none",
) -> str:
    if action not in {"discharge", "charge"}:
        raise ValueError(f"unknown write action: {action}")
    if write_state_gate_mode not in WRITE_STATE_GATE_MODES:
        raise ValueError(f"unknown write state-gate mode: {write_state_gate_mode}")
    bounded_write = write_mode.startswith("bounded_")
    high_node = "whigh" if bounded_write else "vdd"
    low_node = "wlow" if bounded_write else "0"
    write_rails = (
        f"Vwhigh whigh 0 {write_high_v:.12g}\nVwlow wlow 0 {write_low_v:.12g}"
        if bounded_write
        else ""
    )
    if action == "discharge":
        if write_state_gate_mode in {"state_high_discharge", "state_window"}:
            devices = [
                f"Mflow_s w w flow_s 0 NREL W={width_u:.12g}u L=180n",
                f"Mflow_b flow_s bwd flow_b 0 NREL W={width_u:.12g}u L=180n",
                f"Mflow_a flow_b pre flow_a 0 NREL W={width_u:.12g}u L=180n",
                f"Mflow_d flow_a delta {low_node} 0 NSENSE W={width_u:.12g}u L=180n",
            ]
        else:
            devices = [
                f"Mflow_b w bwd flow_b 0 NREL W={width_u:.12g}u L=180n",
                f"Mflow_a flow_b pre flow_a 0 NREL W={width_u:.12g}u L=180n",
                f"Mflow_d flow_a delta {low_node} 0 NSENSE W={width_u:.12g}u L=180n",
            ]
        state_delta_expr = "w_before-w_after"
    else:
        if write_state_gate_mode == "state_window":
            devices = [
                f"Mflow_s {high_node} w flow_s vdd PMOS W={width_u:.12g}u L=180n",
                f"Mflow_b flow_s bwd flow_b 0 NREL W={width_u:.12g}u L=180n",
                f"Mflow_a flow_b pre flow_a 0 NREL W={width_u:.12g}u L=180n",
                f"Mflow_d flow_a delta w 0 NSENSE W={width_u:.12g}u L=180n",
            ]
        else:
            devices = [
                f"Mflow_b {high_node} bwd flow_b 0 NREL W={width_u:.12g}u L=180n",
                f"Mflow_a flow_b pre flow_a 0 NREL W={width_u:.12g}u L=180n",
                f"Mflow_d flow_a delta w 0 NSENSE W={width_u:.12g}u L=180n",
            ]
        state_delta_expr = "w_after-w_before"
    return f"""
{common_header()}
{write_rails}
Vbwd bwd 0 PULSE(0 {{VDD}} 1.00n 20p 20p 2.0n 8n)
Vpre pre 0 PULSE(0 {pre:.12g} 1.00n 20p 20p 2.0n 8n)
Vdelta delta 0 PULSE(0 {delta:.12g} 1.00n 20p 20p 2.0n 8n)
Cw w 0 20f IC={theta:.12g}
Rw w 0 1e15
{chr(10).join(devices)}
Rflow_b flow_b 0 1G
Rflow_a flow_a 0 1G
Rflow_s flow_s 0 1G
Cflow_b flow_b 0 0.02f IC=0
Cflow_a flow_a 0 0.02f IC=0
Cflow_s flow_s 0 0.02f IC=0
.tran 5p 4.5n uic
.meas tran w_before FIND V(w) AT=0.80n
.meas tran w_after FIND V(w) AT=3.75n
.meas tran discharge PARAM='w_before-w_after'
.meas tran charge PARAM='w_after-w_before'
.meas tran state_delta_v PARAM='{state_delta_expr}'
.meas tran signed_state_delta_v PARAM='w_after-w_before'
.control
run
print w_before w_after state_delta_v signed_state_delta_v
.endc
.end
""".lstrip()


def _write_stack(
    cap_node: str,
    prefix: str,
    width_u: float,
    action: str,
    select_gate: str,
    high_node: str,
    low_node: str,
    write_state_gate_mode: str,
) -> list[str]:
    if action == "discharge":
        if write_state_gate_mode in {"state_high_discharge", "state_window"}:
            return [
                f"M{prefix}_s {cap_node} {cap_node} {prefix}_s 0 NREL W={width_u:.12g}u L=180n",
                f"M{prefix}_b {prefix}_s bwd {prefix}_b 0 NREL W={width_u:.12g}u L=180n",
                f"M{prefix}_a {prefix}_b pre {prefix}_a 0 NREL W={width_u:.12g}u L=180n",
                f"M{prefix}_d {prefix}_a {select_gate} {low_node} 0 NSENSE W={width_u:.12g}u L=180n",
                f"R{prefix}_s {prefix}_s 0 1G",
                f"C{prefix}_s {prefix}_s 0 0.02f IC=0",
                f"R{prefix}_b {prefix}_b 0 1G",
                f"C{prefix}_b {prefix}_b 0 0.02f IC=0",
                f"R{prefix}_a {prefix}_a 0 1G",
                f"C{prefix}_a {prefix}_a 0 0.02f IC=0",
            ]
        return [
            f"M{prefix}_b {cap_node} bwd {prefix}_b 0 NREL W={width_u:.12g}u L=180n",
            f"M{prefix}_a {prefix}_b pre {prefix}_a 0 NREL W={width_u:.12g}u L=180n",
            f"M{prefix}_d {prefix}_a {select_gate} {low_node} 0 NSENSE W={width_u:.12g}u L=180n",
            f"R{prefix}_b {prefix}_b 0 1G",
            f"C{prefix}_b {prefix}_b 0 0.02f IC=0",
            f"R{prefix}_a {prefix}_a 0 1G",
            f"C{prefix}_a {prefix}_a 0 0.02f IC=0",
        ]
    if action != "charge":
        raise ValueError(f"unknown write action: {action}")
    if write_state_gate_mode == "state_window":
        return [
            f"M{prefix}_s {high_node} {cap_node} {prefix}_s vdd PMOS W={width_u:.12g}u L=180n",
            f"M{prefix}_b {prefix}_s bwd {prefix}_b 0 NREL W={width_u:.12g}u L=180n",
            f"M{prefix}_a {prefix}_b pre {prefix}_a 0 NREL W={width_u:.12g}u L=180n",
            f"M{prefix}_d {prefix}_a {select_gate} {cap_node} 0 NSENSE W={width_u:.12g}u L=180n",
            f"R{prefix}_s {prefix}_s 0 1G",
            f"C{prefix}_s {prefix}_s 0 0.02f IC=0",
            f"R{prefix}_b {prefix}_b 0 1G",
            f"C{prefix}_b {prefix}_b 0 0.02f IC=0",
            f"R{prefix}_a {prefix}_a 0 1G",
            f"C{prefix}_a {prefix}_a 0 0.02f IC=0",
        ]
    return [
        f"M{prefix}_b {high_node} bwd {prefix}_b 0 NREL W={width_u:.12g}u L=180n",
        f"M{prefix}_a {prefix}_b pre {prefix}_a 0 NREL W={width_u:.12g}u L=180n",
        f"M{prefix}_d {prefix}_a {select_gate} {cap_node} 0 NSENSE W={width_u:.12g}u L=180n",
        f"R{prefix}_b {prefix}_b 0 1G",
        f"C{prefix}_b {prefix}_b 0 0.02f IC=0",
        f"R{prefix}_a {prefix}_a 0 1G",
        f"C{prefix}_a {prefix}_a 0 0.02f IC=0",
    ]


def pair_action_mobility_netlist(
    theta_p: float,
    theta_n: float,
    act: float,
    pre: float,
    delta: float,
    width_u: float,
    signed_action: str,
    write_mode: str,
    pos_write_high_v: float,
    pos_write_low_v: float,
    neg_write_high_v: float,
    neg_write_low_v: float,
    write_state_gate_mode: str = "none",
    pos_width_u: float = 56.0,
    neg_width_u: float = 48.0,
    score_ic: float = 0.30,
    score_cap_f: float = 10.0,
) -> str:
    if signed_action not in SIGNED_ACTIONS:
        raise ValueError(f"unknown signed action: {signed_action}")
    if write_mode not in WRITE_MODES:
        raise ValueError(f"unknown write mode: {write_mode}")
    if write_state_gate_mode not in WRITE_STATE_GATE_MODES:
        raise ValueError(f"unknown write state-gate mode: {write_state_gate_mode}")
    bounded_write = write_mode.startswith("bounded_")
    rail_sources = (
        "\n".join(
            [
                f"Vwphigh wphigh 0 {pos_write_high_v:.12g}",
                f"Vwplow wplow 0 {pos_write_low_v:.12g}",
                f"Vwnhigh wnhigh 0 {neg_write_high_v:.12g}",
                f"Vwnlow wnlow 0 {neg_write_low_v:.12g}",
            ]
        )
        if bounded_write
        else ""
    )
    pos_high_node = "wphigh" if bounded_write else "vdd"
    pos_low_node = "wplow" if bounded_write else "0"
    neg_high_node = "wnhigh" if bounded_write else "vdd"
    neg_low_node = "wnlow" if bounded_write else "0"
    discharge_enabled, charge_enabled = write_actions(write_mode)
    write_devices: list[str] = []
    if signed_action == "increase":
        if charge_enabled:
            write_devices += _write_stack("wp", "wp_ch", width_u, "charge", "delta", pos_high_node, pos_low_node, write_state_gate_mode)
        if discharge_enabled:
            write_devices += _write_stack("wn", "wn_dis", width_u, "discharge", "delta", neg_high_node, neg_low_node, write_state_gate_mode)
        desired_expr = "signed_read_delta"
    else:
        if discharge_enabled:
            write_devices += _write_stack("wp", "wp_dis", width_u, "discharge", "delta", pos_high_node, pos_low_node, write_state_gate_mode)
        if charge_enabled:
            write_devices += _write_stack("wn", "wn_ch", width_u, "charge", "delta", neg_high_node, neg_low_node, write_state_gate_mode)
        desired_expr = "-signed_read_delta"
    if not write_devices:
        raise ValueError(f"write mode {write_mode} enables no write paths")
    read_devices = "\n".join(
        [
            f"Mip_a vdd act_i ip0 0 NSENSE W={pos_width_u:.12g}u L=180n",
            f"Mip_w ip0 wp ip1 0 NREL W={pos_width_u:.12g}u L=180n",
            f"Mip_f ip1 fwd_i score_ip 0 NREL W={pos_width_u:.12g}u L=180n",
            f"Min_f score_in fwd_i in0 0 NREL W={neg_width_u:.12g}u L=180n",
            f"Min_a in0 act_i in1 0 NSENSE W={neg_width_u:.12g}u L=180n",
            f"Min_w in1 wn 0 0 NREL W={neg_width_u:.12g}u L=180n",
            f"Mfp_a vdd act_f fp0 0 NSENSE W={pos_width_u:.12g}u L=180n",
            f"Mfp_w fp0 wp fp1 0 NREL W={pos_width_u:.12g}u L=180n",
            f"Mfp_f fp1 fwd_f score_fp 0 NREL W={pos_width_u:.12g}u L=180n",
            f"Mfn_f score_fn fwd_f fn0 0 NREL W={neg_width_u:.12g}u L=180n",
            f"Mfn_a fn0 act_f fn1 0 NSENSE W={neg_width_u:.12g}u L=180n",
            f"Mfn_w fn1 wn 0 0 NREL W={neg_width_u:.12g}u L=180n",
            "Rip0 ip0 0 1G",
            "Rip1 ip1 0 1G",
            "Rin0 in0 0 1G",
            "Rin1 in1 0 1G",
            "Rfp0 fp0 0 1G",
            "Rfp1 fp1 0 1G",
            "Rfn0 fn0 0 1G",
            "Rfn1 fn1 0 1G",
        ]
    )
    return f"""
{common_header()}
{rail_sources}
Vact_i act_i 0 PULSE(0 {act:.12g} 0.60n 20p 20p 1.20n 8n)
Vfwd_i fwd_i 0 PULSE(0 {{VDD}} 0.60n 20p 20p 1.20n 8n)
Vbwd bwd 0 PULSE(0 {{VDD}} 2.30n 20p 20p 2.00n 8n)
Vpre pre 0 PULSE(0 {pre:.12g} 2.30n 20p 20p 2.00n 8n)
Vdelta delta 0 PULSE(0 {delta:.12g} 2.30n 20p 20p 2.00n 8n)
Vact_f act_f 0 PULSE(0 {act:.12g} 5.00n 20p 20p 1.20n 8n)
Vfwd_f fwd_f 0 PULSE(0 {{VDD}} 5.00n 20p 20p 1.20n 8n)
Cwp wp 0 20f IC={theta_p:.12g}
Cwn wn 0 20f IC={theta_n:.12g}
Rwp wp 0 1e15
Rwn wn 0 1e15
Cscore_ip score_ip 0 {score_cap_f:.12g}f IC=0
Cscore_in score_in 0 {score_cap_f:.12g}f IC={score_ic:.12g}
Cscore_fp score_fp 0 {score_cap_f:.12g}f IC=0
Cscore_fn score_fn 0 {score_cap_f:.12g}f IC={score_ic:.12g}
Rscore_ip score_ip 0 1G
Rscore_in score_in 0 1G
Rscore_fp score_fp 0 1G
Rscore_fn score_fn 0 1G
{read_devices}
{chr(10).join(write_devices)}
.tran 5p 7.0n uic
.meas tran initial_pos_response FIND V(score_ip) AT=1.90n
.meas tran initial_neg_score FIND V(score_in) AT=1.90n
.meas tran initial_neg_response PARAM='{score_ic:.12g}-initial_neg_score'
.meas tran final_pos_response FIND V(score_fp) AT=6.30n
.meas tran final_neg_score FIND V(score_fn) AT=6.30n
.meas tran final_neg_response PARAM='{score_ic:.12g}-final_neg_score'
.meas tran initial_signed_response PARAM='initial_pos_response-initial_neg_response'
.meas tran final_signed_response PARAM='final_pos_response-final_neg_response'
.meas tran signed_read_delta PARAM='final_signed_response-initial_signed_response'
.meas tran desired_signed_read_delta PARAM='{desired_expr}'
.meas tran wp_before FIND V(wp) AT=2.10n
.meas tran wn_before FIND V(wn) AT=2.10n
.meas tran wp_after FIND V(wp) AT=4.70n
.meas tran wn_after FIND V(wn) AT=4.70n
.meas tran wp_state_delta PARAM='wp_after-wp_before'
.meas tran wn_state_delta PARAM='wn_after-wn_before'
.meas tran signed_state_delta PARAM='wp_state_delta-wn_state_delta'
.control
run
print initial_signed_response final_signed_response signed_read_delta desired_signed_read_delta wp_state_delta wn_state_delta signed_state_delta
.endc
.end
""".lstrip()


def error_rule_action_mobility_netlist(
    theta_p: float,
    theta_n: float,
    act: float,
    error_action: str,
    error_rule: str,
    write_mode: str,
    width_u: float,
    pos_write_high_v: float,
    pos_write_low_v: float,
    neg_write_high_v: float,
    neg_write_low_v: float,
    write_state_gate_mode: str = "none",
    pos_width_u: float = 56.0,
    neg_width_u: float = 48.0,
    score_ic: float = 0.30,
    score_cap_f: float = 10.0,
    target_out_v: float = 0.04,
    other_out_v: float = 0.08,
    target_score_v: float = 0.18,
    other_score_v: float = 0.22,
    pre_gate_mode: str = "raw",
    pre_boost_v: float = 0.75,
    pre_boost_cap_f: float = 2.0,
    lead_mode: str = "out_senseamp",
) -> str:
    """Probe the actual training error rails and readout write stacks.

    pair_action_mobility_netlist drives one ideal select gate named ``delta``.
    This diagnostic instead instantiates the same dp/dn error cells used by the
    random-hidden training deck, then connects those rails to the real readout
    flow write topology for both output rows.  It measures whether a simulated
    class-0 or class-1 mistake moves the target row up and the other row down.
    """
    if error_action not in ERROR_RULE_ACTIONS:
        raise ValueError(f"unknown error-rule action: {error_action}")
    if write_mode not in WRITE_MODES:
        raise ValueError(f"unknown write mode: {write_mode}")
    if write_state_gate_mode not in WRITE_STATE_GATE_MODES:
        raise ValueError(f"unknown write state-gate mode: {write_state_gate_mode}")
    if pre_gate_mode not in ERROR_RULE_ACTION_PRE_GATE_MODES:
        raise ValueError(f"unknown error-rule action pre-gate mode: {pre_gate_mode}")
    if pre_gate_mode == "boosted" and pre_boost_cap_f <= 0:
        raise ValueError("boosted pre-gate mode requires positive capacitance")
    target_label = 0 if error_action == "label0_mistake" else 1
    other_label = 1 - target_label
    bounded_write = write_mode.startswith("bounded_")
    pos_high_node = "wphigh" if bounded_write else "vdd"
    pos_low_node = "wplow" if bounded_write else "0"
    neg_high_node = "wnhigh" if bounded_write else "vdd"
    neg_low_node = "wnlow" if bounded_write else "0"
    rail_sources = (
        "\n".join(
            [
                f"Vwphigh wphigh 0 {pos_write_high_v:.12g}",
                f"Vwplow wplow 0 {pos_write_low_v:.12g}",
                f"Vwnhigh wnhigh 0 {neg_write_high_v:.12g}",
                f"Vwnlow wnlow 0 {neg_write_low_v:.12g}",
            ]
        )
        if bounded_write
        else ""
    )
    discharge_enabled, charge_enabled = write_actions(write_mode)
    write_devices: list[str] = []
    for out in range(2):
        if discharge_enabled:
            write_devices += _write_stack(
                f"wn{out}",
                f"w{out}n_dis",
                width_u,
                "discharge",
                f"dp{out}",
                neg_high_node,
                neg_low_node,
                write_state_gate_mode,
            )
            write_devices += _write_stack(
                f"wp{out}",
                f"w{out}p_dis",
                width_u,
                "discharge",
                f"dn{out}",
                pos_high_node,
                pos_low_node,
                write_state_gate_mode,
            )
        if charge_enabled:
            write_devices += _write_stack(
                f"wp{out}",
                f"w{out}p_ch",
                width_u,
                "charge",
                f"dp{out}",
                pos_high_node,
                pos_low_node,
                write_state_gate_mode,
            )
            write_devices += _write_stack(
                f"wn{out}",
                f"w{out}n_ch",
                width_u,
                "charge",
                f"dn{out}",
                neg_high_node,
                neg_low_node,
                write_state_gate_mode,
            )
    if not write_devices:
        raise ValueError(f"write mode {write_mode} enables no write paths")
    read_devices: list[str] = []
    read_caps: list[str] = []
    for phase, prefix, start_ns, at_ns in [("initial", "i", 0.60, 1.90), ("final", "f", 5.00, 6.30)]:
        del phase, at_ns
        read_caps += [
            f"Vact_{prefix} act_{prefix} 0 PULSE(0 {act:.12g} {start_ns:.2f}n 20p 20p 1.20n 8n)",
            f"Vfwd_{prefix} fwd_{prefix} 0 PULSE(0 {{VDD}} {start_ns:.2f}n 20p 20p 1.20n 8n)",
        ]
        for out in range(2):
            read_caps += [
                f"Cscore_{prefix}{out}p score_{prefix}{out}p 0 {score_cap_f:.12g}f IC=0",
                f"Cscore_{prefix}{out}n score_{prefix}{out}n 0 {score_cap_f:.12g}f IC={score_ic:.12g}",
                f"Rscore_{prefix}{out}p score_{prefix}{out}p 0 1G",
                f"Rscore_{prefix}{out}n score_{prefix}{out}n 0 1G",
            ]
            read_devices += [
                f"M{prefix}{out}p_a vdd act_{prefix} {prefix}{out}p0 0 NSENSE W={pos_width_u:.12g}u L=180n",
                f"M{prefix}{out}p_w {prefix}{out}p0 wp{out} {prefix}{out}p1 0 NREL W={pos_width_u:.12g}u L=180n",
                f"M{prefix}{out}p_f {prefix}{out}p1 fwd_{prefix} score_{prefix}{out}p 0 NREL W={pos_width_u:.12g}u L=180n",
                f"M{prefix}{out}n_f score_{prefix}{out}n fwd_{prefix} {prefix}{out}n0 0 NREL W={neg_width_u:.12g}u L=180n",
                f"M{prefix}{out}n_a {prefix}{out}n0 act_{prefix} {prefix}{out}n1 0 NSENSE W={neg_width_u:.12g}u L=180n",
                f"M{prefix}{out}n_w {prefix}{out}n1 wn{out} 0 0 NREL W={neg_width_u:.12g}u L=180n",
                f"R{prefix}{out}p0 {prefix}{out}p0 0 1G",
                f"R{prefix}{out}p1 {prefix}{out}p1 0 1G",
                f"R{prefix}{out}n0 {prefix}{out}n0 0 1G",
                f"R{prefix}{out}n1 {prefix}{out}n1 0 1G",
            ]
    out0 = target_out_v if target_label == 0 else other_out_v
    out1 = target_out_v if target_label == 1 else other_out_v
    score0 = target_score_v if target_label == 0 else other_score_v
    score1 = target_score_v if target_label == 1 else other_score_v
    t0 = VDD if target_label == 0 else 0.0
    t1 = VDD if target_label == 1 else 0.0
    winning_label = other_label
    if lead_mode == "score_direct":
        lead_sources = ""
    elif lead_mode == "out_senseamp":
        lead_sources = "\n".join(
            [
                f"Vlead01 lead01 0 {VDD if winning_label == 1 else 0.0:.12g}",
                f"Vlead10 lead10 0 {VDD if winning_label == 0 else 0.0:.12g}",
            ]
        )
    else:
        lead_sources = "\n".join(
            [
                f"Vlead01 lead01 0 {VDD if winning_label == 0 else 0.0:.12g}",
                f"Vlead10 lead10 0 {VDD if winning_label == 1 else 0.0:.12g}",
            ]
        )
    desired0 = "row0_signed_read_delta" if target_label == 0 else "-row0_signed_read_delta"
    desired1 = "row1_signed_read_delta" if target_label == 1 else "-row1_signed_read_delta"
    if pre_gate_mode == "raw":
        pre_gate_block = f"Vpre pre 0 PULSE(0 {act:.12g} 2.30n 20p 20p 2.00n 8n)"
    else:
        pre_gate_block = "\n".join(
            [
                f"Vpreboost preboost 0 PULSE(0 {pre_boost_v:.12g} 2.30n 20p 20p 2.00n 8n)",
                f"Cpre pre 0 {pre_boost_cap_f:.12g}f IC={act:.12g}",
                f"Cpre_kick preboost pre {pre_boost_cap_f:.12g}f",
                "Rpre pre 0 1G",
            ]
        )
    return f"""
{common_header()}
{rail_sources}
Vt0 t0 0 {t0:.12g}
Vt1 t1 0 {t1:.12g}
Vout0 out0 0 {out0:.12g}
Vout1 out1 0 {out1:.12g}
Vscore0 score0 0 {score0:.12g}
Vscore1 score1 0 {score1:.12g}
{lead_sources}
Verr err 0 PULSE(0 {{VDD}} 2.30n 20p 20p 2.00n 8n)
Vbwd bwd 0 PULSE(0 {{VDD}} 2.30n 20p 20p 2.00n 8n)
{pre_gate_block}
Cwp0 wp0 0 20f IC={theta_p:.12g}
Cwn0 wn0 0 20f IC={theta_n:.12g}
Cwp1 wp1 0 20f IC={theta_p:.12g}
Cwn1 wn1 0 20f IC={theta_n:.12g}
Rwp0 wp0 0 1e15
Rwn0 wn0 0 1e15
Rwp1 wp1 0 1e15
Rwn1 wn1 0 1e15
Cdp0 dp0 0 20f IC=0
Cdn0 dn0 0 20f IC=0
Cdp1 dp1 0 20f IC=0
Cdn1 dn1 0 20f IC=0
Rdp0 dp0 0 1G
Rdn0 dn0 0 1G
Rdp1 dp1 0 1G
Rdn1 dn1 0 1G
{chr(10).join(read_caps)}
{chr(10).join(read_devices)}
{error_cells(error_rule, latch_boost_width_u=0.0, lead_mode=lead_mode)}
{chr(10).join(write_devices)}
.tran 5p 7.0n uic
.meas tran dp0_probe FIND V(dp0) AT=3.00n
.meas tran dn0_probe FIND V(dn0) AT=3.00n
.meas tran dp1_probe FIND V(dp1) AT=3.00n
.meas tran dn1_probe FIND V(dn1) AT=3.00n
.meas tran pre_probe FIND V(pre) AT=3.00n
.meas tran row0_initial_pos FIND V(score_i0p) AT=1.90n
.meas tran row0_initial_neg_score FIND V(score_i0n) AT=1.90n
.meas tran row0_initial_neg PARAM='{score_ic:.12g}-row0_initial_neg_score'
.meas tran row0_final_pos FIND V(score_f0p) AT=6.30n
.meas tran row0_final_neg_score FIND V(score_f0n) AT=6.30n
.meas tran row0_final_neg PARAM='{score_ic:.12g}-row0_final_neg_score'
.meas tran row0_initial_signed PARAM='row0_initial_pos-row0_initial_neg'
.meas tran row0_final_signed PARAM='row0_final_pos-row0_final_neg'
.meas tran row0_signed_read_delta PARAM='row0_final_signed-row0_initial_signed'
.meas tran row0_desired_signed_read_delta PARAM='{desired0}'
.meas tran row1_initial_pos FIND V(score_i1p) AT=1.90n
.meas tran row1_initial_neg_score FIND V(score_i1n) AT=1.90n
.meas tran row1_initial_neg PARAM='{score_ic:.12g}-row1_initial_neg_score'
.meas tran row1_final_pos FIND V(score_f1p) AT=6.30n
.meas tran row1_final_neg_score FIND V(score_f1n) AT=6.30n
.meas tran row1_final_neg PARAM='{score_ic:.12g}-row1_final_neg_score'
.meas tran row1_initial_signed PARAM='row1_initial_pos-row1_initial_neg'
.meas tran row1_final_signed PARAM='row1_final_pos-row1_final_neg'
.meas tran row1_signed_read_delta PARAM='row1_final_signed-row1_initial_signed'
.meas tran row1_desired_signed_read_delta PARAM='{desired1}'
.control
run
print dp0_probe dn0_probe dp1_probe dn1_probe pre_probe row0_signed_read_delta row0_desired_signed_read_delta row1_signed_read_delta row1_desired_signed_read_delta
.endc
.end
""".lstrip()


def run_netlist(spice_bin: str, path: Path, netlist: str, timeout: float) -> dict[str, float]:
    path.write_text(netlist)
    cmd = [spice_bin, "-b", str(path)] if "ngspice" in Path(spice_bin).name.lower() else [spice_bin, str(path)]
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-3000:])
    measures = parse_measures(proc.stdout + "\n" + proc.stderr)
    if not measures:
        raise RuntimeError("ngspice produced no parseable measurements:\n" + (proc.stdout + proc.stderr)[-3000:])
    return measures


def add_slopes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["read_slope"] = np.nan
    out["read_gain"] = np.nan
    read_mask = out["experiment"].astype(str).str.startswith("read_") & out["act"].notna()
    out.loc[read_mask, "read_gain"] = out.loc[read_mask, "read_response"] / out.loc[read_mask, "act"]
    for (experiment, act), idx in out.groupby(["experiment", "act"], dropna=False).groups.items():
        if not str(experiment).startswith("read_"):
            continue
        ordered_idx = list(idx)
        sub = out.loc[ordered_idx].sort_values("theta")
        if len(sub) < 2:
            slopes = np.full(len(sub), np.nan)
        else:
            slopes = np.gradient(sub["read_response"].to_numpy(), sub["theta"].to_numpy())
        out.loc[sub.index, "read_slope"] = slopes
    return out


def signed_mobility_table(df: pd.DataFrame, write_mode: str, write_low_v: float, write_high_v: float) -> pd.DataFrame:
    read_pos = (
        df[df["experiment"] == "read_positive"][["theta", "act", "read_gain", "read_slope"]]
        .rename(columns={"read_gain": "read_positive_gain", "read_slope": "read_positive_slope"})
        .copy()
    )
    read_neg = (
        df[df["experiment"] == "read_negative"][["theta", "act", "read_gain", "read_slope"]]
        .rename(columns={"read_gain": "read_negative_gain", "read_slope": "read_negative_slope"})
        .copy()
    )
    signed = read_pos.merge(read_neg, on=["theta", "act"], how="inner")
    signed["write_mode"] = write_mode
    signed["write_low_v"] = write_low_v if write_mode.startswith("bounded_") else np.nan
    signed["write_high_v"] = write_high_v if write_mode.startswith("bounded_") else np.nan

    signed["write_discharge_v"] = 0.0
    signed["write_charge_v"] = 0.0
    discharge = df[df["experiment"] == "write_discharge"][["theta", "state_delta_v"]].rename(
        columns={"state_delta_v": "write_discharge_v"}
    )
    charge = df[df["experiment"] == "write_charge"][["theta", "state_delta_v"]].rename(
        columns={"state_delta_v": "write_charge_v"}
    )
    if not discharge.empty:
        signed = signed.drop(columns=["write_discharge_v"]).merge(discharge, on="theta", how="left")
    if not charge.empty:
        signed = signed.drop(columns=["write_charge_v"]).merge(charge, on="theta", how="left")
    signed["write_discharge_v"] = signed["write_discharge_v"].fillna(0.0)
    signed["write_charge_v"] = signed["write_charge_v"].fillna(0.0)

    # For normal-polarity signed readout:
    #   positive desired update discharges the negative branch and/or charges the positive branch;
    #   negative desired update discharges the positive branch and/or charges the negative branch.
    signed["signed_increase_mobility"] = (
        signed["read_negative_slope"] * signed["write_discharge_v"]
        + signed["read_positive_slope"] * signed["write_charge_v"]
    )
    signed["signed_decrease_mobility"] = (
        signed["read_positive_slope"] * signed["write_discharge_v"]
        + signed["read_negative_slope"] * signed["write_charge_v"]
    )
    # If a write primitive also multiplies its state update by G_eff'(theta),
    # the induced effective-weight mobility gains another slope factor.  These
    # columns are a diagnostic for physical-state-gradient-style writes; the raw
    # columns above remain the actual measured hardware update.
    signed["physical_gradient_increase_mobility"] = (
        signed["read_negative_slope"] ** 2 * signed["write_discharge_v"]
        + signed["read_positive_slope"] ** 2 * signed["write_charge_v"]
    )
    signed["physical_gradient_decrease_mobility"] = (
        signed["read_positive_slope"] ** 2 * signed["write_discharge_v"]
        + signed["read_negative_slope"] ** 2 * signed["write_charge_v"]
    )
    signed["signed_update_sign_aligned"] = (
        (signed["signed_increase_mobility"] > 0) & (signed["signed_decrease_mobility"] > 0)
    )
    signed["physical_gradient_sign_aligned"] = (
        (signed["physical_gradient_increase_mobility"] > 0)
        & (signed["physical_gradient_decrease_mobility"] > 0)
    )
    max_mobility = signed[["signed_increase_mobility", "signed_decrease_mobility"]].max(axis=1)
    min_mobility = signed[["signed_increase_mobility", "signed_decrease_mobility"]].min(axis=1)
    signed["signed_mobility_balance"] = np.where(max_mobility > 0, min_mobility / max_mobility, np.nan)
    physical_max = signed[
        ["physical_gradient_increase_mobility", "physical_gradient_decrease_mobility"]
    ].max(axis=1)
    physical_min = signed[
        ["physical_gradient_increase_mobility", "physical_gradient_decrease_mobility"]
    ].min(axis=1)
    signed["physical_gradient_mobility_balance"] = np.where(
        physical_max > 0,
        physical_min / physical_max,
        np.nan,
    )
    return signed


def _branch_read_table(df: pd.DataFrame, experiment: str, suffix: str) -> pd.DataFrame:
    return (
        df[df["experiment"] == experiment][["theta", "act", "read_gain", "read_slope"]]
        .rename(
            columns={
                "theta": f"theta_{suffix}",
                "read_gain": f"read_{suffix}_gain",
                "read_slope": f"read_{suffix}_slope",
            }
        )
        .copy()
    )


def _branch_write_table(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    out = pd.DataFrame({f"theta_{suffix}": sorted(df["theta"].dropna().unique())})
    out[f"write_{suffix}_discharge_v"] = 0.0
    out[f"write_{suffix}_charge_v"] = 0.0
    discharge = df[df["experiment"] == "write_discharge"][["theta", "state_delta_v"]].rename(
        columns={"theta": f"theta_{suffix}", "state_delta_v": f"write_{suffix}_discharge_v"}
    )
    charge = df[df["experiment"] == "write_charge"][["theta", "state_delta_v"]].rename(
        columns={"theta": f"theta_{suffix}", "state_delta_v": f"write_{suffix}_charge_v"}
    )
    if not discharge.empty:
        out = out.drop(columns=[f"write_{suffix}_discharge_v"]).merge(discharge, on=f"theta_{suffix}", how="left")
    if not charge.empty:
        out = out.drop(columns=[f"write_{suffix}_charge_v"]).merge(charge, on=f"theta_{suffix}", how="left")
    out[f"write_{suffix}_discharge_v"] = out[f"write_{suffix}_discharge_v"].fillna(0.0)
    out[f"write_{suffix}_charge_v"] = out[f"write_{suffix}_charge_v"].fillna(0.0)
    return out


def branch_pair_signed_mobility_table(
    positive_df: pd.DataFrame,
    negative_df: pd.DataFrame,
    write_mode: str,
) -> pd.DataFrame:
    """Compute signed readout mobility for independently biased p/n branches.

    signed_mobility_table is the same-theta diagnostic.  This table is the
    branch-specific diagnostic: the positive branch can sit at theta_p while
    the negative branch sits at theta_n.  That is the relevant case for the
    branch-range experiments where the two MOS branches use different storage
    windows.
    """
    discharge_enabled, charge_enabled = write_actions(write_mode)
    pos = _branch_read_table(positive_df, "read_positive", "p").merge(
        _branch_write_table(positive_df, "p"),
        on="theta_p",
        how="left",
    )
    neg = _branch_read_table(negative_df, "read_negative", "n").merge(
        _branch_write_table(negative_df, "n"),
        on="theta_n",
        how="left",
    )
    if not discharge_enabled:
        pos["write_p_discharge_v"] = 0.0
        neg["write_n_discharge_v"] = 0.0
    if not charge_enabled:
        pos["write_p_charge_v"] = 0.0
        neg["write_n_charge_v"] = 0.0
    pair = pos.assign(_pair_key=1).merge(neg.assign(_pair_key=1), on=["act", "_pair_key"]).drop(columns=["_pair_key"])
    pair["write_mode"] = write_mode
    pair["signed_read_gain"] = pair["read_p_gain"] - pair["read_n_gain"]
    pair["signed_increase_mobility"] = (
        pair["read_n_slope"] * pair["write_n_discharge_v"]
        + pair["read_p_slope"] * pair["write_p_charge_v"]
    )
    pair["signed_decrease_mobility"] = (
        pair["read_p_slope"] * pair["write_p_discharge_v"]
        + pair["read_n_slope"] * pair["write_n_charge_v"]
    )
    pair["physical_gradient_increase_mobility"] = (
        pair["read_n_slope"] ** 2 * pair["write_n_discharge_v"]
        + pair["read_p_slope"] ** 2 * pair["write_p_charge_v"]
    )
    pair["physical_gradient_decrease_mobility"] = (
        pair["read_p_slope"] ** 2 * pair["write_p_discharge_v"]
        + pair["read_n_slope"] ** 2 * pair["write_n_charge_v"]
    )
    pair["signed_update_sign_aligned"] = (
        (pair["signed_increase_mobility"] > 0) & (pair["signed_decrease_mobility"] > 0)
    )
    pair["physical_gradient_sign_aligned"] = (
        (pair["physical_gradient_increase_mobility"] > 0)
        & (pair["physical_gradient_decrease_mobility"] > 0)
    )
    max_mobility = pair[["signed_increase_mobility", "signed_decrease_mobility"]].max(axis=1)
    min_mobility = pair[["signed_increase_mobility", "signed_decrease_mobility"]].min(axis=1)
    pair["signed_mobility_balance"] = np.where(max_mobility > 0, min_mobility / max_mobility, np.nan)
    physical_max = pair[
        ["physical_gradient_increase_mobility", "physical_gradient_decrease_mobility"]
    ].max(axis=1)
    physical_min = pair[
        ["physical_gradient_increase_mobility", "physical_gradient_decrease_mobility"]
    ].min(axis=1)
    pair["physical_gradient_mobility_balance"] = np.where(
        physical_max > 0,
        physical_min / physical_max,
        np.nan,
    )
    return pair


def summarize_signed_mobility(signed: pd.DataFrame, write_mode: str, write_low_v: float, write_high_v: float) -> dict[str, Any]:
    operating = signed
    if write_mode.startswith("bounded_"):
        operating = signed[(signed["theta"] >= write_low_v) & (signed["theta"] <= write_high_v)]
    if operating.empty:
        operating = signed
    near_zero = 1e-9
    return {
        "signed_mobility_csv": None,
        "operating_theta_min_v": write_low_v if write_mode.startswith("bounded_") else float(signed["theta"].min()),
        "operating_theta_max_v": write_high_v if write_mode.startswith("bounded_") else float(signed["theta"].max()),
        "signed_mobility_rows": int(len(signed)),
        "operating_signed_mobility_rows": int(len(operating)),
        "signed_update_sign_aligned_fraction": float(operating["signed_update_sign_aligned"].mean()),
        "min_signed_increase_mobility": float(operating["signed_increase_mobility"].min()),
        "max_signed_increase_mobility": float(operating["signed_increase_mobility"].max()),
        "min_signed_decrease_mobility": float(operating["signed_decrease_mobility"].min()),
        "max_signed_decrease_mobility": float(operating["signed_decrease_mobility"].max()),
        "min_signed_mobility_balance": float(operating["signed_mobility_balance"].min(skipna=True)),
        "physical_gradient_sign_aligned_fraction": float(operating["physical_gradient_sign_aligned"].mean()),
        "min_physical_gradient_increase_mobility": float(
            operating["physical_gradient_increase_mobility"].min()
        ),
        "max_physical_gradient_increase_mobility": float(
            operating["physical_gradient_increase_mobility"].max()
        ),
        "min_physical_gradient_decrease_mobility": float(
            operating["physical_gradient_decrease_mobility"].min()
        ),
        "max_physical_gradient_decrease_mobility": float(
            operating["physical_gradient_decrease_mobility"].max()
        ),
        "min_physical_gradient_mobility_balance": float(
            operating["physical_gradient_mobility_balance"].min(skipna=True)
        ),
        "near_zero_signed_increase_count": int((operating["signed_increase_mobility"].abs() < near_zero).sum()),
        "near_zero_signed_decrease_count": int((operating["signed_decrease_mobility"].abs() < near_zero).sum()),
    }


def summarize_branch_pair_mobility(
    pair: pd.DataFrame,
    pos_low_v: float,
    pos_high_v: float,
    neg_low_v: float,
    neg_high_v: float,
    summary_act_v: float | None = None,
) -> dict[str, Any]:
    operating_all_act = pair[
        (pair["theta_p"] >= pos_low_v)
        & (pair["theta_p"] <= pos_high_v)
        & (pair["theta_n"] >= neg_low_v)
        & (pair["theta_n"] <= neg_high_v)
    ]
    if operating_all_act.empty:
        operating_all_act = pair
    operating = operating_all_act
    if summary_act_v is not None:
        at_summary_act = operating_all_act[np.isclose(operating_all_act["act"], summary_act_v)]
        if not at_summary_act.empty:
            operating = at_summary_act
    if operating.empty:
        operating = pair
    aligned = operating[operating["signed_update_sign_aligned"]]
    best_source = aligned if not aligned.empty else operating
    best = best_source.sort_values(
        ["signed_mobility_balance", "signed_increase_mobility", "signed_decrease_mobility"],
        ascending=[False, False, False],
    ).iloc[0]
    gain_safe = (
        operating_all_act.groupby(["theta_p", "theta_n"], as_index=False)
        .agg(
            min_signed_read_gain=("signed_read_gain", "min"),
            mean_signed_read_gain=("signed_read_gain", "mean"),
            min_signed_mobility_balance=("signed_mobility_balance", "min"),
            mean_signed_mobility_balance=("signed_mobility_balance", "mean"),
            aligned_fraction=("signed_update_sign_aligned", "mean"),
            min_signed_increase_mobility=("signed_increase_mobility", "min"),
            min_signed_decrease_mobility=("signed_decrease_mobility", "min"),
            min_physical_gradient_mobility_balance=("physical_gradient_mobility_balance", "min"),
            physical_gradient_aligned_fraction=("physical_gradient_sign_aligned", "mean"),
        )
        .copy()
    )
    gain_safe["gain_sign_safe"] = gain_safe["min_signed_read_gain"] > 0
    gain_safe["mobility_sign_safe"] = gain_safe["aligned_fraction"] >= 1.0
    fully_safe = gain_safe[gain_safe["gain_sign_safe"] & gain_safe["mobility_sign_safe"]]
    gain_safe_source = fully_safe if not fully_safe.empty else gain_safe
    best_gain_safe = gain_safe_source.sort_values(
        [
            "gain_sign_safe",
            "mobility_sign_safe",
            "min_signed_read_gain",
            "min_signed_mobility_balance",
            "mean_signed_read_gain",
        ],
        ascending=[False, False, False, False, False],
    ).iloc[0]
    return {
        "branch_pair_mobility_csv": None,
        "branch_pair_mobility_table_csv": None,
        "branch_pair_summary_act_v": summary_act_v,
        "branch_pair_pos_operating_theta_min_v": pos_low_v,
        "branch_pair_pos_operating_theta_max_v": pos_high_v,
        "branch_pair_neg_operating_theta_min_v": neg_low_v,
        "branch_pair_neg_operating_theta_max_v": neg_high_v,
        "branch_pair_rows": int(len(pair)),
        "branch_pair_operating_rows": int(len(operating)),
        "branch_pair_operating_all_act_rows": int(len(operating_all_act)),
        "branch_pair_update_sign_aligned_fraction": float(operating["signed_update_sign_aligned"].mean()),
        "branch_pair_all_act_update_sign_aligned_fraction": float(
            operating_all_act["signed_update_sign_aligned"].mean()
        ),
        "branch_pair_min_signed_read_gain": float(operating["signed_read_gain"].min()),
        "branch_pair_max_signed_read_gain": float(operating["signed_read_gain"].max()),
        "branch_pair_mean_signed_read_gain": float(operating["signed_read_gain"].mean()),
        "branch_pair_all_act_min_signed_read_gain": float(operating_all_act["signed_read_gain"].min()),
        "branch_pair_all_act_negative_signed_read_gain_fraction": float(
            (operating_all_act["signed_read_gain"] <= 0).mean()
        ),
        "branch_pair_min_signed_increase_mobility": float(operating["signed_increase_mobility"].min()),
        "branch_pair_max_signed_increase_mobility": float(operating["signed_increase_mobility"].max()),
        "branch_pair_min_signed_decrease_mobility": float(operating["signed_decrease_mobility"].min()),
        "branch_pair_max_signed_decrease_mobility": float(operating["signed_decrease_mobility"].max()),
        "branch_pair_physical_gradient_sign_aligned_fraction": float(
            operating["physical_gradient_sign_aligned"].mean()
        ),
        "branch_pair_min_physical_gradient_increase_mobility": float(
            operating["physical_gradient_increase_mobility"].min()
        ),
        "branch_pair_max_physical_gradient_increase_mobility": float(
            operating["physical_gradient_increase_mobility"].max()
        ),
        "branch_pair_min_physical_gradient_decrease_mobility": float(
            operating["physical_gradient_decrease_mobility"].min()
        ),
        "branch_pair_max_physical_gradient_decrease_mobility": float(
            operating["physical_gradient_decrease_mobility"].max()
        ),
        "branch_pair_min_physical_gradient_mobility_balance": float(
            operating["physical_gradient_mobility_balance"].min(skipna=True)
        ),
        "branch_pair_best_theta_p_v": float(best["theta_p"]),
        "branch_pair_best_theta_n_v": float(best["theta_n"]),
        "branch_pair_best_signed_read_gain": float(best["signed_read_gain"]),
        "branch_pair_best_signed_increase_mobility": float(best["signed_increase_mobility"]),
        "branch_pair_best_signed_decrease_mobility": float(best["signed_decrease_mobility"]),
        "branch_pair_best_signed_mobility_balance": float(best["signed_mobility_balance"]),
        "branch_pair_best_physical_gradient_increase_mobility": float(
            best["physical_gradient_increase_mobility"]
        ),
        "branch_pair_best_physical_gradient_decrease_mobility": float(
            best["physical_gradient_decrease_mobility"]
        ),
        "branch_pair_best_physical_gradient_mobility_balance": float(
            best["physical_gradient_mobility_balance"]
        ),
        "branch_pair_gain_safe_pair_count": int(len(fully_safe)),
        "branch_pair_best_gain_safe_theta_p_v": float(best_gain_safe["theta_p"]),
        "branch_pair_best_gain_safe_theta_n_v": float(best_gain_safe["theta_n"]),
        "branch_pair_best_gain_safe_min_signed_read_gain": float(best_gain_safe["min_signed_read_gain"]),
        "branch_pair_best_gain_safe_mean_signed_read_gain": float(best_gain_safe["mean_signed_read_gain"]),
        "branch_pair_best_gain_safe_min_signed_mobility_balance": float(
            best_gain_safe["min_signed_mobility_balance"]
        ),
        "branch_pair_best_gain_safe_min_physical_gradient_mobility_balance": float(
            best_gain_safe["min_physical_gradient_mobility_balance"]
        ),
        "branch_pair_best_gain_safe_mean_signed_mobility_balance": float(
            best_gain_safe["mean_signed_mobility_balance"]
        ),
        "branch_pair_best_gain_safe_aligned_fraction": float(best_gain_safe["aligned_fraction"]),
        "branch_pair_best_gain_safe_physical_gradient_aligned_fraction": float(
            best_gain_safe["physical_gradient_aligned_fraction"]
        ),
    }


def summarize_pair_action_mobility(pair_action: pd.DataFrame) -> dict[str, Any]:
    if pair_action.empty:
        return {
            "pair_action_mobility_csv": None,
            "pair_action_mobility_table_csv": None,
            "pair_action_rows": 0,
        }
    out = pair_action.copy()
    out["pair_action_sign_aligned"] = out["desired_signed_read_delta"] > 0
    grouped = (
        out.groupby(["theta_p", "theta_n"], as_index=False)
        .agg(
            aligned_fraction=("pair_action_sign_aligned", "mean"),
            min_desired_signed_read_delta=("desired_signed_read_delta", "min"),
            mean_desired_signed_read_delta=("desired_signed_read_delta", "mean"),
            min_abs_signed_read_delta=("signed_read_delta", lambda s: float(np.abs(s).min())),
            mean_abs_signed_read_delta=("signed_read_delta", lambda s: float(np.abs(s).mean())),
        )
        .copy()
    )
    best = grouped.sort_values(
        ["aligned_fraction", "min_desired_signed_read_delta", "mean_abs_signed_read_delta"],
        ascending=[False, False, False],
    ).iloc[0]
    return {
        "pair_action_mobility_csv": None,
        "pair_action_mobility_table_csv": None,
        "pair_action_rows": int(len(out)),
        "pair_action_sign_aligned_fraction": float(out["pair_action_sign_aligned"].mean()),
        "pair_action_min_desired_signed_read_delta": float(out["desired_signed_read_delta"].min()),
        "pair_action_max_desired_signed_read_delta": float(out["desired_signed_read_delta"].max()),
        "pair_action_min_signed_read_delta": float(out["signed_read_delta"].min()),
        "pair_action_max_signed_read_delta": float(out["signed_read_delta"].max()),
        "pair_action_best_theta_p_v": float(best["theta_p"]),
        "pair_action_best_theta_n_v": float(best["theta_n"]),
        "pair_action_best_aligned_fraction": float(best["aligned_fraction"]),
        "pair_action_best_min_desired_signed_read_delta": float(best["min_desired_signed_read_delta"]),
        "pair_action_best_mean_desired_signed_read_delta": float(best["mean_desired_signed_read_delta"]),
        "pair_action_best_mean_abs_signed_read_delta": float(best["mean_abs_signed_read_delta"]),
    }


def summarize_error_rule_action_mobility(error_rule_action: pd.DataFrame) -> dict[str, Any]:
    if error_rule_action.empty:
        return {
            "error_rule_action_mobility_csv": None,
            "error_rule_action_mobility_table_csv": None,
            "error_rule_action_rows": 0,
        }
    out = error_rule_action.copy()
    out["row0_action_sign_aligned"] = out["row0_desired_signed_read_delta"] > 0
    out["row1_action_sign_aligned"] = out["row1_desired_signed_read_delta"] > 0
    out["both_rows_action_sign_aligned"] = out["row0_action_sign_aligned"] & out["row1_action_sign_aligned"]
    grouped = (
        out.groupby(["theta_p", "theta_n", "error_rule"], as_index=False)
        .agg(
            both_rows_aligned_fraction=("both_rows_action_sign_aligned", "mean"),
            row0_aligned_fraction=("row0_action_sign_aligned", "mean"),
            row1_aligned_fraction=("row1_action_sign_aligned", "mean"),
            min_row0_desired_signed_read_delta=("row0_desired_signed_read_delta", "min"),
            min_row1_desired_signed_read_delta=("row1_desired_signed_read_delta", "min"),
            mean_abs_row0_signed_read_delta=("row0_signed_read_delta", lambda s: float(np.abs(s).mean())),
            mean_abs_row1_signed_read_delta=("row1_signed_read_delta", lambda s: float(np.abs(s).mean())),
        )
        .copy()
    )
    grouped["min_both_desired_signed_read_delta"] = grouped[
        ["min_row0_desired_signed_read_delta", "min_row1_desired_signed_read_delta"]
    ].min(axis=1)
    grouped["mean_abs_signed_read_delta"] = grouped[
        ["mean_abs_row0_signed_read_delta", "mean_abs_row1_signed_read_delta"]
    ].mean(axis=1)
    best = grouped.sort_values(
        ["both_rows_aligned_fraction", "min_both_desired_signed_read_delta", "mean_abs_signed_read_delta"],
        ascending=[False, False, False],
    ).iloc[0]
    return {
        "error_rule_action_mobility_csv": None,
        "error_rule_action_mobility_table_csv": None,
        "error_rule_action_rows": int(len(out)),
        "error_rule_action_both_rows_sign_aligned_fraction": float(out["both_rows_action_sign_aligned"].mean()),
        "error_rule_action_row0_sign_aligned_fraction": float(out["row0_action_sign_aligned"].mean()),
        "error_rule_action_row1_sign_aligned_fraction": float(out["row1_action_sign_aligned"].mean()),
        "error_rule_action_min_row0_desired_signed_read_delta": float(
            out["row0_desired_signed_read_delta"].min()
        ),
        "error_rule_action_min_row1_desired_signed_read_delta": float(
            out["row1_desired_signed_read_delta"].min()
        ),
        "error_rule_action_max_dp0_v": float(out["dp0_probe"].max()),
        "error_rule_action_max_dn0_v": float(out["dn0_probe"].max()),
        "error_rule_action_max_dp1_v": float(out["dp1_probe"].max()),
        "error_rule_action_max_dn1_v": float(out["dn1_probe"].max()),
        "error_rule_action_max_pre_probe_v": float(out["pre_probe"].max())
        if "pre_probe" in out
        else None,
        "error_rule_action_best_error_rule": str(best["error_rule"]),
        "error_rule_action_best_theta_p_v": float(best["theta_p"]),
        "error_rule_action_best_theta_n_v": float(best["theta_n"]),
        "error_rule_action_best_both_rows_aligned_fraction": float(best["both_rows_aligned_fraction"]),
        "error_rule_action_best_min_both_desired_signed_read_delta": float(
            best["min_both_desired_signed_read_delta"]
        ),
        "error_rule_action_best_mean_abs_signed_read_delta": float(best["mean_abs_signed_read_delta"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="device_readout_mobility_sweep")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--write-width-u", type=float, default=0.02)
    ap.add_argument("--write-mode", choices=WRITE_MODES, default="discharge")
    ap.add_argument("--write-state-gate-mode", choices=WRITE_STATE_GATE_MODES, default="none")
    ap.add_argument("--write-high-v", type=float, default=VDD)
    ap.add_argument("--write-low-v", type=float, default=0.0)
    ap.add_argument("--pos-write-high-v", type=float)
    ap.add_argument("--pos-write-low-v", type=float)
    ap.add_argument("--neg-write-high-v", type=float)
    ap.add_argument("--neg-write-low-v", type=float)
    ap.add_argument("--pre", type=float, default=0.65)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--pos-width-u", type=float, default=56.0)
    ap.add_argument("--neg-width-u", type=float, default=48.0)
    ap.add_argument("--negative-score-ic-v", type=float, default=0.30)
    ap.add_argument("--score-cap-f", type=float, default=10.0)
    ap.add_argument("--theta-values", type=parse_float_list)
    ap.add_argument("--act-values", type=parse_float_list)
    ap.add_argument("--pair-action-sweep", action="store_true")
    ap.add_argument("--pair-theta-p-values", type=parse_float_list)
    ap.add_argument("--pair-theta-n-values", type=parse_float_list)
    ap.add_argument("--pair-act-values", type=parse_float_list)
    ap.add_argument("--error-rule-action-sweep", action="store_true")
    ap.add_argument("--error-rule-actions", choices=ERROR_RULE_ACTIONS, nargs="+", default=ERROR_RULE_ACTIONS)
    ap.add_argument("--error-rule-action-rules", nargs="+", default=["perceptron", "out_competitive"])
    ap.add_argument("--error-rule-action-theta-p-values", type=parse_float_list)
    ap.add_argument("--error-rule-action-theta-n-values", type=parse_float_list)
    ap.add_argument("--error-rule-action-act-values", type=parse_float_list)
    ap.add_argument("--error-rule-action-target-out-v", type=float, default=0.04)
    ap.add_argument("--error-rule-action-other-out-v", type=float, default=0.08)
    ap.add_argument("--error-rule-action-target-score-v", type=float, default=0.18)
    ap.add_argument("--error-rule-action-other-score-v", type=float, default=0.22)
    ap.add_argument(
        "--error-rule-action-pre-gate-mode",
        choices=ERROR_RULE_ACTION_PRE_GATE_MODES,
        default="raw",
    )
    ap.add_argument("--error-rule-action-pre-boost-v", type=float, default=0.75)
    ap.add_argument("--error-rule-action-pre-boost-cap-f", type=float, default=2.0)
    ap.add_argument("--error-rule-action-lead-mode", default="out_senseamp")
    args = ap.parse_args()

    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    results = ROOT / "spice/results"
    tables = ROOT / "results/tables"
    for directory in [generated, results, tables]:
        directory.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)

    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag)
    theta_values = args.theta_values or [0.05, 0.10, 0.16, 0.24, 0.34, 0.46, 0.58, 0.70, 0.82, 0.94, 1.06, 1.15]
    act_values = args.act_values or [0.25, 0.50, 0.75]
    pos_write_high_v = args.write_high_v if args.pos_write_high_v is None else args.pos_write_high_v
    pos_write_low_v = args.write_low_v if args.pos_write_low_v is None else args.pos_write_low_v
    neg_write_high_v = args.write_high_v if args.neg_write_high_v is None else args.neg_write_high_v
    neg_write_low_v = args.write_low_v if args.neg_write_low_v is None else args.neg_write_low_v
    rows: list[dict[str, Any]] = []
    pair_action_rows: list[dict[str, Any]] = []
    error_rule_action_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for act in act_values:
        for branch in ["positive", "negative"]:
            for theta in theta_values:
                measures = run_netlist(
                    spice_bin,
                    generated / f"{safe_tag}_read_{branch}_a{act:.2f}_t{theta:.2f}.cir",
                    read_branch_netlist(
                        theta,
                        act,
                        branch,
                        score_ic=args.negative_score_ic_v,
                        pos_width_u=args.pos_width_u,
                        neg_width_u=args.neg_width_u,
                        score_cap_f=args.score_cap_f,
                    ),
                    args.timeout,
                )
                rows.append(
                    {
                        "experiment": f"read_{branch}",
                        "theta": theta,
                        "act": act,
                        "pre": None,
                        "delta": None,
                        "write_width_u": None,
                        **measures,
                    }
                )

    discharge_enabled, charge_enabled = write_actions(args.write_mode)
    for action, enabled in [("discharge", discharge_enabled), ("charge", charge_enabled)]:
        if not enabled:
            continue
        for theta in theta_values:
            measures = run_netlist(
                spice_bin,
                generated / f"{safe_tag}_write_{action}_t{theta:.2f}.cir",
                write_mobility_netlist(
                    theta,
                    args.pre,
                    args.delta,
                    args.write_width_u,
                    action,
                    args.write_mode,
                    args.write_high_v,
                    args.write_low_v,
                    args.write_state_gate_mode,
                ),
                args.timeout,
            )
            rows.append(
                {
                    "experiment": f"write_{action}",
                    "theta": theta,
                    "act": None,
                    "pre": args.pre,
                    "delta": args.delta,
                    "write_width_u": args.write_width_u,
                    **measures,
                }
            )

    if args.pair_action_sweep:
        pair_theta_p_values = args.pair_theta_p_values or theta_values
        pair_theta_n_values = args.pair_theta_n_values or theta_values
        pair_act_values = args.pair_act_values or act_values
        for act in pair_act_values:
            for theta_p in pair_theta_p_values:
                for theta_n in pair_theta_n_values:
                    for signed_action in SIGNED_ACTIONS:
                        measures = run_netlist(
                            spice_bin,
                            generated / (
                                f"{safe_tag}_pair_{signed_action}_a{act:.2f}_"
                                f"tp{theta_p:.2f}_tn{theta_n:.2f}.cir"
                            ),
                            pair_action_mobility_netlist(
                                theta_p,
                                theta_n,
                                act,
                                args.pre,
                                args.delta,
                                args.write_width_u,
                                signed_action,
                                args.write_mode,
                                pos_write_high_v,
                                pos_write_low_v,
                                neg_write_high_v,
                                neg_write_low_v,
                                args.write_state_gate_mode,
                                args.pos_width_u,
                                args.neg_width_u,
                                args.negative_score_ic_v,
                                args.score_cap_f,
                            ),
                            args.timeout,
                        )
                        pair_action_rows.append(
                            {
                                "experiment": f"pair_action_{signed_action}",
                                "theta_p": theta_p,
                                "theta_n": theta_n,
                                "act": act,
                                "pre": args.pre,
                                "delta": args.delta,
                                "write_width_u": args.write_width_u,
                                "write_mode": args.write_mode,
                                "write_state_gate_mode": args.write_state_gate_mode,
                                **measures,
                            }
                        )

    if args.error_rule_action_sweep:
        action_theta_p_values = args.error_rule_action_theta_p_values or args.pair_theta_p_values or theta_values
        action_theta_n_values = args.error_rule_action_theta_n_values or args.pair_theta_n_values or theta_values
        action_act_values = args.error_rule_action_act_values or args.pair_act_values or act_values
        for error_rule in args.error_rule_action_rules:
            for act in action_act_values:
                for theta_p in action_theta_p_values:
                    for theta_n in action_theta_n_values:
                        for error_action in args.error_rule_actions:
                            measures = run_netlist(
                                spice_bin,
                                generated / (
                                    f"{safe_tag}_errail_{error_rule}_{error_action}_"
                                    f"a{act:.2f}_tp{theta_p:.2f}_tn{theta_n:.2f}.cir"
                                ),
                                error_rule_action_mobility_netlist(
                                    theta_p,
                                    theta_n,
                                    act,
                                    error_action,
                                    error_rule,
                                    args.write_mode,
                                    args.write_width_u,
                                    pos_write_high_v,
                                    pos_write_low_v,
                                    neg_write_high_v,
                                    neg_write_low_v,
                                    args.write_state_gate_mode,
                                    args.pos_width_u,
                                    args.neg_width_u,
                                    args.negative_score_ic_v,
                                    args.score_cap_f,
                                    args.error_rule_action_target_out_v,
                                    args.error_rule_action_other_out_v,
                                    args.error_rule_action_target_score_v,
                                    args.error_rule_action_other_score_v,
                                    args.error_rule_action_pre_gate_mode,
                                    args.error_rule_action_pre_boost_v,
                                    args.error_rule_action_pre_boost_cap_f,
                                    args.error_rule_action_lead_mode,
                                ),
                                args.timeout,
                            )
                            error_rule_action_rows.append(
                                {
                                    "experiment": f"error_rule_action_{error_action}",
                                    "error_rule": error_rule,
                                    "theta_p": theta_p,
                                    "theta_n": theta_n,
                                    "act": act,
                                    "error_action": error_action,
                                    "write_width_u": args.write_width_u,
                                    "write_mode": args.write_mode,
                                    "write_state_gate_mode": args.write_state_gate_mode,
                                    "lead_mode": args.error_rule_action_lead_mode,
                                    "pre_gate_mode": args.error_rule_action_pre_gate_mode,
                                    "pre_boost_v": (
                                        args.error_rule_action_pre_boost_v
                                        if args.error_rule_action_pre_gate_mode == "boosted"
                                        else None
                                    ),
                                    "pre_boost_cap_f": (
                                        args.error_rule_action_pre_boost_cap_f
                                        if args.error_rule_action_pre_gate_mode == "boosted"
                                        else None
                                    ),
                                    **measures,
                                }
                            )

    df = add_slopes(pd.DataFrame(rows))
    write = df[df["experiment"] == "write_discharge"][["theta", "state_delta_v"]].rename(
        columns={"state_delta_v": "write_discharge_v"}
    )
    if not write.empty:
        df = df.merge(write, on="theta", how="left")
    else:
        df["write_discharge_v"] = np.nan
    df["effective_mobility"] = df["read_slope"] * df["write_discharge_v"]
    signed = signed_mobility_table(df, args.write_mode, args.write_low_v, args.write_high_v)
    pair_action = pd.DataFrame(pair_action_rows)
    if not pair_action.empty:
        pair_action["pair_action_sign_aligned"] = pair_action["desired_signed_read_delta"] > 0
    error_rule_action = pd.DataFrame(error_rule_action_rows)
    if not error_rule_action.empty:
        error_rule_action["row0_action_sign_aligned"] = (
            error_rule_action["row0_desired_signed_read_delta"] > 0
        )
        error_rule_action["row1_action_sign_aligned"] = (
            error_rule_action["row1_desired_signed_read_delta"] > 0
        )
        error_rule_action["both_rows_action_sign_aligned"] = (
            error_rule_action["row0_action_sign_aligned"] & error_rule_action["row1_action_sign_aligned"]
        )

    csv_path = results / f"{safe_tag}.csv"
    table_path = tables / f"{safe_tag}.csv"
    signed_csv_path = results / f"{safe_tag}_signed_mobility.csv"
    signed_table_path = tables / f"{safe_tag}_signed_mobility.csv"
    pair_action_csv_path = results / f"{safe_tag}_pair_action_mobility.csv"
    pair_action_table_path = tables / f"{safe_tag}_pair_action_mobility.csv"
    error_rule_action_csv_path = results / f"{safe_tag}_error_rule_action_mobility.csv"
    error_rule_action_table_path = tables / f"{safe_tag}_error_rule_action_mobility.csv"
    df.to_csv(csv_path, index=False)
    df.to_csv(table_path, index=False)
    signed.to_csv(signed_csv_path, index=False)
    signed.to_csv(signed_table_path, index=False)
    if not pair_action.empty:
        pair_action.to_csv(pair_action_csv_path, index=False)
        pair_action.to_csv(pair_action_table_path, index=False)
    if not error_rule_action.empty:
        error_rule_action.to_csv(error_rule_action_csv_path, index=False)
        error_rule_action.to_csv(error_rule_action_table_path, index=False)

    read_rows = df[df["experiment"].str.startswith("read_")]
    write_rows = df[df["experiment"].str.startswith("write_")]
    signed_summary = summarize_signed_mobility(signed, args.write_mode, args.write_low_v, args.write_high_v)
    signed_summary["signed_mobility_csv"] = str(signed_csv_path)
    signed_summary["signed_mobility_table_csv"] = str(signed_table_path)
    pair_action_summary = summarize_pair_action_mobility(pair_action)
    if not pair_action.empty:
        pair_action_summary["pair_action_mobility_csv"] = str(pair_action_csv_path)
        pair_action_summary["pair_action_mobility_table_csv"] = str(pair_action_table_path)
    error_rule_action_summary = summarize_error_rule_action_mobility(error_rule_action)
    if not error_rule_action.empty:
        error_rule_action_summary["error_rule_action_mobility_csv"] = str(error_rule_action_csv_path)
        error_rule_action_summary["error_rule_action_mobility_table_csv"] = str(error_rule_action_table_path)
    summary = {
        "tag": safe_tag,
        "simulator": version,
        "model_level": "ngspice built-in LEVEL=1 MOS models; not a foundry PDK.",
        "csv": str(csv_path),
        "table_csv": str(table_path),
        "theta_values": theta_values,
        "act_values": act_values,
        "write_mode": args.write_mode,
        "write_state_gate_mode": args.write_state_gate_mode,
        "write_width_u": args.write_width_u,
        "write_high_v": args.write_high_v if args.write_mode.startswith("bounded_") else None,
        "write_low_v": args.write_low_v if args.write_mode.startswith("bounded_") else None,
        "pos_write_high_v": pos_write_high_v if args.write_mode.startswith("bounded_") else None,
        "pos_write_low_v": pos_write_low_v if args.write_mode.startswith("bounded_") else None,
        "neg_write_high_v": neg_write_high_v if args.write_mode.startswith("bounded_") else None,
        "neg_write_low_v": neg_write_low_v if args.write_mode.startswith("bounded_") else None,
        "write_pre_v": args.pre,
        "write_delta_v": args.delta,
        "read_positive_width_u": args.pos_width_u,
        "read_negative_width_u": args.neg_width_u,
        "negative_read_score_initial_v": args.negative_score_ic_v,
        "score_cap_f": args.score_cap_f,
        "error_rule_action_lead_mode": args.error_rule_action_lead_mode
        if args.error_rule_action_sweep
        else None,
        "min_read_slope": float(read_rows["read_slope"].min()),
        "max_read_slope": float(read_rows["read_slope"].max()),
        "min_write_state_delta_v": float(write_rows["state_delta_v"].min()),
        "max_write_state_delta_v": float(write_rows["state_delta_v"].max()),
        "min_write_discharge_v": float(df["write_discharge_v"].min(skipna=True))
        if "write_discharge_v" in df
        else None,
        "max_write_discharge_v": float(df["write_discharge_v"].max(skipna=True))
        if "write_discharge_v" in df
        else None,
        "min_effective_mobility": float(read_rows["effective_mobility"].min()),
        "max_effective_mobility": float(read_rows["effective_mobility"].max()),
        **signed_summary,
        **pair_action_summary,
        **error_rule_action_summary,
        "wall_time_s": time.perf_counter() - t0,
        "interpretation": (
            "Read-response slopes approximate G_eff'(theta); write state deltas approximate the natural "
            "direct-flow state mobility s_MOS(theta). The signed mobility table combines positive- and "
            "negative-branch read slopes with the enabled write actions to estimate the effective-weight "
            "mobility for desired positive and negative readout updates."
        ),
    }
    summary_path = results / f"{safe_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
