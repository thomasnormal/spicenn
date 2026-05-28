from __future__ import annotations

import argparse
import csv
import json
import time
from typing import Any

import numpy as np

from datasets import dataset_records, parse_counted_mnist_dataset
from parameter_theory import derive_multiclass_margin_correction_sizing
from run_device_mnist01_scalar_training import sanitize_tag
from run_device_sequential_training import mos_models, run_netlist
from run_multiclass_output_head_primitive import (
    class_local_bounded_update_lines,
    class_local_label_descent_gradient_lines,
    class_local_readout_forward_lines,
    class_node,
    signed_store_lines,
)
from run_normalization_subcircuits import APPROACHES as NORMALIZER_APPROACHES
from run_normalization_subcircuits import normalization_subcircuits, spice_subckt_name
from run_score_decision_primitive import low_gain_preamp_lines, low_gain_ref_decision_lines, low_gain_ref_state_lines
from run_spice_sweep import ROOT, detect_spice


CYCLE_NS = 16.0
NORMALIZER_ERROR_MODES = tuple(f"normalizer-{approach}-descent" for approach in NORMALIZER_APPROACHES)
ERROR_MODES = (
    "label-descent",
    "score-gated-nontarget",
    "restored-score-nontarget",
    "live-pairwise-margin-descent",
    *NORMALIZER_ERROR_MODES,
)
WRITER_MODES = ("sampled", "live")


def pwl(points: list[tuple[float, float]]) -> str:
    merged: list[tuple[float, float]] = []
    for time_ns, value in points:
        if merged and abs(merged[-1][0] - time_ns) < 1e-15:
            merged[-1] = (time_ns, value)
        else:
            merged.append((time_ns, value))
    return "PWL(" + " ".join(f"{time_ns:.6g}n {value:.12g}" for time_ns, value in merged) + ")"


def windowed_pwl(
    cycle_values: list[float],
    *,
    start_ns: float,
    end_ns: float,
    cycle_ns: float = CYCLE_NS,
) -> str:
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for cycle, value in enumerate(cycle_values):
        base = cycle * cycle_ns
        points += [
            (base + start_ns - 0.01, 0.0),
            (base + start_ns, value),
            (base + end_ns, value),
            (base + end_ns + 0.01, 0.0),
        ]
    points.append((len(cycle_values) * cycle_ns, 0.0))
    return pwl(points)


def width_scaled_windowed_pwl(
    cycle_values: list[float],
    *,
    start_ns: float,
    end_ns: float,
    width_scale: float,
    cycle_ns: float = CYCLE_NS,
) -> str:
    if width_scale < 0.0 or width_scale > 1.0:
        raise ValueError("width_scale must be in [0, 1]")
    if width_scale >= 1.0:
        return windowed_pwl(cycle_values, start_ns=start_ns, end_ns=end_ns, cycle_ns=cycle_ns)
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    scaled_end_ns = start_ns + (end_ns - start_ns) * width_scale
    for cycle, value in enumerate(cycle_values):
        base = cycle * cycle_ns
        if abs(value) < 1e-15 or scaled_end_ns <= start_ns:
            points += [(base + start_ns - 0.01, 0.0), (base + end_ns + 0.01, 0.0)]
            continue
        points += [
            (base + start_ns - 0.01, 0.0),
            (base + start_ns, value),
            (base + scaled_end_ns, value),
            (base + scaled_end_ns + 0.01, 0.0),
            (base + end_ns + 0.01, 0.0),
        ]
    points.append((len(cycle_values) * cycle_ns, 0.0))
    return pwl(points)


def multi_windowed_pwl(
    cycle_values: list[float],
    *,
    windows: list[tuple[float, float]],
    cycle_ns: float = CYCLE_NS,
) -> str:
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for cycle, value in enumerate(cycle_values):
        base = cycle * cycle_ns
        for start_ns, end_ns in windows:
            points += [
                (base + start_ns - 0.01, 0.0),
                (base + start_ns, value),
                (base + end_ns, value),
                (base + end_ns + 0.01, 0.0),
            ]
    points.append((len(cycle_values) * cycle_ns, 0.0))
    return pwl(points)


def multi_window_phase_pwl(
    cycle_count: int,
    *,
    windows: list[tuple[float, float]],
    high: float = 1.2,
    cycle_ns: float = CYCLE_NS,
) -> str:
    return multi_windowed_pwl([high] * cycle_count, windows=windows, cycle_ns=cycle_ns)


def periodic_phase_pwl(
    cycle_count: int,
    *,
    start_ns: float,
    end_ns: float,
    active_cycles: set[int] | None = None,
    high: float = 1.2,
    cycle_ns: float = CYCLE_NS,
) -> str:
    values = [high if active_cycles is None or cycle in active_cycles else 0.0 for cycle in range(cycle_count)]
    return windowed_pwl(values, start_ns=start_ns, end_ns=end_ns, cycle_ns=cycle_ns)


def active_low_phase_pwl(
    cycle_count: int,
    *,
    start_ns: float,
    end_ns: float,
    active_cycles: set[int],
    cycle_ns: float = CYCLE_NS,
) -> str:
    points: list[tuple[float, float]] = [(0.0, 1.2)]
    for cycle in range(cycle_count):
        base = cycle * cycle_ns
        if cycle in active_cycles:
            points += [
                (base + start_ns - 0.01, 1.2),
                (base + start_ns, 0.0),
                (base + end_ns, 0.0),
                (base + end_ns + 0.01, 1.2),
            ]
        else:
            points += [(base, 1.2), (base + cycle_ns, 1.2)]
    points.append((cycle_count * cycle_ns, 1.2))
    return pwl(points)


def records_to_feature_matrix(records: list[dict[str, Any]], feature_count: int) -> np.ndarray:
    if feature_count <= 0:
        raise ValueError("feature_count must be positive")
    matrix = []
    for record in records:
        inputs = record.get("inputs")
        if not isinstance(inputs, dict):
            raise ValueError("records must contain an inputs dictionary")
        matrix.append([float(inputs[f"x{feature}"]) for feature in range(feature_count)])
    return np.asarray(matrix, dtype=float)


def balanced_train_eval_split(
    records: list[dict[str, Any]],
    *,
    class_count: int,
    train_samples: int,
    eval_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if train_samples % class_count != 0 or eval_samples % class_count != 0:
        raise ValueError("train_samples and eval_samples must be divisible by class_count")
    train_per_class = train_samples // class_count
    eval_per_class = eval_samples // class_count
    train: list[dict[str, Any]] = []
    eval_: list[dict[str, Any]] = []
    for class_idx in range(class_count):
        class_records = [record for record in records if int(record["label"]) == class_idx]
        needed = train_per_class + eval_per_class
        if len(class_records) < needed:
            raise ValueError(f"not enough records for class {class_idx}: need {needed}, got {len(class_records)}")
        train.extend(class_records[:train_per_class])
        eval_.extend(class_records[train_per_class:needed])
    return train, eval_


def pairwise_decision_node(class_idx: int, opponent_idx: int) -> str:
    return f"c{class_idx}_gt_c{opponent_idx}_decision"


def pairwise_low_gain_winner_lines(
    *,
    class_a: int,
    class_b: int,
    amp_clock_node: str = "scoreamp",
    decision_clock_node: str = "scoredec",
    reset_node: str = "scorepre",
    pullup_width: float = 16.0,
    pulldown_width: float = 64.0,
    gain_input_width: float = 1.0,
    gain_tail_width: float = 8.0,
    gain_capacitance_f: float = 8.0,
) -> list[str]:
    prefix = f"c{class_a}_gt_c{class_b}_"
    node_ab = pairwise_decision_node(class_a, class_b)
    node_ba = pairwise_decision_node(class_b, class_a)
    dec_src = f"{prefix}dec_src"
    return [
        f"C{node_ab} {node_ab} 0 4f IC=0",
        f"C{node_ba} {node_ba} 0 4f IC=0",
        f"R{node_ab} {node_ab} 0 1G",
        f"R{node_ba} {node_ba} 0 1G",
        f"Mprecharge_{node_ab} {node_ab} {reset_node} vdd vdd PMOS W=4u L=180n",
        f"Mprecharge_{node_ba} {node_ba} {reset_node} vdd vdd PMOS W=4u L=180n",
        *low_gain_ref_state_lines(
            prefix=prefix,
            gain_capacitance_f=gain_capacitance_f,
            reset_node=reset_node,
        ),
        *low_gain_preamp_lines(
            prefix=prefix,
            score_node=class_node(class_a, "score"),
            scoren_node=class_node(class_b, "score"),
            amp_clock_node=amp_clock_node,
            gain_input_width=gain_input_width,
            gain_tail_width=gain_tail_width,
        ),
        f"M{prefix}dec_pair_p {node_ab} {node_ba} vdd vdd PMOS W={pullup_width:.6g}u L=180n",
        f"M{prefix}decn_pair_p {node_ba} {node_ab} vdd vdd PMOS W={pullup_width:.6g}u L=180n",
        f"M{prefix}dec_pair_n {node_ab} {prefix}scoren_amp {dec_src} 0 NSENSE W={pulldown_width:.6g}u L=180n",
        f"M{prefix}decn_pair_n {node_ba} {prefix}score_amp {dec_src} 0 NSENSE W={pulldown_width:.6g}u L=180n",
        f"M{prefix}dec_pair_tail {dec_src} {decision_clock_node} 0 0 NMOS W={pulldown_width:.6g}u L=180n",
    ]


def pairwise_target_margin_penalty_lines(
    *,
    class_count: int,
    decision_clock_node: str = "scoredec",
    penalty_width_u: float = 0.25,
) -> list[str]:
    if penalty_width_u <= 0.0:
        raise ValueError("penalty_width_u must be positive")
    lines: list[str] = []
    for target_idx in range(class_count):
        for opponent_idx in range(class_count):
            if opponent_idx == target_idx:
                continue
            target_wins = pairwise_decision_node(target_idx, opponent_idx)
            targetp = class_node(target_idx, "targetp")
            prefix = f"mpen_t{target_idx}_o{opponent_idx}_"
            lines += [
                f"R{prefix}i {prefix}i 0 1G",
                f"M{prefix}label {target_wins} {targetp} {prefix}i 0 NSENSE W={penalty_width_u:.6g}u L=180n",
                f"M{prefix}clk {prefix}i {decision_clock_node} 0 0 NMOS W={penalty_width_u:.6g}u L=180n",
            ]
    return lines


def class_local_score_gated_nontarget_gradient_lines(
    *,
    class_idx: int,
    feature_idx: int,
    activation_node: str,
    width_u: float = 24.0,
) -> list[str]:
    prefix = f"c{class_idx}_f{feature_idx}_"
    return [
        f"M{prefix}gvp_a vdd {activation_node} {prefix}gvp_a 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}gvp_d {prefix}gvp_a {class_node(class_idx, 'targetp')} {prefix}gvp_d 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}gvp_g {prefix}gvp_d acc {class_node(class_idx, f'gvp{feature_idx}')} 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_a vdd {activation_node} {prefix}gvn_a 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_label {prefix}gvn_a {class_node(class_idx, 'targetn')} {prefix}gvn_label 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_score {prefix}gvn_label {class_node(class_idx, 'score')} {prefix}gvn_d 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_g {prefix}gvn_d acc {class_node(class_idx, f'gvn{feature_idx}')} 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}rgp_pd {class_node(class_idx, f'rgp{feature_idx}')} {class_node(class_idx, f'gvp{feature_idx}')} 0 0 NSENSE W=16u L=180n",
        f"M{prefix}rgn_pd {class_node(class_idx, f'rgn{feature_idx}')} {class_node(class_idx, f'gvn{feature_idx}')} 0 0 NSENSE W=16u L=180n",
    ]


def class_local_restored_score_nontarget_gradient_lines(
    *,
    class_idx: int,
    feature_idx: int,
    activation_node: str,
    score_gate_node: str,
    width_u: float = 24.0,
) -> list[str]:
    prefix = f"c{class_idx}_f{feature_idx}_"
    return [
        f"M{prefix}gvp_a vdd {activation_node} {prefix}gvp_a 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}gvp_d {prefix}gvp_a {class_node(class_idx, 'targetp')} {prefix}gvp_d 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}gvp_g {prefix}gvp_d acc {class_node(class_idx, f'gvp{feature_idx}')} 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_a vdd {activation_node} {prefix}gvn_a 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_label {prefix}gvn_a {class_node(class_idx, 'targetn')} {prefix}gvn_label 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_score {prefix}gvn_label {score_gate_node} {prefix}gvn_d 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_g {prefix}gvn_d acc {class_node(class_idx, f'gvn{feature_idx}')} 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}rgp_pd {class_node(class_idx, f'rgp{feature_idx}')} {class_node(class_idx, f'gvp{feature_idx}')} 0 0 NSENSE W=16u L=180n",
        f"M{prefix}rgn_pd {class_node(class_idx, f'rgn{feature_idx}')} {class_node(class_idx, f'gvn{feature_idx}')} 0 0 NSENSE W=16u L=180n",
    ]


def class_local_live_label_descent_update_lines(
    *,
    class_idx: int,
    feature_idx: int,
    activation_node: str,
    positive_descent_node: str | None = None,
    negative_descent_node: str | None = None,
    nontarget_guard_node: str | None = None,
    update_guard_node: str | None = None,
    width_u: float = 0.5,
    stack_shunt_resistance_ohm: float = 1.0e9,
    stack_parasitic_capacitance_f: float = 0.05,
    high_side_topology: str = "nmos-stack",
) -> list[str]:
    if high_side_topology not in ("nmos-stack", "pmos-gated", "pmos-differential"):
        raise ValueError("high_side_topology must be nmos-stack, pmos-gated, or pmos-differential")
    if nontarget_guard_node is not None and update_guard_node is not None:
        raise ValueError("nontarget_guard_node and update_guard_node are mutually exclusive")
    if stack_shunt_resistance_ohm <= 0.0:
        raise ValueError("stack_shunt_resistance_ohm must be positive")
    if stack_parasitic_capacitance_f <= 0.0:
        raise ValueError("stack_parasitic_capacitance_f must be positive")
    prefix = f"c{class_idx}_f{feature_idx}_live_"
    vwp = class_node(class_idx, f"vwp{feature_idx}")
    vwn = class_node(class_idx, f"vwn{feature_idx}")
    pos = class_node(class_idx, "targetp") if positive_descent_node is None else positive_descent_node
    neg = class_node(class_idx, "targetn") if negative_descent_node is None else negative_descent_node
    lines = [
        f"R{prefix}pos_up_shunt {prefix}pos_up 0 {stack_shunt_resistance_ohm:.12g}",
        f"R{prefix}pos_dn_shunt {prefix}pos_dn 0 {stack_shunt_resistance_ohm:.12g}",
        f"R{prefix}neg_up_shunt {prefix}neg_up 0 {stack_shunt_resistance_ohm:.12g}",
        f"R{prefix}neg_dn_shunt {prefix}neg_dn 0 {stack_shunt_resistance_ohm:.12g}",
        f"C{prefix}pos_up_par {prefix}pos_up 0 {stack_parasitic_capacitance_f:.12g}f IC=0",
        f"C{prefix}pos_dn_par {prefix}pos_dn 0 {stack_parasitic_capacitance_f:.12g}f IC=0",
        f"C{prefix}neg_up_par {prefix}neg_up 0 {stack_parasitic_capacitance_f:.12g}f IC=0",
        f"C{prefix}neg_dn_par {prefix}neg_dn 0 {stack_parasitic_capacitance_f:.12g}f IC=0",
        f"M{prefix}pos_dn_d {prefix}pos_dn {pos} vwlo_ref 0 NSENSE W={width_u:.6g}u L=180n",
    ]
    if update_guard_node is None:
        lines += [
            f"M{prefix}pos_dn_e {vwn} {activation_node} {prefix}pos_dn 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}neg_dn_e {vwp} {activation_node} {prefix}neg_dn 0 NSENSE W={width_u:.6g}u L=180n",
        ]
    else:
        lines += [
            f"R{prefix}pos_dn_allguard_shunt {prefix}pos_dn_allguard 0 {stack_shunt_resistance_ohm:.12g}",
            f"R{prefix}neg_dn_allguard_shunt {prefix}neg_dn_allguard 0 {stack_shunt_resistance_ohm:.12g}",
            f"C{prefix}pos_dn_allguard_par {prefix}pos_dn_allguard 0 {stack_parasitic_capacitance_f:.12g}f IC=0",
            f"C{prefix}neg_dn_allguard_par {prefix}neg_dn_allguard 0 {stack_parasitic_capacitance_f:.12g}f IC=0",
            f"M{prefix}pos_dn_e {vwn} {activation_node} {prefix}pos_dn_allguard 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}pos_dn_g {prefix}pos_dn_allguard {update_guard_node} {prefix}pos_dn 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}neg_dn_e {vwp} {activation_node} {prefix}neg_dn_allguard 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}neg_dn_g {prefix}neg_dn_allguard {update_guard_node} {prefix}neg_dn 0 NSENSE W={width_u:.6g}u L=180n",
        ]
    if high_side_topology == "nmos-stack":
        if update_guard_node is None:
            lines += [
                f"M{prefix}pos_up_e vwhi_ref {activation_node} {prefix}pos_up 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}pos_up_d {prefix}pos_up {pos} {vwp} 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}neg_up_e vwhi_ref {activation_node} {prefix}neg_up 0 NSENSE W={width_u:.6g}u L=180n",
            ]
        else:
            lines += [
                f"R{prefix}pos_up_allguard_shunt {prefix}pos_up_allguard 0 {stack_shunt_resistance_ohm:.12g}",
                f"R{prefix}neg_up_allguard_shunt {prefix}neg_up_allguard 0 {stack_shunt_resistance_ohm:.12g}",
                f"C{prefix}pos_up_allguard_par {prefix}pos_up_allguard 0 {stack_parasitic_capacitance_f:.12g}f IC=0",
                f"C{prefix}neg_up_allguard_par {prefix}neg_up_allguard 0 {stack_parasitic_capacitance_f:.12g}f IC=0",
                f"M{prefix}pos_up_e vwhi_ref {activation_node} {prefix}pos_up_allguard 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}pos_up_g {prefix}pos_up_allguard {update_guard_node} {prefix}pos_up 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}pos_up_d {prefix}pos_up {pos} {vwp} 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}neg_up_e vwhi_ref {activation_node} {prefix}neg_up_allguard 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}neg_up_g {prefix}neg_up_allguard {update_guard_node} {prefix}neg_up 0 NSENSE W={width_u:.6g}u L=180n",
            ]
    else:
        pos_ctrl = f"{prefix}pos_up_ctrl"
        neg_ctrl = f"{prefix}neg_up_ctrl"
        pos_ctrl_mid = f"{prefix}pos_up_ctrl_mid"
        neg_ctrl_mid = f"{prefix}neg_up_ctrl_mid"
        pmos_width_u = 2.0 * width_u
        ctrl_keeper_resistance = 1.0e6 if high_side_topology == "pmos-differential" else 50000
        lines += [
            f"C{pos_ctrl} {pos_ctrl} 0 2f IC=1.2",
            f"R{pos_ctrl} {pos_ctrl} vdd {ctrl_keeper_resistance:.12g}",
            f"R{pos_ctrl_mid} {pos_ctrl_mid} 0 {stack_shunt_resistance_ohm:.12g}",
            f"C{pos_ctrl_mid} {pos_ctrl_mid} 0 {stack_parasitic_capacitance_f:.12g}f IC=0",
            f"M{prefix}pos_up_p {vwp} {pos_ctrl} vwhi_ref vdd PMOS W={pmos_width_u:.6g}u L=180n",
            f"C{neg_ctrl} {neg_ctrl} 0 2f IC=1.2",
            f"R{neg_ctrl} {neg_ctrl} vdd {ctrl_keeper_resistance:.12g}",
            f"R{neg_ctrl_mid} {neg_ctrl_mid} 0 {stack_shunt_resistance_ohm:.12g}",
            f"C{neg_ctrl_mid} {neg_ctrl_mid} 0 {stack_parasitic_capacitance_f:.12g}f IC=0",
        ]
        if update_guard_node is None:
            lines += [
                f"M{prefix}pos_up_ctrl_e {pos_ctrl} {activation_node} {pos_ctrl_mid} 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}pos_up_ctrl_d {pos_ctrl_mid} {pos} 0 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}neg_up_ctrl_e {neg_ctrl} {activation_node} {neg_ctrl_mid} 0 NSENSE W={width_u:.6g}u L=180n",
            ]
        else:
            pos_ctrl_guard = f"{prefix}pos_up_ctrl_allguard"
            neg_ctrl_guard = f"{prefix}neg_up_ctrl_allguard"
            lines += [
                f"R{pos_ctrl_guard} {pos_ctrl_guard} 0 {stack_shunt_resistance_ohm:.12g}",
                f"C{pos_ctrl_guard} {pos_ctrl_guard} 0 {stack_parasitic_capacitance_f:.12g}f IC=0",
                f"M{prefix}pos_up_ctrl_e {pos_ctrl} {activation_node} {pos_ctrl_guard} 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}pos_up_ctrl_g {pos_ctrl_guard} {update_guard_node} {pos_ctrl_mid} 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}pos_up_ctrl_d {pos_ctrl_mid} {pos} 0 0 NSENSE W={width_u:.6g}u L=180n",
                f"R{neg_ctrl_guard} {neg_ctrl_guard} 0 {stack_shunt_resistance_ohm:.12g}",
                f"C{neg_ctrl_guard} {neg_ctrl_guard} 0 {stack_parasitic_capacitance_f:.12g}f IC=0",
                f"M{prefix}neg_up_ctrl_e {neg_ctrl} {activation_node} {neg_ctrl_guard} 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}neg_up_ctrl_g {neg_ctrl_guard} {update_guard_node} {neg_ctrl_mid} 0 NSENSE W={width_u:.6g}u L=180n",
            ]
        if high_side_topology == "pmos-differential":
            lines += [
                f"M{prefix}pos_up_ctrl_latch {pos_ctrl} {neg_ctrl} vdd vdd PMOS W={pmos_width_u:.6g}u L=180n",
                f"M{prefix}neg_up_ctrl_latch {neg_ctrl} {pos_ctrl} vdd vdd PMOS W={pmos_width_u:.6g}u L=180n",
            ]
    if nontarget_guard_node is None:
        if high_side_topology == "nmos-stack":
            lines.append(f"M{prefix}neg_up_d {prefix}neg_up {neg} {vwn} 0 NSENSE W={width_u:.6g}u L=180n")
        else:
            neg_ctrl = f"{prefix}neg_up_ctrl"
            neg_ctrl_mid = f"{prefix}neg_up_ctrl_mid"
            pmos_width_u = 2.0 * width_u
            lines += [
                f"M{prefix}neg_up_ctrl_d {neg_ctrl_mid} {neg} 0 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}neg_up_p {vwn} {neg_ctrl} vwhi_ref vdd PMOS W={pmos_width_u:.6g}u L=180n",
            ]
        lines.append(f"M{prefix}neg_dn_d {prefix}neg_dn {neg} vwlo_ref 0 NSENSE W={width_u:.6g}u L=180n")
    else:
        if high_side_topology == "nmos-stack":
            neg_up_lines = [
                f"M{prefix}neg_up_g {prefix}neg_up {nontarget_guard_node} {prefix}neg_up_guard 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}neg_up_d {prefix}neg_up_guard {neg} {vwn} 0 NSENSE W={width_u:.6g}u L=180n",
            ]
        else:
            neg_ctrl = f"{prefix}neg_up_ctrl"
            neg_ctrl_mid = f"{prefix}neg_up_ctrl_mid"
            neg_ctrl_guard = f"{prefix}neg_up_ctrl_guard"
            pmos_width_u = 2.0 * width_u
            neg_up_lines = [
                f"R{neg_ctrl_guard} {neg_ctrl_guard} 0 {stack_shunt_resistance_ohm:.12g}",
                f"C{neg_ctrl_guard} {neg_ctrl_guard} 0 {stack_parasitic_capacitance_f:.12g}f IC=0",
                f"M{prefix}neg_up_ctrl_g {neg_ctrl_mid} {nontarget_guard_node} {neg_ctrl_guard} 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}neg_up_ctrl_d {neg_ctrl_guard} {neg} 0 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}neg_up_p {vwn} {neg_ctrl} vwhi_ref vdd PMOS W={pmos_width_u:.6g}u L=180n",
            ]
        lines += [
            f"R{prefix}neg_up_guard_shunt {prefix}neg_up_guard 0 {stack_shunt_resistance_ohm:.12g}",
            f"R{prefix}neg_dn_guard_shunt {prefix}neg_dn_guard 0 {stack_shunt_resistance_ohm:.12g}",
            f"C{prefix}neg_up_guard_par {prefix}neg_up_guard 0 {stack_parasitic_capacitance_f:.12g}f IC=0",
            f"C{prefix}neg_dn_guard_par {prefix}neg_dn_guard 0 {stack_parasitic_capacitance_f:.12g}f IC=0",
            *neg_up_lines,
            f"M{prefix}neg_dn_g {prefix}neg_dn {nontarget_guard_node} {prefix}neg_dn_guard 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}neg_dn_d {prefix}neg_dn_guard {neg} vwlo_ref 0 NSENSE W={width_u:.6g}u L=180n",
        ]
    return lines


def class_local_support_storage_lines(
    *,
    class_idx: int,
    feature_idx: int,
    activation_node: str,
    positive_descent_node: str | None = None,
    capacitance_f: float = 4.0,
    width_u: float = 0.5,
) -> list[str]:
    if min(capacitance_f, width_u) <= 0.0:
        raise ValueError("support storage sizes must be positive")
    support = class_node(class_idx, f"f{feature_idx}_support")
    pos = class_node(class_idx, "targetp") if positive_descent_node is None else positive_descent_node
    return [
        f"C{support} {support} 0 {capacitance_f:.12g}f IC=0",
        f"R{support} {support} 0 1G",
        f"M{support}_e vwhi_ref {activation_node} {support}_mid 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{support}_d {support}_mid {pos} {support} 0 NSENSE W={width_u:.6g}u L=180n",
    ]


def normalizer_approach_from_error_mode(error_mode: str) -> str | None:
    if not error_mode.startswith("normalizer-") or not error_mode.endswith("-descent"):
        return None
    approach = error_mode.removeprefix("normalizer-").removesuffix("-descent")
    if approach not in NORMALIZER_APPROACHES:
        raise ValueError(f"normalizer approach must be one of {NORMALIZER_APPROACHES}")
    return approach


def pairwise_live_margin_error_lines(
    *,
    class_count: int,
    error_clock_node: str,
    reset_node: str,
    error_width_u: float,
    error_capacitance_f: float,
) -> list[str]:
    if min(error_width_u, error_capacitance_f) <= 0.0:
        raise ValueError("pairwise live margin error sizes must be positive")
    lines: list[str] = []
    for class_idx in range(class_count):
        errp = class_node(class_idx, "errp")
        errn = class_node(class_idx, "errn")
        lines += [
            f"C{errp} {errp} 0 {error_capacitance_f:.12g}f IC=0",
            f"C{errn} {errn} 0 {error_capacitance_f:.12g}f IC=0",
            f"R{errp} {errp} 0 1G",
            f"R{errn} {errn} 0 1G",
            f"Mreset_{errp} {errp} {reset_node} 0 0 NMOS W=4u L=180n",
            f"Mreset_{errn} {errn} {reset_node} 0 0 NMOS W=4u L=180n",
        ]
    for target_idx in range(class_count):
        for opponent_idx in range(class_count):
            if opponent_idx == target_idx:
                continue
            decision = pairwise_decision_node(opponent_idx, target_idx)
            opposite_decision = pairwise_decision_node(target_idx, opponent_idx)
            errp = class_node(target_idx, "errp")
            errn = class_node(opponent_idx, "errn")
            targetp = class_node(target_idx, "targetp")
            prefix = f"live_pm_t{target_idx}_o{opponent_idx}_"
            lines += [
                f"R{prefix}errp_sup {prefix}errp_sup 0 1G",
                f"R{prefix}errp_t {prefix}errp_t 0 1G",
                f"R{prefix}errp_w {prefix}errp_w 0 1G",
                f"M{prefix}errp_sup {prefix}errp_sup {opposite_decision} vdd vdd PMOS W={error_width_u:.6g}u L=180n",
                f"M{prefix}errp_label {prefix}errp_sup {targetp} {prefix}errp_t 0 NSENSE W={error_width_u:.6g}u L=180n",
                f"M{prefix}errp_win {prefix}errp_t {decision} {prefix}errp_w 0 NSENSE W={error_width_u:.6g}u L=180n",
                f"M{prefix}errp_clk {prefix}errp_w {error_clock_node} {errp} 0 NSENSE W={error_width_u:.6g}u L=180n",
                f"R{prefix}errn_sup {prefix}errn_sup 0 1G",
                f"R{prefix}errn_t {prefix}errn_t 0 1G",
                f"R{prefix}errn_w {prefix}errn_w 0 1G",
                f"M{prefix}errn_sup {prefix}errn_sup {opposite_decision} vdd vdd PMOS W={error_width_u:.6g}u L=180n",
                f"M{prefix}errn_label {prefix}errn_sup {targetp} {prefix}errn_t 0 NSENSE W={error_width_u:.6g}u L=180n",
                f"M{prefix}errn_win {prefix}errn_t {decision} {prefix}errn_w 0 NSENSE W={error_width_u:.6g}u L=180n",
                f"M{prefix}errn_clk {prefix}errn_w {error_clock_node} {errn} 0 NSENSE W={error_width_u:.6g}u L=180n",
            ]
    return lines


def generate_netlist(
    *,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    class_count: int,
    feature_count: int,
    initial_positive: float = 0.36,
    initial_negative: float = 0.34,
    target_high: float = 1.1,
    nontarget_scale: float = 0.5,
    nontarget_width_scale: float = 1.0,
    error_mode: str = "label-descent",
    writer_mode: str = "sampled",
    readout_center_resistance: float = 0.0,
    readout_center_voltage: float = 0.40,
) -> str:
    if class_count < 2:
        raise ValueError("class_count must be at least 2")
    if not train_records or not eval_records:
        raise ValueError("train_records and eval_records must be nonempty")
    if min(initial_positive, initial_negative, target_high) < 0.0:
        raise ValueError("voltages must be nonnegative")
    if max(initial_positive, initial_negative, target_high) > 1.2:
        raise ValueError("voltages must stay within supply rails")
    if nontarget_scale < 0.0 or nontarget_scale > 1.0:
        raise ValueError("nontarget_scale must be in [0, 1]")
    if nontarget_width_scale < 0.0 or nontarget_width_scale > 1.0:
        raise ValueError("nontarget_width_scale must be in [0, 1]")
    if readout_center_resistance < 0.0:
        raise ValueError("readout_center_resistance must be nonnegative")
    if readout_center_voltage < 0.0 or readout_center_voltage > 1.2:
        raise ValueError("readout_center_voltage must stay within supply rails")
    if error_mode not in ERROR_MODES:
        raise ValueError(f"error_mode must be one of {ERROR_MODES}")
    if writer_mode not in WRITER_MODES:
        raise ValueError(f"writer_mode must be one of {WRITER_MODES}")
    normalizer_approach = normalizer_approach_from_error_mode(error_mode)
    uses_pairwise_margin = error_mode == "live-pairwise-margin-descent"
    if writer_mode == "live" and error_mode != "label-descent" and normalizer_approach is None and not uses_pairwise_margin:
        raise ValueError("live writer_mode currently supports label-descent, pairwise-margin, and normalizer descent modes only")
    if normalizer_approach is not None and (writer_mode != "live" or class_count != 3):
        raise ValueError("normalizer descent currently requires live writer_mode and class_count=3")
    if uses_pairwise_margin and (writer_mode != "live" or class_count != 3):
        raise ValueError("live pairwise-margin descent currently requires live writer_mode and class_count=3")

    all_records = eval_records + train_records + eval_records
    sequence = ["initial_eval"] * len(eval_records) + ["train"] * len(train_records) + ["final_eval"] * len(eval_records)
    train_cycles = {idx for idx, label in enumerate(sequence) if label == "train"}
    labels = [int(record["label"]) for record in all_records]
    if any(label < 0 or label >= class_count for label in labels):
        raise ValueError("record labels must be valid class indices")
    features = records_to_feature_matrix(all_records, feature_count)
    cycle_count = len(all_records)
    stop_ns = cycle_count * CYCLE_NS
    uses_early_score = (
        error_mode in {"score-gated-nontarget", "restored-score-nontarget"}
        or normalizer_approach is not None
        or uses_pairwise_margin
    )
    uses_restored_score = error_mode == "restored-score-nontarget"
    uses_pairwise_timing = uses_restored_score or uses_pairwise_margin
    update_signal_start_ns = 4.25 if uses_pairwise_timing else 1.8
    update_signal_end_ns = 6.35 if uses_pairwise_timing else 4.2
    acc_start_ns = 4.55 if uses_restored_score else 2.0
    acc_end_ns = 6.25 if uses_restored_score else 4.0
    apply_start_ns = 6.65 if uses_restored_score else 5.0
    apply_end_ns = 8.15 if uses_restored_score else 7.0
    score_reset_windows = [(0.2, 1.0), (8.35, 8.85)] if uses_restored_score else [(0.2, 1.0), (7.4, 8.0)]
    margin_sizing = (
        derive_multiclass_margin_correction_sizing(
            class_count=class_count,
            target_margin_v=1.0e-3,
            score_delta_v=1.0e-3,
            error_drive_scale=1.0,
        )
        if uses_pairwise_margin
        else None
    )

    lines = [
        "* Continuous class-local multiclass output-head training sequence.",
        "* Python supplies feature rails, labels, and clocks; weights update only through circuit state.",
        "* No behavioral sources.",
        ".param VDD=1.2",
        mos_models(),
        *(normalization_subcircuits(approaches=(normalizer_approach,)).splitlines() if normalizer_approach is not None else []),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        "Vvwhi_ref vwhi_ref 0 0.42",
        "Vvwlo_ref vwlo_ref 0 0.28",
        f"Vrst rst 0 {periodic_phase_pwl(cycle_count, start_ns=0.2, end_ns=1.0)}",
        f"Vrstscore rstscore 0 {multi_window_phase_pwl(cycle_count, windows=score_reset_windows)}",
    ]
    if readout_center_resistance > 0.0:
        lines.append(f"Vreadout_center readout_center 0 {readout_center_voltage:.12g}")
    if writer_mode == "sampled":
        lines += [
            f"Vacc acc 0 {periodic_phase_pwl(cycle_count, start_ns=acc_start_ns, end_ns=acc_end_ns, active_cycles=train_cycles)}",
            f"Vapply apply 0 {periodic_phase_pwl(cycle_count, start_ns=apply_start_ns, end_ns=apply_end_ns, active_cycles=train_cycles)}",
            f"Vapplyn applyn 0 {active_low_phase_pwl(cycle_count, start_ns=apply_start_ns, end_ns=apply_end_ns, active_cycles=train_cycles)}",
        ]
    if normalizer_approach is not None:
        lines += [
            f"Vscoreerr scoreerr 0 {periodic_phase_pwl(cycle_count, start_ns=update_signal_start_ns, end_ns=update_signal_end_ns, active_cycles=train_cycles)}",
            f"Vscoregaterst scoregaterst 0 {periodic_phase_pwl(cycle_count, start_ns=0.2, end_ns=1.0)}",
        ]
    if uses_pairwise_margin:
        lines += [
            f"Vscoreerr scoreerr 0 {periodic_phase_pwl(cycle_count, start_ns=update_signal_start_ns, end_ns=update_signal_end_ns, active_cycles=train_cycles, high=margin_sizing.error_clock_high_v)}",
        ]
    if uses_restored_score or uses_pairwise_margin:
        lines += [
            f"Vscorepre scorepre 0 {active_low_phase_pwl(cycle_count, start_ns=0.2, end_ns=1.0, active_cycles=set(range(cycle_count)))}",
            f"Vscoreamp scoreamp 0 {periodic_phase_pwl(cycle_count, start_ns=1.75, end_ns=3.15, active_cycles=train_cycles)}",
            f"Vscoredec scoredec 0 {periodic_phase_pwl(cycle_count, start_ns=3.45, end_ns=4.15, active_cycles=train_cycles)}",
        ]
    if uses_restored_score:
        lines += ["Voutref outref 0 0.25"]
    for feature in range(feature_count):
        act_values = [float(value) for value in features[:, feature]]
        actrow_windows = [(1.05, 1.65), (9.0, 12.0)] if uses_early_score else [(9.0, 12.0)]
        lines += [
            f"Vact{feature} act{feature} 0 {windowed_pwl(act_values, start_ns=update_signal_start_ns, end_ns=update_signal_end_ns)}",
            f"Vactrow{feature} actrow{feature} 0 {multi_windowed_pwl(act_values, windows=actrow_windows)}",
        ]
    for class_idx in range(class_count):
        targetp_values = [
            target_high if cycle in train_cycles and labels[cycle] == class_idx else 0.0 for cycle in range(cycle_count)
        ]
        targetn_values = [
            0.0 if cycle not in train_cycles or labels[cycle] == class_idx else target_high * nontarget_scale
            for cycle in range(cycle_count)
        ]
        lines += [
            f"V{class_node(class_idx, 'targetp')} {class_node(class_idx, 'targetp')} 0 {windowed_pwl(targetp_values, start_ns=update_signal_start_ns, end_ns=update_signal_end_ns)}",
            f"V{class_node(class_idx, 'targetn')} {class_node(class_idx, 'targetn')} 0 {width_scaled_windowed_pwl(targetn_values, start_ns=update_signal_start_ns, end_ns=update_signal_end_ns, width_scale=nontarget_width_scale)}",
            f"C{class_node(class_idx, 'score')} {class_node(class_idx, 'score')} 0 10f IC=0",
            f"C{class_node(class_idx, 'scoren')} {class_node(class_idx, 'scoren')} 0 10f IC=0",
            f"R{class_node(class_idx, 'score')} {class_node(class_idx, 'score')} 0 1e6",
            f"R{class_node(class_idx, 'scoren')} {class_node(class_idx, 'scoren')} 0 1e6",
            f"Mreset_{class_node(class_idx, 'score')} {class_node(class_idx, 'score')} rstscore 0 0 NMOS W=4u L=180n",
            f"Mreset_{class_node(class_idx, 'scoren')} {class_node(class_idx, 'scoren')} rstscore 0 0 NMOS W=4u L=180n",
        ]
        if uses_restored_score:
            prefix = f"c{class_idx}_"
            lines += [
                f"C{prefix}decision {prefix}decision 0 20f IC=0",
                f"C{prefix}decisionn {prefix}decisionn 0 20f IC=0",
                f"R{prefix}decision {prefix}decision 0 1G",
                f"R{prefix}decisionn {prefix}decisionn 0 1G",
                f"Mprecharge_{prefix}decision {prefix}decision scorepre vdd vdd PMOS W=4u L=180n",
                f"Mprecharge_{prefix}decisionn {prefix}decisionn scorepre vdd vdd PMOS W=4u L=180n",
                *low_gain_ref_state_lines(prefix=prefix, reset_node="scorepre"),
                *low_gain_ref_decision_lines(
                    prefix=prefix,
                    score_node=class_node(class_idx, "score"),
                    scoren_node=class_node(class_idx, "scoren"),
                    outref_node="outref",
                    amp_clock_node="scoreamp",
                    decision_clock_node="scoredec",
                ),
            ]
        for feature in range(feature_count):
            if writer_mode == "sampled":
                for node in ("gvp", "gvn"):
                    lines += [
                        f"C{class_node(class_idx, f'{node}{feature}')} {class_node(class_idx, f'{node}{feature}')} 0 2f IC=0",
                        f"R{class_node(class_idx, f'{node}{feature}')} {class_node(class_idx, f'{node}{feature}')} 0 1G",
                        f"Mreset_{class_node(class_idx, f'{node}{feature}')} {class_node(class_idx, f'{node}{feature}')} rst 0 0 NMOS W=4u L=180n",
                    ]
                for node in ("rgp", "rgn"):
                    lines += [
                        f"C{class_node(class_idx, f'{node}{feature}')} {class_node(class_idx, f'{node}{feature}')} 0 4f IC=1.2",
                        f"R{class_node(class_idx, f'{node}{feature}')} {class_node(class_idx, f'{node}{feature}')} vdd 50k",
                    ]
            lines += [
                *signed_store_lines(
                    positive_node=class_node(class_idx, f"vwp{feature}"),
                    negative_node=class_node(class_idx, f"vwn{feature}"),
                    positive_ic=initial_positive,
                    negative_ic=initial_negative,
                ),
                *(
                    [
                        f"R{class_node(class_idx, f'vwp{feature}')}_center {class_node(class_idx, f'vwp{feature}')} readout_center {readout_center_resistance:.12g}",
                        f"R{class_node(class_idx, f'vwn{feature}')}_center {class_node(class_idx, f'vwn{feature}')} readout_center {readout_center_resistance:.12g}",
                    ]
                    if readout_center_resistance > 0.0
                    else []
                ),
                *(
                    class_local_live_label_descent_update_lines(
                        class_idx=class_idx,
                        feature_idx=feature,
                        activation_node=f"act{feature}",
                        positive_descent_node=f"c{class_idx}_errp" if normalizer_approach is not None or uses_pairwise_margin else None,
                        negative_descent_node=f"c{class_idx}_errn" if normalizer_approach is not None or uses_pairwise_margin else None,
                    )
                    if writer_mode == "live"
                    else (
                        class_local_score_gated_nontarget_gradient_lines(
                            class_idx=class_idx,
                            feature_idx=feature,
                            activation_node=f"act{feature}",
                        )
                        if error_mode == "score-gated-nontarget"
                        else class_local_restored_score_nontarget_gradient_lines(
                            class_idx=class_idx,
                            feature_idx=feature,
                            activation_node=f"act{feature}",
                            score_gate_node=f"c{class_idx}_decision",
                        )
                        if uses_restored_score
                        else class_local_label_descent_gradient_lines(
                            class_idx=class_idx,
                            feature_idx=feature,
                            activation_node=f"act{feature}",
                        )
                    )
                ),
                *(class_local_bounded_update_lines(class_idx=class_idx, feature_idx=feature) if writer_mode == "sampled" else []),
                *class_local_readout_forward_lines(class_idx=class_idx, feature_idx=feature),
            ]
    if uses_pairwise_margin:
        for class_idx in range(class_count):
            for opponent_idx in range(class_idx + 1, class_count):
                lines += pairwise_low_gain_winner_lines(
                    class_a=class_idx,
                    class_b=opponent_idx,
                    amp_clock_node="scoreamp",
                    decision_clock_node="scoredec",
                    reset_node="scorepre",
                    pullup_width=margin_sizing.pairwise_pullup_width_u,
                    pulldown_width=margin_sizing.pairwise_pulldown_width_u,
                )
        lines += pairwise_target_margin_penalty_lines(
            class_count=class_count,
            decision_clock_node="scoredec",
            penalty_width_u=margin_sizing.margin_penalty_width_u,
        )
        lines += pairwise_live_margin_error_lines(
            class_count=class_count,
            error_clock_node="scoreerr",
            reset_node="rst",
            error_width_u=margin_sizing.error_width_u,
            error_capacitance_f=margin_sizing.error_cap_f,
        )
    if normalizer_approach is not None:
        lines += [
            "Xscore_normalizer "
            + " ".join(class_node(class_idx, "score") for class_idx in range(3))
            + " "
            + " ".join(class_node(class_idx, "targetp") for class_idx in range(3))
            + " "
            + " ".join(class_node(class_idx, "targetn") for class_idx in range(3))
            + " scoreerr scoregaterst "
            + " ".join(f"c{class_idx}_errp c{class_idx}_errn" for class_idx in range(3))
            + f" vdd 0 {spice_subckt_name(normalizer_approach)}"
        ]
    for cycle, (record, seq) in enumerate(zip(all_records, sequence)):
        base = cycle * CYCLE_NS
        for class_idx in range(class_count):
            lines += [
                f".meas tran c{class_idx}_score_{cycle} FIND V({class_node(class_idx, 'score')}) AT={base + 10.5:.2f}n",
                f".meas tran c{class_idx}_scoren_{cycle} FIND V({class_node(class_idx, 'scoren')}) AT={base + 10.5:.2f}n",
                f".meas tran c{class_idx}_score_net_{cycle} PARAM='c{class_idx}_score_{cycle}-c{class_idx}_scoren_{cycle}'",
            ]
            if cycle == len(all_records) - 1:
                for feature in range(feature_count):
                    lines += [
                        f".meas tran c{class_idx}_f{feature}_vwp_final FIND V({class_node(class_idx, f'vwp{feature}')}) AT={stop_ns - 0.5:.2f}n",
                        f".meas tran c{class_idx}_f{feature}_vwn_final FIND V({class_node(class_idx, f'vwn{feature}')}) AT={stop_ns - 0.5:.2f}n",
                        f".meas tran c{class_idx}_f{feature}_signed_final PARAM='c{class_idx}_f{feature}_vwp_final-c{class_idx}_f{feature}_vwn_final'",
                    ]
        lines.append(f"* cycle {cycle} {seq} label={int(record['label'])}")
    lines += [
        f".tran 5p {stop_ns:.2f}n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def rows_from_measures(
    records: list[dict[str, Any]],
    measures: dict[str, float],
    *,
    sequence: list[str],
    class_count: int,
) -> list[dict[str, Any]]:
    rows = []
    for cycle, (record, seq) in enumerate(zip(records, sequence)):
        scores = [float(measures[f"c{class_idx}_score_net_{cycle}"]) for class_idx in range(class_count)]
        prediction = int(np.argmax(scores))
        label = int(record["label"])
        row = {
            "cycle": cycle,
            "sequence": seq,
            "label": label,
            "prediction": prediction,
            "correct": prediction == label,
            "score_margin_v": float(scores[label] - max(score for idx, score in enumerate(scores) if idx != label)),
        }
        row.update({f"score_c{class_idx}_v": score for class_idx, score in enumerate(scores)})
        rows.append(row)
    return rows


def accuracy(rows: list[dict[str, Any]], sequence: str) -> float:
    selected = [row for row in rows if row["sequence"] == sequence]
    if not selected:
        return 0.0
    return float(np.mean([bool(row["correct"]) for row in selected]))


def final_signed_weight_matrix(measures: dict[str, float], *, class_count: int, feature_count: int) -> list[list[float]]:
    return [
        [float(measures[f"c{class_idx}_f{feature}_signed_final"]) for feature in range(feature_count)]
        for class_idx in range(class_count)
    ]


def score_matrix(rows: list[dict[str, Any]], *, sequence: str, class_count: int) -> list[list[float]]:
    return [
        [float(row[f"score_c{class_idx}_v"]) for class_idx in range(class_count)]
        for row in rows
        if row["sequence"] == sequence
    ]


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    generated = ROOT / "spice/generated"
    tables = ROOT / "results/tables"
    generated.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    tag = sanitize_tag(args.tag)
    spice_bin, version = detect_spice(args.spice_bin)
    dataset_info = parse_counted_mnist_dataset(args.dataset)
    if dataset_info is None:
        raise ValueError("dataset must be a counted multiclass MNIST dataset such as mnist3fixed8_6")
    class_count, _frontend, _sample_count = dataset_info
    records = dataset_records(args.dataset, args.seed, root=ROOT, download=args.download)
    train_records, eval_records = balanced_train_eval_split(
        records,
        class_count=class_count,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
    )
    all_records = eval_records + train_records + eval_records
    sequence = ["initial_eval"] * len(eval_records) + ["train"] * len(train_records) + ["final_eval"] * len(eval_records)
    start = time.perf_counter()
    deck = generate_netlist(
        train_records=train_records,
        eval_records=eval_records,
        class_count=class_count,
        feature_count=args.feature_count,
        initial_positive=args.initial_positive,
        initial_negative=args.initial_negative,
        nontarget_scale=args.nontarget_scale,
        nontarget_width_scale=args.nontarget_width_scale,
        error_mode=args.error_mode,
        writer_mode=args.writer_mode,
        readout_center_resistance=args.readout_center_resistance,
        readout_center_voltage=args.readout_center_voltage,
    )
    path = generated / f"{tag}.cir"
    measures = run_netlist(spice_bin, path, deck, timeout=args.timeout)
    rows = rows_from_measures(all_records, measures, sequence=sequence, class_count=class_count)
    initial_acc = accuracy(rows, "initial_eval")
    final_acc = accuracy(rows, "final_eval")
    final_signed = final_signed_weight_matrix(measures, class_count=class_count, feature_count=args.feature_count)
    final_signed_sums = [float(sum(row)) for row in final_signed]
    final_signed_spread = float(max(final_signed_sums) - min(final_signed_sums)) if final_signed_sums else 0.0
    csv_path = tables / f"{tag}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cycle",
                "sequence",
                "label",
                "prediction",
                "correct",
                "score_margin_v",
                *[f"score_c{class_idx}_v" for class_idx in range(class_count)],
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    final_margins = [float(row["score_margin_v"]) for row in rows if row["sequence"] == "final_eval"]
    summary = {
        "simulator": version,
        "architecture": "continuous_class_local_output_head_sequence",
        "dataset": args.dataset,
        "class_count": class_count,
        "feature_count": args.feature_count,
        "initial_positive": args.initial_positive,
        "initial_negative": args.initial_negative,
        "nontarget_scale": args.nontarget_scale,
        "nontarget_width_scale": args.nontarget_width_scale,
        "error_mode": args.error_mode,
        "writer_mode": args.writer_mode,
        "readout_center_resistance_ohm": args.readout_center_resistance if args.readout_center_resistance > 0.0 else None,
        "readout_center_voltage_v": args.readout_center_voltage if args.readout_center_resistance > 0.0 else None,
        "train_samples": args.train_samples,
        "eval_samples": args.eval_samples,
        "initial_eval_accuracy": initial_acc,
        "final_eval_accuracy": final_acc,
        "nontrivial_learning_met": final_acc > initial_acc,
        "final_eval_min_margin_v": min(final_margins) if final_margins else None,
        "final_signed_weight_sums_by_class_v": final_signed_sums,
        "final_signed_weight_sum_spread_v": final_signed_spread,
        "final_signed_weight_matrix_v": final_signed,
        "initial_eval_score_matrix_v": score_matrix(rows, sequence="initial_eval", class_count=class_count),
        "final_eval_score_matrix_v": score_matrix(rows, sequence="final_eval", class_count=class_count),
        "csv": str(csv_path),
        "wall_time_s": time.perf_counter() - start,
    }
    summary_path = tables / f"{tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spice-bin", default=None)
    ap.add_argument("--tag", default="multiclass_output_head_sequence")
    ap.add_argument("--dataset", default="mnist3fixed8_6")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--train-samples", type=int, default=3)
    ap.add_argument("--eval-samples", type=int, default=3)
    ap.add_argument("--feature-count", type=int, default=8)
    ap.add_argument("--initial-positive", type=float, default=0.36)
    ap.add_argument("--initial-negative", type=float, default=0.34)
    ap.add_argument("--nontarget-scale", type=float, default=0.5)
    ap.add_argument("--nontarget-width-scale", type=float, default=1.0)
    ap.add_argument("--error-mode", choices=ERROR_MODES, default="label-descent")
    ap.add_argument("--writer-mode", choices=WRITER_MODES, default="sampled")
    ap.add_argument("--readout-center-resistance", type=float, default=0.0)
    ap.add_argument("--readout-center-voltage", type=float, default=0.40)
    ap.add_argument("--timeout", type=float, default=120.0)
    return ap


def validate_args(args: argparse.Namespace) -> None:
    if args.timeout <= 0.0:
        raise ValueError("timeout must be positive")
    if args.train_samples <= 0 or args.eval_samples <= 0:
        raise ValueError("train-samples and eval-samples must be positive")
    if args.feature_count <= 0:
        raise ValueError("feature-count must be positive")
    if min(args.initial_positive, args.initial_negative) < 0.0:
        raise ValueError("initial-positive and initial-negative must be nonnegative")
    if max(args.initial_positive, args.initial_negative) > 1.2:
        raise ValueError("initial-positive and initial-negative must stay within supply rails")
    if args.nontarget_scale < 0.0 or args.nontarget_scale > 1.0:
        raise ValueError("nontarget-scale must be in [0, 1]")
    if args.nontarget_width_scale < 0.0 or args.nontarget_width_scale > 1.0:
        raise ValueError("nontarget-width-scale must be in [0, 1]")
    if args.readout_center_resistance < 0.0:
        raise ValueError("readout-center-resistance must be nonnegative")
    if args.readout_center_voltage < 0.0 or args.readout_center_voltage > 1.2:
        raise ValueError("readout-center-voltage must stay within supply rails")
    if args.error_mode not in ERROR_MODES:
        raise ValueError(f"error-mode must be one of {ERROR_MODES}")
    if args.writer_mode not in WRITER_MODES:
        raise ValueError(f"writer-mode must be one of {WRITER_MODES}")
    normalizer_approach = normalizer_approach_from_error_mode(args.error_mode)
    uses_pairwise_margin = args.error_mode == "live-pairwise-margin-descent"
    if args.writer_mode == "live" and args.error_mode != "label-descent" and normalizer_approach is None and not uses_pairwise_margin:
        raise ValueError("live writer-mode currently supports label-descent, pairwise-margin, and normalizer descent modes only")
    if normalizer_approach is not None and args.writer_mode != "live":
        raise ValueError("normalizer descent currently requires live writer-mode")
    if uses_pairwise_margin and args.writer_mode != "live":
        raise ValueError("live pairwise-margin descent currently requires live writer-mode")
    if parse_counted_mnist_dataset(args.dataset) is None:
        raise ValueError("dataset must be a counted multiclass MNIST dataset")


def main_for_test(argv: list[str]) -> argparse.Namespace:
    args = build_arg_parser().parse_args(argv)
    validate_args(args)
    return args


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)
    print(json.dumps(run_case(args), indent=2))


if __name__ == "__main__":
    main()
