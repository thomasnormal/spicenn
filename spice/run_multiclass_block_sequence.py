from __future__ import annotations

import argparse
import csv
import json
import time
from typing import Any

import numpy as np

from datasets import dataset_records, parse_counted_mnist_dataset
from run_device_mnist01_scalar_training import sanitize_tag
from run_device_sequential_training import mos_models, run_netlist
from run_multiclass_output_head_primitive import (
    class_local_bounded_update_lines,
    class_local_label_descent_gradient_lines,
    class_local_readout_forward_lines,
    class_node,
    signed_store_lines,
)
from run_multiclass_output_head_sequence import (
    CYCLE_NS,
    active_low_phase_pwl,
    balanced_train_eval_split,
    class_local_restored_score_nontarget_gradient_lines,
    class_local_score_gated_nontarget_gradient_lines,
    periodic_phase_pwl,
    width_scaled_windowed_pwl,
    windowed_pwl,
)
from run_score_decision_primitive import low_gain_preamp_lines, low_gain_ref_decision_lines, low_gain_ref_state_lines
from run_spice_sweep import ROOT, detect_spice


SCENARIOS = ("target-repeat", "one-hot", "mnist")
ERROR_MODES = (
    "label-descent",
    "score-gated-nontarget",
    "residual-score-nontarget",
    "amplified-score-nontarget",
    "restored-score-nontarget",
    "restored-winner-nontarget",
)


def pairwise_decision_node(class_idx: int, opponent_idx: int) -> str:
    return f"c{class_idx}_gt_c{opponent_idx}_decision"


def pairwise_winner_lines(
    *,
    class_a: int,
    class_b: int,
    compare_clock: str = "scoredec",
    reset_node: str = "scorepre",
    width_u: float = 64.0,
) -> list[str]:
    node_ab = pairwise_decision_node(class_a, class_b)
    node_ba = pairwise_decision_node(class_b, class_a)
    keeper_width_u = max(1.0, width_u / 64.0)
    return [
        f"C{node_ab} {node_ab} 0 20f IC=1.2",
        f"C{node_ba} {node_ba} 0 20f IC=1.2",
        f"R{node_ab} {node_ab} 0 1G",
        f"R{node_ba} {node_ba} 0 1G",
        f"Mprecharge_{node_ab} {node_ab} {reset_node} vdd vdd PMOS W=4u L=180n",
        f"Mprecharge_{node_ba} {node_ba} {reset_node} vdd vdd PMOS W=4u L=180n",
        f"M{node_ab}_dis_s {node_ab} {class_node(class_b, 'score')} {node_ab}_dn 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{node_ab}_dis_e {node_ab}_dn {compare_clock} 0 0 NMOS W={width_u:.6g}u L=180n",
        f"M{node_ba}_dis_s {node_ba} {class_node(class_a, 'score')} {node_ba}_dn 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{node_ba}_dis_e {node_ba}_dn {compare_clock} 0 0 NMOS W={width_u:.6g}u L=180n",
        f"M{node_ab}_keep {node_ab} {node_ba} vdd vdd PMOS W={keeper_width_u:.6g}u L=180n",
        f"M{node_ba}_keep {node_ba} {node_ab} vdd vdd PMOS W={keeper_width_u:.6g}u L=180n",
        f"M{node_ab}_nkeep {node_ab} {node_ba} 0 0 NMOS W={keeper_width_u:.6g}u L=180n",
        f"M{node_ba}_nkeep {node_ba} {node_ab} 0 0 NMOS W={keeper_width_u:.6g}u L=180n",
    ]


def class_local_multi_gate_nontarget_gradient_lines(
    *,
    class_idx: int,
    feature_idx: int,
    activation_node: str,
    gate_nodes: list[str],
    width_u: float = 24.0,
) -> list[str]:
    prefix = f"c{class_idx}_f{feature_idx}_"
    lines = [
        f"M{prefix}gvp_a vdd {activation_node} {prefix}gvp_a 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}gvp_d {prefix}gvp_a {class_node(class_idx, 'targetp')} {prefix}gvp_d 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}gvp_g {prefix}gvp_d acc {class_node(class_idx, f'gvp{feature_idx}')} 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_a vdd {activation_node} {prefix}gvn_a 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_label {prefix}gvn_a {class_node(class_idx, 'targetn')} {prefix}gvn_gate0 0 NSENSE W={width_u:.6g}u L=180n",
    ]
    previous = f"{prefix}gvn_gate0"
    for gate_idx, gate_node in enumerate(gate_nodes):
        next_node = f"{prefix}gvn_gate{gate_idx + 1}"
        lines.append(
            f"M{prefix}gvn_gate{gate_idx} {previous} {gate_node} {next_node} 0 NSENSE W={width_u:.6g}u L=180n"
        )
        previous = next_node
    lines += [
        f"M{prefix}gvn_g {previous} acc {class_node(class_idx, f'gvn{feature_idx}')} 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}rgp_pd {class_node(class_idx, f'rgp{feature_idx}')} {class_node(class_idx, f'gvp{feature_idx}')} 0 0 NSENSE W=16u L=180n",
        f"M{prefix}rgn_pd {class_node(class_idx, f'rgn{feature_idx}')} {class_node(class_idx, f'gvn{feature_idx}')} 0 0 NSENSE W=16u L=180n",
    ]
    return lines


def class_local_residual_score_gate_lines(
    *,
    class_idx: int,
    compare_clock: str = "scoredec",
    reset_node: str = "scoregaterst",
    pullup_width_u: float = 12.0,
    pulldown_width_u: float = 24.0,
) -> list[str]:
    prefix = f"c{class_idx}_"
    score_gate = class_node(class_idx, "score_gate")
    score_amp = f"{prefix}score_amp"
    scoren_amp = f"{prefix}scoren_amp"
    return [
        f"C{score_gate} {score_gate} 0 4f IC=0",
        f"R{score_gate} {score_gate} 0 1G",
        f"Mreset_{score_gate} {score_gate} {reset_node} 0 0 NMOS W=4u L=180n",
        f"M{prefix}score_gate_up_v vdd {score_amp} {prefix}score_gate_up_i 0 NREL W={pullup_width_u:.6g}u L=180n",
        f"M{prefix}score_gate_up_t {prefix}score_gate_up_i {compare_clock} {score_gate} 0 NSENSE W={pullup_width_u:.6g}u L=180n",
        f"M{prefix}score_gate_dn_v {score_gate} {scoren_amp} {prefix}score_gate_dn_i 0 NREL W={pulldown_width_u:.6g}u L=180n",
        f"M{prefix}score_gate_dn_t {prefix}score_gate_dn_i {compare_clock} 0 0 NSENSE W={pulldown_width_u:.6g}u L=180n",
    ]


def class_local_residual_score_nontarget_gradient_lines(
    *,
    class_idx: int,
    feature_idx: int,
    activation_node: str,
    score_gate_node: str,
    width_u: float = 24.0,
) -> list[str]:
    prefix = f"c{class_idx}_f{feature_idx}_"
    rgn = class_node(class_idx, f"rgn{feature_idx}")
    return [
        f"M{prefix}gvp_a vdd {activation_node} {prefix}gvp_a 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}gvp_d {prefix}gvp_a {class_node(class_idx, 'targetp')} {prefix}gvp_d 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}gvp_g {prefix}gvp_d acc {class_node(class_idx, f'gvp{feature_idx}')} 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_a vdd {activation_node} {prefix}gvn_a 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_label {prefix}gvn_a {class_node(class_idx, 'targetn')} {prefix}gvn_label 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_score {prefix}gvn_label {score_gate_node} {prefix}gvn_d 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_g {prefix}gvn_d acc {class_node(class_idx, f'gvn{feature_idx}')} 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}rgp_pd {class_node(class_idx, f'rgp{feature_idx}')} {class_node(class_idx, f'gvp{feature_idx}')} 0 0 NSENSE W=16u L=180n",
        f"M{prefix}rgn_pd {rgn} {class_node(class_idx, f'gvn{feature_idx}')} 0 0 NSENSE W=16u L=180n",
        f"M{prefix}rgn_res_a {rgn} {activation_node} {prefix}rgn_res_a 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}rgn_res_label {prefix}rgn_res_a {class_node(class_idx, 'targetn')} {prefix}rgn_res_label 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}rgn_res_score {prefix}rgn_res_label {score_gate_node} {prefix}rgn_res_score 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}rgn_res_acc {prefix}rgn_res_score acc 0 0 NREL W={width_u:.6g}u L=180n",
    ]


def records_to_feature_matrix(records: list[dict[str, Any]], feature_count: int) -> list[list[float]]:
    if feature_count <= 0:
        raise ValueError("feature_count must be positive")
    matrix = []
    for record in records:
        inputs = record.get("inputs")
        if not isinstance(inputs, dict):
            raise ValueError("records must contain an inputs dictionary")
        row = []
        for feature in range(feature_count):
            key = f"x{feature}"
            if key not in inputs:
                raise ValueError(f"records must contain inputs['{key}']")
            value = float(inputs[key])
            if value < 0.0 or value > 1.2:
                raise ValueError("input values must stay within supply rails")
            row.append(value)
        matrix.append(row)
    return matrix


def target_repeat_records(*, count: int, target_class: int, input_value: float) -> list[dict[str, Any]]:
    return [{"label": target_class, "inputs": {"x0": input_value}} for _ in range(count)]


def one_hot_records(*, class_count: int, repeats: int, active_value: float) -> list[dict[str, Any]]:
    records = []
    for _ in range(repeats):
        for label in range(class_count):
            records.append(
                {
                    "label": label,
                    "inputs": {f"x{feature}": active_value if feature == label else 0.0 for feature in range(class_count)},
                }
            )
    return records


def generate_netlist(
    *,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    class_count: int = 3,
    feature_count: int = 1,
    hidden_positive: float = 1.00,
    hidden_negative: float = 0.20,
    hidden_width_u: float = 1.0,
    readout_width_u: float = 64.0,
    score_capacitance_f: float = 10.0,
    score_load_resistance: float = 1e6,
    initial_positive: float = 0.40,
    initial_negative: float = 0.40,
    target_high: float = 1.1,
    nontarget_scale: float = 1.0,
    nontarget_width_scale: float = 1.0,
    error_mode: str = "label-descent",
) -> str:
    if class_count < 2:
        raise ValueError("class_count must be at least 2")
    if feature_count <= 0:
        raise ValueError("feature_count must be positive")
    if not train_records or not eval_records:
        raise ValueError("train_records and eval_records must be nonempty")
    for name, value in {
        "hidden_positive": hidden_positive,
        "hidden_negative": hidden_negative,
        "hidden_width_u": hidden_width_u,
        "readout_width_u": readout_width_u,
        "score_capacitance_f": score_capacitance_f,
        "score_load_resistance": score_load_resistance,
        "initial_positive": initial_positive,
        "initial_negative": initial_negative,
        "target_high": target_high,
    }.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if nontarget_scale < 0.0 or nontarget_scale > 1.0:
        raise ValueError("nontarget_scale must be in [0, 1]")
    if nontarget_width_scale < 0.0 or nontarget_width_scale > 1.0:
        raise ValueError("nontarget_width_scale must be in [0, 1]")
    if error_mode not in ERROR_MODES:
        raise ValueError(f"error_mode must be one of {ERROR_MODES}")
    if max(hidden_positive, hidden_negative, initial_positive, initial_negative, target_high) > 1.2:
        raise ValueError("voltages must stay within supply rails")

    all_records = eval_records + train_records + eval_records
    sequence = ["initial_eval"] * len(eval_records) + ["train"] * len(train_records) + ["final_eval"] * len(eval_records)
    train_cycles = {idx for idx, label in enumerate(sequence) if label == "train"}
    labels = [int(record["label"]) for record in all_records]
    if any(label < 0 or label >= class_count for label in labels):
        raise ValueError("record labels must be valid class indices")
    features = records_to_feature_matrix(all_records, feature_count)
    cycle_count = len(all_records)
    stop_ns = cycle_count * CYCLE_NS
    uses_restored_score = error_mode == "restored-score-nontarget"
    uses_residual_score = error_mode == "residual-score-nontarget"
    uses_amplified_score = error_mode == "amplified-score-nontarget"
    uses_score_preamp = uses_residual_score or uses_amplified_score
    uses_restored_winner = error_mode == "restored-winner-nontarget"
    uses_late_restored_gate = uses_restored_score or uses_restored_winner or uses_score_preamp
    target_start_ns = 10.8 if uses_late_restored_gate else 9.0
    target_end_ns = 12.8 if uses_late_restored_gate else 11.0
    acc_start_ns = 10.8 if uses_late_restored_gate else 9.0
    acc_end_ns = 12.8 if uses_late_restored_gate else 11.0
    apply_start_ns = 13.0 if uses_late_restored_gate else 12.0
    apply_end_ns = 13.1 if uses_late_restored_gate else 12.1

    lines = [
        "* Continuous multiclass block sequence: split-rail hidden features and class-local readout updates.",
        "* Python supplies row inputs, labels, and clocks only; readout weights persist as capacitor state.",
        "* No behavioral sources.",
        ".param VDD=1.2",
        mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        "Vvwhi_ref vwhi_ref 0 0.42",
        "Vvwlo_ref vwlo_ref 0 0.28",
        f"Vrst rst 0 {periodic_phase_pwl(cycle_count, start_ns=0.2, end_ns=1.0)}",
        f"Vsamp samp 0 {periodic_phase_pwl(cycle_count, start_ns=2.5, end_ns=3.5)}",
        f"Vsampn sampn 0 {active_low_phase_pwl(cycle_count, start_ns=2.5, end_ns=3.5, active_cycles=set(range(cycle_count)))}",
        f"Vout out 0 {periodic_phase_pwl(cycle_count, start_ns=5.0, end_ns=8.0)}",
        f"Voutn outn 0 {active_low_phase_pwl(cycle_count, start_ns=5.0, end_ns=8.0, active_cycles=set(range(cycle_count)))}",
        f"Vacc acc 0 {periodic_phase_pwl(cycle_count, start_ns=acc_start_ns, end_ns=acc_end_ns, active_cycles=train_cycles)}",
        f"Vapply apply 0 {periodic_phase_pwl(cycle_count, start_ns=apply_start_ns, end_ns=apply_end_ns, active_cycles=train_cycles)}",
        f"Vapplyn applyn 0 {active_low_phase_pwl(cycle_count, start_ns=apply_start_ns, end_ns=apply_end_ns, active_cycles=train_cycles)}",
    ]
    if uses_late_restored_gate:
        lines += [
            "Voutref outref 0 0.25",
            f"Vscorepre scorepre 0 {active_low_phase_pwl(cycle_count, start_ns=8.10, end_ns=8.35, active_cycles=train_cycles)}",
            f"Vscoreamp scoreamp 0 {periodic_phase_pwl(cycle_count, start_ns=8.60, end_ns=9.50, active_cycles=train_cycles)}",
            f"Vscoredec scoredec 0 {periodic_phase_pwl(cycle_count, start_ns=9.70, end_ns=10.40, active_cycles=train_cycles)}",
        ]
    if uses_residual_score:
        lines.append(
            f"Vscoregaterst scoregaterst 0 {periodic_phase_pwl(cycle_count, start_ns=8.10, end_ns=8.35, active_cycles=train_cycles)}"
        )
    for feature in range(feature_count):
        row_values = [float(row[feature]) for row in features]
        lines += [
            f"Vrow{feature} row{feature} 0 {windowed_pwl(row_values, start_ns=1.1, end_ns=4.0)}",
            f"Cwhp{feature} whp{feature} 0 20f IC={hidden_positive:.12g}",
            f"Cwhn{feature} whn{feature} 0 20f IC={hidden_negative:.12g}",
            f"Rwhp{feature} whp{feature} 0 1e15",
            f"Rwhn{feature} whn{feature} 0 1e15",
            f"Cpre_p{feature} pre_p{feature} 0 20f IC=0",
            f"Cpre_n{feature} pre_n{feature} 0 20f IC=0",
            f"Cact_raw{feature} act_raw{feature} 0 20f IC=0",
            f"Cact_store{feature} act{feature} 0 20f IC=0",
            f"Celig{feature} elig{feature} 0 20f IC=0",
            f"Cactrow{feature} actrow{feature} 0 1f IC=0",
            f"Rpre_p{feature} pre_p{feature} 0 1G",
            f"Rpre_n{feature} pre_n{feature} 0 1G",
            f"Ract_raw{feature} act_raw{feature} 0 1G",
            f"Ract_store{feature} act{feature} 0 1G",
            f"Relig{feature} elig{feature} 0 1G",
            f"Ractrow{feature} actrow{feature} 0 1e12",
            f"Mpre_p{feature}_rst pre_p{feature} rst 0 0 NMOS W=4u L=180n",
            f"Mpre_n{feature}_rst pre_n{feature} rst 0 0 NMOS W=4u L=180n",
            f"Mact_raw{feature}_rst act_raw{feature} rst 0 0 NMOS W=4u L=180n",
            f"Mact_store{feature}_rst act{feature} rst 0 0 NMOS W=4u L=180n",
            f"Melig{feature}_rst elig{feature} rst 0 0 NMOS W=4u L=180n",
            f"Mactrow{feature}_rst actrow{feature} rst 0 0 NMOS W=4u L=180n",
            f"Mhidden_pos{feature} row{feature} whp{feature} pre_p{feature} 0 NMOS W={hidden_width_u:.6g}u L=180n",
            f"Mhidden_neg{feature} row{feature} whn{feature} pre_n{feature} 0 NMOS W={hidden_width_u:.6g}u L=180n",
            f"Mact{feature}_p vdd pre_p{feature} act_raw{feature} 0 NREL W=24u L=180n",
            f"Mact{feature}_n act_raw{feature} pre_n{feature} 0 0 NREL W=24u L=180n",
            f"Mact_store{feature}_n act{feature} samp act_raw{feature} 0 NMOS W=16u L=180n",
            f"Mact_store{feature}_p act{feature} sampn act_raw{feature} vdd PMOS W=32u L=180n",
            f"Melig{feature}_n elig{feature} samp pre_p{feature} 0 NMOS W=16u L=180n",
            f"Melig{feature}_p elig{feature} sampn pre_p{feature} vdd PMOS W=32u L=180n",
            f"Mactrow{feature}_n actrow{feature} out act{feature} 0 NMOS W=16u L=180n",
            f"Mactrow{feature}_p actrow{feature} outn act{feature} vdd PMOS W=32u L=180n",
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
            f"V{class_node(class_idx, 'targetp')} {class_node(class_idx, 'targetp')} 0 {windowed_pwl(targetp_values, start_ns=target_start_ns, end_ns=target_end_ns)}",
            f"V{class_node(class_idx, 'targetn')} {class_node(class_idx, 'targetn')} 0 {width_scaled_windowed_pwl(targetn_values, start_ns=target_start_ns, end_ns=target_end_ns, width_scale=nontarget_width_scale)}",
            f"C{class_node(class_idx, 'score')} {class_node(class_idx, 'score')} 0 {score_capacitance_f:.12g}f IC=0",
            f"C{class_node(class_idx, 'scoren')} {class_node(class_idx, 'scoren')} 0 {score_capacitance_f:.12g}f IC=0",
            f"R{class_node(class_idx, 'score')} {class_node(class_idx, 'score')} 0 {score_load_resistance:.12g}",
            f"R{class_node(class_idx, 'scoren')} {class_node(class_idx, 'scoren')} 0 {score_load_resistance:.12g}",
            f"Mreset_{class_node(class_idx, 'score')} {class_node(class_idx, 'score')} rst 0 0 NMOS W=4u L=180n",
            f"Mreset_{class_node(class_idx, 'scoren')} {class_node(class_idx, 'scoren')} rst 0 0 NMOS W=4u L=180n",
        ]
        if uses_restored_score or uses_score_preamp:
            prefix = f"c{class_idx}_"
            lines += [
                *low_gain_ref_state_lines(prefix=prefix, reset_node="scorepre"),
            ]
            if uses_restored_score:
                lines += [
                    f"C{prefix}decision {prefix}decision 0 20f IC=0",
                    f"C{prefix}decisionn {prefix}decisionn 0 20f IC=0",
                    f"R{prefix}decision {prefix}decision 0 1G",
                    f"R{prefix}decisionn {prefix}decisionn 0 1G",
                    f"Mprecharge_{prefix}decision {prefix}decision scorepre vdd vdd PMOS W=4u L=180n",
                    f"Mprecharge_{prefix}decisionn {prefix}decisionn scorepre vdd vdd PMOS W=4u L=180n",
                    *low_gain_ref_decision_lines(
                        prefix=prefix,
                        score_node=class_node(class_idx, "score"),
                        scoren_node=class_node(class_idx, "scoren"),
                        outref_node="outref",
                        amp_clock_node="scoreamp",
                        decision_clock_node="scoredec",
                    ),
                ]
            else:
                lines += [
                    *low_gain_preamp_lines(
                        prefix=prefix,
                        score_node=class_node(class_idx, "score"),
                        scoren_node=class_node(class_idx, "scoren"),
                        amp_clock_node="scoreamp",
                    ),
                ]
                if uses_residual_score:
                    lines += class_local_residual_score_gate_lines(class_idx=class_idx)
        if uses_restored_winner:
            for opponent_idx in range(class_idx + 1, class_count):
                lines += pairwise_winner_lines(class_a=class_idx, class_b=opponent_idx)
        for feature in range(feature_count):
            if error_mode == "score-gated-nontarget":
                gradient_lines = class_local_score_gated_nontarget_gradient_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=f"elig{feature}",
                )
            elif uses_restored_score:
                gradient_lines = class_local_restored_score_nontarget_gradient_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=f"elig{feature}",
                    score_gate_node=f"c{class_idx}_decision",
                )
            elif uses_residual_score:
                gradient_lines = class_local_residual_score_nontarget_gradient_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=f"elig{feature}",
                    score_gate_node=class_node(class_idx, "score_gate"),
                )
            elif uses_amplified_score:
                gradient_lines = class_local_residual_score_nontarget_gradient_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=f"elig{feature}",
                    score_gate_node=f"c{class_idx}_score_amp",
                )
            elif uses_restored_winner:
                gradient_lines = class_local_multi_gate_nontarget_gradient_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=f"elig{feature}",
                    gate_nodes=[
                        pairwise_decision_node(class_idx, opponent_idx)
                        for opponent_idx in range(class_count)
                        if opponent_idx != class_idx
                    ],
                )
            else:
                gradient_lines = class_local_label_descent_gradient_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=f"elig{feature}",
                )
            lines += [
                f"C{class_node(class_idx, f'gvp{feature}')} {class_node(class_idx, f'gvp{feature}')} 0 2f IC=0",
                f"C{class_node(class_idx, f'gvn{feature}')} {class_node(class_idx, f'gvn{feature}')} 0 2f IC=0",
                f"C{class_node(class_idx, f'rgp{feature}')} {class_node(class_idx, f'rgp{feature}')} 0 4f IC=1.2",
                f"C{class_node(class_idx, f'rgn{feature}')} {class_node(class_idx, f'rgn{feature}')} 0 4f IC=1.2",
                f"R{class_node(class_idx, f'gvp{feature}')} {class_node(class_idx, f'gvp{feature}')} 0 1G",
                f"R{class_node(class_idx, f'gvn{feature}')} {class_node(class_idx, f'gvn{feature}')} 0 1G",
                f"R{class_node(class_idx, f'rgp{feature}')} {class_node(class_idx, f'rgp{feature}')} vdd 50k",
                f"R{class_node(class_idx, f'rgn{feature}')} {class_node(class_idx, f'rgn{feature}')} vdd 50k",
                f"Mreset_{class_node(class_idx, f'gvp{feature}')} {class_node(class_idx, f'gvp{feature}')} rst 0 0 NMOS W=4u L=180n",
                f"Mreset_{class_node(class_idx, f'gvn{feature}')} {class_node(class_idx, f'gvn{feature}')} rst 0 0 NMOS W=4u L=180n",
                *signed_store_lines(
                    positive_node=class_node(class_idx, f"vwp{feature}"),
                    negative_node=class_node(class_idx, f"vwn{feature}"),
                    positive_ic=initial_positive,
                    negative_ic=initial_negative,
                ),
                *class_local_readout_forward_lines(class_idx=class_idx, feature_idx=feature, width_u=readout_width_u),
                *gradient_lines,
                *class_local_bounded_update_lines(class_idx=class_idx, feature_idx=feature),
            ]
    train_seen = 0
    for cycle, (record, seq) in enumerate(zip(all_records, sequence)):
        base = cycle * CYCLE_NS
        for feature in range(feature_count):
            lines += [
                f".meas tran pre_p_f{feature}_{cycle} FIND V(pre_p{feature}) AT={base + 3.2:.2f}n",
                f".meas tran pre_n_f{feature}_{cycle} FIND V(pre_n{feature}) AT={base + 3.2:.2f}n",
                f".meas tran pre_margin_f{feature}_{cycle} PARAM='pre_p_f{feature}_{cycle}-pre_n_f{feature}_{cycle}'",
                f".meas tran act_f{feature}_{cycle} FIND V(act{feature}) AT={base + 4.5:.2f}n",
                f".meas tran elig_f{feature}_{cycle} FIND V(elig{feature}) AT={base + 4.5:.2f}n",
            ]
        lines += [
            f".meas tran pre_margin_{cycle} PARAM='pre_margin_f0_{cycle}'",
            f".meas tran act_{cycle} PARAM='act_f0_{cycle}'",
            f".meas tran elig_{cycle} PARAM='elig_f0_{cycle}'",
        ]
        for class_idx in range(class_count):
            lines += [
                f".meas tran c{class_idx}_score_{cycle} FIND V({class_node(class_idx, 'score')}) AT={base + 8.5:.2f}n",
                f".meas tran c{class_idx}_scoren_{cycle} FIND V({class_node(class_idx, 'scoren')}) AT={base + 8.5:.2f}n",
                f".meas tran c{class_idx}_score_net_{cycle} PARAM='c{class_idx}_score_{cycle}-c{class_idx}_scoren_{cycle}'",
            ]
            if uses_restored_winner:
                for opponent_idx in range(class_count):
                    if opponent_idx == class_idx:
                        continue
                    lines += [
                        f".meas tran c{class_idx}_gt_c{opponent_idx}_decision_{cycle} FIND V({pairwise_decision_node(class_idx, opponent_idx)}) AT={base + 10.6:.2f}n",
                        f".meas tran c{class_idx}_gt_c{opponent_idx}_decisionn_{cycle} FIND V({pairwise_decision_node(opponent_idx, class_idx)}) AT={base + 10.6:.2f}n",
                        f".meas tran c{class_idx}_gt_c{opponent_idx}_diff_{cycle} PARAM='c{class_idx}_gt_c{opponent_idx}_decision_{cycle}-c{class_idx}_gt_c{opponent_idx}_decisionn_{cycle}'",
                    ]
            if uses_score_preamp:
                lines += [
                    f".meas tran c{class_idx}_score_amp_{cycle} FIND V(c{class_idx}_score_amp) AT={base + 9.60:.2f}n",
                    f".meas tran c{class_idx}_scoren_amp_{cycle} FIND V(c{class_idx}_scoren_amp) AT={base + 9.60:.2f}n",
                    f".meas tran c{class_idx}_score_gain_diff_{cycle} PARAM='c{class_idx}_score_amp_{cycle}-c{class_idx}_scoren_amp_{cycle}'",
                ]
            if uses_residual_score:
                lines += [
                    f".meas tran c{class_idx}_score_gate_{cycle} FIND V({class_node(class_idx, 'score_gate')}) AT={base + 10.60:.2f}n",
                ]
        if seq == "train":
            train_seen += 1
            for class_idx in range(class_count):
                for feature in range(feature_count):
                    lines += [
                        f".meas tran c{class_idx}_f{feature}_vwp_after_train{train_seen} FIND V({class_node(class_idx, f'vwp{feature}')}) AT={base + 15.0:.2f}n",
                        f".meas tran c{class_idx}_f{feature}_vwn_after_train{train_seen} FIND V({class_node(class_idx, f'vwn{feature}')}) AT={base + 15.0:.2f}n",
                        f".meas tran c{class_idx}_f{feature}_signed_after_train{train_seen} PARAM='c{class_idx}_f{feature}_vwp_after_train{train_seen}-c{class_idx}_f{feature}_vwn_after_train{train_seen}'",
                    ]
                lines += [
                    f".meas tran c{class_idx}_signed_after_train{train_seen} PARAM='c{class_idx}_f0_signed_after_train{train_seen}'",
                ]
        lines.append(f"* cycle {cycle} {seq} label={int(record['label'])}")
    for class_idx in range(class_count):
        for feature in range(feature_count):
            lines += [
                f".meas tran c{class_idx}_f{feature}_vwp_final FIND V({class_node(class_idx, f'vwp{feature}')}) AT={stop_ns - 0.5:.2f}n",
                f".meas tran c{class_idx}_f{feature}_vwn_final FIND V({class_node(class_idx, f'vwn{feature}')}) AT={stop_ns - 0.5:.2f}n",
                f".meas tran c{class_idx}_f{feature}_signed_final PARAM='c{class_idx}_f{feature}_vwp_final-c{class_idx}_f{feature}_vwn_final'",
            ]
        lines += [f".meas tran c{class_idx}_signed_final PARAM='c{class_idx}_f0_signed_final'"]
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
        label = int(record["label"])
        prediction = int(np.argmax(scores))
        rows.append(
            {
                "cycle": cycle,
                "sequence": seq,
                "label": label,
                "prediction": prediction,
                "correct": prediction == label,
                "score_margin_v": float(scores[label] - max(score for idx, score in enumerate(scores) if idx != label)),
                **{f"score_c{class_idx}_v": scores[class_idx] for class_idx in range(class_count)},
            }
        )
    return rows


def accuracy(rows: list[dict[str, Any]], sequence: str) -> float:
    selected = [row for row in rows if row["sequence"] == sequence]
    if not selected:
        return 0.0
    return float(np.mean([bool(row["correct"]) for row in selected]))


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    generated = ROOT / "spice/generated"
    tables = ROOT / "results/tables"
    generated.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    tag = sanitize_tag(args.tag)
    spice_bin, version = detect_spice(args.spice_bin)
    feature_count = args.feature_count
    if args.scenario == "target-repeat":
        feature_count = 1
        train_records = target_repeat_records(
            count=args.train_samples,
            target_class=args.target_class,
            input_value=args.input_value,
        )
        eval_records = target_repeat_records(
            count=args.eval_samples,
            target_class=args.target_class,
            input_value=args.input_value,
        )
    elif args.scenario == "one-hot":
        feature_count = args.class_count
        train_records = one_hot_records(
            class_count=args.class_count,
            repeats=args.train_repeats,
            active_value=args.input_value,
        )
        eval_records = one_hot_records(
            class_count=args.class_count,
            repeats=args.eval_repeats,
            active_value=args.input_value,
        )
    elif args.scenario == "mnist":
        dataset_info = parse_counted_mnist_dataset(args.dataset)
        if dataset_info is None:
            raise ValueError("dataset must be a counted multiclass MNIST dataset such as mnist3fixed8_6")
        class_count, _frontend, _sample_count = dataset_info
        if class_count != args.class_count:
            raise ValueError("class-count must match the counted MNIST dataset")
        records = dataset_records(args.dataset, args.seed, root=ROOT, download=args.download)
        train_records, eval_records = balanced_train_eval_split(
            records,
            class_count=args.class_count,
            train_samples=args.train_samples,
            eval_samples=args.eval_samples,
        )
    else:
        raise ValueError(f"scenario must be one of {SCENARIOS}")
    all_records = eval_records + train_records + eval_records
    sequence = ["initial_eval"] * len(eval_records) + ["train"] * len(train_records) + ["final_eval"] * len(eval_records)
    start = time.perf_counter()
    path = generated / f"{tag}.cir"
    deck = generate_netlist(
        train_records=train_records,
        eval_records=eval_records,
        class_count=args.class_count,
        feature_count=feature_count,
        readout_width_u=args.readout_width,
        score_capacitance_f=args.score_capacitance_f,
        score_load_resistance=args.score_load_resistance,
        initial_positive=args.initial_positive,
        initial_negative=args.initial_negative,
        nontarget_scale=args.nontarget_scale,
        nontarget_width_scale=args.nontarget_width_scale,
        error_mode=args.error_mode,
    )
    measures = run_netlist(spice_bin, path, deck, timeout=args.timeout)
    rows = rows_from_measures(all_records, measures, sequence=sequence, class_count=args.class_count)
    initial_margins = [float(row["score_margin_v"]) for row in rows if row["sequence"] == "initial_eval"]
    final_margins = [float(row["score_margin_v"]) for row in rows if row["sequence"] == "final_eval"]
    final_signed = [
        [float(measures[f"c{class_idx}_f{feature}_signed_final"]) for feature in range(feature_count)]
        for class_idx in range(args.class_count)
    ]
    train_progress = []
    for train_idx in range(1, len(train_records) + 1):
        train_progress.append(
            [
                [float(measures[f"c{class_idx}_f{feature}_signed_after_train{train_idx}"]) for feature in range(feature_count)]
                for class_idx in range(args.class_count)
            ]
        )
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
                *[f"score_c{class_idx}_v" for class_idx in range(args.class_count)],
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    initial_margin = min(initial_margins)
    final_margin = min(final_margins)
    summary = {
        "simulator": version,
        "architecture": "continuous_multiclass_split_rail_block_sequence",
        "scenario": args.scenario,
        "dataset": args.dataset if args.scenario == "mnist" else None,
        "seed": args.seed if args.scenario == "mnist" else None,
        "class_count": args.class_count,
        "feature_count": feature_count,
        "readout_width_u": args.readout_width,
        "score_capacitance_f": args.score_capacitance_f,
        "score_load_resistance_ohm": args.score_load_resistance,
        "nontarget_scale": args.nontarget_scale,
        "nontarget_width_scale": args.nontarget_width_scale,
        "error_mode": args.error_mode,
        "target_class": args.target_class if args.scenario == "target-repeat" else None,
        "train_samples": len(train_records),
        "eval_samples": len(eval_records),
        "initial_eval_accuracy": accuracy(rows, "initial_eval"),
        "final_eval_accuracy": accuracy(rows, "final_eval"),
        "accuracy_improvement": accuracy(rows, "final_eval") - accuracy(rows, "initial_eval"),
        "initial_eval_min_margin_v": initial_margin,
        "final_eval_min_margin_v": final_margin,
        "margin_improvement_v": final_margin - initial_margin,
        "final_signed_matrix_v": final_signed,
        "signed_after_each_train_v": train_progress,
        "passed": (
            accuracy(rows, "final_eval") > accuracy(rows, "initial_eval")
            if args.scenario == "mnist"
            else (
                final_margin > initial_margin
                and accuracy(rows, "final_eval") >= accuracy(rows, "initial_eval")
                and (
                    args.scenario == "one-hot"
                    or final_signed[args.target_class][0] > args.min_target_signed
                )
            )
        ),
        "csv": str(csv_path),
        "wall_time_s": time.perf_counter() - start,
    }
    summary_path = tables / f"{tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spice-bin", default=None)
    ap.add_argument("--tag", default="multiclass_block_sequence")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--class-count", type=int, default=3)
    ap.add_argument("--feature-count", type=int, default=8)
    ap.add_argument("--scenario", choices=SCENARIOS, default="target-repeat")
    ap.add_argument("--dataset", default="mnist3fixed8_6")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--target-class", type=int, default=0)
    ap.add_argument("--train-samples", type=int, default=2)
    ap.add_argument("--eval-samples", type=int, default=1)
    ap.add_argument("--train-repeats", type=int, default=1)
    ap.add_argument("--eval-repeats", type=int, default=1)
    ap.add_argument("--input-value", type=float, default=0.85)
    ap.add_argument("--readout-width", type=float, default=64.0)
    ap.add_argument("--score-capacitance-f", type=float, default=10.0)
    ap.add_argument("--score-load-resistance", type=float, default=1e6)
    ap.add_argument("--initial-positive", type=float, default=0.40)
    ap.add_argument("--initial-negative", type=float, default=0.40)
    ap.add_argument("--nontarget-scale", type=float, default=1.0)
    ap.add_argument("--nontarget-width-scale", type=float, default=1.0)
    ap.add_argument("--error-mode", choices=ERROR_MODES, default="label-descent")
    ap.add_argument("--min-target-signed", type=float, default=10e-3)
    return ap


def validate_args(args: argparse.Namespace) -> None:
    if args.timeout <= 0.0:
        raise ValueError("timeout must be positive")
    if args.class_count < 2:
        raise ValueError("class-count must be at least 2")
    if args.feature_count <= 0:
        raise ValueError("feature-count must be positive")
    if args.scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}")
    if args.error_mode not in ERROR_MODES:
        raise ValueError(f"error-mode must be one of {ERROR_MODES}")
    if args.scenario == "mnist" and parse_counted_mnist_dataset(args.dataset) is None:
        raise ValueError("dataset must be a counted multiclass MNIST dataset")
    if args.target_class < 0 or args.target_class >= args.class_count:
        raise ValueError("target-class must be a valid class index")
    if args.train_samples <= 0 or args.eval_samples <= 0:
        raise ValueError("train-samples and eval-samples must be positive")
    if args.train_repeats <= 0 or args.eval_repeats <= 0:
        raise ValueError("train-repeats and eval-repeats must be positive")
    if args.input_value < 0.0 or args.input_value > 1.2:
        raise ValueError("input-value must stay within supply rails")
    if args.readout_width <= 0.0:
        raise ValueError("readout-width must be positive")
    if args.score_capacitance_f <= 0.0:
        raise ValueError("score-capacitance-f must be positive")
    if args.score_load_resistance <= 0.0:
        raise ValueError("score-load-resistance must be positive")
    if args.nontarget_scale < 0.0 or args.nontarget_scale > 1.0:
        raise ValueError("nontarget-scale must be in [0, 1]")
    if args.nontarget_width_scale < 0.0 or args.nontarget_width_scale > 1.0:
        raise ValueError("nontarget-width-scale must be in [0, 1]")
    if min(args.initial_positive, args.initial_negative) <= 0.0:
        raise ValueError("initial-positive and initial-negative must be positive")
    if max(args.initial_positive, args.initial_negative) > 1.2:
        raise ValueError("initial-positive and initial-negative must stay within supply rails")
    if args.min_target_signed < 0.0:
        raise ValueError("min-target-signed must be nonnegative")


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
