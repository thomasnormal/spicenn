from __future__ import annotations

import argparse
import csv
import json
import time
from typing import Any

import numpy as np

from parameter_theory import derive_class_evidence_normalizer_sizing, derive_multiclass_margin_correction_sizing
from datasets import dataset_records, parse_counted_mnist_dataset
from run_device_mnist01_scalar_training import sanitize_tag
from run_device_sequential_training import mos_models, run_netlist
from run_normalization_subcircuits import (
    APPROACHES as NORMALIZATION_APPROACHES,
    normalization_subcircuits,
    spice_subckt_name,
)
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
    class_local_live_label_descent_update_lines,
    class_local_restored_score_nontarget_gradient_lines,
    class_local_score_gated_nontarget_gradient_lines,
    periodic_phase_pwl,
    width_scaled_windowed_pwl,
    windowed_pwl,
)
from run_feature_eligibility_competition_primitive import (
    feature_common_contrast_gate_lines,
    feature_loss_suppression_lines,
    feature_rank_suppression_lines,
    gate_node as eligibility_gate_node,
    pairwise_feature_winner_lines,
    shared_eligibility_common_reference_lines,
)
from run_score_decision_primitive import low_gain_preamp_lines, low_gain_ref_decision_lines, low_gain_ref_state_lines
from run_spice_sweep import ROOT, detect_spice


SCENARIOS = ("target-repeat", "one-hot", "mnist")
CLASS_BIAS_MODES = ("none", "target-only", "label-descent")
READOUT_UPDATE_MODES = ("sampled", "live")
HIDDEN_UPDATE_MODES = ("none", "readout-weighted")
SCORE_TIMING_MODES = ("late", "early")
SCORE_SENSE_MODES = ("voltage", "current-clamp")
READOUT_FORWARD_MODES = ("direct", "diode")
ELIGIBILITY_GATE_MODES = ("raw", "competition", "rank", "contrast")
ERROR_MODES = (
    "label-descent",
    "label-rail-descent",
    "score-gated-nontarget",
    "residual-score-nontarget",
    "amplified-score-nontarget",
    "common-ref-score-nontarget",
    "raw-common-ref-score-nontarget",
    "target-ref-score-nontarget",
    "amplified-score-competitive",
    "amplified-score-pairwise",
    "amplified-score-binary-descent",
    "score-mass-descent",
    "common-score-mass-descent",
    "contrast-score-mass-descent",
    "low-gain-contrast-score-mass-descent",
    "contrast-gated-score-mass-descent",
    "target-contrast-score-mass-descent",
    "common-score-mass-pairwise-descent",
    "pairwise-score-competition-descent",
    "pairwise-margin-correction-descent",
    *(f"normalizer-{approach}-descent" for approach in NORMALIZATION_APPROACHES),
    "pairwise-binary-descent",
    "restored-score-binary-descent",
    "restored-score-nontarget",
    "restored-winner-nontarget",
)
NORMALIZER_ERROR_MODES = tuple(f"normalizer-{approach}-descent" for approach in NORMALIZATION_APPROACHES)

ERROR_RAIL_DESCENT_MODES = (
    "label-rail-descent",
    "score-mass-descent",
    "common-score-mass-descent",
    "contrast-score-mass-descent",
    "low-gain-contrast-score-mass-descent",
    "contrast-gated-score-mass-descent",
    "target-contrast-score-mass-descent",
    "common-score-mass-pairwise-descent",
    "pairwise-score-competition-descent",
    "pairwise-margin-correction-descent",
    *NORMALIZER_ERROR_MODES,
)


def normalizer_error_approach(error_mode: str) -> str:
    if error_mode not in NORMALIZER_ERROR_MODES:
        raise ValueError(f"error_mode must be one of {NORMALIZER_ERROR_MODES}")
    return error_mode.removeprefix("normalizer-").removesuffix("-descent")


def pairwise_decision_node(class_idx: int, opponent_idx: int) -> str:
    return f"c{class_idx}_gt_c{opponent_idx}_decision"


def class_local_label_error_rail_lines(
    *,
    class_count: int,
    error_clock_node: str = "scoreerr",
    reset_node: str = "scoregaterst",
    error_width_u: float = 32.0,
    error_capacitance_f: float = 2.0,
) -> list[str]:
    if min(error_width_u, error_capacitance_f) <= 0.0:
        raise ValueError("label error rail widths and capacitances must be positive")
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
            f"R{errp}_label {errp}_label 0 1G",
            f"M{errp}_label vdd {class_node(class_idx, 'targetp')} {errp}_label 0 NSENSE W={error_width_u:.6g}u L=180n",
            f"M{errp}_clk {errp}_label {error_clock_node} {errp} 0 NSENSE W={error_width_u:.6g}u L=180n",
            f"R{errn}_label {errn}_label 0 1G",
            f"M{errn}_label vdd {class_node(class_idx, 'targetn')} {errn}_label 0 NSENSE W={error_width_u:.6g}u L=180n",
            f"M{errn}_clk {errn}_label {error_clock_node} {errn} 0 NSENSE W={error_width_u:.6g}u L=180n",
        ]
    return lines


def hidden_credit_node(feature_idx: int, suffix: str) -> str:
    return f"h{feature_idx}_{suffix}"


def hidden_readout_weighted_credit_lines(
    *,
    class_count: int,
    feature_idx: int,
    error_positive_nodes: list[str],
    error_negative_nodes: list[str],
    width_u: float = 8.0,
) -> list[str]:
    hdp = hidden_credit_node(feature_idx, "hdp")
    hdn = hidden_credit_node(feature_idx, "hdn")
    lines = [
        f"C{hdp} {hdp} 0 12f IC=0",
        f"C{hdn} {hdn} 0 12f IC=0",
        f"R{hdp} {hdp} 0 1G",
        f"R{hdn} {hdn} 0 1G",
        f"Mreset_{hdp} {hdp} rst 0 0 NMOS W=4u L=180n",
        f"Mreset_{hdn} {hdn} rst 0 0 NMOS W=4u L=180n",
    ]
    for class_idx in range(class_count):
        errp = error_positive_nodes[class_idx]
        errn = error_negative_nodes[class_idx]
        vwp = class_node(class_idx, f"vwp{feature_idx}")
        vwn = class_node(class_idx, f"vwn{feature_idx}")
        act = f"act{feature_idx}"
        prefix = f"h{feature_idx}_c{class_idx}_"
        lines += [
            f"R{prefix}pv_e {prefix}pv_e 0 1G",
            f"R{prefix}pv_w {prefix}pv_w 0 1G",
            f"M{prefix}hdp_pv_e vdd {errp} {prefix}pv_e 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}hdp_pv_w {prefix}pv_e {vwp} {prefix}pv_w 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}hdp_pv_a {prefix}pv_w {act} {hdp} 0 NREL W={width_u:.6g}u L=180n",
            f"R{prefix}pn_e {prefix}pn_e 0 1G",
            f"R{prefix}pn_w {prefix}pn_w 0 1G",
            f"M{prefix}hdn_pv_e vdd {errp} {prefix}pn_e 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}hdn_pv_w {prefix}pn_e {vwn} {prefix}pn_w 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}hdn_pv_a {prefix}pn_w {act} {hdn} 0 NREL W={width_u:.6g}u L=180n",
            f"R{prefix}nv_e {prefix}nv_e 0 1G",
            f"R{prefix}nv_w {prefix}nv_w 0 1G",
            f"M{prefix}hdp_nv_e vdd {errn} {prefix}nv_e 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}hdp_nv_w {prefix}nv_e {vwn} {prefix}nv_w 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}hdp_nv_a {prefix}nv_w {act} {hdp} 0 NREL W={width_u:.6g}u L=180n",
            f"R{prefix}nn_e {prefix}nn_e 0 1G",
            f"R{prefix}nn_w {prefix}nn_w 0 1G",
            f"M{prefix}hdn_nv_e vdd {errn} {prefix}nn_e 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}hdn_nv_w {prefix}nn_e {vwp} {prefix}nn_w 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}hdn_nv_a {prefix}nn_w {act} {hdn} 0 NREL W={width_u:.6g}u L=180n",
        ]
    return lines


def hidden_live_weight_update_lines(
    *,
    feature_idx: int,
    eligibility_node: str,
    positive_credit_node: str,
    negative_credit_node: str,
    width_u: float = 0.25,
) -> list[str]:
    whp = f"whp{feature_idx}"
    whn = f"whn{feature_idx}"
    prefix = f"h{feature_idx}_live_"
    return [
        f"M{prefix}pos_up_e vwhi_ref {eligibility_node} {prefix}pos_up 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}pos_up_d {prefix}pos_up {positive_credit_node} {whp} 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}pos_dn_e {whn} {eligibility_node} {prefix}pos_dn 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}pos_dn_d {prefix}pos_dn {positive_credit_node} vwlo_ref 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}neg_up_e vwhi_ref {eligibility_node} {prefix}neg_up 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}neg_up_d {prefix}neg_up {negative_credit_node} {whn} 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}neg_dn_e {whp} {eligibility_node} {prefix}neg_dn 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}neg_dn_d {prefix}neg_dn {negative_credit_node} vwlo_ref 0 NSENSE W={width_u:.6g}u L=180n",
    ]


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


def class_local_pairwise_binary_descent_gradient_lines(
    *,
    class_idx: int,
    feature_idx: int,
    activation_node: str,
    losing_gate_nodes: list[str],
    winning_gate_nodes: list[str],
    width_u: float = 24.0,
    pairwise_width_u: float = 64.0,
) -> list[str]:
    prefix = f"c{class_idx}_f{feature_idx}_"
    lines = [
        f"M{prefix}rgp_pair_base {class_node(class_idx, f'rgp{feature_idx}')} {class_node(class_idx, f'gvp{feature_idx}')} 0 0 NSENSE W=16u L=180n",
        f"M{prefix}rgn_pair_base {class_node(class_idx, f'rgn{feature_idx}')} {class_node(class_idx, f'gvn{feature_idx}')} 0 0 NSENSE W=16u L=180n",
    ]
    for gate_idx, gate_node in enumerate(losing_gate_nodes):
        lines += [
            f"M{prefix}gvp_pair{gate_idx}_a vdd {activation_node} {prefix}gvp_pair{gate_idx}_a 0 NREL W={pairwise_width_u:.6g}u L=180n",
            f"M{prefix}gvp_pair{gate_idx}_label {prefix}gvp_pair{gate_idx}_a {class_node(class_idx, 'targetp')} {prefix}gvp_pair{gate_idx}_label 0 NSENSE W={pairwise_width_u:.6g}u L=180n",
            f"M{prefix}gvp_pair{gate_idx}_gate {prefix}gvp_pair{gate_idx}_label {gate_node} {prefix}gvp_pair{gate_idx}_gate 0 NSENSE W={pairwise_width_u:.6g}u L=180n",
            f"M{prefix}gvp_pair{gate_idx}_g {prefix}gvp_pair{gate_idx}_gate acc {class_node(class_idx, f'gvp{feature_idx}')} 0 NREL W={pairwise_width_u:.6g}u L=180n",
            f"M{prefix}rgp_pair{gate_idx}_a {class_node(class_idx, f'rgp{feature_idx}')} {activation_node} {prefix}rgp_pair{gate_idx}_a 0 NREL W={pairwise_width_u:.6g}u L=180n",
            f"M{prefix}rgp_pair{gate_idx}_label {prefix}rgp_pair{gate_idx}_a {class_node(class_idx, 'targetp')} {prefix}rgp_pair{gate_idx}_label 0 NSENSE W={pairwise_width_u:.6g}u L=180n",
            f"M{prefix}rgp_pair{gate_idx}_gate {prefix}rgp_pair{gate_idx}_label {gate_node} {prefix}rgp_pair{gate_idx}_gate 0 NSENSE W={pairwise_width_u:.6g}u L=180n",
            f"M{prefix}rgp_pair{gate_idx}_acc {prefix}rgp_pair{gate_idx}_gate acc 0 0 NREL W={pairwise_width_u:.6g}u L=180n",
        ]
    for gate_idx, gate_node in enumerate(winning_gate_nodes):
        lines += [
            f"M{prefix}gvn_pair{gate_idx}_a vdd {activation_node} {prefix}gvn_pair{gate_idx}_a 0 NREL W={pairwise_width_u:.6g}u L=180n",
            f"M{prefix}gvn_pair{gate_idx}_label {prefix}gvn_pair{gate_idx}_a {class_node(class_idx, 'targetn')} {prefix}gvn_pair{gate_idx}_label 0 NSENSE W={pairwise_width_u:.6g}u L=180n",
            f"M{prefix}gvn_pair{gate_idx}_gate {prefix}gvn_pair{gate_idx}_label {gate_node} {prefix}gvn_pair{gate_idx}_gate 0 NSENSE W={pairwise_width_u:.6g}u L=180n",
            f"M{prefix}gvn_pair{gate_idx}_g {prefix}gvn_pair{gate_idx}_gate acc {class_node(class_idx, f'gvn{feature_idx}')} 0 NREL W={pairwise_width_u:.6g}u L=180n",
            f"M{prefix}rgn_pair{gate_idx}_a {class_node(class_idx, f'rgn{feature_idx}')} {activation_node} {prefix}rgn_pair{gate_idx}_a 0 NREL W={pairwise_width_u:.6g}u L=180n",
            f"M{prefix}rgn_pair{gate_idx}_label {prefix}rgn_pair{gate_idx}_a {class_node(class_idx, 'targetn')} {prefix}rgn_pair{gate_idx}_label 0 NSENSE W={pairwise_width_u:.6g}u L=180n",
            f"M{prefix}rgn_pair{gate_idx}_gate {prefix}rgn_pair{gate_idx}_label {gate_node} {prefix}rgn_pair{gate_idx}_gate 0 NSENSE W={pairwise_width_u:.6g}u L=180n",
            f"M{prefix}rgn_pair{gate_idx}_acc {prefix}rgn_pair{gate_idx}_gate acc 0 0 NREL W={pairwise_width_u:.6g}u L=180n",
        ]
    if not losing_gate_nodes:
        lines += [
            f"M{prefix}gvp_a vdd {activation_node} {prefix}gvp_a 0 NREL W={width_u:.6g}u L=180n",
            f"M{prefix}gvp_d {prefix}gvp_a {class_node(class_idx, 'targetp')} {prefix}gvp_d 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}gvp_g {prefix}gvp_d acc {class_node(class_idx, f'gvp{feature_idx}')} 0 NREL W={width_u:.6g}u L=180n",
        ]
    return lines


def class_local_pairwise_binary_descent_correction_lines(
    *,
    class_idx: int,
    feature_idx: int,
    activation_node: str,
    losing_gate_nodes: list[str],
    winning_gate_nodes: list[str],
    pairwise_width_u: float = 16.0,
) -> list[str]:
    prefix = f"c{class_idx}_f{feature_idx}_"
    lines: list[str] = []
    for gate_idx, gate_node in enumerate(losing_gate_nodes):
        lines += [
            f"M{prefix}gvp_paircorr{gate_idx}_a vdd {activation_node} {prefix}gvp_paircorr{gate_idx}_a 0 NREL W={pairwise_width_u:.6g}u L=180n",
            f"M{prefix}gvp_paircorr{gate_idx}_label {prefix}gvp_paircorr{gate_idx}_a {class_node(class_idx, 'targetp')} {prefix}gvp_paircorr{gate_idx}_label 0 NSENSE W={pairwise_width_u:.6g}u L=180n",
            f"M{prefix}gvp_paircorr{gate_idx}_gate {prefix}gvp_paircorr{gate_idx}_label {gate_node} {prefix}gvp_paircorr{gate_idx}_gate 0 NSENSE W={pairwise_width_u:.6g}u L=180n",
            f"M{prefix}gvp_paircorr{gate_idx}_g {prefix}gvp_paircorr{gate_idx}_gate acc {class_node(class_idx, f'gvp{feature_idx}')} 0 NREL W={pairwise_width_u:.6g}u L=180n",
            f"M{prefix}rgp_paircorr{gate_idx}_a {class_node(class_idx, f'rgp{feature_idx}')} {activation_node} {prefix}rgp_paircorr{gate_idx}_a 0 NREL W={pairwise_width_u:.6g}u L=180n",
            f"M{prefix}rgp_paircorr{gate_idx}_label {prefix}rgp_paircorr{gate_idx}_a {class_node(class_idx, 'targetp')} {prefix}rgp_paircorr{gate_idx}_label 0 NSENSE W={pairwise_width_u:.6g}u L=180n",
            f"M{prefix}rgp_paircorr{gate_idx}_gate {prefix}rgp_paircorr{gate_idx}_label {gate_node} {prefix}rgp_paircorr{gate_idx}_gate 0 NSENSE W={pairwise_width_u:.6g}u L=180n",
            f"M{prefix}rgp_paircorr{gate_idx}_acc {prefix}rgp_paircorr{gate_idx}_gate acc 0 0 NREL W={pairwise_width_u:.6g}u L=180n",
        ]
    for gate_idx, gate_node in enumerate(winning_gate_nodes):
        lines += [
            f"M{prefix}gvn_paircorr{gate_idx}_a vdd {activation_node} {prefix}gvn_paircorr{gate_idx}_a 0 NREL W={pairwise_width_u:.6g}u L=180n",
            f"M{prefix}gvn_paircorr{gate_idx}_label {prefix}gvn_paircorr{gate_idx}_a {class_node(class_idx, 'targetn')} {prefix}gvn_paircorr{gate_idx}_label 0 NSENSE W={pairwise_width_u:.6g}u L=180n",
            f"M{prefix}gvn_paircorr{gate_idx}_gate {prefix}gvn_paircorr{gate_idx}_label {gate_node} {prefix}gvn_paircorr{gate_idx}_gate 0 NSENSE W={pairwise_width_u:.6g}u L=180n",
            f"M{prefix}gvn_paircorr{gate_idx}_g {prefix}gvn_paircorr{gate_idx}_gate acc {class_node(class_idx, f'gvn{feature_idx}')} 0 NREL W={pairwise_width_u:.6g}u L=180n",
            f"M{prefix}rgn_paircorr{gate_idx}_a {class_node(class_idx, f'rgn{feature_idx}')} {activation_node} {prefix}rgn_paircorr{gate_idx}_a 0 NREL W={pairwise_width_u:.6g}u L=180n",
            f"M{prefix}rgn_paircorr{gate_idx}_label {prefix}rgn_paircorr{gate_idx}_a {class_node(class_idx, 'targetn')} {prefix}rgn_paircorr{gate_idx}_label 0 NSENSE W={pairwise_width_u:.6g}u L=180n",
            f"M{prefix}rgn_paircorr{gate_idx}_gate {prefix}rgn_paircorr{gate_idx}_label {gate_node} {prefix}rgn_paircorr{gate_idx}_gate 0 NSENSE W={pairwise_width_u:.6g}u L=180n",
            f"M{prefix}rgn_paircorr{gate_idx}_acc {prefix}rgn_paircorr{gate_idx}_gate acc 0 0 NREL W={pairwise_width_u:.6g}u L=180n",
        ]
    return lines


def class_local_restored_score_binary_correction_lines(
    *,
    class_idx: int,
    feature_idx: int,
    activation_node: str,
    positive_gate_node: str,
    negative_gate_node: str,
    width_u: float = 16.0,
) -> list[str]:
    prefix = f"c{class_idx}_f{feature_idx}_"
    return [
        f"M{prefix}gvp_bincorr_a vdd {activation_node} {prefix}gvp_bincorr_a 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}gvp_bincorr_label {prefix}gvp_bincorr_a {class_node(class_idx, 'targetp')} {prefix}gvp_bincorr_label 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}gvp_bincorr_gate {prefix}gvp_bincorr_label {negative_gate_node} {prefix}gvp_bincorr_gate 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}gvp_bincorr_g {prefix}gvp_bincorr_gate acc {class_node(class_idx, f'gvp{feature_idx}')} 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}rgp_bincorr_a {class_node(class_idx, f'rgp{feature_idx}')} {activation_node} {prefix}rgp_bincorr_a 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}rgp_bincorr_label {prefix}rgp_bincorr_a {class_node(class_idx, 'targetp')} {prefix}rgp_bincorr_label 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}rgp_bincorr_gate {prefix}rgp_bincorr_label {negative_gate_node} {prefix}rgp_bincorr_gate 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}rgp_bincorr_acc {prefix}rgp_bincorr_gate acc 0 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_bincorr_a vdd {activation_node} {prefix}gvn_bincorr_a 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_bincorr_label {prefix}gvn_bincorr_a {class_node(class_idx, 'targetn')} {prefix}gvn_bincorr_label 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_bincorr_gate {prefix}gvn_bincorr_label {positive_gate_node} {prefix}gvn_bincorr_gate 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_bincorr_g {prefix}gvn_bincorr_gate acc {class_node(class_idx, f'gvn{feature_idx}')} 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}rgn_bincorr_a {class_node(class_idx, f'rgn{feature_idx}')} {activation_node} {prefix}rgn_bincorr_a 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}rgn_bincorr_label {prefix}rgn_bincorr_a {class_node(class_idx, 'targetn')} {prefix}rgn_bincorr_label 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}rgn_bincorr_gate {prefix}rgn_bincorr_label {positive_gate_node} {prefix}rgn_bincorr_gate 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}rgn_bincorr_acc {prefix}rgn_bincorr_gate acc 0 0 NREL W={width_u:.6g}u L=180n",
    ]


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


def shared_score_common_reference_lines(
    *,
    class_count: int,
    common_node: str = "score_common",
    reset_node: str = "scorepre",
    resistance_ohm: float = 20000.0,
    capacitance_f: float = 4.0,
    source_node_template: str = "c{class_idx}_score_amp",
) -> list[str]:
    lines = [
        f"C{common_node} {common_node} 0 {capacitance_f:.12g}f IC=1.2",
        f"R{common_node}_leak {common_node} 0 1G",
        f"Mprecharge_{common_node} {common_node} {reset_node} vdd vdd PMOS W=4u L=180n",
    ]
    for class_idx in range(class_count):
        source_node = source_node_template.format(class_idx=class_idx)
        lines.append(
            f"R{common_node}_c{class_idx} {common_node} {source_node} {resistance_ohm:.12g}"
        )
    return lines


def shared_label_score_reference_lines(
    *,
    class_count: int,
    reference_node: str = "target_score_ref",
    reset_node: str = "scoregaterst",
    select_width_u: float = 24.0,
    capacitance_f: float = 4.0,
    source_node_template: str = "c{class_idx}_score_amp",
) -> list[str]:
    lines = [
        f"C{reference_node} {reference_node} 0 {capacitance_f:.12g}f IC=0",
        f"R{reference_node}_leak {reference_node} 0 1G",
        f"Mreset_{reference_node} {reference_node} {reset_node} 0 0 NMOS W=4u L=180n",
    ]
    for class_idx in range(class_count):
        source_node = source_node_template.format(class_idx=class_idx)
        lines.append(
            f"M{reference_node}_sel_c{class_idx} {reference_node} {class_node(class_idx, 'targetp')} {source_node} 0 NSENSE W={select_width_u:.6g}u L=180n"
        )
    return lines


def class_local_score_common_gate_lines(
    *,
    class_idx: int,
    common_node: str = "score_common",
    score_input_node: str | None = None,
    output_node: str | None = None,
    compare_clock: str = "scoredec",
    reset_node: str = "scoregaterst",
    pullup_width_u: float = 48.0,
    pulldown_width_u: float = 12.0,
    analog_model: str = "NREL",
) -> list[str]:
    prefix = f"c{class_idx}_"
    score_gate = class_node(class_idx, "score_common_gate") if output_node is None else output_node
    score_input = f"{prefix}score_amp" if score_input_node is None else score_input_node
    return [
        f"C{score_gate} {score_gate} 0 4f IC=0",
        f"R{score_gate} {score_gate} 0 1G",
        f"Mreset_{score_gate} {score_gate} {reset_node} 0 0 NMOS W=4u L=180n",
        f"M{prefix}score_common_gate_up_v vdd {score_input} {prefix}score_common_gate_up_i 0 {analog_model} W={pullup_width_u:.6g}u L=180n",
        f"M{prefix}score_common_gate_up_t {prefix}score_common_gate_up_i {compare_clock} {score_gate} 0 NSENSE W={pullup_width_u:.6g}u L=180n",
        f"M{prefix}score_common_gate_dn_v {score_gate} {common_node} {prefix}score_common_gate_dn_i 0 {analog_model} W={pulldown_width_u:.6g}u L=180n",
        f"M{prefix}score_common_gate_dn_t {prefix}score_common_gate_dn_i {compare_clock} 0 0 NSENSE W={pulldown_width_u:.6g}u L=180n",
    ]


def class_local_score_contrast_lines(
    *,
    class_idx: int,
    common_node: str = "score_common",
    score_input_node: str | None = None,
    output_node: str | None = None,
    compare_clock: str = "scoredec",
    reset_node: str = "scoregaterst",
    reset_reference_node: str = "score_contrast_ref",
    capacitance_f: float = 10.0,
    pullup_width_u: float = 192.0,
    pulldown_width_u: float = 24.0,
) -> list[str]:
    if min(capacitance_f, pullup_width_u, pulldown_width_u) <= 0.0:
        raise ValueError("score contrast capacitance and widths must be positive")
    prefix = f"c{class_idx}_"
    contrast = class_node(class_idx, "score_contrast") if output_node is None else output_node
    score_input = f"{prefix}score_amp" if score_input_node is None else score_input_node
    return [
        f"C{contrast} {contrast} 0 {capacitance_f:.12g}f IC=0.6",
        f"R{contrast} {contrast} 0 1G",
        f"Mreset_{contrast} {contrast} {reset_node} {reset_reference_node} 0 NMOS W=4u L=180n",
        f"R{prefix}score_contrast_up {prefix}score_contrast_up 0 1G",
        f"R{prefix}score_contrast_dn {prefix}score_contrast_dn 0 1G",
        f"M{prefix}score_contrast_up_v vdd {score_input} {prefix}score_contrast_up 0 NREL W={pullup_width_u:.6g}u L=180n",
        f"M{prefix}score_contrast_up_t {prefix}score_contrast_up {compare_clock} {contrast} 0 NSENSE W={pullup_width_u:.6g}u L=180n",
        f"M{prefix}score_contrast_dn_v {contrast} {common_node} {prefix}score_contrast_dn 0 NREL W={pulldown_width_u:.6g}u L=180n",
        f"M{prefix}score_contrast_dn_t {prefix}score_contrast_dn {compare_clock} 0 0 NSENSE W={pulldown_width_u:.6g}u L=180n",
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


def class_local_error_rail_gradient_lines(
    *,
    class_idx: int,
    feature_idx: int,
    activation_node: str,
    positive_error_node: str,
    negative_error_node: str,
    width_u: float = 24.0,
) -> list[str]:
    prefix = f"c{class_idx}_f{feature_idx}_"
    return [
        f"M{prefix}gvp_a vdd {activation_node} {prefix}gvp_a 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}gvp_e {prefix}gvp_a {positive_error_node} {prefix}gvp_d 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}gvp_g {prefix}gvp_d acc {class_node(class_idx, f'gvp{feature_idx}')} 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_a vdd {activation_node} {prefix}gvn_a 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_e {prefix}gvn_a {negative_error_node} {prefix}gvn_d 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_g {prefix}gvn_d acc {class_node(class_idx, f'gvn{feature_idx}')} 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}rgp_pd {class_node(class_idx, f'rgp{feature_idx}')} {class_node(class_idx, f'gvp{feature_idx}')} 0 0 NSENSE W=16u L=180n",
        f"M{prefix}rgn_pd {class_node(class_idx, f'rgn{feature_idx}')} {class_node(class_idx, f'gvn{feature_idx}')} 0 0 NSENSE W=16u L=180n",
    ]


def shared_score_mass_error_lines(
    *,
    class_count: int,
    score_input_template: str = "c{class_idx}_score_amp",
    mass_node: str = "score_nontarget_mass",
    mass_clock_node: str = "scoredec",
    error_clock_node: str = "scoreerr",
    reset_node: str = "scoregaterst",
    sum_width_u: float = 32.0,
    error_width_u: float = 32.0,
    mass_capacitance_f: float = 8.0,
    error_capacitance_f: float = 8.0,
) -> list[str]:
    if min(sum_width_u, error_width_u, mass_capacitance_f, error_capacitance_f) <= 0.0:
        raise ValueError("score-mass widths and capacitances must be positive")
    lines = [
        f"C{mass_node} {mass_node} 0 {mass_capacitance_f:.12g}f IC=0",
        f"R{mass_node} {mass_node} 0 1G",
        f"Mreset_{mass_node} {mass_node} {reset_node} 0 0 NMOS W=4u L=180n",
    ]
    for class_idx in range(class_count):
        score_node = score_input_template.format(class_idx=class_idx)
        errp = class_node(class_idx, "errp")
        errn = class_node(class_idx, "errn")
        lines += [
            f"C{errp} {errp} 0 {error_capacitance_f:.12g}f IC=0",
            f"C{errn} {errn} 0 {error_capacitance_f:.12g}f IC=0",
            f"R{errp} {errp} 0 1G",
            f"R{errn} {errn} 0 1G",
            f"Mreset_{errp} {errp} {reset_node} 0 0 NMOS W=4u L=180n",
            f"Mreset_{errn} {errn} {reset_node} 0 0 NMOS W=4u L=180n",
            f"Rmass_nt{class_idx}_a mass_nt{class_idx}_a 0 1G",
            f"Rmass_nt{class_idx}_s mass_nt{class_idx}_s 0 1G",
            f"Mmass_nt{class_idx}_label vdd {class_node(class_idx, 'targetn')} mass_nt{class_idx}_a 0 NSENSE W={sum_width_u:.6g}u L=180n",
            f"Mmass_nt{class_idx}_score mass_nt{class_idx}_a {score_node} mass_nt{class_idx}_s 0 NSENSE W={sum_width_u:.6g}u L=180n",
            f"Mmass_nt{class_idx}_clk mass_nt{class_idx}_s {mass_clock_node} {mass_node} 0 NSENSE W={sum_width_u:.6g}u L=180n",
            f"R{errp}_a {errp}_a 0 1G",
            f"R{errp}_m {errp}_m 0 1G",
            f"M{errp}_label vdd {class_node(class_idx, 'targetp')} {errp}_a 0 NSENSE W={error_width_u:.6g}u L=180n",
            f"M{errp}_mass {errp}_a {mass_node} {errp}_m 0 NSENSE W={error_width_u:.6g}u L=180n",
            f"M{errp}_clk {errp}_m {error_clock_node} {errp} 0 NSENSE W={error_width_u:.6g}u L=180n",
            f"R{errn}_a {errn}_a 0 1G",
            f"R{errn}_s {errn}_s 0 1G",
            f"M{errn}_label vdd {class_node(class_idx, 'targetn')} {errn}_a 0 NSENSE W={error_width_u:.6g}u L=180n",
            f"M{errn}_score {errn}_a {score_node} {errn}_s 0 NSENSE W={error_width_u:.6g}u L=180n",
            f"M{errn}_clk {errn}_s {error_clock_node} {errn} 0 NSENSE W={error_width_u:.6g}u L=180n",
        ]
    return lines


def pairwise_score_competition_error_lines(
    *,
    class_count: int,
    error_clock_node: str = "scoreerr",
    reset_node: str = "scoregaterst",
    error_width_u: float = 128.0,
    error_capacitance_f: float = 4.0,
    create_error_nodes: bool = True,
) -> list[str]:
    if min(error_width_u, error_capacitance_f) <= 0.0:
        raise ValueError("pairwise competition error widths and capacitances must be positive")
    lines: list[str] = []
    if create_error_nodes:
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
            prefix = f"t{target_idx}_o{opponent_idx}_"
            lines += [
                f"R{prefix}errp_sup {prefix}errp_sup 0 1G",
                f"R{prefix}errp_t {prefix}errp_t 0 1G",
                f"R{prefix}errp_w {prefix}errp_w 0 1G",
                f"C{prefix}errp_sup {prefix}errp_sup 0 0.1f IC=0",
                f"C{prefix}errp_t {prefix}errp_t 0 0.1f IC=0",
                f"C{prefix}errp_w {prefix}errp_w 0 0.1f IC=0",
                f"M{prefix}errp_sup {prefix}errp_sup {opposite_decision} vdd vdd PMOS W={error_width_u:.6g}u L=180n",
                f"M{prefix}errp_label {prefix}errp_sup {targetp} {prefix}errp_t 0 NSENSE W={error_width_u:.6g}u L=180n",
                f"M{prefix}errp_win {prefix}errp_t {decision} {prefix}errp_w 0 NSENSE W={error_width_u:.6g}u L=180n",
                f"M{prefix}errp_clk {prefix}errp_w {error_clock_node} {errp} 0 NSENSE W={error_width_u:.6g}u L=180n",
                f"R{prefix}errn_sup {prefix}errn_sup 0 1G",
                f"R{prefix}errn_t {prefix}errn_t 0 1G",
                f"R{prefix}errn_w {prefix}errn_w 0 1G",
                f"C{prefix}errn_sup {prefix}errn_sup 0 0.1f IC=0",
                f"C{prefix}errn_t {prefix}errn_t 0 0.1f IC=0",
                f"C{prefix}errn_w {prefix}errn_w 0 0.1f IC=0",
                f"M{prefix}errn_sup {prefix}errn_sup {opposite_decision} vdd vdd PMOS W={error_width_u:.6g}u L=180n",
                f"M{prefix}errn_label {prefix}errn_sup {targetp} {prefix}errn_t 0 NSENSE W={error_width_u:.6g}u L=180n",
                f"M{prefix}errn_win {prefix}errn_t {decision} {prefix}errn_w 0 NSENSE W={error_width_u:.6g}u L=180n",
                f"M{prefix}errn_clk {prefix}errn_w {error_clock_node} {errn} 0 NSENSE W={error_width_u:.6g}u L=180n",
            ]
    return lines


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


def class_local_amplified_score_competitive_gradient_lines(
    *,
    class_idx: int,
    feature_idx: int,
    activation_node: str,
    own_score_gate_node: str,
    opponent_score_gate_nodes: list[str],
    width_u: float = 24.0,
    boost_width_u: float = 192.0,
) -> list[str]:
    prefix = f"c{class_idx}_f{feature_idx}_"
    lines = class_local_residual_score_nontarget_gradient_lines(
        class_idx=class_idx,
        feature_idx=feature_idx,
        activation_node=activation_node,
        score_gate_node=own_score_gate_node,
        width_u=width_u,
    )
    for opponent_idx, opponent_gate in enumerate(opponent_score_gate_nodes):
        lines += [
            f"M{prefix}gvp_comp{opponent_idx}_a vdd {activation_node} {prefix}gvp_comp{opponent_idx}_a 0 NREL W={boost_width_u:.6g}u L=180n",
            f"M{prefix}gvp_comp{opponent_idx}_label {prefix}gvp_comp{opponent_idx}_a {class_node(class_idx, 'targetp')} {prefix}gvp_comp{opponent_idx}_label 0 NSENSE W={boost_width_u:.6g}u L=180n",
            f"M{prefix}gvp_comp{opponent_idx}_score {prefix}gvp_comp{opponent_idx}_label {opponent_gate} {prefix}gvp_comp{opponent_idx}_score 0 NSENSE W={boost_width_u:.6g}u L=180n",
            f"M{prefix}gvp_comp{opponent_idx}_g {prefix}gvp_comp{opponent_idx}_score acc {class_node(class_idx, f'gvp{feature_idx}')} 0 NREL W={boost_width_u:.6g}u L=180n",
            f"M{prefix}rgp_comp{opponent_idx}_a {class_node(class_idx, f'rgp{feature_idx}')} {activation_node} {prefix}rgp_comp{opponent_idx}_a 0 NREL W={boost_width_u:.6g}u L=180n",
            f"M{prefix}rgp_comp{opponent_idx}_label {prefix}rgp_comp{opponent_idx}_a {class_node(class_idx, 'targetp')} {prefix}rgp_comp{opponent_idx}_label 0 NSENSE W={boost_width_u:.6g}u L=180n",
            f"M{prefix}rgp_comp{opponent_idx}_score {prefix}rgp_comp{opponent_idx}_label {opponent_gate} {prefix}rgp_comp{opponent_idx}_score 0 NSENSE W={boost_width_u:.6g}u L=180n",
            f"M{prefix}rgp_comp{opponent_idx}_acc {prefix}rgp_comp{opponent_idx}_score acc 0 0 NREL W={boost_width_u:.6g}u L=180n",
        ]
    return lines


def class_local_restored_score_binary_descent_gradient_lines(
    *,
    class_idx: int,
    feature_idx: int,
    activation_node: str,
    positive_gate_node: str,
    negative_gate_node: str,
    width_u: float = 24.0,
) -> list[str]:
    prefix = f"c{class_idx}_f{feature_idx}_"
    return [
        f"M{prefix}gvp_a vdd {activation_node} {prefix}gvp_a 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}gvp_label {prefix}gvp_a {class_node(class_idx, 'targetp')} {prefix}gvp_label 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}gvp_decisionn {prefix}gvp_label {negative_gate_node} {prefix}gvp_decisionn 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}gvp_g {prefix}gvp_decisionn acc {class_node(class_idx, f'gvp{feature_idx}')} 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_a vdd {activation_node} {prefix}gvn_a 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_label {prefix}gvn_a {class_node(class_idx, 'targetn')} {prefix}gvn_label 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_decision {prefix}gvn_label {positive_gate_node} {prefix}gvn_decision 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_g {prefix}gvn_decision acc {class_node(class_idx, f'gvn{feature_idx}')} 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}rgp_pd {class_node(class_idx, f'rgp{feature_idx}')} {class_node(class_idx, f'gvp{feature_idx}')} 0 0 NSENSE W=16u L=180n",
        f"M{prefix}rgn_pd {class_node(class_idx, f'rgn{feature_idx}')} {class_node(class_idx, f'gvn{feature_idx}')} 0 0 NSENSE W=16u L=180n",
    ]


def class_local_target_only_gradient_lines(
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
        f"M{prefix}rgp_pd {class_node(class_idx, f'rgp{feature_idx}')} {class_node(class_idx, f'gvp{feature_idx}')} 0 0 NSENSE W=16u L=180n",
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
    score_measure_ns: float = 8.5,
    initial_positive: float = 0.40,
    initial_negative: float = 0.40,
    target_high: float = 1.1,
    nontarget_scale: float = 1.0,
    nontarget_width_scale: float = 1.0,
    score_mass_sum_width: float = 32.0,
    score_mass_error_width: float = 32.0,
    score_mass_pairwise_error_scale: float = 0.0625,
    pairwise_margin_target_v: float = 1.0e-3,
    pairwise_margin_error_drive_scale: float = 1.0,
    normalizer_error_clock_high: float = 1.2,
    error_mode: str = "label-descent",
    readout_update_mode: str = "sampled",
    hidden_update_mode: str = "none",
    hidden_credit_width_u: float = 8.0,
    hidden_update_width_u: float = 0.25,
    score_timing_mode: str = "late",
    score_sense_mode: str = "voltage",
    readout_forward_mode: str = "direct",
    eligibility_gate_mode: str = "raw",
    eligibility_contrast_common_resistance_ohm: float = 1.0e6,
    eligibility_contrast_common_capacitance_f: float = 1.0,
    class_bias_mode: str = "none",
    class_bias_input: float = 0.85,
    readout_center_resistance: float = 0.0,
    readout_center_voltage: float = 0.40,
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
        "score_measure_ns": score_measure_ns,
        "initial_positive": initial_positive,
        "initial_negative": initial_negative,
        "target_high": target_high,
        "score_mass_sum_width": score_mass_sum_width,
        "score_mass_error_width": score_mass_error_width,
        "score_mass_pairwise_error_scale": score_mass_pairwise_error_scale,
        "pairwise_margin_target_v": pairwise_margin_target_v,
        "pairwise_margin_error_drive_scale": pairwise_margin_error_drive_scale,
        "normalizer_error_clock_high": normalizer_error_clock_high,
        "hidden_credit_width_u": hidden_credit_width_u,
        "hidden_update_width_u": hidden_update_width_u,
        "eligibility_contrast_common_resistance_ohm": eligibility_contrast_common_resistance_ohm,
        "eligibility_contrast_common_capacitance_f": eligibility_contrast_common_capacitance_f,
    }.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if score_measure_ns >= CYCLE_NS:
        raise ValueError("score_measure_ns must be inside the cycle")
    if readout_center_resistance < 0.0:
        raise ValueError("readout_center_resistance must be nonnegative")
    if readout_center_voltage < 0.0 or readout_center_voltage > 1.2:
        raise ValueError("readout_center_voltage must stay within supply rails")
    if nontarget_scale < 0.0 or nontarget_scale > 1.0:
        raise ValueError("nontarget_scale must be in [0, 1]")
    if nontarget_width_scale < 0.0 or nontarget_width_scale > 1.0:
        raise ValueError("nontarget_width_scale must be in [0, 1]")
    if error_mode not in ERROR_MODES:
        raise ValueError(f"error_mode must be one of {ERROR_MODES}")
    if readout_update_mode not in READOUT_UPDATE_MODES:
        raise ValueError(f"readout_update_mode must be one of {READOUT_UPDATE_MODES}")
    if hidden_update_mode not in HIDDEN_UPDATE_MODES:
        raise ValueError(f"hidden_update_mode must be one of {HIDDEN_UPDATE_MODES}")
    if score_timing_mode not in SCORE_TIMING_MODES:
        raise ValueError(f"score_timing_mode must be one of {SCORE_TIMING_MODES}")
    if score_sense_mode not in SCORE_SENSE_MODES:
        raise ValueError(f"score_sense_mode must be one of {SCORE_SENSE_MODES}")
    if readout_forward_mode not in READOUT_FORWARD_MODES:
        raise ValueError(f"readout_forward_mode must be one of {READOUT_FORWARD_MODES}")
    if eligibility_gate_mode not in ELIGIBILITY_GATE_MODES:
        raise ValueError(f"eligibility_gate_mode must be one of {ELIGIBILITY_GATE_MODES}")
    if score_sense_mode == "current-clamp" and error_mode != "label-descent":
        raise ValueError("current-clamp score_sense_mode is only a label-descent readout diagnostic")
    if hidden_update_mode != "none" and error_mode not in ERROR_RAIL_DESCENT_MODES:
        raise ValueError("hidden_update_mode requires an error-rail descent mode")
    if class_bias_mode not in CLASS_BIAS_MODES:
        raise ValueError(f"class_bias_mode must be one of {CLASS_BIAS_MODES}")
    if class_bias_input < 0.0 or class_bias_input > 1.2:
        raise ValueError("class_bias_input must stay within supply rails")
    if max(hidden_positive, hidden_negative, initial_positive, initial_negative, target_high) > 1.2:
        raise ValueError("voltages must stay within supply rails")

    all_records = eval_records + train_records + eval_records
    sequence = ["initial_eval"] * len(eval_records) + ["train"] * len(train_records) + ["final_eval"] * len(eval_records)
    train_cycles = {idx for idx, label in enumerate(sequence) if label == "train"}
    cycle_count = len(all_records)
    uses_live_writer = readout_update_mode == "live" or hidden_update_mode != "none"
    live_writer_reset_cycles = set(range(cycle_count)) if uses_live_writer else train_cycles
    labels = [int(record["label"]) for record in all_records]
    if any(label < 0 or label >= class_count for label in labels):
        raise ValueError("record labels must be valid class indices")
    features = records_to_feature_matrix(all_records, feature_count)
    has_class_bias = class_bias_mode != "none"
    bias_feature = feature_count if has_class_bias else None
    total_feature_count = feature_count + (1 if has_class_bias else 0)
    stop_ns = cycle_count * CYCLE_NS
    uses_restored_score = error_mode == "restored-score-nontarget"
    uses_residual_score = error_mode == "residual-score-nontarget"
    uses_amplified_score = error_mode == "amplified-score-nontarget"
    uses_common_ref_score = error_mode == "common-ref-score-nontarget"
    uses_raw_common_ref_score = error_mode == "raw-common-ref-score-nontarget"
    uses_target_ref_score = error_mode == "target-ref-score-nontarget"
    uses_common_score_mass = error_mode == "common-score-mass-descent"
    uses_contrast_score_mass = error_mode == "contrast-score-mass-descent"
    uses_low_gain_contrast_score_mass = error_mode == "low-gain-contrast-score-mass-descent"
    uses_contrast_gated_score_mass = error_mode == "contrast-gated-score-mass-descent"
    uses_target_contrast_score_mass = error_mode == "target-contrast-score-mass-descent"
    uses_common_score_mass_pairwise = error_mode == "common-score-mass-pairwise-descent"
    uses_pairwise_score_competition = error_mode == "pairwise-score-competition-descent"
    uses_pairwise_margin_correction = error_mode == "pairwise-margin-correction-descent"
    uses_normalizer_error = error_mode in NORMALIZER_ERROR_MODES
    uses_hidden_update = hidden_update_mode == "readout-weighted"
    uses_label_rail_descent = error_mode == "label-rail-descent"
    normalizer_approach = normalizer_error_approach(error_mode) if uses_normalizer_error else None
    uses_score_common_gate = uses_common_ref_score or uses_raw_common_ref_score
    uses_score_common_gate_nodes = uses_score_common_gate or uses_common_score_mass or uses_common_score_mass_pairwise
    uses_score_contrast_nodes = (
        uses_contrast_score_mass or uses_low_gain_contrast_score_mass or uses_contrast_gated_score_mass
    )
    uses_amplified_competitive = error_mode == "amplified-score-competitive"
    uses_amplified_pairwise = error_mode == "amplified-score-pairwise"
    uses_amplified_binary = error_mode == "amplified-score-binary-descent"
    uses_score_mass = error_mode == "score-mass-descent"
    uses_pairwise_binary = error_mode == "pairwise-binary-descent"
    uses_restored_binary = error_mode == "restored-score-binary-descent"
    uses_score_preamp = (
        uses_residual_score
        or uses_amplified_score
        or uses_common_ref_score
        or uses_amplified_competitive
        or uses_amplified_pairwise
        or uses_amplified_binary
        or uses_target_ref_score
        or uses_score_mass
        or uses_common_score_mass
        or uses_contrast_score_mass
        or uses_low_gain_contrast_score_mass
        or uses_contrast_gated_score_mass
        or uses_target_contrast_score_mass
        or uses_common_score_mass_pairwise
        or uses_pairwise_score_competition
        or uses_pairwise_margin_correction
        or uses_normalizer_error
    )
    uses_restored_winner = error_mode == "restored-winner-nontarget"
    uses_pairwise_decisions = (
        uses_restored_winner
        or uses_pairwise_binary
        or uses_amplified_pairwise
        or uses_pairwise_score_competition
        or uses_pairwise_margin_correction
        or uses_common_score_mass_pairwise
    )
    uses_late_restored_gate = (
        uses_restored_score
        or uses_restored_binary
        or uses_pairwise_binary
        or uses_restored_winner
        or uses_pairwise_score_competition
        or uses_pairwise_margin_correction
        or uses_common_score_mass_pairwise
        or uses_score_preamp
        or uses_raw_common_ref_score
        or uses_normalizer_error
    )
    uses_score_error_clock = (
        uses_label_rail_descent
        or uses_score_mass
        or uses_common_score_mass
        or uses_contrast_score_mass
        or uses_low_gain_contrast_score_mass
        or uses_contrast_gated_score_mass
        or uses_target_contrast_score_mass
        or uses_common_score_mass_pairwise
        or uses_pairwise_score_competition
        or uses_pairwise_margin_correction
        or uses_normalizer_error
    )
    uses_score_based_target_timing = (
        uses_target_ref_score
        or uses_score_error_clock
    )
    if score_timing_mode == "early":
        out_start_ns = 5.0
        out_end_ns = 5.35
        scorepre_start_ns = 5.45
        scorepre_end_ns = 5.70
        scoreamp_start_ns = 5.95
        scoreamp_end_ns = 6.75
        scoredec_start_ns = 6.95
        scoredec_end_ns = 7.65
        scoreerr_start_ns = 7.80
        scoreerr_end_ns = 8.10
    else:
        out_start_ns = 5.0
        out_end_ns = 8.0
        scorepre_start_ns = 8.10
        scorepre_end_ns = 8.35
        scoreamp_start_ns = 8.60
        scoreamp_end_ns = 9.50
        scoredec_start_ns = 9.70
        scoredec_end_ns = 10.40
        scoreerr_start_ns = 10.73 if uses_contrast_gated_score_mass else 10.45
        scoreerr_end_ns = 10.79 if uses_contrast_gated_score_mass else 10.75
    target_start_ns = (
        max(0.0, scoredec_start_ns - 0.15)
        if uses_score_based_target_timing
        else 10.8 if uses_late_restored_gate else 9.0
    )
    target_end_ns = (
        max(scoreerr_end_ns + 0.20, target_start_ns + 0.50)
        if uses_score_based_target_timing
        else 12.8 if uses_late_restored_gate else 11.0
    )
    acc_start_ns = scoreerr_start_ns if uses_score_error_clock else 10.8 if uses_late_restored_gate else 9.0
    acc_end_ns = target_end_ns if uses_score_error_clock else 12.8 if uses_late_restored_gate else 11.0
    apply_start_ns = acc_end_ns + 0.20 if uses_score_error_clock else 13.0 if uses_late_restored_gate else 12.0
    apply_end_ns = apply_start_ns + 0.10
    score_amp_measure_ns = scoreamp_end_ns + 0.10
    score_decision_measure_ns = scoredec_end_ns + 0.20
    score_error_measure_ns = scoreerr_end_ns + 0.03
    score_mass_measure_ns = max(scorepre_end_ns, scoreerr_start_ns - 0.03)
    low_gain_contrast_sizing = (
        derive_class_evidence_normalizer_sizing(
            class_count=class_count,
            normalized_score_delta_v=4.86e-3,
            error_drive_scale=score_mass_error_width / 32.0,
        )
        if uses_low_gain_contrast_score_mass
        else None
    )
    margin_correction_sizing = (
        derive_multiclass_margin_correction_sizing(
            class_count=class_count,
            target_margin_v=pairwise_margin_target_v,
            score_delta_v=3.0e-3,
            error_drive_scale=pairwise_margin_error_drive_scale,
        )
        if uses_pairwise_margin_correction
        else None
    )
    if readout_update_mode == "live" and error_mode != "label-descent" and error_mode not in ERROR_RAIL_DESCENT_MODES:
        raise ValueError("live readout_update_mode requires label-descent or error-rail descent modes")

    lines = [
        "* Continuous multiclass block sequence: split-rail hidden features and class-local readout updates.",
        "* Python supplies row inputs, labels, and clocks only; readout weights persist as capacitor state.",
        "* No behavioral sources.",
        ".param VDD=1.2",
        mos_models(),
        *(
            [".model NHIGH NMOS LEVEL=1 VTO=0.75 KP=220u LAMBDA=0.03 GAMMA=0.20 PHI=0.60"]
            if eligibility_gate_mode in ("competition", "rank")
            else []
        ),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        "Vvwhi_ref vwhi_ref 0 0.42",
        "Vvwlo_ref vwlo_ref 0 0.28",
        f"Vrst rst 0 {periodic_phase_pwl(cycle_count, start_ns=0.2, end_ns=1.0)}",
        f"Vsamp samp 0 {periodic_phase_pwl(cycle_count, start_ns=2.5, end_ns=3.5)}",
        f"Vsampn sampn 0 {active_low_phase_pwl(cycle_count, start_ns=2.5, end_ns=3.5, active_cycles=set(range(cycle_count)))}",
        *(
            [
                f"Veligpre eligpre 0 {periodic_phase_pwl(cycle_count, start_ns=3.55, end_ns=target_end_ns + 0.20, active_cycles=train_cycles)}",
                f"Veligdec eligdec 0 {periodic_phase_pwl(cycle_count, start_ns=3.65, end_ns=4.35, active_cycles=train_cycles)}",
                f"Veliggate eliggate 0 {periodic_phase_pwl(cycle_count, start_ns=4.45, end_ns=4.75, active_cycles=train_cycles)}",
            ]
            if eligibility_gate_mode in ("competition", "rank")
            else []
        ),
        *(
            [f"Veliggate eliggate 0 {periodic_phase_pwl(cycle_count, start_ns=4.45, end_ns=4.75, active_cycles=train_cycles)}"]
            if eligibility_gate_mode == "contrast"
            else []
        ),
        f"Vout out 0 {periodic_phase_pwl(cycle_count, start_ns=out_start_ns, end_ns=out_end_ns)}",
        f"Voutn outn 0 {active_low_phase_pwl(cycle_count, start_ns=out_start_ns, end_ns=out_end_ns, active_cycles=set(range(cycle_count)))}",
    ]
    if readout_update_mode == "sampled":
        lines += [
            f"Vacc acc 0 {periodic_phase_pwl(cycle_count, start_ns=acc_start_ns, end_ns=acc_end_ns, active_cycles=train_cycles)}",
            f"Vapply apply 0 {periodic_phase_pwl(cycle_count, start_ns=apply_start_ns, end_ns=apply_end_ns, active_cycles=train_cycles)}",
            f"Vapplyn applyn 0 {active_low_phase_pwl(cycle_count, start_ns=apply_start_ns, end_ns=apply_end_ns, active_cycles=train_cycles)}",
        ]
    if readout_center_resistance > 0.0:
        lines.append(f"Vreadout_center readout_center 0 {readout_center_voltage:.12g}")
    if uses_late_restored_gate:
        lines += [
            "Voutref outref 0 0.25",
            *(["Vscore_contrast_ref score_contrast_ref 0 0.6"] if uses_score_contrast_nodes else []),
            f"Vscorepre scorepre 0 {active_low_phase_pwl(cycle_count, start_ns=scorepre_start_ns, end_ns=scorepre_end_ns, active_cycles=train_cycles)}",
            f"Vscoreamp scoreamp 0 {periodic_phase_pwl(cycle_count, start_ns=scoreamp_start_ns, end_ns=scoreamp_end_ns, active_cycles=train_cycles)}",
            f"Vscoredec scoredec 0 {periodic_phase_pwl(cycle_count, start_ns=scoredec_start_ns, end_ns=scoredec_end_ns, active_cycles=train_cycles)}",
            *(
                [
                    f"Vscoregate scoregate 0 {periodic_phase_pwl(cycle_count, start_ns=scoredec_end_ns + 0.02, end_ns=scoredec_end_ns + 0.15, active_cycles=train_cycles)}"
                ]
                if uses_contrast_gated_score_mass
                else []
            ),
            *(
                [
                    f"Vscoremass scoremass 0 {periodic_phase_pwl(cycle_count, start_ns=scoredec_end_ns + 0.18, end_ns=scoredec_end_ns + 0.30, active_cycles=train_cycles)}"
                ]
                if uses_contrast_gated_score_mass
                else []
            ),
        ]
    if uses_score_error_clock:
        scoreerr_high = (
            margin_correction_sizing.error_clock_high_v
            if margin_correction_sizing is not None
            else normalizer_error_clock_high if uses_normalizer_error else 1.2
        )
        lines.append(
            f"Vscoreerr scoreerr 0 {periodic_phase_pwl(cycle_count, start_ns=scoreerr_start_ns, end_ns=scoreerr_end_ns, active_cycles=train_cycles, high=scoreerr_high)}"
        )
    if (
        uses_residual_score
        or uses_score_common_gate_nodes
        or uses_score_contrast_nodes
        or uses_target_ref_score
        or uses_score_mass
        or uses_contrast_score_mass
        or uses_low_gain_contrast_score_mass
        or uses_contrast_gated_score_mass
        or uses_target_contrast_score_mass
        or uses_common_score_mass_pairwise
        or uses_pairwise_score_competition
        or uses_pairwise_margin_correction
        or uses_normalizer_error
        or uses_label_rail_descent
    ):
        scoregaterst_start_ns = 0.2 if uses_live_writer else scorepre_start_ns
        scoregaterst_end_ns = 1.0 if uses_live_writer else scorepre_end_ns
        lines.append(
            f"Vscoregaterst scoregaterst 0 {periodic_phase_pwl(cycle_count, start_ns=scoregaterst_start_ns, end_ns=scoregaterst_end_ns, active_cycles=live_writer_reset_cycles)}"
        )
    for feature in range(total_feature_count):
        row_values = (
            [class_bias_input] * cycle_count
            if bias_feature is not None and feature == bias_feature
            else [float(row[feature]) for row in features]
        )
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
            *(
                [
                    f"Cxelig{feature} xelig{feature} 0 20f IC=0",
                    f"Rxelig{feature} xelig{feature} 0 1G",
                    f"Mxelig{feature}_rst xelig{feature} rst 0 0 NMOS W=4u L=180n",
                ]
                if uses_hidden_update
                else []
            ),
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
            *(
                [
                    f"Mxelig{feature}_n xelig{feature} samp row{feature} 0 NMOS W=16u L=180n",
                    f"Mxelig{feature}_p xelig{feature} sampn row{feature} vdd PMOS W=32u L=180n",
                ]
                if uses_hidden_update
                else []
            ),
            f"Mactrow{feature}_n actrow{feature} out act{feature} 0 NMOS W=16u L=180n",
            f"Mactrow{feature}_p actrow{feature} outn act{feature} vdd PMOS W=32u L=180n",
        ]
    if eligibility_gate_mode == "contrast":
        lines += shared_eligibility_common_reference_lines(
            feature_count=feature_count,
            resistance_ohm=eligibility_contrast_common_resistance_ohm,
            capacitance_f=eligibility_contrast_common_capacitance_f,
        )
        lines += feature_common_contrast_gate_lines(
            feature_count=feature_count,
            compare_clock_node="eliggate",
            reset_node="rst",
        )
    elif eligibility_gate_mode in ("competition", "rank"):
        for feature_a in range(feature_count):
            for feature_b in range(feature_a + 1, feature_count):
                lines += pairwise_feature_winner_lines(
                    feature_a=feature_a,
                    feature_b=feature_b,
                    decision_clock_node="eligdec",
                    reset_node="eligpre",
                    width_u=32.0,
                )
        if eligibility_gate_mode == "rank":
            lines += feature_rank_suppression_lines(
                feature_count=feature_count,
                gate_clock_node="eliggate",
                reset_node="eligpre",
                gate_capacitance_f=50.0,
                loss_width_u=0.50,
            )
        else:
            lines += feature_loss_suppression_lines(
                feature_count=feature_count,
                gate_clock_node="eliggate",
                reset_node="eligpre",
                gate_capacitance_f=8.0,
                loss_width_u=32.0,
            )
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
        ]
        if score_sense_mode == "current-clamp":
            lines += [
                f"V{class_node(class_idx, 'score_clamp')} {class_node(class_idx, 'score')} 0 0",
                f"V{class_node(class_idx, 'scoren_clamp')} {class_node(class_idx, 'scoren')} 0 0",
            ]
        else:
            lines += [
                f"C{class_node(class_idx, 'score')} {class_node(class_idx, 'score')} 0 {score_capacitance_f:.12g}f IC=0",
                f"C{class_node(class_idx, 'scoren')} {class_node(class_idx, 'scoren')} 0 {score_capacitance_f:.12g}f IC=0",
                f"R{class_node(class_idx, 'score')} {class_node(class_idx, 'score')} 0 {score_load_resistance:.12g}",
                f"R{class_node(class_idx, 'scoren')} {class_node(class_idx, 'scoren')} 0 {score_load_resistance:.12g}",
                f"Mreset_{class_node(class_idx, 'score')} {class_node(class_idx, 'score')} rst 0 0 NMOS W=4u L=180n",
                f"Mreset_{class_node(class_idx, 'scoren')} {class_node(class_idx, 'scoren')} rst 0 0 NMOS W=4u L=180n",
            ]
        if uses_restored_score or uses_restored_binary or uses_score_preamp:
            prefix = f"c{class_idx}_"
            lines += [
                *low_gain_ref_state_lines(prefix=prefix, reset_node="scorepre"),
            ]
            if uses_restored_score or uses_restored_binary or uses_amplified_binary:
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
        if uses_raw_common_ref_score:
            lines += class_local_score_common_gate_lines(
                class_idx=class_idx,
                score_input_node=class_node(class_idx, "score"),
                analog_model="NSENSE",
            )
        elif uses_common_ref_score:
            lines += class_local_score_common_gate_lines(class_idx=class_idx)
        elif uses_common_score_mass:
            lines += class_local_score_common_gate_lines(
                class_idx=class_idx,
                pullup_width_u=192.0,
                pulldown_width_u=6.0,
            )
        elif uses_common_score_mass_pairwise:
            lines += class_local_score_common_gate_lines(
                class_idx=class_idx,
                pullup_width_u=192.0,
                pulldown_width_u=6.0,
            )
        elif uses_contrast_score_mass or uses_low_gain_contrast_score_mass or uses_contrast_gated_score_mass:
            lines += class_local_score_contrast_lines(class_idx=class_idx)
            if uses_contrast_gated_score_mass:
                lines += class_local_score_common_gate_lines(
                    class_idx=class_idx,
                    common_node="score_contrast_ref",
                    score_input_node=class_node(class_idx, "score_contrast"),
                    output_node=class_node(class_idx, "score_contrast_gate"),
                    compare_clock="scoregate",
                    pullup_width_u=192.0,
                    pulldown_width_u=24.0,
                )
        elif uses_target_ref_score:
            lines += class_local_score_common_gate_lines(
                class_idx=class_idx,
                common_node="target_score_ref",
                output_node=class_node(class_idx, "score_target_gate"),
            )
        elif uses_target_contrast_score_mass:
            lines += class_local_score_common_gate_lines(
                class_idx=class_idx,
                common_node="target_score_ref",
                output_node=class_node(class_idx, "score_target_gate"),
                pullup_width_u=192.0,
                pulldown_width_u=6.0,
            )
        if uses_pairwise_decisions:
            for opponent_idx in range(class_idx + 1, class_count):
                if uses_pairwise_score_competition or uses_pairwise_margin_correction or uses_common_score_mass_pairwise:
                    lines += pairwise_low_gain_winner_lines(class_a=class_idx, class_b=opponent_idx)
                else:
                    lines += pairwise_winner_lines(class_a=class_idx, class_b=opponent_idx)
    if uses_raw_common_ref_score:
        lines += shared_score_common_reference_lines(
            class_count=class_count,
            source_node_template="c{class_idx}_score",
        )
    elif (
        uses_common_ref_score
        or uses_common_score_mass
        or uses_common_score_mass_pairwise
        or uses_contrast_score_mass
        or uses_low_gain_contrast_score_mass
        or uses_contrast_gated_score_mass
    ):
        lines += shared_score_common_reference_lines(
            class_count=class_count,
            resistance_ohm=(
                low_gain_contrast_sizing.score_common_resistance_ohm
                if low_gain_contrast_sizing is not None
                else 20000.0
            ),
        )
    if uses_target_ref_score or uses_target_contrast_score_mass:
        lines += shared_label_score_reference_lines(class_count=class_count)
    if uses_label_rail_descent:
        lines += class_local_label_error_rail_lines(class_count=class_count)
    if uses_score_mass:
        lines += shared_score_mass_error_lines(
            class_count=class_count,
            score_input_template="c{class_idx}_score_amp",
            sum_width_u=score_mass_sum_width,
            error_width_u=score_mass_error_width,
        )
    if uses_common_score_mass:
        lines += shared_score_mass_error_lines(
            class_count=class_count,
            score_input_template="c{class_idx}_score_common_gate",
            sum_width_u=4.0 * score_mass_sum_width,
            error_width_u=4.0 * score_mass_error_width,
            mass_capacitance_f=0.5,
            error_capacitance_f=0.5,
        )
    if uses_contrast_score_mass:
        lines += shared_score_mass_error_lines(
            class_count=class_count,
            score_input_template="c{class_idx}_score_contrast",
            sum_width_u=4.0 * score_mass_sum_width,
            error_width_u=4.0 * score_mass_error_width,
            mass_capacitance_f=0.5,
            error_capacitance_f=0.5,
        )
    if uses_low_gain_contrast_score_mass:
        lines += shared_score_mass_error_lines(
            class_count=class_count,
            score_input_template="c{class_idx}_score_contrast",
            sum_width_u=low_gain_contrast_sizing.mass_width_u,
            error_width_u=low_gain_contrast_sizing.target_error_width_u,
            mass_capacitance_f=low_gain_contrast_sizing.mass_cap_f,
            error_capacitance_f=low_gain_contrast_sizing.error_cap_f,
        )
    if uses_contrast_gated_score_mass:
        lines += shared_score_mass_error_lines(
            class_count=class_count,
            score_input_template="c{class_idx}_score_contrast_gate",
            mass_clock_node="scoremass",
            sum_width_u=4.0 * score_mass_sum_width,
            error_width_u=4.0 * score_mass_error_width,
            mass_capacitance_f=0.5,
            error_capacitance_f=0.5,
        )
    if uses_common_score_mass_pairwise:
        lines += shared_score_mass_error_lines(
            class_count=class_count,
            score_input_template="c{class_idx}_score_common_gate",
            sum_width_u=4.0 * score_mass_sum_width,
            error_width_u=4.0 * score_mass_error_width,
            mass_capacitance_f=0.5,
            error_capacitance_f=0.5,
        )
        lines += pairwise_score_competition_error_lines(
            class_count=class_count,
            error_width_u=score_mass_pairwise_error_scale * score_mass_error_width,
            error_capacitance_f=0.5,
            create_error_nodes=False,
        )
    if uses_target_contrast_score_mass:
        lines += shared_score_mass_error_lines(
            class_count=class_count,
            score_input_template="c{class_idx}_score_target_gate",
            sum_width_u=4.0 * score_mass_sum_width,
            error_width_u=4.0 * score_mass_error_width,
            mass_capacitance_f=0.5,
            error_capacitance_f=0.5,
        )
    if uses_pairwise_margin_correction:
        lines += pairwise_target_margin_penalty_lines(
            class_count=class_count,
            penalty_width_u=margin_correction_sizing.margin_penalty_width_u,
        )
        lines += pairwise_score_competition_error_lines(
            class_count=class_count,
            error_width_u=margin_correction_sizing.error_width_u,
            error_capacitance_f=margin_correction_sizing.error_cap_f,
        )
    if uses_pairwise_score_competition:
        lines += pairwise_score_competition_error_lines(
            class_count=class_count,
            error_width_u=score_mass_error_width,
            error_capacitance_f=0.5,
        )
    if uses_normalizer_error:
        lines += [
            normalization_subcircuits(approaches=(normalizer_approach,)),
            (
                "Xscore_normalizer "
                + " ".join(class_node(class_idx, "score") for class_idx in range(class_count))
                + " "
                + " ".join(class_node(class_idx, "targetp") for class_idx in range(class_count))
                + " "
                + " ".join(class_node(class_idx, "targetn") for class_idx in range(class_count))
                + " scoreerr scoregaterst "
                + " ".join(
                    f"{class_node(class_idx, 'errp')} {class_node(class_idx, 'errn')}"
                    for class_idx in range(class_count)
                )
                + f" vdd 0 {spice_subckt_name(normalizer_approach)}"
            ),
        ]
    if uses_hidden_update:
        error_positive_nodes = [class_node(class_idx, "errp") for class_idx in range(class_count)]
        error_negative_nodes = [class_node(class_idx, "errn") for class_idx in range(class_count)]
        for feature in range(feature_count):
            lines += hidden_readout_weighted_credit_lines(
                class_count=class_count,
                feature_idx=feature,
                error_positive_nodes=error_positive_nodes,
                error_negative_nodes=error_negative_nodes,
                width_u=hidden_credit_width_u,
            )
            lines += hidden_live_weight_update_lines(
                feature_idx=feature,
                eligibility_node=f"xelig{feature}",
                positive_credit_node=hidden_credit_node(feature, "hdp"),
                negative_credit_node=hidden_credit_node(feature, "hdn"),
                width_u=hidden_update_width_u,
            )
    for class_idx in range(class_count):
        for feature in range(total_feature_count):
            activation_node = (
                eligibility_gate_node(feature)
                if eligibility_gate_mode in ("competition", "rank", "contrast") and feature < feature_count
                else f"elig{feature}"
            )
            if error_mode == "score-gated-nontarget":
                gradient_lines = class_local_score_gated_nontarget_gradient_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=activation_node,
                )
            elif uses_restored_score:
                gradient_lines = class_local_restored_score_nontarget_gradient_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=activation_node,
                    score_gate_node=f"c{class_idx}_decision",
                )
            elif uses_residual_score:
                gradient_lines = class_local_residual_score_nontarget_gradient_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=activation_node,
                    score_gate_node=class_node(class_idx, "score_gate"),
                )
            elif uses_amplified_score:
                gradient_lines = class_local_residual_score_nontarget_gradient_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=activation_node,
                    score_gate_node=f"c{class_idx}_score_amp",
                )
            elif uses_score_common_gate:
                gradient_lines = class_local_residual_score_nontarget_gradient_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=activation_node,
                    score_gate_node=class_node(class_idx, "score_common_gate"),
                )
            elif uses_target_ref_score:
                gradient_lines = class_local_residual_score_nontarget_gradient_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=activation_node,
                    score_gate_node=class_node(class_idx, "score_target_gate"),
                )
            elif uses_amplified_competitive:
                gradient_lines = class_local_amplified_score_competitive_gradient_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=activation_node,
                    own_score_gate_node=f"c{class_idx}_score_amp",
                    opponent_score_gate_nodes=[
                        f"c{opponent_idx}_score_amp"
                        for opponent_idx in range(class_count)
                        if opponent_idx != class_idx
                    ],
                )
            elif uses_amplified_pairwise:
                gradient_lines = class_local_residual_score_nontarget_gradient_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=activation_node,
                    score_gate_node=f"c{class_idx}_score_amp",
                )
                gradient_lines += class_local_pairwise_binary_descent_correction_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=activation_node,
                    losing_gate_nodes=[
                        pairwise_decision_node(opponent_idx, class_idx)
                        for opponent_idx in range(class_count)
                        if opponent_idx != class_idx
                    ],
                    winning_gate_nodes=[
                        pairwise_decision_node(class_idx, opponent_idx)
                        for opponent_idx in range(class_count)
                        if opponent_idx != class_idx
                    ],
                )
            elif uses_amplified_binary:
                gradient_lines = class_local_residual_score_nontarget_gradient_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=activation_node,
                    score_gate_node=f"c{class_idx}_score_amp",
                )
                gradient_lines += class_local_restored_score_binary_correction_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=activation_node,
                    positive_gate_node=f"c{class_idx}_decision",
                    negative_gate_node=f"c{class_idx}_decisionn",
                )
            elif uses_label_rail_descent:
                gradient_lines = class_local_error_rail_gradient_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=activation_node,
                    positive_error_node=class_node(class_idx, "errp"),
                    negative_error_node=class_node(class_idx, "errn"),
                )
            elif uses_score_mass:
                gradient_lines = class_local_error_rail_gradient_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=activation_node,
                    positive_error_node=class_node(class_idx, "errp"),
                    negative_error_node=class_node(class_idx, "errn"),
                )
            elif uses_common_score_mass:
                gradient_lines = class_local_error_rail_gradient_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=activation_node,
                    positive_error_node=class_node(class_idx, "errp"),
                    negative_error_node=class_node(class_idx, "errn"),
                )
            elif uses_common_score_mass_pairwise:
                gradient_lines = class_local_error_rail_gradient_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=activation_node,
                    positive_error_node=class_node(class_idx, "errp"),
                    negative_error_node=class_node(class_idx, "errn"),
                )
            elif uses_contrast_score_mass:
                gradient_lines = class_local_error_rail_gradient_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=activation_node,
                    positive_error_node=class_node(class_idx, "errp"),
                    negative_error_node=class_node(class_idx, "errn"),
                )
            elif uses_low_gain_contrast_score_mass:
                gradient_lines = class_local_error_rail_gradient_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=activation_node,
                    positive_error_node=class_node(class_idx, "errp"),
                    negative_error_node=class_node(class_idx, "errn"),
                )
            elif uses_contrast_gated_score_mass:
                gradient_lines = class_local_error_rail_gradient_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=activation_node,
                    positive_error_node=class_node(class_idx, "errp"),
                    negative_error_node=class_node(class_idx, "errn"),
                )
            elif uses_target_contrast_score_mass:
                gradient_lines = class_local_error_rail_gradient_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=activation_node,
                    positive_error_node=class_node(class_idx, "errp"),
                    negative_error_node=class_node(class_idx, "errn"),
                )
            elif uses_pairwise_score_competition or uses_pairwise_margin_correction:
                gradient_lines = class_local_error_rail_gradient_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=activation_node,
                    positive_error_node=class_node(class_idx, "errp"),
                    negative_error_node=class_node(class_idx, "errn"),
                )
            elif uses_normalizer_error:
                gradient_lines = class_local_error_rail_gradient_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=activation_node,
                    positive_error_node=class_node(class_idx, "errp"),
                    negative_error_node=class_node(class_idx, "errn"),
                )
            elif uses_restored_binary:
                gradient_lines = class_local_restored_score_binary_descent_gradient_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=activation_node,
                    positive_gate_node=f"c{class_idx}_decision",
                    negative_gate_node=f"c{class_idx}_decisionn",
                )
            elif uses_pairwise_binary:
                gradient_lines = class_local_pairwise_binary_descent_gradient_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=activation_node,
                    losing_gate_nodes=[
                        pairwise_decision_node(opponent_idx, class_idx)
                        for opponent_idx in range(class_count)
                        if opponent_idx != class_idx
                    ],
                    winning_gate_nodes=[
                        pairwise_decision_node(class_idx, opponent_idx)
                        for opponent_idx in range(class_count)
                        if opponent_idx != class_idx
                    ],
                )
            elif uses_restored_winner:
                gradient_lines = class_local_multi_gate_nontarget_gradient_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=activation_node,
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
                    activation_node=activation_node,
                )
            if bias_feature is not None and feature == bias_feature and class_bias_mode == "target-only":
                gradient_lines = class_local_target_only_gradient_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=activation_node,
                )
            if readout_update_mode == "live":
                positive_descent_node = (
                    None if error_mode == "label-descent" else class_node(class_idx, "errp")
                )
                negative_descent_node = (
                    None if error_mode == "label-descent" else class_node(class_idx, "errn")
                )
                if bias_feature is not None and feature == bias_feature and class_bias_mode == "target-only":
                    positive_descent_node = None
                    negative_descent_node = "0"
                readout_update_lines = class_local_live_label_descent_update_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=activation_node,
                    positive_descent_node=positive_descent_node,
                    negative_descent_node=negative_descent_node,
                )
            else:
                readout_update_lines = [
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
                    *gradient_lines,
                    *class_local_bounded_update_lines(class_idx=class_idx, feature_idx=feature),
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
                *class_local_readout_forward_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    width_u=readout_width_u,
                    isolation=readout_forward_mode,
                ),
                *readout_update_lines,
            ]
    train_seen = 0
    for cycle, (record, seq) in enumerate(zip(all_records, sequence)):
        base = cycle * CYCLE_NS
        for feature in range(total_feature_count):
            lines += [
                f".meas tran pre_p_f{feature}_{cycle} FIND V(pre_p{feature}) AT={base + 3.2:.2f}n",
                f".meas tran pre_n_f{feature}_{cycle} FIND V(pre_n{feature}) AT={base + 3.2:.2f}n",
                f".meas tran pre_margin_f{feature}_{cycle} PARAM='pre_p_f{feature}_{cycle}-pre_n_f{feature}_{cycle}'",
                f".meas tran act_f{feature}_{cycle} FIND V(act{feature}) AT={base + 4.5:.2f}n",
                f".meas tran elig_f{feature}_{cycle} FIND V(elig{feature}) AT={base + 4.5:.2f}n",
            ]
            if uses_hidden_update and feature < feature_count:
                lines += [
                    f".meas tran xelig_f{feature}_{cycle} FIND V(xelig{feature}) AT={base + 4.5:.2f}n",
                    f".meas tran hdp_f{feature}_{cycle} FIND V({hidden_credit_node(feature, 'hdp')}) AT={base + score_error_measure_ns:.2f}n",
                    f".meas tran hdn_f{feature}_{cycle} FIND V({hidden_credit_node(feature, 'hdn')}) AT={base + score_error_measure_ns:.2f}n",
                    f".meas tran hcredit_f{feature}_{cycle} PARAM='hdp_f{feature}_{cycle}-hdn_f{feature}_{cycle}'",
                ]
        if eligibility_gate_mode in ("competition", "rank", "contrast"):
            for feature in range(feature_count):
                lines.append(
                    f".meas tran egate_f{feature}_{cycle} FIND V({eligibility_gate_node(feature)}) AT={base + 4.85:.2f}n"
                )
        lines += [
            f".meas tran pre_margin_{cycle} PARAM='pre_margin_f0_{cycle}'",
            f".meas tran act_{cycle} PARAM='act_f0_{cycle}'",
            f".meas tran elig_{cycle} PARAM='elig_f0_{cycle}'",
        ]
        for class_idx in range(class_count):
            if score_sense_mode == "current-clamp":
                lines += [
                    f".meas tran c{class_idx}_score_{cycle} FIND I(V{class_node(class_idx, 'score_clamp')}) AT={base + score_measure_ns:.2f}n",
                    f".meas tran c{class_idx}_scoren_{cycle} FIND I(V{class_node(class_idx, 'scoren_clamp')}) AT={base + score_measure_ns:.2f}n",
                    f".meas tran c{class_idx}_score_net_{cycle} PARAM='c{class_idx}_score_{cycle}-c{class_idx}_scoren_{cycle}'",
                ]
            else:
                lines += [
                    f".meas tran c{class_idx}_score_{cycle} FIND V({class_node(class_idx, 'score')}) AT={base + score_measure_ns:.2f}n",
                    f".meas tran c{class_idx}_scoren_{cycle} FIND V({class_node(class_idx, 'scoren')}) AT={base + score_measure_ns:.2f}n",
                    f".meas tran c{class_idx}_score_net_{cycle} PARAM='c{class_idx}_score_{cycle}-c{class_idx}_scoren_{cycle}'",
                ]
            if uses_pairwise_decisions:
                for opponent_idx in range(class_count):
                    if opponent_idx == class_idx:
                        continue
                    lines += [
                        f".meas tran c{class_idx}_gt_c{opponent_idx}_decision_{cycle} FIND V({pairwise_decision_node(class_idx, opponent_idx)}) AT={base + score_decision_measure_ns:.2f}n",
                        f".meas tran c{class_idx}_gt_c{opponent_idx}_decisionn_{cycle} FIND V({pairwise_decision_node(opponent_idx, class_idx)}) AT={base + score_decision_measure_ns:.2f}n",
                        f".meas tran c{class_idx}_gt_c{opponent_idx}_diff_{cycle} PARAM='c{class_idx}_gt_c{opponent_idx}_decision_{cycle}-c{class_idx}_gt_c{opponent_idx}_decisionn_{cycle}'",
                    ]
            if uses_score_preamp:
                lines += [
                    f".meas tran c{class_idx}_score_amp_{cycle} FIND V(c{class_idx}_score_amp) AT={base + score_amp_measure_ns:.2f}n",
                    f".meas tran c{class_idx}_scoren_amp_{cycle} FIND V(c{class_idx}_scoren_amp) AT={base + score_amp_measure_ns:.2f}n",
                    f".meas tran c{class_idx}_score_gain_diff_{cycle} PARAM='c{class_idx}_score_amp_{cycle}-c{class_idx}_scoren_amp_{cycle}'",
                ]
            if uses_score_common_gate_nodes or uses_score_contrast_nodes:
                score_input_measure = (
                    f"c{class_idx}_score_{cycle}"
                    if uses_raw_common_ref_score
                    else f"c{class_idx}_score_amp_{cycle}"
                )
                lines += [
                    f".meas tran score_common_c{class_idx}_{cycle} FIND V(score_common) AT={base + score_amp_measure_ns:.2f}n",
                    f".meas tran c{class_idx}_score_above_common_{cycle} PARAM='{score_input_measure}-score_common_c{class_idx}_{cycle}'",
                ]
                if uses_score_common_gate_nodes:
                    lines.append(
                        f".meas tran c{class_idx}_score_common_gate_{cycle} FIND V({class_node(class_idx, 'score_common_gate')}) AT={base + score_decision_measure_ns:.2f}n"
                    )
                if uses_score_contrast_nodes:
                    lines.append(
                        f".meas tran c{class_idx}_score_contrast_{cycle} FIND V({class_node(class_idx, 'score_contrast')}) AT={base + score_decision_measure_ns:.2f}n"
                    )
                    if uses_contrast_gated_score_mass:
                        lines.append(
                            f".meas tran c{class_idx}_score_contrast_gate_{cycle} FIND V({class_node(class_idx, 'score_contrast_gate')}) AT={base + score_error_measure_ns:.2f}n"
                        )
            if uses_target_ref_score or uses_target_contrast_score_mass:
                lines += [
                    f".meas tran target_score_ref_c{class_idx}_{cycle} FIND V(target_score_ref) AT={base + score_decision_measure_ns:.2f}n",
                    f".meas tran c{class_idx}_score_target_gate_{cycle} FIND V({class_node(class_idx, 'score_target_gate')}) AT={base + score_decision_measure_ns:.2f}n",
                    f".meas tran c{class_idx}_score_above_target_{cycle} PARAM='c{class_idx}_score_amp_{cycle}-target_score_ref_c{class_idx}_{cycle}'",
                ]
            if (
                uses_score_mass
                or uses_label_rail_descent
                or uses_common_score_mass
                or uses_contrast_score_mass
                or uses_low_gain_contrast_score_mass
                or uses_contrast_gated_score_mass
                or uses_target_contrast_score_mass
                or uses_common_score_mass_pairwise
                or uses_pairwise_score_competition
                or uses_pairwise_margin_correction
                or uses_normalizer_error
            ):
                if (
                    uses_score_mass
                    or uses_common_score_mass
                    or uses_contrast_score_mass
                    or uses_low_gain_contrast_score_mass
                    or uses_contrast_gated_score_mass
                    or uses_target_contrast_score_mass
                    or uses_common_score_mass_pairwise
                ):
                    lines.append(
                        f".meas tran score_nontarget_mass_c{class_idx}_{cycle} FIND V(score_nontarget_mass) AT={base + score_mass_measure_ns:.2f}n"
                    )
                lines += [
                    f".meas tran c{class_idx}_errp_{cycle} FIND V({class_node(class_idx, 'errp')}) AT={base + score_error_measure_ns:.2f}n",
                    f".meas tran c{class_idx}_errn_{cycle} FIND V({class_node(class_idx, 'errn')}) AT={base + score_error_measure_ns:.2f}n",
                    f".meas tran c{class_idx}_errdiff_{cycle} PARAM='c{class_idx}_errp_{cycle}-c{class_idx}_errn_{cycle}'",
                ]
            if uses_residual_score:
                lines += [
                    f".meas tran c{class_idx}_score_gate_{cycle} FIND V({class_node(class_idx, 'score_gate')}) AT={base + score_decision_measure_ns:.2f}n",
                ]
        if seq == "train":
            train_seen += 1
            if uses_hidden_update:
                for feature in range(feature_count):
                    lines += [
                        f".meas tran whp_f{feature}_after_train{train_seen} FIND V(whp{feature}) AT={base + 15.0:.2f}n",
                        f".meas tran whn_f{feature}_after_train{train_seen} FIND V(whn{feature}) AT={base + 15.0:.2f}n",
                        f".meas tran whsigned_f{feature}_after_train{train_seen} PARAM='whp_f{feature}_after_train{train_seen}-whn_f{feature}_after_train{train_seen}'",
                    ]
            for class_idx in range(class_count):
                for feature in range(total_feature_count):
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
        for feature in range(total_feature_count):
            lines += [
                f".meas tran c{class_idx}_f{feature}_vwp_final FIND V({class_node(class_idx, f'vwp{feature}')}) AT={stop_ns - 0.5:.2f}n",
                f".meas tran c{class_idx}_f{feature}_vwn_final FIND V({class_node(class_idx, f'vwn{feature}')}) AT={stop_ns - 0.5:.2f}n",
                f".meas tran c{class_idx}_f{feature}_signed_final PARAM='c{class_idx}_f{feature}_vwp_final-c{class_idx}_f{feature}_vwn_final'",
            ]
        lines += [f".meas tran c{class_idx}_signed_final PARAM='c{class_idx}_f0_signed_final'"]
    if uses_hidden_update:
        for feature in range(feature_count):
            lines += [
                f".meas tran whp_f{feature}_final FIND V(whp{feature}) AT={stop_ns - 0.5:.2f}n",
                f".meas tran whn_f{feature}_final FIND V(whn{feature}) AT={stop_ns - 0.5:.2f}n",
                f".meas tran whsigned_f{feature}_final PARAM='whp_f{feature}_final-whn_f{feature}_final'",
            ]
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


def signed_readout_projection_stats(
    measures: dict[str, float],
    *,
    labels: list[int],
    sequence: list[str],
    class_count: int,
    total_feature_count: int,
    final_signed: list[list[float]],
) -> dict[str, Any]:
    """Compare physical score rails with an ideal signed dot-product readout.

    This is a diagnostic only: it uses measured activation capacitors and final
    readout capacitor state after ngspice completes. It does not feed anything
    back into the transient.
    """
    rows: list[dict[str, Any]] = []
    weight_matrix = np.array(final_signed, dtype=float)
    for cycle, seq in enumerate(sequence):
        if seq != "final_eval":
            continue
        keys = [f"act_f{feature}_{cycle}" for feature in range(total_feature_count)]
        if not all(key in measures for key in keys):
            continue
        acts = np.array([float(measures[key]) for key in keys], dtype=float)
        scores = weight_matrix @ acts
        label = int(labels[cycle])
        prediction = int(np.argmax(scores))
        margin = float(scores[label] - max(score for idx, score in enumerate(scores) if idx != label))
        rows.append(
            {
                "cycle": cycle,
                "label": label,
                "prediction": prediction,
                "correct": prediction == label,
                "score_margin_v2": margin,
                **{f"score_c{class_idx}_v2": float(scores[class_idx]) for class_idx in range(class_count)},
            }
        )
    if not rows:
        return {
            "final_eval_signed_projection_rows": [],
            "final_eval_signed_projection_accuracy": None,
            "final_eval_signed_projection_min_margin_v2": None,
            "final_eval_signed_projection_mean_abs_score_v2": None,
        }
    margins = [float(row["score_margin_v2"]) for row in rows]
    score_values = [
        abs(float(row[f"score_c{class_idx}_v2"]))
        for row in rows
        for class_idx in range(class_count)
    ]
    return {
        "final_eval_signed_projection_rows": rows,
        "final_eval_signed_projection_accuracy": float(np.mean([bool(row["correct"]) for row in rows])),
        "final_eval_signed_projection_min_margin_v2": float(np.min(margins)),
        "final_eval_signed_projection_mean_abs_score_v2": float(np.mean(score_values)),
    }


def conductance_readout_projection_stats(
    measures: dict[str, float],
    *,
    labels: list[int],
    sequence: list[str],
    class_count: int,
    total_feature_count: int,
    final_positive: list[list[float]],
    final_negative: list[list[float]],
    positive_vto: float = 0.35,
    negative_width_scale: float = 48.0 / 64.0,
) -> dict[str, Any]:
    """Approximate the physical NMOS conductance readout from final gate rails."""
    positive = np.maximum(np.array(final_positive, dtype=float) - positive_vto, 0.0)
    negative = negative_width_scale * np.maximum(np.array(final_negative, dtype=float) - positive_vto, 0.0)
    weight_matrix = positive - negative
    rows: list[dict[str, Any]] = []
    for cycle, seq in enumerate(sequence):
        if seq != "final_eval":
            continue
        keys = [f"act_f{feature}_{cycle}" for feature in range(total_feature_count)]
        if not all(key in measures for key in keys):
            continue
        acts = np.array([float(measures[key]) for key in keys], dtype=float)
        scores = weight_matrix @ acts
        label = int(labels[cycle])
        prediction = int(np.argmax(scores))
        margin = float(scores[label] - max(score for idx, score in enumerate(scores) if idx != label))
        rows.append(
            {
                "cycle": cycle,
                "label": label,
                "prediction": prediction,
                "correct": prediction == label,
                "score_margin_v2": margin,
                **{f"score_c{class_idx}_v2": float(scores[class_idx]) for class_idx in range(class_count)},
            }
        )
    if not rows:
        return {
            "final_eval_conductance_projection_rows": [],
            "final_eval_conductance_projection_accuracy": None,
            "final_eval_conductance_projection_min_margin_v2": None,
            "final_eval_conductance_projection_active_weights": None,
        }
    margins = [float(row["score_margin_v2"]) for row in rows]
    return {
        "final_eval_conductance_projection_rows": rows,
        "final_eval_conductance_projection_accuracy": float(np.mean([bool(row["correct"]) for row in rows])),
        "final_eval_conductance_projection_min_margin_v2": float(np.min(margins)),
        "final_eval_conductance_projection_active_weights": int(np.count_nonzero(np.abs(weight_matrix) > 0.0)),
    }


def activation_prototype_projection_stats(
    measures: dict[str, float],
    *,
    labels: list[int],
    sequence: list[str],
    class_count: int,
    total_feature_count: int,
) -> dict[str, Any]:
    """Classify final activations with prototypes built from train activations.

    This diagnostic asks whether the measured feature/eligibility space is
    separable before blaming a specific physical readout update rule. It is
    post-run analysis only and does not modify circuit state.
    """
    train_by_class: list[list[np.ndarray]] = [[] for _ in range(class_count)]
    final_rows: list[tuple[int, int, np.ndarray]] = []
    for cycle, seq in enumerate(sequence):
        keys = [f"act_f{feature}_{cycle}" for feature in range(total_feature_count)]
        if not all(key in measures for key in keys):
            continue
        acts = np.array([float(measures[key]) for key in keys], dtype=float)
        label = int(labels[cycle])
        if seq == "train":
            train_by_class[label].append(acts)
        elif seq == "final_eval":
            final_rows.append((cycle, label, acts))
    if any(not rows for rows in train_by_class) or not final_rows:
        return {
            "final_eval_activation_prototype_rows": [],
            "final_eval_activation_prototype_accuracy": None,
            "final_eval_activation_prototype_min_margin_v2": None,
            "final_eval_activation_prototype_pairwise_cosine_mean": None,
        }
    prototypes = np.array([np.mean(rows, axis=0) for rows in train_by_class], dtype=float)
    proto_norms = np.linalg.norm(prototypes, axis=1)
    rows: list[dict[str, Any]] = []
    cosine_rows: list[dict[str, Any]] = []
    for cycle, label, acts in final_rows:
        scores = prototypes @ acts
        prediction = int(np.argmax(scores))
        margin = float(scores[label] - max(score for idx, score in enumerate(scores) if idx != label))
        rows.append(
            {
                "cycle": cycle,
                "label": label,
                "prediction": prediction,
                "correct": prediction == label,
                "score_margin_v2": margin,
                **{f"score_c{class_idx}_v2": float(scores[class_idx]) for class_idx in range(class_count)},
            }
        )
        act_norm = float(np.linalg.norm(acts))
        if act_norm > 0.0 and np.all(proto_norms > 0.0):
            cosine_scores = scores / (proto_norms * act_norm)
            cosine_prediction = int(np.argmax(cosine_scores))
            cosine_margin = float(
                cosine_scores[label] - max(score for idx, score in enumerate(cosine_scores) if idx != label)
            )
            cosine_rows.append(
                {
                    "cycle": cycle,
                    "label": label,
                    "prediction": cosine_prediction,
                    "correct": cosine_prediction == label,
                    "score_margin": cosine_margin,
                    **{f"score_c{class_idx}": float(cosine_scores[class_idx]) for class_idx in range(class_count)},
                }
            )
    pairwise_cosines: list[float] = []
    for left_idx, left in enumerate(prototypes):
        for right_idx in range(left_idx + 1, class_count):
            denom = float(proto_norms[left_idx] * proto_norms[right_idx])
            if denom > 0.0:
                pairwise_cosines.append(float(np.dot(left, prototypes[right_idx]) / denom))
    margins = [float(row["score_margin_v2"]) for row in rows]
    return {
        "final_eval_activation_prototype_rows": rows,
        "final_eval_activation_prototype_accuracy": float(np.mean([bool(row["correct"]) for row in rows])),
        "final_eval_activation_prototype_min_margin_v2": float(np.min(margins)),
        "final_eval_activation_cosine_prototype_rows": cosine_rows,
        "final_eval_activation_cosine_prototype_accuracy": (
            float(np.mean([bool(row["correct"]) for row in cosine_rows])) if cosine_rows else None
        ),
        "final_eval_activation_cosine_prototype_min_margin": (
            float(np.min([float(row["score_margin"]) for row in cosine_rows])) if cosine_rows else None
        ),
        "final_eval_activation_prototype_pairwise_cosine_mean": (
            float(np.mean(pairwise_cosines)) if pairwise_cosines else None
        ),
    }


def error_rail_stats(
    measures: dict[str, float],
    *,
    labels: list[int],
    sequence: list[str],
    class_count: int,
) -> dict[str, Any]:
    target_values: list[float] = []
    nontarget_values: list[float] = []
    rows: list[list[float]] = []
    for cycle, seq in enumerate(sequence):
        if seq != "train":
            continue
        keys = [f"c{class_idx}_errdiff_{cycle}" for class_idx in range(class_count)]
        if not all(key in measures for key in keys):
            continue
        values = [float(measures[key]) for key in keys]
        rows.append(values)
        label = int(labels[cycle])
        target_values.append(values[label])
        nontarget_values.extend(value for class_idx, value in enumerate(values) if class_idx != label)
    if not rows:
        return {
            "train_errdiff_rows_v": [],
            "train_target_errdiff_mean_v": None,
            "train_target_errdiff_min_v": None,
            "train_nontarget_errdiff_mean_v": None,
            "train_nontarget_errdiff_max_v": None,
        }
    return {
        "train_errdiff_rows_v": rows,
        "train_target_errdiff_mean_v": float(np.mean(target_values)),
        "train_target_errdiff_min_v": float(np.min(target_values)),
        "train_nontarget_errdiff_mean_v": float(np.mean(nontarget_values)),
        "train_nontarget_errdiff_max_v": float(np.max(nontarget_values)),
    }


def eligibility_stats(
    measures: dict[str, float],
    *,
    sequence: list[str],
    total_feature_count: int,
) -> dict[str, Any]:
    rows: list[list[float]] = []
    active_25mv: list[int] = []
    active_250mv: list[int] = []
    active_500mv: list[int] = []
    for cycle, seq in enumerate(sequence):
        if seq != "train":
            continue
        keys = [f"elig_f{feature}_{cycle}" for feature in range(total_feature_count)]
        if not all(key in measures for key in keys):
            continue
        values = [float(measures[key]) for key in keys]
        rows.append(values)
        active_25mv.append(sum(value > 25e-3 for value in values))
        active_250mv.append(sum(value > 250e-3 for value in values))
        active_500mv.append(sum(value > 500e-3 for value in values))
    if not rows:
        return {
            "train_eligibility_rows_v": [],
            "train_eligibility_active_features_25mv_mean": None,
            "train_eligibility_active_features_250mv_mean": None,
            "train_eligibility_active_features_500mv_mean": None,
            "train_eligibility_pairwise_cosine_mean": None,
        }
    pairwise_cosines: list[float] = []
    vectors = [np.array(row, dtype=float) for row in rows]
    for left_idx, left in enumerate(vectors):
        for right in vectors[left_idx + 1 :]:
            denom = float(np.linalg.norm(left) * np.linalg.norm(right))
            if denom > 0.0:
                pairwise_cosines.append(float(np.dot(left, right) / denom))
    return {
        "train_eligibility_rows_v": rows,
        "train_eligibility_active_features_25mv_mean": float(np.mean(active_25mv)),
        "train_eligibility_active_features_25mv_max": int(np.max(active_25mv)),
        "train_eligibility_active_features_250mv_mean": float(np.mean(active_250mv)),
        "train_eligibility_active_features_500mv_mean": float(np.mean(active_500mv)),
        "train_eligibility_pairwise_cosine_mean": float(np.mean(pairwise_cosines)) if pairwise_cosines else None,
    }


def eligibility_gate_stats(
    measures: dict[str, float],
    *,
    sequence: list[str],
    feature_count: int,
) -> dict[str, Any]:
    rows: list[list[float]] = []
    active_250mv: list[int] = []
    active_600mv: list[int] = []
    for cycle, seq in enumerate(sequence):
        if seq != "train":
            continue
        keys = [f"egate_f{feature}_{cycle}" for feature in range(feature_count)]
        if not all(key in measures for key in keys):
            continue
        values = [float(measures[key]) for key in keys]
        rows.append(values)
        active_250mv.append(sum(value > 250e-3 for value in values))
        active_600mv.append(sum(value > 600e-3 for value in values))
    if not rows:
        return {
            "train_eligibility_gate_rows_v": [],
            "train_eligibility_gate_active_features_250mv_mean": None,
            "train_eligibility_gate_active_features_250mv_max": None,
            "train_eligibility_gate_active_features_600mv_mean": None,
            "train_eligibility_gate_active_features_600mv_max": None,
            "train_eligibility_gate_max_v": None,
        }
    return {
        "train_eligibility_gate_rows_v": rows,
        "train_eligibility_gate_active_features_250mv_mean": float(np.mean(active_250mv)),
        "train_eligibility_gate_active_features_250mv_max": int(np.max(active_250mv)),
        "train_eligibility_gate_active_features_600mv_mean": float(np.mean(active_600mv)),
        "train_eligibility_gate_active_features_600mv_max": int(np.max(active_600mv)),
        "train_eligibility_gate_max_v": float(np.max(rows)),
    }


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
    labels = [int(record["label"]) for record in all_records]
    total_feature_count = feature_count + (1 if args.class_bias_mode != "none" else 0)
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
        score_measure_ns=args.score_measure_ns,
        initial_positive=args.initial_positive,
        initial_negative=args.initial_negative,
        nontarget_scale=args.nontarget_scale,
        nontarget_width_scale=args.nontarget_width_scale,
        score_mass_sum_width=args.score_mass_sum_width,
        score_mass_error_width=args.score_mass_error_width,
        score_mass_pairwise_error_scale=args.score_mass_pairwise_error_scale,
        pairwise_margin_target_v=args.pairwise_margin_target_v,
        pairwise_margin_error_drive_scale=args.pairwise_margin_error_drive_scale,
        normalizer_error_clock_high=args.normalizer_error_clock_high,
        error_mode=args.error_mode,
        readout_update_mode=args.readout_update_mode,
        hidden_update_mode=args.hidden_update_mode,
        hidden_credit_width_u=args.hidden_credit_width,
        hidden_update_width_u=args.hidden_update_width,
        score_timing_mode=args.score_timing_mode,
        score_sense_mode=args.score_sense_mode,
        readout_forward_mode=args.readout_forward_mode,
        eligibility_gate_mode=args.eligibility_gate_mode,
        eligibility_contrast_common_resistance_ohm=args.eligibility_contrast_common_resistance,
        eligibility_contrast_common_capacitance_f=args.eligibility_contrast_common_capacitance_f,
        class_bias_mode=args.class_bias_mode,
        class_bias_input=args.class_bias_input,
        readout_center_resistance=args.readout_center_resistance,
        readout_center_voltage=args.readout_center_voltage,
    )
    measures = run_netlist(spice_bin, path, deck, timeout=args.timeout)
    rows = rows_from_measures(all_records, measures, sequence=sequence, class_count=args.class_count)
    initial_margins = [float(row["score_margin_v"]) for row in rows if row["sequence"] == "initial_eval"]
    final_margins = [float(row["score_margin_v"]) for row in rows if row["sequence"] == "final_eval"]
    final_signed = [
        [float(measures[f"c{class_idx}_f{feature}_signed_final"]) for feature in range(total_feature_count)]
        for class_idx in range(args.class_count)
    ]
    final_positive = [
        [float(measures[f"c{class_idx}_f{feature}_vwp_final"]) for feature in range(total_feature_count)]
        for class_idx in range(args.class_count)
    ]
    final_negative = [
        [float(measures[f"c{class_idx}_f{feature}_vwn_final"]) for feature in range(total_feature_count)]
        for class_idx in range(args.class_count)
    ]
    final_hidden_signed = (
        [float(measures[f"whsigned_f{feature}_final"]) for feature in range(feature_count)]
        if args.hidden_update_mode != "none"
        else None
    )
    train_progress = []
    for train_idx in range(1, len(train_records) + 1):
        train_progress.append(
            [
                [float(measures[f"c{class_idx}_f{feature}_signed_after_train{train_idx}"]) for feature in range(feature_count)]
                + (
                    [float(measures[f"c{class_idx}_f{feature_count}_signed_after_train{train_idx}"])]
                    if args.class_bias_mode != "none"
                    else []
                )
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
        "total_feature_count": total_feature_count,
        "class_bias_mode": args.class_bias_mode,
        "class_bias_input": args.class_bias_input if args.class_bias_mode != "none" else None,
        "readout_center_resistance_ohm": args.readout_center_resistance if args.readout_center_resistance > 0.0 else None,
        "readout_center_voltage_v": args.readout_center_voltage if args.readout_center_resistance > 0.0 else None,
        "readout_width_u": args.readout_width,
        "score_capacitance_f": args.score_capacitance_f,
        "score_load_resistance_ohm": args.score_load_resistance,
        "score_measure_ns": args.score_measure_ns,
        "nontarget_scale": args.nontarget_scale,
        "nontarget_width_scale": args.nontarget_width_scale,
        "score_mass_sum_width_u": args.score_mass_sum_width if "score-mass" in args.error_mode else None,
        "score_mass_error_width_u": args.score_mass_error_width if "score-mass" in args.error_mode else None,
        "score_mass_pairwise_error_scale": (
            args.score_mass_pairwise_error_scale if args.error_mode == "common-score-mass-pairwise-descent" else None
        ),
        "pairwise_margin_target_v": (
            args.pairwise_margin_target_v if args.error_mode == "pairwise-margin-correction-descent" else None
        ),
        "pairwise_margin_error_drive_scale": (
            args.pairwise_margin_error_drive_scale
            if args.error_mode == "pairwise-margin-correction-descent"
            else None
        ),
        "normalizer_error_clock_high": (
            args.normalizer_error_clock_high if args.error_mode in NORMALIZER_ERROR_MODES else None
        ),
        "score_mass_common_internal_gain": (
            4.0
            if args.error_mode
            in {
                "common-score-mass-descent",
                "low-gain-contrast-score-mass-descent",
                "contrast-gated-score-mass-descent",
                "target-contrast-score-mass-descent",
                "common-score-mass-pairwise-descent",
            }
            else None
        ),
        "error_mode": args.error_mode,
        "readout_update_mode": args.readout_update_mode,
        "hidden_update_mode": args.hidden_update_mode,
        "hidden_credit_width_u": args.hidden_credit_width if args.hidden_update_mode != "none" else None,
        "hidden_update_width_u": args.hidden_update_width if args.hidden_update_mode != "none" else None,
        "score_timing_mode": args.score_timing_mode,
        "score_sense_mode": args.score_sense_mode,
        "readout_forward_mode": args.readout_forward_mode,
        "eligibility_gate_mode": args.eligibility_gate_mode,
        "eligibility_contrast_common_resistance_ohm": (
            args.eligibility_contrast_common_resistance if args.eligibility_gate_mode == "contrast" else None
        ),
        "eligibility_contrast_common_capacitance_f": (
            args.eligibility_contrast_common_capacitance_f if args.eligibility_gate_mode == "contrast" else None
        ),
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
        "final_positive_matrix_v": final_positive,
        "final_negative_matrix_v": final_negative,
        "final_hidden_signed_v": final_hidden_signed,
        "signed_after_each_train_v": train_progress,
        **signed_readout_projection_stats(
            measures,
            labels=labels,
            sequence=sequence,
            class_count=args.class_count,
            total_feature_count=total_feature_count,
            final_signed=final_signed,
        ),
        **conductance_readout_projection_stats(
            measures,
            labels=labels,
            sequence=sequence,
            class_count=args.class_count,
            total_feature_count=total_feature_count,
            final_positive=final_positive,
            final_negative=final_negative,
        ),
        **activation_prototype_projection_stats(
            measures,
            labels=labels,
            sequence=sequence,
            class_count=args.class_count,
            total_feature_count=total_feature_count,
        ),
        **error_rail_stats(measures, labels=labels, sequence=sequence, class_count=args.class_count),
        **eligibility_stats(measures, sequence=sequence, total_feature_count=total_feature_count),
        **eligibility_gate_stats(measures, sequence=sequence, feature_count=feature_count),
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
    ap.add_argument("--score-measure-ns", type=float, default=8.5)
    ap.add_argument("--initial-positive", type=float, default=0.40)
    ap.add_argument("--initial-negative", type=float, default=0.40)
    ap.add_argument("--nontarget-scale", type=float, default=1.0)
    ap.add_argument("--nontarget-width-scale", type=float, default=1.0)
    ap.add_argument("--score-mass-sum-width", type=float, default=32.0)
    ap.add_argument("--score-mass-error-width", type=float, default=32.0)
    ap.add_argument("--score-mass-pairwise-error-scale", type=float, default=0.0625)
    ap.add_argument("--pairwise-margin-target-v", type=float, default=1.0e-3)
    ap.add_argument("--pairwise-margin-error-drive-scale", type=float, default=1.0)
    ap.add_argument("--normalizer-error-clock-high", type=float, default=1.2)
    ap.add_argument("--error-mode", choices=ERROR_MODES, default="label-descent")
    ap.add_argument("--readout-update-mode", choices=READOUT_UPDATE_MODES, default="sampled")
    ap.add_argument("--hidden-update-mode", choices=HIDDEN_UPDATE_MODES, default="none")
    ap.add_argument("--hidden-credit-width", type=float, default=8.0)
    ap.add_argument("--hidden-update-width", type=float, default=0.25)
    ap.add_argument("--score-timing-mode", choices=SCORE_TIMING_MODES, default="late")
    ap.add_argument("--score-sense-mode", choices=SCORE_SENSE_MODES, default="voltage")
    ap.add_argument("--readout-forward-mode", choices=READOUT_FORWARD_MODES, default="direct")
    ap.add_argument("--eligibility-gate-mode", choices=ELIGIBILITY_GATE_MODES, default="raw")
    ap.add_argument("--eligibility-contrast-common-resistance", type=float, default=1.0e6)
    ap.add_argument("--eligibility-contrast-common-capacitance-f", type=float, default=1.0)
    ap.add_argument("--class-bias-mode", choices=CLASS_BIAS_MODES, default="none")
    ap.add_argument("--class-bias-input", type=float, default=0.85)
    ap.add_argument("--readout-center-resistance", type=float, default=0.0)
    ap.add_argument("--readout-center-voltage", type=float, default=0.40)
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
    if args.readout_update_mode not in READOUT_UPDATE_MODES:
        raise ValueError(f"readout-update-mode must be one of {READOUT_UPDATE_MODES}")
    if args.hidden_update_mode not in HIDDEN_UPDATE_MODES:
        raise ValueError(f"hidden-update-mode must be one of {HIDDEN_UPDATE_MODES}")
    if args.score_timing_mode not in SCORE_TIMING_MODES:
        raise ValueError(f"score-timing-mode must be one of {SCORE_TIMING_MODES}")
    if args.score_sense_mode not in SCORE_SENSE_MODES:
        raise ValueError(f"score-sense-mode must be one of {SCORE_SENSE_MODES}")
    if args.readout_forward_mode not in READOUT_FORWARD_MODES:
        raise ValueError(f"readout-forward-mode must be one of {READOUT_FORWARD_MODES}")
    if args.score_sense_mode == "current-clamp" and args.error_mode != "label-descent":
        raise ValueError("current-clamp score-sense-mode is only a label-descent readout diagnostic")
    if (
        args.readout_update_mode == "live"
        and args.error_mode != "label-descent"
        and args.error_mode not in ERROR_RAIL_DESCENT_MODES
    ):
        raise ValueError("live readout-update-mode requires label-descent or error-rail descent modes")
    if args.hidden_update_mode != "none" and args.error_mode not in ERROR_RAIL_DESCENT_MODES:
        raise ValueError("hidden-update-mode requires an error-rail descent mode")
    if args.class_bias_mode not in CLASS_BIAS_MODES:
        raise ValueError(f"class-bias-mode must be one of {CLASS_BIAS_MODES}")
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
    if args.class_bias_input < 0.0 or args.class_bias_input > 1.2:
        raise ValueError("class-bias-input must stay within supply rails")
    if args.readout_center_resistance < 0.0:
        raise ValueError("readout-center-resistance must be nonnegative")
    if args.readout_center_voltage < 0.0 or args.readout_center_voltage > 1.2:
        raise ValueError("readout-center-voltage must stay within supply rails")
    if args.readout_width <= 0.0:
        raise ValueError("readout-width must be positive")
    if args.score_capacitance_f <= 0.0:
        raise ValueError("score-capacitance-f must be positive")
    if args.score_load_resistance <= 0.0:
        raise ValueError("score-load-resistance must be positive")
    if args.score_measure_ns <= 0.0 or args.score_measure_ns >= CYCLE_NS:
        raise ValueError("score-measure-ns must stay inside the cycle")
    if args.nontarget_scale < 0.0 or args.nontarget_scale > 1.0:
        raise ValueError("nontarget-scale must be in [0, 1]")
    if args.nontarget_width_scale < 0.0 or args.nontarget_width_scale > 1.0:
        raise ValueError("nontarget-width-scale must be in [0, 1]")
    if args.score_mass_sum_width <= 0.0:
        raise ValueError("score-mass-sum-width must be positive")
    if args.score_mass_error_width <= 0.0:
        raise ValueError("score-mass-error-width must be positive")
    if args.score_mass_pairwise_error_scale <= 0.0:
        raise ValueError("score-mass-pairwise-error-scale must be positive")
    if args.pairwise_margin_target_v <= 0.0:
        raise ValueError("pairwise-margin-target-v must be positive")
    if args.pairwise_margin_error_drive_scale <= 0.0:
        raise ValueError("pairwise-margin-error-drive-scale must be positive")
    if args.normalizer_error_clock_high <= 0.0 or args.normalizer_error_clock_high > 1.2:
        raise ValueError("normalizer-error-clock-high must stay in (0, 1.2]")
    if args.hidden_credit_width <= 0.0:
        raise ValueError("hidden-credit-width must be positive")
    if args.hidden_update_width <= 0.0:
        raise ValueError("hidden-update-width must be positive")
    if args.eligibility_contrast_common_resistance <= 0.0:
        raise ValueError("eligibility-contrast-common-resistance must be positive")
    if args.eligibility_contrast_common_capacitance_f <= 0.0:
        raise ValueError("eligibility-contrast-common-capacitance-f must be positive")
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
