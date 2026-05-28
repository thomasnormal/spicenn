from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from _util import parse_measures
from run_device_sequential_training import mos_models
from run_mnist01_fixed_feature_divider_training import (
    CYCLE_NS,
    OUTPUTS,
    _clock_lines,
    _sample_feature_wave,
    _sample_plan,
    _target_wave,
    _validate_records,
    load_mnist01_records,
)
from run_multiclass_output_head_primitive import class_node, signed_store_lines
from run_multiclass_output_head_sequence import class_local_live_label_descent_update_lines
from run_spice_sweep import run_text_netlist


HIDDEN = 4


def hidden_block_for_feature(feature: int, image_size: int) -> int:
    if image_size % 2 != 0:
        raise ValueError("image_size must be even for four quadrant hidden units")
    row = feature // image_size
    col = feature % image_size
    if not 0 <= row < image_size or not 0 <= col < image_size:
        raise ValueError("feature index is outside the image")
    return (2 if row >= image_size // 2 else 0) + (1 if col >= image_size // 2 else 0)


def _image_size_from_feature_count(feature_count: int) -> int:
    image_size = int(round(feature_count**0.5))
    if image_size * image_size != feature_count:
        raise ValueError("feature count must be a square image")
    return image_size


def _hidden_weight_node(hidden: int, feature: int, kind: str) -> str:
    return f"wh{hidden}f{feature}{kind}"


def _readout_storage_lines(hidden_count: int, initial_positive: float, initial_negative: float) -> list[str]:
    lines: list[str] = []
    for output in range(OUTPUTS):
        for hidden in range(hidden_count):
            lines += signed_store_lines(
                positive_node=class_node(output, f"vwp{hidden}"),
                negative_node=class_node(output, f"vwn{hidden}"),
                positive_ic=initial_positive,
                negative_ic=initial_negative,
            )
    return lines


def _hidden_storage_lines(
    feature_count: int,
    hidden_count: int,
    *,
    inside_positive: float,
    outside_positive: float,
    inside_negative: float,
    outside_negative: float,
) -> list[str]:
    image_size = _image_size_from_feature_count(feature_count)
    lines: list[str] = []
    for hidden in range(hidden_count):
        for feature in range(feature_count):
            inside = hidden_block_for_feature(feature, image_size) == hidden
            positive = inside_positive if inside else outside_positive
            negative = inside_negative if inside else outside_negative
            lines += [
                f"C{_hidden_weight_node(hidden, feature, 'p')} {_hidden_weight_node(hidden, feature, 'p')} 0 20f IC={positive:.12g}",
                f"R{_hidden_weight_node(hidden, feature, 'p')} {_hidden_weight_node(hidden, feature, 'p')} 0 1e15",
                f"C{_hidden_weight_node(hidden, feature, 'n')} {_hidden_weight_node(hidden, feature, 'n')} 0 20f IC={negative:.12g}",
                f"R{_hidden_weight_node(hidden, feature, 'n')} {_hidden_weight_node(hidden, feature, 'n')} 0 1e15",
            ]
    return lines


def _hidden_state_lines(hidden_count: int) -> list[str]:
    lines: list[str] = []
    for hidden in range(hidden_count):
        for node in (f"pre{hidden}_p", f"pre{hidden}_n", f"act{hidden}", f"hrow{hidden}"):
            lines += [
                f"C{node} {node} 0 12f IC=0",
                f"R{node} {node} 0 1G",
                f"Mreset_{node} {node} rst 0 0 NMOS W=4u L=180n",
            ]
    return lines


def _hidden_forward_lines(feature_count: int, hidden_count: int, width_u: float) -> list[str]:
    lines: list[str] = []
    for hidden in range(hidden_count):
        pre_p = f"pre{hidden}_p"
        pre_n = f"pre{hidden}_n"
        act = f"act{hidden}"
        for feature in range(feature_count):
            lines += [
                f"Rh{hidden}f{feature}pmid h{hidden}f{feature}pmid 0 1G",
                f"Ch{hidden}f{feature}pmid h{hidden}f{feature}pmid 0 0.05f IC=0",
                f"Rh{hidden}f{feature}nmid h{hidden}f{feature}nmid 0 1G",
                f"Ch{hidden}f{feature}nmid h{hidden}f{feature}nmid 0 0.05f IC=0",
                f"Mh{hidden}f{feature}p_phi px{feature} featphi h{hidden}f{feature}pmid 0 NSENSE W={width_u:.6g}u L=180n",
                f"Mh{hidden}f{feature}p_w h{hidden}f{feature}pmid {_hidden_weight_node(hidden, feature, 'p')} {pre_p} 0 NSENSE W={width_u:.6g}u L=180n",
                f"Mh{hidden}f{feature}n_phi px{feature} featphi h{hidden}f{feature}nmid 0 NSENSE W={width_u:.6g}u L=180n",
                f"Mh{hidden}f{feature}n_w h{hidden}f{feature}nmid {_hidden_weight_node(hidden, feature, 'n')} {pre_n} 0 NSENSE W={width_u:.6g}u L=180n",
            ]
        lines += [
            f"Mact{hidden}_p vdd {pre_p} {act} 0 NREL W=24u L=180n",
            f"Mact{hidden}_n {act} {pre_n} 0 0 NSENSE W=24u L=180n",
            f"Chrow{hidden}_ctrl hrow{hidden}_ctrl 0 1f IC=1.2",
            f"Rhrow{hidden}_ctrl hrow{hidden}_ctrl vdd 1G",
            f"Rhrow{hidden}_mid hrow{hidden}_mid 0 1G",
            f"Mhrow{hidden}_ctrl_rst hrow{hidden}_ctrl rstn vdd vdd PMOS W=4u L=180n",
            f"Mhrow{hidden}_ctrl_a hrow{hidden}_ctrl {act} hrow{hidden}_mid 0 NSENSE W=12u L=180n",
            f"Mhrow{hidden}_ctrl_phi hrow{hidden}_mid featphi 0 0 NSENSE W=12u L=180n",
            f"Mhrow{hidden}_restore hrow{hidden} hrow{hidden}_ctrl vdd vdd PMOS W=16u L=180n",
        ]
    return lines


def _score_storage_lines() -> list[str]:
    lines: list[str] = []
    for output in range(OUTPUTS):
        for kind in ("scorep", "scoren", "errp", "errn"):
            node = class_node(output, kind)
            cap_f = 2.0 if kind.startswith("err") else 8.0
            lines += [
                f"C{node} {node} 0 {cap_f:.12g}f IC=0",
                f"R{node} {node} 0 1G",
                f"Mreset_{node} {node} rst 0 0 NMOS W=4u L=180n",
            ]
    return lines


def _score_readout_lines(hidden_count: int, width_u: float) -> list[str]:
    lines: list[str] = []
    for output in range(OUTPUTS):
        scorep = class_node(output, "scorep")
        scoren = class_node(output, "scoren")
        for hidden in range(hidden_count):
            prefix = f"c{output}_h{hidden}_score_"
            lines += [
                f"R{prefix}pa {prefix}pa 0 1G",
                f"R{prefix}pb {prefix}pb 0 1G",
                f"C{prefix}pa {prefix}pa 0 0.05f IC=0",
                f"C{prefix}pb {prefix}pb 0 0.05f IC=0",
                f"M{prefix}pa vdd hrow{hidden} {prefix}pa 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}pw {prefix}pa {class_node(output, f'vwp{hidden}')} {prefix}pb 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}pphi {prefix}pb scorephi {scorep} 0 NSENSE W={width_u:.6g}u L=180n",
                f"R{prefix}na {prefix}na 0 1G",
                f"R{prefix}nb {prefix}nb 0 1G",
                f"C{prefix}na {prefix}na 0 0.05f IC=0",
                f"C{prefix}nb {prefix}nb 0 0.05f IC=0",
                f"M{prefix}na vdd hrow{hidden} {prefix}na 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}nw {prefix}na {class_node(output, f'vwn{hidden}')} {prefix}nb 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}nphi {prefix}nb scorephi {scoren} 0 NSENSE W={width_u:.6g}u L=180n",
            ]
    return lines


def _error_storage_lines() -> list[str]:
    lines: list[str] = [
        "Vnormfloor normfloor 0 0.62",
        "Rrnorm rnorm 0 1G",
        "Rmir0 mir0 0 1G",
        "Rmir1 mir1 0 1G",
        "Vrsen0 rnorm rd0 0",
        "Vrsen1 rnorm rd1 0",
    ]
    for branch in (0, 1):
        low = f"b{branch}low"
        lines += [
            f"C{low} {low} 0 1f IC=1.2",
            f"R{low} {low} vdd 1G",
            f"Mpre_{low} {low} rstn vdd vdd PMOS W=4u L=180n",
            f"R{low}_a {low}_a 0 1G",
            f"M{low}_sink {low} mir{branch} {low}_a 0 NMOS W=0.85u L=180n",
            f"M{low}_phi {low}_a errphi 0 0 NSENSE W=0.85u L=180n",
        ]
    return lines


def _divider_probability_lines(branch_width_u: float, floor_width_u: float) -> list[str]:
    return [
        f"Mnorm0_score rd0 {class_node(0, 'scorep')} mir0 0 NSENSE W={branch_width_u:.6g}u L=180n",
        f"Mnorm0_floor rd0 normfloor mir0 0 NSENSE W={floor_width_u:.6g}u L=180n",
        f"Mnorm1_score rd1 {class_node(1, 'scorep')} mir1 0 NSENSE W={branch_width_u:.6g}u L=180n",
        f"Mnorm1_floor rd1 normfloor mir1 0 NSENSE W={floor_width_u:.6g}u L=180n",
        "Mnorm0_ref mir0 mir0 0 0 NMOS W=2u L=180n",
        "Mnorm1_ref mir1 mir1 0 0 NMOS W=2u L=180n",
    ]


def _route_to_error_rails_lines(route_width_u: float) -> list[str]:
    def charge_lines(name: str, dest: str, low: str, target: str) -> list[str]:
        return [
            f"R{name}_a {name}_a 0 1G",
            f"R{name}_b {name}_b 0 1G",
            f"M{name}_m vdd {low} {name}_a vdd PMOS W={route_width_u:.6g}u L=180n",
            f"M{name}_t {name}_a {target} {name}_b 0 NSENSE W=12u L=180n",
            f"M{name}_phi {name}_b errphi {dest} 0 NSENSE W=12u L=180n",
        ]

    lines: list[str] = []
    lines += charge_lines("err_c0p", class_node(0, "errp"), "b1low", "t0")
    lines += charge_lines("err_c1n", class_node(1, "errn"), "b1low", "t0")
    lines += charge_lines("err_c1p", class_node(1, "errp"), "b0low", "t1")
    lines += charge_lines("err_c0n", class_node(0, "errn"), "b0low", "t1")
    return lines


def _readout_writer_lines(hidden_count: int, width_u: float) -> list[str]:
    lines = [
        "Vvwhi_ref vwhi_ref 0 0.48",
        "Vvwlo_ref vwlo_ref 0 0.22",
    ]
    for output in range(OUTPUTS):
        for hidden in range(hidden_count):
            lines += class_local_live_label_descent_update_lines(
                class_idx=output,
                feature_idx=hidden,
                activation_node=f"hrow{hidden}",
                positive_descent_node=class_node(output, "errp"),
                negative_descent_node=class_node(output, "errn"),
                width_u=width_u,
                high_side_topology="pmos-differential",
            )
    return lines


def _hidden_credit_lines(
    hidden_count: int,
    width_u: float,
    capacitance_f: float,
    internal_cap_f: float,
    internal_shunt_ohm: float,
) -> list[str]:
    lines: list[str] = []

    def term(hidden: int, prefix: str, source: str, err: str, weight: str, dest: str) -> list[str]:
        n0 = f"{prefix}_e"
        n1 = f"{prefix}_w"
        return [
            f"R{n0} {n0} 0 {internal_shunt_ohm:.12g}",
            f"R{n1} {n1} 0 {internal_shunt_ohm:.12g}",
            f"C{n0} {n0} 0 {internal_cap_f:.12g}f IC=0",
            f"C{n1} {n1} 0 {internal_cap_f:.12g}f IC=0",
            f"M{prefix}_e {source} {err} {n0} 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}_w {n0} {weight} {n1} 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}_a {n1} hrow{hidden} {dest} 0 NMOS W={width_u:.6g}u L=180n",
        ]

    for hidden in range(hidden_count):
        hdp = f"h{hidden}_hdp"
        hdn = f"h{hidden}_hdn"
        lines += [
            f"C{hdp} {hdp} 0 {capacitance_f:.12g}f IC=0",
            f"C{hdn} {hdn} 0 {capacitance_f:.12g}f IC=0",
            f"R{hdp} {hdp} 0 1G",
            f"R{hdn} {hdn} 0 1G",
            f"Mreset_{hdp} {hdp} rst 0 0 NMOS W=4u L=180n",
            f"Mreset_{hdn} {hdn} rst 0 0 NMOS W=4u L=180n",
        ]
        for output in range(OUTPUTS):
            errp = class_node(output, "errp")
            errn = class_node(output, "errn")
            vwp = class_node(output, f"vwp{hidden}")
            vwn = class_node(output, f"vwn{hidden}")
            prefix = f"h{hidden}_c{output}_cred"
            lines += term(hidden, f"{prefix}_pv", "vdd", errp, vwp, hdp)
            lines += term(hidden, f"{prefix}_pn", "vdd", errp, vwn, hdn)
            lines += term(hidden, f"{prefix}_nv", "vdd", errn, vwn, hdp)
            lines += term(hidden, f"{prefix}_nn", "vdd", errn, vwp, hdn)
    return lines


def _hidden_credit_gate_lines(hidden_count: int, width_u: float, cap_f: float, pull_scale: float) -> list[str]:
    lines: list[str] = []
    for hidden in range(hidden_count):
        raw_p = f"h{hidden}_hdp"
        raw_n = f"h{hidden}_hdn"
        gate_p = f"h{hidden}_hdp_gate"
        gate_n = f"h{hidden}_hdn_gate"
        mid_p = f"{gate_p}_mid"
        mid_n = f"{gate_n}_mid"
        lines += [
            f"C{gate_p} {gate_p} 0 {cap_f:.12g}f IC=0",
            f"C{gate_n} {gate_n} 0 {cap_f:.12g}f IC=0",
            f"R{gate_p} {gate_p} 0 1G",
            f"R{gate_n} {gate_n} 0 1G",
            f"R{mid_p} {mid_p} 0 1G",
            f"R{mid_n} {mid_n} 0 1G",
            f"Mreset_{gate_p} {gate_p} rst 0 0 NMOS W=4u L=180n",
            f"Mreset_{gate_n} {gate_n} rst 0 0 NMOS W=4u L=180n",
            f"M{gate_p}_up0 vdd {raw_p} {mid_p} 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{gate_p}_up1 {mid_p} {raw_p} {gate_p} 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{gate_p}_dn {gate_p} {raw_n} 0 0 NSENSE W={pull_scale * width_u:.6g}u L=180n",
            f"M{gate_n}_up0 vdd {raw_n} {mid_n} 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{gate_n}_up1 {mid_n} {raw_n} {gate_n} 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{gate_n}_dn {gate_n} {raw_p} 0 0 NSENSE W={pull_scale * width_u:.6g}u L=180n",
        ]
    return lines


def _hidden_writer_lines(
    feature_count: int,
    hidden_count: int,
    width_u: float,
    pmos_width_u: float,
    gate_cap_f: float,
) -> list[str]:
    lines = [
        "Vhidden_whi_ref hidden_whi_ref 0 1.05",
        "Vhidden_wlo_ref hidden_wlo_ref 0 0.15",
    ]

    def pmos_charge_lines(prefix: str, dest: str, selector: str, credit: str) -> list[str]:
        gate = f"{prefix}_pgate"
        mid = f"{prefix}_pgmid"
        phi = f"{prefix}_pgphi"
        return [
            f"C{gate} {gate} 0 {gate_cap_f:.12g}f IC=1.05",
            f"R{gate} {gate} hidden_whi_ref 1G",
            f"R{mid} {mid} 0 1G",
            f"R{phi} {phi} 0 1G",
            f"M{prefix}_pgate_rst hidden_whi_ref rst {gate} 0 NSENSE W=4u L=180n",
            f"M{prefix}_pgate_sel {gate} {selector} {mid} 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}_pgate_cred {mid} {credit} {phi} 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}_pgate_phi {phi} errphi 0 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}_pmos {dest} {gate} hidden_whi_ref vdd PMOS W={pmos_width_u:.6g}u L=180n",
        ]

    for hidden in range(hidden_count):
        hdp = f"h{hidden}_hdp_gate"
        hdn = f"h{hidden}_hdn_gate"
        for feature in range(feature_count):
            whp = _hidden_weight_node(hidden, feature, "p")
            whn = _hidden_weight_node(hidden, feature, "n")
            prefix = f"h{hidden}f{feature}_live_"
            lines += [
                f"R{prefix}pdn {prefix}pdn 0 1G",
                f"R{prefix}ndn {prefix}ndn 0 1G",
                f"M{prefix}pdn_e {whp} px{feature} {prefix}pdn 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}pdn_c {prefix}pdn {hdn} hidden_wlo_ref 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}ndn_e {whn} px{feature} {prefix}ndn 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}ndn_c {prefix}ndn {hdp} hidden_wlo_ref 0 NSENSE W={width_u:.6g}u L=180n",
                *pmos_charge_lines(f"{prefix}pup", whp, f"px{feature}", hdp),
                *pmos_charge_lines(f"{prefix}nup", whn, f"px{feature}", hdn),
            ]
    return lines


def _probe_hidden_feature(sample: dict[str, Any]) -> tuple[int, int]:
    features = np.asarray(sample["features"], dtype=float)
    feature = int(np.argmax(features))
    image_size = _image_size_from_feature_count(features.shape[0])
    return hidden_block_for_feature(feature, image_size), feature


def _measure_lines(samples: list[dict[str, Any]], eval_count: int, train_count: int) -> list[str]:
    lines: list[str] = []
    train_offset = eval_count
    final_offset = eval_count + train_count
    for idx, sample in enumerate(samples):
        base = idx * CYCLE_NS
        label = int(sample["label"])
        other = 1 - label
        if sample["phase"] in {"initial", "final"}:
            phase = str(sample["phase"])
            local = idx if phase == "initial" else idx - final_offset
            at = base + 3.15
            lines += [
                f".meas tran {phase}_target_scorep_{local} FIND V({class_node(label, 'scorep')}) AT={at:.2f}n",
                f".meas tran {phase}_target_scoren_{local} FIND V({class_node(label, 'scoren')}) AT={at:.2f}n",
                f".meas tran {phase}_other_scorep_{local} FIND V({class_node(other, 'scorep')}) AT={at:.2f}n",
                f".meas tran {phase}_other_scoren_{local} FIND V({class_node(other, 'scoren')}) AT={at:.2f}n",
                f".meas tran {phase}_target_signed_{local} PARAM='{phase}_target_scorep_{local}-{phase}_target_scoren_{local}'",
                f".meas tran {phase}_other_signed_{local} PARAM='{phase}_other_scorep_{local}-{phase}_other_scoren_{local}'",
                f".meas tran {phase}_margin_{local} PARAM='{phase}_target_signed_{local}-{phase}_other_signed_{local}'",
            ]
        if sample["phase"] == "train":
            local = idx - train_offset
            before = base + 0.55
            readout_after = base + 7.80
            hidden_after = base + 9.80
            err_at = base + 5.35
            hidden, feature = _probe_hidden_feature(sample)
            lines += [
                f".meas tran train_act_probe_{local} FIND V(act{hidden}) AT={base + 2.00:.2f}n",
                f".meas tran train_hrow_probe_{local} FIND V(hrow{hidden}) AT={base + 2.00:.2f}n",
                f".meas tran train_hrow_ctrl_probe_{local} FIND V(hrow{hidden}_ctrl) AT={base + 2.00:.2f}n",
                f".meas tran train_ir0_{local} FIND I(Vrsen0) AT={base + 5.00:.2f}n",
                f".meas tran train_ir1_{local} FIND I(Vrsen1) AT={base + 5.00:.2f}n",
                f".meas tran train_target_errp_{local} FIND V({class_node(label, 'errp')}) AT={err_at:.2f}n",
                f".meas tran train_target_errn_{local} FIND V({class_node(label, 'errn')}) AT={err_at:.2f}n",
                f".meas tran train_other_errp_{local} FIND V({class_node(other, 'errp')}) AT={err_at:.2f}n",
                f".meas tran train_other_errn_{local} FIND V({class_node(other, 'errn')}) AT={err_at:.2f}n",
                f".meas tran train_hdp_gate_probe_{local} FIND V(h{hidden}_hdp_gate) AT={err_at:.2f}n",
                f".meas tran train_hdn_gate_probe_{local} FIND V(h{hidden}_hdn_gate) AT={err_at:.2f}n",
                f".meas tran train_hcredit_gate_probe_{local} PARAM='train_hdp_gate_probe_{local}-train_hdn_gate_probe_{local}'",
            ]
            for output in (label, other):
                role = "target" if output == label else "other"
                lines += [
                    f".meas tran train_{role}_vwp_before_{local} FIND V({class_node(output, f'vwp{hidden}')}) AT={before:.2f}n",
                    f".meas tran train_{role}_vwn_before_{local} FIND V({class_node(output, f'vwn{hidden}')}) AT={before:.2f}n",
                    f".meas tran train_{role}_vwp_after_{local} FIND V({class_node(output, f'vwp{hidden}')}) AT={readout_after:.2f}n",
                    f".meas tran train_{role}_vwn_after_{local} FIND V({class_node(output, f'vwn{hidden}')}) AT={readout_after:.2f}n",
                    f".meas tran train_{role}_signed_before_{local} PARAM='train_{role}_vwp_before_{local}-train_{role}_vwn_before_{local}'",
                    f".meas tran train_{role}_signed_after_{local} PARAM='train_{role}_vwp_after_{local}-train_{role}_vwn_after_{local}'",
                    f".meas tran train_{role}_signed_delta_{local} PARAM='train_{role}_signed_after_{local}-train_{role}_signed_before_{local}'",
                ]
            lines += [
                f".meas tran train_wh_probe_p_before_{local} FIND V({_hidden_weight_node(hidden, feature, 'p')}) AT={before:.2f}n",
                f".meas tran train_wh_probe_n_before_{local} FIND V({_hidden_weight_node(hidden, feature, 'n')}) AT={before:.2f}n",
                f".meas tran train_wh_probe_p_after_{local} FIND V({_hidden_weight_node(hidden, feature, 'p')}) AT={hidden_after:.2f}n",
                f".meas tran train_wh_probe_n_after_{local} FIND V({_hidden_weight_node(hidden, feature, 'n')}) AT={hidden_after:.2f}n",
                f".meas tran train_wh_probe_signed_before_{local} PARAM='train_wh_probe_p_before_{local}-train_wh_probe_n_before_{local}'",
                f".meas tran train_wh_probe_signed_after_{local} PARAM='train_wh_probe_p_after_{local}-train_wh_probe_n_after_{local}'",
                f".meas tran train_wh_probe_signed_delta_{local} PARAM='train_wh_probe_signed_after_{local}-train_wh_probe_signed_before_{local}'",
            ]
    for idx in range(eval_count):
        lines.append(f".meas tran final_margin_improvement_{idx} PARAM='final_margin_{idx}-initial_margin_{idx}'")
    return lines


def mnist01_live_hidden_netlist(
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    *,
    hidden_count: int = HIDDEN,
    readout_initial_positive: float = 0.40,
    readout_initial_negative: float = 0.40,
    hidden_inside_positive: float = 1.05,
    hidden_outside_positive: float = 0.05,
    hidden_inside_negative: float = 0.05,
    hidden_outside_negative: float = 0.05,
    iref_a: float = 1.0e-6,
    hidden_forward_width_u: float = 8.0,
    readout_width_u: float = 16.0,
    branch_width_u: float = 0.05,
    floor_width_u: float = 0.015,
    route_width_u: float = 3.0,
    readout_update_width_u: float = 0.25,
    hidden_credit_width_u: float = 32.0,
    hidden_credit_cap_f: float = 12.0,
    hidden_credit_internal_cap_f: float = 0.05,
    hidden_credit_internal_shunt_ohm: float = 5.0e7,
    hidden_credit_gate_width_u: float = 4.0,
    hidden_credit_gate_cap_f: float = 2.0,
    hidden_credit_gate_pull_scale: float = 1.0,
    hidden_update_width_u: float = 1.0,
    hidden_writer_pmos_width_u: float = 4.0,
    hidden_writer_gate_cap_f: float = 0.2,
) -> str:
    if hidden_count != HIDDEN:
        raise ValueError("this small MNIST01 rung currently uses exactly four quadrant hidden units")
    if not train_records or not eval_records:
        raise ValueError("train and eval records must not be empty")
    feature_count = len(train_records[0]["features"])
    _validate_records(train_records, feature_count, "train")
    _validate_records(eval_records, feature_count, "eval")
    _image_size_from_feature_count(feature_count)
    if min(
        readout_initial_positive,
        readout_initial_negative,
        hidden_inside_positive,
        hidden_outside_positive,
        hidden_inside_negative,
        hidden_outside_negative,
        iref_a,
        hidden_forward_width_u,
        readout_width_u,
        branch_width_u,
        floor_width_u,
        route_width_u,
        readout_update_width_u,
        hidden_credit_width_u,
        hidden_credit_cap_f,
        hidden_credit_internal_cap_f,
        hidden_credit_internal_shunt_ohm,
        hidden_credit_gate_width_u,
        hidden_credit_gate_cap_f,
        hidden_credit_gate_pull_scale,
        hidden_update_width_u,
        hidden_writer_pmos_width_u,
        hidden_writer_gate_cap_f,
    ) <= 0.0:
        raise ValueError("voltages, currents, widths, and capacitances must be positive")

    samples, eval_count, train_count = _sample_plan(train_records, eval_records)
    stop_ns = len(samples) * CYCLE_NS
    lines = [
        "* Live-hidden MNIST01 learner with conductance-divider normalized error.",
        "* Python supplies pixels, labels, clocks, and diagnostics only.",
        "* Readout and hidden weights move through live transistor/passive writer paths.",
        ".param VDD=1.2",
        mos_models(),
        ".options method=gear maxord=2 reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        f"Vt0 t0 0 {_target_wave(samples, 0, stop_ns)}",
        f"Vt1 t1 0 {_target_wave(samples, 1, stop_ns)}",
        *_clock_lines(samples, stop_ns, iref_a),
    ]
    for feature in range(feature_count):
        lines.append(f"Vpx{feature} px{feature} 0 {_sample_feature_wave(samples, feature, stop_ns)}")
    lines += [
        *_hidden_storage_lines(
            feature_count,
            hidden_count,
            inside_positive=hidden_inside_positive,
            outside_positive=hidden_outside_positive,
            inside_negative=hidden_inside_negative,
            outside_negative=hidden_outside_negative,
        ),
        *_readout_storage_lines(hidden_count, readout_initial_positive, readout_initial_negative),
        *_hidden_state_lines(hidden_count),
        *_score_storage_lines(),
        *_hidden_forward_lines(feature_count, hidden_count, hidden_forward_width_u),
        *_score_readout_lines(hidden_count, readout_width_u),
        *_error_storage_lines(),
        *_divider_probability_lines(branch_width_u, floor_width_u),
        *_route_to_error_rails_lines(route_width_u),
        *_readout_writer_lines(hidden_count, readout_update_width_u),
        *_hidden_credit_lines(
            hidden_count,
            hidden_credit_width_u,
            hidden_credit_cap_f,
            hidden_credit_internal_cap_f,
            hidden_credit_internal_shunt_ohm,
        ),
        *_hidden_credit_gate_lines(
            hidden_count,
            hidden_credit_gate_width_u,
            hidden_credit_gate_cap_f,
            hidden_credit_gate_pull_scale,
        ),
        *_hidden_writer_lines(
            feature_count,
            hidden_count,
            hidden_update_width_u,
            hidden_writer_pmos_width_u,
            hidden_writer_gate_cap_f,
        ),
        f".tran 5p {stop_ns:.2f}n uic",
        *_measure_lines(samples, eval_count, train_count),
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def run_netlist(spice_bin: str, path: Path, netlist: str, timeout: float = 120.0) -> dict[str, float]:
    proc = run_text_netlist(spice_bin, path, netlist, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-3000:])
    parsed = parse_measures(proc.stdout + "\n" + proc.stderr)
    if not parsed:
        raise RuntimeError("SPICE produced no parseable measurements:\n" + (proc.stdout + proc.stderr)[-3000:])
    return parsed
