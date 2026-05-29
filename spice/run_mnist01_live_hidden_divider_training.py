from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from _util import parse_measures
from run_device_sequential_training import mos_models
from run_mnist01_fixed_feature_divider_training import (
    CYCLE_NS,
    OUTPUTS,
    _clock_lines as _fixed_clock_lines,
    _pulse_wave,
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
HIDDEN_INIT_MODES = ("quadrant", "identity", "patch2x2")
HIDDEN_CONNECTIVITY_MODES = ("dense", "identity-sparse", "patch2x2-sparse")
HIDDEN_WRITER_TOPOLOGIES = ("pmos-highside", "pmos-differential", "pmos-signcharge")
HIDDEN_WRITER_PHASE_MODES = ("default", "hidden-write")
READOUT_WRITER_NORMALIZATION_MODES = ("none", "activity-gate")
HIDDEN_CREDIT_ERROR_SOURCES = ("hidden", "readout")


def _clock_lines(
    samples: list[dict[str, Any]],
    stop_ns: float,
    iref_a: float,
    *,
    hidden_write_start_train_index: int = 0,
    hidden_credit_sense_start_ns: float = 5.00,
    hidden_credit_sense_end_ns: float = 6.15,
    hidden_write_start_ns: float = 6.30,
    hidden_write_end_ns: float = 8.40,
) -> list[str]:
    hidden_sense: list[tuple[float, float]] = []
    hidden_write: list[tuple[float, float]] = []
    train_idx = 0
    for idx, sample in enumerate(samples):
        if bool(sample["train"]):
            base = idx * CYCLE_NS
            hidden_sense.append((base + hidden_credit_sense_start_ns, base + hidden_credit_sense_end_ns))
            if train_idx >= hidden_write_start_train_index:
                hidden_write.append((base + hidden_write_start_ns, base + hidden_write_end_ns))
            train_idx += 1
    return [
        *_fixed_clock_lines(samples, stop_ns, iref_a),
        f"Vhcgphi hcgphi 0 {_pulse_wave(hidden_sense, stop_ns)}",
        f"Vhiddenwritephi hiddenwritephi 0 {_pulse_wave(hidden_write, stop_ns)}",
    ]


def hidden_block_for_feature(feature: int, image_size: int) -> int:
    if image_size % 2 != 0:
        raise ValueError("image_size must be even for four quadrant hidden units")
    row = feature // image_size
    col = feature % image_size
    if not 0 <= row < image_size or not 0 <= col < image_size:
        raise ValueError("feature index is outside the image")
    return (2 if row >= image_size // 2 else 0) + (1 if col >= image_size // 2 else 0)


def hidden_unit_for_feature(feature: int, feature_count: int, hidden_count: int, hidden_init_mode: str) -> int:
    if hidden_init_mode not in HIDDEN_INIT_MODES:
        raise ValueError(f"hidden_init_mode must be one of {HIDDEN_INIT_MODES}")
    if hidden_init_mode == "quadrant":
        if hidden_count != HIDDEN:
            raise ValueError("quadrant hidden_init_mode uses exactly four quadrant hidden units")
        image_size = _image_size_from_feature_count(feature_count)
        return hidden_block_for_feature(feature, image_size)
    if hidden_init_mode == "patch2x2":
        expected_hidden = patch2x2_hidden_count(feature_count)
        if hidden_count != expected_hidden:
            raise ValueError("patch2x2 hidden_init_mode uses one hidden unit per sliding 2x2 patch")
        return patch2x2_hidden_for_feature(feature, feature_count)
    if hidden_count != feature_count:
        raise ValueError("identity hidden_init_mode requires one hidden unit per input feature")
    if not 0 <= feature < feature_count:
        raise ValueError("feature index is outside the image")
    return feature


def _image_size_from_feature_count(feature_count: int) -> int:
    image_size = int(round(feature_count**0.5))
    if image_size * image_size != feature_count:
        raise ValueError("feature count must be a square image")
    return image_size


def _patch2x2_image_shape_from_feature_count(feature_count: int) -> tuple[int, int]:
    try:
        return _image_size_from_feature_count(feature_count), 1
    except ValueError:
        pass
    if feature_count % 2 != 0:
        raise ValueError("patch2x2 feature count must be a square image or two square feature channels")
    image_size = int(round((feature_count // 2) ** 0.5))
    if 2 * image_size * image_size != feature_count:
        raise ValueError("patch2x2 feature count must be a square image or two square feature channels")
    if image_size < 2:
        raise ValueError("patch2x2 hidden_init_mode requires at least a 2x2 image")
    return image_size, 2


def patch2x2_hidden_count(feature_count: int) -> int:
    image_size, channels = _patch2x2_image_shape_from_feature_count(feature_count)
    if image_size < 2:
        raise ValueError("patch2x2 hidden_init_mode requires at least a 2x2 image")
    return channels * (image_size - 1) * (image_size - 1)


def patch2x2_hidden_for_feature(feature: int, feature_count: int) -> int:
    image_size, channels = _patch2x2_image_shape_from_feature_count(feature_count)
    if image_size < 2:
        raise ValueError("patch2x2 hidden_init_mode requires at least a 2x2 image")
    pixels = image_size * image_size
    if not 0 <= feature < channels * pixels:
        raise ValueError("feature index is outside the image")
    local_feature = feature % pixels
    row = local_feature // image_size
    col = local_feature % image_size
    patch_row = min(row, image_size - 2)
    patch_col = min(col, image_size - 2)
    patch_count_per_channel = (image_size - 1) * (image_size - 1)
    channel = feature // pixels
    return channel * patch_count_per_channel + patch_row * (image_size - 1) + patch_col


def patch2x2_features_for_hidden(hidden: int, feature_count: int, hidden_count: int) -> tuple[int, ...]:
    image_size, channels = _patch2x2_image_shape_from_feature_count(feature_count)
    expected_hidden = patch2x2_hidden_count(feature_count)
    if hidden_count != expected_hidden:
        raise ValueError("patch2x2 hidden_init_mode uses one hidden unit per sliding 2x2 patch")
    if not 0 <= hidden < hidden_count:
        raise ValueError("hidden index is outside the patch2x2 hidden range")
    patch_count_per_channel = (image_size - 1) * (image_size - 1)
    channel = hidden // patch_count_per_channel
    patch_index = hidden % patch_count_per_channel
    patch_row = patch_index // (image_size - 1)
    patch_col = patch_index % (image_size - 1)
    top_left = patch_row * image_size + patch_col
    local_patch = (
        top_left,
        top_left + 1,
        top_left + image_size,
        top_left + image_size + 1,
    )
    pixels = image_size * image_size
    if channel >= channels:
        raise ValueError("hidden index is outside the patch2x2 hidden range")
    return tuple(channel * pixels + feature for feature in local_patch)


def _hidden_weight_node(hidden: int, feature: int, kind: str) -> str:
    return f"wh{hidden}f{feature}{kind}"


def _connected_features_for_hidden(
    hidden: int,
    feature_count: int,
    hidden_count: int,
    init_mode: str,
    connectivity_mode: str,
) -> range | tuple[int, ...]:
    if connectivity_mode not in HIDDEN_CONNECTIVITY_MODES:
        raise ValueError(f"hidden_connectivity_mode must be one of {HIDDEN_CONNECTIVITY_MODES}")
    if connectivity_mode == "dense":
        return range(feature_count)
    if connectivity_mode == "identity-sparse":
        if init_mode != "identity" or hidden_count != feature_count:
            raise ValueError("identity-sparse connectivity requires identity hidden rows")
        return range(hidden, hidden + 1)
    if init_mode != "patch2x2":
        raise ValueError("patch2x2-sparse connectivity requires patch2x2 hidden rows")
    return patch2x2_features_for_hidden(hidden, feature_count, hidden_count)


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
    init_mode: str,
    inside_positive: float,
    outside_positive: float,
    inside_negative: float,
    outside_negative: float,
    connectivity_mode: str = "dense",
) -> list[str]:
    if init_mode not in HIDDEN_INIT_MODES:
        raise ValueError(f"hidden_init_mode must be one of {HIDDEN_INIT_MODES}")
    if init_mode == "quadrant" and hidden_count != HIDDEN:
        raise ValueError("quadrant hidden_init_mode uses exactly four quadrant hidden units")
    if init_mode == "identity" and hidden_count != feature_count:
        raise ValueError("identity hidden_init_mode requires one hidden unit per input feature")
    if init_mode == "patch2x2" and hidden_count != patch2x2_hidden_count(feature_count):
        raise ValueError("patch2x2 hidden_init_mode uses one hidden unit per sliding 2x2 patch")
    image_size = _image_size_from_feature_count(feature_count) if init_mode == "quadrant" else 0
    lines: list[str] = []
    for hidden in range(hidden_count):
        patch_features = (
            set(patch2x2_features_for_hidden(hidden, feature_count, hidden_count))
            if init_mode == "patch2x2"
            else set()
        )
        for feature in _connected_features_for_hidden(hidden, feature_count, hidden_count, init_mode, connectivity_mode):
            if init_mode == "quadrant":
                inside = hidden_block_for_feature(feature, image_size) == hidden
            elif init_mode == "patch2x2":
                inside = feature in patch_features
            else:
                inside = hidden == feature
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


def _input_feature_common_gate_lines(
    feature_count: int,
    *,
    common_resistance_ohm: float,
    common_capacitance_f: float,
    gate_capacitance_f: float,
    contrast_capacitance_f: float,
    pullup_width_u: float,
    pulldown_width_u: float,
    pass_width_u: float,
) -> list[str]:
    if min(
        feature_count,
        common_resistance_ohm,
        common_capacitance_f,
        gate_capacitance_f,
        contrast_capacitance_f,
        pullup_width_u,
        pulldown_width_u,
        pass_width_u,
    ) <= 0.0:
        raise ValueError("input feature common-gate sizes must be positive")
    lines = [
        f"Cpx_common px_common 0 {common_capacitance_f:.12g}f IC=0",
        "Rpx_common_leak px_common 0 1G",
    ]
    for feature in range(feature_count):
        gate = f"pxgate{feature}"
        contrast = f"px_contrast{feature}"
        up_i = f"{gate}_up_i"
        dn_i = f"{gate}_dn_i"
        pass_mid = f"pxcontrast_f{feature}_pass_mid"
        lines += [
            f"Rpx_common_px{feature} px_common px{feature} {common_resistance_ohm:.12g}",
            f"C{gate} {gate} 0 {gate_capacitance_f:.12g}f IC=0",
            f"R{gate} {gate} 0 1G",
            f"M{gate}_rst {gate} rst 0 0 NMOS W=4u L=180n",
            f"R{up_i} {up_i} 0 1G",
            f"R{dn_i} {dn_i} 0 1G",
            f"M{gate}_up_v vdd px{feature} {up_i} 0 NSENSE W={pullup_width_u:.6g}u L=180n",
            f"M{gate}_up_t {up_i} featphi {gate} 0 NSENSE W={pullup_width_u:.6g}u L=180n",
            f"M{gate}_dn_v {gate} px_common {dn_i} 0 NSENSE W={pulldown_width_u:.6g}u L=180n",
            f"M{gate}_dn_t {dn_i} featphi 0 0 NSENSE W={pulldown_width_u:.6g}u L=180n",
            f"C{contrast} {contrast} 0 {contrast_capacitance_f:.12g}f IC=0",
            f"R{contrast} {contrast} 0 1G",
            f"M{contrast}_rst {contrast} rst 0 0 NMOS W=4u L=180n",
            f"R{pass_mid} {pass_mid} 0 1G",
            f"Mpxcontrast_f{feature}_pass_g px{feature} {gate} {pass_mid} 0 NSENSE W={pass_width_u:.6g}u L=180n",
            f"Mpxcontrast_f{feature}_pass_t {pass_mid} featphi {contrast} 0 NSENSE W={pass_width_u:.6g}u L=180n",
        ]
    return lines


def _input_feature_restored_gate_lines(
    feature_count: int,
    *,
    low_capacitance_f: float,
    drive_capacitance_f: float,
    discharge_width_u: float,
    restore_width_u: float,
) -> list[str]:
    if min(feature_count, low_capacitance_f, drive_capacitance_f, discharge_width_u, restore_width_u) <= 0.0:
        raise ValueError("input feature restored-gate sizes must be positive")
    lines: list[str] = []
    for feature in range(feature_count):
        low = f"pxgate{feature}_low"
        drive = f"pxdrive{feature}"
        low_mid = f"{low}_mid"
        lines += [
            f"C{low} {low} 0 {low_capacitance_f:.12g}f IC=1.2",
            f"R{low} {low} vdd 1G",
            f"M{low}_rst {low} rstn vdd vdd PMOS W=4u L=180n",
            f"R{low_mid} {low_mid} 0 1G",
            f"M{low}_dis_g {low} pxgate{feature} {low_mid} 0 NSENSE W={discharge_width_u:.6g}u L=180n",
            f"M{low}_dis_t {low_mid} featphi 0 0 NSENSE W={discharge_width_u:.6g}u L=180n",
            f"C{drive} {drive} 0 {drive_capacitance_f:.12g}f IC=0",
            f"R{drive} {drive} 0 1G",
            f"M{drive}_rst {drive} rst 0 0 NMOS W=4u L=180n",
            f"M{drive}_rest {drive} {low} vdd vdd PMOS W={restore_width_u:.6g}u L=180n",
        ]
    return lines


def _hidden_activation_common_gate_lines(
    hidden_count: int,
    *,
    common_resistance_ohm: float,
    gate_capacitance_f: float,
    contrast_capacitance_f: float,
    pullup_width_u: float,
    pulldown_width_u: float,
    pass_width_u: float,
) -> list[str]:
    if min(
        hidden_count,
        common_resistance_ohm,
        gate_capacitance_f,
        contrast_capacitance_f,
        pullup_width_u,
        pulldown_width_u,
        pass_width_u,
    ) <= 0.0:
        raise ValueError("hidden activation common-gate sizes must be positive")
    lines = ["Rhidden_act_common hidden_act_common 0 1G"]
    for hidden in range(hidden_count):
        gate = f"hactgate{hidden}"
        act_contrast = f"act_contrast{hidden}"
        up_i = f"{gate}_up_i"
        dn_i = f"{gate}_dn_i"
        pass_mid = f"hactcontrast_h{hidden}_pass_mid"
        lines += [
            f"Rhidden_act_common_act{hidden} hidden_act_common act{hidden} {common_resistance_ohm:.12g}",
            f"C{gate} {gate} 0 {gate_capacitance_f:.12g}f IC=0",
            f"R{gate} {gate} 0 1G",
            f"M{gate}_rst {gate} rst 0 0 NMOS W=4u L=180n",
            f"R{up_i} {up_i} 0 1G",
            f"R{dn_i} {dn_i} 0 1G",
            f"M{gate}_up_v vdd act{hidden} {up_i} 0 NSENSE W={pullup_width_u:.6g}u L=180n",
            f"M{gate}_up_t {up_i} featphi {gate} 0 NSENSE W={pullup_width_u:.6g}u L=180n",
            f"M{gate}_dn_v {gate} hidden_act_common {dn_i} 0 NSENSE W={pulldown_width_u:.6g}u L=180n",
            f"M{gate}_dn_t {dn_i} featphi 0 0 NSENSE W={pulldown_width_u:.6g}u L=180n",
            f"C{act_contrast} {act_contrast} 0 {contrast_capacitance_f:.12g}f IC=0",
            f"R{act_contrast} {act_contrast} 0 1G",
            f"M{act_contrast}_rst {act_contrast} rst 0 0 NMOS W=4u L=180n",
            f"R{pass_mid} {pass_mid} 0 1G",
            f"Mhactcontrast_h{hidden}_pass_g act{hidden} {gate} {pass_mid} 0 NSENSE W={pass_width_u:.6g}u L=180n",
            f"Mhactcontrast_h{hidden}_pass_t {pass_mid} featphi {act_contrast} 0 NSENSE W={pass_width_u:.6g}u L=180n",
        ]
    return lines


def _hidden_forward_lines(
    feature_count: int,
    hidden_count: int,
    width_u: float,
    *,
    activation_mode: str,
    activation_sense_width_u: float,
    input_mode: str = "raw",
    input_common_resistance_ohm: float = 20000.0,
    input_common_capacitance_f: float = 8.0,
    input_gate_capacitance_f: float = 8.0,
    input_contrast_capacitance_f: float = 20.0,
    input_pullup_width_u: float = 128.0,
    input_pulldown_width_u: float = 24.0,
    input_pass_width_u: float = 16.0,
    input_restored_low_capacitance_f: float = 1.0,
    input_restored_drive_capacitance_f: float = 4.0,
    input_restored_discharge_width_u: float = 24.0,
    input_restored_restore_width_u: float = 16.0,
    row_select_mode: str = "act",
    row_select_common_resistance_ohm: float = 100000.0,
    row_select_gate_capacitance_f: float = 8.0,
    row_select_contrast_capacitance_f: float = 20.0,
    row_select_pullup_width_u: float = 128.0,
    row_select_pulldown_width_u: float = 24.0,
    row_select_pass_width_u: float = 16.0,
    hidden_init_mode: str = "quadrant",
    hidden_connectivity_mode: str = "dense",
) -> list[str]:
    if activation_mode not in ("single-ended", "differential-preamp"):
        raise ValueError("hidden_activation_mode must be single-ended or differential-preamp")
    if input_mode not in ("raw", "contrast-common-gate", "restored-common-gate"):
        raise ValueError("hidden_input_mode must be raw, contrast-common-gate, or restored-common-gate")
    if row_select_mode not in ("act", "act-common-gate"):
        raise ValueError("hidden_row_select_mode must be act or act-common-gate")
    lines: list[str] = []
    if input_mode in ("contrast-common-gate", "restored-common-gate"):
        lines += _input_feature_common_gate_lines(
            feature_count,
            common_resistance_ohm=input_common_resistance_ohm,
            common_capacitance_f=input_common_capacitance_f,
            gate_capacitance_f=input_gate_capacitance_f,
            contrast_capacitance_f=input_contrast_capacitance_f,
            pullup_width_u=input_pullup_width_u,
            pulldown_width_u=input_pulldown_width_u,
            pass_width_u=input_pass_width_u,
        )
    if input_mode == "restored-common-gate":
        lines += _input_feature_restored_gate_lines(
            feature_count,
            low_capacitance_f=input_restored_low_capacitance_f,
            drive_capacitance_f=input_restored_drive_capacitance_f,
            discharge_width_u=input_restored_discharge_width_u,
            restore_width_u=input_restored_restore_width_u,
        )
    for hidden in range(hidden_count):
        pre_p = f"pre{hidden}_p"
        pre_n = f"pre{hidden}_n"
        act = f"act{hidden}"
        for feature in _connected_features_for_hidden(
            hidden,
            feature_count,
            hidden_count,
            hidden_init_mode,
            hidden_connectivity_mode,
        ):
            lines += [
                f"Rh{hidden}f{feature}pmid h{hidden}f{feature}pmid 0 1G",
                f"Ch{hidden}f{feature}pmid h{hidden}f{feature}pmid 0 0.05f IC=0",
                f"Rh{hidden}f{feature}nmid h{hidden}f{feature}nmid 0 1G",
                f"Ch{hidden}f{feature}nmid h{hidden}f{feature}nmid 0 0.05f IC=0",
            ]
            if input_mode == "raw":
                lines += [
                    f"Mh{hidden}f{feature}p_phi px{feature} featphi h{hidden}f{feature}pmid 0 NSENSE W={width_u:.6g}u L=180n",
                    f"Mh{hidden}f{feature}n_phi px{feature} featphi h{hidden}f{feature}nmid 0 NSENSE W={width_u:.6g}u L=180n",
                ]
            else:
                gate_node = f"pxdrive{feature}" if input_mode == "restored-common-gate" else f"pxgate{feature}"
                lines += [
                    f"Rh{hidden}f{feature}pinput h{hidden}f{feature}pinput 0 1G",
                    f"Ch{hidden}f{feature}pinput h{hidden}f{feature}pinput 0 0.05f IC=0",
                    f"Rh{hidden}f{feature}ninput h{hidden}f{feature}ninput 0 1G",
                    f"Ch{hidden}f{feature}ninput h{hidden}f{feature}ninput 0 0.05f IC=0",
                    f"Mh{hidden}f{feature}p_phi px{feature} featphi h{hidden}f{feature}pinput 0 NSENSE W={width_u:.6g}u L=180n",
                    f"Mh{hidden}f{feature}p_gate h{hidden}f{feature}pinput {gate_node} h{hidden}f{feature}pmid 0 NSENSE W={width_u:.6g}u L=180n",
                    f"Mh{hidden}f{feature}n_phi px{feature} featphi h{hidden}f{feature}ninput 0 NSENSE W={width_u:.6g}u L=180n",
                    f"Mh{hidden}f{feature}n_gate h{hidden}f{feature}ninput {gate_node} h{hidden}f{feature}nmid 0 NSENSE W={width_u:.6g}u L=180n",
                ]
            lines += [
                f"Mh{hidden}f{feature}p_w h{hidden}f{feature}pmid {_hidden_weight_node(hidden, feature, 'p')} {pre_p} 0 NSENSE W={width_u:.6g}u L=180n",
                f"Mh{hidden}f{feature}n_w h{hidden}f{feature}nmid {_hidden_weight_node(hidden, feature, 'n')} {pre_n} 0 NSENSE W={width_u:.6g}u L=180n",
            ]
        if activation_mode == "single-ended":
            lines += [
                f"Mact{hidden}_p vdd {pre_p} {act} 0 NREL W=24u L=180n",
                f"Mact{hidden}_n {act} {pre_n} 0 0 NSENSE W=24u L=180n",
            ]
        else:
            sense_p = f"h{hidden}_act_sense_p"
            sense_n = f"h{hidden}_act_sense_n"
            tail = f"h{hidden}_act_sense_tail"
            lines += [
                f"C{sense_p} {sense_p} 0 1f IC=1.2",
                f"C{sense_n} {sense_n} 0 1f IC=1.2",
                f"R{sense_p} {sense_p} vdd 1G",
                f"R{sense_n} {sense_n} vdd 1G",
                f"R{tail} {tail} 0 1G",
                f"M{sense_p}_rst {sense_p} rstn vdd vdd PMOS W=4u L=180n",
                f"M{sense_n}_rst {sense_n} rstn vdd vdd PMOS W=4u L=180n",
                f"Mh{hidden}_act_sense_tail {tail} featphi 0 0 NSENSE W={activation_sense_width_u:.6g}u L=180n",
                f"Mh{hidden}_act_sense_p {sense_n} {pre_p} {tail} 0 NSENSE W={activation_sense_width_u:.6g}u L=180n",
                f"Mh{hidden}_act_sense_n {sense_p} {pre_n} {tail} 0 NSENSE W={activation_sense_width_u:.6g}u L=180n",
                f"Mh{hidden}_act_sense_lp {sense_n} {sense_p} vdd vdd PMOS W={max(2.0, activation_sense_width_u / 8.0):.6g}u L=180n",
                f"Mh{hidden}_act_sense_ln {sense_p} {sense_n} vdd vdd PMOS W={max(2.0, activation_sense_width_u / 8.0):.6g}u L=180n",
                f"Mact{hidden}_diff_restore {act} {sense_n} vdd vdd PMOS W={max(8.0, activation_sense_width_u / 2.0):.6g}u L=180n",
            ]
    if row_select_mode == "act-common-gate":
        lines += _hidden_activation_common_gate_lines(
            hidden_count,
            common_resistance_ohm=row_select_common_resistance_ohm,
            gate_capacitance_f=row_select_gate_capacitance_f,
            contrast_capacitance_f=row_select_contrast_capacitance_f,
            pullup_width_u=row_select_pullup_width_u,
            pulldown_width_u=row_select_pulldown_width_u,
            pass_width_u=row_select_pass_width_u,
        )
    for hidden in range(hidden_count):
        hrow_source = f"act_contrast{hidden}" if row_select_mode == "act-common-gate" else f"act{hidden}"
        hrow_select_model = "NMOS" if row_select_mode == "act-common-gate" else "NSENSE"
        lines += [
            f"Chrow{hidden}_ctrl hrow{hidden}_ctrl 0 1f IC=1.2",
            f"Rhrow{hidden}_ctrl hrow{hidden}_ctrl vdd 1G",
            f"Rhrow{hidden}_mid hrow{hidden}_mid 0 1G",
            f"Mhrow{hidden}_ctrl_rst hrow{hidden}_ctrl rstn vdd vdd PMOS W=4u L=180n",
            f"Mhrow{hidden}_ctrl_a hrow{hidden}_ctrl {hrow_source} hrow{hidden}_mid 0 {hrow_select_model} W=12u L=180n",
            f"Mhrow{hidden}_ctrl_phi hrow{hidden}_mid featphi 0 0 NSENSE W=12u L=180n",
            f"Mhrow{hidden}_restore hrow{hidden} hrow{hidden}_ctrl vdd vdd PMOS W=16u L=180n",
        ]
    return lines


def _score_storage_lines() -> list[str]:
    lines: list[str] = []
    for output in range(OUTPUTS):
        for kind in ("scorep", "scoren", "errp", "errn", "herrp", "herrn"):
            node = class_node(output, kind)
            cap_f = 2.0 if "err" in kind else 8.0
            lines += [
                f"C{node} {node} 0 {cap_f:.12g}f IC=0",
                f"R{node} {node} 0 1G",
                f"Mreset_{node} {node} rst 0 0 NMOS W=4u L=180n",
            ]
    return lines


def _score_readout_lines(hidden_count: int, width_u: float, activation_mode: str = "hrow") -> list[str]:
    if activation_mode not in ("hrow", "pre-differential"):
        raise ValueError("readout_activation_mode must be hrow or pre-differential")
    lines: list[str] = []

    def term(prefix: str, activation_node: str, weight_node: str, dest: str) -> list[str]:
        node_a = f"{prefix}a"
        node_b = f"{prefix}b"
        return [
            f"R{node_a} {node_a} 0 1G",
            f"R{node_b} {node_b} 0 1G",
            f"C{node_a} {node_a} 0 0.05f IC=0",
            f"C{node_b} {node_b} 0 0.05f IC=0",
            f"M{prefix}a vdd {activation_node} {node_a} 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}w {node_a} {weight_node} {node_b} 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}phi {node_b} scorephi {dest} 0 NSENSE W={width_u:.6g}u L=180n",
        ]

    for output in range(OUTPUTS):
        scorep = class_node(output, "scorep")
        scoren = class_node(output, "scoren")
        for hidden in range(hidden_count):
            prefix = f"c{output}_h{hidden}_score_"
            if activation_mode == "hrow":
                lines += term(prefix + "p", f"hrow{hidden}", class_node(output, f"vwp{hidden}"), scorep)
                lines += term(prefix + "n", f"hrow{hidden}", class_node(output, f"vwn{hidden}"), scoren)
            else:
                pre_p = f"pre{hidden}_p"
                pre_n = f"pre{hidden}_n"
                vwp = class_node(output, f"vwp{hidden}")
                vwn = class_node(output, f"vwn{hidden}")
                lines += term(prefix + "pp", pre_p, vwp, scorep)
                lines += term(prefix + "nn", pre_n, vwn, scorep)
                lines += term(prefix + "pn", pre_p, vwn, scoren)
                lines += term(prefix + "np", pre_n, vwp, scoren)
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


def _route_to_hidden_error_rails_lines(route_width_u: float) -> list[str]:
    def charge_lines(name: str, dest: str, low: str, target: str) -> list[str]:
        return [
            f"R{name}_a {name}_a 0 1G",
            f"R{name}_b {name}_b 0 1G",
            f"M{name}_m vdd {low} {name}_a vdd PMOS W={route_width_u:.6g}u L=180n",
            f"M{name}_t {name}_a {target} {name}_b 0 NSENSE W=12u L=180n",
            f"M{name}_phi {name}_b errphi {dest} 0 NSENSE W=12u L=180n",
        ]

    lines: list[str] = []
    lines += charge_lines("herr_c0p", class_node(0, "herrp"), "b1low", "t0")
    lines += charge_lines("herr_c1n", class_node(1, "herrn"), "b1low", "t0")
    lines += charge_lines("herr_c1p", class_node(1, "herrp"), "b0low", "t1")
    lines += charge_lines("herr_c0n", class_node(0, "herrn"), "b0low", "t1")
    return lines


def _readout_writer_activity_normalization_lines(
    hidden_count: int,
    *,
    discharge_width_u: float,
    gate_capacitance_f: float,
    phase_node: str = "scorephi",
) -> list[str]:
    if min(
        hidden_count,
        discharge_width_u,
        gate_capacitance_f,
    ) <= 0.0:
        raise ValueError("readout writer activity normalization sizes must be positive")
    lines = [
        f"Chrow_activity_gate hrow_activity_gate 0 {gate_capacitance_f:.12g}f IC=1.2",
        "Rhrow_activity_gate hrow_activity_gate vdd 1G",
        "Mpre_hrow_activity_gate hrow_activity_gate rstn vdd vdd PMOS W=4u L=180n",
    ]
    for hidden in range(hidden_count):
        gate_mid = f"hrow_activity_gate_h{hidden}_mid"
        lines += [
            f"R{gate_mid} {gate_mid} 0 1G",
            f"C{gate_mid} {gate_mid} 0 0.05f IC=0",
            f"Mhrow_activity_gate_h{hidden}_src hrow_activity_gate hrow{hidden} {gate_mid} 0 NSENSE W={discharge_width_u:.6g}u L=180n",
            f"Mhrow_activity_gate_h{hidden}_phi {gate_mid} {phase_node} 0 0 NSENSE W={discharge_width_u:.6g}u L=180n",
        ]
    return lines


def _readout_writer_lines(
    hidden_count: int,
    width_u: float,
    activation_mode: str = "hrow",
    normalization_mode: str = "none",
    *,
    normalization_discharge_width_u: float = 0.02,
    normalization_gate_capacitance_f: float = 80.0,
) -> list[str]:
    if activation_mode not in ("hrow", "pre-differential"):
        raise ValueError("readout_writer_activation_mode must be hrow or pre-differential")
    if normalization_mode not in READOUT_WRITER_NORMALIZATION_MODES:
        raise ValueError(f"readout_writer_normalization_mode must be one of {READOUT_WRITER_NORMALIZATION_MODES}")
    if activation_mode == "pre-differential" and normalization_mode != "none":
        raise ValueError("readout_writer_normalization_mode is only supported for hrow, not pre-differential")
    lines = [
        "Vvwhi_ref vwhi_ref 0 0.48",
        "Vvwlo_ref vwlo_ref 0 0.22",
    ]
    update_guard_node = None
    if normalization_mode == "activity-gate":
        lines += _readout_writer_activity_normalization_lines(
            hidden_count,
            discharge_width_u=normalization_discharge_width_u,
            gate_capacitance_f=normalization_gate_capacitance_f,
        )
        update_guard_node = "hrow_activity_gate"
    for output in range(OUTPUTS):
        for hidden in range(hidden_count):
            if activation_mode == "hrow":
                lines += class_local_live_label_descent_update_lines(
                    class_idx=output,
                    feature_idx=hidden,
                    activation_node=f"hrow{hidden}",
                    positive_descent_node=class_node(output, "errp"),
                    negative_descent_node=class_node(output, "errn"),
                    update_guard_node=update_guard_node,
                    update_guard_model="NREL" if update_guard_node is not None else "NSENSE",
                    width_u=width_u,
                    high_side_topology="pmos-differential",
                )
            else:
                lines += class_local_live_label_descent_update_lines(
                    class_idx=output,
                    feature_idx=hidden,
                    activation_node=f"pre{hidden}_p",
                    positive_descent_node=class_node(output, "errp"),
                    negative_descent_node=class_node(output, "errn"),
                    width_u=width_u,
                    high_side_topology="pmos-differential",
                    prefix_suffix="prep_",
                )
                lines += class_local_live_label_descent_update_lines(
                    class_idx=output,
                    feature_idx=hidden,
                    activation_node=f"pre{hidden}_n",
                    positive_descent_node=class_node(output, "errn"),
                    negative_descent_node=class_node(output, "errp"),
                    width_u=width_u,
                    high_side_topology="pmos-differential",
                    prefix_suffix="pren_",
                )
    return lines


def _hidden_credit_lines(
    hidden_count: int,
    width_u: float,
    capacitance_f: float,
    internal_cap_f: float,
    internal_shunt_ohm: float,
    error_source: str = "hidden",
) -> list[str]:
    if error_source not in HIDDEN_CREDIT_ERROR_SOURCES:
        raise ValueError(f"hidden_credit_error_source must be one of {HIDDEN_CREDIT_ERROR_SOURCES}")
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
            err_prefix = "herr" if error_source == "hidden" else "err"
            errp = class_node(output, f"{err_prefix}p")
            errn = class_node(output, f"{err_prefix}n")
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


def _hidden_credit_dynamic_preamp_gate_lines(
    hidden_count: int,
    *,
    sense_width_u: float,
    latch_pmos_width_u: float,
    output_width_u: float,
    output_pull_width_u: float,
    support_width_u: float,
    write_gate_width_u: float,
    capacitance_f: float,
) -> list[str]:
    lines: list[str] = []
    for hidden in range(hidden_count):
        raw_p = f"h{hidden}_hdp"
        raw_n = f"h{hidden}_hdn"
        gate_p = f"h{hidden}_hdp_gate"
        gate_n = f"h{hidden}_hdn_gate"
        pre_p = f"h{hidden}_hcg_pre_p"
        pre_n = f"h{hidden}_hcg_pre_n"
        tail = f"h{hidden}_hcg_tail"
        pos_mid = f"h{hidden}_hcg_pos_mid"
        neg_mid = f"h{hidden}_hcg_neg_mid"
        support = f"h{hidden}_hcg_support"
        write_mid = f"h{hidden}_hcg_write_mid"
        write = f"h{hidden}_hcg_write"
        lines += [
            f"C{support} {support} 0 {capacitance_f:.12g}f IC=0",
            f"R{support} {support} 0 1G",
            f"M{hidden}_hcg_support_rst {support} rst 0 0 NMOS W=4u L=180n",
            f"M{hidden}_hcg_support_p vdd {raw_p} {support} 0 NSENSE W={support_width_u:.6g}u L=180n",
            f"M{hidden}_hcg_support_n vdd {raw_n} {support} 0 NSENSE W={support_width_u:.6g}u L=180n",
            f"R{write_mid} {write_mid} 0 1G",
            f"R{write} {write} 0 1G",
            f"M{hidden}_hcg_write_support hiddenwritephi {support} {write_mid} 0 NSENSE W={write_gate_width_u:.6g}u L=180n",
            f"M{hidden}_hcg_write_buf {write_mid} hiddenwritephi {write} 0 NSENSE W={write_gate_width_u:.6g}u L=180n",
            f"C{pre_p} {pre_p} 0 {capacitance_f:.12g}f IC=1.2",
            f"C{pre_n} {pre_n} 0 {capacitance_f:.12g}f IC=1.2",
            f"R{pre_p} {pre_p} 0 1G",
            f"R{pre_n} {pre_n} 0 1G",
            f"M{pre_p}_rst {pre_p} rstn vdd vdd PMOS W=8u L=180n",
            f"M{pre_n}_rst {pre_n} rstn vdd vdd PMOS W=8u L=180n",
            f"M{hidden}_hcg_eq {pre_p} rst {pre_n} 0 NMOS W=2u L=180n",
            f"R{tail} {tail} 0 1G",
            f"M{hidden}_hcg_tail {tail} hcgphi 0 0 NSENSE W={sense_width_u:.6g}u L=180n",
            f"M{hidden}_hcg_sense_p {pre_n} {raw_p} {tail} 0 NSENSE W={sense_width_u:.6g}u L=180n",
            f"M{hidden}_hcg_sense_n {pre_p} {raw_n} {tail} 0 NSENSE W={sense_width_u:.6g}u L=180n",
            f"M{hidden}_hcg_latch_p {pre_n} {pre_p} vdd vdd PMOS W={latch_pmos_width_u:.6g}u L=180n",
            f"M{hidden}_hcg_latch_n {pre_p} {pre_n} vdd vdd PMOS W={latch_pmos_width_u:.6g}u L=180n",
            f"C{gate_p} {gate_p} 0 {capacitance_f:.12g}f IC=0",
            f"C{gate_n} {gate_n} 0 {capacitance_f:.12g}f IC=0",
            f"R{gate_p} {gate_p} 0 1G",
            f"R{gate_n} {gate_n} 0 1G",
            f"Mreset_{gate_p} {gate_p} rst 0 0 NMOS W=4u L=180n",
            f"Mreset_{gate_n} {gate_n} rst 0 0 NMOS W=4u L=180n",
            f"R{pos_mid} {pos_mid} 0 1G",
            f"R{neg_mid} {neg_mid} 0 1G",
            f"M{hidden}_hcg_pos_pmos vdd {pre_n} {pos_mid} vdd PMOS W={output_width_u:.6g}u L=180n",
            f"M{hidden}_hcg_pos_nmos {pos_mid} {pre_p} {gate_p} 0 NSENSE W={output_width_u:.6g}u L=180n",
            f"M{hidden}_hcg_pos_pull {gate_p} {pre_n} 0 0 NMOS W={output_pull_width_u:.6g}u L=180n",
            f"M{hidden}_hcg_neg_pmos vdd {pre_p} {neg_mid} vdd PMOS W={output_width_u:.6g}u L=180n",
            f"M{hidden}_hcg_neg_nmos {neg_mid} {pre_n} {gate_n} 0 NSENSE W={output_width_u:.6g}u L=180n",
            f"M{hidden}_hcg_neg_pull {gate_n} {pre_p} 0 0 NMOS W={output_pull_width_u:.6g}u L=180n",
        ]
    return lines


def _hidden_writer_lines(
    feature_count: int,
    hidden_count: int,
    width_u: float,
    pmos_width_u: float,
    gate_cap_f: float,
    topology: str,
    phase_node: str = "errphi",
    phase_low_side: bool = False,
    hidden_init_mode: str = "quadrant",
    hidden_connectivity_mode: str = "dense",
    signcharge_packet_cap_f: float = 0.25,
) -> list[str]:
    if topology not in HIDDEN_WRITER_TOPOLOGIES:
        raise ValueError(f"hidden_writer_topology must be one of {HIDDEN_WRITER_TOPOLOGIES}")
    if signcharge_packet_cap_f <= 0.0:
        raise ValueError("signcharge packet capacitance must be positive")
    lines = [
        "Vhidden_whi_ref hidden_whi_ref 0 1.05",
        "Vhidden_wlo_ref hidden_wlo_ref 0 0.15",
    ]

    def pmos_charge_lines(
        prefix: str,
        dest: str,
        selector: str,
        credit: str,
        phase: str,
        *,
        packet_cap_f: float | None = None,
    ) -> list[str]:
        gate = f"{prefix}_pgate"
        mid = f"{prefix}_pgmid"
        phi = f"{prefix}_pgphi"
        source = "hidden_whi_ref" if packet_cap_f is None else f"{prefix}_packet"
        return [
            f"C{gate} {gate} 0 {gate_cap_f:.12g}f IC=1.05",
            f"R{gate} {gate} hidden_whi_ref 1G",
            f"R{mid} {mid} 0 1G",
            f"R{phi} {phi} 0 1G",
            f"C{mid} {mid} 0 0.001f IC=0",
            f"C{phi} {phi} 0 0.001f IC=0",
            *(
                [
                    f"C{source} {source} 0 {packet_cap_f:.12g}f IC=1.05",
                    f"R{source} {source} hidden_whi_ref 1G",
                    f"M{prefix}_packet_rst {source} rstn hidden_whi_ref vdd PMOS W=4u L=180n",
                ]
                if packet_cap_f is not None
                else []
            ),
            f"M{prefix}_pgate_rst hidden_whi_ref rst {gate} 0 NSENSE W=4u L=180n",
            f"M{prefix}_pgate_sel {gate} {selector} {mid} 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}_pgate_cred {mid} {credit} {phi} 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}_pgate_phi {phi} {phase} 0 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}_pmos {dest} {gate} {source} vdd PMOS W={pmos_width_u:.6g}u L=180n",
        ]

    def pmos_differential_lines(prefix: str, whp: str, whn: str, selector: str, pos: str, neg: str, phase: str) -> list[str]:
        pos_ctrl = f"{prefix}pos_up_ctrl"
        neg_ctrl = f"{prefix}neg_up_ctrl"
        pos_ctrl_mid = f"{prefix}pos_up_ctrl_mid"
        pos_ctrl_phi = f"{prefix}pos_up_ctrl_phi"
        neg_ctrl_mid = f"{prefix}neg_up_ctrl_mid"
        neg_ctrl_phi = f"{prefix}neg_up_ctrl_phi"
        pos_dn = f"{prefix}pos_dn"
        neg_dn = f"{prefix}neg_dn"
        pos_dn_sel = f"{prefix}pos_dn_sel"
        neg_dn_sel = f"{prefix}neg_dn_sel"
        pos_dn_final = f"{prefix}pos_dn_phi" if phase_low_side else pos_dn_sel
        neg_dn_final = f"{prefix}neg_dn_phi" if phase_low_side else neg_dn_sel
        out: list[str] = [
            f"R{prefix}pos_dn_shunt {pos_dn} 0 1G",
            f"R{prefix}neg_dn_shunt {neg_dn} 0 1G",
            f"R{prefix}pos_dn_sel_shunt {pos_dn_sel} 0 1G",
            f"R{prefix}neg_dn_sel_shunt {neg_dn_sel} 0 1G",
            f"C{prefix}pos_dn_par {pos_dn} 0 0.05f IC=0",
            f"C{prefix}neg_dn_par {neg_dn} 0 0.05f IC=0",
            f"C{prefix}pos_dn_sel_par {pos_dn_sel} 0 0.05f IC=0",
            f"C{prefix}neg_dn_sel_par {neg_dn_sel} 0 0.05f IC=0",
            *(
                [
                    f"R{prefix}pos_dn_phi_shunt {prefix}pos_dn_phi 0 1G",
                    f"R{prefix}neg_dn_phi_shunt {prefix}neg_dn_phi 0 1G",
                    f"C{prefix}pos_dn_phi_par {prefix}pos_dn_phi 0 0.05f IC=0",
                    f"C{prefix}neg_dn_phi_par {prefix}neg_dn_phi 0 0.05f IC=0",
                    f"M{prefix}pos_dn_phi {pos_dn_sel} {phase} {prefix}pos_dn_phi 0 NSENSE W={width_u:.6g}u L=180n",
                    f"M{prefix}neg_dn_phi {neg_dn_sel} {phase} {prefix}neg_dn_phi 0 NSENSE W={width_u:.6g}u L=180n",
                ]
                if phase_low_side
                else []
            ),
            f"M{prefix}pos_dn_e {whn} {selector} {pos_dn} 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}neg_dn_e {whp} {selector} {neg_dn} 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}pos_dn_select {pos_dn} {neg_ctrl} {pos_dn_sel} 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}neg_dn_select {neg_dn} {pos_ctrl} {neg_dn_sel} 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}pos_dn_d {pos_dn_final} {pos} hidden_wlo_ref 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}neg_dn_d {neg_dn_final} {neg} hidden_wlo_ref 0 NSENSE W={width_u:.6g}u L=180n",
            f"C{pos_ctrl} {pos_ctrl} 0 {gate_cap_f:.12g}f IC=1.2",
            f"C{neg_ctrl} {neg_ctrl} 0 {gate_cap_f:.12g}f IC=1.2",
            f"R{pos_ctrl} {pos_ctrl} vdd 1G",
            f"R{neg_ctrl} {neg_ctrl} vdd 1G",
            f"R{pos_ctrl_mid} {pos_ctrl_mid} 0 1G",
            f"R{pos_ctrl_phi} {pos_ctrl_phi} 0 1G",
            f"R{neg_ctrl_mid} {neg_ctrl_mid} 0 1G",
            f"R{neg_ctrl_phi} {neg_ctrl_phi} 0 1G",
            f"M{prefix}pos_up_ctrl_rst vdd rst {pos_ctrl} 0 NSENSE W=4u L=180n",
            f"M{prefix}neg_up_ctrl_rst vdd rst {neg_ctrl} 0 NSENSE W=4u L=180n",
            f"M{prefix}pos_up_ctrl_e {pos_ctrl} {selector} {pos_ctrl_mid} 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}pos_up_ctrl_d {pos_ctrl_mid} {pos} {pos_ctrl_phi} 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}pos_up_ctrl_phi {pos_ctrl_phi} {phase} 0 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}neg_up_ctrl_e {neg_ctrl} {selector} {neg_ctrl_mid} 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}neg_up_ctrl_d {neg_ctrl_mid} {neg} {neg_ctrl_phi} 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}neg_up_ctrl_phi {neg_ctrl_phi} {phase} 0 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}pos_up_ctrl_latch {pos_ctrl} {neg_ctrl} vdd vdd PMOS W={pmos_width_u:.6g}u L=180n",
            f"M{prefix}neg_up_ctrl_latch {neg_ctrl} {pos_ctrl} vdd vdd PMOS W={pmos_width_u:.6g}u L=180n",
            f"M{prefix}pos_up_p {whp} {pos_ctrl} hidden_whi_ref vdd PMOS W={pmos_width_u:.6g}u L=180n",
            f"M{prefix}neg_up_p {whn} {neg_ctrl} hidden_whi_ref vdd PMOS W={pmos_width_u:.6g}u L=180n",
        ]
        return out

    for hidden in range(hidden_count):
        hdp = f"h{hidden}_hdp_gate"
        hdn = f"h{hidden}_hdn_gate"
        hidden_phase = phase_node.format(hidden=hidden)
        for feature in _connected_features_for_hidden(
            hidden,
            feature_count,
            hidden_count,
            hidden_init_mode,
            hidden_connectivity_mode,
        ):
            whp = _hidden_weight_node(hidden, feature, "p")
            whn = _hidden_weight_node(hidden, feature, "n")
            prefix = f"h{hidden}f{feature}_live_"
            if topology == "pmos-highside":
                lines += [
                    f"R{prefix}pdn {prefix}pdn 0 1G",
                    f"R{prefix}ndn {prefix}ndn 0 1G",
                    f"M{prefix}pdn_e {whp} px{feature} {prefix}pdn 0 NSENSE W={width_u:.6g}u L=180n",
                    f"M{prefix}pdn_c {prefix}pdn {hdn} hidden_wlo_ref 0 NSENSE W={width_u:.6g}u L=180n",
                    f"M{prefix}ndn_e {whn} px{feature} {prefix}ndn 0 NSENSE W={width_u:.6g}u L=180n",
                    f"M{prefix}ndn_c {prefix}ndn {hdp} hidden_wlo_ref 0 NSENSE W={width_u:.6g}u L=180n",
                    *pmos_charge_lines(f"{prefix}pup", whp, f"px{feature}", hdp, hidden_phase),
                    *pmos_charge_lines(f"{prefix}nup", whn, f"px{feature}", hdn, hidden_phase),
                ]
            elif topology == "pmos-differential":
                lines += pmos_differential_lines(prefix, whp, whn, f"px{feature}", hdp, hdn, hidden_phase)
            else:
                lines += pmos_charge_lines(
                    f"{prefix}pup",
                    whp,
                    f"px{feature}",
                    hdp,
                    hidden_phase,
                    packet_cap_f=signcharge_packet_cap_f,
                )
                lines += pmos_charge_lines(
                    f"{prefix}nup",
                    whn,
                    f"px{feature}",
                    hdn,
                    hidden_phase,
                    packet_cap_f=signcharge_packet_cap_f,
                )
    return lines


def _probe_hidden_feature(sample: dict[str, Any], hidden_count: int = HIDDEN, hidden_init_mode: str = "quadrant") -> tuple[int, int]:
    features = np.asarray(sample["features"], dtype=float)
    feature = int(np.argmax(features))
    return hidden_unit_for_feature(feature, features.shape[0], hidden_count, hidden_init_mode), feature


def _measure_lines(
    samples: list[dict[str, Any]],
    eval_count: int,
    train_count: int,
    *,
    hidden_count: int = HIDDEN,
    hidden_init_mode: str = "quadrant",
    measure_eval_hidden_states: bool = True,
    hidden_write_probe_ns: float = 7.35,
) -> list[str]:
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
            act_at = base + 2.00
            at = base + 3.15
            if measure_eval_hidden_states:
                for hidden in range(hidden_count):
                    lines += [
                        f".meas tran {phase}_prep_h{hidden}_{local} FIND V(pre{hidden}_p) AT={act_at:.2f}n",
                        f".meas tran {phase}_pren_h{hidden}_{local} FIND V(pre{hidden}_n) AT={act_at:.2f}n",
                        f".meas tran {phase}_pre_signed_h{hidden}_{local} PARAM='{phase}_prep_h{hidden}_{local}-{phase}_pren_h{hidden}_{local}'",
                        f".meas tran {phase}_act_h{hidden}_{local} FIND V(act{hidden}) AT={act_at:.2f}n",
                        f".meas tran {phase}_hrow_h{hidden}_{local} FIND V(hrow{hidden}) AT={act_at:.2f}n",
                    ]
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
            score_at = base + 3.15
            readout_after = base + 7.80
            hidden_after = base + 9.80
            err_at = base + 5.35
            hidden, feature = _probe_hidden_feature(sample, hidden_count, hidden_init_mode)
            lines += [
                f".meas tran train_target_scorep_{local} FIND V({class_node(label, 'scorep')}) AT={score_at:.2f}n",
                f".meas tran train_target_scoren_{local} FIND V({class_node(label, 'scoren')}) AT={score_at:.2f}n",
                f".meas tran train_other_scorep_{local} FIND V({class_node(other, 'scorep')}) AT={score_at:.2f}n",
                f".meas tran train_other_scoren_{local} FIND V({class_node(other, 'scoren')}) AT={score_at:.2f}n",
                f".meas tran train_target_signed_{local} PARAM='train_target_scorep_{local}-train_target_scoren_{local}'",
                f".meas tran train_other_signed_{local} PARAM='train_other_scorep_{local}-train_other_scoren_{local}'",
                f".meas tran train_margin_{local} PARAM='train_target_signed_{local}-train_other_signed_{local}'",
                f".meas tran train_act_probe_{local} FIND V(act{hidden}) AT={base + 2.00:.2f}n",
                f".meas tran train_hrow_probe_{local} FIND V(hrow{hidden}) AT={base + 2.00:.2f}n",
                f".meas tran train_hrow_ctrl_probe_{local} FIND V(hrow{hidden}_ctrl) AT={base + 2.00:.2f}n",
                f".meas tran train_ir0_{local} FIND I(Vrsen0) AT={base + 5.00:.2f}n",
                f".meas tran train_ir1_{local} FIND I(Vrsen1) AT={base + 5.00:.2f}n",
                f".meas tran train_target_errp_{local} FIND V({class_node(label, 'errp')}) AT={err_at:.2f}n",
                f".meas tran train_target_errn_{local} FIND V({class_node(label, 'errn')}) AT={err_at:.2f}n",
                f".meas tran train_other_errp_{local} FIND V({class_node(other, 'errp')}) AT={err_at:.2f}n",
                f".meas tran train_other_errn_{local} FIND V({class_node(other, 'errn')}) AT={err_at:.2f}n",
                f".meas tran train_target_herrp_{local} FIND V({class_node(label, 'herrp')}) AT={err_at:.2f}n",
                f".meas tran train_target_herrn_{local} FIND V({class_node(label, 'herrn')}) AT={err_at:.2f}n",
                f".meas tran train_other_herrp_{local} FIND V({class_node(other, 'herrp')}) AT={err_at:.2f}n",
                f".meas tran train_other_herrn_{local} FIND V({class_node(other, 'herrn')}) AT={err_at:.2f}n",
                f".meas tran train_hdp_gate_probe_{local} FIND V(h{hidden}_hdp_gate) AT={err_at:.2f}n",
                f".meas tran train_hdn_gate_probe_{local} FIND V(h{hidden}_hdn_gate) AT={err_at:.2f}n",
                f".meas tran train_hcredit_gate_probe_{local} PARAM='train_hdp_gate_probe_{local}-train_hdn_gate_probe_{local}'",
                f".meas tran train_hdp_gate_write_probe_{local} FIND V(h{hidden}_hdp_gate) AT={base + hidden_write_probe_ns:.2f}n",
                f".meas tran train_hdn_gate_write_probe_{local} FIND V(h{hidden}_hdn_gate) AT={base + hidden_write_probe_ns:.2f}n",
                f".meas tran train_hcredit_gate_write_probe_{local} PARAM='train_hdp_gate_write_probe_{local}-train_hdn_gate_write_probe_{local}'",
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


def forward_metric_rows(
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    measures: dict[str, float],
    *,
    loss_margin_scale_v: float = 1.0e-3,
) -> list[dict[str, Any]]:
    if loss_margin_scale_v <= 0.0:
        raise ValueError("loss_margin_scale_v must be positive")
    feature_count = len(train_records[0]["features"]) if train_records else 0
    _validate_records(train_records, feature_count, "train")
    _validate_records(eval_records, feature_count, "eval")
    samples, eval_count, train_count = _sample_plan(train_records, eval_records)
    final_offset = eval_count + train_count
    rows: list[dict[str, Any]] = []
    total_correct = 0
    total_loss = 0.0
    phase_counts: dict[str, int] = {}
    phase_correct: dict[str, int] = {}
    phase_loss: dict[str, float] = {}
    for sample_idx, sample in enumerate(samples):
        phase = str(sample["phase"])
        if phase == "initial":
            local = sample_idx
        elif phase == "train":
            local = sample_idx - eval_count
        elif phase == "final":
            local = sample_idx - final_offset
        else:
            raise ValueError(f"unknown sample phase {phase!r}")
        prefix = phase
        target_signed = measures[f"{prefix}_target_signed_{local}"]
        other_signed = measures[f"{prefix}_other_signed_{local}"]
        margin = measures[f"{prefix}_margin_{local}"]
        loss = float(np.logaddexp(0.0, -margin / loss_margin_scale_v))
        correct = margin > 0.0
        total_correct += int(correct)
        total_loss += loss
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        phase_correct[phase] = phase_correct.get(phase, 0) + int(correct)
        phase_loss[phase] = phase_loss.get(phase, 0.0) + loss
        count = len(rows) + 1
        rows.append(
            {
                "forward_index": len(rows),
                "cycle_index": sample_idx,
                "phase": phase,
                "phase_index": local,
                "label": int(sample["label"]),
                "target_signed_v": target_signed,
                "other_signed_v": other_signed,
                "margin_v": margin,
                "correct": int(correct),
                "softplus_loss": loss,
                "cumulative_accuracy": total_correct / count,
                "cumulative_mean_loss": total_loss / count,
                "phase_cumulative_accuracy": phase_correct[phase] / phase_counts[phase],
                "phase_cumulative_mean_loss": phase_loss[phase] / phase_counts[phase],
            }
        )
    return rows


def mnist01_live_hidden_netlist(
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    *,
    hidden_count: int = HIDDEN,
    hidden_init_mode: str = "quadrant",
    hidden_connectivity_mode: str = "dense",
    readout_initial_positive: float = 0.40,
    readout_initial_negative: float = 0.40,
    hidden_inside_positive: float = 1.05,
    hidden_outside_positive: float = 0.05,
    hidden_inside_negative: float = 0.05,
    hidden_outside_negative: float = 0.05,
    iref_a: float = 1.0e-6,
    hidden_forward_width_u: float = 8.0,
    hidden_activation_mode: str = "single-ended",
    hidden_activation_sense_width_u: float = 4.0,
    hidden_input_mode: str = "raw",
    hidden_input_common_resistance_ohm: float = 20000.0,
    hidden_input_common_capacitance_f: float = 8.0,
    hidden_input_gate_capacitance_f: float = 8.0,
    hidden_input_contrast_capacitance_f: float = 20.0,
    hidden_input_pullup_width_u: float = 128.0,
    hidden_input_pulldown_width_u: float = 24.0,
    hidden_input_pass_width_u: float = 16.0,
    hidden_input_restored_low_capacitance_f: float = 1.0,
    hidden_input_restored_drive_capacitance_f: float = 4.0,
    hidden_input_restored_discharge_width_u: float = 24.0,
    hidden_input_restored_restore_width_u: float = 16.0,
    hidden_row_select_mode: str = "act",
    hidden_row_select_common_resistance_ohm: float = 100000.0,
    hidden_row_select_gate_capacitance_f: float = 8.0,
    hidden_row_select_contrast_capacitance_f: float = 20.0,
    hidden_row_select_pullup_width_u: float = 128.0,
    hidden_row_select_pulldown_width_u: float = 24.0,
    hidden_row_select_pass_width_u: float = 16.0,
    readout_activation_mode: str = "hrow",
    readout_writer_activation_mode: str = "hrow",
    readout_writer_normalization_mode: str = "none",
    readout_writer_normalization_discharge_width_u: float = 0.02,
    readout_writer_normalization_gate_capacitance_f: float = 80.0,
    readout_width_u: float = 16.0,
    branch_width_u: float = 0.05,
    floor_width_u: float = 0.015,
    route_width_u: float = 3.0,
    hidden_error_route_width_u: float = 16.0,
    readout_update_width_u: float = 0.25,
    hidden_credit_width_u: float = 32.0,
    hidden_credit_cap_f: float = 12.0,
    hidden_credit_internal_cap_f: float = 0.05,
    hidden_credit_internal_shunt_ohm: float = 5.0e7,
    hidden_credit_error_source: str = "hidden",
    hidden_credit_gate_width_u: float = 4.0,
    hidden_credit_gate_cap_f: float = 2.0,
    hidden_credit_gate_pull_scale: float = 2.0,
    hidden_credit_gate_mode: str = "differential-excess",
    hidden_credit_preamp_sense_width_u: float = 32.0,
    hidden_credit_preamp_latch_pmos_width_u: float = 4.0,
    hidden_credit_preamp_output_width_u: float = 2.0,
    hidden_credit_preamp_output_pull_width_u: float = 0.5,
    hidden_credit_preamp_support_width_u: float = 4.0,
    hidden_credit_preamp_write_gate_width_u: float = 8.0,
    hidden_write_start_train_index: int = 0,
    hidden_credit_sense_start_ns: float = 5.00,
    hidden_credit_sense_end_ns: float = 6.15,
    hidden_write_start_ns: float = 6.30,
    hidden_write_end_ns: float = 8.40,
    hidden_update_width_u: float = 1.0,
    hidden_writer_pmos_width_u: float = 4.0,
    hidden_writer_gate_cap_f: float = 0.2,
    hidden_writer_signcharge_packet_cap_f: float = 0.25,
    hidden_writer_topology: str = "pmos-highside",
    hidden_writer_phase_mode: str = "default",
    measure_eval_hidden_states: bool = True,
    tran_step_ps: float = 5.0,
) -> str:
    if not train_records or not eval_records:
        raise ValueError("train and eval records must not be empty")
    feature_count = len(train_records[0]["features"])
    _validate_records(train_records, feature_count, "train")
    _validate_records(eval_records, feature_count, "eval")
    samples, eval_count, train_count = _sample_plan(train_records, eval_records)
    if hidden_init_mode not in HIDDEN_INIT_MODES:
        raise ValueError(f"hidden_init_mode must be one of {HIDDEN_INIT_MODES}")
    if hidden_connectivity_mode not in HIDDEN_CONNECTIVITY_MODES:
        raise ValueError(f"hidden_connectivity_mode must be one of {HIDDEN_CONNECTIVITY_MODES}")
    if hidden_init_mode == "quadrant" and hidden_count != HIDDEN:
        raise ValueError("quadrant hidden_init_mode uses exactly four quadrant hidden units")
    if hidden_init_mode == "identity" and hidden_count != feature_count:
        raise ValueError("identity hidden_init_mode requires one hidden unit per input feature")
    if hidden_connectivity_mode == "identity-sparse" and hidden_init_mode != "identity":
        raise ValueError("identity-sparse connectivity requires identity hidden rows")
    if hidden_connectivity_mode == "patch2x2-sparse" and hidden_init_mode != "patch2x2":
        raise ValueError("patch2x2-sparse connectivity requires patch2x2 hidden rows")
    if hidden_init_mode == "patch2x2" and hidden_count != patch2x2_hidden_count(feature_count):
        raise ValueError("patch2x2 hidden_init_mode uses one hidden unit per sliding 2x2 patch")
    if hidden_init_mode == "quadrant":
        _image_size_from_feature_count(feature_count)
    if hidden_credit_gate_mode not in ("differential-excess", "dynamic-preamp"):
        raise ValueError("hidden_credit_gate_mode must be differential-excess or dynamic-preamp")
    if hidden_credit_error_source not in HIDDEN_CREDIT_ERROR_SOURCES:
        raise ValueError(f"hidden_credit_error_source must be one of {HIDDEN_CREDIT_ERROR_SOURCES}")
    if hidden_activation_mode not in ("single-ended", "differential-preamp"):
        raise ValueError("hidden_activation_mode must be single-ended or differential-preamp")
    if hidden_input_mode not in ("raw", "contrast-common-gate", "restored-common-gate"):
        raise ValueError("hidden_input_mode must be raw, contrast-common-gate, or restored-common-gate")
    if hidden_row_select_mode not in ("act", "act-common-gate"):
        raise ValueError("hidden_row_select_mode must be act or act-common-gate")
    if readout_activation_mode not in ("hrow", "pre-differential"):
        raise ValueError("readout_activation_mode must be hrow or pre-differential")
    if readout_writer_activation_mode not in ("hrow", "pre-differential"):
        raise ValueError("readout_writer_activation_mode must be hrow or pre-differential")
    if readout_writer_normalization_mode not in READOUT_WRITER_NORMALIZATION_MODES:
        raise ValueError(
            f"readout_writer_normalization_mode must be one of {READOUT_WRITER_NORMALIZATION_MODES}"
        )
    if readout_writer_activation_mode == "pre-differential" and readout_writer_normalization_mode != "none":
        raise ValueError("readout_writer_normalization_mode is only supported for hrow, not pre-differential")
    if hidden_writer_topology not in HIDDEN_WRITER_TOPOLOGIES:
        raise ValueError(f"hidden_writer_topology must be one of {HIDDEN_WRITER_TOPOLOGIES}")
    if hidden_writer_phase_mode not in HIDDEN_WRITER_PHASE_MODES:
        raise ValueError(f"hidden_writer_phase_mode must be one of {HIDDEN_WRITER_PHASE_MODES}")
    if hidden_credit_gate_mode == "dynamic-preamp" and hidden_writer_topology == "pmos-highside":
        raise ValueError("dynamic-preamp hidden credit gate requires pmos-differential or pmos-signcharge hidden writer topology")
    if hidden_credit_gate_mode == "dynamic-preamp" and hidden_writer_phase_mode != "default":
        raise ValueError("dynamic-preamp hidden credit gate owns the hidden writer phase")
    if hidden_write_start_train_index < 0:
        raise ValueError("hidden_write_start_train_index must be nonnegative")
    if hidden_writer_phase_mode == "hidden-write" and hidden_write_start_train_index < train_count:
        raise ValueError(
            "active ordinary hiddenwritephi writes are not supported; use dynamic-preamp for active hidden writes "
            "or set hidden_write_start_train_index after the train count to disable ordinary hidden writes"
        )
    if not (0.0 <= hidden_credit_sense_start_ns < hidden_credit_sense_end_ns <= CYCLE_NS):
        raise ValueError("hidden credit sense window must be inside one cycle")
    if not (0.0 <= hidden_write_start_ns < hidden_write_end_ns <= CYCLE_NS):
        raise ValueError("hidden write window must be inside one cycle")
    if min(
        hidden_count,
        readout_initial_positive,
        readout_initial_negative,
        hidden_inside_positive,
        hidden_outside_positive,
        hidden_inside_negative,
        hidden_outside_negative,
        iref_a,
        hidden_forward_width_u,
        hidden_activation_sense_width_u,
        hidden_input_common_resistance_ohm,
        hidden_input_common_capacitance_f,
        hidden_input_gate_capacitance_f,
        hidden_input_contrast_capacitance_f,
        hidden_input_pullup_width_u,
        hidden_input_pulldown_width_u,
        hidden_input_pass_width_u,
        hidden_input_restored_low_capacitance_f,
        hidden_input_restored_drive_capacitance_f,
        hidden_input_restored_discharge_width_u,
        hidden_input_restored_restore_width_u,
        hidden_row_select_common_resistance_ohm,
        hidden_row_select_gate_capacitance_f,
        hidden_row_select_contrast_capacitance_f,
        hidden_row_select_pullup_width_u,
        hidden_row_select_pulldown_width_u,
        hidden_row_select_pass_width_u,
        readout_writer_normalization_discharge_width_u,
        readout_writer_normalization_gate_capacitance_f,
        readout_width_u,
        branch_width_u,
        floor_width_u,
        route_width_u,
        hidden_error_route_width_u,
        readout_update_width_u,
        hidden_credit_width_u,
        hidden_credit_cap_f,
        hidden_credit_internal_cap_f,
        hidden_credit_internal_shunt_ohm,
        hidden_credit_gate_width_u,
        hidden_credit_gate_cap_f,
        hidden_credit_gate_pull_scale,
        hidden_credit_preamp_sense_width_u,
        hidden_credit_preamp_latch_pmos_width_u,
        hidden_credit_preamp_output_width_u,
        hidden_credit_preamp_output_pull_width_u,
        hidden_credit_preamp_support_width_u,
        hidden_credit_preamp_write_gate_width_u,
        hidden_credit_sense_end_ns - hidden_credit_sense_start_ns,
        hidden_write_end_ns - hidden_write_start_ns,
        hidden_update_width_u,
        hidden_writer_pmos_width_u,
        hidden_writer_gate_cap_f,
        hidden_writer_signcharge_packet_cap_f,
        tran_step_ps,
    ) <= 0.0:
        raise ValueError("voltages, currents, widths, and capacitances must be positive")

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
        *_clock_lines(
            samples,
            stop_ns,
            iref_a,
            hidden_write_start_train_index=hidden_write_start_train_index,
            hidden_credit_sense_start_ns=hidden_credit_sense_start_ns,
            hidden_credit_sense_end_ns=hidden_credit_sense_end_ns,
            hidden_write_start_ns=hidden_write_start_ns,
            hidden_write_end_ns=hidden_write_end_ns,
        ),
    ]
    for feature in range(feature_count):
        lines.append(f"Vpx{feature} px{feature} 0 {_sample_feature_wave(samples, feature, stop_ns)}")
    lines += [
        *_hidden_storage_lines(
            feature_count,
            hidden_count,
            init_mode=hidden_init_mode,
            inside_positive=hidden_inside_positive,
            outside_positive=hidden_outside_positive,
            inside_negative=hidden_inside_negative,
            outside_negative=hidden_outside_negative,
            connectivity_mode=hidden_connectivity_mode,
        ),
        *_readout_storage_lines(hidden_count, readout_initial_positive, readout_initial_negative),
        *_hidden_state_lines(hidden_count),
        *_score_storage_lines(),
        *_hidden_forward_lines(
            feature_count,
            hidden_count,
            hidden_forward_width_u,
            activation_mode=hidden_activation_mode,
            activation_sense_width_u=hidden_activation_sense_width_u,
            input_mode=hidden_input_mode,
            input_common_resistance_ohm=hidden_input_common_resistance_ohm,
            input_common_capacitance_f=hidden_input_common_capacitance_f,
            input_gate_capacitance_f=hidden_input_gate_capacitance_f,
            input_contrast_capacitance_f=hidden_input_contrast_capacitance_f,
            input_pullup_width_u=hidden_input_pullup_width_u,
            input_pulldown_width_u=hidden_input_pulldown_width_u,
            input_pass_width_u=hidden_input_pass_width_u,
            input_restored_low_capacitance_f=hidden_input_restored_low_capacitance_f,
            input_restored_drive_capacitance_f=hidden_input_restored_drive_capacitance_f,
            input_restored_discharge_width_u=hidden_input_restored_discharge_width_u,
            input_restored_restore_width_u=hidden_input_restored_restore_width_u,
            row_select_mode=hidden_row_select_mode,
            row_select_common_resistance_ohm=hidden_row_select_common_resistance_ohm,
            row_select_gate_capacitance_f=hidden_row_select_gate_capacitance_f,
            row_select_contrast_capacitance_f=hidden_row_select_contrast_capacitance_f,
            row_select_pullup_width_u=hidden_row_select_pullup_width_u,
            row_select_pulldown_width_u=hidden_row_select_pulldown_width_u,
            row_select_pass_width_u=hidden_row_select_pass_width_u,
            hidden_init_mode=hidden_init_mode,
            hidden_connectivity_mode=hidden_connectivity_mode,
        ),
        *_score_readout_lines(hidden_count, readout_width_u, activation_mode=readout_activation_mode),
        *_error_storage_lines(),
        *_divider_probability_lines(branch_width_u, floor_width_u),
        *_route_to_error_rails_lines(route_width_u),
        *_route_to_hidden_error_rails_lines(hidden_error_route_width_u),
        *_readout_writer_lines(
            hidden_count,
            readout_update_width_u,
            activation_mode=readout_writer_activation_mode,
            normalization_mode=readout_writer_normalization_mode,
            normalization_discharge_width_u=readout_writer_normalization_discharge_width_u,
            normalization_gate_capacitance_f=readout_writer_normalization_gate_capacitance_f,
        ),
        *_hidden_credit_lines(
            hidden_count,
            hidden_credit_width_u,
            hidden_credit_cap_f,
            hidden_credit_internal_cap_f,
            hidden_credit_internal_shunt_ohm,
            hidden_credit_error_source,
        ),
        *(
            _hidden_credit_gate_lines(
                hidden_count,
                hidden_credit_gate_width_u,
                hidden_credit_gate_cap_f,
                hidden_credit_gate_pull_scale,
            )
            if hidden_credit_gate_mode == "differential-excess"
            else _hidden_credit_dynamic_preamp_gate_lines(
                hidden_count,
                sense_width_u=hidden_credit_preamp_sense_width_u,
                latch_pmos_width_u=hidden_credit_preamp_latch_pmos_width_u,
                output_width_u=hidden_credit_preamp_output_width_u,
                output_pull_width_u=hidden_credit_preamp_output_pull_width_u,
                support_width_u=hidden_credit_preamp_support_width_u,
                write_gate_width_u=hidden_credit_preamp_write_gate_width_u,
                capacitance_f=hidden_credit_gate_cap_f,
            )
        ),
        *_hidden_writer_lines(
            feature_count,
            hidden_count,
            hidden_update_width_u,
            hidden_writer_pmos_width_u,
            hidden_writer_gate_cap_f,
            hidden_writer_topology,
            (
                "h{hidden}_hcg_write"
                if hidden_credit_gate_mode == "dynamic-preamp"
                else ("hiddenwritephi" if hidden_writer_phase_mode == "hidden-write" else "errphi")
            ),
            hidden_credit_gate_mode == "dynamic-preamp",
            hidden_init_mode=hidden_init_mode,
            hidden_connectivity_mode=hidden_connectivity_mode,
            signcharge_packet_cap_f=hidden_writer_signcharge_packet_cap_f,
        ),
        f".tran {tran_step_ps:.12g}p {stop_ns:.2f}n uic",
        *_measure_lines(
            samples,
            eval_count,
            train_count,
            hidden_count=hidden_count,
            hidden_init_mode=hidden_init_mode,
            measure_eval_hidden_states=measure_eval_hidden_states,
            hidden_write_probe_ns=(hidden_write_start_ns + hidden_write_end_ns) / 2.0,
        ),
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
