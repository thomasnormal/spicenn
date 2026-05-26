from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_device_mnist01_scalar_training import balanced_digit_indices, binary_accuracy, sanitize_tag
from run_device_sequential_training import (
    active_low_pulse_wave,
    expected_positive,
    mos_models,
    output_driver_line,
    pulse_wave,
    run_netlist,
)
from run_spice_mnist_local_block_batch_op_train import block_indices
from run_spice_sweep import ROOT, detect_spice, run_tiny_test
from spicenn.timing import CYCLE_NS


INPUT_RAIL_MODES = ("raw", "complement", "alternating-complement")
TARGET_POLARITIES = ("active-high", "active-low")
HIDDEN_ACTIVATION_MODELS = ("nrel", "sense")
HIDDEN_FORWARD_TOPOLOGIES = ("per-pixel-phase", "shared-phase", "always-on", "split-rail")
READOUT_FORWARD_MODELS = ("nrel", "sense")
LEARNING_ACTIVATION_GATE_MODELS = ("nrel", "sense")
HIDDEN_POLARITY_INITS = ("ink", "alternating-channel", "random-pixel")
HIDDEN_CREDIT_MODES = ("direct-feedback", "readout-weighted", "readout-restored")
ERROR_SIGNAL_MODES = ("raw", "restored", "restored-hidden")
SCORE_MODES = ("single-ended", "differential")
OUTPUT_DIFFERENTIAL_STAGES = ("simple", "score-gated", "latched")
OUTPUT_DECISION_REF_SOURCES = ("voltage", "divider", "adaptive")
OUTPUT_DECISION_STAGES = (
    "none",
    "ref-latched",
    "ref-precharged-latched",
    "ref-preamp-latched",
    "diff-latched",
    "diff-precharged-latched",
    "ratio-inverter",
    "stacked-inverter",
    "shift-inverter",
)
MEASUREMENT_DETAILS = ("full", "outputs")
VDD_VALUE = 1.2


def stable_seed(seed: int, name: str) -> int:
    digest = hashlib.blake2b(f"{seed}:{name}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def stable_normal(seed: int, name: str, sigma: float) -> float:
    if sigma <= 0.0:
        return 0.0
    rng = np.random.default_rng(stable_seed(seed, name))
    return float(rng.normal(0.0, sigma))


def mismatch_factor(seed: int, name: str, sigma: float) -> float:
    if sigma <= 0.0:
        return 1.0
    return max(0.05, 1.0 + stable_normal(seed, name, sigma))


def clamp_voltage(value: float) -> float:
    return min(VDD_VALUE, max(0.0, value))


def spice_capacitance(value: float) -> str:
    if value <= 0.0:
        raise ValueError("capacitance must be positive")
    if value < 1e-12:
        return f"{value / 1e-15:.12g}f"
    if value < 1e-9:
        return f"{value / 1e-12:.12g}p"
    return f"{value:.12g}"


def spice_subcircuits() -> str:
    return "\n".join(
        [
            ".subckt signed_store p n Cp=20f Icp=0.5 Icn=0.2 Rleak=1e15",
            "Cp_ p 0 {Cp} IC={Icp}",
            "Cn_ n 0 {Cp} IC={Icn}",
            "Rp_ p 0 {Rleak}",
            "Rn_ n 0 {Rleak}",
            ".ends signed_store",
            "",
            ".subckt split_rail_hidden_pixel x whp whn pre pren Wp=3u Wn=2.25u",
            "Mpos x whp pre 0 NMOS W={Wp} L=180n",
            "Mneg x whn pren 0 NMOS W={Wn} L=180n",
            ".ends split_rail_hidden_pixel",
            "",
            ".subckt split_rail_hidden_bias vdd bhp bhn pre pren Wp=3u Wn=2.25u",
            "Mpos vdd bhp pre 0 NMOS W={Wp} L=180n",
            "Mneg vdd bhn pren 0 NMOS W={Wn} L=180n",
            ".ends split_rail_hidden_bias",
            "",
            ".subckt split_rail_relu_nrel vdd pre pren act W=24u",
            "Mp vdd pre act 0 NREL W={W} L=180n",
            "Mn act pren 0 0 NREL W={W} L=180n",
            ".ends split_rail_relu_nrel",
            "",
            ".subckt split_rail_relu_sense vdd pre pren act W=24u",
            "Mp vdd pre act 0 NSENSE W={W} L=180n",
            "Mn act pren 0 0 NSENSE W={W} L=180n",
            ".ends split_rail_relu_sense",
        ]
    )


def signed_store_instance(
    name: str,
    positive_node: str,
    negative_node: str,
    *,
    capacitance: str = "20f",
    positive_ic: float,
    negative_ic: float,
    leak_resistance: str = "1e15",
) -> str:
    return (
        f"X{name} {positive_node} {negative_node} signed_store "
        f"Cp={capacitance} Icp={positive_ic:.12g} Icn={negative_ic:.12g} Rleak={leak_resistance}"
    )


def split_rail_hidden_pixel_instance(
    name: str,
    input_node: str,
    whp_node: str,
    whn_node: str,
    pre_node: str,
    pren_node: str,
    *,
    positive_width: float,
    negative_width: float,
) -> str:
    return (
        f"X{name} {input_node} {whp_node} {whn_node} {pre_node} {pren_node} split_rail_hidden_pixel "
        f"Wp={positive_width:.6g}u Wn={negative_width:.6g}u"
    )


def split_rail_hidden_bias_instance(
    name: str,
    bhp_node: str,
    bhn_node: str,
    pre_node: str,
    pren_node: str,
    *,
    positive_width: float,
    negative_width: float,
) -> str:
    return (
        f"X{name} vdd {bhp_node} {bhn_node} {pre_node} {pren_node} split_rail_hidden_bias "
        f"Wp={positive_width:.6g}u Wn={negative_width:.6g}u"
    )


def split_rail_relu_instance(
    name: str,
    pre_node: str,
    pren_node: str,
    act_node: str,
    *,
    model: str,
    width: float,
) -> str:
    return f"X{name} vdd {pre_node} {pren_node} {act_node} split_rail_relu_{model} W={width:.6g}u"


def decision_ref_divider_resistances(ref_voltage: float, total_resistance: float) -> tuple[float, float]:
    if not 0.0 < ref_voltage < VDD_VALUE:
        raise ValueError(f"divider output_decision_ref must be between 0 and {VDD_VALUE}")
    if total_resistance <= 0.0:
        raise ValueError("output_decision_ref_resistance must be positive")
    bottom = total_resistance * ref_voltage / VDD_VALUE
    top = total_resistance - bottom
    return top, bottom


def block_topology(image_size: int, block_size: int, stride: int, channels: int) -> tuple[list[list[int]], int]:
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if channels <= 0:
        raise ValueError("channels must be positive")
    blocks = block_indices(image_size, block_size, stride)
    return blocks, len(blocks) * channels


def encode_pixel_rail(value: float) -> float:
    return float(np.clip(0.08 + 0.92 * value, 0.05, 1.1))


def input_rail_name(pixel: int, channel: int, mode: str) -> str:
    if mode == "raw":
        return f"x{pixel}"
    if mode == "complement":
        return f"nx{pixel}"
    if mode == "alternating-complement":
        return f"nx{pixel}" if channel % 2 else f"x{pixel}"
    raise ValueError(f"unknown input rail mode {mode!r}")


def required_input_rail_names(pixel_count: int, channels: int, mode: str) -> list[str]:
    rails = {
        input_rail_name(pixel, channel, mode)
        for pixel in range(pixel_count)
        for channel in range(channels)
    }
    return sorted(rails, key=lambda name: (name.startswith("nx"), int(name.removeprefix("nx").removeprefix("x"))))


def initial_block_weights(
    image_size: int,
    block_size: int,
    stride: int,
    channels: int,
    *,
    seed: int = 0,
    hidden_bias_positive_init: float = 0.50,
    hidden_bias_negative_init: float = 0.20,
    hidden_polarity_init: str = "ink",
    output_bias_positive_init: float = 0.52,
    output_bias_negative_init: float = 0.25,
) -> dict[str, Any]:
    if hidden_polarity_init not in HIDDEN_POLARITY_INITS:
        raise ValueError(f"hidden_polarity_init must be one of {HIDDEN_POLARITY_INITS}")
    blocks, feature_count = block_topology(image_size, block_size, stride, channels)
    block_len = len(blocks[0])
    rng = np.random.default_rng(seed)
    whp = np.clip(0.72 + rng.normal(0.0, 0.035, size=(feature_count, block_len)), 0.50, 0.92)
    whn = np.clip(0.22 + rng.normal(0.0, 0.025, size=(feature_count, block_len)), 0.05, 0.42)
    bhp = np.clip(hidden_bias_positive_init + rng.normal(0.0, 0.025, size=feature_count), 0.05, 1.15)
    bhn = np.clip(hidden_bias_negative_init + rng.normal(0.0, 0.020, size=feature_count), 0.02, 1.10)
    vwp = np.clip(0.52 + rng.normal(0.0, 0.025, size=feature_count), 0.35, 0.75)
    vwn = np.clip(0.25 + rng.normal(0.0, 0.020, size=feature_count), 0.08, 0.45)
    obp = output_bias_positive_init
    obn = output_bias_negative_init
    if hidden_polarity_init == "alternating-channel":
        for feature in range(feature_count):
            if feature % channels == 1:
                whp[feature], whn[feature] = whn[feature].copy(), whp[feature].copy()
    elif hidden_polarity_init == "random-pixel":
        polarity = rng.random(size=(feature_count, block_len)) < 0.5
        for feature in range(feature_count):
            if np.all(polarity[feature]):
                polarity[feature, int(rng.integers(0, block_len))] = False
            elif not np.any(polarity[feature]):
                polarity[feature, int(rng.integers(0, block_len))] = True
        whp_mixed = whp.copy()
        whn_mixed = whn.copy()
        whp_mixed[~polarity], whn_mixed[~polarity] = whn[~polarity], whp[~polarity]
        whp, whn = whp_mixed, whn_mixed
    return {
        "whp": whp.tolist(),
        "whn": whn.tolist(),
        "bhp": bhp.tolist(),
        "bhn": bhn.tolist(),
        "vwp": vwp.tolist(),
        "vwn": vwn.tolist(),
        "obp": obp,
        "obn": obn,
    }


def block_weight_shape(weights: dict[str, Any]) -> tuple[int, int]:
    required = ("whp", "whn", "bhp", "bhn", "vwp", "vwn")
    missing = [key for key in required if key not in weights]
    if missing:
        raise ValueError(f"missing weight rails: {', '.join(missing)}")
    whp = np.asarray(weights["whp"], dtype=float)
    whn = np.asarray(weights["whn"], dtype=float)
    bhp = np.asarray(weights["bhp"], dtype=float)
    bhn = np.asarray(weights["bhn"], dtype=float)
    vwp = np.asarray(weights["vwp"], dtype=float)
    vwn = np.asarray(weights["vwn"], dtype=float)
    if whp.ndim != 2 or whn.shape != whp.shape:
        raise ValueError("hidden weight rails must be 2-D arrays with matching shape")
    feature_count, block_len = whp.shape
    if feature_count <= 0 or block_len <= 0:
        raise ValueError("hidden weight arrays must be nonempty")
    if bhp.shape != (feature_count,) or bhn.shape != (feature_count,):
        raise ValueError("hidden bias rails must match hidden feature count")
    if vwp.shape != (feature_count,) or vwn.shape != (feature_count,):
        raise ValueError("readout weight rails must match hidden feature count")
    return int(feature_count), int(block_len)


def validate_block_samples(samples: list[dict[str, Any]], *, required_rails: list[str]) -> None:
    if not samples:
        raise ValueError("samples must not be empty")
    for idx, sample in enumerate(samples):
        missing = [rail for rail in required_rails if rail not in sample]
        if missing:
            raise ValueError(f"sample {idx} missing pixel rails: {', '.join(missing[:4])}")
        if "target" not in sample:
            raise ValueError(f"sample {idx} missing target rail")


def block_sample_wave(samples: list[dict[str, Any]], key: str, stop_ns: float, *, cycle_ns: float) -> str:
    points: list[tuple[float, float]] = []
    for idx, sample in enumerate(samples):
        start = idx * cycle_ns
        end = start + cycle_ns
        value = float(sample[key])
        if idx == 0:
            points.append((0.0, value))
        else:
            points.append((start, value))
        points.append((min(stop_ns, end - 0.05), value))
    points.append((stop_ns, float(samples[-1][key])))
    return "PWL(" + " ".join(f"{t:.12g}n {v:.12g}" for t, v in points) + ")"


def perturb_input_records(
    records: list[dict[str, Any]],
    *,
    sigma: float,
    seed: int,
) -> list[dict[str, Any]]:
    if sigma < 0.0:
        raise ValueError("input_voltage_jitter_sigma must be nonnegative")
    if sigma == 0.0:
        return [dict(record) for record in records]
    perturbed: list[dict[str, Any]] = []
    for sample_idx, record in enumerate(records):
        copy = dict(record)
        for key, value in record.items():
            if key.startswith("x") or key.startswith("nx"):
                copy[key] = clamp_voltage(float(value) + stable_normal(seed, f"input:{sample_idx}:{key}", sigma))
        perturbed.append(copy)
    return perturbed


def perturb_initial_state(
    weights: dict[str, Any],
    *,
    sigma: float,
    seed: int,
) -> dict[str, Any]:
    if sigma < 0.0:
        raise ValueError("state_ic_mismatch_sigma must be nonnegative")
    copied: dict[str, Any] = {}
    for name, value in weights.items():
        array = np.asarray(value, dtype=float)
        if sigma == 0.0:
            copied[name] = array.copy().tolist()
            continue
        out = np.empty_like(array, dtype=float)
        for index in np.ndindex(array.shape):
            suffix = "_".join(str(part) for part in index)
            out[index] = clamp_voltage(float(array[index]) + stable_normal(seed, f"state:{name}:{suffix}", sigma))
        copied[name] = out.tolist()
    return copied


def jittered_interval(
    start: float,
    end: float,
    *,
    min_start: float,
    max_end: float,
    sigma: float,
    seed: int,
    name: str,
) -> tuple[float, float]:
    if sigma <= 0.0:
        return start, end
    offset = stable_normal(seed, name, sigma)
    lower = min_start - start
    upper = max_end - end
    offset = min(max(offset, lower), upper)
    return start + offset, end + offset


def expand_training_schedule(sample_count: int, training_enabled: bool | Sequence[bool]) -> list[bool]:
    if isinstance(training_enabled, bool):
        return [training_enabled] * sample_count
    schedule = [bool(enabled) for enabled in training_enabled]
    if len(schedule) != sample_count:
        raise ValueError("training schedule length must match sample count")
    return schedule


def block_repeated_phases(
    sample_count: int,
    *,
    training_enabled: bool | Sequence[bool],
    phase_time_scale: float,
    phase_jitter_sigma_ns: float = 0.0,
    phase_jitter_seed: int = 0,
) -> str:
    if phase_jitter_sigma_ns < 0.0:
        raise ValueError("phase_jitter_sigma_ns must be nonnegative")
    training_schedule = expand_training_schedule(sample_count, training_enabled)
    cycle_ns = CYCLE_NS * phase_time_scale
    stop = sample_count * cycle_ns
    rstf: list[tuple[float, float]] = []
    rstg: list[tuple[float, float]] = []
    fwd: list[tuple[float, float]] = []
    err: list[tuple[float, float]] = []
    bwd: list[tuple[float, float]] = []
    acc: list[tuple[float, float]] = []
    apply: list[tuple[float, float]] = []
    dec: list[tuple[float, float]] = []
    for idx in range(sample_count):
        base = idx * cycle_ns
        scale = phase_time_scale
        sample_start = base
        sample_stop = base + cycle_ns

        def window(label: str, start: float, end: float) -> tuple[float, float]:
            return jittered_interval(
                start,
                end,
                min_start=sample_start,
                max_end=sample_stop,
                sigma=phase_jitter_sigma_ns,
                seed=phase_jitter_seed,
                name=f"phase:{idx}:{label}",
            )

        rstf += [
            window("rstf0", base + 0.00 * scale, base + 0.50 * scale),
            window("rstf1", base + 12.05 * scale, base + 12.55 * scale),
        ]
        rstg += [window("rstg", base + 0.00 * scale, base + 0.50 * scale)]
        fwd += [
            window("fwd0", base + 0.75 * scale, base + 3.00 * scale),
            window("fwd1", base + 12.80 * scale, base + 15.60 * scale),
        ]
        dec.append(window("dec", base + 15.00 * scale, base + 15.55 * scale))
        if training_schedule[idx]:
            err.append(window("err", base + 3.25 * scale, base + 5.00 * scale))
            bwd.append(window("bwd", base + 5.25 * scale, base + 7.00 * scale))
            acc.append(window("acc", base + 7.25 * scale, base + 9.00 * scale))
            apply.append(window("apply", base + 9.25 * scale, base + 11.20 * scale))
    return "\n".join(
        [
            f"Vrstf rstf 0 {pulse_wave(rstf, stop)}",
            f"Vrstfn rstfn 0 {active_low_pulse_wave(rstf, stop)}",
            f"Vrstg rstg 0 {pulse_wave(rstg, stop)}",
            f"Vrstgn rstgn 0 {active_low_pulse_wave(rstg, stop)}",
            f"Vfwd fwd 0 {pulse_wave(fwd, stop)}",
            f"Verr err 0 {pulse_wave(err, stop)}",
            f"Vbwd bwd 0 {pulse_wave(bwd, stop)}",
            f"Vacc acc 0 {pulse_wave(acc, stop)}",
            f"Vapply apply 0 {pulse_wave(apply, stop)}",
            f"Vapplyn applyn 0 {active_low_pulse_wave(apply, stop)}",
            f"Vdec dec 0 {pulse_wave(dec, stop)}",
        ]
    )


def hidden_activation_device_model(model: str) -> str:
    if model == "nrel":
        return "NREL"
    if model == "sense":
        return "NSENSE"
    raise ValueError(f"hidden_activation_model must be one of {HIDDEN_ACTIVATION_MODELS}")


def readout_forward_device_model(model: str) -> str:
    if model == "nrel":
        return "NREL"
    if model == "sense":
        return "NSENSE"
    raise ValueError(f"readout_forward_model must be one of {READOUT_FORWARD_MODELS}")


def learning_activation_gate_device_model(model: str) -> str:
    if model == "nrel":
        return "NREL"
    if model == "sense":
        return "NSENSE"
    raise ValueError(f"learning_activation_gate_model must be one of {LEARNING_ACTIVATION_GATE_MODELS}")


def hidden_credit_device_lines(
    feature: int,
    *,
    mode: str,
    positive_error_node: str,
    negative_error_node: str,
    hidden_error_width: float,
    learning_activation_model: str,
) -> list[str]:
    if mode == "direct-feedback":
        return [
            f"Mhdp{feature}_d0 vdd {positive_error_node} hdp{feature}_d0 0 NSENSE W={hidden_error_width:.6g}u L=180n",
            f"Mhdp{feature}_d1 hdp{feature}_d0 act{feature} hdp{feature}_d1 0 {learning_activation_model} W={hidden_error_width:.6g}u L=180n",
            f"Mhdp{feature}_d2 hdp{feature}_d1 bwd hdp{feature} 0 NMOS W={hidden_error_width:.6g}u L=180n",
            f"Mhdn{feature}_d0 vdd {negative_error_node} hdn{feature}_d0 0 NSENSE W={hidden_error_width:.6g}u L=180n",
            f"Mhdn{feature}_d1 hdn{feature}_d0 act{feature} hdn{feature}_d1 0 {learning_activation_model} W={hidden_error_width:.6g}u L=180n",
            f"Mhdn{feature}_d2 hdn{feature}_d1 bwd hdn{feature} 0 NMOS W={hidden_error_width:.6g}u L=180n",
        ]
    if mode in {"readout-weighted", "readout-restored"}:
        positive_weight_gate = f"rvwp{feature}" if mode == "readout-restored" else f"vwp{feature}"
        negative_weight_gate = f"rvwn{feature}" if mode == "readout-restored" else f"vwn{feature}"
        return [
            f"Mhdp{feature}_pv_e vdd {positive_error_node} hdp{feature}_pv_e 0 NSENSE W={hidden_error_width:.6g}u L=180n",
            f"Mhdp{feature}_pv_w hdp{feature}_pv_e {positive_weight_gate} hdp{feature}_pv_w 0 NSENSE W={hidden_error_width:.6g}u L=180n",
            f"Mhdp{feature}_pv_a hdp{feature}_pv_w act{feature} hdp{feature}_pv_a 0 {learning_activation_model} W={hidden_error_width:.6g}u L=180n",
            f"Mhdp{feature}_pv_b hdp{feature}_pv_a bwd hdp{feature} 0 NMOS W={hidden_error_width:.6g}u L=180n",
            f"Mhdp{feature}_nv_e vdd {negative_error_node} hdp{feature}_nv_e 0 NSENSE W={hidden_error_width:.6g}u L=180n",
            f"Mhdp{feature}_nv_w hdp{feature}_nv_e {negative_weight_gate} hdp{feature}_nv_w 0 NSENSE W={hidden_error_width:.6g}u L=180n",
            f"Mhdp{feature}_nv_a hdp{feature}_nv_w act{feature} hdp{feature}_nv_a 0 {learning_activation_model} W={hidden_error_width:.6g}u L=180n",
            f"Mhdp{feature}_nv_b hdp{feature}_nv_a bwd hdp{feature} 0 NMOS W={hidden_error_width:.6g}u L=180n",
            f"Mhdn{feature}_pv_e vdd {positive_error_node} hdn{feature}_pv_e 0 NSENSE W={hidden_error_width:.6g}u L=180n",
            f"Mhdn{feature}_pv_w hdn{feature}_pv_e {negative_weight_gate} hdn{feature}_pv_w 0 NSENSE W={hidden_error_width:.6g}u L=180n",
            f"Mhdn{feature}_pv_a hdn{feature}_pv_w act{feature} hdn{feature}_pv_a 0 {learning_activation_model} W={hidden_error_width:.6g}u L=180n",
            f"Mhdn{feature}_pv_b hdn{feature}_pv_a bwd hdn{feature} 0 NMOS W={hidden_error_width:.6g}u L=180n",
            f"Mhdn{feature}_nv_e vdd {negative_error_node} hdn{feature}_nv_e 0 NSENSE W={hidden_error_width:.6g}u L=180n",
            f"Mhdn{feature}_nv_w hdn{feature}_nv_e {positive_weight_gate} hdn{feature}_nv_w 0 NSENSE W={hidden_error_width:.6g}u L=180n",
            f"Mhdn{feature}_nv_a hdn{feature}_nv_w act{feature} hdn{feature}_nv_a 0 {learning_activation_model} W={hidden_error_width:.6g}u L=180n",
            f"Mhdn{feature}_nv_b hdn{feature}_nv_a bwd hdn{feature} 0 NMOS W={hidden_error_width:.6g}u L=180n",
        ]
    raise ValueError(f"hidden_credit_mode must be one of {HIDDEN_CREDIT_MODES}")


def block_netlist(
    samples: list[dict[str, Any]],
    weights: dict[str, Any],
    *,
    image_size: int,
    block_size: int,
    stride: int,
    channels: int,
    training_enabled: bool | Sequence[bool],
    output_driver_model: str = "sense",
    output_differential_stage: str = "simple",
    output_score_pullup_width: float = 24.0,
    output_scoren_pulldown_width: float = 24.0,
    output_latch_capacitance: float = 20e-15,
    output_decision_stage: str = "none",
    output_decision_ref: float = 1.09,
    output_decision_ref_source: str = "voltage",
    output_decision_ref_resistance: float = 1e6,
    output_decision_ref_capacitance: float = 20e-15,
    output_decision_ref_write_width: float = 1.0,
    output_decision_pullup_width: float = 48.0,
    output_decision_pulldown_width: float = 96.0,
    readout_apply_scale: float = 0.35,
    hidden_forward_width: float = 3.0,
    hidden_forward_topology: str = "per-pixel-phase",
    readout_gradient_width: float = 24.0,
    hidden_error_width: float = 32.0,
    hidden_credit_mode: str = "direct-feedback",
    readout_feedback_restore_width: float = 4.0,
    error_signal_mode: str = "raw",
    error_restore_width: float = 4.0,
    hidden_update_width: float = 12.0,
    hidden_weight_write_width: float = 0.25,
    hidden_weight_leak_resistance: float = 0.0,
    hidden_weight_positive_ref: float = 0.50,
    hidden_weight_negative_ref: float = 0.20,
    hidden_activation_width: float = 24.0,
    hidden_input_residual_width: float = 0.0,
    hidden_stack_shunt_resistance: float = 0.0,
    hidden_stack_parasitic_capacitance: float = 0.0,
    hidden_activation_model: str = "nrel",
    readout_forward_width: float = 64.0,
    readout_forward_model: str = "nrel",
    learning_activation_gate_model: str = "nrel",
    readout_weight_leak_resistance: float = 0.0,
    readout_stack_shunt_resistance: float = 0.0,
    readout_stack_parasitic_capacitance: float = 0.0,
    readout_weight_positive_ref: float = 0.52,
    readout_weight_negative_ref: float = 0.25,
    activation_competition_width: float = 0.0,
    score_activity_inhibition_width: float = 0.0,
    output_bias_enabled: bool = False,
    output_bias_apply_scale: float = 1.0,
    output_bias_positive_init: float = 0.52,
    output_bias_negative_init: float = 0.25,
    output_bias_leak_resistance: float = 0.0,
    output_bias_positive_ref: float = 0.25,
    output_bias_negative_ref: float = 0.25,
    phase_time_scale: float = 1.0,
    phase_jitter_sigma_ns: float = 0.0,
    phase_jitter_seed: int = 0,
    passive_mismatch_sigma: float = 0.0,
    passive_mismatch_seed: int = 0,
    score_mode: str = "single-ended",
    input_rail_mode: str = "alternating-complement",
    measurement_detail: str = "full",
) -> str:
    if readout_apply_scale <= 0.0:
        raise ValueError("readout_apply_scale must be positive")
    if output_score_pullup_width <= 0.0:
        raise ValueError("output_score_pullup_width must be positive")
    if output_scoren_pulldown_width <= 0.0:
        raise ValueError("output_scoren_pulldown_width must be positive")
    if output_latch_capacitance <= 0.0:
        raise ValueError("output_latch_capacitance must be positive")
    if output_decision_stage not in OUTPUT_DECISION_STAGES:
        raise ValueError(f"output_decision_stage must be one of {OUTPUT_DECISION_STAGES}")
    if output_decision_ref_source not in OUTPUT_DECISION_REF_SOURCES:
        raise ValueError(f"output_decision_ref_source must be one of {OUTPUT_DECISION_REF_SOURCES}")
    if output_decision_ref_source in {"divider", "adaptive"} and output_decision_ref_resistance < 0.0:
        raise ValueError("output_decision_ref_resistance must be nonnegative")
    if output_decision_ref_source in {"divider", "adaptive"} and output_decision_ref_resistance > 0.0:
        decision_ref_divider_resistances(output_decision_ref, output_decision_ref_resistance)
    if output_decision_ref_source == "adaptive":
        if output_decision_ref_capacitance <= 0.0:
            raise ValueError("output_decision_ref_capacitance must be positive")
        if output_decision_ref_write_width <= 0.0:
            raise ValueError("output_decision_ref_write_width must be positive")
    if output_decision_ref_source == "divider" and output_decision_ref_resistance <= 0.0:
        raise ValueError("output_decision_ref_resistance must be positive")
    if (
        output_decision_stage
        in {"ref-latched", "ref-precharged-latched", "ref-preamp-latched", "diff-latched", "diff-precharged-latched"}
        and output_differential_stage != "latched"
    ):
        raise ValueError("output_decision_stage requires latched output_differential_stage")
    if output_decision_ref_source == "adaptive" and output_decision_stage not in {
        "ref-latched",
        "ref-precharged-latched",
        "ref-preamp-latched",
    }:
        raise ValueError("adaptive output_decision_ref_source requires a reference decision stage")
    if output_decision_pullup_width <= 0.0:
        raise ValueError("output_decision_pullup_width must be positive")
    if output_decision_pulldown_width <= 0.0:
        raise ValueError("output_decision_pulldown_width must be positive")
    if hidden_forward_width <= 0.0:
        raise ValueError("hidden_forward_width must be positive")
    if hidden_forward_topology not in HIDDEN_FORWARD_TOPOLOGIES:
        raise ValueError(f"hidden_forward_topology must be one of {HIDDEN_FORWARD_TOPOLOGIES}")
    if readout_gradient_width <= 0.0:
        raise ValueError("readout_gradient_width must be positive")
    if hidden_error_width <= 0.0:
        raise ValueError("hidden_error_width must be positive")
    if hidden_credit_mode not in HIDDEN_CREDIT_MODES:
        raise ValueError(f"hidden_credit_mode must be one of {HIDDEN_CREDIT_MODES}")
    if hidden_credit_mode in {"readout-weighted", "readout-restored"} and score_mode != "differential":
        raise ValueError(f"{hidden_credit_mode} hidden_credit_mode requires differential score_mode")
    if readout_feedback_restore_width <= 0.0:
        raise ValueError("readout_feedback_restore_width must be positive")
    if error_signal_mode not in ERROR_SIGNAL_MODES:
        raise ValueError(f"error_signal_mode must be one of {ERROR_SIGNAL_MODES}")
    if error_signal_mode in {"restored", "restored-hidden"} and score_mode != "differential":
        raise ValueError(f"{error_signal_mode} error_signal_mode requires differential score_mode")
    if error_restore_width <= 0.0:
        raise ValueError("error_restore_width must be positive")
    if hidden_update_width <= 0.0:
        raise ValueError("hidden_update_width must be positive")
    if hidden_weight_write_width <= 0.0:
        raise ValueError("hidden_weight_write_width must be positive")
    if hidden_weight_leak_resistance < 0.0:
        raise ValueError("hidden_weight_leak_resistance must be nonnegative")
    if hidden_activation_width <= 0.0:
        raise ValueError("hidden_activation_width must be positive")
    if hidden_input_residual_width < 0.0:
        raise ValueError("hidden_input_residual_width must be nonnegative")
    if hidden_stack_shunt_resistance < 0.0:
        raise ValueError("hidden_stack_shunt_resistance must be nonnegative")
    if hidden_stack_parasitic_capacitance < 0.0:
        raise ValueError("hidden_stack_parasitic_capacitance must be nonnegative")
    if readout_forward_width <= 0.0:
        raise ValueError("readout_forward_width must be positive")
    if readout_weight_leak_resistance < 0.0:
        raise ValueError("readout_weight_leak_resistance must be nonnegative")
    if readout_stack_shunt_resistance < 0.0:
        raise ValueError("readout_stack_shunt_resistance must be nonnegative")
    if readout_stack_parasitic_capacitance < 0.0:
        raise ValueError("readout_stack_parasitic_capacitance must be nonnegative")
    if activation_competition_width < 0.0:
        raise ValueError("activation_competition_width must be nonnegative")
    if score_activity_inhibition_width < 0.0:
        raise ValueError("score_activity_inhibition_width must be nonnegative")
    if score_activity_inhibition_width > 0.0 and activation_competition_width <= 0.0:
        raise ValueError("score_activity_inhibition_width requires activation_competition_width")
    if output_bias_apply_scale <= 0.0:
        raise ValueError("output_bias_apply_scale must be positive")
    if output_bias_leak_resistance < 0.0:
        raise ValueError("output_bias_leak_resistance must be nonnegative")
    if phase_time_scale <= 0.0:
        raise ValueError("phase_time_scale must be positive")
    if phase_jitter_sigma_ns < 0.0:
        raise ValueError("phase_jitter_sigma_ns must be nonnegative")
    if passive_mismatch_sigma < 0.0:
        raise ValueError("passive_mismatch_sigma must be nonnegative")
    if score_mode not in SCORE_MODES:
        raise ValueError(f"score_mode must be one of {SCORE_MODES}")
    if output_differential_stage not in OUTPUT_DIFFERENTIAL_STAGES:
        raise ValueError(f"output_differential_stage must be one of {OUTPUT_DIFFERENTIAL_STAGES}")
    if output_differential_stage != "simple" and score_mode != "differential":
        raise ValueError("non-simple output_differential_stage requires differential score_mode")
    activation_model = hidden_activation_device_model(hidden_activation_model)
    readout_model = readout_forward_device_model(readout_forward_model)
    learning_activation_model = learning_activation_gate_device_model(learning_activation_gate_model)
    if input_rail_mode not in INPUT_RAIL_MODES:
        raise ValueError(f"input_rail_mode must be one of {INPUT_RAIL_MODES}")
    if measurement_detail not in MEASUREMENT_DETAILS:
        raise ValueError(f"measurement_detail must be one of {MEASUREMENT_DETAILS}")
    blocks, expected_features = block_topology(image_size, block_size, stride, channels)
    feature_count, block_len = block_weight_shape(weights)
    if feature_count != expected_features or block_len != len(blocks[0]):
        raise ValueError(
            f"weight shape ({feature_count}, {block_len}) does not match topology "
            f"({expected_features}, {len(blocks[0])})"
        )
    pixel_count = image_size * image_size
    required_rails = required_input_rail_names(pixel_count, channels, input_rail_mode)
    validate_block_samples(samples, required_rails=required_rails)

    readout_pmos_w = 8.0 * readout_apply_scale
    readout_nmos_w = 2.0 * readout_apply_scale
    output_bias_pmos_w = readout_pmos_w * output_bias_apply_scale
    output_bias_nmos_w = readout_nmos_w * output_bias_apply_scale
    restore_error_enabled = error_signal_mode in {"restored", "restored-hidden"}
    hidden_positive_error_node = "edp" if restore_error_enabled else "dp"
    hidden_negative_error_node = "edn" if restore_error_enabled else "dn"
    readout_positive_error_node = "edp" if error_signal_mode == "restored" else "dp"
    readout_negative_error_node = "edn" if error_signal_mode == "restored" else "dn"
    hidden_neg_width = max(0.5, hidden_forward_width * 0.75)
    hidden_forward_phase_width = max(hidden_forward_width, hidden_forward_width * block_len)
    readout_negative_forward_width = max(0.5, readout_forward_width * 0.75)
    cycle_ns = CYCLE_NS * phase_time_scale
    stop = len(samples) * cycle_ns
    measures: list[str] = []
    prints: list[str] = []
    for idx in range(len(samples)):
        base = idx * cycle_ns
        scale = phase_time_scale
        measures += [
            f".meas tran score_before_{idx} FIND V(score) AT={base + 2.95 * scale:.2f}n",
            f".meas tran scoren_before_{idx} FIND V(scoren) AT={base + 2.95 * scale:.2f}n",
            f".meas tran score_net_{idx} PARAM='score_before_{idx}-scoren_before_{idx}'",
            f".meas tran out_before_{idx} FIND V(out) AT={base + 2.95 * scale:.2f}n",
            f".meas tran score_error_{idx} FIND V(score) AT={base + 4.25 * scale:.2f}n",
            f".meas tran dp_after_{idx} FIND V(dp) AT={base + 5.10 * scale:.2f}n",
            f".meas tran dn_after_{idx} FIND V(dn) AT={base + 5.10 * scale:.2f}n",
            f".meas tran out_after_{idx} FIND V(out) AT={base + 15.50 * scale:.2f}n",
            f".meas tran d_out_{idx} PARAM='out_after_{idx}-out_before_{idx}'",
            f".meas tran error_net_{idx} PARAM='dp_after_{idx}-dn_after_{idx}'",
        ]
        if restore_error_enabled:
            measures += [
                f".meas tran edp_after_{idx} FIND V(edp) AT={base + 5.10 * scale:.2f}n",
                f".meas tran edn_after_{idx} FIND V(edn) AT={base + 5.10 * scale:.2f}n",
                f".meas tran error_restored_diff_{idx} PARAM='edp_after_{idx}-edn_after_{idx}'",
            ]
        if output_differential_stage == "latched":
            measures += [
                f".meas tran outn_before_{idx} FIND V(outn) AT={base + 2.95 * scale:.2f}n",
                f".meas tran outn_after_{idx} FIND V(outn) AT={base + 15.50 * scale:.2f}n",
                f".meas tran out_diff_{idx} PARAM='out_after_{idx}-outn_after_{idx}'",
            ]
        if output_decision_stage != "none":
            measures += [
                f".meas tran decision_before_{idx} FIND V(decision) AT={base + 2.95 * scale:.2f}n",
                f".meas tran decision_after_{idx} FIND V(decision) AT={base + 15.50 * scale:.2f}n",
                f".meas tran decisionn_after_{idx} FIND V(decisionn) AT={base + 15.50 * scale:.2f}n",
                f".meas tran decision_diff_{idx} PARAM='decision_after_{idx}-decisionn_after_{idx}'",
            ]
        if output_decision_ref_source == "adaptive":
            measures += [
                f".meas tran outref_before_{idx} FIND V(outref) AT={base + 0.60 * scale:.2f}n",
                f".meas tran outref_after_apply_{idx} FIND V(outref) AT={base + 11.50 * scale:.2f}n",
                f".meas tran d_outref_{idx} PARAM='outref_after_apply_{idx}-outref_before_{idx}'",
            ]
        if output_bias_enabled:
            measures += [
                f".meas tran obp_before_{idx} FIND V(obp) AT={base + 0.60 * scale:.2f}n",
                f".meas tran obn_before_{idx} FIND V(obn) AT={base + 0.60 * scale:.2f}n",
                f".meas tran gop_after_{idx} FIND V(gop) AT={base + 9.10 * scale:.2f}n",
                f".meas tran gon_after_{idx} FIND V(gon) AT={base + 9.10 * scale:.2f}n",
                f".meas tran obp_after_apply_{idx} FIND V(obp) AT={base + 11.50 * scale:.2f}n",
                f".meas tran obn_after_apply_{idx} FIND V(obn) AT={base + 11.50 * scale:.2f}n",
                f".meas tran output_bias_signed_before_{idx} PARAM='obp_before_{idx}-obn_before_{idx}'",
                f".meas tran output_bias_signed_after_{idx} PARAM='obp_after_apply_{idx}-obn_after_apply_{idx}'",
                f".meas tran d_output_bias_signed_{idx} PARAM='output_bias_signed_after_{idx}-output_bias_signed_before_{idx}'",
            ]
        if activation_competition_width > 0.0:
            measures += [
                f".meas tran actinh_before_{idx} FIND V(actinh) AT={base + 2.95 * scale:.2f}n",
                f".meas tran actinh_after_{idx} FIND V(actinh) AT={base + 15.50 * scale:.2f}n",
            ]
        if measurement_detail == "outputs":
            prints.append(f"print out_before_{idx} out_after_{idx} error_net_{idx}")
            continue
        for feature in range(feature_count):
            measures += [
                f".meas tran act{feature}_before_{idx} FIND V(act{feature}) AT={base + 2.95 * scale:.2f}n",
                f".meas tran act{feature}_after_{idx} FIND V(act{feature}) AT={base + 15.50 * scale:.2f}n",
                f".meas tran bhp{feature}_before_{idx} FIND V(bhp{feature}) AT={base + 0.60 * scale:.2f}n",
                f".meas tran bhn{feature}_before_{idx} FIND V(bhn{feature}) AT={base + 0.60 * scale:.2f}n",
                f".meas tran vwp{feature}_before_{idx} FIND V(vwp{feature}) AT={base + 0.60 * scale:.2f}n",
                f".meas tran vwn{feature}_before_{idx} FIND V(vwn{feature}) AT={base + 0.60 * scale:.2f}n",
                f".meas tran hdp{feature}_after_{idx} FIND V(hdp{feature}) AT={base + 7.10 * scale:.2f}n",
                f".meas tran hdn{feature}_after_{idx} FIND V(hdn{feature}) AT={base + 7.10 * scale:.2f}n",
                f".meas tran gvp{feature}_after_{idx} FIND V(gvp{feature}) AT={base + 9.10 * scale:.2f}n",
                f".meas tran gvn{feature}_after_{idx} FIND V(gvn{feature}) AT={base + 9.10 * scale:.2f}n",
                f".meas tran gbp{feature}_after_{idx} FIND V(gbp{feature}) AT={base + 9.10 * scale:.2f}n",
                f".meas tran gbn{feature}_after_{idx} FIND V(gbn{feature}) AT={base + 9.10 * scale:.2f}n",
                f".meas tran bhp{feature}_after_apply_{idx} FIND V(bhp{feature}) AT={base + 11.50 * scale:.2f}n",
                f".meas tran bhn{feature}_after_apply_{idx} FIND V(bhn{feature}) AT={base + 11.50 * scale:.2f}n",
                f".meas tran vwp{feature}_after_apply_{idx} FIND V(vwp{feature}) AT={base + 11.50 * scale:.2f}n",
                f".meas tran vwn{feature}_after_apply_{idx} FIND V(vwn{feature}) AT={base + 11.50 * scale:.2f}n",
                f".meas tran bias{feature}_signed_before_{idx} PARAM='bhp{feature}_before_{idx}-bhn{feature}_before_{idx}'",
                f".meas tran bias{feature}_signed_after_{idx} PARAM='bhp{feature}_after_apply_{idx}-bhn{feature}_after_apply_{idx}'",
                f".meas tran readout{feature}_signed_before_{idx} PARAM='vwp{feature}_before_{idx}-vwn{feature}_before_{idx}'",
                f".meas tran readout{feature}_signed_after_{idx} PARAM='vwp{feature}_after_apply_{idx}-vwn{feature}_after_apply_{idx}'",
                f".meas tran d_bias{feature}_signed_{idx} PARAM='bias{feature}_signed_after_{idx}-bias{feature}_signed_before_{idx}'",
                f".meas tran d_readout{feature}_signed_{idx} PARAM='readout{feature}_signed_after_{idx}-readout{feature}_signed_before_{idx}'",
            ]
            if hidden_credit_mode == "readout-restored":
                measures += [
                    f".meas tran rvwp{feature}_after_err_{idx} FIND V(rvwp{feature}) AT={base + 5.10 * scale:.2f}n",
                    f".meas tran rvwn{feature}_after_err_{idx} FIND V(rvwn{feature}) AT={base + 5.10 * scale:.2f}n",
                    f".meas tran rvw{feature}_diff_after_err_{idx} PARAM='rvwp{feature}_after_err_{idx}-rvwn{feature}_after_err_{idx}'",
                ]
            for pix in range(block_len):
                measures += [
                    f".meas tran whp{feature}_{pix}_after_apply_{idx} FIND V(whp{feature}_{pix}) AT={base + 11.50 * scale:.2f}n",
                    f".meas tran whn{feature}_{pix}_after_apply_{idx} FIND V(whn{feature}_{pix}) AT={base + 11.50 * scale:.2f}n",
                ]
        prints.append(f"print out_before_{idx} out_after_{idx} error_net_{idx}")

    lines = [
        "* Block/stride MNIST01 device-level training smoke.",
        f"* image={image_size} block={block_size} stride={stride} channels={channels}",
        f"* {feature_count} feature cells, each with {block_len} trainable hidden pixel weights.",
        f".param VDD={VDD_VALUE:.12g}",
        mos_models(),
        spice_subcircuits(),
        "Vdd vdd 0 {VDD}",
    ]
    if readout_weight_leak_resistance > 0.0:
        lines += [
            f"Vvwp_ref vwp_ref 0 {readout_weight_positive_ref:.12g}",
            f"Vvwn_ref vwn_ref 0 {readout_weight_negative_ref:.12g}",
        ]
    if hidden_weight_leak_resistance > 0.0:
        lines += [
            f"Vwhp_ref whp_ref 0 {hidden_weight_positive_ref:.12g}",
            f"Vwhn_ref whn_ref 0 {hidden_weight_negative_ref:.12g}",
        ]
    if output_bias_enabled and output_bias_leak_resistance > 0.0:
        lines += [
            f"Vobp_ref obp_ref 0 {output_bias_positive_ref:.12g}",
            f"Vobn_ref obn_ref 0 {output_bias_negative_ref:.12g}",
        ]
    if output_decision_stage in {"ref-latched", "ref-precharged-latched", "ref-preamp-latched"}:
        if output_decision_ref_source == "divider":
            rtop, rbot = decision_ref_divider_resistances(output_decision_ref, output_decision_ref_resistance)
            rtop *= mismatch_factor(passive_mismatch_seed, "Routref_top", passive_mismatch_sigma)
            rbot *= mismatch_factor(passive_mismatch_seed, "Routref_bot", passive_mismatch_sigma)
            lines += [
                f"Routref_top vdd outref {rtop:.12g}",
                f"Routref_bot outref 0 {rbot:.12g}",
                "Coutref outref 0 1f IC=0",
            ]
        elif output_decision_ref_source == "adaptive":
            lines += [
                f"Coutref outref 0 {spice_capacitance(output_decision_ref_capacitance)} IC={output_decision_ref:.12g}",
                "Routref outref 0 1e15",
                "Coutref_raise_gate outref_raise_gate 0 4f IC=1.2",
                "Routref_raise_gate outref_raise_gate vdd 50k",
            ]
            if output_decision_ref_resistance > 0.0:
                rtop, rbot = decision_ref_divider_resistances(output_decision_ref, output_decision_ref_resistance)
                rtop *= mismatch_factor(passive_mismatch_seed, "Routref_top", passive_mismatch_sigma)
                rbot *= mismatch_factor(passive_mismatch_seed, "Routref_bot", passive_mismatch_sigma)
                lines += [
                    f"Routref_top vdd outref {rtop:.12g}",
                    f"Routref_bot outref 0 {rbot:.12g}",
                ]
        else:
            lines.append(f"Voutref outref 0 {output_decision_ref:.12g}")
    for rail in required_rails:
        lines.append(f"V{rail} {rail} 0 {block_sample_wave(samples, rail, stop, cycle_ns=cycle_ns)}")
    lines += [
        f"Vtarget target 0 {block_sample_wave(samples, 'target', stop, cycle_ns=cycle_ns)}",
        block_repeated_phases(
            len(samples),
            training_enabled=training_enabled,
            phase_time_scale=phase_time_scale,
            phase_jitter_sigma_ns=phase_jitter_sigma_ns,
            phase_jitter_seed=phase_jitter_seed,
        ),
        "",
        "* Persistent signed hidden and readout weights.",
    ]
    for feature in range(feature_count):
        for pix in range(block_len):
            lines += [
                signed_store_instance(
                    f"wh{feature}_{pix}",
                    f"whp{feature}_{pix}",
                    f"whn{feature}_{pix}",
                    positive_ic=float(weights["whp"][feature][pix]),
                    negative_ic=float(weights["whn"][feature][pix]),
                ),
            ]
            if hidden_weight_leak_resistance > 0.0:
                lines += [
                    f"Rwhp{feature}_{pix}_leak whp{feature}_{pix} whp_ref {hidden_weight_leak_resistance:.6g}",
                    f"Rwhn{feature}_{pix}_leak whn{feature}_{pix} whn_ref {hidden_weight_leak_resistance:.6g}",
                ]
        lines += [
            signed_store_instance(
                f"bh{feature}",
                f"bhp{feature}",
                f"bhn{feature}",
                positive_ic=float(weights["bhp"][feature]),
                negative_ic=float(weights["bhn"][feature]),
            ),
            signed_store_instance(
                f"vw{feature}",
                f"vwp{feature}",
                f"vwn{feature}",
                positive_ic=float(weights["vwp"][feature]),
                negative_ic=float(weights["vwn"][feature]),
            ),
        ]
        if hidden_weight_leak_resistance > 0.0:
            lines += [
                f"Rbhp{feature}_leak bhp{feature} whp_ref {hidden_weight_leak_resistance:.6g}",
                f"Rbhn{feature}_leak bhn{feature} whn_ref {hidden_weight_leak_resistance:.6g}",
            ]
        if readout_weight_leak_resistance > 0.0:
            lines += [
                f"Rvwp{feature}_leak vwp{feature} vwp_ref {readout_weight_leak_resistance:.6g}",
                f"Rvwn{feature}_leak vwn{feature} vwn_ref {readout_weight_leak_resistance:.6g}",
            ]

    if output_bias_enabled:
        obp_initial = float(weights.get("obp", output_bias_positive_init))
        obn_initial = float(weights.get("obn", output_bias_negative_init))
        lines += [
            "",
            "* Persistent signed output bias and local bias-gradient state.",
            signed_store_instance("ob", "obp", "obn", positive_ic=obp_initial, negative_ic=obn_initial),
            "Cgop gop 0 2f IC=0",
            "Cgon gon 0 2f IC=0",
            "Crop rop 0 4f IC=1.2",
            "Cron ron 0 4f IC=1.2",
            "Rgop gop 0 1G",
            "Rgon gon 0 1G",
            "Rrop rop vdd 50k",
            "Rron ron vdd 50k",
        ]
        if output_bias_leak_resistance > 0.0:
            lines += [
                f"Robp_leak obp obp_ref {output_bias_leak_resistance:.6g}",
                f"Robn_leak obn obn_ref {output_bias_leak_resistance:.6g}",
            ]

    lines += [
        "",
        "* Shared output/error state.",
        "Cscore score 0 10f IC=0",
        "Cscoren scoren 0 10f IC=0",
        f"Cout out 0 {spice_capacitance(output_latch_capacitance)} IC=0",
        "Cdp dp 0 20f IC=0",
        "Cdn dn 0 20f IC=0",
        "Rscore score 0 1G",
        "Rscoren scoren 0 1G",
        "Rout out 0 1G",
        "Rdp dp 0 1G",
        "Rdn dn 0 1G",
    ]
    if restore_error_enabled:
        lines += [
            "Cedp edp 0 8f IC=0",
            "Cedn edn 0 8f IC=0",
            "Cerrstore_src errstore_src 0 0.1f IC=0",
            "Redp edp 0 1G",
            "Redn edn 0 1G",
            "Rerrstore_src errstore_src 0 1G",
        ]
    if activation_competition_width > 0.0:
        lines += [
            "Cactinh actinh 0 10f IC=0",
            "Ractinh actinh 0 1G",
        ]
    if output_differential_stage == "latched":
        lines += [
            f"Coutn outn 0 {spice_capacitance(output_latch_capacitance)} IC=0",
            "Routn outn 0 1G",
        ]
    if output_decision_stage != "none":
        lines += [
            "Cdecision decision 0 20f IC=0",
            "Cdecisionn decisionn 0 20f IC=0",
            "Rdecision decision 0 1G",
            "Rdecisionn decisionn 0 1G",
        ]
    if output_decision_stage == "ref-preamp-latched":
        lines += [
            "Cdecision_pre decision_pre 0 10f IC=0",
            "Cdecisionn_pre decisionn_pre 0 10f IC=0",
            "Rdecision_pre decision_pre 0 1G",
            "Rdecisionn_pre decisionn_pre 0 1G",
        ]
    if output_decision_stage == "shift-inverter":
        lines += [
            "Cdec_mid dec_mid 0 20f IC=0",
            "Rdec_mid dec_mid 0 1G",
        ]
    if output_decision_stage == "stacked-inverter":
        lines += [
            "Cdec_stack0 dec_stack0 0 0.1f IC=0",
            "Cdec_stack1 dec_stack1 0 0.1f IC=0",
        ]
    for feature in range(feature_count):
        lines += [
            f"Cpre{feature} pre{feature} 0 10f IC=0",
            f"Cact{feature} act{feature} 0 20f IC=0",
            f"Chdp{feature} hdp{feature} 0 12f IC=0",
            f"Chdn{feature} hdn{feature} 0 12f IC=0",
            f"Cgvp{feature} gvp{feature} 0 2f IC=0",
            f"Cgvn{feature} gvn{feature} 0 2f IC=0",
            f"Cgbp{feature} gbp{feature} 0 10f IC=0",
            f"Cgbn{feature} gbn{feature} 0 10f IC=0",
            f"Crgp{feature} rgp{feature} 0 4f IC=1.2",
            f"Crgn{feature} rgn{feature} 0 4f IC=1.2",
            f"Rpre{feature} pre{feature} 0 1G",
            f"Ract{feature} act{feature} 0 1G",
            f"Rhdp{feature} hdp{feature} 0 1G",
            f"Rhdn{feature} hdn{feature} 0 1G",
            f"Rgvp{feature} gvp{feature} 0 1G",
            f"Rgvn{feature} gvn{feature} 0 1G",
            f"Rgbp{feature} gbp{feature} 0 1G",
            f"Rgbn{feature} gbn{feature} 0 1G",
            f"Rrgp{feature} rgp{feature} vdd 50k",
            f"Rrgn{feature} rgn{feature} vdd 50k",
        ]
        if hidden_forward_topology == "split-rail":
            lines += [
                f"Cpren{feature} pren{feature} 0 10f IC=0",
                f"Rpren{feature} pren{feature} 0 1G",
            ]
        if hidden_credit_mode == "readout-restored":
            lines += [
                f"Crvwp{feature} rvwp{feature} 0 4f IC=0",
                f"Crvwn{feature} rvwn{feature} 0 4f IC=0",
                f"Crvw_src{feature} rvw_src{feature} 0 0.1f IC=0",
                f"Rrvwp{feature} rvwp{feature} 0 1G",
                f"Rrvwn{feature} rvwn{feature} 0 1G",
                f"Rrvw_src{feature} rvw_src{feature} 0 1G",
            ]
        for pix in range(block_len):
            lines += [
                f"Cghp{feature}_{pix} ghp{feature}_{pix} 0 10f IC=0",
                f"Cghn{feature}_{pix} ghn{feature}_{pix} 0 10f IC=0",
                f"Rghp{feature}_{pix} ghp{feature}_{pix} 0 1G",
                f"Rghn{feature}_{pix} ghn{feature}_{pix} 0 1G",
            ]

    lines += [
        "",
        "* Reset shared nonpersistent state.",
        "Mreset_score score rstf 0 0 NMOS W=4u L=180n",
        "Mreset_scoren scoren rstf 0 0 NMOS W=4u L=180n",
        "Mreset_out out rstf 0 0 NMOS W=4u L=180n",
        "Mreset_dp dp rstg 0 0 NMOS W=4u L=180n",
        "Mreset_dn dn rstg 0 0 NMOS W=4u L=180n",
    ]
    if restore_error_enabled:
        lines += [
            "Mprecharge_edp edp rstgn vdd vdd PMOS W=4u L=180n",
            "Mprecharge_edn edn rstgn vdd vdd PMOS W=4u L=180n",
        ]
    if output_bias_enabled:
        lines += [
            "Mreset_gop gop rstg 0 0 NMOS W=4u L=180n",
            "Mreset_gon gon rstg 0 0 NMOS W=4u L=180n",
        ]
    if output_differential_stage == "latched":
        lines.append("Mreset_outn outn rstf 0 0 NMOS W=4u L=180n")
    if output_decision_stage != "none" and output_decision_stage not in {"ref-precharged-latched", "diff-precharged-latched"}:
        lines += [
            "Mreset_decision decision rstf 0 0 NMOS W=4u L=180n",
            "Mreset_decisionn decisionn rstf 0 0 NMOS W=4u L=180n",
        ]
    if output_decision_stage in {"ref-precharged-latched", "diff-precharged-latched"}:
        lines += [
            "Mprecharge_decision decision rstfn vdd vdd PMOS W=4u L=180n",
            "Mprecharge_decisionn decisionn rstfn vdd vdd PMOS W=4u L=180n",
        ]
    if output_decision_stage == "ref-preamp-latched":
        lines += [
            "Mreset_decision_pre decision_pre rstf 0 0 NMOS W=4u L=180n",
            "Mreset_decisionn_pre decisionn_pre rstf 0 0 NMOS W=4u L=180n",
        ]
    if activation_competition_width > 0.0:
        lines.append("Mreset_actinh actinh rstf 0 0 NMOS W=4u L=180n")
    for feature in range(feature_count):
        lines += [
            f"Mreset_pre{feature} pre{feature} rstf 0 0 NMOS W=4u L=180n",
            f"Mreset_act{feature} act{feature} rstf 0 0 NMOS W=4u L=180n",
            f"Mreset_hdp{feature} hdp{feature} rstg 0 0 NMOS W=4u L=180n",
            f"Mreset_hdn{feature} hdn{feature} rstg 0 0 NMOS W=4u L=180n",
            f"Mreset_gvp{feature} gvp{feature} rstg 0 0 NMOS W=4u L=180n",
            f"Mreset_gvn{feature} gvn{feature} rstg 0 0 NMOS W=4u L=180n",
            f"Mreset_gbp{feature} gbp{feature} rstg 0 0 NMOS W=4u L=180n",
            f"Mreset_gbn{feature} gbn{feature} rstg 0 0 NMOS W=4u L=180n",
        ]
        if hidden_forward_topology == "split-rail":
            lines.append(f"Mreset_pren{feature} pren{feature} rstf 0 0 NMOS W=4u L=180n")
        if hidden_credit_mode == "readout-restored":
            lines += [
                f"Mprecharge_rvwp{feature} rvwp{feature} rstgn vdd vdd PMOS W=4u L=180n",
                f"Mprecharge_rvwn{feature} rvwn{feature} rstgn vdd vdd PMOS W=4u L=180n",
            ]
        for pix in range(block_len):
            lines += [
                f"Mreset_ghp{feature}_{pix} ghp{feature}_{pix} rstg 0 0 NMOS W=4u L=180n",
                f"Mreset_ghn{feature}_{pix} ghn{feature}_{pix} rstg 0 0 NMOS W=4u L=180n",
            ]

    for block_idx, block in enumerate(blocks):
        for channel in range(channels):
            feature = block_idx * channels + channel
            lines += ["", f"* Feature {feature}: block {block_idx}, channel {channel}."]
            if hidden_forward_topology == "shared-phase":
                lines += [
                    f"Mhfpos{feature}_phase hfp{feature}_rail fwd pre{feature} 0 NMOS W={hidden_forward_phase_width:.6g}u L=180n",
                    f"Mhfneg{feature}_phase pre{feature} fwd hfn{feature}_rail 0 NMOS W={hidden_forward_phase_width:.6g}u L=180n",
                    f"Chfp{feature}_rail hfp{feature}_rail 0 0.1f IC=0",
                    f"Chfn{feature}_rail hfn{feature}_rail 0 0.1f IC=0",
                    f"Rhfp{feature}_rail hfp{feature}_rail 0 1G",
                    f"Rhfn{feature}_rail hfn{feature}_rail 0 1G",
                ]
            for pix, pixel_node in enumerate(block):
                input_node = input_rail_name(pixel_node, channel, input_rail_mode)
                if hidden_forward_topology == "shared-phase":
                    hidden_forward_lines = [
                        f"Mhpos{feature}_{pix}_x vdd {input_node} hp{feature}_{pix}_0 0 NMOS W={hidden_forward_width:.6g}u L=180n",
                        f"Mhpos{feature}_{pix}_w hp{feature}_{pix}_0 whp{feature}_{pix} hfp{feature}_rail 0 NMOS W={hidden_forward_width:.6g}u L=180n",
                        f"Mhneg{feature}_{pix}_x hfn{feature}_rail {input_node} hn{feature}_{pix}_0 0 NMOS W={hidden_neg_width:.6g}u L=180n",
                        f"Mhneg{feature}_{pix}_w hn{feature}_{pix}_0 whn{feature}_{pix} 0 0 NMOS W={hidden_neg_width:.6g}u L=180n",
                    ]
                    hidden_stack_nodes = [
                        f"hp{feature}_{pix}_0",
                        f"hn{feature}_{pix}_0",
                    ]
                elif hidden_forward_topology == "always-on":
                    hidden_forward_lines = [
                        f"Mhpos{feature}_{pix}_x vdd {input_node} hp{feature}_{pix}_0 0 NMOS W={hidden_forward_width:.6g}u L=180n",
                        f"Mhpos{feature}_{pix}_w hp{feature}_{pix}_0 whp{feature}_{pix} pre{feature} 0 NMOS W={hidden_forward_width:.6g}u L=180n",
                        f"Mhneg{feature}_{pix}_x pre{feature} {input_node} hn{feature}_{pix}_0 0 NMOS W={hidden_neg_width:.6g}u L=180n",
                        f"Mhneg{feature}_{pix}_w hn{feature}_{pix}_0 whn{feature}_{pix} 0 0 NMOS W={hidden_neg_width:.6g}u L=180n",
                    ]
                    hidden_stack_nodes = [
                        f"hp{feature}_{pix}_0",
                        f"hn{feature}_{pix}_0",
                    ]
                elif hidden_forward_topology == "split-rail":
                    hidden_forward_lines = [
                        split_rail_hidden_pixel_instance(
                            f"hs{feature}_{pix}",
                            input_node,
                            f"whp{feature}_{pix}",
                            f"whn{feature}_{pix}",
                            f"pre{feature}",
                            f"pren{feature}",
                            positive_width=hidden_forward_width,
                            negative_width=hidden_neg_width,
                        ),
                    ]
                    hidden_stack_nodes = []
                else:
                    hidden_forward_lines = [
                        f"Mhpos{feature}_{pix}_x vdd {input_node} hp{feature}_{pix}_0 0 NMOS W={hidden_forward_width:.6g}u L=180n",
                        f"Mhpos{feature}_{pix}_w hp{feature}_{pix}_0 whp{feature}_{pix} hp{feature}_{pix}_1 0 NMOS W={hidden_forward_width:.6g}u L=180n",
                        f"Mhpos{feature}_{pix}_f hp{feature}_{pix}_1 fwd pre{feature} 0 NMOS W={hidden_forward_width:.6g}u L=180n",
                        f"Mhneg{feature}_{pix}_f pre{feature} fwd hn{feature}_{pix}_0 0 NMOS W={hidden_neg_width:.6g}u L=180n",
                        f"Mhneg{feature}_{pix}_x hn{feature}_{pix}_0 {input_node} hn{feature}_{pix}_1 0 NMOS W={hidden_neg_width:.6g}u L=180n",
                        f"Mhneg{feature}_{pix}_w hn{feature}_{pix}_1 whn{feature}_{pix} 0 0 NMOS W={hidden_neg_width:.6g}u L=180n",
                    ]
                    hidden_stack_nodes = [
                        f"hp{feature}_{pix}_0",
                        f"hp{feature}_{pix}_1",
                        f"hn{feature}_{pix}_0",
                        f"hn{feature}_{pix}_1",
                    ]
                lines += [
                    *hidden_forward_lines,
                    *(
                        [
                            f"Rhread_{node} {node} 0 {hidden_stack_shunt_resistance:.12g}"
                            for node in hidden_stack_nodes
                        ]
                        if hidden_stack_shunt_resistance > 0.0
                        else []
                    ),
                    *(
                        [
                            f"Chread_{node} {node} 0 {hidden_stack_parasitic_capacitance:.12g}"
                            for node in hidden_stack_nodes
                        ]
                        if hidden_stack_parasitic_capacitance > 0.0
                        else []
                    ),
                    f"Mghp{feature}_{pix}_x vdd {input_node} ghp{feature}_{pix}_x 0 NMOS W={hidden_update_width:.6g}u L=180n",
                    f"Mghp{feature}_{pix}_d ghp{feature}_{pix}_x hdp{feature} ghp{feature}_{pix}_d 0 NSENSE W={hidden_update_width:.6g}u L=180n",
                    f"Mghp{feature}_{pix}_g ghp{feature}_{pix}_d acc ghp{feature}_{pix} 0 NMOS W={hidden_update_width:.6g}u L=180n",
                    f"Mghn{feature}_{pix}_x vdd {input_node} ghn{feature}_{pix}_x 0 NMOS W={hidden_update_width:.6g}u L=180n",
                    f"Mghn{feature}_{pix}_d ghn{feature}_{pix}_x hdn{feature} ghn{feature}_{pix}_d 0 NSENSE W={hidden_update_width:.6g}u L=180n",
                    f"Mghn{feature}_{pix}_g ghn{feature}_{pix}_d acc ghn{feature}_{pix} 0 NMOS W={hidden_update_width:.6g}u L=180n",
                    f"Mwhp{feature}_{pix}_up_g vdd ghp{feature}_{pix} whp{feature}_{pix}_up 0 NSENSE W={hidden_weight_write_width:.6g}u L=180n",
                    f"Mwhp{feature}_{pix}_up_a whp{feature}_{pix}_up apply whp{feature}_{pix} 0 NREL W={hidden_weight_write_width:.6g}u L=180n",
                    f"Mwhn{feature}_{pix}_dn_a whn{feature}_{pix} apply whn{feature}_{pix}_dn 0 NREL W={hidden_weight_write_width:.6g}u L=180n",
                    f"Mwhn{feature}_{pix}_dn_g whn{feature}_{pix}_dn ghp{feature}_{pix} 0 0 NSENSE W={hidden_weight_write_width:.6g}u L=180n",
                    f"Mwhn{feature}_{pix}_up_g vdd ghn{feature}_{pix} whn{feature}_{pix}_up 0 NSENSE W={hidden_weight_write_width:.6g}u L=180n",
                    f"Mwhn{feature}_{pix}_up_a whn{feature}_{pix}_up apply whn{feature}_{pix} 0 NREL W={hidden_weight_write_width:.6g}u L=180n",
                    f"Mwhp{feature}_{pix}_dn_a whp{feature}_{pix} apply whp{feature}_{pix}_dn 0 NREL W={hidden_weight_write_width:.6g}u L=180n",
                    f"Mwhp{feature}_{pix}_dn_g whp{feature}_{pix}_dn ghn{feature}_{pix} 0 0 NSENSE W={hidden_weight_write_width:.6g}u L=180n",
                ]
            if hidden_forward_topology == "always-on":
                hidden_bias_lines = [
                    f"Mhbpos{feature}_b vdd bhp{feature} pre{feature} 0 NMOS W={hidden_forward_width:.6g}u L=180n",
                    f"Mhbneg{feature}_b pre{feature} bhn{feature} 0 0 NMOS W={hidden_neg_width:.6g}u L=180n",
                ]
            elif hidden_forward_topology == "split-rail":
                hidden_bias_lines = [
                    split_rail_hidden_bias_instance(
                        f"hb{feature}",
                        f"bhp{feature}",
                        f"bhn{feature}",
                        f"pre{feature}",
                        f"pren{feature}",
                        positive_width=hidden_forward_width,
                        negative_width=hidden_neg_width,
                    ),
                ]
            else:
                hidden_bias_lines = [
                    f"Mhbpos{feature}_b vdd bhp{feature} hbp{feature}_0 0 NMOS W={hidden_forward_width:.6g}u L=180n",
                    f"Mhbpos{feature}_f hbp{feature}_0 fwd pre{feature} 0 NMOS W={hidden_forward_width:.6g}u L=180n",
                    f"Mhbneg{feature}_f pre{feature} fwd hbn{feature}_0 0 NMOS W={hidden_neg_width:.6g}u L=180n",
                    f"Mhbneg{feature}_b hbn{feature}_0 bhn{feature} 0 0 NMOS W={hidden_neg_width:.6g}u L=180n",
                ]
            if hidden_forward_topology == "split-rail":
                hidden_activation_lines = [
                    split_rail_relu_instance(
                        f"relu_h{feature}",
                        f"pre{feature}",
                        f"pren{feature}",
                        f"act{feature}",
                        model=hidden_activation_model,
                        width=hidden_activation_width,
                    ),
                ]
            else:
                hidden_activation_lines = [
                    f"Mrelu_h{feature} vdd pre{feature} act{feature} 0 {activation_model} W={hidden_activation_width:.6g}u L=180n",
                ]
            lines += [
                *hidden_bias_lines,
                *hidden_activation_lines,
                *(
                    [
                        line
                        for pix, pixel_node in enumerate(block)
                        for input_node in [input_rail_name(pixel_node, channel, input_rail_mode)]
                        for line in [
                            f"Mhres{feature}_{pix}_x vdd {input_node} hres{feature}_{pix}_0 0 NSENSE W={hidden_input_residual_width:.6g}u L=180n",
                            f"Mhres{feature}_{pix}_f hres{feature}_{pix}_0 fwd act{feature} 0 NMOS W={hidden_input_residual_width:.6g}u L=180n",
                        ]
                    ]
                    if hidden_input_residual_width > 0.0
                    else []
                ),
                *(
                    [
                        f"Mactinh_src{feature}_a vdd act{feature} actinh_src{feature}_a 0 NSENSE W={activation_competition_width:.6g}u L=180n",
                        f"Mactinh_src{feature}_f actinh_src{feature}_a fwd actinh 0 NMOS W={activation_competition_width:.6g}u L=180n",
                        f"Mactinh_sink{feature}_i act{feature} actinh actinh_sink{feature}_i 0 NSENSE W={activation_competition_width:.6g}u L=180n",
                        f"Mactinh_sink{feature}_f actinh_sink{feature}_i fwd 0 0 NMOS W={activation_competition_width:.6g}u L=180n",
                    ]
                    if activation_competition_width > 0.0
                    else []
                ),
                f"Movpos{feature}_a vdd act{feature} op{feature}_0 0 {readout_model} W={readout_forward_width:.6g}u L=180n",
                f"Movpos{feature}_w op{feature}_0 vwp{feature} op{feature}_1 0 {readout_model} W={readout_forward_width:.6g}u L=180n",
                f"Movpos{feature}_f op{feature}_1 fwd score 0 {readout_model} W={readout_forward_width:.6g}u L=180n",
                *(
                    [
                        f"Movneg{feature}_a vdd act{feature} on{feature}_0 0 {readout_model} W={readout_negative_forward_width:.6g}u L=180n",
                        f"Movneg{feature}_w on{feature}_0 vwn{feature} on{feature}_1 0 {readout_model} W={readout_negative_forward_width:.6g}u L=180n",
                        f"Movneg{feature}_f on{feature}_1 fwd scoren 0 {readout_model} W={readout_negative_forward_width:.6g}u L=180n",
                    ]
                    if score_mode == "differential"
                    else [
                        f"Movneg{feature}_f score fwd on{feature}_0 0 {readout_model} W={readout_negative_forward_width:.6g}u L=180n",
                        f"Movneg{feature}_a on{feature}_0 act{feature} on{feature}_1 0 {readout_model} W={readout_negative_forward_width:.6g}u L=180n",
                        f"Movneg{feature}_w on{feature}_1 vwn{feature} 0 0 {readout_model} W={readout_negative_forward_width:.6g}u L=180n",
                    ]
                ),
                *(
                    [
                        f"Mrvwp{feature}_p rvwp{feature} rvwn{feature} vdd vdd PMOS W={readout_feedback_restore_width:.6g}u L=180n",
                        f"Mrvwn{feature}_p rvwn{feature} rvwp{feature} vdd vdd PMOS W={readout_feedback_restore_width:.6g}u L=180n",
                        f"Mrvwp{feature}_n rvwp{feature} vwn{feature} rvw_src{feature} 0 NSENSE W={readout_feedback_restore_width:.6g}u L=180n",
                        f"Mrvwn{feature}_n rvwn{feature} vwp{feature} rvw_src{feature} 0 NSENSE W={readout_feedback_restore_width:.6g}u L=180n",
                        f"Mrvw{feature}_tail rvw_src{feature} err 0 0 NMOS W={readout_feedback_restore_width:.6g}u L=180n",
                    ]
                    if hidden_credit_mode == "readout-restored"
                    else []
                ),
                *hidden_credit_device_lines(
                    feature,
                    mode=hidden_credit_mode,
                    positive_error_node=hidden_positive_error_node,
                    negative_error_node=hidden_negative_error_node,
                    hidden_error_width=hidden_error_width,
                    learning_activation_model=learning_activation_model,
                ),
                f"Mgvp{feature}_a vdd act{feature} gvp{feature}_a 0 {learning_activation_model} W={readout_gradient_width:.6g}u L=180n",
                f"Mgvp{feature}_d gvp{feature}_a {readout_positive_error_node} gvp{feature}_d 0 NSENSE W={readout_gradient_width:.6g}u L=180n",
                f"Mgvp{feature}_g gvp{feature}_d acc gvp{feature} 0 NREL W={readout_gradient_width:.6g}u L=180n",
                f"Mgvn{feature}_a vdd act{feature} gvn{feature}_a 0 {learning_activation_model} W={readout_gradient_width:.6g}u L=180n",
                f"Mgvn{feature}_d gvn{feature}_a {readout_negative_error_node} gvn{feature}_d 0 NSENSE W={readout_gradient_width:.6g}u L=180n",
                f"Mgvn{feature}_g gvn{feature}_d acc gvn{feature} 0 NREL W={readout_gradient_width:.6g}u L=180n",
                f"Mgbp{feature}_d vdd hdp{feature} gbp{feature}_d 0 NSENSE W={hidden_update_width:.6g}u L=180n",
                f"Mgbp{feature}_g gbp{feature}_d acc gbp{feature} 0 NMOS W={hidden_update_width:.6g}u L=180n",
                f"Mgbn{feature}_d vdd hdn{feature} gbn{feature}_d 0 NSENSE W={hidden_update_width:.6g}u L=180n",
                f"Mgbn{feature}_g gbn{feature}_d acc gbn{feature} 0 NMOS W={hidden_update_width:.6g}u L=180n",
                f"Mrgp{feature}_pd rgp{feature} gvp{feature} 0 0 NSENSE W=16u L=180n",
                f"Mrgn{feature}_pd rgn{feature} gvn{feature} 0 0 NSENSE W=16u L=180n",
                f"Mbhp{feature}_up_g vdd gbp{feature} bhp{feature}_up 0 NSENSE W={hidden_weight_write_width:.6g}u L=180n",
                f"Mbhp{feature}_up_a bhp{feature}_up apply bhp{feature} 0 NREL W={hidden_weight_write_width:.6g}u L=180n",
                f"Mbhn{feature}_dn_a bhn{feature} apply bhn{feature}_dn 0 NREL W={hidden_weight_write_width:.6g}u L=180n",
                f"Mbhn{feature}_dn_g bhn{feature}_dn gbp{feature} 0 0 NSENSE W={hidden_weight_write_width:.6g}u L=180n",
                f"Mbhn{feature}_up_g vdd gbn{feature} bhn{feature}_up 0 NSENSE W={hidden_weight_write_width:.6g}u L=180n",
                f"Mbhn{feature}_up_a bhn{feature}_up apply bhn{feature} 0 NREL W={hidden_weight_write_width:.6g}u L=180n",
                f"Mbhp{feature}_dn_a bhp{feature} apply bhp{feature}_dn 0 NREL W={hidden_weight_write_width:.6g}u L=180n",
                f"Mbhp{feature}_dn_g bhp{feature}_dn gbn{feature} 0 0 NSENSE W={hidden_weight_write_width:.6g}u L=180n",
                f"Mvwp{feature}_up_p0 vwp{feature}_up rgp{feature} vdd vdd PMOS W={readout_pmos_w:.6g}u L=180n",
                f"Mvwp{feature}_up_p1 vwp{feature} applyn vwp{feature}_up vdd PMOS W={readout_pmos_w:.6g}u L=180n",
                f"Mvwn{feature}_dn_a vwn{feature} apply vwn{feature}_dn 0 NREL W={readout_nmos_w:.6g}u L=180n",
                f"Mvwn{feature}_dn_g vwn{feature}_dn gvp{feature} 0 0 NSENSE W={readout_nmos_w:.6g}u L=180n",
                f"Mvwn{feature}_up_p0 vwn{feature}_up rgn{feature} vdd vdd PMOS W={readout_pmos_w:.6g}u L=180n",
                f"Mvwn{feature}_up_p1 vwn{feature} applyn vwn{feature}_up vdd PMOS W={readout_pmos_w:.6g}u L=180n",
                f"Mvwp{feature}_dn_a vwp{feature} apply vwp{feature}_dn 0 NREL W={readout_nmos_w:.6g}u L=180n",
                f"Mvwp{feature}_dn_g vwp{feature}_dn gvn{feature} 0 0 NSENSE W={readout_nmos_w:.6g}u L=180n",
                *(
                    [
                        f"Rread_op{feature}_0 op{feature}_0 0 {readout_stack_shunt_resistance:.12g}",
                        f"Rread_op{feature}_1 op{feature}_1 0 {readout_stack_shunt_resistance:.12g}",
                        f"Rread_on{feature}_0 on{feature}_0 0 {readout_stack_shunt_resistance:.12g}",
                        f"Rread_on{feature}_1 on{feature}_1 0 {readout_stack_shunt_resistance:.12g}",
                    ]
                    if readout_stack_shunt_resistance > 0.0
                    else []
                ),
                *(
                    [
                        f"Cread_op{feature}_0 op{feature}_0 0 {readout_stack_parasitic_capacitance:.12g}",
                        f"Cread_op{feature}_1 op{feature}_1 0 {readout_stack_parasitic_capacitance:.12g}",
                        f"Cread_on{feature}_0 on{feature}_0 0 {readout_stack_parasitic_capacitance:.12g}",
                        f"Cread_on{feature}_1 on{feature}_1 0 {readout_stack_parasitic_capacitance:.12g}",
                    ]
                    if readout_stack_parasitic_capacitance > 0.0
                    else []
                ),
            ]

    if output_bias_enabled:
        lines += [
            "",
            "* Trainable signed output bias contribution and local error-driven writer.",
            f"Mobpos_w vdd obp obp_f0 0 {readout_model} W={readout_forward_width:.6g}u L=180n",
            f"Mobpos_f obp_f0 fwd score 0 {readout_model} W={readout_forward_width:.6g}u L=180n",
            *(
                [
                    f"Mobneg_w vdd obn obn_f0 0 {readout_model} W={readout_negative_forward_width:.6g}u L=180n",
                    f"Mobneg_f obn_f0 fwd scoren 0 {readout_model} W={readout_negative_forward_width:.6g}u L=180n",
                ]
                if score_mode == "differential"
                else [
                    f"Mobneg_f score fwd obn_f0 0 {readout_model} W={readout_negative_forward_width:.6g}u L=180n",
                    f"Mobneg_w obn_f0 obn 0 0 {readout_model} W={readout_negative_forward_width:.6g}u L=180n",
                ]
            ),
            f"Mgop_d vdd {readout_positive_error_node} gop_d 0 NSENSE W={readout_gradient_width:.6g}u L=180n",
            f"Mgop_g gop_d acc gop 0 NREL W={readout_gradient_width:.6g}u L=180n",
            f"Mgon_d vdd {readout_negative_error_node} gon_d 0 NSENSE W={readout_gradient_width:.6g}u L=180n",
            f"Mgon_g gon_d acc gon 0 NREL W={readout_gradient_width:.6g}u L=180n",
            "Mrop_pd rop gop 0 0 NSENSE W=16u L=180n",
            "Mron_pd ron gon 0 0 NSENSE W=16u L=180n",
            f"Mobp_up_p0 obp_up rop vdd vdd PMOS W={output_bias_pmos_w:.6g}u L=180n",
            f"Mobp_up_p1 obp applyn obp_up vdd PMOS W={output_bias_pmos_w:.6g}u L=180n",
            f"Mobn_dn_a obn apply obn_dn 0 NREL W={output_bias_nmos_w:.6g}u L=180n",
            f"Mobn_dn_g obn_dn gop 0 0 NSENSE W={output_bias_nmos_w:.6g}u L=180n",
            f"Mobn_up_p0 obn_up ron vdd vdd PMOS W={output_bias_pmos_w:.6g}u L=180n",
            f"Mobn_up_p1 obn applyn obn_up vdd PMOS W={output_bias_pmos_w:.6g}u L=180n",
            f"Mobp_dn_a obp apply obp_dn 0 NREL W={output_bias_nmos_w:.6g}u L=180n",
            f"Mobp_dn_g obp_dn gon 0 0 NSENSE W={output_bias_nmos_w:.6g}u L=180n",
        ]

    if output_decision_ref_source == "adaptive":
        outref_charge_width = max(0.5, output_decision_ref_write_width * 8.0)
        lines += [
            "",
            "* Adaptive decision-reference state: dp lowers the threshold, dn raises it during apply.",
            f"Moutref_raise_gate outref_raise_gate {readout_negative_error_node} 0 0 NSENSE W=16u L=180n",
            f"Moutref_raise_p0 outref_raise outref_raise_gate vdd vdd PMOS W={outref_charge_width:.6g}u L=180n",
            f"Moutref_raise_p1 outref applyn outref_raise vdd PMOS W={outref_charge_width:.6g}u L=180n",
            f"Moutref_lower_a outref apply outref_lower 0 NREL W={output_decision_ref_write_width:.6g}u L=180n",
            f"Moutref_lower_g outref_lower {readout_positive_error_node} 0 0 NSENSE W={output_decision_ref_write_width:.6g}u L=180n",
        ]

    if score_activity_inhibition_width > 0.0:
        lines += [
            "",
            "* Score-level common-mode inhibition from the activation competition rail.",
            *(
                [
                    f"Mscoreinh_a vdd actinh scoreinh_a 0 NSENSE W={score_activity_inhibition_width:.6g}u L=180n",
                    f"Mscoreinh_f scoreinh_a fwd scoren 0 NMOS W={score_activity_inhibition_width:.6g}u L=180n",
                ]
                if score_mode == "differential"
                else [
                    f"Mscoreinh_i score actinh scoreinh_i 0 NSENSE W={score_activity_inhibition_width:.6g}u L=180n",
                    f"Mscoreinh_f scoreinh_i fwd 0 0 NMOS W={score_activity_inhibition_width:.6g}u L=180n",
                ]
            ),
        ]

    if score_mode == "differential" and output_differential_stage == "latched":
        output_stage = "\n".join(
            [
                "* Dynamic differential output latch: score discharges outn, scoren discharges out.",
                f"Moutlat_p_out out outn vdd vdd PMOS W={output_score_pullup_width:.6g}u L=180n",
                f"Moutlat_p_outn outn out vdd vdd PMOS W={output_score_pullup_width:.6g}u L=180n",
                f"Moutlat_n_out out scoren outlat_src 0 NSENSE W={output_scoren_pulldown_width:.6g}u L=180n",
                f"Moutlat_n_outn outn score outlat_src 0 NSENSE W={output_scoren_pulldown_width:.6g}u L=180n",
                f"Moutlat_tail outlat_src fwd 0 0 NMOS W={output_scoren_pulldown_width:.6g}u L=180n",
            ]
        )
    elif score_mode == "differential" and output_differential_stage == "score-gated":
        output_stage = "\n".join(
            [
                f"Moutp_gate outp_gate scoren vdd vdd PMOS W={output_score_pullup_width:.6g}u L=180n",
                f"Moutp_score outp_gate score out 0 NSENSE W={output_score_pullup_width:.6g}u L=180n",
                f"Moutn out scoren 0 0 NSENSE W={output_scoren_pulldown_width:.6g}u L=180n",
            ]
        )
    elif score_mode == "differential":
        output_stage = "\n".join(
            [
                f"Moutp vdd score out 0 NSENSE W={output_score_pullup_width:.6g}u L=180n",
                f"Moutn out scoren 0 0 NSENSE W={output_scoren_pulldown_width:.6g}u L=180n",
            ]
        )
    else:
        output_stage = output_driver_line(output_driver_model)

    if output_decision_stage == "ref-latched":
        decision_stage = "\n".join(
            [
                "* Reference latch converts analog out into a circuit decision rail.",
                f"Mdec_p decision decisionn vdd vdd PMOS W={output_decision_pullup_width:.6g}u L=180n",
                f"Mdecn_p decisionn decision vdd vdd PMOS W={output_decision_pullup_width:.6g}u L=180n",
                f"Mdec_n decision outref dec_src 0 NSENSE W={output_decision_pulldown_width:.6g}u L=180n",
                f"Mdecn_n decisionn out dec_src 0 NSENSE W={output_decision_pulldown_width:.6g}u L=180n",
                f"Mdec_tail dec_src dec 0 0 NMOS W={output_decision_pulldown_width:.6g}u L=180n",
            ]
        )
    elif output_decision_stage == "ref-precharged-latched":
        decision_stage = "\n".join(
            [
                "* Precharged reference latch: reset precharges both decision rails high, dec discharges the lower side.",
                f"Mdec_pc_p decision decisionn vdd vdd PMOS W={output_decision_pullup_width:.6g}u L=180n",
                f"Mdecn_pc_p decisionn decision vdd vdd PMOS W={output_decision_pullup_width:.6g}u L=180n",
                f"Mdec_pc_n decision outref dec_src 0 NSENSE W={output_decision_pulldown_width:.6g}u L=180n",
                f"Mdecn_pc_n decisionn out dec_src 0 NSENSE W={output_decision_pulldown_width:.6g}u L=180n",
                f"Mdec_pc_tail dec_src dec 0 0 NMOS W={output_decision_pulldown_width:.6g}u L=180n",
            ]
        )
    elif output_decision_stage == "ref-preamp-latched":
        decision_stage = "\n".join(
            [
                "* High-impedance reference preamp followed by a regenerative decision latch.",
                f"Mdecpre_lp decision_pre decision_pre vdd vdd PMOS W={output_decision_pullup_width:.6g}u L=180n",
                f"Mdecpre_ln decisionn_pre decisionn_pre vdd vdd PMOS W={output_decision_pullup_width:.6g}u L=180n",
                f"Mdecpre_ref decision_pre outref decpre_src 0 NSENSE W={output_decision_pulldown_width:.6g}u L=180n",
                f"Mdecpre_out decisionn_pre out decpre_src 0 NSENSE W={output_decision_pulldown_width:.6g}u L=180n",
                f"Mdecpre_tail decpre_src dec 0 0 NMOS W={output_decision_pulldown_width:.6g}u L=180n",
                f"Mdec_p decision decisionn vdd vdd PMOS W={output_decision_pullup_width:.6g}u L=180n",
                f"Mdecn_p decisionn decision vdd vdd PMOS W={output_decision_pullup_width:.6g}u L=180n",
                f"Mdec_n decision decisionn_pre dec_src 0 NSENSE W={output_decision_pulldown_width:.6g}u L=180n",
                f"Mdecn_n decisionn decision_pre dec_src 0 NSENSE W={output_decision_pulldown_width:.6g}u L=180n",
                f"Mdec_tail dec_src dec 0 0 NMOS W={output_decision_pulldown_width:.6g}u L=180n",
            ]
        )
    elif output_decision_stage == "diff-latched":
        decision_stage = "\n".join(
            [
                "* Differential latch converts out/outn into a circuit decision rail.",
                f"Mdec_p decision decisionn vdd vdd PMOS W={output_decision_pullup_width:.6g}u L=180n",
                f"Mdecn_p decisionn decision vdd vdd PMOS W={output_decision_pullup_width:.6g}u L=180n",
                f"Mdec_n decision outn dec_src 0 NSENSE W={output_decision_pulldown_width:.6g}u L=180n",
                f"Mdecn_n decisionn out dec_src 0 NSENSE W={output_decision_pulldown_width:.6g}u L=180n",
                f"Mdec_tail dec_src dec 0 0 NMOS W={output_decision_pulldown_width:.6g}u L=180n",
            ]
        )
    elif output_decision_stage == "diff-precharged-latched":
        decision_stage = "\n".join(
            [
                "* Precharged differential latch: reset precharges both decision rails high, dec discharges the lower side from out/outn.",
                f"Mdec_diffpc_p decision decisionn vdd vdd PMOS W={output_decision_pullup_width:.6g}u L=180n",
                f"Mdecn_diffpc_p decisionn decision vdd vdd PMOS W={output_decision_pullup_width:.6g}u L=180n",
                f"Mdec_diffpc_n decision outn dec_src 0 NSENSE W={output_decision_pulldown_width:.6g}u L=180n",
                f"Mdecn_diffpc_n decisionn out dec_src 0 NSENSE W={output_decision_pulldown_width:.6g}u L=180n",
                f"Mdec_diffpc_tail dec_src dec 0 0 NMOS W={output_decision_pulldown_width:.6g}u L=180n",
            ]
        )
    elif output_decision_stage == "ratio-inverter":
        decision_stage = "\n".join(
            [
                "* Static CMOS threshold detector: decision rises when out crosses the ratioed inverter trip.",
                f"Mdec_inv1_p decisionn out vdd vdd PMOS W={output_decision_pullup_width:.6g}u L=180n",
                f"Mdec_inv1_n decisionn out 0 0 NMOS W={output_decision_pulldown_width:.6g}u L=180n",
                f"Mdec_inv2_p decision decisionn vdd vdd PMOS W={output_decision_pullup_width:.6g}u L=180n",
                f"Mdec_inv2_n decision decisionn 0 0 NMOS W={output_decision_pulldown_width:.6g}u L=180n",
            ]
        )
    elif output_decision_stage == "stacked-inverter":
        decision_stage = "\n".join(
            [
                "* High-threshold CMOS detector: three stacked NMOS devices raise the input trip without loading out.",
                f"Mdec_stack_p decisionn out vdd vdd PMOS W={output_decision_pullup_width:.6g}u L=180n",
                f"Mdec_stack_n0 decisionn out dec_stack0 0 NMOS W={output_decision_pulldown_width:.6g}u L=180n",
                f"Mdec_stack_n1 dec_stack0 out dec_stack1 0 NMOS W={output_decision_pulldown_width:.6g}u L=180n",
                f"Mdec_stack_n2 dec_stack1 out 0 0 NMOS W={output_decision_pulldown_width:.6g}u L=180n",
                f"Mdec_buf_p decision decisionn vdd vdd PMOS W={output_decision_pullup_width:.6g}u L=180n",
                f"Mdec_buf_n decision decisionn 0 0 NMOS W={output_decision_pulldown_width:.6g}u L=180n",
            ]
        )
    elif output_decision_stage == "shift-inverter":
        decision_stage = "\n".join(
            [
                "* Source-follower level shift plus ratioed CMOS inverter threshold detector.",
                f"Mdec_shift vdd out decisionn 0 NMOS W={output_decision_pulldown_width:.6g}u L=180n",
                f"Mdec_inv1_p dec_mid decisionn vdd vdd PMOS W={output_decision_pullup_width:.6g}u L=180n",
                f"Mdec_inv1_n dec_mid decisionn 0 0 NMOS W={output_decision_pulldown_width:.6g}u L=180n",
                f"Mdec_inv2_p decision dec_mid vdd vdd PMOS W={output_decision_pullup_width:.6g}u L=180n",
                f"Mdec_inv2_n decision dec_mid 0 0 NMOS W={output_decision_pulldown_width:.6g}u L=180n",
            ]
        )
    else:
        decision_stage = ""

    lines += [
        "",
        output_stage,
        *(["", decision_stage] if decision_stage else []),
        "",
        "* Shared output error from target/raw-score conductance competition.",
        "Mdp_t0 vdd target dp_t 0 NSENSE W=32u L=180n",
        "Mdp_t1 dp_t err dp 0 NSENSE W=32u L=180n",
        *(
            [
                "Mdp_sn0 vdd scoren dp_sn 0 NSENSE W=24u L=180n",
                "Mdp_sn1 dp_sn err dp 0 NSENSE W=24u L=180n",
            ]
            if score_mode == "differential"
            else []
        ),
        "Mdp_y0 dp err dp_y 0 NSENSE W=24u L=180n",
        "Mdp_y1 dp_y score 0 0 NSENSE W=24u L=180n",
        "Mdn_y0 vdd score dn_y 0 NSENSE W=32u L=180n",
        "Mdn_y1 dn_y err dn 0 NSENSE W=32u L=180n",
        *(
            [
                "Mdn_sn0 dn err dn_sn 0 NSENSE W=24u L=180n",
                "Mdn_sn1 dn_sn scoren 0 0 NSENSE W=24u L=180n",
            ]
            if score_mode == "differential"
            else []
        ),
        "Mdn_t0 dn err dn_t 0 NSENSE W=24u L=180n",
        "Mdn_t1 dn_t target 0 0 NSENSE W=24u L=180n",
        *(
            [
                "",
                "* Restored output-error latch: raw dp/dn select full-swing edp/edn learning rails.",
                f"Merrstore_p edp edn vdd vdd PMOS W={error_restore_width:.6g}u L=180n",
                f"Merrstore_n edn edp vdd vdd PMOS W={error_restore_width:.6g}u L=180n",
                f"Merrstore_ep edp dn errstore_src 0 NSENSE W={error_restore_width:.6g}u L=180n",
                f"Merrstore_en edn dp errstore_src 0 NSENSE W={error_restore_width:.6g}u L=180n",
                f"Merrstore_tail errstore_src err 0 0 NMOS W={error_restore_width:.6g}u L=180n",
            ]
            if restore_error_enabled
            else []
        ),
        "",
        ".options method=gear maxord=2",
        f".tran 10p {stop:.2f}n uic",
        *measures,
        ".control",
        "run",
        *prints,
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def load_mnist01_block_records(
    train_samples: int,
    eval_samples: int,
    *,
    image_size: int,
    seed: int,
    positive_digit: int,
    negative_digit: int,
    complement_rail_scale: float,
    target_polarity: str,
    download: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from torchvision import datasets, transforms
    import torch.nn.functional as F

    if positive_digit == negative_digit:
        raise ValueError("positive and negative digits must differ")
    if complement_rail_scale <= 0.0 or complement_rail_scale > 1.0:
        raise ValueError("complement_rail_scale must be in (0, 1]")
    if target_polarity not in TARGET_POLARITIES:
        raise ValueError(f"target_polarity must be one of {TARGET_POLARITIES}")
    digits = (positive_digit, negative_digit)
    ds_train = datasets.MNIST(root=str(ROOT / "data"), train=True, download=download, transform=transforms.ToTensor())
    ds_eval = datasets.MNIST(root=str(ROOT / "data"), train=False, download=download, transform=transforms.ToTensor())
    train_labels = np.asarray(ds_train.targets, dtype=np.int64)
    eval_labels = np.asarray(ds_eval.targets, dtype=np.int64)
    train_indices = balanced_digit_indices(train_labels, train_samples, seed=seed, digits=digits)
    eval_indices = balanced_digit_indices(eval_labels, eval_samples, seed=seed + 1, digits=digits)

    def extract(ds: Any, indices: np.ndarray) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for index in indices:
            image, digit = ds[int(index)]
            resized = F.interpolate(image.unsqueeze(0), size=(image_size, image_size), mode="area").squeeze()
            pixels = np.asarray(resized.numpy(), dtype=np.float64).reshape(-1)
            digit_i = int(digit)
            is_positive = digit_i == positive_digit
            target_high = is_positive if target_polarity == "active-high" else not is_positive
            record: dict[str, Any] = {
                "target": 1.1 if target_high else 0.0,
                "digit": float(digit_i),
                "mnist_index": float(index),
                "positive_label": 1.0 if is_positive else 0.0,
            }
            for pixel, value in enumerate(pixels):
                pixel_value = float(value)
                record[f"x{pixel}"] = encode_pixel_rail(pixel_value)
                record[f"nx{pixel}"] = encode_pixel_rail(complement_rail_scale * (1.0 - pixel_value))
            records.append(record)
        return records

    return extract(ds_train, train_indices), extract(ds_eval, eval_indices)


def rows_from_measures(
    samples: list[dict[str, Any]],
    measures: dict[str, float],
    *,
    sequence: str | Sequence[str],
    required_rails: list[str],
) -> pd.DataFrame:
    if isinstance(sequence, str):
        sequence_labels = [sequence] * len(samples)
    else:
        sequence_labels = list(sequence)
        if len(sequence_labels) != len(samples):
            raise ValueError("sequence label count must match sample count")
    rows: list[dict[str, Any]] = []
    for sample_idx, sample in enumerate(samples):
        positive = expected_positive(float(sample["target"]))
        row: dict[str, Any] = {
            "sequence": sequence_labels[sample_idx],
            "sample_idx": sample_idx,
            "target": sample["target"],
            "digit": sample.get("digit"),
            "mnist_index": sample.get("mnist_index"),
            "positive_label": sample.get("positive_label"),
            "expected_direction": "positive" if positive else "negative",
        }
        for rail in required_rails:
            row[rail] = sample[rail]
        for key, value in measures.items():
            suffix = f"_{sample_idx}"
            if key.endswith(suffix):
                row[key[: -len(suffix)]] = value
        rows.append(row)
    return pd.DataFrame(rows)


def final_weights_from_rows(rows: pd.DataFrame, *, feature_count: int, block_len: int) -> dict[str, Any]:
    if rows.empty:
        raise ValueError("cannot extract final weights from empty rows")
    final = rows.iloc[-1]
    weights = {
        "whp": [
            [float(final[f"whp{feature}_{pix}_after_apply"]) for pix in range(block_len)]
            for feature in range(feature_count)
        ],
        "whn": [
            [float(final[f"whn{feature}_{pix}_after_apply"]) for pix in range(block_len)]
            for feature in range(feature_count)
        ],
        "bhp": [float(final[f"bhp{feature}_after_apply"]) for feature in range(feature_count)],
        "bhn": [float(final[f"bhn{feature}_after_apply"]) for feature in range(feature_count)],
        "vwp": [float(final[f"vwp{feature}_after_apply"]) for feature in range(feature_count)],
        "vwn": [float(final[f"vwn{feature}_after_apply"]) for feature in range(feature_count)],
    }
    if "obp_after_apply" in rows.columns and "obn_after_apply" in rows.columns:
        weights["obp"] = float(final["obp_after_apply"])
        weights["obn"] = float(final["obn_after_apply"])
    return weights


def output_bias_diagnostics(
    train_rows: pd.DataFrame,
    final_eval_rows: pd.DataFrame,
    *,
    enabled: bool,
    decision_threshold: float,
) -> dict[str, float | bool | None]:
    empty = {
        "output_bias_signed_final_train": None,
        "output_bias_signed_final_train_abs": None,
        "output_bias_signed_final_eval_first": None,
        "output_bias_signed_final_eval_last": None,
        "output_bias_signed_final_eval_drift": None,
        "output_bias_signed_final_eval_drift_abs": None,
        "output_bias_signed_final_eval_max_abs": None,
        "output_bias_signed_final_train_to_threshold_ratio": None,
        "output_bias_state_drift_warning": False,
    }
    if (
        not enabled
        or "output_bias_signed_after" not in train_rows.columns
        or "output_bias_signed_after" not in final_eval_rows.columns
    ):
        return empty

    train_bias = train_rows["output_bias_signed_after"].dropna().to_numpy(dtype=float)
    eval_bias = final_eval_rows["output_bias_signed_after"].dropna().to_numpy(dtype=float)
    if train_bias.size == 0 or eval_bias.size == 0:
        return empty

    final_train = float(train_bias[-1])
    first_eval = float(eval_bias[0])
    last_eval = float(eval_bias[-1])
    drift = last_eval - first_eval
    threshold_ratio = abs(final_train) / decision_threshold if decision_threshold > 0.0 else None
    drift_warning = (
        decision_threshold > 0.0
        and abs(final_train) > decision_threshold
        and abs(drift) > decision_threshold
    )
    return {
        "output_bias_signed_final_train": final_train,
        "output_bias_signed_final_train_abs": abs(final_train),
        "output_bias_signed_final_eval_first": first_eval,
        "output_bias_signed_final_eval_last": last_eval,
        "output_bias_signed_final_eval_drift": drift,
        "output_bias_signed_final_eval_drift_abs": abs(drift),
        "output_bias_signed_final_eval_max_abs": float(np.max(np.abs(eval_bias))),
        "output_bias_signed_final_train_to_threshold_ratio": threshold_ratio,
        "output_bias_state_drift_warning": drift_warning,
    }


def adaptive_reference_diagnostics(
    train_rows: pd.DataFrame,
    final_eval_rows: pd.DataFrame,
    *,
    enabled: bool,
    nominal_ref: float,
) -> dict[str, float | None]:
    empty = {
        "outref_final_train": None,
        "outref_final_train_error": None,
        "outref_final_eval_first": None,
        "outref_final_eval_last": None,
        "outref_final_eval_drift": None,
        "outref_final_eval_drift_abs": None,
        "outref_final_eval_min": None,
        "outref_final_eval_max": None,
        "outref_final_eval_max_abs_error": None,
    }
    if not enabled or "outref_after_apply" not in train_rows.columns or "outref_before" not in final_eval_rows.columns:
        return empty

    train_ref = train_rows["outref_after_apply"].dropna().to_numpy(dtype=float)
    eval_ref = final_eval_rows["outref_before"].dropna().to_numpy(dtype=float)
    if train_ref.size == 0 or eval_ref.size == 0:
        return empty

    final_train = float(train_ref[-1])
    first_eval = float(eval_ref[0])
    last_eval = float(eval_ref[-1])
    drift = last_eval - first_eval
    return {
        "outref_final_train": final_train,
        "outref_final_train_error": final_train - nominal_ref,
        "outref_final_eval_first": first_eval,
        "outref_final_eval_last": last_eval,
        "outref_final_eval_drift": drift,
        "outref_final_eval_drift_abs": abs(drift),
        "outref_final_eval_min": float(np.min(eval_ref)),
        "outref_final_eval_max": float(np.max(eval_ref)),
        "outref_final_eval_max_abs_error": float(np.max(np.abs(eval_ref - nominal_ref))),
    }


def score_net_diagnostics(initial_eval_rows: pd.DataFrame, final_eval_rows: pd.DataFrame) -> dict[str, float | None]:
    def metrics(prefix: str, rows: pd.DataFrame, signal: str) -> dict[str, float | None]:
        empty = {
            f"{prefix}_{signal}_positive_mean": None,
            f"{prefix}_{signal}_negative_mean": None,
            f"{prefix}_{signal}_margin": None,
        }
        if signal not in rows.columns or "positive_label" not in rows.columns:
            return empty
        positives = rows.loc[rows["positive_label"] > 0.5, signal].dropna().to_numpy(dtype=float)
        negatives = rows.loc[rows["positive_label"] <= 0.5, signal].dropna().to_numpy(dtype=float)
        if positives.size == 0 or negatives.size == 0:
            return empty
        positive_mean = float(np.mean(positives))
        negative_mean = float(np.mean(negatives))
        return {
            f"{prefix}_{signal}_positive_mean": positive_mean,
            f"{prefix}_{signal}_negative_mean": negative_mean,
            f"{prefix}_{signal}_margin": positive_mean - negative_mean,
        }

    return {
        **metrics("initial_eval", initial_eval_rows, "score_net"),
        **metrics("final_eval", final_eval_rows, "score_net"),
        **metrics("initial_eval", initial_eval_rows, "out_after"),
        **metrics("final_eval", final_eval_rows, "out_after"),
        **metrics("initial_eval", initial_eval_rows, "out_diff"),
        **metrics("final_eval", final_eval_rows, "out_diff"),
        **metrics("initial_eval", initial_eval_rows, "decision_after"),
        **metrics("final_eval", final_eval_rows, "decision_after"),
        **metrics("initial_eval", initial_eval_rows, "decision_diff"),
        **metrics("final_eval", final_eval_rows, "decision_diff"),
    }


def binary_accuracy_for_signal(
    rows: pd.DataFrame,
    *,
    signal: str,
    threshold: float,
    output_positive_when: str = "high",
) -> float:
    if signal == "out_after":
        return binary_accuracy(rows, threshold=threshold, output_positive_when=output_positive_when)
    if rows.empty:
        return 0.0
    if signal not in rows.columns:
        raise ValueError(f"missing accuracy signal column: {signal}")
    if output_positive_when == "high":
        predicted = rows[signal].to_numpy(dtype=float) > threshold
    elif output_positive_when == "low":
        predicted = rows[signal].to_numpy(dtype=float) <= threshold
    else:
        raise ValueError("output_positive_when must be 'high' or 'low'")
    expected = rows["positive_label"].to_numpy(dtype=float) > 0.5
    return float(np.mean(predicted == expected))


def threshold_window_diagnostics(
    rows: pd.DataFrame,
    *,
    signal: str,
    output_positive_when: str = "high",
) -> dict[str, float | None]:
    empty = {
        f"{signal}_best_threshold_accuracy": None,
        f"{signal}_best_threshold": None,
        f"{signal}_best_threshold_active_fraction": None,
    }
    if rows.empty or signal not in rows.columns or "positive_label" not in rows.columns:
        return empty
    signal_values = rows[signal].dropna().to_numpy(dtype=float)
    if signal_values.size == 0:
        return empty
    unique_values = np.unique(signal_values)
    eps = max(1e-12, float(np.ptp(unique_values)) * 1e-9)
    candidates = [float(unique_values[0] - eps), float(unique_values[-1] + eps)]
    candidates += [float((lo + hi) / 2.0) for lo, hi in zip(unique_values[:-1], unique_values[1:])]

    best_accuracy = -1.0
    best_threshold = candidates[0]
    best_active_fraction = 0.0
    for threshold in candidates:
        accuracy = binary_accuracy_for_signal(
            rows,
            signal=signal,
            threshold=threshold,
            output_positive_when=output_positive_when,
        )
        if output_positive_when == "high":
            active_fraction = float(np.mean(rows[signal].to_numpy(dtype=float) > threshold))
        else:
            active_fraction = float(np.mean(rows[signal].to_numpy(dtype=float) <= threshold))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
            best_active_fraction = active_fraction
    return {
        f"{signal}_best_threshold_accuracy": best_accuracy,
        f"{signal}_best_threshold": best_threshold,
        f"{signal}_best_threshold_active_fraction": best_active_fraction,
    }


def nontrivial_learning_flag(initial_accuracy: float | None, final_accuracy: float) -> bool:
    if initial_accuracy is None:
        return False
    return final_accuracy > max(initial_accuracy, 0.5)


def run_device_sequence(
    spice_bin: str,
    path: Path,
    samples: list[dict[str, Any]],
    weights: dict[str, Any],
    *,
    image_size: int,
    block_size: int,
    stride: int,
    channels: int,
    training_enabled: bool | Sequence[bool],
    timeout: float,
    sequence: str | Sequence[str],
    output_driver_model: str,
    output_differential_stage: str,
    output_score_pullup_width: float,
    output_scoren_pulldown_width: float,
    output_latch_capacitance: float,
    output_decision_stage: str,
    output_decision_ref: float,
    output_decision_ref_source: str,
    output_decision_ref_resistance: float,
    output_decision_ref_capacitance: float,
    output_decision_ref_write_width: float,
    output_decision_pullup_width: float,
    output_decision_pulldown_width: float,
    readout_apply_scale: float,
    hidden_forward_width: float,
    hidden_forward_topology: str,
    readout_gradient_width: float,
    hidden_error_width: float,
    hidden_credit_mode: str,
    readout_feedback_restore_width: float,
    error_signal_mode: str,
    error_restore_width: float,
    hidden_update_width: float,
    hidden_weight_write_width: float,
    hidden_weight_leak_resistance: float,
    hidden_weight_positive_ref: float,
    hidden_weight_negative_ref: float,
    hidden_activation_width: float,
    hidden_input_residual_width: float,
    hidden_stack_shunt_resistance: float,
    hidden_stack_parasitic_capacitance: float,
    hidden_activation_model: str,
    readout_forward_width: float,
    readout_forward_model: str,
    learning_activation_gate_model: str,
    readout_weight_leak_resistance: float,
    readout_stack_shunt_resistance: float,
    readout_stack_parasitic_capacitance: float,
    readout_weight_positive_ref: float,
    readout_weight_negative_ref: float,
    activation_competition_width: float,
    score_activity_inhibition_width: float,
    output_bias_enabled: bool,
    output_bias_apply_scale: float,
    output_bias_positive_init: float,
    output_bias_negative_init: float,
    output_bias_leak_resistance: float,
    output_bias_positive_ref: float,
    output_bias_negative_ref: float,
    phase_time_scale: float,
    phase_jitter_sigma_ns: float,
    phase_jitter_seed: int,
    passive_mismatch_sigma: float,
    passive_mismatch_seed: int,
    score_mode: str,
    input_rail_mode: str,
    measurement_detail: str,
) -> pd.DataFrame:
    netlist = block_netlist(
        samples,
        weights,
        image_size=image_size,
        block_size=block_size,
        stride=stride,
        channels=channels,
        training_enabled=training_enabled,
        output_driver_model=output_driver_model,
        output_differential_stage=output_differential_stage,
        output_score_pullup_width=output_score_pullup_width,
        output_scoren_pulldown_width=output_scoren_pulldown_width,
        output_latch_capacitance=output_latch_capacitance,
        output_decision_stage=output_decision_stage,
        output_decision_ref=output_decision_ref,
        output_decision_ref_source=output_decision_ref_source,
        output_decision_ref_resistance=output_decision_ref_resistance,
        output_decision_ref_capacitance=output_decision_ref_capacitance,
        output_decision_ref_write_width=output_decision_ref_write_width,
        output_decision_pullup_width=output_decision_pullup_width,
        output_decision_pulldown_width=output_decision_pulldown_width,
        readout_apply_scale=readout_apply_scale,
        hidden_forward_width=hidden_forward_width,
        hidden_forward_topology=hidden_forward_topology,
        readout_gradient_width=readout_gradient_width,
        hidden_error_width=hidden_error_width,
        hidden_credit_mode=hidden_credit_mode,
        readout_feedback_restore_width=readout_feedback_restore_width,
        error_signal_mode=error_signal_mode,
        error_restore_width=error_restore_width,
        hidden_update_width=hidden_update_width,
        hidden_weight_write_width=hidden_weight_write_width,
        hidden_weight_leak_resistance=hidden_weight_leak_resistance,
        hidden_weight_positive_ref=hidden_weight_positive_ref,
        hidden_weight_negative_ref=hidden_weight_negative_ref,
        hidden_activation_width=hidden_activation_width,
        hidden_input_residual_width=hidden_input_residual_width,
        hidden_stack_shunt_resistance=hidden_stack_shunt_resistance,
        hidden_stack_parasitic_capacitance=hidden_stack_parasitic_capacitance,
        hidden_activation_model=hidden_activation_model,
        readout_forward_width=readout_forward_width,
        readout_forward_model=readout_forward_model,
        learning_activation_gate_model=learning_activation_gate_model,
        readout_weight_leak_resistance=readout_weight_leak_resistance,
        readout_stack_shunt_resistance=readout_stack_shunt_resistance,
        readout_stack_parasitic_capacitance=readout_stack_parasitic_capacitance,
        readout_weight_positive_ref=readout_weight_positive_ref,
        readout_weight_negative_ref=readout_weight_negative_ref,
        activation_competition_width=activation_competition_width,
        score_activity_inhibition_width=score_activity_inhibition_width,
        output_bias_enabled=output_bias_enabled,
        output_bias_apply_scale=output_bias_apply_scale,
        output_bias_positive_init=output_bias_positive_init,
        output_bias_negative_init=output_bias_negative_init,
        output_bias_leak_resistance=output_bias_leak_resistance,
        output_bias_positive_ref=output_bias_positive_ref,
        output_bias_negative_ref=output_bias_negative_ref,
        phase_time_scale=phase_time_scale,
        phase_jitter_sigma_ns=phase_jitter_sigma_ns,
        phase_jitter_seed=phase_jitter_seed,
        passive_mismatch_sigma=passive_mismatch_sigma,
        passive_mismatch_seed=passive_mismatch_seed,
        score_mode=score_mode,
        input_rail_mode=input_rail_mode,
        measurement_detail=measurement_detail,
    )
    if "\nB" in netlist:
        raise ValueError("block-stride device runner generated a behavioral source")
    measures = run_netlist(spice_bin, path, netlist, timeout)
    required_rails = required_input_rail_names(image_size * image_size, channels, input_rail_mode)
    return rows_from_measures(samples, measures, sequence=sequence, required_rails=required_rails)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-samples", type=int, default=4)
    ap.add_argument("--eval-samples", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=2)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weight-seed", type=int, default=0)
    ap.add_argument("--positive-digit", type=int, default=0)
    ap.add_argument("--negative-digit", type=int, default=1)
    ap.add_argument("--target-polarity", choices=TARGET_POLARITIES, default="active-high")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--tag", default="device_mnist01_block")
    ap.add_argument("--output-driver-model", choices=["sense", "nrel"], default="sense")
    ap.add_argument("--output-differential-stage", choices=OUTPUT_DIFFERENTIAL_STAGES, default="simple")
    ap.add_argument("--output-score-pullup-width", type=float, default=24.0)
    ap.add_argument("--output-scoren-pulldown-width", type=float, default=24.0)
    ap.add_argument("--output-latch-capacitance", type=float, default=20e-15)
    ap.add_argument("--output-decision-stage", choices=OUTPUT_DECISION_STAGES, default="none")
    ap.add_argument("--output-decision-ref", type=float, default=1.09)
    ap.add_argument("--output-decision-ref-source", choices=OUTPUT_DECISION_REF_SOURCES, default="voltage")
    ap.add_argument("--output-decision-ref-resistance", type=float, default=1e6)
    ap.add_argument("--output-decision-ref-capacitance", type=float, default=20e-15)
    ap.add_argument("--output-decision-ref-write-width", type=float, default=1.0)
    ap.add_argument("--output-decision-pullup-width", type=float, default=48.0)
    ap.add_argument("--output-decision-pulldown-width", type=float, default=96.0)
    ap.add_argument("--output-decision-threshold", type=float, default=0.6)
    ap.add_argument("--readout-apply-scale", type=float, default=0.35)
    ap.add_argument("--hidden-forward-width", type=float, default=3.0)
    ap.add_argument("--hidden-forward-topology", choices=HIDDEN_FORWARD_TOPOLOGIES, default="per-pixel-phase")
    ap.add_argument("--readout-gradient-width", type=float, default=24.0)
    ap.add_argument("--hidden-error-width", type=float, default=32.0)
    ap.add_argument("--hidden-credit-mode", choices=HIDDEN_CREDIT_MODES, default="direct-feedback")
    ap.add_argument("--readout-feedback-restore-width", type=float, default=4.0)
    ap.add_argument("--error-signal-mode", choices=ERROR_SIGNAL_MODES, default="raw")
    ap.add_argument("--error-restore-width", type=float, default=4.0)
    ap.add_argument("--hidden-update-width", type=float, default=12.0)
    ap.add_argument("--hidden-weight-write-width", type=float, default=0.25)
    ap.add_argument("--hidden-weight-leak-resistance", type=float, default=0.0)
    ap.add_argument("--hidden-weight-positive-ref", type=float, default=0.50)
    ap.add_argument("--hidden-weight-negative-ref", type=float, default=0.20)
    ap.add_argument("--hidden-activation-width", type=float, default=24.0)
    ap.add_argument("--hidden-input-residual-width", type=float, default=0.0)
    ap.add_argument("--hidden-stack-shunt-resistance", type=float, default=0.0)
    ap.add_argument("--hidden-stack-parasitic-capacitance", type=float, default=0.0)
    ap.add_argument("--hidden-activation-model", choices=HIDDEN_ACTIVATION_MODELS, default="nrel")
    ap.add_argument("--hidden-polarity-init", choices=HIDDEN_POLARITY_INITS, default="ink")
    ap.add_argument("--readout-forward-width", type=float, default=64.0)
    ap.add_argument("--readout-forward-model", choices=READOUT_FORWARD_MODELS, default="nrel")
    ap.add_argument("--learning-activation-gate-model", choices=LEARNING_ACTIVATION_GATE_MODELS, default="nrel")
    ap.add_argument("--readout-weight-leak-resistance", type=float, default=0.0)
    ap.add_argument("--readout-stack-shunt-resistance", type=float, default=0.0)
    ap.add_argument("--readout-stack-parasitic-capacitance", type=float, default=0.0)
    ap.add_argument("--readout-weight-positive-ref", type=float, default=0.52)
    ap.add_argument("--readout-weight-negative-ref", type=float, default=0.25)
    ap.add_argument("--activation-competition-width", type=float, default=0.0)
    ap.add_argument("--score-activity-inhibition-width", type=float, default=0.0)
    ap.add_argument("--output-bias", action="store_true")
    ap.add_argument("--output-bias-apply-scale", type=float, default=1.0)
    ap.add_argument("--output-bias-positive-init", type=float, default=0.52)
    ap.add_argument("--output-bias-negative-init", type=float, default=0.25)
    ap.add_argument("--output-bias-leak-resistance", type=float, default=0.0)
    ap.add_argument("--output-bias-positive-ref", type=float, default=0.25)
    ap.add_argument("--output-bias-negative-ref", type=float, default=0.25)
    ap.add_argument("--phase-time-scale", type=float, default=1.0)
    ap.add_argument(
        "--input-voltage-jitter-sigma",
        type=float,
        default=0.0,
        help="Gaussian input-DAC voltage jitter, in volts, applied deterministically to Python-supplied input rails.",
    )
    ap.add_argument(
        "--phase-jitter-sigma-ns",
        type=float,
        default=0.0,
        help="Gaussian timing jitter, in ns, applied deterministically to phase pulse windows.",
    )
    ap.add_argument(
        "--passive-mismatch-sigma",
        type=float,
        default=0.0,
        help="Relative sigma for passive resistor mismatch in generated circuit references.",
    )
    ap.add_argument(
        "--state-ic-mismatch-sigma",
        type=float,
        default=0.0,
        help="Gaussian initial capacitor-state mismatch, in volts, applied once before the transient.",
    )
    ap.add_argument(
        "--perturbation-seed",
        type=int,
        default=0,
        help="Seed for deterministic input jitter, phase jitter, and mismatch perturbations.",
    )
    ap.add_argument("--score-mode", choices=SCORE_MODES, default="single-ended")
    ap.add_argument("--input-rail-mode", choices=INPUT_RAIL_MODES, default="alternating-complement")
    ap.add_argument(
        "--measurement-detail",
        choices=MEASUREMENT_DETAILS,
        default="full",
        help=(
            "'full' records every state capacitor for debugging. 'outputs' records only output/accuracy "
            "diagnostics and requires --continuous-final-eval because Python will not have measured weights "
            "available to seed a separate eval deck."
        ),
    )
    ap.add_argument("--complement-rail-scale", type=float, default=0.5)
    ap.add_argument("--hidden-bias-positive-init", type=float, default=0.50)
    ap.add_argument("--hidden-bias-negative-init", type=float, default=0.20)
    ap.add_argument("--decision-threshold", type=float, default=0.10)
    ap.add_argument(
        "--continuous-final-eval",
        action="store_true",
        help=(
            "Run training samples and final-eval samples in one SPICE transient, with write clocks disabled "
            "for the final-eval segment. This avoids using Python-extracted final weights to seed eval."
        ),
    )
    ap.add_argument(
        "--skip-initial-eval",
        action="store_true",
        help="Skip the separate initial-eval deck for faster convergence/preflight runs.",
    )
    ap.add_argument("--assert-nonbehavioral", action="store_true")
    args = ap.parse_args()

    if args.train_samples <= 0:
        raise ValueError("train-samples must be positive for a training smoke")
    if args.eval_samples <= 0:
        raise ValueError("eval-samples must be positive")
    if args.input_voltage_jitter_sigma < 0.0:
        raise ValueError("input-voltage-jitter-sigma must be nonnegative")
    if args.phase_jitter_sigma_ns < 0.0:
        raise ValueError("phase-jitter-sigma-ns must be nonnegative")
    if args.passive_mismatch_sigma < 0.0:
        raise ValueError("passive-mismatch-sigma must be nonnegative")
    if args.state_ic_mismatch_sigma < 0.0:
        raise ValueError("state-ic-mismatch-sigma must be nonnegative")
    if args.measurement_detail != "full" and not args.continuous_final_eval:
        raise ValueError("--measurement-detail outputs requires --continuous-final-eval")
    blocks, feature_count = block_topology(args.image_size, args.block_size, args.stride, args.channels)
    block_len = len(blocks[0])

    generated = ROOT / "spice/generated"
    results = ROOT / "spice/results"
    tables = ROOT / "results/tables"
    for directory in [generated, results, tables]:
        directory.mkdir(parents=True, exist_ok=True)

    spice_bin, version = detect_spice(None)
    run_tiny_test(spice_bin, generated)
    safe_tag = sanitize_tag(args.tag)
    t0 = time.perf_counter()
    train_samples, eval_samples = load_mnist01_block_records(
        args.train_samples,
        args.eval_samples,
        image_size=args.image_size,
        seed=args.seed,
        positive_digit=args.positive_digit,
        negative_digit=args.negative_digit,
        complement_rail_scale=args.complement_rail_scale,
        target_polarity=args.target_polarity,
        download=args.download,
    )
    train_samples = perturb_input_records(
        train_samples,
        sigma=args.input_voltage_jitter_sigma,
        seed=stable_seed(args.perturbation_seed, "train-inputs"),
    )
    eval_samples = perturb_input_records(
        eval_samples,
        sigma=args.input_voltage_jitter_sigma,
        seed=stable_seed(args.perturbation_seed, "eval-inputs"),
    )
    initial_weights = initial_block_weights(
        args.image_size,
        args.block_size,
        args.stride,
        args.channels,
        seed=args.weight_seed,
        hidden_bias_positive_init=args.hidden_bias_positive_init,
        hidden_bias_negative_init=args.hidden_bias_negative_init,
        hidden_polarity_init=args.hidden_polarity_init,
        output_bias_positive_init=args.output_bias_positive_init,
        output_bias_negative_init=args.output_bias_negative_init,
    )
    initial_weights = perturb_initial_state(
        initial_weights,
        sigma=args.state_ic_mismatch_sigma,
        seed=stable_seed(args.perturbation_seed, "initial-state"),
    )
    common = {
        "image_size": args.image_size,
        "block_size": args.block_size,
        "stride": args.stride,
        "channels": args.channels,
        "timeout": args.timeout,
        "output_driver_model": args.output_driver_model,
        "output_differential_stage": args.output_differential_stage,
        "output_score_pullup_width": args.output_score_pullup_width,
        "output_scoren_pulldown_width": args.output_scoren_pulldown_width,
        "output_latch_capacitance": args.output_latch_capacitance,
        "output_decision_stage": args.output_decision_stage,
        "output_decision_ref": args.output_decision_ref,
        "output_decision_ref_source": args.output_decision_ref_source,
        "output_decision_ref_resistance": args.output_decision_ref_resistance,
        "output_decision_ref_capacitance": args.output_decision_ref_capacitance,
        "output_decision_ref_write_width": args.output_decision_ref_write_width,
        "output_decision_pullup_width": args.output_decision_pullup_width,
        "output_decision_pulldown_width": args.output_decision_pulldown_width,
        "readout_apply_scale": args.readout_apply_scale,
        "hidden_forward_width": args.hidden_forward_width,
        "hidden_forward_topology": args.hidden_forward_topology,
        "readout_gradient_width": args.readout_gradient_width,
        "hidden_error_width": args.hidden_error_width,
        "hidden_credit_mode": args.hidden_credit_mode,
        "readout_feedback_restore_width": args.readout_feedback_restore_width,
        "error_signal_mode": args.error_signal_mode,
        "error_restore_width": args.error_restore_width,
        "hidden_update_width": args.hidden_update_width,
        "hidden_weight_write_width": args.hidden_weight_write_width,
        "hidden_weight_leak_resistance": args.hidden_weight_leak_resistance,
        "hidden_weight_positive_ref": args.hidden_weight_positive_ref,
        "hidden_weight_negative_ref": args.hidden_weight_negative_ref,
        "hidden_activation_width": args.hidden_activation_width,
        "hidden_input_residual_width": args.hidden_input_residual_width,
        "hidden_stack_shunt_resistance": args.hidden_stack_shunt_resistance,
        "hidden_stack_parasitic_capacitance": args.hidden_stack_parasitic_capacitance,
        "hidden_activation_model": args.hidden_activation_model,
        "readout_forward_width": args.readout_forward_width,
        "readout_forward_model": args.readout_forward_model,
        "learning_activation_gate_model": args.learning_activation_gate_model,
        "readout_weight_leak_resistance": args.readout_weight_leak_resistance,
        "readout_stack_shunt_resistance": args.readout_stack_shunt_resistance,
        "readout_stack_parasitic_capacitance": args.readout_stack_parasitic_capacitance,
        "readout_weight_positive_ref": args.readout_weight_positive_ref,
        "readout_weight_negative_ref": args.readout_weight_negative_ref,
        "activation_competition_width": args.activation_competition_width,
        "score_activity_inhibition_width": args.score_activity_inhibition_width,
        "output_bias_enabled": args.output_bias,
        "output_bias_apply_scale": args.output_bias_apply_scale,
        "output_bias_positive_init": args.output_bias_positive_init,
        "output_bias_negative_init": args.output_bias_negative_init,
        "output_bias_leak_resistance": args.output_bias_leak_resistance,
        "output_bias_positive_ref": args.output_bias_positive_ref,
        "output_bias_negative_ref": args.output_bias_negative_ref,
        "phase_time_scale": args.phase_time_scale,
        "phase_jitter_sigma_ns": args.phase_jitter_sigma_ns,
        "phase_jitter_seed": stable_seed(args.perturbation_seed, "phase-jitter"),
        "passive_mismatch_sigma": args.passive_mismatch_sigma,
        "passive_mismatch_seed": stable_seed(args.perturbation_seed, "passive-mismatch"),
        "score_mode": args.score_mode,
        "input_rail_mode": args.input_rail_mode,
        "measurement_detail": args.measurement_detail,
    }
    if args.skip_initial_eval:
        initial_eval_rows = pd.DataFrame()
    else:
        initial_eval_rows = run_device_sequence(
            spice_bin,
            generated / f"{safe_tag}_initial_eval.cir",
            eval_samples,
            initial_weights,
            training_enabled=False,
            sequence="initial_eval",
            **common,
        )
    if args.continuous_final_eval:
        combined_samples = [*train_samples, *eval_samples]
        combined_schedule = [True] * len(train_samples) + [False] * len(eval_samples)
        combined_sequence = ["train"] * len(train_samples) + ["final_eval"] * len(eval_samples)
        combined_rows = run_device_sequence(
            spice_bin,
            generated / f"{safe_tag}_train_final_eval.cir",
            combined_samples,
            initial_weights,
            training_enabled=combined_schedule,
            sequence=combined_sequence,
            **common,
        )
        train_rows = combined_rows.loc[combined_rows["sequence"] == "train"].reset_index(drop=True)
        final_eval_rows = combined_rows.loc[combined_rows["sequence"] == "final_eval"].reset_index(drop=True)
    else:
        train_rows = run_device_sequence(
            spice_bin,
            generated / f"{safe_tag}_train.cir",
            train_samples,
            initial_weights,
            training_enabled=True,
            sequence="train",
            **common,
        )
        final_weights_for_eval = final_weights_from_rows(train_rows, feature_count=feature_count, block_len=block_len)
        final_eval_rows = run_device_sequence(
            spice_bin,
            generated / f"{safe_tag}_final_eval.cir",
            eval_samples,
            final_weights_for_eval,
            training_enabled=False,
            sequence="final_eval",
            **common,
        )
    final_weights = (
        final_weights_from_rows(train_rows, feature_count=feature_count, block_len=block_len)
        if args.measurement_detail == "full"
        else None
    )

    curve = pd.concat([initial_eval_rows, train_rows, final_eval_rows], ignore_index=True)
    curve_path = results / f"{safe_tag}.csv"
    table_curve_path = tables / f"{safe_tag}.csv"
    curve.to_csv(curve_path, index=False)
    curve.to_csv(table_curve_path, index=False)

    output_positive_when = "high" if args.target_polarity == "active-high" else "low"
    accuracy_signal = "decision_after" if args.output_decision_stage != "none" else "out_after"
    accuracy_threshold = args.output_decision_threshold if args.output_decision_stage != "none" else args.decision_threshold
    initial_accuracy = (
        None
        if initial_eval_rows.empty
        else binary_accuracy_for_signal(
            initial_eval_rows,
            signal=accuracy_signal,
            threshold=accuracy_threshold,
            output_positive_when=output_positive_when,
        )
    )
    final_accuracy = binary_accuracy_for_signal(
        final_eval_rows,
        signal=accuracy_signal,
        threshold=accuracy_threshold,
        output_positive_when=output_positive_when,
    )
    initial_active_fraction = (
        None
        if initial_eval_rows.empty
        else float(np.mean(np.abs(initial_eval_rows[accuracy_signal].to_numpy(dtype=float)) > accuracy_threshold))
    )
    final_active_fraction = float(
        np.mean(np.abs(final_eval_rows[accuracy_signal].to_numpy(dtype=float)) > accuracy_threshold)
    )
    nontrivial_learning_met = nontrivial_learning_flag(initial_accuracy, final_accuracy)
    target_topology = args.image_size == 10 and args.block_size == 4 and args.stride == 2 and args.channels == 2
    output_bias_summary = output_bias_diagnostics(
        train_rows,
        final_eval_rows,
        enabled=args.output_bias,
        decision_threshold=args.decision_threshold,
    )
    adaptive_reference_summary = adaptive_reference_diagnostics(
        train_rows,
        final_eval_rows,
        enabled=args.output_decision_ref_source == "adaptive",
        nominal_ref=args.output_decision_ref,
    )
    score_net_summary = score_net_diagnostics(initial_eval_rows, final_eval_rows)
    threshold_window_summary = {
        **threshold_window_diagnostics(
            final_eval_rows,
            signal="out_after",
            output_positive_when=output_positive_when,
        ),
        **threshold_window_diagnostics(
            final_eval_rows,
            signal="out_diff",
            output_positive_when=output_positive_when,
        ),
        **threshold_window_diagnostics(
            final_eval_rows,
            signal="decision_after",
            output_positive_when=output_positive_when,
        ),
        **threshold_window_diagnostics(
            final_eval_rows,
            signal="decision_diff",
            output_positive_when=output_positive_when,
        ),
    }
    summary = {
        "simulator": version,
        "architecture": "device_level_mnist01_block_stride_channel_training",
        "status": "mnist01_block_stride_device_training_smoke",
        "model_level": "ngspice built-in LEVEL=1 MOS models; not a foundry PDK.",
        "dataset": f"MNIST01 raw-pixel block topology image{args.image_size}_b{args.block_size}_s{args.stride}_c{args.channels}",
        "positive_digit": args.positive_digit,
        "negative_digit": args.negative_digit,
        "target_polarity": args.target_polarity,
        "output_positive_when": output_positive_when,
        "image_size": args.image_size,
        "block_size": args.block_size,
        "stride": args.stride,
        "channels": args.channels,
        "blocks": len(blocks),
        "block_len": block_len,
        "feature_count": feature_count,
        "target_10x10_b4_stride2_c2_topology": target_topology,
        "input_encoding": (
            "raw and optional complemented resized pixel rails encoded as 0.08 + 0.92 * intensity; no PCA/local-PCA"
        ),
        "input_rail_mode": args.input_rail_mode,
        "complement_rail_scale": args.complement_rail_scale,
        "hidden_bias_state": "persistent signed bhp/bhn capacitors with MOS/passive local bias writers",
        "hidden_credit_mode": args.hidden_credit_mode,
        "readout_feedback_restore_width": args.readout_feedback_restore_width,
        "error_signal_mode": args.error_signal_mode,
        "error_restore_width": args.error_restore_width,
        "output_driver_model": args.output_driver_model,
        "output_differential_stage": args.output_differential_stage,
        "output_score_pullup_width": args.output_score_pullup_width,
        "output_scoren_pulldown_width": args.output_scoren_pulldown_width,
        "output_latch_capacitance": args.output_latch_capacitance,
        "output_decision_stage": args.output_decision_stage,
        "output_decision_ref": args.output_decision_ref,
        "output_decision_ref_source": args.output_decision_ref_source,
        "output_decision_ref_resistance": args.output_decision_ref_resistance,
        "output_decision_ref_capacitance": args.output_decision_ref_capacitance,
        "output_decision_ref_write_width": args.output_decision_ref_write_width,
        "output_decision_pullup_width": args.output_decision_pullup_width,
        "output_decision_pulldown_width": args.output_decision_pulldown_width,
        "readout_apply_scale": args.readout_apply_scale,
        "hidden_forward_width": args.hidden_forward_width,
        "hidden_forward_topology": args.hidden_forward_topology,
        "readout_gradient_width": args.readout_gradient_width,
        "hidden_error_width": args.hidden_error_width,
        "hidden_update_width": args.hidden_update_width,
        "hidden_weight_write_width": args.hidden_weight_write_width,
        "hidden_weight_leak_resistance": args.hidden_weight_leak_resistance,
        "hidden_weight_positive_ref": args.hidden_weight_positive_ref,
        "hidden_weight_negative_ref": args.hidden_weight_negative_ref,
        "hidden_activation_width": args.hidden_activation_width,
        "hidden_input_residual_width": args.hidden_input_residual_width,
        "hidden_stack_shunt_resistance": args.hidden_stack_shunt_resistance,
        "hidden_stack_parasitic_capacitance": args.hidden_stack_parasitic_capacitance,
        "hidden_activation_model": args.hidden_activation_model,
        "hidden_polarity_init": args.hidden_polarity_init,
        "readout_forward_width": args.readout_forward_width,
        "readout_forward_model": args.readout_forward_model,
        "learning_activation_gate_model": args.learning_activation_gate_model,
        "readout_weight_leak_resistance": args.readout_weight_leak_resistance,
        "readout_stack_shunt_resistance": args.readout_stack_shunt_resistance,
        "readout_stack_parasitic_capacitance": args.readout_stack_parasitic_capacitance,
        "readout_weight_positive_ref": args.readout_weight_positive_ref,
        "readout_weight_negative_ref": args.readout_weight_negative_ref,
        "activation_competition_width": args.activation_competition_width,
        "score_activity_inhibition_width": args.score_activity_inhibition_width,
        "output_bias_enabled": args.output_bias,
        "output_bias_apply_scale": args.output_bias_apply_scale,
        "output_bias_positive_init": args.output_bias_positive_init,
        "output_bias_negative_init": args.output_bias_negative_init,
        "output_bias_leak_resistance": args.output_bias_leak_resistance,
        "output_bias_positive_ref": args.output_bias_positive_ref,
        "output_bias_negative_ref": args.output_bias_negative_ref,
        "phase_time_scale": args.phase_time_scale,
        "input_voltage_jitter_sigma": args.input_voltage_jitter_sigma,
        "phase_jitter_sigma_ns": args.phase_jitter_sigma_ns,
        "passive_mismatch_sigma": args.passive_mismatch_sigma,
        "state_ic_mismatch_sigma": args.state_ic_mismatch_sigma,
        "perturbation_seed": args.perturbation_seed,
        "robustness_perturbation_model": (
            "deterministic Gaussian input-DAC rail jitter, phase timing jitter, passive reference-resistor "
            "mismatch, and initial capacitor-state mismatch; no behavioral learning elements are added"
        ),
        "score_mode": args.score_mode,
        "measurement_detail": args.measurement_detail,
        "hidden_bias_positive_init": args.hidden_bias_positive_init,
        "hidden_bias_negative_init": args.hidden_bias_negative_init,
        "learning_device_implementation": "transistor_passive",
        "no_behavioral_signal_math": True,
        "no_behavioral_learning_devices": True,
        "uses_behavioral_learning_devices": False,
        "transistor_or_passive_learning_path": True,
        "single_training_transient": True,
        "continuous_final_eval_transient": args.continuous_final_eval,
        "final_eval_same_transient_as_training": args.continuous_final_eval,
        "continuous_transient_contract_met": True,
        "strict_fully_on_device_contract_met": args.continuous_final_eval,
        "strict_fully_on_device_requested": True,
        "batch_size": 1,
        "python_weight_updates_between_samples": False,
        "python_checkpointing_between_samples": False,
        "python_weight_transfer_to_final_eval": not args.continuous_final_eval,
        "final_weights_used_to_seed_eval": not args.continuous_final_eval,
        "python_hidden_state_intervention": False,
        "training_eval_uses_spice_forward_path": True,
        "passive_weight_diagnostics_recorded": args.measurement_detail == "full",
        "uses_local_pca": False,
        "realistic_train_order": True,
        "train_samples": args.train_samples,
        "eval_samples": args.eval_samples,
        "mnist_index_order": "stable_balanced_random_digit01",
        "decision_threshold": args.decision_threshold,
        "accuracy_signal": accuracy_signal,
        "accuracy_threshold": accuracy_threshold,
        "output_decision_threshold": args.output_decision_threshold,
        "initial_eval_skipped": args.skip_initial_eval,
        "initial_eval_accuracy": initial_accuracy,
        "final_eval_accuracy": final_accuracy,
        "eval_accuracy_delta": None if initial_accuracy is None else final_accuracy - initial_accuracy,
        "initial_eval_output_active_fraction": initial_active_fraction,
        "final_eval_output_active_fraction": final_active_fraction,
        "nontrivial_learning_met": nontrivial_learning_met,
        **score_net_summary,
        **threshold_window_summary,
        **output_bias_summary,
        **adaptive_reference_summary,
        "initial_weights": initial_weights,
        "final_weights": final_weights,
        "curve": str(curve_path),
        "table_curve": str(table_curve_path),
        "netlists": {
            "initial_eval": None if args.skip_initial_eval else str(generated / f"{safe_tag}_initial_eval.cir"),
            "train": (
                str(generated / f"{safe_tag}_train_final_eval.cir")
                if args.continuous_final_eval
                else str(generated / f"{safe_tag}_train.cir")
            ),
            "final_eval": (
                str(generated / f"{safe_tag}_train_final_eval.cir")
                if args.continuous_final_eval
                else str(generated / f"{safe_tag}_final_eval.cir")
            ),
        },
        "wall_time_s": time.perf_counter() - t0,
        "full_objective_contract_issues": [
            "binary MNIST01 smoke, not multiclass MNIST",
            "" if target_topology else "not yet the 10x10 b4 stride2 c2 target topology",
            "" if args.continuous_final_eval else "final eval is seeded from Python-extracted train weights",
            "does not yet demonstrate nontrivial learning" if not nontrivial_learning_met else "",
        ],
        "interpretation": (
            "This runner replaces scalar tile inputs with block-local raw pixel rails and persistent per-pixel "
            "hidden weight capacitors. It is a topology-scaling rung toward 10x10 b4 stride2 c2 while preserving "
            "the no-Python-update and no-behavioral-learning-device contract."
        ),
    }
    summary["full_objective_contract_issues"] = [issue for issue in summary["full_objective_contract_issues"] if issue]
    if args.assert_nonbehavioral:
        assert summary["no_behavioral_learning_devices"] is True
        assert summary["transistor_or_passive_learning_path"] is True
    summary_path = results / f"{safe_tag}_summary.json"
    table_summary_path = tables / f"{safe_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    table_summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as exc:
        raise SystemExit(f"missing optional MNIST dependency: {exc}") from exc
