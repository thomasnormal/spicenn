from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_device_mnist01_quad_training import repeated_phases, sample_wave
from run_device_mnist01_scalar_training import balanced_digit_indices, binary_accuracy, sanitize_tag
from run_device_sequential_training import expected_positive, mos_models, output_driver_line, run_netlist
from run_spice_mnist_local_block_batch_op_train import block_indices
from run_spice_sweep import ROOT, detect_spice, run_tiny_test
from spicenn.timing import CYCLE_NS


INPUT_RAIL_MODES = ("raw", "complement", "alternating-complement")
TARGET_POLARITIES = ("active-high", "active-low")


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
) -> dict[str, Any]:
    blocks, feature_count = block_topology(image_size, block_size, stride, channels)
    block_len = len(blocks[0])
    rng = np.random.default_rng(seed)
    whp = np.clip(0.72 + rng.normal(0.0, 0.035, size=(feature_count, block_len)), 0.50, 0.92)
    whn = np.clip(0.22 + rng.normal(0.0, 0.025, size=(feature_count, block_len)), 0.05, 0.42)
    bhp = np.clip(hidden_bias_positive_init + rng.normal(0.0, 0.025, size=feature_count), 0.05, 1.15)
    bhn = np.clip(hidden_bias_negative_init + rng.normal(0.0, 0.020, size=feature_count), 0.02, 1.10)
    vwp = np.clip(0.52 + rng.normal(0.0, 0.025, size=feature_count), 0.35, 0.75)
    vwn = np.clip(0.25 + rng.normal(0.0, 0.020, size=feature_count), 0.08, 0.45)
    return {
        "whp": whp.tolist(),
        "whn": whn.tolist(),
        "bhp": bhp.tolist(),
        "bhn": bhn.tolist(),
        "vwp": vwp.tolist(),
        "vwn": vwn.tolist(),
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


def block_netlist(
    samples: list[dict[str, Any]],
    weights: dict[str, Any],
    *,
    image_size: int,
    block_size: int,
    stride: int,
    channels: int,
    training_enabled: bool,
    output_driver_model: str = "sense",
    readout_apply_scale: float = 0.35,
    hidden_forward_width: float = 3.0,
    readout_gradient_width: float = 24.0,
    hidden_error_width: float = 32.0,
    hidden_update_width: float = 12.0,
    hidden_weight_write_width: float = 0.25,
    input_rail_mode: str = "alternating-complement",
) -> str:
    if readout_apply_scale <= 0.0:
        raise ValueError("readout_apply_scale must be positive")
    if hidden_forward_width <= 0.0:
        raise ValueError("hidden_forward_width must be positive")
    if readout_gradient_width <= 0.0:
        raise ValueError("readout_gradient_width must be positive")
    if hidden_error_width <= 0.0:
        raise ValueError("hidden_error_width must be positive")
    if hidden_update_width <= 0.0:
        raise ValueError("hidden_update_width must be positive")
    if hidden_weight_write_width <= 0.0:
        raise ValueError("hidden_weight_write_width must be positive")
    if input_rail_mode not in INPUT_RAIL_MODES:
        raise ValueError(f"input_rail_mode must be one of {INPUT_RAIL_MODES}")
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
    hidden_neg_width = max(0.5, hidden_forward_width * 0.75)
    stop = len(samples) * CYCLE_NS
    measures: list[str] = []
    prints: list[str] = []
    for idx in range(len(samples)):
        base = idx * CYCLE_NS
        measures += [
            f".meas tran score_before_{idx} FIND V(score) AT={base + 2.95:.2f}n",
            f".meas tran out_before_{idx} FIND V(out) AT={base + 2.95:.2f}n",
            f".meas tran score_error_{idx} FIND V(score) AT={base + 4.25:.2f}n",
            f".meas tran dp_after_{idx} FIND V(dp) AT={base + 5.10:.2f}n",
            f".meas tran dn_after_{idx} FIND V(dn) AT={base + 5.10:.2f}n",
            f".meas tran out_after_{idx} FIND V(out) AT={base + 15.50:.2f}n",
            f".meas tran d_out_{idx} PARAM='out_after_{idx}-out_before_{idx}'",
            f".meas tran error_net_{idx} PARAM='dp_after_{idx}-dn_after_{idx}'",
        ]
        for feature in range(feature_count):
            measures += [
                f".meas tran act{feature}_before_{idx} FIND V(act{feature}) AT={base + 2.95:.2f}n",
                f".meas tran act{feature}_after_{idx} FIND V(act{feature}) AT={base + 15.50:.2f}n",
                f".meas tran bhp{feature}_before_{idx} FIND V(bhp{feature}) AT={base + 0.60:.2f}n",
                f".meas tran bhn{feature}_before_{idx} FIND V(bhn{feature}) AT={base + 0.60:.2f}n",
                f".meas tran vwp{feature}_before_{idx} FIND V(vwp{feature}) AT={base + 0.60:.2f}n",
                f".meas tran vwn{feature}_before_{idx} FIND V(vwn{feature}) AT={base + 0.60:.2f}n",
                f".meas tran hdp{feature}_after_{idx} FIND V(hdp{feature}) AT={base + 7.10:.2f}n",
                f".meas tran hdn{feature}_after_{idx} FIND V(hdn{feature}) AT={base + 7.10:.2f}n",
                f".meas tran gbp{feature}_after_{idx} FIND V(gbp{feature}) AT={base + 9.10:.2f}n",
                f".meas tran gbn{feature}_after_{idx} FIND V(gbn{feature}) AT={base + 9.10:.2f}n",
                f".meas tran bhp{feature}_after_apply_{idx} FIND V(bhp{feature}) AT={base + 11.50:.2f}n",
                f".meas tran bhn{feature}_after_apply_{idx} FIND V(bhn{feature}) AT={base + 11.50:.2f}n",
                f".meas tran vwp{feature}_after_apply_{idx} FIND V(vwp{feature}) AT={base + 11.50:.2f}n",
                f".meas tran vwn{feature}_after_apply_{idx} FIND V(vwn{feature}) AT={base + 11.50:.2f}n",
                f".meas tran bias{feature}_signed_before_{idx} PARAM='bhp{feature}_before_{idx}-bhn{feature}_before_{idx}'",
                f".meas tran bias{feature}_signed_after_{idx} PARAM='bhp{feature}_after_apply_{idx}-bhn{feature}_after_apply_{idx}'",
                f".meas tran readout{feature}_signed_before_{idx} PARAM='vwp{feature}_before_{idx}-vwn{feature}_before_{idx}'",
                f".meas tran readout{feature}_signed_after_{idx} PARAM='vwp{feature}_after_apply_{idx}-vwn{feature}_after_apply_{idx}'",
                f".meas tran d_bias{feature}_signed_{idx} PARAM='bias{feature}_signed_after_{idx}-bias{feature}_signed_before_{idx}'",
                f".meas tran d_readout{feature}_signed_{idx} PARAM='readout{feature}_signed_after_{idx}-readout{feature}_signed_before_{idx}'",
            ]
            for pix in range(block_len):
                measures += [
                    f".meas tran whp{feature}_{pix}_after_apply_{idx} FIND V(whp{feature}_{pix}) AT={base + 11.50:.2f}n",
                    f".meas tran whn{feature}_{pix}_after_apply_{idx} FIND V(whn{feature}_{pix}) AT={base + 11.50:.2f}n",
                ]
        prints.append(f"print out_before_{idx} out_after_{idx} error_net_{idx}")

    lines = [
        "* Block/stride MNIST01 device-level training smoke.",
        f"* image={image_size} block={block_size} stride={stride} channels={channels}",
        f"* {feature_count} feature cells, each with {block_len} trainable hidden pixel weights.",
        ".param VDD=1.2",
        mos_models(),
        "Vdd vdd 0 {VDD}",
    ]
    for rail in required_rails:
        lines.append(f"V{rail} {rail} 0 {sample_wave(samples, rail, stop)}")
    lines += [
        f"Vtarget target 0 {sample_wave(samples, 'target', stop)}",
        repeated_phases(len(samples), training_enabled=training_enabled),
        "",
        "* Persistent signed hidden and readout weights.",
    ]
    for feature in range(feature_count):
        for pix in range(block_len):
            lines += [
                f"Cwhp{feature}_{pix} whp{feature}_{pix} 0 20f IC={float(weights['whp'][feature][pix]):.12g}",
                f"Cwhn{feature}_{pix} whn{feature}_{pix} 0 20f IC={float(weights['whn'][feature][pix]):.12g}",
                f"Rwhp{feature}_{pix} whp{feature}_{pix} 0 1e15",
                f"Rwhn{feature}_{pix} whn{feature}_{pix} 0 1e15",
            ]
        lines += [
            f"Cbhp{feature} bhp{feature} 0 20f IC={float(weights['bhp'][feature]):.12g}",
            f"Cbhn{feature} bhn{feature} 0 20f IC={float(weights['bhn'][feature]):.12g}",
            f"Cvwp{feature} vwp{feature} 0 20f IC={float(weights['vwp'][feature]):.12g}",
            f"Cvwn{feature} vwn{feature} 0 20f IC={float(weights['vwn'][feature]):.12g}",
            f"Rbhp{feature} bhp{feature} 0 1e15",
            f"Rbhn{feature} bhn{feature} 0 1e15",
            f"Rvwp{feature} vwp{feature} 0 1e15",
            f"Rvwn{feature} vwn{feature} 0 1e15",
        ]

    lines += [
        "",
        "* Shared output/error state.",
        "Cscore score 0 10f IC=0",
        "Cout out 0 20f IC=0",
        "Cdp dp 0 20f IC=0",
        "Cdn dn 0 20f IC=0",
        "Rscore score 0 1G",
        "Rout out 0 1G",
        "Rdp dp 0 1G",
        "Rdn dn 0 1G",
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
        "Mreset_out out rstf 0 0 NMOS W=4u L=180n",
        "Mreset_dp dp rstg 0 0 NMOS W=4u L=180n",
        "Mreset_dn dn rstg 0 0 NMOS W=4u L=180n",
    ]
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
        for pix in range(block_len):
            lines += [
                f"Mreset_ghp{feature}_{pix} ghp{feature}_{pix} rstg 0 0 NMOS W=4u L=180n",
                f"Mreset_ghn{feature}_{pix} ghn{feature}_{pix} rstg 0 0 NMOS W=4u L=180n",
            ]

    for block_idx, block in enumerate(blocks):
        for channel in range(channels):
            feature = block_idx * channels + channel
            lines += ["", f"* Feature {feature}: block {block_idx}, channel {channel}."]
            for pix, pixel_node in enumerate(block):
                input_node = input_rail_name(pixel_node, channel, input_rail_mode)
                lines += [
                    f"Mhpos{feature}_{pix}_x vdd {input_node} hp{feature}_{pix}_0 0 NMOS W={hidden_forward_width:.6g}u L=180n",
                    f"Mhpos{feature}_{pix}_w hp{feature}_{pix}_0 whp{feature}_{pix} hp{feature}_{pix}_1 0 NMOS W={hidden_forward_width:.6g}u L=180n",
                    f"Mhpos{feature}_{pix}_f hp{feature}_{pix}_1 fwd pre{feature} 0 NMOS W={hidden_forward_width:.6g}u L=180n",
                    f"Mhneg{feature}_{pix}_f pre{feature} fwd hn{feature}_{pix}_0 0 NMOS W={hidden_neg_width:.6g}u L=180n",
                    f"Mhneg{feature}_{pix}_x hn{feature}_{pix}_0 {input_node} hn{feature}_{pix}_1 0 NMOS W={hidden_neg_width:.6g}u L=180n",
                    f"Mhneg{feature}_{pix}_w hn{feature}_{pix}_1 whn{feature}_{pix} 0 0 NMOS W={hidden_neg_width:.6g}u L=180n",
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
            lines += [
                f"Mhbpos{feature}_b vdd bhp{feature} hbp{feature}_0 0 NMOS W={hidden_forward_width:.6g}u L=180n",
                f"Mhbpos{feature}_f hbp{feature}_0 fwd pre{feature} 0 NMOS W={hidden_forward_width:.6g}u L=180n",
                f"Mhbneg{feature}_f pre{feature} fwd hbn{feature}_0 0 NMOS W={hidden_neg_width:.6g}u L=180n",
                f"Mhbneg{feature}_b hbn{feature}_0 bhn{feature} 0 0 NMOS W={hidden_neg_width:.6g}u L=180n",
                f"Mrelu_h{feature} vdd pre{feature} act{feature} 0 NREL W=24u L=180n",
                f"Movpos{feature}_a vdd act{feature} op{feature}_0 0 NREL W=64u L=180n",
                f"Movpos{feature}_w op{feature}_0 vwp{feature} op{feature}_1 0 NREL W=64u L=180n",
                f"Movpos{feature}_f op{feature}_1 fwd score 0 NREL W=64u L=180n",
                f"Movneg{feature}_f score fwd on{feature}_0 0 NREL W=48u L=180n",
                f"Movneg{feature}_a on{feature}_0 act{feature} on{feature}_1 0 NREL W=48u L=180n",
                f"Movneg{feature}_w on{feature}_1 vwn{feature} 0 0 NREL W=48u L=180n",
                f"Mhdp{feature}_d0 vdd dp hdp{feature}_d0 0 NSENSE W={hidden_error_width:.6g}u L=180n",
                f"Mhdp{feature}_d1 hdp{feature}_d0 act{feature} hdp{feature}_d1 0 NREL W={hidden_error_width:.6g}u L=180n",
                f"Mhdp{feature}_d2 hdp{feature}_d1 bwd hdp{feature} 0 NMOS W={hidden_error_width:.6g}u L=180n",
                f"Mhdn{feature}_d0 vdd dn hdn{feature}_d0 0 NSENSE W={hidden_error_width:.6g}u L=180n",
                f"Mhdn{feature}_d1 hdn{feature}_d0 act{feature} hdn{feature}_d1 0 NREL W={hidden_error_width:.6g}u L=180n",
                f"Mhdn{feature}_d2 hdn{feature}_d1 bwd hdn{feature} 0 NMOS W={hidden_error_width:.6g}u L=180n",
                f"Mgvp{feature}_a vdd act{feature} gvp{feature}_a 0 NREL W={readout_gradient_width:.6g}u L=180n",
                f"Mgvp{feature}_d gvp{feature}_a dp gvp{feature}_d 0 NSENSE W={readout_gradient_width:.6g}u L=180n",
                f"Mgvp{feature}_g gvp{feature}_d acc gvp{feature} 0 NREL W={readout_gradient_width:.6g}u L=180n",
                f"Mgvn{feature}_a vdd act{feature} gvn{feature}_a 0 NREL W={readout_gradient_width:.6g}u L=180n",
                f"Mgvn{feature}_d gvn{feature}_a dn gvn{feature}_d 0 NSENSE W={readout_gradient_width:.6g}u L=180n",
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
                f"Mvwp{feature}_up_p0 vdd rgp{feature} vwp{feature}_up vdd PMOS W={readout_pmos_w:.6g}u L=180n",
                f"Mvwp{feature}_up_p1 vwp{feature}_up applyn vwp{feature} vdd PMOS W={readout_pmos_w:.6g}u L=180n",
                f"Mvwn{feature}_dn_a vwn{feature} apply vwn{feature}_dn 0 NREL W={readout_nmos_w:.6g}u L=180n",
                f"Mvwn{feature}_dn_g vwn{feature}_dn gvp{feature} 0 0 NSENSE W={readout_nmos_w:.6g}u L=180n",
                f"Mvwn{feature}_up_p0 vdd rgn{feature} vwn{feature}_up vdd PMOS W={readout_pmos_w:.6g}u L=180n",
                f"Mvwn{feature}_up_p1 vwn{feature}_up applyn vwn{feature} vdd PMOS W={readout_pmos_w:.6g}u L=180n",
                f"Mvwp{feature}_dn_a vwp{feature} apply vwp{feature}_dn 0 NREL W={readout_nmos_w:.6g}u L=180n",
                f"Mvwp{feature}_dn_g vwp{feature}_dn gvn{feature} 0 0 NSENSE W={readout_nmos_w:.6g}u L=180n",
            ]

    lines += [
        "",
        output_driver_line(output_driver_model),
        "",
        "* Shared output error from target/raw-score conductance competition.",
        "Mdp_t0 vdd target dp_t 0 NSENSE W=32u L=180n",
        "Mdp_t1 dp_t err dp 0 NSENSE W=32u L=180n",
        "Mdp_y0 dp err dp_y 0 NSENSE W=24u L=180n",
        "Mdp_y1 dp_y score 0 0 NSENSE W=24u L=180n",
        "Mdn_y0 vdd score dn_y 0 NSENSE W=32u L=180n",
        "Mdn_y1 dn_y err dn 0 NSENSE W=32u L=180n",
        "Mdn_t0 dn err dn_t 0 NSENSE W=24u L=180n",
        "Mdn_t1 dn_t target 0 0 NSENSE W=24u L=180n",
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
    sequence: str,
    required_rails: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample_idx, sample in enumerate(samples):
        positive = expected_positive(float(sample["target"]))
        row: dict[str, Any] = {
            "sequence": sequence,
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
    return {
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
    training_enabled: bool,
    timeout: float,
    sequence: str,
    output_driver_model: str,
    readout_apply_scale: float,
    hidden_forward_width: float,
    readout_gradient_width: float,
    hidden_error_width: float,
    hidden_update_width: float,
    hidden_weight_write_width: float,
    input_rail_mode: str,
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
        readout_apply_scale=readout_apply_scale,
        hidden_forward_width=hidden_forward_width,
        readout_gradient_width=readout_gradient_width,
        hidden_error_width=hidden_error_width,
        hidden_update_width=hidden_update_width,
        hidden_weight_write_width=hidden_weight_write_width,
        input_rail_mode=input_rail_mode,
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
    ap.add_argument("--readout-apply-scale", type=float, default=0.35)
    ap.add_argument("--hidden-forward-width", type=float, default=3.0)
    ap.add_argument("--readout-gradient-width", type=float, default=24.0)
    ap.add_argument("--hidden-error-width", type=float, default=32.0)
    ap.add_argument("--hidden-update-width", type=float, default=12.0)
    ap.add_argument("--hidden-weight-write-width", type=float, default=0.25)
    ap.add_argument("--input-rail-mode", choices=INPUT_RAIL_MODES, default="alternating-complement")
    ap.add_argument("--complement-rail-scale", type=float, default=0.5)
    ap.add_argument("--hidden-bias-positive-init", type=float, default=0.50)
    ap.add_argument("--hidden-bias-negative-init", type=float, default=0.20)
    ap.add_argument("--decision-threshold", type=float, default=0.10)
    ap.add_argument("--assert-nonbehavioral", action="store_true")
    args = ap.parse_args()

    if args.train_samples <= 0:
        raise ValueError("train-samples must be positive for a training smoke")
    if args.eval_samples <= 0:
        raise ValueError("eval-samples must be positive")
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
    initial_weights = initial_block_weights(
        args.image_size,
        args.block_size,
        args.stride,
        args.channels,
        seed=args.weight_seed,
        hidden_bias_positive_init=args.hidden_bias_positive_init,
        hidden_bias_negative_init=args.hidden_bias_negative_init,
    )
    common = {
        "image_size": args.image_size,
        "block_size": args.block_size,
        "stride": args.stride,
        "channels": args.channels,
        "timeout": args.timeout,
        "output_driver_model": args.output_driver_model,
        "readout_apply_scale": args.readout_apply_scale,
        "hidden_forward_width": args.hidden_forward_width,
        "readout_gradient_width": args.readout_gradient_width,
        "hidden_error_width": args.hidden_error_width,
        "hidden_update_width": args.hidden_update_width,
        "hidden_weight_write_width": args.hidden_weight_write_width,
        "input_rail_mode": args.input_rail_mode,
    }
    initial_eval_rows = run_device_sequence(
        spice_bin,
        generated / f"{safe_tag}_initial_eval.cir",
        eval_samples,
        initial_weights,
        training_enabled=False,
        sequence="initial_eval",
        **common,
    )
    train_rows = run_device_sequence(
        spice_bin,
        generated / f"{safe_tag}_train.cir",
        train_samples,
        initial_weights,
        training_enabled=True,
        sequence="train",
        **common,
    )
    final_weights = final_weights_from_rows(train_rows, feature_count=feature_count, block_len=block_len)
    final_eval_rows = run_device_sequence(
        spice_bin,
        generated / f"{safe_tag}_final_eval.cir",
        eval_samples,
        final_weights,
        training_enabled=False,
        sequence="final_eval",
        **common,
    )

    curve = pd.concat([initial_eval_rows, train_rows, final_eval_rows], ignore_index=True)
    curve_path = results / f"{safe_tag}.csv"
    table_curve_path = tables / f"{safe_tag}.csv"
    curve.to_csv(curve_path, index=False)
    curve.to_csv(table_curve_path, index=False)

    output_positive_when = "high" if args.target_polarity == "active-high" else "low"
    initial_accuracy = binary_accuracy(
        initial_eval_rows, threshold=args.decision_threshold, output_positive_when=output_positive_when
    )
    final_accuracy = binary_accuracy(
        final_eval_rows, threshold=args.decision_threshold, output_positive_when=output_positive_when
    )
    initial_active_fraction = float(
        np.mean(np.abs(initial_eval_rows["out_after"].to_numpy(dtype=float)) > args.decision_threshold)
    )
    final_active_fraction = float(
        np.mean(np.abs(final_eval_rows["out_after"].to_numpy(dtype=float)) > args.decision_threshold)
    )
    nontrivial_learning_met = final_accuracy > max(initial_accuracy, 0.5)
    target_topology = args.image_size == 10 and args.block_size == 4 and args.stride == 2 and args.channels == 2
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
        "hidden_credit_mode": "direct_feedback",
        "output_driver_model": args.output_driver_model,
        "readout_apply_scale": args.readout_apply_scale,
        "hidden_forward_width": args.hidden_forward_width,
        "readout_gradient_width": args.readout_gradient_width,
        "hidden_error_width": args.hidden_error_width,
        "hidden_update_width": args.hidden_update_width,
        "hidden_weight_write_width": args.hidden_weight_write_width,
        "hidden_bias_positive_init": args.hidden_bias_positive_init,
        "hidden_bias_negative_init": args.hidden_bias_negative_init,
        "learning_device_implementation": "transistor_passive",
        "no_behavioral_signal_math": True,
        "no_behavioral_learning_devices": True,
        "uses_behavioral_learning_devices": False,
        "transistor_or_passive_learning_path": True,
        "single_training_transient": True,
        "continuous_transient_contract_met": True,
        "strict_fully_on_device_contract_met": True,
        "strict_fully_on_device_requested": True,
        "batch_size": 1,
        "python_weight_updates_between_samples": False,
        "python_checkpointing_between_samples": False,
        "python_hidden_state_intervention": False,
        "training_eval_uses_spice_forward_path": True,
        "uses_local_pca": False,
        "realistic_train_order": True,
        "train_samples": args.train_samples,
        "eval_samples": args.eval_samples,
        "mnist_index_order": "stable_balanced_random_digit01",
        "decision_threshold": args.decision_threshold,
        "initial_eval_accuracy": initial_accuracy,
        "final_eval_accuracy": final_accuracy,
        "eval_accuracy_delta": final_accuracy - initial_accuracy,
        "initial_eval_output_active_fraction": initial_active_fraction,
        "final_eval_output_active_fraction": final_active_fraction,
        "nontrivial_learning_met": nontrivial_learning_met,
        "initial_weights": initial_weights,
        "final_weights": final_weights,
        "curve": str(curve_path),
        "table_curve": str(table_curve_path),
        "netlists": {
            "initial_eval": str(generated / f"{safe_tag}_initial_eval.cir"),
            "train": str(generated / f"{safe_tag}_train.cir"),
            "final_eval": str(generated / f"{safe_tag}_final_eval.cir"),
        },
        "wall_time_s": time.perf_counter() - t0,
        "full_objective_contract_issues": [
            "binary MNIST01 smoke, not multiclass MNIST",
            "" if target_topology else "not yet the 10x10 b4 stride2 c2 target topology",
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
