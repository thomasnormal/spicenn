from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_device_multicell_classifier import mos_models, pulse_wave, pwl
from run_device_xor2_learned_features import CYCLE_NS, OUTPUTS, input_value, target_wave, xor_label
from run_spice_sweep import ROOT, detect_spice, run_tiny_test


MEAS_RE = re.compile(r"(?im)^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([-+0-9.eE]+)")
HIDDEN = 8
INPUT_RAILS = ["x0", "nx0", "x1", "nx1"]
HIDDEN_RAILS = ["bias", *INPUT_RAILS]
DATASET_EXAMPLES = [
    "xor2",
    "moons8",
    "moons12",
    "mnist01_8",
    "mnist01_12",
    "mnist01pool16_12",
    "mnist01fixed16_12",
    "mnist01fixed32_12",
    "mnist01sensory64_12",
    "mnist01rand16_12",
]


@dataclass(frozen=True)
class SynapseDesign:
    name: str
    description: str
    output_forward_style: str
    hidden_forward_width_u: float
    output_forward_pos_width_u: float
    output_forward_neg_width_u: float
    hidden_relu_width_u: float
    output_relu_width_u: float
    hidden_delta_width_u: float
    readout_gradient_width_u: float
    hidden_gradient_width_u: float
    output_bias_forward_pos_width_u: float
    output_bias_forward_neg_width_u: float


SYNAPSE_DESIGNS: dict[str, SynapseDesign] = {
    "split_signed_v1": SynapseDesign(
        name="split_signed_v1",
        description=(
            "Differential positive/negative weight capacitors drive separate MOS conductance paths; "
            "the same readout weight capacitor nodes can also gate the hidden-delta path."
        ),
        output_forward_style="gate_stack",
        hidden_forward_width_u=24.0,
        output_forward_pos_width_u=56.0,
        output_forward_neg_width_u=48.0,
        hidden_relu_width_u=24.0,
        output_relu_width_u=24.0,
        hidden_delta_width_u=32.0,
        readout_gradient_width_u=24.0,
        hidden_gradient_width_u=40.0,
        output_bias_forward_pos_width_u=40.0,
        output_bias_forward_neg_width_u=36.0,
    ),
    "split_signed_passact_v1": SynapseDesign(
        name="split_signed_passact_v1",
        description=(
            "Readout positive branch uses the hidden activation voltage as the source through a weight-gated "
            "pass device; negative branch remains a signed discharge stack. This tests a more voltage-mode "
            "readout while keeping capacitor-held signed weights and backprop weight transport."
        ),
        output_forward_style="pass_act_source",
        hidden_forward_width_u=24.0,
        output_forward_pos_width_u=56.0,
        output_forward_neg_width_u=48.0,
        hidden_relu_width_u=24.0,
        output_relu_width_u=24.0,
        hidden_delta_width_u=32.0,
        readout_gradient_width_u=24.0,
        hidden_gradient_width_u=40.0,
        output_bias_forward_pos_width_u=40.0,
        output_bias_forward_neg_width_u=36.0,
    ),
}

HIDDEN_ERROR_RULES = ["backprop", "dfa"]
HIDDEN_DELTA_RELU_GATES = ["act_nrel", "act_nsense", "none"]
HIDDEN_DELTA_WEIGHT_DEVICES = ["nmos", "nrel", "nsense"]
HIDDEN_DELTA_OUTPUT_MODES = ["raw", "senseamp"]
HIDDEN_GRADIENT_ACT_GATES = ["act_nrel", "act_nsense", "none"]
HIDDEN_APPLY_MODES = ["direct", "grad_senseamp"]
HIDDEN_FORWARD_MODES = ["weighted_relu", "rail_buffer"]
OUTPUT_HEAD_MODES = ["source_follower", "score_diff"]
LEARNING_MODES = ["accumulate_apply", "flow"]
FLOW_HIDDEN_WRITES = ["direct", "off"]
FLOW_PRE_STORES = ["shared_node", "synapse_gate", "synapse_consume"]
READOUT_FLOW_POLARITIES = ["normal", "reversed"]
READOUT_FLOW_WRITE_MODES = ["discharge", "charge_discharge"]
HIDDEN_INIT_MODES = ["random", "input_identity"]
MEASURE_DETAILS = ["full", "probe", "light"]
BACKWARD_GATE_MODES = [
    "scheduled",
    "lead_or",
    "target_mistake",
    "target_mistake_latch",
    "target_out_mistake_latch",
]
CAP_DITHER_SCOPES = ["none", "hidden", "readout", "all"]
TRAIN_CHARGE_NOISE_SCOPES = CAP_DITHER_SCOPES
DIFFERENTIAL_SEPARATOR_INITS = {"separator", "csv_separator"}
RECTIFIED_SEPARATOR_INITS = {"rectified_separator", "csv_rectified_separator"}
THRESHOLD_SEPARATOR_INITS = {"threshold_separator", "csv_threshold_separator"}
SEPARATOR_READOUT_INITS = DIFFERENTIAL_SEPARATOR_INITS | RECTIFIED_SEPARATOR_INITS | THRESHOLD_SEPARATOR_INITS


def offset_key(offset_ns: float) -> str:
    return f"{int(round(offset_ns * 1000)):04d}ps"


def parse_offsets(text: str) -> list[float]:
    offsets = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not offsets:
        raise ValueError("at least one readout sample offset is required.")
    if len(set(round(offset, 6) for offset in offsets)) != len(offsets):
        raise ValueError("readout sample offsets must be unique.")
    for offset in offsets:
        if not 0.5 <= offset <= 5.8:
            raise ValueError("readout sample offsets must stay in the forward/compare window, 0.5..5.8 ns.")
    return offsets


def set_hidden_cells(count: int) -> None:
    global HIDDEN
    if count <= 0:
        raise ValueError("hidden cell count must be positive.")
    HIDDEN = count


def set_input_rails(rails: list[str]) -> None:
    global INPUT_RAILS, HIDDEN_RAILS
    if not rails:
        raise ValueError("at least one input rail is required.")
    if len(set(rails)) != len(rails):
        raise ValueError(f"input rail names must be unique: {rails}")
    for rail in rails:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", rail):
            raise ValueError(f"input rail name is not SPICE-safe: {rail!r}")
        if rail == "bias":
            raise ValueError("input rail name 'bias' is reserved.")
    INPUT_RAILS = list(rails)
    HIDDEN_RAILS = ["bias", *INPUT_RAILS]


def scaled_synapse_design(
    name: str,
    hidden_delta_width_scale: float,
    hidden_gradient_width_scale: float,
    readout_gradient_width_scale: float,
    output_forward_width_scale: float = 1.0,
    output_bias_forward_width_scale: float = 1.0,
    output_relu_width_scale: float = 1.0,
) -> SynapseDesign:
    base = SYNAPSE_DESIGNS[name]
    return replace(
        base,
        hidden_delta_width_u=base.hidden_delta_width_u * hidden_delta_width_scale,
        hidden_gradient_width_u=base.hidden_gradient_width_u * hidden_gradient_width_scale,
        readout_gradient_width_u=base.readout_gradient_width_u * readout_gradient_width_scale,
        output_forward_pos_width_u=base.output_forward_pos_width_u * output_forward_width_scale,
        output_forward_neg_width_u=base.output_forward_neg_width_u * output_forward_width_scale,
        output_bias_forward_pos_width_u=base.output_bias_forward_pos_width_u * output_bias_forward_width_scale,
        output_bias_forward_neg_width_u=base.output_bias_forward_neg_width_u * output_bias_forward_width_scale,
        output_relu_width_u=base.output_relu_width_u * output_relu_width_scale,
    )


def parse_measures(text: str) -> dict[str, float]:
    return {name.lower(): float(value) for name, value in MEAS_RE.findall(text)}


def two_moons_records(sample_count: int, seed: int) -> list[dict[str, Any]]:
    if sample_count % 2 != 0:
        raise ValueError("two-moons sample count must be even.")
    rng = np.random.default_rng(seed)
    half = sample_count // 2
    angles = np.linspace(0.15 * np.pi, 0.95 * np.pi, half)
    x0 = np.c_[np.cos(angles), np.sin(angles)]
    x1 = np.c_[1.0 - np.cos(angles), 1.0 - np.sin(angles) - 0.5]
    xy = np.vstack([x0, x1])
    labels = np.array([0] * half + [1] * half, dtype=int)
    xy += rng.normal(0.0, 0.035, size=xy.shape)
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    scaled = 0.08 + 0.84 * (xy - lo) / np.maximum(hi - lo, 1e-9)
    records: list[dict[str, Any]] = []
    for idx, ((x, y), label) in enumerate(zip(scaled, labels)):
        records.append(
            {
                "pattern": idx,
                "label": int(label),
                "inputs": {
                    "x0": float(x),
                    "nx0": float(1.0 - x),
                    "x1": float(y),
                    "nx1": float(1.0 - y),
                },
            }
        )
    return records


def dct2_lowfreq(image: np.ndarray, side: int) -> np.ndarray:
    n = image.shape[0]
    coords = np.arange(n, dtype=np.float64)
    basis = []
    for k in range(side):
        alpha = np.sqrt(1.0 / n) if k == 0 else np.sqrt(2.0 / n)
        basis.append(alpha * np.cos(np.pi * (coords + 0.5) * k / n))
    mat = np.stack(basis)
    return mat @ image @ mat.T


def random_local_relu_features(image: np.ndarray, feature_count: int) -> np.ndarray:
    rng = np.random.default_rng(17_003 + feature_count)
    features: list[float] = []
    for feature_idx in range(feature_count):
        patch_side = int(rng.choice([3, 4, 5]))
        row0 = int(rng.integers(0, image.shape[0] - patch_side + 1))
        col0 = int(rng.integers(0, image.shape[1] - patch_side + 1))
        patch = image[row0 : row0 + patch_side, col0 : col0 + patch_side]
        weights = rng.choice([-1.0, 0.0, 1.0], size=patch.shape, p=[0.35, 0.30, 0.35])
        if not np.any(weights):
            weights[patch_side // 2, patch_side // 2] = 1.0
        response = float(np.sum(weights * patch) / max(1, np.count_nonzero(weights)))
        bias = float(rng.uniform(-0.08, 0.08))
        if feature_idx % 2:
            response = -response
        features.append(max(0.0, response - bias))
    return np.asarray(features, dtype=np.float64)


def mnist01_frontend(image: np.ndarray, frontend: str) -> tuple[np.ndarray, str]:
    if frontend == "pool2":
        return image.reshape(2, 4, 2, 4).mean(axis=(1, 3)).reshape(-1), "2x2_area_downsample"
    if frontend == "pool16":
        return image.reshape(4, 2, 4, 2).mean(axis=(1, 3)).reshape(-1), "4x4_area_downsample"
    if frontend == "fixed8":
        pooled = image.reshape(2, 4, 2, 4).mean(axis=(1, 3)).reshape(-1)
        lr = abs(float(image[:, :4].mean() - image[:, 4:].mean()))
        tb = abs(float(image[:4, :].mean() - image[4:, :].mean()))
        diag = abs(float(np.trace(image) / 8.0 - np.trace(np.fliplr(image)) / 8.0))
        center = float(image[2:6, 2:6].mean())
        return np.r_[pooled, [lr, tb, diag, center]], "2x2_pool_plus_global_haar_energy"
    if frontend == "fixed16":
        pooled = image.reshape(2, 4, 2, 4).mean(axis=(1, 3)).reshape(-1)
        features = [*pooled]
        for br in range(2):
            for bc in range(2):
                block = image[br * 4 : (br + 1) * 4, bc * 4 : (bc + 1) * 4]
                features.extend(
                    [
                        abs(float(block[:, :2].mean() - block[:, 2:].mean())),
                        abs(float(block[:2, :].mean() - block[2:, :].mean())),
                        abs(float(np.trace(block) / 4.0 - np.trace(np.fliplr(block)) / 4.0)),
                    ]
                )
        return np.asarray(features, dtype=np.float64), "2x2_pool_plus_local_haar_energy"
    if frontend == "haar16":
        features = []
        for br in range(2):
            for bc in range(2):
                block = image[br * 4 : (br + 1) * 4, bc * 4 : (bc + 1) * 4]
                features.extend(
                    [
                        float(block.mean()),
                        abs(float(block[:, :2].mean() - block[:, 2:].mean())),
                        abs(float(block[:2, :].mean() - block[2:, :].mean())),
                        abs(float(np.trace(block) / 4.0 - np.trace(np.fliplr(block)) / 4.0)),
                    ]
                )
        return np.asarray(features, dtype=np.float64), "local_haar_dc_and_energy"
    if frontend == "dct16":
        coeff = dct2_lowfreq(image, 4).reshape(-1)
        coeff[1:] = np.abs(coeff[1:])
        return coeff, "low_frequency_4x4_dct_abs_ac"
    if frontend == "fixed32":
        fixed, _fixed_desc = mnist01_frontend(image, "fixed16")
        dct, _dct_desc = mnist01_frontend(image, "dct16")
        return np.r_[fixed, dct], "local_haar_pool_plus_low_frequency_dct"
    sensory_match = re.fullmatch(r"sensory(\d+)", frontend)
    if sensory_match:
        feature_count = int(sensory_match.group(1))
        if not 32 <= feature_count <= 96:
            raise ValueError("sensory frontend feature count must be in 32..96.")
        fixed32, _fixed_desc = mnist01_frontend(image, "fixed32")
        random_count = feature_count - 32
        if random_count:
            random_features = random_local_relu_features(image, random_count)
            features = np.r_[fixed32, random_features]
            desc = f"local_haar_pool_low_frequency_dct_plus_{random_count}_random_local_relu"
        else:
            features = fixed32
            desc = "local_haar_pool_plus_low_frequency_dct"
        return features, desc
    random_match = re.fullmatch(r"rand(\d+)", frontend)
    if random_match:
        feature_count = int(random_match.group(1))
        if not 1 <= feature_count <= 64:
            raise ValueError("random local ReLU frontend feature count must be in 1..64.")
        return (
            random_local_relu_features(image, feature_count),
            f"{feature_count}_fixed_sparse_random_local_relu_filters",
        )
    raise ValueError(
        f"unknown MNIST01 frontend: {frontend}. "
        "Expected pool2, pool16, fixed8, fixed16, fixed32, haar16, dct16, sensoryN, or randN."
    )


def mnist01_records(sample_count: int, seed: int, frontend: str = "pool2") -> list[dict[str, Any]]:
    if sample_count % 2 != 0:
        raise ValueError("MNIST 0/1 sample count must be even.")
    from torch.nn import functional as F
    from torchvision import datasets, transforms

    ds = datasets.MNIST(root=str(ROOT / "data"), train=True, download=False, transform=transforms.ToTensor())
    labels_np = np.asarray(ds.targets)
    rng = np.random.default_rng(seed)
    half = sample_count // 2
    selected: list[tuple[int, int]] = []
    for digit in [0, 1]:
        candidates = np.flatnonzero(labels_np == digit)
        chosen = rng.choice(candidates, size=half, replace=False)
        selected.extend((int(idx), digit) for idx in chosen)
    raw_features: list[np.ndarray] = []
    frontend_description = ""
    for idx, _label in selected:
        x, _digit = ds[idx]
        x8 = F.interpolate(x.unsqueeze(0), size=(8, 8), mode="area").squeeze().numpy().astype(np.float64)
        features, frontend_description = mnist01_frontend(x8, frontend)
        raw_features.append(features)
    raw = np.stack(raw_features)
    lo = raw.min(axis=0)
    hi = raw.max(axis=0)
    scaled = 0.08 + 0.84 * (raw - lo) / np.maximum(hi - lo, 1e-9)
    if frontend == "pool2":
        rail_names = ["x0", "nx0", "x1", "nx1"]
    else:
        rail_names = [f"x{i}" for i in range(scaled.shape[1])]
    records: list[dict[str, Any]] = []
    for pattern, ((idx, label), features) in enumerate(zip(selected, scaled)):
        records.append(
            {
                "pattern": pattern,
                "label": int(label),
                "source_index": idx,
                "source_digit": int(label),
                "input_frontend": f"{frontend_description}_per_feature_selected_subset_minmax_to_0p08_0p92",
                "input_frontend_key": frontend,
                "input_rails": rail_names,
                "inputs": {rail: float(value) for rail, value in zip(rail_names, features)},
            }
        )
    return records


def dataset_records(name: str, seed: int) -> list[dict[str, Any]]:
    if name == "xor2":
        return [{"pattern": p, "label": xor_label(p)} for p in range(4)]
    if name.startswith("moons"):
        suffix = name.removeprefix("moons").removeprefix("_")
        if suffix.isdigit():
            return two_moons_records(int(suffix), seed)
    mnist_match = re.fullmatch(r"mnist01([a-z0-9]*)_(\d+)", name)
    if mnist_match:
        frontend = mnist_match.group(1) or "pool2"
        return mnist01_records(int(mnist_match.group(2)), seed, frontend)
    examples = ", ".join(
        DATASET_EXAMPLES
        + ["moons16", "mnist01_16", "mnist01fixed8_16", "mnist01fixed32_16", "mnist01rand8_16"]
    )
    raise ValueError(f"unknown dataset: {name}. Expected one of {examples} or another even-sized counted variant.")


def input_rails_for_records(records: list[dict[str, Any]]) -> list[str]:
    for record in records:
        inputs = record.get("inputs")
        if inputs is not None:
            rails = record.get("input_rails")
            if rails is not None:
                return [str(rail) for rail in rails]
            return [str(rail) for rail in inputs.keys()]
    return ["x0", "nx0", "x1", "nx1"]


def interleaved_order(records: list[dict[str, Any]]) -> list[int]:
    by_label: dict[int, list[int]] = {}
    for record in records:
        by_label.setdefault(int(record["label"]), []).append(int(record["pattern"]))
    labels = sorted(by_label)
    order: list[int] = []
    max_len = max(len(patterns) for patterns in by_label.values())
    for idx in range(max_len):
        for label in labels:
            patterns = by_label[label]
            if idx < len(patterns):
                order.append(patterns[idx])
    return order


def sample_input_value(sample: dict[str, Any], node: str) -> float:
    inputs = sample.get("inputs")
    if inputs is not None:
        return float(inputs[node])
    return input_value(int(sample["pattern"]), node)


def perceptron_separable(df: pd.DataFrame) -> dict[str, Any]:
    act_cols = [f"act{h}" for h in range(HIDDEN)]
    ordered = df.sort_values("pattern")
    x = ordered[act_cols].to_numpy(dtype=float)
    y = np.where(ordered["label"].to_numpy(dtype=int) == 1, 1.0, -1.0)
    xb = np.c_[x, np.ones(len(x))]
    w = np.zeros(xb.shape[1])
    best_margin = float("-inf")
    best_epoch = 0
    for epoch in range(20_000):
        errors = 0
        margins = y * (xb @ w)
        margin = float(margins.min())
        if margin > best_margin:
            best_margin = margin
            best_epoch = epoch
        for xi, yi, mi in zip(xb, y, margins):
            if mi <= 1e-9:
                w += yi * xi
                errors += 1
        if errors == 0:
            return {
                "linearly_separable": True,
                "perceptron_epochs": epoch,
                "min_margin": float((y * (xb @ w)).min()),
            }
    return {
        "linearly_separable": False,
        "perceptron_epochs": 20_000,
        "best_min_margin": best_margin,
        "best_epoch": best_epoch,
    }


def perceptron_separable_array(x: np.ndarray, y_labels: np.ndarray) -> dict[str, Any]:
    y = np.where(y_labels.astype(int) == 1, 1.0, -1.0)
    xb = np.c_[x, np.ones(len(x))]
    w = np.zeros(xb.shape[1])
    best_margin = float("-inf")
    best_epoch = 0
    for epoch in range(20_000):
        errors = 0
        margins = y * (xb @ w)
        margin = float(margins.min())
        if margin > best_margin:
            best_margin = margin
            best_epoch = epoch
        for xi, yi, mi in zip(xb, y, margins):
            if mi <= 1e-9:
                w += yi * xi
                errors += 1
        if errors == 0:
            return {
                "linearly_separable": True,
                "perceptron_epochs": epoch,
                "min_margin": float((y * (xb @ w)).min()),
            }
    return {
        "linearly_separable": False,
        "perceptron_epochs": 20_000,
        "best_min_margin": best_margin,
        "best_epoch": best_epoch,
    }


def input_feature_separability(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records or records[0].get("inputs") is None:
        return None
    rails = input_rails_for_records(records)
    x = np.asarray([[float(record["inputs"][rail]) for rail in rails] for record in records], dtype=float)
    y = np.asarray([int(record["label"]) for record in records], dtype=int)
    return {
        "input_count": len(rails),
        "input_rails": rails,
        **perceptron_separable_array(x, y),
    }


def sample_wave(samples: list[dict[str, Any]], node: str, stop_ns: float) -> str:
    points: list[tuple[float, float]] = []
    for idx, sample in enumerate(samples):
        start = idx * CYCLE_NS
        end = start + CYCLE_NS
        value = sample_input_value(sample, node)
        if idx == 0:
            points.append((0.0, value))
        else:
            points.append((start - 0.05, sample_input_value(samples[idx - 1], node)))
            points.append((start, value))
        points.append((min(stop_ns, end - 0.05), value))
    points.append((stop_ns, sample_input_value(samples[-1], node)))
    return pwl(points)


def phases(
    samples: list[dict[str, Any]],
    bwd_start_ns: float,
    apply_start_ns: float,
    apply_end_ns: float,
    cmp_start_ns: float,
    cmp_end_ns: float,
    learning_mode: str,
    backward_gate_mode: str,
) -> str:
    if learning_mode not in LEARNING_MODES:
        raise ValueError(f"unknown learning mode: {learning_mode}")
    if backward_gate_mode not in BACKWARD_GATE_MODES:
        raise ValueError(f"unknown backward gate mode: {backward_gate_mode}")
    stop = len(samples) * CYCLE_NS
    rstf: list[tuple[float, float]] = []
    rste: list[tuple[float, float]] = []
    rstg: list[tuple[float, float]] = []
    fwd: list[tuple[float, float]] = []
    cmp: list[tuple[float, float]] = []
    err: list[tuple[float, float]] = []
    bwd: list[tuple[float, float]] = []
    acc: list[tuple[float, float]] = []
    gcmp: list[tuple[float, float]] = []
    apply: list[tuple[float, float]] = []
    for idx, sample in enumerate(samples):
        base = idx * CYCLE_NS
        rstf.append((base + 0.00, base + 0.50))
        rste.append((base + 0.00, base + 0.50))
        if learning_mode == "accumulate_apply" and (
            sample["phase"] != "train" or sample.get("reset_gradient", True)
        ):
            rstg.append((base + 0.00, base + 0.50))
        fwd.append((base + 0.75, base + 3.00))
        if sample["phase"] == "train":
            cmp.append((base + cmp_start_ns, base + cmp_end_ns))
            err.append((base + 5.25, base + 6.50))
            bwd_end = apply_end_ns if learning_mode == "flow" else 8.00
            bwd.append((base + bwd_start_ns, base + bwd_end))
            if learning_mode == "accumulate_apply":
                acc.append((base + 8.25, base + 9.00))
            if sample.get("apply_update", True):
                if learning_mode == "accumulate_apply":
                    gcmp.append((base + 9.05, base + 9.20))
                    apply.append((base + apply_start_ns, base + apply_end_ns))
                elif learning_mode == "flow":
                    apply.append((base + apply_start_ns, base + apply_end_ns))
                rstf.append((base + 12.05, base + 12.55))
                fwd.append((base + 12.80, base + 15.60))
    return "\n".join(
        [
            f"Vrstf rstf 0 {pulse_wave(rstf, stop)}",
            f"Vrste rste 0 {pulse_wave(rste, stop)}",
            f"Vrstg rstg 0 {pulse_wave(rstg, stop)}",
            f"Vfwd fwd 0 {pulse_wave(fwd, stop)}",
            f"Vcmp cmp 0 {pulse_wave(cmp, stop)}",
            f"Verr err 0 {pulse_wave(err, stop)}",
            (
                f"Vbwd bwd 0 {pulse_wave(bwd, stop)}"
                if backward_gate_mode == "scheduled"
                else f"Vbwd_src bwd_src 0 {pulse_wave(bwd, stop)}"
            ),
            f"Vacc acc 0 {pulse_wave(acc, stop)}",
            f"Vgcmp gcmp 0 {pulse_wave(gcmp, stop)}",
            f"Vapply apply 0 {pulse_wave(apply, stop)}",
        ]
    )


def make_samples(records: list[dict[str, Any]], epochs: int, order: list[int], batch_apply: bool) -> list[dict[str, Any]]:
    by_pattern = {int(record["pattern"]): record for record in records}
    samples: list[dict[str, Any]] = []
    for record in records:
        samples.append({**record, "phase": "initial_eval"})
    for _ in range(epochs):
        for pos, pattern in enumerate(order):
            record = by_pattern[pattern]
            samples.append(
                {
                    **record,
                    "phase": "train",
                    "reset_gradient": (not batch_apply) or pos == 0,
                    "apply_update": (not batch_apply) or pos == len(order) - 1,
                }
            )
    for record in records:
        samples.append({**record, "phase": "final_eval"})
    return samples


def hidden_init(seed: int, mode: str) -> dict[str, float]:
    if mode not in HIDDEN_INIT_MODES:
        raise ValueError(f"unknown hidden init mode: {mode}")
    init: dict[str, float] = {}
    for h in range(HIDDEN):
        if mode == "input_identity" and h < len(INPUT_RAILS):
            passthrough_rail = INPUT_RAILS[h]
            for rail in HIDDEN_RAILS:
                init[f"wh{h}_{rail}p"] = 1.05 if rail == passthrough_rail else 0.01
                init[f"wh{h}_{rail}n"] = 0.01
            continue
        init[f"wh{h}_biasp"] = 0.90 - 0.03 * ((h + seed) % 3)
        init[f"wh{h}_biasn"] = 0.42 + 0.02 * ((h + seed) % 2)
        for rail_idx, rail in enumerate(INPUT_RAILS):
            k = h * len(INPUT_RAILS) + rail_idx + seed * 11
            p = 0.38 + 0.54 * (((37 * k + 19) % 101) / 100)
            n = 0.38 + 0.54 * (((61 * k + 7) % 101) / 100)
            if abs(p - n) < 0.08:
                p = min(0.92, p + 0.11)
            init[f"wh{h}_{rail}p"] = p
            init[f"wh{h}_{rail}n"] = n
    return init


SEPARATOR_WEIGHTS = [
    -11.146586,
    -4.6483022,
    0.9671126,
    2.7160453,
    -0.4086213,
    -2.847773,
    -1.037769,
    5.0313706,
]
SEPARATOR_BIAS = 1.0


def csv_separator_weights(path: Path, phase: str) -> tuple[list[float], float]:
    df = pd.read_csv(path)
    subset = df[df["phase"] == phase].sort_values("pattern")
    if len(subset) == 0:
        raise ValueError(f"separator CSV has no rows for phase {phase!r}: {path}")
    act_cols = [f"act{h}" for h in range(HIDDEN)]
    missing = [col for col in ["label", *act_cols] if col not in subset.columns]
    if missing:
        raise ValueError(f"separator CSV is missing columns {missing}: {path}")
    x = subset[act_cols].to_numpy(dtype=float)
    y = np.where(subset["label"].to_numpy(dtype=int) == 1, 1.0, -1.0)
    xb = np.c_[x, np.ones(len(x))]
    w = np.zeros(xb.shape[1])
    for _epoch in range(20_000):
        margins = y * (xb @ w)
        if bool((margins > 1e-9).all()):
            return [float(v) for v in w[:-1]], float(w[-1])
        for xi, yi, margin in zip(xb, y, margins):
            if margin <= 1e-9:
                w += yi * xi
    raise ValueError(f"separator CSV phase is not linearly separable after 20000 perceptron epochs: {path}")


def clamp_cap(v: float) -> float:
    return min(1.15, max(0.01, v))


def lead_class0_wins(lead_mode: str, lead01, lead10):
    if lead_mode == "out_senseamp":
        return lead10 > lead01
    return lead01 > lead10


def lead_win_gate(lead_mode: str, class_index: int) -> str:
    if class_index not in {0, 1}:
        raise ValueError(f"unknown class index: {class_index}")
    if lead_mode == "out_senseamp":
        return "lead10" if class_index == 0 else "lead01"
    if lead_mode in {"score", "lose", "senseamp"}:
        return "lead01" if class_index == 0 else "lead10"
    raise ValueError(f"unknown lead mode: {lead_mode}")


def target_mistake_gate_stats(train: pd.DataFrame, bwd_threshold_v: float = 0.5) -> dict[str, Any]:
    required = {"label", "score0_cmp", "score1_cmp", "bwd_signal"}
    if train.empty or not required.issubset(train.columns):
        return {
            "target_mistake_bwd_threshold_v": bwd_threshold_v,
            "target_mistake_bwd_match_fraction": None,
            "target_mistake_bwd_false_positive_count": None,
            "target_mistake_bwd_false_negative_count": None,
            "target_mistake_score_loses_count": None,
            "target_mistake_bwd_open_count": None,
        }
    score0_wins = train["score0_cmp"] > train["score1_cmp"]
    target_is_class0 = train["label"].astype(int) == 0
    target_loses = np.where(target_is_class0, ~score0_wins, score0_wins)
    bwd_open = train["bwd_signal"] > bwd_threshold_v
    match = bwd_open.to_numpy() == target_loses
    false_positive = bwd_open.to_numpy() & ~target_loses
    false_negative = ~bwd_open.to_numpy() & target_loses
    return {
        "target_mistake_bwd_threshold_v": bwd_threshold_v,
        "target_mistake_bwd_match_fraction": float(match.mean()),
        "target_mistake_bwd_false_positive_count": int(false_positive.sum()),
        "target_mistake_bwd_false_negative_count": int(false_negative.sum()),
        "target_mistake_score_loses_count": int(target_loses.sum()),
        "target_mistake_bwd_open_count": int(bwd_open.sum()),
    }


def readout_init(
    seed: int,
    mode: str,
    separator_scale: float,
    separator_offset_v: float,
    readout_center_v: float,
    random_center_v: float | None,
    random_span_v: float,
    separator_csv: Path | None,
    separator_phase: str,
) -> dict[str, float]:
    if mode in {"separator", "rectified_separator", "threshold_separator"} and HIDDEN != len(SEPARATOR_WEIGHTS):
        raise ValueError(
            f"{mode} is defined for {len(SEPARATOR_WEIGHTS)} hidden cells; use csv_* initialization or random."
        )
    if mode in {"separator", "rectified_separator", "threshold_separator"}:
        weights = SEPARATOR_WEIGHTS
        bias = SEPARATOR_BIAS
    elif mode in {"csv_separator", "csv_rectified_separator", "csv_threshold_separator"}:
        if separator_csv is None:
            raise ValueError(f"--readout-init {mode} requires --separator-csv.")
        weights, bias = csv_separator_weights(separator_csv, separator_phase)
    else:
        weights = []
        bias = 0.0
    if mode in DIFFERENTIAL_SEPARATOR_INITS:
        init: dict[str, float] = {}
        center = readout_center_v
        for out, sign in [(0, -1.0), (1, 1.0)]:
            output_offset = separator_offset_v if out == 0 else -separator_offset_v
            bias_diff = sign * separator_scale * bias + output_offset
            init[f"vbo{out}p"] = clamp_cap(center + bias_diff / 2)
            init[f"vbo{out}n"] = clamp_cap(center - bias_diff / 2)
            for h, weight in enumerate(weights):
                diff = sign * separator_scale * weight
                init[f"vw{out}{h}p"] = clamp_cap(center + diff / 2)
                init[f"vw{out}{h}n"] = clamp_cap(center - diff / 2)
        return init
    if mode in RECTIFIED_SEPARATOR_INITS:
        init = {}
        base = readout_center_v
        for out, sign in [(0, -1.0), (1, 1.0)]:
            output_offset = separator_offset_v if out == 0 else -separator_offset_v
            bias_diff = sign * separator_scale * bias + output_offset
            init[f"vbo{out}p"] = clamp_cap(base + max(0.0, bias_diff))
            init[f"vbo{out}n"] = clamp_cap(base + max(0.0, -bias_diff))
            for h, weight in enumerate(weights):
                diff = sign * separator_scale * weight
                init[f"vw{out}{h}p"] = clamp_cap(base + max(0.0, diff))
                init[f"vw{out}{h}n"] = clamp_cap(base + max(0.0, -diff))
        return init
    if mode in THRESHOLD_SEPARATOR_INITS:
        init = {}
        off = 0.01
        base = readout_center_v
        threshold_drive = abs(separator_offset_v)
        init["vbo0p"] = clamp_cap(base + threshold_drive)
        init["vbo0n"] = off
        init["vbo1p"] = clamp_cap(base + max(0.0, separator_scale * bias))
        init["vbo1n"] = clamp_cap(base + max(0.0, -separator_scale * bias))
        for h, weight in enumerate(weights):
            init[f"vw0{h}p"] = off
            init[f"vw0{h}n"] = off
            diff = separator_scale * weight
            init[f"vw1{h}p"] = clamp_cap(base + max(0.0, diff))
            init[f"vw1{h}n"] = clamp_cap(base + max(0.0, -diff))
        return init
    if mode != "random":
        raise ValueError(f"unknown readout init mode: {mode}")
    init: dict[str, float] = {}
    if random_center_v is not None:
        half_span = random_span_v / 2.0
        for out in range(OUTPUTS):
            kp = out + seed * 7
            kn = out + seed * 11
            init[f"vbo{out}p"] = clamp_cap(
                random_center_v + half_span * (2 * (((17 * kp + 3) % 101) / 100) - 1)
            )
            init[f"vbo{out}n"] = clamp_cap(
                random_center_v + half_span * (2 * (((23 * kn + 13) % 101) / 100) - 1)
            )
            for h in range(HIDDEN):
                k = out * HIDDEN + h + seed * 5
                p = random_center_v + half_span * (2 * (((29 * k + 5) % 101) / 100) - 1)
                n = random_center_v + half_span * (2 * (((43 * k + 17) % 101) / 100) - 1)
                init[f"vw{out}{h}p"] = clamp_cap(p)
                init[f"vw{out}{h}n"] = clamp_cap(n)
        return init
    for out in range(OUTPUTS):
        init[f"vbo{out}p"] = 0.66 - 0.02 * ((out + seed) % 2)
        init[f"vbo{out}n"] = 0.52 + 0.02 * (out % 2)
        for h in range(HIDDEN):
            k = out * HIDDEN + h + seed * 5
            p = 0.56 + 0.16 * (((29 * k + 5) % 101) / 100)
            n = 0.56 + 0.16 * (((43 * k + 17) % 101) / 100)
            init[f"vw{out}{h}p"] = p
            init[f"vw{out}{h}n"] = n
    return init


def apply_output_bias_offset(readout: dict[str, float], offset_v: float) -> dict[str, float]:
    if offset_v == 0.0:
        return readout
    adjusted = dict(readout)
    for out, sign in [(0, 1.0), (1, -1.0)]:
        adjusted[f"vbo{out}p"] = clamp_cap(adjusted[f"vbo{out}p"] + sign * offset_v / 2)
        adjusted[f"vbo{out}n"] = clamp_cap(adjusted[f"vbo{out}n"] - sign * offset_v / 2)
    return adjusted


def dither_persistent_state(
    hidden: dict[str, float],
    readout: dict[str, float],
    amplitude_v: float,
    seed: int,
    scope: str,
) -> tuple[dict[str, float], dict[str, float]]:
    if scope not in CAP_DITHER_SCOPES:
        raise ValueError(f"unknown cap dither scope: {scope}")
    if amplitude_v <= 0.0 or scope == "none":
        return hidden, readout
    rng = np.random.default_rng(seed)

    def dither(values: dict[str, float]) -> dict[str, float]:
        return {
            key: clamp_cap(value + float(rng.uniform(-amplitude_v, amplitude_v)))
            for key, value in sorted(values.items())
        }

    hidden_out = dither(hidden) if scope in {"hidden", "all"} else hidden
    readout_out = dither(readout) if scope in {"readout", "all"} else readout
    return hidden_out, readout_out


def feedback_init(seed: int, scale: float) -> dict[str, float]:
    init: dict[str, float] = {}
    center = 0.64
    for out in range(OUTPUTS):
        for h in range(HIDDEN):
            k = out * HIDDEN + h + seed * 13
            raw = (((97 * k + 23) % 101) / 50.0) - 1.0
            if abs(raw) < 0.2:
                raw = 0.2 if (k % 2 == 0) else -0.2
            diff = scale * raw
            init[f"fb{out}{h}p"] = center + diff / 2
            init[f"fb{out}{h}n"] = center - diff / 2
    return init


def signed_weight_pairs(scope: str) -> list[tuple[str, str, str]]:
    if scope not in TRAIN_CHARGE_NOISE_SCOPES:
        raise ValueError(f"unknown signed-weight scope: {scope}")
    pairs: list[tuple[str, str, str]] = []
    if scope in {"readout", "all"}:
        for out in range(OUTPUTS):
            pairs.append((f"vbo{out}", f"vbo{out}p", f"vbo{out}n"))
            for h in range(HIDDEN):
                pairs.append((f"vw{out}{h}", f"vw{out}{h}p", f"vw{out}{h}n"))
    if scope in {"hidden", "all"}:
        for h in range(HIDDEN):
            for rail in HIDDEN_RAILS:
                pairs.append((f"wh{h}_{rail}", f"wh{h}_{rail}p", f"wh{h}_{rail}n"))
    return pairs


def train_charge_noise(
    samples: list[dict[str, Any]],
    stop_ns: float,
    width_u: float,
    probability: float,
    seed: int,
    scope: str,
    pulse_width_ns: float,
    bwd_start_ns: float,
) -> str:
    if scope not in TRAIN_CHARGE_NOISE_SCOPES:
        raise ValueError(f"unknown train charge noise scope: {scope}")
    if width_u <= 0.0 or probability <= 0.0 or scope == "none":
        return "* Training-time charge noise disabled."
    if not 0.0 <= probability <= 1.0:
        raise ValueError("training charge noise probability must be in 0..1.")
    if pulse_width_ns <= 0.0:
        raise ValueError("training charge noise pulse width must be positive.")
    rng = np.random.default_rng(seed)
    pulses_by_node: dict[str, list[tuple[float, float]]] = {}
    signed_pairs = signed_weight_pairs(scope)
    for idx, sample in enumerate(samples):
        if sample["phase"] != "train":
            continue
        base = idx * CYCLE_NS
        start = base + bwd_start_ns + 0.35
        end = min(base + 11.05, start + pulse_width_ns)
        for _, pos_node, neg_node in signed_pairs:
            if float(rng.random()) >= probability:
                continue
            # Discharging the negative branch increases the signed weight;
            # discharging the positive branch decreases it.
            node = neg_node if bool(rng.integers(0, 2)) else pos_node
            pulses_by_node.setdefault(node, []).append((start, end))
    if not pulses_by_node:
        return "* Training-time charge noise sampled no active pulses."
    lines = [
        "* Training-time stochastic charge bleed. Noise is transistor-gated by bwd plus a Python-seeded random pulse rail.",
    ]
    parasitic_nodes: list[str] = []
    for node, pulses in sorted(pulses_by_node.items()):
        gate = f"nz_{node}"
        mid = f"{node}_nz_mid"
        lines += [
            f"V{gate} {gate} 0 {pulse_wave(pulses, stop_ns)}",
            f"Mnoise_{node}_b {node} bwd {mid} 0 NREL W={width_u:.12g}u L=180n",
            f"Mnoise_{node}_g {mid} {gate} 0 0 NREL W={width_u:.12g}u L=180n",
        ]
        parasitic_nodes.append(mid)
    lines += node_parasitics(*parasitic_nodes)
    return "\n".join(lines)


def persistent_caps(hidden: dict[str, float], readout: dict[str, float], cap_f: float) -> str:
    lines: list[str] = []
    for h in range(HIDDEN):
        for rail in HIDDEN_RAILS:
            lines += [
                f"Cwh{h}_{rail}p wh{h}_{rail}p 0 {cap_f:.12g}f IC={hidden[f'wh{h}_{rail}p']:.12g}",
                f"Cwh{h}_{rail}n wh{h}_{rail}n 0 {cap_f:.12g}f IC={hidden[f'wh{h}_{rail}n']:.12g}",
                f"Rwh{h}_{rail}p wh{h}_{rail}p 0 1e15",
                f"Rwh{h}_{rail}n wh{h}_{rail}n 0 1e15",
            ]
    for out in range(OUTPUTS):
        lines += [
            f"Cvbo{out}p vbo{out}p 0 {cap_f:.12g}f IC={readout[f'vbo{out}p']:.12g}",
            f"Cvbo{out}n vbo{out}n 0 {cap_f:.12g}f IC={readout[f'vbo{out}n']:.12g}",
            f"Rvbo{out}p vbo{out}p 0 1e15",
            f"Rvbo{out}n vbo{out}n 0 1e15",
        ]
        for h in range(HIDDEN):
            lines += [
                f"Cvw{out}{h}p vw{out}{h}p 0 {cap_f:.12g}f IC={readout[f'vw{out}{h}p']:.12g}",
                f"Cvw{out}{h}n vw{out}{h}n 0 {cap_f:.12g}f IC={readout[f'vw{out}{h}n']:.12g}",
                f"Rvw{out}{h}p vw{out}{h}p 0 1e15",
                f"Rvw{out}{h}n vw{out}{h}n 0 1e15",
            ]
    return "\n".join(lines)


def feedback_caps(feedback: dict[str, float], cap_f: float) -> str:
    lines: list[str] = []
    for out in range(OUTPUTS):
        for h in range(HIDDEN):
            lines += [
                f"Cfb{out}{h}p fb{out}{h}p 0 {cap_f:.12g}f IC={feedback[f'fb{out}{h}p']:.12g}",
                f"Cfb{out}{h}n fb{out}{h}n 0 {cap_f:.12g}f IC={feedback[f'fb{out}{h}n']:.12g}",
                f"Rfb{out}{h}p fb{out}{h}p 0 1e15",
                f"Rfb{out}{h}n fb{out}{h}n 0 1e15",
            ]
    return "\n".join(lines)


def temporary_caps(
    gradient_cap_f: float,
    hidden_gradient_cap_f: float,
    hidden_delta_cap_f: float,
    lead_cap_f: float,
    include_gradient_caps: bool,
    score_reset_v: float,
) -> str:
    lines: list[str] = []
    for h in range(HIDDEN):
        lines += [
            f"Cpre{h} pre{h} 0 10f IC=0",
            f"Cact{h} act{h} 0 20f IC=0",
            f"Chdp{h} hdp{h} 0 {hidden_delta_cap_f:.12g}f IC=0",
            f"Chdn{h} hdn{h} 0 {hidden_delta_cap_f:.12g}f IC=0",
            f"Rpre{h} pre{h} 0 1G",
            f"Ract{h} act{h} 0 1G",
            f"Rhdp{h} hdp{h} 0 1G",
            f"Rhdn{h} hdn{h} 0 1G",
        ]
        if include_gradient_caps:
            for rail in HIDDEN_RAILS:
                lines += [
                    f"Cghp{h}_{rail} ghp{h}_{rail} 0 {hidden_gradient_cap_f:.12g}f IC=0",
                    f"Cghn{h}_{rail} ghn{h}_{rail} 0 {hidden_gradient_cap_f:.12g}f IC=0",
                    f"Rghp{h}_{rail} ghp{h}_{rail} 0 1G",
                    f"Rghn{h}_{rail} ghn{h}_{rail} 0 1G",
                ]
    for out in range(OUTPUTS):
        lines += [
            f"Cscore{out} score{out} 0 10f IC={score_reset_v:.12g}",
            f"Cout{out} out{out} 0 20f IC=0",
            f"Cdp{out} dp{out} 0 20f IC=0",
            f"Cdn{out} dn{out} 0 20f IC=0",
            f"Rscore{out} score{out} 0 1G",
            f"Rout{out} out{out} 0 1G",
            f"Rdp{out} dp{out} 0 1G",
            f"Rdn{out} dn{out} 0 1G",
        ]
        if include_gradient_caps:
            lines += [
                f"Cgvpb{out} gvpb{out} 0 {gradient_cap_f:.12g}f IC=0",
                f"Cgvnb{out} gvnb{out} 0 {gradient_cap_f:.12g}f IC=0",
                f"Rgvpb{out} gvpb{out} 0 1G",
                f"Rgvnb{out} gvnb{out} 0 1G",
            ]
            for h in range(HIDDEN):
                lines += [
                    f"Cgvp{out}{h} gvp{out}{h} 0 {gradient_cap_f:.12g}f IC=0",
                    f"Cgvn{out}{h} gvn{out}{h} 0 {gradient_cap_f:.12g}f IC=0",
                    f"Rgvp{out}{h} gvp{out}{h} 0 1G",
                    f"Rgvn{out}{h} gvn{out}{h} 0 1G",
                ]
    lines += [
        f"Clead01 lead01 0 {lead_cap_f:.12g}f IC=0",
        f"Clead10 lead10 0 {lead_cap_f:.12g}f IC=0",
        "Rlead01 lead01 0 1G",
        "Rlead10 lead10 0 1G",
    ]
    return "\n".join(lines)


def resets(lead_mode: str, include_gradient_resets: bool, score_reset_v: float) -> str:
    lines: list[str] = []
    for h in range(HIDDEN):
        lines += [
            f"Mreset_pre{h} pre{h} rstf 0 0 NMOS W=4u L=180n",
            f"Mreset_act{h} act{h} rstf 0 0 NMOS W=4u L=180n",
            f"Mreset_hdp{h} hdp{h} rste 0 0 NMOS W=4u L=180n",
            f"Mreset_hdn{h} hdn{h} rste 0 0 NMOS W=4u L=180n",
        ]
        if include_gradient_resets:
            for rail in HIDDEN_RAILS:
                lines += [
                    f"Mreset_ghp{h}_{rail} ghp{h}_{rail} rstg 0 0 NMOS W=4u L=180n",
                    f"Mreset_ghn{h}_{rail} ghn{h}_{rail} rstg 0 0 NMOS W=4u L=180n",
                ]
    for out in range(OUTPUTS):
        lines += [
            f"Mreset_score{out} score{out} rstf scorecm 0 NMOS W=4u L=180n",
            f"Mreset_out{out} out{out} rstf 0 0 NMOS W=4u L=180n",
            f"Mreset_dp{out} dp{out} rste 0 0 NMOS W=4u L=180n",
            f"Mreset_dn{out} dn{out} rste 0 0 NMOS W=4u L=180n",
        ]
        if include_gradient_resets:
            lines += [
                f"Mreset_gvpb{out} gvpb{out} rstg 0 0 NMOS W=4u L=180n",
                f"Mreset_gvnb{out} gvnb{out} rstg 0 0 NMOS W=4u L=180n",
            ]
            for h in range(HIDDEN):
                lines += [
                    f"Mreset_gvp{out}{h} gvp{out}{h} rstg 0 0 NMOS W=4u L=180n",
                    f"Mreset_gvn{out}{h} gvn{out}{h} rstg 0 0 NMOS W=4u L=180n",
                ]
    if lead_mode in {"senseamp", "out_senseamp"}:
        lines += [
            "Mreset_lead01_high vdd rste lead01 0 NSENSE W=32u L=180n",
            "Mreset_lead10_high vdd rste lead10 0 NSENSE W=32u L=180n",
        ]
    else:
        lines += [
            "Mreset_lead01 lead01 rste 0 0 NMOS W=4u L=180n",
            "Mreset_lead10 lead10 rste 0 0 NMOS W=4u L=180n",
        ]
    return "\n".join(lines)


def hidden_forward(design: SynapseDesign, hidden_forward_mode: str) -> str:
    if hidden_forward_mode not in HIDDEN_FORWARD_MODES:
        raise ValueError(f"unknown hidden forward mode: {hidden_forward_mode}")
    lines: list[str] = []
    syn_w = design.hidden_forward_width_u
    for h in range(HIDDEN):
        if hidden_forward_mode == "rail_buffer" and h < len(INPUT_RAILS):
            rail = INPUT_RAILS[h]
            lines += [
                f"* Buffered hidden {h}: forward pass-gate copy from input rail {rail} into activation/pre caps.",
                f"Mhbuf{h}_act act{h} fwd {rail} 0 NMOS W={syn_w:.12g}u L=180n",
                f"Mhbuf{h}_pre pre{h} fwd {rail} 0 NMOS W={syn_w:.12g}u L=180n",
            ]
            continue
        lines.append(f"* General hidden {h}: fully connected signed conductance from input rails plus one bias rail.")
        for rail in HIDDEN_RAILS:
            lines += [
                f"Mh{h}_{rail}p_x vdd {rail} h{h}_{rail}p0 0 NMOS W={syn_w:.12g}u L=180n",
                f"Mh{h}_{rail}p_w h{h}_{rail}p0 wh{h}_{rail}p h{h}_{rail}p1 0 NMOS W={syn_w:.12g}u L=180n",
                f"Mh{h}_{rail}p_f h{h}_{rail}p1 fwd pre{h} 0 NMOS W={syn_w:.12g}u L=180n",
                f"Mh{h}_{rail}n_f pre{h} fwd h{h}_{rail}n0 0 NMOS W={syn_w:.12g}u L=180n",
                f"Mh{h}_{rail}n_x h{h}_{rail}n0 {rail} h{h}_{rail}n1 0 NMOS W={syn_w:.12g}u L=180n",
                f"Mh{h}_{rail}n_w h{h}_{rail}n1 wh{h}_{rail}n 0 0 NMOS W={syn_w:.12g}u L=180n",
                f"Rh{h}_{rail}p0 h{h}_{rail}p0 0 1e9",
                f"Rh{h}_{rail}p1 h{h}_{rail}p1 0 1e9",
                f"Rh{h}_{rail}n0 h{h}_{rail}n0 0 1e9",
                f"Rh{h}_{rail}n1 h{h}_{rail}n1 0 1e9",
                f"Ch{h}_{rail}p0 h{h}_{rail}p0 0 0.02f IC=0",
                f"Ch{h}_{rail}p1 h{h}_{rail}p1 0 0.02f IC=0",
                f"Ch{h}_{rail}n0 h{h}_{rail}n0 0 0.02f IC=0",
                f"Ch{h}_{rail}n1 h{h}_{rail}n1 0 0.02f IC=0",
            ]
        lines.append(f"Mrelu_h{h} vdd pre{h} act{h} 0 NREL W={design.hidden_relu_width_u:.12g}u L=180n")
    return "\n".join(lines)


def output_forward(design: SynapseDesign, output_head: str) -> str:
    if output_head not in OUTPUT_HEAD_MODES:
        raise ValueError(f"unknown output head: {output_head}")
    lines: list[str] = []
    for out in range(OUTPUTS):
        lines.append(f"* Output {out}: signed readout from all general hidden activations.")
        for h in range(HIDDEN):
            readout_internal_nodes = [
                f"o{out}{h}p0",
                f"o{out}{h}p1",
                f"o{out}{h}n0",
                f"o{out}{h}n1",
            ]
            if design.output_forward_style == "gate_stack":
                lines += [
                    f"Mo{out}{h}pos_a vdd act{h} o{out}{h}p0 0 NSENSE W={design.output_forward_pos_width_u:.12g}u L=180n",
                    f"Mo{out}{h}pos_w o{out}{h}p0 vw{out}{h}p o{out}{h}p1 0 NREL W={design.output_forward_pos_width_u:.12g}u L=180n",
                    f"Mo{out}{h}pos_f o{out}{h}p1 fwd score{out} 0 NREL W={design.output_forward_pos_width_u:.12g}u L=180n",
                    f"Mo{out}{h}neg_f score{out} fwd o{out}{h}n0 0 NREL W={design.output_forward_neg_width_u:.12g}u L=180n",
                    f"Mo{out}{h}neg_a o{out}{h}n0 act{h} o{out}{h}n1 0 NSENSE W={design.output_forward_neg_width_u:.12g}u L=180n",
                    f"Mo{out}{h}neg_w o{out}{h}n1 vw{out}{h}n 0 0 NREL W={design.output_forward_neg_width_u:.12g}u L=180n",
                ]
            elif design.output_forward_style == "pass_act_source":
                lines += [
                    f"Mo{out}{h}pos_w act{h} vw{out}{h}p o{out}{h}p1 0 NREL W={design.output_forward_pos_width_u:.12g}u L=180n",
                    f"Mo{out}{h}pos_f o{out}{h}p1 fwd score{out} 0 NREL W={design.output_forward_pos_width_u:.12g}u L=180n",
                    f"Mo{out}{h}neg_f score{out} fwd o{out}{h}n0 0 NREL W={design.output_forward_neg_width_u:.12g}u L=180n",
                    f"Mo{out}{h}neg_a o{out}{h}n0 act{h} o{out}{h}n1 0 NSENSE W={design.output_forward_neg_width_u:.12g}u L=180n",
                    f"Mo{out}{h}neg_w o{out}{h}n1 vw{out}{h}n 0 0 NREL W={design.output_forward_neg_width_u:.12g}u L=180n",
                ]
            else:
                raise ValueError(f"unknown output forward style: {design.output_forward_style}")
            lines += node_parasitics(*readout_internal_nodes)
        lines += [
            (
                f"Mo{out}bpos_a vdd bias o{out}bp0 0 NSENSE W={design.output_bias_forward_pos_width_u:.12g}u L=180n"
                if design.output_forward_style == "gate_stack"
                else f"Mo{out}bpos_src vdd vbo{out}p o{out}bp0 0 NREL W={design.output_bias_forward_pos_width_u:.12g}u L=180n"
            ),
            (
                f"Mo{out}bpos_w o{out}bp0 vbo{out}p o{out}bp1 0 NREL W={design.output_bias_forward_pos_width_u:.12g}u L=180n"
                if design.output_forward_style == "gate_stack"
                else f"Mo{out}bpos_gate o{out}bp0 bias o{out}bp1 0 NSENSE W={design.output_bias_forward_pos_width_u:.12g}u L=180n"
            ),
            f"Mo{out}bpos_f o{out}bp1 fwd score{out} 0 NREL W={design.output_bias_forward_pos_width_u:.12g}u L=180n",
            f"Mo{out}bneg_f score{out} fwd o{out}bn0 0 NREL W={design.output_bias_forward_neg_width_u:.12g}u L=180n",
            f"Mo{out}bneg_a o{out}bn0 bias o{out}bn1 0 NSENSE W={design.output_bias_forward_neg_width_u:.12g}u L=180n",
            f"Mo{out}bneg_w o{out}bn1 vbo{out}n 0 0 NREL W={design.output_bias_forward_neg_width_u:.12g}u L=180n",
        ]
        lines += node_parasitics(f"o{out}bp0", f"o{out}bp1", f"o{out}bn0", f"o{out}bn1")
        if output_head == "source_follower":
            lines.append(f"Mrelu_o{out} vdd score{out} out{out} 0 NSENSE W={design.output_relu_width_u:.12g}u L=180n")
        elif output_head == "score_diff":
            other = 1 - out
            pos_mid = f"out{out}_diff_pos"
            neg_mid = f"out{out}_diff_neg"
            lines += [
                f"* Common-mode rejecting output head: score{out} charges out{out}; score{other} discharges it.",
                f"Mout{out}_diff_pos_s vdd score{out} {pos_mid} 0 NSENSE W={design.output_relu_width_u:.12g}u L=180n",
                f"Mout{out}_diff_pos_f {pos_mid} fwd out{out} 0 NREL W={design.output_relu_width_u:.12g}u L=180n",
                f"Mout{out}_diff_neg_f out{out} fwd {neg_mid} 0 NREL W={design.output_relu_width_u:.12g}u L=180n",
                f"Mout{out}_diff_neg_s {neg_mid} score{other} 0 0 NSENSE W={design.output_relu_width_u:.12g}u L=180n",
            ]
            lines += node_parasitics(pos_mid, neg_mid)
    return "\n".join(lines)


def low_score_gate_cells(lose_pull_kohm: float, lose_width_u: float) -> str:
    lines: list[str] = []
    for out in range(OUTPUTS):
        lines += [
            f"Close{out} lose{out} 0 5f IC=0",
            f"Rlose{out}_pull lose{out} vdd {lose_pull_kohm:.12g}k",
            f"Mlose{out}_dn lose{out} score{out} 0 0 NSENSE W={lose_width_u:.12g}u L=180n",
        ]
    return "\n".join(lines)


def node_parasitics(*nodes: str) -> list[str]:
    lines: list[str] = []
    for node in nodes:
        lines += [
            f"Rpar_{node} {node} 0 1e9",
            f"Cpar_{node} {node} 0 0.02f IC=0",
        ]
    return lines


def score_lead_gate_cells(lead_width_u: float, lead_mode: str) -> str:
    if lead_mode == "score":
        return "\n".join(
            [
                "* lead01 rises when score0 conducts more strongly than score1 during compare.",
                f"Mlead01_up_s vdd score0 lead01_up 0 NSENSE W={lead_width_u:.12g}u L=180n",
                f"Mlead01_up_e lead01_up cmp lead01 0 NSENSE W={lead_width_u:.12g}u L=180n",
                f"Mlead01_dn_e lead01 cmp lead01_dn 0 NSENSE W={lead_width_u:.12g}u L=180n",
                f"Mlead01_dn_s lead01_dn score1 0 0 NSENSE W={lead_width_u:.12g}u L=180n",
                "* lead10 rises when score1 conducts more strongly than score0 during compare.",
                f"Mlead10_up_s vdd score1 lead10_up 0 NSENSE W={lead_width_u:.12g}u L=180n",
                f"Mlead10_up_e lead10_up cmp lead10 0 NSENSE W={lead_width_u:.12g}u L=180n",
                f"Mlead10_dn_e lead10 cmp lead10_dn 0 NSENSE W={lead_width_u:.12g}u L=180n",
                f"Mlead10_dn_s lead10_dn score0 0 0 NSENSE W={lead_width_u:.12g}u L=180n",
            ]
            + node_parasitics("lead01_up", "lead01_dn", "lead10_up", "lead10_dn")
        )
    if lead_mode == "lose":
        return "\n".join(
            [
                "* lead01 rises when lose1 is high and lose0 is low, i.e. score0 should lead.",
                f"Mlead01_up_s vdd lose1 lead01_up 0 NSENSE W={lead_width_u:.12g}u L=180n",
                f"Mlead01_up_e lead01_up cmp lead01 0 NSENSE W={lead_width_u:.12g}u L=180n",
                f"Mlead01_dn_e lead01 cmp lead01_dn 0 NSENSE W={lead_width_u:.12g}u L=180n",
                f"Mlead01_dn_s lead01_dn lose0 0 0 NSENSE W={lead_width_u:.12g}u L=180n",
                "* lead10 rises when lose0 is high and lose1 is low, i.e. score1 should lead.",
                f"Mlead10_up_s vdd lose0 lead10_up 0 NSENSE W={lead_width_u:.12g}u L=180n",
                f"Mlead10_up_e lead10_up cmp lead10 0 NSENSE W={lead_width_u:.12g}u L=180n",
                f"Mlead10_dn_e lead10 cmp lead10_dn 0 NSENSE W={lead_width_u:.12g}u L=180n",
                f"Mlead10_dn_s lead10_dn lose1 0 0 NSENSE W={lead_width_u:.12g}u L=180n",
            ]
            + node_parasitics("lead01_up", "lead01_dn", "lead10_up", "lead10_dn")
        )
    if lead_mode == "senseamp":
        keeper_width_u = max(1.0, lead_width_u / 64.0)
        return "\n".join(
            [
                "* Dynamic score sense amp: rste precharges both lead nodes high; cmp discharges the losing side.",
                f"Mlead01_dis_s lead01 score1 lead01_dn 0 NSENSE W={lead_width_u:.12g}u L=180n",
                f"Mlead01_dis_e lead01_dn cmp 0 0 NSENSE W={lead_width_u:.12g}u L=180n",
                f"Mlead10_dis_s lead10 score0 lead10_dn 0 NSENSE W={lead_width_u:.12g}u L=180n",
                f"Mlead10_dis_e lead10_dn cmp 0 0 NSENSE W={lead_width_u:.12g}u L=180n",
                f"Mlead01_keep lead01 lead10 vdd vdd PMOS W={keeper_width_u:.12g}u L=180n",
                f"Mlead10_keep lead10 lead01 vdd vdd PMOS W={keeper_width_u:.12g}u L=180n",
            ]
            + node_parasitics("lead01_dn", "lead10_dn")
        )
    if lead_mode == "out_senseamp":
        keeper_width_u = max(1.0, lead_width_u / 64.0)
        return "\n".join(
            [
                "* Dynamic output sense amp: rste precharges both lead nodes high; cmp discharges the losing side.",
                f"Mlead01_dis_s lead01 out0 lead01_dn 0 NSENSE W={lead_width_u:.12g}u L=180n",
                f"Mlead01_dis_e lead01_dn cmp 0 0 NSENSE W={lead_width_u:.12g}u L=180n",
                f"Mlead10_dis_s lead10 out1 lead10_dn 0 NSENSE W={lead_width_u:.12g}u L=180n",
                f"Mlead10_dis_e lead10_dn cmp 0 0 NSENSE W={lead_width_u:.12g}u L=180n",
                f"Mlead01_keep lead01 lead10 vdd vdd PMOS W={keeper_width_u:.12g}u L=180n",
                f"Mlead10_keep lead10 lead01 vdd vdd PMOS W={keeper_width_u:.12g}u L=180n",
            ]
            + node_parasitics("lead01_dn", "lead10_dn")
        )
    raise ValueError(f"unknown lead mode: {lead_mode}")


def backward_gate_cells(mode: str, width_u: float, cap_f: float, lead_mode: str = "out_senseamp") -> str:
    if mode not in BACKWARD_GATE_MODES:
        raise ValueError(f"unknown backward gate mode: {mode}")
    if mode == "scheduled":
        return "* Backward rail is driven directly by the scheduled Python guide waveform."
    if mode == "target_mistake_latch":
        target0_wins_gate = lead_win_gate(lead_mode, 0)
        target1_wins_gate = lead_win_gate(lead_mode, 1)
        return "\n".join(
            [
                "* Latched mistake-gated backward rail: target-loss events are captured during compare,",
                "* then replayed later when bwd_src opens the backward/write window.",
                f"* Target winner gates: class 0 uses {target0_wins_gate}; class 1 uses {target1_wins_gate}.",
                f"Cbwd_gate bwd 0 {cap_f:.12g}f IC=0",
                "Rbwd_gate bwd 0 1G",
                "Mreset_bwd_gate bwd rste 0 0 NMOS W=4u L=180n",
                f"Cmerr0 merr0 0 {cap_f:.12g}f IC=0",
                f"Cmerr1 merr1 0 {cap_f:.12g}f IC=0",
                "Rmerr0 merr0 0 1G",
                "Rmerr1 merr1 0 1G",
                "Mreset_merr0 merr0 rste 0 0 NMOS W=4u L=180n",
                "Mreset_merr1 merr1 rste 0 0 NMOS W=4u L=180n",
                f"Mmerr0_p vdd {target0_wins_gate} merr0_p vdd PMOS W={width_u:.12g}u L=180n",
                f"Mmerr0_t merr0_p t0 merr0_t 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mmerr0_l merr0_t {target1_wins_gate} merr0_l 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mmerr0_c merr0_l cmp merr0 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mmerr1_p vdd {target1_wins_gate} merr1_p vdd PMOS W={width_u:.12g}u L=180n",
                f"Mmerr1_t merr1_p t1 merr1_t 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mmerr1_l merr1_t {target0_wins_gate} merr1_l 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mmerr1_c merr1_l cmp merr1 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mbwd_merr0_a vdd merr0 bwd_merr0_a 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mbwd_merr0_b bwd_merr0_a bwd_src bwd 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mbwd_merr1_a vdd merr1 bwd_merr1_a 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mbwd_merr1_b bwd_merr1_a bwd_src bwd 0 NSENSE W={width_u:.12g}u L=180n",
            ]
            + node_parasitics(
                "merr0_p",
                "merr0_t",
                "merr0_l",
                "merr1_p",
                "merr1_t",
                "merr1_l",
                "bwd_merr0_a",
                "bwd_merr1_a",
            )
        )
    if mode == "target_out_mistake_latch":
        return "\n".join(
            [
                "* Output-capacitor mistake latch: captures target-low/other-high during the error window,",
                "* then uses the stored event to open the later backward/write stream.",
                f"Cbwd_gate bwd 0 {cap_f:.12g}f IC=0",
                "Rbwd_gate bwd 0 1G",
                "Mreset_bwd_gate bwd rste 0 0 NMOS W=4u L=180n",
                f"Cmerr0 merr0 0 {cap_f:.12g}f IC=0",
                f"Cmerr1 merr1 0 {cap_f:.12g}f IC=0",
                "Rmerr0 merr0 0 1G",
                "Rmerr1 merr1 0 1G",
                "Mreset_merr0 merr0 rste 0 0 NMOS W=4u L=180n",
                "Mreset_merr1 merr1 rste 0 0 NMOS W=4u L=180n",
                f"Mmerr0_p vdd out0 merr0_p vdd PMOS W={width_u:.12g}u L=180n",
                f"Mmerr0_t merr0_p t0 merr0_t 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mmerr0_o merr0_t out1 merr0_o 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mmerr0_e merr0_o err merr0 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mmerr1_p vdd out1 merr1_p vdd PMOS W={width_u:.12g}u L=180n",
                f"Mmerr1_t merr1_p t1 merr1_t 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mmerr1_o merr1_t out0 merr1_o 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mmerr1_e merr1_o err merr1 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mbwd_merr0_a vdd merr0 bwd_merr0_a 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mbwd_merr0_b bwd_merr0_a bwd_src bwd 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mbwd_merr1_a vdd merr1 bwd_merr1_a 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mbwd_merr1_b bwd_merr1_a bwd_src bwd 0 NSENSE W={width_u:.12g}u L=180n",
            ]
            + node_parasitics(
                "merr0_p",
                "merr0_t",
                "merr0_o",
                "merr1_p",
                "merr1_t",
                "merr1_o",
                "bwd_merr0_a",
                "bwd_merr1_a",
            )
        )
    if mode == "target_mistake":
        target0_wins_gate = lead_win_gate(lead_mode, 0)
        target1_wins_gate = lead_win_gate(lead_mode, 1)
        return "\n".join(
            [
                "* Mistake-gated backward rail: bwd rises only when the target class loses the output sense latch.",
                "* The PMOS inhibit requires the target's winning lead to be low, suppressing ambiguous both-high latches.",
                f"* Target winner gates: class 0 uses {target0_wins_gate}; class 1 uses {target1_wins_gate}.",
                f"Cbwd_gate bwd 0 {cap_f:.12g}f IC=0",
                "Rbwd_gate bwd 0 1G",
                "Mreset_bwd_gate bwd rste 0 0 NMOS W=4u L=180n",
                f"Mbwd_t0_p vdd {target0_wins_gate} bwd_t0_p vdd PMOS W={width_u:.12g}u L=180n",
                f"Mbwd_t0_a bwd_t0_p t0 bwd_t0_a 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mbwd_t0_l bwd_t0_a {target1_wins_gate} bwd_t0_l 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mbwd_t0_b bwd_t0_l bwd_src bwd 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mbwd_t1_p vdd {target1_wins_gate} bwd_t1_p vdd PMOS W={width_u:.12g}u L=180n",
                f"Mbwd_t1_a bwd_t1_p t1 bwd_t1_a 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mbwd_t1_l bwd_t1_a {target0_wins_gate} bwd_t1_l 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mbwd_t1_b bwd_t1_l bwd_src bwd 0 NSENSE W={width_u:.12g}u L=180n",
            ]
            + node_parasitics("bwd_t0_p", "bwd_t0_a", "bwd_t0_l", "bwd_t1_p", "bwd_t1_a", "bwd_t1_l")
        )
    return "\n".join(
        [
            "* Self-timed backward rail: bwd rises only after the scheduled window and an output lead latch.",
            f"Cbwd_gate bwd 0 {cap_f:.12g}f IC=0",
            "Rbwd_gate bwd 0 1G",
            "Mreset_bwd_gate bwd rste 0 0 NMOS W=4u L=180n",
            f"Mbwd_lead01_a vdd lead01 bwd_lead01_a 0 NSENSE W={width_u:.12g}u L=180n",
            f"Mbwd_lead01_b bwd_lead01_a bwd_src bwd 0 NSENSE W={width_u:.12g}u L=180n",
            f"Mbwd_lead10_a vdd lead10 bwd_lead10_a 0 NSENSE W={width_u:.12g}u L=180n",
            f"Mbwd_lead10_b bwd_lead10_a bwd_src bwd 0 NSENSE W={width_u:.12g}u L=180n",
        ]
        + node_parasitics("bwd_lead01_a", "bwd_lead10_a")
    )


def error_cells(
    error_rule: str,
    latch_boost_width_u: float,
    residual_target_width_u: float = 96.0,
    residual_output_width_u: float = 64.0,
) -> str:
    lines: list[str] = []
    for out in range(OUTPUTS):
        if error_rule == "score":
            lines += [
                f"Mdp{out}_t0 vdd t{out} dp{out}_t 0 NSENSE W=32u L=180n",
                f"Mdp{out}_t1 dp{out}_t err dp{out} 0 NSENSE W=32u L=180n",
                f"Mdp{out}_y0 dp{out} err dp{out}_y 0 NSENSE W=24u L=180n",
                f"Mdp{out}_y1 dp{out}_y score{out} 0 0 NSENSE W=24u L=180n",
                f"Mdn{out}_y0 vdd score{out} dn{out}_y 0 NSENSE W=32u L=180n",
                f"Mdn{out}_y1 dn{out}_y err dn{out} 0 NSENSE W=32u L=180n",
                f"Mdn{out}_t0 dn{out} err dn{out}_t 0 NSENSE W=24u L=180n",
                f"Mdn{out}_t1 dn{out}_t t{out} 0 0 NSENSE W=24u L=180n",
            ]
            lines += node_parasitics(f"dp{out}_t", f"dp{out}_y", f"dn{out}_y", f"dn{out}_t")
        elif error_rule == "perceptron":
            other = 1 - out
            lines += [
                f"Mdp{out}_t0 vdd t{out} dp{out}_t 0 NSENSE W=32u L=180n",
                f"Mdp{out}_t1 dp{out}_t err dp{out} 0 NSENSE W=32u L=180n",
                f"Mdn{out}_o0 vdd t{other} dn{out}_o 0 NSENSE W=32u L=180n",
                f"Mdn{out}_o1 dn{out}_o err dn{out} 0 NSENSE W=32u L=180n",
            ]
            lines += node_parasitics(f"dp{out}_t", f"dn{out}_o")
        elif error_rule == "margin":
            other = 1 - out
            lines += [
                f"Mdp{out}_t0 vdd t{out} dp{out}_t 0 NSENSE W=160u L=180n",
                f"Mdp{out}_o0 dp{out}_t score{other} dp{out}_o 0 NSENSE W=160u L=180n",
                f"Mdp{out}_e0 dp{out}_o err dp{out} 0 NSENSE W=160u L=180n",
                f"Mdp{out}_d0 dp{out} err dp{out}_d 0 NSENSE W=96u L=180n",
                f"Mdp{out}_d1 dp{out}_d score{out} 0 0 NSENSE W=96u L=180n",
                f"Mdn{out}_t0 vdd t{other} dn{out}_t 0 NSENSE W=160u L=180n",
                f"Mdn{out}_s0 dn{out}_t score{out} dn{out}_s 0 NSENSE W=160u L=180n",
                f"Mdn{out}_e0 dn{out}_s err dn{out} 0 NSENSE W=160u L=180n",
                f"Mdn{out}_d0 dn{out} err dn{out}_d 0 NSENSE W=96u L=180n",
                f"Mdn{out}_d1 dn{out}_d score{other} 0 0 NSENSE W=96u L=180n",
            ]
            lines += node_parasitics(
                f"dp{out}_t",
                f"dp{out}_o",
                f"dp{out}_d",
                f"dn{out}_t",
                f"dn{out}_s",
                f"dn{out}_d",
            )
        elif error_rule == "competitive":
            other = 1 - out
            lines += [
                f"Mdp{out}_t0 vdd t{out} dp{out}_t 0 NSENSE W=96u L=180n",
                f"Mdp{out}_o0 dp{out}_t score{other} dp{out}_o 0 NSENSE W=96u L=180n",
                f"Mdp{out}_e0 dp{out}_o err dp{out} 0 NSENSE W=96u L=180n",
                f"Mdn{out}_t0 vdd t{other} dn{out}_t 0 NSENSE W=96u L=180n",
                f"Mdn{out}_s0 dn{out}_t score{out} dn{out}_s 0 NSENSE W=96u L=180n",
                f"Mdn{out}_e0 dn{out}_s err dn{out} 0 NSENSE W=96u L=180n",
            ]
            lines += node_parasitics(f"dp{out}_t", f"dp{out}_o", f"dn{out}_t", f"dn{out}_s")
        elif error_rule == "out_competitive":
            other = 1 - out
            lines += [
                f"Mdp{out}_t0 vdd t{out} dp{out}_t 0 NSENSE W=96u L=180n",
                f"Mdp{out}_o0 dp{out}_t out{other} dp{out}_o 0 NSENSE W=96u L=180n",
                f"Mdp{out}_e0 dp{out}_o err dp{out} 0 NSENSE W=96u L=180n",
                f"Mdn{out}_t0 vdd t{other} dn{out}_t 0 NSENSE W=96u L=180n",
                f"Mdn{out}_s0 dn{out}_t out{out} dn{out}_s 0 NSENSE W=96u L=180n",
                f"Mdn{out}_e0 dn{out}_s err dn{out} 0 NSENSE W=96u L=180n",
            ]
            lines += node_parasitics(f"dp{out}_t", f"dp{out}_o", f"dn{out}_t", f"dn{out}_s")
        elif error_rule == "out_residual":
            tw = residual_target_width_u
            yw = residual_output_width_u
            lines += [
                f"Mdp{out}_t0 vdd t{out} dp{out}_t 0 NSENSE W={tw:.12g}u L=180n",
                f"Mdp{out}_t1 dp{out}_t err dp{out} 0 NSENSE W={tw:.12g}u L=180n",
                f"Mdp{out}_y0 dp{out} err dp{out}_y 0 NSENSE W={yw:.12g}u L=180n",
                f"Mdp{out}_y1 dp{out}_y out{out} 0 0 NSENSE W={yw:.12g}u L=180n",
                f"Mdn{out}_y0 vdd out{out} dn{out}_y 0 NSENSE W={tw:.12g}u L=180n",
                f"Mdn{out}_y1 dn{out}_y err dn{out} 0 NSENSE W={tw:.12g}u L=180n",
                f"Mdn{out}_t0 dn{out} err dn{out}_t 0 NSENSE W={yw:.12g}u L=180n",
                f"Mdn{out}_t1 dn{out}_t t{out} 0 0 NSENSE W={yw:.12g}u L=180n",
            ]
            lines += node_parasitics(f"dp{out}_t", f"dp{out}_y", f"dn{out}_y", f"dn{out}_t")
        elif error_rule == "out_competitive_latchboost":
            other = 1 - out
            other_wins_gate = "lead01" if out == 0 else "lead10"
            self_wins_gate = "lead10" if out == 0 else "lead01"
            lines += [
                f"Mdp{out}_t0 vdd t{out} dp{out}_t 0 NSENSE W=96u L=180n",
                f"Mdp{out}_o0 dp{out}_t out{other} dp{out}_o 0 NSENSE W=96u L=180n",
                f"Mdp{out}_e0 dp{out}_o err dp{out} 0 NSENSE W=96u L=180n",
                f"Mdn{out}_t0 vdd t{other} dn{out}_t 0 NSENSE W=96u L=180n",
                f"Mdn{out}_s0 dn{out}_t out{out} dn{out}_s 0 NSENSE W=96u L=180n",
                f"Mdn{out}_e0 dn{out}_s err dn{out} 0 NSENSE W=96u L=180n",
            ]
            base_nodes = [f"dp{out}_t", f"dp{out}_o", f"dn{out}_t", f"dn{out}_s"]
            if latch_boost_width_u > 0.0:
                w = latch_boost_width_u
                lines += [
                    f"Mdp{out}_bt0 vdd t{out} dp{out}_bt 0 NSENSE W={w:.12g}u L=180n",
                    f"Mdp{out}_bl0 dp{out}_bt {other_wins_gate} dp{out}_bl 0 NSENSE W={w:.12g}u L=180n",
                    f"Mdp{out}_bo0 dp{out}_bl out{other} dp{out}_bo 0 NSENSE W={w:.12g}u L=180n",
                    f"Mdp{out}_be0 dp{out}_bo err dp{out} 0 NSENSE W={w:.12g}u L=180n",
                    f"Mdn{out}_bt0 vdd t{other} dn{out}_bt 0 NSENSE W={w:.12g}u L=180n",
                    f"Mdn{out}_bl0 dn{out}_bt {self_wins_gate} dn{out}_bl 0 NSENSE W={w:.12g}u L=180n",
                    f"Mdn{out}_bs0 dn{out}_bl out{out} dn{out}_bs 0 NSENSE W={w:.12g}u L=180n",
                    f"Mdn{out}_be0 dn{out}_bs err dn{out} 0 NSENSE W={w:.12g}u L=180n",
                ]
                base_nodes += [
                    f"dp{out}_bt",
                    f"dp{out}_bl",
                    f"dp{out}_bo",
                    f"dn{out}_bt",
                    f"dn{out}_bl",
                    f"dn{out}_bs",
                ]
            lines += node_parasitics(*base_nodes)
        elif error_rule == "out_mistake":
            losing_gate = "lead10" if out == 0 else "lead01"
            winning_gate = "lead01" if out == 0 else "lead10"
            other = 1 - out
            lines += [
                f"Mdp{out}_t0 vdd t{out} dp{out}_t 0 NSENSE W=128u L=180n",
                f"Mdp{out}_l0 dp{out}_t {losing_gate} dp{out}_l 0 NSENSE W=128u L=180n",
                f"Mdp{out}_o0 dp{out}_l out{other} dp{out}_o 0 NSENSE W=128u L=180n",
                f"Mdp{out}_e0 dp{out}_o err dp{out} 0 NSENSE W=128u L=180n",
                f"Mdn{out}_t0 vdd t{other} dn{out}_t 0 NSENSE W=128u L=180n",
                f"Mdn{out}_l0 dn{out}_t {winning_gate} dn{out}_l 0 NSENSE W=128u L=180n",
                f"Mdn{out}_s0 dn{out}_l out{out} dn{out}_s 0 NSENSE W=128u L=180n",
                f"Mdn{out}_e0 dn{out}_s err dn{out} 0 NSENSE W=128u L=180n",
            ]
            lines += node_parasitics(
                f"dp{out}_t",
                f"dp{out}_l",
                f"dp{out}_o",
                f"dn{out}_t",
                f"dn{out}_l",
                f"dn{out}_s",
            )
        elif error_rule == "out_latch_mistake":
            other = 1 - out
            # In out_senseamp mode lead01 is discharged by out0 and lead10 by out1.
            # Therefore lead10 high means class 0 is winning; lead01 high means class 1 is winning.
            other_wins_gate = "lead01" if out == 0 else "lead10"
            self_wins_gate = "lead10" if out == 0 else "lead01"
            lines += [
                f"Mdp{out}_t0 vdd t{out} dp{out}_t 0 NSENSE W=128u L=180n",
                f"Mdp{out}_l0 dp{out}_t {other_wins_gate} dp{out}_l 0 NSENSE W=128u L=180n",
                f"Mdp{out}_o0 dp{out}_l out{other} dp{out}_o 0 NSENSE W=128u L=180n",
                f"Mdp{out}_e0 dp{out}_o err dp{out} 0 NSENSE W=128u L=180n",
                f"Mdn{out}_t0 vdd t{other} dn{out}_t 0 NSENSE W=128u L=180n",
                f"Mdn{out}_l0 dn{out}_t {self_wins_gate} dn{out}_l 0 NSENSE W=128u L=180n",
                f"Mdn{out}_s0 dn{out}_l out{out} dn{out}_s 0 NSENSE W=128u L=180n",
                f"Mdn{out}_e0 dn{out}_s err dn{out} 0 NSENSE W=128u L=180n",
            ]
            lines += node_parasitics(
                f"dp{out}_t",
                f"dp{out}_l",
                f"dp{out}_o",
                f"dn{out}_t",
                f"dn{out}_l",
                f"dn{out}_s",
            )
        elif error_rule == "lowtarget":
            other = 1 - out
            lines += [
                f"Mdp{out}_t0 vdd t{out} dp{out}_t 0 NSENSE W=96u L=180n",
                f"Mdp{out}_l0 dp{out}_t lose{out} dp{out}_l 0 NSENSE W=96u L=180n",
                f"Mdp{out}_e0 dp{out}_l err dp{out} 0 NSENSE W=96u L=180n",
                f"Mdn{out}_t0 vdd t{other} dn{out}_t 0 NSENSE W=96u L=180n",
                f"Mdn{out}_l0 dn{out}_t lose{other} dn{out}_l 0 NSENSE W=96u L=180n",
                f"Mdn{out}_e0 dn{out}_l err dn{out} 0 NSENSE W=96u L=180n",
            ]
            lines += node_parasitics(f"dp{out}_t", f"dp{out}_l", f"dn{out}_t", f"dn{out}_l")
        elif error_rule == "mistake":
            losing_gate = "lead10" if out == 0 else "lead01"
            winning_gate = "lead01" if out == 0 else "lead10"
            other = 1 - out
            lines += [
                f"Mdp{out}_t0 vdd t{out} dp{out}_t 0 NSENSE W=128u L=180n",
                f"Mdp{out}_l0 dp{out}_t {losing_gate} dp{out}_l 0 NSENSE W=128u L=180n",
                f"Mdp{out}_e0 dp{out}_l err dp{out} 0 NSENSE W=128u L=180n",
                f"Mdn{out}_t0 vdd t{other} dn{out}_t 0 NSENSE W=128u L=180n",
                f"Mdn{out}_l0 dn{out}_t {winning_gate} dn{out}_l 0 NSENSE W=128u L=180n",
                f"Mdn{out}_e0 dn{out}_l err dn{out} 0 NSENSE W=128u L=180n",
            ]
            lines += node_parasitics(f"dp{out}_t", f"dp{out}_l", f"dn{out}_t", f"dn{out}_l")
        elif error_rule == "local_loss":
            other = 1 - out
            lines += [
                f"Mdp{out}_t0 vdd t{out} dp{out}_t 0 NSENSE W=128u L=180n",
                f"Mdp{out}_l0 dp{out}_t lose{out} dp{out}_l 0 NSENSE W=128u L=180n",
                f"Mdp{out}_e0 dp{out}_l err dp{out} 0 NSENSE W=128u L=180n",
                f"Mdn{out}_t0 vdd t{other} dn{out}_t 0 NSENSE W=128u L=180n",
                f"Mdn{out}_s0 dn{out}_t score{out} dn{out}_s 0 NSENSE W=128u L=180n",
                f"Mdn{out}_e0 dn{out}_s err dn{out} 0 NSENSE W=128u L=180n",
            ]
            lines += node_parasitics(f"dp{out}_t", f"dp{out}_l", f"dn{out}_t", f"dn{out}_s")
        else:
            raise ValueError(f"unknown error rule: {error_rule}")
    return "\n".join(lines)


def hidden_delta(
    hidden_error_rule: str,
    hidden_delta_relu_gate: str,
    hidden_delta_weight_device: str,
    design: SynapseDesign,
    internal_cap_f: float,
    internal_leak_ohm: float,
) -> str:
    if hidden_error_rule not in HIDDEN_ERROR_RULES:
        raise ValueError(f"unknown hidden error rule: {hidden_error_rule}")
    if hidden_delta_relu_gate not in HIDDEN_DELTA_RELU_GATES:
        raise ValueError(f"unknown hidden delta ReLU gate: {hidden_delta_relu_gate}")
    if hidden_delta_weight_device not in HIDDEN_DELTA_WEIGHT_DEVICES:
        raise ValueError(f"unknown hidden delta weight device: {hidden_delta_weight_device}")
    weight_model = {
        "nmos": "NMOS",
        "nrel": "NREL",
        "nsense": "NSENSE",
    }[hidden_delta_weight_device]
    lines: list[str] = []
    for h in range(HIDDEN):
        if hidden_error_rule == "backprop":
            lines.append(
                f"* Hidden delta for general hidden {h}: backprop through capacitor-held readout weights."
            )
        else:
            lines.append(
                f"* Hidden delta for general hidden {h}: direct feedback alignment through fixed feedback caps."
            )
        for out in range(OUTPUTS):
            pos_node = f"vw{out}{h}p" if hidden_error_rule == "backprop" else f"fb{out}{h}p"
            neg_node = f"vw{out}{h}n" if hidden_error_rule == "backprop" else f"fb{out}{h}n"
            w = design.hidden_delta_width_u
            relu_model = "NSENSE" if hidden_delta_relu_gate == "act_nsense" else "NREL"
            for prefix, delta_node, weight_node, target in [
                ("p_a", f"dp{out}", pos_node, f"hdp{h}"),
                ("p_b", f"dn{out}", neg_node, f"hdp{h}"),
                ("n_a", f"dn{out}", pos_node, f"hdn{h}"),
                ("n_b", f"dp{out}", neg_node, f"hdn{h}"),
            ]:
                stem = f"hd{prefix}{h}{out}"
                n0 = f"{stem}_0"
                n1 = f"{stem}_1"
                lines += [
                    f"M{stem}_d vdd {delta_node} {n0} 0 NSENSE W={w:.12g}u L=180n",
                    f"M{stem}_w {n0} {weight_node} {n1} 0 {weight_model} W={w:.12g}u L=180n",
                ]
                internal_nodes = [n0, n1]
                if hidden_delta_relu_gate == "none":
                    lines.append(f"M{stem}_b {n1} bwd {target} 0 NMOS W={w:.12g}u L=180n")
                else:
                    n2 = f"{stem}_2"
                    internal_nodes.append(n2)
                    lines += [
                        f"M{stem}_r {n1} act{h} {n2} 0 {relu_model} W={w:.12g}u L=180n",
                        f"M{stem}_b {n2} bwd {target} 0 NMOS W={w:.12g}u L=180n",
                    ]
                for node in internal_nodes:
                    if internal_leak_ohm > 0:
                        lines.append(f"Rhdpar_{node} {node} 0 {internal_leak_ohm:.12g}")
                    if internal_cap_f > 0:
                        lines.append(f"Chdpar_{node} {node} 0 {internal_cap_f:.12g}f IC=0")
    return "\n".join(lines)


def hidden_delta_senseamps(mode: str, width_u: float, cap_f: float) -> str:
    if mode not in HIDDEN_DELTA_OUTPUT_MODES:
        raise ValueError(f"unknown hidden delta output mode: {mode}")
    if mode == "raw":
        return "* Hidden delta output mode: raw hdp/hdn nodes directly gate hidden writes."
    if width_u <= 0 or cap_f <= 0:
        raise ValueError("hidden delta sense width and capacitance must be positive.")
    keeper_width_u = max(1.0, width_u / 64.0)
    lines: list[str] = [
        "* Hidden delta output mode: local sense amps amplify hdp/hdn before the hidden write path."
    ]
    for h in range(HIDDEN):
        lines += [
            f"Chdpg{h} hdpg{h} 0 {cap_f:.12g}f IC=0",
            f"Chdng{h} hdng{h} 0 {cap_f:.12g}f IC=0",
            f"Rhdpg{h} hdpg{h} 0 1G",
            f"Rhdng{h} hdng{h} 0 1G",
            f"Mreset_hdpg{h}_high vdd rste hdpg{h} 0 NSENSE W=32u L=180n",
            f"Mreset_hdng{h}_high vdd rste hdng{h} 0 NSENSE W=32u L=180n",
            f"Mhdpg{h}_dis_s hdpg{h} hdn{h} hdpg{h}_dn 0 NSENSE W={width_u:.12g}u L=180n",
            f"Mhdpg{h}_dis_e hdpg{h}_dn bwd 0 0 NSENSE W={width_u:.12g}u L=180n",
            f"Mhdng{h}_dis_s hdng{h} hdp{h} hdng{h}_dn 0 NSENSE W={width_u:.12g}u L=180n",
            f"Mhdng{h}_dis_e hdng{h}_dn bwd 0 0 NSENSE W={width_u:.12g}u L=180n",
            f"Mhdpg{h}_keep hdpg{h} hdng{h} vdd vdd PMOS W={keeper_width_u:.12g}u L=180n",
            f"Mhdng{h}_keep hdng{h} hdpg{h} vdd vdd PMOS W={keeper_width_u:.12g}u L=180n",
        ]
        lines += node_parasitics(f"hdpg{h}_dn", f"hdng{h}_dn")
    return "\n".join(lines)


def readout_gradients_and_updates(
    readout_update_width_u: float,
    output_bias_update_width_u: float,
    design: SynapseDesign,
) -> str:
    lines: list[str] = []
    for out in range(OUTPUTS):
        grad_w = design.readout_gradient_width_u
        lines += [
            f"Mgvpb{out}_a vdd bias gvpb{out}_a 0 NREL W={grad_w:.12g}u L=180n",
            f"Mgvpb{out}_d gvpb{out}_a dp{out} gvpb{out}_d 0 NSENSE W={grad_w:.12g}u L=180n",
            f"Mgvpb{out}_g gvpb{out}_d acc gvpb{out} 0 NREL W={grad_w:.12g}u L=180n",
            f"Mgvnb{out}_a vdd bias gvnb{out}_a 0 NREL W={grad_w:.12g}u L=180n",
            f"Mgvnb{out}_d gvnb{out}_a dn{out} gvnb{out}_d 0 NSENSE W={grad_w:.12g}u L=180n",
            f"Mgvnb{out}_g gvnb{out}_d acc gvnb{out} 0 NREL W={grad_w:.12g}u L=180n",
            f"Mvbo{out}n_dn_a vbo{out}n apply vbo{out}n_dn 0 NREL W={output_bias_update_width_u:.12g}u L=180n",
            f"Mvbo{out}n_dn_g vbo{out}n_dn gvpb{out} 0 0 NSENSE W={output_bias_update_width_u:.12g}u L=180n",
            f"Mvbo{out}p_dn_a vbo{out}p apply vbo{out}p_dn 0 NREL W={output_bias_update_width_u:.12g}u L=180n",
            f"Mvbo{out}p_dn_g vbo{out}p_dn gvnb{out} 0 0 NSENSE W={output_bias_update_width_u:.12g}u L=180n",
        ]
        for h in range(HIDDEN):
            lines += [
                f"Mgvp{out}{h}_a vdd act{h} gvp{out}{h}_a 0 NREL W={grad_w:.12g}u L=180n",
                f"Mgvp{out}{h}_d gvp{out}{h}_a dp{out} gvp{out}{h}_d 0 NSENSE W={grad_w:.12g}u L=180n",
                f"Mgvp{out}{h}_g gvp{out}{h}_d acc gvp{out}{h} 0 NREL W={grad_w:.12g}u L=180n",
                f"Mgvn{out}{h}_a vdd act{h} gvn{out}{h}_a 0 NREL W={grad_w:.12g}u L=180n",
                f"Mgvn{out}{h}_d gvn{out}{h}_a dn{out} gvn{out}{h}_d 0 NSENSE W={grad_w:.12g}u L=180n",
                f"Mgvn{out}{h}_g gvn{out}{h}_d acc gvn{out}{h} 0 NREL W={grad_w:.12g}u L=180n",
                f"Mvw{out}{h}n_dn_a vw{out}{h}n apply vw{out}{h}n_dn 0 NREL W={readout_update_width_u:.12g}u L=180n",
                f"Mvw{out}{h}n_dn_g vw{out}{h}n_dn gvp{out}{h} 0 0 NSENSE W={readout_update_width_u:.12g}u L=180n",
                f"Mvw{out}{h}p_dn_a vw{out}{h}p apply vw{out}{h}p_dn 0 NREL W={readout_update_width_u:.12g}u L=180n",
                f"Mvw{out}{h}p_dn_g vw{out}{h}p_dn gvn{out}{h} 0 0 NSENSE W={readout_update_width_u:.12g}u L=180n",
            ]
    return "\n".join(lines)


def readout_flow_updates(
    readout_update_width_u: float,
    output_bias_update_width_u: float,
    flow_pre_store: str,
    readout_flow_polarity: str,
    readout_flow_write_mode: str = "discharge",
) -> str:
    if flow_pre_store not in FLOW_PRE_STORES:
        raise ValueError(f"unknown flow pre-store mode: {flow_pre_store}")
    if readout_flow_polarity not in READOUT_FLOW_POLARITIES:
        raise ValueError(f"unknown readout flow polarity: {readout_flow_polarity}")
    if readout_flow_write_mode not in READOUT_FLOW_WRITE_MODES:
        raise ValueError(f"unknown readout flow write mode: {readout_flow_write_mode}")
    if readout_update_width_u < 0 or output_bias_update_width_u < 0:
        raise ValueError("readout flow update widths must be nonnegative.")
    n_gate, p_gate = ("dp", "dn") if readout_flow_polarity == "normal" else ("dn", "dp")
    lines: list[str] = []
    for out in range(OUTPUTS):
        if output_bias_update_width_u > 0:
            lines += [
                f"Mvbo{out}n_flow_b vbo{out}n bwd vbo{out}n_flow_b 0 NREL W={output_bias_update_width_u:.12g}u L=180n",
                f"Mvbo{out}n_flow_d vbo{out}n_flow_b {n_gate}{out} 0 0 NSENSE W={output_bias_update_width_u:.12g}u L=180n",
                f"Mvbo{out}p_flow_b vbo{out}p bwd vbo{out}p_flow_b 0 NREL W={output_bias_update_width_u:.12g}u L=180n",
                f"Mvbo{out}p_flow_d vbo{out}p_flow_b {p_gate}{out} 0 0 NSENSE W={output_bias_update_width_u:.12g}u L=180n",
            ]
            lines += node_parasitics(f"vbo{out}n_flow_b", f"vbo{out}p_flow_b")
            if readout_flow_write_mode == "charge_discharge":
                lines += [
                    f"Mvbo{out}p_ch_b vdd bwd vbo{out}p_ch_b 0 NREL W={output_bias_update_width_u:.12g}u L=180n",
                    f"Mvbo{out}p_ch_d vbo{out}p_ch_b {n_gate}{out} vbo{out}p 0 NSENSE W={output_bias_update_width_u:.12g}u L=180n",
                    f"Mvbo{out}n_ch_b vdd bwd vbo{out}n_ch_b 0 NREL W={output_bias_update_width_u:.12g}u L=180n",
                    f"Mvbo{out}n_ch_d vbo{out}n_ch_b {p_gate}{out} vbo{out}n 0 NSENSE W={output_bias_update_width_u:.12g}u L=180n",
                ]
                lines += node_parasitics(f"vbo{out}p_ch_b", f"vbo{out}n_ch_b")
        for h in range(HIDDEN):
            pre_gate = f"fpro{out}{h}" if flow_pre_store != "shared_node" else f"act{h}"
            if readout_update_width_u > 0:
                lines += [
                    f"Mvw{out}{h}n_flow_b vw{out}{h}n bwd vw{out}{h}n_flow_b 0 NREL W={readout_update_width_u:.12g}u L=180n",
                    f"Mvw{out}{h}n_flow_a vw{out}{h}n_flow_b {pre_gate} vw{out}{h}n_flow_a 0 NREL W={readout_update_width_u:.12g}u L=180n",
                    f"Mvw{out}{h}n_flow_d vw{out}{h}n_flow_a {n_gate}{out} 0 0 NSENSE W={readout_update_width_u:.12g}u L=180n",
                    f"Mvw{out}{h}p_flow_b vw{out}{h}p bwd vw{out}{h}p_flow_b 0 NREL W={readout_update_width_u:.12g}u L=180n",
                    f"Mvw{out}{h}p_flow_a vw{out}{h}p_flow_b {pre_gate} vw{out}{h}p_flow_a 0 NREL W={readout_update_width_u:.12g}u L=180n",
                    f"Mvw{out}{h}p_flow_d vw{out}{h}p_flow_a {p_gate}{out} 0 0 NSENSE W={readout_update_width_u:.12g}u L=180n",
                ]
                if readout_flow_write_mode == "charge_discharge":
                    lines += [
                        f"Mvw{out}{h}p_ch_b vdd bwd vw{out}{h}p_ch_b 0 NREL W={readout_update_width_u:.12g}u L=180n",
                        f"Mvw{out}{h}p_ch_a vw{out}{h}p_ch_b {pre_gate} vw{out}{h}p_ch_a 0 NREL W={readout_update_width_u:.12g}u L=180n",
                        f"Mvw{out}{h}p_ch_d vw{out}{h}p_ch_a {n_gate}{out} vw{out}{h}p 0 NSENSE W={readout_update_width_u:.12g}u L=180n",
                        f"Mvw{out}{h}n_ch_b vdd bwd vw{out}{h}n_ch_b 0 NREL W={readout_update_width_u:.12g}u L=180n",
                        f"Mvw{out}{h}n_ch_a vw{out}{h}n_ch_b {pre_gate} vw{out}{h}n_ch_a 0 NREL W={readout_update_width_u:.12g}u L=180n",
                        f"Mvw{out}{h}n_ch_d vw{out}{h}n_ch_a {p_gate}{out} vw{out}{h}n 0 NSENSE W={readout_update_width_u:.12g}u L=180n",
                    ]
                lines += node_parasitics(
                    f"vw{out}{h}n_flow_b",
                    f"vw{out}{h}n_flow_a",
                    f"vw{out}{h}p_flow_b",
                    f"vw{out}{h}p_flow_a",
                )
                if readout_flow_write_mode == "charge_discharge":
                    lines += node_parasitics(
                        f"vw{out}{h}p_ch_b",
                        f"vw{out}{h}p_ch_a",
                        f"vw{out}{h}n_ch_b",
                        f"vw{out}{h}n_ch_a",
                    )
    return "\n".join(lines)


def flow_pre_activation_stores(
    mode: str,
    cap_f: float,
    consume_width_u: float,
) -> str:
    if mode not in FLOW_PRE_STORES:
        raise ValueError(f"unknown flow pre-store mode: {mode}")
    if mode == "shared_node":
        return "* Flow pre-activation storage: using shared source activation/input nodes."
    if cap_f <= 0 or consume_width_u <= 0:
        raise ValueError("flow pre-store capacitance and consume width must be positive.")
    lines: list[str] = [
        "* Per-synapse pre-activation traces are charged through MOS store paths during fwd for local direct-flow writes."
    ]
    consume = mode == "synapse_consume"
    for out in range(OUTPUTS):
        for h in range(HIDDEN):
            node = f"fpro{out}{h}"
            lines += [
                f"C{node} {node} 0 {cap_f:.12g}f IC=0",
                f"R{node} {node} 0 1G",
                f"Mreset_{node} {node} rstf 0 0 NMOS W=4u L=180n",
                f"Mstore_{node} {node} fwd act{h} 0 NREL W=4u L=180n",
            ]
            if consume:
                lines.append(f"Mconsume_{node} {node} bwd 0 0 NREL W={consume_width_u:.12g}u L=180n")
    for h in range(HIDDEN):
        for rail in HIDDEN_RAILS:
            node = f"fphi{h}_{rail}"
            lines += [
                f"C{node} {node} 0 {cap_f:.12g}f IC=0",
                f"R{node} {node} 0 1G",
                f"Mreset_{node} {node} rstf 0 0 NMOS W=4u L=180n",
                f"Mstore_{node} {node} fwd {rail} 0 NREL W=4u L=180n",
            ]
            if consume:
                lines.append(f"Mconsume_{node} {node} bwd 0 0 NREL W={consume_width_u:.12g}u L=180n")
    return "\n".join(lines)


def hidden_gradients_and_updates(
    update_width_u: float,
    hidden_gradient_act_gate: str,
    hidden_apply_mode: str,
    hidden_grad_sense_width_u: float,
    hidden_grad_sense_cap_f: float,
    design: SynapseDesign,
) -> str:
    if hidden_gradient_act_gate not in HIDDEN_GRADIENT_ACT_GATES:
        raise ValueError(f"unknown hidden gradient activation gate: {hidden_gradient_act_gate}")
    if hidden_apply_mode not in HIDDEN_APPLY_MODES:
        raise ValueError(f"unknown hidden apply mode: {hidden_apply_mode}")
    lines: list[str] = []
    grad_w = design.hidden_gradient_width_u
    relu_model = "NSENSE" if hidden_gradient_act_gate == "act_nsense" else "NREL"
    for h in range(HIDDEN):
        for rail in HIDDEN_RAILS:
            for sign, delta_node, grad_node in [
                ("p", f"hdp{h}", f"ghp{h}_{rail}"),
                ("n", f"hdn{h}", f"ghn{h}_{rail}"),
            ]:
                n0 = f"gh{sign}{h}_{rail}_x"
                n1 = f"gh{sign}{h}_{rail}_d"
                lines += [
                    f"Mgh{sign}{h}_{rail}_x vdd {rail} {n0} 0 NMOS W={grad_w:.12g}u L=180n",
                    f"Mgh{sign}{h}_{rail}_d {n0} {delta_node} {n1} 0 NSENSE W={grad_w:.12g}u L=180n",
                ]
                if hidden_gradient_act_gate == "none":
                    lines.append(f"Mgh{sign}{h}_{rail}_g {n1} acc {grad_node} 0 NMOS W={grad_w:.12g}u L=180n")
                    lines += node_parasitics(n0, n1)
                else:
                    n2 = f"gh{sign}{h}_{rail}_a"
                    lines += [
                        f"Mgh{sign}{h}_{rail}_a {n1} act{h} {n2} 0 {relu_model} W={grad_w:.12g}u L=180n",
                        f"Mgh{sign}{h}_{rail}_g {n2} acc {grad_node} 0 NMOS W={grad_w:.12g}u L=180n",
                    ]
                    lines += node_parasitics(n0, n1, n2)
            if hidden_apply_mode == "direct":
                pos_gate = f"ghp{h}_{rail}"
                neg_gate = f"ghn{h}_{rail}"
            else:
                pos_gate = f"hgwp{h}_{rail}"
                neg_gate = f"hgwn{h}_{rail}"
                keeper_width_u = max(1.0, hidden_grad_sense_width_u / 64.0)
                lines += [
                    f"Chgwp{h}_{rail} {pos_gate} 0 {hidden_grad_sense_cap_f:.12g}f IC=0",
                    f"Chgwn{h}_{rail} {neg_gate} 0 {hidden_grad_sense_cap_f:.12g}f IC=0",
                    f"Rhgwp{h}_{rail} {pos_gate} 0 1G",
                    f"Rhgwn{h}_{rail} {neg_gate} 0 1G",
                    f"Mreset_hgwp{h}_{rail}_high vdd rstg {pos_gate} 0 NSENSE W=32u L=180n",
                    f"Mreset_hgwn{h}_{rail}_high vdd rstg {neg_gate} 0 NSENSE W=32u L=180n",
                    f"Mhgwp{h}_{rail}_dis_s {pos_gate} ghn{h}_{rail} hgwp{h}_{rail}_dn 0 NSENSE W={hidden_grad_sense_width_u:.12g}u L=180n",
                    f"Mhgwp{h}_{rail}_dis_e hgwp{h}_{rail}_dn gcmp 0 0 NSENSE W={hidden_grad_sense_width_u:.12g}u L=180n",
                    f"Mhgwn{h}_{rail}_dis_s {neg_gate} ghp{h}_{rail} hgwn{h}_{rail}_dn 0 NSENSE W={hidden_grad_sense_width_u:.12g}u L=180n",
                    f"Mhgwn{h}_{rail}_dis_e hgwn{h}_{rail}_dn gcmp 0 0 NSENSE W={hidden_grad_sense_width_u:.12g}u L=180n",
                    f"Mhgwp{h}_{rail}_keep {pos_gate} {neg_gate} vdd vdd PMOS W={keeper_width_u:.12g}u L=180n",
                    f"Mhgwn{h}_{rail}_keep {neg_gate} {pos_gate} vdd vdd PMOS W={keeper_width_u:.12g}u L=180n",
                ]
                lines += node_parasitics(f"hgwp{h}_{rail}_dn", f"hgwn{h}_{rail}_dn")
            lines += [
                f"Mwh{h}_{rail}n_dn_a wh{h}_{rail}n apply wh{h}_{rail}n_dn 0 NREL W={update_width_u:.12g}u L=180n",
                f"Mwh{h}_{rail}n_dn_g wh{h}_{rail}n_dn {pos_gate} 0 0 NSENSE W={update_width_u:.12g}u L=180n",
                f"Mwh{h}_{rail}p_dn_a wh{h}_{rail}p apply wh{h}_{rail}p_dn 0 NREL W={update_width_u:.12g}u L=180n",
                f"Mwh{h}_{rail}p_dn_g wh{h}_{rail}p_dn {neg_gate} 0 0 NSENSE W={update_width_u:.12g}u L=180n",
            ]
    return "\n".join(lines)


def hidden_flow_updates(update_width_u: float, flow_pre_store: str, hidden_delta_output_mode: str) -> str:
    if flow_pre_store not in FLOW_PRE_STORES:
        raise ValueError(f"unknown flow pre-store mode: {flow_pre_store}")
    if hidden_delta_output_mode not in HIDDEN_DELTA_OUTPUT_MODES:
        raise ValueError(f"unknown hidden delta output mode: {hidden_delta_output_mode}")
    lines: list[str] = []
    for h in range(HIDDEN):
        pos_delta_gate = f"hdpg{h}" if hidden_delta_output_mode == "senseamp" else f"hdp{h}"
        neg_delta_gate = f"hdng{h}" if hidden_delta_output_mode == "senseamp" else f"hdn{h}"
        for rail in HIDDEN_RAILS:
            pre_gate = f"fphi{h}_{rail}" if flow_pre_store != "shared_node" else rail
            if hidden_delta_output_mode == "raw":
                lines += [
                    f"Mwh{h}_{rail}n_flow_b wh{h}_{rail}n bwd wh{h}_{rail}n_flow_b 0 NREL W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}n_flow_x wh{h}_{rail}n_flow_b {pre_gate} wh{h}_{rail}n_flow_x 0 NMOS W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}n_flow_d wh{h}_{rail}n_flow_x {pos_delta_gate} 0 0 NSENSE W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}p_flow_b wh{h}_{rail}p bwd wh{h}_{rail}p_flow_b 0 NREL W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}p_flow_x wh{h}_{rail}p_flow_b {pre_gate} wh{h}_{rail}p_flow_x 0 NMOS W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}p_flow_d wh{h}_{rail}p_flow_x {neg_delta_gate} 0 0 NSENSE W={update_width_u:.12g}u L=180n",
                ]
                lines += node_parasitics(
                    f"wh{h}_{rail}n_flow_b",
                    f"wh{h}_{rail}n_flow_x",
                    f"wh{h}_{rail}p_flow_b",
                    f"wh{h}_{rail}p_flow_x",
                )
            else:
                lines += [
                    f"Mwh{h}_{rail}n_flow_b wh{h}_{rail}n bwd wh{h}_{rail}n_flow_b 0 NREL W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}n_flow_x wh{h}_{rail}n_flow_b {pre_gate} wh{h}_{rail}n_flow_x 0 NMOS W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}n_flow_d wh{h}_{rail}n_flow_x {pos_delta_gate} wh{h}_{rail}n_flow_d 0 NSENSE W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}n_flow_a wh{h}_{rail}n_flow_d apply 0 0 NREL W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}p_flow_b wh{h}_{rail}p bwd wh{h}_{rail}p_flow_b 0 NREL W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}p_flow_x wh{h}_{rail}p_flow_b {pre_gate} wh{h}_{rail}p_flow_x 0 NMOS W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}p_flow_d wh{h}_{rail}p_flow_x {neg_delta_gate} wh{h}_{rail}p_flow_d 0 NSENSE W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}p_flow_a wh{h}_{rail}p_flow_d apply 0 0 NREL W={update_width_u:.12g}u L=180n",
                ]
                lines += node_parasitics(
                    f"wh{h}_{rail}n_flow_b",
                    f"wh{h}_{rail}n_flow_x",
                    f"wh{h}_{rail}n_flow_d",
                    f"wh{h}_{rail}p_flow_b",
                    f"wh{h}_{rail}p_flow_x",
                    f"wh{h}_{rail}p_flow_d",
                )
    return "\n".join(lines)


def measure_lines(
    samples: list[dict[str, Any]],
    hidden_apply_mode: str,
    learning_mode: str,
    hidden_delta_output_mode: str,
    measure_detail: str,
    readout_sample_offsets_ns: list[float],
    cmp_start_ns: float,
    cmp_end_ns: float,
    bwd_start_ns: float,
    apply_end_ns: float,
    backward_gate_mode: str,
    hidden_delta_network_enabled: bool = True,
) -> tuple[str, str]:
    if hidden_apply_mode not in HIDDEN_APPLY_MODES:
        raise ValueError(f"unknown hidden apply mode: {hidden_apply_mode}")
    if learning_mode not in LEARNING_MODES:
        raise ValueError(f"unknown learning mode: {learning_mode}")
    if hidden_delta_output_mode not in HIDDEN_DELTA_OUTPUT_MODES:
        raise ValueError(f"unknown hidden delta output mode: {hidden_delta_output_mode}")
    if measure_detail not in MEASURE_DETAILS:
        raise ValueError(f"unknown measurement detail level: {measure_detail}")
    if backward_gate_mode not in BACKWARD_GATE_MODES:
        raise ValueError(f"unknown backward gate mode: {backward_gate_mode}")
    if not readout_sample_offsets_ns:
        raise ValueError("at least one readout sample offset is required.")
    include_hidden_grad_measures = learning_mode == "accumulate_apply"
    include_train_detail = measure_detail == "full"
    include_signal_probe = measure_detail in {"full", "probe"}
    default_offset = readout_sample_offsets_ns[0]
    cmp_probe_offset_ns = (cmp_start_ns + cmp_end_ns) / 2.0
    lead_probe_offset_ns = min(5.00, cmp_end_ns + 0.10)
    bwd_probe_offset_ns = min(apply_end_ns - 0.05, bwd_start_ns + 0.50)
    if bwd_probe_offset_ns <= bwd_start_ns:
        bwd_probe_offset_ns = (bwd_start_ns + apply_end_ns) / 2.0
    update_probe_offset_ns = min(apply_end_ns - 0.05, max(bwd_probe_offset_ns, 10.50))
    lines: list[str] = []
    prints: list[str] = []
    for idx, sample in enumerate(samples):
        base = idx * CYCLE_NS
        label = int(sample["label"])
        other = 1 - label
        lines += [
            f".meas tran target_out_{idx} FIND V(out{label}) AT={base + default_offset:.2f}n",
            f".meas tran other_out_{idx} FIND V(out{other}) AT={base + default_offset:.2f}n",
            f".meas tran margin_{idx} PARAM='target_out_{idx}-other_out_{idx}'",
            f".meas tran score0_{idx} FIND V(score0) AT={base + default_offset:.2f}n",
            f".meas tran score1_{idx} FIND V(score1) AT={base + default_offset:.2f}n",
            f".meas tran score0_cmp_{idx} FIND V(score0) AT={base + cmp_probe_offset_ns:.2f}n",
            f".meas tran score1_cmp_{idx} FIND V(score1) AT={base + cmp_probe_offset_ns:.2f}n",
            f".meas tran out0_cmp_{idx} FIND V(out0) AT={base + cmp_probe_offset_ns:.2f}n",
            f".meas tran out1_cmp_{idx} FIND V(out1) AT={base + cmp_probe_offset_ns:.2f}n",
        ]
        for offset in readout_sample_offsets_ns:
            key = offset_key(offset)
            lines += [
                f".meas tran target_out_{key}_{idx} FIND V(out{label}) AT={base + offset:.2f}n",
                f".meas tran other_out_{key}_{idx} FIND V(out{other}) AT={base + offset:.2f}n",
                f".meas tran margin_{key}_{idx} PARAM='target_out_{key}_{idx}-other_out_{key}_{idx}'",
            ]
        for h in range(HIDDEN):
            lines.append(f".meas tran act{h}_{idx} FIND V(act{h}) AT={base + default_offset:.2f}n")
        for out in range(OUTPUTS):
            lines.append(f".meas tran lose{out}_{idx} FIND V(lose{out}) AT={base + 3.20:.2f}n")
        lines += [
            f".meas tran lead01_{idx} FIND V(lead01) AT={base + lead_probe_offset_ns:.2f}n",
            f".meas tran lead10_{idx} FIND V(lead10) AT={base + lead_probe_offset_ns:.2f}n",
        ]
        if sample["phase"] == "train":
            lines.append(f".meas tran bwd_signal_{idx} FIND V(bwd) AT={base + bwd_probe_offset_ns:.2f}n")
            if "mistake_latch" in backward_gate_mode:
                lines += [
                    f".meas tran merr0_{idx} FIND V(merr0) AT={base + bwd_probe_offset_ns:.2f}n",
                    f".meas tran merr1_{idx} FIND V(merr1) AT={base + bwd_probe_offset_ns:.2f}n",
                ]
            applies_update = sample.get("apply_update", True)
            if include_train_detail and hidden_delta_network_enabled:
                lines += [
                    f".meas tran hdp0_guard_{idx} FIND V(hdp0) AT={base + bwd_probe_offset_ns:.2f}n",
                ]
            if applies_update and include_train_detail:
                lines += [
                    f".meas tran train_target_after_{idx} FIND V(out{label}) AT={base + 15.50:.2f}n",
                    f".meas tran train_other_after_{idx} FIND V(out{other}) AT={base + 15.50:.2f}n",
                    f".meas tran train_margin_after_{idx} PARAM='train_target_after_{idx}-train_other_after_{idx}'",
                    f".meas tran train_d_margin_{idx} PARAM='train_margin_after_{idx}-margin_{idx}'",
                ]
                for out in range(OUTPUTS):
                    lines += [
                        f".meas tran vbo{out}p_before_{idx} FIND V(vbo{out}p) AT={base + 0.60:.2f}n",
                        f".meas tran vbo{out}n_before_{idx} FIND V(vbo{out}n) AT={base + 0.60:.2f}n",
                        f".meas tran vbo{out}p_after_{idx} FIND V(vbo{out}p) AT={base + 11.50:.2f}n",
                        f".meas tran vbo{out}n_after_{idx} FIND V(vbo{out}n) AT={base + 11.50:.2f}n",
                        f".meas tran vbo{out}_signed_before_{idx} PARAM='vbo{out}p_before_{idx}-vbo{out}n_before_{idx}'",
                        f".meas tran vbo{out}_signed_after_{idx} PARAM='vbo{out}p_after_{idx}-vbo{out}n_after_{idx}'",
                        f".meas tran d_vbo{out}_signed_{idx} PARAM='vbo{out}_signed_after_{idx}-vbo{out}_signed_before_{idx}'",
                    ]
                    for h in range(HIDDEN):
                        lines += [
                            f".meas tran vw{out}{h}p_before_{idx} FIND V(vw{out}{h}p) AT={base + 0.60:.2f}n",
                            f".meas tran vw{out}{h}n_before_{idx} FIND V(vw{out}{h}n) AT={base + 0.60:.2f}n",
                            f".meas tran vw{out}{h}p_after_{idx} FIND V(vw{out}{h}p) AT={base + 11.50:.2f}n",
                            f".meas tran vw{out}{h}n_after_{idx} FIND V(vw{out}{h}n) AT={base + 11.50:.2f}n",
                            f".meas tran vw{out}{h}_signed_before_{idx} PARAM='vw{out}{h}p_before_{idx}-vw{out}{h}n_before_{idx}'",
                            f".meas tran vw{out}{h}_signed_after_{idx} PARAM='vw{out}{h}p_after_{idx}-vw{out}{h}n_after_{idx}'",
                            f".meas tran d_vw{out}{h}_signed_{idx} PARAM='vw{out}{h}_signed_after_{idx}-vw{out}{h}_signed_before_{idx}'",
                        ]
            if include_signal_probe:
                for out in range(OUTPUTS):
                    lines += [
                        f".meas tran dp{out}_{idx} FIND V(dp{out}) AT={base + bwd_probe_offset_ns:.2f}n",
                        f".meas tran dn{out}_{idx} FIND V(dn{out}) AT={base + bwd_probe_offset_ns:.2f}n",
                        f".meas tran output_delta_net_{out}_{idx} PARAM='dp{out}_{idx}-dn{out}_{idx}'",
                    ]
                if hidden_delta_network_enabled:
                    for h in range(HIDDEN):
                        lines += [
                            f".meas tran hdp{h}_{idx} FIND V(hdp{h}) AT={base + bwd_probe_offset_ns:.2f}n",
                            f".meas tran hdn{h}_{idx} FIND V(hdn{h}) AT={base + bwd_probe_offset_ns:.2f}n",
                            f".meas tran hidden_delta_net_{h}_{idx} PARAM='hdp{h}_{idx}-hdn{h}_{idx}'",
                            f".meas tran hdp{h}_update_{idx} FIND V(hdp{h}) AT={base + update_probe_offset_ns:.2f}n",
                            f".meas tran hdn{h}_update_{idx} FIND V(hdn{h}) AT={base + update_probe_offset_ns:.2f}n",
                            f".meas tran hidden_delta_update_net_{h}_{idx} PARAM='hdp{h}_update_{idx}-hdn{h}_update_{idx}'",
                        ]
                        if hidden_delta_output_mode == "senseamp":
                            lines += [
                                f".meas tran hdpg{h}_{idx} FIND V(hdpg{h}) AT={base + update_probe_offset_ns:.2f}n",
                                f".meas tran hdng{h}_{idx} FIND V(hdng{h}) AT={base + update_probe_offset_ns:.2f}n",
                                f".meas tran hidden_delta_gate_net_{h}_{idx} PARAM='hdpg{h}_{idx}-hdng{h}_{idx}'",
                            ]
                for h in range(HIDDEN):
                    if not include_train_detail:
                        continue
                    for rail in HIDDEN_RAILS:
                        if include_hidden_grad_measures:
                            lines += [
                                f".meas tran ghp{h}_{rail}_{idx} FIND V(ghp{h}_{rail}) AT={base + 8.95:.2f}n",
                                f".meas tran ghn{h}_{rail}_{idx} FIND V(ghn{h}_{rail}) AT={base + 8.95:.2f}n",
                                f".meas tran hidden_grad_net_{h}_{rail}_{idx} PARAM='ghp{h}_{rail}_{idx}-ghn{h}_{rail}_{idx}'",
                            ]
                        if (
                            learning_mode == "accumulate_apply"
                            and hidden_apply_mode == "grad_senseamp"
                            and applies_update
                        ):
                            lines += [
                                f".meas tran hgwp{h}_{rail}_{idx} FIND V(hgwp{h}_{rail}) AT={base + 9.22:.2f}n",
                                f".meas tran hgwn{h}_{rail}_{idx} FIND V(hgwn{h}_{rail}) AT={base + 9.22:.2f}n",
                                f".meas tran hidden_apply_gate_net_{h}_{rail}_{idx} PARAM='hgwp{h}_{rail}_{idx}-hgwn{h}_{rail}_{idx}'",
                            ]
                    if applies_update:
                        for rail in HIDDEN_RAILS:
                            lines += [
                                f".meas tran wh{h}_{rail}p_before_{idx} FIND V(wh{h}_{rail}p) AT={base + 0.60:.2f}n",
                                f".meas tran wh{h}_{rail}n_before_{idx} FIND V(wh{h}_{rail}n) AT={base + 0.60:.2f}n",
                                f".meas tran wh{h}_{rail}p_after_{idx} FIND V(wh{h}_{rail}p) AT={base + 11.50:.2f}n",
                                f".meas tran wh{h}_{rail}n_after_{idx} FIND V(wh{h}_{rail}n) AT={base + 11.50:.2f}n",
                                f".meas tran wh{h}_{rail}_signed_before_{idx} PARAM='wh{h}_{rail}p_before_{idx}-wh{h}_{rail}n_before_{idx}'",
                                f".meas tran wh{h}_{rail}_signed_after_{idx} PARAM='wh{h}_{rail}p_after_{idx}-wh{h}_{rail}n_after_{idx}'",
                                f".meas tran d_wh{h}_{rail}_signed_{idx} PARAM='wh{h}_{rail}_signed_after_{idx}-wh{h}_{rail}_signed_before_{idx}'",
                            ]
        prints.append(f"print target_out_{idx} other_out_{idx} margin_{idx}")
    final_base = (len(samples) - 1) * CYCLE_NS
    for out in range(OUTPUTS):
        lines += [
            f".meas tran vbo{out}p_initial FIND V(vbo{out}p) AT=0.60n",
            f".meas tran vbo{out}n_initial FIND V(vbo{out}n) AT=0.60n",
            f".meas tran vbo{out}p_final FIND V(vbo{out}p) AT={final_base + 0.60:.2f}n",
            f".meas tran vbo{out}n_final FIND V(vbo{out}n) AT={final_base + 0.60:.2f}n",
            f".meas tran vbo{out}_signed_initial PARAM='vbo{out}p_initial-vbo{out}n_initial'",
            f".meas tran vbo{out}_signed_final PARAM='vbo{out}p_final-vbo{out}n_final'",
            f".meas tran d_vbo{out}_signed_total PARAM='vbo{out}_signed_final-vbo{out}_signed_initial'",
        ]
        for h in range(HIDDEN):
            lines += [
                f".meas tran vw{out}{h}p_initial FIND V(vw{out}{h}p) AT=0.60n",
                f".meas tran vw{out}{h}n_initial FIND V(vw{out}{h}n) AT=0.60n",
                f".meas tran vw{out}{h}p_final FIND V(vw{out}{h}p) AT={final_base + 0.60:.2f}n",
                f".meas tran vw{out}{h}n_final FIND V(vw{out}{h}n) AT={final_base + 0.60:.2f}n",
                f".meas tran vw{out}{h}_signed_initial PARAM='vw{out}{h}p_initial-vw{out}{h}n_initial'",
                f".meas tran vw{out}{h}_signed_final PARAM='vw{out}{h}p_final-vw{out}{h}n_final'",
                f".meas tran d_vw{out}{h}_signed_total PARAM='vw{out}{h}_signed_final-vw{out}{h}_signed_initial'",
            ]
    for h in range(HIDDEN):
        for rail in HIDDEN_RAILS:
            lines += [
                f".meas tran wh{h}_{rail}p_initial FIND V(wh{h}_{rail}p) AT=0.60n",
                f".meas tran wh{h}_{rail}n_initial FIND V(wh{h}_{rail}n) AT=0.60n",
                f".meas tran wh{h}_{rail}p_final FIND V(wh{h}_{rail}p) AT={final_base + 0.60:.2f}n",
                f".meas tran wh{h}_{rail}n_final FIND V(wh{h}_{rail}n) AT={final_base + 0.60:.2f}n",
                f".meas tran wh{h}_{rail}_signed_initial PARAM='wh{h}_{rail}p_initial-wh{h}_{rail}n_initial'",
                f".meas tran wh{h}_{rail}_signed_final PARAM='wh{h}_{rail}p_final-wh{h}_{rail}n_final'",
                f".meas tran d_wh{h}_{rail}_signed_total PARAM='wh{h}_{rail}_signed_final-wh{h}_{rail}_signed_initial'",
            ]
    return "\n".join(lines), "\n".join(prints)


def random_hidden_netlist(
    epochs: int,
    seed: int,
    init_seed: int | None,
    dataset_name: str,
    train_order: list[int],
    batch_apply: bool,
    synapse_design_name: str,
    hidden_forward_mode: str,
    hidden_delta_width_scale: float,
    hidden_gradient_width_scale: float,
    readout_gradient_width_scale: float,
    output_forward_width_scale: float,
    output_bias_forward_width_scale: float,
    output_relu_width_scale: float,
    output_head: str,
    hidden_error_rule: str,
    hidden_delta_relu_gate: str,
    hidden_delta_weight_device: str,
    hidden_delta_output_mode: str,
    hidden_delta_sense_width_u: float,
    hidden_delta_sense_cap_f: float,
    hidden_delta_internal_cap_f: float,
    hidden_delta_internal_leak_ohm: float,
    hidden_gradient_act_gate: str,
    hidden_apply_mode: str,
    learning_mode: str,
    flow_hidden_write: str,
    flow_pre_store: str,
    flow_pre_cap_f: float,
    flow_pre_consume_width_u: float,
    hidden_grad_sense_width_u: float,
    hidden_grad_sense_cap_f: float,
    feedback_scale: float,
    hidden_init_mode: str,
    readout_init_mode: str,
    separator_scale: float,
    separator_offset_v: float,
    readout_center_v: float,
    readout_random_center_v: float | None,
    readout_random_span_v: float,
    output_bias_offset_v: float,
    separator_csv: Path | None,
    separator_phase: str,
    hidden_cap_f: float,
    cap_dither_v: float,
    cap_dither_seed: int,
    cap_dither_scope: str,
    train_charge_noise_width_u: float,
    train_charge_noise_probability: float,
    train_charge_noise_seed: int,
    train_charge_noise_scope: str,
    train_charge_noise_pulse_ns: float,
    gradient_cap_f: float,
    hidden_gradient_cap_f: float,
    hidden_delta_cap_f: float,
    lead_cap_f: float,
    score_reset_v: float,
    readout_update_width_u: float,
    output_bias_update_width_u: float,
    readout_flow_polarity: str,
    readout_flow_write_mode: str,
    hidden_update_width_u: float,
    error_rule: str,
    latch_boost_width_u: float,
    residual_target_width_u: float,
    residual_output_width_u: float,
    lose_pull_kohm: float,
    lose_width_u: float,
    lead_mode: str,
    lead_width_u: float,
    backward_gate_mode: str,
    backward_gate_width_u: float,
    backward_gate_cap_f: float,
    bwd_start_ns: float,
    cmp_start_ns: float,
    cmp_end_ns: float,
    apply_start_ns: float,
    apply_end_ns: float,
    measure_detail: str,
    readout_sample_offsets_ns: list[float],
) -> tuple[str, list[dict[str, Any]]]:
    design = scaled_synapse_design(
        synapse_design_name,
        hidden_delta_width_scale,
        hidden_gradient_width_scale,
        readout_gradient_width_scale,
        output_forward_width_scale,
        output_bias_forward_width_scale,
        output_relu_width_scale,
    )
    include_gradient_caps = learning_mode == "accumulate_apply"
    state_seed = seed if init_seed is None else init_seed
    records = dataset_records(dataset_name, seed)
    set_input_rails(input_rails_for_records(records))
    samples = make_samples(records, epochs, train_order, batch_apply)
    stop = len(samples) * CYCLE_NS
    hidden_delta_network_enabled = learning_mode != "flow" or flow_hidden_write == "direct"
    input_sources = "\n".join(
        f"V{rail} {rail} 0 {sample_wave(samples, rail, stop)}" for rail in INPUT_RAILS
    )
    meas, prints = measure_lines(
        samples,
        hidden_apply_mode,
        learning_mode,
        hidden_delta_output_mode,
        measure_detail,
        readout_sample_offsets_ns,
        cmp_start_ns,
        cmp_end_ns,
        bwd_start_ns,
        apply_end_ns,
        backward_gate_mode,
        hidden_delta_network_enabled,
    )
    if learning_mode == "accumulate_apply":
        hidden_delta_block = hidden_delta(
            hidden_error_rule,
            hidden_delta_relu_gate,
            hidden_delta_weight_device,
            design,
            hidden_delta_internal_cap_f,
            hidden_delta_internal_leak_ohm,
        )
        hidden_delta_sense_block = hidden_delta_senseamps(
            hidden_delta_output_mode,
            hidden_delta_sense_width_u,
            hidden_delta_sense_cap_f,
        )
        learning_block = "\n".join(
            [
                readout_gradients_and_updates(readout_update_width_u, output_bias_update_width_u, design),
                hidden_delta_sense_block,
                hidden_gradients_and_updates(
                    hidden_update_width_u,
                    hidden_gradient_act_gate,
                    hidden_apply_mode,
                    hidden_grad_sense_width_u,
                    hidden_grad_sense_cap_f,
                    design,
                ),
            ]
        )
    elif learning_mode == "flow":
        if flow_hidden_write not in FLOW_HIDDEN_WRITES:
            raise ValueError(f"unknown flow hidden write mode: {flow_hidden_write}")
        if flow_pre_store not in FLOW_PRE_STORES:
            raise ValueError(f"unknown flow pre-store mode: {flow_pre_store}")
        hidden_delta_block = (
            hidden_delta(
                hidden_error_rule,
                hidden_delta_relu_gate,
                hidden_delta_weight_device,
                design,
                hidden_delta_internal_cap_f,
                hidden_delta_internal_leak_ohm,
            )
            if flow_hidden_write == "direct"
            else "* Hidden delta network omitted: flow hidden writes disabled for readout-only direct-flow test."
        )
        hidden_delta_sense_block = (
            hidden_delta_senseamps(
                hidden_delta_output_mode,
                hidden_delta_sense_width_u,
                hidden_delta_sense_cap_f,
            )
            if flow_hidden_write == "direct"
            else "* Hidden delta output omitted: flow hidden writes disabled for readout-only direct-flow test."
        )
        flow_blocks = [
            "* Direct backward/write flow: no gradient accumulator caps are used in the weight update path.",
            readout_flow_updates(
                readout_update_width_u,
                output_bias_update_width_u,
                flow_pre_store,
                readout_flow_polarity,
                readout_flow_write_mode,
            ),
            hidden_delta_sense_block,
        ]
        if flow_hidden_write == "direct":
            flow_blocks.append(hidden_flow_updates(hidden_update_width_u, flow_pre_store, hidden_delta_output_mode))
        else:
            flow_blocks.append("* Hidden weight capacitors are held during this direct-flow run.")
        learning_block = "\n".join(
            flow_blocks
        )
    else:
        raise ValueError(f"unknown learning mode: {learning_mode}")
    feedback_block = (
        "\n* Fixed signed feedback-alignment weights.\n"
        + feedback_caps(feedback_init(state_seed, feedback_scale), hidden_cap_f)
        if hidden_error_rule == "dfa"
        else ""
    )
    hidden_state, readout_state = dither_persistent_state(
        hidden_init(state_seed, hidden_init_mode),
        apply_output_bias_offset(
            readout_init(
                state_seed,
                readout_init_mode,
                separator_scale,
                separator_offset_v,
                readout_center_v,
                readout_random_center_v,
                readout_random_span_v,
                separator_csv,
                separator_phase,
            ),
            output_bias_offset_v,
        ),
        cap_dither_v,
        cap_dither_seed,
        cap_dither_scope,
    )
    return (
        f"""
* Device-level binary dataset with general random hidden layer.
* Dataset: {dataset_name}.
* {HIDDEN} hidden ReLU cells are fully connected to {len(INPUT_RAILS)} input rails plus a bias rail.
* Output ReLU cells also have capacitor-held trainable bias weights.
* No hidden cell is wired to a specific literal pattern.
* Synapse design: {design.name}; hidden error rule: {hidden_error_rule}; learning mode: {learning_mode}.
.param VDD=1.2
{mos_models()}
Vdd vdd 0 {{VDD}}
Vbias bias 0 {{VDD}}
Vscorecm scorecm 0 {score_reset_v:.12g}
{input_sources}
Vt0 t0 0 {target_wave(samples, 0, stop)}
Vt1 t1 0 {target_wave(samples, 1, stop)}
{phases(samples, bwd_start_ns, apply_start_ns, apply_end_ns, cmp_start_ns, cmp_end_ns, learning_mode, backward_gate_mode)}

{persistent_caps(hidden_state, readout_state, hidden_cap_f)}
{feedback_block}
{temporary_caps(gradient_cap_f, hidden_gradient_cap_f, hidden_delta_cap_f, lead_cap_f, include_gradient_caps, score_reset_v)}
{resets(lead_mode, include_gradient_caps, score_reset_v)}
{flow_pre_activation_stores(flow_pre_store, flow_pre_cap_f, flow_pre_consume_width_u) if learning_mode == "flow" else ""}
{train_charge_noise(samples, stop, train_charge_noise_width_u, train_charge_noise_probability, train_charge_noise_seed, train_charge_noise_scope, train_charge_noise_pulse_ns, bwd_start_ns)}

    {hidden_forward(design, hidden_forward_mode)}
{output_forward(design, output_head)}
{low_score_gate_cells(lose_pull_kohm, lose_width_u)}
{score_lead_gate_cells(lead_width_u, lead_mode)}
{backward_gate_cells(backward_gate_mode, backward_gate_width_u, backward_gate_cap_f, lead_mode)}
{error_cells(error_rule, latch_boost_width_u, residual_target_width_u, residual_output_width_u)}
{hidden_delta_block}
{learning_block}

.options method=gear maxord=2 rshunt=1e12 gmin=1e-12
.tran 10p {stop:.2f}n uic
{meas}
.control
run
{prints}
.endc
.end
""".lstrip(),
        samples,
    )


def run_netlist(spice_bin: str, path: Path, netlist: str, timeout: float) -> dict[str, float]:
    path.write_text(netlist)
    cmd = [spice_bin, "-b", str(path)] if "ngspice" in Path(spice_bin).name.lower() else [spice_bin, str(path)]
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-3000:])
    parsed = parse_measures(proc.stdout + "\n" + proc.stderr)
    if not parsed:
        raise RuntimeError("ngspice produced no parseable measurements:\n" + (proc.stdout + proc.stderr)[-3000:])
    return parsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--tag", default="device_xor2_random_hidden")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--hidden-cells", type=int, default=HIDDEN)
    ap.add_argument(
        "--init-seed",
        type=int,
        default=None,
        help="Optional separate seed for hidden/readout/feedback capacitor initial conditions.",
    )
    ap.add_argument(
        "--dataset",
        default="xor2",
        metavar="DATASET",
        help=(
            "Dataset name. Examples: "
            + ", ".join(DATASET_EXAMPLES)
            + "; counted variants such as moons16 or mnist01_16 are also accepted."
        ),
    )
    ap.add_argument("--order", default="auto")
    ap.add_argument("--batch-apply", action="store_true")
    ap.add_argument("--synapse-design", choices=sorted(SYNAPSE_DESIGNS), default="split_signed_v1")
    ap.add_argument("--hidden-delta-width-scale", type=float, default=1.0)
    ap.add_argument("--hidden-gradient-width-scale", type=float, default=1.0)
    ap.add_argument("--readout-gradient-width-scale", type=float, default=1.0)
    ap.add_argument(
        "--output-forward-width-scale",
        type=float,
        default=1.0,
        help="Scale the forward readout synapse devices that integrate hidden activations onto score caps.",
    )
    ap.add_argument(
        "--output-bias-forward-width-scale",
        type=float,
        default=1.0,
        help="Scale the output-bias forward devices independently of the readout synapses.",
    )
    ap.add_argument(
        "--output-relu-width-scale",
        type=float,
        default=1.0,
        help="Scale the output source-follower/ReLU device that charges the class output caps from score caps.",
    )
    ap.add_argument(
        "--output-head",
        choices=OUTPUT_HEAD_MODES,
        default="source_follower",
        help=(
            "Output cell driven by score caps. score_diff cross-couples the two scores so "
            "score common-mode is rejected before the output/lead path."
        ),
    )
    ap.add_argument("--hidden-error-rule", choices=HIDDEN_ERROR_RULES, default="backprop")
    ap.add_argument("--hidden-delta-relu-gate", choices=HIDDEN_DELTA_RELU_GATES, default="act_nrel")
    ap.add_argument(
        "--hidden-delta-weight-device",
        choices=HIDDEN_DELTA_WEIGHT_DEVICES,
        default="nmos",
        help="MOS model used by the readout-weight-gated transistor in the hidden-delta backprop path.",
    )
    ap.add_argument("--hidden-delta-output-mode", choices=HIDDEN_DELTA_OUTPUT_MODES, default="raw")
    ap.add_argument("--hidden-delta-sense-width-u", type=float, default=512.0)
    ap.add_argument("--hidden-delta-sense-cap-f", type=float, default=2.0)
    ap.add_argument("--hidden-delta-internal-cap-f", type=float, default=0.0)
    ap.add_argument("--hidden-delta-internal-leak-ohm", type=float, default=0.0)
    ap.add_argument("--hidden-gradient-act-gate", choices=HIDDEN_GRADIENT_ACT_GATES, default="act_nrel")
    ap.add_argument("--hidden-apply-mode", choices=HIDDEN_APPLY_MODES, default="direct")
    ap.add_argument(
        "--hidden-forward-mode",
        choices=HIDDEN_FORWARD_MODES,
        default="weighted_relu",
        help=(
            "Hidden forward circuit. weighted_relu uses trainable signed conductance into a ReLU cell; "
            "rail_buffer pass-gate copies input rails into hidden activation capacitors during fwd."
        ),
    )
    ap.add_argument("--learning-mode", choices=LEARNING_MODES, default="accumulate_apply")
    ap.add_argument("--flow-hidden-write", choices=FLOW_HIDDEN_WRITES, default="direct")
    ap.add_argument("--flow-pre-store", choices=FLOW_PRE_STORES, default="shared_node")
    ap.add_argument("--flow-pre-cap-f", type=float, default=2.0)
    ap.add_argument("--flow-pre-consume-width-u", type=float, default=0.05)
    ap.add_argument("--hidden-grad-sense-width-u", type=float, default=512.0)
    ap.add_argument("--hidden-grad-sense-cap-f", type=float, default=2.0)
    ap.add_argument("--feedback-scale", type=float, default=0.3)
    ap.add_argument(
        "--hidden-init",
        choices=HIDDEN_INIT_MODES,
        default="random",
        help="Initial hidden synapse capacitor pattern. input_identity maps input rail i to hidden cell i.",
    )
    ap.add_argument(
        "--readout-init",
        choices=[
            "random",
            "separator",
            "csv_separator",
            "rectified_separator",
            "csv_rectified_separator",
            "threshold_separator",
            "csv_threshold_separator",
        ],
        default="random",
    )
    ap.add_argument("--separator-scale", type=float, default=0.02)
    ap.add_argument("--separator-offset-v", type=float, default=0.0)
    ap.add_argument("--readout-center-v", type=float, default=0.64)
    ap.add_argument(
        "--readout-random-center-v",
        type=float,
        default=None,
        help=(
            "Optional center voltage for random readout capacitor initialization. "
            "Use this to place random caps in a measured high-slope conductance region."
        ),
    )
    ap.add_argument(
        "--readout-random-span-v",
        type=float,
        default=0.20,
        help="Peak-to-peak spread around --readout-random-center-v when random readout centering is enabled.",
    )
    ap.add_argument(
        "--output-bias-offset-v",
        type=float,
        default=0.0,
        help="Additional signed output-bias capacitor offset; positive favors class 0, negative favors class 1.",
    )
    ap.add_argument("--separator-csv", type=Path)
    ap.add_argument("--separator-phase", default="initial_eval")
    ap.add_argument("--hidden-cap-f", type=float, default=4.0)
    ap.add_argument("--cap-dither-v", type=float, default=0.0)
    ap.add_argument("--cap-dither-seed", type=int, default=0)
    ap.add_argument("--cap-dither-scope", choices=CAP_DITHER_SCOPES, default="none")
    ap.add_argument("--train-charge-noise-width-u", type=float, default=0.0)
    ap.add_argument("--train-charge-noise-prob", type=float, default=0.0)
    ap.add_argument("--train-charge-noise-seed", type=int, default=0)
    ap.add_argument("--train-charge-noise-scope", choices=TRAIN_CHARGE_NOISE_SCOPES, default="none")
    ap.add_argument("--train-charge-noise-pulse-ns", type=float, default=0.20)
    ap.add_argument("--gradient-cap-f", type=float, default=4.0)
    ap.add_argument("--hidden-gradient-cap-f", type=float)
    ap.add_argument("--hidden-delta-cap-f", type=float, default=12.0)
    ap.add_argument("--lead-cap-f", type=float, default=2.0)
    ap.add_argument(
        "--score-reset-v",
        type=float,
        default=0.0,
        help=(
            "Forward-phase reset/precharge voltage for output score capacitors. "
            "Nonzero values give negative readout branches discharge headroom."
        ),
    )
    ap.add_argument("--update-width-u", type=float, default=120.0)
    ap.add_argument("--readout-update-width-u", type=float)
    ap.add_argument("--output-bias-update-width-u", type=float)
    ap.add_argument(
        "--readout-flow-polarity",
        choices=READOUT_FLOW_POLARITIES,
        default="normal",
        help=(
            "Polarity of direct-flow readout discharges. normal drains negative caps on dp "
            "and positive caps on dn; reversed swaps those gates for sign-control experiments."
        ),
    )
    ap.add_argument(
        "--readout-flow-write-mode",
        choices=READOUT_FLOW_WRITE_MODES,
        default="discharge",
        help=(
            "Physical direct-flow readout write primitive. charge_discharge charges the branch "
            "matching the desired sign while draining the opposite branch."
        ),
    )
    ap.add_argument("--hidden-update-width-u", type=float)
    ap.add_argument(
        "--error-rule",
        choices=[
            "score",
            "perceptron",
            "margin",
            "competitive",
            "out_competitive",
            "out_residual",
            "out_competitive_latchboost",
            "out_mistake",
            "out_latch_mistake",
            "lowtarget",
            "mistake",
            "local_loss",
        ],
        default="score",
    )
    ap.add_argument("--latch-boost-width-u", type=float, default=64.0)
    ap.add_argument(
        "--residual-target-width-u",
        type=float,
        default=96.0,
        help="Target/source device width for the out_residual error cell.",
    )
    ap.add_argument(
        "--residual-output-width-u",
        type=float,
        default=64.0,
        help="Own-output feedback device width for the out_residual error cell.",
    )
    ap.add_argument("--lose-pull-kohm", type=float, default=100.0)
    ap.add_argument("--lose-width-u", type=float, default=24.0)
    ap.add_argument("--lead-mode", choices=["score", "lose", "senseamp", "out_senseamp"], default="score")
    ap.add_argument("--lead-width-u", type=float, default=96.0)
    ap.add_argument("--backward-gate-mode", choices=BACKWARD_GATE_MODES, default="scheduled")
    ap.add_argument("--backward-gate-width-u", type=float, default=64.0)
    ap.add_argument("--backward-gate-cap-f", type=float, default=2.0)
    ap.add_argument("--bwd-start-ns", type=float, default=6.75)
    ap.add_argument(
        "--cmp-start-ns",
        type=float,
        default=3.25,
        help="Start of the output/score compare window within each training cycle.",
    )
    ap.add_argument("--cmp-end-ns", type=float, default=4.10)
    ap.add_argument("--apply-start-ns", type=float, default=9.25)
    ap.add_argument("--apply-end-ns", type=float, default=11.20)
    ap.add_argument("--measure-detail", choices=MEASURE_DETAILS, default="full")
    ap.add_argument(
        "--readout-sample-offsets-ns",
        default="2.95",
        help="Comma-separated readout sampling offsets within each cycle. The first offset remains the compatibility accuracy point.",
    )
    args = ap.parse_args()
    if args.epochs < 0:
        raise SystemExit("--epochs must be nonnegative.")
    try:
        set_hidden_cells(args.hidden_cells)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if (
        args.hidden_delta_width_scale <= 0
        or args.hidden_gradient_width_scale <= 0
        or args.readout_gradient_width_scale <= 0
        or args.output_forward_width_scale <= 0
        or args.output_bias_forward_width_scale <= 0
        or args.output_relu_width_scale <= 0
    ):
        raise SystemExit("synapse width scales must be positive.")
    if args.hidden_delta_internal_cap_f < 0 or args.hidden_delta_internal_leak_ohm < 0:
        raise SystemExit("hidden delta internal damping values must be nonnegative.")
    if args.hidden_delta_sense_width_u <= 0 or args.hidden_delta_sense_cap_f <= 0:
        raise SystemExit("hidden delta sense width and capacitance must be positive.")
    hidden_gradient_cap_f = args.hidden_gradient_cap_f if args.hidden_gradient_cap_f is not None else args.gradient_cap_f
    if args.gradient_cap_f <= 0 or hidden_gradient_cap_f <= 0:
        raise SystemExit("gradient capacitances must be positive.")
    if args.hidden_delta_cap_f <= 0:
        raise SystemExit("hidden delta capacitance must be positive.")
    if not 0.0 <= args.score_reset_v <= 0.8:
        raise SystemExit("--score-reset-v must be in 0..0.8 V.")
    if args.hidden_grad_sense_width_u <= 0 or args.hidden_grad_sense_cap_f <= 0:
        raise SystemExit("hidden gradient sense width and capacitance must be positive.")
    if args.flow_pre_cap_f <= 0 or args.flow_pre_consume_width_u <= 0:
        raise SystemExit("flow pre-store capacitance and consume width must be positive.")
    if args.hidden_cap_f <= 0:
        raise SystemExit("--hidden-cap-f must be positive.")
    if args.cap_dither_v < 0:
        raise SystemExit("--cap-dither-v must be nonnegative.")
    if args.train_charge_noise_width_u < 0:
        raise SystemExit("--train-charge-noise-width-u must be nonnegative.")
    if not 0.0 <= args.train_charge_noise_prob <= 1.0:
        raise SystemExit("--train-charge-noise-prob must be in 0..1.")
    if args.train_charge_noise_pulse_ns <= 0:
        raise SystemExit("--train-charge-noise-pulse-ns must be positive.")
    if args.readout_update_width_u is not None and args.readout_update_width_u < 0:
        raise SystemExit("--readout-update-width-u must be nonnegative.")
    if args.output_bias_update_width_u is not None and args.output_bias_update_width_u < 0:
        raise SystemExit("--output-bias-update-width-u must be nonnegative.")
    if args.hidden_update_width_u is not None and args.hidden_update_width_u < 0:
        raise SystemExit("--hidden-update-width-u must be nonnegative.")
    if args.backward_gate_width_u <= 0 or args.backward_gate_cap_f <= 0:
        raise SystemExit("backward gate width and capacitance must be positive.")
    if args.latch_boost_width_u < 0:
        raise SystemExit("--latch-boost-width-u must be nonnegative.")
    if not 0.01 <= args.readout_center_v <= 1.15:
        raise SystemExit("--readout-center-v must be in the capacitor-voltage range 0.01..1.15 V.")
    if args.readout_random_center_v is not None and not 0.01 <= args.readout_random_center_v <= 1.15:
        raise SystemExit("--readout-random-center-v must be in the capacitor-voltage range 0.01..1.15 V.")
    if args.readout_random_span_v < 0:
        raise SystemExit("--readout-random-span-v must be nonnegative.")
    if abs(args.output_bias_offset_v) > 1.0:
        raise SystemExit("--output-bias-offset-v must be within +/-1.0 V.")
    try:
        readout_sample_offsets_ns = parse_offsets(args.readout_sample_offsets_ns)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not 2.40 <= args.cmp_start_ns < args.cmp_end_ns <= 5.00:
        raise SystemExit("--cmp-start-ns/--cmp-end-ns must stay inside 2.40..5.00 ns with start < end.")
    if not 6.50 <= args.bwd_start_ns < args.apply_end_ns:
        raise SystemExit("--bwd-start-ns must start after error storage and before the backward/update window ends.")
    if not 9.0 <= args.apply_start_ns < args.apply_end_ns <= 11.8:
        raise SystemExit("--apply-start-ns/--apply-end-ns must stay inside the update window before refiring.")
    records = dataset_records(args.dataset, args.seed)
    try:
        set_input_rails(input_rails_for_records(records))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.hidden_init == "input_identity" and args.hidden_cells < len(INPUT_RAILS):
        raise SystemExit("--hidden-init input_identity requires --hidden-cells >= the dataset input rail count.")
    all_patterns = [int(record["pattern"]) for record in records]
    if args.order == "auto":
        train_order = all_patterns
    elif args.order == "interleave":
        train_order = interleaved_order(records)
    else:
        train_order = [int(part) for part in args.order.split(",") if part.strip()]
    if sorted(train_order) != sorted(all_patterns):
        expected = ",".join(str(pattern) for pattern in all_patterns)
        raise SystemExit(f"--order must be 'auto', 'interleave', or a comma-separated permutation of {expected}.")
    if args.readout_init in {"separator", "rectified_separator", "threshold_separator"} and args.dataset != "xor2":
        raise SystemExit(f"--readout-init {args.readout_init} is only calibrated for --dataset xor2.")
    if args.readout_init in {"csv_separator", "csv_rectified_separator", "csv_threshold_separator"} and args.separator_csv is None:
        raise SystemExit(f"--readout-init {args.readout_init} requires --separator-csv.")
    if args.residual_target_width_u <= 0 or args.residual_output_width_u <= 0:
        raise SystemExit("--residual-target-width-u and --residual-output-width-u must be positive.")

    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    results = ROOT / "spice/results"
    tables = ROOT / "results/tables"
    for directory in [generated, results, tables]:
        directory.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)

    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag)
    netlist, samples = random_hidden_netlist(
        args.epochs,
        args.seed,
        args.init_seed,
        args.dataset,
        train_order,
        args.batch_apply,
        args.synapse_design,
        args.hidden_forward_mode,
        args.hidden_delta_width_scale,
        args.hidden_gradient_width_scale,
        args.readout_gradient_width_scale,
        args.output_forward_width_scale,
        args.output_bias_forward_width_scale,
        args.output_relu_width_scale,
        args.output_head,
        args.hidden_error_rule,
        args.hidden_delta_relu_gate,
        args.hidden_delta_weight_device,
        args.hidden_delta_output_mode,
        args.hidden_delta_sense_width_u,
        args.hidden_delta_sense_cap_f,
        args.hidden_delta_internal_cap_f,
        args.hidden_delta_internal_leak_ohm,
        args.hidden_gradient_act_gate,
        args.hidden_apply_mode,
        args.learning_mode,
        args.flow_hidden_write,
        args.flow_pre_store,
        args.flow_pre_cap_f,
        args.flow_pre_consume_width_u,
        args.hidden_grad_sense_width_u,
        args.hidden_grad_sense_cap_f,
        args.feedback_scale,
        args.hidden_init,
        args.readout_init,
        args.separator_scale,
        args.separator_offset_v,
        args.readout_center_v,
        args.readout_random_center_v,
        args.readout_random_span_v,
        args.output_bias_offset_v,
        args.separator_csv,
        args.separator_phase,
        args.hidden_cap_f,
        args.cap_dither_v,
        args.cap_dither_seed,
        args.cap_dither_scope,
        args.train_charge_noise_width_u,
        args.train_charge_noise_prob,
        args.train_charge_noise_seed,
        args.train_charge_noise_scope,
        args.train_charge_noise_pulse_ns,
        args.gradient_cap_f,
        hidden_gradient_cap_f,
        args.hidden_delta_cap_f,
        args.lead_cap_f,
        args.score_reset_v,
        args.readout_update_width_u if args.readout_update_width_u is not None else args.update_width_u,
        args.output_bias_update_width_u
        if args.output_bias_update_width_u is not None
        else (args.readout_update_width_u if args.readout_update_width_u is not None else args.update_width_u),
        args.readout_flow_polarity,
        args.readout_flow_write_mode,
        args.hidden_update_width_u if args.hidden_update_width_u is not None else args.update_width_u,
        args.error_rule,
        args.latch_boost_width_u,
        args.residual_target_width_u,
        args.residual_output_width_u,
        args.lose_pull_kohm,
        args.lose_width_u,
        args.lead_mode,
        args.lead_width_u,
        args.backward_gate_mode,
        args.backward_gate_width_u,
        args.backward_gate_cap_f,
        args.bwd_start_ns,
        args.cmp_start_ns,
        args.cmp_end_ns,
        args.apply_start_ns,
        args.apply_end_ns,
        args.measure_detail,
        readout_sample_offsets_ns,
    )
    t0 = time.perf_counter()
    parsed = run_netlist(spice_bin, generated / f"{safe_tag}.cir", netlist, args.timeout)

    hidden_delta_network_enabled = args.learning_mode != "flow" or args.flow_hidden_write == "direct"
    rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        phase = str(sample["phase"])
        row: dict[str, Any] = {
            "cycle": idx,
            "phase": phase,
            "pattern": int(sample["pattern"]),
            "label": int(sample["label"]),
            "reset_gradient": bool(sample.get("reset_gradient", False)),
            "applied_update": bool(sample.get("apply_update", False)),
            "target_out": parsed[f"target_out_{idx}"],
            "other_out": parsed[f"other_out_{idx}"],
            "margin": parsed[f"margin_{idx}"],
            "score0": parsed[f"score0_{idx}"],
            "score1": parsed[f"score1_{idx}"],
            "score0_cmp": parsed[f"score0_cmp_{idx}"],
            "score1_cmp": parsed[f"score1_cmp_{idx}"],
            "out0_cmp": parsed[f"out0_cmp_{idx}"],
            "out1_cmp": parsed[f"out1_cmp_{idx}"],
            "correct": parsed[f"margin_{idx}"] > 0,
            "mean_hidden_act": sum(parsed[f"act{h}_{idx}"] for h in range(HIDDEN)) / HIDDEN,
        }
        for offset in readout_sample_offsets_ns:
            key = offset_key(offset)
            margin = parsed[f"margin_{key}_{idx}"]
            row[f"target_out_{key}"] = parsed[f"target_out_{key}_{idx}"]
            row[f"other_out_{key}"] = parsed[f"other_out_{key}_{idx}"]
            row[f"margin_{key}"] = margin
            row[f"correct_{key}"] = margin > 0
        for h in range(HIDDEN):
            row[f"act{h}"] = parsed[f"act{h}_{idx}"]
        for out in range(OUTPUTS):
            row[f"lose{out}"] = parsed[f"lose{out}_{idx}"]
        row["lead01"] = parsed[f"lead01_{idx}"]
        row["lead10"] = parsed[f"lead10_{idx}"]
        if phase == "train":
            row["bwd_signal"] = parsed[f"bwd_signal_{idx}"]
            if "mistake_latch" in args.backward_gate_mode:
                row["merr0"] = parsed[f"merr0_{idx}"]
                row["merr1"] = parsed[f"merr1_{idx}"]
        if phase == "train" and args.measure_detail in {"full", "probe"}:
            row["max_abs_output_delta_signal"] = max(
                abs(parsed[f"output_delta_net_{out}_{idx}"]) for out in range(OUTPUTS)
            )
            row["max_output_delta_node"] = max(
                max(abs(parsed[f"dp{out}_{idx}"]), abs(parsed[f"dn{out}_{idx}"])) for out in range(OUTPUTS)
            )
            if hidden_delta_network_enabled:
                row["max_abs_hidden_delta_signal"] = max(
                    abs(parsed[f"hidden_delta_net_{h}_{idx}"]) for h in range(HIDDEN)
                )
                row["max_hidden_delta_node"] = max(
                    max(abs(parsed[f"hdp{h}_{idx}"]), abs(parsed[f"hdn{h}_{idx}"])) for h in range(HIDDEN)
                )
                row["max_abs_hidden_delta_update_signal"] = max(
                    abs(parsed[f"hidden_delta_update_net_{h}_{idx}"]) for h in range(HIDDEN)
                )
                row["max_hidden_delta_update_node"] = max(
                    max(abs(parsed[f"hdp{h}_update_{idx}"]), abs(parsed[f"hdn{h}_update_{idx}"]))
                    for h in range(HIDDEN)
                )
                if args.hidden_delta_output_mode == "senseamp":
                    row["max_abs_hidden_delta_gate_signal"] = max(
                        abs(parsed[f"hidden_delta_gate_net_{h}_{idx}"]) for h in range(HIDDEN)
                    )
                    row["max_hidden_delta_gate_node"] = max(
                        max(abs(parsed[f"hdpg{h}_{idx}"]), abs(parsed[f"hdng{h}_{idx}"]))
                        for h in range(HIDDEN)
                    )
        if phase == "train" and args.measure_detail == "full":
            if args.learning_mode == "accumulate_apply":
                row["max_abs_hidden_grad_signal"] = max(
                    abs(parsed[f"hidden_grad_net_{h}_{rail}_{idx}"])
                    for h in range(HIDDEN)
                    for rail in HIDDEN_RAILS
                )
                row["max_hidden_grad_node"] = max(
                    max(parsed[f"ghp{h}_{rail}_{idx}"], parsed[f"ghn{h}_{rail}_{idx}"])
                    for h in range(HIDDEN)
                    for rail in HIDDEN_RAILS
                )
            if (
                args.learning_mode == "accumulate_apply"
                and args.hidden_apply_mode == "grad_senseamp"
                and sample.get("apply_update", True)
            ):
                row["max_abs_hidden_apply_gate_signal"] = max(
                    abs(parsed[f"hidden_apply_gate_net_{h}_{rail}_{idx}"])
                    for h in range(HIDDEN)
                    for rail in HIDDEN_RAILS
                )
                row["max_hidden_apply_gate_node"] = max(
                    max(parsed[f"hgwp{h}_{rail}_{idx}"], parsed[f"hgwn{h}_{rail}_{idx}"])
                    for h in range(HIDDEN)
                    for rail in HIDDEN_RAILS
                )
        if phase == "train" and sample.get("apply_update", True) and args.measure_detail == "full":
            row.update(
                {
                    "post_update_margin": parsed[f"train_margin_after_{idx}"],
                    "d_margin_after_update": parsed[f"train_d_margin_{idx}"],
                    "max_abs_readout_weight_signed_delta": max(
                        abs(parsed[f"d_vw{out}{h}_signed_{idx}"])
                        for out in range(OUTPUTS)
                        for h in range(HIDDEN)
                    ),
                    "max_abs_output_bias_signed_delta": max(
                        abs(parsed[f"d_vbo{out}_signed_{idx}"]) for out in range(OUTPUTS)
                    ),
                    "max_abs_readout_signed_delta": max(
                        [
                            abs(parsed[f"d_vw{out}{h}_signed_{idx}"])
                            for out in range(OUTPUTS)
                            for h in range(HIDDEN)
                        ]
                        + [abs(parsed[f"d_vbo{out}_signed_{idx}"]) for out in range(OUTPUTS)]
                    ),
                    "max_abs_hidden_signed_delta": max(
                        abs(parsed[f"d_wh{h}_{rail}_signed_{idx}"])
                        for h in range(HIDDEN)
                        for rail in HIDDEN_RAILS
                    ),
                }
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    curve_path = results / f"{safe_tag}.csv"
    table_path = tables / f"{safe_tag}.csv"
    df.to_csv(curve_path, index=False)
    df.to_csv(table_path, index=False)

    initial_eval = df[df["phase"] == "initial_eval"]
    final_eval = df[df["phase"] == "final_eval"]
    train = df[df["phase"] == "train"]
    applied_train = train[train["applied_update"]]
    has_applied_train = not applied_train.empty
    if train.empty:
        lead_tracks_score_winner = None
        lead_score_winner_fraction = None
        mean_train_lead01 = 0.0
        mean_train_lead10 = 0.0
        mean_abs_train_lead_diff = 0.0
    else:
        score0_wins = ((train["label"] == 0) & (train["margin"] > 0)) | (
            (train["label"] == 1) & (train["margin"] < 0)
        )
        lead0_wins = lead_class0_wins(args.lead_mode, train["lead01"], train["lead10"])
        lead_score_winner_fraction = float((lead0_wins.to_numpy() == score0_wins.to_numpy()).mean())
        lead_tracks_score_winner = bool(lead_score_winner_fraction >= 0.75)
        mean_train_lead01 = float(train["lead01"].mean())
        mean_train_lead10 = float(train["lead10"].mean())
        mean_abs_train_lead_diff = float((train["lead01"] - train["lead10"]).abs().mean())
    total_readout_deltas = [
        parsed[f"d_vw{out}{h}_signed_total"]
        for out in range(OUTPUTS)
        for h in range(HIDDEN)
    ]
    total_output_bias_deltas = [parsed[f"d_vbo{out}_signed_total"] for out in range(OUTPUTS)]
    total_hidden_deltas = [
        parsed[f"d_wh{h}_{rail}_signed_total"]
        for h in range(HIDDEN)
        for rail in HIDDEN_RAILS
    ]
    effective_design = scaled_synapse_design(
        args.synapse_design,
        args.hidden_delta_width_scale,
        args.hidden_gradient_width_scale,
        args.readout_gradient_width_scale,
        args.output_forward_width_scale,
        args.output_bias_forward_width_scale,
        args.output_relu_width_scale,
    )
    has_hidden_apply_gate_metrics = (
        has_applied_train and "max_abs_hidden_apply_gate_signal" in applied_train.columns
    )
    has_hidden_grad_metrics = not train.empty and "max_abs_hidden_grad_signal" in train.columns
    has_train_delta_metrics = has_applied_train and "max_abs_readout_signed_delta" in applied_train.columns
    has_output_delta_metrics = not train.empty and "max_abs_output_delta_signal" in train.columns
    has_hidden_delta_metrics = not train.empty and "max_abs_hidden_delta_signal" in train.columns
    has_mistake_latch_metrics = not train.empty and {"merr0", "merr1"}.issubset(train.columns)
    has_hidden_delta_update_metrics = (
        not train.empty and "max_abs_hidden_delta_update_signal" in train.columns
    )
    has_hidden_delta_gate_metrics = (
        not train.empty and "max_abs_hidden_delta_gate_signal" in train.columns
    )
    has_bwd_metrics = not train.empty and "bwd_signal" in train.columns
    mistake_gate_stats = target_mistake_gate_stats(train) if args.backward_gate_mode == "target_mistake" else {}
    hidden_delta_network_enabled = args.learning_mode != "flow" or args.flow_hidden_write == "direct"
    hidden_weight_updates_enabled = args.epochs > 0 and not (
        args.learning_mode == "flow" and args.flow_hidden_write == "off"
    )
    readout_offset_stats = []
    for offset in readout_sample_offsets_ns:
        key = offset_key(offset)
        readout_offset_stats.append(
            {
                "offset_ns": offset,
                "key": key,
                "initial_accuracy": float(initial_eval[f"correct_{key}"].mean()),
                "final_accuracy": float(final_eval[f"correct_{key}"].mean()),
                "initial_min_margin_v": float(initial_eval[f"margin_{key}"].min()),
                "final_min_margin_v": float(final_eval[f"margin_{key}"].min()),
            }
        )
    best_final_transient = max(
        readout_offset_stats,
        key=lambda item: (float(item["final_accuracy"]), float(item["final_min_margin_v"])),
    )
    summary = {
        "tag": safe_tag,
        "simulator": version,
        "architecture": "device_level_binary_general_random_hidden",
        "status": "tiny_general_hidden_device_experiment",
        "benchmark": args.dataset,
        "dataset": args.dataset,
        "dataset_records": records,
        "model_level": "ngspice built-in LEVEL=1 MOS models; not a foundry PDK.",
        "synapse_design": args.synapse_design,
        "synapse_design_description": SYNAPSE_DESIGNS[args.synapse_design].description,
        "hidden_delta_width_scale": args.hidden_delta_width_scale,
        "hidden_gradient_width_scale": args.hidden_gradient_width_scale,
        "readout_gradient_width_scale": args.readout_gradient_width_scale,
        "output_forward_width_scale": args.output_forward_width_scale,
        "output_bias_forward_width_scale": args.output_bias_forward_width_scale,
        "output_relu_width_scale": args.output_relu_width_scale,
        "output_head": args.output_head,
        "effective_hidden_delta_width_u": effective_design.hidden_delta_width_u,
        "effective_hidden_gradient_width_u": effective_design.hidden_gradient_width_u,
        "effective_readout_gradient_width_u": effective_design.readout_gradient_width_u,
        "effective_output_forward_pos_width_u": effective_design.output_forward_pos_width_u,
        "effective_output_forward_neg_width_u": effective_design.output_forward_neg_width_u,
        "effective_output_bias_forward_pos_width_u": effective_design.output_bias_forward_pos_width_u,
        "effective_output_bias_forward_neg_width_u": effective_design.output_bias_forward_neg_width_u,
        "effective_output_relu_width_u": effective_design.output_relu_width_u,
        "hidden_error_rule": args.hidden_error_rule,
        "hidden_delta_relu_gate": args.hidden_delta_relu_gate,
        "hidden_delta_weight_device": args.hidden_delta_weight_device,
        "hidden_delta_output_mode": args.hidden_delta_output_mode,
        "hidden_delta_sense_width_u": args.hidden_delta_sense_width_u
        if args.hidden_delta_output_mode == "senseamp"
        else None,
        "hidden_delta_sense_cap_f": args.hidden_delta_sense_cap_f
        if args.hidden_delta_output_mode == "senseamp"
        else None,
        "hidden_delta_internal_cap_f": args.hidden_delta_internal_cap_f or None,
        "hidden_delta_internal_leak_ohm": args.hidden_delta_internal_leak_ohm or None,
        "hidden_gradient_act_gate": args.hidden_gradient_act_gate,
        "hidden_apply_mode": args.hidden_apply_mode,
        "learning_mode": args.learning_mode,
        "measure_detail": args.measure_detail,
        "flow_hidden_write": args.flow_hidden_write if args.learning_mode == "flow" else None,
        "flow_pre_store": args.flow_pre_store if args.learning_mode == "flow" else None,
        "flow_pre_cap_f": args.flow_pre_cap_f
        if args.learning_mode == "flow" and args.flow_pre_store != "shared_node"
        else None,
        "flow_pre_consume_width_u": args.flow_pre_consume_width_u
        if args.learning_mode == "flow" and args.flow_pre_store == "synapse_consume"
        else None,
        "uses_gradient_accumulators": args.learning_mode == "accumulate_apply",
        "uses_separate_apply_phase": args.learning_mode == "accumulate_apply"
        or (
            args.learning_mode == "flow"
            and args.flow_hidden_write == "direct"
            and args.hidden_delta_output_mode == "senseamp"
        ),
        "uses_direct_backward_write_flow": args.learning_mode == "flow",
        "uses_hidden_write_flow": args.learning_mode == "flow" and args.flow_hidden_write == "direct",
        "uses_per_synapse_pre_activation_trace": args.learning_mode == "flow"
        and args.flow_pre_store != "shared_node",
        "uses_destructive_pre_activation_trace_read": args.learning_mode == "flow"
        and args.flow_pre_store == "synapse_consume",
        "pre_activation_capture_path": (
            "mos_store_trace_caps"
            if args.learning_mode == "flow" and args.flow_pre_store != "shared_node"
            else "shared_source_nodes"
        ),
        "hidden_delta_passes_through_activation_gate": hidden_delta_network_enabled
        and args.hidden_delta_relu_gate != "none",
        "hidden_delta_output_latched": hidden_delta_network_enabled
        and args.hidden_delta_output_mode == "senseamp",
        "direct_weight_write_path": args.learning_mode == "flow",
        "hidden_grad_sense_width_u": args.hidden_grad_sense_width_u
        if args.learning_mode == "accumulate_apply" and args.hidden_apply_mode == "grad_senseamp"
        else None,
        "hidden_grad_sense_cap_f": args.hidden_grad_sense_cap_f
        if args.learning_mode == "accumulate_apply" and args.hidden_apply_mode == "grad_senseamp"
        else None,
        "hidden_delta_network_enabled": hidden_delta_network_enabled,
        "real_backprop_through_readout_synapses": args.hidden_error_rule == "backprop" and hidden_delta_network_enabled,
        "uses_readout_weight_transport_for_hidden_delta": args.hidden_error_rule == "backprop" and hidden_delta_network_enabled,
        "fixed_feedback_caps": args.hidden_error_rule == "dfa",
        "feedback_scale": args.feedback_scale if args.hidden_error_rule == "dfa" else None,
        "input_rails": INPUT_RAILS,
        "input_count": len(INPUT_RAILS),
        "input_frontend": records[0].get("input_frontend") if records else None,
        "input_frontend_key": records[0].get("input_frontend_key") if records else None,
        "hidden_forward_mode": args.hidden_forward_mode,
        "signal_path": (
            (
                f"{min(HIDDEN, len(INPUT_RAILS))} hidden activation capacitors are MOS pass-gate buffered "
                f"from externally driven input rails; any remaining hidden cells use signed conductance from "
                f"{len(INPUT_RAILS)} input rails plus a bias rail. "
            )
            if args.hidden_forward_mode == "rail_buffer"
            else (
                f"{HIDDEN} fully connected hidden ReLU cells receive signed conductance from "
                f"{len(INPUT_RAILS)} externally driven input rails plus a bias rail. "
            )
        )
        + (
            "Readout, output-bias, and hidden weights are capacitor-held signed states. "
            f"Readout flow writes use {args.readout_flow_write_mode} signed updates."
        ),
        "no_behavioral_signal_math": True,
        "uses_behavioral_tanh": False,
        "uses_behavioral_multipliers": False,
        "hidden_topology_programmed_as_literals": False,
        "hidden_bias_rail": True,
        "hidden_cells": HIDDEN,
        "output_bias_weights_trained": True,
        "hidden_weight_initialization": (
            "input_identity_passthrough_signed_caps"
            if args.hidden_init == "input_identity"
            else "deterministic_pseudorandom_dense_signed"
        ),
        "hidden_init": args.hidden_init,
        "readout_initialization": args.readout_init,
        "separator_scale": args.separator_scale if args.readout_init in SEPARATOR_READOUT_INITS else None,
        "separator_offset_v": args.separator_offset_v if args.readout_init in SEPARATOR_READOUT_INITS else None,
        "readout_center_v": args.readout_center_v if args.readout_init in SEPARATOR_READOUT_INITS else None,
        "readout_random_center_v": args.readout_random_center_v if args.readout_init == "random" else None,
        "readout_random_span_v": (
            args.readout_random_span_v
            if args.readout_init == "random" and args.readout_random_center_v is not None
            else None
        ),
        "output_bias_offset_v": args.output_bias_offset_v,
        "separator_csv": str(args.separator_csv)
        if args.readout_init in {"csv_separator", "csv_rectified_separator", "csv_threshold_separator"}
        else None,
        "separator_phase": args.separator_phase
        if args.readout_init in {"csv_separator", "csv_rectified_separator", "csv_threshold_separator"}
        else None,
        "readout_weights_trained": args.epochs > 0,
        "hidden_feature_weights_trained": hidden_weight_updates_enabled,
        "epochs": args.epochs,
        "seed": args.seed,
        "dataset_seed": args.seed,
        "initialization_seed": args.seed if args.init_seed is None else args.init_seed,
        "order_mode": args.order,
        "train_order": train_order,
        "batch_apply": args.batch_apply,
        "error_rule": args.error_rule,
        "latch_boost_width_u": args.latch_boost_width_u
        if args.error_rule == "out_competitive_latchboost"
        else None,
        "residual_target_width_u": args.residual_target_width_u if args.error_rule == "out_residual" else None,
        "residual_output_width_u": args.residual_output_width_u if args.error_rule == "out_residual" else None,
        "lose_pull_kohm": args.lose_pull_kohm,
        "lose_width_u": args.lose_width_u,
        "lead_mode": args.lead_mode,
        "lead_width_u": args.lead_width_u,
        "backward_gate_mode": args.backward_gate_mode,
        "backward_gate_width_u": args.backward_gate_width_u if args.backward_gate_mode != "scheduled" else None,
        "backward_gate_cap_f": args.backward_gate_cap_f if args.backward_gate_mode != "scheduled" else None,
        "bwd_start_ns": args.bwd_start_ns,
        "cmp_start_ns": args.cmp_start_ns,
        "cmp_end_ns": args.cmp_end_ns,
        "lead_gate_tracks_score_winner": lead_tracks_score_winner,
        "lead_gate_score_winner_fraction": lead_score_winner_fraction,
        "mean_train_lead01_v": mean_train_lead01,
        "mean_train_lead10_v": mean_train_lead10,
        "mean_abs_train_lead_diff_v": mean_abs_train_lead_diff,
        "max_train_bwd_signal_v": float(train["bwd_signal"].max()) if has_bwd_metrics else 0.0,
        "mean_train_bwd_signal_v": float(train["bwd_signal"].mean()) if has_bwd_metrics else 0.0,
        **mistake_gate_stats,
        "max_train_mistake_latch_v": float(train[["merr0", "merr1"]].max().max())
        if has_mistake_latch_metrics
        else None,
        "mean_train_mistake_latch_v": float(train[["merr0", "merr1"]].to_numpy().mean())
        if has_mistake_latch_metrics
        else None,
        "train_cycles": int(len(train)),
        "train_apply_cycles": int(len(applied_train)),
        "hidden_cap_f": args.hidden_cap_f,
        "cap_dither_v": args.cap_dither_v,
        "cap_dither_seed": args.cap_dither_seed if args.cap_dither_v > 0 else None,
        "cap_dither_scope": args.cap_dither_scope if args.cap_dither_v > 0 else None,
        "uses_train_charge_noise": args.train_charge_noise_width_u > 0
        and args.train_charge_noise_prob > 0
        and args.train_charge_noise_scope != "none",
        "train_charge_noise_width_u": args.train_charge_noise_width_u,
        "train_charge_noise_prob": args.train_charge_noise_prob,
        "train_charge_noise_seed": args.train_charge_noise_seed
        if args.train_charge_noise_width_u > 0 and args.train_charge_noise_prob > 0
        else None,
        "train_charge_noise_scope": args.train_charge_noise_scope
        if args.train_charge_noise_width_u > 0 and args.train_charge_noise_prob > 0
        else None,
        "train_charge_noise_pulse_ns": args.train_charge_noise_pulse_ns
        if args.train_charge_noise_width_u > 0 and args.train_charge_noise_prob > 0
        else None,
        "gradient_cap_f": args.gradient_cap_f if args.learning_mode == "accumulate_apply" else None,
        "hidden_gradient_cap_f": hidden_gradient_cap_f if args.learning_mode == "accumulate_apply" else None,
        "hidden_delta_cap_f": args.hidden_delta_cap_f,
        "lead_cap_f": args.lead_cap_f,
        "score_reset_v": args.score_reset_v,
        "update_width_u": args.update_width_u,
        "readout_update_width_u": args.readout_update_width_u if args.readout_update_width_u is not None else args.update_width_u,
        "output_bias_update_width_u": args.output_bias_update_width_u
        if args.output_bias_update_width_u is not None
        else (args.readout_update_width_u if args.readout_update_width_u is not None else args.update_width_u),
        "readout_flow_polarity": args.readout_flow_polarity if args.learning_mode == "flow" else None,
        "readout_flow_write_mode": args.readout_flow_write_mode if args.learning_mode == "flow" else None,
        "hidden_update_width_u": args.hidden_update_width_u if args.hidden_update_width_u is not None else args.update_width_u,
        "apply_start_ns": args.apply_start_ns,
        "apply_end_ns": args.apply_end_ns,
        "apply_duration_ns": args.apply_end_ns - args.apply_start_ns,
        "readout_sample_offsets_ns": readout_sample_offsets_ns,
        "readout_offset_stats": readout_offset_stats,
        "best_final_transient_offset_ns": best_final_transient["offset_ns"],
        "best_final_transient_accuracy": best_final_transient["final_accuracy"],
        "best_final_transient_min_margin_v": best_final_transient["final_min_margin_v"],
        "initial_eval_accuracy": float(initial_eval["correct"].mean()),
        "final_eval_accuracy": float(final_eval["correct"].mean()),
        "input_feature_separability": input_feature_separability(records),
        "initial_hidden_feature_separability": perceptron_separable(initial_eval),
        "final_hidden_feature_separability": perceptron_separable(final_eval),
        "initial_min_margin_v": float(initial_eval["margin"].min()),
        "final_min_margin_v": float(final_eval["margin"].min()),
        "min_margin_gain_v": float((final_eval["margin"].to_numpy() - initial_eval["margin"].to_numpy()).min()),
        "mean_hidden_activation_initial_v": float(initial_eval["mean_hidden_act"].mean()),
        "mean_hidden_activation_final_v": float(final_eval["mean_hidden_act"].mean()),
        "all_train_cycles_update_readout": bool((applied_train["max_abs_readout_signed_delta"] > 1e-7).all())
        if has_train_delta_metrics
        else None,
        "all_train_cycles_update_hidden": (
            bool((applied_train["max_abs_hidden_signed_delta"] > 1e-7).all())
            if has_train_delta_metrics and hidden_weight_updates_enabled
            else False if has_applied_train and not hidden_weight_updates_enabled
            else None
        ),
        "max_train_readout_signed_delta_v": float(applied_train["max_abs_readout_signed_delta"].max())
        if has_train_delta_metrics
        else 0.0,
        "max_train_readout_weight_signed_delta_v": float(applied_train["max_abs_readout_weight_signed_delta"].max())
        if has_train_delta_metrics
        else 0.0,
        "max_train_output_bias_signed_delta_v": float(applied_train["max_abs_output_bias_signed_delta"].max())
        if has_train_delta_metrics
        else 0.0,
        "max_train_hidden_signed_delta_v": float(applied_train["max_abs_hidden_signed_delta"].max())
        if has_train_delta_metrics
        else 0.0,
        "max_train_output_delta_signal_v": float(train["max_abs_output_delta_signal"].max())
        if has_output_delta_metrics
        else 0.0,
        "max_train_output_delta_node_v": float(train["max_output_delta_node"].max())
        if has_output_delta_metrics
        else 0.0,
        "max_train_hidden_delta_signal_v": float(train["max_abs_hidden_delta_signal"].max())
        if has_hidden_delta_metrics
        else 0.0,
        "max_train_hidden_delta_node_v": float(train["max_hidden_delta_node"].max())
        if has_hidden_delta_metrics
        else 0.0,
        "max_train_hidden_delta_update_signal_v": float(train["max_abs_hidden_delta_update_signal"].max())
        if has_hidden_delta_update_metrics
        else 0.0,
        "max_train_hidden_delta_update_node_v": float(train["max_hidden_delta_update_node"].max())
        if has_hidden_delta_update_metrics
        else 0.0,
        "max_train_hidden_delta_gate_signal_v": float(train["max_abs_hidden_delta_gate_signal"].max())
        if has_hidden_delta_gate_metrics
        else 0.0,
        "max_train_hidden_delta_gate_node_v": float(train["max_hidden_delta_gate_node"].max())
        if has_hidden_delta_gate_metrics
        else 0.0,
        "max_train_hidden_grad_signal_v": float(train["max_abs_hidden_grad_signal"].max())
        if has_hidden_grad_metrics
        else 0.0,
        "max_train_hidden_grad_node_v": float(train["max_hidden_grad_node"].max())
        if has_hidden_grad_metrics
        else 0.0,
        "max_train_hidden_apply_gate_signal_v": float(applied_train["max_abs_hidden_apply_gate_signal"].max())
        if has_hidden_apply_gate_metrics
        else 0.0,
        "max_train_hidden_apply_gate_node_v": float(applied_train["max_hidden_apply_gate_node"].max())
        if has_hidden_apply_gate_metrics
        else 0.0,
        "max_abs_total_readout_weight_signed_delta_v": float(max(abs(x) for x in total_readout_deltas)),
        "max_abs_total_output_bias_signed_delta_v": float(max(abs(x) for x in total_output_bias_deltas)),
        "max_abs_total_readout_signed_delta_v": float(
            max([abs(x) for x in total_readout_deltas] + [abs(x) for x in total_output_bias_deltas])
        ),
        "max_abs_total_hidden_signed_delta_v": float(max(abs(x) for x in total_hidden_deltas)),
        "curve": str(curve_path),
        "table_curve": str(table_path),
        "wall_time_s": time.perf_counter() - t0,
        "interpretation": (
            "This removes literal-detector hidden topology and tests whether a general dense hidden layer "
            "can run and update at device level on tiny binary datasets before moving to 8x8 MNIST."
        ),
    }
    summary_path = results / f"{safe_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
