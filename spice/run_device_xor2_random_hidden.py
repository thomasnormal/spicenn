from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from spicenn import (
        CapState,
        CapStateArray,
        DiffPairBleedWriteSelector,
        DifferentialCapStateArray,
        DirectFlowWeightCell,
        FanInTopology,
        load_readout_cap_state_csv,
        NetlistBuilder,
        NodeParasitics,
        PreTraceArray,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from spicenn import (
        CapState,
        CapStateArray,
        DiffPairBleedWriteSelector,
        DifferentialCapStateArray,
        DirectFlowWeightCell,
        FanInTopology,
        load_readout_cap_state_csv,
        NetlistBuilder,
        NodeParasitics,
        PreTraceArray,
    )

from run_device_multicell_classifier import mos_models, pulse_wave, pwl
from run_device_xor2_learned_features import CYCLE_NS, input_value, xor_label
from run_spice_sweep import ROOT, detect_spice, run_text_netlist, run_tiny_test
from _util import MEAS_RE, parse_measures
from datasets import (
    DATASET_EXAMPLES,
    dataset_records,
    dct2_lowfreq,
    mnist01_frontend,
    mnist01_records,
    mnist_records,
    parse_counted_mnist_dataset,
    random_local_relu_features,
    two_moons_records,
)


HIDDEN = 8
OUTPUTS = 2
INPUT_RAILS = ["x0", "nx0", "x1", "nx1"]
HIDDEN_RAILS = ["bias", *INPUT_RAILS]
ReadoutFanins = dict[int, tuple[int, ...]]
HiddenFanins = dict[int, tuple[str, ...]]


def dense_readout_fanins(hidden_cells: int | None = None, outputs: int | None = None) -> ReadoutFanins:
    hidden_count = HIDDEN if hidden_cells is None else hidden_cells
    output_count = OUTPUTS if outputs is None else outputs
    topology = FanInTopology.dense(tuple(range(hidden_count)), output_count)
    return {out: tuple(int(src) for src in srcs) for out, srcs in topology.as_fanins().items()}


def random_readout_fanins(
    hidden_cells: int,
    outputs: int,
    *,
    seed: int,
    fan_in: int | None = None,
    fan_out: int | None = None,
) -> ReadoutFanins:
    """Return random hidden->output connectivity as output-indexed fanins.

    fan_in constrains each output class to sample a fixed number of hidden
    activation rails. fan_out constrains each hidden activation rail to drive a
    fixed number of output/class rails, which is the hardware interpretation of
    "3 output synapses per neuron".
    """
    if hidden_cells <= 0 or outputs <= 0:
        raise ValueError("hidden_cells and outputs must be positive.")
    if (fan_in is None) == (fan_out is None):
        raise ValueError("provide exactly one of fan_in or fan_out.")
    if fan_in is not None:
        topology = FanInTopology.random_fanin(tuple(range(hidden_cells)), outputs, seed=seed, fan_in=fan_in)
        return {out: tuple(int(src) for src in srcs) for out, srcs in topology.as_fanins().items()}
    assert fan_out is not None
    topology = FanInTopology.random_fanout(tuple(range(hidden_cells)), outputs, seed=seed, fan_out=fan_out)
    return {out: tuple(int(src) for src in srcs) for out, srcs in topology.as_fanins().items()}


def balanced_random_readout_fanins(
    hidden_cells: int,
    outputs: int,
    *,
    seed: int,
    fan_out: int,
) -> ReadoutFanins:
    topology = FanInTopology.balanced_random_fanout(tuple(range(hidden_cells)), outputs, seed=seed, fan_out=fan_out)
    return {out: tuple(int(src) for src in srcs) for out, srcs in topology.as_fanins().items()}


def readout_fanouts_from_fanins(fanins: ReadoutFanins, hidden_cells: int | None = None) -> dict[int, tuple[int, ...]]:
    hidden_count = HIDDEN if hidden_cells is None else hidden_cells
    output_count = max(OUTPUTS, max(fanins.keys(), default=-1) + 1)
    topology = FanInTopology.from_fanins(tuple(range(hidden_count)), output_count, fanins)
    return {int(hidden): tuple(int(out) for out in outs) for hidden, outs in topology.fanouts().items()}


def effective_readout_fanins(readout_fanins: ReadoutFanins | None = None) -> ReadoutFanins:
    if readout_fanins is None:
        return dense_readout_fanins()
    return {out: tuple(readout_fanins.get(out, ())) for out in range(OUTPUTS)}


def readout_topology_summary(fanins: ReadoutFanins) -> dict[str, Any]:
    topology = FanInTopology.from_fanins(tuple(range(HIDDEN)), OUTPUTS, fanins)
    return topology.summary(prefix="readout")


def build_readout_fanins(
    mode: str,
    *,
    fan_in: int,
    fan_out: int,
    seed: int,
) -> ReadoutFanins:
    if mode == "dense":
        return dense_readout_fanins()
    if mode == "random_fanin":
        return random_readout_fanins(HIDDEN, OUTPUTS, seed=seed, fan_in=fan_in)
    if mode == "random_fanout":
        return random_readout_fanins(HIDDEN, OUTPUTS, seed=seed, fan_out=fan_out)
    if mode == "balanced_random_fanout":
        return balanced_random_readout_fanins(HIDDEN, OUTPUTS, seed=seed, fan_out=fan_out)
    raise ValueError(f"unknown readout topology: {mode}")


def dense_hidden_fanins(hidden_cells: int | None = None, input_rails: list[str] | None = None) -> HiddenFanins:
    hidden_count = HIDDEN if hidden_cells is None else hidden_cells
    rails = HIDDEN_RAILS if input_rails is None else ["bias", *input_rails]
    topology = FanInTopology.dense(tuple(rails), hidden_count)
    return {hidden: tuple(str(src) for src in srcs) for hidden, srcs in topology.as_fanins().items()}


def random_hidden_fanins(
    hidden_cells: int,
    input_rails: list[str],
    *,
    seed: int,
    fan_in: int,
) -> HiddenFanins:
    if hidden_cells <= 0:
        raise ValueError("hidden_cells must be positive.")
    topology = FanInTopology.random_fanin(
        tuple(input_rails),
        hidden_cells,
        seed=seed,
        fan_in=fan_in,
        always_sources=("bias",),
    )
    return {hidden: tuple(str(src) for src in srcs) for hidden, srcs in topology.as_fanins().items()}


def effective_hidden_fanins(hidden_fanins: HiddenFanins | None = None) -> HiddenFanins:
    if hidden_fanins is None:
        return dense_hidden_fanins()
    return {hidden: tuple(hidden_fanins.get(hidden, ("bias",))) for hidden in range(HIDDEN)}


def hidden_fanouts_from_fanins(fanins: HiddenFanins) -> dict[str, tuple[int, ...]]:
    topology = FanInTopology.from_fanins(tuple(HIDDEN_RAILS), HIDDEN, fanins)
    return {str(rail): tuple(int(hidden) for hidden in hidden_ids) for rail, hidden_ids in topology.fanouts().items()}


def hidden_topology_summary(fanins: HiddenFanins) -> dict[str, Any]:
    topology = FanInTopology.from_fanins(tuple(HIDDEN_RAILS), HIDDEN, fanins)
    fanouts = topology.fanouts()
    fanin_counts = topology.fanin_counts(exclude_sources=("bias",))
    fanout_counts = topology.fanout_counts(sources=tuple(INPUT_RAILS))
    return {
        "hidden_input_edge_count": int(sum(fanin_counts)),
        "hidden_input_fanin_counts": fanin_counts,
        "hidden_input_fanout_counts": fanout_counts,
        "hidden_fanins": {str(hidden): list(fanins.get(hidden, ())) for hidden in range(HIDDEN)},
        "hidden_input_fanouts": {rail: list(fanouts.get(rail, ())) for rail in INPUT_RAILS},
    }


def build_hidden_fanins(
    mode: str,
    *,
    fan_in: int,
    seed: int,
) -> HiddenFanins:
    if mode == "dense":
        return dense_hidden_fanins()
    if mode == "random_fanin":
        return random_hidden_fanins(HIDDEN, INPUT_RAILS, seed=seed, fan_in=fan_in)
    raise ValueError(f"unknown hidden input topology: {mode}")


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
    "split_signed_passact_buffered_v1": SynapseDesign(
        name="split_signed_passact_buffered_v1",
        description=(
            "Readout positive branch uses a source-follower replica of each hidden activation as the pass-device "
            "source, so readout fan-out does not directly drain the stored activation capacitor."
        ),
        output_forward_style="pass_act_buffered",
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
HIDDEN_FORWARD_MODES = ["weighted_relu", "weighted_relu_pass_input", "rail_buffer"]
OUTPUT_HEAD_MODES = [
    "source_follower",
    "score_diff",
    "split_score_none",
    "split_score_caps",
    "split_score_diffgate",
    "split_score_chargegate",
    "split_score_diffpair",
    "split_score_diode_diffpair",
    "split_score_compete_tail",
    "split_score_diode_mirror_gate_caps",
    "split_score_diode_mirror_caps",
]
SPLIT_SCORE_OUTPUT_HEADS = {
    "split_score_none",
    "split_score_caps",
    "split_score_diffgate",
    "split_score_chargegate",
    "split_score_diffpair",
    "split_score_diode_diffpair",
    "split_score_compete_tail",
    "split_score_diode_mirror_gate_caps",
    "split_score_diode_mirror_caps",
}
DIODE_MIRROR_OUTPUT_HEADS = {
    "split_score_diode_mirror_gate_caps",
    "split_score_diode_mirror_caps",
}
COMMON_MODE_OUT_RESET_HEADS = {"split_score_caps", "split_score_diffgate"}
LOW_TRUE_OUTPUT_HEADS = {
    "split_score_diffpair",
    "split_score_diode_diffpair",
    "split_score_compete_tail",
    "split_score_diode_mirror_gate_caps",
    "split_score_diode_mirror_caps",
}
DECISION_SOURCES = ["out", "score"]
UPDATE_ERROR_RULES = {
    "out_residual",
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
    "ce_mirror_hybrid_limited",
    "ce_mirror_compete_limited",
}
LEARNING_MODES = ["accumulate_apply", "flow"]
FLOW_HIDDEN_WRITES = ["direct", "off"]
FLOW_PRE_STORES = ["shared_node", "synapse_gate", "synapse_consume", "synapse_boost", "synapse_spike"]
READOUT_FLOW_POLARITIES = ["normal", "reversed"]
READOUT_CENTER_PULL_GATES = ["bwd", "apply"]
READOUT_CENTER_PULL_MODES = ["always", "state_high"]
READOUT_WRITE_STATE_GATE_MODES = ["none", "state_high_discharge", "state_window"]
OUTPUT_BIAS_WRITE_PRE_GATES = ["none", "bias"]
OUTPUT_BIAS_FLOW_POLARITIES = ["follow_readout", "normal", "reversed"]
WRITE_ERROR_EXCLUSION_MODES = ["none", "pmos_inhibit", "pmos_inhibit_decay", "diffpair_bleed"]
WRITE_GATE_DEVICES = ["NSENSE", "NREL", "NMOS"]
READOUT_TOPOLOGIES = ["dense", "random_fanin", "random_fanout", "balanced_random_fanout"]
HIDDEN_INPUT_TOPOLOGIES = ["dense", "random_fanin"]
READOUT_FLOW_WRITE_MODES = [
    "discharge",
    "bounded_discharge",
    "charge_only",
    "bounded_charge_only",
    "charge_discharge",
    "bounded_charge_discharge",
    "bounded_cmos_charge_discharge",
    "bounded_pmos_charge_only",
    "bounded_pmos_charge_discharge",
]
HIGH_SIDE_PMOS_READOUT_WRITE_MODES = {
    "bounded_cmos_charge_discharge",
    "bounded_pmos_charge_only",
    "bounded_pmos_charge_discharge",
}
HIDDEN_FLOW_WRITE_MODES = [
    "discharge",
    "bounded_discharge",
    "charge_only",
    "bounded_charge_only",
    "charge_discharge",
    "bounded_charge_discharge",
]
HIDDEN_INIT_MODES = ["random", "input_identity"]
MEASURE_DETAILS = ["full", "probe", "light"]
SPICE_ACCURACY_PRESETS = {
    "standard": "method=gear maxord=2 rshunt=1e12 gmin=1e-12",
    "fast": "method=gear maxord=2 rshunt=1e11 gmin=1e-11 reltol=3e-3 vntol=1e-5 abstol=1e-10",
    "loose": "method=gear maxord=2 rshunt=1e10 gmin=1e-10 reltol=1e-2 vntol=3e-5 abstol=1e-9",
}
BACKWARD_GATE_MODES = [
    "scheduled",
    "lead_or",
    "target_mistake",
    "target_mistake_latch",
    "target_mistake_latch_simple",
    "target_out_mistake_latch",
    "target_out_mistake_latch_restore",
    "target_out_mistake_latch_restore_stacked",
    "target_out_mistake_latch_restore_stacked_timed",
]
CAP_DITHER_SCOPES = ["none", "hidden", "readout", "all"]
TRAIN_CHARGE_NOISE_SCOPES = CAP_DITHER_SCOPES
DIFFERENTIAL_SEPARATOR_INITS = {"separator", "csv_separator"}
RECTIFIED_SEPARATOR_INITS = {"rectified_separator", "csv_rectified_separator"}
THRESHOLD_SEPARATOR_INITS = {"threshold_separator", "csv_threshold_separator"}
SEPARATOR_READOUT_INITS = DIFFERENTIAL_SEPARATOR_INITS | RECTIFIED_SEPARATOR_INITS | THRESHOLD_SEPARATOR_INITS
PROGRAMMED_READOUT_INITS = {"csv_readout", "csv_readout_rectified", "csv_readout_sparse_rectified"}
CAP_STATE_READOUT_INITS = {"csv_cap_state"}
CSV_READOUT_INITS = (
    {"csv_separator", "csv_rectified_separator", "csv_threshold_separator"}
    | PROGRAMMED_READOUT_INITS
    | CAP_STATE_READOUT_INITS
)
SPARSE_READOUT_INACTIVE_V = 0.16


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


def spice_options_for_preset(preset: str) -> str:
    try:
        return SPICE_ACCURACY_PRESETS[preset]
    except KeyError as exc:
        allowed = ", ".join(sorted(SPICE_ACCURACY_PRESETS))
        raise ValueError(f"unknown SPICE accuracy preset: {preset}. Expected one of {allowed}.") from exc


def default_readout_write_high_v(readout_flow_write_mode: str) -> float:
    if readout_flow_write_mode in HIGH_SIDE_PMOS_READOUT_WRITE_MODES:
        return 1.0
    return 0.58


def prediction_histogram_for(df: pd.DataFrame, output_count: int, prediction_column: str) -> dict[str, int]:
    hist = {str(out): 0 for out in range(output_count)}
    if df.empty or prediction_column not in df:
        return hist
    for label, count in df[prediction_column].dropna().astype(int).value_counts().items():
        hist[str(label)] = int(count)
    return hist


def prediction_histogram(df: pd.DataFrame, output_count: int) -> dict[str, int]:
    return prediction_histogram_for(df, output_count, "predicted_label")


def per_class_accuracy_for(df: pd.DataFrame, output_count: int, correct_column: str) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    if df.empty or "label" not in df or correct_column not in df:
        return {str(out): None for out in range(output_count)}
    for label in range(output_count):
        subset = df[df["label"] == label]
        result[str(label)] = None if subset.empty else float(subset[correct_column].mean())
    return result


def per_class_accuracy(df: pd.DataFrame, output_count: int) -> dict[str, float | None]:
    return per_class_accuracy_for(df, output_count, "correct")


def column_centering_metrics(prefix: str, source: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Report class accuracy after subtracting each output column's measured mean."""
    if source.size == 0 or labels.size == 0:
        return {
            f"{prefix}_column_centered_accuracy": None,
            f"{prefix}_column_centered_min_margin_v": None,
            f"{prefix}_mean_by_output_v": [],
        }
    means = source.mean(axis=0)
    centered = source - means
    predictions = np.argmax(centered, axis=1)
    margins = []
    for row, label in zip(centered, labels):
        target = row[int(label)]
        other = max(value for out, value in enumerate(row) if out != int(label))
        margins.append(float(target - other))
    return {
        f"{prefix}_column_centered_accuracy": float((predictions == labels).mean()),
        f"{prefix}_column_centered_min_margin_v": float(min(margins)) if margins else None,
        f"{prefix}_mean_by_output_v": [float(value) for value in means],
    }


def score_matrix(rows: pd.DataFrame, output_count: int) -> np.ndarray:
    cols = [f"score{out}" for out in range(output_count)]
    if rows.empty or any(col not in rows for col in cols):
        return np.empty((0, output_count), dtype=float)
    return rows[cols].to_numpy(dtype=float)


def output_decision_matrix(rows: pd.DataFrame, output_count: int, *, low_true_output: bool) -> np.ndarray:
    out_cols = [f"out{out}" for out in range(output_count)]
    if not rows.empty and all(col in rows for col in out_cols):
        values = rows[out_cols].to_numpy(dtype=float)
        return -values if low_true_output else values
    if (
        output_count == 2
        and not rows.empty
        and {"label", "output_target", "output_other"}.issubset(rows.columns)
    ):
        values = np.empty((len(rows), output_count), dtype=float)
        for idx, (_row_index, row) in enumerate(rows.iterrows()):
            label = int(row["label"])
            other = 1 - label
            values[idx, label] = float(row["output_target"])
            values[idx, other] = float(row["output_other"])
        return values
    return np.empty((0, output_count), dtype=float)


def output_delta_alignment_metrics(train: pd.DataFrame, output_count: int) -> dict[str, float | None]:
    """Summarize whether measured output error rails match one-vs-rest class updates."""
    delta_cols = [f"output_delta_net_{out}" for out in range(output_count)]
    empty = {
        "train_output_delta_sign_alignment_fraction": None,
        "train_output_delta_target_gt_all_others_fraction": None,
        "train_output_delta_target_positive_fraction": None,
        "train_output_delta_other_negative_fraction": None,
        "train_output_delta_all_other_negative_fraction": None,
        "train_output_delta_target_minus_max_other_mean_v": None,
        "train_output_delta_target_mean_v": None,
        "train_output_delta_correct_target_mean_v": None,
        "train_output_delta_wrong_target_mean_v": None,
        "train_output_delta_target_mistake_gain_v": None,
        "train_output_delta_wrong_to_correct_target_ratio": None,
        "train_output_delta_max_other_mean_v": None,
        "train_output_delta_mean_other_mean_v": None,
    }
    if train.empty or "label" not in train or any(col not in train for col in delta_cols):
        return empty

    rows = train.dropna(subset=["label", *delta_cols])
    if rows.empty:
        return empty

    labels = rows["label"].astype(int).to_numpy()
    deltas = rows[delta_cols].to_numpy(dtype=float)
    sample_indices = np.arange(len(rows))
    target_delta = deltas[sample_indices, labels]
    is_other = np.ones_like(deltas, dtype=bool)
    is_other[sample_indices, labels] = False
    other_deltas = deltas[is_other].reshape(len(rows), output_count - 1)
    desired_sign = np.full_like(deltas, -1.0)
    desired_sign[sample_indices, labels] = 1.0
    sign_matches = deltas * desired_sign >= 0.0
    correct = rows["score_correct"].astype(bool).to_numpy() if "score_correct" in rows else None
    correct_target = target_delta[correct] if correct is not None else np.array([], dtype=float)
    wrong_target = target_delta[~correct] if correct is not None else np.array([], dtype=float)
    correct_mean = float(correct_target.mean()) if correct_target.size else None
    wrong_mean = float(wrong_target.mean()) if wrong_target.size else None
    if correct_mean is not None and abs(correct_mean) > 1e-12 and wrong_mean is not None:
        wrong_to_correct = float(wrong_mean / correct_mean)
    else:
        wrong_to_correct = None

    return {
        "train_output_delta_sign_alignment_fraction": float(sign_matches.mean()),
        "train_output_delta_target_gt_all_others_fraction": float((target_delta > other_deltas.max(axis=1)).mean()),
        "train_output_delta_target_positive_fraction": float((target_delta >= 0.0).mean()),
        "train_output_delta_other_negative_fraction": float((other_deltas <= 0.0).mean()),
        "train_output_delta_all_other_negative_fraction": float((other_deltas <= 0.0).all(axis=1).mean()),
        "train_output_delta_target_minus_max_other_mean_v": float((target_delta - other_deltas.max(axis=1)).mean()),
        "train_output_delta_target_mean_v": float(target_delta.mean()),
        "train_output_delta_correct_target_mean_v": correct_mean,
        "train_output_delta_wrong_target_mean_v": wrong_mean,
        "train_output_delta_target_mistake_gain_v": (
            float(wrong_mean - correct_mean) if wrong_mean is not None and correct_mean is not None else None
        ),
        "train_output_delta_wrong_to_correct_target_ratio": wrong_to_correct,
        "train_output_delta_max_other_mean_v": float(other_deltas.max(axis=1).mean()),
        "train_output_delta_mean_other_mean_v": float(other_deltas.mean(axis=1).mean()),
    }


def output_delta_sums_by_out(train: pd.DataFrame, output_count: int) -> list[float] | None:
    delta_cols = [f"output_delta_net_{out}" for out in range(output_count)]
    if train.empty or any(col not in train for col in delta_cols):
        return None
    rows = train.dropna(subset=delta_cols)
    if rows.empty:
        return None
    return [float(rows[col].sum()) for col in delta_cols]


def resolve_error_source_rails(
    *,
    error_rule: str,
    output_count: int,
    target_high_v: float,
    error_target_source_v: float | None,
    error_nontarget_source_v: float | None,
    error_source_balance: str,
    error_nontarget_balance_scale: float,
) -> tuple[float | None, float | None]:
    if error_source_balance == "none":
        return error_target_source_v, error_nontarget_source_v
    if error_rule != "onehot_limited":
        raise ValueError("--error-source-balance currently applies only to --error-rule onehot_limited.")
    if output_count < 2:
        raise ValueError("--error-source-balance requires at least two output classes.")
    if error_nontarget_balance_scale <= 0.0:
        raise ValueError("--error-nontarget-balance-scale must be positive.")

    target_source = error_target_source_v if error_target_source_v is not None else target_high_v
    if error_source_balance == "onehot_average":
        nontarget_source = (
            error_nontarget_source_v
            if error_nontarget_source_v is not None
            else target_source * error_nontarget_balance_scale / (output_count - 1)
        )
        return target_source, nontarget_source
    raise ValueError(f"unknown --error-source-balance {error_source_balance!r}")


def signed_alignment_fraction(a: list[float], b: list[float], *, eps: float = 1e-12) -> float | None:
    if len(a) != len(b) or not a:
        return None
    pairs = [(x, y) for x, y in zip(a, b) if abs(x) > eps and abs(y) > eps]
    if not pairs:
        return None
    return float(np.mean([np.sign(x) == np.sign(y) for x, y in pairs]))


def cosine_similarity(a: list[float], b: list[float], *, eps: float = 1e-12) -> float | None:
    if len(a) != len(b) or not a:
        return None
    av = np.array(a, dtype=float)
    bv = np.array(b, dtype=float)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= eps:
        return None
    return float(np.dot(av, bv) / denom)


def selected_decision(
    *,
    decision_source: str,
    output_target: float,
    output_other: float,
    output_predicted_label: int,
    score_target: float,
    score_other: float,
    score_predicted_label: int,
) -> tuple[float, float, float, int]:
    """Return the target/other/margin/prediction for the configured analog decision rails."""
    if decision_source == "score":
        return score_target, score_other, score_target - score_other, score_predicted_label
    if decision_source == "out":
        return output_target, output_other, output_target - output_other, output_predicted_label
    allowed = ", ".join(DECISION_SOURCES)
    raise ValueError(f"unknown decision source: {decision_source}. Expected one of {allowed}.")


def set_hidden_cells(count: int) -> None:
    global HIDDEN
    if count <= 0:
        raise ValueError("hidden cell count must be positive.")
    HIDDEN = count


def set_output_count(count: int) -> None:
    global OUTPUTS
    if count < 2:
        raise ValueError("output count must be at least two.")
    OUTPUTS = count


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
    output_forward_pos_width_scale: float = 1.0,
    output_forward_neg_width_scale: float = 1.0,
    output_bias_forward_width_scale: float = 1.0,
    output_relu_width_scale: float = 1.0,
) -> SynapseDesign:
    base = SYNAPSE_DESIGNS[name]
    return replace(
        base,
        hidden_delta_width_u=base.hidden_delta_width_u * hidden_delta_width_scale,
        hidden_gradient_width_u=base.hidden_gradient_width_u * hidden_gradient_width_scale,
        readout_gradient_width_u=base.readout_gradient_width_u * readout_gradient_width_scale,
        output_forward_pos_width_u=base.output_forward_pos_width_u
        * output_forward_width_scale
        * output_forward_pos_width_scale,
        output_forward_neg_width_u=base.output_forward_neg_width_u
        * output_forward_width_scale
        * output_forward_neg_width_scale,
        output_bias_forward_pos_width_u=base.output_bias_forward_pos_width_u * output_bias_forward_width_scale,
        output_bias_forward_neg_width_u=base.output_bias_forward_neg_width_u * output_bias_forward_width_scale,
        output_relu_width_u=base.output_relu_width_u * output_relu_width_scale,
    )




def label_shuffled_records(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Return records with labels randomly reassigned while features stay fixed."""
    if not records:
        return []
    rng = np.random.default_rng(seed)
    labels = np.asarray([int(record["label"]) for record in records], dtype=int)
    shuffled = rng.permutation(labels)
    if len(labels) > 1 and np.array_equal(shuffled, labels):
        shuffled = np.roll(shuffled, 1)
    return [
        {
            **record,
            "true_label": int(record["label"]),
            "label": int(label),
            "label_shuffle_seed": seed,
        }
        for record, label in zip(records, shuffled)
    ]


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
    y = ordered["label"].to_numpy(dtype=int)
    return perceptron_separable_array(x, y)


def binary_perceptron_separable_array(x: np.ndarray, y_labels: np.ndarray) -> dict[str, Any]:
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
                "perceptron_type": "binary",
                "perceptron_epochs": epoch,
                "min_margin": float((y * (xb @ w)).min()),
            }
    return {
        "linearly_separable": False,
        "perceptron_type": "binary",
        "perceptron_epochs": 20_000,
        "best_min_margin": best_margin,
        "best_epoch": best_epoch,
    }


def multiclass_perceptron_separable_array(x: np.ndarray, y_labels: np.ndarray) -> dict[str, Any]:
    labels = y_labels.astype(int)
    classes = sorted(int(label) for label in np.unique(labels))
    class_to_row = {label: idx for idx, label in enumerate(classes)}
    y = np.asarray([class_to_row[int(label)] for label in labels], dtype=int)
    xb = np.c_[x, np.ones(len(x))]
    w = np.zeros((len(classes), xb.shape[1]))
    best_margin = float("-inf")
    best_epoch = 0
    best_accuracy = 0.0
    for epoch in range(20_000):
        scores = xb @ w.T
        predicted = np.argmax(scores, axis=1)
        true_scores = scores[np.arange(len(y)), y]
        masked = scores.copy()
        masked[np.arange(len(y)), y] = -np.inf
        margins = true_scores - masked.max(axis=1)
        accuracy = float((predicted == y).mean())
        margin = float(margins.min())
        if (accuracy, margin) > (best_accuracy, best_margin):
            best_accuracy = accuracy
            best_margin = margin
            best_epoch = epoch
        errors = 0
        for xi, yi, pred in zip(xb, y, predicted):
            if pred != yi:
                w[yi] += xi
                w[pred] -= xi
                errors += 1
        if errors == 0:
            final_scores = xb @ w.T
            final_true = final_scores[np.arange(len(y)), y]
            final_masked = final_scores.copy()
            final_masked[np.arange(len(y)), y] = -np.inf
            final_margins = final_true - final_masked.max(axis=1)
            return {
                "linearly_separable": True,
                "perceptron_type": "multiclass",
                "classes": classes,
                "perceptron_epochs": epoch,
                "min_margin": float(final_margins.min()),
                "training_accuracy": 1.0,
            }
    return {
        "linearly_separable": False,
        "perceptron_type": "multiclass",
        "classes": classes,
        "perceptron_epochs": 20_000,
        "best_min_margin": best_margin,
        "best_epoch": best_epoch,
        "best_training_accuracy": best_accuracy,
    }


def perceptron_separable_array(x: np.ndarray, y_labels: np.ndarray) -> dict[str, Any]:
    classes = sorted(int(label) for label in np.unique(y_labels.astype(int)))
    if set(classes) <= {0, 1}:
        return binary_perceptron_separable_array(x, y_labels)
    return multiclass_perceptron_separable_array(x, y_labels)


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


def target_wave(
    samples: list[dict[str, Any]],
    output: int,
    stop_ns: float,
    high_v: float = 1.1,
    low_v: float = 0.0,
    complement: bool = False,
) -> str:
    points: list[tuple[float, float]] = []

    def value_for(sample: dict[str, Any]) -> float:
        is_target = int(sample["label"]) == output
        if complement:
            is_target = not is_target
        return high_v if is_target else low_v

    for idx, sample in enumerate(samples):
        start = idx * CYCLE_NS
        end = start + CYCLE_NS
        value = value_for(sample)
        if idx == 0:
            points.append((0.0, value))
        else:
            points.append((start - 0.05, value_for(samples[idx - 1])))
            points.append((start, value))
        points.append((min(stop_ns, end - 0.05), value))
    points.append((stop_ns, value_for(samples[-1])))
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
    preboost_high_v: float | None = None,
    train_refire: bool = True,
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
    outg: list[tuple[float, float]] = []
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
        outg.append((base + 2.70, base + 3.00))
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
                if train_refire:
                    rstf.append((base + 12.05, base + 12.55))
                    fwd.append((base + 12.80, base + 15.60))
                    outg.append((base + 15.30, base + 15.60))
    sources = [
        f"Vrstf rstf 0 {pulse_wave(rstf, stop)}",
        f"Vrste rste 0 {pulse_wave(rste, stop)}",
        f"Vrstg rstg 0 {pulse_wave(rstg, stop)}",
        f"Vfwd fwd 0 {pulse_wave(fwd, stop)}",
        f"Voutg outg 0 {pulse_wave(outg, stop)}",
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
    if preboost_high_v is not None:
        sources.insert(7, f"Vpreboost preboost 0 {pulse_wave(bwd, stop, preboost_high_v)}")
    return "\n".join(sources)


def make_samples(
    records: list[dict[str, Any]],
    epochs: int,
    order: list[int],
    batch_apply: bool,
    eval_each_epoch: bool = False,
) -> list[dict[str, Any]]:
    by_pattern = {int(record["pattern"]): record for record in records}
    samples: list[dict[str, Any]] = []
    for record in records:
        samples.append({**record, "phase": "initial_eval", "epoch": 0})
    for epoch in range(1, epochs + 1):
        for pos, pattern in enumerate(order):
            record = by_pattern[pattern]
            samples.append(
                {
                    **record,
                    "phase": "train",
                    "epoch": epoch,
                    "reset_gradient": (not batch_apply) or pos == 0,
                    "apply_update": (not batch_apply) or pos == len(order) - 1,
                }
            )
        if eval_each_epoch:
            for record in records:
                samples.append({**record, "phase": f"epoch{epoch}_eval", "epoch": epoch})
    for record in records:
        samples.append({**record, "phase": "final_eval", "epoch": epochs})
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


def csv_readout_weights(path: Path) -> tuple[list[list[float]], list[float]]:
    df = pd.read_csv(path)
    weight_cols = [f"w{h}" for h in range(HIDDEN)]
    missing = [col for col in ["out", "bias", *weight_cols] if col not in df.columns]
    if missing:
        raise ValueError(f"programmed readout CSV is missing columns {missing}: {path}")
    weights = [[0.0 for _h in range(HIDDEN)] for _out in range(OUTPUTS)]
    biases = [0.0 for _out in range(OUTPUTS)]
    seen: set[int] = set()
    for row in df.itertuples(index=False):
        out = int(getattr(row, "out"))
        if not 0 <= out < OUTPUTS:
            raise ValueError(f"programmed readout CSV output {out} is outside 0..{OUTPUTS - 1}: {path}")
        if out in seen:
            raise ValueError(f"programmed readout CSV has duplicate output row {out}: {path}")
        seen.add(out)
        biases[out] = float(getattr(row, "bias"))
        weights[out] = [float(getattr(row, col)) for col in weight_cols]
    missing_outs = sorted(set(range(OUTPUTS)) - seen)
    if missing_outs:
        raise ValueError(f"programmed readout CSV is missing output rows {missing_outs}: {path}")
    return weights, biases


def csv_readout_cap_state(path: Path) -> dict[str, float]:
    return load_readout_cap_state_csv(path, hidden_count=HIDDEN, output_count=OUTPUTS)


def clamp_cap(v: float) -> float:
    return min(1.15, max(0.01, v))


def lead_class0_wins(lead_mode: str, lead01, lead10):
    if lead_mode == "out_senseamp":
        return lead10 > lead01
    return lead01 > lead10


def lead_win_gate(lead_mode: str, class_index: int) -> str:
    if class_index not in {0, 1}:
        raise ValueError(f"unknown class index: {class_index}")
    if lead_mode == "score_direct":
        return "score0" if class_index == 0 else "score1"
    if lead_mode == "out_senseamp":
        return "lead10" if class_index == 0 else "lead01"
    if lead_mode in {"score", "score_charge", "lose", "senseamp", "senseamp_strong"}:
        return "lead01" if class_index == 0 else "lead10"
    raise ValueError(f"unknown lead mode: {lead_mode}")


def target_loss_mask(train: pd.DataFrame, lead_mode: str = "score_direct") -> np.ndarray:
    if lead_mode != "score_direct" and {"lead01", "lead10"}.issubset(train.columns):
        score0_wins = lead_class0_wins(lead_mode, train["lead01"], train["lead10"])
    else:
        score0_wins = train["score0_cmp"] > train["score1_cmp"]
    target_is_class0 = train["label"].astype(int) == 0
    return np.where(target_is_class0, ~score0_wins, score0_wins)


def target_mistake_gate_stats(
    train: pd.DataFrame,
    lead_mode: str = "score_direct",
    bwd_threshold_v: float = 0.5,
) -> dict[str, Any]:
    required = {"label", "score0_cmp", "score1_cmp", "bwd_signal"}
    if train.empty or not required.issubset(train.columns):
        return {
            "target_mistake_bwd_threshold_v": bwd_threshold_v,
            "target_mistake_reference": None,
            "target_mistake_bwd_match_fraction": None,
            "target_mistake_bwd_false_positive_count": None,
            "target_mistake_bwd_false_negative_count": None,
            "target_mistake_score_loses_count": None,
            "target_mistake_bwd_open_count": None,
            "target_mistake_bwd_target_loses_mean_v": None,
            "target_mistake_bwd_target_wins_mean_v": None,
            "target_mistake_bwd_target_loses_min_v": None,
            "target_mistake_bwd_target_wins_max_v": None,
            "target_mistake_bwd_voltage_separation_v": None,
            "target_mistake_bwd_best_threshold_v": None,
            "target_mistake_bwd_best_threshold_match_fraction": None,
        }
    target_loses = target_loss_mask(train, lead_mode)
    bwd_signal = train["bwd_signal"].to_numpy()
    bwd_open = bwd_signal > bwd_threshold_v
    match = bwd_open == target_loses
    false_positive = bwd_open & ~target_loses
    false_negative = ~bwd_open & target_loses
    loses_signal = bwd_signal[target_loses]
    wins_signal = bwd_signal[~target_loses]
    thresholds = [-1.0]
    unique_signal = np.unique(bwd_signal)
    if len(unique_signal) > 1:
        thresholds.extend(float((a + b) / 2) for a, b in zip(unique_signal[:-1], unique_signal[1:]))
    thresholds.append(2.0)
    best_threshold = max(
        thresholds,
        key=lambda threshold: float(((bwd_signal > threshold) == target_loses).mean()),
    )
    best_match = float(((bwd_signal > best_threshold) == target_loses).mean())
    return {
        "target_mistake_bwd_threshold_v": bwd_threshold_v,
        "target_mistake_reference": lead_mode,
        "target_mistake_bwd_match_fraction": float(match.mean()),
        "target_mistake_bwd_false_positive_count": int(false_positive.sum()),
        "target_mistake_bwd_false_negative_count": int(false_negative.sum()),
        "target_mistake_score_loses_count": int(target_loses.sum()),
        "target_mistake_bwd_open_count": int(bwd_open.sum()),
        "target_mistake_bwd_target_loses_mean_v": float(loses_signal.mean()) if len(loses_signal) else None,
        "target_mistake_bwd_target_wins_mean_v": float(wins_signal.mean()) if len(wins_signal) else None,
        "target_mistake_bwd_target_loses_min_v": float(loses_signal.min()) if len(loses_signal) else None,
        "target_mistake_bwd_target_wins_max_v": float(wins_signal.max()) if len(wins_signal) else None,
        "target_mistake_bwd_voltage_separation_v": (
            float(loses_signal.min() - wins_signal.max()) if len(loses_signal) and len(wins_signal) else None
        ),
        "target_mistake_bwd_best_threshold_v": float(best_threshold),
        "target_mistake_bwd_best_threshold_match_fraction": best_match,
    }


def target_mistake_latch_stats(
    train: pd.DataFrame,
    lead_mode: str = "score_direct",
    latch_threshold_v: float = 0.5,
) -> dict[str, Any]:
    empty = {
        "target_mistake_latch_threshold_v": latch_threshold_v,
        "target_mistake_latch_reference": None,
        "target_mistake_latch_match_fraction": None,
        "target_mistake_latch_false_positive_count": None,
        "target_mistake_latch_false_negative_count": None,
        "target_mistake_latch_open_count": None,
        "target_mistake_latch_best_threshold_v": None,
        "target_mistake_latch_best_threshold_match_fraction": None,
    }
    if train.empty or not {"label", "score0_cmp", "score1_cmp", "merr0", "merr1"}.issubset(train.columns):
        return empty
    target_loses = target_loss_mask(train, lead_mode)
    labels = train["label"].astype(int).to_numpy()
    expected0 = target_loses & (labels == 0)
    expected1 = target_loses & (labels == 1)
    merr0 = train["merr0"].to_numpy()
    merr1 = train["merr1"].to_numpy()
    open0 = merr0 > latch_threshold_v
    open1 = merr1 > latch_threshold_v
    match = (open0 == expected0) & (open1 == expected1)
    any_open = open0 | open1
    false_positive = any_open & ~target_loses
    false_negative = ~any_open & target_loses
    latch_signal = np.maximum(merr0, merr1)
    thresholds = [-1.0]
    unique_signal = np.unique(latch_signal)
    if len(unique_signal) > 1:
        thresholds.extend(float((a + b) / 2) for a, b in zip(unique_signal[:-1], unique_signal[1:]))
    thresholds.append(2.0)
    best_threshold = max(
        thresholds,
        key=lambda threshold: float(((latch_signal > threshold) == target_loses).mean()),
    )
    best_match = float(((latch_signal > best_threshold) == target_loses).mean())
    return {
        "target_mistake_latch_threshold_v": latch_threshold_v,
        "target_mistake_latch_reference": lead_mode,
        "target_mistake_latch_match_fraction": float(match.mean()),
        "target_mistake_latch_false_positive_count": int(false_positive.sum()),
        "target_mistake_latch_false_negative_count": int(false_negative.sum()),
        "target_mistake_latch_open_count": int(any_open.sum()),
        "target_mistake_latch_best_threshold_v": float(best_threshold),
        "target_mistake_latch_best_threshold_match_fraction": best_match,
    }


def output_error_rail_stats(
    train: pd.DataFrame,
    lead_mode: str,
    rail_threshold_v: float = 0.5,
) -> dict[str, Any]:
    required = {"label", "score0_cmp", "score1_cmp", "dp0", "dn0", "dp1", "dn1"}
    if train.empty or not required.issubset(train.columns):
        return {
            "output_error_rail_threshold_v": rail_threshold_v,
            "output_error_rail_match_fraction": None,
            "output_error_rail_false_positive_count": None,
            "output_error_rail_false_negative_count": None,
            "output_error_rail_target_loses_count": None,
            "output_error_rail_open_count": None,
        }
    if lead_mode != "score_direct" and {"lead01", "lead10"}.issubset(train.columns):
        score0_wins = lead_class0_wins(lead_mode, train["lead01"], train["lead10"])
    else:
        score0_wins = train["score0_cmp"] > train["score1_cmp"]
    target_is_class0 = train["label"] == 0
    target_loses = np.where(target_is_class0, ~score0_wins, score0_wins)
    expected = pd.DataFrame(index=train.index)
    expected["dp0"] = target_loses & target_is_class0
    expected["dn0"] = target_loses & ~target_is_class0
    expected["dp1"] = target_loses & ~target_is_class0
    expected["dn1"] = target_loses & target_is_class0
    observed = train[["dp0", "dn0", "dp1", "dn1"]] > rail_threshold_v
    match = (observed == expected).all(axis=1)
    false_positive = observed & ~expected
    false_negative = expected & ~observed
    return {
        "output_error_rail_threshold_v": rail_threshold_v,
        "output_error_rail_match_fraction": float(match.mean()),
        "output_error_rail_false_positive_count": int(false_positive.to_numpy().sum()),
        "output_error_rail_false_negative_count": int(false_negative.to_numpy().sum()),
        "output_error_rail_target_loses_count": int(target_loses.sum()),
        "output_error_rail_open_count": int(observed.to_numpy().sum()),
    }


def readout_init(
    seed: int,
    mode: str,
    separator_scale: float,
    separator_offset_v: float,
    readout_center_v: float,
    random_center_v: float | None,
    random_span_v: float,
    random_pos_center_v: float | None,
    random_neg_center_v: float | None,
    random_pos_span_v: float | None,
    random_neg_span_v: float | None,
    separator_csv: Path | None,
    separator_phase: str,
) -> dict[str, float]:
    if mode in CAP_STATE_READOUT_INITS:
        if separator_csv is None:
            raise ValueError(f"--readout-init {mode} requires --separator-csv.")
        return csv_readout_cap_state(separator_csv)
    if mode in PROGRAMMED_READOUT_INITS:
        if separator_csv is None:
            raise ValueError(f"--readout-init {mode} requires --separator-csv.")
        weights_by_out, biases = csv_readout_weights(separator_csv)
        init: dict[str, float] = {}
        center = readout_center_v
        for out in range(OUTPUTS):
            bias_diff = separator_scale * biases[out]
            if mode == "csv_readout_rectified":
                init[f"vbo{out}p"] = clamp_cap(center + max(0.0, bias_diff))
                init[f"vbo{out}n"] = clamp_cap(center + max(0.0, -bias_diff))
            elif mode == "csv_readout_sparse_rectified":
                init[f"vbo{out}p"] = clamp_cap(center + max(0.0, bias_diff)) if bias_diff > 0.0 else SPARSE_READOUT_INACTIVE_V
                init[f"vbo{out}n"] = clamp_cap(center + max(0.0, -bias_diff)) if bias_diff < 0.0 else SPARSE_READOUT_INACTIVE_V
            else:
                init[f"vbo{out}p"] = clamp_cap(center + bias_diff / 2)
                init[f"vbo{out}n"] = clamp_cap(center - bias_diff / 2)
            for h, weight in enumerate(weights_by_out[out]):
                diff = separator_scale * weight
                if mode == "csv_readout_rectified":
                    init[f"vw{out}{h}p"] = clamp_cap(center + max(0.0, diff))
                    init[f"vw{out}{h}n"] = clamp_cap(center + max(0.0, -diff))
                elif mode == "csv_readout_sparse_rectified":
                    init[f"vw{out}{h}p"] = clamp_cap(center + diff) if diff > 0.0 else SPARSE_READOUT_INACTIVE_V
                    init[f"vw{out}{h}n"] = clamp_cap(center - diff) if diff < 0.0 else SPARSE_READOUT_INACTIVE_V
                else:
                    init[f"vw{out}{h}p"] = clamp_cap(center + diff / 2)
                    init[f"vw{out}{h}n"] = clamp_cap(center - diff / 2)
        return init
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
    if random_center_v is not None or random_pos_center_v is not None or random_neg_center_v is not None:
        pos_center = random_pos_center_v if random_pos_center_v is not None else random_center_v
        neg_center = random_neg_center_v if random_neg_center_v is not None else random_center_v
        pos_half_span = (random_pos_span_v if random_pos_span_v is not None else random_span_v) / 2.0
        neg_half_span = (random_neg_span_v if random_neg_span_v is not None else random_span_v) / 2.0
        for out in range(OUTPUTS):
            kp = out + seed * 7
            kn = out + seed * 11
            init[f"vbo{out}p"] = (
                clamp_cap(pos_center + pos_half_span * (2 * (((17 * kp + 3) % 101) / 100) - 1))
                if pos_center is not None
                else 0.66 - 0.02 * ((out + seed) % 2)
            )
            init[f"vbo{out}n"] = (
                clamp_cap(neg_center + neg_half_span * (2 * (((23 * kn + 13) % 101) / 100) - 1))
                if neg_center is not None
                else 0.52 + 0.02 * (out % 2)
            )
            for h in range(HIDDEN):
                k = out * HIDDEN + h + seed * 5
                default_p = 0.56 + 0.16 * (((29 * k + 5) % 101) / 100)
                default_n = 0.56 + 0.16 * (((43 * k + 17) % 101) / 100)
                p = (
                    pos_center + pos_half_span * (2 * (((29 * k + 5) % 101) / 100) - 1)
                    if pos_center is not None
                    else default_p
                )
                n = (
                    neg_center + neg_half_span * (2 * (((43 * k + 17) % 101) / 100) - 1)
                    if neg_center is not None
                    else default_n
                )
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
    if OUTPUTS != 2:
        raise ValueError("signed output-bias offset is only defined for binary runs.")
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


def signed_weight_pairs(
    scope: str,
    readout_fanins: ReadoutFanins | None = None,
    hidden_fanins: HiddenFanins | None = None,
) -> list[tuple[str, str, str]]:
    if scope not in TRAIN_CHARGE_NOISE_SCOPES:
        raise ValueError(f"unknown signed-weight scope: {scope}")
    readout_edges = effective_readout_fanins(readout_fanins)
    hidden_edges = effective_hidden_fanins(hidden_fanins)
    pairs: list[tuple[str, str, str]] = []
    if scope in {"readout", "all"}:
        for out in range(OUTPUTS):
            pairs.append((f"vbo{out}", f"vbo{out}p", f"vbo{out}n"))
            for h in readout_edges[out]:
                pairs.append((f"vw{out}{h}", f"vw{out}{h}p", f"vw{out}{h}n"))
    if scope in {"hidden", "all"}:
        for h in range(HIDDEN):
            for rail in hidden_edges[h]:
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
    readout_fanins: ReadoutFanins | None = None,
    hidden_fanins: HiddenFanins | None = None,
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
    signed_pairs = signed_weight_pairs(scope, readout_fanins, hidden_fanins)
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


def persistent_caps(
    hidden: dict[str, float],
    readout: dict[str, float],
    cap_f: float,
    readout_fanins: ReadoutFanins | None = None,
    hidden_fanins: HiddenFanins | None = None,
) -> str:
    fanins = effective_readout_fanins(readout_fanins)
    hidden_edges = effective_hidden_fanins(hidden_fanins)
    deck = NetlistBuilder()
    hidden_bases: list[str] = []
    for h in range(HIDDEN):
        for rail in hidden_edges[h]:
            hidden_bases.append(f"wh{h}_{rail}")
    deck.render_component(
        DifferentialCapStateArray(
            "hidden_weight_caps",
            tuple(hidden_bases),
            hidden,
            cap_f,
        )
    )
    readout_bases: list[str] = []
    for out in range(OUTPUTS):
        readout_bases.append(f"vbo{out}")
        for h in fanins[out]:
            readout_bases.append(f"vw{out}{h}")
    deck.render_component(
        DifferentialCapStateArray(
            "readout_weight_caps",
            tuple(readout_bases),
            readout,
            cap_f,
        )
    )
    return deck.render_body()


def feedback_caps(
    feedback: dict[str, float],
    cap_f: float,
    readout_fanins: ReadoutFanins | None = None,
) -> str:
    fanins = effective_readout_fanins(readout_fanins)
    lines: list[str] = []
    for out in range(OUTPUTS):
        for h in fanins[out]:
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
    score_cap_f: float = 10.0,
    output_cap_f: float = 20.0,
    output_head: str = "source_follower",
    readout_fanins: ReadoutFanins | None = None,
    hidden_fanins: HiddenFanins | None = None,
) -> str:
    deck = NetlistBuilder()
    fanins = effective_readout_fanins(readout_fanins)
    hidden_edges = effective_hidden_fanins(hidden_fanins)
    if output_head not in OUTPUT_HEAD_MODES:
        raise ValueError(f"unknown output head: {output_head}")
    if output_head in COMMON_MODE_OUT_RESET_HEADS:
        out_ic = score_reset_v
        out_leak_node = "scorecm"
    elif output_head in LOW_TRUE_OUTPUT_HEADS:
        out_ic = 1.2
        out_leak_node = "vdd"
    else:
        out_ic = 0.0
        out_leak_node = "0"
    for h in range(HIDDEN):
        deck.render_component(
            CapStateArray(
                f"hidden_dynamic_caps_{h}",
                (
                    CapState(f"pre{h}", f"pre{h}", 10.0, 0.0, "0", "1G"),
                    CapState(f"act{h}", f"act{h}", 20.0, 0.0, "0", "1G"),
                    CapState(f"hdp{h}", f"hdp{h}", hidden_delta_cap_f, 0.0, "0", "1G"),
                    CapState(f"hdn{h}", f"hdn{h}", hidden_delta_cap_f, 0.0, "0", "1G"),
                ),
            )
        )
        if include_gradient_caps:
            deck.render_component(
                CapStateArray.from_nodes(
                    f"hidden_gradient_caps_{h}",
                    tuple(node for rail in hidden_edges[h] for node in (f"ghp{h}_{rail}", f"ghn{h}_{rail}")),
                    cap_f=hidden_gradient_cap_f,
                    ic_v=0.0,
                    leak_to="0",
                    leak_ohm="1G",
                )
            )
    for out in range(OUTPUTS):
        deck.render_component(
            CapStateArray(
                f"output_dynamic_caps_{out}",
                (
                    CapState(f"score{out}", f"score{out}", score_cap_f, score_reset_v, "0", "1G"),
                    CapState(f"scorep{out}", f"scorep{out}", score_cap_f, score_reset_v, "0", "1G"),
                    CapState(f"scoren{out}", f"scoren{out}", score_cap_f, score_reset_v, "0", "1G"),
                    CapState(f"out{out}", f"out{out}", output_cap_f, out_ic, out_leak_node, "1G"),
                    CapState(f"dp{out}", f"dp{out}", 20.0, 0.0, "0", "1G"),
                    CapState(f"dn{out}", f"dn{out}", 20.0, 0.0, "0", "1G"),
                    CapState(f"ybar{out}", f"ybar{out}", 20.0, 1.2, "0", "1G"),
                ),
            )
        )
        if include_gradient_caps:
            readout_gradient_nodes = [f"gvpb{out}", f"gvnb{out}"]
            for h in fanins[out]:
                readout_gradient_nodes.extend([f"gvp{out}{h}", f"gvn{out}{h}"])
            deck.render_component(
                CapStateArray.from_nodes(
                    f"readout_gradient_caps_{out}",
                    tuple(readout_gradient_nodes),
                    cap_f=gradient_cap_f,
                    ic_v=0.0,
                    leak_to="0",
                    leak_ohm="1G",
                )
            )
    deck.render_component(
        CapStateArray.from_nodes(
            "lead_caps",
            ("lead01", "lead10"),
            cap_f=lead_cap_f,
            ic_v=0.0,
            leak_to="0",
            leak_ohm="1G",
        )
    )
    return deck.render_body()


def resets(
    lead_mode: str,
    include_gradient_resets: bool,
    score_reset_v: float,
    output_head: str = "source_follower",
    readout_fanins: ReadoutFanins | None = None,
    hidden_fanins: HiddenFanins | None = None,
) -> str:
    lines: list[str] = []
    fanins = effective_readout_fanins(readout_fanins)
    hidden_edges = effective_hidden_fanins(hidden_fanins)
    if output_head not in OUTPUT_HEAD_MODES:
        raise ValueError(f"unknown output head: {output_head}")
    if output_head in COMMON_MODE_OUT_RESET_HEADS:
        out_reset_node = "scorecm"
    elif output_head in LOW_TRUE_OUTPUT_HEADS:
        out_reset_node = "vdd"
    else:
        out_reset_node = "0"
    for h in range(HIDDEN):
        lines += [
            f"Mreset_pre{h} pre{h} rstf 0 0 NMOS W=4u L=180n",
            f"Mreset_act{h} act{h} rstf 0 0 NMOS W=4u L=180n",
            f"Mreset_hdp{h} hdp{h} rste 0 0 NMOS W=4u L=180n",
            f"Mreset_hdn{h} hdn{h} rste 0 0 NMOS W=4u L=180n",
        ]
        if include_gradient_resets:
            for rail in hidden_edges[h]:
                lines += [
                    f"Mreset_ghp{h}_{rail} ghp{h}_{rail} rstg 0 0 NMOS W=4u L=180n",
                    f"Mreset_ghn{h}_{rail} ghn{h}_{rail} rstg 0 0 NMOS W=4u L=180n",
                ]
    for out in range(OUTPUTS):
        lines += [
            f"Mreset_score{out} score{out} rstf scorecm 0 NMOS W=4u L=180n",
            f"Mreset_scorep{out} scorep{out} rstf scorecm 0 NMOS W=4u L=180n",
            f"Mreset_scoren{out} scoren{out} rstf scorecm 0 NMOS W=4u L=180n",
            f"Mreset_out{out} out{out} rstf {out_reset_node} 0 NMOS W=4u L=180n",
            f"Mreset_dp{out} dp{out} rste 0 0 NMOS W=4u L=180n",
            f"Mreset_dn{out} dn{out} rste 0 0 NMOS W=4u L=180n",
            f"Mreset_ybar{out}_high vdd rste ybar{out} 0 NSENSE W=32u L=180n",
        ]
        if include_gradient_resets:
            lines += [
                f"Mreset_gvpb{out} gvpb{out} rstg 0 0 NMOS W=4u L=180n",
                f"Mreset_gvnb{out} gvnb{out} rstg 0 0 NMOS W=4u L=180n",
            ]
            for h in fanins[out]:
                lines += [
                    f"Mreset_gvp{out}{h} gvp{out}{h} rstg 0 0 NMOS W=4u L=180n",
                    f"Mreset_gvn{out}{h} gvn{out}{h} rstg 0 0 NMOS W=4u L=180n",
                ]
    if lead_mode in {"senseamp", "senseamp_strong", "out_senseamp"}:
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


def hidden_forward(
    design: SynapseDesign,
    hidden_forward_mode: str,
    hidden_fanins: HiddenFanins | None = None,
) -> str:
    if hidden_forward_mode not in HIDDEN_FORWARD_MODES:
        raise ValueError(f"unknown hidden forward mode: {hidden_forward_mode}")
    fanins = effective_hidden_fanins(hidden_fanins)
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
        lines.append(f"* General hidden {h}: signed conductance from {len(fanins[h])} selected input/bias rails.")
        for rail in fanins[h]:
            if hidden_forward_mode == "weighted_relu_pass_input":
                lines += [
                    f"Mh{h}_{rail}p_w {rail} wh{h}_{rail}p h{h}_{rail}p1 0 NREL W={syn_w:.12g}u L=180n",
                    f"Mh{h}_{rail}p_f h{h}_{rail}p1 fwd pre{h} 0 NREL W={syn_w:.12g}u L=180n",
                    f"Mh{h}_{rail}n_f pre{h} fwd h{h}_{rail}n0 0 NREL W={syn_w:.12g}u L=180n",
                    f"Mh{h}_{rail}n_x h{h}_{rail}n0 {rail} h{h}_{rail}n1 0 NREL W={syn_w:.12g}u L=180n",
                    f"Mh{h}_{rail}n_w h{h}_{rail}n1 wh{h}_{rail}n 0 0 NREL W={syn_w:.12g}u L=180n",
                    f"Rh{h}_{rail}p1 h{h}_{rail}p1 0 1e9",
                    f"Rh{h}_{rail}n0 h{h}_{rail}n0 0 1e9",
                    f"Rh{h}_{rail}n1 h{h}_{rail}n1 0 1e9",
                    f"Ch{h}_{rail}p1 h{h}_{rail}p1 0 0.02f IC=0",
                    f"Ch{h}_{rail}n0 h{h}_{rail}n0 0 0.02f IC=0",
                    f"Ch{h}_{rail}n1 h{h}_{rail}n1 0 0.02f IC=0",
                ]
            else:
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


def output_head_from_scores(
    design: SynapseDesign,
    output_head: str,
    out: int,
    score_diode_width_u: float = 1024.0,
    score_mirror_cap_f: float = 20.0,
) -> str:
    """Production score-cap to output-cap conversion cell for one class."""
    lines: list[str] = []
    if output_head == "source_follower":
        lines.append(f"Mrelu_o{out} vdd score{out} out{out} 0 NSENSE W={design.output_relu_width_u:.12g}u L=180n")
    elif output_head == "split_score_none":
        lines.append(
            (
                f"* Score-rail-only split output: scorep{out}/scoren{out} are the class output. "
                "No score-to-out conversion cell is attached, so the readout score capacitors are not loaded."
            )
        )
    elif output_head == "split_score_caps":
        pos_mid = f"out{out}_split_pos"
        neg_mid = f"out{out}_split_neg"
        lines += [
            f"* Differential score-cap head: scorep{out} charges out{out}; scoren{out} discharges it.",
            f"Mout{out}_split_pos_s vdd scorep{out} {pos_mid} 0 NSENSE W={design.output_relu_width_u:.12g}u L=180n",
            f"Mout{out}_split_pos_f {pos_mid} fwd out{out} 0 NREL W={design.output_relu_width_u:.12g}u L=180n",
            f"Mout{out}_split_neg_f out{out} fwd {neg_mid} 0 NREL W={design.output_relu_width_u:.12g}u L=180n",
            f"Mout{out}_split_neg_s {neg_mid} scoren{out} 0 0 NSENSE W={design.output_relu_width_u:.12g}u L=180n",
        ]
        lines += node_parasitics(pos_mid, neg_mid)
    elif output_head == "split_score_diffgate":
        pos_low = f"out{out}_diffgate_pos_low"
        pos_mid = f"out{out}_diffgate_pos_mid"
        neg_low = f"out{out}_diffgate_neg_low"
        neg_mid = f"out{out}_diffgate_neg_mid"
        lines += [
            (
                f"* Differential-gated split-score head: scorep{out} charges out{out} only when "
                f"scoren{out} is low; scoren{out} discharges it only when scorep{out} is low."
            ),
            f"Mout{out}_dg_pos_inhibit vdd scoren{out} {pos_low} vdd PMOS W={design.output_relu_width_u:.12g}u L=180n",
            f"Mout{out}_dg_pos_score {pos_low} scorep{out} {pos_mid} 0 NREL W={design.output_relu_width_u:.12g}u L=180n",
            f"Mout{out}_dg_pos_f {pos_mid} fwd out{out} 0 NREL W={design.output_relu_width_u:.12g}u L=180n",
            f"Mout{out}_dg_neg_inhibit out{out} scorep{out} {neg_low} vdd PMOS W={design.output_relu_width_u:.12g}u L=180n",
            f"Mout{out}_dg_neg_score {neg_low} scoren{out} {neg_mid} 0 NREL W={design.output_relu_width_u:.12g}u L=180n",
            f"Mout{out}_dg_neg_f {neg_mid} fwd 0 0 NREL W={design.output_relu_width_u:.12g}u L=180n",
        ]
        lines += node_parasitics(pos_low, pos_mid, neg_low, neg_mid)
    elif output_head == "split_score_chargegate":
        pos_low = f"out{out}_chargegate_pos_low"
        pos_mid = f"out{out}_chargegate_pos_mid"
        lines += [
            (
                f"* Unipolar split-score charge gate: scorep{out} charges out{out} only when "
                f"scoren{out} is low.  There is no local discharge leg; reset defines the low state."
            ),
            f"Mout{out}_cg_inhibit vdd scoren{out} {pos_low} vdd PMOS W={design.output_relu_width_u:.12g}u L=180n",
            f"Mout{out}_cg_score {pos_low} scorep{out} {pos_mid} 0 NREL W={design.output_relu_width_u:.12g}u L=180n",
            f"Mout{out}_cg_f {pos_mid} fwd out{out} 0 NREL W={design.output_relu_width_u:.12g}u L=180n",
        ]
        lines += node_parasitics(pos_low, pos_mid)
    elif output_head == "split_score_diffpair":
        pairsrc = f"out{out}_dpair_src"
        lines += [
            (
                f"* Source-coupled split-score output: scorep{out}/scoren{out} first compete in "
                f"a local differential pair.  out{out} is active-low: the positive branch discharges "
                f"it when the signed score is high."
            ),
            f"Rout{out}_dpair_pull out{out} vdd 1e12",
            f"Mout{out}_dpair_pos out{out} scorep{out} {pairsrc} 0 NREL W={design.output_relu_width_u:.12g}u L=180n",
            f"Mout{out}_dpair_neg vdd scoren{out} {pairsrc} 0 NREL W={design.output_relu_width_u:.12g}u L=180n",
            f"Mout{out}_dpair_tail {pairsrc} fwd 0 0 NMOS W={design.output_relu_width_u:.12g}u L=180n",
        ]
        lines += node_parasitics(pairsrc)
    elif output_head == "split_score_diode_diffpair":
        pairsrc = f"out{out}_ddpair_src"
        lines += [
            (
                f"* Diode-loaded source-coupled split-score output: scorep{out}/scoren{out} are "
                f"loaded by diode-connected MOS devices, approximating a current-mirror input before "
                f"the local active-low differential output stage."
            ),
            f"Mscorep{out}_diode scorep{out} scorep{out} 0 0 NSENSE W={score_diode_width_u:.12g}u L=180n",
            f"Mscoren{out}_diode scoren{out} scoren{out} 0 0 NSENSE W={score_diode_width_u:.12g}u L=180n",
            f"Rout{out}_ddpair_pull out{out} vdd 1e12",
            f"Mout{out}_ddpair_pos out{out} scorep{out} {pairsrc} 0 NREL W={design.output_relu_width_u:.12g}u L=180n",
            f"Mout{out}_ddpair_neg vdd scoren{out} {pairsrc} 0 NREL W={design.output_relu_width_u:.12g}u L=180n",
            f"Mout{out}_ddpair_tail {pairsrc} fwd 0 0 NMOS W={design.output_relu_width_u:.12g}u L=180n",
        ]
        lines += node_parasitics(pairsrc)
    elif output_head == "split_score_compete_tail":
        mid = f"out{out}_ctail_mid"
        lines += [
            (
                f"* Shared-tail split-score current competition output: scorep{out} opens the "
                f"class branch while scoren{out} inhibits it through a PMOS source device. "
                f"out{out} is active-low; all classes share one tail current."
            ),
            f"Rout{out}_ctail_pull out{out} vdd 1e12",
            f"Mout{out}_ctail_inhibit {mid} scoren{out} out{out} vdd PMOS W={design.output_relu_width_u:.12g}u L=180n",
            f"Mout{out}_ctail_score {mid} scorep{out} out_compete_src 0 NREL W={design.output_relu_width_u:.12g}u L=180n",
        ]
        lines += node_parasitics(mid)
    elif output_head == "split_score_diode_mirror_gate_caps":
        mid = f"out{out}_dmgate_mid"
        pairsrc = f"out{out}_dmgate_src"
        lines += [
            (
                f"* Diode/mirror gated split-score output: scorep{out}/scoren{out} discharge "
                f"mirror caps scorepm{out}/scorenm{out}.  out{out} is active-low and discharges "
                f"only when scorepm{out} is low and scorenm{out} is high, matching the measured "
                f"current-derived score scorenm-scorepm without a source-coupled dump branch."
            ),
            f"Mscorep{out}_diode scorep{out} scorep{out} 0 0 NSENSE W={score_diode_width_u:.12g}u L=180n",
            f"Mscoren{out}_diode scoren{out} scoren{out} 0 0 NSENSE W={score_diode_width_u:.12g}u L=180n",
            f"Cscorepm{out} scorepm{out} 0 {score_mirror_cap_f:.12g}f IC=1.2",
            f"Cscorenm{out} scorenm{out} 0 {score_mirror_cap_f:.12g}f IC=1.2",
            f"Rscorepm{out} scorepm{out} 0 1e12",
            f"Rscorenm{out} scorenm{out} 0 1e12",
            f"Mreset_scorepm{out}_high vdd rstf scorepm{out} 0 NSENSE W=16u L=180n",
            f"Mreset_scorenm{out}_high vdd rstf scorenm{out} 0 NSENSE W=16u L=180n",
            f"Mscorep{out}_mirror scorepm{out} scorep{out} 0 0 NSENSE W={score_diode_width_u:.12g}u L=180n",
            f"Mscoren{out}_mirror scorenm{out} scoren{out} 0 0 NSENSE W={score_diode_width_u:.12g}u L=180n",
            f"Rout{out}_dmgate_pull out{out} vdd 1e12",
            f"Mout{out}_dmgate_inhibit out{out} scorepm{out} {mid} vdd PMOS W={design.output_relu_width_u:.12g}u L=180n",
            f"Mout{out}_dmgate_score {mid} scorenm{out} {pairsrc} 0 NREL W={design.output_relu_width_u:.12g}u L=180n",
            f"Mout{out}_dmgate_tail {pairsrc} outg 0 0 NMOS W={design.output_relu_width_u:.12g}u L=180n",
        ]
        lines += node_parasitics(mid, pairsrc)
    elif output_head == "split_score_diode_mirror_caps":
        pairsrc = f"out{out}_dmcap_src"
        lines += [
            (
                f"* Diode/mirror split-score output: scorep{out}/scoren{out} are low-impedance "
                f"diode input nodes.  Matched mirror sinks discharge local caps scorepm{out}/"
                f"scorenm{out}; the output pair then classifies from those current-derived "
                f"mirror-cap voltages instead of directly from the compressed diode voltages."
            ),
            f"Mscorep{out}_diode scorep{out} scorep{out} 0 0 NSENSE W={score_diode_width_u:.12g}u L=180n",
            f"Mscoren{out}_diode scoren{out} scoren{out} 0 0 NSENSE W={score_diode_width_u:.12g}u L=180n",
            f"Cscorepm{out} scorepm{out} 0 {score_mirror_cap_f:.12g}f IC=1.2",
            f"Cscorenm{out} scorenm{out} 0 {score_mirror_cap_f:.12g}f IC=1.2",
            f"Rscorepm{out} scorepm{out} 0 1e12",
            f"Rscorenm{out} scorenm{out} 0 1e12",
            f"Mreset_scorepm{out}_high vdd rstf scorepm{out} 0 NSENSE W=16u L=180n",
            f"Mreset_scorenm{out}_high vdd rstf scorenm{out} 0 NSENSE W=16u L=180n",
            f"Mscorep{out}_mirror scorepm{out} scorep{out} 0 0 NSENSE W={score_diode_width_u:.12g}u L=180n",
            f"Mscoren{out}_mirror scorenm{out} scoren{out} 0 0 NSENSE W={score_diode_width_u:.12g}u L=180n",
            f"Rout{out}_dmcap_pull out{out} vdd 1e12",
            f"Mout{out}_dmcap_pos out{out} scorenm{out} {pairsrc} 0 NREL W={design.output_relu_width_u:.12g}u L=180n",
            f"Mout{out}_dmcap_neg vdd scorepm{out} {pairsrc} 0 NREL W={design.output_relu_width_u:.12g}u L=180n",
            f"Mout{out}_dmcap_tail {pairsrc} outg 0 0 NMOS W={design.output_relu_width_u:.12g}u L=180n",
        ]
        lines += node_parasitics(pairsrc)
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
    else:
        raise ValueError(f"unknown output head: {output_head}")
    return "\n".join(lines)


def output_head_shared_cells(design: SynapseDesign, output_head: str) -> str:
    """Shared score-to-output cells that are emitted once per output layer."""
    if output_head == "split_score_compete_tail":
        lines = [
            "* Shared tail for split-score current competition output head.",
            f"Mout_ctail_tail out_compete_src outg 0 0 NMOS W={design.output_relu_width_u:.12g}u L=180n",
        ]
        lines += node_parasitics("out_compete_src")
        return "\n".join(lines)
    return ""


def output_forward(
    design: SynapseDesign,
    output_head: str,
    score_diode_width_u: float = 1024.0,
    score_mirror_cap_f: float = 20.0,
    readout_fanins: ReadoutFanins | None = None,
) -> str:
    if output_head not in OUTPUT_HEAD_MODES:
        raise ValueError(f"unknown output head: {output_head}")
    if output_head == "score_diff" and OUTPUTS != 2:
        raise ValueError("score_diff output head is only defined for two output classes.")
    fanins = effective_readout_fanins(readout_fanins)
    missing_outputs = [out for out in range(OUTPUTS) if out not in fanins]
    if missing_outputs:
        raise ValueError(f"readout fanins missing outputs: {missing_outputs}")
    for out, srcs in fanins.items():
        for h in srcs:
            if h < 0 or h >= HIDDEN:
                raise ValueError(f"readout fanin hidden index {h} for output {out} is outside 0..{HIDDEN - 1}.")
    lines: list[str] = []
    if design.output_forward_style == "pass_act_buffered":
        lines.append("* Buffered hidden activation replicas for voltage-mode readout.")
        active_hidden = sorted({h for srcs in fanins.values() for h in srcs})
        for h in active_hidden:
            lines += [
                f"Mactbuf{h}_src vdd act{h} actbuf{h} 0 NSENSE W={design.output_forward_pos_width_u:.12g}u L=180n",
                f"Mactbuf{h}_rst actbuf{h} rstf 0 0 NREL W=4u L=180n",
            ]
            lines += node_parasitics(f"actbuf{h}")
    for out in range(OUTPUTS):
        lines.append(f"* Output {out}: signed readout from {len(fanins[out])} general hidden activations.")
        for h in fanins[out]:
            readout_internal_nodes = [
                f"o{out}{h}p0",
                f"o{out}{h}p1",
                f"o{out}{h}n0",
                f"o{out}{h}n1",
            ]
            if design.output_forward_style == "gate_stack":
                if output_head in SPLIT_SCORE_OUTPUT_HEADS:
                    lines += [
                        f"Mo{out}{h}pos_a vdd act{h} o{out}{h}p0 0 NSENSE W={design.output_forward_pos_width_u:.12g}u L=180n",
                        f"Mo{out}{h}pos_w o{out}{h}p0 vw{out}{h}p o{out}{h}p1 0 NREL W={design.output_forward_pos_width_u:.12g}u L=180n",
                        f"Mo{out}{h}pos_f o{out}{h}p1 fwd scorep{out} 0 NREL W={design.output_forward_pos_width_u:.12g}u L=180n",
                        f"Mo{out}{h}neg_a vdd act{h} o{out}{h}n0 0 NSENSE W={design.output_forward_neg_width_u:.12g}u L=180n",
                        f"Mo{out}{h}neg_w o{out}{h}n0 vw{out}{h}n o{out}{h}n1 0 NREL W={design.output_forward_neg_width_u:.12g}u L=180n",
                        f"Mo{out}{h}neg_f o{out}{h}n1 fwd scoren{out} 0 NREL W={design.output_forward_neg_width_u:.12g}u L=180n",
                    ]
                else:
                    lines += [
                        f"Mo{out}{h}pos_a vdd act{h} o{out}{h}p0 0 NSENSE W={design.output_forward_pos_width_u:.12g}u L=180n",
                        f"Mo{out}{h}pos_w o{out}{h}p0 vw{out}{h}p o{out}{h}p1 0 NREL W={design.output_forward_pos_width_u:.12g}u L=180n",
                        f"Mo{out}{h}pos_f o{out}{h}p1 fwd score{out} 0 NREL W={design.output_forward_pos_width_u:.12g}u L=180n",
                        f"Mo{out}{h}neg_f score{out} fwd o{out}{h}n0 0 NREL W={design.output_forward_neg_width_u:.12g}u L=180n",
                        f"Mo{out}{h}neg_a o{out}{h}n0 act{h} o{out}{h}n1 0 NSENSE W={design.output_forward_neg_width_u:.12g}u L=180n",
                        f"Mo{out}{h}neg_w o{out}{h}n1 vw{out}{h}n 0 0 NREL W={design.output_forward_neg_width_u:.12g}u L=180n",
                    ]
            elif design.output_forward_style in {"pass_act_source", "pass_act_buffered"}:
                act_source = f"actbuf{h}" if design.output_forward_style == "pass_act_buffered" else f"act{h}"
                if output_head in SPLIT_SCORE_OUTPUT_HEADS:
                    lines += [
                        f"Mo{out}{h}pos_w {act_source} vw{out}{h}p o{out}{h}p1 0 NREL W={design.output_forward_pos_width_u:.12g}u L=180n",
                        f"Mo{out}{h}pos_f o{out}{h}p1 fwd scorep{out} 0 NREL W={design.output_forward_pos_width_u:.12g}u L=180n",
                        f"Mo{out}{h}neg_w {act_source} vw{out}{h}n o{out}{h}n1 0 NREL W={design.output_forward_neg_width_u:.12g}u L=180n",
                        f"Mo{out}{h}neg_f o{out}{h}n1 fwd scoren{out} 0 NREL W={design.output_forward_neg_width_u:.12g}u L=180n",
                    ]
                else:
                    lines += [
                        f"Mo{out}{h}pos_w {act_source} vw{out}{h}p o{out}{h}p1 0 NREL W={design.output_forward_pos_width_u:.12g}u L=180n",
                        f"Mo{out}{h}pos_f o{out}{h}p1 fwd score{out} 0 NREL W={design.output_forward_pos_width_u:.12g}u L=180n",
                        f"Mo{out}{h}neg_f score{out} fwd o{out}{h}n0 0 NREL W={design.output_forward_neg_width_u:.12g}u L=180n",
                        f"Mo{out}{h}neg_a o{out}{h}n0 act{h} o{out}{h}n1 0 NSENSE W={design.output_forward_neg_width_u:.12g}u L=180n",
                        f"Mo{out}{h}neg_w o{out}{h}n1 vw{out}{h}n 0 0 NREL W={design.output_forward_neg_width_u:.12g}u L=180n",
                    ]
            else:
                raise ValueError(f"unknown output forward style: {design.output_forward_style}")
            lines += node_parasitics(*readout_internal_nodes)
        bias_internal_nodes = [f"o{out}bp0", f"o{out}bp1", f"o{out}bn0", f"o{out}bn1"]
        if output_head in SPLIT_SCORE_OUTPUT_HEADS:
            if design.output_forward_style == "gate_stack":
                lines += [
                    f"Mo{out}bpos_a vdd bias o{out}bp0 0 NSENSE W={design.output_bias_forward_pos_width_u:.12g}u L=180n",
                    f"Mo{out}bpos_w o{out}bp0 vbo{out}p o{out}bp1 0 NREL W={design.output_bias_forward_pos_width_u:.12g}u L=180n",
                    f"Mo{out}bpos_f o{out}bp1 fwd scorep{out} 0 NREL W={design.output_bias_forward_pos_width_u:.12g}u L=180n",
                    f"Mo{out}bneg_a vdd bias o{out}bn0 0 NSENSE W={design.output_bias_forward_neg_width_u:.12g}u L=180n",
                    f"Mo{out}bneg_w o{out}bn0 vbo{out}n o{out}bn1 0 NREL W={design.output_bias_forward_neg_width_u:.12g}u L=180n",
                    f"Mo{out}bneg_f o{out}bn1 fwd scoren{out} 0 NREL W={design.output_bias_forward_neg_width_u:.12g}u L=180n",
                ]
            else:
                lines += [
                    f"Mo{out}bpos_w bias vbo{out}p o{out}bp1 0 NREL W={design.output_bias_forward_pos_width_u:.12g}u L=180n",
                    f"Mo{out}bpos_f o{out}bp1 fwd scorep{out} 0 NREL W={design.output_bias_forward_pos_width_u:.12g}u L=180n",
                    f"Mo{out}bneg_w bias vbo{out}n o{out}bn1 0 NREL W={design.output_bias_forward_neg_width_u:.12g}u L=180n",
                    f"Mo{out}bneg_f o{out}bn1 fwd scoren{out} 0 NREL W={design.output_bias_forward_neg_width_u:.12g}u L=180n",
                ]
                bias_internal_nodes = [f"o{out}bp1", f"o{out}bn1"]
        else:
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
        lines += node_parasitics(*bias_internal_nodes)
        lines.append(output_head_from_scores(design, output_head, out, score_diode_width_u, score_mirror_cap_f))
    shared_head_cells = output_head_shared_cells(design, output_head)
    if shared_head_cells:
        lines.append(shared_head_cells)
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
    return NodeParasitics("parasitics", tuple(nodes)).render_lines()


def diffpair_bleed_write_selector_lines(
    prefix: str,
    positive_error_gate: str,
    negative_error_gate: str,
    positive_write_gate: str,
    negative_write_gate: str,
    width_u: float,
    label: str,
) -> list[str]:
    """Build high-true write rails from a differential error comparison.

    The older pmos_inhibit selector treats dp and dn mostly as two independent
    absolute gates.  This cell adds a shared-tail comparison stage: common-mode
    dp/dn drives both internal bar nodes similarly, and the weak bwd-gated
    output bleeds keep both write rails low unless one side wins strongly.
    """
    return DiffPairBleedWriteSelector(
        prefix,
        positive_error_gate,
        negative_error_gate,
        positive_write_gate,
        negative_write_gate,
        width_u,
        label,
    ).render_lines()


def score_lead_gate_cells(lead_width_u: float, lead_mode: str) -> str:
    if OUTPUTS != 2 and lead_mode != "score_direct":
        raise ValueError("only score_direct lead mode is available for multi-class device runs.")
    if lead_mode == "score_direct":
        return "\n".join(
            [
                "* Direct score lead mode: target-mistake stacks use score0/score1",
                "* capacitor voltages as their winner gates, without an intermediate lead latch.",
            ]
        )
    if lead_mode == "score_charge":
        return "\n".join(
            [
                "* Charge-only score lead: lead01/lead10 start low and are charged by score0/score1 during compare.",
                "* This avoids the score-mode discharge path that can cancel small score voltages.",
                f"Mlead01_up_s vdd score0 lead01_up 0 NSENSE W={lead_width_u:.12g}u L=180n",
                f"Mlead01_up_e lead01_up cmp lead01 0 NSENSE W={lead_width_u:.12g}u L=180n",
                f"Mlead10_up_s vdd score1 lead10_up 0 NSENSE W={lead_width_u:.12g}u L=180n",
                f"Mlead10_up_e lead10_up cmp lead10 0 NSENSE W={lead_width_u:.12g}u L=180n",
            ]
            + node_parasitics("lead01_up", "lead10_up")
        )
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
    if lead_mode in {"senseamp", "senseamp_strong"}:
        keeper_width_u = max(1.0, lead_width_u / 64.0)
        lines = [
            "* Dynamic score sense amp: rste precharges both lead nodes high; cmp discharges the losing side.",
            f"Mlead01_dis_s lead01 score1 lead01_dn 0 NSENSE W={lead_width_u:.12g}u L=180n",
            f"Mlead01_dis_e lead01_dn cmp 0 0 NSENSE W={lead_width_u:.12g}u L=180n",
            f"Mlead10_dis_s lead10 score0 lead10_dn 0 NSENSE W={lead_width_u:.12g}u L=180n",
            f"Mlead10_dis_e lead10_dn cmp 0 0 NSENSE W={lead_width_u:.12g}u L=180n",
            f"Mlead01_keep lead01 lead10 vdd vdd PMOS W={keeper_width_u:.12g}u L=180n",
            f"Mlead10_keep lead10 lead01 vdd vdd PMOS W={keeper_width_u:.12g}u L=180n",
        ]
        if lead_mode == "senseamp_strong":
            lines += [
                "* Strong variant adds cross-coupled NMOS pull-downs for regenerative lead separation.",
                f"Mlead01_nkeep lead01 lead10 0 0 NMOS W={keeper_width_u:.12g}u L=180n",
                f"Mlead10_nkeep lead10 lead01 0 0 NMOS W={keeper_width_u:.12g}u L=180n",
            ]
        return "\n".join(lines + node_parasitics("lead01_dn", "lead10_dn"))
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
    if OUTPUTS != 2 and mode != "scheduled":
        raise ValueError("self-timed/target-mistake backward gates are currently binary; use scheduled for multi-class.")
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
    if mode == "target_mistake_latch_simple":
        target0_wins_gate = lead_win_gate(lead_mode, 0)
        target1_wins_gate = lead_win_gate(lead_mode, 1)
        return "\n".join(
            [
                "* Short-stack latched mistake gate: trusts a regenerated winner lead and captures",
                "* target-and-other-wins events during compare, then replays them during bwd_src.",
                f"* Winner gates: class 0 uses {target0_wins_gate}; class 1 uses {target1_wins_gate}.",
                f"Cbwd_gate bwd 0 {cap_f:.12g}f IC=0",
                "Rbwd_gate bwd 0 1G",
                "Mreset_bwd_gate bwd rste 0 0 NMOS W=4u L=180n",
                f"Cmerr0 merr0 0 {cap_f:.12g}f IC=0",
                f"Cmerr1 merr1 0 {cap_f:.12g}f IC=0",
                "Rmerr0 merr0 0 1G",
                "Rmerr1 merr1 0 1G",
                "Mreset_merr0 merr0 rste 0 0 NMOS W=4u L=180n",
                "Mreset_merr1 merr1 rste 0 0 NMOS W=4u L=180n",
                f"Mmerr0_t vdd t0 merr0_t 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mmerr0_l merr0_t {target1_wins_gate} merr0_l 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mmerr0_c merr0_l cmp merr0 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mmerr1_t vdd t1 merr1_t 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mmerr1_l merr1_t {target0_wins_gate} merr1_l 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mmerr1_c merr1_l cmp merr1 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mbwd_merr0_a vdd merr0 bwd_merr0_a 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mbwd_merr0_b bwd_merr0_a bwd_src bwd 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mbwd_merr1_a vdd merr1 bwd_merr1_a 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mbwd_merr1_b bwd_merr1_a bwd_src bwd 0 NSENSE W={width_u:.12g}u L=180n",
            ]
            + node_parasitics(
                "merr0_t",
                "merr0_l",
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
    if mode in {
        "target_out_mistake_latch_restore",
        "target_out_mistake_latch_restore_stacked",
        "target_out_mistake_latch_restore_stacked_timed",
    }:
        stacked_restore = "stacked" in mode
        timed_restore = mode.endswith("_timed")
        lines = [
            "* Restored output-capacitor mistake latch: captures target-low/other-high during the error window,",
            "* then inverts the low-voltage event cap into a restored PMOS pull-up during bwd_src.",
            (
                "* Restore discriminator: "
                + ("two-device event stack" if stacked_restore else "single event device")
                + (" gated by bwd_src." if timed_restore else ".")
            ),
            f"Cbwd_gate bwd 0 {cap_f:.12g}f IC=0",
            "Rbwd_gate bwd 0 1G",
            "Mreset_bwd_gate bwd rste 0 0 NMOS W=4u L=180n",
            f"Cmerr0 merr0 0 {cap_f:.12g}f IC=0",
            f"Cmerr1 merr1 0 {cap_f:.12g}f IC=0",
            f"Cmerr0_bar merr0_bar 0 {cap_f:.12g}f IC=1.2",
            f"Cmerr1_bar merr1_bar 0 {cap_f:.12g}f IC=1.2",
            "Rmerr0 merr0 0 1G",
            "Rmerr1 merr1 0 1G",
            "Rmerr0_bar merr0_bar vdd 1G",
            "Rmerr1_bar merr1_bar vdd 1G",
            "Mreset_merr0 merr0 rste 0 0 NMOS W=4u L=180n",
            "Mreset_merr1 merr1 rste 0 0 NMOS W=4u L=180n",
            "Mreset_merr0_bar vdd rste merr0_bar 0 NREL W=4u L=180n",
            "Mreset_merr1_bar vdd rste merr1_bar 0 NREL W=4u L=180n",
            f"Mmerr0_p vdd out0 merr0_p vdd PMOS W={width_u:.12g}u L=180n",
            f"Mmerr0_t merr0_p t0 merr0_t 0 NSENSE W={width_u:.12g}u L=180n",
            f"Mmerr0_o merr0_t out1 merr0_o 0 NSENSE W={width_u:.12g}u L=180n",
            f"Mmerr0_e merr0_o err merr0 0 NSENSE W={width_u:.12g}u L=180n",
            f"Mmerr1_p vdd out1 merr1_p vdd PMOS W={width_u:.12g}u L=180n",
            f"Mmerr1_t merr1_p t1 merr1_t 0 NSENSE W={width_u:.12g}u L=180n",
            f"Mmerr1_o merr1_t out0 merr1_o 0 NSENSE W={width_u:.12g}u L=180n",
            f"Mmerr1_e merr1_o err merr1 0 NSENSE W={width_u:.12g}u L=180n",
        ]
        parasitic_nodes = [
            "merr0_p",
            "merr0_t",
            "merr0_o",
            "merr1_p",
            "merr1_t",
            "merr1_o",
            "merr0_bar",
            "merr1_bar",
            "bwd_merr0_p",
            "bwd_merr1_p",
        ]
        if stacked_restore:
            lines.extend(
                [
                    f"Mmerr0_restore_a merr0_bar merr0 merr0_bar_a 0 NREL W={width_u:.12g}u L=180n",
                    f"Mmerr1_restore_a merr1_bar merr1 merr1_bar_a 0 NREL W={width_u:.12g}u L=180n",
                ]
            )
            if timed_restore:
                lines.extend(
                    [
                        f"Mmerr0_restore_b merr0_bar_a merr0 merr0_bar_b 0 NREL W={width_u:.12g}u L=180n",
                        f"Mmerr0_restore_t merr0_bar_b bwd_src 0 0 NREL W={width_u:.12g}u L=180n",
                        f"Mmerr1_restore_b merr1_bar_a merr1 merr1_bar_b 0 NREL W={width_u:.12g}u L=180n",
                        f"Mmerr1_restore_t merr1_bar_b bwd_src 0 0 NREL W={width_u:.12g}u L=180n",
                    ]
                )
                parasitic_nodes.extend(["merr0_bar_b", "merr1_bar_b"])
            else:
                lines.extend(
                    [
                        f"Mmerr0_restore_b merr0_bar_a merr0 0 0 NREL W={width_u:.12g}u L=180n",
                        f"Mmerr1_restore_b merr1_bar_a merr1 0 0 NREL W={width_u:.12g}u L=180n",
                    ]
                )
            parasitic_nodes.extend(["merr0_bar_a", "merr1_bar_a"])
        else:
            lines.extend(
                [
                    f"Mmerr0_restore merr0_bar merr0 0 0 NREL W={width_u:.12g}u L=180n",
                    f"Mmerr1_restore merr1_bar merr1 0 0 NREL W={width_u:.12g}u L=180n",
                ]
            )
        lines.extend(
            [
                f"Mbwd_merr0_p bwd_merr0_p merr0_bar vdd vdd PMOS W={width_u:.12g}u L=180n",
                f"Mbwd_merr0_b bwd_merr0_p bwd_src bwd 0 NSENSE W={width_u:.12g}u L=180n",
                f"Mbwd_merr1_p bwd_merr1_p merr1_bar vdd vdd PMOS W={width_u:.12g}u L=180n",
                f"Mbwd_merr1_b bwd_merr1_p bwd_src bwd 0 NSENSE W={width_u:.12g}u L=180n",
            ]
        )
        return "\n".join(lines + node_parasitics(*parasitic_nodes))
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
    lead_mode: str = "out_senseamp",
    error_target_source_v: float | None = None,
    error_nontarget_source_v: float | None = None,
) -> str:
    if OUTPUTS != 2 and error_rule not in {
        "score",
        "out_residual",
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
        "ce_mirror_hybrid_limited",
        "ce_mirror_compete_limited",
    }:
        raise ValueError(
            "multi-class direct-flow runs currently require score, out_residual, onehot, onehot_limited, onehot_out, ce_out, ce_split_score, ce_split_diffgate, ce_split_dpair, ce_split_compete, ce_split_current, ce_split_hybrid, ce_split_limited, ce_mirror_limited, ce_mirror_winner_limited, ce_mirror_hybrid_limited, or ce_mirror_compete_limited error rails."
        )
    lines: list[str] = []
    target_source_node = "vdd"
    nontarget_source_node = "vdd"
    if error_target_source_v is not None:
        target_source_node = "ctsrch"
        lines.append(f"Vctsrch ctsrch 0 {error_target_source_v:.12g}")
    if error_nontarget_source_v is not None:
        nontarget_source_node = "cesrch"
        lines.append(f"Vcesrch cesrch 0 {error_nontarget_source_v:.12g}")
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
        elif error_rule == "onehot":
            target_w = residual_target_width_u
            nontarget_w = residual_output_width_u
            lines += [
                f"* One-vs-rest target rails: dp for target class, dn for complement target.",
                f"Mdp{out}_t0 vdd t{out} dp{out}_t 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdp{out}_t1 dp{out}_t err dp{out} 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdn{out}_nt0 vdd nt{out} dn{out}_nt 0 NSENSE W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_nt1 dn{out}_nt err dn{out} 0 NSENSE W={nontarget_w:.12g}u L=180n",
            ]
            lines += node_parasitics(f"dp{out}_t", f"dn{out}_nt")
        elif error_rule == "onehot_limited":
            target_w = residual_target_width_u
            nontarget_w = residual_output_width_u
            lines += [
                f"* Current-limited one-vs-rest rails: fixed target dp and complement dn.",
                f"* This is a hardware-native positive-average surrogate for softmax CE; tune",
                f"* target/non-target source widths so one target pulse balances OUTPUTS-1",
                f"* non-target pulses.",
                f"Mdp{out}_t0 ctsrc t{out} dp{out}_t 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdp{out}_err0 dp{out}_t err dp{out} 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdn{out}_nt0 cesrc nt{out} dn{out}_nt 0 NSENSE W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_err0 dn{out}_nt err dn{out} 0 NSENSE W={nontarget_w:.12g}u L=180n",
            ]
            lines += node_parasitics(f"dp{out}_t", f"dn{out}_nt")
        elif error_rule == "onehot_out":
            target_w = residual_target_width_u
            nontarget_w = residual_output_width_u
            lines += [
                f"* Output-gated one-vs-rest rails: target dp, active non-target dn.",
                f"Mdp{out}_t0 vdd t{out} dp{out}_t 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdp{out}_t1 dp{out}_t err dp{out} 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdn{out}_nt0 vdd nt{out} dn{out}_nt 0 NSENSE W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_out0 dn{out}_nt out{out} dn{out}_out 0 NSENSE W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_err0 dn{out}_out err dn{out} 0 NSENSE W={nontarget_w:.12g}u L=180n",
            ]
            lines += node_parasitics(f"dp{out}_t", f"dn{out}_nt", f"dn{out}_out")
        elif error_rule == "ce_out":
            target_w = residual_target_width_u
            nontarget_w = residual_output_width_u
            lines += [
                f"* CE-like rails: target dp is inhibited by its own output; non-target dn is output gated.",
                f"Mdp{out}_low0 vdd out{out} dp{out}_low vdd PMOS W={target_w:.12g}u L=180n",
                f"Mdp{out}_t0 dp{out}_low t{out} dp{out}_t 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdp{out}_err0 dp{out}_t err dp{out} 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdn{out}_nt0 vdd nt{out} dn{out}_nt 0 NSENSE W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_out0 dn{out}_nt out{out} dn{out}_out 0 NSENSE W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_err0 dn{out}_out err dn{out} 0 NSENSE W={nontarget_w:.12g}u L=180n",
            ]
            lines += node_parasitics(f"dp{out}_low", f"dp{out}_t", f"dn{out}_nt", f"dn{out}_out")
        elif error_rule == "ce_split_score":
            target_w = residual_target_width_u
            nontarget_w = residual_output_width_u
            lines += [
                f"* Split-score CE-like rails: target dp is inhibited by scorep; non-target dn is scorep gated.",
                f"Mdp{out}_low0 vdd scorep{out} dp{out}_low vdd PMOS W={target_w:.12g}u L=180n",
                f"Mdp{out}_t0 dp{out}_low t{out} dp{out}_t 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdp{out}_err0 dp{out}_t err dp{out} 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdn{out}_nt0 vdd nt{out} dn{out}_nt 0 NSENSE W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_score0 dn{out}_nt scorep{out} dn{out}_score 0 NSENSE W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_err0 dn{out}_score err dn{out} 0 NSENSE W={nontarget_w:.12g}u L=180n",
            ]
            lines += node_parasitics(f"dp{out}_low", f"dp{out}_t", f"dn{out}_nt", f"dn{out}_score")
        elif error_rule == "ce_split_diffgate":
            target_w = residual_target_width_u
            nontarget_w = residual_output_width_u
            lines += [
                f"* Differential split-score CE-like rails: target dp needs high scoren and low scorep;",
                f"* non-target dn needs high scorep and low scoren. This rejects score common-mode before writing.",
                f"Mdp{out}_low0 vdd scorep{out} dp{out}_low vdd PMOS W={target_w:.12g}u L=180n",
                f"Mdp{out}_t0 dp{out}_low t{out} dp{out}_t 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdp{out}_neg0 dp{out}_t scoren{out} dp{out}_neg 0 NREL W={target_w:.12g}u L=180n",
                f"Mdp{out}_err0 dp{out}_neg err dp{out} 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdn{out}_low0 vdd scoren{out} dn{out}_low vdd PMOS W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_nt0 dn{out}_low nt{out} dn{out}_nt 0 NSENSE W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_pos0 dn{out}_nt scorep{out} dn{out}_pos 0 NREL W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_err0 dn{out}_pos err dn{out} 0 NSENSE W={nontarget_w:.12g}u L=180n",
            ]
            lines += node_parasitics(
                f"dp{out}_low",
                f"dp{out}_t",
                f"dp{out}_neg",
                f"dn{out}_low",
                f"dn{out}_nt",
                f"dn{out}_pos",
            )
        elif error_rule == "ce_split_dpair":
            target_w = residual_target_width_u
            nontarget_w = residual_output_width_u
            lines += [
                f"* Differential-pair split-score CE rails: ybar{out} is an active-low predicted-class current rail.",
                f"Myp{out}_pos ybar{out} scorep{out} ysrc{out} 0 NREL W={nontarget_w:.12g}u L=180n",
                f"Myp{out}_neg vdd scoren{out} ysrc{out} 0 NREL W={nontarget_w:.12g}u L=180n",
                f"Myp{out}_tail ysrc{out} err 0 0 NMOS W={nontarget_w:.12g}u L=180n",
                f"Mdp{out}_t0 vdd t{out} dp{out}_t 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdp{out}_yp0 dp{out}_t ybar{out} dp{out}_yp 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdp{out}_err0 dp{out}_yp err dp{out} 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdn{out}_pred0 vdd ybar{out} dn{out}_pred vdd PMOS W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_nt0 dn{out}_pred nt{out} dn{out}_nt 0 NSENSE W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_err0 dn{out}_nt err dn{out} 0 NSENSE W={nontarget_w:.12g}u L=180n",
            ]
            lines += node_parasitics(f"ysrc{out}", f"dp{out}_t", f"dp{out}_yp", f"dn{out}_pred", f"dn{out}_nt")
        elif error_rule == "ce_split_compete":
            target_w = residual_target_width_u
            nontarget_w = residual_output_width_u
            lines += [
                f"* Class-coupled split-score CE rails: ybar{out} is discharged by a shared-tail",
                f"* scorep/scoren current competition before target/non-target write gates use it.",
                f"Mcc{out}_inh cc{out}_mid scoren{out} ybar{out} vdd PMOS W={nontarget_w:.12g}u L=180n",
                f"Mcc{out}_branch cc{out}_mid scorep{out} ccsrc 0 NREL W={nontarget_w:.12g}u L=180n",
                f"Mdp{out}_t0 vdd t{out} dp{out}_t 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdp{out}_yp0 dp{out}_t ybar{out} dp{out}_yp 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdp{out}_err0 dp{out}_yp err dp{out} 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdn{out}_pred0 vdd ybar{out} dn{out}_pred vdd PMOS W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_nt0 dn{out}_pred nt{out} dn{out}_nt 0 NSENSE W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_err0 dn{out}_nt err dn{out} 0 NSENSE W={nontarget_w:.12g}u L=180n",
            ]
            lines += node_parasitics(f"cc{out}_mid", f"dp{out}_t", f"dp{out}_yp", f"dn{out}_pred", f"dn{out}_nt")
        elif error_rule == "ce_split_current":
            target_w = residual_target_width_u
            nontarget_w = residual_output_width_u
            lines += [
                f"* Split-score current CE rails: target dp is one-hot; non-target dn is charged",
                f"* directly from a shared-source scorep/scoren branch, avoiding a thresholded ybar conversion.",
                f"Mdp{out}_t0 vdd t{out} dp{out}_t 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdp{out}_err0 dp{out}_t err dp{out} 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdn{out}_nt0 cesrc nt{out} dn{out}_nt 0 NSENSE W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_inh0 dn{out}_nt scoren{out} dn{out}_inh vdd PMOS W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_score0 dn{out}_inh scorep{out} dn{out}_score 0 NREL W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_err0 dn{out}_score err dn{out} 0 NSENSE W={nontarget_w:.12g}u L=180n",
            ]
            lines += node_parasitics(f"dp{out}_t", f"dn{out}_nt", f"dn{out}_inh", f"dn{out}_score")
        elif error_rule == "ce_split_hybrid":
            target_w = residual_target_width_u
            nontarget_w = residual_output_width_u
            lines += [
                f"* Hybrid split-score CE rails: shared-tail ybar suppresses target dp when the",
                f"* target already wins; shared-source score current charges non-target dn directly.",
                f"Mcc{out}_inh cc{out}_mid scoren{out} ybar{out} vdd PMOS W={nontarget_w:.12g}u L=180n",
                f"Mcc{out}_branch cc{out}_mid scorep{out} ccsrc 0 NREL W={nontarget_w:.12g}u L=180n",
                f"Mdp{out}_t0 vdd t{out} dp{out}_t 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdp{out}_yp0 dp{out}_t ybar{out} dp{out}_yp 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdp{out}_err0 dp{out}_yp err dp{out} 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdn{out}_nt0 cesrc nt{out} dn{out}_nt 0 NSENSE W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_inh0 dn{out}_nt scoren{out} dn{out}_inh vdd PMOS W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_score0 dn{out}_inh scorep{out} dn{out}_score 0 NREL W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_err0 dn{out}_score err dn{out} 0 NSENSE W={nontarget_w:.12g}u L=180n",
            ]
            lines += node_parasitics(
                f"cc{out}_mid",
                f"dp{out}_t",
                f"dp{out}_yp",
                f"dn{out}_nt",
                f"dn{out}_inh",
                f"dn{out}_score",
            )
        elif error_rule == "ce_split_limited":
            target_w = residual_target_width_u
            nontarget_w = residual_output_width_u
            lines += [
                f"* Current-limited hybrid split-score CE rails: ybar suppresses target dp when",
                f"* the target already wins, but target dp is fed from ctsrc instead of vdd so",
                f"* the target correction can be current-starved independently of target logic rails.",
                f"Mcc{out}_inh cc{out}_mid scoren{out} ybar{out} vdd PMOS W={nontarget_w:.12g}u L=180n",
                f"Mcc{out}_branch cc{out}_mid scorep{out} ccsrc 0 NREL W={nontarget_w:.12g}u L=180n",
                f"Mdp{out}_t0 ctsrc t{out} dp{out}_t 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdp{out}_yp0 dp{out}_t ybar{out} dp{out}_yp 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdp{out}_err0 dp{out}_yp err dp{out} 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdn{out}_nt0 cesrc nt{out} dn{out}_nt 0 NSENSE W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_inh0 dn{out}_nt scoren{out} dn{out}_inh vdd PMOS W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_score0 dn{out}_inh scorep{out} dn{out}_score 0 NREL W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_err0 dn{out}_score err dn{out} 0 NSENSE W={nontarget_w:.12g}u L=180n",
            ]
            lines += node_parasitics(
                f"cc{out}_mid",
                f"dp{out}_t",
                f"dp{out}_yp",
                f"dn{out}_nt",
                f"dn{out}_inh",
                f"dn{out}_score",
            )
        elif error_rule == "ce_mirror_limited":
            target_w = residual_target_width_u
            nontarget_w = residual_output_width_u
            lines += [
                f"* Diode-mirror CE rails: target dp is current-limited one-hot; non-target dn",
                f"* is gated by mirror-cap evidence because scorep/scoren diode nodes are too compressed.",
                f"Mdp{out}_t0 ctsrc t{out} dp{out}_t 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdp{out}_err0 dp{out}_t err dp{out} 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdn{out}_nt0 cesrc nt{out} dn{out}_nt 0 NSENSE W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_pm0 dn{out}_nt scorepm{out} dn{out}_pm vdd PMOS W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_nm0 dn{out}_pm scorenm{out} dn{out}_nm 0 NREL W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_err0 dn{out}_nm err dn{out} 0 NSENSE W={nontarget_w:.12g}u L=180n",
            ]
            lines += node_parasitics(
                f"dp{out}_t",
                f"dn{out}_nt",
                f"dn{out}_pm",
                f"dn{out}_nm",
            )
        elif error_rule == "ce_mirror_winner_limited":
            target_w = residual_target_width_u
            nontarget_w = residual_output_width_u
            lines += [
                f"* Winner-gated diode-mirror CE rails: out{out} is active-low for the",
                f"* mirror-derived class winner.  Target dp flows only when the target is not",
                f"* already active; non-target dn flows only for an active low non-target.",
                f"Mdp{out}_t0 ctsrc t{out} dp{out}_t 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdp{out}_out0 dp{out}_t out{out} dp{out}_out 0 NREL W={target_w:.12g}u L=180n",
                f"Mdp{out}_err0 dp{out}_out err dp{out} 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdn{out}_pred0 cesrc out{out} dn{out}_pred vdd PMOS W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_nt0 dn{out}_pred nt{out} dn{out}_nt 0 NSENSE W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_err0 dn{out}_nt err dn{out} 0 NSENSE W={nontarget_w:.12g}u L=180n",
            ]
            lines += node_parasitics(
                f"dp{out}_t",
                f"dp{out}_out",
                f"dn{out}_pred",
                f"dn{out}_nt",
            )
        elif error_rule == "ce_mirror_hybrid_limited":
            target_w = residual_target_width_u
            nontarget_w = residual_output_width_u
            lines += [
                f"* Hybrid diode-mirror CE rails: target dp uses the hard active-low",
                f"* winner output gate to suppress correct-sample target pumping, while",
                f"* non-target dn still follows analog mirror-cap evidence instead of only",
                f"* the hard winner. This tests whether soft non-target pressure fixes",
                f"* mirror-winner sign/ranking loss without reintroducing target overwrite.",
                f"Mdp{out}_t0 ctsrc t{out} dp{out}_t 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdp{out}_out0 dp{out}_t out{out} dp{out}_out 0 NREL W={target_w:.12g}u L=180n",
                f"Mdp{out}_err0 dp{out}_out err dp{out} 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdn{out}_nt0 cesrc nt{out} dn{out}_nt 0 NSENSE W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_pm0 dn{out}_nt scorepm{out} dn{out}_pm vdd PMOS W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_nm0 dn{out}_pm scorenm{out} dn{out}_nm 0 NREL W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_err0 dn{out}_nm err dn{out} 0 NSENSE W={nontarget_w:.12g}u L=180n",
            ]
            lines += node_parasitics(
                f"dp{out}_t",
                f"dp{out}_out",
                f"dn{out}_nt",
                f"dn{out}_pm",
                f"dn{out}_nm",
            )
        elif error_rule == "ce_mirror_compete_limited":
            target_w = residual_target_width_u
            nontarget_w = residual_output_width_u
            lines += [
                f"* Shared-tail diode-mirror CE rails: scorepm{out}/scorenm{out} form an",
                f"* active-low ybar{out} competition rail before target/non-target writes.",
                f"Mmc{out}_inh mc{out}_mid scorepm{out} ybar{out} vdd PMOS W={nontarget_w:.12g}u L=180n",
                f"Mmc{out}_branch mc{out}_mid scorenm{out} mcsrc 0 NREL W={nontarget_w:.12g}u L=180n",
                f"Mdp{out}_t0 ctsrc t{out} dp{out}_t 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdp{out}_yp0 dp{out}_t ybar{out} dp{out}_yp 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdp{out}_err0 dp{out}_yp err dp{out} 0 NSENSE W={target_w:.12g}u L=180n",
                f"Mdn{out}_pred0 cesrc ybar{out} dn{out}_pred vdd PMOS W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_nt0 dn{out}_pred nt{out} dn{out}_nt 0 NSENSE W={nontarget_w:.12g}u L=180n",
                f"Mdn{out}_err0 dn{out}_nt err dn{out} 0 NSENSE W={nontarget_w:.12g}u L=180n",
            ]
            lines += node_parasitics(
                f"mc{out}_mid",
                f"dp{out}_t",
                f"dp{out}_yp",
                f"dn{out}_pred",
                f"dn{out}_nt",
            )
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
        elif error_rule == "lead_mistake":
            other = 1 - out
            self_wins_gate = lead_win_gate(lead_mode, out)
            other_wins_gate = lead_win_gate(lead_mode, other)
            lines += [
                f"* Full-swing lead-mistake rails: class {out} gets dp only when its target loses.",
                f"Mdp{out}_t0 vdd t{out} dp{out}_t 0 NSENSE W=128u L=180n",
                f"Mdp{out}_l0 dp{out}_t {other_wins_gate} dp{out}_l 0 NSENSE W=128u L=180n",
                f"Mdp{out}_e0 dp{out}_l err dp{out} 0 NSENSE W=128u L=180n",
                f"Mdn{out}_t0 vdd t{other} dn{out}_t 0 NSENSE W=128u L=180n",
                f"Mdn{out}_l0 dn{out}_t {self_wins_gate} dn{out}_l 0 NSENSE W=128u L=180n",
                f"Mdn{out}_e0 dn{out}_l err dn{out} 0 NSENSE W=128u L=180n",
            ]
            lines += node_parasitics(f"dp{out}_t", f"dp{out}_l", f"dn{out}_t", f"dn{out}_l")
        elif error_rule == "lead_mistake_lowtarget":
            other = 1 - out
            self_wins_gate = lead_win_gate(lead_mode, out)
            other_wins_gate = lead_win_gate(lead_mode, other)
            lines += [
                f"* Soft lead-mistake rails: class {out} updates only when the target loses and the target score is low.",
                f"Mdp{out}_t0 vdd t{out} dp{out}_t 0 NSENSE W=128u L=180n",
                f"Mdp{out}_l0 dp{out}_t {other_wins_gate} dp{out}_l 0 NSENSE W=128u L=180n",
                f"Mdp{out}_s0 dp{out}_l lose{out} dp{out}_s 0 NSENSE W=128u L=180n",
                f"Mdp{out}_e0 dp{out}_s err dp{out} 0 NSENSE W=128u L=180n",
                f"Mdn{out}_t0 vdd t{other} dn{out}_t 0 NSENSE W=128u L=180n",
                f"Mdn{out}_l0 dn{out}_t {self_wins_gate} dn{out}_l 0 NSENSE W=128u L=180n",
                f"Mdn{out}_s0 dn{out}_l lose{other} dn{out}_s 0 NSENSE W=128u L=180n",
                f"Mdn{out}_e0 dn{out}_s err dn{out} 0 NSENSE W=128u L=180n",
            ]
            lines += node_parasitics(
                f"dp{out}_t",
                f"dp{out}_l",
                f"dp{out}_s",
                f"dn{out}_t",
                f"dn{out}_l",
                f"dn{out}_s",
            )
        elif error_rule == "lead_mistake_outlow":
            other = 1 - out
            self_wins_gate = lead_win_gate(lead_mode, out)
            other_wins_gate = lead_win_gate(lead_mode, other)
            lines += [
                f"* Soft lead-mistake rails: class {out} updates through an analog target-output-low PMOS source gate.",
                f"Mdp{out}_p0 vdd out{out} dp{out}_p vdd PMOS W=128u L=180n",
                f"Mdp{out}_t0 dp{out}_p t{out} dp{out}_t 0 NSENSE W=128u L=180n",
                f"Mdp{out}_l0 dp{out}_t {other_wins_gate} dp{out}_l 0 NSENSE W=128u L=180n",
                f"Mdp{out}_e0 dp{out}_l err dp{out} 0 NSENSE W=128u L=180n",
                f"Mdn{out}_p0 vdd out{other} dn{out}_p vdd PMOS W=128u L=180n",
                f"Mdn{out}_t0 dn{out}_p t{other} dn{out}_t 0 NSENSE W=128u L=180n",
                f"Mdn{out}_l0 dn{out}_t {self_wins_gate} dn{out}_l 0 NSENSE W=128u L=180n",
                f"Mdn{out}_e0 dn{out}_l err dn{out} 0 NSENSE W=128u L=180n",
            ]
            lines += node_parasitics(
                f"dp{out}_p",
                f"dp{out}_t",
                f"dp{out}_l",
                f"dn{out}_p",
                f"dn{out}_t",
                f"dn{out}_l",
            )
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
    if error_rule in {"ce_split_compete", "ce_split_hybrid", "ce_split_limited", "ce_mirror_compete_limited"}:
        tail_w = max(residual_output_width_u, 1e-9)
        tail_node = "mcsrc" if error_rule == "ce_mirror_compete_limited" else "ccsrc"
        tail_name = "Mmc_tail" if error_rule == "ce_mirror_compete_limited" else "Mcc_tail"
        lines += [
            f"* Shared-tail class competition for {error_rule}.",
            f"R{tail_node} {tail_node} 0 1e12",
            f"{tail_name} {tail_node} err 0 0 NMOS W={tail_w:.12g}u L=180n",
        ]
        lines += node_parasitics(tail_node)
    if error_rule in {
        "onehot_limited",
        "ce_split_limited",
        "ce_mirror_limited",
        "ce_mirror_winner_limited",
        "ce_mirror_hybrid_limited",
        "ce_mirror_compete_limited",
    }:
        source_w = max(residual_target_width_u, 1e-9)
        lines += [
            f"* Current-starved target source rail for {error_rule}.",
            "Cctsrc ctsrc 0 2f IC=0",
            "Rctsrc ctsrc 0 1G",
            "Mreset_ctsrc ctsrc rste 0 0 NMOS W=4u L=180n",
            f"Mctsrc {target_source_node} err ctsrc 0 NSENSE W={source_w:.12g}u L=180n",
        ]
    if error_rule in {
        "onehot_limited",
        "ce_split_current",
        "ce_split_hybrid",
        "ce_split_limited",
        "ce_mirror_limited",
        "ce_mirror_winner_limited",
        "ce_mirror_hybrid_limited",
        "ce_mirror_compete_limited",
    }:
        source_w = max(residual_output_width_u, 1e-9)
        lines += [
            f"* Shared source rail for {error_rule} non-target probability current.",
            "Ccesrc cesrc 0 0.2f IC=0",
            "Rcesrc cesrc 0 1G",
            "Mreset_cesrc cesrc rste 0 0 NMOS W=4u L=180n",
            f"Mcesrc {nontarget_source_node} err cesrc 0 NSENSE W={source_w:.12g}u L=180n",
        ]
    return "\n".join(lines)


def hidden_delta(
    hidden_error_rule: str,
    hidden_delta_relu_gate: str,
    hidden_delta_weight_device: str,
    design: SynapseDesign,
    internal_cap_f: float,
    internal_leak_ohm: float,
    internal_reset_width_u: float,
    readout_fanins: ReadoutFanins | None = None,
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
    fanouts = readout_fanouts_from_fanins(effective_readout_fanins(readout_fanins))
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
        for out in fanouts[h]:
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
                    if internal_reset_width_u > 0:
                        lines.append(
                            f"Mreset_{node} {node} rste 0 0 NMOS W={internal_reset_width_u:.12g}u L=180n"
                        )
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
    readout_fanins: ReadoutFanins | None = None,
) -> str:
    lines: list[str] = []
    fanins = effective_readout_fanins(readout_fanins)
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
        for h in fanins[out]:
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


def output_bias_flow_action_widths(
    output_bias_update_width_u: float,
    readout_charge_update_width_u: float | None,
    readout_discharge_update_width_u: float | None,
) -> tuple[float, float]:
    discharge_width_u = (
        output_bias_update_width_u
        if readout_discharge_update_width_u is None
        else min(output_bias_update_width_u, readout_discharge_update_width_u)
    )
    charge_width_u = (
        output_bias_update_width_u
        if readout_charge_update_width_u is None
        else min(output_bias_update_width_u, readout_charge_update_width_u)
    )
    return discharge_width_u, charge_width_u


def readout_flow_updates(
    readout_update_width_u: float,
    output_bias_update_width_u: float,
    flow_pre_store: str,
    readout_flow_polarity: str,
    readout_flow_write_mode: str = "discharge",
    readout_pos_update_width_u: float | None = None,
    readout_neg_update_width_u: float | None = None,
    readout_charge_update_width_u: float | None = None,
    readout_discharge_update_width_u: float | None = None,
    readout_dp_gate_update_width_u: float | None = None,
    readout_dn_gate_update_width_u: float | None = None,
    readout_dp_discharge_gate_update_width_u: float | None = None,
    readout_dp_charge_gate_update_width_u: float | None = None,
    readout_dn_discharge_gate_update_width_u: float | None = None,
    readout_dn_charge_gate_update_width_u: float | None = None,
    readout_center_pull_width_u: float = 0.0,
    output_bias_center_pull_width_u: float = 0.0,
    readout_center_pull_gate: str = "bwd",
    readout_center_pull_mode: str = "always",
    readout_write_state_gate_mode: str = "none",
    readout_write_gate_device: str = "NSENSE",
    output_bias_write_pre_gate: str = "none",
    output_bias_flow_polarity: str = "follow_readout",
    readout_pos_write_high_node: str = "whigh",
    readout_pos_write_low_node: str = "wlow",
    readout_neg_write_high_node: str = "whigh",
    readout_neg_write_low_node: str = "wlow",
    readout_pos_center_pull_node: str = "wcenter",
    readout_neg_center_pull_node: str = "wcenter",
    output_bias_pos_center_pull_node: str = "wcenter",
    output_bias_neg_center_pull_node: str = "wcenter",
    write_error_exclusion: str = "none",
    write_error_exclusion_width_u: float = 8.0,
    readout_fanins: ReadoutFanins | None = None,
) -> str:
    if flow_pre_store not in FLOW_PRE_STORES:
        raise ValueError(f"unknown flow pre-store mode: {flow_pre_store}")
    if readout_flow_polarity not in READOUT_FLOW_POLARITIES:
        raise ValueError(f"unknown readout flow polarity: {readout_flow_polarity}")
    if readout_flow_write_mode not in READOUT_FLOW_WRITE_MODES:
        raise ValueError(f"unknown readout flow write mode: {readout_flow_write_mode}")
    if readout_center_pull_gate not in READOUT_CENTER_PULL_GATES:
        raise ValueError(f"unknown readout center-pull gate: {readout_center_pull_gate}")
    if readout_center_pull_mode not in READOUT_CENTER_PULL_MODES:
        raise ValueError(f"unknown readout center-pull mode: {readout_center_pull_mode}")
    if readout_write_state_gate_mode not in READOUT_WRITE_STATE_GATE_MODES:
        raise ValueError(f"unknown readout write state-gate mode: {readout_write_state_gate_mode}")
    if readout_write_gate_device not in WRITE_GATE_DEVICES:
        raise ValueError(f"unknown readout write-gate device: {readout_write_gate_device}")
    if output_bias_write_pre_gate not in OUTPUT_BIAS_WRITE_PRE_GATES:
        raise ValueError(f"unknown output-bias write pre-gate: {output_bias_write_pre_gate}")
    if output_bias_flow_polarity not in OUTPUT_BIAS_FLOW_POLARITIES:
        raise ValueError(f"unknown output-bias flow polarity: {output_bias_flow_polarity}")
    if write_error_exclusion not in WRITE_ERROR_EXCLUSION_MODES:
        raise ValueError(f"unknown write error exclusion mode: {write_error_exclusion}")
    pos_update_width_u = (
        readout_update_width_u if readout_pos_update_width_u is None else readout_pos_update_width_u
    )
    neg_update_width_u = (
        readout_update_width_u if readout_neg_update_width_u is None else readout_neg_update_width_u
    )
    pos_discharge_width_u = (
        pos_update_width_u if readout_discharge_update_width_u is None else readout_discharge_update_width_u
    )
    neg_discharge_width_u = (
        neg_update_width_u if readout_discharge_update_width_u is None else readout_discharge_update_width_u
    )
    pos_charge_width_u = (
        pos_update_width_u if readout_charge_update_width_u is None else readout_charge_update_width_u
    )
    neg_charge_width_u = (
        neg_update_width_u if readout_charge_update_width_u is None else readout_charge_update_width_u
    )
    bias_discharge_width_u, bias_charge_width_u = output_bias_flow_action_widths(
        output_bias_update_width_u,
        readout_charge_update_width_u,
        readout_discharge_update_width_u,
    )
    if (
        readout_update_width_u < 0
        or pos_update_width_u < 0
        or neg_update_width_u < 0
        or pos_discharge_width_u < 0
        or neg_discharge_width_u < 0
        or pos_charge_width_u < 0
        or neg_charge_width_u < 0
        or bias_discharge_width_u < 0
        or bias_charge_width_u < 0
        or (readout_dp_gate_update_width_u is not None and readout_dp_gate_update_width_u < 0)
        or (readout_dn_gate_update_width_u is not None and readout_dn_gate_update_width_u < 0)
        or (
            readout_dp_discharge_gate_update_width_u is not None
            and readout_dp_discharge_gate_update_width_u < 0
        )
        or (
            readout_dp_charge_gate_update_width_u is not None
            and readout_dp_charge_gate_update_width_u < 0
        )
        or (
            readout_dn_discharge_gate_update_width_u is not None
            and readout_dn_discharge_gate_update_width_u < 0
        )
        or (
            readout_dn_charge_gate_update_width_u is not None
            and readout_dn_charge_gate_update_width_u < 0
        )
        or output_bias_update_width_u < 0
        or readout_center_pull_width_u < 0
        or output_bias_center_pull_width_u < 0
        or write_error_exclusion_width_u < 0
    ):
        raise ValueError("readout flow update widths must be nonnegative.")

    def error_gate_width(gate: str, fallback_width_u: float, action: str) -> float:
        if action not in {"charge", "discharge"}:
            raise ValueError(f"unknown write-gate action: {action}")
        if (
            gate.startswith("dp")
            and action == "discharge"
            and readout_dp_discharge_gate_update_width_u is not None
        ):
            return readout_dp_discharge_gate_update_width_u
        if gate.startswith("dp") and action == "charge" and readout_dp_charge_gate_update_width_u is not None:
            return readout_dp_charge_gate_update_width_u
        if gate.startswith("dp") and readout_dp_gate_update_width_u is not None:
            return readout_dp_gate_update_width_u
        if (
            gate.startswith("dn")
            and action == "discharge"
            and readout_dn_discharge_gate_update_width_u is not None
        ):
            return readout_dn_discharge_gate_update_width_u
        if gate.startswith("dn") and action == "charge" and readout_dn_charge_gate_update_width_u is not None:
            return readout_dn_charge_gate_update_width_u
        if gate.startswith("dn") and readout_dn_gate_update_width_u is not None:
            return readout_dn_gate_update_width_u
        return fallback_width_u

    n_gate, p_gate = ("dp", "dn") if readout_flow_polarity == "normal" else ("dn", "dp")
    bounded_write = readout_flow_write_mode.startswith("bounded_")
    pmos_charge_write = readout_flow_write_mode in {
        "bounded_pmos_charge_only",
        "bounded_pmos_charge_discharge",
    }
    cmos_complementary_write = readout_flow_write_mode == "bounded_cmos_charge_discharge"
    pmos_charge_discharge = readout_flow_write_mode == "bounded_pmos_charge_discharge"
    discharge_enabled = readout_flow_write_mode in {
        "discharge",
        "bounded_discharge",
        "charge_discharge",
        "bounded_charge_discharge",
        "bounded_cmos_charge_discharge",
        "bounded_pmos_charge_discharge",
    }
    charge_enabled = readout_flow_write_mode in {
        "charge_only",
        "bounded_charge_only",
        "charge_discharge",
        "bounded_charge_discharge",
    }
    high_node = "whigh" if bounded_write else "vdd"
    low_node = "wlow" if bounded_write else "0"
    pos_high_node = readout_pos_write_high_node if bounded_write else high_node
    pos_low_node = readout_pos_write_low_node if bounded_write else low_node
    neg_high_node = readout_neg_write_high_node if bounded_write else high_node
    neg_low_node = readout_neg_write_low_node if bounded_write else low_node
    state_gate_discharge = readout_write_state_gate_mode in {"state_high_discharge", "state_window"}
    state_gate_charge = readout_write_state_gate_mode == "state_window"
    if pmos_charge_write and write_error_exclusion != "diffpair_bleed":
        raise ValueError(f"{readout_flow_write_mode} requires write_error_exclusion='diffpair_bleed'.")
    if pmos_charge_write and readout_write_state_gate_mode != "none":
        raise ValueError(f"{readout_flow_write_mode} does not support readout write state-gate modes yet.")
    if cmos_complementary_write and write_error_exclusion != "diffpair_bleed":
        raise ValueError(f"{readout_flow_write_mode} requires write_error_exclusion='diffpair_bleed'.")
    if cmos_complementary_write and flow_pre_store != "synapse_spike":
        raise ValueError(f"{readout_flow_write_mode} requires flow_pre_store='synapse_spike'.")
    if cmos_complementary_write and readout_write_state_gate_mode != "none":
        raise ValueError(f"{readout_flow_write_mode} does not support readout write state-gate modes yet.")
    lines: list[str] = []
    fanins = effective_readout_fanins(readout_fanins)

    def bias_pre_stack(prefix: str, source_node: str, width_u: float) -> tuple[str, list[str], list[str]]:
        if output_bias_write_pre_gate == "none":
            return source_node, [], []
        pre_node = f"{prefix}_a"
        return (
            pre_node,
            [
                f"M{prefix}_a {source_node} bias {pre_node} 0 NREL W={width_u:.12g}u L=180n",
            ],
            [pre_node],
        )

    for out in range(OUTPUTS):
        positive_error_gate = f"{n_gate}{out}"
        negative_error_gate = f"{p_gate}{out}"
        overlap_write_gate: str | None = None
        positive_write_gate_low_true: str | None = None
        negative_write_gate_low_true: str | None = None
        if write_error_exclusion in {"pmos_inhibit", "pmos_inhibit_decay"}:
            positive_write_gate = f"rwpos{out}"
            negative_write_gate = f"rwneg{out}"
            lines += [
                f"* Exclusive readout write rails for output {out}: block ambiguous dp/dn overlap.",
                f"Crwpos{out} {positive_write_gate} 0 0.1f IC=0",
                f"Crwneg{out} {negative_write_gate} 0 0.1f IC=0",
                f"Rrwpos{out} {positive_write_gate} 0 1G",
                f"Rrwneg{out} {negative_write_gate} 0 1G",
                f"Mreset_rwpos{out} {positive_write_gate} rste 0 0 NMOS W=4u L=180n",
                f"Mreset_rwneg{out} {negative_write_gate} rste 0 0 NMOS W=4u L=180n",
                f"Mrwpos{out}_inh vdd {negative_error_gate} rwpos{out}_src vdd PMOS W={write_error_exclusion_width_u:.12g}u L=180n",
                f"Mrwpos{out}_gate rwpos{out}_src {positive_error_gate} {positive_write_gate} 0 NSENSE W={write_error_exclusion_width_u:.12g}u L=180n",
                f"Mrwneg{out}_inh vdd {positive_error_gate} rwneg{out}_src vdd PMOS W={write_error_exclusion_width_u:.12g}u L=180n",
                f"Mrwneg{out}_gate rwneg{out}_src {negative_error_gate} {negative_write_gate} 0 NSENSE W={write_error_exclusion_width_u:.12g}u L=180n",
                f"Mrwpos{out}_kill {positive_write_gate} {negative_error_gate} 0 0 NMOS W={write_error_exclusion_width_u:.12g}u L=180n",
                f"Mrwneg{out}_kill {negative_write_gate} {positive_error_gate} 0 0 NMOS W={write_error_exclusion_width_u:.12g}u L=180n",
            ]
            lines += node_parasitics(f"rwpos{out}_src", f"rwneg{out}_src")
            if write_error_exclusion == "pmos_inhibit_decay":
                overlap_write_gate = f"rwov{out}"
                lines += [
                    f"* Readout overlap-decay rail for output {out}: dp AND dn gives symmetric common-mode weight decay.",
                    f"Crwov{out} {overlap_write_gate} 0 0.1f IC=0",
                    f"Rrwov{out} {overlap_write_gate} 0 1G",
                    f"Mreset_rwov{out} {overlap_write_gate} rste 0 0 NMOS W=4u L=180n",
                    f"Mrwov{out}_p vdd {positive_error_gate} rwov{out}_mid 0 NMOS W={write_error_exclusion_width_u:.12g}u L=180n",
                    f"Mrwov{out}_n rwov{out}_mid {negative_error_gate} {overlap_write_gate} 0 NMOS W={write_error_exclusion_width_u:.12g}u L=180n",
                ]
                lines += node_parasitics(f"rwov{out}_mid")
        elif write_error_exclusion == "diffpair_bleed":
            positive_write_gate = f"rwpos{out}"
            negative_write_gate = f"rwneg{out}"
            positive_write_gate_low_true = f"rwsel{out}_posbar"
            negative_write_gate_low_true = f"rwsel{out}_negbar"
            lines += diffpair_bleed_write_selector_lines(
                f"rwsel{out}",
                positive_error_gate,
                negative_error_gate,
                positive_write_gate,
                negative_write_gate,
                write_error_exclusion_width_u,
                f"readout output {out}",
            )
        else:
            positive_write_gate = positive_error_gate
            negative_write_gate = negative_error_gate
        positive_gate_neg_discharge_width_u = error_gate_width(
            positive_error_gate, neg_discharge_width_u, "discharge"
        )
        negative_gate_pos_discharge_width_u = error_gate_width(
            negative_error_gate, pos_discharge_width_u, "discharge"
        )
        positive_gate_pos_charge_width_u = error_gate_width(positive_error_gate, pos_charge_width_u, "charge")
        negative_gate_neg_charge_width_u = error_gate_width(negative_error_gate, neg_charge_width_u, "charge")
        # The output-bias cell has no hidden pre-activation gate, so its useful
        # polarity can differ from the row synapse polarity.  By default this
        # preserves the historical opposite-row mapping; explicit normal or
        # reversed lets full experiments combine reversed row writes with
        # normal-polarity bias writes.
        if output_bias_flow_polarity == "follow_readout":
            bias_positive_write_gate = negative_write_gate
            bias_negative_write_gate = positive_write_gate
        else:
            bias_n_gate, bias_p_gate = ("dp", "dn") if output_bias_flow_polarity == "normal" else ("dn", "dp")
            bias_positive_write_gate = f"{bias_p_gate}{out}"
            bias_negative_write_gate = f"{bias_n_gate}{out}"
        bias_low_true_by_gate = {
            positive_write_gate: positive_write_gate_low_true,
            negative_write_gate: negative_write_gate_low_true,
            positive_error_gate: positive_write_gate_low_true,
            negative_error_gate: negative_write_gate_low_true,
        }
        bias_positive_write_gate_low_true = bias_low_true_by_gate.get(bias_positive_write_gate)
        bias_negative_write_gate_low_true = bias_low_true_by_gate.get(bias_negative_write_gate)
        if bias_discharge_width_u > 0 and discharge_enabled:
            if state_gate_discharge:
                n_flow_source, n_flow_pre, n_flow_pre_nodes = bias_pre_stack(
                    f"vbo{out}n_flow",
                    f"vbo{out}n_flow_b",
                    bias_discharge_width_u,
                )
                p_flow_source, p_flow_pre, p_flow_pre_nodes = bias_pre_stack(
                    f"vbo{out}p_flow",
                    f"vbo{out}p_flow_b",
                    bias_discharge_width_u,
                )
                lines += [
                    f"Mvbo{out}n_flow_s vbo{out}n vbo{out}n vbo{out}n_flow_s 0 NREL W={bias_discharge_width_u:.12g}u L=180n",
                    f"Mvbo{out}n_flow_b vbo{out}n_flow_s bwd vbo{out}n_flow_b 0 NREL W={bias_discharge_width_u:.12g}u L=180n",
                    *n_flow_pre,
                    f"Mvbo{out}n_flow_d {n_flow_source} {bias_positive_write_gate} {neg_low_node} 0 {readout_write_gate_device} W={bias_discharge_width_u:.12g}u L=180n",
                    f"Mvbo{out}p_flow_s vbo{out}p vbo{out}p vbo{out}p_flow_s 0 NREL W={bias_discharge_width_u:.12g}u L=180n",
                    f"Mvbo{out}p_flow_b vbo{out}p_flow_s bwd vbo{out}p_flow_b 0 NREL W={bias_discharge_width_u:.12g}u L=180n",
                    *p_flow_pre,
                    f"Mvbo{out}p_flow_d {p_flow_source} {bias_negative_write_gate} {pos_low_node} 0 {readout_write_gate_device} W={bias_discharge_width_u:.12g}u L=180n",
                ]
                lines += node_parasitics(
                    f"vbo{out}n_flow_s",
                    f"vbo{out}n_flow_b",
                    *n_flow_pre_nodes,
                    f"vbo{out}p_flow_s",
                    f"vbo{out}p_flow_b",
                    *p_flow_pre_nodes,
                )
            else:
                n_flow_source, n_flow_pre, n_flow_pre_nodes = bias_pre_stack(
                    f"vbo{out}n_flow",
                    f"vbo{out}n_flow_b",
                    bias_discharge_width_u,
                )
                p_flow_source, p_flow_pre, p_flow_pre_nodes = bias_pre_stack(
                    f"vbo{out}p_flow",
                    f"vbo{out}p_flow_b",
                    bias_discharge_width_u,
                )
                lines += [
                    f"Mvbo{out}n_flow_b vbo{out}n bwd vbo{out}n_flow_b 0 NREL W={bias_discharge_width_u:.12g}u L=180n",
                    *n_flow_pre,
                    f"Mvbo{out}n_flow_d {n_flow_source} {bias_positive_write_gate} {neg_low_node} 0 {readout_write_gate_device} W={bias_discharge_width_u:.12g}u L=180n",
                    f"Mvbo{out}p_flow_b vbo{out}p bwd vbo{out}p_flow_b 0 NREL W={bias_discharge_width_u:.12g}u L=180n",
                    *p_flow_pre,
                    f"Mvbo{out}p_flow_d {p_flow_source} {bias_negative_write_gate} {pos_low_node} 0 {readout_write_gate_device} W={bias_discharge_width_u:.12g}u L=180n",
                ]
                lines += node_parasitics(
                    f"vbo{out}n_flow_b",
                    *n_flow_pre_nodes,
                    f"vbo{out}p_flow_b",
                    *p_flow_pre_nodes,
                )
        if bias_charge_width_u > 0 and charge_enabled:
            if state_gate_charge:
                p_ch_source, p_ch_pre, p_ch_pre_nodes = bias_pre_stack(
                    f"vbo{out}p_ch",
                    f"vbo{out}p_ch_b",
                    bias_charge_width_u,
                )
                n_ch_source, n_ch_pre, n_ch_pre_nodes = bias_pre_stack(
                    f"vbo{out}n_ch",
                    f"vbo{out}n_ch_b",
                    bias_charge_width_u,
                )
                lines += [
                    f"Mvbo{out}p_ch_s {pos_high_node} vbo{out}p vbo{out}p_ch_s vdd PMOS W={bias_charge_width_u:.12g}u L=180n",
                    f"Mvbo{out}p_ch_b vbo{out}p_ch_s bwd vbo{out}p_ch_b 0 NREL W={bias_charge_width_u:.12g}u L=180n",
                    *p_ch_pre,
                    f"Mvbo{out}p_ch_d {p_ch_source} {bias_positive_write_gate} vbo{out}p 0 {readout_write_gate_device} W={bias_charge_width_u:.12g}u L=180n",
                    f"Mvbo{out}n_ch_s {neg_high_node} vbo{out}n vbo{out}n_ch_s vdd PMOS W={bias_charge_width_u:.12g}u L=180n",
                    f"Mvbo{out}n_ch_b vbo{out}n_ch_s bwd vbo{out}n_ch_b 0 NREL W={bias_charge_width_u:.12g}u L=180n",
                    *n_ch_pre,
                    f"Mvbo{out}n_ch_d {n_ch_source} {bias_negative_write_gate} vbo{out}n 0 {readout_write_gate_device} W={bias_charge_width_u:.12g}u L=180n",
                ]
                lines += node_parasitics(
                    f"vbo{out}p_ch_s",
                    f"vbo{out}p_ch_b",
                    *p_ch_pre_nodes,
                    f"vbo{out}n_ch_s",
                    f"vbo{out}n_ch_b",
                    *n_ch_pre_nodes,
                )
            else:
                p_ch_source, p_ch_pre, p_ch_pre_nodes = bias_pre_stack(
                    f"vbo{out}p_ch",
                    f"vbo{out}p_ch_b",
                    bias_charge_width_u,
                )
                n_ch_source, n_ch_pre, n_ch_pre_nodes = bias_pre_stack(
                    f"vbo{out}n_ch",
                    f"vbo{out}n_ch_b",
                    bias_charge_width_u,
                )
                lines += [
                    f"Mvbo{out}p_ch_b {pos_high_node} bwd vbo{out}p_ch_b 0 NREL W={bias_charge_width_u:.12g}u L=180n",
                    *p_ch_pre,
                    f"Mvbo{out}p_ch_d {p_ch_source} {bias_positive_write_gate} vbo{out}p 0 {readout_write_gate_device} W={bias_charge_width_u:.12g}u L=180n",
                    f"Mvbo{out}n_ch_b {neg_high_node} bwd vbo{out}n_ch_b 0 NREL W={bias_charge_width_u:.12g}u L=180n",
                    *n_ch_pre,
                    f"Mvbo{out}n_ch_d {n_ch_source} {bias_negative_write_gate} vbo{out}n 0 {readout_write_gate_device} W={bias_charge_width_u:.12g}u L=180n",
                ]
                lines += node_parasitics(
                    f"vbo{out}p_ch_b",
                    *p_ch_pre_nodes,
                    f"vbo{out}n_ch_b",
                    *n_ch_pre_nodes,
                )
        if pmos_charge_write and bias_charge_width_u > 0:
            if bias_positive_write_gate_low_true is None or bias_negative_write_gate_low_true is None:
                raise ValueError("PMOS readout bias charge requires low-true diffpair write selector rails.")
            if output_bias_write_pre_gate == "none":
                lines += [
                    f"Mvbo{out}p_pch_s vbo{out}p_pch_b {bias_positive_write_gate_low_true} {pos_high_node} vdd PMOS W={bias_charge_width_u:.12g}u L=180n",
                    f"Mvbo{out}p_pch_b vbo{out}p_pch_b bwd vbo{out}p 0 NREL W={bias_charge_width_u:.12g}u L=180n",
                    f"Mvbo{out}n_pch_s vbo{out}n_pch_b {bias_negative_write_gate_low_true} {neg_high_node} vdd PMOS W={bias_charge_width_u:.12g}u L=180n",
                    f"Mvbo{out}n_pch_b vbo{out}n_pch_b bwd vbo{out}n 0 NREL W={bias_charge_width_u:.12g}u L=180n",
                ]
                lines += node_parasitics(f"vbo{out}p_pch_b", f"vbo{out}n_pch_b")
            else:
                lines += [
                    f"Mvbo{out}p_pch_s vbo{out}p_pch_b {bias_positive_write_gate_low_true} {pos_high_node} vdd PMOS W={bias_charge_width_u:.12g}u L=180n",
                    f"Mvbo{out}p_pch_g vbo{out}p_pch_b bwd vbo{out}p_pch_g 0 NREL W={bias_charge_width_u:.12g}u L=180n",
                    f"Mvbo{out}p_pch_a vbo{out}p_pch_g bias vbo{out}p 0 NREL W={bias_charge_width_u:.12g}u L=180n",
                    f"Mvbo{out}n_pch_s vbo{out}n_pch_b {bias_negative_write_gate_low_true} {neg_high_node} vdd PMOS W={bias_charge_width_u:.12g}u L=180n",
                    f"Mvbo{out}n_pch_g vbo{out}n_pch_b bwd vbo{out}n_pch_g 0 NREL W={bias_charge_width_u:.12g}u L=180n",
                    f"Mvbo{out}n_pch_a vbo{out}n_pch_g bias vbo{out}n 0 NREL W={bias_charge_width_u:.12g}u L=180n",
                ]
                lines += node_parasitics(
                    f"vbo{out}p_pch_b",
                    f"vbo{out}p_pch_g",
                    f"vbo{out}n_pch_b",
                    f"vbo{out}n_pch_g",
                )
        if overlap_write_gate is not None and bias_discharge_width_u > 0 and discharge_enabled:
            if state_gate_discharge:
                p_ov_source, p_ov_pre, p_ov_pre_nodes = bias_pre_stack(
                    f"vbo{out}p_ov",
                    f"vbo{out}p_ov_b",
                    bias_discharge_width_u,
                )
                n_ov_source, n_ov_pre, n_ov_pre_nodes = bias_pre_stack(
                    f"vbo{out}n_ov",
                    f"vbo{out}n_ov_b",
                    bias_discharge_width_u,
                )
                lines += [
                    f"Mvbo{out}p_ov_s vbo{out}p vbo{out}p vbo{out}p_ov_s 0 NREL W={bias_discharge_width_u:.12g}u L=180n",
                    f"Mvbo{out}p_ov_b vbo{out}p_ov_s bwd vbo{out}p_ov_b 0 NREL W={bias_discharge_width_u:.12g}u L=180n",
                    *p_ov_pre,
                    f"Mvbo{out}p_ov_d {p_ov_source} {overlap_write_gate} {pos_low_node} 0 {readout_write_gate_device} W={bias_discharge_width_u:.12g}u L=180n",
                    f"Mvbo{out}n_ov_s vbo{out}n vbo{out}n vbo{out}n_ov_s 0 NREL W={bias_discharge_width_u:.12g}u L=180n",
                    f"Mvbo{out}n_ov_b vbo{out}n_ov_s bwd vbo{out}n_ov_b 0 NREL W={bias_discharge_width_u:.12g}u L=180n",
                    *n_ov_pre,
                    f"Mvbo{out}n_ov_d {n_ov_source} {overlap_write_gate} {neg_low_node} 0 {readout_write_gate_device} W={bias_discharge_width_u:.12g}u L=180n",
                ]
                lines += node_parasitics(
                    f"vbo{out}p_ov_s",
                    f"vbo{out}p_ov_b",
                    *p_ov_pre_nodes,
                    f"vbo{out}n_ov_s",
                    f"vbo{out}n_ov_b",
                    *n_ov_pre_nodes,
                )
            else:
                p_ov_source, p_ov_pre, p_ov_pre_nodes = bias_pre_stack(
                    f"vbo{out}p_ov",
                    f"vbo{out}p_ov_b",
                    bias_discharge_width_u,
                )
                n_ov_source, n_ov_pre, n_ov_pre_nodes = bias_pre_stack(
                    f"vbo{out}n_ov",
                    f"vbo{out}n_ov_b",
                    bias_discharge_width_u,
                )
                lines += [
                    f"Mvbo{out}p_ov_b vbo{out}p bwd vbo{out}p_ov_b 0 NREL W={bias_discharge_width_u:.12g}u L=180n",
                    *p_ov_pre,
                    f"Mvbo{out}p_ov_d {p_ov_source} {overlap_write_gate} {pos_low_node} 0 {readout_write_gate_device} W={bias_discharge_width_u:.12g}u L=180n",
                    f"Mvbo{out}n_ov_b vbo{out}n bwd vbo{out}n_ov_b 0 NREL W={bias_discharge_width_u:.12g}u L=180n",
                    *n_ov_pre,
                    f"Mvbo{out}n_ov_d {n_ov_source} {overlap_write_gate} {neg_low_node} 0 {readout_write_gate_device} W={bias_discharge_width_u:.12g}u L=180n",
                ]
                lines += node_parasitics(
                    f"vbo{out}p_ov_b",
                    *p_ov_pre_nodes,
                    f"vbo{out}n_ov_b",
                    *n_ov_pre_nodes,
                )
        if output_bias_center_pull_width_u > 0:
            if readout_center_pull_mode == "always":
                lines += [
                    f"Mvbo{out}p_center vbo{out}p {readout_center_pull_gate} {output_bias_pos_center_pull_node} 0 NREL W={output_bias_center_pull_width_u:.12g}u L=180n",
                    f"Mvbo{out}n_center vbo{out}n {readout_center_pull_gate} {output_bias_neg_center_pull_node} 0 NREL W={output_bias_center_pull_width_u:.12g}u L=180n",
                ]
            else:
                lines += [
                    f"Mvbo{out}p_center_g vbo{out}p {readout_center_pull_gate} vbo{out}p_center_g 0 NREL W={output_bias_center_pull_width_u:.12g}u L=180n",
                    f"Mvbo{out}p_center_s vbo{out}p_center_g vbo{out}p {output_bias_pos_center_pull_node} 0 NREL W={output_bias_center_pull_width_u:.12g}u L=180n",
                    f"Mvbo{out}n_center_g vbo{out}n {readout_center_pull_gate} vbo{out}n_center_g 0 NREL W={output_bias_center_pull_width_u:.12g}u L=180n",
                    f"Mvbo{out}n_center_s vbo{out}n_center_g vbo{out}n {output_bias_neg_center_pull_node} 0 NREL W={output_bias_center_pull_width_u:.12g}u L=180n",
                ]
                lines += node_parasitics(f"vbo{out}p_center_g", f"vbo{out}n_center_g")
        for h in fanins[out]:
            if flow_pre_store == "shared_node":
                pre_gate = f"act{h}"
                pre_gate_low_true = None
            elif flow_pre_store == "synapse_boost":
                pre_gate = f"fprb{out}{h}"
                pre_gate_low_true = None
            elif flow_pre_store == "synapse_spike":
                pre_gate = f"fprg{out}{h}"
                pre_gate_low_true = f"fprbar{out}{h}"
            else:
                pre_gate = f"fpro{out}{h}"
                pre_gate_low_true = None
            lines += DirectFlowWeightCell(
                f"vw{out}{h}",
                pos_weight_node=f"vw{out}{h}p",
                neg_weight_node=f"vw{out}{h}n",
                pre_gate=pre_gate,
                positive_write_gate=positive_write_gate,
                negative_write_gate=negative_write_gate,
                pos_discharge_width_u=pos_discharge_width_u,
                neg_discharge_width_u=neg_discharge_width_u,
                pos_charge_width_u=pos_charge_width_u,
                neg_charge_width_u=neg_charge_width_u,
                positive_gate_neg_discharge_width_u=positive_gate_neg_discharge_width_u,
                negative_gate_pos_discharge_width_u=negative_gate_pos_discharge_width_u,
                positive_gate_pos_charge_width_u=positive_gate_pos_charge_width_u,
                negative_gate_neg_charge_width_u=negative_gate_neg_charge_width_u,
                pos_high_node=pos_high_node,
                pos_low_node=pos_low_node,
                neg_high_node=neg_high_node,
                neg_low_node=neg_low_node,
                write_gate_device=readout_write_gate_device,
                discharge_enabled=discharge_enabled,
                charge_enabled=charge_enabled,
                state_gate_discharge=state_gate_discharge,
                state_gate_charge=state_gate_charge,
                pmos_charge_write=pmos_charge_write,
                cmos_complementary_charge=cmos_complementary_write,
                positive_write_gate_low_true=positive_write_gate_low_true,
                negative_write_gate_low_true=negative_write_gate_low_true,
                pre_gate_low_true=pre_gate_low_true,
                overlap_write_gate=overlap_write_gate,
                center_pull_width_u=readout_center_pull_width_u,
                center_pull_gate=readout_center_pull_gate,
                center_pull_mode=readout_center_pull_mode,
                pos_center_pull_node=readout_pos_center_pull_node,
                neg_center_pull_node=readout_neg_center_pull_node,
            ).render_lines()
    return "\n".join(lines)


def flow_pre_activation_stores(
    mode: str,
    cap_f: float,
    consume_width_u: float,
    boost_width_u: float = 4.0,
    spike_ref_node: str = "spikeref",
    readout_fanins: ReadoutFanins | None = None,
    hidden_fanins: HiddenFanins | None = None,
) -> str:
    if mode not in FLOW_PRE_STORES:
        raise ValueError(f"unknown flow pre-store mode: {mode}")
    if mode == "shared_node":
        return "* Flow pre-activation storage: using shared source activation/input nodes."
    if cap_f <= 0 or consume_width_u <= 0:
        raise ValueError("flow pre-store capacitance and consume width must be positive.")
    if mode == "synapse_boost" and boost_width_u <= 0:
        raise ValueError("flow pre-store boost width must be positive for synapse_boost mode.")
    spike = mode == "synapse_spike"
    fanins = effective_readout_fanins(readout_fanins)
    hidden_edges = effective_hidden_fanins(hidden_fanins)
    deck = NetlistBuilder()
    readout_traces = PreTraceArray.from_readout_topology(
        "readout_pretrace_caps",
        FanInTopology.from_fanins(tuple(range(HIDDEN)), OUTPUTS, fanins),
        mode=mode,
        cap_f=cap_f,
        consume_width_u=consume_width_u,
        boost_width_u=boost_width_u,
        spike_ref_node=spike_ref_node,
    )
    readout_traces.comment = (
        "Per-synapse pre-activation traces are charged through MOS store paths during fwd for local direct-flow writes."
    )
    hidden_traces = PreTraceArray.from_hidden_topology(
        "hidden_pretrace_caps",
        FanInTopology.from_fanins(tuple(HIDDEN_RAILS), HIDDEN, hidden_edges),
        mode=mode,
        cap_f=cap_f,
        consume_width_u=consume_width_u,
        boost_width_u=boost_width_u,
        spike_ref_node=spike_ref_node,
    )
    deck.render_component(readout_traces)
    deck.render_component(hidden_traces)
    if spike:
        deck.extend(node_parasitics(*readout_traces.spike_mid_nodes(), *hidden_traces.spike_mid_nodes()))
    return deck.render_body()


def hidden_gradients_and_updates(
    update_width_u: float,
    hidden_gradient_act_gate: str,
    hidden_apply_mode: str,
    hidden_grad_sense_width_u: float,
    hidden_grad_sense_cap_f: float,
    design: SynapseDesign,
    hidden_fanins: HiddenFanins | None = None,
) -> str:
    if hidden_gradient_act_gate not in HIDDEN_GRADIENT_ACT_GATES:
        raise ValueError(f"unknown hidden gradient activation gate: {hidden_gradient_act_gate}")
    if hidden_apply_mode not in HIDDEN_APPLY_MODES:
        raise ValueError(f"unknown hidden apply mode: {hidden_apply_mode}")
    lines: list[str] = []
    fanins = effective_hidden_fanins(hidden_fanins)
    grad_w = design.hidden_gradient_width_u
    relu_model = "NSENSE" if hidden_gradient_act_gate == "act_nsense" else "NREL"
    for h in range(HIDDEN):
        for rail in fanins[h]:
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


def hidden_flow_updates(
    update_width_u: float,
    flow_pre_store: str,
    hidden_delta_output_mode: str,
    hidden_flow_write_mode: str,
    write_error_exclusion: str = "none",
    write_error_exclusion_width_u: float = 8.0,
    hidden_fanins: HiddenFanins | None = None,
) -> str:
    if flow_pre_store not in FLOW_PRE_STORES:
        raise ValueError(f"unknown flow pre-store mode: {flow_pre_store}")
    if hidden_delta_output_mode not in HIDDEN_DELTA_OUTPUT_MODES:
        raise ValueError(f"unknown hidden delta output mode: {hidden_delta_output_mode}")
    if hidden_flow_write_mode not in HIDDEN_FLOW_WRITE_MODES:
        raise ValueError(f"unknown hidden flow write mode: {hidden_flow_write_mode}")
    if write_error_exclusion not in WRITE_ERROR_EXCLUSION_MODES:
        raise ValueError(f"unknown write error exclusion mode: {write_error_exclusion}")
    if write_error_exclusion_width_u < 0:
        raise ValueError("write error exclusion width must be nonnegative.")
    bounded_write = hidden_flow_write_mode.startswith("bounded_")
    discharge_enabled = hidden_flow_write_mode in {
        "discharge",
        "bounded_discharge",
        "charge_discharge",
        "bounded_charge_discharge",
    }
    charge_enabled = hidden_flow_write_mode in {
        "charge_only",
        "bounded_charge_only",
        "charge_discharge",
        "bounded_charge_discharge",
    }
    high_node = "whigh" if bounded_write else "vdd"
    low_node = "wlow" if bounded_write else "0"
    lines: list[str] = []
    fanins = effective_hidden_fanins(hidden_fanins)
    for h in range(HIDDEN):
        positive_delta_raw = f"hdpg{h}" if hidden_delta_output_mode == "senseamp" else f"hdp{h}"
        negative_delta_raw = f"hdng{h}" if hidden_delta_output_mode == "senseamp" else f"hdn{h}"
        overlap_delta_gate: str | None = None
        if write_error_exclusion in {"pmos_inhibit", "pmos_inhibit_decay"}:
            pos_delta_gate = f"hwpos{h}"
            neg_delta_gate = f"hwneg{h}"
            lines += [
                f"* Exclusive hidden write rails for hidden {h}: block ambiguous hdp/hdn overlap.",
                f"Chwpos{h} {pos_delta_gate} 0 0.1f IC=0",
                f"Chwneg{h} {neg_delta_gate} 0 0.1f IC=0",
                f"Rhwpos{h} {pos_delta_gate} 0 1G",
                f"Rhwneg{h} {neg_delta_gate} 0 1G",
                f"Mreset_hwpos{h} {pos_delta_gate} rste 0 0 NMOS W=4u L=180n",
                f"Mreset_hwneg{h} {neg_delta_gate} rste 0 0 NMOS W=4u L=180n",
                f"Mhwpos{h}_inh vdd {negative_delta_raw} hwpos{h}_src vdd PMOS W={write_error_exclusion_width_u:.12g}u L=180n",
                f"Mhwpos{h}_gate hwpos{h}_src {positive_delta_raw} {pos_delta_gate} 0 NSENSE W={write_error_exclusion_width_u:.12g}u L=180n",
                f"Mhwneg{h}_inh vdd {positive_delta_raw} hwneg{h}_src vdd PMOS W={write_error_exclusion_width_u:.12g}u L=180n",
                f"Mhwneg{h}_gate hwneg{h}_src {negative_delta_raw} {neg_delta_gate} 0 NSENSE W={write_error_exclusion_width_u:.12g}u L=180n",
                f"Mhwpos{h}_kill {pos_delta_gate} {negative_delta_raw} 0 0 NMOS W={write_error_exclusion_width_u:.12g}u L=180n",
                f"Mhwneg{h}_kill {neg_delta_gate} {positive_delta_raw} 0 0 NMOS W={write_error_exclusion_width_u:.12g}u L=180n",
            ]
            lines += node_parasitics(f"hwpos{h}_src", f"hwneg{h}_src")
            if write_error_exclusion == "pmos_inhibit_decay":
                overlap_delta_gate = f"hwov{h}"
                lines += [
                    f"* Hidden overlap-decay rail for hidden {h}: hdp AND hdn gives symmetric common-mode weight decay.",
                    f"Chwov{h} {overlap_delta_gate} 0 0.1f IC=0",
                    f"Rhwov{h} {overlap_delta_gate} 0 1G",
                    f"Mreset_hwov{h} {overlap_delta_gate} rste 0 0 NMOS W=4u L=180n",
                    f"Mhwov{h}_p vdd {positive_delta_raw} hwov{h}_mid 0 NMOS W={write_error_exclusion_width_u:.12g}u L=180n",
                    f"Mhwov{h}_n hwov{h}_mid {negative_delta_raw} {overlap_delta_gate} 0 NMOS W={write_error_exclusion_width_u:.12g}u L=180n",
                ]
                lines += node_parasitics(f"hwov{h}_mid")
        elif write_error_exclusion == "diffpair_bleed":
            pos_delta_gate = f"hwpos{h}"
            neg_delta_gate = f"hwneg{h}"
            lines += diffpair_bleed_write_selector_lines(
                f"hwsel{h}",
                positive_delta_raw,
                negative_delta_raw,
                pos_delta_gate,
                neg_delta_gate,
                write_error_exclusion_width_u,
                f"hidden cell {h}",
            )
        else:
            pos_delta_gate = positive_delta_raw
            neg_delta_gate = negative_delta_raw
        for rail in fanins[h]:
            if flow_pre_store == "shared_node":
                pre_gate = rail
            elif flow_pre_store == "synapse_boost":
                pre_gate = f"fphib{h}_{rail}"
            elif flow_pre_store == "synapse_spike":
                pre_gate = f"fphig{h}_{rail}"
            else:
                pre_gate = f"fphi{h}_{rail}"
            if hidden_delta_output_mode == "raw" and discharge_enabled:
                lines += [
                    f"Mwh{h}_{rail}n_flow_b wh{h}_{rail}n bwd wh{h}_{rail}n_flow_b 0 NREL W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}n_flow_x wh{h}_{rail}n_flow_b {pre_gate} wh{h}_{rail}n_flow_x 0 NMOS W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}n_flow_d wh{h}_{rail}n_flow_x {pos_delta_gate} {low_node} 0 NSENSE W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}p_flow_b wh{h}_{rail}p bwd wh{h}_{rail}p_flow_b 0 NREL W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}p_flow_x wh{h}_{rail}p_flow_b {pre_gate} wh{h}_{rail}p_flow_x 0 NMOS W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}p_flow_d wh{h}_{rail}p_flow_x {neg_delta_gate} {low_node} 0 NSENSE W={update_width_u:.12g}u L=180n",
                ]
                lines += node_parasitics(
                    f"wh{h}_{rail}n_flow_b",
                    f"wh{h}_{rail}n_flow_x",
                    f"wh{h}_{rail}p_flow_b",
                    f"wh{h}_{rail}p_flow_x",
                )
            if hidden_delta_output_mode == "raw" and charge_enabled:
                lines += [
                    f"Mwh{h}_{rail}p_ch_b {high_node} bwd wh{h}_{rail}p_ch_b 0 NREL W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}p_ch_x wh{h}_{rail}p_ch_b {pre_gate} wh{h}_{rail}p_ch_x 0 NMOS W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}p_ch_d wh{h}_{rail}p_ch_x {pos_delta_gate} wh{h}_{rail}p 0 NSENSE W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}n_ch_b {high_node} bwd wh{h}_{rail}n_ch_b 0 NREL W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}n_ch_x wh{h}_{rail}n_ch_b {pre_gate} wh{h}_{rail}n_ch_x 0 NMOS W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}n_ch_d wh{h}_{rail}n_ch_x {neg_delta_gate} wh{h}_{rail}n 0 NSENSE W={update_width_u:.12g}u L=180n",
                ]
                lines += node_parasitics(
                    f"wh{h}_{rail}p_ch_b",
                    f"wh{h}_{rail}p_ch_x",
                    f"wh{h}_{rail}n_ch_b",
                    f"wh{h}_{rail}n_ch_x",
                )
            if overlap_delta_gate is not None and hidden_delta_output_mode == "raw" and discharge_enabled:
                lines += [
                    f"Mwh{h}_{rail}p_ov_b wh{h}_{rail}p bwd wh{h}_{rail}p_ov_b 0 NREL W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}p_ov_x wh{h}_{rail}p_ov_b {pre_gate} wh{h}_{rail}p_ov_x 0 NMOS W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}p_ov_d wh{h}_{rail}p_ov_x {overlap_delta_gate} {low_node} 0 NSENSE W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}n_ov_b wh{h}_{rail}n bwd wh{h}_{rail}n_ov_b 0 NREL W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}n_ov_x wh{h}_{rail}n_ov_b {pre_gate} wh{h}_{rail}n_ov_x 0 NMOS W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}n_ov_d wh{h}_{rail}n_ov_x {overlap_delta_gate} {low_node} 0 NSENSE W={update_width_u:.12g}u L=180n",
                ]
                lines += node_parasitics(
                    f"wh{h}_{rail}p_ov_b",
                    f"wh{h}_{rail}p_ov_x",
                    f"wh{h}_{rail}n_ov_b",
                    f"wh{h}_{rail}n_ov_x",
                )
            if hidden_delta_output_mode != "raw" and discharge_enabled:
                lines += [
                    f"Mwh{h}_{rail}n_flow_b wh{h}_{rail}n bwd wh{h}_{rail}n_flow_b 0 NREL W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}n_flow_x wh{h}_{rail}n_flow_b {pre_gate} wh{h}_{rail}n_flow_x 0 NMOS W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}n_flow_d wh{h}_{rail}n_flow_x {pos_delta_gate} wh{h}_{rail}n_flow_d 0 NSENSE W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}n_flow_a wh{h}_{rail}n_flow_d apply {low_node} 0 NREL W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}p_flow_b wh{h}_{rail}p bwd wh{h}_{rail}p_flow_b 0 NREL W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}p_flow_x wh{h}_{rail}p_flow_b {pre_gate} wh{h}_{rail}p_flow_x 0 NMOS W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}p_flow_d wh{h}_{rail}p_flow_x {neg_delta_gate} wh{h}_{rail}p_flow_d 0 NSENSE W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}p_flow_a wh{h}_{rail}p_flow_d apply {low_node} 0 NREL W={update_width_u:.12g}u L=180n",
                ]
                lines += node_parasitics(
                    f"wh{h}_{rail}n_flow_b",
                    f"wh{h}_{rail}n_flow_x",
                    f"wh{h}_{rail}n_flow_d",
                    f"wh{h}_{rail}p_flow_b",
                    f"wh{h}_{rail}p_flow_x",
                    f"wh{h}_{rail}p_flow_d",
                )
            if hidden_delta_output_mode != "raw" and charge_enabled:
                lines += [
                    f"Mwh{h}_{rail}p_ch_b {high_node} bwd wh{h}_{rail}p_ch_b 0 NREL W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}p_ch_x wh{h}_{rail}p_ch_b {pre_gate} wh{h}_{rail}p_ch_x 0 NMOS W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}p_ch_d wh{h}_{rail}p_ch_x {pos_delta_gate} wh{h}_{rail}p_ch_d 0 NSENSE W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}p_ch_a wh{h}_{rail}p_ch_d apply wh{h}_{rail}p 0 NREL W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}n_ch_b {high_node} bwd wh{h}_{rail}n_ch_b 0 NREL W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}n_ch_x wh{h}_{rail}n_ch_b {pre_gate} wh{h}_{rail}n_ch_x 0 NMOS W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}n_ch_d wh{h}_{rail}n_ch_x {neg_delta_gate} wh{h}_{rail}n_ch_d 0 NSENSE W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}n_ch_a wh{h}_{rail}n_ch_d apply wh{h}_{rail}n 0 NREL W={update_width_u:.12g}u L=180n",
                ]
                lines += node_parasitics(
                    f"wh{h}_{rail}p_ch_b",
                    f"wh{h}_{rail}p_ch_x",
                    f"wh{h}_{rail}p_ch_d",
                    f"wh{h}_{rail}n_ch_b",
                    f"wh{h}_{rail}n_ch_x",
                    f"wh{h}_{rail}n_ch_d",
                )
            if overlap_delta_gate is not None and hidden_delta_output_mode != "raw" and discharge_enabled:
                lines += [
                    f"Mwh{h}_{rail}p_ov_b wh{h}_{rail}p bwd wh{h}_{rail}p_ov_b 0 NREL W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}p_ov_x wh{h}_{rail}p_ov_b {pre_gate} wh{h}_{rail}p_ov_x 0 NMOS W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}p_ov_d wh{h}_{rail}p_ov_x {overlap_delta_gate} wh{h}_{rail}p_ov_d 0 NSENSE W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}p_ov_a wh{h}_{rail}p_ov_d apply {low_node} 0 NREL W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}n_ov_b wh{h}_{rail}n bwd wh{h}_{rail}n_ov_b 0 NREL W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}n_ov_x wh{h}_{rail}n_ov_b {pre_gate} wh{h}_{rail}n_ov_x 0 NMOS W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}n_ov_d wh{h}_{rail}n_ov_x {overlap_delta_gate} wh{h}_{rail}n_ov_d 0 NSENSE W={update_width_u:.12g}u L=180n",
                    f"Mwh{h}_{rail}n_ov_a wh{h}_{rail}n_ov_d apply {low_node} 0 NREL W={update_width_u:.12g}u L=180n",
                ]
                lines += node_parasitics(
                    f"wh{h}_{rail}p_ov_b",
                    f"wh{h}_{rail}p_ov_x",
                    f"wh{h}_{rail}p_ov_d",
                    f"wh{h}_{rail}n_ov_b",
                    f"wh{h}_{rail}n_ov_x",
                    f"wh{h}_{rail}n_ov_d",
                )
    return "\n".join(lines)


def measure_lines(
    samples: list[dict[str, Any]],
    hidden_apply_mode: str,
    learning_mode: str,
    hidden_delta_output_mode: str,
    measure_detail: str,
    readout_sample_offsets_ns: list[float],
    activation_sample_offsets_ns: list[float],
    cmp_start_ns: float,
    cmp_end_ns: float,
    bwd_start_ns: float,
    apply_end_ns: float,
    backward_gate_mode: str,
    hidden_delta_network_enabled: bool = True,
    output_head: str = "source_follower",
    *,
    flow_pre_store: str = "shared_node",
    readout_write_error_exclusion: str = "none",
    readout_fanins: ReadoutFanins | None = None,
    hidden_fanins: HiddenFanins | None = None,
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
    if output_head not in OUTPUT_HEAD_MODES:
        raise ValueError(f"unknown output head: {output_head}")
    if not readout_sample_offsets_ns:
        raise ValueError("at least one readout sample offset is required.")
    include_hidden_grad_measures = learning_mode == "accumulate_apply"
    include_train_detail = measure_detail == "full"
    include_signal_probe = measure_detail in {"full", "probe"}
    fanins = effective_readout_fanins(readout_fanins)
    hidden_edges = effective_hidden_fanins(hidden_fanins)
    default_offset = readout_sample_offsets_ns[0]
    cmp_probe_offset_ns = (cmp_start_ns + cmp_end_ns) / 2.0
    lead_probe_offset_ns = min(5.00, cmp_end_ns + 0.10)
    bwd_probe_offset_ns = min(apply_end_ns - 0.05, bwd_start_ns + 0.50)
    if bwd_probe_offset_ns <= bwd_start_ns:
        bwd_probe_offset_ns = (bwd_start_ns + apply_end_ns) / 2.0
    update_probe_offset_ns = min(apply_end_ns - 0.05, max(bwd_probe_offset_ns, 10.50))
    lines: list[str] = []
    prints: list[str] = []

    def append_score_measurements(out: int, idx: int, at: float, suffix: str) -> None:
        if output_head in DIODE_MIRROR_OUTPUT_HEADS:
            lines.extend(
                [
                    f".meas tran scorep{out}{suffix}_{idx} FIND V(scorep{out}) AT={at:.2f}n",
                    f".meas tran scoren{out}{suffix}_{idx} FIND V(scoren{out}) AT={at:.2f}n",
                    f".meas tran scorepm{out}{suffix}_{idx} FIND V(scorepm{out}) AT={at:.2f}n",
                    f".meas tran scorenm{out}{suffix}_{idx} FIND V(scorenm{out}) AT={at:.2f}n",
                    f".meas tran score{out}{suffix}_{idx} PARAM='scorenm{out}{suffix}_{idx}-scorepm{out}{suffix}_{idx}'",
                ]
            )
        elif output_head in SPLIT_SCORE_OUTPUT_HEADS:
            lines.extend(
                [
                    f".meas tran scorep{out}{suffix}_{idx} FIND V(scorep{out}) AT={at:.2f}n",
                    f".meas tran scoren{out}{suffix}_{idx} FIND V(scoren{out}) AT={at:.2f}n",
                    f".meas tran score{out}{suffix}_{idx} PARAM='scorep{out}{suffix}_{idx}-scoren{out}{suffix}_{idx}'",
                ]
            )
        else:
            lines.append(f".meas tran score{out}{suffix}_{idx} FIND V(score{out}) AT={at:.2f}n")

    activation_offsets = sorted({default_offset, *activation_sample_offsets_ns})
    for idx, sample in enumerate(samples):
        base = idx * CYCLE_NS
        label = int(sample["label"])
        if not 0 <= label < OUTPUTS:
            raise ValueError(f"sample label {label} is outside configured output count {OUTPUTS}.")
        lines += [
            f".meas tran target_out_{idx} FIND V(out{label}) AT={base + default_offset:.2f}n",
        ]
        if OUTPUTS == 2:
            other = 1 - label
            lines += [
                f".meas tran other_out_{idx} FIND V(out{other}) AT={base + default_offset:.2f}n",
                f".meas tran margin_{idx} PARAM='target_out_{idx}-other_out_{idx}'",
                f".meas tran out0_cmp_{idx} FIND V(out0) AT={base + cmp_probe_offset_ns:.2f}n",
                f".meas tran out1_cmp_{idx} FIND V(out1) AT={base + cmp_probe_offset_ns:.2f}n",
            ]
            append_score_measurements(0, idx, base + default_offset, "")
            append_score_measurements(1, idx, base + default_offset, "")
            append_score_measurements(0, idx, base + cmp_probe_offset_ns, "_cmp")
            append_score_measurements(1, idx, base + cmp_probe_offset_ns, "_cmp")
        else:
            for out in range(OUTPUTS):
                lines += [
                    f".meas tran out{out}_{idx} FIND V(out{out}) AT={base + default_offset:.2f}n",
                    f".meas tran out{out}_cmp_{idx} FIND V(out{out}) AT={base + cmp_probe_offset_ns:.2f}n",
                ]
                append_score_measurements(out, idx, base + default_offset, "")
                append_score_measurements(out, idx, base + cmp_probe_offset_ns, "_cmp")
        for offset in readout_sample_offsets_ns:
            key = offset_key(offset)
            if OUTPUTS == 2:
                other = 1 - label
                lines += [
                    f".meas tran target_out_{key}_{idx} FIND V(out{label}) AT={base + offset:.2f}n",
                    f".meas tran other_out_{key}_{idx} FIND V(out{other}) AT={base + offset:.2f}n",
                    f".meas tran margin_{key}_{idx} PARAM='target_out_{key}_{idx}-other_out_{key}_{idx}'",
                ]
                append_score_measurements(0, idx, base + offset, f"_{key}")
                append_score_measurements(1, idx, base + offset, f"_{key}")
            else:
                for out in range(OUTPUTS):
                    lines.append(f".meas tran out{out}_{key}_{idx} FIND V(out{out}) AT={base + offset:.2f}n")
                    append_score_measurements(out, idx, base + offset, f"_{key}")
        for h in range(HIDDEN):
            lines.append(f".meas tran act{h}_{idx} FIND V(act{h}) AT={base + default_offset:.2f}n")
            for offset in activation_offsets:
                key = offset_key(offset)
                lines.append(f".meas tran act{h}_{key}_{idx} FIND V(act{h}) AT={base + offset:.2f}n")
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
                if backward_gate_mode in {
                    "target_out_mistake_latch_restore",
                    "target_out_mistake_latch_restore_stacked",
                    "target_out_mistake_latch_restore_stacked_timed",
                }:
                    lines += [
                        f".meas tran merr0_bar_{idx} FIND V(merr0_bar) AT={base + bwd_probe_offset_ns:.2f}n",
                        f".meas tran merr1_bar_{idx} FIND V(merr1_bar) AT={base + bwd_probe_offset_ns:.2f}n",
                    ]
            applies_update = sample.get("apply_update", True)
            if include_train_detail and hidden_delta_network_enabled:
                lines += [
                    f".meas tran hdp0_guard_{idx} FIND V(hdp0) AT={base + bwd_probe_offset_ns:.2f}n",
                ]
            if applies_update and include_train_detail and OUTPUTS == 2:
                other = 1 - label
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
                    for h in fanins[out]:
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
                    if readout_write_error_exclusion == "diffpair_bleed":
                        lines += [
                            f".meas tran rwpos{out}_{idx} FIND V(rwpos{out}) AT={base + bwd_probe_offset_ns:.2f}n",
                            f".meas tran rwneg{out}_{idx} FIND V(rwneg{out}) AT={base + bwd_probe_offset_ns:.2f}n",
                            f".meas tran rwsel{out}_posbar_{idx} FIND V(rwsel{out}_posbar) AT={base + bwd_probe_offset_ns:.2f}n",
                            f".meas tran rwsel{out}_negbar_{idx} FIND V(rwsel{out}_negbar) AT={base + bwd_probe_offset_ns:.2f}n",
                            f".meas tran readout_write_select_net_{out}_{idx} PARAM='rwpos{out}_{idx}-rwneg{out}_{idx}'",
                            f".meas tran readout_write_select_bar_net_{out}_{idx} PARAM='rwsel{out}_negbar_{idx}-rwsel{out}_posbar_{idx}'",
                        ]
                    if flow_pre_store == "synapse_spike":
                        for h in fanins[out]:
                            lines += [
                                f".meas tran fprg{out}{h}_{idx} FIND V(fprg{out}{h}) AT={base + bwd_probe_offset_ns:.2f}n",
                                f".meas tran fprbar{out}{h}_{idx} FIND V(fprbar{out}{h}) AT={base + bwd_probe_offset_ns:.2f}n",
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
                    for rail in hidden_edges[h]:
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
                        for rail in hidden_edges[h]:
                            lines += [
                                f".meas tran wh{h}_{rail}p_before_{idx} FIND V(wh{h}_{rail}p) AT={base + 0.60:.2f}n",
                                f".meas tran wh{h}_{rail}n_before_{idx} FIND V(wh{h}_{rail}n) AT={base + 0.60:.2f}n",
                                f".meas tran wh{h}_{rail}p_after_{idx} FIND V(wh{h}_{rail}p) AT={base + 11.50:.2f}n",
                                f".meas tran wh{h}_{rail}n_after_{idx} FIND V(wh{h}_{rail}n) AT={base + 11.50:.2f}n",
                                f".meas tran wh{h}_{rail}_signed_before_{idx} PARAM='wh{h}_{rail}p_before_{idx}-wh{h}_{rail}n_before_{idx}'",
                                f".meas tran wh{h}_{rail}_signed_after_{idx} PARAM='wh{h}_{rail}p_after_{idx}-wh{h}_{rail}n_after_{idx}'",
                                f".meas tran d_wh{h}_{rail}_signed_{idx} PARAM='wh{h}_{rail}_signed_after_{idx}-wh{h}_{rail}_signed_before_{idx}'",
                            ]
        if OUTPUTS == 2:
            prints.append(f"print target_out_{idx} other_out_{idx} margin_{idx}")
        else:
            prints.append(f"print target_out_{idx}")
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
        for h in fanins[out]:
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
        for rail in hidden_edges[h]:
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
    label_shuffle_seed: int | None,
    dataset_name: str,
    train_order: list[int],
    batch_apply: bool,
    synapse_design_name: str,
    hidden_forward_mode: str,
    hidden_input_topology: str,
    hidden_input_fan_in: int,
    hidden_input_topology_seed: int,
    hidden_delta_width_scale: float,
    hidden_gradient_width_scale: float,
    readout_gradient_width_scale: float,
    output_forward_width_scale: float,
    output_forward_pos_width_scale: float,
    output_forward_neg_width_scale: float,
    output_bias_forward_width_scale: float,
    output_relu_width_scale: float,
    output_head: str,
    readout_topology: str,
    readout_fan_in: int,
    readout_fan_out: int,
    readout_topology_seed: int,
    hidden_error_rule: str,
    hidden_delta_relu_gate: str,
    hidden_delta_weight_device: str,
    hidden_delta_output_mode: str,
    hidden_delta_sense_width_u: float,
    hidden_delta_sense_cap_f: float,
    hidden_delta_internal_cap_f: float,
    hidden_delta_internal_leak_ohm: float,
    hidden_delta_internal_reset_width_u: float,
    hidden_gradient_act_gate: str,
    hidden_apply_mode: str,
    learning_mode: str,
    flow_hidden_write: str,
    flow_pre_store: str,
    readout_write_error_exclusion: str,
    readout_write_error_exclusion_width_u: float,
    hidden_write_error_exclusion: str,
    hidden_write_error_exclusion_width_u: float,
    flow_pre_cap_f: float,
    flow_pre_consume_width_u: float,
    flow_pre_boost_v: float,
    flow_pre_boost_width_u: float,
    flow_pre_spike_ref_v: float,
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
    readout_random_pos_center_v: float | None,
    readout_random_neg_center_v: float | None,
    readout_random_pos_span_v: float | None,
    readout_random_neg_span_v: float | None,
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
    score_cap_f: float,
    score_diode_width_u: float,
    score_mirror_cap_f: float,
    output_cap_f: float,
    readout_update_width_u: float,
    readout_pos_update_width_u: float | None,
    readout_neg_update_width_u: float | None,
    readout_charge_update_width_u: float | None,
    readout_discharge_update_width_u: float | None,
    readout_dp_gate_update_width_u: float | None,
    readout_dn_gate_update_width_u: float | None,
    readout_dp_discharge_gate_update_width_u: float | None,
    readout_dp_charge_gate_update_width_u: float | None,
    readout_dn_discharge_gate_update_width_u: float | None,
    readout_dn_charge_gate_update_width_u: float | None,
    output_bias_update_width_u: float,
    readout_center_pull_width_u: float,
    output_bias_center_pull_width_u: float,
    readout_center_pull_v: float,
    readout_pos_center_pull_v: float | None,
    readout_neg_center_pull_v: float | None,
    output_bias_pos_center_pull_v: float | None,
    output_bias_neg_center_pull_v: float | None,
    readout_center_pull_gate: str,
    readout_center_pull_mode: str,
    readout_write_state_gate_mode: str,
    readout_write_gate_device: str,
    output_bias_write_pre_gate: str,
    output_bias_flow_polarity: str,
    readout_write_high_v: float,
    readout_write_low_v: float,
    readout_pos_write_high_v: float | None,
    readout_pos_write_low_v: float | None,
    readout_neg_write_high_v: float | None,
    readout_neg_write_low_v: float | None,
    readout_flow_polarity: str,
    readout_flow_write_mode: str,
    hidden_update_width_u: float,
    hidden_flow_write_mode: str,
    error_rule: str,
    target_high_v: float,
    target_low_v: float,
    latch_boost_width_u: float,
    residual_target_width_u: float,
    residual_output_width_u: float,
    error_target_source_v: float | None,
    error_nontarget_source_v: float | None,
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
    activation_sample_offsets_ns: list[float],
    tran_step_ps: float = 10.0,
    spice_accuracy_preset: str = "standard",
    train_refire: bool = True,
    eval_each_epoch: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    design = scaled_synapse_design(
        synapse_design_name,
        hidden_delta_width_scale,
        hidden_gradient_width_scale,
        readout_gradient_width_scale,
        output_forward_width_scale,
        output_forward_pos_width_scale,
        output_forward_neg_width_scale,
        output_bias_forward_width_scale,
        output_relu_width_scale,
    )
    readout_pos_write_high = readout_write_high_v if readout_pos_write_high_v is None else readout_pos_write_high_v
    readout_pos_write_low = readout_write_low_v if readout_pos_write_low_v is None else readout_pos_write_low_v
    readout_neg_write_high = readout_write_high_v if readout_neg_write_high_v is None else readout_neg_write_high_v
    readout_neg_write_low = readout_write_low_v if readout_neg_write_low_v is None else readout_neg_write_low_v
    readout_pos_center_pull = (
        readout_center_pull_v if readout_pos_center_pull_v is None else readout_pos_center_pull_v
    )
    readout_neg_center_pull = (
        readout_center_pull_v if readout_neg_center_pull_v is None else readout_neg_center_pull_v
    )
    output_bias_pos_center_pull = (
        readout_center_pull_v if output_bias_pos_center_pull_v is None else output_bias_pos_center_pull_v
    )
    output_bias_neg_center_pull = (
        readout_center_pull_v if output_bias_neg_center_pull_v is None else output_bias_neg_center_pull_v
    )
    if tran_step_ps <= 0:
        raise ValueError("transient step must be positive.")
    spice_options = spice_options_for_preset(spice_accuracy_preset)
    include_gradient_caps = learning_mode == "accumulate_apply"
    state_seed = seed if init_seed is None else init_seed
    records = dataset_records(dataset_name, seed, root=ROOT)
    if label_shuffle_seed is not None:
        records = label_shuffled_records(records, label_shuffle_seed)
    set_output_count(max(int(record["label"]) for record in records) + 1)
    set_input_rails(input_rails_for_records(records))
    readout_fanins = build_readout_fanins(
        readout_topology,
        fan_in=readout_fan_in,
        fan_out=readout_fan_out,
        seed=readout_topology_seed,
    )
    hidden_fanins = build_hidden_fanins(
        hidden_input_topology,
        fan_in=hidden_input_fan_in,
        seed=hidden_input_topology_seed,
    )
    topology_summary = readout_topology_summary(readout_fanins)
    hidden_topology_info = hidden_topology_summary(hidden_fanins)
    samples = make_samples(records, epochs, train_order, batch_apply, eval_each_epoch)
    stop = len(samples) * CYCLE_NS
    hidden_delta_network_enabled = learning_mode != "flow" or flow_hidden_write == "direct"
    input_sources = "\n".join(
        f"V{rail} {rail} 0 {sample_wave(samples, rail, stop)}" for rail in INPUT_RAILS
    )
    target_sources = "\n".join(
        [
            *(
                f"Vt{out} t{out} 0 {target_wave(samples, out, stop, target_high_v, target_low_v)}"
                for out in range(OUTPUTS)
            ),
            *(
                f"Vnt{out} nt{out} 0 {target_wave(samples, out, stop, target_high_v, target_low_v, complement=True)}"
                for out in range(OUTPUTS)
            ),
        ]
    )
    meas, prints = measure_lines(
        samples,
        hidden_apply_mode,
        learning_mode,
        hidden_delta_output_mode,
        measure_detail,
        readout_sample_offsets_ns,
        activation_sample_offsets_ns,
        cmp_start_ns,
        cmp_end_ns,
        bwd_start_ns,
        apply_end_ns,
        backward_gate_mode,
        hidden_delta_network_enabled,
        output_head,
        flow_pre_store=flow_pre_store,
        readout_write_error_exclusion=readout_write_error_exclusion,
        readout_fanins=readout_fanins,
        hidden_fanins=hidden_fanins,
    )
    if learning_mode == "accumulate_apply":
        hidden_delta_block = hidden_delta(
            hidden_error_rule,
            hidden_delta_relu_gate,
            hidden_delta_weight_device,
            design,
            hidden_delta_internal_cap_f,
            hidden_delta_internal_leak_ohm,
            hidden_delta_internal_reset_width_u,
            readout_fanins,
        )
        hidden_delta_sense_block = hidden_delta_senseamps(
            hidden_delta_output_mode,
            hidden_delta_sense_width_u,
            hidden_delta_sense_cap_f,
        )
        learning_block = "\n".join(
            [
                readout_gradients_and_updates(
                    readout_update_width_u,
                    output_bias_update_width_u,
                    design,
                    readout_fanins,
                ),
                hidden_delta_sense_block,
                hidden_gradients_and_updates(
                    hidden_update_width_u,
                    hidden_gradient_act_gate,
                    hidden_apply_mode,
                    hidden_grad_sense_width_u,
                    hidden_grad_sense_cap_f,
                    design,
                    hidden_fanins,
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
                hidden_delta_internal_reset_width_u,
                readout_fanins,
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
                readout_pos_update_width_u,
                readout_neg_update_width_u,
                readout_charge_update_width_u,
                readout_discharge_update_width_u,
                readout_dp_gate_update_width_u,
                readout_dn_gate_update_width_u,
                readout_dp_discharge_gate_update_width_u,
                readout_dp_charge_gate_update_width_u,
                readout_dn_discharge_gate_update_width_u,
                readout_dn_charge_gate_update_width_u,
                readout_center_pull_width_u,
                output_bias_center_pull_width_u,
                readout_center_pull_gate,
                readout_center_pull_mode,
                readout_write_state_gate_mode,
                readout_write_gate_device,
                output_bias_write_pre_gate,
                output_bias_flow_polarity,
                "wphigh",
                "wplow",
                "wnhigh",
                "wnlow",
                "wcenterp",
                "wcentern",
                "wbocenterp",
                "wbocentern",
                readout_write_error_exclusion,
                readout_write_error_exclusion_width_u,
                readout_fanins,
            ),
            hidden_delta_sense_block,
        ]
        if flow_hidden_write == "direct":
            flow_blocks.append(
                hidden_flow_updates(
                    hidden_update_width_u,
                    flow_pre_store,
                    hidden_delta_output_mode,
                    hidden_flow_write_mode,
                    hidden_write_error_exclusion,
                    hidden_write_error_exclusion_width_u,
                    hidden_fanins,
                )
            )
        else:
            flow_blocks.append("* Hidden weight capacitors are held during this direct-flow run.")
        learning_block = "\n".join(
            flow_blocks
        )
    else:
        raise ValueError(f"unknown learning mode: {learning_mode}")
    feedback_block = (
        "\n* Fixed signed feedback-alignment weights.\n"
        + feedback_caps(feedback_init(state_seed, feedback_scale), hidden_cap_f, readout_fanins)
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
                readout_random_pos_center_v,
                readout_random_neg_center_v,
                readout_random_pos_span_v,
                readout_random_neg_span_v,
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
* Hidden input topology: {hidden_input_topology}; data edges={hidden_topology_info["hidden_input_edge_count"]}; fan-in={hidden_topology_info["hidden_input_fanin_counts"]}; fan-out={hidden_topology_info["hidden_input_fanout_counts"]}. Bias rails are retained per hidden cell.
* Readout topology: {readout_topology}; edges={topology_summary["readout_edge_count"]}; fan-in={topology_summary["readout_fanin_counts"]}; fan-out={topology_summary["readout_fanout_counts"]}.
* Output ReLU cells also have capacitor-held trainable bias weights.
* No hidden cell is wired to a specific literal pattern.
* Synapse design: {design.name}; hidden error rule: {hidden_error_rule}; learning mode: {learning_mode}.
.param VDD=1.2
{mos_models()}
Vdd vdd 0 {{VDD}}
Vbias bias 0 {{VDD}}
Vspikeref spikeref 0 {flow_pre_spike_ref_v:.12g}
Vscorecm scorecm 0 {score_reset_v:.12g}
Vwcenter wcenter 0 {readout_center_pull_v:.12g}
Vwcenterp wcenterp 0 {readout_pos_center_pull:.12g}
Vwcentern wcentern 0 {readout_neg_center_pull:.12g}
Vwbocenterp wbocenterp 0 {output_bias_pos_center_pull:.12g}
Vwbocentern wbocentern 0 {output_bias_neg_center_pull:.12g}
Vwhigh whigh 0 {readout_write_high_v:.12g}
Vwlow wlow 0 {readout_write_low_v:.12g}
Vwphigh wphigh 0 {readout_pos_write_high:.12g}
Vwplow wplow 0 {readout_pos_write_low:.12g}
Vwnhigh wnhigh 0 {readout_neg_write_high:.12g}
Vwnlow wnlow 0 {readout_neg_write_low:.12g}
{input_sources}
{target_sources}
{phases(samples, bwd_start_ns, apply_start_ns, apply_end_ns, cmp_start_ns, cmp_end_ns, learning_mode, backward_gate_mode, flow_pre_boost_v if learning_mode == "flow" and flow_pre_store == "synapse_boost" else None, train_refire)}

{persistent_caps(hidden_state, readout_state, hidden_cap_f, readout_fanins, hidden_fanins)}
{feedback_block}
{temporary_caps(gradient_cap_f, hidden_gradient_cap_f, hidden_delta_cap_f, lead_cap_f, include_gradient_caps, score_reset_v, score_cap_f, output_cap_f, output_head, readout_fanins, hidden_fanins)}
{resets(lead_mode, include_gradient_caps, score_reset_v, output_head, readout_fanins, hidden_fanins)}
{flow_pre_activation_stores(flow_pre_store, flow_pre_cap_f, flow_pre_consume_width_u, flow_pre_boost_width_u, "spikeref", readout_fanins, hidden_fanins) if learning_mode == "flow" else ""}
{train_charge_noise(samples, stop, train_charge_noise_width_u, train_charge_noise_probability, train_charge_noise_seed, train_charge_noise_scope, train_charge_noise_pulse_ns, bwd_start_ns, readout_fanins, hidden_fanins)}

    {hidden_forward(design, hidden_forward_mode, hidden_fanins)}
{output_forward(design, output_head, score_diode_width_u, score_mirror_cap_f, readout_fanins)}
{low_score_gate_cells(lose_pull_kohm, lose_width_u)}
{score_lead_gate_cells(lead_width_u, lead_mode)}
{backward_gate_cells(backward_gate_mode, backward_gate_width_u, backward_gate_cap_f, lead_mode)}
{error_cells(error_rule, latch_boost_width_u, residual_target_width_u, residual_output_width_u, lead_mode, error_target_source_v, error_nontarget_source_v)}
{hidden_delta_block}
{learning_block}

.options {spice_options}
.tran {tran_step_ps:.12g}p {stop:.2f}n uic
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
    proc = run_text_netlist(spice_bin, path, netlist, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-3000:])
    parsed = parse_measures(proc.stdout + "\n" + proc.stderr)
    if not parsed:
        raise RuntimeError("ngspice produced no parseable measurements:\n" + (proc.stdout + proc.stderr)[-3000:])
    return parsed


def experiment_interpretation(output_count: int) -> str:
    if output_count == 2:
        return (
            "This removes literal-detector hidden topology and tests whether a general dense hidden layer "
            "can run and update at device level on tiny binary datasets before moving to 8x8 MNIST."
        )
    return (
        "This exercises the general dense hidden/readout device path on a tiny multiclass dataset, "
        "including per-class score rails, one-vs-rest error rails, and transistor/passive readout writes."
    )


def write_summary_files(summary: dict[str, Any], summary_path: Path, table_summary_path: Path) -> None:
    text = json.dumps(summary, indent=2) + "\n"
    summary_path.write_text(text)
    table_summary_path.write_text(text)


def main() -> None:
    global CYCLE_NS
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument(
        "--simulator",
        default=None,
        help=(
            "Optional SPICE executable override. Use auto for ngspice-first compatibility, "
            "auto-fast for Xyce/XyceNF-first sweeps, or pass an executable path."
        ),
    )
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
        "--label-shuffle-seed",
        type=int,
        default=None,
        help="Negative-control mode: shuffle labels across fixed input records with this seed.",
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
    ap.add_argument(
        "--eval-each-epoch",
        action="store_true",
        help="Insert no-write evaluation cycles after each training epoch to expose transient best accuracy.",
    )
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
        "--output-forward-pos-width-scale",
        type=float,
        default=1.0,
        help="Additional scale for positive-branch forward readout devices.",
    )
    ap.add_argument(
        "--output-forward-neg-width-scale",
        type=float,
        default=1.0,
        help="Additional scale for negative-branch forward readout devices.",
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
    ap.add_argument(
        "--decision-source",
        choices=DECISION_SOURCES,
        default="out",
        help=(
            "Analog node family used as the classifier decision. 'out' uses the optional "
            "sense/output capacitors. 'score' uses the hardware score rails directly, which "
            "is useful when the score circuit is good but the output sense head is still under design."
        ),
    )
    ap.add_argument(
        "--readout-topology",
        choices=READOUT_TOPOLOGIES,
        default="dense",
        help=(
            "Hidden-to-output connectivity. random_fanout gives each hidden cell a fixed number of "
            "output synapses; balanced_random_fanout additionally keeps class fan-ins nearly equal."
        ),
    )
    ap.add_argument(
        "--readout-fan-in",
        type=int,
        default=3,
        help="For random_fanin readout topology, number of hidden synapses per output class.",
    )
    ap.add_argument(
        "--readout-fan-out",
        type=int,
        default=3,
        help="For random_fanout/balanced_random_fanout readout topology, number of output synapses per hidden cell.",
    )
    ap.add_argument(
        "--readout-topology-seed",
        type=int,
        default=0,
        help="Seed used to sample sparse hidden-to-output readout connectivity.",
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
    ap.add_argument(
        "--hidden-delta-internal-reset-width-u",
        type=float,
        default=0.0,
        help=(
            "Optional rste-gated NMOS reset width for hidden-delta internal stack nodes. "
            "Use with internal cap/leak damping to prevent stale charge from becoming state."
        ),
    )
    ap.add_argument("--hidden-gradient-act-gate", choices=HIDDEN_GRADIENT_ACT_GATES, default="act_nrel")
    ap.add_argument("--hidden-apply-mode", choices=HIDDEN_APPLY_MODES, default="direct")
    ap.add_argument(
        "--hidden-forward-mode",
        choices=HIDDEN_FORWARD_MODES,
        default="weighted_relu",
        help=(
            "Hidden forward circuit. weighted_relu uses trainable signed conductance into a ReLU cell; "
            "weighted_relu_pass_input passes the input/bias rail as the positive-source voltage through "
            "a weight-gated device; "
            "rail_buffer pass-gate copies input rails into hidden activation capacitors during fwd."
        ),
    )
    ap.add_argument(
        "--hidden-input-topology",
        choices=HIDDEN_INPUT_TOPOLOGIES,
        default="dense",
        help="Input-to-hidden connectivity. random_fanin keeps the bias rail and samples data input rails per hidden cell.",
    )
    ap.add_argument(
        "--hidden-input-fan-in",
        type=int,
        default=3,
        help="For random_fanin hidden input topology, number of data input rails per hidden cell; bias is added separately.",
    )
    ap.add_argument(
        "--hidden-input-topology-seed",
        type=int,
        default=0,
        help="Seed used to sample sparse input-to-hidden connectivity.",
    )
    ap.add_argument("--learning-mode", choices=LEARNING_MODES, default="accumulate_apply")
    ap.add_argument("--flow-hidden-write", choices=FLOW_HIDDEN_WRITES, default="direct")
    ap.add_argument("--flow-pre-store", choices=FLOW_PRE_STORES, default="shared_node")
    ap.add_argument(
        "--write-error-exclusion",
        choices=WRITE_ERROR_EXCLUSION_MODES,
        default="none",
        help=(
            "Optional transistor-generated mutual inhibition for local write rails. "
            "pmos_inhibit creates positive-write = positive-error AND not negative-error, "
            "and the symmetric negative-write rail. pmos_inhibit_decay also creates an "
            "overlap rail that applies symmetric common-mode decay when both error rails are high. "
            "diffpair_bleed uses a shared-tail differential pair plus weak bwd bleed, so equal "
            "common-mode dp/dn rails mostly decay instead of opening both write branches."
        ),
    )
    ap.add_argument(
        "--write-error-exclusion-width-u",
        type=float,
        default=8.0,
        help="PMOS/NMOS width for the optional exclusive write-rail generator.",
    )
    ap.add_argument(
        "--readout-write-error-exclusion",
        choices=WRITE_ERROR_EXCLUSION_MODES,
        default=None,
        help="Override --write-error-exclusion for readout weight and output-bias writes.",
    )
    ap.add_argument(
        "--readout-write-error-exclusion-width-u",
        type=float,
        default=None,
        help="Override --write-error-exclusion-width-u for readout write rails.",
    )
    ap.add_argument(
        "--hidden-write-error-exclusion",
        choices=WRITE_ERROR_EXCLUSION_MODES,
        default=None,
        help="Override --write-error-exclusion for hidden-weight writes.",
    )
    ap.add_argument(
        "--hidden-write-error-exclusion-width-u",
        type=float,
        default=None,
        help="Override --write-error-exclusion-width-u for hidden write rails.",
    )
    ap.add_argument("--flow-pre-cap-f", type=float, default=2.0)
    ap.add_argument("--flow-pre-consume-width-u", type=float, default=0.05)
    ap.add_argument(
        "--flow-pre-boost-v",
        type=float,
        default=0.75,
        help=(
            "Boost rail used by flow-pre-store=synapse_boost to generate a stronger "
            "backward/write pre-gate from the stored forward activation trace."
        ),
    )
    ap.add_argument(
        "--flow-pre-boost-width-u",
        type=float,
        default=4.0,
        help="Forward-store pass width for the bootstrapped write gate used by flow-pre-store=synapse_boost.",
    )
    ap.add_argument(
        "--flow-pre-spike-ref-v",
        type=float,
        default=0.30,
        help=(
            "Source reference for flow-pre-store=synapse_spike.  The stored activation must exceed "
            "this rail plus the NREL threshold before the full-swing eligibility gate fires."
        ),
    )
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
            "csv_readout",
            "csv_readout_rectified",
            "csv_readout_sparse_rectified",
            "csv_cap_state",
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
        "--readout-random-pos-center-v",
        type=float,
        default=None,
        help="Optional random initialization center for positive readout branch capacitors.",
    )
    ap.add_argument(
        "--readout-random-neg-center-v",
        type=float,
        default=None,
        help="Optional random initialization center for negative readout branch capacitors.",
    )
    ap.add_argument(
        "--readout-random-pos-span-v",
        type=float,
        default=None,
        help="Optional peak-to-peak random span for positive readout branch capacitors.",
    )
    ap.add_argument(
        "--readout-random-neg-span-v",
        type=float,
        default=None,
        help="Optional peak-to-peak random span for negative readout branch capacitors.",
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
    ap.add_argument(
        "--score-cap-f",
        type=float,
        default=10.0,
        help="Output score capacitor size in fF. Larger values keep the readout branch in a smaller-signal regime.",
    )
    ap.add_argument(
        "--score-diode-width-u",
        type=float,
        default=1024.0,
        help=(
            "Diode-connected score-load width for diode-loaded split-score heads. "
            "This is the production current-mirror input approximation for split score nodes."
        ),
    )
    ap.add_argument(
        "--score-mirror-cap-f",
        type=float,
        default=20.0,
        help="Current-mirror output capacitor size in fF for split_score_diode_mirror_caps.",
    )
    ap.add_argument(
        "--output-cap-f",
        type=float,
        default=20.0,
        help="Output capacitor size in fF. Larger values keep active-low/current-style output heads from saturating early.",
    )
    ap.add_argument("--update-width-u", type=float, default=120.0)
    ap.add_argument("--readout-update-width-u", type=float)
    ap.add_argument(
        "--readout-pos-update-width-u",
        type=float,
        help="Optional direct-flow update width for positive readout weight branches.",
    )
    ap.add_argument(
        "--readout-neg-update-width-u",
        type=float,
        help="Optional direct-flow update width for negative readout weight branches.",
    )
    ap.add_argument(
        "--readout-charge-update-width-u",
        type=float,
        help=(
            "Optional direct-flow update width for readout charge devices. "
            "When set, it overrides branch widths for the charge half of charge/discharge write modes."
        ),
    )
    ap.add_argument(
        "--readout-discharge-update-width-u",
        type=float,
        help=(
            "Optional direct-flow update width for readout discharge devices. "
            "When set, it overrides branch widths for the discharge half of discharge/charge_discharge write modes."
        ),
    )
    ap.add_argument(
        "--readout-dp-gate-update-width-u",
        type=float,
        help=(
            "Optional final write-gate width for devices controlled by dp error rails. "
            "This tunes target-row mobility in one-hot multiclass experiments without changing error-rail voltage."
        ),
    )
    ap.add_argument(
        "--readout-dn-gate-update-width-u",
        type=float,
        help=(
            "Optional final write-gate width for devices controlled by dn error rails. "
            "This tunes non-target-row mobility in one-hot multiclass experiments without changing error-rail voltage."
        ),
    )
    ap.add_argument(
        "--readout-dp-discharge-gate-update-width-u",
        type=float,
        help="Optional dp-controlled discharge gate width; overrides --readout-dp-gate-update-width-u for discharge legs.",
    )
    ap.add_argument(
        "--readout-dp-charge-gate-update-width-u",
        type=float,
        help="Optional dp-controlled charge gate width; overrides --readout-dp-gate-update-width-u for charge legs.",
    )
    ap.add_argument(
        "--readout-dn-discharge-gate-update-width-u",
        type=float,
        help="Optional dn-controlled discharge gate width; overrides --readout-dn-gate-update-width-u for discharge legs.",
    )
    ap.add_argument(
        "--readout-dn-charge-gate-update-width-u",
        type=float,
        help="Optional dn-controlled charge gate width; overrides --readout-dn-gate-update-width-u for charge legs.",
    )
    ap.add_argument("--output-bias-update-width-u", type=float)
    ap.add_argument(
        "--readout-center-pull-width-u",
        type=float,
        default=0.0,
        help=(
            "Optional weak MOS pass width that pulls each readout weight capacitor "
            "toward --readout-center-pull-v during the backward/write window."
        ),
    )
    ap.add_argument(
        "--output-bias-center-pull-width-u",
        type=float,
        default=0.0,
        help=(
            "Optional weak MOS pass width that pulls each output-bias capacitor "
            "toward --readout-center-pull-v during the backward/write window."
        ),
    )
    ap.add_argument(
        "--readout-center-pull-v",
        type=float,
        default=0.64,
        help="Global center rail used by the optional readout/output-bias center-pull pass devices.",
    )
    ap.add_argument(
        "--readout-pos-center-pull-v",
        type=float,
        default=None,
        help="Optional positive-branch center-pull rail; defaults to --readout-center-pull-v.",
    )
    ap.add_argument(
        "--readout-neg-center-pull-v",
        type=float,
        default=None,
        help="Optional negative-branch center-pull rail; defaults to --readout-center-pull-v.",
    )
    ap.add_argument(
        "--output-bias-pos-center-pull-v",
        type=float,
        default=None,
        help="Optional positive output-bias center-pull rail; defaults to --readout-center-pull-v.",
    )
    ap.add_argument(
        "--output-bias-neg-center-pull-v",
        type=float,
        default=None,
        help="Optional negative output-bias center-pull rail; defaults to --readout-center-pull-v.",
    )
    ap.add_argument(
        "--readout-center-pull-gate",
        choices=READOUT_CENTER_PULL_GATES,
        default="bwd",
        help=(
            "Waveform gate for optional readout center-pull devices. 'bwd' couples the pull to the "
            "analog mistake rail; 'apply' uses the later scheduled write window to avoid loading bwd."
        ),
    )
    ap.add_argument(
        "--readout-center-pull-mode",
        choices=READOUT_CENTER_PULL_MODES,
        default="always",
        help=(
            "Topology for optional readout center-pull devices. 'always' is a direct pass to the "
            "center rail; 'state_high' adds a second pass device gated by the weight cap itself, "
            "so centering only turns on after the state rises above the center rail."
        ),
    )
    ap.add_argument(
        "--readout-write-state-gate-mode",
        choices=READOUT_WRITE_STATE_GATE_MODES,
        default="none",
        help=(
            "Optional state-dependent gating inside the direct-flow readout write stack. "
            "'state_high_discharge' lets discharge flow only from high stored states; "
            "'state_window' also turns charge paths off as the stored state approaches the high rail."
        ),
    )
    ap.add_argument(
        "--readout-write-gate-device",
        choices=WRITE_GATE_DEVICES,
        default="NSENSE",
        help=(
            "MOS model for final error-controlled readout write gates. NSENSE is the historical "
            "low-threshold gate; NREL/NMOS trade write magnitude for stronger suppression of weak error rails."
        ),
    )
    ap.add_argument(
        "--output-bias-write-pre-gate",
        choices=OUTPUT_BIAS_WRITE_PRE_GATES,
        default="none",
        help=(
            "Optional pre-activation device inserted into output-bias write stacks. 'bias' makes "
            "the bias capacitor write topology match ordinary readout synapses with a constant "
            "bias pre-activation rail, reducing bias mobility mismatch."
        ),
    )
    ap.add_argument(
        "--output-bias-flow-polarity",
        choices=OUTPUT_BIAS_FLOW_POLARITIES,
        default="follow_readout",
        help=(
            "Output-bias write polarity for direct-flow mode. The default preserves the historical "
            "opposite-row mapping; explicit normal/reversed lets row and bias write signs be tested independently."
        ),
    )
    ap.add_argument(
        "--readout-flow-polarity",
        choices=READOUT_FLOW_POLARITIES,
        default="normal",
        help=(
            "Polarity of direct-flow readout discharges. normal drains negative caps on dp "
            "and positive caps on dn; reversed swaps those gates.  The isolated multiclass "
            "write-selectivity contract currently passes with reversed polarity for the "
            "scorep/scoren p-n score convention."
        ),
    )
    ap.add_argument(
        "--readout-flow-write-mode",
        choices=READOUT_FLOW_WRITE_MODES,
        default="discharge",
        help=(
            "Physical direct-flow readout write primitive. discharge drains the opposite branch; "
            "charge_only charges the branch matching the desired sign; charge_discharge does both. "
            "bounded_* modes use --readout-write-high-v/--readout-write-low-v instead of VDD/ground. "
            "bounded_pmos_charge_only and bounded_pmos_charge_discharge use low-true diffpair selector "
            "bars for high-side PMOS charge. bounded_cmos_charge_discharge additionally uses the "
            "low-true synapse_spike pretrace bar for a true PMOS charge stack."
        ),
    )
    ap.add_argument(
        "--readout-write-high-v",
        type=float,
        default=None,
        help=(
            "High rail for bounded_* local selected-branch readout/hidden writes. "
            "Defaults to 1.0 V for high-side PMOS charge writes and 0.58 V for "
            "the older NMOS bounded write modes."
        ),
    )
    ap.add_argument(
        "--readout-write-low-v",
        type=float,
        default=0.16,
        help="Low rail for bounded_* local selected-branch readout/hidden writes.",
    )
    ap.add_argument(
        "--readout-pos-write-high-v",
        type=float,
        default=None,
        help="Optional high rail for bounded positive readout-branch writes; defaults to --readout-write-high-v.",
    )
    ap.add_argument(
        "--readout-pos-write-low-v",
        type=float,
        default=None,
        help="Optional low rail for bounded positive readout-branch writes; defaults to --readout-write-low-v.",
    )
    ap.add_argument(
        "--readout-neg-write-high-v",
        type=float,
        default=None,
        help="Optional high rail for bounded negative readout-branch writes; defaults to --readout-write-high-v.",
    )
    ap.add_argument(
        "--readout-neg-write-low-v",
        type=float,
        default=None,
        help="Optional low rail for bounded negative readout-branch writes; defaults to --readout-write-low-v.",
    )
    ap.add_argument("--hidden-update-width-u", type=float)
    ap.add_argument(
        "--hidden-flow-write-mode",
        choices=HIDDEN_FLOW_WRITE_MODES,
        default="discharge",
        help=(
            "Physical direct-flow hidden-weight write primitive. Uses the same mode names "
            "as --readout-flow-write-mode; bounded_* modes use the readout write rails."
        ),
    )
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
            "ce_mirror_hybrid_limited",
            "ce_mirror_compete_limited",
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
            "lead_mistake",
            "lead_mistake_lowtarget",
            "lead_mistake_outlow",
            "local_loss",
        ],
        default="score",
    )
    ap.add_argument(
        "--target-high-v",
        type=float,
        default=1.1,
        help="High voltage used by target rails t* and complement target rails nt*.",
    )
    ap.add_argument(
        "--target-low-v",
        type=float,
        default=0.0,
        help="Low voltage used by target rails t* and complement target rails nt*.",
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
    ap.add_argument(
        "--error-target-source-v",
        type=float,
        default=None,
        help=(
            "Optional high rail for current-limited target error source ctsrc. "
            "Defaults to VDD; lower values tune analog CE/one-vs-rest target amplitude."
        ),
    )
    ap.add_argument(
        "--error-nontarget-source-v",
        type=float,
        default=None,
        help=(
            "Optional high rail for current-limited non-target error source cesrc. "
            "Defaults to VDD; lower values tune softmax-like negative pressure."
        ),
    )
    ap.add_argument(
        "--error-source-balance",
        choices=["none", "onehot_average"],
        default="none",
        help=(
            "Optional source-rail heuristic for onehot_limited. onehot_average sets "
            "ctsrc to the target high rail unless overridden and cesrc to ctsrc/(classes-1), "
            "so one target pulse approximately balances the non-target pulses."
        ),
    )
    ap.add_argument(
        "--error-nontarget-balance-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the auto-balanced onehot_limited non-target source rail.",
    )
    ap.add_argument("--lose-pull-kohm", type=float, default=100.0)
    ap.add_argument("--lose-width-u", type=float, default=24.0)
    ap.add_argument(
        "--lead-mode",
        choices=[
            "score",
            "score_charge",
            "score_direct",
            "lose",
            "senseamp",
            "senseamp_strong",
            "out_senseamp",
        ],
        default="score",
    )
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
    ap.add_argument(
        "--cycle-ns",
        type=float,
        default=CYCLE_NS,
        help=(
            "Per-sample transient cycle length. The default 16 ns keeps the historical post-write "
            "refire window; shorter screening cycles require --skip-train-refire."
        ),
    )
    ap.add_argument(
        "--skip-train-refire",
        action="store_true",
        help=(
            "Skip the diagnostic reset/refire window after each write. This preserves the main "
            "forward/error/backward/write phases and lets screening runs use shorter cycles."
        ),
    )
    ap.add_argument("--measure-detail", choices=MEASURE_DETAILS, default="full")
    ap.add_argument(
        "--readout-sample-offsets-ns",
        default="2.95",
        help="Comma-separated readout sampling offsets within each cycle. The first offset remains the compatibility accuracy point.",
    )
    ap.add_argument(
        "--activation-sample-offsets-ns",
        default="",
        help=(
            "Optional comma-separated activation sampling offsets within each cycle. "
            "Rows keep the compatibility act* columns at the first readout offset and add act*_<offset> columns."
        ),
    )
    ap.add_argument(
        "--tran-step-ps",
        type=float,
        default=10.0,
        help="Requested transient print/measurement step in ps. Larger values speed exploratory runs but must be calibrated.",
    )
    ap.add_argument(
        "--spice-accuracy-preset",
        choices=sorted(SPICE_ACCURACY_PRESETS),
        default="standard",
        help="SPICE option preset: standard keeps historical anchors; fast/loose relax tolerances for screening.",
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
        or args.output_forward_pos_width_scale <= 0
        or args.output_forward_neg_width_scale <= 0
        or args.output_bias_forward_width_scale <= 0
        or args.output_relu_width_scale <= 0
    ):
        raise SystemExit("synapse width scales must be positive.")
    if (
        args.hidden_delta_internal_cap_f < 0
        or args.hidden_delta_internal_leak_ohm < 0
        or args.hidden_delta_internal_reset_width_u < 0
    ):
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
    if args.score_cap_f <= 0:
        raise SystemExit("--score-cap-f must be positive.")
    if args.score_diode_width_u <= 0:
        raise SystemExit("--score-diode-width-u must be positive.")
    if args.score_mirror_cap_f <= 0:
        raise SystemExit("--score-mirror-cap-f must be positive.")
    if args.output_cap_f <= 0:
        raise SystemExit("--output-cap-f must be positive.")
    if args.hidden_grad_sense_width_u <= 0 or args.hidden_grad_sense_cap_f <= 0:
        raise SystemExit("hidden gradient sense width and capacitance must be positive.")
    if args.flow_pre_cap_f <= 0 or args.flow_pre_consume_width_u <= 0:
        raise SystemExit("flow pre-store capacitance and consume width must be positive.")
    if not 0.0 <= args.flow_pre_boost_v <= 1.2:
        raise SystemExit("--flow-pre-boost-v must be in 0..1.2 V.")
    if args.flow_pre_boost_width_u <= 0:
        raise SystemExit("--flow-pre-boost-width-u must be positive.")
    if not 0.0 <= args.flow_pre_spike_ref_v <= 1.2:
        raise SystemExit("--flow-pre-spike-ref-v must be in 0..1.2 V.")
    if args.write_error_exclusion_width_u <= 0:
        raise SystemExit("--write-error-exclusion-width-u must be positive.")
    readout_write_error_exclusion = (
        args.write_error_exclusion
        if args.readout_write_error_exclusion is None
        else args.readout_write_error_exclusion
    )
    hidden_write_error_exclusion = (
        args.write_error_exclusion
        if args.hidden_write_error_exclusion is None
        else args.hidden_write_error_exclusion
    )
    readout_write_error_exclusion_width_u = (
        args.write_error_exclusion_width_u
        if args.readout_write_error_exclusion_width_u is None
        else args.readout_write_error_exclusion_width_u
    )
    hidden_write_error_exclusion_width_u = (
        args.write_error_exclusion_width_u
        if args.hidden_write_error_exclusion_width_u is None
        else args.hidden_write_error_exclusion_width_u
    )
    if readout_write_error_exclusion_width_u <= 0:
        raise SystemExit("--readout-write-error-exclusion-width-u must be positive.")
    if hidden_write_error_exclusion_width_u <= 0:
        raise SystemExit("--hidden-write-error-exclusion-width-u must be positive.")
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
    if args.readout_pos_update_width_u is not None and args.readout_pos_update_width_u < 0:
        raise SystemExit("--readout-pos-update-width-u must be nonnegative.")
    if args.readout_neg_update_width_u is not None and args.readout_neg_update_width_u < 0:
        raise SystemExit("--readout-neg-update-width-u must be nonnegative.")
    if args.readout_dp_gate_update_width_u is not None and args.readout_dp_gate_update_width_u < 0:
        raise SystemExit("--readout-dp-gate-update-width-u must be nonnegative.")
    if args.readout_dn_gate_update_width_u is not None and args.readout_dn_gate_update_width_u < 0:
        raise SystemExit("--readout-dn-gate-update-width-u must be nonnegative.")
    if (
        args.readout_dp_discharge_gate_update_width_u is not None
        and args.readout_dp_discharge_gate_update_width_u < 0
    ):
        raise SystemExit("--readout-dp-discharge-gate-update-width-u must be nonnegative.")
    if args.readout_dp_charge_gate_update_width_u is not None and args.readout_dp_charge_gate_update_width_u < 0:
        raise SystemExit("--readout-dp-charge-gate-update-width-u must be nonnegative.")
    if (
        args.readout_dn_discharge_gate_update_width_u is not None
        and args.readout_dn_discharge_gate_update_width_u < 0
    ):
        raise SystemExit("--readout-dn-discharge-gate-update-width-u must be nonnegative.")
    if args.readout_dn_charge_gate_update_width_u is not None and args.readout_dn_charge_gate_update_width_u < 0:
        raise SystemExit("--readout-dn-charge-gate-update-width-u must be nonnegative.")
    if args.readout_charge_update_width_u is not None and args.readout_charge_update_width_u < 0:
        raise SystemExit("--readout-charge-update-width-u must be nonnegative.")
    if args.readout_discharge_update_width_u is not None and args.readout_discharge_update_width_u < 0:
        raise SystemExit("--readout-discharge-update-width-u must be nonnegative.")
    if args.output_bias_update_width_u is not None and args.output_bias_update_width_u < 0:
        raise SystemExit("--output-bias-update-width-u must be nonnegative.")
    if args.readout_center_pull_width_u < 0 or args.output_bias_center_pull_width_u < 0:
        raise SystemExit("--readout/output-bias center-pull widths must be nonnegative.")
    if not 0.0 <= args.readout_center_pull_v <= 1.2:
        raise SystemExit("--readout-center-pull-v must be in 0..1.2 V.")
    for name in [
        "readout_pos_center_pull_v",
        "readout_neg_center_pull_v",
        "output_bias_pos_center_pull_v",
        "output_bias_neg_center_pull_v",
    ]:
        value = getattr(args, name)
        if value is not None and not 0.0 <= value <= 1.2:
            raise SystemExit(f"--{name.replace('_', '-')} must be in 0..1.2 V.")
    if args.readout_write_high_v is None:
        args.readout_write_high_v = default_readout_write_high_v(args.readout_flow_write_mode)
    if not 0.0 <= args.readout_write_low_v < args.readout_write_high_v <= 1.2:
        raise SystemExit("--readout-write-low-v must be below --readout-write-high-v, both in 0..1.2 V.")
    readout_pos_write_high_v = (
        args.readout_write_high_v if args.readout_pos_write_high_v is None else args.readout_pos_write_high_v
    )
    readout_pos_write_low_v = (
        args.readout_write_low_v if args.readout_pos_write_low_v is None else args.readout_pos_write_low_v
    )
    readout_neg_write_high_v = (
        args.readout_write_high_v if args.readout_neg_write_high_v is None else args.readout_neg_write_high_v
    )
    readout_neg_write_low_v = (
        args.readout_write_low_v if args.readout_neg_write_low_v is None else args.readout_neg_write_low_v
    )
    if not 0.0 <= readout_pos_write_low_v < readout_pos_write_high_v <= 1.2:
        raise SystemExit("positive readout write rails must be ordered within 0..1.2 V.")
    if not 0.0 <= readout_neg_write_low_v < readout_neg_write_high_v <= 1.2:
        raise SystemExit("negative readout write rails must be ordered within 0..1.2 V.")
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
    if args.readout_random_pos_center_v is not None and not 0.01 <= args.readout_random_pos_center_v <= 1.15:
        raise SystemExit("--readout-random-pos-center-v must be in the capacitor-voltage range 0.01..1.15 V.")
    if args.readout_random_neg_center_v is not None and not 0.01 <= args.readout_random_neg_center_v <= 1.15:
        raise SystemExit("--readout-random-neg-center-v must be in the capacitor-voltage range 0.01..1.15 V.")
    if args.readout_random_span_v < 0:
        raise SystemExit("--readout-random-span-v must be nonnegative.")
    if args.readout_random_pos_span_v is not None and args.readout_random_pos_span_v < 0:
        raise SystemExit("--readout-random-pos-span-v must be nonnegative.")
    if args.readout_random_neg_span_v is not None and args.readout_random_neg_span_v < 0:
        raise SystemExit("--readout-random-neg-span-v must be nonnegative.")
    if abs(args.output_bias_offset_v) > 1.0:
        raise SystemExit("--output-bias-offset-v must be within +/-1.0 V.")
    try:
        readout_sample_offsets_ns = parse_offsets(args.readout_sample_offsets_ns)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        activation_sample_offsets_ns = parse_offsets(args.activation_sample_offsets_ns) if args.activation_sample_offsets_ns.strip() else []
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.tran_step_ps <= 0:
        raise SystemExit("--tran-step-ps must be positive.")
    if not 2.40 <= args.cmp_start_ns < args.cmp_end_ns <= 5.00:
        raise SystemExit("--cmp-start-ns/--cmp-end-ns must stay inside 2.40..5.00 ns with start < end.")
    if not 6.50 <= args.bwd_start_ns < args.apply_end_ns:
        raise SystemExit("--bwd-start-ns must start after error storage and before the backward/update window ends.")
    if not 9.0 <= args.apply_start_ns < args.apply_end_ns <= 11.8:
        raise SystemExit("--apply-start-ns/--apply-end-ns must stay inside the update window before refiring.")
    if args.cycle_ns <= 0:
        raise SystemExit("--cycle-ns must be positive.")
    if args.skip_train_refire:
        if args.cycle_ns < 11.60:
            raise SystemExit("--cycle-ns must be at least 11.60 ns when --skip-train-refire is enabled.")
    elif args.cycle_ns < 15.80:
        raise SystemExit("--cycle-ns below 15.80 ns requires --skip-train-refire.")
    CYCLE_NS = args.cycle_ns
    records = dataset_records(args.dataset, args.seed, root=ROOT)
    if args.label_shuffle_seed is not None:
        records = label_shuffled_records(records, args.label_shuffle_seed)
    try:
        set_output_count(max(int(record["label"]) for record in records) + 1)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        set_input_rails(input_rails_for_records(records))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        readout_fanins = build_readout_fanins(
            args.readout_topology,
            fan_in=args.readout_fan_in,
            fan_out=args.readout_fan_out,
            seed=args.readout_topology_seed,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    readout_topology_info = readout_topology_summary(readout_fanins)
    try:
        hidden_fanins = build_hidden_fanins(
            args.hidden_input_topology,
            fan_in=args.hidden_input_fan_in,
            seed=args.hidden_input_topology_seed,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    hidden_topology_info = hidden_topology_summary(hidden_fanins)
    if OUTPUTS != 2:
        if args.output_head not in {"source_follower"} | SPLIT_SCORE_OUTPUT_HEADS:
            raise SystemExit(
                "multi-class direct-flow runs currently require --output-head source_follower or a split-score head."
            )
        if args.error_rule not in {
            "score",
            "out_residual",
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
            "ce_mirror_hybrid_limited",
            "ce_mirror_compete_limited",
        }:
            raise SystemExit(
                "multi-class direct-flow runs currently require --error-rule score, out_residual, onehot, onehot_limited, onehot_out, ce_out, ce_split_score, ce_split_diffgate, ce_split_dpair, ce_split_compete, ce_split_current, ce_split_hybrid, ce_split_limited, ce_mirror_limited, ce_mirror_winner_limited, ce_mirror_hybrid_limited, or ce_mirror_compete_limited."
            )
        if args.backward_gate_mode != "scheduled":
            raise SystemExit("multi-class direct-flow runs currently require --backward-gate-mode scheduled.")
        if args.lead_mode != "score_direct":
            raise SystemExit("multi-class direct-flow runs currently require --lead-mode score_direct.")
        if args.readout_init not in {
            "random",
            "csv_readout",
            "csv_readout_rectified",
            "csv_readout_sparse_rectified",
            "csv_cap_state",
        }:
            raise SystemExit(
                "multi-class direct-flow runs currently require --readout-init random, csv_readout, csv_readout_rectified, csv_readout_sparse_rectified, or csv_cap_state."
            )
        if args.output_bias_offset_v != 0.0:
            raise SystemExit("multi-class direct-flow runs do not support --output-bias-offset-v yet.")
        if args.measure_detail == "full":
            raise SystemExit("multi-class direct-flow runs currently require --measure-detail light or probe.")
    if (
        args.error_rule
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
            "ce_mirror_hybrid_limited",
            "ce_mirror_compete_limited",
        }
        and args.output_head not in SPLIT_SCORE_OUTPUT_HEADS
    ):
        raise SystemExit(f"--error-rule {args.error_rule} requires a split-score output head.")
    if args.error_rule in {
        "ce_mirror_limited",
        "ce_mirror_winner_limited",
        "ce_mirror_hybrid_limited",
        "ce_mirror_compete_limited",
    } and args.output_head not in DIODE_MIRROR_OUTPUT_HEADS:
        raise SystemExit(f"--error-rule {args.error_rule} requires a diode-mirror split-score output head.")
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
    if args.readout_init in CSV_READOUT_INITS and args.separator_csv is None:
        raise SystemExit(f"--readout-init {args.readout_init} requires --separator-csv.")
    if not 0.0 <= args.target_low_v < args.target_high_v <= 1.2:
        raise SystemExit("--target-low-v must be below --target-high-v, both in 0..1.2 V.")
    if args.residual_target_width_u <= 0 or args.residual_output_width_u <= 0:
        raise SystemExit("--residual-target-width-u and --residual-output-width-u must be positive.")
    try:
        args.error_target_source_v, args.error_nontarget_source_v = resolve_error_source_rails(
            error_rule=args.error_rule,
            output_count=OUTPUTS,
            target_high_v=args.target_high_v,
            error_target_source_v=args.error_target_source_v,
            error_nontarget_source_v=args.error_nontarget_source_v,
            error_source_balance=args.error_source_balance,
            error_nontarget_balance_scale=args.error_nontarget_balance_scale,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.error_target_source_v is not None and not 0.0 < args.error_target_source_v <= 1.2:
        raise SystemExit("--error-target-source-v must be in 0..1.2 V.")
    if args.error_nontarget_source_v is not None and not 0.0 < args.error_nontarget_source_v <= 1.2:
        raise SystemExit("--error-nontarget-source-v must be in 0..1.2 V.")

    spice_bin, version = detect_spice(args.simulator)
    generated = ROOT / "spice/generated"
    results = ROOT / "spice/results"
    tables = ROOT / "results/tables"
    for directory in [generated, results, tables]:
        directory.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)

    readout_update_width = args.readout_update_width_u if args.readout_update_width_u is not None else args.update_width_u
    readout_pos_update_width = (
        readout_update_width if args.readout_pos_update_width_u is None else args.readout_pos_update_width_u
    )
    readout_neg_update_width = (
        readout_update_width if args.readout_neg_update_width_u is None else args.readout_neg_update_width_u
    )
    readout_charge_update_width = (
        None if args.readout_charge_update_width_u is None else args.readout_charge_update_width_u
    )
    readout_discharge_update_width = (
        None if args.readout_discharge_update_width_u is None else args.readout_discharge_update_width_u
    )
    output_bias_update_width = (
        args.output_bias_update_width_u
        if args.output_bias_update_width_u is not None
        else readout_update_width
    )
    effective_output_bias_discharge_width, effective_output_bias_charge_width = output_bias_flow_action_widths(
        output_bias_update_width,
        readout_charge_update_width,
        readout_discharge_update_width,
    )

    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag)
    netlist, samples = random_hidden_netlist(
        args.epochs,
        args.seed,
        args.init_seed,
        args.label_shuffle_seed,
        args.dataset,
        train_order,
        args.batch_apply,
        args.synapse_design,
        args.hidden_forward_mode,
        args.hidden_input_topology,
        args.hidden_input_fan_in,
        args.hidden_input_topology_seed,
        args.hidden_delta_width_scale,
        args.hidden_gradient_width_scale,
        args.readout_gradient_width_scale,
        args.output_forward_width_scale,
        args.output_forward_pos_width_scale,
        args.output_forward_neg_width_scale,
        args.output_bias_forward_width_scale,
        args.output_relu_width_scale,
        args.output_head,
        args.readout_topology,
        args.readout_fan_in,
        args.readout_fan_out,
        args.readout_topology_seed,
        args.hidden_error_rule,
        args.hidden_delta_relu_gate,
        args.hidden_delta_weight_device,
        args.hidden_delta_output_mode,
        args.hidden_delta_sense_width_u,
        args.hidden_delta_sense_cap_f,
        args.hidden_delta_internal_cap_f,
        args.hidden_delta_internal_leak_ohm,
        args.hidden_delta_internal_reset_width_u,
        args.hidden_gradient_act_gate,
        args.hidden_apply_mode,
        args.learning_mode,
        args.flow_hidden_write,
        args.flow_pre_store,
        readout_write_error_exclusion,
        readout_write_error_exclusion_width_u,
        hidden_write_error_exclusion,
        hidden_write_error_exclusion_width_u,
        args.flow_pre_cap_f,
        args.flow_pre_consume_width_u,
        args.flow_pre_boost_v,
        args.flow_pre_boost_width_u,
        args.flow_pre_spike_ref_v,
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
        args.readout_random_pos_center_v,
        args.readout_random_neg_center_v,
        args.readout_random_pos_span_v,
        args.readout_random_neg_span_v,
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
        args.score_cap_f,
        args.score_diode_width_u,
        args.score_mirror_cap_f,
        args.output_cap_f,
        readout_update_width,
        args.readout_pos_update_width_u,
        args.readout_neg_update_width_u,
        args.readout_charge_update_width_u,
        args.readout_discharge_update_width_u,
        args.readout_dp_gate_update_width_u,
        args.readout_dn_gate_update_width_u,
        args.readout_dp_discharge_gate_update_width_u,
        args.readout_dp_charge_gate_update_width_u,
        args.readout_dn_discharge_gate_update_width_u,
        args.readout_dn_charge_gate_update_width_u,
        output_bias_update_width,
        args.readout_center_pull_width_u,
        args.output_bias_center_pull_width_u,
        args.readout_center_pull_v,
        args.readout_pos_center_pull_v,
        args.readout_neg_center_pull_v,
        args.output_bias_pos_center_pull_v,
        args.output_bias_neg_center_pull_v,
        args.readout_center_pull_gate,
        args.readout_center_pull_mode,
        args.readout_write_state_gate_mode,
        args.readout_write_gate_device,
        args.output_bias_write_pre_gate,
        args.output_bias_flow_polarity,
        args.readout_write_high_v,
        args.readout_write_low_v,
        readout_pos_write_high_v,
        readout_pos_write_low_v,
        readout_neg_write_high_v,
        readout_neg_write_low_v,
        args.readout_flow_polarity,
        args.readout_flow_write_mode,
        args.hidden_update_width_u if args.hidden_update_width_u is not None else args.update_width_u,
        args.hidden_flow_write_mode,
        args.error_rule,
        args.target_high_v,
        args.target_low_v,
        args.latch_boost_width_u,
        args.residual_target_width_u,
        args.residual_output_width_u,
        args.error_target_source_v,
        args.error_nontarget_source_v,
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
        activation_sample_offsets_ns,
        args.tran_step_ps,
        args.spice_accuracy_preset,
        not args.skip_train_refire,
        args.eval_each_epoch,
    )
    t0 = time.perf_counter()
    parsed = run_netlist(spice_bin, generated / f"{safe_tag}.cir", netlist, args.timeout)

    hidden_delta_network_enabled = args.learning_mode != "flow" or args.flow_hidden_write == "direct"
    low_true_output = args.output_head in LOW_TRUE_OUTPUT_HEADS
    rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        phase = str(sample["phase"])
        label = int(sample["label"])
        if OUTPUTS == 2:
            other = 1 - label
            target_raw = parsed[f"target_out_{idx}"]
            other_raw = parsed[f"other_out_{idx}"]
            output_target = -target_raw if low_true_output else target_raw
            output_other = -other_raw if low_true_output else other_raw
            output_margin = output_target - output_other
            output_predicted_label = label if output_margin > 0 else 1 - label
            score_values = [parsed[f"score{out}_{idx}"] for out in range(OUTPUTS)]
            score_cmp_values = [parsed[f"score{out}_cmp_{idx}"] for out in range(OUTPUTS)]
            out_cmp_values = [parsed[f"out{out}_cmp_{idx}"] for out in range(OUTPUTS)]
        else:
            out_values = [parsed[f"out{out}_{idx}"] for out in range(OUTPUTS)]
            output_decision_values = [-value for value in out_values] if low_true_output else out_values
            output_target = output_decision_values[label]
            output_other = max(value for out, value in enumerate(output_decision_values) if out != label)
            output_margin = output_target - output_other
            output_predicted_label = int(np.argmax(output_decision_values))
            score_values = [parsed[f"score{out}_{idx}"] for out in range(OUTPUTS)]
            score_cmp_values = [parsed[f"score{out}_cmp_{idx}"] for out in range(OUTPUTS)]
            out_cmp_values = [parsed[f"out{out}_cmp_{idx}"] for out in range(OUTPUTS)]
        score_target = score_values[label]
        score_other = max(value for out, value in enumerate(score_values) if out != label)
        score_margin = score_target - score_other
        score_predicted_label = int(np.argmax(score_values))
        target_out, other_out, margin, predicted_label = selected_decision(
            decision_source=args.decision_source,
            output_target=output_target,
            output_other=output_other,
            output_predicted_label=output_predicted_label,
            score_target=score_target,
            score_other=score_other,
            score_predicted_label=score_predicted_label,
        )
        row: dict[str, Any] = {
            "cycle": idx,
            "phase": phase,
            "epoch": int(sample.get("epoch", 0)),
            "pattern": int(sample["pattern"]),
            "label": label,
            "reset_gradient": bool(sample.get("reset_gradient", False)),
            "applied_update": bool(sample.get("apply_update", False)),
            "target_out": target_out,
            "other_out": other_out,
            "margin": margin,
            "predicted_label": predicted_label,
            "correct": predicted_label == label,
            "output_target": output_target,
            "output_other": output_other,
            "output_margin": output_margin,
            "output_predicted_label": output_predicted_label,
            "output_correct": output_predicted_label == label,
            "target_score": score_target,
            "other_score": score_other,
            "score_margin": score_margin,
            "score_predicted_label": score_predicted_label,
            "score_correct": score_predicted_label == label,
            "mean_hidden_act": sum(parsed[f"act{h}_{idx}"] for h in range(HIDDEN)) / HIDDEN,
        }
        for out, value in enumerate(score_values):
            row[f"score{out}"] = value
            if args.output_head in SPLIT_SCORE_OUTPUT_HEADS:
                row[f"scorep{out}"] = parsed[f"scorep{out}_{idx}"]
                row[f"scoren{out}"] = parsed[f"scoren{out}_{idx}"]
                if args.output_head in DIODE_MIRROR_OUTPUT_HEADS:
                    row[f"scorepm{out}"] = parsed[f"scorepm{out}_{idx}"]
                    row[f"scorenm{out}"] = parsed[f"scorenm{out}_{idx}"]
        for out, value in enumerate(score_cmp_values):
            row[f"score{out}_cmp"] = value
            if args.output_head in SPLIT_SCORE_OUTPUT_HEADS:
                row[f"scorep{out}_cmp"] = parsed[f"scorep{out}_cmp_{idx}"]
                row[f"scoren{out}_cmp"] = parsed[f"scoren{out}_cmp_{idx}"]
                if args.output_head in DIODE_MIRROR_OUTPUT_HEADS:
                    row[f"scorepm{out}_cmp"] = parsed[f"scorepm{out}_cmp_{idx}"]
                    row[f"scorenm{out}_cmp"] = parsed[f"scorenm{out}_cmp_{idx}"]
        for out, value in enumerate(out_cmp_values):
            row[f"out{out}_cmp"] = value
        if OUTPUTS != 2:
            for out in range(OUTPUTS):
                row[f"out{out}"] = parsed[f"out{out}_{idx}"]
        for offset in readout_sample_offsets_ns:
            key = offset_key(offset)
            if OUTPUTS == 2:
                target_raw = parsed[f"target_out_{key}_{idx}"]
                other_raw = parsed[f"other_out_{key}_{idx}"]
                offset_target = -target_raw if low_true_output else target_raw
                offset_other = -other_raw if low_true_output else other_raw
                offset_output_margin = offset_target - offset_other
                offset_score_values = [parsed[f"score{out}_{key}_{idx}"] for out in range(OUTPUTS)]
                offset_score_target = offset_score_values[label]
                offset_score_other = offset_score_values[other]
                offset_score_margin = offset_score_target - offset_score_other
                row[f"target_out_{key}"] = offset_target
                row[f"other_out_{key}"] = offset_other
                row[f"output_margin_{key}"] = offset_output_margin
                row[f"target_score_{key}"] = offset_score_target
                row[f"other_score_{key}"] = offset_score_other
                row[f"score_margin_{key}"] = offset_score_margin
                if args.decision_source == "score":
                    row[f"margin_{key}"] = offset_score_margin
                    row[f"correct_{key}"] = offset_score_margin > 0
                else:
                    row[f"margin_{key}"] = offset_output_margin
                    row[f"correct_{key}"] = offset_output_margin > 0
            else:
                offset_values = [parsed[f"out{out}_{key}_{idx}"] for out in range(OUTPUTS)]
                offset_decision_values = [-value for value in offset_values] if low_true_output else offset_values
                offset_target = offset_decision_values[label]
                offset_other = max(value for out, value in enumerate(offset_decision_values) if out != label)
                offset_predicted = int(np.argmax(offset_decision_values))
                offset_score_values = [parsed[f"score{out}_{key}_{idx}"] for out in range(OUTPUTS)]
                offset_score_target = offset_score_values[label]
                offset_score_other = max(value for out, value in enumerate(offset_score_values) if out != label)
                offset_score_predicted = int(np.argmax(offset_score_values))
                row[f"target_out_{key}"] = offset_target
                row[f"other_out_{key}"] = offset_other
                row[f"output_margin_{key}"] = offset_target - offset_other
                row[f"target_score_{key}"] = offset_score_target
                row[f"other_score_{key}"] = offset_score_other
                row[f"score_margin_{key}"] = offset_score_target - offset_score_other
                if args.decision_source == "score":
                    row[f"margin_{key}"] = offset_score_target - offset_score_other
                    row[f"correct_{key}"] = offset_score_predicted == label
                else:
                    row[f"margin_{key}"] = offset_target - offset_other
                    row[f"correct_{key}"] = offset_predicted == label
        for h in range(HIDDEN):
            row[f"act{h}"] = parsed[f"act{h}_{idx}"]
            for offset in sorted({readout_sample_offsets_ns[0], *activation_sample_offsets_ns}):
                key = offset_key(offset)
                row[f"act{h}_{key}"] = parsed[f"act{h}_{key}_{idx}"]
        for out in range(OUTPUTS):
            row[f"lose{out}"] = parsed[f"lose{out}_{idx}"]
        row["lead01"] = parsed[f"lead01_{idx}"]
        row["lead10"] = parsed[f"lead10_{idx}"]
        if phase == "train":
            row["bwd_signal"] = parsed[f"bwd_signal_{idx}"]
            if "mistake_latch" in args.backward_gate_mode:
                row["merr0"] = parsed[f"merr0_{idx}"]
                row["merr1"] = parsed[f"merr1_{idx}"]
                if args.backward_gate_mode in {
                    "target_out_mistake_latch_restore",
                    "target_out_mistake_latch_restore_stacked",
                    "target_out_mistake_latch_restore_stacked_timed",
                }:
                    row["merr0_bar"] = parsed[f"merr0_bar_{idx}"]
                    row["merr1_bar"] = parsed[f"merr1_bar_{idx}"]
        if phase == "train" and args.measure_detail in {"full", "probe"}:
            for out in range(OUTPUTS):
                row[f"dp{out}"] = parsed[f"dp{out}_{idx}"]
                row[f"dn{out}"] = parsed[f"dn{out}_{idx}"]
                row[f"output_delta_net_{out}"] = parsed[f"output_delta_net_{out}_{idx}"]
                if args.readout_write_error_exclusion == "diffpair_bleed":
                    row[f"rwpos{out}"] = parsed[f"rwpos{out}_{idx}"]
                    row[f"rwneg{out}"] = parsed[f"rwneg{out}_{idx}"]
                    row[f"rwsel{out}_posbar"] = parsed[f"rwsel{out}_posbar_{idx}"]
                    row[f"rwsel{out}_negbar"] = parsed[f"rwsel{out}_negbar_{idx}"]
                    row[f"readout_write_select_net_{out}"] = parsed[f"readout_write_select_net_{out}_{idx}"]
                    row[f"readout_write_select_bar_net_{out}"] = parsed[
                        f"readout_write_select_bar_net_{out}_{idx}"
                    ]
                if args.flow_pre_store == "synapse_spike":
                    fprg_values = [parsed[f"fprg{out}{h}_{idx}"] for h in readout_fanins[out]]
                    fprbar_values = [parsed[f"fprbar{out}{h}_{idx}"] for h in readout_fanins[out]]
                    if fprg_values:
                        row[f"fprg{out}_on_fraction"] = float(np.mean(np.asarray(fprg_values) > 0.6))
                        row[f"fprg{out}_min"] = float(min(fprg_values))
                        row[f"fprg{out}_max"] = float(max(fprg_values))
                        row[f"fprbar{out}_min"] = float(min(fprbar_values))
                        row[f"fprbar{out}_max"] = float(max(fprbar_values))
            row["max_abs_output_delta_signal"] = max(
                abs(parsed[f"output_delta_net_{out}_{idx}"]) for out in range(OUTPUTS)
            )
            row["max_output_delta_node"] = max(
                max(abs(parsed[f"dp{out}_{idx}"]), abs(parsed[f"dn{out}_{idx}"])) for out in range(OUTPUTS)
            )
            if args.readout_write_error_exclusion == "diffpair_bleed":
                row["max_abs_readout_write_select_signal"] = max(
                    abs(parsed[f"readout_write_select_net_{out}_{idx}"]) for out in range(OUTPUTS)
                )
                row["max_abs_readout_write_select_bar_signal"] = max(
                    abs(parsed[f"readout_write_select_bar_net_{out}_{idx}"]) for out in range(OUTPUTS)
                )
                row["max_readout_write_select_node"] = max(
                    max(abs(parsed[f"rwpos{out}_{idx}"]), abs(parsed[f"rwneg{out}_{idx}"]))
                    for out in range(OUTPUTS)
                )
                row["min_readout_write_select_bar"] = min(
                    min(parsed[f"rwsel{out}_posbar_{idx}"], parsed[f"rwsel{out}_negbar_{idx}"])
                    for out in range(OUTPUTS)
                )
            if args.flow_pre_store == "synapse_spike":
                all_fprg_values = [
                    parsed[f"fprg{out}{h}_{idx}"] for out in range(OUTPUTS) for h in readout_fanins[out]
                ]
                all_fprbar_values = [
                    parsed[f"fprbar{out}{h}_{idx}"] for out in range(OUTPUTS) for h in readout_fanins[out]
                ]
                row["flow_pre_spike_on_fraction"] = float(np.mean(np.asarray(all_fprg_values) > 0.6))
                row["flow_pre_spike_gate_min"] = float(min(all_fprg_values))
                row["flow_pre_spike_gate_max"] = float(max(all_fprg_values))
                row["flow_pre_spike_bar_min"] = float(min(all_fprbar_values))
                row["flow_pre_spike_bar_max"] = float(max(all_fprbar_values))
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
                    for rail in hidden_fanins[h]
                )
                row["max_hidden_grad_node"] = max(
                    max(parsed[f"ghp{h}_{rail}_{idx}"], parsed[f"ghn{h}_{rail}_{idx}"])
                    for h in range(HIDDEN)
                    for rail in hidden_fanins[h]
                )
            if (
                args.learning_mode == "accumulate_apply"
                and args.hidden_apply_mode == "grad_senseamp"
                and sample.get("apply_update", True)
            ):
                row["max_abs_hidden_apply_gate_signal"] = max(
                    abs(parsed[f"hidden_apply_gate_net_{h}_{rail}_{idx}"])
                    for h in range(HIDDEN)
                    for rail in hidden_fanins[h]
                )
                row["max_hidden_apply_gate_node"] = max(
                    max(parsed[f"hgwp{h}_{rail}_{idx}"], parsed[f"hgwn{h}_{rail}_{idx}"])
                    for h in range(HIDDEN)
                    for rail in hidden_fanins[h]
                )
        if phase == "train" and sample.get("apply_update", True) and args.measure_detail == "full" and OUTPUTS == 2:
            readout_delta_values = [
                abs(parsed[f"d_vw{out}{h}_signed_{idx}"])
                for out in range(OUTPUTS)
                for h in readout_fanins[out]
            ]
            readout_weight_delta_by_out = {
                f"sum_d_readout_out{out}_signed": sum(
                    parsed[f"d_vw{out}{h}_signed_{idx}"] for h in readout_fanins[out]
                )
                for out in range(OUTPUTS)
            }
            output_bias_delta_by_out = {
                f"d_output_bias_out{out}_signed": parsed[f"d_vbo{out}_signed_{idx}"]
                for out in range(OUTPUTS)
            }
            row.update(
                {
                    "post_update_margin": parsed[f"train_margin_after_{idx}"],
                    "d_margin_after_update": parsed[f"train_d_margin_{idx}"],
                    "max_abs_readout_weight_signed_delta": max(
                        readout_delta_values or [0.0]
                    ),
                    "max_abs_output_bias_signed_delta": max(
                        abs(parsed[f"d_vbo{out}_signed_{idx}"]) for out in range(OUTPUTS)
                    ),
                    "max_abs_readout_signed_delta": max(
                        readout_delta_values
                        + [abs(parsed[f"d_vbo{out}_signed_{idx}"]) for out in range(OUTPUTS)]
                    ),
                    "max_abs_hidden_signed_delta": max(
                        abs(parsed[f"d_wh{h}_{rail}_signed_{idx}"])
                        for h in range(HIDDEN)
                        for rail in hidden_fanins[h]
                    ),
                }
                | readout_weight_delta_by_out
                | output_bias_delta_by_out
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
    if train.empty or OUTPUTS != 2:
        lead_tracks_score_winner = None
        lead_score_winner_fraction = None
        mean_train_lead01 = 0.0
        mean_train_lead10 = 0.0
        mean_abs_train_lead_diff = 0.0
    else:
        score0_wins = ((train["label"] == 0) & (train["margin"] > 0)) | (
            (train["label"] == 1) & (train["margin"] < 0)
        )
        if args.lead_mode == "score_direct":
            lead0_wins = train["score0_cmp"] > train["score1_cmp"]
        else:
            lead0_wins = lead_class0_wins(args.lead_mode, train["lead01"], train["lead10"])
        lead_score_winner_fraction = float((lead0_wins.to_numpy() == score0_wins.to_numpy()).mean())
        lead_tracks_score_winner = bool(lead_score_winner_fraction >= 0.75)
        mean_train_lead01 = float(train["lead01"].mean())
        mean_train_lead10 = float(train["lead10"].mean())
        mean_abs_train_lead_diff = float((train["lead01"] - train["lead10"]).abs().mean())
    total_readout_deltas = [
        parsed[f"d_vw{out}{h}_signed_total"]
        for out in range(OUTPUTS)
        for h in readout_fanins[out]
    ]
    total_readout_pos_branch_deltas = [
        parsed[f"vw{out}{h}p_final"] - parsed[f"vw{out}{h}p_initial"]
        for out in range(OUTPUTS)
        for h in readout_fanins[out]
    ]
    total_readout_neg_branch_deltas = [
        parsed[f"vw{out}{h}n_final"] - parsed[f"vw{out}{h}n_initial"]
        for out in range(OUTPUTS)
        for h in readout_fanins[out]
    ]
    total_readout_deltas_by_out = {
        f"total_readout_out{out}_signed_delta_v": sum(
            parsed[f"d_vw{out}{h}_signed_total"] for h in readout_fanins[out]
        )
        for out in range(OUTPUTS)
    }
    total_readout_pos_branch_deltas_by_out = {
        f"total_readout_out{out}_pos_branch_delta_v": sum(
            parsed[f"vw{out}{h}p_final"] - parsed[f"vw{out}{h}p_initial"] for h in readout_fanins[out]
        )
        for out in range(OUTPUTS)
    }
    total_readout_neg_branch_deltas_by_out = {
        f"total_readout_out{out}_neg_branch_delta_v": sum(
            parsed[f"vw{out}{h}n_final"] - parsed[f"vw{out}{h}n_initial"] for h in readout_fanins[out]
        )
        for out in range(OUTPUTS)
    }
    total_readout_common_deltas_by_out = {
        f"total_readout_out{out}_branch_common_delta_v": (
            total_readout_pos_branch_deltas_by_out[f"total_readout_out{out}_pos_branch_delta_v"]
            + total_readout_neg_branch_deltas_by_out[f"total_readout_out{out}_neg_branch_delta_v"]
        )
        for out in range(OUTPUTS)
    }
    total_readout_common_deltas = list(total_readout_common_deltas_by_out.values())
    readout_initial_branch_values = [
        parsed[f"vw{out}{h}{suffix}_initial"]
        for out in range(OUTPUTS)
        for h in readout_fanins[out]
        for suffix in ("p", "n")
    ]
    readout_final_branch_values = [
        parsed[f"vw{out}{h}{suffix}_final"]
        for out in range(OUTPUTS)
        for h in readout_fanins[out]
        for suffix in ("p", "n")
    ]
    max_abs_total_readout_signed_delta = max(abs(x) for x in total_readout_deltas)
    max_abs_total_readout_common_delta = max(abs(x) for x in total_readout_common_deltas)
    row_signed_deltas = list(total_readout_deltas_by_out.values())
    total_output_bias_deltas = [parsed[f"d_vbo{out}_signed_total"] for out in range(OUTPUTS)]
    total_output_bias_deltas_by_out = {
        f"total_output_bias_out{out}_signed_delta_v": parsed[f"d_vbo{out}_signed_total"]
        for out in range(OUTPUTS)
    }
    total_hidden_deltas = [
        parsed[f"d_wh{h}_{rail}_signed_total"]
        for h in range(HIDDEN)
        for rail in hidden_fanins[h]
    ]
    effective_design = scaled_synapse_design(
        args.synapse_design,
        args.hidden_delta_width_scale,
        args.hidden_gradient_width_scale,
        args.readout_gradient_width_scale,
        args.output_forward_width_scale,
        args.output_forward_pos_width_scale,
        args.output_forward_neg_width_scale,
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
    has_mistake_latch_bar_metrics = not train.empty and {"merr0_bar", "merr1_bar"}.issubset(train.columns)
    has_hidden_delta_update_metrics = (
        not train.empty and "max_abs_hidden_delta_update_signal" in train.columns
    )
    has_hidden_delta_gate_metrics = (
        not train.empty and "max_abs_hidden_delta_gate_signal" in train.columns
    )
    has_bwd_metrics = not train.empty and "bwd_signal" in train.columns
    has_readout_write_select_metrics = (
        not train.empty and "max_abs_readout_write_select_signal" in train.columns
    )
    readout_write_select_stats = (
        {
            "max_train_readout_write_select_signal_v": float(
                train["max_abs_readout_write_select_signal"].max()
            ),
            "mean_train_readout_write_select_signal_v": float(
                train["max_abs_readout_write_select_signal"].mean()
            ),
            "max_train_readout_write_select_bar_signal_v": float(
                train["max_abs_readout_write_select_bar_signal"].max()
            ),
            "mean_train_readout_write_select_bar_signal_v": float(
                train["max_abs_readout_write_select_bar_signal"].mean()
            ),
            "max_train_readout_write_select_node_v": float(train["max_readout_write_select_node"].max()),
            "min_train_readout_write_select_bar_v": float(train["min_readout_write_select_bar"].min()),
        }
        if has_readout_write_select_metrics
        else {}
    )
    has_flow_pre_spike_metrics = not train.empty and "flow_pre_spike_on_fraction" in train.columns
    flow_pre_spike_stats = (
        {
            "mean_train_flow_pre_spike_on_fraction": float(train["flow_pre_spike_on_fraction"].mean()),
            "min_train_flow_pre_spike_on_fraction": float(train["flow_pre_spike_on_fraction"].min()),
            "max_train_flow_pre_spike_on_fraction": float(train["flow_pre_spike_on_fraction"].max()),
            "min_train_flow_pre_spike_gate_v": float(train["flow_pre_spike_gate_min"].min()),
            "max_train_flow_pre_spike_gate_v": float(train["flow_pre_spike_gate_max"].max()),
            "min_train_flow_pre_spike_bar_v": float(train["flow_pre_spike_bar_min"].min()),
            "max_train_flow_pre_spike_bar_v": float(train["flow_pre_spike_bar_max"].max()),
        }
        if has_flow_pre_spike_metrics
        else {}
    )
    mistake_gate_stats = (
        target_mistake_gate_stats(train, args.lead_mode)
        if OUTPUTS == 2
        and args.backward_gate_mode in {
            "target_mistake",
            "target_mistake_latch",
            "target_mistake_latch_simple",
            "target_out_mistake_latch",
            "target_out_mistake_latch_restore",
            "target_out_mistake_latch_restore_stacked",
            "target_out_mistake_latch_restore_stacked_timed",
        }
        else {}
    )
    mistake_latch_stats = (
        target_mistake_latch_stats(train, args.lead_mode)
        if OUTPUTS == 2 and "mistake_latch" in args.backward_gate_mode
        else {}
    )
    output_error_stats = output_error_rail_stats(train, args.lead_mode) if OUTPUTS == 2 and has_output_delta_metrics else {}
    output_delta_alignment_stats = (
        output_delta_alignment_metrics(train, OUTPUTS) if has_output_delta_metrics else {}
    )
    train_output_delta_sums = output_delta_sums_by_out(train, OUTPUTS) if has_output_delta_metrics else None
    train_output_delta_sums_by_out = (
        {f"train_output_delta_out{out}_sum_v": value for out, value in enumerate(train_output_delta_sums)}
        if train_output_delta_sums is not None
        else {}
    )
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
    epoch_eval_stats = []
    for phase_name in sorted(
        (str(phase) for phase in df["phase"].unique() if re.fullmatch(r"epoch\d+_eval", str(phase))),
        key=lambda phase: int(re.search(r"\d+", phase).group(0)),
    ):
        subset = df[df["phase"] == phase_name]
        offset_stats = []
        for offset in readout_sample_offsets_ns:
            key = offset_key(offset)
            offset_stats.append(
                {
                    "offset_ns": offset,
                    "key": key,
                    "accuracy": float(subset[f"correct_{key}"].mean()),
                    "min_margin_v": float(subset[f"margin_{key}"].min()),
                }
            )
        best_offset = max(
            offset_stats,
            key=lambda item: (float(item["accuracy"]), float(item["min_margin_v"])),
        )
        epoch_eval_stats.append(
            {
                "phase": phase_name,
                "epoch": int(re.search(r"\d+", phase_name).group(0)),
                "accuracy": float(subset["correct"].mean()),
                "min_margin_v": float(subset["margin"].min()),
                "mean_hidden_activation_v": float(subset["mean_hidden_act"].mean()),
                "readout_offset_stats": offset_stats,
                "best_transient_offset_ns": best_offset["offset_ns"],
                "best_transient_accuracy": best_offset["accuracy"],
                "best_transient_min_margin_v": best_offset["min_margin_v"],
            }
        )
    eval_phase_stats = [
        {
            "phase": "initial_eval",
            "epoch": 0,
            "accuracy": float(initial_eval["correct"].mean()),
            "min_margin_v": float(initial_eval["margin"].min()),
        },
        *epoch_eval_stats,
        {
            "phase": "final_eval",
            "epoch": args.epochs,
            "accuracy": float(final_eval["correct"].mean()),
            "min_margin_v": float(final_eval["margin"].min()),
        },
    ]
    best_eval_phase = max(
        eval_phase_stats,
        key=lambda item: (float(item["accuracy"]), float(item["min_margin_v"])),
    )
    best_final_transient = max(
        readout_offset_stats,
        key=lambda item: (float(item["final_accuracy"]), float(item["final_min_margin_v"])),
    )
    initial_labels = initial_eval["label"].to_numpy(dtype=int)
    final_labels = final_eval["label"].to_numpy(dtype=int)
    initial_score_values = score_matrix(initial_eval, OUTPUTS)
    final_score_values = score_matrix(final_eval, OUTPUTS)
    initial_output_values = output_decision_matrix(initial_eval, OUTPUTS, low_true_output=low_true_output)
    final_output_values = output_decision_matrix(final_eval, OUTPUTS, low_true_output=low_true_output)
    summary = {
        "tag": safe_tag,
        "simulator": version,
        "architecture": (
            "device_level_binary_general_random_hidden"
            if OUTPUTS == 2
            else "device_level_multiclass_general_random_hidden"
        ),
        "status": "tiny_general_hidden_device_experiment",
        "benchmark": args.dataset,
        "dataset": args.dataset,
        "dataset_records": records,
        "label_shuffle_seed": args.label_shuffle_seed,
        "model_level": "ngspice built-in LEVEL=1 MOS models; not a foundry PDK.",
        "synapse_design": args.synapse_design,
        "synapse_design_description": SYNAPSE_DESIGNS[args.synapse_design].description,
        "readout_topology": args.readout_topology,
        "readout_topology_seed": args.readout_topology_seed,
        "readout_fan_in": args.readout_fan_in if args.readout_topology == "random_fanin" else None,
        "readout_fan_out": args.readout_fan_out
        if args.readout_topology in {"random_fanout", "balanced_random_fanout"}
        else None,
        **readout_topology_info,
        "hidden_delta_width_scale": args.hidden_delta_width_scale,
        "hidden_gradient_width_scale": args.hidden_gradient_width_scale,
        "readout_gradient_width_scale": args.readout_gradient_width_scale,
        "output_forward_width_scale": args.output_forward_width_scale,
        "output_forward_pos_width_scale": args.output_forward_pos_width_scale,
        "output_forward_neg_width_scale": args.output_forward_neg_width_scale,
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
        "hidden_delta_internal_reset_width_u": args.hidden_delta_internal_reset_width_u or None,
        "hidden_gradient_act_gate": args.hidden_gradient_act_gate,
        "hidden_apply_mode": args.hidden_apply_mode,
        "learning_mode": args.learning_mode,
        "measure_detail": args.measure_detail,
        "tran_step_ps": args.tran_step_ps,
        "spice_accuracy_preset": args.spice_accuracy_preset,
        "spice_options": spice_options_for_preset(args.spice_accuracy_preset),
        "cycle_ns": args.cycle_ns,
        "train_refire": not args.skip_train_refire,
        "flow_hidden_write": args.flow_hidden_write if args.learning_mode == "flow" else None,
        "flow_pre_store": args.flow_pre_store if args.learning_mode == "flow" else None,
        "flow_pre_cap_f": args.flow_pre_cap_f
        if args.learning_mode == "flow" and args.flow_pre_store != "shared_node"
        else None,
        "flow_pre_consume_width_u": args.flow_pre_consume_width_u
        if args.learning_mode == "flow" and args.flow_pre_store == "synapse_consume"
        else None,
        "flow_pre_boost_v": args.flow_pre_boost_v
        if args.learning_mode == "flow" and args.flow_pre_store == "synapse_boost"
        else None,
        "flow_pre_boost_width_u": args.flow_pre_boost_width_u
        if args.learning_mode == "flow" and args.flow_pre_store == "synapse_boost"
        else None,
        "flow_pre_spike_ref_v": args.flow_pre_spike_ref_v
        if args.learning_mode == "flow" and args.flow_pre_store == "synapse_spike"
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
        "uses_boosted_pre_activation_write_gate": args.learning_mode == "flow"
        and args.flow_pre_store == "synapse_boost",
        "uses_spiking_pre_activation_write_gate": args.learning_mode == "flow"
        and args.flow_pre_store == "synapse_spike",
        "pre_activation_capture_path": (
            "mos_store_trace_caps_plus_boosted_write_gate"
            if args.learning_mode == "flow" and args.flow_pre_store == "synapse_boost"
            else (
                "mos_store_trace_caps_plus_thresholded_full_swing_eligibility_gate"
                if args.learning_mode == "flow" and args.flow_pre_store == "synapse_spike"
                else (
                    "mos_store_trace_caps"
                    if args.learning_mode == "flow" and args.flow_pre_store != "shared_node"
                    else "shared_source_nodes"
                )
            )
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
        "output_count": OUTPUTS,
        "input_frontend": records[0].get("input_frontend") if records else None,
        "input_frontend_key": records[0].get("input_frontend_key") if records else None,
        "hidden_forward_mode": args.hidden_forward_mode,
        "hidden_input_topology": args.hidden_input_topology,
        "hidden_input_topology_seed": args.hidden_input_topology_seed,
        "hidden_input_fan_in": args.hidden_input_fan_in if args.hidden_input_topology == "random_fanin" else None,
        **hidden_topology_info,
        "signal_path": (
            (
                f"{min(HIDDEN, len(INPUT_RAILS))} hidden activation capacitors are MOS pass-gate buffered "
                f"from externally driven input rails; any remaining hidden cells use signed conductance from "
                f"{len(INPUT_RAILS)} input rails plus a bias rail. "
            )
            if args.hidden_forward_mode == "rail_buffer"
            else (
                f"{HIDDEN} hidden ReLU cells pass selected externally driven input/bias rails "
                f"as the positive-source voltage through capacitor-held weight gates, with signed negative "
                f"discharge branches, into the hidden pre-activation capacitors. "
            )
            if args.hidden_forward_mode == "weighted_relu_pass_input"
            else (
                f"{HIDDEN} hidden ReLU cells receive signed conductance from "
                f"{hidden_topology_info['hidden_input_edge_count']} selected data-input synapses plus "
                f"one bias rail per hidden cell. "
            )
        )
        + (
            "Readout, output-bias, and hidden weights are capacitor-held signed states. "
            f"Readout flow writes use {args.readout_flow_write_mode} signed updates. "
            f"Hidden flow writes use {args.hidden_flow_write_mode} signed updates."
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
        "separator_scale": args.separator_scale if args.readout_init in SEPARATOR_READOUT_INITS | PROGRAMMED_READOUT_INITS else None,
        "separator_offset_v": args.separator_offset_v if args.readout_init in SEPARATOR_READOUT_INITS else None,
        "readout_center_v": args.readout_center_v if args.readout_init in SEPARATOR_READOUT_INITS | PROGRAMMED_READOUT_INITS else None,
        "readout_random_center_v": args.readout_random_center_v if args.readout_init == "random" else None,
        "readout_random_span_v": (
            args.readout_random_span_v
            if args.readout_init == "random" and args.readout_random_center_v is not None
            else None
        ),
        "readout_random_pos_center_v": args.readout_random_pos_center_v if args.readout_init == "random" else None,
        "readout_random_neg_center_v": args.readout_random_neg_center_v if args.readout_init == "random" else None,
        "readout_random_pos_span_v": args.readout_random_pos_span_v if args.readout_init == "random" else None,
        "readout_random_neg_span_v": args.readout_random_neg_span_v if args.readout_init == "random" else None,
        "output_bias_offset_v": args.output_bias_offset_v,
        "separator_csv": str(args.separator_csv)
        if args.readout_init in CSV_READOUT_INITS
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
        "target_high_v": args.target_high_v,
        "target_low_v": args.target_low_v,
        "latch_boost_width_u": args.latch_boost_width_u
        if args.error_rule == "out_competitive_latchboost"
        else None,
        "residual_target_width_u": args.residual_target_width_u if args.error_rule in UPDATE_ERROR_RULES else None,
        "residual_output_width_u": args.residual_output_width_u if args.error_rule in UPDATE_ERROR_RULES else None,
        "error_target_source_v": args.error_target_source_v if args.error_rule in UPDATE_ERROR_RULES else None,
        "error_nontarget_source_v": args.error_nontarget_source_v if args.error_rule in UPDATE_ERROR_RULES else None,
        "error_source_balance": args.error_source_balance if args.error_rule in UPDATE_ERROR_RULES else "none",
        "error_nontarget_balance_scale": (
            args.error_nontarget_balance_scale
            if args.error_rule in UPDATE_ERROR_RULES and args.error_source_balance != "none"
            else None
        ),
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
        **mistake_latch_stats,
        **output_error_stats,
        **output_delta_alignment_stats,
        **train_output_delta_sums_by_out,
        **readout_write_select_stats,
        **flow_pre_spike_stats,
        "readout_row_delta_matches_output_delta_sum_fraction": (
            signed_alignment_fraction(row_signed_deltas, train_output_delta_sums)
            if train_output_delta_sums is not None
            else None
        ),
        "readout_row_delta_vs_output_delta_sum_cosine": (
            cosine_similarity(row_signed_deltas, train_output_delta_sums)
            if train_output_delta_sums is not None
            else None
        ),
        "max_train_mistake_latch_v": float(train[["merr0", "merr1"]].max().max())
        if has_mistake_latch_metrics
        else None,
        "mean_train_mistake_latch_v": float(train[["merr0", "merr1"]].to_numpy().mean())
        if has_mistake_latch_metrics
        else None,
        "min_train_mistake_latch_bar_v": float(train[["merr0_bar", "merr1_bar"]].min().min())
        if has_mistake_latch_bar_metrics
        else None,
        "mean_train_mistake_latch_bar_v": float(train[["merr0_bar", "merr1_bar"]].to_numpy().mean())
        if has_mistake_latch_bar_metrics
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
        "score_cap_f": args.score_cap_f,
        "score_diode_width_u": (
            args.score_diode_width_u
            if args.output_head in {"split_score_diode_diffpair", *DIODE_MIRROR_OUTPUT_HEADS}
            else None
        ),
        "score_mirror_cap_f": args.score_mirror_cap_f if args.output_head in DIODE_MIRROR_OUTPUT_HEADS else None,
        "output_cap_f": args.output_cap_f,
        "update_width_u": args.update_width_u,
        "readout_update_width_u": readout_update_width,
        "readout_pos_update_width_u": readout_pos_update_width if args.learning_mode == "flow" else None,
        "readout_neg_update_width_u": readout_neg_update_width if args.learning_mode == "flow" else None,
        "readout_charge_update_width_u": readout_charge_update_width if args.learning_mode == "flow" else None,
        "readout_discharge_update_width_u": readout_discharge_update_width if args.learning_mode == "flow" else None,
        "readout_dp_gate_update_width_u": args.readout_dp_gate_update_width_u
        if args.learning_mode == "flow"
        else None,
        "readout_dn_gate_update_width_u": args.readout_dn_gate_update_width_u
        if args.learning_mode == "flow"
        else None,
        "readout_dp_discharge_gate_update_width_u": args.readout_dp_discharge_gate_update_width_u
        if args.learning_mode == "flow"
        else None,
        "readout_dp_charge_gate_update_width_u": args.readout_dp_charge_gate_update_width_u
        if args.learning_mode == "flow"
        else None,
        "readout_dn_discharge_gate_update_width_u": args.readout_dn_discharge_gate_update_width_u
        if args.learning_mode == "flow"
        else None,
        "readout_dn_charge_gate_update_width_u": args.readout_dn_charge_gate_update_width_u
        if args.learning_mode == "flow"
        else None,
        "output_bias_update_width_u": output_bias_update_width,
        "effective_output_bias_discharge_update_width_u": (
            effective_output_bias_discharge_width if args.learning_mode == "flow" else None
        ),
        "effective_output_bias_charge_update_width_u": (
            effective_output_bias_charge_width if args.learning_mode == "flow" else None
        ),
        "readout_center_pull_width_u": args.readout_center_pull_width_u if args.learning_mode == "flow" else None,
        "output_bias_center_pull_width_u": args.output_bias_center_pull_width_u
        if args.learning_mode == "flow"
        else None,
        "readout_center_pull_v": args.readout_center_pull_v if args.learning_mode == "flow" else None,
        "readout_pos_center_pull_v": (
            (args.readout_pos_center_pull_v if args.readout_pos_center_pull_v is not None else args.readout_center_pull_v)
            if args.learning_mode == "flow"
            else None
        ),
        "readout_neg_center_pull_v": (
            (args.readout_neg_center_pull_v if args.readout_neg_center_pull_v is not None else args.readout_center_pull_v)
            if args.learning_mode == "flow"
            else None
        ),
        "output_bias_pos_center_pull_v": (
            (
                args.output_bias_pos_center_pull_v
                if args.output_bias_pos_center_pull_v is not None
                else args.readout_center_pull_v
            )
            if args.learning_mode == "flow"
            else None
        ),
        "output_bias_neg_center_pull_v": (
            (
                args.output_bias_neg_center_pull_v
                if args.output_bias_neg_center_pull_v is not None
                else args.readout_center_pull_v
            )
            if args.learning_mode == "flow"
            else None
        ),
        "readout_center_pull_gate": args.readout_center_pull_gate if args.learning_mode == "flow" else None,
        "readout_center_pull_mode": args.readout_center_pull_mode if args.learning_mode == "flow" else None,
        "readout_write_state_gate_mode": (
            args.readout_write_state_gate_mode if args.learning_mode == "flow" else None
        ),
        "readout_write_gate_device": args.readout_write_gate_device if args.learning_mode == "flow" else None,
        "output_bias_write_pre_gate": args.output_bias_write_pre_gate if args.learning_mode == "flow" else None,
        "output_bias_flow_polarity": args.output_bias_flow_polarity if args.learning_mode == "flow" else None,
        "readout_write_high_v": (
            args.readout_write_high_v
            if args.learning_mode == "flow"
            and (args.readout_flow_write_mode.startswith("bounded_") or args.hidden_flow_write_mode.startswith("bounded_"))
            else None
        ),
        "readout_write_low_v": (
            args.readout_write_low_v
            if args.learning_mode == "flow"
            and (args.readout_flow_write_mode.startswith("bounded_") or args.hidden_flow_write_mode.startswith("bounded_"))
            else None
        ),
        "readout_pos_write_high_v": readout_pos_write_high_v if args.learning_mode == "flow" else None,
        "readout_pos_write_low_v": readout_pos_write_low_v if args.learning_mode == "flow" else None,
        "readout_neg_write_high_v": readout_neg_write_high_v if args.learning_mode == "flow" else None,
        "readout_neg_write_low_v": readout_neg_write_low_v if args.learning_mode == "flow" else None,
        "readout_flow_polarity": args.readout_flow_polarity if args.learning_mode == "flow" else None,
        "readout_flow_write_mode": args.readout_flow_write_mode if args.learning_mode == "flow" else None,
        "write_error_exclusion": args.write_error_exclusion if args.learning_mode == "flow" else None,
        "write_error_exclusion_width_u": (
            args.write_error_exclusion_width_u
            if args.learning_mode == "flow" and args.write_error_exclusion != "none"
            else None
        ),
        "readout_write_error_exclusion": readout_write_error_exclusion if args.learning_mode == "flow" else None,
        "readout_write_error_exclusion_width_u": (
            readout_write_error_exclusion_width_u
            if args.learning_mode == "flow" and readout_write_error_exclusion != "none"
            else None
        ),
        "hidden_write_error_exclusion": (
            hidden_write_error_exclusion
            if args.learning_mode == "flow" and args.flow_hidden_write == "direct"
            else None
        ),
        "hidden_write_error_exclusion_width_u": (
            hidden_write_error_exclusion_width_u
            if args.learning_mode == "flow"
            and args.flow_hidden_write == "direct"
            and hidden_write_error_exclusion != "none"
            else None
        ),
        "hidden_update_width_u": args.hidden_update_width_u if args.hidden_update_width_u is not None else args.update_width_u,
        "hidden_flow_write_mode": (
            args.hidden_flow_write_mode
            if args.learning_mode == "flow" and args.flow_hidden_write == "direct"
            else None
        ),
        "apply_start_ns": args.apply_start_ns,
        "apply_end_ns": args.apply_end_ns,
        "apply_duration_ns": args.apply_end_ns - args.apply_start_ns,
        "readout_sample_offsets_ns": readout_sample_offsets_ns,
        "activation_sample_offsets_ns": sorted({readout_sample_offsets_ns[0], *activation_sample_offsets_ns}),
        "eval_each_epoch": args.eval_each_epoch,
        "epoch_eval_stats": epoch_eval_stats,
        "best_eval_phase": best_eval_phase["phase"],
        "best_eval_epoch": best_eval_phase["epoch"],
        "best_eval_accuracy": best_eval_phase["accuracy"],
        "best_eval_min_margin_v": best_eval_phase["min_margin_v"],
        "readout_offset_stats": readout_offset_stats,
        "readout_offset_decision_source": args.decision_source,
        "best_final_transient_offset_ns": best_final_transient["offset_ns"],
        "best_final_transient_accuracy": best_final_transient["final_accuracy"],
        "best_final_transient_min_margin_v": best_final_transient["final_min_margin_v"],
        "decision_source": args.decision_source,
        "initial_eval_accuracy": float(initial_eval["correct"].mean()),
        "final_eval_accuracy": float(final_eval["correct"].mean()),
        "initial_output_accuracy": float(initial_eval["output_correct"].mean()),
        "final_output_accuracy": float(final_eval["output_correct"].mean()),
        "initial_score_accuracy": float(initial_eval["score_correct"].mean()),
        "final_score_accuracy": float(final_eval["score_correct"].mean()),
        **column_centering_metrics("initial_score", initial_score_values, initial_labels),
        **column_centering_metrics("final_score", final_score_values, final_labels),
        **column_centering_metrics("initial_output", initial_output_values, initial_labels),
        **column_centering_metrics("final_output", final_output_values, final_labels),
        "initial_prediction_histogram": prediction_histogram(initial_eval, OUTPUTS),
        "final_prediction_histogram": prediction_histogram(final_eval, OUTPUTS),
        "initial_output_prediction_histogram": prediction_histogram_for(
            initial_eval, OUTPUTS, "output_predicted_label"
        ),
        "final_output_prediction_histogram": prediction_histogram_for(
            final_eval, OUTPUTS, "output_predicted_label"
        ),
        "initial_score_prediction_histogram": prediction_histogram_for(
            initial_eval, OUTPUTS, "score_predicted_label"
        ),
        "final_score_prediction_histogram": prediction_histogram_for(
            final_eval, OUTPUTS, "score_predicted_label"
        ),
        "initial_class_accuracy": per_class_accuracy(initial_eval, OUTPUTS),
        "final_class_accuracy": per_class_accuracy(final_eval, OUTPUTS),
        "initial_output_class_accuracy": per_class_accuracy_for(initial_eval, OUTPUTS, "output_correct"),
        "final_output_class_accuracy": per_class_accuracy_for(final_eval, OUTPUTS, "output_correct"),
        "initial_score_class_accuracy": per_class_accuracy_for(initial_eval, OUTPUTS, "score_correct"),
        "final_score_class_accuracy": per_class_accuracy_for(final_eval, OUTPUTS, "score_correct"),
        "input_feature_separability": input_feature_separability(records),
        "initial_hidden_feature_separability": perceptron_separable(initial_eval),
        "final_hidden_feature_separability": perceptron_separable(final_eval),
        "initial_min_margin_v": float(initial_eval["margin"].min()),
        "final_min_margin_v": float(final_eval["margin"].min()),
        "initial_output_min_margin_v": float(initial_eval["output_margin"].min()),
        "final_output_min_margin_v": float(final_eval["output_margin"].min()),
        "initial_score_min_margin_v": float(initial_eval["score_margin"].min()),
        "final_score_min_margin_v": float(final_eval["score_margin"].min()),
        "min_margin_gain_v": float((final_eval["margin"].to_numpy() - initial_eval["margin"].to_numpy()).min()),
        "output_min_margin_gain_v": float(
            (final_eval["output_margin"].to_numpy() - initial_eval["output_margin"].to_numpy()).min()
        ),
        "score_min_margin_gain_v": float(
            (final_eval["score_margin"].to_numpy() - initial_eval["score_margin"].to_numpy()).min()
        ),
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
        "initial_readout_branch_min_v": float(min(readout_initial_branch_values)),
        "initial_readout_branch_max_v": float(max(readout_initial_branch_values)),
        "final_readout_branch_min_v": float(min(readout_final_branch_values)),
        "final_readout_branch_max_v": float(max(readout_final_branch_values)),
        "max_abs_total_readout_weight_signed_delta_v": float(max_abs_total_readout_signed_delta),
        "max_abs_total_readout_pos_branch_delta_v": float(
            max(abs(x) for x in total_readout_pos_branch_deltas)
        ),
        "max_abs_total_readout_neg_branch_delta_v": float(
            max(abs(x) for x in total_readout_neg_branch_deltas)
        ),
        "max_abs_total_readout_common_delta_v": float(max_abs_total_readout_common_delta),
        "readout_common_to_signed_delta_ratio": float(
            max_abs_total_readout_common_delta / max(max_abs_total_readout_signed_delta, 1e-12)
        ),
        "max_abs_readout_common_delta_per_applied_sample_v": float(
            max_abs_total_readout_common_delta / max(len(applied_train), 1)
        ),
        "readout_row_signed_delta_mean_v": float(np.mean(row_signed_deltas)),
        "readout_row_signed_delta_std_v": float(np.std(row_signed_deltas)),
        "readout_row_negative_delta_count": int(sum(delta < 0 for delta in row_signed_deltas)),
        "readout_row_positive_delta_count": int(sum(delta > 0 for delta in row_signed_deltas)),
        "max_abs_total_output_bias_signed_delta_v": float(max(abs(x) for x in total_output_bias_deltas)),
        "max_abs_total_readout_signed_delta_v": float(
            max([abs(x) for x in total_readout_deltas] + [abs(x) for x in total_output_bias_deltas])
        ),
        **total_readout_deltas_by_out,
        **total_readout_pos_branch_deltas_by_out,
        **total_readout_neg_branch_deltas_by_out,
        **total_readout_common_deltas_by_out,
        **total_output_bias_deltas_by_out,
        "max_abs_total_hidden_signed_delta_v": float(max(abs(x) for x in total_hidden_deltas)),
        "curve": str(curve_path),
        "table_curve": str(table_path),
        "wall_time_s": time.perf_counter() - t0,
        "interpretation": experiment_interpretation(OUTPUTS),
    }
    summary_path = results / f"{safe_tag}_summary.json"
    table_summary_path = tables / f"{safe_tag}_summary.json"
    write_summary_files(summary, summary_path, table_summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
