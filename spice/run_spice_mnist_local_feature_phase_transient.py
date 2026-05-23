from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from run_spice_mnist_batch_op_train import read_wrdata_row
from local_feature_error import (
    append_target_margin_gate,
    class_centered_expr,
    mean_centered_expr,
    softmax_delta_exprs,
    softmax_exp_expr,
)
from run_spice_mnist_local_block_batch_op_train import (
    local_activation_deriv_expr as block_local_activation_deriv_expr,
    local_activation_expr as block_local_activation_expr,
    block_indices,
    readout_feedback_expr,
    synapse_transfer_expr,
)
from run_spice_mnist_local_feature_batch_op_train import run_eval, run_train_batch, wrap_xyce_behavioral_rhs, xyce_prn_path
from run_spice_mnist_train import load_mnist_sequence, mnist_index_splits
from run_spice_sweep import ROOT, detect_spice, is_xyce, prepare_netlist_for_simulator, run_tiny_test, run_simulator_netlist
from spice_adapter import SPICE_SIMULATOR_ARGS_ENV

TrainState = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
MNIST_TRAIN_COUNT = 60000
MNIST_TEST_COUNT = 10000


@dataclass(frozen=True)
class SampleDrive:
    node: str
    source: str
    expr: str
    line: str | None
    constant_value: float | None

    @property
    def elided(self) -> bool:
        return self.line is None


def sanitize_tag(tag: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in tag)


def phase_pwl(pulses: list[tuple[float, float]], t_stop: float, edge: float) -> str:
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for t_on, t_off in pulses:
        points.append((max(0.0, t_on - edge), 0.0))
        points.append((t_on, 1.0))
        points.append((t_off, 1.0))
        points.append((min(t_stop, t_off + edge), 0.0))
    cleaned: list[tuple[float, float]] = []
    for t, val in sorted(points):
        if cleaned and abs(cleaned[-1][0] - t) < 1e-18:
            cleaned[-1] = (t, val)
        else:
            cleaned.append((t, val))
    return "PWL(" + " ".join(f"{t:.12g} {val:.12g}" for t, val in cleaned) + ")"


def direct_phase_clock_period(phase: float, gap: float) -> float:
    if phase <= 0.0:
        raise ValueError("phase must be positive")
    if gap < 0.0:
        raise ValueError("gap must be non-negative")
    return 5.0 * phase + 6.0 * gap


def analytic_phase_clock_expr(
    pulses: list[tuple[float, float]],
    phase: float,
    gap: float,
    edge: float,
) -> str:
    if not pulses:
        return "0"
    if edge <= 0.0:
        raise ValueError("analytic phase clocks require positive edge")
    period = direct_phase_clock_period(phase, gap)
    first_on, first_off = pulses[0]
    last_on, last_off = pulses[-1]
    width_tol = max(1e-18, abs(phase) * 1e-9)
    period_tol = max(1e-18, abs(period) * 1e-9)
    for idx, (t_on, t_off) in enumerate(pulses):
        if abs((t_off - t_on) - phase) > width_tol:
            raise ValueError("analytic phase clocks require uniform phase width")
        if idx and abs((t_on - pulses[idx - 1][0]) - period) > period_tol:
            raise ValueError("analytic phase clocks require direct-mode periodic phases")
    start = first_on - edge
    stop = last_off + edge
    rel = f"(time-({start:.12g}))"
    r = f"({rel}-floor({rel}/({period:.12g}))*({period:.12g}))"
    return (
        f"if((time < {start:.12g}) | (time > {stop:.12g}), 0, "
        f"if({r} < {edge:.12g}, {r}/({edge:.12g}), "
        f"if({r} < {(edge + phase):.12g}, 1, "
        f"if({r} < {(edge + phase + edge):.12g}, 1-({r}-{edge:.12g}-{phase:.12g})/({edge:.12g}), 0))))"
    )


def phase_clock_source_line(
    name: str,
    node: str,
    pulses: list[tuple[float, float]],
    t_stop: float,
    phase: float,
    gap: float,
    edge: float,
    phase_clock_mode: str,
    direct_update: bool,
    update_batch_size: int,
) -> str:
    if phase_clock_mode == "pwl":
        return f"V{name} {node} 0 {phase_pwl(pulses, t_stop, edge)}"
    if phase_clock_mode == "analytic":
        if not direct_update or update_batch_size != 1:
            raise ValueError("analytic phase clocks require direct update mode with batch_size=1")
        return f"B{name} {node} 0 V = {analytic_phase_clock_expr(pulses, phase, gap, edge)}"
    raise ValueError("phase_clock_mode must be 'pwl' or 'analytic'")


def phase_pulse_area(phase: float, edge: float) -> float:
    if phase <= 0.0:
        raise ValueError("phase must be positive")
    if edge < 0.0:
        raise ValueError("edge must be non-negative")
    return phase + edge


def estimate_transient_points(t_stop: float, transient_step: float) -> int:
    if transient_step <= 0.0:
        raise ValueError("transient_step must be positive")
    if t_stop < 0.0:
        raise ValueError("t_stop must be non-negative")
    return int(math.ceil(t_stop / transient_step)) + 1


def phase_output_vector_count(
    w: np.ndarray,
    hb: np.ndarray,
    readout: np.ndarray,
    output_bias: np.ndarray,
    *,
    include_output_bias_vectors: bool = True,
    include_y_vectors: bool = True,
) -> int:
    count = int(w.size + hb.size + readout.size)
    if include_output_bias_vectors:
        count += int(output_bias.size)
    if include_y_vectors:
        count += int(readout.shape[0])
    return count


def hidden_preactivation_source_count(mode: str, blocks: int, channels: int) -> int:
    if mode == "node":
        return int(blocks) * int(channels)
    if mode == "inline":
        return 0
    raise ValueError("hidden_preactivation_mode must be 'node' or 'inline'")


def hidden_activation_state_count(mode: str, blocks: int, channels: int) -> int:
    if mode == "stored":
        return int(blocks) * int(channels)
    if mode == "inline":
        return 0
    raise ValueError("hidden_activation_mode must be 'stored' or 'inline'")


def hidden_delta_state_count(blocks: int, channels: int) -> int:
    return int(blocks) * int(channels)


def score_state_count(classes: int, mode: str = "stored") -> int:
    if mode == "stored":
        return int(classes)
    if mode == "inline":
        return 0
    raise ValueError("score_state_mode must be 'stored' or 'inline'")


def score_calculation_source_count(mode: str, classes: int) -> int:
    if mode == "node":
        return int(classes)
    if mode == "inline":
        return 0
    raise ValueError("score_calculation_mode must be 'node' or 'inline'")


def output_rail_source_count(mode: str, classes: int) -> int:
    if mode == "node":
        return int(classes)
    if mode == "inline":
        return 0
    raise ValueError("output_rail_mode must be 'node' or 'inline'")


def output_delta_state_count(mode: str, classes: int) -> int:
    if mode == "node":
        return int(classes)
    if mode == "inline":
        return 0
    raise ValueError("output_delta_mode must be 'node' or 'inline'")


def gradient_accumulator_state_count(
    update_mode: str,
    blocks: list[list[int]],
    channels: int,
    classes: int,
    x_batch: np.ndarray | None = None,
    *,
    local_updates_enabled: bool = True,
    readout_updates_enabled: bool = True,
    output_bias_updates_enabled: bool = True,
) -> int:
    if update_mode == "direct":
        return 0
    if update_mode != "phased":
        raise ValueError("update_mode must be 'phased' or 'direct'")
    count = 0
    n_blocks = len(blocks)
    channels = int(channels)
    classes = int(classes)
    if local_updates_enabled:
        count += n_blocks * channels
        if x_batch is None:
            active_block_pixels = sum(len(idxs) for idxs in blocks)
        else:
            x_arr = np.asarray(x_batch)
            active_block_pixels = sum(
                1
                for idxs in blocks
                for idx in idxs
                if np.any(x_arr[:, idx] != 0.0)
            )
        count += active_block_pixels * channels
    if readout_updates_enabled:
        count += classes * n_blocks * channels
    if output_bias_updates_enabled:
        count += classes
    return count


def temporary_state_count(
    *,
    hidden_activation_states: int,
    hidden_delta_states: int,
    score_states: int,
    output_delta_states: int,
    gradient_accumulator_states: int,
) -> int:
    counts = [
        hidden_activation_states,
        hidden_delta_states,
        score_states,
        output_delta_states,
        gradient_accumulator_states,
    ]
    if any(int(count) < 0 for count in counts):
        raise ValueError("temporary state counts must be non-negative")
    return sum(int(count) for count in counts)


def auxiliary_algebraic_source_count(
    hidden_preactivation_sources: int,
    score_calculation_sources: int,
    output_rail_sources: int,
) -> int:
    return int(hidden_preactivation_sources) + int(score_calculation_sources) + int(output_rail_sources)


def validate_transient_point_budget(estimated_points: int, max_points: int) -> None:
    if max_points < 0:
        raise ValueError("--max-transient-points must be non-negative")
    if max_points and estimated_points > max_points:
        raise ValueError(
            f"estimated transient points ({estimated_points}) exceed --max-transient-points ({max_points}); "
            "reduce --updates, increase --transient-step, or raise the budget intentionally"
        )


def validate_source_point_budget(source_complexity: dict[str, int], max_points: int) -> None:
    if max_points < 0:
        raise ValueError("--max-source-pwl-points must be non-negative")
    estimated_points = total_source_pwl_points(source_complexity)
    if max_points and estimated_points > max_points:
        raise ValueError(
            f"estimated source PWL points ({estimated_points}) exceed --max-source-pwl-points ({max_points}); "
            "reduce --updates, choose a smaller training prefix, or raise the budget intentionally"
        )


def validate_sample_source_budget(source_complexity: dict[str, int], max_sources: int) -> None:
    if max_sources < 0:
        raise ValueError("--max-sample-sources must be non-negative")
    estimated_sources = int(source_complexity.get("sample_source_count", 0))
    if max_sources and estimated_sources > max_sources:
        raise ValueError(
            f"estimated sample sources ({estimated_sources}) exceed --max-sample-sources ({max_sources}); "
            "use compact/elided source modes, reduce input dimensionality, or raise the budget intentionally"
        )


def validate_total_source_budget(source_complexity: dict[str, int], max_sources: int) -> None:
    if max_sources < 0:
        raise ValueError("--max-total-sources must be non-negative")
    estimated_sources = int(source_complexity.get("total_source_count", 0))
    if max_sources and estimated_sources > max_sources:
        raise ValueError(
            f"estimated total source elements ({estimated_sources}) exceed --max-total-sources ({max_sources}); "
            "use compact/elided source modes, reduce clocks/controls, or raise the budget intentionally"
        )


def validate_output_vector_budget(vector_count: int, max_vectors: int) -> None:
    if max_vectors < 0:
        raise ValueError("--max-output-vectors must be non-negative")
    if max_vectors and vector_count > max_vectors:
        raise ValueError(
            f"estimated output vectors ({vector_count}) exceed --max-output-vectors ({max_vectors}); "
            "omit optional diagnostic rails or raise the budget intentionally"
        )


def validate_auxiliary_algebraic_source_budget(source_count: int, max_sources: int) -> None:
    if max_sources < 0:
        raise ValueError("--max-auxiliary-algebraic-sources must be non-negative")
    if max_sources and source_count > max_sources:
        raise ValueError(
            f"estimated auxiliary algebraic sources ({source_count}) exceed --max-auxiliary-algebraic-sources ({max_sources}); "
            "use inline fused deck modes or raise the budget intentionally"
        )


def total_source_pwl_points(source_complexity: dict[str, int]) -> int:
    return (
        int(source_complexity["sample_source_pwl_points"])
        + int(source_complexity["phase_clock_source_pwl_points"])
        + int(source_complexity.get("control_source_pwl_points", 0))
    )


def total_source_count(source_complexity: dict[str, int]) -> int:
    return (
        int(source_complexity["sample_source_count"])
        + int(source_complexity.get("target_behavioral_source_count", 0))
        + int(source_complexity["phase_clock_source_count"])
        + int(source_complexity.get("control_source_count", 0))
    )


def final_state_measure_time(t_stop: float, transient_step: float, final_update_stop: float) -> float:
    if transient_step <= 0.0:
        raise ValueError("transient_step must be positive")
    measure_time = max(0.0, t_stop - transient_step)
    if measure_time <= final_update_stop:
        slack = max(t_stop - final_update_stop, 0.0)
        raise ValueError(
            "--transient-step places the final-state measurement before the final update has settled; "
            f"use a step smaller than the final measurement slack ({slack:.12g}s)"
        )
    return measure_time


def phase_state_descriptions(
    update_mode: str,
    output_bias_state_frozen: bool = False,
    output_delta_mode: str = "node",
    hidden_activation_mode: str = "stored",
    score_state_mode: str = "stored",
) -> dict[str, str]:
    if output_delta_mode not in {"node", "inline"}:
        raise ValueError("output_delta_mode must be 'node' or 'inline'")
    if hidden_activation_mode not in {"stored", "inline"}:
        raise ValueError("hidden_activation_mode must be 'stored' or 'inline'")
    if score_state_mode not in {"stored", "inline"}:
        raise ValueError("score_state_mode must be 'stored' or 'inline'")
    stored_feature_prefix = (
        "feature activations, "
        if hidden_activation_mode == "stored"
        else ""
    )
    stored_score_prefix = "class scores, " if score_state_mode == "stored" else ""
    inline_feature_note = (
        ""
        if hidden_activation_mode == "stored"
        else "; hidden activations are inline expressions"
    )
    inline_score_note = (
        ""
        if score_state_mode == "stored"
        else "; class scores are inline expressions"
    )
    inline_notes = inline_feature_note + inline_score_note
    if update_mode == "phased":
        if output_delta_mode == "node":
            temporary = (
                f"{stored_feature_prefix}{stored_score_prefix}class deltas, hidden/backward feature deltas, "
                f"and gradient accumulators are capacitor voltages{inline_notes}"
            )
        else:
            temporary = (
                f"{stored_feature_prefix}{stored_score_prefix}hidden/backward feature deltas, and gradient accumulators "
                f"are capacitor voltages; class deltas are inline expressions{inline_notes}"
            )
    elif update_mode == "direct":
        if output_delta_mode == "node":
            temporary = (
                f"{stored_feature_prefix}{stored_score_prefix}class deltas, and hidden/backward feature deltas "
                f"are capacitor voltages{inline_notes}; weights are updated directly during each per-sample update phase"
            )
        else:
            temporary = (
                f"{stored_feature_prefix}{stored_score_prefix}hidden/backward feature deltas are capacitor voltages; "
                f"class deltas are inline expressions{inline_notes}; "
                "weights are updated directly during each per-sample update phase"
            )
    else:
        raise ValueError("update_mode must be 'phased' or 'direct'")
    if output_bias_state_frozen:
        persistent = (
            "local feature weights, local biases, and class readout weights are persistent capacitor "
            "voltages initialized once at the start of the transient; output biases are frozen constants"
        )
    else:
        persistent = (
            "local feature weights, local biases, class readout weights, and output biases are "
            "persistent capacitor voltages initialized once at the start of the transient"
        )
    return {
        "persistent_state": persistent,
        "temporary_state": temporary,
    }


def phase_deck_mode_fields(
    *,
    phase_clock_mode: str,
    target_source_mode: str,
    sample_edge: float,
    hidden_preactivation_mode: str,
    hidden_preactivation_source_count: int,
    hidden_activation_mode: str = "stored",
    hidden_activation_state_count: int | None = None,
    hidden_delta_state_count: int | None = None,
    score_state_mode: str = "stored",
    score_state_count: int | None = None,
    gradient_accumulator_state_count: int | None = None,
    temporary_state_count: int | None = None,
    score_calculation_mode: str,
    score_calculation_source_count: int,
    output_rail_mode: str,
    output_rail_source_count: int,
    output_delta_mode: str,
    output_delta_state_count: int,
) -> dict[str, object]:
    auxiliary_sources = auxiliary_algebraic_source_count(
        hidden_preactivation_source_count,
        score_calculation_source_count,
        output_rail_source_count,
    )
    return {
        "phase_clock_mode": phase_clock_mode,
        "target_source_mode": target_source_mode,
        "sample_edge_s": sample_edge,
        "hidden_preactivation_mode": hidden_preactivation_mode,
        "hidden_preactivation_source_count": hidden_preactivation_source_count,
        "hidden_activation_mode": hidden_activation_mode,
        "hidden_activation_state_count": (
            hidden_activation_state_count
            if hidden_activation_state_count is not None
            else None
        ),
        "hidden_delta_state_count": hidden_delta_state_count,
        "score_state_mode": score_state_mode,
        "score_state_count": score_state_count,
        "gradient_accumulator_state_count": gradient_accumulator_state_count,
        "temporary_state_count": temporary_state_count,
        "score_calculation_mode": score_calculation_mode,
        "score_calculation_source_count": score_calculation_source_count,
        "output_rail_mode": output_rail_mode,
        "output_rail_source_count": output_rail_source_count,
        "output_delta_mode": output_delta_mode,
        "output_delta_state_count": output_delta_state_count,
        "auxiliary_algebraic_source_count": auxiliary_sources,
    }


def strict_fully_on_device_contract_met(
    batch_size: int,
    reference_mode: str,
    init_weights: str,
) -> bool:
    return batch_size == 1 and reference_mode == "none" and not init_weights


def validate_strict_fully_on_device_args(
    batch_size: int,
    reference_mode: str,
    init_weights: str,
) -> None:
    if batch_size != 1:
        raise ValueError("--strict-fully-on-device requires --batch-size 1")
    if reference_mode != "none":
        raise ValueError("--strict-fully-on-device requires --reference-mode none")
    if init_weights:
        raise ValueError("--strict-fully-on-device requires random init; do not pass --init-weights")


def phase_execution_contract_fields(
    batch_size: int,
    reference_mode: str,
    init_weights: str = "",
    strict_fully_on_device: bool = False,
) -> dict[str, bool | str]:
    if reference_mode not in {"spice", "none"}:
        raise ValueError("reference_mode must be 'spice' or 'none'")
    strict_contract_met = strict_fully_on_device_contract_met(batch_size, reference_mode, init_weights)
    return {
        "single_phase_training_transient": True,
        "weights_persist_inside_phase_transient": True,
        "python_weight_updates_between_samples": False,
        "python_checkpointing_between_samples": False,
        "reference_replay_used_for_diagnostics": reference_mode == "spice",
        "fully_on_device_execution_contract_met": batch_size == 1,
        "strict_fully_on_device_requested": strict_fully_on_device,
        "strict_fully_on_device_contract_met": strict_contract_met,
        "random_init_used": not bool(init_weights),
        "initial_weights_source": "random_init" if not init_weights else "checkpoint",
        "execution_contract_note": (
            "This gate covers the executed phase-transient training path only: one persistent-state "
            "training transient, batch_size=1 online updates, and no Python weight writes between "
            "samples. Reference/eval runs are diagnostics after the training transient."
        ),
    }


def phase_preflight_summary(
    *,
    simulator_selector: str | None,
    image_size: int,
    block_size: int,
    stride: int,
    blocks: int,
    channels: int,
    train_samples: int,
    eval_samples: int,
    batch_size: int,
    updates: int,
    total_samples: int,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    labels: np.ndarray,
    lr: float,
    lr_schedule: str,
    lr_final_scale: float,
    update_mode: str,
    phase_clock_mode: str,
    target_source_mode: str,
    hidden_preactivation_mode: str,
    hidden_preactivation_source_count: int,
    hidden_activation_mode: str,
    hidden_activation_state_count: int,
    hidden_delta_state_count: int,
    score_state_mode: str,
    score_state_count: int,
    gradient_accumulator_state_count: int,
    temporary_state_count: int,
    score_calculation_mode: str,
    score_calculation_source_count: int,
    output_rail_mode: str,
    output_rail_source_count: int,
    output_delta_mode: str,
    output_delta_state_count: int,
    output_bias_state_frozen: bool,
    phase_output_vector_count: int,
    phase_output_includes_y: bool,
    reference_mode: str,
    init_weights: str,
    strict_fully_on_device: bool,
    estimated_transient_points: int,
    max_transient_points: int,
    max_source_pwl_points: int,
    max_sample_sources: int,
    max_total_sources: int,
    max_output_vectors: int,
    max_auxiliary_algebraic_sources: int,
    t_stop: float,
    transient_step: float,
    phase: float,
    sample_edge: float,
    settle_ratio: float,
    source_complexity: dict[str, int],
) -> dict[str, object]:
    cost_fields = phase_cost_summary_fields(
        updates=updates,
        estimated_transient_points=estimated_transient_points,
        phase_output_vector_count=phase_output_vector_count,
        auxiliary_algebraic_source_count=auxiliary_algebraic_source_count(
            hidden_preactivation_source_count,
            score_calculation_source_count,
            output_rail_source_count,
        ),
        source_complexity=source_complexity,
        max_transient_points=max_transient_points,
        max_source_pwl_points=max_source_pwl_points,
        max_sample_sources=max_sample_sources,
        max_total_sources=max_total_sources,
        max_output_vectors=max_output_vectors,
        max_auxiliary_algebraic_sources=max_auxiliary_algebraic_sources,
    )
    return {
        "simulator_selector": simulator_selector,
        "architecture": "phase_resolved_transient_local_feature_readout",
        "status": "phase_preflight_only",
        "preflight_only": True,
        "would_launch_simulator": False,
        "image_size": image_size,
        "block_size": block_size,
        "stride": stride,
        "blocks": blocks,
        "channels": channels,
        "classes": 10,
        "mnist_index_order": "stable_permutation_prefix",
        "train_index_metadata": index_prefix_metadata(train_indices),
        "eval_index_metadata": index_prefix_metadata(eval_indices),
        "train_label_metadata": label_sequence_metadata(labels, 10),
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "batch_size": batch_size,
        "updates": updates,
        "total_samples": total_samples,
        "lr": lr,
        "lr_schedule": lr_schedule,
        "lr_final_scale": lr_final_scale,
        "update_mode": update_mode,
        **phase_deck_mode_fields(
            phase_clock_mode=phase_clock_mode,
            target_source_mode=target_source_mode,
            sample_edge=sample_edge,
            hidden_preactivation_mode=hidden_preactivation_mode,
            hidden_preactivation_source_count=hidden_preactivation_source_count,
            hidden_activation_mode=hidden_activation_mode,
            hidden_activation_state_count=hidden_activation_state_count,
            hidden_delta_state_count=hidden_delta_state_count,
            score_state_mode=score_state_mode,
            score_state_count=score_state_count,
            gradient_accumulator_state_count=gradient_accumulator_state_count,
            temporary_state_count=temporary_state_count,
            score_calculation_mode=score_calculation_mode,
            score_calculation_source_count=score_calculation_source_count,
            output_rail_mode=output_rail_mode,
            output_rail_source_count=output_rail_source_count,
            output_delta_mode=output_delta_mode,
            output_delta_state_count=output_delta_state_count,
        ),
        "output_bias_state_frozen": output_bias_state_frozen,
        "phase_output_vector_count": phase_output_vector_count,
        "phase_output_includes_y": phase_output_includes_y,
        "reference_mode": reference_mode,
        "init_weights": init_weights,
        "phase_netlist": None,
        "phase_data": None,
        "final_weights": None,
        "equivalence_metrics": None,
        **phase_execution_contract_fields(batch_size, reference_mode, init_weights, strict_fully_on_device),
        "continuous_transient_contract_met": None,
        "t_stop_s": t_stop,
        "transient_step_s": transient_step,
        "estimated_transient_points": estimated_transient_points,
        "max_transient_points": max_transient_points,
        "max_source_pwl_points": max_source_pwl_points,
        "max_sample_sources": max_sample_sources,
        "max_total_sources": max_total_sources,
        "max_output_vectors": max_output_vectors,
        "max_auxiliary_algebraic_sources": max_auxiliary_algebraic_sources,
        **source_complexity,
        **cost_fields,
        "phase_s": phase,
        "settle_ratio": settle_ratio,
        **phase_state_descriptions(
            update_mode,
            output_bias_state_frozen,
            output_delta_mode,
            hidden_activation_mode,
            score_state_mode,
        ),
    }


def phase_cost_summary_fields(
    *,
    updates: int,
    estimated_transient_points: int,
    phase_output_vector_count: int,
    auxiliary_algebraic_source_count: int,
    source_complexity: dict[str, int],
    max_transient_points: int,
    max_source_pwl_points: int,
    max_sample_sources: int,
    max_total_sources: int,
    max_output_vectors: int,
    max_auxiliary_algebraic_sources: int,
) -> dict[str, object]:
    if updates <= 0:
        raise ValueError("updates must be positive")
    total_source_points = total_source_pwl_points(source_complexity)
    sample_source_count = int(source_complexity.get("sample_source_count", 0))
    total_sources = int(source_complexity.get("total_source_count", 0))
    return {
        "estimated_transient_points_per_update": float(estimated_transient_points) / float(updates),
        "sample_source_pwl_points_per_update": float(source_complexity["sample_source_pwl_points"]) / float(updates),
        "phase_clock_source_pwl_points_per_update": float(source_complexity["phase_clock_source_pwl_points"]) / float(updates),
        "control_source_pwl_points_per_update": float(source_complexity.get("control_source_pwl_points", 0)) / float(updates),
        "total_source_pwl_points_per_update": float(total_source_points) / float(updates),
        "transient_budget_met": bool(not max_transient_points or estimated_transient_points <= max_transient_points),
        "source_pwl_budget_met": bool(not max_source_pwl_points or total_source_points <= max_source_pwl_points),
        "sample_source_budget_met": bool(not max_sample_sources or sample_source_count <= max_sample_sources),
        "total_source_budget_met": bool(not max_total_sources or total_sources <= max_total_sources),
        "output_vector_budget_met": bool(not max_output_vectors or phase_output_vector_count <= max_output_vectors),
        "auxiliary_algebraic_source_budget_met": bool(
            not max_auxiliary_algebraic_sources
            or auxiliary_algebraic_source_count <= max_auxiliary_algebraic_sources
        ),
    }


def index_prefix_metadata(indices: np.ndarray, prefix_len: int = 16) -> dict[str, object]:
    idx = np.asarray(indices, dtype=np.int64)
    return {
        "count": int(idx.size),
        "sha256": hashlib.sha256(idx.tobytes()).hexdigest(),
        "prefix": [int(value) for value in idx[:prefix_len]],
    }


def label_sequence_metadata(labels: np.ndarray, n_classes: int, prefix_len: int = 32) -> dict[str, object]:
    y = np.asarray(labels, dtype=np.int64)
    if n_classes <= 0:
        raise ValueError("n_classes must be positive")
    hist = np.bincount(y, minlength=n_classes)
    total = max(int(y.size), 1)
    dominant_count = int(np.max(hist)) if hist.size else 0
    dominant_label = int(np.argmax(hist)) if hist.size else None
    return {
        "count": int(y.size),
        "sha256": hashlib.sha256(y.tobytes()).hexdigest(),
        "prefix": [int(value) for value in y[:prefix_len]],
        "histogram": [int(value) for value in hist],
        "dominant_label": dominant_label,
        "dominant_label_fraction": float(dominant_count / total),
        "unique_labels": int(np.count_nonzero(hist)),
    }


def sample_source_pwl(values: np.ndarray, sample_starts: list[float], t_stop: float, edge: float) -> str:
    if len(values) == 0:
        raise ValueError("values must not be empty")
    if len(values) != len(sample_starts):
        raise ValueError("values length must match sample_starts length")
    if edge < 0.0:
        raise ValueError("edge must be non-negative")
    if np.all(values == values[0]):
        return f"{float(values[0]):.12g}"
    points: list[tuple[float, float]] = [(0.0, float(values[0]))]
    for s, val in enumerate(values):
        current = float(val)
        prev = float(values[s - 1] if s > 0 else val)
        if current == prev:
            continue
        t = sample_starts[s]
        points.append((max(0.0, t - edge), prev))
        points.append((t, current))
    cleaned: list[tuple[float, float]] = []
    for t, val in points:
        if cleaned and abs(cleaned[-1][0] - t) < 1e-18:
            cleaned[-1] = (t, val)
        else:
            cleaned.append((t, val))
    return "PWL(" + " ".join(f"{t:.12g} {val:.12g}" for t, val in cleaned) + ")"


def dc_source_value(source: str) -> float | None:
    if source.startswith("PWL("):
        return None
    try:
        value = float(source)
    except ValueError:
        return None
    return 0.0 if value == 0.0 else value


def spice_literal(value: float) -> str:
    return "0" if value == 0.0 else f"{value:.12g}"


def sample_drive(
    source_name: str,
    node: str,
    values: np.ndarray,
    sample_starts: list[float],
    t_stop: float,
    edge: float,
    *,
    elide_dc: bool = False,
) -> SampleDrive:
    source = sample_source_pwl(values, sample_starts, t_stop, edge)
    constant = dc_source_value(source)
    if elide_dc and constant is not None:
        return SampleDrive(node=node, source=source, expr=spice_literal(constant), line=None, constant_value=constant)
    return SampleDrive(
        node=node,
        source=source,
        expr=f"V({node})",
        line=f"{source_name} {node} 0 {source}",
        constant_value=constant,
    )


def emitted_sources(drives: list[SampleDrive]) -> list[str]:
    return [drive.source for drive in drives if not drive.elided]


def elided_dc_count(drives: list[SampleDrive]) -> int:
    return sum(1 for drive in drives if drive.elided and drive.constant_value is not None)


def lr_schedule_values(base_lr: float, updates: int, schedule: str = "constant", final_scale: float = 1.0) -> np.ndarray:
    if base_lr < 0.0:
        raise ValueError("base_lr must be non-negative")
    if updates <= 0:
        raise ValueError("updates must be positive")
    if final_scale < 0.0:
        raise ValueError("lr_final_scale must be non-negative")
    if schedule == "constant":
        return np.full(updates, float(base_lr))
    if schedule == "linear-decay":
        if updates == 1:
            scales = np.array([1.0], dtype=float)
        else:
            scales = np.linspace(1.0, final_scale, updates)
        return float(base_lr) * scales
    raise ValueError("lr_schedule must be 'constant' or 'linear-decay'")


def pwl_point_count(source: str) -> int:
    if not source.startswith("PWL(") or not source.endswith(")"):
        return 0
    tokens = source[4:-1].split()
    if len(tokens) % 2:
        raise ValueError("PWL source has an odd number of time/value tokens")
    return len(tokens) // 2


def phase_source_complexity(
    x_batch: np.ndarray,
    targets: np.ndarray,
    phases: dict[str, list[tuple[float, float]]],
    sample_starts: list[float],
    t_stop: float,
    edge: float,
    direct_update: bool,
    phase_clock_mode: str = "pwl",
    lr_values: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    target_source_mode: str = "rails",
    sample_edge: float | None = None,
) -> dict[str, int]:
    if target_source_mode not in {"rails", "label"}:
        raise ValueError("target_source_mode must be 'rails' or 'label'")
    sample_transition_edge = edge if sample_edge is None else sample_edge
    pixel_drives = [
        sample_drive(f"Vpix{i}", f"pix{i}", x_batch[:, i], sample_starts, t_stop, sample_transition_edge, elide_dc=True)
        for i in range(x_batch.shape[1])
    ]
    pixel_sources = emitted_sources(pixel_drives)
    target_behavioral_source_count = 0
    if target_source_mode == "rails":
        target_sources = [sample_source_pwl(targets[:, k], sample_starts, t_stop, sample_transition_edge) for k in range(targets.shape[1])]
    else:
        label_values = np.asarray(labels if labels is not None else np.argmax(targets, axis=1), dtype=float)
        target_sources = [sample_source_pwl(label_values, sample_starts, t_stop, sample_transition_edge)]
        target_behavioral_source_count = int(targets.shape[1])
    phase_names = ["act", "score", "err", "bwd", "acc"] if direct_update else ["act", "score", "err", "bwd", "acc", "apply", "clear"]
    if phase_clock_mode == "pwl":
        phase_sources = [phase_pwl(phases[name], t_stop, edge) for name in phase_names]
    elif phase_clock_mode == "analytic":
        phase_sources = ["0" for _name in phase_names]
    else:
        raise ValueError("phase_clock_mode must be 'pwl' or 'analytic'")
    control_sources = []
    if lr_values is not None:
        control_sources.append(sample_source_pwl(np.asarray(lr_values, dtype=float), sample_starts, t_stop, sample_transition_edge))

    def count_dc(sources: list[str]) -> int:
        return sum(not source.startswith("PWL(") for source in sources)

    def count_pwl(sources: list[str]) -> int:
        return sum(source.startswith("PWL(") for source in sources)

    def count_points(sources: list[str]) -> int:
        return sum(pwl_point_count(source) for source in sources)

    sample_sources = pixel_sources + target_sources
    sample_elided_dc_count = elided_dc_count(pixel_drives)
    complexity = {
        "sample_source_count": len(sample_sources),
        "sample_source_dc_count": count_dc(sample_sources),
        "sample_source_pwl_count": count_pwl(sample_sources),
        "sample_source_pwl_points": count_points(sample_sources),
        "sample_source_elided_dc_count": sample_elided_dc_count,
        "pixel_source_count": len(pixel_sources),
        "pixel_source_dc_count": count_dc(pixel_sources),
        "pixel_source_pwl_count": count_pwl(pixel_sources),
        "pixel_source_pwl_points": count_points(pixel_sources),
        "pixel_source_elided_dc_count": sample_elided_dc_count,
        "target_source_count": len(target_sources),
        "target_source_dc_count": count_dc(target_sources),
        "target_source_pwl_count": count_pwl(target_sources),
        "target_source_pwl_points": count_points(target_sources),
        "target_source_elided_dc_count": 0,
        "target_behavioral_source_count": target_behavioral_source_count,
        "target_source_mode_label": int(target_source_mode == "label"),
        "phase_clock_source_count": len(phase_sources),
        "phase_clock_source_pwl_count": count_pwl(phase_sources),
        "phase_clock_source_pwl_points": count_points(phase_sources),
        "control_source_count": len(control_sources),
        "control_source_dc_count": count_dc(control_sources),
        "control_source_pwl_count": count_pwl(control_sources),
        "control_source_pwl_points": count_points(control_sources),
    }
    complexity["total_source_count"] = total_source_count(complexity)
    complexity["total_source_pwl_points"] = total_source_pwl_points(complexity)
    return complexity


def tanh_expr(expr: str) -> str:
    return f"(2/(1+exp(-2*({expr})))-1)"


def local_activation_expr(
    expr: str,
    local_activation: str,
    relu_clip: float,
    relu_leak: float = 0.01,
    softplus_beta: float = 10.0,
) -> str:
    return block_local_activation_expr(expr, local_activation, relu_clip, relu_leak, softplus_beta)


def local_activation_deriv_expr(
    preactivation_expr: str,
    activation_ref: str,
    local_activation: str,
    relu_clip: float,
    activation_derivative: str = "exact",
    derivative_floor: float = 0.0,
    derivative_gate_threshold: float = 1e-6,
    relu_leak: float = 0.01,
    softplus_beta: float = 10.0,
) -> str:
    activation_expr = (
        activation_ref
        if activation_ref.startswith(("V(", "("))
        else f"V({activation_ref})"
    )
    return block_local_activation_deriv_expr(
        preactivation_expr,
        activation_expr,
        local_activation,
        relu_clip,
        activation_derivative,
        derivative_floor,
        derivative_gate_threshold,
        relu_leak,
        softplus_beta,
    )


def target_matrix(labels: np.ndarray, n_classes: int, softmax_output: bool = False) -> np.ndarray:
    targets = np.zeros((len(labels), n_classes)) if softmax_output else -np.ones((len(labels), n_classes))
    for s, label in enumerate(labels):
        targets[s, int(label)] = 1.0
    return targets


def target_from_label_expr(class_index: int, softmax_output: bool) -> str:
    active = f"(0.5*(1+tanh((0.5-abs(V(label)-{class_index}))/{{TARGET_LABEL_SMOOTH}})))"
    return active if softmax_output else f"(2*({active})-1)"


def target_source_lines(
    labels: np.ndarray,
    targets: np.ndarray,
    sample_starts: list[float],
    t_stop: float,
    edge: float,
    *,
    target_source_mode: str,
    softmax_output: bool,
    sample_edge: float | None = None,
) -> list[str]:
    sample_transition_edge = edge if sample_edge is None else sample_edge
    if target_source_mode == "rails":
        return [
            f"Vtarget{k} target{k} 0 {sample_source_pwl(targets[:, k], sample_starts, t_stop, sample_transition_edge)}"
            for k in range(targets.shape[1])
        ]
    if target_source_mode == "label":
        lines = [
            f"Vlabel label 0 {sample_source_pwl(np.asarray(labels, dtype=float), sample_starts, t_stop, sample_transition_edge)}"
        ]
        lines.extend(
            f"Btarget{k} target{k} 0 V = {target_from_label_expr(k, softmax_output)}" for k in range(targets.shape[1])
        )
        return lines
    raise ValueError("target_source_mode must be 'rails' or 'label'")


def apply_readout_class_centering_np(readout: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return readout
    if mode == "mean":
        return readout - np.mean(readout, axis=0, keepdims=True)
    raise ValueError("readout_class_centering must be 'none' or 'mean'")


def parse_measured_vector(stdout: str, n_vec: int) -> np.ndarray:
    return parse_named_measured_vector(stdout, "m", n_vec)


def parse_named_measured_vector(stdout: str, prefix: str, n_vec: int) -> np.ndarray:
    vals = np.empty(n_vec, dtype=float)
    for i in range(n_vec):
        name = f"{prefix}{i:05d}"
        m = re.search(rf"(?im)^\s*{name}\s*=\s*([-+0-9.eE]+)", stdout)
        if not m:
            raise ValueError(f"missing final-state measurement {name}")
        vals[i] = float(m.group(1))
    return vals


def parse_probe_update_list(raw: str, updates: int) -> tuple[int, ...]:
    if not raw:
        return ()
    selected: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if item in {"final", "last"}:
            selected.add(updates)
            continue
        if item == "powers2":
            value = 1
            while value <= updates:
                selected.add(value)
                value *= 2
            selected.add(updates)
            continue
        if "-" in item:
            bounds = item.split("-", 1)
            if len(bounds) != 2 or not bounds[0] or not bounds[1]:
                raise ValueError(f"invalid probe update range {item!r}")
            start, stop = (int(bounds[0]), int(bounds[1]))
            if start > stop:
                raise ValueError(f"invalid descending probe update range {item!r}")
            selected.update(range(start, stop + 1))
            continue
        selected.add(int(item))
    bad = sorted(value for value in selected if value < 1 or value > updates)
    if bad:
        raise ValueError(f"probe updates must be in [1, {updates}], got {bad}")
    return tuple(sorted(selected))


def parse_probe_measurements(stdout: str, probe_updates: tuple[int, ...], n_vec: int) -> dict[int, np.ndarray]:
    return {
        update: parse_named_measured_vector(stdout, f"p{probe_idx:03d}_", n_vec)
        for probe_idx, update in enumerate(probe_updates)
    }


def read_xyce_print_last_row(path: Path, expected_values: int) -> np.ndarray:
    rows = read_xyce_print_rows(path, expected_values)
    if not rows:
        raise ValueError(f"no Xyce print data rows found in {path}")
    return rows[-1][1]


def read_xyce_print_rows(path: Path, expected_values: int) -> list[tuple[float, np.ndarray]]:
    rows: list[tuple[float, np.ndarray]] = []
    for raw in path.read_text().splitlines():
        parts = raw.split()
        if not parts or parts[0].lower() in {"index", "end"}:
            continue
        try:
            int(float(parts[0]))
            values = np.array([float(item) for item in parts[1:]], dtype=float)
        except ValueError:
            continue
        if values.size < expected_values:
            raise ValueError(f"Xyce print row in {path} had {values.size} values, expected {expected_values}")
        rows.append((float(values[0]), values[-expected_values:]))
    if not rows:
        raise ValueError(f"no Xyce print data rows found in {path}")
    return rows


def xyce_print_vectors_at_times(
    rows: list[tuple[float, np.ndarray]],
    requested_times: dict[int, float],
) -> dict[int, np.ndarray]:
    if not requested_times:
        return {}
    out: dict[int, np.ndarray] = {}
    for update, target_time in requested_times.items():
        row_time, values = min(rows, key=lambda row: abs(row[0] - target_time))
        tolerance = max(1e-15, abs(target_time) * 1e-6)
        if abs(row_time - target_time) > tolerance:
            raise ValueError(
                f"no Xyce print row near probe update {update} time {target_time:.12g}; closest was {row_time:.12g}"
            )
        out[update] = values
    return out


def prepare_phase_netlist(netlist: str, spice_bin: str) -> str:
    rendered = prepare_netlist_for_simulator(netlist, spice_bin)
    return wrap_xyce_behavioral_rhs(rendered) if is_xyce(spice_bin) else rendered


def make_phase_schedule(
    update_batch_size: int,
    updates: int,
    phase: float,
    gap: float,
    direct_update: bool = False,
) -> tuple[dict[str, list[tuple[float, float]]], list[float], float]:
    phases = {"act": [], "score": [], "err": [], "bwd": [], "acc": [], "apply": [], "clear": []}
    sample_starts: list[float] = []
    t = phase
    for _update in range(updates):
        for _sample in range(update_batch_size):
            sample_starts.append(t)
            t += gap
            phases["act"].append((t, t + phase))
            t += phase + gap
            phases["score"].append((t, t + phase))
            t += phase + gap
            phases["err"].append((t, t + phase))
            t += phase + gap
            phases["bwd"].append((t, t + phase))
            t += phase + gap
            phases["acc"].append((t, t + phase))
            t += phase + gap
        if not direct_update:
            phases["apply"].append((t, t + phase))
            t += phase + gap
            phases["clear"].append((t, t + phase))
            t += phase + gap
    return phases, sample_starts, t + phase


def probe_measure_times(
    clear_phases: list[tuple[float, float]],
    probe_updates: tuple[int, ...],
    gap: float,
    t_stop: float,
    final_measure_time: float,
) -> dict[int, float]:
    if not probe_updates:
        return {}
    if gap <= 0.0:
        raise ValueError("probe measurements require --gap > 0")
    times: dict[int, float] = {}
    for update in probe_updates:
        _clear_start, clear_stop = clear_phases[update - 1]
        times[update] = min(final_measure_time, clear_stop + 0.5 * gap, t_stop)
    return times


def make_phase_transient_netlist(
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    w: np.ndarray,
    hb: np.ndarray,
    readout: np.ndarray,
    output_bias: np.ndarray,
    blocks: list[list[int]],
    lr: float,
    out_path: Path,
    linear_output: bool,
    update_batch_size: int,
    updates: int,
    phase: float,
    gap: float,
    edge: float,
    settle_ratio: float,
    transient_step: float,
    cw: float,
    cstate: float,
    cgrad: float,
    rleak: float,
    softmax_output: bool = False,
    output_mode: str = "wrdata",
    probe_updates: tuple[int, ...] = (),
    local_activation: str = "tanh",
    relu_clip: float = 1.0,
    activation_derivative: str = "exact",
    derivative_floor: float = 0.0,
    derivative_gate_threshold: float = 1e-6,
    readout_feedback_mode: str = "readout",
    readout_feedback_clip: float = 0.05,
    relu_leak: float = 0.01,
    softplus_beta: float = 10.0,
    hidden_synapse_mode: str = "linear",
    readout_synapse_mode: str = "linear",
    synapse_clip: float = 1.0,
    update_mode: str = "phased",
    output_bias_update_scale: float = 1.0,
    readout_update_scale: float = 1.0,
    local_update_scale: float = 1.0,
    state_decay: float = 0.0,
    softmax_negative_scale: float = 1.0,
    softmax_error_centering: str = "none",
    softmax_temperature: float = 1.0,
    softmax_competition_mode: str = "all",
    softmax_competitor_power: int = 2,
    softmax_error_gate: str = "none",
    softmax_margin: float = 1.0,
    readout_class_centering: str = "none",
    phase_clock_mode: str = "pwl",
    lr_schedule: str = "constant",
    lr_final_scale: float = 1.0,
    target_source_mode: str = "rails",
    include_output_y_vectors: bool = True,
    sample_edge: float | None = None,
    hidden_preactivation_mode: str = "node",
    hidden_activation_mode: str = "stored",
    score_state_mode: str = "stored",
    score_calculation_mode: str = "node",
    output_rail_mode: str = "node",
    output_delta_mode: str = "node",
) -> tuple[str, int, float]:
    total_samples = x_batch.shape[0]
    if total_samples != update_batch_size * updates:
        raise ValueError("x_batch length must equal update_batch_size * updates")
    if rleak < 0:
        raise ValueError("rleak must be non-negative")
    if update_mode not in {"phased", "direct"}:
        raise ValueError("update_mode must be 'phased' or 'direct'")
    if output_bias_update_scale < 0.0:
        raise ValueError("output_bias_update_scale must be non-negative")
    if readout_update_scale < 0.0:
        raise ValueError("readout_update_scale must be non-negative")
    if local_update_scale < 0.0:
        raise ValueError("local_update_scale must be non-negative")
    if state_decay < 0.0 or state_decay >= 1.0:
        raise ValueError("state_decay must be in [0, 1)")
    if softmax_negative_scale < 0.0:
        raise ValueError("softmax_negative_scale must be non-negative")
    if softmax_temperature <= 0.0:
        raise ValueError("softmax_temperature must be positive")
    if softmax_competition_mode not in {"all", "normalized-power"}:
        raise ValueError("softmax_competition_mode must be 'all' or 'normalized-power'")
    if softmax_competitor_power < 1:
        raise ValueError("softmax_competitor_power must be positive")
    if softmax_error_centering not in {"none", "mean"}:
        raise ValueError("softmax_error_centering must be 'none' or 'mean'")
    if softmax_error_gate not in {"none", "target-margin"}:
        raise ValueError("softmax_error_gate must be 'none' or 'target-margin'")
    if softmax_error_gate == "target-margin" and softmax_margin <= 0.0:
        raise ValueError("softmax_margin must be positive when softmax_error_gate is target-margin")
    if readout_class_centering not in {"none", "mean"}:
        raise ValueError("readout_class_centering must be 'none' or 'mean'")
    if phase_clock_mode not in {"pwl", "analytic"}:
        raise ValueError("phase_clock_mode must be 'pwl' or 'analytic'")
    if lr_schedule not in {"constant", "linear-decay"}:
        raise ValueError("lr_schedule must be 'constant' or 'linear-decay'")
    if target_source_mode not in {"rails", "label"}:
        raise ValueError("target_source_mode must be 'rails' or 'label'")
    if hidden_preactivation_mode not in {"node", "inline"}:
        raise ValueError("hidden_preactivation_mode must be 'node' or 'inline'")
    if hidden_activation_mode not in {"stored", "inline"}:
        raise ValueError("hidden_activation_mode must be 'stored' or 'inline'")
    if score_state_mode not in {"stored", "inline"}:
        raise ValueError("score_state_mode must be 'stored' or 'inline'")
    if score_calculation_mode not in {"node", "inline"}:
        raise ValueError("score_calculation_mode must be 'node' or 'inline'")
    if output_rail_mode not in {"node", "inline"}:
        raise ValueError("output_rail_mode must be 'node' or 'inline'")
    if output_delta_mode not in {"node", "inline"}:
        raise ValueError("output_delta_mode must be 'node' or 'inline'")
    if output_rail_mode == "inline" and include_output_y_vectors:
        raise ValueError("output_rail_mode=inline cannot print final y vectors")
    sample_transition_edge = edge if sample_edge is None else sample_edge
    if sample_transition_edge < 0.0:
        raise ValueError("sample_edge must be non-negative")
    lr_values = lr_schedule_values(lr, updates, lr_schedule, lr_final_scale)
    direct_update = update_mode == "direct"
    if direct_update and update_batch_size != 1:
        raise ValueError("direct update mode requires update_batch_size=1")
    if phase_clock_mode == "analytic" and (not direct_update or update_batch_size != 1):
        raise ValueError("analytic phase clocks require direct update mode with batch_size=1")
    decay_phase = "pacc" if direct_update else "papply"
    n_blocks, channels, block_len = w.shape
    n_classes = readout.shape[0]
    phases, sample_starts, t_stop = make_phase_schedule(update_batch_size, updates, phase, gap, direct_update)
    lr_sample_values = np.repeat(lr_values, update_batch_size)
    lr_control = "{LR}" if lr_schedule == "constant" else "V(lrctrl)"
    local_updates_enabled = local_update_scale != 0.0
    readout_updates_enabled = readout_update_scale != 0.0
    output_bias_updates_enabled = output_bias_update_scale != 0.0
    output_bias_state_frozen = not output_bias_updates_enabled and state_decay == 0.0
    tau = phase / settle_ratio
    phase_area = phase_pulse_area(phase, edge)
    targets = target_matrix(y_batch, n_classes, softmax_output)
    pixel_drives = [
        sample_drive(f"Vpix{i}", f"pix{i}", x_batch[:, i], sample_starts, t_stop, sample_transition_edge, elide_dc=True)
        for i in range(x_batch.shape[1])
    ]
    state_descriptions = phase_state_descriptions(
        update_mode,
        output_bias_state_frozen,
        output_delta_mode,
        hidden_activation_mode,
        score_state_mode,
    )
    lines = [
        "* Phase-resolved transient local-feature/readout training deck.",
        f"* {state_descriptions['persistent_state']}.",
        f"* {state_descriptions['temporary_state']}.",
        f".param LR={lr:.12g}",
        f".param BS={update_batch_size}",
        f".param CW={cw:.12g}",
        f".param CSTATE={cstate:.12g}",
        f".param CGRAD={cgrad:.12g}",
        f".param RLEAK={rleak:.12g}",
        f".param TAU={tau:.12g}",
        f".param TPHASE={phase:.12g}",
        f".param TAREA={phase_area:.12g}",
        f".param LOCAL_UPDATE_SCALE={local_update_scale:.12g}",
        f".param OB_UPDATE_SCALE={output_bias_update_scale:.12g}",
        f".param READOUT_UPDATE_SCALE={readout_update_scale:.12g}",
        f".param STATE_DECAY={state_decay:.12g}",
        f".param SOFTMAX_NEGATIVE_SCALE={softmax_negative_scale:.12g}",
        f".param SOFTMAX_TEMPERATURE={softmax_temperature:.12g}",
        f".param SOFTMAX_COMPETITOR_POWER={softmax_competitor_power}",
        f".param SOFTMAX_MARGIN={softmax_margin:.12g}",
        ".param TARGET_LABEL_SMOOTH=0.02",
        "",
        phase_clock_source_line("pact", "pact", phases["act"], t_stop, phase, gap, edge, phase_clock_mode, direct_update, update_batch_size),
        phase_clock_source_line("pscore", "pscore", phases["score"], t_stop, phase, gap, edge, phase_clock_mode, direct_update, update_batch_size),
        phase_clock_source_line("perr", "perr", phases["err"], t_stop, phase, gap, edge, phase_clock_mode, direct_update, update_batch_size),
        phase_clock_source_line("pbwd", "pbwd", phases["bwd"], t_stop, phase, gap, edge, phase_clock_mode, direct_update, update_batch_size),
        phase_clock_source_line("pacc", "pacc", phases["acc"], t_stop, phase, gap, edge, phase_clock_mode, direct_update, update_batch_size),
        "",
    ]
    if lr_control == "V(lrctrl)":
        lines.append(f"Vlrctrl lrctrl 0 {sample_source_pwl(lr_sample_values, sample_starts, t_stop, sample_transition_edge)}")
        lines.append("")
    if not direct_update:
        lines[-1:-1] = [
            phase_clock_source_line("papply", "papply", phases["apply"], t_stop, phase, gap, edge, phase_clock_mode, direct_update, update_batch_size),
            phase_clock_source_line("pclear", "pclear", phases["clear"], t_stop, phase, gap, edge, phase_clock_mode, direct_update, update_batch_size),
        ]
    elided_pixels = [f"{drive.node}={drive.expr}" for drive in pixel_drives if drive.elided]
    if elided_pixels:
        lines.append("* elided constant pixel sources: " + ", ".join(elided_pixels))
    for drive in pixel_drives:
        if drive.line is not None:
            lines.append(drive.line)
    lines.extend(
        target_source_lines(
            y_batch,
            targets,
            sample_starts,
            t_stop,
            edge,
            target_source_mode=target_source_mode,
            softmax_output=softmax_output,
            sample_edge=sample_transition_edge,
        )
    )
    lines.append("")
    for b in range(n_blocks):
        for c in range(channels):
            for p in range(block_len):
                lines.append(f"Cw{b}_{c}_{p} w{b}_{c}_{p} 0 {{CW}} IC={w[b, c, p]:.12g}")
                if rleak > 0:
                    lines.append(f"Rw{b}_{c}_{p} w{b}_{c}_{p} 0 {{RLEAK}}")
                if not direct_update and local_updates_enabled and pixel_drives[blocks[b][p]].expr != "0":
                    lines.append(f"Cgw{b}_{c}_{p} gw{b}_{c}_{p} 0 {{CGRAD}} IC=0")
            lines.append(f"Chb{b}_{c} hb{b}_{c} 0 {{CW}} IC={hb[b, c]:.12g}")
            if rleak > 0:
                lines.append(f"Rhb{b}_{c} hb{b}_{c} 0 {{RLEAK}}")
            if not direct_update and local_updates_enabled:
                lines.append(f"Cghb{b}_{c} ghb{b}_{c} 0 {{CGRAD}} IC=0")
            if hidden_activation_mode == "stored":
                lines.append(f"Ch{b}_{c} h{b}_{c} 0 {{CSTATE}} IC=0")
            lines.append(f"Cdh{b}_{c} dh{b}_{c} 0 {{CSTATE}} IC=0")
    for k in range(n_classes):
        for b in range(n_blocks):
            for c in range(channels):
                lines.append(f"Cv{k}_{b}_{c} v{k}_{b}_{c} 0 {{CW}} IC={readout[k, b, c]:.12g}")
                if rleak > 0:
                    lines.append(f"Rv{k}_{b}_{c} v{k}_{b}_{c} 0 {{RLEAK}}")
                if not direct_update and readout_updates_enabled:
                    lines.append(f"Cgv{k}_{b}_{c} gv{k}_{b}_{c} 0 {{CGRAD}} IC=0")
        if not output_bias_state_frozen:
            lines.append(f"Cob{k} ob{k} 0 {{CW}} IC={output_bias[k]:.12g}")
            if rleak > 0:
                lines.append(f"Rob{k} ob{k} 0 {{RLEAK}}")
            if not direct_update and output_bias_updates_enabled:
                lines.append(f"Cgob{k} gob{k} 0 {{CGRAD}} IC=0")
        if score_state_mode == "stored":
            lines.append(f"Cscore{k} score{k} 0 {{CSTATE}} IC=0")
        if output_delta_mode == "node":
            lines.append(f"Cd{k} d{k} 0 {{CSTATE}} IC=0")
    lines.append("")
    hidden_preactivation_exprs: dict[tuple[int, int], str] = {}
    hidden_activation_refs: dict[tuple[int, int], str] = {}
    hidden_activation_deriv_refs: dict[tuple[int, int], str] = {}
    for b, idxs in enumerate(blocks):
        for c in range(channels):
            terms = []
            for p, idx in enumerate(idxs):
                pixel_expr = pixel_drives[idx].expr
                if pixel_expr == "0":
                    continue
                terms.append(f"{synapse_transfer_expr(f'V(w{b}_{c}_{p})', hidden_synapse_mode, synapse_clip)}*{pixel_expr}")
            terms.append(f"V(hb{b}_{c})")
            preactivation_expr = " + ".join(terms)
            hidden_preactivation_exprs[(b, c)] = preactivation_expr
            activation_input = f"V(ah{b}_{c})"
            if hidden_preactivation_mode == "node":
                lines.append(f"Bpre_h{b}_{c} ah{b}_{c} 0 V = {preactivation_expr}")
            else:
                activation_input = f"({preactivation_expr})"
            h_calc = local_activation_expr(activation_input, local_activation, relu_clip, relu_leak, softplus_beta)
            if hidden_activation_mode == "stored":
                lines.append(f"Bstore_h{b}_{c} h{b}_{c} 0 I = V(pact)*{{CSTATE}}/{{TAU}}*(V(h{b}_{c})-({h_calc}))")
                hidden_activation_refs[(b, c)] = f"V(h{b}_{c})"
                hidden_activation_deriv_refs[(b, c)] = f"h{b}_{c}"
            else:
                hidden_activation_refs[(b, c)] = f"({h_calc})"
                hidden_activation_deriv_refs[(b, c)] = f"({h_calc})"
    score_refs: list[str] = []
    for k in range(n_classes):
        readout_exprs_for_class: dict[tuple[int, int], str] = {}
        for b in range(n_blocks):
            for c in range(channels):
                class_exprs = [
                    synapse_transfer_expr(f"V(v{kk}_{b}_{c})", readout_synapse_mode, synapse_clip)
                    for kk in range(n_classes)
                ]
                readout_exprs_for_class[(b, c)] = class_centered_expr(class_exprs, k, readout_class_centering)
        score_terms = [
            f"{readout_exprs_for_class[(b, c)]}*{hidden_activation_refs[(b, c)]}"
            for b in range(n_blocks)
            for c in range(channels)
        ]
        score_terms.append(f"{output_bias[k]:.12g}" if output_bias_state_frozen else f"V(ob{k})")
        score_expr = " + ".join(score_terms)
        if score_calculation_mode == "node":
            lines.append(f"Bscore{k} scorecalc{k} 0 V = {score_expr}")
            store_score_expr = f"V(scorecalc{k})"
        else:
            store_score_expr = f"({score_expr})"
        if score_state_mode == "stored":
            lines.append(f"Bstore_score{k} score{k} 0 I = V(pscore)*{{CSTATE}}/{{TAU}}*(V(score{k})-({store_score_expr}))")
            score_refs.append(f"V(score{k})")
        else:
            score_refs.append(store_score_expr)
    delta_refs: list[str] = []
    if softmax_output:
        denom = " + ".join(softmax_exp_expr(score_refs[k]) for k in range(n_classes))
        y_exprs = [f"{softmax_exp_expr(score_refs[k])}/({denom})" for k in range(n_classes)]
        for k in range(n_classes):
            if output_rail_mode == "node":
                lines.append(f"By{k} y{k} 0 V = {y_exprs[k]}")
        raw_delta_exprs = softmax_delta_exprs(
            [f"V(target{k})" for k in range(n_classes)],
            [f"V(y{k})" for k in range(n_classes)] if output_rail_mode == "node" else y_exprs,
            softmax_competition_mode,
            softmax_competitor_power,
        )
        if softmax_error_gate == "target-margin":
            append_target_margin_gate(
                lines,
                "gerr",
                score_refs,
                [f"V(target{k})" for k in range(n_classes)],
            )
        for k in range(n_classes):
            delta_expr = (
                mean_centered_expr(raw_delta_exprs, k)
                if softmax_error_centering == "mean"
                else raw_delta_exprs[k]
            )
            if softmax_error_gate == "target-margin":
                delta_expr = f"V(gerr)*({delta_expr})"
            if output_delta_mode == "node":
                lines.append(f"Bstore_d{k} d{k} 0 I = V(perr)*{{CSTATE}}/{{TAU}}*(V(d{k})-({delta_expr}))")
                delta_refs.append(f"V(d{k})")
            else:
                delta_refs.append(f"({delta_expr})")
    else:
        for k in range(n_classes):
            if linear_output:
                y_expr = score_refs[k]
                if output_rail_mode == "node":
                    lines.append(f"By{k} y{k} 0 V = {y_expr}")
                    y_expr = f"V(y{k})"
                delta_expr = f"(V(target{k})-({y_expr}))"
            else:
                y_expr = tanh_expr(score_refs[k])
                if output_rail_mode == "node":
                    lines.append(f"By{k} y{k} 0 V = {y_expr}")
                delta_expr = f"(V(target{k})-({y_expr}))*(1-({y_expr})*({y_expr}))"
            if output_delta_mode == "node":
                lines.append(f"Bstore_d{k} d{k} 0 I = V(perr)*{{CSTATE}}/{{TAU}}*(V(d{k})-({delta_expr}))")
                delta_refs.append(f"V(d{k})")
            else:
                delta_refs.append(f"({delta_expr})")
    lines.append("")
    for b, idxs in enumerate(blocks):
        for c in range(channels):
            feedback = " + ".join(
                readout_feedback_expr(
                    class_centered_expr(
                        [
                            synapse_transfer_expr(f"V(v{kk}_{b}_{c})", readout_synapse_mode, synapse_clip)
                            for kk in range(n_classes)
                        ],
                        k,
                        readout_class_centering,
                    ),
                    delta_refs[k],
                    readout_feedback_mode,
                    readout_feedback_clip,
                )
                for k in range(n_classes)
            )
            deriv = local_activation_deriv_expr(
                f"V(ah{b}_{c})" if hidden_preactivation_mode == "node" else f"({hidden_preactivation_exprs[(b, c)]})",
                hidden_activation_deriv_refs[(b, c)],
                local_activation,
                relu_clip,
                activation_derivative,
                derivative_floor,
                derivative_gate_threshold,
                relu_leak,
                softplus_beta,
            )
            local_delta = f"({feedback})*{deriv}"
            lines.append(f"Bstore_dh{b}_{c} dh{b}_{c} 0 I = V(pbwd)*{{CSTATE}}/{{TAU}}*(V(dh{b}_{c})-({local_delta}))")
            for p, idx in enumerate(idxs):
                pixel_expr = pixel_drives[idx].expr
                grad = f"V(dh{b}_{c})*{pixel_expr}"
                if direct_update and local_updates_enabled and pixel_expr != "0":
                    lines.append(f"Bupd_w{b}_{c}_{p} w{b}_{c}_{p} 0 I = -V(pacc)*{{CW}}*{lr_control}*{{LOCAL_UPDATE_SCALE}}/({{BS}}*{{TAREA}})*({grad})")
                elif not direct_update and local_updates_enabled and pixel_expr != "0":
                    lines.append(f"Bacc_w{b}_{c}_{p} gw{b}_{c}_{p} 0 I = -V(pacc)*{{CGRAD}}/{{TAREA}}*({grad})")
                    lines.append(f"Bupd_w{b}_{c}_{p} w{b}_{c}_{p} 0 I = -V(papply)*{{CW}}*{lr_control}*{{LOCAL_UPDATE_SCALE}}/({{BS}}*{{TAREA}})*V(gw{b}_{c}_{p})")
                    lines.append(f"Bclear_gw{b}_{c}_{p} gw{b}_{c}_{p} 0 I = V(pclear)*{{CGRAD}}/{{TAU}}*V(gw{b}_{c}_{p})")
                if state_decay > 0.0:
                    lines.append(f"Bdecay_w{b}_{c}_{p} w{b}_{c}_{p} 0 I = V({decay_phase})*{{CW}}*{{STATE_DECAY}}/{{TAREA}}*V(w{b}_{c}_{p})")
            if direct_update and local_updates_enabled:
                lines.append(f"Bupd_hb{b}_{c} hb{b}_{c} 0 I = -V(pacc)*{{CW}}*{lr_control}*{{LOCAL_UPDATE_SCALE}}/({{BS}}*{{TAREA}})*V(dh{b}_{c})")
            elif not direct_update and local_updates_enabled:
                lines.append(f"Bacc_hb{b}_{c} ghb{b}_{c} 0 I = -V(pacc)*{{CGRAD}}/{{TAREA}}*V(dh{b}_{c})")
                lines.append(f"Bupd_hb{b}_{c} hb{b}_{c} 0 I = -V(papply)*{{CW}}*{lr_control}*{{LOCAL_UPDATE_SCALE}}/({{BS}}*{{TAREA}})*V(ghb{b}_{c})")
                lines.append(f"Bclear_ghb{b}_{c} ghb{b}_{c} 0 I = V(pclear)*{{CGRAD}}/{{TAU}}*V(ghb{b}_{c})")
            if state_decay > 0.0:
                lines.append(f"Bdecay_hb{b}_{c} hb{b}_{c} 0 I = V({decay_phase})*{{CW}}*{{STATE_DECAY}}/{{TAREA}}*V(hb{b}_{c})")
    for k in range(n_classes):
        for b in range(n_blocks):
            for c in range(channels):
                grad = f"{delta_refs[k]}*{hidden_activation_refs[(b, c)]}"
                if direct_update and readout_updates_enabled:
                    lines.append(f"Bupd_v{k}_{b}_{c} v{k}_{b}_{c} 0 I = -V(pacc)*{{CW}}*{lr_control}*{{READOUT_UPDATE_SCALE}}/({{BS}}*{{TAREA}})*({grad})")
                elif not direct_update and readout_updates_enabled:
                    lines.append(f"Bacc_v{k}_{b}_{c} gv{k}_{b}_{c} 0 I = -V(pacc)*{{CGRAD}}/{{TAREA}}*({grad})")
                    lines.append(f"Bupd_v{k}_{b}_{c} v{k}_{b}_{c} 0 I = -V(papply)*{{CW}}*{lr_control}*{{READOUT_UPDATE_SCALE}}/({{BS}}*{{TAREA}})*V(gv{k}_{b}_{c})")
                    lines.append(f"Bclear_gv{k}_{b}_{c} gv{k}_{b}_{c} 0 I = V(pclear)*{{CGRAD}}/{{TAU}}*V(gv{k}_{b}_{c})")
                if state_decay > 0.0:
                    lines.append(f"Bdecay_v{k}_{b}_{c} v{k}_{b}_{c} 0 I = V({decay_phase})*{{CW}}*{{STATE_DECAY}}/{{TAREA}}*V(v{k}_{b}_{c})")
        if direct_update and output_bias_updates_enabled:
            lines.append(f"Bupd_ob{k} ob{k} 0 I = -V(pacc)*{{CW}}*{lr_control}*{{OB_UPDATE_SCALE}}/({{BS}}*{{TAREA}})*{delta_refs[k]}")
        elif not direct_update and output_bias_updates_enabled:
            lines.append(f"Bacc_ob{k} gob{k} 0 I = -V(pacc)*{{CGRAD}}/{{TAREA}}*{delta_refs[k]}")
            lines.append(f"Bupd_ob{k} ob{k} 0 I = -V(papply)*{{CW}}*{lr_control}*{{OB_UPDATE_SCALE}}/({{BS}}*{{TAREA}})*V(gob{k})")
            lines.append(f"Bclear_gob{k} gob{k} 0 I = V(pclear)*{{CGRAD}}/{{TAU}}*V(gob{k})")
        if state_decay > 0.0:
            lines.append(f"Bdecay_ob{k} ob{k} 0 I = V({decay_phase})*{{CW}}*{{STATE_DECAY}}/{{TAREA}}*V(ob{k})")
    vectors = [f"V(w{b}_{c}_{p})" for b in range(n_blocks) for c in range(channels) for p in range(block_len)]
    vectors += [f"V(hb{b}_{c})" for b in range(n_blocks) for c in range(channels)]
    vectors += [f"V(v{k}_{b}_{c})" for k in range(n_classes) for b in range(n_blocks) for c in range(channels)]
    if not output_bias_state_frozen:
        vectors += [f"V(ob{k})" for k in range(n_classes)]
    if include_output_y_vectors:
        vectors += [f"V(y{k})" for k in range(n_classes)]
    if output_mode == "native_measure":
        output_mode = "measure"
    if output_mode not in {"wrdata", "control_measure", "measure", "print"}:
        raise ValueError("output_mode must be 'wrdata', 'control_measure', 'measure', or 'print'")
    final_state_phase = phases["acc" if direct_update else "clear"][-1]
    measure_time = final_state_measure_time(t_stop, transient_step, final_state_phase[1])
    probe_times = probe_measure_times(phases["acc" if direct_update else "clear"], probe_updates, gap, t_stop, measure_time)
    lines += ["", ".options method=gear maxord=2"]
    if output_mode == "print":
        print_times = sorted({measure_time, *probe_times.values()})
        print_time_list = ",".join(f"{time:.12g}" for time in print_times)
        lines += [
            f".options output OUTPUTTIMEPOINTS={print_time_list}",
            f".tran {transient_step:.12g} {t_stop:.12g} uic",
            ".print TRAN " + " ".join(vectors),
        ]
    elif output_mode == "measure":
        lines.append(f".tran {transient_step:.12g} {t_stop:.12g} uic")
        for i, vec in enumerate(vectors):
            lines.append(f".measure TRAN m{i:05d} FIND {vec} AT={measure_time:.12g}")
        for probe_idx, update in enumerate(probe_updates):
            for i, vec in enumerate(vectors):
                lines.append(f".measure TRAN p{probe_idx:03d}_{i:05d} FIND {vec} AT={probe_times[update]:.12g}")
    else:
        lines += [".control", f"tran {transient_step:.12g} {t_stop:.12g} uic"]
        if output_mode == "control_measure":
            for i, vec in enumerate(vectors):
                lines.append(f"meas tran m{i:05d} FIND {vec} AT={measure_time:.12g}")
            for probe_idx, update in enumerate(probe_updates):
                for i, vec in enumerate(vectors):
                    lines.append(f"meas tran p{probe_idx:03d}_{i:05d} FIND {vec} AT={probe_times[update]:.12g}")
        else:
            if probe_updates:
                raise ValueError("probe measurements require 'measure' or 'control_measure' output mode")
            lines.append(f"wrdata {out_path} " + " ".join(vectors))
        lines.append(".endc")
    lines += [".end", ""]
    return "\n".join(lines), len(vectors), t_stop


def unpack_state(
    vals: np.ndarray,
    w: np.ndarray,
    hb: np.ndarray,
    readout: np.ndarray,
    output_bias: np.ndarray,
    include_output_bias_vectors: bool = True,
    include_y_vectors: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_classes = readout.shape[0]
    offset = 0
    nw = vals[offset : offset + w.size].reshape(w.shape)
    offset += w.size
    nhb = vals[offset : offset + hb.size].reshape(hb.shape)
    offset += hb.size
    nv = vals[offset : offset + readout.size].reshape(readout.shape)
    offset += readout.size
    if include_output_bias_vectors:
        nob = vals[offset : offset + output_bias.size]
        offset += output_bias.size
    else:
        nob = output_bias.copy()
    if include_y_vectors:
        y = vals[offset : offset + n_classes]
    else:
        y = np.array([], dtype=float)
    return nw, nhb, nv, nob, y


def state_metrics(
    ref: TrainState,
    got: TrainState,
) -> dict[str, float]:
    names = ["local_weights", "local_bias", "readout", "output_bias"]
    metrics: dict[str, float] = {}
    for name, a, b in zip(names, ref, got):
        diff = np.asarray(b) - np.asarray(a)
        metrics[f"{name}_max_abs_diff"] = float(np.max(np.abs(diff)))
        metrics[f"{name}_mean_abs_diff"] = float(np.mean(np.abs(diff)))
        metrics[f"{name}_rms_diff"] = float(np.sqrt(np.mean(diff * diff)))
    all_diff = np.concatenate([(np.asarray(b) - np.asarray(a)).ravel() for a, b in zip(ref, got)])
    metrics["state_max_abs_diff"] = float(np.max(np.abs(all_diff)))
    metrics["state_mean_abs_diff"] = float(np.mean(np.abs(all_diff)))
    metrics["state_rms_diff"] = float(np.sqrt(np.mean(all_diff * all_diff)))
    return metrics


def update_direction_metrics(
    initial: TrainState,
    ref: TrainState,
    got: TrainState,
    eps: float = 1e-12,
) -> dict[str, float]:
    ref_delta = np.concatenate([(np.asarray(after) - np.asarray(before)).ravel() for before, after in zip(initial, ref)])
    got_delta = np.concatenate([(np.asarray(after) - np.asarray(before)).ravel() for before, after in zip(initial, got)])
    ref_norm = float(np.linalg.norm(ref_delta))
    got_norm = float(np.linalg.norm(got_delta))
    metrics = {
        "reference_update_l2": ref_norm,
        "phase_update_l2": got_norm,
    }
    if ref_norm > eps and got_norm > eps:
        metrics["state_update_direction_cosine"] = float(np.dot(ref_delta, got_delta) / (ref_norm * got_norm))
    else:
        metrics["state_update_direction_cosine"] = float("nan")
    mask = np.abs(ref_delta) > eps
    if np.any(mask):
        aligned = np.sign(ref_delta[mask]) == np.sign(got_delta[mask])
        metrics["state_update_sign_alignment_fraction"] = float(np.mean(aligned))
        metrics["state_update_wrong_sign_count"] = float(np.size(aligned) - int(np.sum(aligned)))
    else:
        metrics["state_update_sign_alignment_fraction"] = float("nan")
        metrics["state_update_wrong_sign_count"] = 0.0
    return metrics


def phase_only_update_metrics(
    initial: TrainState,
    got: TrainState,
) -> dict[str, float | None]:
    got_delta = np.concatenate([(np.asarray(after) - np.asarray(before)).ravel() for before, after in zip(initial, got)])
    return {
        "reference_update_l2": None,
        "phase_update_l2": float(np.linalg.norm(got_delta)),
        "state_update_direction_cosine": None,
        "state_update_sign_alignment_fraction": None,
        "state_update_wrong_sign_count": None,
    }


def empty_reference_metrics() -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    for name in ["local_weights", "local_bias", "readout", "output_bias"]:
        metrics[f"{name}_max_abs_diff"] = None
        metrics[f"{name}_mean_abs_diff"] = None
        metrics[f"{name}_rms_diff"] = None
    metrics.update(
        {
            "state_max_abs_diff": None,
            "state_mean_abs_diff": None,
            "state_rms_diff": None,
            "reference_update_l2": None,
            "phase_update_l2": None,
            "state_update_direction_cosine": None,
            "state_update_sign_alignment_fraction": None,
            "state_update_wrong_sign_count": None,
        }
    )
    return metrics


def probe_diagnostic_rows(
    probe_updates: tuple[int, ...],
    probe_vals: dict[int, np.ndarray],
    initial_state: TrainState,
    op_probe_states: dict[int, TrainState],
    include_output_bias_vectors: bool = True,
    include_y_vectors: bool = True,
) -> tuple[list[dict[str, float | int | None]], dict[int, TrainState]]:
    w, hb, readout, output_bias = initial_state
    rows: list[dict[str, float | int | None]] = []
    phase_states: dict[int, TrainState] = {}
    for update in probe_updates:
        if update not in probe_vals:
            raise ValueError(f"missing probe measurements for update {update}")
        probe_w, probe_hb, probe_readout, probe_ob, _probe_y = unpack_state(
            probe_vals[update],
            w,
            hb,
            readout,
            output_bias,
            include_output_bias_vectors,
            include_y_vectors,
        )
        phase_state = (probe_w, probe_hb, probe_readout, probe_ob)
        phase_states[update] = phase_state
        row: dict[str, float | int | None] = {"update": update}
        op_state = op_probe_states.get(update)
        if op_state is None:
            row.update(empty_reference_metrics())
            row.update(phase_only_update_metrics(initial_state, phase_state))
        else:
            row.update(state_metrics(op_state, phase_state))
            row.update(update_direction_metrics(initial_state, op_state, phase_state))
        rows.append(row)
    return rows, phase_states


def finite_row_value(row: dict[str, object], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def summarize_probe_rows(probe_rows: list[dict[str, object]]) -> dict[str, int | float | None]:
    if not probe_rows:
        return {
            "probe_count": 0,
            "final_probe_update": None,
            "final_probe_phase_update_l2": None,
            "max_probe_phase_update_l2": None,
            "max_probe_phase_update_l2_update": None,
            "best_probe_phase_eval_accuracy": None,
            "best_probe_phase_eval_update": None,
            "best_probe_phase_eval_improvement": None,
        }

    final_row = max(probe_rows, key=lambda row: int(row["update"]))
    phase_l2_rows = [(row, finite_row_value(row, "phase_update_l2")) for row in probe_rows]
    phase_l2_rows = [(row, value) for row, value in phase_l2_rows if value is not None]
    max_l2_row, max_l2 = max(phase_l2_rows, key=lambda item: item[1]) if phase_l2_rows else (None, None)

    eval_rows = [(row, finite_row_value(row, "phase_eval_accuracy")) for row in probe_rows]
    eval_rows = [(row, value) for row, value in eval_rows if value is not None]
    best_eval_row, best_eval = max(eval_rows, key=lambda item: item[1]) if eval_rows else (None, None)

    return {
        "probe_count": len(probe_rows),
        "final_probe_update": int(final_row["update"]),
        "final_probe_phase_update_l2": finite_row_value(final_row, "phase_update_l2"),
        "max_probe_phase_update_l2": max_l2,
        "max_probe_phase_update_l2_update": int(max_l2_row["update"]) if max_l2_row is not None else None,
        "best_probe_phase_eval_accuracy": best_eval,
        "best_probe_phase_eval_update": int(best_eval_row["update"]) if best_eval_row is not None else None,
        "best_probe_phase_eval_improvement": (
            finite_row_value(best_eval_row, "phase_eval_improvement") if best_eval_row is not None else None
        ),
    }


def simulator_sidecar_paths(netlist_path: Path) -> tuple[Path, ...]:
    return tuple(Path(str(netlist_path) + suffix) for suffix in (".prn", ".mt0", ".ms0", ".ma0"))


def cleanup_simulator_sidecars(netlist_paths: list[Path]) -> int:
    cleaned = 0
    seen: set[Path] = set()
    for netlist_path in netlist_paths:
        for sidecar in simulator_sidecar_paths(netlist_path):
            if sidecar in seen:
                continue
            seen.add(sidecar)
            if sidecar.exists():
                sidecar.unlink()
                cleaned += 1
    return cleaned


def block_tensor_np(x: np.ndarray, blocks: list[list[int]]) -> np.ndarray:
    return np.stack([x[:, idxs] for idxs in blocks], axis=1)


def synapse_transfer_np(weight: np.ndarray, mode: str, clip: float) -> np.ndarray:
    if mode in {"linear", "full", "ideal"}:
        return weight
    clip = max(float(clip), 1e-12)
    if mode in {"tanh-clipped", "smooth-clipped", "clipped"}:
        return clip * np.tanh(weight / clip)
    if mode in {"hard-clipped", "bounded"}:
        return np.clip(weight, -clip, clip)
    if mode in {"sign", "binary"}:
        return clip * weight / (np.abs(weight) + 1e-9)
    raise ValueError(f"unknown synapse transfer mode {mode!r}")


def local_activation_np(x: np.ndarray, mode: str, relu_clip: float, relu_leak: float, softplus_beta: float) -> np.ndarray:
    if mode == "tanh":
        return np.tanh(x)
    if mode == "relu":
        return np.maximum(x, 0.0)
    if mode in {"clipped-relu", "clipped_relu"}:
        return np.clip(x, 0.0, relu_clip)
    if mode in {"diff-clipped-relu", "differential-clipped-relu", "diff_clipped_relu"}:
        return np.clip(x, 0.0, relu_clip) - np.clip(-x, 0.0, relu_clip)
    if mode in {"leaky-relu", "leaky_relu"}:
        return np.where(x >= 0.0, x, relu_leak * x)
    if mode in {"softplus", "softplus-relu", "softplus_relu"}:
        beta = max(float(softplus_beta), 1e-12)
        return np.logaddexp(0.0, beta * x) / beta
    raise ValueError(f"unknown local activation {mode!r}")


def output_activation_np(
    score: np.ndarray,
    linear_output: bool,
    softmax_output: bool,
    softmax_temperature: float = 1.0,
) -> np.ndarray:
    if linear_output:
        return score
    if softmax_output:
        if softmax_temperature <= 0.0:
            raise ValueError("softmax_temperature must be positive")
        scaled = score / softmax_temperature
        shifted = scaled - np.max(scaled, axis=1, keepdims=True)
        exp_score = np.exp(shifted)
        return exp_score / np.sum(exp_score, axis=1, keepdims=True)
    return np.tanh(score)


def numpy_eval_accuracy(
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    state: TrainState,
    blocks: list[list[int]],
    batch_size: int,
    *,
    linear_output: bool,
    softmax_output: bool,
    local_activation: str,
    relu_clip: float,
    relu_leak: float,
    softplus_beta: float,
    hidden_synapse_mode: str,
    readout_synapse_mode: str,
    synapse_clip: float,
    softmax_temperature: float = 1.0,
    readout_class_centering: str = "none",
) -> float:
    correct = 0
    w, hb, readout, output_bias = state
    eff_w = synapse_transfer_np(w, hidden_synapse_mode, synapse_clip)
    eff_readout = apply_readout_class_centering_np(
        synapse_transfer_np(readout, readout_synapse_mode, synapse_clip),
        readout_class_centering,
    )
    for start in range(0, len(y_eval), batch_size):
        x = x_eval[start : start + batch_size]
        labels = y_eval[start : start + batch_size]
        xb = block_tensor_np(x, blocks)
        pre = np.einsum("nbp,bcp->nbc", xb, eff_w) + hb
        h = local_activation_np(pre, local_activation, relu_clip, relu_leak, softplus_beta)
        score = np.einsum("nbc,kbc->nk", h, eff_readout) + output_bias
        y = output_activation_np(score, linear_output, softmax_output, softmax_temperature)
        correct += int(np.sum(np.argmax(y, axis=1) == labels))
    return correct / max(len(y_eval), 1)


def numpy_eval_diagnostics(
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    state: TrainState,
    blocks: list[list[int]],
    batch_size: int,
    *,
    linear_output: bool,
    softmax_output: bool,
    local_activation: str,
    relu_clip: float,
    relu_leak: float,
    softplus_beta: float,
    hidden_synapse_mode: str,
    readout_synapse_mode: str,
    synapse_clip: float,
    softmax_temperature: float = 1.0,
    readout_class_centering: str = "none",
) -> dict[str, object]:
    _w, _hb, _readout, output_bias = state
    n_classes = int(output_bias.shape[0])
    preds: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    w, hb, readout, output_bias = state
    eff_w = synapse_transfer_np(w, hidden_synapse_mode, synapse_clip)
    eff_readout = apply_readout_class_centering_np(
        synapse_transfer_np(readout, readout_synapse_mode, synapse_clip),
        readout_class_centering,
    )
    for start in range(0, len(y_eval), batch_size):
        x = x_eval[start : start + batch_size]
        labels = y_eval[start : start + batch_size]
        xb = block_tensor_np(x, blocks)
        pre = np.einsum("nbp,bcp->nbc", xb, eff_w) + hb
        h = local_activation_np(pre, local_activation, relu_clip, relu_leak, softplus_beta)
        score = np.einsum("nbc,kbc->nk", h, eff_readout) + output_bias
        y = output_activation_np(score, linear_output, softmax_output, softmax_temperature)
        preds.append(np.argmax(y, axis=1))
        labels_all.append(labels)

    pred = np.concatenate(preds) if preds else np.zeros((0,), dtype=int)
    labels = np.concatenate(labels_all) if labels_all else np.zeros((0,), dtype=int)
    correct_mask = pred == labels
    label_hist = np.bincount(labels, minlength=n_classes)
    pred_hist = np.bincount(pred, minlength=n_classes)
    correct_hist = np.bincount(labels[correct_mask], minlength=n_classes)
    per_class_accuracy = [
        None if int(total) == 0 else float(correct / total)
        for correct, total in zip(correct_hist, label_hist)
    ]
    total = max(int(labels.size), 1)
    dominant_pred_count = int(np.max(pred_hist)) if pred_hist.size else 0
    dominant_pred_class = int(np.argmax(pred_hist)) if pred_hist.size else None
    return {
        "accuracy": float(np.sum(correct_mask) / total),
        "label_histogram": [int(value) for value in label_hist],
        "prediction_histogram": [int(value) for value in pred_hist],
        "correct_by_label": [int(value) for value in correct_hist],
        "per_class_accuracy": per_class_accuracy,
        "dominant_pred_class": dominant_pred_class,
        "dominant_pred_fraction": float(dominant_pred_count / total),
        "unique_predicted_classes": int(np.count_nonzero(pred_hist)),
    }


def diagnostic_eval_accuracy(
    eval_backend: str,
    spice_bin: str,
    netlist_path: Path,
    data_path: Path,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    state: TrainState,
    blocks: list[list[int]],
    batch_size: int,
    timeout: float,
    *,
    linear_output: bool,
    softmax_output: bool,
    local_activation: str,
    relu_clip: float,
    relu_leak: float,
    softplus_beta: float,
    hidden_synapse_mode: str,
    readout_synapse_mode: str,
    synapse_clip: float,
    softmax_temperature: float = 1.0,
    readout_class_centering: str = "none",
) -> float:
    if eval_backend == "spice":
        w, hb, readout, output_bias = state
        return run_eval(
            spice_bin,
            netlist_path,
            data_path,
            x_eval,
            y_eval,
            w,
            hb,
            readout,
            output_bias,
            blocks,
            batch_size,
            timeout,
            linear_output=linear_output,
            softmax_output=softmax_output,
            local_activation=local_activation,
            relu_clip=relu_clip,
            relu_leak=relu_leak,
            softplus_beta=softplus_beta,
            hidden_synapse_mode=hidden_synapse_mode,
            readout_synapse_mode=readout_synapse_mode,
            synapse_clip=synapse_clip,
            softmax_temperature=softmax_temperature,
            readout_class_centering=readout_class_centering,
        )
    if eval_backend == "numpy":
        return numpy_eval_accuracy(
            x_eval,
            y_eval,
            state,
            blocks,
            batch_size,
            linear_output=linear_output,
            softmax_output=softmax_output,
            local_activation=local_activation,
            relu_clip=relu_clip,
            relu_leak=relu_leak,
            softplus_beta=softplus_beta,
            hidden_synapse_mode=hidden_synapse_mode,
            readout_synapse_mode=readout_synapse_mode,
            synapse_clip=synapse_clip,
            softmax_temperature=softmax_temperature,
            readout_class_centering=readout_class_centering,
        )
    raise ValueError("eval_backend must be 'spice' or 'numpy'")


def diagnostic_eval_accuracies(
    eval_backend: str,
    spice_bin: str,
    netlist_path: Path,
    data_path: Path,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    state: TrainState,
    blocks: list[list[int]],
    batch_size: int,
    timeout: float,
    **eval_kwargs,
) -> dict[str, float]:
    if eval_backend in {"spice", "numpy"}:
        return {
            eval_backend: diagnostic_eval_accuracy(
                eval_backend,
                spice_bin,
                netlist_path,
                data_path,
                x_eval,
                y_eval,
                state,
                blocks,
                batch_size,
                timeout,
                **eval_kwargs,
            )
        }
    if eval_backend == "both":
        spice_accuracy = diagnostic_eval_accuracy(
            "spice",
            spice_bin,
            netlist_path,
            data_path,
            x_eval,
            y_eval,
            state,
            blocks,
            batch_size,
            timeout,
            **eval_kwargs,
        )
        numpy_accuracy = diagnostic_eval_accuracy(
            "numpy",
            spice_bin,
            netlist_path,
            data_path,
            x_eval,
            y_eval,
            state,
            blocks,
            batch_size,
            timeout,
            **eval_kwargs,
        )
        return {"spice": spice_accuracy, "numpy": numpy_accuracy}
    raise ValueError("eval_backend must be 'spice', 'numpy', or 'both'")


def primary_eval_accuracy(eval_backend: str, accuracies: dict[str, float]) -> float:
    if eval_backend == "numpy":
        return accuracies["numpy"]
    if eval_backend in {"spice", "both"}:
        return accuracies["spice"]
    raise ValueError("eval_backend must be 'spice', 'numpy', or 'both'")


def eval_backend_abs_diff(accuracies: dict[str, float]) -> float | None:
    if "spice" in accuracies and "numpy" in accuracies:
        return abs(accuracies["spice"] - accuracies["numpy"])
    return None


def select_phase_output_mode(
    requested_mode: str,
    spice_bin: str,
    final_measures: bool,
    probe_updates: tuple[int, ...],
) -> str:
    if requested_mode != "auto":
        return requested_mode
    if is_xyce(spice_bin):
        return "measure" if final_measures else "print"
    return "control_measure" if final_measures or probe_updates else "wrdata"


def load_or_init_weights(
    init_weights: str,
    rng: np.random.Generator,
    n_blocks: int,
    channels: int,
    block_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    expected = ((n_blocks, channels, block_len), (n_blocks, channels), (10, n_blocks, channels), (10,))
    if init_weights:
        init = np.load(init_weights)
        if not {"local_weights", "local_bias", "readout", "output_bias"}.issubset(init.files):
            raise ValueError(f"expected local-feature checkpoint keys, got {init.files}")
        w = init["local_weights"].copy()
        hb = init["local_bias"].copy()
        readout = init["readout"].copy()
        output_bias = init["output_bias"].copy()
        actual = (w.shape, hb.shape, readout.shape, output_bias.shape)
        if actual != expected:
            raise ValueError(f"initial weight shapes {actual} do not match expected {expected}")
        return w, hb, readout, output_bias
    w = rng.normal(0.0, 0.05, size=expected[0])
    hb = np.zeros(expected[1])
    readout = rng.normal(0.0, 0.05, size=expected[2])
    output_bias = np.zeros(expected[3])
    return w, hb, readout, output_bias


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-samples", type=int, default=200)
    ap.add_argument("--eval-samples", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=4)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--channels", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--updates", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--lr-schedule", choices=["constant", "linear-decay"], default="constant")
    ap.add_argument("--lr-final-scale", type=float, default=1.0)
    ap.add_argument("--linear-output", action="store_true")
    ap.add_argument("--softmax-output", action="store_true")
    ap.add_argument(
        "--local-activation",
        default="tanh",
        choices=[
            "tanh",
            "relu",
            "clipped-relu",
            "clipped_relu",
            "diff-clipped-relu",
            "differential-clipped-relu",
            "diff_clipped_relu",
            "leaky-relu",
            "leaky_relu",
            "softplus",
            "softplus-relu",
            "softplus_relu",
        ],
    )
    ap.add_argument("--relu-clip", type=float, default=1.0)
    ap.add_argument("--relu-leak", type=float, default=0.01)
    ap.add_argument("--softplus-beta", type=float, default=10.0)
    ap.add_argument("--activation-derivative", choices=["exact", "stored-gate", "unity", "floor-exact"], default="exact")
    ap.add_argument("--derivative-floor", type=float, default=0.0)
    ap.add_argument("--derivative-gate-threshold", type=float, default=1e-6)
    ap.add_argument("--readout-feedback-mode", choices=["readout", "full-readout", "exact", "sign-readout", "sign", "clipped-readout", "clipped"], default="readout")
    ap.add_argument("--readout-feedback-clip", type=float, default=0.05)
    ap.add_argument("--output-bias-update-scale", type=float, default=1.0)
    ap.add_argument("--readout-update-scale", type=float, default=1.0)
    ap.add_argument("--local-update-scale", type=float, default=1.0)
    ap.add_argument(
        "--state-decay",
        type=float,
        default=0.0,
        help="Per-update on-device decay fraction for persistent local/readout/bias state; must be in [0, 1).",
    )
    ap.add_argument(
        "--softmax-negative-scale",
        type=float,
        default=1.0,
        help="Scale non-target softmax error rails: target error is 1-y_target, non-target error is -scale*y_k.",
    )
    ap.add_argument(
        "--softmax-error-centering",
        choices=["none", "mean"],
        default="none",
        help="Optionally subtract the per-sample mean softmax error so class error rails remain zero-sum.",
    )
    ap.add_argument(
        "--softmax-temperature",
        type=float,
        default=1.0,
        help="Divide score rails by this positive value before softmax error generation.",
    )
    ap.add_argument(
        "--softmax-competition-mode",
        choices=["all", "normalized-power"],
        default="all",
        help="Choose whether non-target softmax error is spread over all classes or focused by normalized y^p competition.",
    )
    ap.add_argument(
        "--softmax-competitor-power",
        type=int,
        default=2,
        help="Positive integer p for --softmax-competition-mode normalized-power.",
    )
    ap.add_argument(
        "--softmax-error-gate",
        choices=["none", "target-margin"],
        default="none",
        help="Optionally gate softmax error by target-vs-competitor score margin before storing class deltas.",
    )
    ap.add_argument(
        "--softmax-margin",
        type=float,
        default=1.0,
        help="Positive target score margin used by --softmax-error-gate target-margin.",
    )
    synapse_modes = ["linear", "full", "ideal", "tanh-clipped", "smooth-clipped", "clipped", "hard-clipped", "bounded", "sign", "binary"]
    ap.add_argument("--hidden-synapse-mode", choices=synapse_modes, default="linear")
    ap.add_argument("--readout-synapse-mode", choices=synapse_modes, default="linear")
    ap.add_argument("--synapse-clip", type=float, default=1.0)
    ap.add_argument("--readout-class-centering", choices=["none", "mean"], default="none")
    ap.add_argument(
        "--hidden-preactivation-mode",
        choices=["node", "inline"],
        default="node",
        help=(
            "Use separate hidden preactivation behavioral nodes, or inline each hidden preactivation "
            "expression into the activation/derivative equations for fused-cell experiments."
        ),
    )
    ap.add_argument(
        "--hidden-activation-mode",
        choices=["stored", "inline"],
        default="stored",
        help=(
            "Store hidden activations on phase-latched capacitors, or inline the activation expression "
            "directly into score/readout-gradient paths for aggressive fused-cell experiments."
        ),
    )
    ap.add_argument(
        "--score-calculation-mode",
        choices=["node", "inline"],
        default="node",
        help=(
            "Use separate per-class score calculation behavioral nodes, or inline each score expression "
            "into the score storage current source for fused readout/summing experiments."
        ),
    )
    ap.add_argument(
        "--score-state-mode",
        choices=["stored", "inline"],
        default="stored",
        help=(
            "Store class scores on phase-latched capacitors, or inline score expressions directly "
            "into output/error paths for aggressive fused readout experiments."
        ),
    )
    ap.add_argument(
        "--output-rail-mode",
        choices=["node", "inline"],
        default="node",
        help=(
            "Use separate y/output behavioral rails, or inline output activation/probability expressions "
            "directly into error sources. Inline mode requires final y-vector printing to be disabled."
        ),
    )
    ap.add_argument(
        "--output-delta-mode",
        choices=["node", "inline"],
        default="node",
        help=(
            "Use separate stored class-delta capacitors, or inline class-delta expressions into "
            "backward and update sources for direct fused error-head experiments."
        ),
    )
    ap.add_argument(
        "--target-source-mode",
        choices=["rails", "label"],
        default="rails",
        help="Drive target rails directly as per-class PWL sources, or decode them from one label PWL source.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--simulator", default=None)
    ap.add_argument(
        "--max-transient-points",
        type=int,
        default=0,
        help="Optional preflight guard on ceil(t_stop / transient_step) + 1; 0 disables the guard.",
    )
    ap.add_argument(
        "--max-source-pwl-points",
        type=int,
        default=0,
        help="Optional preflight guard on input/target plus phase-clock PWL point count; 0 disables the guard.",
    )
    ap.add_argument(
        "--max-sample-sources",
        type=int,
        default=0,
        help="Optional preflight guard on emitted input/target voltage-source count; 0 disables the guard.",
    )
    ap.add_argument(
        "--max-total-sources",
        type=int,
        default=0,
        help="Optional preflight guard on emitted voltage/behavioral source element count; 0 disables the guard.",
    )
    ap.add_argument(
        "--max-output-vectors",
        type=int,
        default=0,
        help="Optional preflight guard on final/probe output vector count; 0 disables the guard.",
    )
    ap.add_argument(
        "--max-auxiliary-algebraic-sources",
        type=int,
        default=0,
        help=(
            "Optional preflight guard on non-persistent algebraic source families controlled by fused deck modes; "
            "0 disables the guard."
        ),
    )
    ap.add_argument("--init-weights", default="")
    ap.add_argument("--phase", type=float, default=2e-9)
    ap.add_argument("--gap", type=float, default=0.2e-9)
    ap.add_argument("--edge", type=float, default=10e-12)
    ap.add_argument(
        "--sample-edge",
        type=float,
        default=None,
        help=(
            "Optional transition edge for input, label, and LR-control PWL sources. "
            "Unset uses --edge; 0 emits sharp sample steps while keeping phase-clock edges finite."
        ),
    )
    ap.add_argument("--settle-ratio", type=float, default=40.0)
    ap.add_argument("--transient-step", type=float, default=20e-12)
    ap.add_argument("--cw", type=float, default=1e-12)
    ap.add_argument("--cstate", type=float, default=1e-12)
    ap.add_argument("--cgrad", type=float, default=1e-12)
    ap.add_argument("--rleak", type=float, default=1e18)
    ap.add_argument("--direction-cosine-threshold", type=float, default=0.999)
    ap.add_argument("--sign-alignment-threshold", type=float, default=0.98)
    ap.add_argument("--eval-accuracy-diff-threshold", type=float, default=0.0)
    ap.add_argument("--random-accuracy-threshold", type=float, default=0.10)
    ap.add_argument("--learning-improvement-threshold", type=float, default=0.02)
    ap.add_argument("--eval-backend", choices=["spice", "numpy", "both"], default="spice")
    ap.add_argument("--reference-mode", choices=["spice", "none"], default="spice")
    ap.add_argument("--phase-output-mode", choices=["auto", "measure", "print", "control_measure", "wrdata"], default="auto")
    ap.add_argument(
        "--phase-output-include-y",
        action="store_true",
        help="Also print/measure final class output rails. Final eval recomputes outputs from weights, so strict runs omit these by default.",
    )
    ap.add_argument("--update-mode", choices=["phased", "direct"], default="phased")
    ap.add_argument(
        "--phase-clock-mode",
        choices=["pwl", "analytic"],
        default="pwl",
        help=(
            "Emit phase clocks as explicit PWL voltage sources or as bounded analytic behavioral clocks. "
            "Analytic clocks are Xyce-only and require --update-mode direct with --batch-size 1."
        ),
    )
    ap.add_argument(
        "--strict-fully-on-device",
        action="store_true",
        help=(
            "Fail unless the run is random-init, batch_size=1, and has no Python-side "
            "reference replay; final SPICE/NumPy diagnostics are still allowed after the transient."
        ),
    )
    ap.add_argument("--simulator-extra-args", default="", help=f"Extra simulator command-line arguments, also available via {SPICE_SIMULATOR_ARGS_ENV}.")
    ap.add_argument("--final-measures", action="store_true")
    ap.add_argument(
        "--probe-updates",
        default="",
        help="Comma-separated 1-based update numbers, ranges, 'final', or 'powers2' to measure inside the same transient.",
    )
    ap.add_argument(
        "--eval-probe-updates",
        action="store_true",
        help="After the uninterrupted transient finishes, run diagnostic evals on measured probe states.",
    )
    ap.add_argument(
        "--preflight-only",
        action="store_true",
        help="Load the deterministic training prefix, report source/transient complexity, and exit before simulator setup.",
    )
    ap.add_argument("--tag", default="phase_local_feature")
    args = ap.parse_args()

    if args.batch_size <= 0 or args.updates <= 0:
        raise ValueError("--batch-size and --updates must be positive")
    if args.lr < 0:
        raise ValueError("--lr must be non-negative")
    if args.lr_final_scale < 0:
        raise ValueError("--lr-final-scale must be non-negative")
    if args.lr_schedule != "constant" and args.reference_mode != "none":
        raise ValueError("--lr-schedule non-constant requires --reference-mode none")
    if args.linear_output and args.softmax_output:
        raise ValueError("--linear-output and --softmax-output are mutually exclusive")
    if args.channels <= 0:
        raise ValueError("--channels must be positive")
    if args.relu_clip <= 0:
        raise ValueError("--relu-clip must be positive")
    if args.relu_leak < 0:
        raise ValueError("--relu-leak must be non-negative")
    if args.softplus_beta <= 0:
        raise ValueError("--softplus-beta must be positive")
    if args.derivative_floor < 0 or args.derivative_floor > 1:
        raise ValueError("--derivative-floor must be between 0 and 1")
    if args.derivative_gate_threshold < 0:
        raise ValueError("--derivative-gate-threshold must be non-negative")
    if args.readout_feedback_clip <= 0:
        raise ValueError("--readout-feedback-clip must be positive")
    if args.output_bias_update_scale < 0:
        raise ValueError("--output-bias-update-scale must be non-negative")
    if args.readout_update_scale < 0:
        raise ValueError("--readout-update-scale must be non-negative")
    if args.local_update_scale < 0:
        raise ValueError("--local-update-scale must be non-negative")
    if args.state_decay < 0 or args.state_decay >= 1:
        raise ValueError("--state-decay must be in [0, 1)")
    if args.softmax_negative_scale < 0:
        raise ValueError("--softmax-negative-scale must be non-negative")
    if args.softmax_temperature <= 0:
        raise ValueError("--softmax-temperature must be positive")
    if args.softmax_competitor_power < 1:
        raise ValueError("--softmax-competitor-power must be positive")
    if args.softmax_error_gate == "target-margin" and args.softmax_margin <= 0:
        raise ValueError("--softmax-margin must be positive when --softmax-error-gate is target-margin")
    if args.synapse_clip <= 0:
        raise ValueError("--synapse-clip must be positive")
    if args.phase <= 0 or args.settle_ratio <= 0:
        raise ValueError("--phase and --settle-ratio must be positive")
    sample_edge = args.edge if args.sample_edge is None else args.sample_edge
    if sample_edge < 0.0:
        raise ValueError("--sample-edge must be non-negative")
    if args.max_transient_points < 0:
        raise ValueError("--max-transient-points must be non-negative")
    if args.max_source_pwl_points < 0:
        raise ValueError("--max-source-pwl-points must be non-negative")
    if args.max_sample_sources < 0:
        raise ValueError("--max-sample-sources must be non-negative")
    if args.max_total_sources < 0:
        raise ValueError("--max-total-sources must be non-negative")
    if args.max_output_vectors < 0:
        raise ValueError("--max-output-vectors must be non-negative")
    if args.max_auxiliary_algebraic_sources < 0:
        raise ValueError("--max-auxiliary-algebraic-sources must be non-negative")
    if args.direction_cosine_threshold < -1 or args.direction_cosine_threshold > 1:
        raise ValueError("--direction-cosine-threshold must be between -1 and 1")
    if args.sign_alignment_threshold < 0 or args.sign_alignment_threshold > 1:
        raise ValueError("--sign-alignment-threshold must be between 0 and 1")
    if args.eval_accuracy_diff_threshold < 0:
        raise ValueError("--eval-accuracy-diff-threshold must be non-negative")
    if args.random_accuracy_threshold < 0 or args.random_accuracy_threshold > 1:
        raise ValueError("--random-accuracy-threshold must be between 0 and 1")
    if args.learning_improvement_threshold < 0:
        raise ValueError("--learning-improvement-threshold must be non-negative")
    probe_updates = parse_probe_update_list(args.probe_updates, args.updates)
    if args.eval_probe_updates and not probe_updates:
        raise ValueError("--eval-probe-updates requires --probe-updates")
    if args.eval_probe_updates and args.eval_samples <= 0:
        raise ValueError("--eval-probe-updates requires --eval-samples > 0")
    if args.output_rail_mode == "inline" and args.phase_output_include_y:
        raise ValueError("--output-rail-mode inline cannot be combined with --phase-output-include-y")
    if args.update_mode == "direct" and args.batch_size != 1:
        raise ValueError("--update-mode direct requires --batch-size 1")
    if args.phase_clock_mode == "analytic" and (args.update_mode != "direct" or args.batch_size != 1):
        raise ValueError("--phase-clock-mode analytic requires --update-mode direct with --batch-size 1")
    if args.strict_fully_on_device:
        validate_strict_fully_on_device_args(args.batch_size, args.reference_mode, args.init_weights)
    output_bias_state_frozen = args.output_bias_update_scale == 0.0 and args.state_decay == 0.0
    _preflight_phases, _preflight_sample_starts, preflight_t_stop = make_phase_schedule(
        args.batch_size,
        args.updates,
        args.phase,
        args.gap,
        args.update_mode == "direct",
    )
    final_state_phase = _preflight_phases["acc" if args.update_mode == "direct" else "clear"][-1]
    final_state_measure_time(preflight_t_stop, args.transient_step, final_state_phase[1])
    estimated_transient_points = estimate_transient_points(preflight_t_stop, args.transient_step)
    validate_transient_point_budget(estimated_transient_points, args.max_transient_points)
    if args.simulator_extra_args:
        os.environ[SPICE_SIMULATOR_ARGS_ENV] = args.simulator_extra_args

    stride = args.block_size if args.stride is None else args.stride
    blocks = block_indices(args.image_size, args.block_size, stride)
    expected_w_shape = (len(blocks), args.channels, args.block_size * args.block_size)
    expected_hb_shape = (len(blocks), args.channels)
    expected_readout_shape = (10, len(blocks), args.channels)
    expected_ob_shape = (10,)
    preflight_hidden_preactivation_source_count = hidden_preactivation_source_count(
        args.hidden_preactivation_mode,
        len(blocks),
        args.channels,
    )
    preflight_hidden_activation_state_count = hidden_activation_state_count(
        args.hidden_activation_mode,
        len(blocks),
        args.channels,
    )
    preflight_hidden_delta_state_count = hidden_delta_state_count(len(blocks), args.channels)
    preflight_score_state_count = score_state_count(10, args.score_state_mode)
    preflight_score_calculation_source_count = score_calculation_source_count(args.score_calculation_mode, 10)
    preflight_output_rail_source_count = output_rail_source_count(args.output_rail_mode, 10)
    preflight_output_delta_state_count = output_delta_state_count(args.output_delta_mode, 10)
    preflight_auxiliary_algebraic_source_count = auxiliary_algebraic_source_count(
        preflight_hidden_preactivation_source_count,
        preflight_score_calculation_source_count,
        preflight_output_rail_source_count,
    )
    preflight_phase_output_vector_count = int(
        np.prod(expected_w_shape)
        + np.prod(expected_hb_shape)
        + np.prod(expected_readout_shape)
        + (0 if output_bias_state_frozen else np.prod(expected_ob_shape))
        + (np.prod(expected_ob_shape) if args.phase_output_include_y else 0)
    )
    validate_output_vector_budget(preflight_phase_output_vector_count, args.max_output_vectors)
    validate_auxiliary_algebraic_source_budget(
        preflight_auxiliary_algebraic_source_count,
        args.max_auxiliary_algebraic_sources,
    )
    total_samples = args.batch_size * args.updates
    if args.train_samples < total_samples:
        raise ValueError("--train-samples must cover --batch-size * --updates")
    x_train, y_train, x_test, y_test = load_mnist_sequence(args.train_samples, max(1, args.eval_samples), args.image_size, args.seed)
    train_indices, eval_indices = mnist_index_splits(
        args.train_samples,
        max(1, args.eval_samples),
        MNIST_TRAIN_COUNT,
        MNIST_TEST_COUNT,
        args.seed,
    )
    x_batch = x_train[:total_samples]
    y_batch = y_train[:total_samples]
    preflight_gradient_accumulator_state_count = gradient_accumulator_state_count(
        args.update_mode,
        blocks,
        args.channels,
        10,
        x_batch,
        local_updates_enabled=args.local_update_scale != 0.0,
        readout_updates_enabled=args.readout_update_scale != 0.0,
        output_bias_updates_enabled=args.output_bias_update_scale != 0.0,
    )
    preflight_temporary_state_count = temporary_state_count(
        hidden_activation_states=preflight_hidden_activation_state_count,
        hidden_delta_states=preflight_hidden_delta_state_count,
        score_states=preflight_score_state_count,
        output_delta_states=preflight_output_delta_state_count,
        gradient_accumulator_states=preflight_gradient_accumulator_state_count,
    )
    source_phases, source_sample_starts, source_t_stop = make_phase_schedule(
        args.batch_size,
        args.updates,
        args.phase,
        args.gap,
        args.update_mode == "direct",
    )
    source_lr_values = None
    if args.lr_schedule != "constant":
        source_lr_values = np.repeat(lr_schedule_values(args.lr, args.updates, args.lr_schedule, args.lr_final_scale), args.batch_size)
    source_complexity = phase_source_complexity(
        x_batch,
        target_matrix(y_batch, 10, args.softmax_output),
        source_phases,
        source_sample_starts,
        source_t_stop,
        args.edge,
        args.update_mode == "direct",
        args.phase_clock_mode,
        source_lr_values,
        labels=y_batch,
        target_source_mode=args.target_source_mode,
        sample_edge=sample_edge,
    )
    validate_source_point_budget(source_complexity, args.max_source_pwl_points)
    validate_sample_source_budget(source_complexity, args.max_sample_sources)
    validate_total_source_budget(source_complexity, args.max_total_sources)
    if args.preflight_only:
        print(
            json.dumps(
                phase_preflight_summary(
                    simulator_selector=args.simulator,
                    image_size=args.image_size,
                    block_size=args.block_size,
                    stride=stride,
                    blocks=len(blocks),
                    channels=args.channels,
                    train_samples=args.train_samples,
                    eval_samples=args.eval_samples,
                    batch_size=args.batch_size,
                    updates=args.updates,
                    total_samples=total_samples,
                    train_indices=train_indices,
                    eval_indices=eval_indices,
                    labels=y_batch,
                    lr=args.lr,
                    lr_schedule=args.lr_schedule,
                    lr_final_scale=args.lr_final_scale,
                    update_mode=args.update_mode,
                    phase_clock_mode=args.phase_clock_mode,
                    target_source_mode=args.target_source_mode,
                    hidden_preactivation_mode=args.hidden_preactivation_mode,
                    hidden_preactivation_source_count=preflight_hidden_preactivation_source_count,
                    hidden_activation_mode=args.hidden_activation_mode,
                    hidden_activation_state_count=preflight_hidden_activation_state_count,
                    hidden_delta_state_count=preflight_hidden_delta_state_count,
                    score_state_mode=args.score_state_mode,
                    score_state_count=preflight_score_state_count,
                    gradient_accumulator_state_count=preflight_gradient_accumulator_state_count,
                    temporary_state_count=preflight_temporary_state_count,
                    score_calculation_mode=args.score_calculation_mode,
                    score_calculation_source_count=preflight_score_calculation_source_count,
                    output_rail_mode=args.output_rail_mode,
                    output_rail_source_count=preflight_output_rail_source_count,
                    output_delta_mode=args.output_delta_mode,
                    output_delta_state_count=preflight_output_delta_state_count,
                    output_bias_state_frozen=output_bias_state_frozen,
                    phase_output_vector_count=preflight_phase_output_vector_count,
                    phase_output_includes_y=args.phase_output_include_y,
                    reference_mode=args.reference_mode,
                    init_weights=args.init_weights,
                    strict_fully_on_device=args.strict_fully_on_device,
                    estimated_transient_points=estimated_transient_points,
                    max_transient_points=args.max_transient_points,
                    max_source_pwl_points=args.max_source_pwl_points,
                    max_sample_sources=args.max_sample_sources,
                    max_total_sources=args.max_total_sources,
                    max_output_vectors=args.max_output_vectors,
                    max_auxiliary_algebraic_sources=args.max_auxiliary_algebraic_sources,
                    t_stop=preflight_t_stop,
                    transient_step=args.transient_step,
                    phase=args.phase,
                    sample_edge=sample_edge,
                    settle_ratio=args.settle_ratio,
                    source_complexity=source_complexity,
                ),
                indent=2,
            )
        )
        return
    rng = np.random.default_rng(args.seed)
    w, hb, readout, output_bias = load_or_init_weights(
        args.init_weights,
        rng,
        len(blocks),
        args.channels,
        args.block_size * args.block_size,
    )

    spice_bin, version = detect_spice(args.simulator)
    if args.phase_clock_mode == "analytic" and not is_xyce(spice_bin):
        raise ValueError("--phase-clock-mode analytic is Xyce-only")
    generated = ROOT / "spice/generated"
    results = ROOT / "spice/results"
    generated.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)

    stem = f"spice_mnist_local_feature_phase_{sanitize_tag(args.tag)}"
    phase_netlist = generated / f"{stem}.cir"
    phase_data = results / f"{stem}.dat"
    op_netlist = generated / f"{stem}_op_reference.cir"
    op_data = results / f"{stem}_op_reference.dat"
    owned_netlists = [phase_netlist]
    if args.reference_mode == "spice":
        owned_netlists.append(op_netlist)
    phase_output_mode = select_phase_output_mode(args.phase_output_mode, spice_bin, args.final_measures, probe_updates)
    include_output_bias_vectors = not output_bias_state_frozen
    include_y_vectors = args.phase_output_include_y

    netlist, n_vec, t_stop = make_phase_transient_netlist(
        x_batch,
        y_batch,
        w,
        hb,
        readout,
        output_bias,
        blocks,
        args.lr,
        phase_data,
        args.linear_output,
        args.batch_size,
        args.updates,
        args.phase,
        args.gap,
        args.edge,
        args.settle_ratio,
        args.transient_step,
        args.cw,
        args.cstate,
        args.cgrad,
        args.rleak,
        args.softmax_output,
        phase_output_mode,
        probe_updates,
        args.local_activation,
        args.relu_clip,
        args.activation_derivative,
        args.derivative_floor,
        args.derivative_gate_threshold,
        args.readout_feedback_mode,
        args.readout_feedback_clip,
        args.relu_leak,
        args.softplus_beta,
        args.hidden_synapse_mode,
        args.readout_synapse_mode,
        args.synapse_clip,
        args.update_mode,
        args.output_bias_update_scale,
        args.readout_update_scale,
        args.local_update_scale,
        args.state_decay,
        args.softmax_negative_scale,
        args.softmax_error_centering,
        args.softmax_temperature,
        args.softmax_competition_mode,
        args.softmax_competitor_power,
        args.softmax_error_gate,
        args.softmax_margin,
        args.readout_class_centering,
        args.phase_clock_mode,
        args.lr_schedule,
        args.lr_final_scale,
        args.target_source_mode,
        include_y_vectors,
        sample_edge,
        args.hidden_preactivation_mode,
        args.hidden_activation_mode,
        args.score_state_mode,
        args.score_calculation_mode,
        args.output_rail_mode,
        args.output_delta_mode,
    )
    phase_netlist.write_text(prepare_phase_netlist(netlist, spice_bin))

    t0 = time.perf_counter()
    proc = run_simulator_netlist(spice_bin, phase_netlist, timeout=args.timeout)
    phase_wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-3000:] or proc.stdout[-3000:])
    if phase_output_mode in {"measure", "control_measure"}:
        measured_text = proc.stdout + "\n" + proc.stderr
        try:
            vals = parse_measured_vector(measured_text, n_vec)
        except ValueError:
            if not is_xyce(spice_bin):
                raise
            vals = read_xyce_print_last_row(xyce_prn_path(phase_netlist), n_vec)
        probe_vals = parse_probe_measurements(measured_text, probe_updates, n_vec) if probe_updates else {}
    elif phase_output_mode == "print":
        xyce_rows = read_xyce_print_rows(xyce_prn_path(phase_netlist), n_vec)
        vals = xyce_rows[-1][1]
        if probe_updates:
            phases, _sample_starts, _t_stop = make_phase_schedule(
                args.batch_size,
                args.updates,
                args.phase,
                args.gap,
                args.update_mode == "direct",
            )
            measure_time = max(0.0, t_stop - args.transient_step)
            probe_times = probe_measure_times(
                phases["acc" if args.update_mode == "direct" else "clear"],
                probe_updates,
                args.gap,
                t_stop,
                measure_time,
            )
            probe_vals = xyce_print_vectors_at_times(xyce_rows, probe_times)
        else:
            probe_vals = {}
    else:
        vals = read_wrdata_row(phase_data, n_vec)
        probe_vals = {}
    phase_w, phase_hb, phase_readout, phase_ob, phase_y = unpack_state(
        vals,
        w,
        hb,
        readout,
        output_bias,
        include_output_bias_vectors,
        include_y_vectors,
    )

    t1 = time.perf_counter()
    op_w = op_hb = op_readout = op_ob = None
    op_probe_states: dict[int, TrainState] = {}
    if args.reference_mode == "spice":
        op_w, op_hb, op_readout, op_ob = w.copy(), hb.copy(), readout.copy(), output_bias.copy()
        for update in range(args.updates):
            start = update * args.batch_size
            stop = start + args.batch_size
            op_w, op_hb, op_readout, op_ob = run_train_batch(
                spice_bin,
                op_netlist,
                op_data,
                x_batch[start:stop],
                y_batch[start:stop],
                op_w,
                op_hb,
                op_readout,
                op_ob,
                blocks,
                args.lr,
                args.timeout,
                linear_output=args.linear_output,
                softmax_output=args.softmax_output,
                local_activation=args.local_activation,
                relu_clip=args.relu_clip,
                activation_derivative=args.activation_derivative,
                derivative_floor=args.derivative_floor,
                derivative_gate_threshold=args.derivative_gate_threshold,
                readout_feedback_mode=args.readout_feedback_mode,
                readout_feedback_clip=args.readout_feedback_clip,
                relu_leak=args.relu_leak,
                softplus_beta=args.softplus_beta,
                hidden_synapse_mode=args.hidden_synapse_mode,
                readout_synapse_mode=args.readout_synapse_mode,
                synapse_clip=args.synapse_clip,
                readout_class_centering=args.readout_class_centering,
                softmax_negative_scale=args.softmax_negative_scale,
                softmax_error_centering=args.softmax_error_centering,
                softmax_temperature=args.softmax_temperature,
                softmax_competition_mode=args.softmax_competition_mode,
                softmax_competitor_power=args.softmax_competitor_power,
                softmax_error_gate=args.softmax_error_gate,
                softmax_margin=args.softmax_margin,
                state_decay=args.state_decay,
            )
            if update + 1 in probe_updates:
                op_probe_states[update + 1] = (op_w.copy(), op_hb.copy(), op_readout.copy(), op_ob.copy())
    op_wall = time.perf_counter() - t1
    if args.reference_mode == "spice":
        assert op_w is not None and op_hb is not None and op_readout is not None and op_ob is not None
        metrics = state_metrics((op_w, op_hb, op_readout, op_ob), (phase_w, phase_hb, phase_readout, phase_ob))
        metrics.update(
            update_direction_metrics(
                (w, hb, readout, output_bias),
                (op_w, op_hb, op_readout, op_ob),
                (phase_w, phase_hb, phase_readout, phase_ob),
            )
        )
    else:
        metrics = empty_reference_metrics()
        metrics.update(
            phase_only_update_metrics(
                (w, hb, readout, output_bias),
                (phase_w, phase_hb, phase_readout, phase_ob),
            )
        )
    probe_rows, probe_phase_states = probe_diagnostic_rows(
        probe_updates,
        probe_vals,
        (w, hb, readout, output_bias),
        op_probe_states,
        include_output_bias_vectors,
        include_y_vectors,
    )
    phase_eval_accuracy = None
    op_reference_eval_accuracy = None
    initial_eval_accuracy = None
    spice_initial_eval_accuracy = None
    spice_phase_eval_accuracy = None
    spice_op_reference_eval_accuracy = None
    numpy_initial_eval_accuracy = None
    numpy_phase_eval_accuracy = None
    numpy_op_reference_eval_accuracy = None
    initial_numpy_eval_diagnostics = None
    phase_numpy_eval_diagnostics = None
    op_reference_numpy_eval_diagnostics = None
    initial_eval_backend_abs_diff = None
    phase_eval_backend_abs_diff = None
    op_reference_eval_backend_abs_diff = None
    eval_wall = 0.0
    if args.eval_samples > 0:
        t2 = time.perf_counter()
        initial_eval_netlist = generated / f"{stem}_initial_eval.cir"
        phase_eval_netlist = generated / f"{stem}_phase_eval.cir"
        if args.eval_backend in {"spice", "both"}:
            owned_netlists.extend([initial_eval_netlist, phase_eval_netlist])
        eval_kwargs = {
            "linear_output": args.linear_output,
            "softmax_output": args.softmax_output,
            "local_activation": args.local_activation,
            "relu_clip": args.relu_clip,
            "relu_leak": args.relu_leak,
            "softplus_beta": args.softplus_beta,
            "hidden_synapse_mode": args.hidden_synapse_mode,
            "readout_synapse_mode": args.readout_synapse_mode,
            "synapse_clip": args.synapse_clip,
            "softmax_temperature": args.softmax_temperature,
            "readout_class_centering": args.readout_class_centering,
        }
        initial_evals = diagnostic_eval_accuracies(
            args.eval_backend,
            spice_bin,
            initial_eval_netlist,
            results / f"{stem}_initial_eval.dat",
            x_test[: args.eval_samples],
            y_test[: args.eval_samples],
            (w, hb, readout, output_bias),
            blocks,
            max(1, min(args.eval_samples, 50)),
            args.timeout,
            **eval_kwargs,
        )
        initial_eval_accuracy = primary_eval_accuracy(args.eval_backend, initial_evals)
        spice_initial_eval_accuracy = initial_evals.get("spice")
        numpy_initial_eval_accuracy = initial_evals.get("numpy")
        initial_eval_backend_abs_diff = eval_backend_abs_diff(initial_evals)
        if args.eval_backend in {"numpy", "both"}:
            initial_numpy_eval_diagnostics = numpy_eval_diagnostics(
                x_test[: args.eval_samples],
                y_test[: args.eval_samples],
                (w, hb, readout, output_bias),
                blocks,
                max(1, min(args.eval_samples, 50)),
                **eval_kwargs,
            )
        phase_evals = diagnostic_eval_accuracies(
            args.eval_backend,
            spice_bin,
            phase_eval_netlist,
            results / f"{stem}_phase_eval.dat",
            x_test[: args.eval_samples],
            y_test[: args.eval_samples],
            (phase_w, phase_hb, phase_readout, phase_ob),
            blocks,
            max(1, min(args.eval_samples, 50)),
            args.timeout,
            **eval_kwargs,
        )
        phase_eval_accuracy = primary_eval_accuracy(args.eval_backend, phase_evals)
        spice_phase_eval_accuracy = phase_evals.get("spice")
        numpy_phase_eval_accuracy = phase_evals.get("numpy")
        phase_eval_backend_abs_diff = eval_backend_abs_diff(phase_evals)
        if args.eval_backend in {"numpy", "both"}:
            phase_numpy_eval_diagnostics = numpy_eval_diagnostics(
                x_test[: args.eval_samples],
                y_test[: args.eval_samples],
                (phase_w, phase_hb, phase_readout, phase_ob),
                blocks,
                max(1, min(args.eval_samples, 50)),
                **eval_kwargs,
            )
        if args.reference_mode == "spice":
            assert op_w is not None and op_hb is not None and op_readout is not None and op_ob is not None
            op_reference_eval_netlist = generated / f"{stem}_op_reference_eval.cir"
            if args.eval_backend in {"spice", "both"}:
                owned_netlists.append(op_reference_eval_netlist)
            op_reference_evals = diagnostic_eval_accuracies(
                args.eval_backend,
                spice_bin,
                op_reference_eval_netlist,
                results / f"{stem}_op_reference_eval.dat",
                x_test[: args.eval_samples],
                y_test[: args.eval_samples],
                (op_w, op_hb, op_readout, op_ob),
                blocks,
                max(1, min(args.eval_samples, 50)),
                args.timeout,
                **eval_kwargs,
            )
            op_reference_eval_accuracy = primary_eval_accuracy(args.eval_backend, op_reference_evals)
            spice_op_reference_eval_accuracy = op_reference_evals.get("spice")
            numpy_op_reference_eval_accuracy = op_reference_evals.get("numpy")
            op_reference_eval_backend_abs_diff = eval_backend_abs_diff(op_reference_evals)
            if args.eval_backend in {"numpy", "both"}:
                op_reference_numpy_eval_diagnostics = numpy_eval_diagnostics(
                    x_test[: args.eval_samples],
                    y_test[: args.eval_samples],
                    (op_w, op_hb, op_readout, op_ob),
                    blocks,
                    max(1, min(args.eval_samples, 50)),
                    **eval_kwargs,
                )
        if args.eval_probe_updates:
            for row in probe_rows:
                update = int(row["update"])
                probe_w, probe_hb, probe_readout, probe_ob = probe_phase_states[update]
                phase_probe_eval_netlist = generated / f"{stem}_probe_{update}_phase_eval.cir"
                if args.eval_backend in {"spice", "both"}:
                    owned_netlists.append(phase_probe_eval_netlist)
                phase_probe_evals = diagnostic_eval_accuracies(
                    args.eval_backend,
                    spice_bin,
                    phase_probe_eval_netlist,
                    results / f"{stem}_probe_{update}_phase_eval.dat",
                    x_test[: args.eval_samples],
                    y_test[: args.eval_samples],
                    (probe_w, probe_hb, probe_readout, probe_ob),
                    blocks,
                    max(1, min(args.eval_samples, 50)),
                    args.timeout,
                    **eval_kwargs,
                )
                phase_probe_eval = primary_eval_accuracy(args.eval_backend, phase_probe_evals)
                row["phase_eval_accuracy"] = phase_probe_eval
                row["phase_spice_eval_accuracy"] = phase_probe_evals.get("spice")
                row["phase_numpy_eval_accuracy"] = phase_probe_evals.get("numpy")
                row["phase_eval_backend_abs_diff"] = eval_backend_abs_diff(phase_probe_evals)
                if args.eval_backend in {"numpy", "both"}:
                    phase_probe_stats = numpy_eval_diagnostics(
                        x_test[: args.eval_samples],
                        y_test[: args.eval_samples],
                        (probe_w, probe_hb, probe_readout, probe_ob),
                        blocks,
                        max(1, min(args.eval_samples, 50)),
                        **eval_kwargs,
                    )
                    row["phase_numpy_eval_dominant_pred_class"] = phase_probe_stats["dominant_pred_class"]
                    row["phase_numpy_eval_dominant_pred_fraction"] = phase_probe_stats["dominant_pred_fraction"]
                    row["phase_numpy_eval_unique_predicted_classes"] = phase_probe_stats["unique_predicted_classes"]
                    row["phase_numpy_eval_prediction_histogram"] = phase_probe_stats["prediction_histogram"]
                row["phase_eval_improvement"] = (
                    phase_probe_eval - initial_eval_accuracy
                    if initial_eval_accuracy is not None
                    else None
                )
                op_probe_state = op_probe_states.get(update)
                if op_probe_state is None:
                    row["op_reference_eval_accuracy"] = None
                    row["eval_accuracy_abs_diff"] = None
                else:
                    op_probe_w, op_probe_hb, op_probe_readout, op_probe_ob = op_probe_state
                    op_probe_eval_netlist = generated / f"{stem}_probe_{update}_op_reference_eval.cir"
                    if args.eval_backend in {"spice", "both"}:
                        owned_netlists.append(op_probe_eval_netlist)
                    op_probe_evals = diagnostic_eval_accuracies(
                        args.eval_backend,
                        spice_bin,
                        op_probe_eval_netlist,
                        results / f"{stem}_probe_{update}_op_reference_eval.dat",
                        x_test[: args.eval_samples],
                        y_test[: args.eval_samples],
                        (op_probe_w, op_probe_hb, op_probe_readout, op_probe_ob),
                        blocks,
                        max(1, min(args.eval_samples, 50)),
                        args.timeout,
                        **eval_kwargs,
                    )
                    op_probe_eval = primary_eval_accuracy(args.eval_backend, op_probe_evals)
                    row["op_reference_eval_accuracy"] = op_probe_eval
                    row["op_reference_spice_eval_accuracy"] = op_probe_evals.get("spice")
                    row["op_reference_numpy_eval_accuracy"] = op_probe_evals.get("numpy")
                    row["op_reference_eval_backend_abs_diff"] = eval_backend_abs_diff(op_probe_evals)
                    row["eval_accuracy_abs_diff"] = abs(phase_probe_eval - op_probe_eval)
        eval_wall = time.perf_counter() - t2
    eval_accuracy_abs_diff = (
        abs(phase_eval_accuracy - op_reference_eval_accuracy)
        if phase_eval_accuracy is not None and op_reference_eval_accuracy is not None
        else None
    )
    direction_matches_reference = (
        None
        if args.reference_mode == "none"
        else bool(
            np.isfinite(metrics["state_update_direction_cosine"])
            and metrics["state_update_direction_cosine"] >= args.direction_cosine_threshold
            and np.isfinite(metrics["state_update_sign_alignment_fraction"])
            and metrics["state_update_sign_alignment_fraction"] >= args.sign_alignment_threshold
        )
    )
    eval_matches_reference = (
        None
        if eval_accuracy_abs_diff is None
        else bool(eval_accuracy_abs_diff <= args.eval_accuracy_diff_threshold)
    )
    phase_eval_improvement = (
        phase_eval_accuracy - initial_eval_accuracy
        if phase_eval_accuracy is not None and initial_eval_accuracy is not None
        else None
    )
    nontrivial_learning_met = (
        None
        if phase_eval_accuracy is None or phase_eval_improvement is None
        else bool(
            phase_eval_accuracy > args.random_accuracy_threshold
            and phase_eval_improvement >= args.learning_improvement_threshold
        )
    )
    final_weights_path = results / f"{stem}_final_weights.npz"
    reference_weights_path = results / f"{stem}_op_reference_final_weights.npz"
    metrics_path = results / f"{stem}_equivalence_metrics.csv"
    probe_metrics_path = results / f"{stem}_probe_metrics.csv"
    np.savez_compressed(
        final_weights_path,
        local_weights=phase_w,
        local_bias=phase_hb,
        readout=phase_readout,
        output_bias=phase_ob,
        y=phase_y,
    )
    if args.reference_mode == "spice":
        assert op_w is not None and op_hb is not None and op_readout is not None and op_ob is not None
        np.savez_compressed(
            reference_weights_path,
            local_weights=op_w,
            local_bias=op_hb,
            readout=op_readout,
            output_bias=op_ob,
        )
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    if probe_rows:
        pd.DataFrame(probe_rows).to_csv(probe_metrics_path, index=False)
    simulator_sidecars_cleaned = cleanup_simulator_sidecars(owned_netlists)
    state_descriptions = phase_state_descriptions(
        args.update_mode,
        output_bias_state_frozen,
        args.output_delta_mode,
        args.hidden_activation_mode,
        args.score_state_mode,
    )
    execution_contract = phase_execution_contract_fields(
        args.batch_size,
        args.reference_mode,
        args.init_weights,
        args.strict_fully_on_device,
    )
    probe_summary = summarize_probe_rows(probe_rows)
    cost_fields = phase_cost_summary_fields(
        updates=args.updates,
        estimated_transient_points=estimated_transient_points,
        phase_output_vector_count=n_vec,
        auxiliary_algebraic_source_count=preflight_auxiliary_algebraic_source_count,
        source_complexity=source_complexity,
        max_transient_points=args.max_transient_points,
        max_source_pwl_points=args.max_source_pwl_points,
        max_sample_sources=args.max_sample_sources,
        max_total_sources=args.max_total_sources,
        max_output_vectors=args.max_output_vectors,
        max_auxiliary_algebraic_sources=args.max_auxiliary_algebraic_sources,
    )
    summary = {
        "simulator": version,
        "simulator_selector": args.simulator,
        "architecture": "phase_resolved_transient_local_feature_readout",
        "status": (
            ("one_batch_equivalence_check" if args.updates == 1 else "multi_update_equivalence_check")
            if args.reference_mode == "spice"
            else "continuous_phase_train_no_reference"
        ),
        "image_size": args.image_size,
        "block_size": args.block_size,
        "stride": stride,
        "blocks": len(blocks),
        "channels": args.channels,
        "classes": 10,
        "mnist_index_order": "stable_permutation_prefix",
        "train_index_metadata": index_prefix_metadata(train_indices),
        "eval_index_metadata": index_prefix_metadata(eval_indices),
        "train_label_metadata": label_sequence_metadata(y_batch, 10),
        "train_samples": args.train_samples,
        "eval_samples": args.eval_samples,
        "batch_size": args.batch_size,
        "updates": args.updates,
        "probe_updates": list(probe_updates),
        "eval_probe_updates": bool(args.eval_probe_updates),
        "eval_backend": args.eval_backend,
        "total_samples": total_samples,
        "lr": args.lr,
        "lr_schedule": args.lr_schedule,
        "lr_final_scale": args.lr_final_scale,
        "linear_output": bool(args.linear_output),
        "softmax_output": bool(args.softmax_output),
        "local_activation": args.local_activation,
        "relu_clip": args.relu_clip,
        "relu_leak": args.relu_leak,
        "softplus_beta": args.softplus_beta,
        "activation_derivative": args.activation_derivative,
        "derivative_floor": args.derivative_floor,
        "derivative_gate_threshold": args.derivative_gate_threshold,
        "readout_feedback_mode": args.readout_feedback_mode,
        "readout_feedback_clip": args.readout_feedback_clip,
        "output_bias_update_scale": args.output_bias_update_scale,
        "readout_update_scale": args.readout_update_scale,
        "local_update_scale": args.local_update_scale,
        "state_decay": args.state_decay,
        "softmax_negative_scale": args.softmax_negative_scale,
        "softmax_error_centering": args.softmax_error_centering,
        "softmax_temperature": args.softmax_temperature,
        "softmax_competition_mode": args.softmax_competition_mode,
        "softmax_competitor_power": args.softmax_competitor_power,
        "softmax_error_gate": args.softmax_error_gate,
        "softmax_margin": args.softmax_margin,
        "hidden_synapse_mode": args.hidden_synapse_mode,
        "readout_synapse_mode": args.readout_synapse_mode,
        "synapse_clip": args.synapse_clip,
        "readout_class_centering": args.readout_class_centering,
        "reference_mode": args.reference_mode,
        "phase_output_mode_requested": args.phase_output_mode,
        "update_mode": args.update_mode,
        **phase_deck_mode_fields(
            phase_clock_mode=args.phase_clock_mode,
            target_source_mode=args.target_source_mode,
            sample_edge=sample_edge,
            hidden_preactivation_mode=args.hidden_preactivation_mode,
            hidden_preactivation_source_count=preflight_hidden_preactivation_source_count,
            hidden_activation_mode=args.hidden_activation_mode,
            hidden_activation_state_count=preflight_hidden_activation_state_count,
            hidden_delta_state_count=preflight_hidden_delta_state_count,
            score_state_mode=args.score_state_mode,
            score_state_count=preflight_score_state_count,
            gradient_accumulator_state_count=preflight_gradient_accumulator_state_count,
            temporary_state_count=preflight_temporary_state_count,
            score_calculation_mode=args.score_calculation_mode,
            score_calculation_source_count=preflight_score_calculation_source_count,
            output_rail_mode=args.output_rail_mode,
            output_rail_source_count=preflight_output_rail_source_count,
            output_delta_mode=args.output_delta_mode,
            output_delta_state_count=preflight_output_delta_state_count,
        ),
        "output_bias_state_frozen": output_bias_state_frozen,
        "phase_output_includes_y": include_y_vectors,
        "simulator_extra_args": args.simulator_extra_args or os.environ.get(SPICE_SIMULATOR_ARGS_ENV, ""),
        "init_weights": args.init_weights,
        "phase_netlist": str(phase_netlist),
        "phase_data": str(phase_data),
        "op_reference_netlist": str(op_netlist) if args.reference_mode == "spice" else None,
        "op_reference_data": str(op_data) if args.reference_mode == "spice" else None,
        "final_weights": str(final_weights_path),
        "op_reference_final_weights": str(reference_weights_path) if args.reference_mode == "spice" else None,
        "equivalence_metrics": str(metrics_path),
        "probe_metrics": str(probe_metrics_path) if probe_rows else None,
        **probe_summary,
        "simulator_sidecars_cleaned": simulator_sidecars_cleaned,
        "phase_wall_time_s": phase_wall,
        "op_reference_wall_time_s": op_wall,
        "eval_wall_time_s": eval_wall,
        "initial_eval_accuracy": initial_eval_accuracy,
        "phase_eval_accuracy": phase_eval_accuracy,
        "op_reference_eval_accuracy": op_reference_eval_accuracy,
        "spice_initial_eval_accuracy": spice_initial_eval_accuracy,
        "spice_phase_eval_accuracy": spice_phase_eval_accuracy,
        "spice_op_reference_eval_accuracy": spice_op_reference_eval_accuracy,
        "numpy_initial_eval_accuracy": numpy_initial_eval_accuracy,
        "numpy_phase_eval_accuracy": numpy_phase_eval_accuracy,
        "numpy_op_reference_eval_accuracy": numpy_op_reference_eval_accuracy,
        "initial_numpy_eval_diagnostics": initial_numpy_eval_diagnostics,
        "phase_numpy_eval_diagnostics": phase_numpy_eval_diagnostics,
        "op_reference_numpy_eval_diagnostics": op_reference_numpy_eval_diagnostics,
        "initial_eval_backend_abs_diff": initial_eval_backend_abs_diff,
        "phase_eval_backend_abs_diff": phase_eval_backend_abs_diff,
        "op_reference_eval_backend_abs_diff": op_reference_eval_backend_abs_diff,
        "eval_accuracy_abs_diff": eval_accuracy_abs_diff,
        "phase_eval_improvement": phase_eval_improvement,
        "direction_cosine_threshold": args.direction_cosine_threshold,
        "sign_alignment_threshold": args.sign_alignment_threshold,
        "eval_accuracy_diff_threshold": args.eval_accuracy_diff_threshold,
        "random_accuracy_threshold": args.random_accuracy_threshold,
        "learning_improvement_threshold": args.learning_improvement_threshold,
        "direction_matches_batch_op_reference": direction_matches_reference,
        "eval_accuracy_matches_batch_op_reference": eval_matches_reference,
        "nontrivial_learning_met": nontrivial_learning_met,
        "online_batch_size_one": args.batch_size == 1,
        **execution_contract,
        "continuous_transient_contract_met": (
            None
            if args.reference_mode == "none"
            else bool(args.batch_size == 1 and direction_matches_reference and (eval_matches_reference is not False))
        ),
        "t_stop_s": t_stop,
        "transient_step_s": args.transient_step,
        "estimated_transient_points": estimated_transient_points,
        "max_transient_points": args.max_transient_points,
        "phase_output_vector_count": n_vec,
        "max_output_vectors": args.max_output_vectors,
        "max_source_pwl_points": args.max_source_pwl_points,
        "max_sample_sources": args.max_sample_sources,
        "max_total_sources": args.max_total_sources,
        "max_auxiliary_algebraic_sources": args.max_auxiliary_algebraic_sources,
        **source_complexity,
        **cost_fields,
        "phase_s": args.phase,
        "settle_ratio": args.settle_ratio,
        "output_mode": phase_output_mode,
        **state_descriptions,
        "python_role": "Python generates guiding waveforms and compares final state; it does not carry training state during the transient run.",
        "note": (
            "Local-feature phase-transient all-SPICE update run. reference_mode=spice compares against the existing "
            "operating-point SPICE update; reference_mode=none skips that replay for longer final-diagnostic runs."
        ),
        **metrics,
    }
    summary_path = results / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
