from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

import run_spice_mnist_local_feature_fast_online_reference as fast_ref
from run_spice_mnist_local_block_batch_op_train import block_indices
from run_spice_mnist_local_feature_phase_transient import (
    estimate_transient_points,
    hidden_preactivation_source_count,
    lr_schedule_values,
    make_phase_schedule,
    phase_output_vector_count,
    phase_source_complexity,
    score_calculation_source_count,
    target_matrix,
)
from run_spice_mnist_local_feature_phase_variant_sweep import activation_clip_pairs, parse_csv, parse_float_csv, variant_tag
from run_spice_mnist_train import load_mnist_sequence
from run_spice_sweep import ROOT

DEFAULT_PROMOTION_PHASE_CLOCK_MODE = "pwl"
DEFAULT_PROMOTION_SAMPLE_EDGE = 0.0
DEFAULT_PROMOTION_HIDDEN_PREACTIVATION_MODE = "inline"
DEFAULT_PROMOTION_SCORE_CALCULATION_MODE = "inline"

TrainState = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
PROBE_ACCURACY_RE = re.compile(r"^probe_eval_accuracy_u(\d+)$")


def float_axis(args: argparse.Namespace, plural_name: str, single_name: str, flag_name: str) -> list[float]:
    raw = getattr(args, plural_name, "")
    values = parse_float_csv(raw) if raw else [float(getattr(args, single_name))]
    if not values:
        raise ValueError(f"--{flag_name} must not be empty")
    return values


def lr_schedule_scale_axis(args: argparse.Namespace) -> list[tuple[str, float]]:
    raw = getattr(args, "lr_schedules", "")
    schedules = parse_csv(raw) if raw else [args.lr_schedule]
    if not schedules:
        raise ValueError("--lr-schedules must not be empty")
    invalid = sorted(set(schedules) - {"constant", "linear-decay"})
    if invalid:
        raise ValueError(f"unsupported --lr-schedules values: {', '.join(invalid)}")
    final_scales = float_axis(args, "lr_final_scales", "lr_final_scale", "lr-final-scales")
    if any(scale < 0 for scale in final_scales):
        raise ValueError("--lr-final-scale/--lr-final-scales values must be non-negative")

    pairs: list[tuple[str, float]] = []
    for schedule in schedules:
        if schedule == "constant":
            pairs.append((schedule, 1.0))
        else:
            pairs.extend((schedule, scale) for scale in final_scales)
    return pairs


def hparam_tag(
    base_tag: str,
    lr: float,
    lr_schedule: str,
    lr_final_scale: float,
    output_bias_update_scale: float,
    readout_update_scale: float,
    local_update_scale: float,
    state_decay: float,
    softmax_temperature: float,
) -> str:
    return fast_ref.sanitize_tag(
        f"{base_tag}_lr{lr:g}_{lr_schedule}_lrfs{lr_final_scale:g}_obs{output_bias_update_scale:g}_rs{readout_update_scale:g}"
        f"_ls{local_update_scale:g}_decay{state_decay:g}_temp{softmax_temperature:g}"
    )


def variant_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    activations = parse_csv(args.activations)
    relu_clips = parse_float_csv(args.relu_clips)
    derivative_modes = parse_csv(args.derivative_modes)
    feedback_modes = parse_csv(args.feedback_modes)
    hidden_synapse_modes = parse_csv(args.hidden_synapse_modes)
    readout_synapse_modes = parse_csv(args.readout_synapse_modes)
    synapse_clips = parse_float_csv(args.synapse_clips) if args.synapse_clips else [args.synapse_clip]
    lrs = float_axis(args, "lrs", "lr", "lrs")
    lr_schedule_scales = lr_schedule_scale_axis(args)
    output_bias_update_scales = float_axis(
        args,
        "output_bias_update_scales",
        "output_bias_update_scale",
        "output-bias-update-scales",
    )
    readout_update_scales = float_axis(args, "readout_update_scales", "readout_update_scale", "readout-update-scales")
    local_update_scales = float_axis(args, "local_update_scales", "local_update_scale", "local-update-scales")
    state_decays = float_axis(args, "state_decays", "state_decay", "state-decays")
    softmax_temperatures = float_axis(args, "softmax_temperatures", "softmax_temperature", "softmax-temperatures")
    if not activations:
        raise ValueError("--activations must not be empty")
    if not relu_clips:
        raise ValueError("--relu-clips must not be empty")
    if not derivative_modes:
        raise ValueError("--derivative-modes must not be empty")
    if not feedback_modes:
        raise ValueError("--feedback-modes must not be empty")
    if not hidden_synapse_modes:
        raise ValueError("--hidden-synapse-modes must not be empty")
    if not readout_synapse_modes:
        raise ValueError("--readout-synapse-modes must not be empty")
    if any(clip <= 0 for clip in synapse_clips):
        raise ValueError("--synapse-clip/--synapse-clips values must be positive")
    if any(lr < 0 for lr in lrs):
        raise ValueError("--lr/--lrs values must be non-negative")
    if any(scale < 0 for scale in output_bias_update_scales):
        raise ValueError("--output-bias-update-scale/--output-bias-update-scales values must be non-negative")
    if any(scale < 0 for scale in readout_update_scales):
        raise ValueError("--readout-update-scale/--readout-update-scales values must be non-negative")
    if any(scale < 0 for scale in local_update_scales):
        raise ValueError("--local-update-scale/--local-update-scales values must be non-negative")
    if any(decay < 0 or decay >= 1 for decay in state_decays):
        raise ValueError("--state-decay/--state-decays values must be in [0, 1)")
    if any(temp <= 0 for temp in softmax_temperatures):
        raise ValueError("--softmax-temperature/--softmax-temperatures values must be positive")

    rows: list[dict[str, Any]] = []
    for activation, relu_clip in activation_clip_pairs(activations, relu_clips):
        for derivative_mode in derivative_modes:
            for feedback_mode in feedback_modes:
                for hidden_synapse_mode in hidden_synapse_modes:
                    for readout_synapse_mode in readout_synapse_modes:
                        for synapse_clip in synapse_clips:
                            family_tag = variant_tag(
                                args.tag,
                                activation,
                                relu_clip,
                                derivative_mode,
                                feedback_mode,
                                hidden_synapse_mode,
                                readout_synapse_mode,
                                synapse_clip,
                            )
                            for lr in lrs:
                                for lr_schedule, lr_final_scale in lr_schedule_scales:
                                    for output_bias_update_scale in output_bias_update_scales:
                                        for readout_update_scale in readout_update_scales:
                                            for local_update_scale in local_update_scales:
                                                for state_decay in state_decays:
                                                    for softmax_temperature in softmax_temperatures:
                                                        rows.append(
                                                            {
                                                                "local_activation": activation,
                                                                "relu_clip": relu_clip,
                                                                "activation_derivative": derivative_mode,
                                                                "readout_feedback_mode": feedback_mode,
                                                                "hidden_synapse_mode": hidden_synapse_mode,
                                                                "readout_synapse_mode": readout_synapse_mode,
                                                                "synapse_clip": synapse_clip,
                                                                "lr": lr,
                                                                "lr_schedule": lr_schedule,
                                                                "lr_final_scale": lr_final_scale,
                                                                "output_bias_update_scale": output_bias_update_scale,
                                                                "readout_update_scale": readout_update_scale,
                                                                "local_update_scale": local_update_scale,
                                                                "state_decay": state_decay,
                                                                "softmax_temperature": softmax_temperature,
                                                                "tag": hparam_tag(
                                                                    family_tag,
                                                                    lr,
                                                                    lr_schedule,
                                                                    lr_final_scale,
                                                                    output_bias_update_scale,
                                                                    readout_update_scale,
                                                                    local_update_scale,
                                                                    state_decay,
                                                                    softmax_temperature,
                                                                ),
                                                            }
                                                        )
    return rows


def forward_kwargs(args: argparse.Namespace, variant: dict[str, Any]) -> dict[str, Any]:
    return {
        "local_activation": variant["local_activation"],
        "relu_clip": variant["relu_clip"],
        "relu_leak": args.relu_leak,
        "softplus_beta": args.softplus_beta,
        "hidden_synapse_mode": variant["hidden_synapse_mode"],
        "readout_synapse_mode": variant["readout_synapse_mode"],
        "synapse_clip": variant["synapse_clip"],
        "linear_output": args.linear_output,
        "softmax_output": args.softmax_output,
        "softmax_temperature": variant["softmax_temperature"],
        "readout_class_centering": args.readout_class_centering,
    }


def update_kwargs(args: argparse.Namespace, variant: dict[str, Any]) -> dict[str, Any]:
    return {
        "lr": variant["lr"],
        "lr_schedule": variant.get("lr_schedule", "constant"),
        "lr_final_scale": variant.get("lr_final_scale", 1.0),
        "activation_derivative": variant["activation_derivative"],
        "derivative_floor": args.derivative_floor,
        "derivative_gate_threshold": args.derivative_gate_threshold,
        "readout_feedback_mode": variant["readout_feedback_mode"],
        "readout_feedback_clip": args.readout_feedback_clip,
        "output_bias_update_scale": variant["output_bias_update_scale"],
        "readout_update_scale": variant["readout_update_scale"],
        "local_update_scale": variant["local_update_scale"],
        "state_decay": variant["state_decay"],
        "softmax_negative_scale": args.softmax_negative_scale,
        "softmax_error_centering": args.softmax_error_centering,
        "softmax_competition_mode": args.softmax_competition_mode,
        "softmax_competitor_power": args.softmax_competitor_power,
        "softmax_error_gate": args.softmax_error_gate,
        "softmax_margin": args.softmax_margin,
        **forward_kwargs(args, variant),
    }


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def parse_accuracy_thresholds(raw: str) -> list[float]:
    if not raw:
        return []
    thresholds = parse_float_csv(raw)
    if any(threshold < 0.0 or threshold > 1.0 for threshold in thresholds):
        raise ValueError("--learning-thresholds values must be in [0, 1]")
    return thresholds


def probe_accuracy_items(row: dict[str, Any]) -> list[tuple[int, float]]:
    items: list[tuple[int, float]] = []
    for key, value in row.items():
        match = PROBE_ACCURACY_RE.match(key)
        if match is None or value is None:
            continue
        items.append((int(match.group(1)), float(value)))
    return sorted(items)


def compact_probe_variant(row: dict[str, Any], update: int, accuracy: float) -> dict[str, Any]:
    initial = row.get("initial_eval_accuracy")
    return {
        "probe_update": update,
        "probe_eval_accuracy": accuracy,
        "probe_eval_improvement": accuracy - float(initial) if initial is not None else None,
        "tag": row.get("tag"),
        "local_activation": row.get("local_activation"),
        "relu_clip": row.get("relu_clip"),
        "activation_derivative": row.get("activation_derivative"),
        "readout_feedback_mode": row.get("readout_feedback_mode"),
        "hidden_synapse_mode": row.get("hidden_synapse_mode"),
        "readout_synapse_mode": row.get("readout_synapse_mode"),
        "synapse_clip": row.get("synapse_clip"),
        "lr": row.get("lr"),
        "lr_schedule": row.get("lr_schedule", "constant"),
        "lr_final_scale": row.get("lr_final_scale", 1.0),
        "output_bias_update_scale": row.get("output_bias_update_scale"),
        "readout_update_scale": row.get("readout_update_scale"),
        "local_update_scale": row.get("local_update_scale"),
        "state_decay": row.get("state_decay"),
        "softmax_temperature": row.get("softmax_temperature"),
        "initial_eval_accuracy": row.get("initial_eval_accuracy"),
        "final_eval_accuracy": row.get("final_eval_accuracy"),
        "best_probe_update": row.get("best_probe_update"),
        "best_probe_eval_accuracy": row.get("best_probe_eval_accuracy"),
    }


def best_probe_variants_by_update(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_update: dict[int, list[tuple[float, dict[str, Any]]]] = {}
    for row in rows:
        for update, accuracy in probe_accuracy_items(row):
            by_update.setdefault(update, []).append((accuracy, row))
    best: list[dict[str, Any]] = []
    for update in sorted(by_update):
        accuracy, row = max(
            by_update[update],
            key=lambda item: (
                item[0],
                item[1].get("eval_improvement") if item[1].get("eval_improvement") is not None else -999.0,
                item[1].get("final_eval_accuracy") if item[1].get("final_eval_accuracy") is not None else -1.0,
            ),
        )
        best.append(compact_probe_variant(row, update, accuracy))
    return best


def learning_threshold_hits(
    probe_best: list[dict[str, Any]],
    thresholds: list[float],
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    fallback = max(probe_best, key=lambda row: row["probe_eval_accuracy"]) if probe_best else None
    for threshold in thresholds:
        first_hit: dict[str, Any] | None = None
        for row in probe_best:
            accuracy = row.get("probe_eval_accuracy")
            if accuracy is None:
                continue
            if float(accuracy) >= threshold:
                first_hit = row
                break
        if first_hit is not None:
            hits.append({"threshold": threshold, "met": True, **first_hit})
        else:
            if fallback is None:
                hits.append({"threshold": threshold, "met": False, "probe_update": None, "probe_eval_accuracy": None})
            else:
                hits.append(
                    {
                        "threshold": threshold,
                        "met": False,
                        "best_available_probe_update": fallback["probe_update"],
                        "best_available_probe_eval_accuracy": fallback["probe_eval_accuracy"],
                        "best_available_tag": fallback.get("tag"),
                    }
                )
    return hits


def fast_reference_objective_fields(
    *,
    final_eval_accuracy: float | None,
    eval_samples: int,
    full_objective_eval_samples: int,
    full_objective_accuracy: float,
) -> dict[str, Any]:
    return fast_ref.fast_reference_objective_fields(
        final_eval_accuracy,
        eval_samples,
        full_objective_eval_samples,
        full_objective_accuracy,
    )


def best_fast_reference_full_objective_variant(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get("fast_reference_full_objective_candidate") is True]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            row.get("final_eval_accuracy") if row.get("final_eval_accuracy") is not None else -1.0,
            row.get("eval_improvement") if row.get("eval_improvement") is not None else -999.0,
        ),
    )


def cost_projection_budget_met(row: dict[str, Any]) -> bool:
    return (
        row.get("strict_phase_cost_projection_transient_budget_met") is True
        and row.get("strict_phase_cost_projection_source_pwl_budget_met") is True
        and row.get("strict_phase_cost_projection_sample_source_budget_met", True) is True
        and row.get("strict_phase_cost_projection_total_source_budget_met", True) is True
        and row.get("strict_phase_cost_projection_output_vector_budget_met", True) is True
    )


def cost_projection_summary_fields(rows: list[dict[str, Any]], cost_projection_updates: int) -> dict[str, Any]:
    projection_rows = [row for row in rows if row.get("strict_phase_cost_projection_updates") is not None]
    full_objective_rows = [row for row in rows if row.get("fast_reference_full_objective_candidate") is True]
    full_objective_cost_feasible_rows = [row for row in full_objective_rows if cost_projection_budget_met(row)]

    def count_met(selected_rows: list[dict[str, Any]], field: str) -> int:
        return sum(row.get(field, True) is True for row in selected_rows)

    def count_failed(selected_rows: list[dict[str, Any]], field: str) -> int:
        return sum(row.get(field, True) is not True for row in selected_rows)

    return {
        "cost_projection_enabled": int(cost_projection_updates) > 0,
        "cost_projection_updates": int(cost_projection_updates),
        "cost_projection_rows": len(projection_rows),
        "cost_projection_budget_feasible_rows": sum(cost_projection_budget_met(row) for row in projection_rows),
        "cost_projection_transient_budget_feasible_rows": count_met(
            projection_rows, "strict_phase_cost_projection_transient_budget_met"
        ),
        "cost_projection_source_pwl_budget_feasible_rows": count_met(
            projection_rows, "strict_phase_cost_projection_source_pwl_budget_met"
        ),
        "cost_projection_sample_source_budget_feasible_rows": count_met(
            projection_rows, "strict_phase_cost_projection_sample_source_budget_met"
        ),
        "cost_projection_total_source_budget_feasible_rows": count_met(
            projection_rows, "strict_phase_cost_projection_total_source_budget_met"
        ),
        "cost_projection_output_vector_budget_feasible_rows": count_met(
            projection_rows, "strict_phase_cost_projection_output_vector_budget_met"
        ),
        "fast_reference_full_objective_candidate_count": len(full_objective_rows),
        "fast_reference_full_objective_cost_feasible_candidate_count": len(full_objective_cost_feasible_rows),
        "fast_reference_full_objective_cost_infeasible_candidate_count": len(full_objective_rows) - len(full_objective_cost_feasible_rows),
        "fast_reference_full_objective_transient_budget_infeasible_count": count_failed(
            full_objective_rows, "strict_phase_cost_projection_transient_budget_met"
        ),
        "fast_reference_full_objective_source_pwl_budget_infeasible_count": count_failed(
            full_objective_rows, "strict_phase_cost_projection_source_pwl_budget_met"
        ),
        "fast_reference_full_objective_sample_source_budget_infeasible_count": count_failed(
            full_objective_rows, "strict_phase_cost_projection_sample_source_budget_met"
        ),
        "fast_reference_full_objective_total_source_budget_infeasible_count": count_failed(
            full_objective_rows, "strict_phase_cost_projection_total_source_budget_met"
        ),
        "fast_reference_full_objective_output_vector_budget_infeasible_count": count_failed(
            full_objective_rows, "strict_phase_cost_projection_output_vector_budget_met"
        ),
    }


def best_fast_reference_full_objective_cost_feasible_variant(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("fast_reference_full_objective_candidate") is True and cost_projection_budget_met(row)
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            row.get("final_eval_accuracy") if row.get("final_eval_accuracy") is not None else -1.0,
            row.get("eval_improvement") if row.get("eval_improvement") is not None else -999.0,
            -float(row.get("strict_phase_cost_projection_total_source_pwl_points") or 0.0),
        ),
    )


def best_promotion_variant(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get("promotion_probe_eval_accuracy") is not None]
    if not candidates:
        return None
    feasible = [
        row
        for row in candidates
        if promotion_budget_met(row)
    ]
    if feasible:
        candidates = feasible
    return max(candidates, key=lambda row: (row["promotion_probe_eval_accuracy"], row["eval_improvement"]))


def budget_feasible_promotion_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("promotion_probe_eval_accuracy") is not None
        and promotion_budget_met(row)
    ]


def promotion_budget_met(row: dict[str, Any]) -> bool:
    return (
        row.get("strict_phase_promotion_transient_budget_met") is True
        and row.get("strict_phase_promotion_source_pwl_budget_met") is True
        and row.get("strict_phase_promotion_sample_source_budget_met", True) is True
        and row.get("strict_phase_promotion_total_source_budget_met", True) is True
        and row.get("strict_phase_promotion_output_vector_budget_met", True) is True
    )


def best_promotion_efficiency_variant(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in budget_feasible_promotion_rows(rows)
        if row.get("strict_phase_promotion_eval_improvement_per_1k_source_pwl") is not None
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            row["strict_phase_promotion_eval_improvement_per_1k_source_pwl"],
            row.get("promotion_probe_eval_accuracy") or -1.0,
            row.get("promotion_probe_eval_improvement") or -999.0,
        ),
    )


def promotion_timeout_seconds(args: argparse.Namespace, updates: int) -> float:
    configured = float(getattr(args, "promotion_timeout", 0.0))
    if configured > 0.0:
        return configured
    return max(240.0, 1.25 * float(updates))


def promotion_efficiency_fields(
    *,
    initial_eval_accuracy: float,
    promotion_probe_eval_accuracy: float | None,
    promotion_updates: int,
    promotion_costs: dict[str, Any],
) -> dict[str, float | None]:
    if promotion_updates <= 0:
        raise ValueError("promotion_updates must be positive")
    total_source_points = float(promotion_costs.get("strict_phase_promotion_total_source_pwl_points") or 0.0)
    transient_points = float(promotion_costs.get("strict_phase_promotion_estimated_transient_points") or 0.0)
    source_points_per_update = total_source_points / float(promotion_updates) if total_source_points > 0.0 else None
    if promotion_probe_eval_accuracy is None:
        return {
            "promotion_probe_eval_improvement": None,
            "strict_phase_promotion_source_pwl_points_per_update": source_points_per_update,
            "strict_phase_promotion_eval_improvement_per_1k_source_pwl": None,
            "strict_phase_promotion_eval_improvement_per_1k_transient_points": None,
        }
    improvement = float(promotion_probe_eval_accuracy) - float(initial_eval_accuracy)
    return {
        "promotion_probe_eval_improvement": improvement,
        "strict_phase_promotion_source_pwl_points_per_update": source_points_per_update,
        "strict_phase_promotion_eval_improvement_per_1k_source_pwl": (
            1000.0 * improvement / total_source_points if total_source_points > 0.0 else None
        ),
        "strict_phase_promotion_eval_improvement_per_1k_transient_points": (
            1000.0 * improvement / transient_points if transient_points > 0.0 else None
        ),
    }


def strict_phase_promotion_command(args: argparse.Namespace, variant: dict[str, Any]) -> list[str]:
    updates = int(getattr(args, "promotion_updates", 0) or args.train_samples)
    if updates <= 0:
        raise ValueError("--promotion-updates must be positive when set")
    timeout = promotion_timeout_seconds(args, updates)
    promotion_tag = fast_ref.sanitize_tag(f"{getattr(args, 'promotion_tag_prefix', 'promote')}_{variant['tag']}")
    command = [
        sys.executable,
        str(ROOT / "spice/run_spice_mnist_local_feature_phase_transient.py"),
        "--simulator",
        getattr(args, "promotion_simulator", "Xyce"),
        "--train-samples",
        str(updates),
        "--eval-samples",
        str(args.eval_samples),
        "--image-size",
        str(args.image_size),
        "--block-size",
        str(args.block_size),
        "--stride",
        str(args.stride),
        "--channels",
        str(args.channels),
        "--batch-size",
        "1",
        "--updates",
        str(updates),
        "--lr",
        str(variant["lr"]),
        "--lr-schedule",
        variant.get("lr_schedule", "constant"),
        "--lr-final-scale",
        str(variant.get("lr_final_scale", 1.0)),
        "--phase",
        str(getattr(args, "promotion_phase", 0.5e-9)),
        "--gap",
        str(getattr(args, "promotion_gap", 0.05e-9)),
        "--edge",
        str(getattr(args, "promotion_edge", 5e-12)),
        "--settle-ratio",
        str(getattr(args, "promotion_settle_ratio", 20.0)),
        "--transient-step",
        str(getattr(args, "promotion_transient_step", 200e-12)),
        "--timeout",
        str(timeout),
        "--max-transient-points",
        str(getattr(args, "promotion_max_transient_points", 2000)),
        "--max-source-pwl-points",
        str(getattr(args, "promotion_max_source_pwl_points", 0)),
        "--max-sample-sources",
        str(getattr(args, "promotion_max_sample_sources", 0)),
        "--max-total-sources",
        str(getattr(args, "promotion_max_total_sources", 0)),
        "--max-output-vectors",
        str(getattr(args, "promotion_max_output_vectors", 0)),
        "--reference-mode",
        "none",
        "--phase-output-mode",
        "print",
        "--update-mode",
        "direct",
        "--phase-clock-mode",
        getattr(args, "promotion_phase_clock_mode", DEFAULT_PROMOTION_PHASE_CLOCK_MODE),
        "--target-source-mode",
        getattr(args, "promotion_target_source_mode", "label"),
        "--eval-backend",
        "numpy",
        "--local-activation",
        variant["local_activation"],
        "--relu-clip",
        str(variant["relu_clip"]),
        "--relu-leak",
        str(args.relu_leak),
        "--softplus-beta",
        str(args.softplus_beta),
        "--activation-derivative",
        variant["activation_derivative"],
        "--derivative-floor",
        str(args.derivative_floor),
        "--derivative-gate-threshold",
        str(args.derivative_gate_threshold),
        "--readout-feedback-mode",
        variant["readout_feedback_mode"],
        "--readout-feedback-clip",
        str(args.readout_feedback_clip),
        "--output-bias-update-scale",
        str(variant["output_bias_update_scale"]),
        "--readout-update-scale",
        str(variant["readout_update_scale"]),
        "--local-update-scale",
        str(variant["local_update_scale"]),
        "--state-decay",
        str(variant["state_decay"]),
        "--softmax-negative-scale",
        str(args.softmax_negative_scale),
        "--softmax-error-centering",
        args.softmax_error_centering,
        "--softmax-temperature",
        str(variant["softmax_temperature"]),
        "--softmax-competition-mode",
        args.softmax_competition_mode,
        "--softmax-competitor-power",
        str(args.softmax_competitor_power),
        "--softmax-error-gate",
        args.softmax_error_gate,
        "--softmax-margin",
        str(args.softmax_margin),
        "--hidden-synapse-mode",
        variant["hidden_synapse_mode"],
        "--readout-synapse-mode",
        variant["readout_synapse_mode"],
        "--synapse-clip",
        str(variant["synapse_clip"]),
        "--readout-class-centering",
        args.readout_class_centering,
        "--hidden-preactivation-mode",
        getattr(args, "promotion_hidden_preactivation_mode", DEFAULT_PROMOTION_HIDDEN_PREACTIVATION_MODE),
        "--score-calculation-mode",
        getattr(args, "promotion_score_calculation_mode", DEFAULT_PROMOTION_SCORE_CALCULATION_MODE),
        "--tag",
        promotion_tag,
    ]
    if args.softmax_output:
        command.append("--softmax-output")
    if args.linear_output:
        command.append("--linear-output")
    promotion_sample_edge = getattr(args, "promotion_sample_edge", DEFAULT_PROMOTION_SAMPLE_EDGE)
    if promotion_sample_edge is not None:
        command.extend(["--sample-edge", str(promotion_sample_edge)])
    probe_updates = getattr(args, "promotion_probe_updates", "")
    if probe_updates:
        command.extend(["--probe-updates", probe_updates])
    if getattr(args, "promotion_phase_output_include_y", False):
        command.append("--phase-output-include-y")
    command.append("--strict-fully-on-device")
    return command


def strict_phase_promotion_cost_fields(
    args: argparse.Namespace,
    variant: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, Any]:
    updates = int(getattr(args, "promotion_updates", 0) or args.train_samples)
    return strict_phase_cost_fields_for_updates(
        args,
        variant,
        x_train,
        y_train,
        updates=updates,
        prefix="strict_phase_promotion",
    )


def strict_phase_cost_fields_for_updates(
    args: argparse.Namespace,
    variant: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    updates: int,
    prefix: str,
) -> dict[str, Any]:
    if updates <= 0:
        raise ValueError("strict phase cost projection updates must be positive")
    if len(x_train) < updates or len(y_train) < updates:
        raise ValueError("training prefix is shorter than strict phase cost projection updates")
    phase = float(getattr(args, "promotion_phase", 0.5e-9))
    gap = float(getattr(args, "promotion_gap", 0.05e-9))
    edge = float(getattr(args, "promotion_edge", 5e-12))
    sample_edge = getattr(args, "promotion_sample_edge", DEFAULT_PROMOTION_SAMPLE_EDGE)
    if sample_edge is not None:
        sample_edge = float(sample_edge)
    transient_step = float(getattr(args, "promotion_transient_step", 200e-12))
    phase_clock_mode = getattr(args, "promotion_phase_clock_mode", DEFAULT_PROMOTION_PHASE_CLOCK_MODE)
    hidden_preactivation_mode = getattr(
        args,
        "promotion_hidden_preactivation_mode",
        DEFAULT_PROMOTION_HIDDEN_PREACTIVATION_MODE,
    )
    score_calculation_mode = getattr(
        args,
        "promotion_score_calculation_mode",
        DEFAULT_PROMOTION_SCORE_CALCULATION_MODE,
    )
    phases, sample_starts, t_stop = make_phase_schedule(1, updates, phase, gap, True)
    lr_values = None
    lr_schedule = variant.get("lr_schedule", "constant")
    lr_final_scale = float(variant.get("lr_final_scale", 1.0))
    if lr_schedule != "constant":
        lr_values = lr_schedule_values(float(variant["lr"]), updates, lr_schedule, lr_final_scale)
    source_complexity = phase_source_complexity(
        x_train[:updates],
        target_matrix(y_train[:updates], 10, bool(args.softmax_output)),
        phases,
        sample_starts,
        t_stop,
        edge,
        True,
        phase_clock_mode,
        lr_values,
        labels=y_train[:updates],
        target_source_mode=getattr(args, "promotion_target_source_mode", "label"),
        sample_edge=sample_edge,
    )
    estimated_points = estimate_transient_points(t_stop, transient_step)
    max_transient_points = int(getattr(args, "promotion_max_transient_points", 0))
    max_source_pwl_points = int(getattr(args, "promotion_max_source_pwl_points", 0))
    max_sample_sources = int(getattr(args, "promotion_max_sample_sources", 0))
    max_total_sources = int(getattr(args, "promotion_max_total_sources", 0))
    max_output_vectors = int(getattr(args, "promotion_max_output_vectors", 0))
    inferred_image_size = int(round(float(x_train.shape[1]) ** 0.5))
    image_size = int(getattr(args, "image_size", inferred_image_size))
    block_size = int(getattr(args, "block_size", image_size))
    stride = int(getattr(args, "stride", block_size))
    channels = int(getattr(args, "channels", 1))
    blocks = block_indices(image_size, block_size, stride)
    output_bias_state_frozen = float(variant.get("output_bias_update_scale", 1.0)) == 0.0 and float(variant.get("state_decay", 0.0)) == 0.0
    output_vector_count = phase_output_vector_count(
        np.empty((len(blocks), channels, block_size * block_size)),
        np.empty((len(blocks), channels)),
        np.empty((10, len(blocks), channels)),
        np.empty((10,)),
        include_output_bias_vectors=not output_bias_state_frozen,
        include_y_vectors=bool(getattr(args, "promotion_phase_output_include_y", False)),
    )
    return {
        f"{prefix}_updates": updates,
        f"{prefix}_phase_clock_mode": phase_clock_mode,
        f"{prefix}_sample_edge_s": edge if sample_edge is None else sample_edge,
        f"{prefix}_hidden_preactivation_mode": hidden_preactivation_mode,
        f"{prefix}_hidden_preactivation_source_count": hidden_preactivation_source_count(
            hidden_preactivation_mode,
            len(blocks),
            channels,
        ),
        f"{prefix}_score_calculation_mode": score_calculation_mode,
        f"{prefix}_score_calculation_source_count": score_calculation_source_count(score_calculation_mode, 10),
        f"{prefix}_target_source_mode": getattr(args, "promotion_target_source_mode", "label"),
        f"{prefix}_output_bias_state_frozen": output_bias_state_frozen,
        f"{prefix}_phase_output_includes_y": bool(getattr(args, "promotion_phase_output_include_y", False)),
        f"{prefix}_output_vector_count": output_vector_count,
        f"{prefix}_output_vector_budget_met": bool(not max_output_vectors or output_vector_count <= max_output_vectors),
        f"{prefix}_estimated_transient_points": estimated_points,
        f"{prefix}_transient_budget_met": bool(not max_transient_points or estimated_points <= max_transient_points),
        f"{prefix}_sample_source_count": source_complexity["sample_source_count"],
        f"{prefix}_sample_source_budget_met": bool(
            not max_sample_sources or source_complexity["sample_source_count"] <= max_sample_sources
        ),
        f"{prefix}_sample_source_elided_dc_count": source_complexity["sample_source_elided_dc_count"],
        f"{prefix}_pixel_source_count": source_complexity["pixel_source_count"],
        f"{prefix}_pixel_source_elided_dc_count": source_complexity["pixel_source_elided_dc_count"],
        f"{prefix}_target_source_count": source_complexity["target_source_count"],
        f"{prefix}_target_behavioral_source_count": source_complexity["target_behavioral_source_count"],
        f"{prefix}_total_source_count": source_complexity["total_source_count"],
        f"{prefix}_total_source_budget_met": bool(
            not max_total_sources or source_complexity["total_source_count"] <= max_total_sources
        ),
        f"{prefix}_sample_source_pwl_points": source_complexity["sample_source_pwl_points"],
        f"{prefix}_phase_clock_source_pwl_points": source_complexity["phase_clock_source_pwl_points"],
        f"{prefix}_control_source_pwl_points": source_complexity["control_source_pwl_points"],
        f"{prefix}_total_source_pwl_points": source_complexity["total_source_pwl_points"],
        f"{prefix}_source_pwl_points_per_update": source_complexity["total_source_pwl_points"] / float(updates),
        f"{prefix}_source_pwl_budget_met": bool(
            not max_source_pwl_points or source_complexity["total_source_pwl_points"] <= max_source_pwl_points
        ),
    }


def strict_phase_cost_projection_fields(
    args: argparse.Namespace,
    variant: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, Any]:
    updates = int(getattr(args, "cost_projection_updates", 0) or 0)
    if updates <= 0:
        return {}
    return strict_phase_cost_fields_for_updates(
        args,
        variant,
        x_train,
        y_train,
        updates=updates,
        prefix="strict_phase_cost_projection",
    )


def run_variant(
    args: argparse.Namespace,
    variant: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    initial_state: TrainState,
    blocks: list[list[int]],
    probe_updates: tuple[int, ...],
) -> dict[str, Any]:
    fwd = forward_kwargs(args, variant)
    upd = update_kwargs(args, variant)
    state0 = tuple(arr.copy() for arr in initial_state)
    t0 = time.perf_counter()
    initial_eval = fast_ref.accuracy_np(x_eval, y_eval, state0, blocks, args.eval_batch_size, **fwd)
    final_state, probe_states = fast_ref.run_online(x_train, y_train, state0, blocks, probe_updates, **upd)
    final_eval = fast_ref.accuracy_np(x_eval, y_eval, final_state, blocks, args.eval_batch_size, **fwd)
    objective_fields = fast_reference_objective_fields(
        final_eval_accuracy=final_eval,
        eval_samples=int(args.eval_samples),
        full_objective_eval_samples=int(args.full_objective_eval_samples),
        full_objective_accuracy=float(args.full_objective_accuracy),
    )
    probe_rows = [
        {
            "update": update,
            "eval_accuracy": fast_ref.accuracy_np(x_eval, y_eval, probe_states[update], blocks, args.eval_batch_size, **fwd),
        }
        for update in probe_updates
        if update in probe_states
    ]
    best_probe = max(probe_rows, key=lambda row: row["eval_accuracy"]) if probe_rows else None
    probe_columns = {f"probe_eval_accuracy_u{int(row['update'])}": row["eval_accuracy"] for row in probe_rows}
    promotion_updates = int(getattr(args, "promotion_updates", 0) or args.train_samples)
    promotion_probe_eval_accuracy = probe_columns.get(f"probe_eval_accuracy_u{promotion_updates}")
    if promotion_probe_eval_accuracy is None and promotion_updates == args.train_samples:
        promotion_probe_eval_accuracy = final_eval
    wall = time.perf_counter() - t0
    phase_command = strict_phase_promotion_command(args, variant)
    promotion_costs = strict_phase_promotion_cost_fields(args, variant, x_train, y_train)
    cost_projection = strict_phase_cost_projection_fields(args, variant, x_train, y_train)
    promotion_efficiency = promotion_efficiency_fields(
        initial_eval_accuracy=initial_eval,
        promotion_probe_eval_accuracy=promotion_probe_eval_accuracy,
        promotion_updates=promotion_updates,
        promotion_costs=promotion_costs,
    )
    return {
        **variant,
        "initial_eval_accuracy": initial_eval,
        "final_eval_accuracy": final_eval,
        "eval_improvement": final_eval - initial_eval,
        **objective_fields,
        "best_probe_eval_accuracy": best_probe["eval_accuracy"] if best_probe is not None else None,
        "best_probe_update": best_probe["update"] if best_probe is not None else None,
        "promotion_probe_eval_accuracy": promotion_probe_eval_accuracy,
        "strict_phase_promotion_updates": promotion_updates,
        "strict_phase_promotion_timeout_s": promotion_timeout_seconds(args, promotion_updates),
        "strict_phase_promotion_max_transient_points": int(getattr(args, "promotion_max_transient_points", 2000)),
        "strict_phase_promotion_max_source_pwl_points": int(getattr(args, "promotion_max_source_pwl_points", 0)),
        "strict_phase_promotion_max_sample_sources": int(getattr(args, "promotion_max_sample_sources", 0)),
        "strict_phase_promotion_max_total_sources": int(getattr(args, "promotion_max_total_sources", 0)),
        "strict_phase_promotion_max_output_vectors": int(getattr(args, "promotion_max_output_vectors", 0)),
        **promotion_costs,
        **cost_projection,
        **promotion_efficiency,
        "strict_phase_promotion_command": command_text(phase_command),
        **probe_columns,
        "wall_time_s": wall,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-samples", type=int, default=256)
    ap.add_argument("--eval-samples", type=int, default=300)
    ap.add_argument("--image-size", type=int, default=10)
    ap.add_argument("--block-size", type=int, default=4)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.8)
    ap.add_argument("--lrs", default="")
    ap.add_argument("--lr-schedule", choices=["constant", "linear-decay"], default="constant")
    ap.add_argument("--lr-schedules", default="")
    ap.add_argument("--lr-final-scale", type=float, default=1.0)
    ap.add_argument("--lr-final-scales", default="")
    ap.add_argument("--linear-output", action="store_true")
    ap.add_argument("--softmax-output", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--activations",
        "--local-activation",
        dest="activations",
        default="tanh,diff-clipped-relu,softplus,leaky-relu",
    )
    ap.add_argument("--relu-clips", default="1.0")
    ap.add_argument("--relu-leak", type=float, default=0.01)
    ap.add_argument("--softplus-beta", type=float, default=10.0)
    ap.add_argument("--derivative-modes", default="exact")
    ap.add_argument("--derivative-floor", type=float, default=0.0)
    ap.add_argument("--derivative-gate-threshold", type=float, default=1e-6)
    ap.add_argument("--feedback-modes", default="full-readout")
    ap.add_argument("--readout-feedback-clip", type=float, default=0.05)
    ap.add_argument("--output-bias-update-scale", type=float, default=0.0)
    ap.add_argument("--output-bias-update-scales", default="")
    ap.add_argument("--readout-update-scale", type=float, default=0.25)
    ap.add_argument("--readout-update-scales", default="")
    ap.add_argument("--local-update-scale", type=float, default=1.0)
    ap.add_argument("--local-update-scales", default="")
    ap.add_argument("--state-decay", type=float, default=0.0)
    ap.add_argument("--state-decays", default="")
    ap.add_argument("--softmax-negative-scale", type=float, default=1.0)
    ap.add_argument("--softmax-error-centering", choices=["none", "mean"], default="none")
    ap.add_argument("--softmax-temperature", type=float, default=4.0)
    ap.add_argument("--softmax-temperatures", default="")
    ap.add_argument("--softmax-competition-mode", choices=["all", "normalized-power"], default="all")
    ap.add_argument("--softmax-competitor-power", type=int, default=2)
    ap.add_argument("--softmax-error-gate", choices=["none", "target-margin"], default="none")
    ap.add_argument("--softmax-margin", type=float, default=1.0)
    ap.add_argument("--hidden-synapse-modes", default="tanh-clipped")
    ap.add_argument("--readout-synapse-modes", default="linear")
    ap.add_argument("--synapse-clip", type=float, default=2.0)
    ap.add_argument("--synapse-clips", default="")
    ap.add_argument("--readout-class-centering", choices=["none", "mean"], default="none")
    ap.add_argument("--init-weights", default="")
    ap.add_argument("--probe-updates", default="powers2")
    ap.add_argument("--eval-batch-size", type=int, default=1024)
    ap.add_argument("--promotion-updates", type=int, default=0)
    ap.add_argument("--promotion-simulator", default="Xyce")
    ap.add_argument("--promotion-phase", type=float, default=0.5e-9)
    ap.add_argument("--promotion-gap", type=float, default=0.05e-9)
    ap.add_argument("--promotion-edge", type=float, default=5e-12)
    ap.add_argument(
        "--promotion-sample-edge",
        type=float,
        default=DEFAULT_PROMOTION_SAMPLE_EDGE,
        help=(
            "Sample/label/LR-control PWL transition edge for generated strict promotion commands and "
            "cost projections. The default 0 emits sharp sample steps while phase clocks keep finite edges."
        ),
    )
    ap.add_argument("--promotion-settle-ratio", type=float, default=20.0)
    ap.add_argument("--promotion-transient-step", type=float, default=200e-12)
    ap.add_argument(
        "--promotion-timeout",
        type=float,
        default=0.0,
        help="Strict promotion simulator timeout in seconds; 0 auto-scales with --promotion-updates.",
    )
    ap.add_argument("--promotion-max-transient-points", type=int, default=2000)
    ap.add_argument("--promotion-max-source-pwl-points", type=int, default=0)
    ap.add_argument("--promotion-max-sample-sources", type=int, default=0)
    ap.add_argument("--promotion-max-total-sources", type=int, default=0)
    ap.add_argument("--promotion-max-output-vectors", type=int, default=0)
    ap.add_argument(
        "--promotion-phase-clock-mode",
        choices=["pwl", "analytic"],
        default=DEFAULT_PROMOTION_PHASE_CLOCK_MODE,
        help=(
            "Clock source style for generated strict Xyce promotion commands. PWL is the robust default; "
            "analytic removes per-update clock PWL points but is currently an opt-in experiment knob."
        ),
    )
    ap.add_argument("--promotion-target-source-mode", choices=["rails", "label"], default="label")
    ap.add_argument(
        "--promotion-hidden-preactivation-mode",
        choices=["node", "inline"],
        default=DEFAULT_PROMOTION_HIDDEN_PREACTIVATION_MODE,
    )
    ap.add_argument(
        "--promotion-score-calculation-mode",
        choices=["node", "inline"],
        default=DEFAULT_PROMOTION_SCORE_CALCULATION_MODE,
    )
    ap.add_argument("--promotion-phase-output-include-y", action="store_true")
    ap.add_argument("--promotion-probe-updates", default="")
    ap.add_argument("--promotion-tag-prefix", default="promote")
    ap.add_argument(
        "--cost-projection-updates",
        type=int,
        default=0,
        help=(
            "Optional strict phase training-cost projection horizon. This can differ from --promotion-updates "
            "to estimate a long fast-reference trajectory while generating a shorter promotion command."
        ),
    )
    ap.add_argument(
        "--learning-thresholds",
        default="0.2,0.5,0.75,0.9",
        help="Comma-separated accuracy thresholds summarized by earliest configured probe horizon.",
    )
    ap.add_argument(
        "--full-objective-eval-samples",
        type=int,
        default=10000,
        help="Eval sample count required before a fast-reference row can be called a full-objective candidate.",
    )
    ap.add_argument(
        "--full-objective-accuracy",
        type=float,
        default=0.9,
        help="Accuracy threshold for fast-reference full-objective candidate metadata.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="fast_online_variant_sweep")
    args = ap.parse_args()

    if args.linear_output and args.softmax_output:
        raise ValueError("--linear-output and --softmax-output are mutually exclusive")
    if args.train_samples <= 0 or args.eval_samples <= 0:
        raise ValueError("sample counts must be positive")
    if args.channels <= 0:
        raise ValueError("--channels must be positive")
    if args.synapse_clip <= 0:
        raise ValueError("--synapse-clip must be positive")
    if args.softmax_negative_scale < 0:
        raise ValueError("--softmax-negative-scale must be non-negative")
    if args.softmax_temperature <= 0:
        raise ValueError("--softmax-temperature must be positive")
    if args.softmax_competitor_power < 1:
        raise ValueError("--softmax-competitor-power must be positive")
    if args.softmax_error_gate == "target-margin" and args.softmax_margin <= 0:
        raise ValueError("--softmax-margin must be positive when --softmax-error-gate is target-margin")
    if args.lr_final_scale < 0:
        raise ValueError("--lr-final-scale must be non-negative")
    if args.output_bias_update_scale < 0 or args.readout_update_scale < 0 or args.local_update_scale < 0:
        raise ValueError("update scales must be non-negative")
    if args.state_decay < 0 or args.state_decay >= 1:
        raise ValueError("--state-decay must be in [0, 1)")
    if args.promotion_updates < 0:
        raise ValueError("--promotion-updates must be non-negative")
    if args.cost_projection_updates < 0:
        raise ValueError("--cost-projection-updates must be non-negative")
    if args.cost_projection_updates and args.cost_projection_updates > args.train_samples:
        raise ValueError("--cost-projection-updates cannot exceed --train-samples")
    if args.promotion_timeout < 0:
        raise ValueError("--promotion-timeout must be non-negative")
    if args.promotion_sample_edge is not None and args.promotion_sample_edge < 0:
        raise ValueError("--promotion-sample-edge must be non-negative")
    if args.promotion_max_transient_points < 0:
        raise ValueError("--promotion-max-transient-points must be non-negative")
    if args.promotion_max_source_pwl_points < 0:
        raise ValueError("--promotion-max-source-pwl-points must be non-negative")
    if args.promotion_max_sample_sources < 0:
        raise ValueError("--promotion-max-sample-sources must be non-negative")
    if args.promotion_max_total_sources < 0:
        raise ValueError("--promotion-max-total-sources must be non-negative")
    if args.promotion_max_output_vectors < 0:
        raise ValueError("--promotion-max-output-vectors must be non-negative")
    if args.full_objective_eval_samples <= 0:
        raise ValueError("--full-objective-eval-samples must be positive")
    if args.full_objective_accuracy < 0.0 or args.full_objective_accuracy > 1.0:
        raise ValueError("--full-objective-accuracy must be in [0, 1]")
    learning_thresholds = parse_accuracy_thresholds(args.learning_thresholds)

    stride = args.block_size if args.stride is None else args.stride
    blocks = block_indices(args.image_size, args.block_size, stride)
    probe_updates = fast_ref.parse_probe_update_list(args.probe_updates, args.train_samples)
    promotion_updates = int(args.promotion_updates or args.train_samples)
    if 1 <= promotion_updates <= args.train_samples:
        probe_updates = tuple(sorted(set(probe_updates) | {promotion_updates}))
    x_train, y_train, x_eval, y_eval = load_mnist_sequence(args.train_samples, args.eval_samples, args.image_size, args.seed)
    rng = np.random.default_rng(args.seed)
    initial_state = fast_ref.load_or_init_weights(
        args.init_weights,
        rng,
        len(blocks),
        args.channels,
        args.block_size * args.block_size,
    )
    rows = [
        run_variant(args, variant, x_train, y_train, x_eval, y_eval, initial_state, blocks, probe_updates)
        for variant in variant_grid(args)
    ]
    rows.sort(key=lambda row: (row["final_eval_accuracy"], row["eval_improvement"]), reverse=True)
    probe_best = best_probe_variants_by_update(rows)

    results = ROOT / "spice/results"
    tables = ROOT / "results/tables"
    results.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    stem = f"spice_mnist_local_feature_fast_online_variant_{fast_ref.sanitize_tag(args.tag)}"
    csv_path = results / f"{stem}.csv"
    table_csv_path = tables / f"{stem}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    pd.DataFrame(rows).to_csv(table_csv_path, index=False)
    summary = {
        "architecture": "local_feature_fast_online_variant_sweep",
        "python_role": "Fast NumPy reference only; not an on-device/SPICE result.",
        "train_samples": args.train_samples,
        "eval_samples": args.eval_samples,
        "image_size": args.image_size,
        "block_size": args.block_size,
        "stride": stride,
        "channels": args.channels,
        "batch_size": 1,
        "variants": len(rows),
        "best_variant": rows[0] if rows else None,
        "best_promotion_variant": best_promotion_variant(rows),
        "best_promotion_efficiency_variant": best_promotion_efficiency_variant(rows),
        "best_fast_reference_full_objective_variant": best_fast_reference_full_objective_variant(rows),
        "best_fast_reference_full_objective_cost_feasible_variant": best_fast_reference_full_objective_cost_feasible_variant(rows),
        "best_probe_variants_by_update": probe_best,
        "learning_thresholds": learning_thresholds,
        "learning_threshold_hits": learning_threshold_hits(probe_best, learning_thresholds),
        "full_objective_eval_samples": args.full_objective_eval_samples,
        "full_objective_accuracy": args.full_objective_accuracy,
        "fast_reference_full_eval_sample_count_met": args.eval_samples >= args.full_objective_eval_samples,
        **cost_projection_summary_fields(rows, args.cost_projection_updates),
        "csv": str(csv_path),
        "table_csv": str(table_csv_path),
    }
    summary_path = results / f"{stem}_summary.json"
    table_summary_path = tables / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    table_summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
