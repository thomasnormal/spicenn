from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TARGET_TOPOLOGY = {
    "image_size": 10,
    "block_size": 4,
    "stride": 2,
    "channels": 2,
}


FIELDS = [
    "tag",
    "topology",
    "target_topology",
    "strict_target_contract_met",
    "strict_target_contract_issues",
    "strict_target_nontrivial_learning_met",
    "full_eval_10k_met",
    "full_objective_accuracy_met",
    "full_objective_accuracy_gap",
    "milestone_b_nontrivial_learning_met",
    "milestone_c_target_topology_met",
    "milestone_d_full_objective_met",
    "simulator",
    "updates",
    "eval_samples",
    "mnist_index_order",
    "train_index_sha256",
    "eval_index_sha256",
    "train_label_sha256",
    "train_label_prefix",
    "train_label_histogram",
    "train_dominant_label",
    "train_dominant_label_fraction",
    "train_unique_labels",
    "batch_size",
    "lr",
    "lr_schedule",
    "lr_final_scale",
    "lr_control_mode",
    "softmax_negative_scale",
    "softmax_error_centering",
    "softmax_temperature",
    "softmax_competition_mode",
    "softmax_competitor_power",
    "softmax_error_gate",
    "softmax_margin",
    "update_mode",
    "phase_clock_mode",
    "input_source_mode",
    "input_quantization_levels",
    "robust_sample_transitions",
    "target_source_mode",
    "hidden_activation_mode",
    "hidden_activation_state_count",
    "hidden_delta_mode",
    "hidden_delta_state_count",
    "score_state_mode",
    "score_state_count",
    "gradient_accumulator_state_count",
    "temporary_state_count",
    "output_delta_mode",
    "output_delta_state_count",
    "output_bias_state_frozen",
    "reference_mode",
    "eval_backend",
    "output_mode",
    "phase_output_includes_y",
    "local_activation",
    "local_update_scale",
    "output_bias_update_scale",
    "readout_update_scale",
    "state_decay",
    "hidden_synapse_mode",
    "readout_synapse_mode",
    "readout_class_centering",
    "fully_on_device_execution_contract_met",
    "strict_fully_on_device_contract_met",
    "strict_fully_on_device_requested",
    "strict_contract_inferred_from_legacy_summary",
    "random_init_used",
    "initial_weights_source",
    "continuous_transient_contract_met",
    "direction_matches_batch_op_reference",
    "eval_accuracy_matches_batch_op_reference",
    "python_weight_updates_between_samples",
    "python_checkpointing_between_samples",
    "initial_eval_accuracy",
    "phase_eval_accuracy",
    "phase_eval_improvement",
    "random_accuracy_threshold",
    "learning_improvement_threshold",
    "nontrivial_learning_met",
    "phase_dominant_pred_class",
    "phase_dominant_pred_fraction",
    "phase_unique_predicted_classes",
    "spice_phase_eval_accuracy",
    "numpy_phase_eval_accuracy",
    "phase_eval_backend_abs_diff",
    "phase_update_l2",
    "state_update_direction_cosine",
    "state_update_sign_alignment_fraction",
    "transient_step_s",
    "final_measure_tail_s",
    "estimated_transient_points",
    "estimated_transient_points_per_update",
    "max_transient_points",
    "transient_budget_met",
    "phase_output_vector_count",
    "max_output_vectors",
    "output_vector_budget_met",
    "max_source_pwl_points",
    "source_pwl_budget_met",
    "max_sample_sources",
    "sample_source_budget_met",
    "max_total_sources",
    "total_source_budget_met",
    "max_auxiliary_algebraic_sources",
    "auxiliary_algebraic_source_budget_met",
    "sample_source_count",
    "total_source_count",
    "sample_source_dc_count",
    "sample_source_pwl_count",
    "sample_source_elided_dc_count",
    "sample_source_pwl_points",
    "sample_source_pwl_points_per_update",
    "pixel_source_count",
    "pixel_source_dc_count",
    "pixel_source_pwl_count",
    "pixel_source_elided_dc_count",
    "pixel_source_pwl_points",
    "target_source_dc_count",
    "target_source_count",
    "target_source_pwl_count",
    "target_source_elided_dc_count",
    "target_source_pwl_points",
    "target_behavioral_source_count",
    "phase_clock_source_count",
    "phase_clock_source_pwl_count",
    "phase_clock_source_pwl_points",
    "phase_clock_source_pwl_points_per_update",
    "control_source_count",
    "control_source_pwl_points",
    "control_source_pwl_points_per_update",
    "total_source_pwl_points",
    "total_source_pwl_points_per_update",
    "phase_wall_time_s",
    "eval_wall_time_s",
    "summary_path",
]


def is_phase_transient_summary(data: dict[str, Any]) -> bool:
    status = str(data.get("status") or "")
    return (
        status.startswith("continuous_phase_train")
        or data.get("architecture") == "phase_resolved_transient_local_feature_readout"
        or "fully_on_device_execution_contract_met" in data
        or "single_phase_training_transient" in data
    )


def topology_label(data: dict[str, Any]) -> str:
    image_size = data.get("image_size")
    block_size = data.get("block_size")
    stride = data.get("stride")
    channels = data.get("channels")
    return f"{image_size}x{image_size} b{block_size} s{stride} c{channels}"


def nested(data: dict[str, Any], *keys: str) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def derived_total_source_pwl_points(data: dict[str, Any]) -> int | None:
    total = data.get("total_source_pwl_points")
    if total is not None:
        return total
    sample_points = data.get("sample_source_pwl_points")
    clock_points = data.get("phase_clock_source_pwl_points")
    control_points = data.get("control_source_pwl_points", 0)
    if sample_points is None or clock_points is None:
        return None
    return int(sample_points) + int(clock_points) + int(control_points or 0)


def as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_phase_architecture(data: dict[str, Any]) -> bool:
    return data.get("architecture") == "phase_resolved_transient_local_feature_readout"


def is_continuous_phase_status(data: dict[str, Any]) -> bool:
    return str(data.get("status") or "").startswith("continuous_phase_train")


def inferred_random_init_used(data: dict[str, Any]) -> bool | None:
    random_init = data.get("random_init_used")
    if random_init is not None:
        return bool(random_init)
    source = data.get("initial_weights_source")
    if source is not None:
        return source == "random_init"
    init_weights = data.get("init_weights")
    if init_weights is not None:
        return init_weights == ""
    return None


def inferred_initial_weights_source(data: dict[str, Any]) -> str | None:
    source = data.get("initial_weights_source")
    if source is not None:
        return str(source)
    if data.get("random_init_used") is True:
        return "random_init"
    if data.get("init_weights") == "":
        return "random_init"
    return None


def inferred_python_weight_updates_between_samples(data: dict[str, Any]) -> bool | None:
    value = data.get("python_weight_updates_between_samples")
    if value is not None:
        return bool(value)
    if (
        is_phase_architecture(data)
        and is_continuous_phase_status(data)
        and data.get("python_checkpointing_between_samples") is False
    ):
        return False
    return None


def inferred_fully_on_device_execution_contract_met(data: dict[str, Any]) -> bool | None:
    value = data.get("fully_on_device_execution_contract_met")
    if value is not None:
        return bool(value)
    no_checkpointing = data.get("python_checkpointing_between_samples") is False
    no_weight_updates = inferred_python_weight_updates_between_samples(data) is False
    if (
        is_phase_architecture(data)
        and is_continuous_phase_status(data)
        and data.get("batch_size") == 1
        and no_checkpointing
        and no_weight_updates
    ):
        return True
    return None


def inferred_strict_fully_on_device_contract_met(data: dict[str, Any]) -> bool | None:
    value = data.get("strict_fully_on_device_contract_met")
    if value is not None:
        return bool(value)
    random_init = inferred_random_init_used(data) is True
    no_checkpointing = data.get("python_checkpointing_between_samples") is False
    no_weight_updates = inferred_python_weight_updates_between_samples(data) is False
    fully_on_device = inferred_fully_on_device_execution_contract_met(data) is True
    if (
        fully_on_device
        and data.get("batch_size") == 1
        and data.get("reference_mode") == "none"
        and random_init
        and no_checkpointing
        and no_weight_updates
    ):
        return True
    return None


def strict_contract_inferred_from_legacy_summary(data: dict[str, Any]) -> bool:
    return data.get("strict_fully_on_device_contract_met") is None and inferred_strict_fully_on_device_contract_met(data) is True


def inferred_strict_fully_on_device_requested(data: dict[str, Any]) -> bool | None:
    value = data.get("strict_fully_on_device_requested")
    if value is not None:
        return bool(value)
    if strict_contract_inferred_from_legacy_summary(data):
        return True
    return None


def inferred_eval_backend(data: dict[str, Any]) -> str | None:
    backend = data.get("eval_backend")
    if backend is not None:
        return str(backend)
    has_spice_eval = data.get("spice_phase_eval_accuracy") is not None
    has_numpy_eval = data.get("numpy_phase_eval_accuracy") is not None
    if has_spice_eval and has_numpy_eval:
        return "both"
    if has_spice_eval:
        return "spice"
    if has_numpy_eval:
        return "numpy"
    if data.get("phase_eval_accuracy") is not None:
        return "legacy_phase_eval"
    return None


def derived_nontrivial_learning_met(row: dict[str, Any]) -> bool:
    if row.get("nontrivial_learning_met") is True:
        return True
    accuracy = as_float_or_none(row.get("phase_eval_accuracy"))
    improvement = as_float_or_none(row.get("phase_eval_improvement"))
    random_threshold = as_float_or_none(row.get("random_accuracy_threshold"))
    improvement_threshold = as_float_or_none(row.get("learning_improvement_threshold"))
    if accuracy is None or improvement is None:
        return False
    if random_threshold is None:
        random_threshold = 0.1
    if improvement_threshold is None:
        improvement_threshold = 0.02
    return accuracy > random_threshold and improvement >= improvement_threshold


def robust_sample_transitions(row: dict[str, Any]) -> bool:
    sample_edge = row.get("sample_edge_s")
    if sample_edge is None:
        return True
    edge = as_float_or_none(sample_edge)
    return edge is None or edge > 0.0


def full_objective_accuracy_gap(row: dict[str, Any], threshold: float = 0.9) -> float | None:
    accuracy = as_float_or_none(row.get("phase_eval_accuracy"))
    if accuracy is None:
        return None
    return max(0.0, threshold - accuracy)


def row_from_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    tag = path.name.removesuffix("_summary.json")
    row = {
        "tag": tag,
        "topology": topology_label(data),
        "image_size": data.get("image_size"),
        "block_size": data.get("block_size"),
        "stride": data.get("stride"),
        "channels": data.get("channels"),
        "simulator": data.get("simulator_selector") or data.get("simulator"),
        "updates": data.get("updates"),
        "eval_samples": data.get("eval_samples"),
        "mnist_index_order": data.get("mnist_index_order"),
        "train_index_sha256": nested(data, "train_index_metadata", "sha256"),
        "eval_index_sha256": nested(data, "eval_index_metadata", "sha256"),
        "train_label_sha256": nested(data, "train_label_metadata", "sha256"),
        "train_label_prefix": nested(data, "train_label_metadata", "prefix"),
        "train_label_histogram": nested(data, "train_label_metadata", "histogram"),
        "train_dominant_label": nested(data, "train_label_metadata", "dominant_label"),
        "train_dominant_label_fraction": nested(data, "train_label_metadata", "dominant_label_fraction"),
        "train_unique_labels": nested(data, "train_label_metadata", "unique_labels"),
        "batch_size": data.get("batch_size"),
        "lr": data.get("lr"),
        "lr_schedule": data.get("lr_schedule", "constant"),
        "lr_final_scale": data.get("lr_final_scale", 1.0),
        "lr_control_mode": data.get("lr_control_mode", "pwl"),
        "softmax_negative_scale": data.get("softmax_negative_scale"),
        "softmax_error_centering": data.get("softmax_error_centering"),
        "softmax_temperature": data.get("softmax_temperature"),
        "softmax_competition_mode": data.get("softmax_competition_mode"),
        "softmax_competitor_power": data.get("softmax_competitor_power"),
        "softmax_error_gate": data.get("softmax_error_gate"),
        "softmax_margin": data.get("softmax_margin"),
        "update_mode": data.get("update_mode"),
        "phase_clock_mode": data.get("phase_clock_mode", "pwl"),
        "input_source_mode": data.get("input_source_mode", "pwl"),
        "input_quantization_levels": data.get("input_quantization_levels", 0),
        "sample_edge_s": data.get("sample_edge_s"),
        "hidden_preactivation_mode": data.get("hidden_preactivation_mode", "node"),
        "hidden_preactivation_source_count": data.get("hidden_preactivation_source_count"),
        "hidden_activation_mode": data.get("hidden_activation_mode", "stored"),
        "hidden_activation_state_count": data.get("hidden_activation_state_count"),
        "hidden_delta_mode": data.get("hidden_delta_mode", "stored"),
        "hidden_delta_state_count": data.get("hidden_delta_state_count"),
        "score_state_mode": data.get("score_state_mode", "stored"),
        "score_state_count": data.get("score_state_count"),
        "gradient_accumulator_state_count": data.get("gradient_accumulator_state_count"),
        "temporary_state_count": data.get("temporary_state_count"),
        "score_calculation_mode": data.get("score_calculation_mode", "node"),
        "score_calculation_source_count": data.get("score_calculation_source_count"),
        "output_rail_mode": data.get("output_rail_mode", "node"),
        "output_rail_source_count": data.get("output_rail_source_count"),
        "output_delta_mode": data.get("output_delta_mode", "node"),
        "output_delta_state_count": data.get("output_delta_state_count"),
        "auxiliary_algebraic_source_count": data.get("auxiliary_algebraic_source_count"),
        "target_source_mode": data.get("target_source_mode", "rails"),
        "output_bias_state_frozen": data.get("output_bias_state_frozen"),
        "reference_mode": data.get("reference_mode"),
        "eval_backend": inferred_eval_backend(data),
        "output_mode": data.get("output_mode"),
        "phase_output_includes_y": data.get("phase_output_includes_y"),
        "local_activation": data.get("local_activation"),
        "local_update_scale": data.get("local_update_scale"),
        "output_bias_update_scale": data.get("output_bias_update_scale"),
        "readout_update_scale": data.get("readout_update_scale"),
        "state_decay": data.get("state_decay"),
        "hidden_synapse_mode": data.get("hidden_synapse_mode"),
        "readout_synapse_mode": data.get("readout_synapse_mode"),
        "readout_class_centering": data.get("readout_class_centering"),
        "fully_on_device_execution_contract_met": inferred_fully_on_device_execution_contract_met(data),
        "strict_fully_on_device_contract_met": inferred_strict_fully_on_device_contract_met(data),
        "strict_fully_on_device_requested": inferred_strict_fully_on_device_requested(data),
        "strict_contract_inferred_from_legacy_summary": strict_contract_inferred_from_legacy_summary(data),
        "random_init_used": inferred_random_init_used(data),
        "initial_weights_source": inferred_initial_weights_source(data),
        "continuous_transient_contract_met": data.get("continuous_transient_contract_met"),
        "direction_matches_batch_op_reference": data.get("direction_matches_batch_op_reference"),
        "eval_accuracy_matches_batch_op_reference": data.get("eval_accuracy_matches_batch_op_reference"),
        "python_weight_updates_between_samples": inferred_python_weight_updates_between_samples(data),
        "python_checkpointing_between_samples": data.get("python_checkpointing_between_samples"),
        "initial_eval_accuracy": data.get("initial_eval_accuracy"),
        "phase_eval_accuracy": data.get("phase_eval_accuracy"),
        "phase_eval_improvement": data.get("phase_eval_improvement"),
        "random_accuracy_threshold": data.get("random_accuracy_threshold"),
        "learning_improvement_threshold": data.get("learning_improvement_threshold"),
        "nontrivial_learning_met": data.get("nontrivial_learning_met"),
        "phase_dominant_pred_class": nested(data, "phase_numpy_eval_diagnostics", "dominant_pred_class"),
        "phase_dominant_pred_fraction": nested(data, "phase_numpy_eval_diagnostics", "dominant_pred_fraction"),
        "phase_unique_predicted_classes": nested(data, "phase_numpy_eval_diagnostics", "unique_predicted_classes"),
        "spice_phase_eval_accuracy": data.get("spice_phase_eval_accuracy"),
        "numpy_phase_eval_accuracy": data.get("numpy_phase_eval_accuracy"),
        "phase_eval_backend_abs_diff": data.get("phase_eval_backend_abs_diff"),
        "phase_update_l2": data.get("phase_update_l2"),
        "state_update_direction_cosine": data.get("state_update_direction_cosine"),
        "state_update_sign_alignment_fraction": data.get("state_update_sign_alignment_fraction"),
        "transient_step_s": data.get("transient_step_s"),
        "final_measure_tail_s": data.get("final_measure_tail_s", 0.0),
        "estimated_transient_points": data.get("estimated_transient_points"),
        "estimated_transient_points_per_update": data.get("estimated_transient_points_per_update"),
        "max_transient_points": data.get("max_transient_points"),
        "transient_budget_met": data.get("transient_budget_met"),
        "phase_output_vector_count": data.get("phase_output_vector_count"),
        "max_output_vectors": data.get("max_output_vectors"),
        "output_vector_budget_met": data.get("output_vector_budget_met"),
        "max_source_pwl_points": data.get("max_source_pwl_points"),
        "source_pwl_budget_met": data.get("source_pwl_budget_met"),
        "max_sample_sources": data.get("max_sample_sources"),
        "sample_source_budget_met": data.get("sample_source_budget_met"),
        "max_total_sources": data.get("max_total_sources"),
        "total_source_budget_met": data.get("total_source_budget_met"),
        "max_auxiliary_algebraic_sources": data.get("max_auxiliary_algebraic_sources"),
        "auxiliary_algebraic_source_budget_met": data.get("auxiliary_algebraic_source_budget_met"),
        "sample_source_count": data.get("sample_source_count"),
        "total_source_count": data.get("total_source_count"),
        "sample_source_dc_count": data.get("sample_source_dc_count"),
        "sample_source_pwl_count": data.get("sample_source_pwl_count"),
        "sample_source_elided_dc_count": data.get("sample_source_elided_dc_count"),
        "sample_source_pwl_points": data.get("sample_source_pwl_points"),
        "sample_source_pwl_points_per_update": data.get("sample_source_pwl_points_per_update"),
        "pixel_source_count": data.get("pixel_source_count"),
        "pixel_source_dc_count": data.get("pixel_source_dc_count"),
        "pixel_source_pwl_count": data.get("pixel_source_pwl_count"),
        "pixel_source_elided_dc_count": data.get("pixel_source_elided_dc_count"),
        "pixel_source_pwl_points": data.get("pixel_source_pwl_points"),
        "pixel_rom_index_source_count": data.get("pixel_rom_index_source_count"),
        "pixel_rom_behavioral_source_count": data.get("pixel_rom_behavioral_source_count"),
        "pixel_rom_value_count": data.get("pixel_rom_value_count"),
        "target_source_dc_count": data.get("target_source_dc_count"),
        "target_source_count": data.get("target_source_count"),
        "target_source_pwl_count": data.get("target_source_pwl_count"),
        "target_source_elided_dc_count": data.get("target_source_elided_dc_count"),
        "target_source_pwl_points": data.get("target_source_pwl_points"),
        "target_behavioral_source_count": data.get("target_behavioral_source_count"),
        "phase_clock_source_count": data.get("phase_clock_source_count"),
        "phase_clock_source_pwl_count": data.get("phase_clock_source_pwl_count"),
        "phase_clock_source_pwl_points": data.get("phase_clock_source_pwl_points"),
        "phase_clock_source_pwl_points_per_update": data.get("phase_clock_source_pwl_points_per_update"),
        "control_source_count": data.get("control_source_count"),
        "control_source_pwl_points": data.get("control_source_pwl_points"),
        "control_source_pwl_points_per_update": data.get("control_source_pwl_points_per_update"),
        "total_source_pwl_points": derived_total_source_pwl_points(data),
        "total_source_pwl_points_per_update": data.get("total_source_pwl_points_per_update"),
        "phase_wall_time_s": data.get("phase_wall_time_s"),
        "eval_wall_time_s": data.get("eval_wall_time_s"),
        "summary_mtime_s": path.stat().st_mtime,
        "summary_path": str(path),
    }
    row["target_topology"] = same_target_topology(row)
    row["robust_sample_transitions"] = robust_sample_transitions(row)
    issues = strict_target_contract_issues(row)
    row["strict_target_contract_met"] = not issues
    row["strict_target_contract_issues"] = issues
    nontrivial_learning = derived_nontrivial_learning_met(row)
    full_eval_10k = as_int(row.get("eval_samples")) >= 10000
    full_accuracy = (as_float_or_none(row.get("phase_eval_accuracy")) or -1.0) >= 0.9
    row["strict_target_nontrivial_learning_met"] = row["strict_target_contract_met"] and nontrivial_learning
    row["full_eval_10k_met"] = full_eval_10k
    row["full_objective_accuracy_met"] = full_accuracy
    row["full_objective_accuracy_gap"] = full_objective_accuracy_gap(row)
    row["milestone_b_nontrivial_learning_met"] = row["strict_target_nontrivial_learning_met"]
    row["milestone_c_target_topology_met"] = row["strict_target_contract_met"]
    row["milestone_d_full_objective_met"] = row["strict_target_contract_met"] and full_eval_10k and full_accuracy
    return row


def discover_summary_paths(root: Path = ROOT) -> list[Path]:
    paths_by_name: dict[str, Path] = {}
    for summary_dir in (root / "results" / "tables", root / "spice" / "results"):
        if not summary_dir.exists():
            continue
        for path in sorted(summary_dir.glob("spice_mnist_local_feature_phase*_summary.json")):
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if is_phase_transient_summary(data):
                paths_by_name.setdefault(path.name, path)
    return sorted(paths_by_name.values(), key=lambda path: path.name)


def as_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def same_target_topology(row: dict[str, Any]) -> bool:
    return all(row.get(key) == value for key, value in TARGET_TOPOLOGY.items())


def strict_target_contract_issues(row: dict[str, Any]) -> list[str]:
    checks = [
        ("target_topology", same_target_topology(row)),
        ("batch_size_1", row.get("batch_size") == 1),
        ("robust_sample_transitions", robust_sample_transitions(row)),
        ("strict_contract", row.get("strict_fully_on_device_contract_met") is True),
        ("strict_requested", row.get("strict_fully_on_device_requested") is True),
        ("random_init", row.get("random_init_used") is True),
        ("random_init_source", row.get("initial_weights_source") == "random_init"),
        ("reference_mode_none", row.get("reference_mode") == "none"),
        ("no_python_weight_updates", row.get("python_weight_updates_between_samples") is False),
        ("no_python_checkpointing", row.get("python_checkpointing_between_samples") is False),
    ]
    return [name for name, ok in checks if not ok]


def filter_rows(
    rows: list[dict[str, Any]],
    *,
    target_topology: bool = False,
    contract_only: bool = False,
    strict_contract_only: bool = False,
    min_updates: int | None = None,
) -> list[dict[str, Any]]:
    selected = rows
    if target_topology:
        selected = [row for row in selected if same_target_topology(row)]
    if contract_only:
        selected = [
            row
            for row in selected
            if row.get("fully_on_device_execution_contract_met") is True
        ]
    if strict_contract_only:
        selected = [
            row
            for row in selected
            if row.get("strict_fully_on_device_contract_met") is True
        ]
    if min_updates is not None:
        selected = [row for row in selected if as_int(row.get("updates")) >= min_updates]
    return selected


def sort_rows(rows: list[dict[str, Any]], sort_key: str) -> list[dict[str, Any]]:
    if sort_key == "accuracy":
        return sorted(
            rows,
            key=lambda row: (
                float(row.get("phase_eval_accuracy") or -1.0),
                as_int(row.get("updates")),
                float(row.get("phase_eval_improvement") or -999.0),
            ),
            reverse=True,
        )
    if sort_key == "updates":
        return sorted(
            rows,
            key=lambda row: (as_int(row.get("updates")), row.get("tag") or ""),
            reverse=True,
        )
    if sort_key == "improvement":
        return sorted(
            rows,
            key=lambda row: (
                float(row.get("phase_eval_improvement") or -999.0),
                as_int(row.get("updates")),
            ),
            reverse=True,
        )
    return sorted(rows, key=lambda row: float(row.get("summary_mtime_s") or 0.0), reverse=True)


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return str(value)


def print_markdown(rows: list[dict[str, Any]]) -> None:
    columns = [
        "tag",
        "topology",
        "updates",
        "eval_samples",
        "eval_backend",
        "phase_clock_mode",
        "input_source_mode",
        "input_quantization_levels",
        "sample_edge_s",
        "robust_sample_transitions",
        "hidden_preactivation_mode",
        "hidden_preactivation_source_count",
        "hidden_activation_mode",
        "hidden_activation_state_count",
        "hidden_delta_mode",
        "hidden_delta_state_count",
        "score_state_mode",
        "score_state_count",
        "gradient_accumulator_state_count",
        "temporary_state_count",
        "score_calculation_mode",
        "score_calculation_source_count",
        "output_rail_mode",
        "output_rail_source_count",
        "output_delta_mode",
        "output_delta_state_count",
        "auxiliary_algebraic_source_count",
        "target_source_mode",
        "output_bias_state_frozen",
        "lr",
        "lr_schedule",
        "lr_final_scale",
        "lr_control_mode",
        "softmax_negative_scale",
        "softmax_error_centering",
        "softmax_temperature",
        "softmax_competition_mode",
        "softmax_competitor_power",
        "softmax_error_gate",
        "softmax_margin",
        "train_label_prefix",
        "train_label_histogram",
        "local_update_scale",
        "output_bias_update_scale",
        "readout_update_scale",
        "state_decay",
        "readout_class_centering",
        "strict_target_contract_met",
        "strict_target_contract_issues",
        "strict_target_nontrivial_learning_met",
        "full_eval_10k_met",
        "full_objective_accuracy_met",
        "full_objective_accuracy_gap",
        "milestone_d_full_objective_met",
        "fully_on_device_execution_contract_met",
        "strict_fully_on_device_contract_met",
        "random_init_used",
        "strict_contract_inferred_from_legacy_summary",
        "reference_mode",
        "initial_eval_accuracy",
        "phase_eval_accuracy",
        "phase_eval_improvement",
        "nontrivial_learning_met",
        "phase_dominant_pred_class",
        "phase_dominant_pred_fraction",
        "phase_unique_predicted_classes",
        "phase_eval_backend_abs_diff",
        "phase_update_l2",
        "transient_step_s",
        "final_measure_tail_s",
        "estimated_transient_points",
        "estimated_transient_points_per_update",
        "phase_output_vector_count",
        "auxiliary_algebraic_source_count",
        "sample_source_count",
        "total_source_count",
        "sample_source_elided_dc_count",
        "sample_source_pwl_points",
        "sample_source_pwl_points_per_update",
        "pixel_source_pwl_points",
        "pixel_rom_behavioral_source_count",
        "pixel_rom_value_count",
        "target_source_pwl_points",
        "phase_clock_source_pwl_points",
        "phase_clock_source_pwl_points_per_update",
        "control_source_pwl_points",
        "control_source_pwl_points_per_update",
        "total_source_pwl_points",
        "total_source_pwl_points_per_update",
        "phase_wall_time_s",
    ]
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        print("| " + " | ".join(format_value(row.get(column)) for column in columns) + " |")


def resolve_input_paths(paths: list[str]) -> list[Path]:
    resolved = [Path(path) if Path(path).is_absolute() else ROOT / path for path in paths]
    missing = [path for path in resolved if not path.exists()]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise SystemExit(f"Missing summary path(s):\n{joined}")
    return resolved


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Summarize fully-on-device local-feature phase-transient result JSONs."
    )
    ap.add_argument(
        "summaries",
        nargs="*",
        help="Summary JSON paths. Defaults to discovered phase summaries.",
    )
    ap.add_argument("--target-topology", action="store_true", help="Keep only 10x10 b4 stride2 c2 runs.")
    ap.add_argument("--contract-only", action="store_true", help="Keep only runs meeting the execution contract.")
    ap.add_argument(
        "--strict-contract-only",
        action="store_true",
        help="Keep only random-init, batch_size=1, no-reference fully-on-device runs.",
    )
    ap.add_argument("--min-updates", type=int, help="Keep only runs with at least this many online updates.")
    ap.add_argument("--limit", type=int, help="Maximum rows to print or write after sorting.")
    ap.add_argument(
        "--sort",
        choices=["latest", "updates", "improvement", "accuracy"],
        default="latest",
        help="Row ordering before applying --limit.",
    )
    ap.add_argument("--out", type=Path, help="Optional CSV output path.")
    ap.add_argument("--json-out", type=Path, help="Optional JSON output path.")
    ap.add_argument("--markdown", action="store_true", help="Print a compact Markdown table.")
    args = ap.parse_args()

    paths = resolve_input_paths(args.summaries) if args.summaries else discover_summary_paths()
    rows = [row_from_summary(path) for path in paths]
    rows = filter_rows(
        rows,
        target_topology=args.target_topology,
        contract_only=args.contract_only,
        strict_contract_only=args.strict_contract_only,
        min_updates=args.min_updates,
    )
    rows = sort_rows(rows, args.sort)
    if args.limit is not None:
        rows = rows[: args.limit]

    if args.out:
        out_path = args.out if args.out.is_absolute() else ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=FIELDS,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
    if args.json_out:
        out_path = args.json_out if args.json_out.is_absolute() else ROOT / args.json_out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, indent=2) + "\n")
    if args.markdown or (not args.out and not args.json_out):
        print_markdown(rows)


if __name__ == "__main__":
    main()
