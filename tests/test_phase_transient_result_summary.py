from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_SCRIPT = ROOT / "experiments" / "27_summarize_phase_transient_results.py"


def load_summary_module():
    spec = importlib.util.spec_from_file_location("summarize_phase_transient_results", SUMMARY_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase_summary_row_keeps_execution_contract_and_backend_fields(tmp_path: Path) -> None:
    module = load_summary_module()
    summary_path = tmp_path / "spice_mnist_local_feature_phase_demo_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "simulator_selector": "Xyce",
                "status": "continuous_phase_train_no_reference",
                "architecture": "phase_resolved_transient_local_feature_readout",
                "image_size": 10,
                "block_size": 4,
                "stride": 2,
                "channels": 2,
                "mnist_index_order": "stable_permutation_prefix",
                "train_index_metadata": {"sha256": "trainhash", "count": 16, "prefix": [1, 2]},
                "eval_index_metadata": {"sha256": "evalhash", "count": 100, "prefix": [3, 4]},
                "train_label_metadata": {
                    "sha256": "labelhash",
                    "prefix": [7, 2, 7, 1],
                    "histogram": [0, 1, 1, 0, 0, 0, 0, 2, 0, 0],
                    "dominant_label": 7,
                    "dominant_label_fraction": 0.5,
                    "unique_labels": 3,
                },
                "batch_size": 1,
                "updates": 16,
                "eval_samples": 100,
                "lr": 0.1,
                "lr_schedule": "linear-decay",
                "lr_final_scale": 0.25,
                "softmax_negative_scale": 0.25,
                "softmax_error_centering": "mean",
                "update_mode": "direct",
                "phase_clock_mode": "analytic",
                "reference_mode": "none",
                "eval_backend": "both",
                "output_mode": "print",
                "local_activation": "tanh",
                "local_update_scale": 0.75,
                "output_bias_update_scale": 0.25,
                "readout_update_scale": 0.5,
                "hidden_synapse_mode": "tanh-clipped",
                "readout_synapse_mode": "linear",
                "readout_class_centering": "mean",
                "single_phase_training_transient": True,
                "weights_persist_inside_phase_transient": True,
                "python_weight_updates_between_samples": False,
                "python_checkpointing_between_samples": False,
                "fully_on_device_execution_contract_met": True,
                "strict_fully_on_device_contract_met": True,
                "strict_fully_on_device_requested": True,
                "random_init_used": True,
                "initial_weights_source": "random_init",
                "continuous_transient_contract_met": None,
                "initial_eval_accuracy": 0.05,
                "phase_eval_accuracy": 0.12,
                "phase_eval_improvement": 0.07,
                "random_accuracy_threshold": 0.1,
                "learning_improvement_threshold": 0.02,
                "phase_numpy_eval_diagnostics": {
                    "dominant_pred_class": 3,
                    "dominant_pred_fraction": 0.4,
                    "unique_predicted_classes": 5,
                },
                "phase_update_l2": 0.364,
                "estimated_transient_points": 902,
                "max_transient_points": 2000,
                "phase_output_vector_count": 884,
                "sample_source_count": 103,
                "max_source_pwl_points": 50000,
                "max_sample_sources": 120,
                "max_total_sources": 130,
                "total_source_count": 109,
                "sample_source_dc_count": 0,
                "sample_source_pwl_count": 100,
                "sample_source_elided_dc_count": 7,
                "sample_source_pwl_points": 25478,
                "pixel_source_count": 93,
                "pixel_source_dc_count": 0,
                "pixel_source_pwl_count": 93,
                "pixel_source_elided_dc_count": 7,
                "target_source_dc_count": 0,
                "target_source_elided_dc_count": 0,
                "phase_clock_source_count": 5,
                "phase_clock_source_pwl_count": 0,
                "phase_clock_source_pwl_points": 0,
                "control_source_count": 1,
                "control_source_pwl_points": 4,
                "total_source_pwl_points": 25482,
                "phase_wall_time_s": 36.1,
                "eval_wall_time_s": 9.5,
                "spice_phase_eval_accuracy": 0.12,
                "numpy_phase_eval_accuracy": 0.12,
                "phase_eval_backend_abs_diff": 0.0,
            }
        )
        + "\n"
    )

    row = module.row_from_summary(summary_path)

    assert row["tag"] == "spice_mnist_local_feature_phase_demo"
    assert row["topology"] == "10x10 b4 s2 c2"
    assert row["simulator"] == "Xyce"
    assert row["updates"] == 16
    assert row["mnist_index_order"] == "stable_permutation_prefix"
    assert row["train_index_sha256"] == "trainhash"
    assert row["eval_index_sha256"] == "evalhash"
    assert row["train_label_sha256"] == "labelhash"
    assert row["train_label_prefix"] == [7, 2, 7, 1]
    assert row["train_label_histogram"] == [0, 1, 1, 0, 0, 0, 0, 2, 0, 0]
    assert row["train_dominant_label"] == 7
    assert row["train_dominant_label_fraction"] == 0.5
    assert row["train_unique_labels"] == 3
    assert row["lr"] == 0.1
    assert row["lr_schedule"] == "linear-decay"
    assert row["lr_final_scale"] == 0.25
    assert row["softmax_negative_scale"] == 0.25
    assert row["softmax_error_centering"] == "mean"
    assert row["phase_clock_mode"] == "analytic"
    assert row["local_update_scale"] == 0.75
    assert row["output_bias_update_scale"] == 0.25
    assert row["readout_update_scale"] == 0.5
    assert row["readout_class_centering"] == "mean"
    assert row["eval_backend"] == "both"
    assert row["phase_dominant_pred_class"] == 3
    assert row["phase_dominant_pred_fraction"] == 0.4
    assert row["phase_unique_predicted_classes"] == 5
    assert row["fully_on_device_execution_contract_met"] is True
    assert row["strict_fully_on_device_contract_met"] is True
    assert row["strict_fully_on_device_requested"] is True
    assert row["random_init_used"] is True
    assert row["initial_weights_source"] == "random_init"
    assert row["python_weight_updates_between_samples"] is False
    assert row["python_checkpointing_between_samples"] is False
    assert row["phase_eval_accuracy"] == 0.12
    assert row["phase_eval_backend_abs_diff"] == 0.0
    assert row["target_topology"] is True
    assert row["strict_target_contract_met"] is True
    assert row["strict_target_contract_issues"] == []
    assert row["strict_target_nontrivial_learning_met"] is True
    assert row["full_eval_10k_met"] is False
    assert row["full_objective_accuracy_met"] is False
    assert row["full_objective_accuracy_gap"] == 0.78
    assert row["milestone_b_nontrivial_learning_met"] is True
    assert row["milestone_c_target_topology_met"] is True
    assert row["milestone_d_full_objective_met"] is False
    assert row["estimated_transient_points"] == 902
    assert row["max_transient_points"] == 2000
    assert row["phase_output_vector_count"] == 884
    assert row["sample_source_count"] == 103
    assert row["max_source_pwl_points"] == 50000
    assert row["max_sample_sources"] == 120
    assert row["max_total_sources"] == 130
    assert row["total_source_count"] == 109
    assert row["sample_source_dc_count"] == 0
    assert row["sample_source_elided_dc_count"] == 7
    assert row["pixel_source_count"] == 93
    assert row["pixel_source_elided_dc_count"] == 7
    assert row["sample_source_pwl_points"] == 25478
    assert row["phase_clock_source_count"] == 5
    assert row["phase_clock_source_pwl_count"] == 0
    assert row["phase_clock_source_pwl_points"] == 0
    assert row["control_source_count"] == 1
    assert row["control_source_pwl_points"] == 4
    assert row["total_source_pwl_points"] == 25482


def test_phase_summary_row_derives_total_source_points_for_older_summaries(tmp_path: Path) -> None:
    module = load_summary_module()
    summary_path = tmp_path / "spice_mnist_local_feature_phase_old_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "continuous_phase_train_no_reference",
                "architecture": "phase_resolved_transient_local_feature_readout",
                "sample_source_pwl_points": 25,
                "phase_clock_source_pwl_points": 15,
                "control_source_pwl_points": 7,
            }
        )
        + "\n"
    )

    row = module.row_from_summary(summary_path)

    assert row["phase_clock_mode"] == "pwl"
    assert row["control_source_pwl_points"] == 7
    assert row["total_source_pwl_points"] == 47


def test_discovery_filters_to_phase_transient_summaries_and_prefers_tables(tmp_path: Path) -> None:
    module = load_summary_module()
    tables = tmp_path / "results" / "tables"
    spice_results = tmp_path / "spice" / "results"
    tables.mkdir(parents=True)
    spice_results.mkdir(parents=True)
    duplicate = "spice_mnist_local_feature_phase_demo_summary.json"
    phase_payload = {
        "status": "continuous_phase_train_no_reference",
        "single_phase_training_transient": True,
    }
    (tables / duplicate).write_text(json.dumps(phase_payload) + "\n")
    (spice_results / duplicate).write_text(json.dumps({**phase_payload, "updates": 99}) + "\n")
    (tables / "spice_mnist_local_feature_phase_train_old_summary.json").write_text("{}\n")
    (tables / "spice_mnist_local_feature_batch_summary.json").write_text("{}\n")

    paths = module.discover_summary_paths(tmp_path)

    assert paths == [tables / duplicate]


def test_target_topology_filter_uses_requested_milestone_c_shape(tmp_path: Path) -> None:
    module = load_summary_module()
    target = {"image_size": 10, "block_size": 4, "stride": 2, "channels": 2}
    other = {"image_size": 8, "block_size": 4, "stride": 2, "channels": 2}

    rows = [
        {"tag": "target", **target},
        {"tag": "other", **other},
    ]

    assert module.filter_rows(rows, target_topology=True) == [{"tag": "target", **target}]


def test_strict_contract_filter_keeps_random_init_no_reference_runs() -> None:
    module = load_summary_module()
    rows = [
        {
            "tag": "strict",
            "fully_on_device_execution_contract_met": True,
            "strict_fully_on_device_contract_met": True,
        },
        {
            "tag": "reference_replay",
            "fully_on_device_execution_contract_met": True,
            "strict_fully_on_device_contract_met": False,
        },
    ]

    assert module.filter_rows(rows, contract_only=True) == rows
    assert module.filter_rows(rows, strict_contract_only=True) == [rows[0]]


def test_strict_target_contract_audit_reports_missing_requirements() -> None:
    module = load_summary_module()
    row = {
        "image_size": 10,
        "block_size": 4,
        "stride": 2,
        "channels": 1,
        "batch_size": 2,
        "strict_fully_on_device_contract_met": False,
        "strict_fully_on_device_requested": False,
        "random_init_used": False,
        "initial_weights_source": "checkpoint",
        "reference_mode": "spice",
        "python_weight_updates_between_samples": True,
        "python_checkpointing_between_samples": True,
    }

    assert module.strict_target_contract_issues(row) == [
        "target_topology",
        "batch_size_1",
        "strict_contract",
        "strict_requested",
        "random_init",
        "random_init_source",
        "reference_mode_none",
        "no_python_weight_updates",
        "no_python_checkpointing",
    ]


def test_phase_summary_full_objective_requires_full_eval_accuracy_and_strict_contract(tmp_path: Path) -> None:
    module = load_summary_module()
    strict_full_path = tmp_path / "spice_mnist_local_feature_phase_full_summary.json"
    strict_full_path.write_text(
        json.dumps(
            {
                "status": "continuous_phase_train_no_reference",
                "architecture": "phase_resolved_transient_local_feature_readout",
                "image_size": 10,
                "block_size": 4,
                "stride": 2,
                "channels": 2,
                "batch_size": 1,
                "eval_samples": 10000,
                "reference_mode": "none",
                "strict_fully_on_device_contract_met": True,
                "strict_fully_on_device_requested": True,
                "random_init_used": True,
                "initial_weights_source": "random_init",
                "python_weight_updates_between_samples": False,
                "python_checkpointing_between_samples": False,
                "phase_eval_accuracy": 0.91,
                "phase_eval_improvement": 0.8,
            }
        )
        + "\n"
    )
    small_eval_path = tmp_path / "spice_mnist_local_feature_phase_smalleval_summary.json"
    small_eval_path.write_text(
        json.dumps(
            {
                "status": "continuous_phase_train_no_reference",
                "architecture": "phase_resolved_transient_local_feature_readout",
                "image_size": 10,
                "block_size": 4,
                "stride": 2,
                "channels": 2,
                "batch_size": 1,
                "eval_samples": 300,
                "reference_mode": "none",
                "strict_fully_on_device_contract_met": True,
                "strict_fully_on_device_requested": True,
                "random_init_used": True,
                "initial_weights_source": "random_init",
                "python_weight_updates_between_samples": False,
                "python_checkpointing_between_samples": False,
                "phase_eval_accuracy": 0.91,
                "phase_eval_improvement": 0.8,
            }
        )
        + "\n"
    )
    weak_accuracy_path = tmp_path / "spice_mnist_local_feature_phase_weak_summary.json"
    weak_accuracy_path.write_text(
        json.dumps(
            {
                "status": "continuous_phase_train_no_reference",
                "architecture": "phase_resolved_transient_local_feature_readout",
                "image_size": 10,
                "block_size": 4,
                "stride": 2,
                "channels": 2,
                "batch_size": 1,
                "eval_samples": 10000,
                "reference_mode": "none",
                "strict_fully_on_device_contract_met": True,
                "strict_fully_on_device_requested": True,
                "random_init_used": True,
                "initial_weights_source": "random_init",
                "python_weight_updates_between_samples": False,
                "python_checkpointing_between_samples": False,
                "phase_eval_accuracy": 0.89,
                "phase_eval_improvement": 0.8,
            }
        )
        + "\n"
    )

    strict_full = module.row_from_summary(strict_full_path)
    small_eval = module.row_from_summary(small_eval_path)
    weak_accuracy = module.row_from_summary(weak_accuracy_path)

    assert strict_full["milestone_d_full_objective_met"] is True
    assert strict_full["full_eval_10k_met"] is True
    assert strict_full["full_objective_accuracy_met"] is True
    assert strict_full["full_objective_accuracy_gap"] == 0.0
    assert small_eval["milestone_d_full_objective_met"] is False
    assert small_eval["full_eval_10k_met"] is False
    assert small_eval["full_objective_accuracy_met"] is True
    assert weak_accuracy["milestone_d_full_objective_met"] is False
    assert weak_accuracy["full_eval_10k_met"] is True
    assert weak_accuracy["full_objective_accuracy_met"] is False
    assert weak_accuracy["full_objective_accuracy_gap"] == 0.010000000000000009


def test_accuracy_sort_ranks_phase_accuracy_then_updates() -> None:
    module = load_summary_module()
    rows = [
        {"tag": "low", "phase_eval_accuracy": 0.2, "updates": 256},
        {"tag": "high_short", "phase_eval_accuracy": 0.4, "updates": 64},
        {"tag": "high_long", "phase_eval_accuracy": 0.4, "updates": 128},
    ]

    assert [row["tag"] for row in module.sort_rows(rows, "accuracy")] == ["high_long", "high_short", "low"]
