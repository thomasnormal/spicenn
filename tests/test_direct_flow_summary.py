from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_SCRIPT = ROOT / "experiments" / "26_summarize_direct_flow_results.py"


def load_summary_module():
    spec = importlib.util.spec_from_file_location("summarize_direct_flow_results", SUMMARY_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direct_flow_summary_extracts_nested_separability(tmp_path: Path) -> None:
    module = load_summary_module()
    summary_path = tmp_path / "device_mnist01_24_random_hidden_v190_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "dataset": "mnist01_24",
                "input_frontend_key": "pool2",
                "hidden_cells": 8,
                "hidden_forward_mode": "rail_buffer",
                "hidden_init": "random",
                "flow_pre_store": "synapse_gate",
                "flow_hidden_write": "direct",
                "hidden_update_width_u": 2e-7,
                "epochs": 2,
                "order_mode": "interleave",
                "error_rule": "out_residual",
                "backward_gate_mode": "target_mistake",
                "target_mistake_bwd_match_fraction": 0.875,
                "target_mistake_bwd_false_positive_count": 2,
                "target_mistake_bwd_false_negative_count": 1,
                "target_mistake_bwd_voltage_separation_v": -0.03,
                "target_mistake_bwd_best_threshold_v": 0.42,
                "target_mistake_bwd_best_threshold_match_fraction": 0.9375,
                "target_mistake_latch_match_fraction": 0.8125,
                "target_mistake_latch_false_positive_count": 3,
                "target_mistake_latch_false_negative_count": 2,
                "target_mistake_latch_best_threshold_match_fraction": 0.875,
                "readout_flow_polarity": "normal",
                "readout_flow_write_mode": "bounded_charge_discharge",
                "hidden_flow_write_mode": "bounded_discharge",
                "readout_center_pull_width_u": 0.0002,
                "output_bias_center_pull_width_u": 0.0001,
                "readout_center_pull_v": 0.64,
                "readout_write_high_v": 0.58,
                "readout_write_low_v": 0.16,
                "final_eval_accuracy": 0.9166666667,
                "best_final_transient_accuracy": 0.9166666667,
                "best_final_transient_min_margin_v": -0.0019,
                "input_feature_separability": {"linearly_separable": True, "min_margin": 0.42},
                "initial_hidden_feature_separability": {
                    "linearly_separable": False,
                    "best_min_margin": -0.01,
                },
                "final_hidden_feature_separability": {
                    "linearly_separable": True,
                    "min_margin": 0.00055,
                },
                "max_abs_total_readout_signed_delta_v": 0.08,
                "max_abs_total_hidden_signed_delta_v": 0.023,
                "wall_time_s": 650.0,
            }
        )
        + "\n"
    )

    row = module.row_from_summary(summary_path)

    assert row["tag"] == "device_mnist01_24_random_hidden_v190"
    assert row["dataset"] == "mnist01_24"
    assert row["hidden_forward_mode"] == "rail_buffer"
    assert row["flow_pre_store"] == "synapse_gate"
    assert row["flow_hidden_write"] == "direct"
    assert row["order_mode"] == "interleave"
    assert row["error_rule"] == "out_residual"
    assert row["backward_gate_mode"] == "target_mistake"
    assert row["target_mistake_bwd_match_fraction"] == 0.875
    assert row["target_mistake_bwd_false_positive_count"] == 2
    assert row["target_mistake_bwd_false_negative_count"] == 1
    assert row["target_mistake_bwd_voltage_separation_v"] == -0.03
    assert row["target_mistake_bwd_best_threshold_v"] == 0.42
    assert row["target_mistake_bwd_best_threshold_match_fraction"] == 0.9375
    assert row["target_mistake_latch_match_fraction"] == 0.8125
    assert row["target_mistake_latch_false_positive_count"] == 3
    assert row["target_mistake_latch_false_negative_count"] == 2
    assert row["target_mistake_latch_best_threshold_match_fraction"] == 0.875
    assert row["readout_flow_polarity"] == "normal"
    assert row["readout_flow_write_mode"] == "bounded_charge_discharge"
    assert row["hidden_flow_write_mode"] == "bounded_discharge"
    assert row["readout_center_pull_width_u"] == 0.0002
    assert row["output_bias_center_pull_width_u"] == 0.0001
    assert row["readout_center_pull_v"] == 0.64
    assert row["readout_write_high_v"] == 0.58
    assert row["readout_write_low_v"] == 0.16
    assert row["input_feature_separable"] is True
    assert row["input_feature_min_margin"] == 0.42
    assert row["initial_hidden_separable"] is False
    assert row["initial_hidden_min_margin"] == -0.01
    assert row["final_hidden_separable"] is True
    assert row["final_hidden_min_margin"] == 0.00055
