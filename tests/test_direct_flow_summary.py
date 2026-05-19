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
                "hidden_init": "random",
                "flow_pre_store": "synapse_gate",
                "flow_hidden_write": "direct",
                "hidden_update_width_u": 2e-7,
                "epochs": 2,
                "final_eval_accuracy": 0.9166666667,
                "best_final_transient_accuracy": 0.9166666667,
                "best_final_transient_min_margin_v": -0.0019,
                "input_feature_separability": {"linearly_separable": True},
                "initial_hidden_feature_separability": {"linearly_separable": False},
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
    assert row["flow_pre_store"] == "synapse_gate"
    assert row["flow_hidden_write"] == "direct"
    assert row["input_feature_separable"] is True
    assert row["initial_hidden_separable"] is False
    assert row["final_hidden_separable"] is True
    assert row["final_hidden_min_margin"] == 0.00055
