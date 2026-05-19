import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spice"))

from run_spice_mnist_settling_pareto import (  # noqa: E402
    accuracy_at_time,
    frontier_targets,
    pareto_frontier,
    steady_state_accuracy,
)


def test_finite_readout_time_can_outperform_steady_state():
    evidence = np.array([[0.0, 1.0]])
    output_bias = np.array([0.6, 0.0])
    y = np.array([0])

    transient_acc, transient_correct = accuracy_at_time(
        evidence,
        output_bias,
        y,
        0.5e-9,
        1.0e-9,
        0.25e-9,
    )
    steady_acc, steady_correct = steady_state_accuracy(evidence, output_bias, y)

    assert transient_correct == 1
    assert transient_acc == 1.0
    assert steady_correct == 0
    assert steady_acc == 0.0


def test_pareto_frontier_and_targets_report_fastest_matching_time():
    df = pd.DataFrame(
        [
            {"readout_time_ns": 0.0, "accuracy": 0.4, "correct": 4, "total": 10, "tau_act_ns": 1.0, "tau_score_ns": 1.0},
            {"readout_time_ns": 1.0, "accuracy": 0.6, "correct": 6, "total": 10, "tau_act_ns": 1.0, "tau_score_ns": 1.0},
            {"readout_time_ns": 1.0, "accuracy": 0.5, "correct": 5, "total": 10, "tau_act_ns": 0.5, "tau_score_ns": 1.0},
            {"readout_time_ns": 2.0, "accuracy": 0.55, "correct": 5, "total": 10, "tau_act_ns": 0.5, "tau_score_ns": 0.5},
            {"readout_time_ns": 3.0, "accuracy": 0.8, "correct": 8, "total": 10, "tau_act_ns": 0.5, "tau_score_ns": 0.5},
        ]
    )

    frontier = pareto_frontier(df)
    assert frontier["readout_time_ns"].to_list() == [0.0, 1.0, 3.0]
    assert frontier["accuracy"].to_list() == [0.4, 0.6, 0.8]

    targets = frontier_targets(frontier, best_accuracy=0.8, margins_pp=[0.0, 20.0])
    exact = targets[targets["within_best_pp"] == 0.0].iloc[0]
    within_20pp = targets[targets["within_best_pp"] == 20.0].iloc[0]

    assert exact["readout_time_ns"] == 3.0
    assert exact["accuracy"] == 0.8
    assert within_20pp["readout_time_ns"] == 1.0
    assert within_20pp["accuracy"] == 0.6
