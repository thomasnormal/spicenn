import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spice"))

from run_spice_mnist_local_feature_settling_pareto import (  # noqa: E402
    load_local_feature_checkpoint,
    local_feature_evidence,
)
from run_spice_mnist_settling_pareto import accuracy_at_time, steady_state_accuracy  # noqa: E402


def test_local_feature_transient_can_outperform_steady_state():
    x = np.array([[1.0]])
    local_weights = np.array([[[10.0]]])
    local_bias = np.array([[0.0]])
    readout = np.array([[[0.0]], [[1.0]]])
    output_bias = np.array([0.6, 0.0])
    y = np.array([0])

    evidence = local_feature_evidence(x, local_weights, local_bias, readout, [[0]])
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


def test_load_local_feature_checkpoint_validates_shapes(tmp_path):
    path = tmp_path / "weights.npz"
    np.savez_compressed(
        path,
        local_weights=np.zeros((2, 3, 4)),
        local_bias=np.zeros((2, 3)),
        readout=np.zeros((10, 2, 3)),
        output_bias=np.zeros(10),
    )

    local_weights, local_bias, readout, output_bias = load_local_feature_checkpoint(path, 10, 2, 4)

    assert local_weights.shape == (2, 3, 4)
    assert local_bias.shape == (2, 3)
    assert readout.shape == (10, 2, 3)
    assert output_bias.shape == (10,)

    with pytest.raises(ValueError, match="local_weights"):
        load_local_feature_checkpoint(path, 10, 3, 4)
