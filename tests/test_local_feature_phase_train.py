import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPICE_DIR = ROOT / "spice"
if str(SPICE_DIR) not in sys.path:
    sys.path.insert(0, str(SPICE_DIR))

import run_spice_mnist_local_feature_phase_train as phase_train  # noqa: E402


def test_cli_exposes_simulator_selector() -> None:
    proc = subprocess.run(
        [sys.executable, str(SPICE_DIR / "run_spice_mnist_local_feature_phase_train.py"), "--help"],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--simulator" in proc.stdout


def test_eval_accuracy_improved_requires_finite_strict_improvement() -> None:
    assert phase_train.eval_accuracy_improved(0.9, None)
    assert phase_train.eval_accuracy_improved(0.91, 0.9)
    assert not phase_train.eval_accuracy_improved(0.9, 0.9)
    assert not phase_train.eval_accuracy_improved(0.89, 0.9)
    assert not phase_train.eval_accuracy_improved(float("nan"), 0.9)


def test_maybe_save_best_eval_checkpoint_writes_only_on_enabled_improvement(tmp_path: Path) -> None:
    path = tmp_path / "best_weights.npz"
    state = (
        np.arange(8, dtype=float).reshape(1, 2, 4),
        np.array([[0.1, 0.2]]),
        np.arange(20, dtype=float).reshape(10, 1, 2),
        np.linspace(-0.5, 0.5, 10),
    )

    best, fields = phase_train.maybe_save_best_eval_checkpoint(False, path, state, 0.9, None)
    assert best is None
    assert fields == {}
    assert not path.exists()

    best, fields = phase_train.maybe_save_best_eval_checkpoint(True, path, state, 0.9, None)
    assert best == pytest.approx(0.9)
    assert fields == {"best_eval_checkpoint": str(path)}
    with np.load(path) as checkpoint:
        assert checkpoint["local_weights"] == pytest.approx(state[0])
        assert checkpoint["local_bias"] == pytest.approx(state[1])
        assert checkpoint["readout"] == pytest.approx(state[2])
        assert checkpoint["output_bias"] == pytest.approx(state[3])

    path.unlink()
    best, fields = phase_train.maybe_save_best_eval_checkpoint(True, path, state, 0.89, best)
    assert best == pytest.approx(0.9)
    assert fields == {}
    assert not path.exists()
