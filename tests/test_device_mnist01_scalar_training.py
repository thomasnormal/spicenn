from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPICE_DIR = ROOT / "spice"


def test_device_mnist01_scalar_script_help_runs_from_repo_root() -> None:
    proc = subprocess.run(
        [sys.executable, str(SPICE_DIR / "run_device_mnist01_scalar_training.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--train-samples" in proc.stdout
    assert "--eval-samples" in proc.stdout
    assert "--assert-nonbehavioral" in proc.stdout


def test_scalar_feature_from_image_is_positive_bounded_and_monotone() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_scalar_training as mnist01

    dark = np.zeros((8, 8), dtype=np.float64)
    mid = np.full((8, 8), 0.2, dtype=np.float64)
    bright = np.ones((8, 8), dtype=np.float64)

    assert 0.0 < mnist01.scalar_feature_from_image(dark) < mnist01.scalar_feature_from_image(mid)
    assert mnist01.scalar_feature_from_image(mid) < mnist01.scalar_feature_from_image(bright)
    assert mnist01.scalar_feature_from_image(bright) <= 1.1


def test_balanced_digit_indices_are_stable_balanced_and_shuffled() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_scalar_training as mnist01

    labels = np.asarray([0, 1, 2, 0, 1, 0, 1, 1, 0, 2])
    first = mnist01.balanced_digit_indices(labels, 6, seed=4, digits=(0, 1))
    second = mnist01.balanced_digit_indices(labels, 6, seed=4, digits=(0, 1))

    assert first.tolist() == second.tolist()
    assert sorted(labels[first].tolist()) == [0, 0, 0, 1, 1, 1]
    assert first.tolist() != sorted(first.tolist())


def test_binary_accuracy_uses_spice_output_threshold_against_positive_label() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_scalar_training as mnist01

    rows = pd.DataFrame(
        [
            {"out_after": 0.2, "positive_label": 1.0},
            {"out_after": 0.0, "positive_label": 0.0},
            {"out_after": 0.1, "positive_label": 0.0},
            {"out_after": 0.0, "positive_label": 1.0},
        ]
    )

    assert mnist01.binary_accuracy(rows, threshold=0.05) == 0.5
