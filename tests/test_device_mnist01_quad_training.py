from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPICE_DIR = ROOT / "spice"


def test_device_mnist01_quad_script_help_runs_from_repo_root() -> None:
    proc = subprocess.run(
        [sys.executable, str(SPICE_DIR / "run_device_mnist01_quad_training.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--train-samples" in proc.stdout
    assert "--eval-samples" in proc.stdout
    assert "--grid-rows" in proc.stdout
    assert "--grid-cols" in proc.stdout
    assert "--output-driver-model" in proc.stdout
    assert "--readout-apply-scale" in proc.stdout
    assert "--assert-nonbehavioral" in proc.stdout


def test_quad_features_are_four_positive_bounded_quadrant_rails() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_quad_training as quad

    image = np.zeros((8, 8), dtype=np.float64)
    image[:4, :4] = 0.3
    image[4:, 4:] = 1.0
    features = quad.quad_features_from_image(image)

    assert features.shape == (4,)
    assert np.all(features > 0.0)
    assert np.all(features <= 1.1)
    assert features[0] > features[1]
    assert features[3] > features[0]


def test_grid_features_generalize_quadrants_row_major() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_quad_training as quad

    image = np.zeros((6, 9), dtype=np.float64)
    image[:, 3:6] = 0.2
    image[:, 6:] = 0.7
    features = quad.grid_features_from_image(image, 2, 3)

    assert features.shape == (6,)
    assert np.all(features > 0.0)
    assert np.all(features <= 1.1)
    assert np.allclose(features[:3], features[3:])
    assert features[0] < features[1] < features[2]


def test_sample_wave_has_strictly_increasing_time_points() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_quad_training as quad

    wave = quad.sample_wave([{"x0": 0.1}, {"x0": 0.9}, {"x0": 0.2}], "x0", stop_ns=48.0)
    pairs = wave.removeprefix("PWL(").removesuffix(")").split()
    times = [float(token.removesuffix("n")) for token in pairs[0::2]]

    assert times == sorted(times)
    assert len(times) == len(set(times))


def test_quad_netlist_uses_four_transistor_feature_slices_and_no_behavioral_sources() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_quad_training as quad

    samples = [
        {"x0": 0.7, "x1": 0.8, "x2": 0.9, "x3": 1.0, "target": 1.1},
        {"x0": 0.6, "x1": 0.65, "x2": 0.7, "x3": 0.75, "target": 0.0},
    ]
    netlist = quad.quad_netlist(samples, quad.initial_quad_weights(), training_enabled=True, readout_apply_scale=0.5)

    assert "\nB" not in netlist
    assert netlist.count("Cwhp") == quad.FEATURE_COUNT
    assert "Vx0 x0 0 PWL" in netlist
    assert "Vx3 x3 0 PWL" in netlist
    assert "Movpos3_f op3_1 fwd score 0 NREL" in netlist
    assert "Mvwp0_up_p0 vdd rgp0 vwp0_up vdd PMOS W=4u" in netlist
    assert "Mvwn0_dn_a vwn0 apply vwn0_dn 0 NREL W=1u" in netlist
    assert "Mrelu_o vdd score out 0 NSENSE" in netlist


def test_quad_netlist_supports_larger_grid_feature_count_without_behavioral_sources() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_quad_training as quad

    feature_count = quad.grid_feature_count(3, 3)
    samples = [{f"x{q}": 0.65 + 0.02 * q for q in range(feature_count)} | {"target": 1.1}]
    netlist = quad.quad_netlist(
        samples,
        quad.initial_quad_weights(feature_count),
        training_enabled=True,
        readout_apply_scale=0.5,
    )

    assert "\nB" not in netlist
    assert netlist.count("Cwhp") == feature_count
    assert "Vx8 x8 0 PWL" in netlist
    assert "Movpos8_f op8_1 fwd score 0 NREL" in netlist
    assert "Vx9 x9 0 PWL" not in netlist


def test_quad_netlist_rejects_nonpositive_readout_apply_scale() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import pytest
    import run_device_mnist01_quad_training as quad

    with pytest.raises(ValueError, match="readout_apply_scale"):
        quad.quad_netlist(
            [{"x0": 0.7, "x1": 0.8, "x2": 0.9, "x3": 1.0, "target": 1.1}],
            quad.initial_quad_weights(),
            training_enabled=True,
            readout_apply_scale=0.0,
        )


def test_quad_netlist_rejects_missing_grid_feature_rail() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import pytest
    import run_device_mnist01_quad_training as quad

    with pytest.raises(ValueError, match="sample 0 missing feature rails: x4"):
        quad.quad_netlist(
            [{"x0": 0.7, "x1": 0.8, "x2": 0.9, "x3": 1.0, "target": 1.1}],
            quad.initial_quad_weights(5),
            training_enabled=True,
        )
