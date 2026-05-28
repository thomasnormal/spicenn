from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPICE_DIR = ROOT / "spice"
sys.path.insert(0, str(SPICE_DIR))

import run_mnist01_fixed_feature_divider_training as mnist01_live  # noqa: E402


def _require_mnist_raw() -> None:
    raw = ROOT / "data/MNIST/raw"
    required = [
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    ]
    if not all((raw / name).exists() for name in required):
        pytest.skip("raw MNIST IDX files are not available")


def test_mnist01_fixed_feature_records_are_real_balanced_4x4_mnist() -> None:
    _require_mnist_raw()

    train, evals = mnist01_live.load_mnist01_records(
        train_count_per_digit=1,
        eval_count_per_digit=1,
        image_size=4,
    )

    assert [sample["label"] for sample in train] == [0, 1]
    assert [sample["label"] for sample in evals] == [0, 1]
    assert train[0]["mnist_index"] != evals[0]["mnist_index"]
    for sample in train + evals:
        assert len(sample["features"]) == 16
        assert all(0.0 <= value <= 1.1 for value in sample["features"])
        assert max(sample["features"]) > 0.4


def test_mnist01_pixel_downsample_and_voltage_encoding_are_bounded() -> None:
    image = np.zeros((28, 28), dtype=np.float64)
    image[7:14, 7:14] = 0.65

    small = mnist01_live.downsample_image_area(image, 4)
    encoded = [mnist01_live.pixel_to_feature_voltage(value) for value in small.reshape(-1)]

    assert small.shape == (4, 4)
    assert max(encoded) == pytest.approx(1.1)
    assert min(encoded) == 0.0


def test_mnist01_add_complement_features_preserves_records_and_adds_absence_rows() -> None:
    records = [
        {
            "features": [0.0, 0.55, 1.1],
            "label": 1,
            "digit": 1,
            "mnist_index": 7,
        }
    ]

    augmented = mnist01_live.add_complement_features(records, scale=0.5)

    assert records[0]["features"] == [0.0, 0.55, 1.1]
    assert augmented[0]["label"] == 1
    assert augmented[0]["digit"] == 1
    assert augmented[0]["mnist_index"] == 7
    assert augmented[0]["features"] == pytest.approx([0.0, 0.55, 1.1, 0.55, 0.275, 0.0])


def test_mnist01_fixed_feature_netlist_is_live_transistor_path() -> None:
    train = [
        {"features": [1.0, 0.0, 0.0, 0.0], "label": 0},
        {"features": [0.0, 1.0, 0.0, 0.0], "label": 1},
    ]
    evals = train
    netlist = mnist01_live.mnist01_fixed_feature_netlist(train, evals)

    assert "\nB" not in netlist
    assert "Vapply" not in netlist
    assert "gvp" not in netlist
    assert "ghp" not in netlist
    assert "Vpx0 px0 0 PWL" in netlist
    assert "Mfeat0_sample px0 featphi act0 0 NSENSE" in netlist
    assert "Mnorm0_score rd0 c0_scorep mir0 0 NSENSE" in netlist
    assert "Mc0_f0_live_pos_up_p c0_vwp0 c0_f0_live_pos_up_ctrl vwhi_ref vdd PMOS" in netlist
    assert ".meas tran final_margin_improvement_1" in netlist


def test_mnist01_fixed_feature_netlist_validation() -> None:
    with pytest.raises(ValueError, match="empty"):
        mnist01_live.mnist01_fixed_feature_netlist([], [])
    with pytest.raises(ValueError, match="labels"):
        mnist01_live.mnist01_fixed_feature_netlist(
            [{"features": [1.0], "label": 2}],
            [{"features": [1.0], "label": 0}],
        )
    with pytest.raises(ValueError, match="supply"):
        mnist01_live.mnist01_fixed_feature_netlist(
            [{"features": [1.3], "label": 0}],
            [{"features": [1.0], "label": 0}],
        )
    with pytest.raises(ValueError, match="positive"):
        mnist01_live.mnist01_fixed_feature_netlist(
            [{"features": [1.0], "label": 0}],
            [{"features": [1.0], "label": 0}],
            update_width_u=0.0,
        )


@pytest.mark.ngspice
def test_mnist01_fixed_feature_divider_ngspice_improves_two_real_mnist01_margins(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    _require_mnist_raw()
    train, evals = mnist01_live.load_mnist01_records(
        train_count_per_digit=1,
        eval_count_per_digit=1,
        image_size=4,
    )

    parsed = mnist01_live.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_fixed_feature_divider.cir",
        mnist01_live.mnist01_fixed_feature_netlist(train, evals),
        timeout=120.0,
    )

    for sample_idx in range(2):
        assert abs(parsed[f"initial_margin_{sample_idx}"]) < 1e-6
        assert parsed[f"final_margin_{sample_idx}"] > 0.25e-3
        assert parsed[f"final_margin_improvement_{sample_idx}"] > 0.25e-3

    for train_idx in range(2):
        ir_sum = abs(parsed[f"train_ir0_{train_idx}"]) + abs(parsed[f"train_ir1_{train_idx}"])
        assert ir_sum == pytest.approx(1.0e-6, rel=0.08)
        assert parsed[f"train_target_errp_{train_idx}"] > parsed[f"train_target_errn_{train_idx}"] + 30e-3
        assert parsed[f"train_other_errn_{train_idx}"] > parsed[f"train_other_errp_{train_idx}"] + 30e-3
        assert parsed[f"train_target_signed_delta_{train_idx}"] > 1e-3
        assert parsed[f"train_other_signed_delta_{train_idx}"] < -1e-3


@pytest.mark.ngspice
def test_mnist01_fixed_feature_complement_rows_learn_four_real_mnist01_margins(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    _require_mnist_raw()
    train, evals = mnist01_live.load_mnist01_records(
        train_count_per_digit=2,
        eval_count_per_digit=2,
        image_size=4,
    )
    train = mnist01_live.add_complement_features(train, scale=0.5)
    evals = mnist01_live.add_complement_features(evals, scale=0.5)

    parsed = mnist01_live.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_fixed_feature_complement_rows.cir",
        mnist01_live.mnist01_fixed_feature_netlist(train, evals),
        timeout=180.0,
    )

    for sample_idx in range(4):
        assert parsed[f"final_margin_{sample_idx}"] > 1.0e-3
        assert parsed[f"final_margin_improvement_{sample_idx}"] > 1.0e-3
