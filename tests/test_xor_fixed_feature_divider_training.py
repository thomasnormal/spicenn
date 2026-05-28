from __future__ import annotations

import sys
from pathlib import Path

import pytest


SPICE_DIR = Path(__file__).resolve().parents[1] / "spice"
sys.path.insert(0, str(SPICE_DIR))

import run_xor_fixed_feature_divider_training as xor_fixed  # noqa: E402


def test_xor_fixed_feature_divider_training_netlist_is_transistor_owned() -> None:
    netlist = xor_fixed.xor_fixed_feature_netlist([0, 1, 2, 3])

    assert "\nB" not in netlist
    assert "Vapply" not in netlist
    assert "Iprobref vdd rnorm PWL" in netlist
    assert "Mfeat0_a vdd nx0 feat0_a 0 NSENSE" in netlist
    assert "Mnorm0_score rd0 c0_scorep mir0 0 NSENSE" in netlist
    assert "Merr_c0p_m vdd b1low err_c0p_a vdd PMOS" in netlist
    assert "Mc0_f0_live_pos_up_p c0_vwp0 c0_f0_live_pos_up_ctrl vwhi_ref vdd PMOS" in netlist
    assert ".meas tran train_target_signed_delta_0" in netlist
    assert ".meas tran final_margin_3" in netlist


def test_xor_fixed_feature_divider_training_validation() -> None:
    with pytest.raises(ValueError, match="train_order"):
        xor_fixed.xor_fixed_feature_netlist([])
    with pytest.raises(ValueError, match="0..3"):
        xor_fixed.xor_fixed_feature_netlist([4])
    with pytest.raises(ValueError, match="positive"):
        xor_fixed.xor_fixed_feature_netlist([0], iref_a=0.0)


@pytest.mark.ngspice
def test_xor_fixed_feature_divider_training_ngspice_learns_all_patterns(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    parsed = xor_fixed.run_netlist(
        ngspice_path,
        tmp_path / "xor_fixed_feature_divider_training.cir",
        xor_fixed.xor_fixed_feature_netlist([0, 1, 2, 3]),
        timeout=90.0,
    )

    for pattern in range(4):
        assert abs(parsed[f"initial_margin_{pattern}"]) < 1e-6
        assert parsed[f"final_margin_{pattern}"] > 5e-3
        assert parsed[f"final_margin_improvement_{pattern}"] > 5e-3

    for slot in range(4):
        ir_sum = abs(parsed[f"train_ir0_{slot}"]) + abs(parsed[f"train_ir1_{slot}"])
        assert ir_sum == pytest.approx(1.0e-6, rel=0.08)
        assert 0.0 < parsed[f"train_rnorm_{slot}"] < 0.9
        assert parsed[f"train_target_errp_{slot}"] > parsed[f"train_target_errn_{slot}"] + 20e-3
        assert parsed[f"train_other_errn_{slot}"] > parsed[f"train_other_errp_{slot}"] + 20e-3
        assert parsed[f"train_target_signed_delta_{slot}"] > 1e-3
        assert parsed[f"train_other_signed_delta_{slot}"] < -1e-3
