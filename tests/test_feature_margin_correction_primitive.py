from __future__ import annotations

import sys
from pathlib import Path

import pytest


SPICE_DIR = Path(__file__).resolve().parents[1] / "spice"
sys.path.insert(0, str(SPICE_DIR))

import run_feature_margin_correction_primitive as featmargin  # noqa: E402
from run_device_sequential_training import run_netlist  # noqa: E402


def test_feature_margin_correction_primitive_emits_live_gradient_flow_writer() -> None:
    netlist = featmargin.generate_netlist(case="wrong_feature_active")

    assert "\nB" not in netlist
    assert "Vapply" not in netlist
    assert "_gvp" not in netlist
    assert "_gvn" not in netlist
    assert "Mf0_wrong_contrib_cond actrow c0_vwp0 f0_wrong_contrib 0 NMOS" in netlist
    assert "Mf0_target_contrib_cond actrow c1_vwp0 f0_target_contrib 0 NMOS" in netlist
    assert "Mf0_dec_pair_tail f0_dec_src scoredec 0 0 NMOS" in netlist
    assert "Mf0_correction_gate_boot_p f0_correction_gate f0_correction_gate_boot_ctrl vdd vdd PMOS" in netlist
    assert "Mf0_correction_gate_target_discharge f0_correction_gate f0_target_ge_wrong 0 0 NMOS" in netlist
    assert "Mc1_f0_live_pos_up_p c1_vwp0 c1_f0_live_pos_up_ctrl vwhi_ref vdd PMOS" in netlist
    assert "Mc0_f0_live_neg_up_p c0_vwn0 c0_f0_live_neg_up_ctrl vwhi_ref vdd PMOS" in netlist
    assert ".meas tran target_signed_delta" in netlist
    assert ".meas tran wrong_signed_delta" in netlist


def test_feature_margin_correction_primitive_validation() -> None:
    with pytest.raises(ValueError, match="case"):
        featmargin.generate_netlist(case="bad")
    with pytest.raises(ValueError, match="feature_idx"):
        featmargin.generate_netlist(case="wrong_feature_active", feature_idx=-1)
    with pytest.raises(ValueError, match="device widths"):
        featmargin.generate_netlist(case="wrong_feature_active", writer_width_u=0.0)
    with pytest.raises(ValueError, match="device widths"):
        featmargin.generate_netlist(case="wrong_feature_active", bootstrap_width_u=0.0)
    with pytest.raises(ValueError, match="writer-width-u"):
        featmargin.main_for_test(["--writer-width-u", "0"])
    with pytest.raises(ValueError, match="bootstrap-width-u"):
        featmargin.main_for_test(["--bootstrap-width-u", "0"])
    with pytest.raises(ValueError, match="min-delta-v"):
        featmargin.main_for_test(["--min-delta-v", "0"])


def test_feature_margin_correction_ngspice_updates_only_wrong_contributing_feature(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "feature_margin_wrong_active.cir",
        featmargin.generate_netlist(case="wrong_feature_active"),
        timeout=20.0,
    )

    assert float(measures["contrib_margin"]) > 0.020
    assert float(measures["wrong_gt_target_after"]) > float(measures["target_ge_wrong_after"])
    assert float(measures["target_signed_delta"]) > 0.010
    assert float(measures["wrong_signed_delta"]) < -0.010


def test_feature_margin_correction_ngspice_quiet_when_target_feature_already_stronger(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "feature_margin_target_active.cir",
        featmargin.generate_netlist(case="target_feature_active"),
        timeout=20.0,
    )

    assert float(measures["contrib_margin"]) < -0.020
    assert float(measures["target_ge_wrong_after"]) > float(measures["wrong_gt_target_after"])
    assert abs(float(measures["target_signed_delta"])) < 0.010
    assert abs(float(measures["wrong_signed_delta"])) < 0.010


def test_feature_margin_correction_ngspice_bootstraps_tied_active_feature(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "feature_margin_tied_active.cir",
        featmargin.generate_netlist(case="tied_feature_active"),
        timeout=20.0,
    )

    assert abs(float(measures["contrib_margin"])) < 0.010
    assert float(measures["target_signed_delta"]) > 0.005
    assert float(measures["wrong_signed_delta"]) < -0.005


def test_feature_margin_correction_ngspice_quiet_when_feature_inactive(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "feature_margin_inactive.cir",
        featmargin.generate_netlist(case="inactive_feature"),
        timeout=20.0,
    )

    assert abs(float(measures["contrib_margin"])) < 0.010
    assert abs(float(measures["target_signed_delta"])) < 0.010
    assert abs(float(measures["wrong_signed_delta"])) < 0.010


def test_feature_margin_correction_ngspice_rotated_classes_keep_same_direction(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "feature_margin_rotated.cir",
        featmargin.generate_netlist(case="rotated_wrong_feature_active"),
        timeout=20.0,
    )

    assert float(measures["contrib_margin"]) > 0.020
    assert float(measures["target_signed_delta"]) > 0.010
    assert float(measures["wrong_signed_delta"]) < -0.010
