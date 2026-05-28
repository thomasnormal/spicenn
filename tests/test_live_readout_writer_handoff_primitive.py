from __future__ import annotations

import sys
from pathlib import Path

import pytest


SPICE_DIR = Path(__file__).resolve().parents[1] / "spice"
sys.path.insert(0, str(SPICE_DIR))

import run_live_readout_writer_handoff_primitive as handoff  # noqa: E402
from run_device_sequential_training import run_netlist  # noqa: E402


def test_live_readout_writer_handoff_primitive_uses_real_pmos_differential_writer() -> None:
    netlist = handoff.generate_netlist(cases=handoff.MEASURED_COMMON_MODE_CASES, waveform="pwl")

    assert "\nB" not in netlist
    assert "Vapply" not in netlist
    assert "Mc0_f0_live_pos_up_p c0_vwp0 c0_f0_live_pos_up_ctrl vwhi_ref vdd PMOS" in netlist
    assert "Mc1_f0_live_neg_up_p c1_vwn0 c1_f0_live_neg_up_ctrl vwhi_ref vdd PMOS" in netlist
    assert "Mc0_f0_live_pos_dn_select c0_f0_live_pos_dn c0_f0_live_neg_up_ctrl c0_f0_live_pos_dn_sel 0 NSENSE" in netlist
    assert "Mc1_f0_live_neg_dn_select c1_f0_live_neg_dn c1_f0_live_pos_up_ctrl c1_f0_live_neg_dn_sel 0 NSENSE" in netlist
    assert "Mc0_f0_live_pos_up_ctrl_latch" in netlist
    assert ".meas tran c0_pos_ctrl_min MIN V(c0_f0_live_pos_up_ctrl)" in netlist
    assert ".meas tran c1_neg_ctrl_min MIN V(c1_f0_live_neg_up_ctrl)" in netlist


def test_live_readout_writer_handoff_primitive_validation() -> None:
    with pytest.raises(ValueError, match="waveform"):
        handoff.generate_netlist(waveform="bad")
    with pytest.raises(SystemExit):
        handoff.main_for_test(["--case-set", "bad"])
    with pytest.raises(ValueError, match="update-width"):
        handoff.main_for_test(["--update-width", "0"])
    with pytest.raises(ValueError, match="references"):
        handoff.generate_netlist(low_ref=0.5, high_ref=0.4)
    with pytest.raises(ValueError, match="PWL edge"):
        handoff.generate_netlist(waveform="pwl", eligibility_skew_ns=-2.0)


@pytest.mark.ngspice
def test_live_readout_writer_handoff_ngspice_ideal_rails_select_correct_direction(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "live_readout_writer_handoff_ideal.cir",
        handoff.generate_netlist(waveform="dc"),
        timeout=30.0,
    )
    rows = handoff.rows_from_measures(measures, handoff.DEFAULT_CASES)

    assert all(
        handoff.case_passed(row, min_signed_delta_v=1e-3, ctrl_on_max_v=0.65, ctrl_off_min_v=0.9)
        for row in rows
    )


@pytest.mark.parametrize("eligibility_skew_ns", [0.0, 0.05, -0.05])
@pytest.mark.ngspice
def test_live_readout_writer_handoff_ngspice_measured_common_mode_keeps_sign_under_small_skew(
    tmp_path: Path,
    ngspice_path: str,
    eligibility_skew_ns: float,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / f"live_readout_writer_handoff_measured_skew_{eligibility_skew_ns:+.2f}.cir",
        handoff.generate_netlist(
            cases=handoff.MEASURED_COMMON_MODE_CASES,
            waveform="pwl",
            update_width_u=4.0,
            eligibility_skew_ns=eligibility_skew_ns,
        ),
        timeout=30.0,
    )
    rows = handoff.rows_from_measures(measures, handoff.MEASURED_COMMON_MODE_CASES)

    assert all(
        handoff.case_passed(row, min_signed_delta_v=0.2e-3, ctrl_on_max_v=0.65, ctrl_off_min_v=0.75)
        for row in rows
    )


@pytest.mark.ngspice
def test_live_readout_writer_handoff_ngspice_depressed_negative_state_still_moves_more_negative(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    case = handoff.HandoffCase(
        "depressed_nontarget",
        positive_drive_v=0.28155,
        negative_drive_v=0.35536,
        eligibility_v=1.19968,
        support_v=1.2,
        expected_sign=-1,
    )
    measures = run_netlist(
        ngspice_path,
        tmp_path / "live_readout_writer_handoff_depressed_nontarget.cir",
        handoff.generate_netlist(
            cases=(case,),
            waveform="pwl",
            update_width_u=0.25,
            initial_positive=0.22,
            initial_negative=0.352,
            high_ref=0.48,
            low_ref=0.22,
        ),
        timeout=30.0,
    )
    rows = handoff.rows_from_measures(measures, (case,))

    assert rows[0]["signed_delta_v"] < -0.2e-3
    assert rows[0]["neg_ctrl_min_v"] < 0.65
    assert rows[0]["pos_ctrl_min_v"] > 0.75
