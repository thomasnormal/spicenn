from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPICE_DIR = ROOT / "spice"


def test_alignment_netlist_can_emit_latch_free_weighted_credit() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_readout_hidden_credit_alignment as align

    netlist = align.generate_netlist(mode="readout-weighted", vwp=0.37, vwn=0.33, width=12.0)

    assert "\nB" not in netlist
    assert "rvwp" not in netlist
    assert "rvwn" not in netlist
    assert "Mhdp_pv_w hdp_pv_e vwp hdp_pv_w 0 NSENSE W=12u" in netlist
    assert "Mhdp_nv_w hdp_nv_e vwn hdp_nv_w 0 NSENSE W=12u" in netlist
    assert "Mhdn_pv_w hdn_pv_e vwn hdn_pv_w 0 NSENSE W=12u" in netlist
    assert "Mhdn_nv_w hdn_nv_e vwp hdn_nv_w 0 NSENSE W=12u" in netlist
    assert ".meas tran hidden_credit_margin PARAM='hdp_after-hdn_after'" in netlist


def test_alignment_netlist_can_emit_restored_hardgate_credit() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_readout_hidden_credit_alignment as align

    netlist = align.generate_netlist(mode="readout-restored-hardgate", vwp=0.37, vwn=0.33, width=12.0)

    assert "\nB" not in netlist
    assert "Mrvwp_p rvwp rvwn vdd vdd PMOS W=6u" in netlist
    assert "Mrvwp_n rvwp vwn rvw_src 0 NSENSE W=6u" in netlist
    assert "Mrvwn_n rvwn vwp rvw_src 0 NSENSE W=6u" in netlist
    assert "Mhdp_pv_w hdp_pv_e rvwp hdp_pv_w 0 NMOS W=12u" in netlist
    assert "Mhdn_pv_w hdn_pv_e rvwn hdn_pv_w 0 NMOS W=12u" in netlist
    assert ".meas tran rvw_margin PARAM='rvwp_after-rvwn_after'" in netlist


def test_alignment_classification_tracks_sign_and_dead_zone() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_readout_hidden_credit_alignment as align

    assert align.classify_row({"delta": 0.02, "hidden_credit_margin": 0.01}, min_abs_margin=0.001) == "aligned"
    assert align.classify_row({"delta": -0.02, "hidden_credit_margin": -0.01}, min_abs_margin=0.001) == "aligned"
    assert align.classify_row({"delta": 0.02, "hidden_credit_margin": -0.01}, min_abs_margin=0.001) == "flipped"
    assert align.classify_row({"delta": 0.02, "hidden_credit_margin": 0.0001}, min_abs_margin=0.001) == "weak"
    assert align.classify_row({"delta": 0.0, "hidden_credit_margin": 0.0001}, min_abs_margin=0.001) == "dead_zone"
    assert align.classify_row({"delta": 0.0, "hidden_credit_margin": 0.01}, min_abs_margin=0.001) == "biased"


def test_alignment_cli_validation() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_readout_hidden_credit_alignment as align

    with pytest.raises(ValueError, match="width"):
        align.main_for_test(["--width", "0"])
    with pytest.raises(ValueError, match="deltas"):
        align.main_for_test(["--deltas", ""])
