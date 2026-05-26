from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPICE_DIR = ROOT / "spice"


def test_row_conductance_netlist_uses_differential_conductance_compute() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_row_conductance_primitive as primitive

    netlist = primitive.generate_netlist(wp=0.7, wn=0.25, row=0.85, update_mode="positive", credit_mode="positive")

    assert "\nB" not in netlist
    assert "Vrst rst 0 PULSE(0 1.2 0.0n 10p 10p 0.55n 24n)" in netlist
    assert "Mrow_n row fwd row_src 0 NMOS W=12u L=180n" in netlist
    assert "Mrow_p row fwdn row_src vdd PMOS W=24u L=180n" in netlist
    assert "Mwp_fwd row wp pre_p 0 NMOS W=1u L=180n" in netlist
    assert "Mwn_fwd row wn pre_n 0 NMOS W=1u L=180n" in netlist
    assert "Mhdp_p edp vwp hdp 0 NSENSE W=8u L=180n" in netlist
    assert "Mhdn_p edp vwn hdn 0 NSENSE W=8u L=180n" in netlist
    assert ".meas tran forward_margin PARAM='pre_p_after-pre_n_after'" in netlist
    assert ".meas tran signed_weight_delta PARAM='signed_weight_after-signed_weight_before'" in netlist
    assert ".meas tran hidden_credit_margin PARAM='hdp_after-hdn_after'" in netlist


def test_row_conductance_classification_tracks_expected_signs() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_row_conductance_primitive as primitive

    row = {
        "wp": 0.7,
        "wn": 0.25,
        "readout_wp": 0.7,
        "readout_wn": 0.25,
        "row": 0.85,
        "update_mode": "positive",
        "credit_mode": "positive",
        "forward_margin": 0.1,
        "signed_weight_delta": 0.02,
        "hidden_credit_margin": 0.03,
    }

    assert primitive.classify_row(row, min_abs_margin=0.001) == {
        "forward_classification": "aligned",
        "update_classification": "aligned",
        "hidden_credit_classification": "aligned",
    }
    row["credit_mode"] = "negative"
    row["hidden_credit_margin"] = -0.03
    assert primitive.classify_row(row, min_abs_margin=0.001)["hidden_credit_classification"] == "aligned"
    row["readout_wp"] = 0.4
    row["readout_wn"] = 0.4
    row["hidden_credit_margin"] = 0.0
    assert primitive.classify_row(row, min_abs_margin=0.001)["hidden_credit_classification"] == "dead_zone"
    row["row"] = 0.0
    row["forward_margin"] = 0.0
    assert primitive.classify_row(row, min_abs_margin=0.001)["forward_classification"] == "dead_zone"


def test_row_conductance_cli_validation() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_row_conductance_primitive as primitive

    with pytest.raises(ValueError, match="syn-width"):
        primitive.main_for_test(["--syn-width", "0"])
    with pytest.raises(ValueError, match="timeout"):
        primitive.main_for_test(["--timeout", "0"])
    with pytest.raises(ValueError, match="min-abs-margin"):
        primitive.main_for_test(["--min-abs-margin", "-1"])
