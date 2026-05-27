from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPICE_DIR = ROOT / "spice"


def _run_ngspice_case(tmp_path: Path, ngspice_path: str, name: str, **kwargs: float | str) -> dict[str, float]:
    sys.path.insert(0, str(SPICE_DIR))
    import run_row_conductance_primitive as primitive
    from run_device_sequential_training import run_netlist

    return run_netlist(
        ngspice_path,
        tmp_path / f"row_conductance_{name}.cir",
        primitive.generate_netlist(**kwargs),
        timeout=20.0,
    )


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
    assert "Mpre_p_rst pre_p rst 0 0 NMOS W=6u L=180n" in netlist
    assert "Mpre_n_rst pre_n rst 0 0 NMOS W=6u L=180n" in netlist
    assert "Mwp_up_e vdd ep wp_up_e 0 NSENSE W=0.25u L=180n" in netlist
    assert "Mhdp_p edp vwp hdp 0 NSENSE W=8u L=180n" in netlist
    assert "Mhdn_p edp vwn hdn 0 NSENSE W=8u L=180n" in netlist
    assert ".meas tran forward_margin PARAM='pre_p_after-pre_n_after'" in netlist
    assert ".meas tran signed_weight_delta PARAM='signed_weight_after-signed_weight_before'" in netlist
    assert ".meas tran hidden_credit_margin PARAM='hdp_after-hdn_after'" in netlist

    sequence = primitive.generate_netlist(
        wp=0.45,
        wn=0.40,
        cycles=2,
        cycle_rows=[0.85, 0.0],
        cycle_update_modes=["positive", "none"],
    )
    assert "\nB" not in sequence
    assert "Vrow_src row_src 0 PWL(" in sequence
    assert "Vep ep 0 PWL(" in sequence
    assert "Ven en 0 PWL(" in sequence
    assert "25.01n 0" in sequence


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
    with pytest.raises(ValueError, match="cycles"):
        primitive.main_for_test(["--cycles", "0"])
    with pytest.raises(ValueError, match="cycle_rows"):
        primitive.generate_netlist(wp=0.45, wn=0.40, cycles=2, cycle_rows=[0.85])
    with pytest.raises(ValueError, match="cycle_update_modes"):
        primitive.generate_netlist(wp=0.45, wn=0.40, cycles=2, cycle_update_modes=["positive", "bad"])


@pytest.mark.parametrize(
    ("name", "wp", "wn", "row", "expected_sign"),
    [
        ("positive_weight", 0.70, 0.25, 0.85, 1.0),
        ("negative_weight", 0.25, 0.70, 0.85, -1.0),
        ("inactive_row", 0.70, 0.25, 0.0, 0.0),
    ],
)
def test_row_conductance_primitive_ngspice_forward_polarity(
    tmp_path: Path,
    ngspice_path: str,
    name: str,
    wp: float,
    wn: float,
    row: float,
    expected_sign: float,
) -> None:
    measures = _run_ngspice_case(
        tmp_path,
        ngspice_path,
        f"forward_{name}",
        wp=wp,
        wn=wn,
        row=row,
        update_mode="none",
        credit_mode="none",
    )

    margin = float(measures["forward_margin"])
    if expected_sign > 0.0:
        assert margin > 0.10
        assert float(measures["pre_p_after"]) > 0.10
        assert float(measures["pre_n_after"]) < 1e-3
    elif expected_sign < 0.0:
        assert margin < -0.10
        assert float(measures["pre_n_after"]) > 0.10
        assert float(measures["pre_p_after"]) < 1e-3
    else:
        assert abs(margin) < 1e-3


def test_row_conductance_primitive_ngspice_repeated_cycle_resets_dynamic_nodes_and_retains_weights(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = _run_ngspice_case(
        tmp_path,
        ngspice_path,
        "repeated_cycle_reset_retention",
        wp=0.70,
        wn=0.25,
        row=0.85,
        update_mode="none",
        credit_mode="none",
        cycles=2,
    )

    assert float(measures["forward_margin"]) > 0.10
    assert float(measures["pre_p_after_cycle2_reset"]) < 1e-3
    assert float(measures["pre_n_after_cycle2_reset"]) < 1e-3
    assert float(measures["forward_margin_cycle2"]) > 0.10
    assert float(measures["pre_p_after_cycle2"]) > 0.10
    assert float(measures["pre_n_after_cycle2"]) < 1e-3
    assert abs(float(measures["signed_weight_drift_cycle2"])) < 1e-3


def test_row_conductance_primitive_ngspice_sequence_update_persists_across_next_sample(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = _run_ngspice_case(
        tmp_path,
        ngspice_path,
        "sequence_positive_then_hold",
        wp=0.45,
        wn=0.40,
        cycles=2,
        cycle_rows=[0.85, 0.85],
        cycle_update_modes=["positive", "none"],
        update_mode="none",
        credit_mode="none",
    )

    first_delta = float(measures["signed_weight_delta"])
    final_delta = float(measures["signed_weight_drift_cycle2"])
    assert first_delta > 0.05
    assert abs(final_delta - first_delta) < 2e-3
    assert float(measures["pre_p_after_cycle2_reset"]) < 1e-3
    assert float(measures["pre_n_after_cycle2_reset"]) < 1e-3
    assert float(measures["forward_margin_cycle2"]) > 0.03


def test_row_conductance_primitive_ngspice_sequence_opposite_update_cancels_signed_state(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = _run_ngspice_case(
        tmp_path,
        ngspice_path,
        "sequence_positive_then_negative",
        wp=0.45,
        wn=0.40,
        cycles=2,
        cycle_rows=[0.85, 0.85],
        cycle_update_modes=["positive", "negative"],
        update_mode="none",
        credit_mode="none",
    )

    assert float(measures["signed_weight_delta"]) > 0.05
    assert abs(float(measures["signed_weight_drift_cycle2"])) < 0.01


@pytest.mark.parametrize(
    ("mode", "expected_sign"),
    [
        ("positive", 1.0),
        ("negative", -1.0),
        ("none", 0.0),
    ],
)
def test_row_conductance_primitive_ngspice_update_moves_stored_weight(
    tmp_path: Path,
    ngspice_path: str,
    mode: str,
    expected_sign: float,
) -> None:
    measures = _run_ngspice_case(
        tmp_path,
        ngspice_path,
        f"update_{mode}",
        wp=0.45,
        wn=0.40,
        row=0.85,
        update_mode=mode,
        credit_mode="none",
    )

    delta = float(measures["signed_weight_delta"])
    if expected_sign > 0.0:
        assert delta > 0.05
    elif expected_sign < 0.0:
        assert delta < -0.05
    else:
        assert abs(delta) < 1e-3


@pytest.mark.parametrize(
    ("name", "credit_mode", "readout_wp", "readout_wn", "expected_sign"),
    [
        ("positive_error_positive_weight", "positive", 0.55, 0.30, 1.0),
        ("positive_error_negative_weight", "positive", 0.30, 0.55, -1.0),
        ("negative_error_negative_weight", "negative", 0.30, 0.55, 1.0),
        ("neutral_weight_dead_zone", "positive", 0.40, 0.40, 0.0),
    ],
)
def test_row_conductance_primitive_ngspice_latch_free_hidden_credit(
    tmp_path: Path,
    ngspice_path: str,
    name: str,
    credit_mode: str,
    readout_wp: float,
    readout_wn: float,
    expected_sign: float,
) -> None:
    measures = _run_ngspice_case(
        tmp_path,
        ngspice_path,
        f"credit_{name}",
        wp=0.45,
        wn=0.40,
        row=0.85,
        update_mode="none",
        credit_mode=credit_mode,
        readout_wp=readout_wp,
        readout_wn=readout_wn,
    )

    margin = float(measures["hidden_credit_margin"])
    if expected_sign > 0.0:
        assert margin > 0.05
        assert float(measures["hdp_after"]) > float(measures["hdn_after"])
    elif expected_sign < 0.0:
        assert margin < -0.05
        assert float(measures["hdn_after"]) > float(measures["hdp_after"])
    else:
        assert abs(margin) < 1e-3
