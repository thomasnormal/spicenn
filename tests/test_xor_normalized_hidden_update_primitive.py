from __future__ import annotations

import sys
from pathlib import Path

import pytest


SPICE_DIR = Path(__file__).resolve().parents[1] / "spice"
sys.path.insert(0, str(SPICE_DIR))

import run_xor_normalized_hidden_update_primitive as xor_hidden  # noqa: E402


def _run_case(
    tmp_path: Path,
    ngspice_path: str,
    *,
    name: str,
    evidence: tuple[float, float],
    target: int,
    **kwargs: float | str,
) -> dict[str, float]:
    netlist = xor_hidden.generate_netlist(
        evidence=evidence,
        target=target,
        **kwargs,
    )
    assert "\nB" not in netlist
    assert "Vapply" not in netlist
    return xor_hidden.run_netlist(
        ngspice_path,
        tmp_path / f"{name}.cir",
        netlist,
        timeout=30.0,
    )


def test_xor_normalized_hidden_update_primitive_uses_real_subcircuits() -> None:
    netlist = xor_hidden.generate_netlist(
        evidence=(1.0, 3.0),
        target=0,
        readout_polarity="positive",
        conductance_floor=0.0,
    )

    assert "\nB" not in netlist
    assert "Vapply" not in netlist
    assert ".subckt xor_current_to_writer_descent2 " in netlist
    assert "Xwroute rnorm rd0 rd1 t0 t1 phip c0_errp c0_errn c1_errp c1_errn vdd 0 xor_current_to_writer_descent2" in netlist
    assert "Mh0_c0_hdp_pv_e vdd c0_errp" in netlist
    assert "Mh0_c1_hdn_nv_w h0_c1_nn_e c1_vwp0" in netlist
    assert "Mh0_live_pos_up_e vwhi_ref xelig0" in netlist
    assert "Mh0_live_neg_up_d h0_live_neg_up h0_hdn whn0" in netlist


def test_xor_normalized_hidden_update_primitive_validation() -> None:
    with pytest.raises(ValueError, match="two values"):
        xor_hidden.generate_netlist(evidence=(1.0,), target=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonnegative"):
        xor_hidden.generate_netlist(evidence=(-1.0, 1.0), target=0)
    with pytest.raises(ValueError, match="target"):
        xor_hidden.generate_netlist(evidence=(1.0, 1.0), target=2)
    with pytest.raises(ValueError, match="readout_polarity"):
        xor_hidden.generate_netlist(evidence=(1.0, 1.0), target=0, readout_polarity="bad")
    with pytest.raises(ValueError, match="conductance_floor"):
        xor_hidden.generate_netlist(evidence=(1.0, 1.0), target=0, conductance_floor=-0.1)
    with pytest.raises(ValueError, match="supply rails"):
        xor_hidden.generate_netlist(evidence=(1.0, 1.0), target=0, activation_level=1.3)


def test_conductance_divider_error_drives_readout_weighted_hidden_credit_sign(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    positive = _run_case(
        tmp_path,
        ngspice_path,
        name="hidden_positive",
        evidence=(1.0, 3.0),
        target=0,
        readout_polarity="positive",
        conductance_floor=0.0,
    )
    negative = _run_case(
        tmp_path,
        ngspice_path,
        name="hidden_negative",
        evidence=(1.0, 3.0),
        target=0,
        readout_polarity="negative",
        conductance_floor=0.0,
    )
    neutral = _run_case(
        tmp_path,
        ngspice_path,
        name="hidden_neutral",
        evidence=(1.0, 3.0),
        target=0,
        readout_polarity="neutral",
        conductance_floor=0.0,
    )

    for measures in (positive, negative, neutral):
        ir0 = abs(measures["ir0_raw"])
        ir1 = abs(measures["ir1_raw"])
        assert ir0 + ir1 == pytest.approx(1.0e-6, rel=0.08)
        assert measures["e0p_after"] > 1.0
        assert measures["e1n_after"] > 1.0
        assert measures["e0n_after"] < 1.0e-6
        assert measures["e1p_after"] < 1.0e-6

    assert positive["hcredit_diff"] > 40e-3
    assert positive["signed_delta"] > 20e-3
    assert negative["hcredit_diff"] < -40e-3
    assert negative["signed_delta"] < -20e-3
    assert abs(neutral["hcredit_diff"]) < 2e-6
    assert abs(neutral["signed_delta"]) < 0.5e-3


def test_label_permutation_preserves_hidden_update_direction(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    target0 = _run_case(
        tmp_path,
        ngspice_path,
        name="target0_positive_credit",
        evidence=(1.0, 3.0),
        target=0,
        readout_polarity="positive",
        conductance_floor=0.0,
    )
    target1 = _run_case(
        tmp_path,
        ngspice_path,
        name="target1_positive_credit",
        evidence=(3.0, 1.0),
        target=1,
        readout_polarity="negative",
        conductance_floor=0.0,
    )

    assert target0["e0p_after"] > 1.0
    assert target0["e1n_after"] > 1.0
    assert target1["e1p_after"] > 1.0
    assert target1["e0n_after"] > 1.0
    assert target0["hcredit_diff"] > 40e-3
    assert target1["hcredit_diff"] > 40e-3
    assert target1["hcredit_diff"] == pytest.approx(target0["hcredit_diff"], rel=0.05)
    assert target0["signed_delta"] > 20e-3
    assert target1["signed_delta"] == pytest.approx(target0["signed_delta"], rel=0.08)


def test_hidden_update_requires_activation_and_eligibility(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    inactive_activation = _run_case(
        tmp_path,
        ngspice_path,
        name="inactive_activation",
        evidence=(1.0, 3.0),
        target=0,
        readout_polarity="positive",
        conductance_floor=0.0,
        activation_level=0.0,
    )
    inactive_eligibility = _run_case(
        tmp_path,
        ngspice_path,
        name="inactive_eligibility",
        evidence=(1.0, 3.0),
        target=0,
        readout_polarity="positive",
        conductance_floor=0.0,
        eligibility_level=0.0,
    )

    assert abs(inactive_activation["hcredit_diff"]) < 1e-6
    assert abs(inactive_activation["signed_delta"]) < 0.5e-3
    assert inactive_eligibility["hcredit_diff"] > 40e-3
    assert abs(inactive_eligibility["signed_delta"]) < 0.5e-3


def test_hidden_credit_tracks_normalized_wrong_probability(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    clear = _run_case(
        tmp_path,
        ngspice_path,
        name="clear_target",
        evidence=(3.0, 1.0),
        target=0,
        readout_polarity="positive",
        conductance_floor=0.0,
    )
    tie = _run_case(
        tmp_path,
        ngspice_path,
        name="tie_target",
        evidence=(1.0, 1.0),
        target=0,
        readout_polarity="positive",
        conductance_floor=0.0,
    )
    wrong = _run_case(
        tmp_path,
        ngspice_path,
        name="wrong_target",
        evidence=(1.0, 3.0),
        target=0,
        readout_polarity="positive",
        conductance_floor=0.0,
    )

    assert abs(clear["hcredit_diff"]) < 1e-6
    assert tie["hcredit_diff"] > clear["hcredit_diff"] + 5e-3
    assert wrong["hcredit_diff"] > tie["hcredit_diff"] + 40e-3
    assert abs(clear["signed_delta"]) < 0.5e-3
    assert abs(tie["signed_delta"]) < 0.5e-3
    assert wrong["signed_delta"] > 20e-3
