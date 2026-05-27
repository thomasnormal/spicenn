from __future__ import annotations

import sys
from pathlib import Path

import pytest


SPICE_DIR = Path(__file__).resolve().parents[1] / "spice"
sys.path.insert(0, str(SPICE_DIR))

import run_conductance_readout_primitive as readout  # noqa: E402
from run_device_sequential_training import run_netlist  # noqa: E402


def test_conductance_readout_primitive_emits_row_conductance_path() -> None:
    netlist = readout.generate_netlist(readout_case="positive")

    assert "\nB" not in netlist
    assert "Mactrow_n actrow fwd act 0 NMOS" in netlist
    assert "Mactrow_p actrow fwdn act vdd PMOS" in netlist
    assert "Movpos_cond actrow vwp score 0 NMOS W=64u L=180n" in netlist
    assert "Movneg_cond actrow vwn scoren 0 NMOS W=48u L=180n" in netlist
    assert ".meas tran score_margin PARAM='score_after-scoren_after'" in netlist


def test_conductance_readout_primitive_validation() -> None:
    with pytest.raises(ValueError, match="readout_case"):
        readout.generate_netlist(readout_case="bad")
    with pytest.raises(ValueError, match="sum_case"):
        readout.generate_sum_netlist(sum_case="bad")
    with pytest.raises(ValueError, match="isolation"):
        readout.generate_sum_netlist(sum_case="single_positive", isolation="bad")
    with pytest.raises(ValueError, match="sense_mode"):
        readout.generate_sum_netlist(sum_case="single_positive", sense_mode="bad")
    with pytest.raises(ValueError, match="current-clamp"):
        readout.generate_sum_netlist(sum_case="single_positive", sense_mode="current-clamp", include_decision=True)
    with pytest.raises(ValueError, match="readout_width"):
        readout.generate_netlist(readout_case="positive", readout_width=0.0)
    with pytest.raises(ValueError, match="readout_negative_width_scale"):
        readout.generate_sum_netlist(sum_case="single_positive", readout_negative_width_scale=0.0)
    with pytest.raises(ValueError, match="current_clamp_voltage"):
        readout.generate_sum_netlist(sum_case="single_positive", sense_mode="current-clamp", current_clamp_voltage=0.0)
    with pytest.raises(ValueError, match="decision_pullup_width"):
        readout.generate_sum_netlist(sum_case="single_positive", include_decision=True, decision_pullup_width=0.0)
    with pytest.raises(ValueError, match="bias_width"):
        readout.generate_sum_netlist(sum_case="single_positive", include_score_bias=True, bias_width=0.0)
    with pytest.raises(ValueError, match="timeout"):
        readout.main_for_test(["--timeout", "0"])


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("positive", 1.0),
        ("negative", -1.0),
        ("neutral", 0.0),
    ],
)
def test_conductance_readout_primitive_ngspice_score_polarity(
    tmp_path: Path,
    ngspice_path: str,
    case: str,
    expected: float,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / f"conductance_readout_{case}.cir",
        readout.generate_netlist(readout_case=case, positive_weight=0.50, negative_weight=0.34),
        timeout=20.0,
    )

    margin = float(measures["score_margin"])
    if expected > 0.0:
        assert margin > 1e-3
    elif expected < 0.0:
        assert margin < -1e-3
    else:
        assert abs(margin) < 1e-3


def test_conductance_readout_primitive_ngspice_inactive_row_stays_quiet(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "conductance_readout_inactive.cir",
        readout.generate_netlist(readout_case="inactive", positive_weight=0.50, negative_weight=0.34),
        timeout=20.0,
    )

    assert abs(float(measures["score_margin"])) < 1e-3
    assert float(measures["score_common"]) < 5e-3


def _run_sum_case(
    tmp_path: Path,
    ngspice_path: str,
    case: str,
    *,
    isolation: str = "direct",
    score_load_resistance: float = 1e9,
    sense_mode: str = "voltage",
    readout_negative_width_scale: float = 0.75,
) -> dict[str, float]:
    return run_netlist(
        ngspice_path,
        tmp_path / f"conductance_readout_sum_{case}_{isolation}_{score_load_resistance:.0f}.cir",
        readout.generate_sum_netlist(
            sum_case=case,
            positive_weight=0.50,
            negative_weight=0.34,
            isolation=isolation,
            sense_mode=sense_mode,
            readout_negative_width_scale=readout_negative_width_scale,
            score_load_resistance=score_load_resistance,
        ),
        timeout=20.0,
    )


def test_conductance_readout_sum_primitive_emits_current_clamp_probe() -> None:
    netlist = readout.generate_sum_netlist(
        sum_case="single_positive",
        isolation="diode",
        sense_mode="current-clamp",
        readout_negative_width_scale=1.0,
    )

    assert "\nB" not in netlist
    assert "Vscore_clamp score 0 0.1" in netlist
    assert "Vscoren_clamp scoren 0 0.1" in netlist
    assert "Cscore score" not in netlist
    assert "Rscore score" not in netlist
    assert "Movneg0_cond actrow0 vwn0 midn0 0 NMOS W=64u L=180n" in netlist
    assert ".meas tran score_current_margin PARAM='score_current-scoren_current'" in netlist


def test_conductance_readout_sum_primitive_ngspice_direct_floating_is_not_additive(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    single = _run_sum_case(tmp_path, ngspice_path, "single_positive")
    double = _run_sum_case(tmp_path, ngspice_path, "two_positive")

    single_margin = float(single["score_margin"])
    double_margin = float(double["score_margin"])
    assert single_margin > 0.10
    assert abs(double_margin - single_margin) < 5e-3


def test_conductance_readout_sum_primitive_ngspice_cancels_mixed_signs(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    mixed = _run_sum_case(tmp_path, ngspice_path, "mixed_cancel")

    assert abs(float(mixed["score_margin"])) < 2e-3
    assert float(mixed["score_common"]) > 0.05


def test_conductance_readout_sum_primitive_ngspice_direct_inactive_extra_branch_shunts(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    single = _run_sum_case(tmp_path, ngspice_path, "single_positive")
    inactive_extra = _run_sum_case(tmp_path, ngspice_path, "inactive_extra")

    assert float(inactive_extra["score_margin"]) < 0.75 * float(single["score_margin"])


def test_conductance_readout_sum_primitive_ngspice_diode_isolation_blocks_inactive_shunt(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    single = _run_sum_case(tmp_path, ngspice_path, "single_positive", isolation="diode")
    inactive_extra = _run_sum_case(tmp_path, ngspice_path, "inactive_extra", isolation="diode")

    assert float(single["score_margin"]) > 0.05
    assert abs(float(single["score_margin"]) - float(inactive_extra["score_margin"])) < 5e-3


def test_conductance_readout_sum_primitive_ngspice_low_impedance_load_increases_increment(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    single = _run_sum_case(tmp_path, ngspice_path, "single_positive", isolation="diode", score_load_resistance=1e4)
    double = _run_sum_case(tmp_path, ngspice_path, "two_positive", isolation="diode", score_load_resistance=1e4)

    single_margin = float(single["score_margin"])
    double_margin = float(double["score_margin"])
    assert single_margin > 1e-3
    assert double_margin > 2.0 * single_margin


def test_conductance_readout_sum_primitive_ngspice_mid_load_is_additive_and_latchable(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    single = _run_sum_case(tmp_path, ngspice_path, "single_positive", isolation="diode", score_load_resistance=3e4)
    double = _run_sum_case(tmp_path, ngspice_path, "two_positive", isolation="diode", score_load_resistance=3e4)
    inactive_extra = _run_sum_case(tmp_path, ngspice_path, "inactive_extra", isolation="diode", score_load_resistance=3e4)

    single_margin = float(single["score_margin"])
    double_margin = float(double["score_margin"])
    assert single_margin > 0.03
    assert double_margin > 1.8 * single_margin
    assert abs(float(inactive_extra["score_margin"]) - single_margin) < 5e-3


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("single_positive", 1.0),
        ("mixed_cancel", 0.0),
    ],
)
def test_conductance_readout_sum_primitive_ngspice_mid_load_drives_score_latch(
    tmp_path: Path,
    ngspice_path: str,
    case: str,
    expected: float,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / f"conductance_readout_sum_latched_{case}.cir",
        readout.generate_sum_netlist(
            sum_case=case,
            positive_weight=0.50,
            negative_weight=0.34,
            isolation="diode",
            score_load_resistance=3e4,
            include_decision=True,
        ),
        timeout=20.0,
    )

    decision_diff = float(measures["decision_diff"])
    if expected > 0.0:
        assert float(measures["score_margin"]) > 0.03
        assert decision_diff > 0.05
    else:
        assert abs(float(measures["score_margin"])) < 2e-3
        # The regenerative decision latch is not a neutral dead-zone detector.
        assert abs(decision_diff) > 0.05


def test_conductance_readout_sum_primitive_ngspice_current_clamp_is_additive(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    single = _run_sum_case(
        tmp_path,
        ngspice_path,
        "single_positive",
        isolation="diode",
        sense_mode="current-clamp",
        readout_negative_width_scale=1.0,
    )
    double = _run_sum_case(
        tmp_path,
        ngspice_path,
        "two_positive",
        isolation="diode",
        sense_mode="current-clamp",
        readout_negative_width_scale=1.0,
    )
    inactive_extra = _run_sum_case(
        tmp_path,
        ngspice_path,
        "inactive_extra",
        isolation="diode",
        sense_mode="current-clamp",
        readout_negative_width_scale=1.0,
    )

    single_margin = float(single["score_current_margin"])
    double_margin = float(double["score_current_margin"])
    assert single_margin > 1e-7
    assert 1.95 * single_margin < double_margin < 2.05 * single_margin
    assert abs(float(inactive_extra["score_current_margin"]) - single_margin) < 0.02 * single_margin


def test_conductance_readout_sum_primitive_ngspice_current_clamp_cancels_mixed_signs(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    single = _run_sum_case(
        tmp_path,
        ngspice_path,
        "single_positive",
        isolation="diode",
        sense_mode="current-clamp",
        readout_negative_width_scale=1.0,
    )
    mixed = _run_sum_case(
        tmp_path,
        ngspice_path,
        "mixed_cancel",
        isolation="diode",
        sense_mode="current-clamp",
        readout_negative_width_scale=1.0,
    )

    assert abs(float(mixed["score_current_margin"])) < 0.01 * float(single["score_current_margin"])
    assert float(mixed["score_current_common"]) > 1e-7


def test_conductance_readout_sum_primitive_ngspice_conductance_bias_shifts_score_without_erasing_delta(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    unbiased_mixed = _run_sum_case(tmp_path, ngspice_path, "mixed_cancel", isolation="diode", score_load_resistance=3e4)
    biased_mixed = _run_sum_case_with_bias(tmp_path, ngspice_path, "mixed_cancel")
    biased_single = _run_sum_case_with_bias(tmp_path, ngspice_path, "single_positive")

    bias_shift = float(biased_mixed["score_margin"]) - float(unbiased_mixed["score_margin"])
    biased_increment = float(biased_single["score_margin"]) - float(biased_mixed["score_margin"])
    assert bias_shift > 0.015
    assert biased_increment > 0.025


def _run_sum_case_with_bias(tmp_path: Path, ngspice_path: str, case: str) -> dict[str, float]:
    return run_netlist(
        ngspice_path,
        tmp_path / f"conductance_readout_sum_{case}_biased.cir",
        readout.generate_sum_netlist(
            sum_case=case,
            positive_weight=0.50,
            negative_weight=0.34,
            isolation="diode",
            score_load_resistance=3e4,
            include_score_bias=True,
            bias_positive_weight=0.45,
            bias_negative_weight=0.34,
            bias_width=8.0,
        ),
        timeout=20.0,
    )
