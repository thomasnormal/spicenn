from __future__ import annotations

import sys
from pathlib import Path

import pytest


SPICE_DIR = Path(__file__).resolve().parents[1] / "spice"
sys.path.insert(0, str(SPICE_DIR))

import run_multiclass_score_competition_primitive as comp  # noqa: E402
from run_device_sequential_training import run_netlist  # noqa: E402


def test_multiclass_score_competition_primitive_emits_pairwise_error_rails() -> None:
    netlist = comp.generate_netlist(case="target1_low_wrong0")

    assert "\nB" not in netlist
    assert "Vc0_score c0_score 0 0.0075" in netlist
    assert "Vc1_targetp c1_targetp 0 1.2" in netlist
    assert "Cc1_gt_c0_decision c1_gt_c0_decision 0 4f IC=0" in netlist
    assert "Mc0_gt_c1_scoreamp_score_p c0_gt_c1_score_amp c0_score c0_gt_c1_scoreamp_score_i vdd PMOS" in netlist
    assert "Mt1_o0_errp_sup t1_o0_errp_sup c1_gt_c0_decision vdd vdd PMOS W=128u" in netlist
    assert "Mt1_o0_errp_win t1_o0_errp_t c0_gt_c1_decision t1_o0_errp_w 0 NSENSE W=128u" in netlist
    assert "Mt1_o0_errp_clk t1_o0_errp_w scoreerr c1_errp 0 NSENSE W=128u" in netlist
    assert "Mt1_o0_errn_clk t1_o0_errn_w scoreerr c0_errn 0 NSENSE W=128u" in netlist
    assert ".meas tran c1_errdiff PARAM='c1_errp_after-c1_errn_after'" in netlist


def test_multiclass_score_competition_primitive_validation() -> None:
    with pytest.raises(ValueError, match="case"):
        comp.generate_netlist(case="bad")
    with pytest.raises(ValueError, match="class_count"):
        comp.generate_netlist(case="target1_clear", class_count=4)
    with pytest.raises(ValueError, match="score_values"):
        comp.generate_netlist(case="target1_clear", score_values=(0.1, 0.2))
    with pytest.raises(ValueError, match="target_class"):
        comp.generate_netlist(case="target1_clear", target_class=3)
    with pytest.raises(ValueError, match="supply rails"):
        comp.generate_netlist(case="target1_clear", score_values=(0.1, 1.3, 0.2))
    with pytest.raises(ValueError, match="pairwise-width"):
        comp.main_for_test(["--pairwise-width", "0"])
    with pytest.raises(ValueError, match="error-width"):
        comp.main_for_test(["--error-width", "0"])
    with pytest.raises(ValueError, match="error-capacitance-f"):
        comp.main_for_test(["--error-capacitance-f", "0"])


@pytest.mark.parametrize(
    ("case", "winner", "loser"),
    [
        ("target1_clear", 1, 0),
        ("target1_clear", 1, 2),
        ("target1_low_wrong0", 0, 1),
        ("target2_wrong1", 1, 2),
    ],
)
def test_multiclass_score_competition_primitive_ngspice_pairwise_polarity(
    tmp_path: Path,
    ngspice_path: str,
    case: str,
    winner: int,
    loser: int,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / f"score_competition_{case}_{winner}_gt_{loser}.cir",
        comp.generate_netlist(case=case),
        timeout=20.0,
    )

    assert float(measures[f"c{winner}_gt_c{loser}_diff"]) > 0.10


def test_multiclass_score_competition_primitive_ngspice_corrects_target_below_runnerup(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "score_competition_target1_low_wrong0.cir",
        comp.generate_netlist(case="target1_low_wrong0"),
        timeout=20.0,
    )

    assert float(measures["c1_errdiff"]) > 0.025
    assert float(measures["c0_errdiff"]) < -0.025
    assert float(measures["c2_errdiff"]) < -0.025


def test_multiclass_score_competition_primitive_ngspice_leaves_clear_target_in_dead_zone(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "score_competition_target1_clear.cir",
        comp.generate_netlist(case="target1_clear"),
        timeout=20.0,
    )

    assert abs(float(measures["c0_errdiff"])) < 0.025
    assert abs(float(measures["c1_errdiff"])) < 0.025
    assert abs(float(measures["c2_errdiff"])) < 0.025


def test_multiclass_score_competition_primitive_ngspice_near_tie_builds_margin(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "score_competition_near_tie_target1.cir",
        comp.generate_netlist(case="near_tie_target1"),
        timeout=20.0,
    )

    assert float(measures["c1_errdiff"]) > 0.025
    assert float(measures["c0_errdiff"]) < -0.025
    assert float(measures["c2_errdiff"]) < -0.025


def test_multiclass_score_competition_primitive_ngspice_pairwise_margin_grows_with_gap(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    small_gap = run_netlist(
        ngspice_path,
        tmp_path / "score_competition_target1_small_gap.cir",
        comp.generate_netlist(
            case="target1_low_wrong0",
            score_values=(0.0025, 0.0015, 0.0010),
            target_class=1,
        ),
        timeout=20.0,
    )
    large_gap = run_netlist(
        ngspice_path,
        tmp_path / "score_competition_target1_large_gap.cir",
        comp.generate_netlist(
            case="target1_low_wrong0",
            score_values=(0.0075, 0.0015, 0.0010),
            target_class=1,
        ),
        timeout=20.0,
    )

    assert float(large_gap["c0_gt_c1_diff"]) > float(small_gap["c0_gt_c1_diff"]) + 0.005
