from __future__ import annotations

import sys
from pathlib import Path

import pytest


SPICE_DIR = Path(__file__).resolve().parents[1] / "spice"
sys.path.insert(0, str(SPICE_DIR))

import run_multiclass_margin_correction_primitive as margin  # noqa: E402
from run_device_sequential_training import run_netlist  # noqa: E402


def test_multiclass_margin_correction_primitive_emits_margin_shifted_pairwise_writer() -> None:
    netlist = margin.generate_netlist(case="target1_barely_wins", target_margin_v=1.0e-3)

    assert "\nB" not in netlist
    assert "Vc1_score_raw c1_score_raw 0 0.004" in netlist
    assert "Vc1_score c1_score 0 0.003" in netlist
    assert "Vc1_targetp c1_targetp 0 1.2" in netlist
    assert "Cc1_gt_c0_decision c1_gt_c0_decision 0 4f IC=0" in netlist
    assert "Mc0_gt_c1_scoreamp_scoren_p c0_gt_c1_scoren_amp c1_score c0_gt_c1_scoreamp_scoren_i vdd PMOS" in netlist
    assert "Mmt1_o0_errp_win mt1_o0_errp_t c0_gt_c1_decision mt1_o0_errp_w 0 NSENSE W=128u" in netlist
    assert "Mmt1_o0_errn_clk mt1_o0_errn_w scoreerr c0_errn 0 NSENSE W=128u" in netlist
    assert ".meas tran c1_errdiff PARAM='c1_errp_after-c1_errn_after'" in netlist


def test_multiclass_margin_correction_primitive_validation() -> None:
    with pytest.raises(ValueError, match="case"):
        margin.generate_netlist(case="bad")
    with pytest.raises(ValueError, match="class_count"):
        margin.generate_netlist(case="target1_clear", class_count=4)
    with pytest.raises(ValueError, match="score_values"):
        margin.generate_netlist(case="target1_clear", score_values=(0.1, 0.2))
    with pytest.raises(ValueError, match="target_class"):
        margin.generate_netlist(case="target1_clear", target_class=3)
    with pytest.raises(ValueError, match="supply rails"):
        margin.generate_netlist(case="target1_clear", score_values=(0.1, 1.3, 0.2))
    with pytest.raises(ValueError, match="target-margin-v"):
        margin.main_for_test(["--target-margin-v", "0"])
    with pytest.raises(ValueError, match="error-drive-scale"):
        margin.main_for_test(["--error-drive-scale", "0"])


def test_multiclass_margin_correction_primitive_ngspice_leaves_clear_target_quiet(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "margin_target1_clear.cir",
        margin.generate_netlist(case="target1_clear"),
        timeout=20.0,
    )

    assert abs(float(measures["c0_errdiff"])) < 0.025
    assert abs(float(measures["c1_errdiff"])) < 0.025
    assert abs(float(measures["c2_errdiff"])) < 0.025


def test_multiclass_margin_correction_primitive_ngspice_updates_barely_winning_target(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "margin_target1_barely_wins.cir",
        margin.generate_netlist(case="target1_barely_wins"),
        timeout=20.0,
    )

    assert float(measures["c1_errdiff"]) > 0.025
    assert float(measures["c0_errdiff"]) < -0.025
    assert abs(float(measures["c2_errdiff"])) < 0.025
    assert float(measures["c1_errdiff"]) < 0.15
    assert float(measures["c0_errdiff"]) > -0.15


@pytest.mark.parametrize(
    ("case", "target", "offender", "quiet"),
    [
        ("target1_loses0", 1, 0, 2),
        ("target1_loses2", 1, 2, 0),
        ("target0_loses2", 0, 2, 1),
        ("target2_loses1", 2, 1, 0),
    ],
)
def test_multiclass_margin_correction_primitive_ngspice_corrects_offending_class_only(
    tmp_path: Path,
    ngspice_path: str,
    case: str,
    target: int,
    offender: int,
    quiet: int,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / f"margin_{case}.cir",
        margin.generate_netlist(case=case),
        timeout=20.0,
    )

    assert float(measures[f"c{target}_errdiff"]) > 0.025
    assert float(measures[f"c{offender}_errdiff"]) < -0.025
    assert abs(float(measures[f"c{quiet}_errdiff"])) < 0.025


def test_multiclass_margin_correction_primitive_ngspice_common_mode_preserves_direction(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    low_common = run_netlist(
        ngspice_path,
        tmp_path / "margin_low_common.cir",
        margin.generate_netlist(
            case="target1_loses0",
            score_values=(0.0060, 0.0040, 0.0020),
            target_class=1,
        ),
        timeout=20.0,
    )
    higher_common = run_netlist(
        ngspice_path,
        tmp_path / "margin_higher_common.cir",
        margin.generate_netlist(
            case="target1_loses0",
            score_values=(0.0260, 0.0240, 0.0220),
            target_class=1,
        ),
        timeout=20.0,
    )

    assert float(low_common["c1_errdiff"]) > 0.025
    assert float(low_common["c0_errdiff"]) < -0.025
    assert abs(float(low_common["c2_errdiff"])) < 0.025
    assert float(higher_common["c1_errdiff"]) > 0.025
    assert float(higher_common["c0_errdiff"]) < -0.025
    assert abs(float(higher_common["c2_errdiff"])) < 0.025
