from __future__ import annotations

import sys
from pathlib import Path

import pytest


SPICE_DIR = Path(__file__).resolve().parents[1] / "spice"
sys.path.insert(0, str(SPICE_DIR))

import run_score_decision_primitive as decision  # noqa: E402
from run_device_sequential_training import run_netlist  # noqa: E402


def test_score_decision_primitive_emits_direct_score_sensing_latch() -> None:
    netlist = decision.generate_netlist(score_case="positive")

    assert "\nB" not in netlist
    assert "Vscore score 0 0.12" in netlist
    assert "Vscoren scoren 0 0.08" in netlist
    assert "Mprecharge_decision decision rstfn vdd vdd PMOS W=4u L=180n" in netlist
    assert "Mdec_scorepc_n decision scoren dec_src 0 NSENSE W=12u L=180n" in netlist
    assert "Mdecn_scorepc_n decisionn score dec_src 0 NSENSE W=12u L=180n" in netlist
    assert "Mdec_scorepc_tail dec_src dec 0 0 NMOS W=12u L=180n" in netlist


def test_score_decision_primitive_validation() -> None:
    with pytest.raises(ValueError, match="score_case"):
        decision.generate_netlist(score_case="bad")
    with pytest.raises(ValueError, match="score rails"):
        decision.generate_netlist(score_case="positive", score_center=1.19, score_delta=0.04)
    with pytest.raises(ValueError, match="timeout"):
        decision.main_for_test(["--timeout", "0"])
    with pytest.raises(ValueError, match="scoren-pulldown-scale"):
        decision.main_for_test(["--scoren-pulldown-scale", "0"])
    with pytest.raises(ValueError, match="decision_topology"):
        decision.generate_netlist(score_case="positive", decision_topology="bad")


def test_score_decision_primitive_emits_two_phase_reject_reference_stage() -> None:
    netlist = decision.generate_netlist(
        score_case="positive",
        decision_topology="score-diff-reject-ref",
        reject_ref=0.075,
        scoren_pulldown_scale=0.35,
    )

    assert "\nB" not in netlist
    assert "Voutref outref 0 0.075" in netlist
    assert "Vdec2 dec2 0 PULSE" in netlist
    assert "Mprecharge_decision_pre decision_pre rstfn vdd vdd PMOS W=4u" in netlist
    assert "Mdec_scorepre_n decision_pre scoren dec_src 0 NSENSE W=4.2u" in netlist
    assert "Mdecn_scorepre_n decisionn_pre score dec_src 0 NSENSE W=12u" in netlist
    assert "Mdec_reject_n decision decisionn_pre dec2_src 0 NSENSE W=12u" in netlist
    assert "Mdecn_reject_n decisionn outref dec2_src 0 NSENSE W=12u" in netlist
    assert "Mdec_reject_tail dec2_src dec2 0 0 NMOS W=12u" in netlist
    assert ".meas tran decision_pre_after FIND V(decision_pre) AT=4.5n" in netlist


def test_score_decision_primitive_emits_pre_regeneration_window_stage() -> None:
    netlist = decision.generate_netlist(
        score_case="positive",
        decision_topology="score-diff-window",
        reject_ref=0.075,
    )

    assert "\nB" not in netlist
    assert "Voutref outref 0 0.075" in netlist
    assert "Mdec_win_pos_scoren decision scoren pos_src 0 NSENSE W=12u" in netlist
    assert "Mdec_win_pos_ref decision outref pos_src 0 NSENSE W=12u" in netlist
    assert "Mdecn_win_pos_score decision_posn score pos_src 0 NSENSE W=12u" in netlist
    assert "Mdec_win_neg_score decisionn score neg_src 0 NSENSE W=12u" in netlist
    assert "Mdec_win_neg_ref decisionn outref neg_src 0 NSENSE W=12u" in netlist
    assert "Mdecn_win_neg_scoren decision_negn scoren neg_src 0 NSENSE W=12u" in netlist
    assert ".meas tran positive_window_diff PARAM='decision_after-decision_posn_after'" in netlist
    assert ".meas tran negative_window_diff PARAM='decisionn_after-decision_negn_after'" in netlist


def test_score_decision_primitive_emits_gain_before_window_stage() -> None:
    netlist = decision.generate_netlist(
        score_case="tiny_positive",
        decision_topology="score-diff-gain-window",
        reject_ref=0.05,
    )

    assert "\nB" not in netlist
    assert "Vscore score 0 0.1011" in netlist
    assert "Vscoren scoren 0 0.0989" in netlist
    assert "Cscore_amp score_amp 0 8f IC=1.2" in netlist
    assert "Mprecharge_score_amp score_amp rstfn vdd vdd PMOS W=4u" in netlist
    assert "Mprecharge_scoren_amp scoren_amp rstfn vdd vdd PMOS W=4u" in netlist
    assert "Mscoreamp_score score_amp score scoreamp_score_i 0 NSENSE W=1u" in netlist
    assert "Mscoreamp_scoren scoren_amp scoren scoreamp_scoren_i 0 NSENSE W=1u" in netlist
    assert "Mscoreamp_score_tail scoreamp_score_i amp 0 0 NMOS W=8u" in netlist
    assert "Mscoreamp_scoren_tail scoreamp_scoren_i amp 0 0 NMOS W=8u" in netlist
    assert "Mdec_gain_win_pos_scoreamp decision score_amp pos_src 0 NSENSE W=12u" in netlist
    assert "Mdecn_gain_win_pos_scorenamp decision_posn scoren_amp pos_src 0 NSENSE W=12u" in netlist
    assert "Mdec_gain_win_neg_scorenamp decisionn scoren_amp neg_src 0 NSENSE W=12u" in netlist
    assert "Mdecn_gain_win_neg_scoreamp decision_negn score_amp neg_src 0 NSENSE W=12u" in netlist
    assert ".meas tran score_gain_diff PARAM='scoren_amp_after-score_amp_after'" in netlist


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("positive", 1.0),
        ("negative", -1.0),
    ],
)
def test_score_decision_primitive_ngspice_polarity(
    tmp_path: Path,
    ngspice_path: str,
    case: str,
    expected: float,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / f"score_decision_{case}.cir",
        decision.generate_netlist(score_case=case, score_center=0.10, score_delta=0.04),
        timeout=20.0,
    )

    margin = float(measures["decision_diff"])
    if expected > 0.0:
        assert margin > 0.05
    else:
        assert margin < -0.05


def test_score_decision_primitive_ngspice_neutral_is_metastable_not_dead_zone(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "score_decision_neutral.cir",
        decision.generate_netlist(score_case="neutral", score_center=0.10, score_delta=0.04),
        timeout=20.0,
    )

    # A regenerative latch has no analog dead zone at exact equality; it resolves
    # according to numerical/device imbalance. Training should not rely on
    # neutral score rails producing a small decision margin.
    assert abs(float(measures["decision_diff"])) > 0.05


def test_score_decision_primitive_ngspice_balance_recenters_shifted_scores(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    default_positive = run_netlist(
        ngspice_path,
        tmp_path / "score_decision_shifted_positive_default.cir",
        decision.generate_netlist(
            score_case="shifted_positive",
            score_center=0.20,
            score_delta=0.07,
        ),
        timeout=20.0,
    )
    balanced_positive = run_netlist(
        ngspice_path,
        tmp_path / "score_decision_shifted_positive_balanced.cir",
        decision.generate_netlist(
            score_case="shifted_positive",
            score_center=0.20,
            score_delta=0.07,
            scoren_pulldown_scale=0.25,
        ),
        timeout=20.0,
    )
    balanced_negative = run_netlist(
        ngspice_path,
        tmp_path / "score_decision_shifted_negative_balanced.cir",
        decision.generate_netlist(
            score_case="shifted_negative",
            score_center=0.20,
            score_delta=0.07,
            scoren_pulldown_scale=0.25,
        ),
        timeout=20.0,
    )

    assert float(default_positive["decision_diff"]) < -0.05
    assert float(balanced_positive["decision_diff"]) > 0.05
    assert float(balanced_negative["decision_diff"]) < -0.05


def test_score_decision_primitive_ngspice_reject_reference_stage_resolves_clear_cases(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    positive = run_netlist(
        ngspice_path,
        tmp_path / "score_decision_reject_positive.cir",
        decision.generate_netlist(
            score_case="neutral",
            score=0.148332,
            scoren=0.084363,
            decision_topology="score-diff-reject-ref",
            scoren_pulldown_scale=0.35,
            reject_ref=0.075,
        ),
        timeout=20.0,
    )
    negative = run_netlist(
        ngspice_path,
        tmp_path / "score_decision_reject_negative.cir",
        decision.generate_netlist(
            score_case="neutral",
            score=0.112036,
            scoren=0.209658,
            decision_topology="score-diff-reject-ref",
            scoren_pulldown_scale=0.35,
            reject_ref=0.075,
        ),
        timeout=20.0,
    )

    assert float(positive["decisionn_pre_after"]) < 0.075
    assert float(positive["decision_diff"]) > 0.05
    assert float(negative["decisionn_pre_after"]) > 0.075
    assert float(negative["decision_diff"]) < -0.05


def test_score_decision_primitive_ngspice_window_stage_rejects_near_zero_before_regeneration(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    positive = run_netlist(
        ngspice_path,
        tmp_path / "score_decision_window_positive.cir",
        decision.generate_netlist(
            score_case="neutral",
            score=0.148332,
            scoren=0.084363,
            decision_topology="score-diff-window",
            reject_ref=0.075,
        ),
        timeout=20.0,
    )
    negative = run_netlist(
        ngspice_path,
        tmp_path / "score_decision_window_negative.cir",
        decision.generate_netlist(
            score_case="neutral",
            score=0.0748,
            scoren=0.1938,
            decision_topology="score-diff-window",
            reject_ref=0.075,
        ),
        timeout=20.0,
    )
    near_zero = run_netlist(
        ngspice_path,
        tmp_path / "score_decision_window_near_zero.cir",
        decision.generate_netlist(
            score_case="neutral",
            score=0.14007,
            scoren=0.13993,
            decision_topology="score-diff-window",
            reject_ref=0.075,
        ),
        timeout=20.0,
    )

    assert float(positive["positive_window_diff"]) > 0.05
    assert float(positive["negative_window_diff"]) < -0.05
    assert float(positive["decision_diff"]) > 0.05

    assert float(negative["positive_window_diff"]) < -0.05
    assert float(negative["negative_window_diff"]) > 0.05
    assert float(negative["decision_diff"]) < -0.05

    assert float(near_zero["positive_window_diff"]) < -0.05
    assert float(near_zero["negative_window_diff"]) < -0.05
    assert abs(float(near_zero["decision_diff"])) < 0.05


def test_score_decision_primitive_ngspice_gain_window_amplifies_tiny_margins_and_rejects_tiny_neutral(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    tiny_positive = run_netlist(
        ngspice_path,
        tmp_path / "score_decision_gain_window_tiny_positive.cir",
        decision.generate_netlist(
            score_case="tiny_positive",
            decision_topology="score-diff-gain-window",
            reject_ref=0.05,
        ),
        timeout=20.0,
    )
    tiny_neutral = run_netlist(
        ngspice_path,
        tmp_path / "score_decision_gain_window_tiny_neutral.cir",
        decision.generate_netlist(
            score_case="tiny_neutral",
            decision_topology="score-diff-gain-window",
            reject_ref=0.05,
        ),
        timeout=20.0,
    )
    tiny_negative = run_netlist(
        ngspice_path,
        tmp_path / "score_decision_gain_window_tiny_negative.cir",
        decision.generate_netlist(
            score_case="tiny_negative",
            decision_topology="score-diff-gain-window",
            reject_ref=0.05,
        ),
        timeout=20.0,
    )
    direct_tiny_positive = run_netlist(
        ngspice_path,
        tmp_path / "score_decision_window_tiny_positive_direct.cir",
        decision.generate_netlist(
            score_case="tiny_positive",
            decision_topology="score-diff-window",
            reject_ref=0.05,
        ),
        timeout=20.0,
    )

    assert float(direct_tiny_positive["positive_window_diff"]) < -0.05

    assert float(tiny_positive["score_gain_diff"]) > 0.04
    assert float(tiny_positive["positive_window_diff"]) > 0.05
    assert float(tiny_positive["negative_window_diff"]) < -0.05
    assert float(tiny_positive["decision_diff"]) > 0.05

    assert float(tiny_negative["score_gain_diff"]) < -0.04
    assert float(tiny_negative["positive_window_diff"]) < -0.05
    assert float(tiny_negative["negative_window_diff"]) > 0.05
    assert float(tiny_negative["decision_diff"]) < -0.05

    assert abs(float(tiny_neutral["score_gain_diff"])) < 0.05
    assert float(tiny_neutral["positive_window_diff"]) < -0.05
    assert float(tiny_neutral["negative_window_diff"]) < -0.05
    assert abs(float(tiny_neutral["decision_diff"])) < 0.05
