from __future__ import annotations

import shutil
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


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("positive", 1.0),
        ("negative", -1.0),
    ],
)
def test_score_decision_primitive_ngspice_polarity(tmp_path: Path, case: str, expected: float) -> None:
    ngspice = shutil.which("ngspice")
    if ngspice is None:
        pytest.skip("ngspice is not installed")

    measures = run_netlist(
        ngspice,
        tmp_path / f"score_decision_{case}.cir",
        decision.generate_netlist(score_case=case, score_center=0.10, score_delta=0.04),
        timeout=20.0,
    )

    margin = float(measures["decision_diff"])
    if expected > 0.0:
        assert margin > 0.05
    else:
        assert margin < -0.05


def test_score_decision_primitive_ngspice_neutral_is_metastable_not_dead_zone(tmp_path: Path) -> None:
    ngspice = shutil.which("ngspice")
    if ngspice is None:
        pytest.skip("ngspice is not installed")

    measures = run_netlist(
        ngspice,
        tmp_path / "score_decision_neutral.cir",
        decision.generate_netlist(score_case="neutral", score_center=0.10, score_delta=0.04),
        timeout=20.0,
    )

    # A regenerative latch has no analog dead zone at exact equality; it resolves
    # according to numerical/device imbalance. Training should not rely on
    # neutral score rails producing a small decision margin.
    assert abs(float(measures["decision_diff"])) > 0.05
