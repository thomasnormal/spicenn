from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPICE_DIR = ROOT / "spice"


def test_shared_phase_cell_robustness_netlist_adds_margin_and_stress_hooks() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_shared_phase_cell_robustness as robust

    netlist = robust.generate_netlist(
        robust.CellStress(
            dvdd=0.01,
            dx=-0.02,
            j_bwd=20e-12,
            dwhp=0.005,
            dvto_rvwp_n=0.01,
            dkp_rvwn_n=-0.02,
            cfeed=0.05e-15,
            cpar_dyn=2e-15,
            temp_c=85.0,
        )
    )

    assert "\nB" not in netlist
    assert ".temp 85" in netlist
    assert "Vdd vdd 0 1.21" in netlist
    assert "Vx x 0 0.83" in netlist
    assert "Vbwd bwd 0 PULSE(0 1.2 9.02e-09" in netlist
    assert "Cwhp whp 0 20f IC=0.705" in netlist
    assert ".model NS_RVWP NMOS LEVEL=1 VTO=0.04" in netlist
    assert "Mrvwp_n rvwp vwn rvw_src 0 NS_RVWP" in netlist
    assert "Mrvwn_n rvwn vwp rvw_src 0 NS_RVWN" in netlist
    assert "Cfeed_fwd_pre fwd pre 5e-17" in netlist
    assert "Cpar_pre pre 0 2e-15" in netlist
    assert ".meas tran score_margin PARAM='score_forward-scoren_forward'" in netlist
    assert ".meas tran rvw_margin PARAM='rvwp_after_err-rvwn_after_err'" in netlist
    assert ".meas tran hidden_credit_margin PARAM='hdp_after_bwd-hdn_after_bwd'" in netlist


def test_shared_phase_cell_robustness_sampling_is_deterministic() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_shared_phase_cell_robustness as robust

    kwargs = dict(
        count=3,
        seed=7,
        vdd_sigma=0.02,
        input_sigma=0.01,
        phase_jitter_sigma=20e-12,
        state_sigma=0.005,
        latch_vto_sigma=0.01,
        latch_kp_sigma=0.02,
        cfeed=0.05e-15,
        cpar_dyn=2e-15,
        temp_c=27.0,
    )

    assert robust.sample_stresses(**kwargs) == robust.sample_stresses(**kwargs)
    assert robust.sample_stresses(**kwargs) != robust.sample_stresses(**{**kwargs, "seed": 8})


def test_shared_phase_cell_robustness_classifies_margin_failures() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_shared_phase_cell_robustness as robust

    thresholds = robust.RobustnessThresholds(
        score_margin_min=0.05,
        rvw_margin_min=0.2,
        hdp_min=0.05,
        hdn_max=0.08,
    )

    assert robust.classify(
        {"score_margin": 0.06, "rvw_margin": 0.4, "hdp_after_bwd": 0.07, "hdn_after_bwd": 0.02},
        thresholds,
    ) == {"passed": True, "failures": []}
    assert robust.classify(
        {"score_margin": -0.01, "rvw_margin": 0.1, "hdp_after_bwd": 0.01, "hdn_after_bwd": 0.2},
        thresholds,
    ) == {"passed": False, "failures": ["score_margin", "rvw_margin", "hdp", "hdn_leak"]}


def test_shared_phase_cell_robustness_rejects_negative_cli_sigmas() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_shared_phase_cell_robustness as robust

    with pytest.raises(ValueError, match="runs"):
        robust.main_for_test(["--runs", "0"])
