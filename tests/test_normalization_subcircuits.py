from __future__ import annotations

import sys
from pathlib import Path

import pytest


SPICE_DIR = Path(__file__).resolve().parents[1] / "spice"
sys.path.insert(0, str(SPICE_DIR))

import run_normalization_subcircuits as norm  # noqa: E402
from run_device_sequential_training import run_netlist  # noqa: E402


def test_normalization_subcircuits_define_all_ten_candidates() -> None:
    text = norm.normalization_subcircuits()

    assert "\nB" not in text
    assert len(norm.APPROACHES) == 10
    for approach in norm.APPROACHES:
        assert f".subckt {norm.spice_subckt_name(approach)} " in text
        assert f".ends {norm.spice_subckt_name(approach)}" in text


@pytest.mark.parametrize("approach", norm.APPROACHES)
def test_normalization_subcircuit_netlist_instantiates_candidate(approach: str) -> None:
    netlist = norm.generate_netlist(approach=approach, case="target1_low_wrong0")

    assert "\nB" not in netlist
    assert f"Xnorm s0 s1 s2 tp0 tp1 tp2 tn0 tn1 tn2 phi rst e0p e0n e1p e1n e2p e2n vdd 0 {norm.spice_subckt_name(approach)}" in netlist
    assert "Vtp1 tp1 0 1.2" in netlist
    assert "Vtn0 tn0 0 1.2" in netlist
    assert ".meas tran e1_diff PARAM='e1p_after-e1n_after'" in netlist


def test_normalization_subcircuit_validation() -> None:
    with pytest.raises(ValueError, match="approach"):
        norm.generate_netlist(approach="bad", case="target1_clear")
    with pytest.raises(ValueError, match="case"):
        norm.generate_netlist(approach="current-sum", case="bad")
    with pytest.raises(ValueError, match="three"):
        norm.generate_netlist(approach="current-sum", case="target1_clear", score_values=(0.1, 0.2))
    with pytest.raises(ValueError, match="target_class"):
        norm.generate_netlist(approach="current-sum", case="target1_clear", target_class=3)
    with pytest.raises(ValueError, match="supply rails"):
        norm.generate_netlist(approach="current-sum", case="target1_clear", score_values=(0.1, 0.2, 1.3))
    with pytest.raises(ValueError, match="timeout"):
        norm.main_for_test(["--timeout", "0"])
    with pytest.raises(ValueError, match="min-abs-margin"):
        norm.main_for_test(["--min-abs-margin", "0"])


@pytest.mark.parametrize("approach", norm.APPROACHES)
def test_normalization_subcircuit_ngspice_dense_bootstrap_signs(
    tmp_path: Path,
    ngspice_path: str,
    approach: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / f"normalizer_{approach}_wrong0.cir",
        norm.generate_netlist(approach=approach, case="target1_low_wrong0"),
        timeout=30.0,
    )

    assert float(measures["e1_diff"]) > 0.02
    assert float(measures["e0_diff"]) < -0.02
    assert float(measures["e2_diff"]) < -0.02


@pytest.mark.parametrize("approach", norm.APPROACHES)
def test_normalization_subcircuit_ngspice_target_pressure_tracks_nontarget_mass(
    tmp_path: Path,
    ngspice_path: str,
    approach: str,
) -> None:
    clear = run_netlist(
        ngspice_path,
        tmp_path / f"normalizer_{approach}_clear.cir",
        norm.generate_netlist(approach=approach, case="target1_clear"),
        timeout=30.0,
    )
    wrong = run_netlist(
        ngspice_path,
        tmp_path / f"normalizer_{approach}_wrong0_mass.cir",
        norm.generate_netlist(approach=approach, case="target1_low_wrong0"),
        timeout=30.0,
    )

    assert float(wrong["e1_diff"]) >= float(clear["e1_diff"]) - 1e-3


@pytest.mark.parametrize("approach", norm.APPROACHES)
def test_normalization_subcircuit_ngspice_is_class_permutation_symmetric(
    tmp_path: Path,
    ngspice_path: str,
    approach: str,
) -> None:
    for target_class in range(3):
        wrong_class = (target_class + 1) % 3
        other_class = (target_class + 2) % 3
        scores = [0.0045, 0.0045, 0.0045]
        scores[target_class] = 0.0015
        scores[wrong_class] = 0.0075
        scores[other_class] = 0.0045

        measures = run_netlist(
            ngspice_path,
            tmp_path / f"normalizer_{approach}_target{target_class}_wrong{wrong_class}.cir",
            norm.generate_netlist(
                approach=approach,
                case="target1_low_wrong0",
                score_values=tuple(scores),
                target_class=target_class,
            ),
            timeout=30.0,
        )

        assert float(measures[f"e{target_class}_diff"]) > 0.02
        assert float(measures[f"e{wrong_class}_diff"]) < -0.02
        assert float(measures[f"e{other_class}_diff"]) < -0.02


@pytest.mark.parametrize("approach", norm.APPROACHES)
def test_normalization_subcircuit_ngspice_preserves_direction_across_score_common_mode(
    tmp_path: Path,
    ngspice_path: str,
    approach: str,
) -> None:
    low_scores = (0.0075, 0.0015, 0.0045)
    high_scores = tuple(score + 0.04 for score in low_scores)

    low = run_netlist(
        ngspice_path,
        tmp_path / f"normalizer_{approach}_low_common.cir",
        norm.generate_netlist(
            approach=approach,
            case="target1_low_wrong0",
            score_values=low_scores,
            target_class=1,
        ),
        timeout=30.0,
    )
    high = run_netlist(
        ngspice_path,
        tmp_path / f"normalizer_{approach}_high_common.cir",
        norm.generate_netlist(
            approach=approach,
            case="target1_low_wrong0",
            score_values=high_scores,
            target_class=1,
        ),
        timeout=30.0,
    )

    for measures in (low, high):
        assert float(measures["e1_diff"]) > 0.02
        assert float(measures["e0_diff"]) < -0.02
        assert float(measures["e2_diff"]) < -0.02


def test_normalization_subcircuit_summary_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_detect_spice(spice_bin):
        return "/usr/bin/ngspice", "fake-ngspice"

    def fake_run_netlist(spice_bin, path, deck, *, timeout):
        assert "\nB" not in deck
        return {
            "e0_diff": -0.05,
            "e1_diff": 0.07,
            "e2_diff": -0.05,
            "e0p_after": 0.0,
            "e0n_after": 0.05,
            "e1p_after": 0.07,
            "e1n_after": 0.0,
            "e2p_after": 0.0,
            "e2n_after": 0.05,
        }

    monkeypatch.setattr(norm, "ROOT", tmp_path)
    monkeypatch.setattr(norm, "detect_spice", fake_detect_spice)
    monkeypatch.setattr(norm, "run_netlist", fake_run_netlist)

    args = norm.main_for_test(["--tag", "unit_normalizers", "--approach", "current-sum"])
    summary = norm.run_cases(args)

    assert summary["architecture"] == "normalization_subcircuit_library"
    assert summary["passed"] is True
    assert (tmp_path / "results/tables/unit_normalizers_summary.json").exists()
