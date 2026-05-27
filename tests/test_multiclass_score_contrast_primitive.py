from __future__ import annotations

import sys
from pathlib import Path

import pytest


SPICE_DIR = Path(__file__).resolve().parents[1] / "spice"
sys.path.insert(0, str(SPICE_DIR))

import run_multiclass_score_contrast_primitive as contrast  # noqa: E402
from run_device_sequential_training import run_netlist  # noqa: E402


def _run_case(tmp_path: Path, ngspice_path: str, case: str) -> dict[str, float]:
    return run_netlist(
        ngspice_path,
        tmp_path / f"score_contrast_{case}.cir",
        contrast.generate_netlist(case=case),
        timeout=20.0,
    )


def test_multiclass_score_contrast_primitive_emits_physical_contrast_caps() -> None:
    netlist = contrast.generate_netlist(case="ordered")

    assert "\nB" not in netlist
    assert "Rscore_common_c0 score_common score0 20000" in netlist
    assert "Ccontrast0 contrast0 0 10f IC=0.6" in netlist
    assert "Mcontrast0_up_v vdd score0 contrast0_up 0 NREL W=192u L=180n" in netlist
    assert "Mcontrast0_dn_v contrast0 score_common contrast0_dn 0 NREL W=24u L=180n" in netlist
    assert ".meas tran contrast_spread PARAM='contrast0_after-contrast2_after'" in netlist


def test_multiclass_score_contrast_primitive_emits_low_gain_score_normalizer() -> None:
    netlist = contrast.generate_netlist(case="low_common", input_stage="low-gain")

    assert "\nB" not in netlist
    assert "Vscore_raw0 score_raw0 0 0.0075" in netlist
    assert "Cscore0 score0 0 8f IC=1.2" in netlist
    assert "Mprecharge_score0 score0 rstfn vdd vdd PMOS W=4u L=180n" in netlist
    assert "Mscore0_amp_p score0 score_raw0 score0_amp_i vdd PMOS W=1u L=180n" in netlist
    assert "Mscore0_amp_tail score0_amp_i amp 0 0 NMOS W=8u L=180n" in netlist
    assert "Rscore_common_c0 score_common score0 10000000" in netlist
    assert ".meas tran score0_norm_after FIND V(score0)" in netlist


def test_multiclass_score_contrast_primitive_emits_low_gain_score_mass_error_stage() -> None:
    netlist = contrast.generate_netlist(
        case="low_common",
        input_stage="low-gain",
        error_stage="score-mass",
        target_class=1,
    )

    assert "\nB" not in netlist
    assert "Vtargetp1 targetp1 0 1.2" in netlist
    assert "Vtargetn0 targetn0 0 1.2" in netlist
    assert "Cscore_nontarget_mass score_nontarget_mass 0 0.5f IC=0" in netlist
    assert "Mmass_nt0_score mass_nt0_a contrast0 mass_nt0_s 0 NSENSE W=128u" in netlist
    assert "Mdp1_mass dp1_a score_nontarget_mass dp1_m 0 NSENSE W=128u" in netlist
    assert "Mdn0_score dn0_a contrast0 dn0_s 0 NSENSE W=128u" in netlist
    assert ".meas tran err1_diff PARAM='dp1_after-dn1_after'" in netlist


def test_multiclass_score_contrast_primitive_validation() -> None:
    with pytest.raises(ValueError, match="case"):
        contrast.generate_netlist(case="bad")
    with pytest.raises(ValueError, match="input_stage"):
        contrast.generate_netlist(case="flat", input_stage="bad")
    with pytest.raises(ValueError, match="error_stage"):
        contrast.generate_netlist(case="flat", error_stage="bad")
    with pytest.raises(ValueError, match="target_class"):
        contrast.generate_netlist(case="flat", error_stage="score-mass", target_class=4)
    with pytest.raises(ValueError, match="class_count"):
        contrast.generate_netlist(case="flat", class_count=4)
    with pytest.raises(ValueError, match="score_values"):
        contrast.generate_netlist(case="flat", score_values=(0.1, 0.2))
    with pytest.raises(ValueError, match="supply rails"):
        contrast.generate_netlist(case="flat", score_values=(0.1, 0.2, 1.3))
    with pytest.raises(ValueError, match="common-resistance"):
        contrast.main_for_test(["--common-resistance", "0"])
    with pytest.raises(ValueError, match="min-margin"):
        contrast.main_for_test(["--min-margin", "-1"])


def test_multiclass_score_contrast_primitive_ngspice_orders_score_amp_rails(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = _run_case(tmp_path, ngspice_path, "ordered")

    assert abs(float(measures["score_common_after"]) - 0.45) < 0.01
    assert float(measures["contrast0_after"]) > float(measures["contrast1_after"]) + 0.025
    assert float(measures["contrast1_after"]) > float(measures["contrast2_after"]) + 0.025
    assert float(measures["contrast_spread"]) > 0.20


def test_multiclass_score_contrast_primitive_ngspice_flat_scores_have_dead_zone(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = _run_case(tmp_path, ngspice_path, "flat")

    assert abs(float(measures["score_common_after"]) - 0.45) < 0.01
    assert abs(float(measures["contrast_spread"])) < 1e-3
    assert abs(float(measures["contrast_0_1_margin"])) < 1e-3
    assert abs(float(measures["contrast_1_2_margin"])) < 1e-3


def test_multiclass_score_contrast_primitive_ngspice_spread_grows_with_gap(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    small = _run_case(tmp_path, ngspice_path, "small_gap")
    large = _run_case(tmp_path, ngspice_path, "large_gap")

    assert float(small["contrast_spread"]) > 0.04
    assert float(large["contrast_spread"]) > float(small["contrast_spread"]) + 0.05


def test_multiclass_score_contrast_primitive_ngspice_raw_low_common_is_quiet(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = _run_case(tmp_path, ngspice_path, "low_common")

    assert float(measures["score_common_after"]) < 0.01
    assert abs(float(measures["contrast_spread"])) < 1e-3


def test_multiclass_score_contrast_primitive_ngspice_low_gain_recovers_low_common_ordering(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "score_contrast_low_common_low_gain.cir",
        contrast.generate_netlist(case="low_common", input_stage="low-gain"),
        timeout=20.0,
    )

    assert float(measures["score0_norm_after"]) > float(measures["score1_norm_after"]) + 0.002
    assert float(measures["score1_norm_after"]) > float(measures["score2_norm_after"]) + 0.002
    assert float(measures["contrast0_after"]) > float(measures["contrast1_after"]) + 0.001
    assert float(measures["contrast1_after"]) > float(measures["contrast2_after"]) + 0.001
    assert float(measures["contrast_spread"]) > 0.003


def test_multiclass_score_contrast_primitive_ngspice_low_gain_flat_scores_have_dead_zone(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "score_contrast_flat_low_gain.cir",
        contrast.generate_netlist(case="flat", score_values=(0.0045, 0.0045, 0.0045), input_stage="low-gain"),
        timeout=20.0,
    )

    assert abs(float(measures["score0_norm_after"]) - float(measures["score2_norm_after"])) < 1e-3
    assert abs(float(measures["contrast_spread"])) < 1e-3


def test_multiclass_score_contrast_primitive_ngspice_low_gain_score_mass_drives_writer_scale_errors(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "score_contrast_low_gain_score_mass.cir",
        contrast.generate_netlist(
            case="low_common",
            input_stage="low-gain",
            error_stage="score-mass",
            target_class=1,
        ),
        timeout=20.0,
    )

    assert float(measures["contrast_spread"]) > 0.003
    assert float(measures["score_nontarget_mass_after"]) > 0.08
    assert float(measures["err1_diff"]) > 0.08
    assert float(measures["err0_diff"]) < -0.08
    assert float(measures["err2_diff"]) < -0.08


def test_multiclass_score_contrast_primitive_summary_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_detect_spice(spice_bin):
        return "/usr/bin/ngspice", "fake-ngspice"

    def fake_run_netlist(spice_bin, path, deck, *, timeout):
        assert "\nB" not in deck
        name = str(path)
        if "flat" in name:
            return {"contrast_spread": 0.0, "contrast_0_1_margin": 0.0, "contrast_1_2_margin": 0.0}
        if "low_common" in name:
            return {"contrast_spread": 0.0, "contrast_0_1_margin": 0.0, "contrast_1_2_margin": 0.0}
        return {"contrast_spread": 0.12, "contrast_0_1_margin": 0.06, "contrast_1_2_margin": 0.06}

    monkeypatch.setattr(contrast, "ROOT", tmp_path)
    monkeypatch.setattr(contrast, "detect_spice", fake_detect_spice)
    monkeypatch.setattr(contrast, "run_netlist", fake_run_netlist)

    args = contrast.main_for_test(["--tag", "unit_score_contrast", "--min-margin", "0.025"])
    summary = contrast.run_cases(args)

    assert summary["architecture"] == "multiclass_score_contrast_primitive"
    assert summary["passed"] is True
    assert summary["classification_counts"]["ordered"] == 3
    assert (tmp_path / "results/tables/unit_score_contrast_summary.json").exists()


def test_multiclass_score_contrast_primitive_low_gain_summary_counts_low_common_as_ordered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_detect_spice(spice_bin):
        return "/usr/bin/ngspice", "fake-ngspice"

    def fake_run_netlist(spice_bin, path, deck, *, timeout):
        assert "\nB" not in deck
        name = str(path)
        if "flat" in name:
            return {"contrast_spread": 0.0, "contrast_0_1_margin": 0.0, "contrast_1_2_margin": 0.0}
        return {"contrast_spread": 0.012, "contrast_0_1_margin": 0.006, "contrast_1_2_margin": 0.006}

    monkeypatch.setattr(contrast, "ROOT", tmp_path)
    monkeypatch.setattr(contrast, "detect_spice", fake_detect_spice)
    monkeypatch.setattr(contrast, "run_netlist", fake_run_netlist)

    args = contrast.main_for_test(
        ["--tag", "unit_score_contrast_low_gain", "--input-stage", "low-gain", "--min-margin", "0.003"]
    )
    summary = contrast.run_cases(args)

    assert summary["input_stage"] == "low-gain"
    assert summary["passed"] is True
    assert summary["classification_counts"]["ordered"] == 4
