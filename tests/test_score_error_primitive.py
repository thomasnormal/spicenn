from __future__ import annotations

import sys
from pathlib import Path

import pytest


SPICE_DIR = Path(__file__).resolve().parents[1] / "spice"
sys.path.insert(0, str(SPICE_DIR))

import run_score_error_primitive as error  # noqa: E402
from run_device_sequential_training import run_netlist  # noqa: E402


def _run_ngspice_case(
    tmp_path: Path,
    ngspice_path: str,
    case: str,
    *,
    restore_error: bool = True,
    error_topology: str = "competition",
) -> dict[str, float]:
    return run_netlist(
        ngspice_path,
        tmp_path / f"score_error_{case}_{restore_error}_{error_topology}.cir",
        error.generate_netlist(error_case=case, restore_error=restore_error, error_topology=error_topology),
        timeout=20.0,
    )


def test_score_error_primitive_emits_block_error_motif() -> None:
    netlist = error.generate_netlist(error_case="target_positive_score_negative")

    assert "\nB" not in netlist
    assert "Vtarget target 0 1.2" in netlist
    assert "Vtargetp targetp 0 1.2" in netlist
    assert "Vtargetn targetn 0 0" in netlist
    assert "Vscore score 0 0.06" in netlist
    assert "Vscoren scoren 0 0.14" in netlist
    assert "Mdp_t0 vdd target dp_t 0 NSENSE W=32u L=180n" in netlist
    assert "Mdp_sn0 vdd scoren dp_sn 0 NSENSE W=24u L=180n" in netlist
    assert "Mdn_y0 vdd score dn_y 0 NSENSE W=32u L=180n" in netlist
    assert "Merrstore_ep edp dn errstore_src 0 NSENSE W=7u L=180n" in netlist
    assert "Merrstore_en edn dp errstore_src 0 NSENSE W=7u L=180n" in netlist
    assert ".meas tran raw_error_diff PARAM='dp_after-dn_after'" in netlist
    assert ".meas tran restored_error_diff PARAM='edp_after-edn_after'" in netlist


def test_score_error_primitive_validation() -> None:
    with pytest.raises(ValueError, match="error_case"):
        error.generate_netlist(error_case="bad")
    with pytest.raises(ValueError, match="score_delta"):
        error.generate_netlist(error_case="neutral", score_delta=-0.1)
    with pytest.raises(ValueError, match="error_topology"):
        error.generate_netlist(error_case="neutral", error_topology="bad")
    with pytest.raises(ValueError, match="target and score rails"):
        error.generate_netlist(error_case="target_positive_score_positive", score_center=1.18, score_delta=0.08)
    with pytest.raises(ValueError, match="timeout"):
        error.main_for_test(["--timeout", "0"])
    with pytest.raises(ValueError, match="error-restore-width"):
        error.main_for_test(["--error-restore-width", "0"])
    with pytest.raises(ValueError, match="multiclass-sum-width"):
        error.main_for_test(["--multiclass-sum-width", "0"])
    with pytest.raises(ValueError, match="multiclass-error-width"):
        error.main_for_test(["--multiclass-error-width", "0"])


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("target_positive_score_negative", 1.0),
        ("target_negative_score_positive", -1.0),
        ("target_positive_score_positive", 1.0),
    ],
)
def test_score_error_primitive_ngspice_raw_error_polarity(
    tmp_path: Path,
    ngspice_path: str,
    case: str,
    expected: float,
) -> None:
    measures = _run_ngspice_case(tmp_path, ngspice_path, case, restore_error=False)

    margin = float(measures["raw_error_diff"])
    if expected > 0.0:
        assert margin > 0.025
        assert float(measures["dp_after"]) > float(measures["dn_after"])
    else:
        assert margin < -0.025
        assert float(measures["dn_after"]) > float(measures["dp_after"])


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("target_positive_score_negative", 1.0),
        ("target_negative_score_positive", -1.0),
        ("target_positive_score_positive", 1.0),
    ],
)
def test_score_error_primitive_ngspice_restored_error_polarity(
    tmp_path: Path,
    ngspice_path: str,
    case: str,
    expected: float,
) -> None:
    measures = _run_ngspice_case(tmp_path, ngspice_path, case, restore_error=True)

    raw_margin = float(measures["raw_error_diff"])
    restored_margin = float(measures["restored_error_diff"])
    if expected > 0.0:
        assert raw_margin > 0.025
        assert restored_margin > 0.10
        assert float(measures["edp_after"]) > float(measures["edn_after"])
    else:
        assert raw_margin < -0.025
        # The current restored latch only weakly amplifies negative raw error at
        # the integrated default width. Keep this as a sign guarantee, not as a
        # full-swing promise.
        assert restored_margin < -0.005
        assert float(measures["edn_after"]) > float(measures["edp_after"])


def test_score_error_primitive_ngspice_negative_label_negative_score_exposes_bias(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = _run_ngspice_case(tmp_path, ngspice_path, "target_negative_score_negative", restore_error=False)

    # This is intentionally documented as an observed circuit behavior rather
    # than a desired learning contract: the current raw error circuit treats
    # the high scoren rail as positive pressure even when target is low.
    assert float(measures["raw_error_diff"]) > 0.025


def test_score_error_primitive_ngspice_neutral_raw_case_stays_small_without_restore(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = _run_ngspice_case(tmp_path, ngspice_path, "neutral", restore_error=False)

    assert abs(float(measures["raw_error_diff"])) < 0.025


def test_score_error_primitive_emits_binary_descent_error_motif() -> None:
    netlist = error.generate_netlist(
        error_case="target_negative_score_positive",
        error_topology="binary-descent",
    )

    assert "\nB" not in netlist
    assert "Vtargetp targetp 0 0" in netlist
    assert "Vtargetn targetn 0 1.2" in netlist
    assert "Mdp_bd_t vdd targetp dp_bd_t 0 NSENSE W=48u L=180n" in netlist
    assert "Mdp_bd_s dp_bd_t scoren dp_bd_s 0 NSENSE W=48u L=180n" in netlist
    assert "Mdn_bd_t vdd targetn dn_bd_t 0 NSENSE W=48u L=180n" in netlist
    assert "Mdn_bd_s dn_bd_t score dn_bd_s 0 NSENSE W=48u L=180n" in netlist
    assert "Mdp_sn0" not in netlist
    assert "Mdn_t0" not in netlist


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("target_positive_score_negative", 1.0),
        ("target_negative_score_positive", -1.0),
        ("target_positive_score_positive", 1.0),
        ("target_negative_score_negative", -1.0),
    ],
)
def test_score_error_primitive_ngspice_binary_descent_raw_error_polarity(
    tmp_path: Path,
    ngspice_path: str,
    case: str,
    expected: float,
) -> None:
    measures = _run_ngspice_case(tmp_path, ngspice_path, case, restore_error=False, error_topology="binary-descent")

    margin = float(measures["raw_error_diff"])
    if expected > 0.0:
        assert margin > 0.025
        assert float(measures["dp_after"]) > float(measures["dn_after"])
    else:
        assert margin < -0.025
        assert float(measures["dn_after"]) > float(measures["dp_after"])


def test_score_error_primitive_ngspice_binary_descent_removes_negative_scoren_positive_bias(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    old = _run_ngspice_case(tmp_path, ngspice_path, "target_negative_score_negative", restore_error=False)
    new = _run_ngspice_case(
        tmp_path,
        ngspice_path,
        "target_negative_score_negative",
        restore_error=False,
        error_topology="binary-descent",
    )

    assert float(old["raw_error_diff"]) > 0.025
    assert float(new["raw_error_diff"]) < -0.025
    assert abs(float(new["raw_error_diff"])) < abs(float(old["raw_error_diff"]))


def test_score_error_primitive_ngspice_binary_descent_neutral_stays_small(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = _run_ngspice_case(tmp_path, ngspice_path, "neutral", restore_error=False, error_topology="binary-descent")

    assert abs(float(measures["raw_error_diff"])) < 0.025


def test_score_error_primitive_emits_label_descent_error_motif() -> None:
    netlist = error.generate_netlist(
        error_case="target_negative_score_positive",
        error_topology="label-descent",
    )

    assert "\nB" not in netlist
    assert "Vtargetp targetp 0 0" in netlist
    assert "Vtargetn targetn 0 1.2" in netlist
    assert "* Label descent output error: dp ~= targetp, dn ~= targetn during err." in netlist
    assert "Mdp_ld_t vdd targetp dp_ld_t 0 NSENSE W=48u L=180n" in netlist
    assert "Mdp_ld_e dp_ld_t err dp 0 NSENSE W=48u L=180n" in netlist
    assert "Mdn_ld_t vdd targetn dn_ld_t 0 NSENSE W=48u L=180n" in netlist
    assert "Mdn_ld_e dn_ld_t err dn 0 NSENSE W=48u L=180n" in netlist
    assert "Mdp_bd_t" not in netlist
    assert "Mdn_bd_t" not in netlist
    assert "Mdp_sn0" not in netlist
    assert "Mdn_t0" not in netlist


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("target_positive_score_negative", 1.0),
        ("target_negative_score_positive", -1.0),
        ("target_positive_score_positive", 1.0),
        ("target_negative_score_negative", -1.0),
    ],
)
def test_score_error_primitive_ngspice_label_descent_raw_error_polarity(
    tmp_path: Path,
    ngspice_path: str,
    case: str,
    expected: float,
) -> None:
    measures = _run_ngspice_case(tmp_path, ngspice_path, case, restore_error=False, error_topology="label-descent")

    margin = float(measures["raw_error_diff"])
    if expected > 0.0:
        assert margin > 0.025
        assert float(measures["dp_after"]) > float(measures["dn_after"])
    else:
        assert margin < -0.025
        assert float(measures["dn_after"]) > float(measures["dp_after"])


def test_score_error_primitive_ngspice_label_descent_neutral_stays_small(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = _run_ngspice_case(tmp_path, ngspice_path, "neutral", restore_error=False, error_topology="label-descent")

    assert abs(float(measures["raw_error_diff"])) < 0.025


def test_score_error_primitive_emits_multiclass_nontarget_mass_error_motif() -> None:
    netlist = error.generate_multiclass_netlist(case="target1_high_wrong0")

    assert "\nB" not in netlist
    assert "Cnontarget_mass nontarget_mass 0 8f IC=0" in netlist
    assert "Vtargetp1 targetp1 0 1.2" in netlist
    assert "Vtargetn0 targetn0 0 1.2" in netlist
    assert "Vtargetn1 targetn1 0 0" in netlist
    assert "Mmass_nt0_label vdd targetn0 mass_nt0_a 0 NSENSE W=32u" in netlist
    assert "Mmass_nt0_score mass_nt0_a score0 mass_nt0_s 0 NSENSE W=32u" in netlist
    assert "Mdp1_mass dp1_a nontarget_mass dp1_m 0 NSENSE W=32u" in netlist
    assert "Mdn0_score dn0_a score0 dn0_s 0 NSENSE W=32u" in netlist
    assert ".meas tran err1_diff PARAM='dp1_after-dn1_after'" in netlist


def test_score_error_primitive_multiclass_validation() -> None:
    with pytest.raises(ValueError, match="case"):
        error.generate_multiclass_netlist(case="bad")
    with pytest.raises(ValueError, match="class_count"):
        error.generate_multiclass_netlist(case="neutral", class_count=4)
    with pytest.raises(ValueError, match="score_values"):
        error.generate_multiclass_netlist(case="neutral", score_values=(0.1, 0.2))
    with pytest.raises(ValueError, match="score rails"):
        error.generate_multiclass_netlist(case="neutral", score_values=(0.1, 0.2, 1.3))
    with pytest.raises(ValueError, match="target_class"):
        error.generate_multiclass_netlist(case="neutral", target_class=3)


def _run_multiclass_case(tmp_path: Path, ngspice_path: str, case: str) -> dict[str, float]:
    return run_netlist(
        ngspice_path,
        tmp_path / f"score_error_multiclass_{case}.cir",
        error.generate_multiclass_netlist(case=case),
        timeout=20.0,
    )


def test_score_error_primitive_ngspice_multiclass_target_gets_nontarget_mass_positive_credit(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    high_wrong = _run_multiclass_case(tmp_path, ngspice_path, "target1_high_wrong0")
    low_wrong = _run_multiclass_case(tmp_path, ngspice_path, "target1_low_wrong0")

    assert float(high_wrong["nontarget_mass_after"]) > 0.20
    assert float(high_wrong["err1_diff"]) > 0.025
    assert float(low_wrong["err1_diff"]) > float(high_wrong["err1_diff"]) - 5e-3


def test_score_error_primitive_ngspice_multiclass_nontarget_negative_credit_tracks_score(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = _run_multiclass_case(tmp_path, ngspice_path, "target1_high_wrong0")

    assert float(measures["err0_diff"]) < -0.025
    assert float(measures["err2_diff"]) < -0.005
    assert abs(float(measures["err1_diff"])) > abs(float(measures["err2_diff"]))
    assert abs(float(measures["err0_diff"])) > abs(float(measures["err2_diff"]))


def test_score_error_primitive_ngspice_multiclass_clear_target_has_smaller_positive_pressure(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    clear = _run_multiclass_case(tmp_path, ngspice_path, "target0_clear")
    wrong = _run_multiclass_case(tmp_path, ngspice_path, "target1_high_wrong0")

    assert float(clear["err0_diff"]) > 0.0
    assert float(clear["err0_diff"]) < float(wrong["err1_diff"])
    assert float(clear["err1_diff"]) < -0.005
    assert float(clear["err2_diff"]) < -0.001


def test_score_error_primitive_ngspice_multiclass_neutral_no_label_stays_quiet(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = _run_multiclass_case(tmp_path, ngspice_path, "neutral")

    assert float(measures["nontarget_mass_after"]) < 1e-3
    for class_idx in range(3):
        assert abs(float(measures[f"err{class_idx}_diff"])) < 1e-3


def test_score_error_primitive_multiclass_summary_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_detect_spice(spice_bin):
        return "/usr/bin/ngspice", "fake-ngspice"

    def fake_run_netlist(spice_bin, path, deck, *, timeout):
        assert "\nB" not in deck
        if "target1_high_wrong0" in str(path):
            return {"err0_diff": -0.08, "err1_diff": 0.06, "err2_diff": -0.02, "nontarget_mass_after": 0.3}
        if "target1_low_wrong0" in str(path):
            return {"err0_diff": -0.08, "err1_diff": 0.07, "err2_diff": -0.04, "nontarget_mass_after": 0.3}
        if "target0_clear" in str(path):
            return {"err0_diff": 0.03, "err1_diff": -0.02, "err2_diff": -0.01, "nontarget_mass_after": 0.1}
        if "target2_wrong1" in str(path):
            return {"err0_diff": -0.01, "err1_diff": -0.08, "err2_diff": 0.06, "nontarget_mass_after": 0.3}
        return {"err0_diff": 0.0, "err1_diff": 0.0, "err2_diff": 0.0, "nontarget_mass_after": 0.0}

    monkeypatch.setattr(error, "ROOT", tmp_path)
    monkeypatch.setattr(error, "detect_spice", fake_detect_spice)
    monkeypatch.setattr(error, "run_netlist", fake_run_netlist)

    args = error.main_for_test(["--multiclass", "--tag", "unit_multiclass_error", "--min-abs-margin", "0.005"])
    summary = error.run_multiclass_cases(args)

    assert summary["architecture"] == "multiclass_score_error_primitive"
    assert summary["passed"] is True
    assert summary["cases"] == len(error.MULTICLASS_CASES)
    assert (tmp_path / "results/tables/unit_multiclass_error_summary.json").exists()
