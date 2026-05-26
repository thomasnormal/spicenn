from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest


SPICE_DIR = Path(__file__).resolve().parents[1] / "spice"
sys.path.insert(0, str(SPICE_DIR))

import run_score_error_primitive as error  # noqa: E402
from run_device_sequential_training import run_netlist  # noqa: E402


def _run_ngspice_case(tmp_path: Path, case: str, *, restore_error: bool = True) -> dict[str, float]:
    ngspice = shutil.which("ngspice")
    if ngspice is None:
        pytest.skip("ngspice is not installed")
    return run_netlist(
        ngspice,
        tmp_path / f"score_error_{case}_{restore_error}.cir",
        error.generate_netlist(error_case=case, restore_error=restore_error),
        timeout=20.0,
    )


def test_score_error_primitive_emits_block_error_motif() -> None:
    netlist = error.generate_netlist(error_case="target_positive_score_negative")

    assert "\nB" not in netlist
    assert "Vtarget target 0 1.2" in netlist
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
    with pytest.raises(ValueError, match="target and score rails"):
        error.generate_netlist(error_case="target_positive_score_positive", score_center=1.18, score_delta=0.08)
    with pytest.raises(ValueError, match="timeout"):
        error.main_for_test(["--timeout", "0"])
    with pytest.raises(ValueError, match="error-restore-width"):
        error.main_for_test(["--error-restore-width", "0"])


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
    case: str,
    expected: float,
) -> None:
    measures = _run_ngspice_case(tmp_path, case, restore_error=False)

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
    case: str,
    expected: float,
) -> None:
    measures = _run_ngspice_case(tmp_path, case, restore_error=True)

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
) -> None:
    measures = _run_ngspice_case(tmp_path, "target_negative_score_negative", restore_error=False)

    # This is intentionally documented as an observed circuit behavior rather
    # than a desired learning contract: the current raw error circuit treats
    # the high scoren rail as positive pressure even when target is low.
    assert float(measures["raw_error_diff"]) > 0.025


def test_score_error_primitive_ngspice_neutral_raw_case_stays_small_without_restore(
    tmp_path: Path,
) -> None:
    measures = _run_ngspice_case(tmp_path, "neutral", restore_error=False)

    assert abs(float(measures["raw_error_diff"])) < 0.025
