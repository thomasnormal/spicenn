from __future__ import annotations

import sys
from pathlib import Path

import pytest


SPICE_DIR = Path(__file__).resolve().parents[1] / "spice"
sys.path.insert(0, str(SPICE_DIR))

import run_hidden_credit_rc_screen as screen  # noqa: E402


def test_hidden_credit_rc_screen_builds_live_hidden_args() -> None:
    args = screen.main_for_test(
        [
            "--tag",
            "unit",
            "--variant",
            "drive-nmos",
            "--scenario",
            "mnist",
            "--train-samples",
            "3",
            "--eval-samples",
            "3",
        ]
    )

    argv = screen._screen_argv(args, variant="drive-nmos", child_tag="child")

    assert "--readout-update-mode" in argv
    assert "live" in argv
    assert "--hidden-update-mode" in argv
    assert "readout-weighted" in argv
    assert "--hidden-credit-capacitance-f" in argv
    assert "50.0" in argv
    assert "--hidden-credit-shunt-resistance" in argv
    assert "250000.0" in argv
    assert "--hidden-credit-activation-model" in argv
    assert "NMOS" in argv
    assert "--eligibility-source-mode" in argv
    assert "act" in argv


def test_hidden_credit_rc_screen_validation() -> None:
    with pytest.raises(ValueError, match="variant"):
        screen._variant_list("missing")
    with pytest.raises(ValueError, match="timeout"):
        screen.main_for_test(["--timeout", "0"])
    with pytest.raises(ValueError, match="train-samples"):
        screen.main_for_test(["--train-samples", "0"])
    with pytest.raises(ValueError, match="score-capacitance-f"):
        screen.main_for_test(["--score-capacitance-f", "0"])


def test_hidden_credit_rc_screen_summary_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_run_screen(args):
        calls.append(args.tag)
        return {
            "scenario": "mnist",
            "by_scenario": {
                "mnist": {
                    "best_final_accuracy": 0.5,
                    "best_final_margin_v": -0.01,
                    "passed_count": 1,
                    "failed_count": 0,
                }
            },
            "hidden_credit_capacitance_f": 50.0,
            "hidden_credit_shunt_resistance": 250000.0,
            "hidden_credit_activation_model": "NMOS",
            "hidden_update_width": 0.25,
            "normalizer_error_clock_high": 0.45,
            "csv": str(tmp_path / "child.csv"),
            "wall_time_s": 1.0,
        }

    monkeypatch.setattr(screen, "ROOT", tmp_path)
    monkeypatch.setattr(screen.normalizer_screen, "run_screen", fake_run_screen)

    args = screen.main_for_test(["--tag", "unit_hc", "--variant", "drive-nmos"])
    summary = screen.run_screen(args)

    assert summary["architecture"] == "hidden_credit_rc_screen"
    assert summary["best_accuracy_variant"] == "drive-nmos"
    assert calls == ["unit_hc_drive_nmos"]
    assert (tmp_path / "results/tables/unit_hc.csv").exists()
    assert (tmp_path / "results/tables/unit_hc_summary.json").exists()
