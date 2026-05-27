from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest


SPICE_DIR = Path(__file__).resolve().parents[1] / "spice"
sys.path.insert(0, str(SPICE_DIR))

import run_normalizer_block_screen as screen  # noqa: E402


def test_normalizer_block_screen_builds_block_args() -> None:
    args = screen.main_for_test(
        [
            "--tag",
            "unit",
            "--approach",
            "current-sum",
            "--scenario",
            "mnist",
            "--train-samples",
            "3",
            "--eval-samples",
            "3",
            "--score-capacitance-f",
            "7",
            "--normalizer-error-clock-high",
            "0.45",
            "--readout-update-mode",
            "live",
            "--hidden-update-mode",
            "readout-weighted",
            "--hidden-credit-width",
            "6",
            "--hidden-update-width",
            "0.2",
            "--score-timing-mode",
            "early",
            "--readout-forward-mode",
            "diode",
        ]
    )

    argv = screen._block_argv(args, approach="current-sum", scenario="mnist", child_tag="child")

    assert "--error-mode" in argv
    assert "normalizer-current-sum-descent" in argv
    assert "--scenario" in argv
    assert "mnist" in argv
    assert "--score-capacitance-f" in argv
    assert "7.0" in argv
    assert "--normalizer-error-clock-high" in argv
    assert "0.45" in argv
    assert "--readout-update-mode" in argv
    assert "live" in argv
    assert "--hidden-update-mode" in argv
    assert "readout-weighted" in argv
    assert "--hidden-credit-width" in argv
    assert "6.0" in argv
    assert "--hidden-update-width" in argv
    assert "0.2" in argv
    assert "--score-timing-mode" in argv
    assert "early" in argv
    assert "--readout-forward-mode" in argv
    assert "diode" in argv


def test_normalizer_block_screen_validation() -> None:
    with pytest.raises(ValueError, match="timeout"):
        screen.main_for_test(["--timeout", "0"])
    with pytest.raises(ValueError, match="train-samples"):
        screen.main_for_test(["--train-samples", "0"])
    with pytest.raises(ValueError, match="score-capacitance"):
        screen.main_for_test(["--score-capacitance-f", "0"])
    with pytest.raises(ValueError, match="normalizer-error-clock-high"):
        screen.main_for_test(["--normalizer-error-clock-high", "0"])
    with pytest.raises(ValueError, match="hidden-credit-width"):
        screen.main_for_test(["--hidden-credit-width", "0"])
    with pytest.raises(ValueError, match="hidden-update-width"):
        screen.main_for_test(["--hidden-update-width", "0"])


def test_normalizer_block_screen_summary_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_run_case(args):
        calls.append(args.error_mode)
        suffix = args.error_mode.removeprefix("normalizer-").removesuffix("-descent")
        final_margin = 0.1 if suffix == "current-sum" else 0.2
        return {
            "passed": True,
            "initial_eval_accuracy": 0.3333333333,
            "final_eval_accuracy": 1.0,
            "accuracy_improvement": 0.6666666667,
            "initial_eval_min_margin_v": 0.0,
            "final_eval_min_margin_v": final_margin,
            "margin_improvement_v": final_margin,
            "final_eval_signed_projection_accuracy": 1.0,
            "final_eval_conductance_projection_accuracy": 1.0,
            "final_eval_activation_prototype_accuracy": 0.6666666667,
            "final_eval_activation_cosine_prototype_accuracy": 0.5,
            "train_eligibility_active_features_25mv_mean": 3.0,
            "train_eligibility_active_features_250mv_mean": 2.0,
            "train_eligibility_active_features_500mv_mean": 1.0,
            "train_eligibility_pairwise_cosine_mean": 0.25,
            "csv": str(tmp_path / f"{suffix}.csv"),
            "wall_time_s": 1.0,
        }

    monkeypatch.setattr(screen, "ROOT", tmp_path)
    monkeypatch.setattr(screen.seq, "run_case", fake_run_case)

    args = screen.main_for_test(["--tag", "unit_screen", "--scenario", "one-hot", "--approach", "all"])
    summary = screen.run_screen(args)

    assert summary["architecture"] == "normalizer_block_screen"
    assert len(calls) == len(screen.APPROACHES)
    assert summary["by_scenario"]["one-hot"]["passed_count"] == len(screen.APPROACHES)
    assert summary["readout_update_mode"] == "sampled"
    assert summary["hidden_update_mode"] == "none"
    assert (tmp_path / "results/tables/unit_screen_summary.json").exists()

    with (tmp_path / "results/tables/unit_screen.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["final_eval_signed_projection_accuracy"] == "1.0"
    assert rows[0]["train_eligibility_pairwise_cosine_mean"] == "0.25"
