from __future__ import annotations

import argparse
import csv
import json
import time
from typing import Any

import run_multiclass_block_sequence as seq
from run_device_mnist01_scalar_training import sanitize_tag
from run_normalization_subcircuits import APPROACHES
from run_spice_sweep import ROOT


SCENARIOS = ("one-hot", "mnist", "both")


def _scenario_list(scenario: str) -> tuple[str, ...]:
    if scenario == "both":
        return ("one-hot", "mnist")
    if scenario in {"one-hot", "mnist"}:
        return (scenario,)
    raise ValueError(f"scenario must be one of {SCENARIOS}")


def _approach_list(approach: str) -> tuple[str, ...]:
    if approach == "all":
        return APPROACHES
    if approach in APPROACHES:
        return (approach,)
    raise ValueError(f"approach must be 'all' or one of {APPROACHES}")


def _block_argv(args: argparse.Namespace, *, approach: str, scenario: str, child_tag: str) -> list[str]:
    argv = [
        "--tag",
        child_tag,
        "--timeout",
        str(args.timeout),
        "--spice-bin",
        args.spice_bin or "",
        "--scenario",
        scenario,
        "--error-mode",
        f"normalizer-{approach}-descent",
        "--score-capacitance-f",
        str(args.score_capacitance_f),
        "--score-load-resistance",
        str(args.score_load_resistance),
        "--readout-width",
        str(args.readout_width),
        "--initial-positive",
        str(args.initial_positive),
        "--initial-negative",
        str(args.initial_negative),
        "--normalizer-error-clock-high",
        str(args.normalizer_error_clock_high),
        "--readout-update-mode",
        args.readout_update_mode,
        "--hidden-update-mode",
        args.hidden_update_mode,
        "--hidden-credit-width",
        str(args.hidden_credit_width),
        "--hidden-update-width",
        str(args.hidden_update_width),
        "--score-timing-mode",
        args.score_timing_mode,
        "--readout-forward-mode",
        args.readout_forward_mode,
        "--eligibility-source-mode",
        args.eligibility_source_mode,
    ]
    if scenario == "one-hot":
        argv += [
            "--train-repeats",
            str(args.onehot_train_repeats),
            "--eval-repeats",
            str(args.onehot_eval_repeats),
        ]
    else:
        argv += [
            "--dataset",
            args.dataset,
            "--seed",
            str(args.seed),
            "--train-samples",
            str(args.train_samples),
            "--eval-samples",
            str(args.eval_samples),
            "--class-bias-mode",
            args.class_bias_mode,
            "--class-bias-input",
            str(args.class_bias_input),
        ]
        if args.download:
            argv.append("--download")
    if args.spice_bin is None:
        idx = argv.index("--spice-bin")
        del argv[idx : idx + 2]
    return argv


def _summary_row(summary: dict[str, Any], *, approach: str, scenario: str, tag: str) -> dict[str, Any]:
    return {
        "approach": approach,
        "scenario": scenario,
        "tag": tag,
        "status": "ok",
        "error": "",
        "passed": bool(summary["passed"]),
        "initial_eval_accuracy": float(summary["initial_eval_accuracy"]),
        "final_eval_accuracy": float(summary["final_eval_accuracy"]),
        "accuracy_improvement": float(summary["accuracy_improvement"]),
        "initial_eval_min_margin_v": float(summary["initial_eval_min_margin_v"]),
        "final_eval_min_margin_v": float(summary["final_eval_min_margin_v"]),
        "margin_improvement_v": float(summary["margin_improvement_v"]),
        "train_target_errdiff_mean_v": summary.get("train_target_errdiff_mean_v"),
        "train_target_errdiff_min_v": summary.get("train_target_errdiff_min_v"),
        "train_nontarget_errdiff_mean_v": summary.get("train_nontarget_errdiff_mean_v"),
        "train_nontarget_errdiff_max_v": summary.get("train_nontarget_errdiff_max_v"),
        "final_eval_signed_projection_accuracy": summary.get("final_eval_signed_projection_accuracy"),
        "final_eval_conductance_projection_accuracy": summary.get("final_eval_conductance_projection_accuracy"),
        "final_eval_activation_prototype_accuracy": summary.get("final_eval_activation_prototype_accuracy"),
        "final_eval_activation_cosine_prototype_accuracy": summary.get(
            "final_eval_activation_cosine_prototype_accuracy"
        ),
        "train_eligibility_active_features_25mv_mean": summary.get("train_eligibility_active_features_25mv_mean"),
        "train_eligibility_active_features_250mv_mean": summary.get("train_eligibility_active_features_250mv_mean"),
        "train_eligibility_active_features_500mv_mean": summary.get("train_eligibility_active_features_500mv_mean"),
        "train_eligibility_pairwise_cosine_mean": summary.get("train_eligibility_pairwise_cosine_mean"),
        "csv": str(summary["csv"]),
        "wall_time_s": float(summary["wall_time_s"]),
    }


def _failure_row(error: Exception, *, approach: str, scenario: str, tag: str) -> dict[str, Any]:
    message = str(error).splitlines()[-1] if str(error).splitlines() else repr(error)
    return {
        "approach": approach,
        "scenario": scenario,
        "tag": tag,
        "status": "failed",
        "error": message[:500],
        "passed": False,
        "initial_eval_accuracy": "",
        "final_eval_accuracy": "",
        "accuracy_improvement": "",
        "initial_eval_min_margin_v": "",
        "final_eval_min_margin_v": "",
        "margin_improvement_v": "",
        "train_target_errdiff_mean_v": "",
        "train_target_errdiff_min_v": "",
        "train_nontarget_errdiff_mean_v": "",
        "train_nontarget_errdiff_max_v": "",
        "final_eval_signed_projection_accuracy": "",
        "final_eval_conductance_projection_accuracy": "",
        "final_eval_activation_prototype_accuracy": "",
        "final_eval_activation_cosine_prototype_accuracy": "",
        "train_eligibility_active_features_25mv_mean": "",
        "train_eligibility_active_features_250mv_mean": "",
        "train_eligibility_active_features_500mv_mean": "",
        "train_eligibility_pairwise_cosine_mean": "",
        "csv": "",
        "wall_time_s": "",
    }


def run_screen(args: argparse.Namespace) -> dict[str, Any]:
    tables = ROOT / "results/tables"
    tables.mkdir(parents=True, exist_ok=True)
    tag = sanitize_tag(args.tag)
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for approach in _approach_list(args.approach):
        for scenario in _scenario_list(args.scenario):
            child_tag = sanitize_tag(f"{tag}_{scenario}_{approach}")
            child_args = seq.main_for_test(_block_argv(args, approach=approach, scenario=scenario, child_tag=child_tag))
            try:
                summary = seq.run_case(child_args)
            except Exception as exc:
                if not args.keep_going:
                    raise
                rows.append(_failure_row(exc, approach=approach, scenario=scenario, tag=child_tag))
                continue
            rows.append(_summary_row(summary, approach=approach, scenario=scenario, tag=child_tag))

    csv_path = tables / f"{tag}.csv"
    fieldnames = [
        "approach",
        "scenario",
        "tag",
        "status",
        "error",
        "passed",
        "initial_eval_accuracy",
        "final_eval_accuracy",
        "accuracy_improvement",
        "initial_eval_min_margin_v",
        "final_eval_min_margin_v",
        "margin_improvement_v",
        "train_target_errdiff_mean_v",
        "train_target_errdiff_min_v",
        "train_nontarget_errdiff_mean_v",
        "train_nontarget_errdiff_max_v",
        "final_eval_signed_projection_accuracy",
        "final_eval_conductance_projection_accuracy",
        "final_eval_activation_prototype_accuracy",
        "final_eval_activation_cosine_prototype_accuracy",
        "train_eligibility_active_features_25mv_mean",
        "train_eligibility_active_features_250mv_mean",
        "train_eligibility_active_features_500mv_mean",
        "train_eligibility_pairwise_cosine_mean",
        "csv",
        "wall_time_s",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_scenario: dict[str, Any] = {}
    for scenario in _scenario_list(args.scenario):
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        if not scenario_rows:
            continue
        ok_rows = [row for row in scenario_rows if row["status"] == "ok"]
        best_margin = max(ok_rows, key=lambda row: float(row["final_eval_min_margin_v"])) if ok_rows else None
        best_accuracy = max(ok_rows, key=lambda row: float(row["final_eval_accuracy"])) if ok_rows else None
        by_scenario[scenario] = {
            "best_final_margin_approach": best_margin["approach"] if best_margin is not None else None,
            "best_final_margin_v": best_margin["final_eval_min_margin_v"] if best_margin is not None else None,
            "best_final_accuracy_approach": best_accuracy["approach"] if best_accuracy is not None else None,
            "best_final_accuracy": best_accuracy["final_eval_accuracy"] if best_accuracy is not None else None,
            "passed_count": sum(1 for row in ok_rows if row["passed"]),
            "failed_count": sum(1 for row in scenario_rows if row["status"] == "failed"),
            "total_count": len(scenario_rows),
        }

    summary = {
        "architecture": "normalizer_block_screen",
        "approaches": list(_approach_list(args.approach)),
        "scenario": args.scenario,
        "dataset": args.dataset if args.scenario in {"mnist", "both"} else None,
        "train_samples": args.train_samples if args.scenario in {"mnist", "both"} else None,
        "eval_samples": args.eval_samples if args.scenario in {"mnist", "both"} else None,
        "class_bias_mode": args.class_bias_mode if args.scenario in {"mnist", "both"} else None,
        "score_capacitance_f": args.score_capacitance_f,
        "normalizer_error_clock_high": args.normalizer_error_clock_high,
        "readout_update_mode": args.readout_update_mode,
        "hidden_update_mode": args.hidden_update_mode,
        "hidden_credit_width": args.hidden_credit_width if args.hidden_update_mode != "none" else None,
        "hidden_update_width": args.hidden_update_width if args.hidden_update_mode != "none" else None,
        "score_timing_mode": args.score_timing_mode,
        "readout_forward_mode": args.readout_forward_mode,
        "eligibility_source_mode": args.eligibility_source_mode,
        "csv": str(csv_path),
        "by_scenario": by_scenario,
        "wall_time_s": time.perf_counter() - start,
    }
    summary_path = tables / f"{tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spice-bin", default=None)
    ap.add_argument("--tag", default="normalizer_block_screen")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--approach", choices=("all", *APPROACHES), default="all")
    ap.add_argument("--scenario", choices=SCENARIOS, default="mnist")
    ap.add_argument("--dataset", default="mnist3fixed8_12")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--train-samples", type=int, default=6)
    ap.add_argument("--eval-samples", type=int, default=6)
    ap.add_argument("--onehot-train-repeats", type=int, default=1)
    ap.add_argument("--onehot-eval-repeats", type=int, default=1)
    ap.add_argument("--class-bias-mode", choices=seq.CLASS_BIAS_MODES, default="target-only")
    ap.add_argument("--class-bias-input", type=float, default=0.85)
    ap.add_argument("--score-capacitance-f", type=float, default=5.0)
    ap.add_argument("--score-load-resistance", type=float, default=1e6)
    ap.add_argument("--readout-width", type=float, default=64.0)
    ap.add_argument("--initial-positive", type=float, default=0.40)
    ap.add_argument("--initial-negative", type=float, default=0.40)
    ap.add_argument("--normalizer-error-clock-high", type=float, default=1.2)
    ap.add_argument("--readout-update-mode", choices=seq.READOUT_UPDATE_MODES, default="sampled")
    ap.add_argument("--hidden-update-mode", choices=seq.HIDDEN_UPDATE_MODES, default="none")
    ap.add_argument("--hidden-credit-width", type=float, default=8.0)
    ap.add_argument("--hidden-update-width", type=float, default=0.25)
    ap.add_argument("--score-timing-mode", choices=seq.SCORE_TIMING_MODES, default="late")
    ap.add_argument("--readout-forward-mode", choices=seq.READOUT_FORWARD_MODES, default="direct")
    ap.add_argument("--eligibility-source-mode", choices=seq.ELIGIBILITY_SOURCE_MODES, default="pre-p")
    ap.add_argument("--keep-going", action=argparse.BooleanOptionalAction, default=True)
    return ap


def main_for_test(argv: list[str]) -> argparse.Namespace:
    args = build_arg_parser().parse_args(argv)
    if args.timeout <= 0.0:
        raise ValueError("timeout must be positive")
    if args.train_samples <= 0 or args.eval_samples <= 0:
        raise ValueError("train-samples and eval-samples must be positive")
    if args.onehot_train_repeats <= 0 or args.onehot_eval_repeats <= 0:
        raise ValueError("onehot repeats must be positive")
    if args.score_capacitance_f <= 0.0:
        raise ValueError("score-capacitance-f must be positive")
    if args.score_load_resistance <= 0.0:
        raise ValueError("score-load-resistance must be positive")
    if args.readout_width <= 0.0:
        raise ValueError("readout-width must be positive")
    if min(args.initial_positive, args.initial_negative) <= 0.0:
        raise ValueError("initial weight rails must be positive")
    if args.normalizer_error_clock_high <= 0.0 or args.normalizer_error_clock_high > 1.2:
        raise ValueError("normalizer-error-clock-high must stay in (0, 1.2]")
    if args.hidden_credit_width <= 0.0:
        raise ValueError("hidden-credit-width must be positive")
    if args.hidden_update_width <= 0.0:
        raise ValueError("hidden-update-width must be positive")
    return args


def main() -> None:
    print(json.dumps(run_screen(main_for_test(None)), indent=2))  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
