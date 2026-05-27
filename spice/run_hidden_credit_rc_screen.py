from __future__ import annotations

import argparse
import csv
import json
import time
from typing import Any

from run_device_mnist01_scalar_training import sanitize_tag
from run_normalization_subcircuits import APPROACHES
import run_normalizer_block_screen as normalizer_screen
from run_spice_sweep import ROOT


VARIANTS: dict[str, dict[str, Any]] = {
    "damped-nmos": {
        "hidden_credit_capacitance_f": 50.0,
        "hidden_credit_shunt_resistance": 250000.0,
        "hidden_credit_activation_model": "NMOS",
        "hidden_update_width": 0.10,
        "normalizer_error_clock_high": 0.35,
    },
    "drive-nmos": {
        "hidden_credit_capacitance_f": 50.0,
        "hidden_credit_shunt_resistance": 250000.0,
        "hidden_credit_activation_model": "NMOS",
        "hidden_update_width": 0.25,
        "normalizer_error_clock_high": 0.45,
    },
    "hold-nmos": {
        "hidden_credit_capacitance_f": 50.0,
        "hidden_credit_shunt_resistance": 1000000.0,
        "hidden_credit_activation_model": "NMOS",
        "hidden_update_width": 0.25,
        "normalizer_error_clock_high": 0.45,
    },
    "drive-nsense": {
        "hidden_credit_capacitance_f": 50.0,
        "hidden_credit_shunt_resistance": 250000.0,
        "hidden_credit_activation_model": "NSENSE",
        "hidden_update_width": 0.25,
        "normalizer_error_clock_high": 0.45,
    },
}


HIDDEN_CREDIT_DIAGNOSTIC_FIELDS = (
    "train_hidden_credit_abs_mean_v",
    "train_hidden_credit_abs_max_v",
    "train_hidden_credit_positive_mean_v",
    "train_hidden_credit_negative_mean_v",
    "final_hidden_signed_delta_mean_v",
    "final_hidden_signed_delta_abs_mean_v",
    "final_hidden_signed_delta_min_v",
    "final_hidden_signed_delta_max_v",
)


def _variant_list(variant: str) -> tuple[str, ...]:
    if variant == "all":
        return tuple(VARIANTS)
    if variant in VARIANTS:
        return (variant,)
    raise ValueError(f"variant must be 'all' or one of {tuple(VARIANTS)}")


def _variant_tag_suffix(variant: str) -> str:
    return sanitize_tag(variant.replace("-", "_"))


def _child_metric(summary: dict[str, Any], key: str) -> Any:
    csv_path = summary.get("csv")
    if not csv_path:
        return None
    try:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "ok" and row.get(key, "") != "":
                    return row[key]
    except FileNotFoundError:
        return None
    return None


def _screen_argv(args: argparse.Namespace, *, variant: str, child_tag: str) -> list[str]:
    values = VARIANTS[variant]
    argv = [
        "--tag",
        child_tag,
        "--timeout",
        str(args.timeout),
        "--approach",
        args.approach,
        "--scenario",
        args.scenario,
        "--dataset",
        args.dataset,
        "--seed",
        str(args.seed),
        "--train-samples",
        str(args.train_samples),
        "--eval-samples",
        str(args.eval_samples),
        "--score-capacitance-f",
        str(args.score_capacitance_f),
        "--readout-update-mode",
        "live",
        "--hidden-update-mode",
        "readout-weighted",
        "--hidden-credit-capacitance-f",
        str(values["hidden_credit_capacitance_f"]),
        "--hidden-credit-shunt-resistance",
        str(values["hidden_credit_shunt_resistance"]),
        "--hidden-credit-activation-model",
        str(values["hidden_credit_activation_model"]),
        "--hidden-update-width",
        str(values["hidden_update_width"]),
        "--normalizer-error-clock-high",
        str(values["normalizer_error_clock_high"]),
        "--score-timing-mode",
        args.score_timing_mode,
        "--readout-forward-mode",
        args.readout_forward_mode,
        "--eligibility-source-mode",
        args.eligibility_source_mode,
    ]
    if args.spice_bin is not None:
        argv += ["--spice-bin", args.spice_bin]
    if args.download:
        argv.append("--download")
    return argv


def _row(summary: dict[str, Any], *, variant: str, tag: str) -> dict[str, Any]:
    scenario = str(summary["scenario"])
    scenario_summary = summary["by_scenario"][scenario]
    return {
        "variant": variant,
        "tag": tag,
        "status": "ok",
        "error": "",
        "best_final_accuracy": scenario_summary["best_final_accuracy"],
        "best_final_margin_v": scenario_summary["best_final_margin_v"],
        "passed_count": scenario_summary["passed_count"],
        "failed_count": scenario_summary["failed_count"],
        "hidden_credit_capacitance_f": summary.get("hidden_credit_capacitance_f"),
        "hidden_credit_shunt_resistance": summary.get("hidden_credit_shunt_resistance"),
        "hidden_credit_activation_model": summary.get("hidden_credit_activation_model"),
        "hidden_update_width": summary.get("hidden_update_width"),
        "normalizer_error_clock_high": summary.get("normalizer_error_clock_high"),
        **{field: _child_metric(summary, field) for field in HIDDEN_CREDIT_DIAGNOSTIC_FIELDS},
        "csv": summary["csv"],
        "wall_time_s": summary["wall_time_s"],
    }


def _failure_row(error: Exception, *, variant: str, tag: str) -> dict[str, Any]:
    message = str(error).splitlines()[-1] if str(error).splitlines() else repr(error)
    values = VARIANTS[variant]
    return {
        "variant": variant,
        "tag": tag,
        "status": "failed",
        "error": message[:500],
        "best_final_accuracy": "",
        "best_final_margin_v": "",
        "passed_count": 0,
        "failed_count": 1,
        "hidden_credit_capacitance_f": values["hidden_credit_capacitance_f"],
        "hidden_credit_shunt_resistance": values["hidden_credit_shunt_resistance"],
        "hidden_credit_activation_model": values["hidden_credit_activation_model"],
        "hidden_update_width": values["hidden_update_width"],
        "normalizer_error_clock_high": values["normalizer_error_clock_high"],
        **{field: "" for field in HIDDEN_CREDIT_DIAGNOSTIC_FIELDS},
        "csv": "",
        "wall_time_s": "",
    }


def run_screen(args: argparse.Namespace) -> dict[str, Any]:
    tables = ROOT / "results/tables"
    tables.mkdir(parents=True, exist_ok=True)
    tag = sanitize_tag(args.tag)
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for variant in _variant_list(args.variant):
        child_tag = sanitize_tag(f"{tag}_{_variant_tag_suffix(variant)}")
        child_args = normalizer_screen.main_for_test(_screen_argv(args, variant=variant, child_tag=child_tag))
        try:
            summary = normalizer_screen.run_screen(child_args)
        except Exception as exc:
            if not args.keep_going:
                raise
            rows.append(_failure_row(exc, variant=variant, tag=child_tag))
            continue
        rows.append(_row(summary, variant=variant, tag=child_tag))

    csv_path = tables / f"{tag}.csv"
    fieldnames = [
        "variant",
        "tag",
        "status",
        "error",
        "best_final_accuracy",
        "best_final_margin_v",
        "passed_count",
        "failed_count",
        "hidden_credit_capacitance_f",
        "hidden_credit_shunt_resistance",
        "hidden_credit_activation_model",
        "hidden_update_width",
        "normalizer_error_clock_high",
        *HIDDEN_CREDIT_DIAGNOSTIC_FIELDS,
        "csv",
        "wall_time_s",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [row for row in rows if row["status"] == "ok"]
    best_accuracy = max(ok_rows, key=lambda row: float(row["best_final_accuracy"])) if ok_rows else None
    best_margin = max(ok_rows, key=lambda row: float(row["best_final_margin_v"])) if ok_rows else None
    summary = {
        "architecture": "hidden_credit_rc_screen",
        "variants": list(_variant_list(args.variant)),
        "approach": args.approach,
        "scenario": args.scenario,
        "dataset": args.dataset if args.scenario == "mnist" else None,
        "train_samples": args.train_samples if args.scenario == "mnist" else None,
        "eval_samples": args.eval_samples if args.scenario == "mnist" else None,
        "score_timing_mode": args.score_timing_mode,
        "readout_forward_mode": args.readout_forward_mode,
        "eligibility_source_mode": args.eligibility_source_mode,
        "best_accuracy_variant": best_accuracy["variant"] if best_accuracy is not None else None,
        "best_final_accuracy": best_accuracy["best_final_accuracy"] if best_accuracy is not None else None,
        "best_margin_variant": best_margin["variant"] if best_margin is not None else None,
        "best_final_margin_v": best_margin["best_final_margin_v"] if best_margin is not None else None,
        "csv": str(csv_path),
        "wall_time_s": time.perf_counter() - start,
    }
    summary_path = tables / f"{tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spice-bin", default=None)
    ap.add_argument("--tag", default="hidden_credit_rc_screen")
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--variant", choices=("all", *VARIANTS), default="all")
    ap.add_argument("--approach", choices=APPROACHES, default="current-sum")
    ap.add_argument("--scenario", choices=normalizer_screen.SCENARIOS, default="mnist")
    ap.add_argument("--dataset", default="mnist3fixed8_12")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--train-samples", type=int, default=6)
    ap.add_argument("--eval-samples", type=int, default=6)
    ap.add_argument("--score-capacitance-f", type=float, default=5.0)
    ap.add_argument("--score-timing-mode", choices=normalizer_screen.seq.SCORE_TIMING_MODES, default="early")
    ap.add_argument("--readout-forward-mode", choices=normalizer_screen.seq.READOUT_FORWARD_MODES, default="diode")
    ap.add_argument("--eligibility-source-mode", choices=normalizer_screen.seq.ELIGIBILITY_SOURCE_MODES, default="act")
    ap.add_argument("--keep-going", action=argparse.BooleanOptionalAction, default=True)
    return ap


def main_for_test(argv: list[str]) -> argparse.Namespace:
    args = build_arg_parser().parse_args(argv)
    if args.timeout <= 0.0:
        raise ValueError("timeout must be positive")
    if args.train_samples <= 0 or args.eval_samples <= 0:
        raise ValueError("train-samples and eval-samples must be positive")
    if args.score_capacitance_f <= 0.0:
        raise ValueError("score-capacitance-f must be positive")
    return args


def main() -> None:
    print(json.dumps(run_screen(main_for_test(None)), indent=2))  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
