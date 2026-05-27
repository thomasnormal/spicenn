from __future__ import annotations

import argparse
import csv
import json
import time
from typing import Any

from run_device_mnist01_scalar_training import sanitize_tag
from run_device_sequential_training import mos_models, run_netlist
from run_multiclass_block_sequence import class_node, pairwise_decision_node, pairwise_low_gain_winner_lines
from run_spice_sweep import ROOT, detect_spice


COMPETITION_CASES = (
    "target1_clear",
    "target1_low_wrong0",
    "target0_clear",
    "target2_wrong1",
    "near_tie_target1",
)


def case_scores(case: str) -> tuple[tuple[float, float, float], int]:
    if case == "target1_clear":
        return (0.0020, 0.0080, 0.0035), 1
    if case == "target1_low_wrong0":
        return (0.0075, 0.0015, 0.0045), 1
    if case == "target0_clear":
        return (0.0075, 0.0025, 0.0015), 0
    if case == "target2_wrong1":
        return (0.0015, 0.0075, 0.0035), 2
    if case == "near_tie_target1":
        return (0.00350, 0.00351, 0.00349), 1
    raise ValueError(f"case must be one of {COMPETITION_CASES}")


def expected_error_signs(case: str) -> tuple[float, float, float]:
    scores, target = case_scores(case)
    if case == "near_tie_target1":
        return tuple(1.0 if class_idx == target else -1.0 for class_idx in range(3))
    signs = [0.0, 0.0, 0.0]
    target_score = scores[target]
    for class_idx, score in enumerate(scores):
        if class_idx == target:
            continue
        if score > target_score:
            signs[target] = 1.0
            signs[class_idx] = -1.0
    return tuple(signs)


def target_margin(scores: tuple[float, ...], target: int) -> float:
    return scores[target] - max(score for idx, score in enumerate(scores) if idx != target)


def _target_vs_opponent_error_lines(
    *,
    target_idx: int,
    opponent_idx: int,
    error_clock: str,
    width_u: float,
) -> list[str]:
    decision = pairwise_decision_node(opponent_idx, target_idx)
    opposite_decision = pairwise_decision_node(target_idx, opponent_idx)
    errp = class_node(target_idx, "errp")
    errn = class_node(opponent_idx, "errn")
    targetp = class_node(target_idx, "targetp")
    prefix = f"t{target_idx}_o{opponent_idx}_"
    return [
        f"R{prefix}errp_sup {prefix}errp_sup 0 1G",
        f"R{prefix}errp_t {prefix}errp_t 0 1G",
        f"R{prefix}errp_w {prefix}errp_w 0 1G",
        f"M{prefix}errp_sup {prefix}errp_sup {opposite_decision} vdd vdd PMOS W={width_u:.6g}u L=180n",
        f"M{prefix}errp_label {prefix}errp_sup {targetp} {prefix}errp_t 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}errp_win {prefix}errp_t {decision} {prefix}errp_w 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}errp_clk {prefix}errp_w {error_clock} {errp} 0 NSENSE W={width_u:.6g}u L=180n",
        f"R{prefix}errn_sup {prefix}errn_sup 0 1G",
        f"R{prefix}errn_t {prefix}errn_t 0 1G",
        f"R{prefix}errn_w {prefix}errn_w 0 1G",
        f"M{prefix}errn_sup {prefix}errn_sup {opposite_decision} vdd vdd PMOS W={width_u:.6g}u L=180n",
        f"M{prefix}errn_label {prefix}errn_sup {targetp} {prefix}errn_t 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}errn_win {prefix}errn_t {decision} {prefix}errn_w 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}errn_clk {prefix}errn_w {error_clock} {errn} 0 NSENSE W={width_u:.6g}u L=180n",
    ]


def generate_netlist(
    *,
    case: str,
    class_count: int = 3,
    score_values: tuple[float, ...] | None = None,
    target_class: int | None = None,
    pairwise_width_u: float = 64.0,
    error_width_u: float = 128.0,
    error_capacitance_f: float = 4.0,
) -> str:
    if class_count != 3:
        raise ValueError("class_count must currently be 3")
    if case not in COMPETITION_CASES:
        raise ValueError(f"case must be one of {COMPETITION_CASES}")
    if min(pairwise_width_u, error_width_u, error_capacitance_f) <= 0.0:
        raise ValueError("widths and capacitances must be positive")
    default_scores, default_target = case_scores(case)
    scores = default_scores if score_values is None else tuple(float(value) for value in score_values)
    target = default_target if target_class is None else int(target_class)
    if len(scores) != class_count:
        raise ValueError("score_values must have class_count entries")
    if target < 0 or target >= class_count:
        raise ValueError("target_class must be a valid class index")
    if min(scores) < 0.0 or max(scores) > 1.2:
        raise ValueError("score rails must stay within supply rails")

    lines = [
        "* Multiclass score-competition primitive smoke.",
        "* Pairwise score decisions feed mistake-driven target/opponent error rails.",
        "* Python supplies only forced score/label/clock sources; no behavioral sources.",
        ".param VDD=1.2",
        mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        "Vrst rst 0 PULSE(1.2 0 0.50n 10p 10p 8n 10n)",
        "Vscorepre scorepre 0 PULSE(0 1.2 0.50n 10p 10p 8n 10n)",
        "Vscoreamp scoreamp 0 PULSE(0 1.2 0.80n 10p 10p 1.20n 10n)",
        "Vscoredec scoredec 0 PULSE(0 1.2 2.10n 10p 10p 1.20n 10n)",
        "Vscoreerr scoreerr 0 PULSE(0 1.2 3.50n 10p 10p 1.20n 10n)",
    ]
    for class_idx, score in enumerate(scores):
        targetp = 1.2 if class_idx == target else 0.0
        lines += [
            f"V{class_node(class_idx, 'score')} {class_node(class_idx, 'score')} 0 {score:.12g}",
            f"V{class_node(class_idx, 'targetp')} {class_node(class_idx, 'targetp')} 0 {targetp:.12g}",
            f"C{class_node(class_idx, 'errp')} {class_node(class_idx, 'errp')} 0 {error_capacitance_f:.12g}f IC=0",
            f"C{class_node(class_idx, 'errn')} {class_node(class_idx, 'errn')} 0 {error_capacitance_f:.12g}f IC=0",
            f"R{class_node(class_idx, 'errp')} {class_node(class_idx, 'errp')} 0 1G",
            f"R{class_node(class_idx, 'errn')} {class_node(class_idx, 'errn')} 0 1G",
            f"Mreset_{class_node(class_idx, 'errp')} {class_node(class_idx, 'errp')} rst 0 0 NMOS W=4u L=180n",
            f"Mreset_{class_node(class_idx, 'errn')} {class_node(class_idx, 'errn')} rst 0 0 NMOS W=4u L=180n",
        ]
    for class_idx in range(class_count):
        for opponent_idx in range(class_idx + 1, class_count):
            lines += pairwise_low_gain_winner_lines(
                class_a=class_idx,
                class_b=opponent_idx,
                amp_clock_node="scoreamp",
                decision_clock_node="scoredec",
                reset_node="scorepre",
                pullup_width=max(8.0, pairwise_width_u / 4.0),
                pulldown_width=max(12.0, pairwise_width_u),
            )
    for target_idx in range(class_count):
        for opponent_idx in range(class_count):
            if opponent_idx == target_idx:
                continue
            lines += _target_vs_opponent_error_lines(
                target_idx=target_idx,
                opponent_idx=opponent_idx,
                error_clock="scoreerr",
                width_u=error_width_u,
            )
    for class_idx in range(class_count):
        lines += [
            f".meas tran c{class_idx}_errp_after FIND V({class_node(class_idx, 'errp')}) AT=4.90n",
            f".meas tran c{class_idx}_errn_after FIND V({class_node(class_idx, 'errn')}) AT=4.90n",
            f".meas tran c{class_idx}_errdiff PARAM='c{class_idx}_errp_after-c{class_idx}_errn_after'",
        ]
        for opponent_idx in range(class_count):
            if opponent_idx == class_idx:
                continue
            lines += [
                f".meas tran c{class_idx}_gt_c{opponent_idx}_after FIND V({pairwise_decision_node(class_idx, opponent_idx)}) AT=3.40n",
            ]
    for class_idx in range(class_count):
        for opponent_idx in range(class_count):
            if opponent_idx != class_idx:
                lines.append(
                    f".meas tran c{class_idx}_gt_c{opponent_idx}_diff PARAM='c{class_idx}_gt_c{opponent_idx}_after-c{opponent_idx}_gt_c{class_idx}_after'"
                )
    lines += [
        ".tran 2p 5.1n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def classify_sign(actual: float, expected: float, *, min_abs_margin: float) -> str:
    if abs(expected) < 1e-15:
        return "dead_zone" if abs(actual) < min_abs_margin else "active"
    if expected > 0.0 and actual >= min_abs_margin:
        return "aligned"
    if expected < 0.0 and actual <= -min_abs_margin:
        return "aligned"
    if abs(actual) < min_abs_margin:
        return "weak"
    return "flipped"


def classify_case(case: str, measures: dict[str, Any], *, min_abs_margin: float) -> dict[str, str]:
    expected = expected_error_signs(case)
    return {
        f"c{class_idx}_err_classification": classify_sign(
            float(measures[f"c{class_idx}_errdiff"]),
            expected[class_idx],
            min_abs_margin=min_abs_margin,
        )
        for class_idx in range(3)
    }


def run_cases(args: argparse.Namespace) -> dict[str, Any]:
    generated = ROOT / "spice/generated"
    tables = ROOT / "results/tables"
    generated.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    tag = sanitize_tag(args.tag)
    spice_bin, version = detect_spice(args.spice_bin)
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for case in COMPETITION_CASES:
        path = generated / f"{tag}_{sanitize_tag(case)}.cir"
        scores, target = case_scores(case)
        measures = run_netlist(
            spice_bin,
            path,
            generate_netlist(
                case=case,
                pairwise_width_u=args.pairwise_width,
                error_width_u=args.error_width,
                error_capacitance_f=args.error_capacitance_f,
            ),
            timeout=args.timeout,
        )
        row = {
            "case": case,
            "target_class": target,
            "target_margin_v": target_margin(scores, target),
            **measures,
        }
        row.update(classify_case(case, row, min_abs_margin=args.min_abs_margin))
        rows.append(row)
    csv_path = tables / f"{tag}.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    classification_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        for key, value in row.items():
            if key.endswith("_classification"):
                counts = classification_counts.setdefault(key, {})
                cls = str(value)
                counts[cls] = counts.get(cls, 0) + 1
    passed = all(
        str(value) in {"aligned", "dead_zone"}
        for row in rows
        for key, value in row.items()
        if key.endswith("_classification")
    )
    summary = {
        "simulator": version,
        "architecture": "multiclass_pairwise_score_competition_primitive",
        "model_level": "ngspice built-in LEVEL=1 MOS models; not a foundry PDK.",
        "cases": len(rows),
        "passed": passed,
        "classification_counts": classification_counts,
        "csv": str(csv_path),
        "wall_time_s": time.perf_counter() - start,
    }
    summary_path = tables / f"{tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spice-bin", default=None)
    ap.add_argument("--tag", default="multiclass_score_competition_primitive")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--pairwise-width", type=float, default=64.0)
    ap.add_argument("--error-width", type=float, default=128.0)
    ap.add_argument("--error-capacitance-f", type=float, default=4.0)
    ap.add_argument("--min-abs-margin", type=float, default=25e-3)
    return ap


def validate_args(args: argparse.Namespace) -> None:
    if args.timeout <= 0.0:
        raise ValueError("timeout must be positive")
    if args.pairwise_width <= 0.0:
        raise ValueError("pairwise-width must be positive")
    if args.error_width <= 0.0:
        raise ValueError("error-width must be positive")
    if args.error_capacitance_f <= 0.0:
        raise ValueError("error-capacitance-f must be positive")
    if args.min_abs_margin < 0.0:
        raise ValueError("min-abs-margin must be nonnegative")


def main_for_test(argv: list[str]) -> argparse.Namespace:
    args = build_arg_parser().parse_args(argv)
    validate_args(args)
    return args


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)
    print(json.dumps(run_cases(args), indent=2))


if __name__ == "__main__":
    main()
