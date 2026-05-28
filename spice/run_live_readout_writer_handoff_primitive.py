from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from run_device_mnist01_scalar_training import sanitize_tag
from run_device_sequential_training import mos_models, run_netlist
from run_multiclass_output_head_primitive import class_node, signed_store_lines
from run_multiclass_output_head_sequence import class_local_live_label_descent_update_lines
from run_spice_sweep import ROOT, detect_spice


@dataclass(frozen=True)
class HandoffCase:
    name: str
    positive_drive_v: float
    negative_drive_v: float
    eligibility_v: float
    support_v: float
    expected_sign: int


DEFAULT_CASES = (
    HandoffCase("target", positive_drive_v=0.75, negative_drive_v=0.02, eligibility_v=1.2, support_v=1.2, expected_sign=1),
    HandoffCase("nontarget", positive_drive_v=0.02, negative_drive_v=0.75, eligibility_v=1.2, support_v=1.2, expected_sign=-1),
    HandoffCase("off", positive_drive_v=0.75, negative_drive_v=0.02, eligibility_v=0.0, support_v=1.2, expected_sign=0),
    HandoffCase("guarded", positive_drive_v=0.75, negative_drive_v=0.02, eligibility_v=1.2, support_v=0.0, expected_sign=0),
)

MEASURED_COMMON_MODE_CASES = (
    HandoffCase("target_measured", positive_drive_v=0.351, negative_drive_v=0.313, eligibility_v=0.129, support_v=1.2, expected_sign=1),
    HandoffCase("nontarget_measured", positive_drive_v=0.235, negative_drive_v=0.340, eligibility_v=0.129, support_v=1.2, expected_sign=-1),
    HandoffCase("off_measured", positive_drive_v=0.351, negative_drive_v=0.313, eligibility_v=0.0, support_v=1.2, expected_sign=0),
    HandoffCase("guarded_measured", positive_drive_v=0.351, negative_drive_v=0.313, eligibility_v=0.129, support_v=0.0, expected_sign=0),
)

CASE_SETS = {
    "ideal": DEFAULT_CASES,
    "measured-common-mode": MEASURED_COMMON_MODE_CASES,
}


def _constant_or_pulse(node: str, value: float, *, waveform: str, start_ns: float, end_ns: float) -> str:
    if waveform == "dc":
        return f"V{node} {node} 0 {value:.12g}"
    if waveform == "pwl":
        return (
            f"V{node} {node} 0 PWL(0n 0 {start_ns - 0.05:.12g}n 0 "
            f"{start_ns:.12g}n {value:.12g} {end_ns:.12g}n {value:.12g} {end_ns + 0.05:.12g}n 0 8n 0)"
        )
    raise ValueError("waveform must be dc or pwl")


def generate_netlist(
    *,
    cases: tuple[HandoffCase, ...] = DEFAULT_CASES,
    waveform: str = "dc",
    update_width_u: float = 0.25,
    initial_positive: float = 0.40,
    initial_negative: float = 0.40,
    high_ref: float = 0.48,
    low_ref: float = 0.22,
    update_start_ns: float = 2.0,
    update_end_ns: float = 4.0,
    eligibility_skew_ns: float = 0.0,
) -> str:
    if waveform not in {"dc", "pwl"}:
        raise ValueError("waveform must be dc or pwl")
    if min(update_width_u, initial_positive, initial_negative, high_ref, low_ref) < 0.0:
        raise ValueError("voltages and widths must be nonnegative")
    if not low_ref < high_ref <= 1.2:
        raise ValueError("references must satisfy low_ref < high_ref <= 1.2")
    if not update_start_ns < update_end_ns:
        raise ValueError("update_start_ns must be before update_end_ns")
    if update_start_ns + eligibility_skew_ns <= 0.05:
        raise ValueError("eligibility skew moves the PWL edge before the initial zero point")
    if not cases:
        raise ValueError("at least one handoff case is required")
    for case in cases:
        values = (case.positive_drive_v, case.negative_drive_v, case.eligibility_v, case.support_v)
        if min(values) < 0.0 or max(values) > 1.2:
            raise ValueError("case rail voltages must stay within supply rails")

    lines = [
        "* Live readout writer handoff primitive.",
        "* Frozen writer-domain rails drive the same PMOS-differential live writer used in integration.",
        "* No behavioral sources.",
        ".param VDD=1.2",
        mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        f"Vvwhi_ref vwhi_ref 0 {high_ref:.12g}",
        f"Vvwlo_ref vwlo_ref 0 {low_ref:.12g}",
    ]
    for class_idx, case in enumerate(cases):
        pos = class_node(class_idx, "errp")
        neg = class_node(class_idx, "errn")
        elig = class_node(class_idx, "relig0")
        support = class_node(class_idx, "support0")
        pos_ctrl = f"c{class_idx}_f0_live_pos_up_ctrl"
        neg_ctrl = f"c{class_idx}_f0_live_neg_up_ctrl"
        lines += [
            f"* case {class_idx}: {case.name}",
            _constant_or_pulse(pos, case.positive_drive_v, waveform=waveform, start_ns=update_start_ns, end_ns=update_end_ns),
            _constant_or_pulse(neg, case.negative_drive_v, waveform=waveform, start_ns=update_start_ns, end_ns=update_end_ns),
            _constant_or_pulse(
                elig,
                case.eligibility_v,
                waveform=waveform,
                start_ns=update_start_ns + eligibility_skew_ns,
                end_ns=update_end_ns + eligibility_skew_ns,
            ),
            _constant_or_pulse(support, case.support_v, waveform=waveform, start_ns=update_start_ns, end_ns=update_end_ns),
            *signed_store_lines(
                positive_node=class_node(class_idx, "vwp0"),
                negative_node=class_node(class_idx, "vwn0"),
                positive_ic=initial_positive,
                negative_ic=initial_negative,
            ),
            *class_local_live_label_descent_update_lines(
                class_idx=class_idx,
                feature_idx=0,
                activation_node=elig,
                positive_descent_node=pos,
                negative_descent_node=neg,
                update_guard_node=support,
                width_u=update_width_u,
                high_side_topology="pmos-differential",
            ),
            f".meas tran c{class_idx}_vwp_before FIND V({class_node(class_idx, 'vwp0')}) AT={update_start_ns - 0.1:.12g}n",
            f".meas tran c{class_idx}_vwn_before FIND V({class_node(class_idx, 'vwn0')}) AT={update_start_ns - 0.1:.12g}n",
            f".meas tran c{class_idx}_signed_before PARAM='c{class_idx}_vwp_before-c{class_idx}_vwn_before'",
            f".meas tran c{class_idx}_vwp_after FIND V({class_node(class_idx, 'vwp0')}) AT={update_end_ns + 1.0:.12g}n",
            f".meas tran c{class_idx}_vwn_after FIND V({class_node(class_idx, 'vwn0')}) AT={update_end_ns + 1.0:.12g}n",
            f".meas tran c{class_idx}_signed_after PARAM='c{class_idx}_vwp_after-c{class_idx}_vwn_after'",
            f".meas tran c{class_idx}_signed_delta PARAM='c{class_idx}_signed_after-c{class_idx}_signed_before'",
            f".meas tran c{class_idx}_pos_ctrl_min MIN V({pos_ctrl}) FROM={update_start_ns:.12g}n TO={update_end_ns:.12g}n",
            f".meas tran c{class_idx}_neg_ctrl_min MIN V({neg_ctrl}) FROM={update_start_ns:.12g}n TO={update_end_ns:.12g}n",
        ]
    lines += [
        ".tran 2p 8n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def rows_from_measures(measures: dict[str, float], cases: tuple[HandoffCase, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_idx, case in enumerate(cases):
        rows.append(
            {
                "case": case.name,
                "expected_sign": case.expected_sign,
                "positive_drive_v": case.positive_drive_v,
                "negative_drive_v": case.negative_drive_v,
                "eligibility_v": case.eligibility_v,
                "support_v": case.support_v,
                "signed_delta_v": float(measures[f"c{class_idx}_signed_delta"]),
                "pos_ctrl_min_v": float(measures[f"c{class_idx}_pos_ctrl_min"]),
                "neg_ctrl_min_v": float(measures[f"c{class_idx}_neg_ctrl_min"]),
            }
        )
    return rows


def case_passed(row: dict[str, Any], *, min_signed_delta_v: float, ctrl_on_max_v: float, ctrl_off_min_v: float) -> bool:
    expected = int(row["expected_sign"])
    delta = float(row["signed_delta_v"])
    pos_ctrl_min = float(row["pos_ctrl_min_v"])
    neg_ctrl_min = float(row["neg_ctrl_min_v"])
    if expected > 0:
        return delta > min_signed_delta_v and pos_ctrl_min < ctrl_on_max_v and neg_ctrl_min > ctrl_off_min_v
    if expected < 0:
        return delta < -min_signed_delta_v and neg_ctrl_min < ctrl_on_max_v and pos_ctrl_min > ctrl_off_min_v
    return abs(delta) < min_signed_delta_v


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    generated = ROOT / "spice/generated"
    tables = ROOT / "results/tables"
    generated.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    tag = sanitize_tag(args.tag)
    spice_bin, version = detect_spice(args.spice_bin)
    start = time.perf_counter()
    cases = CASE_SETS[args.case_set]
    deck = generate_netlist(
        cases=cases,
        waveform=args.waveform,
        update_width_u=args.update_width,
        high_ref=args.high_ref,
        low_ref=args.low_ref,
        eligibility_skew_ns=args.eligibility_skew_ns,
    )
    path = generated / f"{tag}.cir"
    measures = run_netlist(spice_bin, path, deck, timeout=args.timeout)
    rows = rows_from_measures(measures, cases)
    for row in rows:
        row["passed"] = case_passed(
            row,
            min_signed_delta_v=args.min_signed_delta,
            ctrl_on_max_v=args.ctrl_on_max,
            ctrl_off_min_v=args.ctrl_off_min,
        )
    passed = all(bool(row["passed"]) for row in rows)
    csv_path = tables / f"{tag}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "simulator": version,
        "architecture": "live_readout_writer_handoff_primitive",
        "case_set": args.case_set,
        "waveform": args.waveform,
        "eligibility_skew_ns": args.eligibility_skew_ns,
        "update_width_u": args.update_width,
        "passed": passed,
        "rows": rows,
        "csv": str(csv_path),
        "wall_time_s": time.perf_counter() - start,
    }
    summary_path = tables / f"{tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spice-bin", default=None)
    ap.add_argument("--tag", default="live_readout_writer_handoff_primitive")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--waveform", choices=("dc", "pwl"), default="dc")
    ap.add_argument("--case-set", choices=tuple(CASE_SETS), default="ideal")
    ap.add_argument("--update-width", type=float, default=0.25)
    ap.add_argument("--high-ref", type=float, default=0.48)
    ap.add_argument("--low-ref", type=float, default=0.22)
    ap.add_argument("--eligibility-skew-ns", type=float, default=0.0)
    ap.add_argument("--min-signed-delta", type=float, default=1e-3)
    ap.add_argument("--ctrl-on-max", type=float, default=0.65)
    ap.add_argument("--ctrl-off-min", type=float, default=0.9)
    return ap


def validate_args(args: argparse.Namespace) -> None:
    if args.timeout <= 0.0:
        raise ValueError("timeout must be positive")
    if args.update_width <= 0.0:
        raise ValueError("update-width must be positive")
    if args.case_set not in CASE_SETS:
        raise ValueError(f"case-set must be one of {tuple(CASE_SETS)}")
    if not 0.0 <= args.low_ref < args.high_ref <= 1.2:
        raise ValueError("references must satisfy 0 <= low < high <= 1.2")
    if args.min_signed_delta < 0.0:
        raise ValueError("min-signed-delta must be nonnegative")
    if not 0.0 <= args.ctrl_on_max <= 1.2:
        raise ValueError("ctrl-on-max must stay within supply")
    if not 0.0 <= args.ctrl_off_min <= 1.2:
        raise ValueError("ctrl-off-min must stay within supply")


def main_for_test(argv: list[str]) -> argparse.Namespace:
    args = build_arg_parser().parse_args(argv)
    validate_args(args)
    return args


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)
    print(json.dumps(run_case(args), indent=2))


if __name__ == "__main__":
    main()
