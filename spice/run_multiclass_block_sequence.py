from __future__ import annotations

import argparse
import csv
import json
import time
from typing import Any

import numpy as np

from run_device_mnist01_scalar_training import sanitize_tag
from run_device_sequential_training import mos_models, run_netlist
from run_multiclass_output_head_primitive import (
    class_local_bounded_update_lines,
    class_local_label_descent_gradient_lines,
    class_local_readout_forward_lines,
    class_node,
    signed_store_lines,
)
from run_multiclass_output_head_sequence import (
    CYCLE_NS,
    active_low_phase_pwl,
    periodic_phase_pwl,
    windowed_pwl,
)
from run_spice_sweep import ROOT, detect_spice


def records_to_rows(records: list[dict[str, Any]]) -> list[float]:
    values = []
    for record in records:
        inputs = record.get("inputs")
        if not isinstance(inputs, dict) or "x0" not in inputs:
            raise ValueError("records must contain inputs['x0']")
        value = float(inputs["x0"])
        if value < 0.0 or value > 1.2:
            raise ValueError("input x0 values must stay within supply rails")
        values.append(value)
    return values


def generate_netlist(
    *,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    class_count: int = 3,
    hidden_positive: float = 1.00,
    hidden_negative: float = 0.20,
    hidden_width_u: float = 1.0,
    readout_width_u: float = 64.0,
    initial_positive: float = 0.40,
    initial_negative: float = 0.40,
    target_high: float = 1.1,
) -> str:
    if class_count < 2:
        raise ValueError("class_count must be at least 2")
    if not train_records or not eval_records:
        raise ValueError("train_records and eval_records must be nonempty")
    for name, value in {
        "hidden_positive": hidden_positive,
        "hidden_negative": hidden_negative,
        "hidden_width_u": hidden_width_u,
        "readout_width_u": readout_width_u,
        "initial_positive": initial_positive,
        "initial_negative": initial_negative,
        "target_high": target_high,
    }.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if max(hidden_positive, hidden_negative, initial_positive, initial_negative, target_high) > 1.2:
        raise ValueError("voltages must stay within supply rails")

    all_records = eval_records + train_records + eval_records
    sequence = ["initial_eval"] * len(eval_records) + ["train"] * len(train_records) + ["final_eval"] * len(eval_records)
    train_cycles = {idx for idx, label in enumerate(sequence) if label == "train"}
    labels = [int(record["label"]) for record in all_records]
    if any(label < 0 or label >= class_count for label in labels):
        raise ValueError("record labels must be valid class indices")
    rows = records_to_rows(all_records)
    cycle_count = len(all_records)
    stop_ns = cycle_count * CYCLE_NS

    lines = [
        "* Continuous multiclass block sequence: split-rail hidden feature and class-local readout updates.",
        "* Python supplies row inputs, labels, and clocks only; readout weights persist as capacitor state.",
        "* No behavioral sources.",
        ".param VDD=1.2",
        mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        "Vvwhi_ref vwhi_ref 0 0.42",
        "Vvwlo_ref vwlo_ref 0 0.28",
        f"Vrst rst 0 {periodic_phase_pwl(cycle_count, start_ns=0.2, end_ns=1.0)}",
        f"Vrow0 row0 0 {windowed_pwl(rows, start_ns=1.1, end_ns=4.0)}",
        f"Vsamp samp 0 {periodic_phase_pwl(cycle_count, start_ns=2.5, end_ns=3.5)}",
        f"Vsampn sampn 0 {active_low_phase_pwl(cycle_count, start_ns=2.5, end_ns=3.5, active_cycles=set(range(cycle_count)))}",
        f"Vout out 0 {periodic_phase_pwl(cycle_count, start_ns=5.0, end_ns=8.0)}",
        f"Voutn outn 0 {active_low_phase_pwl(cycle_count, start_ns=5.0, end_ns=8.0, active_cycles=set(range(cycle_count)))}",
        f"Vacc acc 0 {periodic_phase_pwl(cycle_count, start_ns=9.0, end_ns=11.0, active_cycles=train_cycles)}",
        f"Vapply apply 0 {periodic_phase_pwl(cycle_count, start_ns=12.0, end_ns=12.1, active_cycles=train_cycles)}",
        f"Vapplyn applyn 0 {active_low_phase_pwl(cycle_count, start_ns=12.0, end_ns=12.1, active_cycles=train_cycles)}",
        f"Cwhp whp 0 20f IC={hidden_positive:.12g}",
        f"Cwhn whn 0 20f IC={hidden_negative:.12g}",
        "Rwhp whp 0 1e15",
        "Rwhn whn 0 1e15",
        "Cpre_p pre_p 0 20f IC=0",
        "Cpre_n pre_n 0 20f IC=0",
        "Cact_raw act_raw 0 20f IC=0",
        "Cact_store act0 0 20f IC=0",
        "Celig elig0 0 20f IC=0",
        "Cactrow actrow0 0 1f IC=0",
        "Rpre_p pre_p 0 1G",
        "Rpre_n pre_n 0 1G",
        "Ract_raw act_raw 0 1G",
        "Ract_store act0 0 1G",
        "Relig elig0 0 1G",
        "Ractrow actrow0 0 1e12",
        "Mpre_p_rst pre_p rst 0 0 NMOS W=4u L=180n",
        "Mpre_n_rst pre_n rst 0 0 NMOS W=4u L=180n",
        "Mact_raw_rst act_raw rst 0 0 NMOS W=4u L=180n",
        "Mact_store_rst act0 rst 0 0 NMOS W=4u L=180n",
        "Melig_rst elig0 rst 0 0 NMOS W=4u L=180n",
        "Mactrow_rst actrow0 rst 0 0 NMOS W=4u L=180n",
        f"Mhidden_pos row0 whp pre_p 0 NMOS W={hidden_width_u:.6g}u L=180n",
        f"Mhidden_neg row0 whn pre_n 0 NMOS W={hidden_width_u:.6g}u L=180n",
        "Mact_p vdd pre_p act_raw 0 NREL W=24u L=180n",
        "Mact_n act_raw pre_n 0 0 NREL W=24u L=180n",
        "Mact_store_n act0 samp act_raw 0 NMOS W=16u L=180n",
        "Mact_store_p act0 sampn act_raw vdd PMOS W=32u L=180n",
        "Melig_n elig0 samp pre_p 0 NMOS W=16u L=180n",
        "Melig_p elig0 sampn pre_p vdd PMOS W=32u L=180n",
        "Mactrow_n actrow0 out act0 0 NMOS W=16u L=180n",
        "Mactrow_p actrow0 outn act0 vdd PMOS W=32u L=180n",
    ]
    for class_idx in range(class_count):
        targetp_values = [
            target_high if cycle in train_cycles and labels[cycle] == class_idx else 0.0 for cycle in range(cycle_count)
        ]
        targetn_values = [
            0.0 if cycle not in train_cycles or labels[cycle] == class_idx else target_high
            for cycle in range(cycle_count)
        ]
        lines += [
            f"V{class_node(class_idx, 'targetp')} {class_node(class_idx, 'targetp')} 0 {windowed_pwl(targetp_values, start_ns=9.0, end_ns=11.0)}",
            f"V{class_node(class_idx, 'targetn')} {class_node(class_idx, 'targetn')} 0 {windowed_pwl(targetn_values, start_ns=9.0, end_ns=11.0)}",
            f"C{class_node(class_idx, 'score')} {class_node(class_idx, 'score')} 0 10f IC=0",
            f"C{class_node(class_idx, 'scoren')} {class_node(class_idx, 'scoren')} 0 10f IC=0",
            f"R{class_node(class_idx, 'score')} {class_node(class_idx, 'score')} 0 1e6",
            f"R{class_node(class_idx, 'scoren')} {class_node(class_idx, 'scoren')} 0 1e6",
            f"Mreset_{class_node(class_idx, 'score')} {class_node(class_idx, 'score')} rst 0 0 NMOS W=4u L=180n",
            f"Mreset_{class_node(class_idx, 'scoren')} {class_node(class_idx, 'scoren')} rst 0 0 NMOS W=4u L=180n",
            f"C{class_node(class_idx, 'gvp0')} {class_node(class_idx, 'gvp0')} 0 2f IC=0",
            f"C{class_node(class_idx, 'gvn0')} {class_node(class_idx, 'gvn0')} 0 2f IC=0",
            f"C{class_node(class_idx, 'rgp0')} {class_node(class_idx, 'rgp0')} 0 4f IC=1.2",
            f"C{class_node(class_idx, 'rgn0')} {class_node(class_idx, 'rgn0')} 0 4f IC=1.2",
            f"R{class_node(class_idx, 'gvp0')} {class_node(class_idx, 'gvp0')} 0 1G",
            f"R{class_node(class_idx, 'gvn0')} {class_node(class_idx, 'gvn0')} 0 1G",
            f"R{class_node(class_idx, 'rgp0')} {class_node(class_idx, 'rgp0')} vdd 50k",
            f"R{class_node(class_idx, 'rgn0')} {class_node(class_idx, 'rgn0')} vdd 50k",
            f"Mreset_{class_node(class_idx, 'gvp0')} {class_node(class_idx, 'gvp0')} rst 0 0 NMOS W=4u L=180n",
            f"Mreset_{class_node(class_idx, 'gvn0')} {class_node(class_idx, 'gvn0')} rst 0 0 NMOS W=4u L=180n",
            *signed_store_lines(
                positive_node=class_node(class_idx, "vwp0"),
                negative_node=class_node(class_idx, "vwn0"),
                positive_ic=initial_positive,
                negative_ic=initial_negative,
            ),
            *class_local_readout_forward_lines(class_idx=class_idx, feature_idx=0, width_u=readout_width_u),
            *class_local_label_descent_gradient_lines(class_idx=class_idx, feature_idx=0, activation_node="elig0"),
            *class_local_bounded_update_lines(class_idx=class_idx, feature_idx=0),
        ]
    train_seen = 0
    for cycle, (record, seq) in enumerate(zip(all_records, sequence)):
        base = cycle * CYCLE_NS
        lines += [
            f".meas tran pre_p_{cycle} FIND V(pre_p) AT={base + 3.2:.2f}n",
            f".meas tran pre_n_{cycle} FIND V(pre_n) AT={base + 3.2:.2f}n",
            f".meas tran pre_margin_{cycle} PARAM='pre_p_{cycle}-pre_n_{cycle}'",
            f".meas tran act_{cycle} FIND V(act0) AT={base + 4.5:.2f}n",
            f".meas tran elig_{cycle} FIND V(elig0) AT={base + 4.5:.2f}n",
        ]
        for class_idx in range(class_count):
            lines += [
                f".meas tran c{class_idx}_score_{cycle} FIND V({class_node(class_idx, 'score')}) AT={base + 8.5:.2f}n",
                f".meas tran c{class_idx}_scoren_{cycle} FIND V({class_node(class_idx, 'scoren')}) AT={base + 8.5:.2f}n",
                f".meas tran c{class_idx}_score_net_{cycle} PARAM='c{class_idx}_score_{cycle}-c{class_idx}_scoren_{cycle}'",
            ]
        if seq == "train":
            train_seen += 1
            for class_idx in range(class_count):
                lines += [
                    f".meas tran c{class_idx}_vwp_after_train{train_seen} FIND V({class_node(class_idx, 'vwp0')}) AT={base + 15.0:.2f}n",
                    f".meas tran c{class_idx}_vwn_after_train{train_seen} FIND V({class_node(class_idx, 'vwn0')}) AT={base + 15.0:.2f}n",
                    f".meas tran c{class_idx}_signed_after_train{train_seen} PARAM='c{class_idx}_vwp_after_train{train_seen}-c{class_idx}_vwn_after_train{train_seen}'",
                ]
        lines.append(f"* cycle {cycle} {seq} label={int(record['label'])}")
    for class_idx in range(class_count):
        lines += [
            f".meas tran c{class_idx}_vwp_final FIND V({class_node(class_idx, 'vwp0')}) AT={stop_ns - 0.5:.2f}n",
            f".meas tran c{class_idx}_vwn_final FIND V({class_node(class_idx, 'vwn0')}) AT={stop_ns - 0.5:.2f}n",
            f".meas tran c{class_idx}_signed_final PARAM='c{class_idx}_vwp_final-c{class_idx}_vwn_final'",
        ]
    lines += [
        f".tran 5p {stop_ns:.2f}n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def rows_from_measures(
    records: list[dict[str, Any]],
    measures: dict[str, float],
    *,
    sequence: list[str],
    class_count: int,
) -> list[dict[str, Any]]:
    rows = []
    for cycle, (record, seq) in enumerate(zip(records, sequence)):
        scores = [float(measures[f"c{class_idx}_score_net_{cycle}"]) for class_idx in range(class_count)]
        label = int(record["label"])
        prediction = int(np.argmax(scores))
        rows.append(
            {
                "cycle": cycle,
                "sequence": seq,
                "label": label,
                "prediction": prediction,
                "correct": prediction == label,
                "score_margin_v": float(scores[label] - max(score for idx, score in enumerate(scores) if idx != label)),
                **{f"score_c{class_idx}_v": scores[class_idx] for class_idx in range(class_count)},
            }
        )
    return rows


def accuracy(rows: list[dict[str, Any]], sequence: str) -> float:
    selected = [row for row in rows if row["sequence"] == sequence]
    if not selected:
        return 0.0
    return float(np.mean([bool(row["correct"]) for row in selected]))


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    generated = ROOT / "spice/generated"
    tables = ROOT / "results/tables"
    generated.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    tag = sanitize_tag(args.tag)
    spice_bin, version = detect_spice(args.spice_bin)
    train_records = [{"label": args.target_class, "inputs": {"x0": args.input_value}} for _ in range(args.train_samples)]
    eval_records = [{"label": args.target_class, "inputs": {"x0": args.input_value}} for _ in range(args.eval_samples)]
    all_records = eval_records + train_records + eval_records
    sequence = ["initial_eval"] * len(eval_records) + ["train"] * len(train_records) + ["final_eval"] * len(eval_records)
    start = time.perf_counter()
    path = generated / f"{tag}.cir"
    deck = generate_netlist(
        train_records=train_records,
        eval_records=eval_records,
        class_count=args.class_count,
        initial_positive=args.initial_positive,
        initial_negative=args.initial_negative,
    )
    measures = run_netlist(spice_bin, path, deck, timeout=args.timeout)
    rows = rows_from_measures(all_records, measures, sequence=sequence, class_count=args.class_count)
    initial_margins = [float(row["score_margin_v"]) for row in rows if row["sequence"] == "initial_eval"]
    final_margins = [float(row["score_margin_v"]) for row in rows if row["sequence"] == "final_eval"]
    final_signed = [float(measures[f"c{class_idx}_signed_final"]) for class_idx in range(args.class_count)]
    train_progress = []
    for train_idx in range(1, args.train_samples + 1):
        train_progress.append(
            [float(measures[f"c{class_idx}_signed_after_train{train_idx}"]) for class_idx in range(args.class_count)]
        )
    csv_path = tables / f"{tag}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cycle",
                "sequence",
                "label",
                "prediction",
                "correct",
                "score_margin_v",
                *[f"score_c{class_idx}_v" for class_idx in range(args.class_count)],
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    initial_margin = min(initial_margins)
    final_margin = min(final_margins)
    summary = {
        "simulator": version,
        "architecture": "continuous_multiclass_split_rail_block_sequence",
        "class_count": args.class_count,
        "target_class": args.target_class,
        "train_samples": args.train_samples,
        "eval_samples": args.eval_samples,
        "initial_eval_accuracy": accuracy(rows, "initial_eval"),
        "final_eval_accuracy": accuracy(rows, "final_eval"),
        "initial_eval_min_margin_v": initial_margin,
        "final_eval_min_margin_v": final_margin,
        "margin_improvement_v": final_margin - initial_margin,
        "final_signed_by_class_v": final_signed,
        "signed_after_each_train_v": train_progress,
        "passed": final_margin > initial_margin and final_signed[args.target_class] > args.min_target_signed,
        "csv": str(csv_path),
        "wall_time_s": time.perf_counter() - start,
    }
    summary_path = tables / f"{tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spice-bin", default=None)
    ap.add_argument("--tag", default="multiclass_block_sequence")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--class-count", type=int, default=3)
    ap.add_argument("--target-class", type=int, default=0)
    ap.add_argument("--train-samples", type=int, default=2)
    ap.add_argument("--eval-samples", type=int, default=1)
    ap.add_argument("--input-value", type=float, default=0.85)
    ap.add_argument("--initial-positive", type=float, default=0.40)
    ap.add_argument("--initial-negative", type=float, default=0.40)
    ap.add_argument("--min-target-signed", type=float, default=10e-3)
    return ap


def validate_args(args: argparse.Namespace) -> None:
    if args.timeout <= 0.0:
        raise ValueError("timeout must be positive")
    if args.class_count < 2:
        raise ValueError("class-count must be at least 2")
    if args.target_class < 0 or args.target_class >= args.class_count:
        raise ValueError("target-class must be a valid class index")
    if args.train_samples <= 0 or args.eval_samples <= 0:
        raise ValueError("train-samples and eval-samples must be positive")
    if args.input_value < 0.0 or args.input_value > 1.2:
        raise ValueError("input-value must stay within supply rails")
    if min(args.initial_positive, args.initial_negative) <= 0.0:
        raise ValueError("initial-positive and initial-negative must be positive")
    if max(args.initial_positive, args.initial_negative) > 1.2:
        raise ValueError("initial-positive and initial-negative must stay within supply rails")
    if args.min_target_signed < 0.0:
        raise ValueError("min-target-signed must be nonnegative")


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
