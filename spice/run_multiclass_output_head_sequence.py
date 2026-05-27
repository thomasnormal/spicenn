from __future__ import annotations

import argparse
import csv
import json
import time
from typing import Any

import numpy as np

from datasets import dataset_records, parse_counted_mnist_dataset
from run_device_mnist01_scalar_training import sanitize_tag
from run_device_sequential_training import mos_models, run_netlist
from run_multiclass_output_head_primitive import (
    class_local_bounded_update_lines,
    class_local_label_descent_gradient_lines,
    class_local_readout_forward_lines,
    class_node,
    signed_store_lines,
)
from run_spice_sweep import ROOT, detect_spice


CYCLE_NS = 16.0


def pwl(points: list[tuple[float, float]]) -> str:
    merged: list[tuple[float, float]] = []
    for time_ns, value in points:
        if merged and abs(merged[-1][0] - time_ns) < 1e-15:
            merged[-1] = (time_ns, value)
        else:
            merged.append((time_ns, value))
    return "PWL(" + " ".join(f"{time_ns:.6g}n {value:.12g}" for time_ns, value in merged) + ")"


def windowed_pwl(
    cycle_values: list[float],
    *,
    start_ns: float,
    end_ns: float,
    cycle_ns: float = CYCLE_NS,
) -> str:
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for cycle, value in enumerate(cycle_values):
        base = cycle * cycle_ns
        points += [
            (base + start_ns - 0.01, 0.0),
            (base + start_ns, value),
            (base + end_ns, value),
            (base + end_ns + 0.01, 0.0),
        ]
    points.append((len(cycle_values) * cycle_ns, 0.0))
    return pwl(points)


def periodic_phase_pwl(
    cycle_count: int,
    *,
    start_ns: float,
    end_ns: float,
    active_cycles: set[int] | None = None,
    high: float = 1.2,
    cycle_ns: float = CYCLE_NS,
) -> str:
    values = [high if active_cycles is None or cycle in active_cycles else 0.0 for cycle in range(cycle_count)]
    return windowed_pwl(values, start_ns=start_ns, end_ns=end_ns, cycle_ns=cycle_ns)


def active_low_phase_pwl(
    cycle_count: int,
    *,
    start_ns: float,
    end_ns: float,
    active_cycles: set[int],
    cycle_ns: float = CYCLE_NS,
) -> str:
    points: list[tuple[float, float]] = [(0.0, 1.2)]
    for cycle in range(cycle_count):
        base = cycle * cycle_ns
        if cycle in active_cycles:
            points += [
                (base + start_ns - 0.01, 1.2),
                (base + start_ns, 0.0),
                (base + end_ns, 0.0),
                (base + end_ns + 0.01, 1.2),
            ]
        else:
            points += [(base, 1.2), (base + cycle_ns, 1.2)]
    points.append((cycle_count * cycle_ns, 1.2))
    return pwl(points)


def records_to_feature_matrix(records: list[dict[str, Any]], feature_count: int) -> np.ndarray:
    if feature_count <= 0:
        raise ValueError("feature_count must be positive")
    matrix = []
    for record in records:
        inputs = record.get("inputs")
        if not isinstance(inputs, dict):
            raise ValueError("records must contain an inputs dictionary")
        matrix.append([float(inputs[f"x{feature}"]) for feature in range(feature_count)])
    return np.asarray(matrix, dtype=float)


def balanced_train_eval_split(
    records: list[dict[str, Any]],
    *,
    class_count: int,
    train_samples: int,
    eval_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if train_samples % class_count != 0 or eval_samples % class_count != 0:
        raise ValueError("train_samples and eval_samples must be divisible by class_count")
    train_per_class = train_samples // class_count
    eval_per_class = eval_samples // class_count
    train: list[dict[str, Any]] = []
    eval_: list[dict[str, Any]] = []
    for class_idx in range(class_count):
        class_records = [record for record in records if int(record["label"]) == class_idx]
        needed = train_per_class + eval_per_class
        if len(class_records) < needed:
            raise ValueError(f"not enough records for class {class_idx}: need {needed}, got {len(class_records)}")
        train.extend(class_records[:train_per_class])
        eval_.extend(class_records[train_per_class:needed])
    return train, eval_


def generate_netlist(
    *,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    class_count: int,
    feature_count: int,
    initial_positive: float = 0.36,
    initial_negative: float = 0.34,
    target_high: float = 1.1,
    nontarget_scale: float = 0.5,
) -> str:
    if class_count < 2:
        raise ValueError("class_count must be at least 2")
    if not train_records or not eval_records:
        raise ValueError("train_records and eval_records must be nonempty")
    if min(initial_positive, initial_negative, target_high) < 0.0:
        raise ValueError("voltages must be nonnegative")
    if max(initial_positive, initial_negative, target_high) > 1.2:
        raise ValueError("voltages must stay within supply rails")
    if nontarget_scale < 0.0 or nontarget_scale > 1.0:
        raise ValueError("nontarget_scale must be in [0, 1]")

    all_records = eval_records + train_records + eval_records
    sequence = ["initial_eval"] * len(eval_records) + ["train"] * len(train_records) + ["final_eval"] * len(eval_records)
    train_cycles = {idx for idx, label in enumerate(sequence) if label == "train"}
    labels = [int(record["label"]) for record in all_records]
    if any(label < 0 or label >= class_count for label in labels):
        raise ValueError("record labels must be valid class indices")
    features = records_to_feature_matrix(all_records, feature_count)
    cycle_count = len(all_records)
    stop_ns = cycle_count * CYCLE_NS

    lines = [
        "* Continuous class-local multiclass output-head training sequence.",
        "* Python supplies feature rails, labels, and clocks; weights update only through circuit state.",
        "* No behavioral sources.",
        ".param VDD=1.2",
        mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        "Vvwhi_ref vwhi_ref 0 0.42",
        "Vvwlo_ref vwlo_ref 0 0.28",
        f"Vacc acc 0 {periodic_phase_pwl(cycle_count, start_ns=2.0, end_ns=4.0, active_cycles=train_cycles)}",
        f"Vapply apply 0 {periodic_phase_pwl(cycle_count, start_ns=5.0, end_ns=7.0, active_cycles=train_cycles)}",
        f"Vapplyn applyn 0 {active_low_phase_pwl(cycle_count, start_ns=5.0, end_ns=7.0, active_cycles=train_cycles)}",
        f"Vrst rst 0 {periodic_phase_pwl(cycle_count, start_ns=0.2, end_ns=1.0)}",
    ]
    for feature in range(feature_count):
        act_values = [float(value) for value in features[:, feature]]
        lines += [
            f"Vact{feature} act{feature} 0 {windowed_pwl(act_values, start_ns=1.8, end_ns=4.2)}",
            f"Vactrow{feature} actrow{feature} 0 {windowed_pwl(act_values, start_ns=9.0, end_ns=12.0)}",
        ]
    for class_idx in range(class_count):
        targetp_values = [
            target_high if cycle in train_cycles and labels[cycle] == class_idx else 0.0 for cycle in range(cycle_count)
        ]
        targetn_values = [
            0.0 if cycle not in train_cycles or labels[cycle] == class_idx else target_high * nontarget_scale
            for cycle in range(cycle_count)
        ]
        lines += [
            f"V{class_node(class_idx, 'targetp')} {class_node(class_idx, 'targetp')} 0 {windowed_pwl(targetp_values, start_ns=1.8, end_ns=4.2)}",
            f"V{class_node(class_idx, 'targetn')} {class_node(class_idx, 'targetn')} 0 {windowed_pwl(targetn_values, start_ns=1.8, end_ns=4.2)}",
            f"C{class_node(class_idx, 'score')} {class_node(class_idx, 'score')} 0 10f IC=0",
            f"C{class_node(class_idx, 'scoren')} {class_node(class_idx, 'scoren')} 0 10f IC=0",
            f"R{class_node(class_idx, 'score')} {class_node(class_idx, 'score')} 0 1e6",
            f"R{class_node(class_idx, 'scoren')} {class_node(class_idx, 'scoren')} 0 1e6",
            f"Mreset_{class_node(class_idx, 'score')} {class_node(class_idx, 'score')} rst 0 0 NMOS W=4u L=180n",
            f"Mreset_{class_node(class_idx, 'scoren')} {class_node(class_idx, 'scoren')} rst 0 0 NMOS W=4u L=180n",
        ]
        for feature in range(feature_count):
            for node in ("gvp", "gvn"):
                lines += [
                    f"C{class_node(class_idx, f'{node}{feature}')} {class_node(class_idx, f'{node}{feature}')} 0 2f IC=0",
                    f"R{class_node(class_idx, f'{node}{feature}')} {class_node(class_idx, f'{node}{feature}')} 0 1G",
                    f"Mreset_{class_node(class_idx, f'{node}{feature}')} {class_node(class_idx, f'{node}{feature}')} rst 0 0 NMOS W=4u L=180n",
                ]
            for node in ("rgp", "rgn"):
                lines += [
                    f"C{class_node(class_idx, f'{node}{feature}')} {class_node(class_idx, f'{node}{feature}')} 0 4f IC=1.2",
                    f"R{class_node(class_idx, f'{node}{feature}')} {class_node(class_idx, f'{node}{feature}')} vdd 50k",
                ]
            lines += [
                *signed_store_lines(
                    positive_node=class_node(class_idx, f"vwp{feature}"),
                    negative_node=class_node(class_idx, f"vwn{feature}"),
                    positive_ic=initial_positive,
                    negative_ic=initial_negative,
                ),
                *class_local_label_descent_gradient_lines(
                    class_idx=class_idx,
                    feature_idx=feature,
                    activation_node=f"act{feature}",
                ),
                *class_local_bounded_update_lines(class_idx=class_idx, feature_idx=feature),
                *class_local_readout_forward_lines(class_idx=class_idx, feature_idx=feature),
            ]
    for cycle, (record, seq) in enumerate(zip(all_records, sequence)):
        base = cycle * CYCLE_NS
        for class_idx in range(class_count):
            lines += [
                f".meas tran c{class_idx}_score_{cycle} FIND V({class_node(class_idx, 'score')}) AT={base + 10.5:.2f}n",
                f".meas tran c{class_idx}_scoren_{cycle} FIND V({class_node(class_idx, 'scoren')}) AT={base + 10.5:.2f}n",
                f".meas tran c{class_idx}_score_net_{cycle} PARAM='c{class_idx}_score_{cycle}-c{class_idx}_scoren_{cycle}'",
            ]
            if cycle == len(all_records) - 1:
                for feature in range(feature_count):
                    lines += [
                        f".meas tran c{class_idx}_f{feature}_vwp_final FIND V({class_node(class_idx, f'vwp{feature}')}) AT={stop_ns - 0.5:.2f}n",
                        f".meas tran c{class_idx}_f{feature}_vwn_final FIND V({class_node(class_idx, f'vwn{feature}')}) AT={stop_ns - 0.5:.2f}n",
                        f".meas tran c{class_idx}_f{feature}_signed_final PARAM='c{class_idx}_f{feature}_vwp_final-c{class_idx}_f{feature}_vwn_final'",
                    ]
        lines.append(f"* cycle {cycle} {seq} label={int(record['label'])}")
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
        prediction = int(np.argmax(scores))
        label = int(record["label"])
        rows.append(
            {
                "cycle": cycle,
                "sequence": seq,
                "label": label,
                "prediction": prediction,
                "correct": prediction == label,
                "score_margin_v": float(scores[label] - max(score for idx, score in enumerate(scores) if idx != label)),
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
    dataset_info = parse_counted_mnist_dataset(args.dataset)
    if dataset_info is None:
        raise ValueError("dataset must be a counted multiclass MNIST dataset such as mnist3fixed8_6")
    class_count, _frontend, _sample_count = dataset_info
    records = dataset_records(args.dataset, args.seed, root=ROOT, download=args.download)
    train_records, eval_records = balanced_train_eval_split(
        records,
        class_count=class_count,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
    )
    all_records = eval_records + train_records + eval_records
    sequence = ["initial_eval"] * len(eval_records) + ["train"] * len(train_records) + ["final_eval"] * len(eval_records)
    start = time.perf_counter()
    deck = generate_netlist(
        train_records=train_records,
        eval_records=eval_records,
        class_count=class_count,
        feature_count=args.feature_count,
        nontarget_scale=args.nontarget_scale,
    )
    path = generated / f"{tag}.cir"
    measures = run_netlist(spice_bin, path, deck, timeout=args.timeout)
    rows = rows_from_measures(all_records, measures, sequence=sequence, class_count=class_count)
    initial_acc = accuracy(rows, "initial_eval")
    final_acc = accuracy(rows, "final_eval")
    csv_path = tables / f"{tag}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cycle", "sequence", "label", "prediction", "correct", "score_margin_v"])
        writer.writeheader()
        writer.writerows(rows)
    final_margins = [float(row["score_margin_v"]) for row in rows if row["sequence"] == "final_eval"]
    summary = {
        "simulator": version,
        "architecture": "continuous_class_local_output_head_sequence",
        "dataset": args.dataset,
        "class_count": class_count,
        "feature_count": args.feature_count,
        "nontarget_scale": args.nontarget_scale,
        "train_samples": args.train_samples,
        "eval_samples": args.eval_samples,
        "initial_eval_accuracy": initial_acc,
        "final_eval_accuracy": final_acc,
        "nontrivial_learning_met": final_acc > initial_acc,
        "final_eval_min_margin_v": min(final_margins) if final_margins else None,
        "csv": str(csv_path),
        "wall_time_s": time.perf_counter() - start,
    }
    summary_path = tables / f"{tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spice-bin", default=None)
    ap.add_argument("--tag", default="multiclass_output_head_sequence")
    ap.add_argument("--dataset", default="mnist3fixed8_6")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--train-samples", type=int, default=3)
    ap.add_argument("--eval-samples", type=int, default=3)
    ap.add_argument("--feature-count", type=int, default=8)
    ap.add_argument("--nontarget-scale", type=float, default=0.5)
    ap.add_argument("--timeout", type=float, default=120.0)
    return ap


def validate_args(args: argparse.Namespace) -> None:
    if args.timeout <= 0.0:
        raise ValueError("timeout must be positive")
    if args.train_samples <= 0 or args.eval_samples <= 0:
        raise ValueError("train-samples and eval-samples must be positive")
    if args.feature_count <= 0:
        raise ValueError("feature-count must be positive")
    if args.nontarget_scale < 0.0 or args.nontarget_scale > 1.0:
        raise ValueError("nontarget-scale must be in [0, 1]")
    if parse_counted_mnist_dataset(args.dataset) is None:
        raise ValueError("dataset must be a counted multiclass MNIST dataset")


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
