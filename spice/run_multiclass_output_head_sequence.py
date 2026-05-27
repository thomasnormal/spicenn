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
from run_score_decision_primitive import low_gain_ref_decision_lines, low_gain_ref_state_lines
from run_spice_sweep import ROOT, detect_spice


CYCLE_NS = 16.0
ERROR_MODES = ("label-descent", "score-gated-nontarget", "restored-score-nontarget")


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


def width_scaled_windowed_pwl(
    cycle_values: list[float],
    *,
    start_ns: float,
    end_ns: float,
    width_scale: float,
    cycle_ns: float = CYCLE_NS,
) -> str:
    if width_scale < 0.0 or width_scale > 1.0:
        raise ValueError("width_scale must be in [0, 1]")
    if width_scale >= 1.0:
        return windowed_pwl(cycle_values, start_ns=start_ns, end_ns=end_ns, cycle_ns=cycle_ns)
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    scaled_end_ns = start_ns + (end_ns - start_ns) * width_scale
    for cycle, value in enumerate(cycle_values):
        base = cycle * cycle_ns
        if abs(value) < 1e-15 or scaled_end_ns <= start_ns:
            points += [(base + start_ns - 0.01, 0.0), (base + end_ns + 0.01, 0.0)]
            continue
        points += [
            (base + start_ns - 0.01, 0.0),
            (base + start_ns, value),
            (base + scaled_end_ns, value),
            (base + scaled_end_ns + 0.01, 0.0),
            (base + end_ns + 0.01, 0.0),
        ]
    points.append((len(cycle_values) * cycle_ns, 0.0))
    return pwl(points)


def multi_windowed_pwl(
    cycle_values: list[float],
    *,
    windows: list[tuple[float, float]],
    cycle_ns: float = CYCLE_NS,
) -> str:
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for cycle, value in enumerate(cycle_values):
        base = cycle * cycle_ns
        for start_ns, end_ns in windows:
            points += [
                (base + start_ns - 0.01, 0.0),
                (base + start_ns, value),
                (base + end_ns, value),
                (base + end_ns + 0.01, 0.0),
            ]
    points.append((len(cycle_values) * cycle_ns, 0.0))
    return pwl(points)


def multi_window_phase_pwl(
    cycle_count: int,
    *,
    windows: list[tuple[float, float]],
    high: float = 1.2,
    cycle_ns: float = CYCLE_NS,
) -> str:
    return multi_windowed_pwl([high] * cycle_count, windows=windows, cycle_ns=cycle_ns)


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


def class_local_score_gated_nontarget_gradient_lines(
    *,
    class_idx: int,
    feature_idx: int,
    activation_node: str,
    width_u: float = 24.0,
) -> list[str]:
    prefix = f"c{class_idx}_f{feature_idx}_"
    return [
        f"M{prefix}gvp_a vdd {activation_node} {prefix}gvp_a 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}gvp_d {prefix}gvp_a {class_node(class_idx, 'targetp')} {prefix}gvp_d 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}gvp_g {prefix}gvp_d acc {class_node(class_idx, f'gvp{feature_idx}')} 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_a vdd {activation_node} {prefix}gvn_a 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_label {prefix}gvn_a {class_node(class_idx, 'targetn')} {prefix}gvn_label 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_score {prefix}gvn_label {class_node(class_idx, 'score')} {prefix}gvn_d 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_g {prefix}gvn_d acc {class_node(class_idx, f'gvn{feature_idx}')} 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}rgp_pd {class_node(class_idx, f'rgp{feature_idx}')} {class_node(class_idx, f'gvp{feature_idx}')} 0 0 NSENSE W=16u L=180n",
        f"M{prefix}rgn_pd {class_node(class_idx, f'rgn{feature_idx}')} {class_node(class_idx, f'gvn{feature_idx}')} 0 0 NSENSE W=16u L=180n",
    ]


def class_local_restored_score_nontarget_gradient_lines(
    *,
    class_idx: int,
    feature_idx: int,
    activation_node: str,
    score_gate_node: str,
    width_u: float = 24.0,
) -> list[str]:
    prefix = f"c{class_idx}_f{feature_idx}_"
    return [
        f"M{prefix}gvp_a vdd {activation_node} {prefix}gvp_a 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}gvp_d {prefix}gvp_a {class_node(class_idx, 'targetp')} {prefix}gvp_d 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}gvp_g {prefix}gvp_d acc {class_node(class_idx, f'gvp{feature_idx}')} 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_a vdd {activation_node} {prefix}gvn_a 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_label {prefix}gvn_a {class_node(class_idx, 'targetn')} {prefix}gvn_label 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_score {prefix}gvn_label {score_gate_node} {prefix}gvn_d 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_g {prefix}gvn_d acc {class_node(class_idx, f'gvn{feature_idx}')} 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}rgp_pd {class_node(class_idx, f'rgp{feature_idx}')} {class_node(class_idx, f'gvp{feature_idx}')} 0 0 NSENSE W=16u L=180n",
        f"M{prefix}rgn_pd {class_node(class_idx, f'rgn{feature_idx}')} {class_node(class_idx, f'gvn{feature_idx}')} 0 0 NSENSE W=16u L=180n",
    ]


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
    nontarget_width_scale: float = 1.0,
    error_mode: str = "label-descent",
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
    if nontarget_width_scale < 0.0 or nontarget_width_scale > 1.0:
        raise ValueError("nontarget_width_scale must be in [0, 1]")
    if error_mode not in ERROR_MODES:
        raise ValueError(f"error_mode must be one of {ERROR_MODES}")

    all_records = eval_records + train_records + eval_records
    sequence = ["initial_eval"] * len(eval_records) + ["train"] * len(train_records) + ["final_eval"] * len(eval_records)
    train_cycles = {idx for idx, label in enumerate(sequence) if label == "train"}
    labels = [int(record["label"]) for record in all_records]
    if any(label < 0 or label >= class_count for label in labels):
        raise ValueError("record labels must be valid class indices")
    features = records_to_feature_matrix(all_records, feature_count)
    cycle_count = len(all_records)
    stop_ns = cycle_count * CYCLE_NS
    uses_early_score = error_mode in {"score-gated-nontarget", "restored-score-nontarget"}
    uses_restored_score = error_mode == "restored-score-nontarget"
    update_signal_start_ns = 4.25 if uses_restored_score else 1.8
    update_signal_end_ns = 6.35 if uses_restored_score else 4.2
    acc_start_ns = 4.55 if uses_restored_score else 2.0
    acc_end_ns = 6.25 if uses_restored_score else 4.0
    apply_start_ns = 6.65 if uses_restored_score else 5.0
    apply_end_ns = 8.15 if uses_restored_score else 7.0
    score_reset_windows = [(0.2, 1.0), (8.35, 8.85)] if uses_restored_score else [(0.2, 1.0), (7.4, 8.0)]

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
        f"Vacc acc 0 {periodic_phase_pwl(cycle_count, start_ns=acc_start_ns, end_ns=acc_end_ns, active_cycles=train_cycles)}",
        f"Vapply apply 0 {periodic_phase_pwl(cycle_count, start_ns=apply_start_ns, end_ns=apply_end_ns, active_cycles=train_cycles)}",
        f"Vapplyn applyn 0 {active_low_phase_pwl(cycle_count, start_ns=apply_start_ns, end_ns=apply_end_ns, active_cycles=train_cycles)}",
        f"Vrst rst 0 {periodic_phase_pwl(cycle_count, start_ns=0.2, end_ns=1.0)}",
        f"Vrstscore rstscore 0 {multi_window_phase_pwl(cycle_count, windows=score_reset_windows)}",
    ]
    if uses_restored_score:
        lines += [
            "Voutref outref 0 0.25",
            f"Vscorepre scorepre 0 {active_low_phase_pwl(cycle_count, start_ns=0.2, end_ns=1.0, active_cycles=set(range(cycle_count)))}",
            f"Vscoreamp scoreamp 0 {periodic_phase_pwl(cycle_count, start_ns=1.75, end_ns=3.15, active_cycles=train_cycles)}",
            f"Vscoredec scoredec 0 {periodic_phase_pwl(cycle_count, start_ns=3.45, end_ns=4.15, active_cycles=train_cycles)}",
        ]
    for feature in range(feature_count):
        act_values = [float(value) for value in features[:, feature]]
        actrow_windows = [(1.05, 1.65), (9.0, 12.0)] if uses_early_score else [(9.0, 12.0)]
        lines += [
            f"Vact{feature} act{feature} 0 {windowed_pwl(act_values, start_ns=update_signal_start_ns, end_ns=update_signal_end_ns)}",
            f"Vactrow{feature} actrow{feature} 0 {multi_windowed_pwl(act_values, windows=actrow_windows)}",
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
            f"V{class_node(class_idx, 'targetp')} {class_node(class_idx, 'targetp')} 0 {windowed_pwl(targetp_values, start_ns=update_signal_start_ns, end_ns=update_signal_end_ns)}",
            f"V{class_node(class_idx, 'targetn')} {class_node(class_idx, 'targetn')} 0 {width_scaled_windowed_pwl(targetn_values, start_ns=update_signal_start_ns, end_ns=update_signal_end_ns, width_scale=nontarget_width_scale)}",
            f"C{class_node(class_idx, 'score')} {class_node(class_idx, 'score')} 0 10f IC=0",
            f"C{class_node(class_idx, 'scoren')} {class_node(class_idx, 'scoren')} 0 10f IC=0",
            f"R{class_node(class_idx, 'score')} {class_node(class_idx, 'score')} 0 1e6",
            f"R{class_node(class_idx, 'scoren')} {class_node(class_idx, 'scoren')} 0 1e6",
            f"Mreset_{class_node(class_idx, 'score')} {class_node(class_idx, 'score')} rstscore 0 0 NMOS W=4u L=180n",
            f"Mreset_{class_node(class_idx, 'scoren')} {class_node(class_idx, 'scoren')} rstscore 0 0 NMOS W=4u L=180n",
        ]
        if uses_restored_score:
            prefix = f"c{class_idx}_"
            lines += [
                f"C{prefix}decision {prefix}decision 0 20f IC=0",
                f"C{prefix}decisionn {prefix}decisionn 0 20f IC=0",
                f"R{prefix}decision {prefix}decision 0 1G",
                f"R{prefix}decisionn {prefix}decisionn 0 1G",
                f"Mprecharge_{prefix}decision {prefix}decision scorepre vdd vdd PMOS W=4u L=180n",
                f"Mprecharge_{prefix}decisionn {prefix}decisionn scorepre vdd vdd PMOS W=4u L=180n",
                *low_gain_ref_state_lines(prefix=prefix, reset_node="scorepre"),
                *low_gain_ref_decision_lines(
                    prefix=prefix,
                    score_node=class_node(class_idx, "score"),
                    scoren_node=class_node(class_idx, "scoren"),
                    outref_node="outref",
                    amp_clock_node="scoreamp",
                    decision_clock_node="scoredec",
                ),
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
                *(
                    class_local_score_gated_nontarget_gradient_lines(
                        class_idx=class_idx,
                        feature_idx=feature,
                        activation_node=f"act{feature}",
                    )
                    if error_mode == "score-gated-nontarget"
                    else class_local_restored_score_nontarget_gradient_lines(
                        class_idx=class_idx,
                        feature_idx=feature,
                        activation_node=f"act{feature}",
                        score_gate_node=f"c{class_idx}_decision",
                    )
                    if uses_restored_score
                    else class_local_label_descent_gradient_lines(
                        class_idx=class_idx,
                        feature_idx=feature,
                        activation_node=f"act{feature}",
                    )
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
        row = {
            "cycle": cycle,
            "sequence": seq,
            "label": label,
            "prediction": prediction,
            "correct": prediction == label,
            "score_margin_v": float(scores[label] - max(score for idx, score in enumerate(scores) if idx != label)),
        }
        row.update({f"score_c{class_idx}_v": score for class_idx, score in enumerate(scores)})
        rows.append(row)
    return rows


def accuracy(rows: list[dict[str, Any]], sequence: str) -> float:
    selected = [row for row in rows if row["sequence"] == sequence]
    if not selected:
        return 0.0
    return float(np.mean([bool(row["correct"]) for row in selected]))


def final_signed_weight_matrix(measures: dict[str, float], *, class_count: int, feature_count: int) -> list[list[float]]:
    return [
        [float(measures[f"c{class_idx}_f{feature}_signed_final"]) for feature in range(feature_count)]
        for class_idx in range(class_count)
    ]


def score_matrix(rows: list[dict[str, Any]], *, sequence: str, class_count: int) -> list[list[float]]:
    return [
        [float(row[f"score_c{class_idx}_v"]) for class_idx in range(class_count)]
        for row in rows
        if row["sequence"] == sequence
    ]


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
        initial_positive=args.initial_positive,
        initial_negative=args.initial_negative,
        nontarget_scale=args.nontarget_scale,
        nontarget_width_scale=args.nontarget_width_scale,
        error_mode=args.error_mode,
    )
    path = generated / f"{tag}.cir"
    measures = run_netlist(spice_bin, path, deck, timeout=args.timeout)
    rows = rows_from_measures(all_records, measures, sequence=sequence, class_count=class_count)
    initial_acc = accuracy(rows, "initial_eval")
    final_acc = accuracy(rows, "final_eval")
    final_signed = final_signed_weight_matrix(measures, class_count=class_count, feature_count=args.feature_count)
    final_signed_sums = [float(sum(row)) for row in final_signed]
    final_signed_spread = float(max(final_signed_sums) - min(final_signed_sums)) if final_signed_sums else 0.0
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
                *[f"score_c{class_idx}_v" for class_idx in range(class_count)],
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    final_margins = [float(row["score_margin_v"]) for row in rows if row["sequence"] == "final_eval"]
    summary = {
        "simulator": version,
        "architecture": "continuous_class_local_output_head_sequence",
        "dataset": args.dataset,
        "class_count": class_count,
        "feature_count": args.feature_count,
        "initial_positive": args.initial_positive,
        "initial_negative": args.initial_negative,
        "nontarget_scale": args.nontarget_scale,
        "nontarget_width_scale": args.nontarget_width_scale,
        "error_mode": args.error_mode,
        "train_samples": args.train_samples,
        "eval_samples": args.eval_samples,
        "initial_eval_accuracy": initial_acc,
        "final_eval_accuracy": final_acc,
        "nontrivial_learning_met": final_acc > initial_acc,
        "final_eval_min_margin_v": min(final_margins) if final_margins else None,
        "final_signed_weight_sums_by_class_v": final_signed_sums,
        "final_signed_weight_sum_spread_v": final_signed_spread,
        "final_signed_weight_matrix_v": final_signed,
        "initial_eval_score_matrix_v": score_matrix(rows, sequence="initial_eval", class_count=class_count),
        "final_eval_score_matrix_v": score_matrix(rows, sequence="final_eval", class_count=class_count),
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
    ap.add_argument("--initial-positive", type=float, default=0.36)
    ap.add_argument("--initial-negative", type=float, default=0.34)
    ap.add_argument("--nontarget-scale", type=float, default=0.5)
    ap.add_argument("--nontarget-width-scale", type=float, default=1.0)
    ap.add_argument("--error-mode", choices=ERROR_MODES, default="label-descent")
    ap.add_argument("--timeout", type=float, default=120.0)
    return ap


def validate_args(args: argparse.Namespace) -> None:
    if args.timeout <= 0.0:
        raise ValueError("timeout must be positive")
    if args.train_samples <= 0 or args.eval_samples <= 0:
        raise ValueError("train-samples and eval-samples must be positive")
    if args.feature_count <= 0:
        raise ValueError("feature-count must be positive")
    if min(args.initial_positive, args.initial_negative) < 0.0:
        raise ValueError("initial-positive and initial-negative must be nonnegative")
    if max(args.initial_positive, args.initial_negative) > 1.2:
        raise ValueError("initial-positive and initial-negative must stay within supply rails")
    if args.nontarget_scale < 0.0 or args.nontarget_scale > 1.0:
        raise ValueError("nontarget-scale must be in [0, 1]")
    if args.nontarget_width_scale < 0.0 or args.nontarget_width_scale > 1.0:
        raise ValueError("nontarget-width-scale must be in [0, 1]")
    if args.error_mode not in ERROR_MODES:
        raise ValueError(f"error-mode must be one of {ERROR_MODES}")
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
