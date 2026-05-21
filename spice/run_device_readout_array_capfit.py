#!/usr/bin/env python3
"""Fit readout capacitor states against the production readout-array deck.

This is slower than the branch-surface fitter, but it optimizes against the
actual simultaneous score-cap dynamics used by the full device-level network.
Use narrow scopes first, for example only output-bias caps, then expand once
the array-level objective is no longer the bottleneck.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_device_readout_array_eval as array_eval
import run_device_readout_capfit as branch_capfit
import run_device_readout_transfer as readout_transfer
import run_device_xor2_random_hidden as direct_flow
from run_spice_sweep import ROOT, detect_spice, run_tiny_test


RESULTS = ROOT / "results/tables"
SPICE_RESULTS = ROOT / "spice/results"
GENERATED = ROOT / "spice/generated"


def safe_tag(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in text)


def parse_float_list(text: str) -> list[float]:
    values = [float(part) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return values


def cap_variable_names(
    state: dict[str, float],
    scope: str,
    readout_fanins: direct_flow.ReadoutFanins | None = None,
) -> list[str]:
    if scope == "all":
        names = list(state)
    elif scope == "bias":
        names = [name for name in state if name.startswith("vbo")]
    elif scope == "readout":
        names = [name for name in state if name.startswith("vw")]
    else:
        raise ValueError(f"unknown variable scope: {scope}")
    if readout_fanins is not None:
        active_readout = {
            f"vw{out}{hidden}{branch}"
            for out, srcs in readout_fanins.items()
            for hidden in srcs
            for branch in ("p", "n")
        }
        names = [
            name
            for name in names
            if name.startswith("vbo") or not name.startswith("vw") or name in active_readout
        ]
    return sorted(names)


def cap_role(name: str, *, hidden_cells: int, outputs: int) -> tuple[str, int, int | None, str] | None:
    for out in range(outputs):
        for branch in ("p", "n"):
            if name == f"vbo{out}{branch}":
                return ("bias", out, None, branch)
        for hidden in range(hidden_cells):
            for branch in ("p", "n"):
                if name == f"vw{out}{hidden}{branch}":
                    return ("readout", out, hidden, branch)
    return None


def activation_priority_value(activations: pd.DataFrame, sample_idx: int, hidden: int) -> float:
    """Return the relevant activation magnitude for search ordering.

    Full-deck tables can carry multiple measured activation samples per cycle
    as act<h>_<offset>.  The compatibility act<h> column is not necessarily the
    readout decision instant, so ranking by the largest measured excursion is a
    better guide for which readout caps can affect current mistakes.
    """
    prefix = f"act{hidden}_"
    columns = [col for col in activations.columns if col.startswith(prefix)]
    if f"act{hidden}" in activations:
        columns.append(f"act{hidden}")
    if not columns:
        return 0.0
    return max(abs(float(activations.iloc[sample_idx].get(col, 0.0))) for col in columns)


def error_activation_priorities(
    variables: list[str],
    activations: pd.DataFrame,
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    hidden_cells: int,
    outputs: int,
    near_margin_v: float = 0.05,
) -> dict[str, float]:
    """Rank cap states by how much they can affect current mistakes.

    This is a search heuristic only.  It does not assume a linear readout model:
    it just points the expensive SPICE coordinate search at caps connected to
    high-activation hidden cells on samples with bad or fragile score margins.
    """
    predictions = np.argmax(scores, axis=1)
    other_scores = np.where(np.eye(outputs, dtype=bool)[labels], -np.inf, scores)
    margins = scores[np.arange(len(labels)), labels] - np.max(other_scores, axis=1)
    priority = {name: 0.0 for name in variables}
    roles = {name: cap_role(name, hidden_cells=hidden_cells, outputs=outputs) for name in variables}
    for sample_idx, (label, pred, margin) in enumerate(zip(labels, predictions, margins)):
        if pred == label:
            sample_weight = max(0.0, near_margin_v - float(margin))
            active_outputs = {int(label)}
        else:
            sample_weight = 1.0 + max(0.0, -float(margin))
            active_outputs = {int(label), int(pred)}
        if sample_weight <= 0.0:
            continue
        act_values = [
            activation_priority_value(activations, sample_idx, hidden)
            for hidden in range(hidden_cells)
        ]
        for name, role in roles.items():
            if role is None:
                continue
            kind, out, hidden, _branch = role
            if out not in active_outputs:
                continue
            if kind == "bias":
                priority[name] += sample_weight
            elif hidden is not None:
                priority[name] += sample_weight * act_values[hidden]
    return priority


def order_cap_variables(
    variables: list[str],
    activations: pd.DataFrame,
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    hidden_cells: int,
    outputs: int,
    variable_order: str,
) -> list[str]:
    variables = sorted(variables)
    if variable_order == "alphabetical":
        return variables
    if variable_order != "error_activation":
        raise ValueError(f"unknown variable order: {variable_order}")
    priority = error_activation_priorities(
        variables,
        activations,
        scores,
        labels,
        hidden_cells=hidden_cells,
        outputs=outputs,
    )
    return sorted(variables, key=lambda name: (-priority[name], name))


def _readout_pair_key(name: str, *, hidden_cells: int, outputs: int) -> tuple[int, int] | None:
    role = cap_role(name, hidden_cells=hidden_cells, outputs=outputs)
    if role is None:
        return None
    kind, out, hidden, _branch = role
    if kind != "readout" or hidden is None:
        return None
    return (out, hidden)


def variable_groups(
    variables: list[str],
    *,
    hidden_cells: int,
    outputs: int,
    group_mode: str,
) -> list[tuple[str, ...]]:
    if group_mode == "independent":
        return [(name,) for name in variables]
    if group_mode != "readout_pn_mirror":
        raise ValueError(f"unknown variable group mode: {group_mode}")
    seen: set[str] = set()
    groups: list[tuple[str, ...]] = []
    available = set(variables)
    for name in variables:
        if name in seen:
            continue
        key = _readout_pair_key(name, hidden_cells=hidden_cells, outputs=outputs)
        if key is None:
            groups.append((name,))
            seen.add(name)
            continue
        out, hidden = key
        pair = (f"vw{out}{hidden}n", f"vw{out}{hidden}p")
        if pair[0] in available and pair[1] in available:
            groups.append(pair)
            seen.update(pair)
        else:
            groups.append((name,))
            seen.add(name)
    return groups


def group_candidate_states(
    state: dict[str, float],
    group: tuple[str, ...],
    cap_values: list[float],
    *,
    group_mode: str,
) -> list[dict[str, float]]:
    if len(group) == 1 or group_mode == "independent":
        name = group[0]
        return [
            {name: float(candidate)}
            for candidate in cap_values
            if abs(float(candidate) - float(state[name])) >= 1e-12
        ]
    if group_mode != "readout_pn_mirror" or len(group) != 2:
        raise ValueError(f"unsupported variable group {group!r} for mode {group_mode!r}")
    n_name = next(name for name in group if name.endswith("n"))
    p_name = next(name for name in group if name.endswith("p"))
    values = sorted(float(value) for value in cap_values)
    candidates: list[dict[str, float]] = []
    seen: set[tuple[float, float]] = set()
    for low, high in zip(values, reversed(values)):
        for p_value, n_value in ((high, low), (low, high)):
            key = (round(p_value, 12), round(n_value, 12))
            if key in seen:
                continue
            seen.add(key)
            if abs(p_value - float(state[p_name])) < 1e-12 and abs(n_value - float(state[n_name])) < 1e-12:
                continue
            candidates.append({p_name: p_value, n_name: n_value})
    return candidates


def score_matrix(df: pd.DataFrame, outputs: int) -> np.ndarray:
    return df[[f"score{out}" for out in range(outputs)]].to_numpy(dtype=float)


def metric_tuple(scores: np.ndarray, labels: np.ndarray, selection_metric: str) -> tuple[float, float, float]:
    acc = branch_capfit.accuracy(scores, labels)
    loss = branch_capfit.stable_ce(scores, labels)
    margins = scores[np.arange(len(labels)), labels] - np.max(
        np.where(
            np.eye(scores.shape[1], dtype=bool)[labels],
            -np.inf,
            scores,
        ),
        axis=1,
    )
    min_margin = float(np.min(margins)) if len(margins) else 0.0
    if selection_metric == "accuracy":
        return (acc, min_margin, -loss)
    if selection_metric == "loss":
        return (-loss, acc, min_margin)
    if selection_metric == "margin":
        return (min_margin, acc, -loss)
    raise ValueError(f"unknown selection metric: {selection_metric}")


def summarize_scores(scores: np.ndarray, labels: np.ndarray, selection_metric: str) -> dict[str, Any]:
    margins = scores[np.arange(len(labels)), labels] - np.max(
        np.where(np.eye(scores.shape[1], dtype=bool)[labels], -np.inf, scores),
        axis=1,
    )
    predictions = np.argmax(scores, axis=1)
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "loss": branch_capfit.stable_ce(scores, labels),
        "min_margin_v": float(np.min(margins)),
        "prediction_histogram": {
            str(label): int(count)
            for label, count in zip(*np.unique(predictions.astype(int), return_counts=True))
        },
        "selection_key": list(metric_tuple(scores, labels, selection_metric)),
    }


def state_key(state: dict[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple((name, round(float(value), 12)) for name, value in sorted(state.items()))


def evaluate_state(
    *,
    spice_bin: str,
    tag: str,
    eval_index: int,
    activations: pd.DataFrame,
    labels: np.ndarray,
    state: dict[str, float],
    hidden_cells: int,
    outputs: int,
    design: str,
    output_head: str,
    hidden_cap_f: float,
    score_reset_v: float,
    score_cap_f: float,
    output_cap_f: float,
    sample_ns: float,
    cycle_ns: float,
    activation_drive: str,
    activation_settle_ns: float,
    activation_sample_offsets_ns: list[float],
    readout_load_mode: str,
    score_sense_mode: str,
    tran_step_ps: float,
    spice_accuracy_preset: str,
    score_diode_width_u: float,
    readout_fanins: direct_flow.ReadoutFanins | None,
    timeout: float,
    selection_metric: str,
    cache: dict[tuple[tuple[str, float], ...], dict[str, Any]],
) -> dict[str, Any]:
    key = state_key(state)
    if key in cache:
        return cache[key]
    deck, _samples = array_eval.readout_array_netlist(
        activations=activations,
        readout_state=state,
        design_name=design,
        output_head=output_head,
        hidden_cells=hidden_cells,
        outputs=outputs,
        hidden_cap_f=hidden_cap_f,
        score_reset_v=score_reset_v,
        score_cap_f=score_cap_f,
        output_cap_f=output_cap_f,
        sample_ns=sample_ns,
        cycle_ns=cycle_ns,
        activation_drive=activation_drive,
        activation_settle_ns=activation_settle_ns,
        activation_sample_offsets_ns=activation_sample_offsets_ns,
        readout_load_mode=readout_load_mode,
        tran_step_ps=tran_step_ps,
        spice_accuracy_preset=spice_accuracy_preset,
        score_sense_mode=score_sense_mode,
        score_diode_width_u=score_diode_width_u,
        readout_fanins=readout_fanins,
    )
    parsed = array_eval.run_netlist(spice_bin, GENERATED / f"{safe_tag(tag)}_eval{eval_index:05d}.cir", deck, timeout)
    df = array_eval.rows_from_measures(parsed, activations, outputs)
    scores = score_matrix(df, outputs)
    result = {
        "scores": scores,
        "rows": df,
        **summarize_scores(scores, labels, selection_metric),
    }
    cache[key] = result
    return result


def write_cap_csv(state: dict[str, float], path: Path) -> None:
    pd.DataFrame([{"cap": k, "value": v} for k, v in sorted(state.items())]).to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="device_readout_array_capfit")
    ap.add_argument("--simulator")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--activation-csv", type=Path, required=True)
    ap.add_argument("--activation-phase", default="initial_eval")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--init-cap-csv", type=Path, required=True)
    ap.add_argument("--hidden-cells", type=int, default=8)
    ap.add_argument("--outputs", type=int, default=3)
    ap.add_argument("--design", choices=sorted(direct_flow.SYNAPSE_DESIGNS), default="split_signed_v1")
    ap.add_argument("--output-head", choices=sorted(direct_flow.OUTPUT_HEAD_MODES), default="split_score_none")
    ap.add_argument("--hidden-cap-f", type=float, default=4.0)
    ap.add_argument("--score-reset-v", type=float, default=0.0)
    ap.add_argument("--score-cap-f", type=float, default=10.0)
    ap.add_argument("--output-cap-f", type=float, default=20.0)
    ap.add_argument("--sample-ns", type=float, default=2.95)
    ap.add_argument("--cycle-ns", type=float, default=4.0)
    ap.add_argument("--activation-drive", choices=["held", "ramp", "measured"], default="held")
    ap.add_argument("--activation-settle-ns", type=float, default=2.95)
    ap.add_argument("--activation-sample-offsets-ns", default="")
    ap.add_argument("--readout-load-mode", choices=["forward_only", "flow_offstate"], default="forward_only")
    ap.add_argument("--score-sense-mode", choices=array_eval.SCORE_SENSE_MODES, default="score_caps")
    ap.add_argument("--tran-step-ps", type=float, default=10.0)
    ap.add_argument("--spice-accuracy-preset", choices=sorted(direct_flow.SPICE_ACCURACY_PRESETS), default="fast")
    ap.add_argument("--score-diode-width-u", type=float, default=1024.0)
    ap.add_argument("--cap-values", type=parse_float_list, default=parse_float_list("0.01,0.08,0.16,0.24,0.32,0.4,0.48,0.56,0.64,0.72,0.8,0.88,0.96,1.04,1.12,1.15"))
    ap.add_argument("--variable-scope", choices=["bias", "readout", "all"], default="bias")
    ap.add_argument("--max-vars", type=int)
    ap.add_argument("--max-evals", type=int, default=200)
    ap.add_argument("--sweeps", type=int, default=1)
    ap.add_argument("--target-accuracy", type=float)
    ap.add_argument("--selection-metric", choices=["accuracy", "loss", "margin"], default="accuracy")
    ap.add_argument("--variable-order", choices=["error_activation", "alphabetical"], default="error_activation")
    ap.add_argument(
        "--variable-group-mode",
        choices=["independent", "readout_pn_mirror"],
        default="independent",
        help="independent sweeps one capacitor at a time; readout_pn_mirror tests complementary p/n pairs for readout caps.",
    )
    ap.add_argument("--readout-topology", choices=["dense", "random_fanin", "random_fanout"], default="dense")
    ap.add_argument("--readout-fan-in", type=int, default=3)
    ap.add_argument("--readout-fan-out", type=int, default=3)
    ap.add_argument("--readout-topology-seed", type=int, default=0)
    args = ap.parse_args()

    if args.hidden_cells <= 0 or args.outputs <= 1:
        raise SystemExit("--hidden-cells must be positive and --outputs must be at least two.")
    if args.max_evals <= 0 or args.sweeps <= 0:
        raise SystemExit("--max-evals and --sweeps must be positive.")
    if args.target_accuracy is not None and not 0.0 <= args.target_accuracy <= 1.0:
        raise SystemExit("--target-accuracy must be in 0..1.")
    if args.score_diode_width_u <= 0.0:
        raise SystemExit("--score-diode-width-u must be positive.")
    if args.readout_fan_in <= 0 or args.readout_fan_out <= 0:
        raise SystemExit("--readout-fan-in and --readout-fan-out must be positive.")
    if any(value < 0.0 or value > 1.2 for value in args.cap_values):
        raise SystemExit("--cap-values must stay inside 0..1.2 V.")
    try:
        activation_sample_offsets_ns = (
            direct_flow.parse_offsets(args.activation_sample_offsets_ns)
            if args.activation_sample_offsets_ns.strip()
            else []
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.activation_drive == "measured" and not activation_sample_offsets_ns:
        raise SystemExit("--activation-drive measured requires --activation-sample-offsets-ns.")

    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(args.hidden_cells)
        direct_flow.set_output_count(args.outputs)
        current_state = direct_flow.csv_readout_cap_state(args.init_cap_csv)
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    try:
        readout_fanins = array_eval.build_readout_fanins(
            args.readout_topology,
            hidden_cells=args.hidden_cells,
            outputs=args.outputs,
            fan_in=args.readout_fan_in,
            fan_out=min(args.readout_fan_out, args.outputs),
            seed=args.readout_topology_seed,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    candidate_variables = cap_variable_names(current_state, args.variable_scope, readout_fanins)
    if not candidate_variables:
        raise SystemExit(f"no variables selected by --variable-scope {args.variable_scope!r}")

    activations = readout_transfer.load_activations(
        args.activation_csv,
        args.activation_phase,
        args.hidden_cells,
        args.limit,
    )
    labels = activations["label"].astype(int).to_numpy()
    spice_bin, version = detect_spice(args.simulator)
    for directory in (GENERATED, RESULTS, SPICE_RESULTS):
        directory.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, GENERATED)

    safe = safe_tag(args.tag)
    cache: dict[tuple[tuple[str, float], ...], dict[str, Any]] = {}
    t0 = time.perf_counter()
    eval_count = 0

    current = evaluate_state(
        spice_bin=spice_bin,
        tag=safe,
        eval_index=eval_count,
        activations=activations,
        labels=labels,
        state=current_state,
        hidden_cells=args.hidden_cells,
        outputs=args.outputs,
        design=args.design,
        output_head=args.output_head,
        hidden_cap_f=args.hidden_cap_f,
        score_reset_v=args.score_reset_v,
        score_cap_f=args.score_cap_f,
        output_cap_f=args.output_cap_f,
        sample_ns=args.sample_ns,
        cycle_ns=args.cycle_ns,
        activation_drive=args.activation_drive,
        activation_settle_ns=args.activation_settle_ns,
        activation_sample_offsets_ns=activation_sample_offsets_ns,
        readout_load_mode=args.readout_load_mode,
        score_sense_mode=args.score_sense_mode,
        tran_step_ps=args.tran_step_ps,
        spice_accuracy_preset=args.spice_accuracy_preset,
        score_diode_width_u=args.score_diode_width_u,
        readout_fanins=readout_fanins,
        timeout=args.timeout,
        selection_metric=args.selection_metric,
        cache=cache,
    )
    eval_count += 1
    current_key = tuple(current["selection_key"])
    variables = order_cap_variables(
        candidate_variables,
        activations,
        current["scores"],
        labels,
        hidden_cells=args.hidden_cells,
        outputs=args.outputs,
        variable_order=args.variable_order,
    )
    if args.max_vars is not None:
        variables = variables[: args.max_vars]
    groups = variable_groups(
        variables,
        hidden_cells=args.hidden_cells,
        outputs=args.outputs,
        group_mode=args.variable_group_mode,
    )
    history: list[dict[str, Any]] = [
        {"event": "initial", "evals": eval_count, **{k: current[k] for k in ["accuracy", "loss", "min_margin_v"]}}
    ]

    stop = False
    for sweep in range(args.sweeps):
        changed = 0
        for group in groups:
            if eval_count >= args.max_evals or (
                args.target_accuracy is not None and current["accuracy"] >= args.target_accuracy
            ):
                stop = True
                break
            base_values = {name: float(current_state[name]) for name in group}
            best_values = dict(base_values)
            best_result = current
            best_key = current_key
            for candidate_updates in group_candidate_states(
                current_state,
                group,
                args.cap_values,
                group_mode=args.variable_group_mode,
            ):
                if eval_count >= args.max_evals or (
                    args.target_accuracy is not None and current["accuracy"] >= args.target_accuracy
                ):
                    stop = True
                    break
                trial_state = dict(current_state)
                trial_state.update(candidate_updates)
                result = evaluate_state(
                    spice_bin=spice_bin,
                    tag=safe,
                    eval_index=eval_count,
                    activations=activations,
                    labels=labels,
                    state=trial_state,
                    hidden_cells=args.hidden_cells,
                    outputs=args.outputs,
                    design=args.design,
                    output_head=args.output_head,
                    hidden_cap_f=args.hidden_cap_f,
                    score_reset_v=args.score_reset_v,
                    score_cap_f=args.score_cap_f,
                    output_cap_f=args.output_cap_f,
                    sample_ns=args.sample_ns,
                    cycle_ns=args.cycle_ns,
                    activation_drive=args.activation_drive,
                    activation_settle_ns=args.activation_settle_ns,
                    activation_sample_offsets_ns=activation_sample_offsets_ns,
                    readout_load_mode=args.readout_load_mode,
                    score_sense_mode=args.score_sense_mode,
                    tran_step_ps=args.tran_step_ps,
                    spice_accuracy_preset=args.spice_accuracy_preset,
                    score_diode_width_u=args.score_diode_width_u,
                    readout_fanins=readout_fanins,
                    timeout=args.timeout,
                    selection_metric=args.selection_metric,
                    cache=cache,
                )
                eval_count += 1
                key = tuple(result["selection_key"])
                if key > best_key:
                    best_key = key
                    best_values = {name: float(trial_state[name]) for name in group}
                    best_result = result
            if any(abs(best_values[name] - base_values[name]) >= 1e-12 for name in group):
                for name, value in best_values.items():
                    current_state[name] = value
                current = best_result
                current_key = best_key
                changed += 1
                history.append(
                    {
                        "event": "update",
                        "sweep": sweep,
                        "variable": ",".join(group),
                        "old_value": base_values if len(group) > 1 else next(iter(base_values.values())),
                        "new_value": best_values if len(group) > 1 else next(iter(best_values.values())),
                        "evals": eval_count,
                        **{k: current[k] for k in ["accuracy", "loss", "min_margin_v"]},
                    }
                )
        history.append(
            {
                "event": "sweep_end",
                "sweep": sweep,
                "changed": changed,
                "evals": eval_count,
                **{k: current[k] for k in ["accuracy", "loss", "min_margin_v"]},
            }
        )
        if changed == 0 or stop:
            break

    cap_csv = SPICE_RESULTS / f"{safe}_caps.csv"
    write_cap_csv(current_state, cap_csv)
    result_df = current["rows"].copy()
    table_csv = RESULTS / f"{safe}.csv"
    result_df.to_csv(table_csv, index=False)
    summary = {
        "tag": safe,
        "simulator": version,
        "activation_csv": str(args.activation_csv),
        "activation_phase": args.activation_phase,
        "init_cap_csv": str(args.init_cap_csv),
        "cap_csv": str(cap_csv),
        "table_csv": str(table_csv),
        "samples": int(len(result_df)),
        "hidden_cells": int(args.hidden_cells),
        "outputs": int(args.outputs),
        "design": args.design,
        "output_head": args.output_head,
        "readout_load_mode": args.readout_load_mode,
        "score_sense_mode": args.score_sense_mode,
        "score_diode_width_u": float(args.score_diode_width_u),
        "readout_topology": args.readout_topology,
        "readout_fan_in": int(args.readout_fan_in),
        "readout_fan_out": int(args.readout_fan_out),
        "readout_topology_seed": int(args.readout_topology_seed),
        "activation_drive": args.activation_drive,
        "activation_sample_offsets_ns": [float(v) for v in activation_sample_offsets_ns],
        "variable_scope": args.variable_scope,
        "variable_order": args.variable_order,
        "variable_group_mode": args.variable_group_mode,
        "variables_available": candidate_variables,
        "variables_considered": variables,
        "variable_groups_considered": [list(group) for group in groups],
        "max_evals": int(args.max_evals),
        "target_accuracy": float(args.target_accuracy) if args.target_accuracy is not None else None,
        "evals": int(eval_count),
        "selection_metric": args.selection_metric,
        "cap_values": [float(v) for v in args.cap_values],
        "accuracy": current["accuracy"],
        "loss": current["loss"],
        "min_margin_v": current["min_margin_v"],
        "prediction_histogram": current["prediction_histogram"],
        "history": history,
        "wall_time_s": time.perf_counter() - t0,
    }
    summary.update(array_eval.readout_topology_summary(readout_fanins, args.hidden_cells))
    summary.update(array_eval.score_diagnostics(result_df, args.outputs))
    summary_path = SPICE_RESULTS / f"{safe}_summary.json"
    table_summary_path = RESULTS / f"{safe}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    table_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
