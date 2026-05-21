#!/usr/bin/env python3
"""Fit readout capacitor voltages against a measured MOS branch basis.

This is a diagnostic for the current multiclass blocker.  The old programmed
separator path maps ideal software weights linearly onto capacitor voltages.
This script instead asks a narrower question: if each readout branch contributes
according to the measured MOS kernel, can a direct cap program separate the
held hidden activations?
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_device_readout_branch_surface as branch_surface
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


def stable_ce(scores: np.ndarray, labels: np.ndarray) -> float:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    log_z = np.log(np.sum(np.exp(shifted), axis=1))
    return float(np.mean(log_z - shifted[np.arange(len(labels)), labels]))


def accuracy(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.argmax(scores, axis=1) == labels)) if len(labels) else 0.0


def select_candidate_index(losses: list[float], accuracies: list[float], selection_metric: str) -> int:
    if len(losses) != len(accuracies) or not losses:
        raise ValueError("losses and accuracies must be nonempty lists of the same length")
    if selection_metric == "accuracy":
        return max(range(len(losses)), key=lambda idx: (accuracies[idx], -losses[idx]))
    if selection_metric == "loss":
        return min(range(len(losses)), key=lambda idx: (losses[idx], -accuracies[idx]))
    raise ValueError(f"unknown selection metric: {selection_metric}")


def measure_branch_surface(
    *,
    tag: str,
    spice_bin: str,
    simulator_version: str,
    synapse_design: str,
    act_values: list[float],
    cap_values: list[float],
    width_scale: float,
    score_reset_v: float,
    score_cap_f: float,
    tran_step_ps: float,
    sample_ns: float,
    surface_mode: str,
    diode_width_u: float,
    mirror_cap_f: float,
    timeout: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    GENERATED.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, GENERATED)
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for branch in ("pos", "neg"):
        design = direct_flow.SYNAPSE_DESIGNS[synapse_design]
        for act_v in act_values:
            for cap_v in cap_values:
                netlist = branch_surface.readout_branch_netlist(
                    design_name=synapse_design,
                    branch=branch,
                    act_v=float(act_v),
                    weight_v=float(cap_v),
                    width_scale=width_scale,
                    score_reset_v=score_reset_v,
                    cap_f=score_cap_f,
                    tran_step_ps=tran_step_ps,
                    sample_ns=sample_ns,
                    stop_ns=max(3.2, sample_ns + 0.4),
                    surface_mode=surface_mode,
                    diode_width_u=diode_width_u,
                    mirror_cap_f=mirror_cap_f,
                )
                point_tag = f"{tag}_{synapse_design}_{branch}_a{act_v:.3f}_w{cap_v:.3f}"
                parsed = branch_surface.run_netlist(
                    spice_bin,
                    GENERATED / f"{branch_surface.safe_tag(point_tag)}.cir",
                    netlist,
                    timeout,
                )
                rows.append(
                    {
                        "design": synapse_design,
                        "style": design.output_forward_style,
                        "branch": branch,
                        "act_v": float(act_v),
                        "weight_v": float(cap_v),
                        "score_before_v": parsed["score_before"],
                        "score_after_v": parsed["score_after"],
                        "score_delta_v": parsed["score_delta"],
                    }
                )
    df = pd.DataFrame(rows)
    summary = {
        "tag": tag,
        "simulator": simulator_version,
        "elapsed_s": time.perf_counter() - t0,
        "designs": [synapse_design],
        "branches": ["pos", "neg"],
        "act_values": [float(v) for v in act_values],
        "weight_values": [float(v) for v in cap_values],
        "width_scale": float(width_scale),
        "score_reset_v": float(score_reset_v),
        "cap_f": float(score_cap_f),
        "tran_step_ps": float(tran_step_ps),
        "sample_ns": float(sample_ns),
        "surface_mode": surface_mode,
        "diode_width_u": float(diode_width_u) if surface_mode in branch_surface.DIODE_SURFACE_MODES else None,
        "mirror_cap_f": float(mirror_cap_f) if surface_mode == "diode_mirror_voltage" else None,
        **branch_surface.summarize(df, tolerance=1e-5),
    }
    return df, summary


def branch_candidate_tensor(
    surface: pd.DataFrame,
    *,
    branch: str,
    source_values: np.ndarray,
    cap_values: list[float],
) -> np.ndarray:
    """Return source x cap-candidate x sample score deltas."""
    branch_df = surface[surface["branch"] == branch].copy()
    if branch_df.empty:
        raise ValueError(f"surface is missing branch {branch!r}")
    act_grid = sorted(float(v) for v in branch_df["act_v"].unique())
    weight_grid = sorted(float(v) for v in branch_df["weight_v"].unique())
    grid = (
        branch_df.pivot_table(index="act_v", columns="weight_v", values="score_delta_v", aggfunc="mean")
        .reindex(index=act_grid, columns=weight_grid)
        .to_numpy(dtype=float)
    )
    if np.isnan(grid).any():
        raise ValueError(f"surface is missing one or more act/weight points for branch={branch!r}")
    cap_profiles = np.vstack(
        [
            np.interp(
                np.asarray(cap_values, dtype=float),
                np.asarray(weight_grid, dtype=float),
                grid[act_idx],
                left=grid[act_idx, 0],
                right=grid[act_idx, -1],
            )
            for act_idx in range(len(act_grid))
        ]
    )
    tensor = np.zeros((source_values.shape[1], len(cap_values), source_values.shape[0]), dtype=float)
    for cap_idx, cap_v in enumerate(cap_values):
        y = cap_profiles[:, cap_idx]
        for source_idx in range(source_values.shape[1]):
            tensor[source_idx, cap_idx, :] = np.interp(
                source_values[:, source_idx],
                act_grid,
                y,
                left=y[0],
                right=y[-1],
            )
    return tensor


def source_matrix_from_activations(activations: pd.DataFrame, hidden_cells: int, *, bias_source_v: float) -> np.ndarray:
    hidden_values = activations[[f"act{h}" for h in range(hidden_cells)]].to_numpy(dtype=float)
    bias = np.full(len(hidden_values), float(bias_source_v), dtype=float)
    return np.column_stack([hidden_values, bias])


def transfer_curve_from_surfaces(current_surface: pd.DataFrame, mirror_surface: pd.DataFrame) -> dict[str, list[float]]:
    keys = ["design", "style", "branch", "act_v", "weight_v"]
    merged = current_surface[keys + ["score_delta_v"]].merge(
        mirror_surface[keys + ["score_delta_v"]],
        on=keys,
        suffixes=("_current", "_mirror"),
    )
    if merged.empty:
        raise ValueError("current and mirror surfaces do not share any characterized points")
    points = (
        merged[["score_delta_v_current", "score_delta_v_mirror"]]
        .rename(columns={"score_delta_v_current": "current", "score_delta_v_mirror": "drop"})
        .copy()
    )
    points["current"] = points["current"].clip(lower=0.0)
    points["drop"] = points["drop"].clip(lower=0.0, upper=1.2)
    points = pd.concat(
        [
            pd.DataFrame([{"current": 0.0, "drop": 0.0}]),
            points,
        ],
        ignore_index=True,
    )
    grouped = points.groupby("current", as_index=False)["drop"].max().sort_values("current")
    current = grouped["current"].to_numpy(dtype=float)
    drop = np.maximum.accumulate(grouped["drop"].to_numpy(dtype=float))
    return {"current": [float(v) for v in current], "drop": [float(v) for v in drop]}


def apply_transfer(current: np.ndarray, curve: dict[str, list[float]]) -> np.ndarray:
    x = np.asarray(curve["current"], dtype=float)
    y = np.asarray(curve["drop"], dtype=float)
    if len(x) < 2:
        raise ValueError("transfer curve needs at least two points")
    return np.interp(np.maximum(current, 0.0), x, y, left=y[0], right=y[-1])


def greedy_cap_fit(
    *,
    pos_tensor: np.ndarray,
    neg_tensor: np.ndarray,
    labels: np.ndarray,
    outputs: int,
    cap_values: list[float],
    sweeps: int,
    seed: int,
    selection_metric: str = "loss",
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    sources = pos_tensor.shape[0]
    low_idx = int(np.argmin(cap_values))
    pos_idx = np.full((outputs, sources), low_idx, dtype=int)
    neg_idx = np.full((outputs, sources), low_idx, dtype=int)
    scores = np.zeros((len(labels), outputs), dtype=float)
    for out in range(outputs):
        for source in range(sources):
            scores[:, out] += pos_tensor[source, low_idx] - neg_tensor[source, low_idx]

    best_scores = scores.copy()
    best_pos_idx = pos_idx.copy()
    best_neg_idx = neg_idx.copy()
    best_loss = stable_ce(scores, labels)
    best_accuracy = accuracy(scores, labels)

    def maybe_store_best(current_loss: float, current_accuracy: float) -> None:
        nonlocal best_scores, best_pos_idx, best_neg_idx, best_loss, best_accuracy
        if selection_metric == "accuracy":
            better = current_accuracy > best_accuracy or (
                current_accuracy == best_accuracy and current_loss < best_loss
            )
        else:
            better = current_loss < best_loss or (current_loss == best_loss and current_accuracy > best_accuracy)
        if better:
            best_scores = scores.copy()
            best_pos_idx = pos_idx.copy()
            best_neg_idx = neg_idx.copy()
            best_loss = current_loss
            best_accuracy = current_accuracy

    variables = [(out, source, rail) for out in range(outputs) for source in range(sources) for rail in ("p", "n")]
    history: list[dict[str, float]] = []
    for sweep in range(sweeps):
        rng.shuffle(variables)
        changed = 0
        for out, source, rail in variables:
            tensor = pos_tensor if rail == "p" else neg_tensor
            sign = 1.0 if rail == "p" else -1.0
            current = pos_idx[out, source] if rail == "p" else neg_idx[out, source]
            base_col = scores[:, out] - sign * tensor[source, current]
            losses = []
            accuracies = []
            for candidate in range(len(cap_values)):
                trial_scores = scores.copy()
                trial_scores[:, out] = base_col + sign * tensor[source, candidate]
                losses.append(stable_ce(trial_scores, labels))
                accuracies.append(accuracy(trial_scores, labels))
            best = select_candidate_index(losses, accuracies, selection_metric)
            if best != current:
                changed += 1
                scores[:, out] = base_col + sign * tensor[source, best]
                if rail == "p":
                    pos_idx[out, source] = best
                else:
                    neg_idx[out, source] = best
        current_loss = stable_ce(scores, labels)
        current_accuracy = accuracy(scores, labels)
        maybe_store_best(current_loss, current_accuracy)
        history.append({"sweep": float(sweep), "loss": current_loss, "accuracy": current_accuracy})
        if changed == 0:
            break
    return {
        "scores": best_scores,
        "pos_idx": best_pos_idx,
        "neg_idx": best_neg_idx,
        "history": history,
        "loss": best_loss,
        "accuracy": best_accuracy,
        "selection_metric": selection_metric,
    }


def greedy_cap_fit_summed_transfer(
    *,
    pos_current_tensor: np.ndarray,
    neg_current_tensor: np.ndarray,
    labels: np.ndarray,
    outputs: int,
    cap_values: list[float],
    sweeps: int,
    seed: int,
    transfer_curve: dict[str, list[float]],
    selection_metric: str = "loss",
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    sources = pos_current_tensor.shape[0]
    samples = pos_current_tensor.shape[2]
    low_idx = int(np.argmin(cap_values))
    pos_idx = np.full((outputs, sources), low_idx, dtype=int)
    neg_idx = np.full((outputs, sources), low_idx, dtype=int)
    pos_current = np.zeros((outputs, samples), dtype=float)
    neg_current = np.zeros((outputs, samples), dtype=float)
    for out in range(outputs):
        for source in range(sources):
            pos_current[out] += pos_current_tensor[source, low_idx]
            neg_current[out] += neg_current_tensor[source, low_idx]

    def scores_from_currents(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
        return apply_transfer(pos, transfer_curve).T - apply_transfer(neg, transfer_curve).T

    scores = scores_from_currents(pos_current, neg_current)
    best_scores = scores.copy()
    best_pos_idx = pos_idx.copy()
    best_neg_idx = neg_idx.copy()
    best_loss = stable_ce(scores, labels)
    best_accuracy = accuracy(scores, labels)

    def maybe_store_best(current_loss: float, current_accuracy: float) -> None:
        nonlocal best_scores, best_pos_idx, best_neg_idx, best_loss, best_accuracy
        if selection_metric == "accuracy":
            better = current_accuracy > best_accuracy or (
                current_accuracy == best_accuracy and current_loss < best_loss
            )
        else:
            better = current_loss < best_loss or (current_loss == best_loss and current_accuracy > best_accuracy)
        if better:
            best_scores = scores.copy()
            best_pos_idx = pos_idx.copy()
            best_neg_idx = neg_idx.copy()
            best_loss = current_loss
            best_accuracy = current_accuracy

    variables = [(out, source, rail) for out in range(outputs) for source in range(sources) for rail in ("p", "n")]
    history: list[dict[str, float]] = []
    for sweep in range(sweeps):
        rng.shuffle(variables)
        changed = 0
        for out, source, rail in variables:
            if rail == "p":
                tensor = pos_current_tensor
                current_idx = pos_idx[out, source]
                base_col = pos_current[out] - tensor[source, current_idx]
                fixed_other = apply_transfer(neg_current[out], transfer_curve)
            else:
                tensor = neg_current_tensor
                current_idx = neg_idx[out, source]
                base_col = neg_current[out] - tensor[source, current_idx]
                fixed_other = apply_transfer(pos_current[out], transfer_curve)
            losses = []
            accuracies = []
            trial_cols = []
            for candidate in range(len(cap_values)):
                transferred = apply_transfer(base_col + tensor[source, candidate], transfer_curve)
                trial_col = transferred - fixed_other if rail == "p" else fixed_other - transferred
                trial_scores = scores.copy()
                trial_scores[:, out] = trial_col
                losses.append(stable_ce(trial_scores, labels))
                accuracies.append(accuracy(trial_scores, labels))
                trial_cols.append(trial_col)
            best = select_candidate_index(losses, accuracies, selection_metric)
            if best != current_idx:
                changed += 1
                if rail == "p":
                    pos_current[out] = base_col + tensor[source, best]
                    pos_idx[out, source] = best
                else:
                    neg_current[out] = base_col + tensor[source, best]
                    neg_idx[out, source] = best
                scores[:, out] = trial_cols[best]
        current_loss = stable_ce(scores, labels)
        current_accuracy = accuracy(scores, labels)
        maybe_store_best(current_loss, current_accuracy)
        history.append({"sweep": float(sweep), "loss": current_loss, "accuracy": current_accuracy})
        if changed == 0:
            break
    return {
        "scores": best_scores,
        "pos_idx": best_pos_idx,
        "neg_idx": best_neg_idx,
        "history": history,
        "loss": best_loss,
        "accuracy": best_accuracy,
        "selection_metric": selection_metric,
    }


def cap_init_from_indices(pos_idx: np.ndarray, neg_idx: np.ndarray, cap_values: list[float], *, hidden: int) -> dict[str, float]:
    init: dict[str, float] = {}
    outputs, sources = pos_idx.shape
    if sources != hidden + 1:
        raise ValueError("cap index matrix must include hidden sources plus one bias source")
    for out in range(outputs):
        for h in range(hidden):
            init[f"vw{out}{h}p"] = float(cap_values[int(pos_idx[out, h])])
            init[f"vw{out}{h}n"] = float(cap_values[int(neg_idx[out, h])])
        bias_source = hidden
        init[f"vbo{out}p"] = float(cap_values[int(pos_idx[out, bias_source])])
        init[f"vbo{out}n"] = float(cap_values[int(neg_idx[out, bias_source])])
    return init


def write_cap_csv(init: dict[str, float], path: Path) -> None:
    rows = [{"cap": cap, "value": value} for cap, value in sorted(init.items())]
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="device_readout_capfit")
    ap.add_argument("--simulator")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--activation-csv", type=Path, required=True)
    ap.add_argument("--activation-phase", default="initial_eval")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--hidden-cells", type=int, default=8)
    ap.add_argument("--outputs", type=int, default=3)
    ap.add_argument("--synapse-design", choices=sorted(direct_flow.SYNAPSE_DESIGNS), default="split_signed_v1")
    ap.add_argument("--surface-csv", type=Path)
    ap.add_argument("--transfer-surface-csv", type=Path)
    ap.add_argument(
        "--fit-model",
        choices=["additive_branch", "summed_branch_transfer"],
        default="additive_branch",
        help=(
            "additive_branch sums the measured branch observable directly. "
            "summed_branch_transfer sums clamped branch currents, then applies a measured diode/mirror transfer curve."
        ),
    )
    ap.add_argument("--act-values", type=parse_float_list, default=parse_float_list("0,0.3,0.6,0.9,1.2"))
    ap.add_argument("--cap-values", type=parse_float_list, default=parse_float_list("0.16,0.30,0.44,0.58,0.72,0.86,1.00"))
    ap.add_argument("--width-scale", type=float, default=1.0)
    ap.add_argument("--score-reset-v", type=float, default=0.0)
    ap.add_argument("--score-cap-f", type=float, default=100.0)
    ap.add_argument(
        "--bias-source-v",
        type=float,
        default=1.2,
        help="Voltage of the readout bias source in the full deck. The production circuit drives bias at VDD.",
    )
    ap.add_argument("--tran-step-ps", type=float, default=10.0)
    ap.add_argument("--sample-ns", type=float, default=2.95)
    ap.add_argument(
        "--surface-mode",
        choices=sorted(branch_surface.SURFACE_MODES),
        default="floating_delta",
    )
    ap.add_argument("--diode-width-u", type=float, default=256.0)
    ap.add_argument("--mirror-cap-f", type=float, default=20.0)
    ap.add_argument("--sweeps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--selection-metric",
        choices=["loss", "accuracy"],
        default="loss",
        help="Choose whether the written cap CSV is selected by CE loss or by training-set accuracy.",
    )
    args = ap.parse_args()

    if args.hidden_cells <= 0 or args.outputs <= 1:
        raise SystemExit("--hidden-cells must be positive and --outputs must be at least 2.")
    if args.sweeps <= 0:
        raise SystemExit("--sweeps must be positive.")
    if any(v < 0.0 or v > 1.2 for v in args.cap_values):
        raise SystemExit("--cap-values must stay inside 0..1.2 V.")
    if args.diode_width_u <= 0:
        raise SystemExit("--diode-width-u must be positive.")
    if args.mirror_cap_f <= 0:
        raise SystemExit("--mirror-cap-f must be positive.")
    if args.bias_source_v < 0.0 or args.bias_source_v > 1.2:
        raise SystemExit("--bias-source-v must stay inside 0..1.2 V.")

    direct_flow.set_hidden_cells(args.hidden_cells)
    direct_flow.set_output_count(args.outputs)
    activations = readout_transfer.load_activations(args.activation_csv, args.activation_phase, args.hidden_cells, args.limit)
    labels = activations["label"].astype(int).to_numpy()
    source_values = source_matrix_from_activations(
        activations,
        args.hidden_cells,
        bias_source_v=args.bias_source_v,
    )

    spice_bin, simulator_version = detect_spice(args.simulator)
    safe = safe_tag(args.tag)
    transfer_surface: pd.DataFrame | None = None
    transfer_summary: dict[str, Any] | None = None
    transfer_curve: dict[str, list[float]] | None = None
    if args.fit_model == "summed_branch_transfer" and args.surface_mode != "diode_mirror_voltage":
        raise SystemExit("--fit-model summed_branch_transfer currently requires --surface-mode diode_mirror_voltage.")

    if args.surface_csv is None:
        measured_surface_mode = "clamped_current" if args.fit_model == "summed_branch_transfer" else args.surface_mode
        surface, surface_summary = measure_branch_surface(
            tag=f"{safe}_surface",
            spice_bin=spice_bin,
            simulator_version=simulator_version,
            synapse_design=args.synapse_design,
            act_values=args.act_values,
            cap_values=args.cap_values,
            width_scale=args.width_scale,
            score_reset_v=args.score_reset_v,
            score_cap_f=args.score_cap_f,
            tran_step_ps=args.tran_step_ps,
            sample_ns=args.sample_ns,
            surface_mode=measured_surface_mode,
            diode_width_u=args.diode_width_u,
            mirror_cap_f=args.mirror_cap_f,
            timeout=args.timeout,
        )
    else:
        surface = pd.read_csv(args.surface_csv)
        surface_summary = {"source": str(args.surface_csv)}

    if args.fit_model == "summed_branch_transfer":
        if args.transfer_surface_csv is None:
            transfer_surface, transfer_summary = measure_branch_surface(
                tag=f"{safe}_transfer_surface",
                spice_bin=spice_bin,
                simulator_version=simulator_version,
                synapse_design=args.synapse_design,
                act_values=args.act_values,
                cap_values=args.cap_values,
                width_scale=args.width_scale,
                score_reset_v=args.score_reset_v,
                score_cap_f=args.score_cap_f,
                tran_step_ps=args.tran_step_ps,
                sample_ns=args.sample_ns,
                surface_mode="diode_mirror_voltage",
                diode_width_u=args.diode_width_u,
                mirror_cap_f=args.mirror_cap_f,
                timeout=args.timeout,
            )
        else:
            transfer_surface = pd.read_csv(args.transfer_surface_csv)
            transfer_summary = {"source": str(args.transfer_surface_csv)}
        transfer_curve = transfer_curve_from_surfaces(surface, transfer_surface)

    pos_tensor = branch_candidate_tensor(surface, branch="pos", source_values=source_values, cap_values=args.cap_values)
    neg_tensor = branch_candidate_tensor(surface, branch="neg", source_values=source_values, cap_values=args.cap_values)
    if args.fit_model == "summed_branch_transfer":
        if transfer_curve is None:
            raise RuntimeError("summed transfer fit did not produce a transfer curve")
        fit = greedy_cap_fit_summed_transfer(
            pos_current_tensor=pos_tensor,
            neg_current_tensor=neg_tensor,
            labels=labels,
            outputs=args.outputs,
            cap_values=args.cap_values,
            sweeps=args.sweeps,
            seed=args.seed,
            transfer_curve=transfer_curve,
            selection_metric=args.selection_metric,
        )
    else:
        fit = greedy_cap_fit(
            pos_tensor=pos_tensor,
            neg_tensor=neg_tensor,
            labels=labels,
            outputs=args.outputs,
            cap_values=args.cap_values,
            sweeps=args.sweeps,
            seed=args.seed,
            selection_metric=args.selection_metric,
        )
    init = cap_init_from_indices(fit["pos_idx"], fit["neg_idx"], args.cap_values, hidden=args.hidden_cells)

    RESULTS.mkdir(parents=True, exist_ok=True)
    SPICE_RESULTS.mkdir(parents=True, exist_ok=True)
    cap_csv = SPICE_RESULTS / f"{safe}_caps.csv"
    write_cap_csv(init, cap_csv)
    surface_csv = SPICE_RESULTS / f"{safe}_surface.csv"
    surface.to_csv(surface_csv, index=False)
    transfer_csv = None
    if transfer_surface is not None:
        transfer_csv = SPICE_RESULTS / f"{safe}_transfer_surface.csv"
        transfer_surface.to_csv(transfer_csv, index=False)

    score_df = activations[["label", *[f"act{h}" for h in range(args.hidden_cells)]]].copy()
    scores = fit["scores"]
    for out in range(args.outputs):
        score_df[f"surrogate_score{out}"] = scores[:, out]
    score_df["surrogate_predicted_label"] = np.argmax(scores, axis=1)
    score_df["surrogate_correct"] = score_df["surrogate_predicted_label"].to_numpy() == labels
    score_csv = RESULTS / f"{safe}_surrogate_scores.csv"
    score_df.to_csv(score_csv, index=False)

    summary = {
        "tag": safe,
        "simulator": simulator_version,
        "activation_csv": str(args.activation_csv),
        "activation_phase": args.activation_phase,
        "samples": int(len(activations)),
        "hidden_cells": int(args.hidden_cells),
        "outputs": int(args.outputs),
        "synapse_design": args.synapse_design,
        "cap_values": [float(v) for v in args.cap_values],
        "width_scale": float(args.width_scale),
        "score_reset_v": float(args.score_reset_v),
        "score_cap_f": float(args.score_cap_f),
        "bias_source_v": float(args.bias_source_v),
        "sample_ns": float(args.sample_ns),
        "surface_mode": args.surface_mode,
        "fit_model": args.fit_model,
        "diode_width_u": float(args.diode_width_u) if args.surface_mode in branch_surface.DIODE_SURFACE_MODES else None,
        "mirror_cap_f": float(args.mirror_cap_f) if args.surface_mode == "diode_mirror_voltage" else None,
        "sweeps": int(args.sweeps),
        "seed": int(args.seed),
        "selection_metric": args.selection_metric,
        "surrogate_accuracy": float(fit["accuracy"]),
        "surrogate_loss": float(fit["loss"]),
        "history": fit["history"],
        "cap_csv": str(cap_csv),
        "surface_csv": str(surface_csv),
        "transfer_surface_csv": str(transfer_csv) if transfer_csv is not None else str(args.transfer_surface_csv) if args.transfer_surface_csv else None,
        "score_csv": str(score_csv),
        "surface_summary": surface_summary,
        "transfer_summary": transfer_summary,
        "transfer_curve": transfer_curve,
    }
    summary_path = SPICE_RESULTS / f"{safe}_summary.json"
    table_summary_path = RESULTS / f"{safe}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    table_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
