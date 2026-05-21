#!/usr/bin/env python3
"""Evaluate the production readout synapse array with measured activations.

The branch-surface cap fitter is intentionally cheap, but it treats each
readout branch as independent.  This harness keeps the hidden layer out of the
way while preserving the simultaneous score-cap dynamics of the full readout
array: activation voltages are externally driven, stored readout capacitor
states are loaded, and the production MOS readout fragment integrates all
branches onto the shared class score rails.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_device_readout_transfer as readout_transfer
import run_device_xor2_random_hidden as direct_flow
from run_device_multicell_classifier import mos_models, pulse_wave, pwl
from run_spice_sweep import ROOT, detect_spice, run_text_netlist, run_tiny_test
from _util import MEAS_RE, parse_measures


RESULTS = ROOT / "results/tables"
SPICE_RESULTS = ROOT / "spice/results"
GENERATED = ROOT / "spice/generated"
SCORE_SENSE_MODES = [
    "score_caps",
    "clamped_current",
    "diode_voltage",
    "diode_current",
]


def safe_tag(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in text)


def activation_wave(
    values: np.ndarray,
    stop_ns: float,
    cycle_ns: float,
    *,
    mode: str = "held",
    fwd_start_ns: float = 0.75,
    settle_ns: float = 2.95,
) -> str:
    if mode not in {"held", "ramp"}:
        raise ValueError(f"unknown activation drive mode: {mode}")
    points: list[tuple[float, float]] = []
    for idx, value in enumerate(values):
        start = idx * cycle_ns
        end = start + cycle_ns
        v = float(value)
        if mode == "held":
            if idx == 0:
                points.append((0.0, v))
            else:
                points.append((start - 0.05, float(values[idx - 1])))
                points.append((start, v))
            points.append((min(stop_ns, end - 0.05), v))
        else:
            if idx == 0:
                points.append((0.0, 0.0))
            else:
                points.append((start - 0.05, float(values[idx - 1])))
                points.append((start, 0.0))
            points.append((start + fwd_start_ns, 0.0))
            points.append((start + settle_ns, v))
            points.append((min(stop_ns, end - 0.05), v))
    points.append((stop_ns, float(values[-1]) if mode == "held" else 0.0))
    return pwl(points)


def measured_activation_wave(
    activations: pd.DataFrame,
    hidden: int,
    stop_ns: float,
    cycle_ns: float,
    offsets_ns: list[float],
) -> str:
    if not offsets_ns:
        raise ValueError("measured activation drive requires at least one activation sample offset.")
    sorted_offsets = sorted(offsets_ns)
    points: list[tuple[float, float]] = []
    last_value = 0.0
    for idx in range(len(activations)):
        start = idx * cycle_ns
        points.extend([(start, 0.0), (start + 0.50, 0.0)])
        for offset in sorted_offsets:
            key = direct_flow.offset_key(offset)
            col = f"act{hidden}_{key}"
            if col in activations:
                value = float(activations.iloc[idx][col])
            elif abs(offset - sorted_offsets[-1]) < 1e-12 and f"act{hidden}" in activations:
                value = float(activations.iloc[idx][f"act{hidden}"])
            else:
                raise ValueError(f"activation table is missing measured waveform column {col!r}")
            points.append((start + offset, value))
            last_value = value
        points.append((min(stop_ns, start + cycle_ns - 0.05), last_value))
    points.append((stop_ns, last_value))
    return pwl(points)


def activation_source_wave(
    activations: pd.DataFrame,
    hidden: int,
    stop_ns: float,
    cycle_ns: float,
    *,
    mode: str,
    settle_ns: float,
    sample_offsets_ns: list[float],
) -> str:
    if mode == "measured":
        return measured_activation_wave(activations, hidden, stop_ns, cycle_ns, sample_offsets_ns)
    return activation_wave(
        activations[f"act{hidden}"].to_numpy(dtype=float),
        stop_ns,
        cycle_ns,
        mode=mode,
        settle_ns=settle_ns,
    )


def eval_samples(activations: pd.DataFrame, hidden_cells: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for idx, row in enumerate(activations.itertuples(index=False)):
        samples.append(
            {
                "phase": "eval",
                "epoch": 0,
                "pattern": idx,
                "label": int(getattr(row, "label")),
                "apply_update": False,
                "reset_gradient": False,
            }
        )
    if not samples:
        raise ValueError("activation table is empty")
    for h in range(hidden_cells):
        col = f"act{h}"
        if col not in activations:
            raise ValueError(f"activation table is missing {col!r}")
    return samples


def build_readout_fanins(
    mode: str,
    *,
    hidden_cells: int,
    outputs: int,
    fan_in: int,
    fan_out: int,
    seed: int,
) -> direct_flow.ReadoutFanins:
    if mode == "dense":
        return direct_flow.dense_readout_fanins(hidden_cells, outputs)
    if mode == "random_fanin":
        return direct_flow.random_readout_fanins(hidden_cells, outputs, seed=seed, fan_in=fan_in)
    if mode == "random_fanout":
        return direct_flow.random_readout_fanins(hidden_cells, outputs, seed=seed, fan_out=fan_out)
    raise ValueError(f"unknown readout topology: {mode}")


def readout_topology_summary(fanins: direct_flow.ReadoutFanins, hidden_cells: int) -> dict[str, Any]:
    fanouts = direct_flow.readout_fanouts_from_fanins(fanins, hidden_cells)
    return {
        "readout_edge_count": int(sum(len(srcs) for srcs in fanins.values())),
        "readout_fanins": {str(out): list(srcs) for out, srcs in fanins.items()},
        "readout_fanouts": {str(hidden): list(outs) for hidden, outs in fanouts.items()},
        "readout_min_fanin": int(min((len(srcs) for srcs in fanins.values()), default=0)),
        "readout_max_fanin": int(max((len(srcs) for srcs in fanins.values()), default=0)),
        "readout_min_fanout": int(min((len(outs) for outs in fanouts.values()), default=0)),
        "readout_max_fanout": int(max((len(outs) for outs in fanouts.values()), default=0)),
    }


def readout_state_sources(
    readout_state: dict[str, float],
    cap_f: float,
    readout_fanins: direct_flow.ReadoutFanins | None = None,
) -> str:
    """Use the same capacitor-held state nodes as the full deck, without writes."""
    lines: list[str] = []
    fanins = direct_flow.dense_readout_fanins() if readout_fanins is None else readout_fanins
    for out in range(direct_flow.OUTPUTS):
        lines += [
            f"Cvbo{out}p vbo{out}p 0 {cap_f:.12g}f IC={readout_state[f'vbo{out}p']:.12g}",
            f"Cvbo{out}n vbo{out}n 0 {cap_f:.12g}f IC={readout_state[f'vbo{out}n']:.12g}",
            f"Rvbo{out}p vbo{out}p 0 1e15",
            f"Rvbo{out}n vbo{out}n 0 1e15",
        ]
        for h in fanins[out]:
            lines += [
                f"Cvw{out}{h}p vw{out}{h}p 0 {cap_f:.12g}f IC={readout_state[f'vw{out}{h}p']:.12g}",
                f"Cvw{out}{h}n vw{out}{h}n 0 {cap_f:.12g}f IC={readout_state[f'vw{out}{h}n']:.12g}",
                f"Rvw{out}{h}p vw{out}{h}p 0 1e15",
                f"Rvw{out}{h}n vw{out}{h}n 0 1e15",
            ]
    return "\n".join(lines)


def score_caps_and_resets(score_reset_v: float, score_cap_f: float, output_cap_f: float, output_head: str) -> str:
    if output_head in direct_flow.COMMON_MODE_OUT_RESET_HEADS:
        out_ic = score_reset_v
        out_leak_node = "scorecm"
    elif output_head in direct_flow.LOW_TRUE_OUTPUT_HEADS:
        out_ic = 1.2
        out_leak_node = "vdd"
    else:
        out_ic = 0.0
        out_leak_node = "0"
    lines: list[str] = []
    for out in range(direct_flow.OUTPUTS):
        lines += [
            f"Cscore{out} score{out} 0 {score_cap_f:.12g}f IC={score_reset_v:.12g}",
            f"Cscorep{out} scorep{out} 0 {score_cap_f:.12g}f IC={score_reset_v:.12g}",
            f"Cscoren{out} scoren{out} 0 {score_cap_f:.12g}f IC={score_reset_v:.12g}",
            f"Cout{out} out{out} 0 {output_cap_f:.12g}f IC={out_ic:.12g}",
            f"Rscore{out} score{out} 0 1G",
            f"Rscorep{out} scorep{out} 0 1G",
            f"Rscoren{out} scoren{out} 0 1G",
            f"Rout{out} out{out} {out_leak_node} 1G",
            f"Mreset_score{out} score{out} rstf scorecm 0 NMOS W=4u L=180n",
            f"Mreset_scorep{out} scorep{out} rstf scorecm 0 NMOS W=4u L=180n",
            f"Mreset_scoren{out} scoren{out} rstf scorecm 0 NMOS W=4u L=180n",
            f"Mreset_out{out} out{out} rstf {out_leak_node} 0 NMOS W=4u L=180n",
        ]
    return "\n".join(lines)


def score_sense_cells(
    *,
    score_sense_mode: str,
    score_reset_v: float,
    score_cap_f: float,
    output_cap_f: float,
    output_head: str,
    score_diode_width_u: float,
) -> str:
    if score_sense_mode == "score_caps":
        return score_caps_and_resets(score_reset_v, score_cap_f, output_cap_f, output_head)
    if output_head != "split_score_none":
        raise ValueError(f"{score_sense_mode} score sensing requires --output-head split_score_none.")
    if score_sense_mode == "clamped_current":
        return readout_transfer.clamped_score_sources(score_reset_v)
    if score_sense_mode == "diode_voltage":
        return readout_transfer.diode_score_nodes(score_cap_f, score_diode_width_u)
    if score_sense_mode == "diode_current":
        return readout_transfer.diode_score_nodes(score_cap_f, score_diode_width_u, current_probe=True)
    raise ValueError(f"unknown score sense mode: {score_sense_mode}")


def inactive_error_sources(outputs: int) -> str:
    lines: list[str] = []
    for out in range(outputs):
        lines += [
            f"Vdp{out} dp{out} 0 0",
            f"Vdn{out} dn{out} 0 0",
        ]
    return "\n".join(lines)


def flow_offstate_loads() -> str:
    """Attach the full direct-flow readout write/pretrace load with writes off."""
    return "\n".join(
        [
            "* Off-state direct-flow loads: same pretrace/write stacks as the full deck, with inactive error rails.",
            inactive_error_sources(direct_flow.OUTPUTS),
            direct_flow.flow_pre_activation_stores("synapse_spike", 2.0, 0.05, 4.0, "spikeref"),
            direct_flow.readout_flow_updates(
                readout_update_width_u=120.0,
                output_bias_update_width_u=120.0,
                flow_pre_store="synapse_spike",
                readout_flow_polarity="normal",
                readout_flow_write_mode="bounded_pmos_charge_only",
                readout_center_pull_gate="bwd",
                readout_center_pull_mode="always",
                readout_write_state_gate_mode="none",
                readout_write_gate_device="NSENSE",
                output_bias_write_pre_gate="none",
                output_bias_flow_polarity="follow_readout",
                readout_pos_write_high_node="wphigh",
                readout_pos_write_low_node="wplow",
                readout_neg_write_high_node="wnhigh",
                readout_neg_write_low_node="wnlow",
                readout_pos_center_pull_node="wcenterp",
                readout_neg_center_pull_node="wcentern",
                output_bias_pos_center_pull_node="wbocenterp",
                output_bias_neg_center_pull_node="wbocentern",
                write_error_exclusion="diffpair_bleed",
                write_error_exclusion_width_u=8.0,
            ),
        ]
    )


def readout_array_netlist(
    *,
    activations: pd.DataFrame,
    readout_state: dict[str, float],
    design_name: str,
    output_head: str,
    hidden_cells: int,
    outputs: int,
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
    tran_step_ps: float,
    spice_accuracy_preset: str,
    score_sense_mode: str = "score_caps",
    output_forward_width_scale: float = 1.0,
    output_forward_pos_width_scale: float = 1.0,
    output_forward_neg_width_scale: float = 1.0,
    output_bias_forward_width_scale: float = 1.0,
    output_relu_width_scale: float = 1.0,
    score_diode_width_u: float = 1024.0,
    score_mirror_cap_f: float = 20.0,
    readout_fanins: direct_flow.ReadoutFanins | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    if readout_load_mode not in {"forward_only", "flow_offstate"}:
        raise ValueError(f"unknown readout load mode: {readout_load_mode}")
    if score_sense_mode not in SCORE_SENSE_MODES:
        raise ValueError(f"unknown score sense mode: {score_sense_mode}")
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    original_cycle = direct_flow.CYCLE_NS
    try:
        direct_flow.set_hidden_cells(hidden_cells)
        direct_flow.set_output_count(outputs)
        direct_flow.CYCLE_NS = cycle_ns
        samples = eval_samples(activations, hidden_cells)
        active_fanins = (
            direct_flow.dense_readout_fanins(hidden_cells, outputs)
            if readout_fanins is None
            else readout_fanins
        )
        dense_fanins = direct_flow.dense_readout_fanins(hidden_cells, outputs)
        if readout_load_mode == "flow_offstate" and active_fanins != dense_fanins:
            raise ValueError("sparse readout topology is not yet wired into flow_offstate load stacks.")
        stop_ns = len(samples) * cycle_ns
        design = direct_flow.scaled_synapse_design(
            design_name,
            hidden_delta_width_scale=1.0,
            hidden_gradient_width_scale=1.0,
            readout_gradient_width_scale=1.0,
            output_forward_width_scale=output_forward_width_scale,
            output_forward_pos_width_scale=output_forward_pos_width_scale,
            output_forward_neg_width_scale=output_forward_neg_width_scale,
            output_bias_forward_width_scale=output_bias_forward_width_scale,
            output_relu_width_scale=output_relu_width_scale,
        )
        act_sources = "\n".join(
            f"Vact{h} act{h} 0 "
            f"{activation_source_wave(activations, h, stop_ns, cycle_ns, mode=activation_drive, settle_ns=activation_settle_ns, sample_offsets_ns=activation_sample_offsets_ns)}"
            for h in range(hidden_cells)
        )
        phase_sources = direct_flow.phases(
            samples,
            bwd_start_ns=6.75,
            apply_start_ns=9.25,
            apply_end_ns=11.2,
            cmp_start_ns=3.25,
            cmp_end_ns=4.1,
            learning_mode="flow",
            backward_gate_mode="scheduled",
            train_refire=False,
        )
        load_block = flow_offstate_loads() if readout_load_mode == "flow_offstate" else ""
        measures: list[str] = []
        prints: list[str] = []
        for idx in range(len(samples)):
            at = idx * cycle_ns + sample_ns
            for out in range(outputs):
                if score_sense_mode == "clamped_current":
                    measures += [
                        f".meas tran scorep{out}_{idx} FIND I(Vscorep{out}_clamp) AT={at:.12g}n",
                        f".meas tran scoren{out}_{idx} FIND I(Vscoren{out}_clamp) AT={at:.12g}n",
                        f".meas tran score{out}_{idx} PARAM='scorep{out}_{idx}-scoren{out}_{idx}'",
                    ]
                    prints += [f"scorep{out}_{idx}", f"scoren{out}_{idx}", f"score{out}_{idx}"]
                elif score_sense_mode == "diode_current":
                    measures += [
                        f".meas tran scorep{out}_{idx} FIND I(Vscorep{out}_sense) AT={at:.12g}n",
                        f".meas tran scoren{out}_{idx} FIND I(Vscoren{out}_sense) AT={at:.12g}n",
                        f".meas tran score{out}_{idx} PARAM='scorep{out}_{idx}-scoren{out}_{idx}'",
                    ]
                    prints += [f"scorep{out}_{idx}", f"scoren{out}_{idx}", f"score{out}_{idx}"]
                elif score_sense_mode == "diode_voltage":
                    measures += [
                        f".meas tran scorep{out}_{idx} FIND V(scorep{out}) AT={at:.12g}n",
                        f".meas tran scoren{out}_{idx} FIND V(scoren{out}) AT={at:.12g}n",
                        f".meas tran score{out}_{idx} PARAM='scorep{out}_{idx}-scoren{out}_{idx}'",
                    ]
                    prints += [f"scorep{out}_{idx}", f"scoren{out}_{idx}", f"score{out}_{idx}"]
                elif output_head in direct_flow.DIODE_MIRROR_OUTPUT_HEADS:
                    measures += [
                        f".meas tran scorep{out}_{idx} FIND V(scorep{out}) AT={at:.12g}n",
                        f".meas tran scoren{out}_{idx} FIND V(scoren{out}) AT={at:.12g}n",
                        f".meas tran scorepm{out}_{idx} FIND V(scorepm{out}) AT={at:.12g}n",
                        f".meas tran scorenm{out}_{idx} FIND V(scorenm{out}) AT={at:.12g}n",
                        f".meas tran score{out}_{idx} PARAM='scorenm{out}_{idx}-scorepm{out}_{idx}'",
                    ]
                    prints += [
                        f"scorep{out}_{idx}",
                        f"scoren{out}_{idx}",
                        f"scorepm{out}_{idx}",
                        f"scorenm{out}_{idx}",
                        f"score{out}_{idx}",
                    ]
                elif output_head in direct_flow.SPLIT_SCORE_OUTPUT_HEADS:
                    measures += [
                        f".meas tran scorep{out}_{idx} FIND V(scorep{out}) AT={at:.12g}n",
                        f".meas tran scoren{out}_{idx} FIND V(scoren{out}) AT={at:.12g}n",
                        f".meas tran score{out}_{idx} PARAM='scorep{out}_{idx}-scoren{out}_{idx}'",
                    ]
                    prints += [f"scorep{out}_{idx}", f"scoren{out}_{idx}", f"score{out}_{idx}"]
                else:
                    measures.append(f".meas tran score{out}_{idx} FIND V(score{out}) AT={at:.12g}n")
                    prints.append(f"score{out}_{idx}")
                if score_sense_mode == "score_caps":
                    measures.append(f".meas tran out{out}_{idx} FIND V(out{out}) AT={at:.12g}n")
                    prints.append(f"out{out}_{idx}")
        deck = f"""
* Production readout-array evaluation with externally driven hidden activations.
.param VDD=1.2
{spice_options(spice_accuracy_preset)}
{mos_models()}
Vdd vdd 0 {{VDD}}
Vbias bias 0 {{VDD}}
Vspikeref spikeref 0 0.05
Vscorecm scorecm 0 {score_reset_v:.12g}
Vwcenter wcenter 0 0.64
Vwcenterp wcenterp 0 0.64
Vwcentern wcentern 0 0.64
Vwbocenterp wbocenterp 0 0.64
Vwbocentern wbocentern 0 0.64
Vwhigh whigh 0 1
Vwlow wlow 0 0.16
Vwphigh wphigh 0 1
Vwplow wplow 0 0.16
Vwnhigh wnhigh 0 1
Vwnlow wnlow 0 0.16
{act_sources}
{phase_sources}

{readout_state_sources(readout_state, hidden_cap_f, active_fanins)}
{score_sense_cells(score_sense_mode=score_sense_mode, score_reset_v=score_reset_v, score_cap_f=score_cap_f, output_cap_f=output_cap_f, output_head=output_head, score_diode_width_u=score_diode_width_u)}
{load_block}
{direct_flow.output_forward(design, output_head, score_diode_width_u, score_mirror_cap_f, readout_fanins=active_fanins)}

.tran {tran_step_ps:.12g}p {stop_ns:.12g}n uic
{chr(10).join(measures)}
.control
run
print {' '.join(prints)}
.endc
.end
""".lstrip()
        return deck, samples
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)
        direct_flow.CYCLE_NS = original_cycle


def spice_options(preset: str) -> str:
    options = direct_flow.spice_options_for_preset(preset)
    return f".option {options}" if options else ""


def run_netlist(spice_bin: str, path: Path, text: str, timeout: float) -> dict[str, float]:
    proc = run_text_netlist(spice_bin, path, text, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-3000:])
    parsed = parse_measures(proc.stdout + "\n" + proc.stderr)
    if not parsed:
        raise RuntimeError("SPICE produced no parseable measurements:\n" + (proc.stdout + proc.stderr)[-3000:])
    return parsed


def rows_from_measures(parsed: dict[str, float], activations: pd.DataFrame, outputs: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = activations["label"].astype(int).to_numpy()
    for idx, label in enumerate(labels):
        scores = [parsed[f"score{out}_{idx}"] for out in range(outputs)]
        outs = [parsed.get(f"out{out}_{idx}", scores[out]) for out in range(outputs)]
        pred = int(np.argmax(scores))
        row: dict[str, Any] = {
            "sample": idx,
            "label": int(label),
            "predicted_label": pred,
            "correct": pred == int(label),
            "target_score": scores[int(label)],
            "other_score": max(score for out, score in enumerate(scores) if out != int(label)),
        }
        row["score_margin"] = row["target_score"] - row["other_score"]
        for out, value in enumerate(scores):
            row[f"score{out}"] = value
            row[f"out{out}"] = outs[out]
            for prefix in ("scorep", "scoren", "scorepm", "scorenm"):
                key = f"{prefix}{out}_{idx}"
                if key in parsed:
                    row[f"{prefix}{out}"] = parsed[key]
        rows.append(row)
    return pd.DataFrame(rows)


def _margin_values(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return np.asarray([], dtype=float)
    other = np.where(np.eye(scores.shape[1], dtype=bool)[labels], -np.inf, scores)
    return scores[np.arange(len(labels)), labels] - np.max(other, axis=1)


def _accuracy(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.argmax(scores, axis=1) == labels)) if len(labels) else 0.0


def score_diagnostics(df: pd.DataFrame, outputs: int) -> dict[str, Any]:
    """Report class-order diagnostics for measured score rails.

    Column centering is a hardware-relevant diagnostic: per-class output-bias
    caps can shift score rails, so this estimates how much of the error is
    static score offset rather than loss of separability.
    """
    labels = df["label"].astype(int).to_numpy()
    scores = df[[f"score{out}" for out in range(outputs)]].to_numpy(dtype=float)
    centered = scores - scores.mean(axis=0, keepdims=True)
    inverted = -scores
    inverted_centered = inverted - inverted.mean(axis=0, keepdims=True)
    margins = _margin_values(scores, labels)
    centered_margins = _margin_values(centered, labels)
    diagnostics: dict[str, Any] = {
        "column_centered_accuracy": _accuracy(centered, labels),
        "column_centered_min_margin_v": float(np.min(centered_margins)) if len(centered_margins) else 0.0,
        "inverted_accuracy": _accuracy(inverted, labels),
        "inverted_column_centered_accuracy": _accuracy(inverted_centered, labels),
        "score_mean_by_output_v": {
            str(out): float(scores[:, out].mean()) if len(scores) else 0.0 for out in range(outputs)
        },
        "score_span_by_output_v": {
            str(out): float(scores[:, out].max() - scores[:, out].min()) if len(scores) else 0.0 for out in range(outputs)
        },
        "score_margin_mean_v": float(np.mean(margins)) if len(margins) else 0.0,
    }
    confusion = np.zeros((outputs, outputs), dtype=int)
    predicted = np.argmax(scores, axis=1) if len(scores) else np.asarray([], dtype=int)
    for label, pred in zip(labels, predicted):
        if 0 <= label < outputs and 0 <= pred < outputs:
            confusion[int(label), int(pred)] += 1
    diagnostics["confusion_matrix"] = {
        str(label): {str(pred): int(confusion[label, pred]) for pred in range(outputs)}
        for label in range(outputs)
    }
    diagnostics["accuracy_by_label"] = {
        str(label): (
            float(confusion[label, label] / confusion[label].sum())
            if int(confusion[label].sum()) > 0
            else None
        )
        for label in range(outputs)
    }
    return diagnostics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="device_readout_array_eval")
    ap.add_argument("--simulator")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--activation-csv", type=Path, required=True)
    ap.add_argument("--activation-phase", default="initial_eval")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--cap-csv", type=Path, required=True)
    ap.add_argument("--hidden-cells", type=int, default=8)
    ap.add_argument("--outputs", type=int, default=3)
    ap.add_argument("--design", choices=sorted(direct_flow.SYNAPSE_DESIGNS), default="split_signed_v1")
    ap.add_argument("--output-head", choices=sorted(direct_flow.OUTPUT_HEAD_MODES), default="split_score_none")
    ap.add_argument("--hidden-cap-f", type=float, default=4.0)
    ap.add_argument("--score-reset-v", type=float, default=0.0)
    ap.add_argument("--score-cap-f", type=float, default=10.0)
    ap.add_argument("--output-cap-f", type=float, default=20.0)
    ap.add_argument("--sample-ns", type=float, default=2.95)
    ap.add_argument("--cycle-ns", type=float, default=16.0)
    ap.add_argument(
        "--activation-drive",
        choices=["held", "ramp", "measured"],
        default="held",
        help="Drive activation rails as held sampled voltages, simple ramps, or measured act*_<offset> waveform samples.",
    )
    ap.add_argument("--activation-settle-ns", type=float, default=2.95)
    ap.add_argument(
        "--activation-sample-offsets-ns",
        default="",
        help="Comma-separated offsets used by --activation-drive measured, matching act*_<offset> columns.",
    )
    ap.add_argument(
        "--readout-load-mode",
        choices=["forward_only", "flow_offstate"],
        default="forward_only",
        help="forward_only keeps only the output-forward array; flow_offstate also attaches inactive direct-flow write/pretrace stacks.",
    )
    ap.add_argument(
        "--score-sense-mode",
        choices=SCORE_SENSE_MODES,
        default="score_caps",
        help="How scorep/scoren rails are sensed: normal score caps, ideal current clamp, diode voltage load, or diode current probe.",
    )
    ap.add_argument("--tran-step-ps", type=float, default=10.0)
    ap.add_argument("--spice-accuracy-preset", choices=sorted(direct_flow.SPICE_ACCURACY_PRESETS), default="fast")
    ap.add_argument("--output-forward-width-scale", type=float, default=1.0)
    ap.add_argument("--output-forward-pos-width-scale", type=float, default=1.0)
    ap.add_argument("--output-forward-neg-width-scale", type=float, default=1.0)
    ap.add_argument("--output-bias-forward-width-scale", type=float, default=1.0)
    ap.add_argument("--output-relu-width-scale", type=float, default=1.0)
    ap.add_argument("--score-diode-width-u", type=float, default=1024.0)
    ap.add_argument("--score-mirror-cap-f", type=float, default=20.0)
    ap.add_argument("--readout-topology", choices=["dense", "random_fanin", "random_fanout"], default="dense")
    ap.add_argument("--readout-fan-in", type=int, default=3)
    ap.add_argument("--readout-fan-out", type=int, default=3)
    ap.add_argument("--readout-topology-seed", type=int, default=0)
    args = ap.parse_args()

    if args.hidden_cells <= 0 or args.outputs <= 1:
        raise SystemExit("--hidden-cells must be positive and --outputs must be at least 2.")
    if args.sample_ns <= 0.0 or args.cycle_ns <= 0.0 or args.tran_step_ps <= 0.0 or args.activation_settle_ns <= 0.0:
        raise SystemExit("--sample-ns, --cycle-ns, --activation-settle-ns, and --tran-step-ps must be positive.")
    if args.score_diode_width_u <= 0.0:
        raise SystemExit("--score-diode-width-u must be positive.")
    if args.readout_fan_in <= 0 or args.readout_fan_out <= 0:
        raise SystemExit("--readout-fan-in and --readout-fan-out must be positive.")
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
        readout_state = direct_flow.csv_readout_cap_state(args.cap_csv)
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    activations = readout_transfer.load_activations(
        args.activation_csv,
        args.activation_phase,
        args.hidden_cells,
        args.limit,
    )
    try:
        readout_fanins = build_readout_fanins(
            args.readout_topology,
            hidden_cells=args.hidden_cells,
            outputs=args.outputs,
            fan_in=args.readout_fan_in,
            fan_out=min(args.readout_fan_out, args.outputs),
            seed=args.readout_topology_seed,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    spice_bin, version = detect_spice(args.simulator)
    for directory in (GENERATED, RESULTS, SPICE_RESULTS):
        directory.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, GENERATED)
    safe = safe_tag(args.tag)
    t0 = time.perf_counter()
    deck, _samples = readout_array_netlist(
        activations=activations,
        readout_state=readout_state,
        design_name=args.design,
        output_head=args.output_head,
        hidden_cells=args.hidden_cells,
        outputs=args.outputs,
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
        tran_step_ps=args.tran_step_ps,
        spice_accuracy_preset=args.spice_accuracy_preset,
        score_sense_mode=args.score_sense_mode,
        output_forward_width_scale=args.output_forward_width_scale,
        output_forward_pos_width_scale=args.output_forward_pos_width_scale,
        output_forward_neg_width_scale=args.output_forward_neg_width_scale,
        output_bias_forward_width_scale=args.output_bias_forward_width_scale,
        output_relu_width_scale=args.output_relu_width_scale,
        score_diode_width_u=args.score_diode_width_u,
        score_mirror_cap_f=args.score_mirror_cap_f,
        readout_fanins=readout_fanins,
    )
    parsed = run_netlist(spice_bin, GENERATED / f"{safe}.cir", deck, args.timeout)
    df = rows_from_measures(parsed, activations, args.outputs)
    csv_path = RESULTS / f"{safe}.csv"
    df.to_csv(csv_path, index=False)
    summary = {
        "tag": safe,
        "simulator": version,
        "activation_csv": str(args.activation_csv),
        "activation_phase": args.activation_phase,
        "cap_csv": str(args.cap_csv),
        "samples": int(len(df)),
        "hidden_cells": int(args.hidden_cells),
        "outputs": int(args.outputs),
        "design": args.design,
        "output_head": args.output_head,
        "score_reset_v": float(args.score_reset_v),
        "score_cap_f": float(args.score_cap_f),
        "sample_ns": float(args.sample_ns),
        "cycle_ns": float(args.cycle_ns),
        "activation_drive": args.activation_drive,
        "activation_settle_ns": float(args.activation_settle_ns),
        "activation_sample_offsets_ns": [float(v) for v in activation_sample_offsets_ns],
        "readout_load_mode": args.readout_load_mode,
        "score_sense_mode": args.score_sense_mode,
        "readout_topology": args.readout_topology,
        "readout_fan_in": int(args.readout_fan_in),
        "readout_fan_out": int(args.readout_fan_out),
        "readout_topology_seed": int(args.readout_topology_seed),
        "spice_accuracy_preset": args.spice_accuracy_preset,
        "accuracy": float(df["correct"].mean()),
        "min_margin_v": float(df["score_margin"].min()),
        "prediction_histogram": {str(k): int(v) for k, v in df["predicted_label"].value_counts().sort_index().items()},
        "table_csv": str(csv_path),
        "wall_time_s": time.perf_counter() - t0,
    }
    summary.update(readout_topology_summary(readout_fanins, args.hidden_cells))
    summary.update(score_diagnostics(df, args.outputs))
    summary_path = SPICE_RESULTS / f"{safe}_summary.json"
    table_summary_path = RESULTS / f"{safe}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    table_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
