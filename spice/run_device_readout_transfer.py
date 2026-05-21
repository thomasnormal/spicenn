#!/usr/bin/env python3
"""Characterize the MOS readout transfer from stored caps to class scores.

This is a primitive diagnostic for the dense device-level network.  It holds
measured hidden activations on the `act*` nodes, programs readout capacitor
initial conditions from a CSV separator, and asks SPICE what class scores the
readout/output head actually produces.  The goal is to separate hidden-feature
quality from readout transfer errors.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from spicenn import CapStateProgram, load_readout_cap_state_csv
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from spicenn import CapStateProgram, load_readout_cap_state_csv

import run_device_xor2_random_hidden as direct_flow
from run_spice_sweep import detect_spice


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/tables"
SPICE_RESULTS = ROOT / "spice/results"
GENERATED = ROOT / "spice/generated"
TRANSFER_TOPOLOGIES = [
    "main",
    "split_score_caps",
    "split_score_clamped_current",
    "split_score_diode_clamp",
    "split_score_diode_current",
]


def piecewise_signal(points: list[tuple[float, float]]) -> str:
    deduped: list[tuple[float, float]] = []
    for t, v in points:
        if deduped and abs(deduped[-1][0] - t) < 1e-12:
            deduped[-1] = (t, v)
        else:
            deduped.append((t, v))
    return "PWL(" + " ".join(f"{t:.6g}n {v:.12g}" for t, v in deduped) + ")"


def guide_signal(
    cycles: int,
    cycle_ns: float,
    high_windows: list[tuple[float, float, float]],
    final_value: float = 0.0,
) -> str:
    points: list[tuple[float, float]] = [(0.0, final_value)]
    for idx in range(cycles):
        base = idx * cycle_ns
        for start, end, high in high_windows:
            points.extend(
                [
                    (base + start, final_value),
                    (base + start + 0.02, high),
                    (base + end, high),
                    (base + end + 0.02, final_value),
                ]
            )
    points.append((cycles * cycle_ns, final_value))
    return piecewise_signal(points)


def activation_signal(values: np.ndarray, cycle_ns: float, active_start_ns: float = 0.58) -> str:
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for idx, value in enumerate(values):
        base = idx * cycle_ns
        points.extend(
            [
                (base, 0.0),
                (base + active_start_ns, 0.0),
                (base + active_start_ns + 0.02, float(value)),
                (base + cycle_ns - 0.05, float(value)),
            ]
        )
    points.append((len(values) * cycle_ns, float(values[-1]) if len(values) else 0.0))
    return piecewise_signal(points)


def load_activations(path: Path, phase: str, hidden: int, limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "phase" in df.columns:
        df = df[df["phase"] == phase].copy()
    required = ["label", *[f"act{h}" for h in range(hidden)]]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"activation CSV is missing columns {missing}: {path}")
    if limit is not None:
        df = df.head(limit).copy()
    if df.empty:
        raise ValueError(f"activation CSV has no rows for phase {phase!r}: {path}")
    return df.reset_index(drop=True)


def readout_caps(init: dict[str, float], cap_f: float) -> str:
    return CapStateProgram("readout_caps", init, cap_f).render_spice()


def load_readout_cap_init(path: Path, *, hidden: int, outputs: int) -> dict[str, float]:
    return load_readout_cap_state_csv(path, hidden_count=hidden, output_count=outputs)


def split_score_caps() -> str:
    lines: list[str] = []
    for out in range(direct_flow.OUTPUTS):
        lines += [
            f"Cscorep{out} scorep{out} 0 10f IC=0",
            f"Cscoren{out} scoren{out} 0 10f IC=0",
            f"Rscorep{out} scorep{out} 0 1G",
            f"Rscoren{out} scoren{out} 0 1G",
            f"Mreset_scorep{out} scorep{out} rstf scorecm 0 NMOS W=4u L=180n",
            f"Mreset_scoren{out} scoren{out} rstf scorecm 0 NMOS W=4u L=180n",
        ]
    return "\n".join(lines)


def clamped_score_sources(score_v: float) -> str:
    lines: list[str] = []
    for out in range(direct_flow.OUTPUTS):
        lines += [
            f"Vscorep{out}_clamp scorep{out} 0 {score_v:.12g}",
            f"Vscoren{out}_clamp scoren{out} 0 {score_v:.12g}",
        ]
    return "\n".join(lines)


def diode_score_nodes(score_cap_f: float, diode_width_u: float, *, current_probe: bool = False) -> str:
    lines: list[str] = []
    for out in range(direct_flow.OUTPUTS):
        if current_probe:
            pos_source = f"scorep{out}_sense"
            neg_source = f"scoren{out}_sense"
            sense = [
                f"Vscorep{out}_sense {pos_source} 0 0",
                f"Vscoren{out}_sense {neg_source} 0 0",
            ]
        else:
            pos_source = "0"
            neg_source = "0"
            sense = []
        lines += [
            f"Cscorep{out} scorep{out} 0 {score_cap_f:.12g}f IC=0",
            f"Cscoren{out} scoren{out} 0 {score_cap_f:.12g}f IC=0",
            f"Rscorep{out} scorep{out} 0 1e12",
            f"Rscoren{out} scoren{out} 0 1e12",
            *sense,
            f"Mscorep{out}_diode scorep{out} scorep{out} {pos_source} 0 NSENSE W={diode_width_u:.12g}u L=180n",
            f"Mscoren{out}_diode scoren{out} scoren{out} {neg_source} 0 NSENSE W={diode_width_u:.12g}u L=180n",
            f"Mreset_scorep{out}_diode scorep{out} rstf 0 0 NMOS W=4u L=180n",
            f"Mreset_scoren{out}_diode scoren{out} rstf 0 0 NMOS W=4u L=180n",
        ]
    return "\n".join(lines)


def split_score_caps_output_forward(design: direct_flow.SynapseDesign) -> str:
    """Readout-only test topology with separate positive and negative score caps."""
    lines: list[str] = []
    if design.output_forward_style == "pass_act_buffered":
        lines.append("* Buffered hidden activation replicas for differential score-cap readout.")
        for h in range(direct_flow.HIDDEN):
            lines += [
                f"Mactbuf{h}_src vdd act{h} actbuf{h} 0 NSENSE W={design.output_forward_pos_width_u:.12g}u L=180n",
                f"Mactbuf{h}_rst actbuf{h} rstf 0 0 NREL W=4u L=180n",
            ]
            lines += direct_flow.node_parasitics(f"actbuf{h}")
    for out in range(direct_flow.OUTPUTS):
        lines.append(f"* Output {out}: split positive/negative score capacitors.")
        for h in range(direct_flow.HIDDEN):
            readout_internal_nodes = [
                f"sp{out}{h}p0",
                f"sp{out}{h}p1",
                f"sn{out}{h}n0",
                f"sn{out}{h}n1",
            ]
            if design.output_forward_style == "gate_stack":
                lines += [
                    f"Msp{out}{h}pos_a vdd act{h} sp{out}{h}p0 0 NSENSE W={design.output_forward_pos_width_u:.12g}u L=180n",
                    f"Msp{out}{h}pos_w sp{out}{h}p0 vw{out}{h}p sp{out}{h}p1 0 NREL W={design.output_forward_pos_width_u:.12g}u L=180n",
                    f"Msp{out}{h}pos_f sp{out}{h}p1 fwd scorep{out} 0 NREL W={design.output_forward_pos_width_u:.12g}u L=180n",
                    f"Msn{out}{h}neg_a vdd act{h} sn{out}{h}n0 0 NSENSE W={design.output_forward_neg_width_u:.12g}u L=180n",
                    f"Msn{out}{h}neg_w sn{out}{h}n0 vw{out}{h}n sn{out}{h}n1 0 NREL W={design.output_forward_neg_width_u:.12g}u L=180n",
                    f"Msn{out}{h}neg_f sn{out}{h}n1 fwd scoren{out} 0 NREL W={design.output_forward_neg_width_u:.12g}u L=180n",
                ]
            elif design.output_forward_style in {"pass_act_source", "pass_act_buffered"}:
                act_source = f"actbuf{h}" if design.output_forward_style == "pass_act_buffered" else f"act{h}"
                lines += [
                    f"Msp{out}{h}pos_w {act_source} vw{out}{h}p sp{out}{h}p1 0 NREL W={design.output_forward_pos_width_u:.12g}u L=180n",
                    f"Msp{out}{h}pos_f sp{out}{h}p1 fwd scorep{out} 0 NREL W={design.output_forward_pos_width_u:.12g}u L=180n",
                    f"Msn{out}{h}neg_w {act_source} vw{out}{h}n sn{out}{h}n1 0 NREL W={design.output_forward_neg_width_u:.12g}u L=180n",
                    f"Msn{out}{h}neg_f sn{out}{h}n1 fwd scoren{out} 0 NREL W={design.output_forward_neg_width_u:.12g}u L=180n",
                ]
            else:
                raise ValueError(f"unknown output forward style: {design.output_forward_style}")
            lines += direct_flow.node_parasitics(*readout_internal_nodes)
        bias_internal_nodes = [f"sp{out}bp0", f"sp{out}bp1", f"sn{out}bn0", f"sn{out}bn1"]
        if design.output_forward_style == "gate_stack":
            lines += [
                f"Msp{out}bpos_a vdd bias sp{out}bp0 0 NSENSE W={design.output_bias_forward_pos_width_u:.12g}u L=180n",
                f"Msp{out}bpos_w sp{out}bp0 vbo{out}p sp{out}bp1 0 NREL W={design.output_bias_forward_pos_width_u:.12g}u L=180n",
                f"Msp{out}bpos_f sp{out}bp1 fwd scorep{out} 0 NREL W={design.output_bias_forward_pos_width_u:.12g}u L=180n",
                f"Msn{out}bneg_a vdd bias sn{out}bn0 0 NSENSE W={design.output_bias_forward_neg_width_u:.12g}u L=180n",
                f"Msn{out}bneg_w sn{out}bn0 vbo{out}n sn{out}bn1 0 NREL W={design.output_bias_forward_neg_width_u:.12g}u L=180n",
                f"Msn{out}bneg_f sn{out}bn1 fwd scoren{out} 0 NREL W={design.output_bias_forward_neg_width_u:.12g}u L=180n",
            ]
        else:
            lines += [
                f"Msp{out}bpos_w bias vbo{out}p sp{out}bp1 0 NREL W={design.output_bias_forward_pos_width_u:.12g}u L=180n",
                f"Msp{out}bpos_f sp{out}bp1 fwd scorep{out} 0 NREL W={design.output_bias_forward_pos_width_u:.12g}u L=180n",
                f"Msn{out}bneg_w bias vbo{out}n sn{out}bn1 0 NREL W={design.output_bias_forward_neg_width_u:.12g}u L=180n",
                f"Msn{out}bneg_f sn{out}bn1 fwd scoren{out} 0 NREL W={design.output_bias_forward_neg_width_u:.12g}u L=180n",
            ]
            bias_internal_nodes = [f"sp{out}bp1", f"sn{out}bn1"]
        lines += direct_flow.node_parasitics(*bias_internal_nodes)
    return "\n".join(lines)


def build_readout_transfer_netlist(
    *,
    activations: pd.DataFrame,
    separator_csv: Path,
    synapse_design: str,
    separator_scale: float,
    readout_center_v: float,
    score_reset_v: float,
    output_forward_width_scale: float,
    output_forward_pos_width_scale: float,
    output_forward_neg_width_scale: float,
    output_bias_forward_width_scale: float,
    output_relu_width_scale: float,
    output_head: str,
    transfer_topology: str,
    cycle_ns: float,
    sample_offset_ns: float,
    tran_step_ps: float,
    cap_f: float,
    score_cap_f: float = 10.0,
    readout_init_mode: str = "csv_readout",
    readout_cap_csv: Path | None = None,
    score_diode_width_u: float = 256.0,
    output_bias_offset_v: float = 0.0,
) -> tuple[str, pd.DataFrame]:
    hidden = direct_flow.HIDDEN
    outputs = direct_flow.OUTPUTS
    design = direct_flow.scaled_synapse_design(
        synapse_design,
        hidden_delta_width_scale=1.0,
        hidden_gradient_width_scale=1.0,
        readout_gradient_width_scale=1.0,
        output_forward_width_scale=output_forward_width_scale,
        output_forward_pos_width_scale=output_forward_pos_width_scale,
        output_forward_neg_width_scale=output_forward_neg_width_scale,
        output_bias_forward_width_scale=output_bias_forward_width_scale,
        output_relu_width_scale=output_relu_width_scale,
    )
    if readout_cap_csv is not None:
        init = load_readout_cap_init(readout_cap_csv, hidden=hidden, outputs=outputs)
    else:
        init = direct_flow.apply_output_bias_offset(
            direct_flow.readout_init(
                seed=0,
                mode=readout_init_mode,
                separator_scale=separator_scale,
                separator_offset_v=0.0,
                readout_center_v=readout_center_v,
                random_center_v=None,
                random_span_v=0.0,
                random_pos_center_v=None,
                random_neg_center_v=None,
                random_pos_span_v=None,
                random_neg_span_v=None,
                separator_csv=separator_csv,
                separator_phase="initial_eval",
            ),
            output_bias_offset_v,
        )
    weights, biases = direct_flow.csv_readout_weights(separator_csv)
    act = activations[[f"act{h}" for h in range(hidden)]].to_numpy(dtype=float)
    ideal_logits = separator_scale * (act @ np.asarray(weights, dtype=float).T + np.asarray(biases, dtype=float))
    rows = activations[["label", *[f"act{h}" for h in range(hidden)]]].copy()
    for out in range(outputs):
        rows[f"ideal{out}"] = ideal_logits[:, out]
    rows["ideal_predicted_label"] = ideal_logits.argmax(axis=1)
    rows["ideal_correct"] = rows["ideal_predicted_label"].to_numpy() == rows["label"].astype(int).to_numpy()

    cycles = len(activations)
    measures: list[str] = []
    for idx in range(cycles):
        at = idx * cycle_ns + sample_offset_ns
        for out in range(outputs):
            if transfer_topology in {"split_score_caps", "split_score_diode_clamp"}:
                measures += [
                    f".meas tran scorep{out}_{idx} FIND V(scorep{out}) AT={at:.6g}n",
                    f".meas tran scoren{out}_{idx} FIND V(scoren{out}) AT={at:.6g}n",
                ]
            elif transfer_topology == "split_score_diode_current":
                measures += [
                    f".meas tran scorep{out}_{idx} FIND I(Vscorep{out}_sense) AT={at:.6g}n",
                    f".meas tran scoren{out}_{idx} FIND I(Vscoren{out}_sense) AT={at:.6g}n",
                ]
            elif transfer_topology == "split_score_clamped_current":
                measures += [
                    f".meas tran scorep{out}_{idx} FIND I(Vscorep{out}_clamp) AT={at:.6g}n",
                    f".meas tran scoren{out}_{idx} FIND I(Vscoren{out}_clamp) AT={at:.6g}n",
                ]
            else:
                measures += [
                    f".meas tran score{out}_{idx} FIND V(score{out}) AT={at:.6g}n",
                    f".meas tran out{out}_{idx} FIND V(out{out}) AT={at:.6g}n",
                ]
        for h in range(hidden):
            measures.append(f".meas tran act{h}_{idx} FIND V(act{h}) AT={at:.6g}n")

    activation_sources = []
    for h in range(hidden):
        activation_sources.append(f"Vact{h} act{h} 0 {activation_signal(act[:, h], cycle_ns)}")
    if transfer_topology == "split_score_caps":
        # Production temporary_caps already creates scorep/scoren caps and
        # reset devices.  Keep this harness wired to the same primitive instead
        # of shadowing those cells with duplicate diagnostic definitions.
        score_cells = ""
        readout_forward = split_score_caps_output_forward(design)
    elif transfer_topology == "split_score_clamped_current":
        score_cells = clamped_score_sources(score_reset_v)
        readout_forward = split_score_caps_output_forward(design)
    elif transfer_topology == "split_score_diode_clamp":
        score_cells = diode_score_nodes(score_cap_f, score_diode_width_u)
        readout_forward = split_score_caps_output_forward(design)
    elif transfer_topology == "split_score_diode_current":
        score_cells = diode_score_nodes(score_cap_f, score_diode_width_u, current_probe=True)
        readout_forward = split_score_caps_output_forward(design)
    elif transfer_topology == "main":
        score_cells = ""
        readout_forward = direct_flow.output_forward(design, output_head)
    else:
        raise ValueError(f"unknown transfer topology: {transfer_topology}")

    netlist = f"""
* Readout transfer characterization deck.
* Synapse design: {synapse_design}; output head: {output_head}
.param VDD=1.2
.model NMOS NMOS LEVEL=1 VTO=0.35 KP=220u LAMBDA=0.03 GAMMA=0.20 PHI=0.60
.model NREL NMOS LEVEL=1 VTO=0.12 KP=260u LAMBDA=0.03 GAMMA=0.10 PHI=0.60
.model NSENSE NMOS LEVEL=1 VTO=0.03 KP=260u LAMBDA=0.03 GAMMA=0.05 PHI=0.60
.model PMOS PMOS LEVEL=1 VTO=-0.35 KP=90u LAMBDA=0.03 GAMMA=0.20 PHI=0.60
Vdd vdd 0 {{VDD}}
Vbias bias 0 {{VDD}}
Vscorecm scorecm 0 {score_reset_v:.12g}
Vfwd fwd 0 {guide_signal(cycles, cycle_ns, [(0.75, 3.00, 1.2)])}
Vrstf rstf 0 {guide_signal(cycles, cycle_ns, [(0.00, 0.50, 1.2)])}
Vrste rste 0 0
Vcmp cmp 0 0
Verr err 0 0
Vbwd bwd 0 0
Vacc acc 0 0
Vgcmp gcmp 0 0
Vapply apply 0 0
{chr(10).join(activation_sources)}
{"" if transfer_topology in {"split_score_clamped_current", "split_score_diode_clamp", "split_score_diode_current"} else direct_flow.temporary_caps(4.0, 4.0, 12.0, 2.0, False, score_reset_v, score_cap_f)}
{"" if transfer_topology in {"split_score_clamped_current", "split_score_diode_clamp", "split_score_diode_current"} else direct_flow.resets("score_direct", False, score_reset_v)}
{score_cells}
{readout_caps(init, cap_f)}
{readout_forward}
{chr(10).join(measures)}
.options method=gear maxord=2 rshunt=1e12 gmin=1e-12
.tran {tran_step_ps:.12g}p {cycles * cycle_ns:.12g}n uic
.end
""".strip()
    return netlist + "\n", rows


def attach_measurements(rows: pd.DataFrame, parsed: dict[str, float], transfer_topology: str) -> pd.DataFrame:
    outputs = direct_flow.OUTPUTS
    hidden = direct_flow.HIDDEN
    rows = rows.copy()
    for idx in range(len(rows)):
        for out in range(outputs):
            if transfer_topology in {
                "split_score_caps",
                "split_score_clamped_current",
                "split_score_diode_clamp",
                "split_score_diode_current",
            }:
                scorep = parsed[f"scorep{out}_{idx}"]
                scoren = parsed[f"scoren{out}_{idx}"]
                rows.loc[idx, f"scorep{out}"] = scorep
                rows.loc[idx, f"scoren{out}"] = scoren
                rows.loc[idx, f"score{out}"] = scorep - scoren
                rows.loc[idx, f"out{out}"] = scorep - scoren
            else:
                rows.loc[idx, f"score{out}"] = parsed[f"score{out}_{idx}"]
                rows.loc[idx, f"out{out}"] = parsed[f"out{out}_{idx}"]
        for h in range(hidden):
            rows.loc[idx, f"measured_act{h}"] = parsed[f"act{h}_{idx}"]
    score_values = rows[[f"score{out}" for out in range(outputs)]].to_numpy()
    out_values = rows[[f"out{out}" for out in range(outputs)]].to_numpy()
    labels = rows["label"].astype(int).to_numpy()
    rows["score_predicted_label"] = score_values.argmax(axis=1)
    rows["out_predicted_label"] = out_values.argmax(axis=1)
    rows["score_correct"] = rows["score_predicted_label"].to_numpy() == labels
    rows["out_correct"] = rows["out_predicted_label"].to_numpy() == labels
    return rows


def _argmax_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.argmax(logits, axis=1) == labels)) if len(labels) else 0.0


def _diagonal_ideal_fit(source: np.ndarray, ideal: np.ndarray) -> tuple[np.ndarray, float]:
    fitted = np.zeros_like(ideal, dtype=float)
    rmses: list[float] = []
    for out in range(ideal.shape[1]):
        x = np.column_stack([source[:, out], np.ones(source.shape[0])])
        coef, *_ = np.linalg.lstsq(x, ideal[:, out], rcond=None)
        fitted[:, out] = x @ coef
        rmses.append(float(np.sqrt(np.mean((fitted[:, out] - ideal[:, out]) ** 2))))
    return fitted, float(np.mean(rmses)) if rmses else 0.0


def _full_ideal_fit(source: np.ndarray, ideal: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.column_stack([source, np.ones(source.shape[0])])
    coef, *_ = np.linalg.lstsq(x, ideal, rcond=None)
    fitted = x @ coef
    return fitted, float(np.sqrt(np.mean((fitted - ideal) ** 2))) if fitted.size else 0.0


def affine_recovery_metrics(prefix: str, source: np.ndarray, ideal: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Measure whether a simple downstream affine calibration could recover class order.

    This is an oracle diagnostic, not a training result.  It answers whether the
    MOS readout scores still contain the separator information up to per-class
    gain/offset or a small full affine class mixer.
    """
    diag_logits, diag_rmse = _diagonal_ideal_fit(source, ideal)
    full_logits, full_rmse = _full_ideal_fit(source, ideal)
    return {
        f"{prefix}_diag_idealfit_accuracy": _argmax_accuracy(diag_logits, labels),
        f"{prefix}_diag_idealfit_rmse_v": diag_rmse,
        f"{prefix}_full_idealfit_accuracy": _argmax_accuracy(full_logits, labels),
        f"{prefix}_full_idealfit_rmse_v": full_rmse,
    }


def column_centering_metrics(prefix: str, source: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    means = source.mean(axis=0) if source.size else np.zeros(source.shape[1] if source.ndim == 2 else 0)
    centered = source - means
    if len(labels):
        predictions = np.argmax(centered, axis=1)
        accuracy = float(np.mean(predictions == labels))
        margins = []
        for row, label in zip(centered, labels):
            others = np.delete(row, int(label))
            margins.append(float(row[int(label)] - np.max(others)))
        min_margin = float(np.min(margins)) if margins else 0.0
    else:
        accuracy = 0.0
        min_margin = 0.0
    return {
        f"{prefix}_column_centered_accuracy": accuracy,
        f"{prefix}_column_centered_min_margin_v": min_margin,
        f"{prefix}_mean_by_output_v": {str(out): float(value) for out, value in enumerate(means)},
    }


def transfer_summary(rows: pd.DataFrame, tag: str, synapse_design: str, separator_scale: float, wall_time_s: float) -> dict[str, Any]:
    outputs = 0
    while f"ideal{outputs}" in rows.columns:
        outputs += 1
    if outputs == 0:
        outputs = direct_flow.OUTPUTS
    labels = rows["label"].astype(int).to_numpy()
    ideal_values = rows[[f"ideal{out}" for out in range(outputs)]].to_numpy(dtype=float)
    score_values = rows[[f"score{out}" for out in range(outputs)]].to_numpy(dtype=float)
    out_values = rows[[f"out{out}" for out in range(outputs)]].to_numpy(dtype=float)
    summary: dict[str, Any] = {
        "tag": tag,
        "synapse_design": synapse_design,
        "separator_scale": separator_scale,
        "samples": int(len(rows)),
        "ideal_accuracy": float(rows["ideal_correct"].mean()),
        "score_accuracy": float(rows["score_correct"].mean()),
        "out_accuracy": float(rows["out_correct"].mean()),
        "wall_time_s": wall_time_s,
    }
    summary.update(affine_recovery_metrics("score", score_values, ideal_values, labels))
    summary.update(affine_recovery_metrics("out", out_values, ideal_values, labels))
    summary.update(column_centering_metrics("score", score_values, labels))
    summary.update(column_centering_metrics("out", out_values, labels))
    for out in range(outputs):
        ideal = ideal_values[:, out]
        score = score_values[:, out]
        outv = out_values[:, out]
        summary[f"score_corr_out{out}"] = float(np.corrcoef(ideal, score)[0, 1]) if np.std(ideal) and np.std(score) else None
        summary[f"out_corr_out{out}"] = float(np.corrcoef(ideal, outv)[0, 1]) if np.std(ideal) and np.std(outv) else None
        summary[f"score_span_out{out}_v"] = float(score.max() - score.min())
        summary[f"out_span_out{out}_v"] = float(outv.max() - outv.min())
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="device_readout_transfer")
    ap.add_argument("--simulator")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--activation-csv", type=Path, required=True)
    ap.add_argument("--activation-phase", default="final_eval")
    ap.add_argument("--separator-csv", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--hidden-cells", type=int, default=8)
    ap.add_argument("--outputs", type=int, default=3)
    ap.add_argument("--synapse-design", choices=sorted(direct_flow.SYNAPSE_DESIGNS), default="split_signed_v1")
    ap.add_argument("--separator-scale", type=float, default=1.0)
    ap.add_argument("--readout-init-mode", choices=sorted(direct_flow.PROGRAMMED_READOUT_INITS), default="csv_readout")
    ap.add_argument(
        "--readout-cap-csv",
        type=Path,
        help="Optional direct cap program with cap,value rows. Overrides --readout-init-mode for transfer checks.",
    )
    ap.add_argument("--readout-center-v", type=float, default=0.64)
    ap.add_argument(
        "--output-bias-offset-v",
        type=float,
        default=0.0,
        help="Binary differential output-bias cap offset; negative favors class 1 and positive favors class 0.",
    )
    ap.add_argument("--score-reset-v", type=float, default=0.0)
    ap.add_argument("--output-head", choices=direct_flow.OUTPUT_HEAD_MODES, default="source_follower")
    ap.add_argument("--output-forward-width-scale", type=float, default=1.0)
    ap.add_argument("--output-forward-pos-width-scale", type=float, default=1.0)
    ap.add_argument("--output-forward-neg-width-scale", type=float, default=1.0)
    ap.add_argument("--output-bias-forward-width-scale", type=float, default=1.0)
    ap.add_argument("--output-relu-width-scale", type=float, default=1.0)
    ap.add_argument("--transfer-topology", choices=TRANSFER_TOPOLOGIES, default="main")
    ap.add_argument("--cycle-ns", type=float, default=4.0)
    ap.add_argument("--sample-offset-ns", type=float, default=2.95)
    ap.add_argument("--tran-step-ps", type=float, default=10.0)
    ap.add_argument("--cap-f", type=float, default=4.0)
    ap.add_argument("--score-cap-f", type=float, default=10.0)
    ap.add_argument("--score-diode-width-u", type=float, default=256.0)
    args = ap.parse_args()

    if args.hidden_cells <= 0 or args.outputs <= 1:
        raise SystemExit("--hidden-cells must be positive and --outputs must be at least 2.")
    if args.score_diode_width_u <= 0:
        raise SystemExit("--score-diode-width-u must be positive.")
    direct_flow.set_hidden_cells(args.hidden_cells)
    direct_flow.set_output_count(args.outputs)
    activations = load_activations(args.activation_csv, args.activation_phase, args.hidden_cells, args.limit)
    netlist, rows = build_readout_transfer_netlist(
        activations=activations,
        separator_csv=args.separator_csv,
        synapse_design=args.synapse_design,
        separator_scale=args.separator_scale,
        readout_center_v=args.readout_center_v,
        output_bias_offset_v=args.output_bias_offset_v,
        score_reset_v=args.score_reset_v,
        output_forward_width_scale=args.output_forward_width_scale,
        output_forward_pos_width_scale=args.output_forward_pos_width_scale,
        output_forward_neg_width_scale=args.output_forward_neg_width_scale,
        output_bias_forward_width_scale=args.output_bias_forward_width_scale,
        output_relu_width_scale=args.output_relu_width_scale,
        output_head=args.output_head,
        transfer_topology=args.transfer_topology,
        cycle_ns=args.cycle_ns,
        sample_offset_ns=args.sample_offset_ns,
        tran_step_ps=args.tran_step_ps,
        cap_f=args.cap_f,
        score_cap_f=args.score_cap_f,
        readout_init_mode=args.readout_init_mode,
        readout_cap_csv=args.readout_cap_csv,
        score_diode_width_u=args.score_diode_width_u,
    )
    spice_bin, simulator_version = detect_spice(args.simulator)
    GENERATED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    SPICE_RESULTS.mkdir(parents=True, exist_ok=True)
    safe_tag = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in args.tag)
    t0 = time.perf_counter()
    parsed = direct_flow.run_netlist(spice_bin, GENERATED / f"{safe_tag}.cir", netlist, args.timeout)
    wall_time_s = time.perf_counter() - t0
    measured = attach_measurements(rows, parsed, args.transfer_topology)
    summary = transfer_summary(measured, safe_tag, args.synapse_design, args.separator_scale, wall_time_s)
    summary.update(
        {
            "simulator": simulator_version,
            "activation_csv": str(args.activation_csv),
            "activation_phase": args.activation_phase,
            "separator_csv": str(args.separator_csv),
            "readout_init_mode": args.readout_init_mode,
            "readout_cap_csv": str(args.readout_cap_csv) if args.readout_cap_csv is not None else None,
            "readout_center_v": args.readout_center_v,
            "output_bias_offset_v": args.output_bias_offset_v,
            "score_reset_v": args.score_reset_v,
            "output_head": args.output_head,
            "transfer_topology": args.transfer_topology,
            "cycle_ns": args.cycle_ns,
            "sample_offset_ns": args.sample_offset_ns,
            "tran_step_ps": args.tran_step_ps,
            "score_cap_f": args.score_cap_f,
            "score_diode_width_u": args.score_diode_width_u
            if args.transfer_topology in {"split_score_diode_clamp", "split_score_diode_current"}
            else None,
        }
    )
    curve = RESULTS / f"{safe_tag}.csv"
    measured.to_csv(curve, index=False)
    (SPICE_RESULTS / f"{safe_tag}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
