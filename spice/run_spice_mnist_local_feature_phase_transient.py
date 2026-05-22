from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_spice_mnist_batch_op_train import read_wrdata_row
from run_spice_mnist_local_block_batch_op_train import (
    local_activation_deriv_expr as block_local_activation_deriv_expr,
    local_activation_expr as block_local_activation_expr,
    block_indices,
    readout_feedback_expr,
    synapse_transfer_expr,
)
from run_spice_mnist_local_feature_batch_op_train import run_eval, run_train_batch, wrap_xyce_behavioral_rhs, xyce_prn_path
from run_spice_mnist_train import load_mnist_sequence, mnist_index_splits
from run_spice_sweep import ROOT, detect_spice, is_xyce, prepare_netlist_for_simulator, run_tiny_test, run_simulator_netlist
from spice_adapter import SPICE_SIMULATOR_ARGS_ENV

TrainState = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
MNIST_TRAIN_COUNT = 60000
MNIST_TEST_COUNT = 10000


def sanitize_tag(tag: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in tag)


def phase_pwl(pulses: list[tuple[float, float]], t_stop: float, edge: float) -> str:
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for t_on, t_off in pulses:
        points.append((max(0.0, t_on - edge), 0.0))
        points.append((t_on, 1.0))
        points.append((t_off, 1.0))
        points.append((min(t_stop, t_off + edge), 0.0))
    points.append((t_stop, 0.0))
    cleaned: list[tuple[float, float]] = []
    for t, val in sorted(points):
        if cleaned and abs(cleaned[-1][0] - t) < 1e-18:
            cleaned[-1] = (t, val)
        else:
            cleaned.append((t, val))
    return "PWL(" + " ".join(f"{t:.12g} {val:.12g}" for t, val in cleaned) + ")"


def phase_pulse_area(phase: float, edge: float) -> float:
    if phase <= 0.0:
        raise ValueError("phase must be positive")
    if edge < 0.0:
        raise ValueError("edge must be non-negative")
    return phase + edge


def phase_state_descriptions(update_mode: str) -> dict[str, str]:
    if update_mode == "phased":
        temporary = (
            "feature activations, class scores, class deltas, hidden/backward feature deltas, "
            "and gradient accumulators are capacitor voltages"
        )
    elif update_mode == "direct":
        temporary = (
            "feature activations, class scores, class deltas, and hidden/backward feature deltas "
            "are capacitor voltages; weights are updated directly during each per-sample update phase"
        )
    else:
        raise ValueError("update_mode must be 'phased' or 'direct'")
    return {
        "persistent_state": (
            "local feature weights, local biases, class readout weights, and output biases are "
            "persistent capacitor voltages initialized once at the start of the transient"
        ),
        "temporary_state": temporary,
    }


def strict_fully_on_device_contract_met(
    batch_size: int,
    reference_mode: str,
    init_weights: str,
) -> bool:
    return batch_size == 1 and reference_mode == "none" and not init_weights


def validate_strict_fully_on_device_args(
    batch_size: int,
    reference_mode: str,
    init_weights: str,
) -> None:
    if batch_size != 1:
        raise ValueError("--strict-fully-on-device requires --batch-size 1")
    if reference_mode != "none":
        raise ValueError("--strict-fully-on-device requires --reference-mode none")
    if init_weights:
        raise ValueError("--strict-fully-on-device requires random init; do not pass --init-weights")


def phase_execution_contract_fields(
    batch_size: int,
    reference_mode: str,
    init_weights: str = "",
    strict_fully_on_device: bool = False,
) -> dict[str, bool | str]:
    if reference_mode not in {"spice", "none"}:
        raise ValueError("reference_mode must be 'spice' or 'none'")
    strict_contract_met = strict_fully_on_device_contract_met(batch_size, reference_mode, init_weights)
    return {
        "single_phase_training_transient": True,
        "weights_persist_inside_phase_transient": True,
        "python_weight_updates_between_samples": False,
        "python_checkpointing_between_samples": False,
        "reference_replay_used_for_diagnostics": reference_mode == "spice",
        "fully_on_device_execution_contract_met": batch_size == 1,
        "strict_fully_on_device_requested": strict_fully_on_device,
        "strict_fully_on_device_contract_met": strict_contract_met,
        "random_init_used": not bool(init_weights),
        "initial_weights_source": "random_init" if not init_weights else "checkpoint",
        "execution_contract_note": (
            "This gate covers the executed phase-transient training path only: one persistent-state "
            "training transient, batch_size=1 online updates, and no Python weight writes between "
            "samples. Reference/eval runs are diagnostics after the training transient."
        ),
    }


def index_prefix_metadata(indices: np.ndarray, prefix_len: int = 16) -> dict[str, object]:
    idx = np.asarray(indices, dtype=np.int64)
    return {
        "count": int(idx.size),
        "sha256": hashlib.sha256(idx.tobytes()).hexdigest(),
        "prefix": [int(value) for value in idx[:prefix_len]],
    }


def sample_source_pwl(values: np.ndarray, sample_starts: list[float], t_stop: float, edge: float) -> str:
    points: list[tuple[float, float]] = [(0.0, float(values[0]))]
    for s, val in enumerate(values):
        t = sample_starts[s]
        prev = float(values[s - 1] if s > 0 else val)
        points.append((max(0.0, t - edge), prev))
        points.append((t, float(val)))
    points.append((t_stop, float(values[-1])))
    cleaned: list[tuple[float, float]] = []
    for t, val in points:
        if cleaned and abs(cleaned[-1][0] - t) < 1e-18:
            cleaned[-1] = (t, val)
        else:
            cleaned.append((t, val))
    return "PWL(" + " ".join(f"{t:.12g} {val:.12g}" for t, val in cleaned) + ")"


def tanh_expr(expr: str) -> str:
    return f"(2/(1+exp(-2*({expr})))-1)"


def local_activation_expr(
    expr: str,
    local_activation: str,
    relu_clip: float,
    relu_leak: float = 0.01,
    softplus_beta: float = 10.0,
) -> str:
    return block_local_activation_expr(expr, local_activation, relu_clip, relu_leak, softplus_beta)


def local_activation_deriv_expr(
    preactivation_expr: str,
    activation_node: str,
    local_activation: str,
    relu_clip: float,
    activation_derivative: str = "exact",
    derivative_floor: float = 0.0,
    derivative_gate_threshold: float = 1e-6,
    relu_leak: float = 0.01,
    softplus_beta: float = 10.0,
) -> str:
    return block_local_activation_deriv_expr(
        preactivation_expr,
        f"V({activation_node})",
        local_activation,
        relu_clip,
        activation_derivative,
        derivative_floor,
        derivative_gate_threshold,
        relu_leak,
        softplus_beta,
    )


def target_matrix(labels: np.ndarray, n_classes: int, softmax_output: bool = False) -> np.ndarray:
    targets = np.zeros((len(labels), n_classes)) if softmax_output else -np.ones((len(labels), n_classes))
    for s, label in enumerate(labels):
        targets[s, int(label)] = 1.0
    return targets


def parse_measured_vector(stdout: str, n_vec: int) -> np.ndarray:
    return parse_named_measured_vector(stdout, "m", n_vec)


def parse_named_measured_vector(stdout: str, prefix: str, n_vec: int) -> np.ndarray:
    vals = np.empty(n_vec, dtype=float)
    for i in range(n_vec):
        name = f"{prefix}{i:05d}"
        m = re.search(rf"(?im)^\s*{name}\s*=\s*([-+0-9.eE]+)", stdout)
        if not m:
            raise ValueError(f"missing final-state measurement {name}")
        vals[i] = float(m.group(1))
    return vals


def parse_probe_update_list(raw: str, updates: int) -> tuple[int, ...]:
    if not raw:
        return ()
    selected: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if item in {"final", "last"}:
            selected.add(updates)
            continue
        if item == "powers2":
            value = 1
            while value <= updates:
                selected.add(value)
                value *= 2
            selected.add(updates)
            continue
        if "-" in item:
            bounds = item.split("-", 1)
            if len(bounds) != 2 or not bounds[0] or not bounds[1]:
                raise ValueError(f"invalid probe update range {item!r}")
            start, stop = (int(bounds[0]), int(bounds[1]))
            if start > stop:
                raise ValueError(f"invalid descending probe update range {item!r}")
            selected.update(range(start, stop + 1))
            continue
        selected.add(int(item))
    bad = sorted(value for value in selected if value < 1 or value > updates)
    if bad:
        raise ValueError(f"probe updates must be in [1, {updates}], got {bad}")
    return tuple(sorted(selected))


def parse_probe_measurements(stdout: str, probe_updates: tuple[int, ...], n_vec: int) -> dict[int, np.ndarray]:
    return {
        update: parse_named_measured_vector(stdout, f"p{probe_idx:03d}_", n_vec)
        for probe_idx, update in enumerate(probe_updates)
    }


def read_xyce_print_last_row(path: Path, expected_values: int) -> np.ndarray:
    rows = read_xyce_print_rows(path, expected_values)
    if not rows:
        raise ValueError(f"no Xyce print data rows found in {path}")
    return rows[-1][1]


def read_xyce_print_rows(path: Path, expected_values: int) -> list[tuple[float, np.ndarray]]:
    rows: list[tuple[float, np.ndarray]] = []
    for raw in path.read_text().splitlines():
        parts = raw.split()
        if not parts or parts[0].lower() in {"index", "end"}:
            continue
        try:
            int(float(parts[0]))
            values = np.array([float(item) for item in parts[1:]], dtype=float)
        except ValueError:
            continue
        if values.size < expected_values:
            raise ValueError(f"Xyce print row in {path} had {values.size} values, expected {expected_values}")
        rows.append((float(values[0]), values[-expected_values:]))
    if not rows:
        raise ValueError(f"no Xyce print data rows found in {path}")
    return rows


def xyce_print_vectors_at_times(
    rows: list[tuple[float, np.ndarray]],
    requested_times: dict[int, float],
) -> dict[int, np.ndarray]:
    if not requested_times:
        return {}
    out: dict[int, np.ndarray] = {}
    for update, target_time in requested_times.items():
        row_time, values = min(rows, key=lambda row: abs(row[0] - target_time))
        tolerance = max(1e-15, abs(target_time) * 1e-6)
        if abs(row_time - target_time) > tolerance:
            raise ValueError(
                f"no Xyce print row near probe update {update} time {target_time:.12g}; closest was {row_time:.12g}"
            )
        out[update] = values
    return out


def prepare_phase_netlist(netlist: str, spice_bin: str) -> str:
    rendered = prepare_netlist_for_simulator(netlist, spice_bin)
    return wrap_xyce_behavioral_rhs(rendered) if is_xyce(spice_bin) else rendered


def make_phase_schedule(
    update_batch_size: int,
    updates: int,
    phase: float,
    gap: float,
    direct_update: bool = False,
) -> tuple[dict[str, list[tuple[float, float]]], list[float], float]:
    phases = {"act": [], "score": [], "err": [], "bwd": [], "acc": [], "apply": [], "clear": []}
    sample_starts: list[float] = []
    t = phase
    for _update in range(updates):
        for _sample in range(update_batch_size):
            sample_starts.append(t)
            t += gap
            phases["act"].append((t, t + phase))
            t += phase + gap
            phases["score"].append((t, t + phase))
            t += phase + gap
            phases["err"].append((t, t + phase))
            t += phase + gap
            phases["bwd"].append((t, t + phase))
            t += phase + gap
            phases["acc"].append((t, t + phase))
            t += phase + gap
        if not direct_update:
            phases["apply"].append((t, t + phase))
            t += phase + gap
            phases["clear"].append((t, t + phase))
            t += phase + gap
    return phases, sample_starts, t + phase


def probe_measure_times(
    clear_phases: list[tuple[float, float]],
    probe_updates: tuple[int, ...],
    gap: float,
    t_stop: float,
    final_measure_time: float,
) -> dict[int, float]:
    if not probe_updates:
        return {}
    if gap <= 0.0:
        raise ValueError("probe measurements require --gap > 0")
    times: dict[int, float] = {}
    for update in probe_updates:
        _clear_start, clear_stop = clear_phases[update - 1]
        times[update] = min(final_measure_time, clear_stop + 0.5 * gap, t_stop)
    return times


def make_phase_transient_netlist(
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    w: np.ndarray,
    hb: np.ndarray,
    readout: np.ndarray,
    output_bias: np.ndarray,
    blocks: list[list[int]],
    lr: float,
    out_path: Path,
    linear_output: bool,
    update_batch_size: int,
    updates: int,
    phase: float,
    gap: float,
    edge: float,
    settle_ratio: float,
    transient_step: float,
    cw: float,
    cstate: float,
    cgrad: float,
    rleak: float,
    softmax_output: bool = False,
    output_mode: str = "wrdata",
    probe_updates: tuple[int, ...] = (),
    local_activation: str = "tanh",
    relu_clip: float = 1.0,
    activation_derivative: str = "exact",
    derivative_floor: float = 0.0,
    derivative_gate_threshold: float = 1e-6,
    readout_feedback_mode: str = "readout",
    readout_feedback_clip: float = 0.05,
    relu_leak: float = 0.01,
    softplus_beta: float = 10.0,
    hidden_synapse_mode: str = "linear",
    readout_synapse_mode: str = "linear",
    synapse_clip: float = 1.0,
    update_mode: str = "phased",
) -> tuple[str, int, float]:
    total_samples = x_batch.shape[0]
    if total_samples != update_batch_size * updates:
        raise ValueError("x_batch length must equal update_batch_size * updates")
    if rleak < 0:
        raise ValueError("rleak must be non-negative")
    if update_mode not in {"phased", "direct"}:
        raise ValueError("update_mode must be 'phased' or 'direct'")
    direct_update = update_mode == "direct"
    if direct_update and update_batch_size != 1:
        raise ValueError("direct update mode requires update_batch_size=1")
    n_blocks, channels, block_len = w.shape
    n_classes = readout.shape[0]
    phases, sample_starts, t_stop = make_phase_schedule(update_batch_size, updates, phase, gap, direct_update)
    tau = phase / settle_ratio
    phase_area = phase_pulse_area(phase, edge)
    targets = target_matrix(y_batch, n_classes, softmax_output)
    state_descriptions = phase_state_descriptions(update_mode)
    lines = [
        "* Phase-resolved transient local-feature/readout training deck.",
        f"* {state_descriptions['persistent_state']}.",
        f"* {state_descriptions['temporary_state']}.",
        f".param LR={lr:.12g}",
        f".param BS={update_batch_size}",
        f".param CW={cw:.12g}",
        f".param CSTATE={cstate:.12g}",
        f".param CGRAD={cgrad:.12g}",
        f".param RLEAK={rleak:.12g}",
        f".param TAU={tau:.12g}",
        f".param TPHASE={phase:.12g}",
        f".param TAREA={phase_area:.12g}",
        "",
        f"Vpact pact 0 {phase_pwl(phases['act'], t_stop, edge)}",
        f"Vpscore pscore 0 {phase_pwl(phases['score'], t_stop, edge)}",
        f"Vperr perr 0 {phase_pwl(phases['err'], t_stop, edge)}",
        f"Vpbwd pbwd 0 {phase_pwl(phases['bwd'], t_stop, edge)}",
        f"Vpacc pacc 0 {phase_pwl(phases['acc'], t_stop, edge)}",
        "",
    ]
    if not direct_update:
        lines[-1:-1] = [
            f"Vpapply papply 0 {phase_pwl(phases['apply'], t_stop, edge)}",
            f"Vpclear pclear 0 {phase_pwl(phases['clear'], t_stop, edge)}",
        ]
    for i in range(x_batch.shape[1]):
        lines.append(f"Vpix{i} pix{i} 0 {sample_source_pwl(x_batch[:, i], sample_starts, t_stop, edge)}")
    for k in range(n_classes):
        lines.append(f"Vtarget{k} target{k} 0 {sample_source_pwl(targets[:, k], sample_starts, t_stop, edge)}")
    lines.append("")
    for b in range(n_blocks):
        for c in range(channels):
            for p in range(block_len):
                lines.append(f"Cw{b}_{c}_{p} w{b}_{c}_{p} 0 {{CW}} IC={w[b, c, p]:.12g}")
                if rleak > 0:
                    lines.append(f"Rw{b}_{c}_{p} w{b}_{c}_{p} 0 {{RLEAK}}")
                if not direct_update:
                    lines.append(f"Cgw{b}_{c}_{p} gw{b}_{c}_{p} 0 {{CGRAD}} IC=0")
            lines.append(f"Chb{b}_{c} hb{b}_{c} 0 {{CW}} IC={hb[b, c]:.12g}")
            if rleak > 0:
                lines.append(f"Rhb{b}_{c} hb{b}_{c} 0 {{RLEAK}}")
            if not direct_update:
                lines.append(f"Cghb{b}_{c} ghb{b}_{c} 0 {{CGRAD}} IC=0")
            lines.append(f"Ch{b}_{c} h{b}_{c} 0 {{CSTATE}} IC=0")
            lines.append(f"Cdh{b}_{c} dh{b}_{c} 0 {{CSTATE}} IC=0")
    for k in range(n_classes):
        for b in range(n_blocks):
            for c in range(channels):
                lines.append(f"Cv{k}_{b}_{c} v{k}_{b}_{c} 0 {{CW}} IC={readout[k, b, c]:.12g}")
                if rleak > 0:
                    lines.append(f"Rv{k}_{b}_{c} v{k}_{b}_{c} 0 {{RLEAK}}")
                if not direct_update:
                    lines.append(f"Cgv{k}_{b}_{c} gv{k}_{b}_{c} 0 {{CGRAD}} IC=0")
        lines.append(f"Cob{k} ob{k} 0 {{CW}} IC={output_bias[k]:.12g}")
        if rleak > 0:
            lines.append(f"Rob{k} ob{k} 0 {{RLEAK}}")
        if not direct_update:
            lines.append(f"Cgob{k} gob{k} 0 {{CGRAD}} IC=0")
        lines.append(f"Cscore{k} score{k} 0 {{CSTATE}} IC=0")
        lines.append(f"Cd{k} d{k} 0 {{CSTATE}} IC=0")
    lines.append("")
    for b, idxs in enumerate(blocks):
        for c in range(channels):
            terms = [
                f"{synapse_transfer_expr(f'V(w{b}_{c}_{p})', hidden_synapse_mode, synapse_clip)}*V(pix{idx})"
                for p, idx in enumerate(idxs)
            ]
            terms.append(f"V(hb{b}_{c})")
            lines.append(f"Bpre_h{b}_{c} ah{b}_{c} 0 V = " + " + ".join(terms))
            h_calc = local_activation_expr(f"V(ah{b}_{c})", local_activation, relu_clip, relu_leak, softplus_beta)
            lines.append(f"Bstore_h{b}_{c} h{b}_{c} 0 I = V(pact)*{{CSTATE}}/{{TAU}}*(V(h{b}_{c})-({h_calc}))")
    for k in range(n_classes):
        score_terms = [
            f"{synapse_transfer_expr(f'V(v{k}_{b}_{c})', readout_synapse_mode, synapse_clip)}*V(h{b}_{c})"
            for b in range(n_blocks)
            for c in range(channels)
        ]
        score_terms.append(f"V(ob{k})")
        lines.append(f"Bscore{k} scorecalc{k} 0 V = " + " + ".join(score_terms))
        lines.append(f"Bstore_score{k} score{k} 0 I = V(pscore)*{{CSTATE}}/{{TAU}}*(V(score{k})-V(scorecalc{k}))")
    if softmax_output:
        denom = " + ".join(f"exp(V(score{k}))" for k in range(n_classes))
        for k in range(n_classes):
            lines.append(f"By{k} y{k} 0 V = exp(V(score{k}))/({denom})")
            lines.append(f"Bstore_d{k} d{k} 0 I = V(perr)*{{CSTATE}}/{{TAU}}*(V(d{k})-(V(target{k})-V(y{k})))")
    else:
        for k in range(n_classes):
            if linear_output:
                lines.append(f"By{k} y{k} 0 V = V(score{k})")
                delta_expr = f"(V(target{k})-V(y{k}))"
            else:
                y_expr = tanh_expr(f"V(score{k})")
                lines.append(f"By{k} y{k} 0 V = {y_expr}")
                delta_expr = f"(V(target{k})-({y_expr}))*(1-({y_expr})*({y_expr}))"
            lines.append(f"Bstore_d{k} d{k} 0 I = V(perr)*{{CSTATE}}/{{TAU}}*(V(d{k})-({delta_expr}))")
    lines.append("")
    for b, idxs in enumerate(blocks):
        for c in range(channels):
            feedback = " + ".join(
                readout_feedback_expr(
                    synapse_transfer_expr(f"V(v{k}_{b}_{c})", readout_synapse_mode, synapse_clip),
                    f"V(d{k})",
                    readout_feedback_mode,
                    readout_feedback_clip,
                )
                for k in range(n_classes)
            )
            deriv = local_activation_deriv_expr(
                f"V(ah{b}_{c})",
                f"h{b}_{c}",
                local_activation,
                relu_clip,
                activation_derivative,
                derivative_floor,
                derivative_gate_threshold,
                relu_leak,
                softplus_beta,
            )
            local_delta = f"({feedback})*{deriv}"
            lines.append(f"Bstore_dh{b}_{c} dh{b}_{c} 0 I = V(pbwd)*{{CSTATE}}/{{TAU}}*(V(dh{b}_{c})-({local_delta}))")
            for p, idx in enumerate(idxs):
                grad = f"V(dh{b}_{c})*V(pix{idx})"
                if direct_update:
                    lines.append(f"Bupd_w{b}_{c}_{p} w{b}_{c}_{p} 0 I = -V(pacc)*{{CW}}*{{LR}}/({{BS}}*{{TAREA}})*({grad})")
                else:
                    lines.append(f"Bacc_w{b}_{c}_{p} gw{b}_{c}_{p} 0 I = -V(pacc)*{{CGRAD}}/{{TAREA}}*({grad})")
                    lines.append(f"Bupd_w{b}_{c}_{p} w{b}_{c}_{p} 0 I = -V(papply)*{{CW}}*{{LR}}/({{BS}}*{{TAREA}})*V(gw{b}_{c}_{p})")
                    lines.append(f"Bclear_gw{b}_{c}_{p} gw{b}_{c}_{p} 0 I = V(pclear)*{{CGRAD}}/{{TAU}}*V(gw{b}_{c}_{p})")
            if direct_update:
                lines.append(f"Bupd_hb{b}_{c} hb{b}_{c} 0 I = -V(pacc)*{{CW}}*{{LR}}/({{BS}}*{{TAREA}})*V(dh{b}_{c})")
            else:
                lines.append(f"Bacc_hb{b}_{c} ghb{b}_{c} 0 I = -V(pacc)*{{CGRAD}}/{{TAREA}}*V(dh{b}_{c})")
                lines.append(f"Bupd_hb{b}_{c} hb{b}_{c} 0 I = -V(papply)*{{CW}}*{{LR}}/({{BS}}*{{TAREA}})*V(ghb{b}_{c})")
                lines.append(f"Bclear_ghb{b}_{c} ghb{b}_{c} 0 I = V(pclear)*{{CGRAD}}/{{TAU}}*V(ghb{b}_{c})")
    for k in range(n_classes):
        for b in range(n_blocks):
            for c in range(channels):
                grad = f"V(d{k})*V(h{b}_{c})"
                if direct_update:
                    lines.append(f"Bupd_v{k}_{b}_{c} v{k}_{b}_{c} 0 I = -V(pacc)*{{CW}}*{{LR}}/({{BS}}*{{TAREA}})*({grad})")
                else:
                    lines.append(f"Bacc_v{k}_{b}_{c} gv{k}_{b}_{c} 0 I = -V(pacc)*{{CGRAD}}/{{TAREA}}*({grad})")
                    lines.append(f"Bupd_v{k}_{b}_{c} v{k}_{b}_{c} 0 I = -V(papply)*{{CW}}*{{LR}}/({{BS}}*{{TAREA}})*V(gv{k}_{b}_{c})")
                    lines.append(f"Bclear_gv{k}_{b}_{c} gv{k}_{b}_{c} 0 I = V(pclear)*{{CGRAD}}/{{TAU}}*V(gv{k}_{b}_{c})")
        if direct_update:
            lines.append(f"Bupd_ob{k} ob{k} 0 I = -V(pacc)*{{CW}}*{{LR}}/({{BS}}*{{TAREA}})*V(d{k})")
        else:
            lines.append(f"Bacc_ob{k} gob{k} 0 I = -V(pacc)*{{CGRAD}}/{{TAREA}}*V(d{k})")
            lines.append(f"Bupd_ob{k} ob{k} 0 I = -V(papply)*{{CW}}*{{LR}}/({{BS}}*{{TAREA}})*V(gob{k})")
            lines.append(f"Bclear_gob{k} gob{k} 0 I = V(pclear)*{{CGRAD}}/{{TAU}}*V(gob{k})")
    vectors = [f"V(w{b}_{c}_{p})" for b in range(n_blocks) for c in range(channels) for p in range(block_len)]
    vectors += [f"V(hb{b}_{c})" for b in range(n_blocks) for c in range(channels)]
    vectors += [f"V(v{k}_{b}_{c})" for k in range(n_classes) for b in range(n_blocks) for c in range(channels)]
    vectors += [f"V(ob{k})" for k in range(n_classes)]
    vectors += [f"V(y{k})" for k in range(n_classes)]
    if output_mode == "native_measure":
        output_mode = "measure"
    if output_mode not in {"wrdata", "control_measure", "measure", "print"}:
        raise ValueError("output_mode must be 'wrdata', 'control_measure', 'measure', or 'print'")
    measure_time = max(0.0, t_stop - transient_step)
    probe_times = probe_measure_times(phases["acc" if direct_update else "clear"], probe_updates, gap, t_stop, measure_time)
    lines += ["", ".options method=gear maxord=2"]
    if output_mode == "print":
        print_times = sorted({measure_time, *probe_times.values()})
        print_time_list = ",".join(f"{time:.12g}" for time in print_times)
        lines += [
            f".options output OUTPUTTIMEPOINTS={print_time_list}",
            f".tran {transient_step:.12g} {t_stop:.12g} uic",
            ".print TRAN " + " ".join(vectors),
        ]
    elif output_mode == "measure":
        lines.append(f".tran {transient_step:.12g} {t_stop:.12g} uic")
        for i, vec in enumerate(vectors):
            lines.append(f".measure TRAN m{i:05d} FIND {vec} AT={measure_time:.12g}")
        for probe_idx, update in enumerate(probe_updates):
            for i, vec in enumerate(vectors):
                lines.append(f".measure TRAN p{probe_idx:03d}_{i:05d} FIND {vec} AT={probe_times[update]:.12g}")
    else:
        lines += [".control", f"tran {transient_step:.12g} {t_stop:.12g} uic"]
        if output_mode == "control_measure":
            for i, vec in enumerate(vectors):
                lines.append(f"meas tran m{i:05d} FIND {vec} AT={measure_time:.12g}")
            for probe_idx, update in enumerate(probe_updates):
                for i, vec in enumerate(vectors):
                    lines.append(f"meas tran p{probe_idx:03d}_{i:05d} FIND {vec} AT={probe_times[update]:.12g}")
        else:
            if probe_updates:
                raise ValueError("probe measurements require 'measure' or 'control_measure' output mode")
            lines.append(f"wrdata {out_path} " + " ".join(vectors))
        lines.append(".endc")
    lines += [".end", ""]
    return "\n".join(lines), len(vectors), t_stop


def unpack_state(
    vals: np.ndarray,
    w: np.ndarray,
    hb: np.ndarray,
    readout: np.ndarray,
    output_bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_classes = readout.shape[0]
    offset = 0
    nw = vals[offset : offset + w.size].reshape(w.shape)
    offset += w.size
    nhb = vals[offset : offset + hb.size].reshape(hb.shape)
    offset += hb.size
    nv = vals[offset : offset + readout.size].reshape(readout.shape)
    offset += readout.size
    nob = vals[offset : offset + output_bias.size]
    offset += output_bias.size
    y = vals[offset : offset + n_classes]
    return nw, nhb, nv, nob, y


def state_metrics(
    ref: TrainState,
    got: TrainState,
) -> dict[str, float]:
    names = ["local_weights", "local_bias", "readout", "output_bias"]
    metrics: dict[str, float] = {}
    for name, a, b in zip(names, ref, got):
        diff = np.asarray(b) - np.asarray(a)
        metrics[f"{name}_max_abs_diff"] = float(np.max(np.abs(diff)))
        metrics[f"{name}_mean_abs_diff"] = float(np.mean(np.abs(diff)))
        metrics[f"{name}_rms_diff"] = float(np.sqrt(np.mean(diff * diff)))
    all_diff = np.concatenate([(np.asarray(b) - np.asarray(a)).ravel() for a, b in zip(ref, got)])
    metrics["state_max_abs_diff"] = float(np.max(np.abs(all_diff)))
    metrics["state_mean_abs_diff"] = float(np.mean(np.abs(all_diff)))
    metrics["state_rms_diff"] = float(np.sqrt(np.mean(all_diff * all_diff)))
    return metrics


def update_direction_metrics(
    initial: TrainState,
    ref: TrainState,
    got: TrainState,
    eps: float = 1e-12,
) -> dict[str, float]:
    ref_delta = np.concatenate([(np.asarray(after) - np.asarray(before)).ravel() for before, after in zip(initial, ref)])
    got_delta = np.concatenate([(np.asarray(after) - np.asarray(before)).ravel() for before, after in zip(initial, got)])
    ref_norm = float(np.linalg.norm(ref_delta))
    got_norm = float(np.linalg.norm(got_delta))
    metrics = {
        "reference_update_l2": ref_norm,
        "phase_update_l2": got_norm,
    }
    if ref_norm > eps and got_norm > eps:
        metrics["state_update_direction_cosine"] = float(np.dot(ref_delta, got_delta) / (ref_norm * got_norm))
    else:
        metrics["state_update_direction_cosine"] = float("nan")
    mask = np.abs(ref_delta) > eps
    if np.any(mask):
        aligned = np.sign(ref_delta[mask]) == np.sign(got_delta[mask])
        metrics["state_update_sign_alignment_fraction"] = float(np.mean(aligned))
        metrics["state_update_wrong_sign_count"] = float(np.size(aligned) - int(np.sum(aligned)))
    else:
        metrics["state_update_sign_alignment_fraction"] = float("nan")
        metrics["state_update_wrong_sign_count"] = 0.0
    return metrics


def phase_only_update_metrics(
    initial: TrainState,
    got: TrainState,
) -> dict[str, float | None]:
    got_delta = np.concatenate([(np.asarray(after) - np.asarray(before)).ravel() for before, after in zip(initial, got)])
    return {
        "reference_update_l2": None,
        "phase_update_l2": float(np.linalg.norm(got_delta)),
        "state_update_direction_cosine": None,
        "state_update_sign_alignment_fraction": None,
        "state_update_wrong_sign_count": None,
    }


def empty_reference_metrics() -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    for name in ["local_weights", "local_bias", "readout", "output_bias"]:
        metrics[f"{name}_max_abs_diff"] = None
        metrics[f"{name}_mean_abs_diff"] = None
        metrics[f"{name}_rms_diff"] = None
    metrics.update(
        {
            "state_max_abs_diff": None,
            "state_mean_abs_diff": None,
            "state_rms_diff": None,
            "reference_update_l2": None,
            "phase_update_l2": None,
            "state_update_direction_cosine": None,
            "state_update_sign_alignment_fraction": None,
            "state_update_wrong_sign_count": None,
        }
    )
    return metrics


def probe_diagnostic_rows(
    probe_updates: tuple[int, ...],
    probe_vals: dict[int, np.ndarray],
    initial_state: TrainState,
    op_probe_states: dict[int, TrainState],
) -> tuple[list[dict[str, float | int | None]], dict[int, TrainState]]:
    w, hb, readout, output_bias = initial_state
    rows: list[dict[str, float | int | None]] = []
    phase_states: dict[int, TrainState] = {}
    for update in probe_updates:
        if update not in probe_vals:
            raise ValueError(f"missing probe measurements for update {update}")
        probe_w, probe_hb, probe_readout, probe_ob, _probe_y = unpack_state(
            probe_vals[update],
            w,
            hb,
            readout,
            output_bias,
        )
        phase_state = (probe_w, probe_hb, probe_readout, probe_ob)
        phase_states[update] = phase_state
        row: dict[str, float | int | None] = {"update": update}
        op_state = op_probe_states.get(update)
        if op_state is None:
            row.update(empty_reference_metrics())
            row.update(phase_only_update_metrics(initial_state, phase_state))
        else:
            row.update(state_metrics(op_state, phase_state))
            row.update(update_direction_metrics(initial_state, op_state, phase_state))
        rows.append(row)
    return rows, phase_states


def finite_row_value(row: dict[str, object], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def summarize_probe_rows(probe_rows: list[dict[str, object]]) -> dict[str, int | float | None]:
    if not probe_rows:
        return {
            "probe_count": 0,
            "final_probe_update": None,
            "final_probe_phase_update_l2": None,
            "max_probe_phase_update_l2": None,
            "max_probe_phase_update_l2_update": None,
            "best_probe_phase_eval_accuracy": None,
            "best_probe_phase_eval_update": None,
            "best_probe_phase_eval_improvement": None,
        }

    final_row = max(probe_rows, key=lambda row: int(row["update"]))
    phase_l2_rows = [(row, finite_row_value(row, "phase_update_l2")) for row in probe_rows]
    phase_l2_rows = [(row, value) for row, value in phase_l2_rows if value is not None]
    max_l2_row, max_l2 = max(phase_l2_rows, key=lambda item: item[1]) if phase_l2_rows else (None, None)

    eval_rows = [(row, finite_row_value(row, "phase_eval_accuracy")) for row in probe_rows]
    eval_rows = [(row, value) for row, value in eval_rows if value is not None]
    best_eval_row, best_eval = max(eval_rows, key=lambda item: item[1]) if eval_rows else (None, None)

    return {
        "probe_count": len(probe_rows),
        "final_probe_update": int(final_row["update"]),
        "final_probe_phase_update_l2": finite_row_value(final_row, "phase_update_l2"),
        "max_probe_phase_update_l2": max_l2,
        "max_probe_phase_update_l2_update": int(max_l2_row["update"]) if max_l2_row is not None else None,
        "best_probe_phase_eval_accuracy": best_eval,
        "best_probe_phase_eval_update": int(best_eval_row["update"]) if best_eval_row is not None else None,
        "best_probe_phase_eval_improvement": (
            finite_row_value(best_eval_row, "phase_eval_improvement") if best_eval_row is not None else None
        ),
    }


def simulator_sidecar_paths(netlist_path: Path) -> tuple[Path, ...]:
    return tuple(Path(str(netlist_path) + suffix) for suffix in (".prn", ".mt0", ".ms0", ".ma0"))


def cleanup_simulator_sidecars(netlist_paths: list[Path]) -> int:
    cleaned = 0
    seen: set[Path] = set()
    for netlist_path in netlist_paths:
        for sidecar in simulator_sidecar_paths(netlist_path):
            if sidecar in seen:
                continue
            seen.add(sidecar)
            if sidecar.exists():
                sidecar.unlink()
                cleaned += 1
    return cleaned


def block_tensor_np(x: np.ndarray, blocks: list[list[int]]) -> np.ndarray:
    return np.stack([x[:, idxs] for idxs in blocks], axis=1)


def synapse_transfer_np(weight: np.ndarray, mode: str, clip: float) -> np.ndarray:
    if mode in {"linear", "full", "ideal"}:
        return weight
    clip = max(float(clip), 1e-12)
    if mode in {"tanh-clipped", "smooth-clipped", "clipped"}:
        return clip * np.tanh(weight / clip)
    if mode in {"hard-clipped", "bounded"}:
        return np.clip(weight, -clip, clip)
    if mode in {"sign", "binary"}:
        return clip * weight / (np.abs(weight) + 1e-9)
    raise ValueError(f"unknown synapse transfer mode {mode!r}")


def local_activation_np(x: np.ndarray, mode: str, relu_clip: float, relu_leak: float, softplus_beta: float) -> np.ndarray:
    if mode == "tanh":
        return np.tanh(x)
    if mode == "relu":
        return np.maximum(x, 0.0)
    if mode in {"clipped-relu", "clipped_relu"}:
        return np.clip(x, 0.0, relu_clip)
    if mode in {"diff-clipped-relu", "differential-clipped-relu", "diff_clipped_relu"}:
        return np.clip(x, 0.0, relu_clip) - np.clip(-x, 0.0, relu_clip)
    if mode in {"leaky-relu", "leaky_relu"}:
        return np.where(x >= 0.0, x, relu_leak * x)
    if mode in {"softplus", "softplus-relu", "softplus_relu"}:
        beta = max(float(softplus_beta), 1e-12)
        return np.logaddexp(0.0, beta * x) / beta
    raise ValueError(f"unknown local activation {mode!r}")


def output_activation_np(score: np.ndarray, linear_output: bool, softmax_output: bool) -> np.ndarray:
    if linear_output:
        return score
    if softmax_output:
        shifted = score - np.max(score, axis=1, keepdims=True)
        exp_score = np.exp(shifted)
        return exp_score / np.sum(exp_score, axis=1, keepdims=True)
    return np.tanh(score)


def numpy_eval_accuracy(
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    state: TrainState,
    blocks: list[list[int]],
    batch_size: int,
    *,
    linear_output: bool,
    softmax_output: bool,
    local_activation: str,
    relu_clip: float,
    relu_leak: float,
    softplus_beta: float,
    hidden_synapse_mode: str,
    readout_synapse_mode: str,
    synapse_clip: float,
) -> float:
    correct = 0
    w, hb, readout, output_bias = state
    eff_w = synapse_transfer_np(w, hidden_synapse_mode, synapse_clip)
    eff_readout = synapse_transfer_np(readout, readout_synapse_mode, synapse_clip)
    for start in range(0, len(y_eval), batch_size):
        x = x_eval[start : start + batch_size]
        labels = y_eval[start : start + batch_size]
        xb = block_tensor_np(x, blocks)
        pre = np.einsum("nbp,bcp->nbc", xb, eff_w) + hb
        h = local_activation_np(pre, local_activation, relu_clip, relu_leak, softplus_beta)
        score = np.einsum("nbc,kbc->nk", h, eff_readout) + output_bias
        y = output_activation_np(score, linear_output, softmax_output)
        correct += int(np.sum(np.argmax(y, axis=1) == labels))
    return correct / max(len(y_eval), 1)


def diagnostic_eval_accuracy(
    eval_backend: str,
    spice_bin: str,
    netlist_path: Path,
    data_path: Path,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    state: TrainState,
    blocks: list[list[int]],
    batch_size: int,
    timeout: float,
    *,
    linear_output: bool,
    softmax_output: bool,
    local_activation: str,
    relu_clip: float,
    relu_leak: float,
    softplus_beta: float,
    hidden_synapse_mode: str,
    readout_synapse_mode: str,
    synapse_clip: float,
) -> float:
    if eval_backend == "spice":
        w, hb, readout, output_bias = state
        return run_eval(
            spice_bin,
            netlist_path,
            data_path,
            x_eval,
            y_eval,
            w,
            hb,
            readout,
            output_bias,
            blocks,
            batch_size,
            timeout,
            linear_output=linear_output,
            softmax_output=softmax_output,
            local_activation=local_activation,
            relu_clip=relu_clip,
            relu_leak=relu_leak,
            softplus_beta=softplus_beta,
            hidden_synapse_mode=hidden_synapse_mode,
            readout_synapse_mode=readout_synapse_mode,
            synapse_clip=synapse_clip,
        )
    if eval_backend == "numpy":
        return numpy_eval_accuracy(
            x_eval,
            y_eval,
            state,
            blocks,
            batch_size,
            linear_output=linear_output,
            softmax_output=softmax_output,
            local_activation=local_activation,
            relu_clip=relu_clip,
            relu_leak=relu_leak,
            softplus_beta=softplus_beta,
            hidden_synapse_mode=hidden_synapse_mode,
            readout_synapse_mode=readout_synapse_mode,
            synapse_clip=synapse_clip,
        )
    raise ValueError("eval_backend must be 'spice' or 'numpy'")


def diagnostic_eval_accuracies(
    eval_backend: str,
    spice_bin: str,
    netlist_path: Path,
    data_path: Path,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    state: TrainState,
    blocks: list[list[int]],
    batch_size: int,
    timeout: float,
    **eval_kwargs,
) -> dict[str, float]:
    if eval_backend in {"spice", "numpy"}:
        return {
            eval_backend: diagnostic_eval_accuracy(
                eval_backend,
                spice_bin,
                netlist_path,
                data_path,
                x_eval,
                y_eval,
                state,
                blocks,
                batch_size,
                timeout,
                **eval_kwargs,
            )
        }
    if eval_backend == "both":
        spice_accuracy = diagnostic_eval_accuracy(
            "spice",
            spice_bin,
            netlist_path,
            data_path,
            x_eval,
            y_eval,
            state,
            blocks,
            batch_size,
            timeout,
            **eval_kwargs,
        )
        numpy_accuracy = diagnostic_eval_accuracy(
            "numpy",
            spice_bin,
            netlist_path,
            data_path,
            x_eval,
            y_eval,
            state,
            blocks,
            batch_size,
            timeout,
            **eval_kwargs,
        )
        return {"spice": spice_accuracy, "numpy": numpy_accuracy}
    raise ValueError("eval_backend must be 'spice', 'numpy', or 'both'")


def primary_eval_accuracy(eval_backend: str, accuracies: dict[str, float]) -> float:
    if eval_backend == "numpy":
        return accuracies["numpy"]
    if eval_backend in {"spice", "both"}:
        return accuracies["spice"]
    raise ValueError("eval_backend must be 'spice', 'numpy', or 'both'")


def eval_backend_abs_diff(accuracies: dict[str, float]) -> float | None:
    if "spice" in accuracies and "numpy" in accuracies:
        return abs(accuracies["spice"] - accuracies["numpy"])
    return None


def select_phase_output_mode(
    requested_mode: str,
    spice_bin: str,
    final_measures: bool,
    probe_updates: tuple[int, ...],
) -> str:
    if requested_mode != "auto":
        return requested_mode
    if is_xyce(spice_bin):
        return "measure" if final_measures else "print"
    return "control_measure" if final_measures or probe_updates else "wrdata"


def load_or_init_weights(
    init_weights: str,
    rng: np.random.Generator,
    n_blocks: int,
    channels: int,
    block_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    expected = ((n_blocks, channels, block_len), (n_blocks, channels), (10, n_blocks, channels), (10,))
    if init_weights:
        init = np.load(init_weights)
        if not {"local_weights", "local_bias", "readout", "output_bias"}.issubset(init.files):
            raise ValueError(f"expected local-feature checkpoint keys, got {init.files}")
        w = init["local_weights"].copy()
        hb = init["local_bias"].copy()
        readout = init["readout"].copy()
        output_bias = init["output_bias"].copy()
        actual = (w.shape, hb.shape, readout.shape, output_bias.shape)
        if actual != expected:
            raise ValueError(f"initial weight shapes {actual} do not match expected {expected}")
        return w, hb, readout, output_bias
    w = rng.normal(0.0, 0.05, size=expected[0])
    hb = np.zeros(expected[1])
    readout = rng.normal(0.0, 0.05, size=expected[2])
    output_bias = np.zeros(expected[3])
    return w, hb, readout, output_bias


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-samples", type=int, default=200)
    ap.add_argument("--eval-samples", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=4)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--channels", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--updates", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--linear-output", action="store_true")
    ap.add_argument("--softmax-output", action="store_true")
    ap.add_argument(
        "--local-activation",
        default="tanh",
        choices=[
            "tanh",
            "relu",
            "clipped-relu",
            "clipped_relu",
            "diff-clipped-relu",
            "differential-clipped-relu",
            "diff_clipped_relu",
            "leaky-relu",
            "leaky_relu",
            "softplus",
            "softplus-relu",
            "softplus_relu",
        ],
    )
    ap.add_argument("--relu-clip", type=float, default=1.0)
    ap.add_argument("--relu-leak", type=float, default=0.01)
    ap.add_argument("--softplus-beta", type=float, default=10.0)
    ap.add_argument("--activation-derivative", choices=["exact", "stored-gate", "unity", "floor-exact"], default="exact")
    ap.add_argument("--derivative-floor", type=float, default=0.0)
    ap.add_argument("--derivative-gate-threshold", type=float, default=1e-6)
    ap.add_argument("--readout-feedback-mode", choices=["readout", "full-readout", "exact", "sign-readout", "sign", "clipped-readout", "clipped"], default="readout")
    ap.add_argument("--readout-feedback-clip", type=float, default=0.05)
    synapse_modes = ["linear", "full", "ideal", "tanh-clipped", "smooth-clipped", "clipped", "hard-clipped", "bounded", "sign", "binary"]
    ap.add_argument("--hidden-synapse-mode", choices=synapse_modes, default="linear")
    ap.add_argument("--readout-synapse-mode", choices=synapse_modes, default="linear")
    ap.add_argument("--synapse-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--simulator", default=None)
    ap.add_argument("--init-weights", default="")
    ap.add_argument("--phase", type=float, default=2e-9)
    ap.add_argument("--gap", type=float, default=0.2e-9)
    ap.add_argument("--edge", type=float, default=10e-12)
    ap.add_argument("--settle-ratio", type=float, default=40.0)
    ap.add_argument("--transient-step", type=float, default=20e-12)
    ap.add_argument("--cw", type=float, default=1e-12)
    ap.add_argument("--cstate", type=float, default=1e-12)
    ap.add_argument("--cgrad", type=float, default=1e-12)
    ap.add_argument("--rleak", type=float, default=1e18)
    ap.add_argument("--direction-cosine-threshold", type=float, default=0.999)
    ap.add_argument("--sign-alignment-threshold", type=float, default=0.98)
    ap.add_argument("--eval-accuracy-diff-threshold", type=float, default=0.0)
    ap.add_argument("--random-accuracy-threshold", type=float, default=0.10)
    ap.add_argument("--learning-improvement-threshold", type=float, default=0.02)
    ap.add_argument("--eval-backend", choices=["spice", "numpy", "both"], default="spice")
    ap.add_argument("--reference-mode", choices=["spice", "none"], default="spice")
    ap.add_argument("--phase-output-mode", choices=["auto", "measure", "print", "control_measure", "wrdata"], default="auto")
    ap.add_argument("--update-mode", choices=["phased", "direct"], default="phased")
    ap.add_argument(
        "--strict-fully-on-device",
        action="store_true",
        help=(
            "Fail unless the run is random-init, batch_size=1, and has no Python-side "
            "reference replay; final SPICE/NumPy diagnostics are still allowed after the transient."
        ),
    )
    ap.add_argument("--simulator-extra-args", default="", help=f"Extra simulator command-line arguments, also available via {SPICE_SIMULATOR_ARGS_ENV}.")
    ap.add_argument("--final-measures", action="store_true")
    ap.add_argument(
        "--probe-updates",
        default="",
        help="Comma-separated 1-based update numbers, ranges, 'final', or 'powers2' to measure inside the same transient.",
    )
    ap.add_argument(
        "--eval-probe-updates",
        action="store_true",
        help="After the uninterrupted transient finishes, run diagnostic evals on measured probe states.",
    )
    ap.add_argument("--tag", default="phase_local_feature")
    args = ap.parse_args()

    if args.batch_size <= 0 or args.updates <= 0:
        raise ValueError("--batch-size and --updates must be positive")
    if args.linear_output and args.softmax_output:
        raise ValueError("--linear-output and --softmax-output are mutually exclusive")
    if args.channels <= 0:
        raise ValueError("--channels must be positive")
    if args.relu_clip <= 0:
        raise ValueError("--relu-clip must be positive")
    if args.relu_leak < 0:
        raise ValueError("--relu-leak must be non-negative")
    if args.softplus_beta <= 0:
        raise ValueError("--softplus-beta must be positive")
    if args.derivative_floor < 0 or args.derivative_floor > 1:
        raise ValueError("--derivative-floor must be between 0 and 1")
    if args.derivative_gate_threshold < 0:
        raise ValueError("--derivative-gate-threshold must be non-negative")
    if args.readout_feedback_clip <= 0:
        raise ValueError("--readout-feedback-clip must be positive")
    if args.synapse_clip <= 0:
        raise ValueError("--synapse-clip must be positive")
    if args.phase <= 0 or args.settle_ratio <= 0:
        raise ValueError("--phase and --settle-ratio must be positive")
    if args.direction_cosine_threshold < -1 or args.direction_cosine_threshold > 1:
        raise ValueError("--direction-cosine-threshold must be between -1 and 1")
    if args.sign_alignment_threshold < 0 or args.sign_alignment_threshold > 1:
        raise ValueError("--sign-alignment-threshold must be between 0 and 1")
    if args.eval_accuracy_diff_threshold < 0:
        raise ValueError("--eval-accuracy-diff-threshold must be non-negative")
    if args.random_accuracy_threshold < 0 or args.random_accuracy_threshold > 1:
        raise ValueError("--random-accuracy-threshold must be between 0 and 1")
    if args.learning_improvement_threshold < 0:
        raise ValueError("--learning-improvement-threshold must be non-negative")
    probe_updates = parse_probe_update_list(args.probe_updates, args.updates)
    if args.eval_probe_updates and not probe_updates:
        raise ValueError("--eval-probe-updates requires --probe-updates")
    if args.eval_probe_updates and args.eval_samples <= 0:
        raise ValueError("--eval-probe-updates requires --eval-samples > 0")
    if args.update_mode == "direct" and args.batch_size != 1:
        raise ValueError("--update-mode direct requires --batch-size 1")
    if args.strict_fully_on_device:
        validate_strict_fully_on_device_args(args.batch_size, args.reference_mode, args.init_weights)
    if args.simulator_extra_args:
        os.environ[SPICE_SIMULATOR_ARGS_ENV] = args.simulator_extra_args

    stride = args.block_size if args.stride is None else args.stride
    blocks = block_indices(args.image_size, args.block_size, stride)
    total_samples = args.batch_size * args.updates
    if args.train_samples < total_samples:
        raise ValueError("--train-samples must cover --batch-size * --updates")
    x_train, y_train, x_test, y_test = load_mnist_sequence(args.train_samples, max(1, args.eval_samples), args.image_size, args.seed)
    train_indices, eval_indices = mnist_index_splits(
        args.train_samples,
        max(1, args.eval_samples),
        MNIST_TRAIN_COUNT,
        MNIST_TEST_COUNT,
        args.seed,
    )
    x_batch = x_train[:total_samples]
    y_batch = y_train[:total_samples]
    rng = np.random.default_rng(args.seed)
    w, hb, readout, output_bias = load_or_init_weights(
        args.init_weights,
        rng,
        len(blocks),
        args.channels,
        args.block_size * args.block_size,
    )

    spice_bin, version = detect_spice(args.simulator)
    generated = ROOT / "spice/generated"
    results = ROOT / "spice/results"
    generated.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)

    stem = f"spice_mnist_local_feature_phase_{sanitize_tag(args.tag)}"
    phase_netlist = generated / f"{stem}.cir"
    phase_data = results / f"{stem}.dat"
    op_netlist = generated / f"{stem}_op_reference.cir"
    op_data = results / f"{stem}_op_reference.dat"
    owned_netlists = [phase_netlist]
    if args.reference_mode == "spice":
        owned_netlists.append(op_netlist)
    phase_output_mode = select_phase_output_mode(args.phase_output_mode, spice_bin, args.final_measures, probe_updates)

    netlist, n_vec, t_stop = make_phase_transient_netlist(
        x_batch,
        y_batch,
        w,
        hb,
        readout,
        output_bias,
        blocks,
        args.lr,
        phase_data,
        args.linear_output,
        args.batch_size,
        args.updates,
        args.phase,
        args.gap,
        args.edge,
        args.settle_ratio,
        args.transient_step,
        args.cw,
        args.cstate,
        args.cgrad,
        args.rleak,
        args.softmax_output,
        phase_output_mode,
        probe_updates,
        args.local_activation,
        args.relu_clip,
        args.activation_derivative,
        args.derivative_floor,
        args.derivative_gate_threshold,
        args.readout_feedback_mode,
        args.readout_feedback_clip,
        args.relu_leak,
        args.softplus_beta,
        args.hidden_synapse_mode,
        args.readout_synapse_mode,
        args.synapse_clip,
        args.update_mode,
    )
    phase_netlist.write_text(prepare_phase_netlist(netlist, spice_bin))

    t0 = time.perf_counter()
    proc = run_simulator_netlist(spice_bin, phase_netlist, timeout=args.timeout)
    phase_wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-3000:] or proc.stdout[-3000:])
    if phase_output_mode in {"measure", "control_measure"}:
        measured_text = proc.stdout + "\n" + proc.stderr
        try:
            vals = parse_measured_vector(measured_text, n_vec)
        except ValueError:
            if not is_xyce(spice_bin):
                raise
            vals = read_xyce_print_last_row(xyce_prn_path(phase_netlist), n_vec)
        probe_vals = parse_probe_measurements(measured_text, probe_updates, n_vec) if probe_updates else {}
    elif phase_output_mode == "print":
        xyce_rows = read_xyce_print_rows(xyce_prn_path(phase_netlist), n_vec)
        vals = xyce_rows[-1][1]
        if probe_updates:
            phases, _sample_starts, _t_stop = make_phase_schedule(
                args.batch_size,
                args.updates,
                args.phase,
                args.gap,
                args.update_mode == "direct",
            )
            measure_time = max(0.0, t_stop - args.transient_step)
            probe_times = probe_measure_times(
                phases["acc" if args.update_mode == "direct" else "clear"],
                probe_updates,
                args.gap,
                t_stop,
                measure_time,
            )
            probe_vals = xyce_print_vectors_at_times(xyce_rows, probe_times)
        else:
            probe_vals = {}
    else:
        vals = read_wrdata_row(phase_data, n_vec)
        probe_vals = {}
    phase_w, phase_hb, phase_readout, phase_ob, phase_y = unpack_state(vals, w, hb, readout, output_bias)

    t1 = time.perf_counter()
    op_w = op_hb = op_readout = op_ob = None
    op_probe_states: dict[int, TrainState] = {}
    if args.reference_mode == "spice":
        op_w, op_hb, op_readout, op_ob = w.copy(), hb.copy(), readout.copy(), output_bias.copy()
        for update in range(args.updates):
            start = update * args.batch_size
            stop = start + args.batch_size
            op_w, op_hb, op_readout, op_ob = run_train_batch(
                spice_bin,
                op_netlist,
                op_data,
                x_batch[start:stop],
                y_batch[start:stop],
                op_w,
                op_hb,
                op_readout,
                op_ob,
                blocks,
                args.lr,
                args.timeout,
                linear_output=args.linear_output,
                softmax_output=args.softmax_output,
                local_activation=args.local_activation,
                relu_clip=args.relu_clip,
                activation_derivative=args.activation_derivative,
                derivative_floor=args.derivative_floor,
                derivative_gate_threshold=args.derivative_gate_threshold,
                readout_feedback_mode=args.readout_feedback_mode,
                readout_feedback_clip=args.readout_feedback_clip,
                relu_leak=args.relu_leak,
                softplus_beta=args.softplus_beta,
                hidden_synapse_mode=args.hidden_synapse_mode,
                readout_synapse_mode=args.readout_synapse_mode,
                synapse_clip=args.synapse_clip,
            )
            if update + 1 in probe_updates:
                op_probe_states[update + 1] = (op_w.copy(), op_hb.copy(), op_readout.copy(), op_ob.copy())
    op_wall = time.perf_counter() - t1
    if args.reference_mode == "spice":
        assert op_w is not None and op_hb is not None and op_readout is not None and op_ob is not None
        metrics = state_metrics((op_w, op_hb, op_readout, op_ob), (phase_w, phase_hb, phase_readout, phase_ob))
        metrics.update(
            update_direction_metrics(
                (w, hb, readout, output_bias),
                (op_w, op_hb, op_readout, op_ob),
                (phase_w, phase_hb, phase_readout, phase_ob),
            )
        )
    else:
        metrics = empty_reference_metrics()
        metrics.update(
            phase_only_update_metrics(
                (w, hb, readout, output_bias),
                (phase_w, phase_hb, phase_readout, phase_ob),
            )
        )
    probe_rows, probe_phase_states = probe_diagnostic_rows(
        probe_updates,
        probe_vals,
        (w, hb, readout, output_bias),
        op_probe_states,
    )
    phase_eval_accuracy = None
    op_reference_eval_accuracy = None
    initial_eval_accuracy = None
    spice_initial_eval_accuracy = None
    spice_phase_eval_accuracy = None
    spice_op_reference_eval_accuracy = None
    numpy_initial_eval_accuracy = None
    numpy_phase_eval_accuracy = None
    numpy_op_reference_eval_accuracy = None
    initial_eval_backend_abs_diff = None
    phase_eval_backend_abs_diff = None
    op_reference_eval_backend_abs_diff = None
    eval_wall = 0.0
    if args.eval_samples > 0:
        t2 = time.perf_counter()
        initial_eval_netlist = generated / f"{stem}_initial_eval.cir"
        phase_eval_netlist = generated / f"{stem}_phase_eval.cir"
        if args.eval_backend in {"spice", "both"}:
            owned_netlists.extend([initial_eval_netlist, phase_eval_netlist])
        eval_kwargs = {
            "linear_output": args.linear_output,
            "softmax_output": args.softmax_output,
            "local_activation": args.local_activation,
            "relu_clip": args.relu_clip,
            "relu_leak": args.relu_leak,
            "softplus_beta": args.softplus_beta,
            "hidden_synapse_mode": args.hidden_synapse_mode,
            "readout_synapse_mode": args.readout_synapse_mode,
            "synapse_clip": args.synapse_clip,
        }
        initial_evals = diagnostic_eval_accuracies(
            args.eval_backend,
            spice_bin,
            initial_eval_netlist,
            results / f"{stem}_initial_eval.dat",
            x_test[: args.eval_samples],
            y_test[: args.eval_samples],
            (w, hb, readout, output_bias),
            blocks,
            max(1, min(args.eval_samples, 50)),
            args.timeout,
            **eval_kwargs,
        )
        initial_eval_accuracy = primary_eval_accuracy(args.eval_backend, initial_evals)
        spice_initial_eval_accuracy = initial_evals.get("spice")
        numpy_initial_eval_accuracy = initial_evals.get("numpy")
        initial_eval_backend_abs_diff = eval_backend_abs_diff(initial_evals)
        phase_evals = diagnostic_eval_accuracies(
            args.eval_backend,
            spice_bin,
            phase_eval_netlist,
            results / f"{stem}_phase_eval.dat",
            x_test[: args.eval_samples],
            y_test[: args.eval_samples],
            (phase_w, phase_hb, phase_readout, phase_ob),
            blocks,
            max(1, min(args.eval_samples, 50)),
            args.timeout,
            **eval_kwargs,
        )
        phase_eval_accuracy = primary_eval_accuracy(args.eval_backend, phase_evals)
        spice_phase_eval_accuracy = phase_evals.get("spice")
        numpy_phase_eval_accuracy = phase_evals.get("numpy")
        phase_eval_backend_abs_diff = eval_backend_abs_diff(phase_evals)
        if args.reference_mode == "spice":
            assert op_w is not None and op_hb is not None and op_readout is not None and op_ob is not None
            op_reference_eval_netlist = generated / f"{stem}_op_reference_eval.cir"
            if args.eval_backend in {"spice", "both"}:
                owned_netlists.append(op_reference_eval_netlist)
            op_reference_evals = diagnostic_eval_accuracies(
                args.eval_backend,
                spice_bin,
                op_reference_eval_netlist,
                results / f"{stem}_op_reference_eval.dat",
                x_test[: args.eval_samples],
                y_test[: args.eval_samples],
                (op_w, op_hb, op_readout, op_ob),
                blocks,
                max(1, min(args.eval_samples, 50)),
                args.timeout,
                **eval_kwargs,
            )
            op_reference_eval_accuracy = primary_eval_accuracy(args.eval_backend, op_reference_evals)
            spice_op_reference_eval_accuracy = op_reference_evals.get("spice")
            numpy_op_reference_eval_accuracy = op_reference_evals.get("numpy")
            op_reference_eval_backend_abs_diff = eval_backend_abs_diff(op_reference_evals)
        if args.eval_probe_updates:
            for row in probe_rows:
                update = int(row["update"])
                probe_w, probe_hb, probe_readout, probe_ob = probe_phase_states[update]
                phase_probe_eval_netlist = generated / f"{stem}_probe_{update}_phase_eval.cir"
                if args.eval_backend in {"spice", "both"}:
                    owned_netlists.append(phase_probe_eval_netlist)
                phase_probe_evals = diagnostic_eval_accuracies(
                    args.eval_backend,
                    spice_bin,
                    phase_probe_eval_netlist,
                    results / f"{stem}_probe_{update}_phase_eval.dat",
                    x_test[: args.eval_samples],
                    y_test[: args.eval_samples],
                    (probe_w, probe_hb, probe_readout, probe_ob),
                    blocks,
                    max(1, min(args.eval_samples, 50)),
                    args.timeout,
                    **eval_kwargs,
                )
                phase_probe_eval = primary_eval_accuracy(args.eval_backend, phase_probe_evals)
                row["phase_eval_accuracy"] = phase_probe_eval
                row["phase_spice_eval_accuracy"] = phase_probe_evals.get("spice")
                row["phase_numpy_eval_accuracy"] = phase_probe_evals.get("numpy")
                row["phase_eval_backend_abs_diff"] = eval_backend_abs_diff(phase_probe_evals)
                row["phase_eval_improvement"] = (
                    phase_probe_eval - initial_eval_accuracy
                    if initial_eval_accuracy is not None
                    else None
                )
                op_probe_state = op_probe_states.get(update)
                if op_probe_state is None:
                    row["op_reference_eval_accuracy"] = None
                    row["eval_accuracy_abs_diff"] = None
                else:
                    op_probe_w, op_probe_hb, op_probe_readout, op_probe_ob = op_probe_state
                    op_probe_eval_netlist = generated / f"{stem}_probe_{update}_op_reference_eval.cir"
                    if args.eval_backend in {"spice", "both"}:
                        owned_netlists.append(op_probe_eval_netlist)
                    op_probe_evals = diagnostic_eval_accuracies(
                        args.eval_backend,
                        spice_bin,
                        op_probe_eval_netlist,
                        results / f"{stem}_probe_{update}_op_reference_eval.dat",
                        x_test[: args.eval_samples],
                        y_test[: args.eval_samples],
                        (op_probe_w, op_probe_hb, op_probe_readout, op_probe_ob),
                        blocks,
                        max(1, min(args.eval_samples, 50)),
                        args.timeout,
                        **eval_kwargs,
                    )
                    op_probe_eval = primary_eval_accuracy(args.eval_backend, op_probe_evals)
                    row["op_reference_eval_accuracy"] = op_probe_eval
                    row["op_reference_spice_eval_accuracy"] = op_probe_evals.get("spice")
                    row["op_reference_numpy_eval_accuracy"] = op_probe_evals.get("numpy")
                    row["op_reference_eval_backend_abs_diff"] = eval_backend_abs_diff(op_probe_evals)
                    row["eval_accuracy_abs_diff"] = abs(phase_probe_eval - op_probe_eval)
        eval_wall = time.perf_counter() - t2
    eval_accuracy_abs_diff = (
        abs(phase_eval_accuracy - op_reference_eval_accuracy)
        if phase_eval_accuracy is not None and op_reference_eval_accuracy is not None
        else None
    )
    direction_matches_reference = (
        None
        if args.reference_mode == "none"
        else bool(
            np.isfinite(metrics["state_update_direction_cosine"])
            and metrics["state_update_direction_cosine"] >= args.direction_cosine_threshold
            and np.isfinite(metrics["state_update_sign_alignment_fraction"])
            and metrics["state_update_sign_alignment_fraction"] >= args.sign_alignment_threshold
        )
    )
    eval_matches_reference = (
        None
        if eval_accuracy_abs_diff is None
        else bool(eval_accuracy_abs_diff <= args.eval_accuracy_diff_threshold)
    )
    phase_eval_improvement = (
        phase_eval_accuracy - initial_eval_accuracy
        if phase_eval_accuracy is not None and initial_eval_accuracy is not None
        else None
    )
    nontrivial_learning_met = (
        None
        if phase_eval_accuracy is None or phase_eval_improvement is None
        else bool(
            phase_eval_accuracy > args.random_accuracy_threshold
            and phase_eval_improvement >= args.learning_improvement_threshold
        )
    )
    final_weights_path = results / f"{stem}_final_weights.npz"
    reference_weights_path = results / f"{stem}_op_reference_final_weights.npz"
    metrics_path = results / f"{stem}_equivalence_metrics.csv"
    probe_metrics_path = results / f"{stem}_probe_metrics.csv"
    np.savez_compressed(
        final_weights_path,
        local_weights=phase_w,
        local_bias=phase_hb,
        readout=phase_readout,
        output_bias=phase_ob,
        y=phase_y,
    )
    if args.reference_mode == "spice":
        assert op_w is not None and op_hb is not None and op_readout is not None and op_ob is not None
        np.savez_compressed(
            reference_weights_path,
            local_weights=op_w,
            local_bias=op_hb,
            readout=op_readout,
            output_bias=op_ob,
        )
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    if probe_rows:
        pd.DataFrame(probe_rows).to_csv(probe_metrics_path, index=False)
    simulator_sidecars_cleaned = cleanup_simulator_sidecars(owned_netlists)
    state_descriptions = phase_state_descriptions(args.update_mode)
    execution_contract = phase_execution_contract_fields(
        args.batch_size,
        args.reference_mode,
        args.init_weights,
        args.strict_fully_on_device,
    )
    probe_summary = summarize_probe_rows(probe_rows)
    summary = {
        "simulator": version,
        "simulator_selector": args.simulator,
        "architecture": "phase_resolved_transient_local_feature_readout",
        "status": (
            ("one_batch_equivalence_check" if args.updates == 1 else "multi_update_equivalence_check")
            if args.reference_mode == "spice"
            else "continuous_phase_train_no_reference"
        ),
        "image_size": args.image_size,
        "block_size": args.block_size,
        "stride": stride,
        "blocks": len(blocks),
        "channels": args.channels,
        "classes": 10,
        "mnist_index_order": "stable_permutation_prefix",
        "train_index_metadata": index_prefix_metadata(train_indices),
        "eval_index_metadata": index_prefix_metadata(eval_indices),
        "train_samples": args.train_samples,
        "eval_samples": args.eval_samples,
        "batch_size": args.batch_size,
        "updates": args.updates,
        "probe_updates": list(probe_updates),
        "eval_probe_updates": bool(args.eval_probe_updates),
        "eval_backend": args.eval_backend,
        "total_samples": total_samples,
        "lr": args.lr,
        "linear_output": bool(args.linear_output),
        "softmax_output": bool(args.softmax_output),
        "local_activation": args.local_activation,
        "relu_clip": args.relu_clip,
        "relu_leak": args.relu_leak,
        "softplus_beta": args.softplus_beta,
        "activation_derivative": args.activation_derivative,
        "derivative_floor": args.derivative_floor,
        "derivative_gate_threshold": args.derivative_gate_threshold,
        "readout_feedback_mode": args.readout_feedback_mode,
        "readout_feedback_clip": args.readout_feedback_clip,
        "hidden_synapse_mode": args.hidden_synapse_mode,
        "readout_synapse_mode": args.readout_synapse_mode,
        "synapse_clip": args.synapse_clip,
        "reference_mode": args.reference_mode,
        "phase_output_mode_requested": args.phase_output_mode,
        "update_mode": args.update_mode,
        "simulator_extra_args": args.simulator_extra_args or os.environ.get(SPICE_SIMULATOR_ARGS_ENV, ""),
        "init_weights": args.init_weights,
        "phase_netlist": str(phase_netlist),
        "phase_data": str(phase_data),
        "op_reference_netlist": str(op_netlist) if args.reference_mode == "spice" else None,
        "op_reference_data": str(op_data) if args.reference_mode == "spice" else None,
        "final_weights": str(final_weights_path),
        "op_reference_final_weights": str(reference_weights_path) if args.reference_mode == "spice" else None,
        "equivalence_metrics": str(metrics_path),
        "probe_metrics": str(probe_metrics_path) if probe_rows else None,
        **probe_summary,
        "simulator_sidecars_cleaned": simulator_sidecars_cleaned,
        "phase_wall_time_s": phase_wall,
        "op_reference_wall_time_s": op_wall,
        "eval_wall_time_s": eval_wall,
        "initial_eval_accuracy": initial_eval_accuracy,
        "phase_eval_accuracy": phase_eval_accuracy,
        "op_reference_eval_accuracy": op_reference_eval_accuracy,
        "spice_initial_eval_accuracy": spice_initial_eval_accuracy,
        "spice_phase_eval_accuracy": spice_phase_eval_accuracy,
        "spice_op_reference_eval_accuracy": spice_op_reference_eval_accuracy,
        "numpy_initial_eval_accuracy": numpy_initial_eval_accuracy,
        "numpy_phase_eval_accuracy": numpy_phase_eval_accuracy,
        "numpy_op_reference_eval_accuracy": numpy_op_reference_eval_accuracy,
        "initial_eval_backend_abs_diff": initial_eval_backend_abs_diff,
        "phase_eval_backend_abs_diff": phase_eval_backend_abs_diff,
        "op_reference_eval_backend_abs_diff": op_reference_eval_backend_abs_diff,
        "eval_accuracy_abs_diff": eval_accuracy_abs_diff,
        "phase_eval_improvement": phase_eval_improvement,
        "direction_cosine_threshold": args.direction_cosine_threshold,
        "sign_alignment_threshold": args.sign_alignment_threshold,
        "eval_accuracy_diff_threshold": args.eval_accuracy_diff_threshold,
        "random_accuracy_threshold": args.random_accuracy_threshold,
        "learning_improvement_threshold": args.learning_improvement_threshold,
        "direction_matches_batch_op_reference": direction_matches_reference,
        "eval_accuracy_matches_batch_op_reference": eval_matches_reference,
        "nontrivial_learning_met": nontrivial_learning_met,
        "online_batch_size_one": args.batch_size == 1,
        **execution_contract,
        "continuous_transient_contract_met": (
            None
            if args.reference_mode == "none"
            else bool(args.batch_size == 1 and direction_matches_reference and (eval_matches_reference is not False))
        ),
        "t_stop_s": t_stop,
        "transient_step_s": args.transient_step,
        "phase_s": args.phase,
        "settle_ratio": args.settle_ratio,
        "output_mode": phase_output_mode,
        **state_descriptions,
        "python_role": "Python generates guiding waveforms and compares final state; it does not carry training state during the transient run.",
        "note": (
            "Local-feature phase-transient all-SPICE update run. reference_mode=spice compares against the existing "
            "operating-point SPICE update; reference_mode=none skips that replay for longer final-diagnostic runs."
        ),
        **metrics,
    }
    summary_path = results / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
