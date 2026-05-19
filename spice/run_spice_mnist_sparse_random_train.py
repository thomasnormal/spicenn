from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from run_spice_mnist_batch_op_train import read_wrdata_row
from run_spice_mnist_local_block_batch_op_train import (
    clipped_relu_deriv_expr,
    clipped_relu_expr,
    plot_curve,
    relu_deriv_expr,
    relu_expr,
)
from run_spice_mnist_train import load_mnist_sequence
from run_spice_sweep import ROOT, detect_spice, run_tiny_test


GradientMode = Literal[
    "analog",
    "clipped",
    "quantized",
    "symmetric-quantized",
    "pulse-count",
    "pulse-dithered",
    "pulse-residual",
]


def make_sparse_fanins(
    image_size: int,
    hidden: int,
    fan_in: int,
    radius: int,
    shortcut_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_in = image_size * image_size
    coords = np.array([(i // image_size, i % image_size) for i in range(n_in)])
    centers = rng.integers(0, n_in, size=hidden)
    fanins = np.zeros((hidden, fan_in), dtype=int)
    for j, center in enumerate(centers):
        cy, cx = coords[center]
        if radius < 0:
            local_candidates = np.arange(n_in)
        else:
            dist = np.abs(coords[:, 0] - cy) + np.abs(coords[:, 1] - cx)
            local_candidates = np.flatnonzero(dist <= radius)
        if len(local_candidates) == 0:
            local_candidates = np.arange(n_in)
        chosen = rng.choice(local_candidates, size=fan_in, replace=len(local_candidates) < fan_in)
        n_short = int(round(shortcut_fraction * fan_in))
        if n_short > 0:
            chosen[:n_short] = rng.choice(n_in, size=n_short, replace=n_in < n_short)
            rng.shuffle(chosen)
        fanins[j] = chosen
    return centers, fanins


def make_output_edges(n_classes: int, hidden: int, density: float, rng: np.random.Generator) -> list[list[int]]:
    density = min(max(float(density), 1.0 / max(hidden, 1)), 1.0)
    n_edges = max(1, int(round(density * hidden)))
    return [sorted(rng.choice(hidden, size=n_edges, replace=False).tolist()) for _ in range(n_classes)]


def hidden_activation(
    lines: list[str],
    sample: int,
    hidden_idx: int,
    summed: str,
    mode: str,
    clip: float,
) -> tuple[str, str]:
    a_node = f"a{sample}_{hidden_idx}"
    h_node = f"h{sample}_{hidden_idx}"
    a_expr = f"V({a_node})"
    lines.append(f"Ba{sample}_{hidden_idx} {a_node} 0 V = {summed}")
    if mode == "relu":
        lines.append(f"Bh{sample}_{hidden_idx} {h_node} 0 V = {relu_expr(a_expr)}")
        deriv = relu_deriv_expr(a_expr)
    elif mode in {"clipped-relu", "clipped_relu"}:
        c = max(float(clip), 1e-12)
        lines.append(f"Bh{sample}_{hidden_idx} {h_node} 0 V = {clipped_relu_expr(a_expr, c)}")
        deriv = clipped_relu_deriv_expr(a_expr, c)
    elif mode in {"diff-clipped-relu", "differential-clipped-relu", "diff_clipped_relu"}:
        c = max(float(clip), 1e-12)
        neg_a = f"0-({a_expr})"
        lines.append(f"Bh{sample}_{hidden_idx} {h_node} 0 V = {clipped_relu_expr(a_expr, c)}-{clipped_relu_expr(neg_a, c)}")
        deriv = f"({clipped_relu_deriv_expr(a_expr, c)}+{clipped_relu_deriv_expr(neg_a, c)})"
    else:
        lines.append(f"Bh{sample}_{hidden_idx} {h_node} 0 V = 2/(1+exp(-2*V({a_node})))-1")
        deriv = f"(1-V({h_node})*V({h_node}))"
    return f"V({h_node})", deriv


def clamp_expr(expr: str, limit: float) -> str:
    lim = max(float(limit), 1e-12)
    return f"({clipped_relu_expr(f'({expr})+{lim:.12g}', 2.0 * lim)}-{lim:.12g})"


def gradient_expr(avg_expr: str, mode: GradientMode, bits: int, clip: float) -> str:
    if mode == "analog":
        return avg_expr
    clamped = clamp_expr(avg_expr, clip)
    if mode == "clipped":
        return clamped
    levels = max(2, (1 << max(1, int(bits))) - 1)
    step = 2.0 * max(float(clip), 1e-12) / levels
    return f"({step:.12g}*floor((({clamped})+{clip:.12g})/{step:.12g}+0.5)-{clip:.12g})"


def signed_magnitude_quantized_expr(avg_expr: str, bits: int, clip: float) -> str:
    clamped = clamp_expr(avg_expr, clip)
    max_count = max(1, (1 << max(1, int(bits) - 1)) - 1)
    step = max(float(clip), 1e-12) / max_count
    mag_count = f"floor(abs({clamped})/{step:.12g}+0.5)"
    limited_count = clipped_relu_expr(mag_count, float(max_count))
    sign = f"({clamped})/(abs({clamped})+1e-18)"
    return f"({sign})*({limited_count})*{step:.12g}"


def deterministic_phase(*parts: int) -> float:
    state = 0x9E3779B9
    for part in parts:
        state ^= int(part) + 0x9E3779B9 + ((state << 6) & 0xFFFFFFFF) + (state >> 2)
        state &= 0xFFFFFFFF
    return ((1664525 * state + 1013904223) & 0xFFFFFFFF) / 2**32


def pulse_update_expr(
    avg_expr: str,
    mode: GradientMode,
    bits: int,
    clip: float,
    lr_expr: str,
    pulse_step: float,
    pulse_max_count: int,
    dither_phase: float,
) -> str:
    if mode in {"analog", "clipped", "quantized"}:
        return f"({lr_expr})*({gradient_expr(avg_expr, mode, bits, clip)})"
    if mode == "symmetric-quantized":
        return f"({lr_expr})*({signed_magnitude_quantized_expr(avg_expr, bits, clip)})"
    if mode == "pulse-residual":
        raise ValueError("pulse-residual requires a local residual state node")

    clamped = clamp_expr(avg_expr, clip)
    max_count = max(1, int(pulse_max_count) if pulse_max_count > 0 else (1 << max(1, int(bits))) - 1)
    if pulse_step > 0:
        step_expr = f"{pulse_step:.12g}"
    else:
        step_expr = f"(({lr_expr})*{max(float(clip), 1e-12):.12g}/{max_count})"
    raw_update = f"({lr_expr})*({clamped})"
    round_offset = 0.5 if mode == "pulse-count" else float(dither_phase) % 1.0
    mag_count = f"floor(abs({raw_update})/({step_expr})+{round_offset:.12g})"
    limited_count = clipped_relu_expr(mag_count, float(max_count))
    sign = f"({raw_update})/(abs({raw_update})+1e-18)"
    return f"({sign})*({limited_count})*({step_expr})"


def pulse_residual_update_expr(
    avg_expr: str,
    residual_expr: str,
    lr_expr: str,
    clip: float,
    pulse_step: float,
    pulse_max_count: int,
) -> tuple[str, str]:
    """First-order pulse/residue update: local charge residue plus integer programming pulses."""
    clamped = clamp_expr(avg_expr, clip)
    max_count = max(1, int(pulse_max_count) if pulse_max_count > 0 else 15)
    if pulse_step > 0:
        step_expr = f"{pulse_step:.12g}"
    else:
        step_expr = f"(({lr_expr})*{max(float(clip), 1e-12):.12g}/{max_count})"
    wanted = f"({residual_expr})+({lr_expr})*({clamped})"
    mag_count = f"floor(abs({wanted})/({step_expr}))"
    limited_count = clipped_relu_expr(mag_count, float(max_count))
    sign = f"({wanted})/(abs({wanted})+1e-18)"
    delta = f"({sign})*({limited_count})*({step_expr})"
    new_residual = f"({wanted})-({delta})"
    return delta, new_residual


def make_train_netlist(
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    fanins: np.ndarray,
    output_edges: list[list[int]],
    feedback: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    r1: np.ndarray,
    rb1: np.ndarray,
    r2: np.ndarray,
    rb2: np.ndarray,
    lr: float,
    out_path: Path,
    activation: str,
    activation_clip: float,
    output_mode: str,
    gradient_mode: GradientMode,
    gradient_bits: int,
    gradient_clip: float,
    pulse_step: float,
    pulse_max_count: int,
    hidden_error_rule: str,
) -> str:
    batch, n_in = x_batch.shape
    hidden, fan_in = fanins.shape
    n_classes = w2.shape[0]
    use_residual = gradient_mode == "pulse-residual"
    hidden_to_classes = [[k for k, edges in enumerate(output_edges) if j in edges] for j in range(hidden)]
    lines = [
        "* Random sparse hidden network batch operating-point SPICE training.",
        "* Sparse trainable input fan-in, simple bounded activation, SPICE backprop, and configurable gradient coding.",
        f".param LR={lr:.12g}",
        f".param BS={batch}",
        "",
    ]
    for s in range(batch):
        for i in range(n_in):
            lines.append(f"Vx{s}_{i} x{s}_{i} 0 DC {float(x_batch[s, i]):.12g}")
        if output_mode == "softmax":
            target = np.zeros(n_classes)
            target[int(y_batch[s])] = 1.0
        else:
            target = -np.ones(n_classes)
            target[int(y_batch[s])] = 1.0
        for k in range(n_classes):
            lines.append(f"Vt{s}_{k} t{s}_{k} 0 DC {target[k]:.12g}")
    lines.append("")
    for j in range(hidden):
        for p in range(fan_in):
            lines.append(f"Vw1_{j}_{p} w1_{j}_{p} 0 DC {w1[j, p]:.12g}")
        lines.append(f"Vb1_{j} b1_{j} 0 DC {b1[j]:.12g}")
    for k, edges in enumerate(output_edges):
        for j in edges:
            lines.append(f"Vw2_{k}_{j} w2_{k}_{j} 0 DC {w2[k, j]:.12g}")
        lines.append(f"Vb2_{k} b2_{k} 0 DC {b2[k]:.12g}")
    if hidden_error_rule == "dfa":
        for j in range(hidden):
            for k in range(n_classes):
                lines.append(f"Vfb_{j}_{k} fb_{j}_{k} 0 DC {feedback[j, k]:.12g}")
    if use_residual:
        for j in range(hidden):
            for p in range(fan_in):
                lines.append(f"Vr1_{j}_{p} r1_{j}_{p} 0 DC {r1[j, p]:.12g}")
            lines.append(f"Vrb1_{j} rb1_{j} 0 DC {rb1[j]:.12g}")
        for k, edges in enumerate(output_edges):
            for j in edges:
                lines.append(f"Vr2_{k}_{j} r2_{k}_{j} 0 DC {r2[k, j]:.12g}")
            lines.append(f"Vrb2_{k} rb2_{k} 0 DC {rb2[k]:.12g}")
    lines.append("")
    h_derivs: dict[tuple[int, int], str] = {}
    for s in range(batch):
        for j in range(hidden):
            terms = [f"V(w1_{j}_{p})*V(x{s}_{int(fanins[j, p])})" for p in range(fan_in)]
            terms.append(f"V(b1_{j})")
            _h_expr, deriv = hidden_activation(lines, s, j, " + ".join(terms), activation, activation_clip)
            h_derivs[(s, j)] = deriv
        for k, edges in enumerate(output_edges):
            terms = [f"V(w2_{k}_{j})*V(h{s}_{j})" for j in edges] + [f"V(b2_{k})"]
            lines.append(f"Bz{s}_{k} z{s}_{k} 0 V = {' + '.join(terms)}")
        if output_mode == "softmax":
            denom = " + ".join(f"exp(V(z{s}_{kk}))" for kk in range(n_classes))
            for k in range(n_classes):
                lines.append(f"By{s}_{k} y{s}_{k} 0 V = exp(V(z{s}_{k}))/({denom})")
                lines.append(f"Bd{s}_{k} d{s}_{k} 0 V = V(t{s}_{k})-V(y{s}_{k})")
        else:
            for k in range(n_classes):
                lines.append(f"By{s}_{k} y{s}_{k} 0 V = 2/(1+exp(-2*V(z{s}_{k})))-1")
                lines.append(f"Be{s}_{k} e{s}_{k} 0 V = V(t{s}_{k})-V(y{s}_{k})")
                lines.append(f"Bd{s}_{k} d{s}_{k} 0 V = V(e{s}_{k})*(1-V(y{s}_{k})*V(y{s}_{k}))")
        for j in range(hidden):
            if hidden_error_rule == "dfa":
                back_terms = [f"V(d{s}_{k})*V(fb_{j}_{k})" for k in range(n_classes)]
            else:
                back_terms = [f"V(d{s}_{k})*V(w2_{k}_{j})" for k in hidden_to_classes[j]]
            back = " + ".join(back_terms) if back_terms else "0"
            lines.append(f"Bdh{s}_{j} dh{s}_{j} 0 V = ({back})*{h_derivs[(s, j)]}")
    lines.append("")
    for j in range(hidden):
        for p in range(fan_in):
            grad_sum = " + ".join(f"V(dh{s}_{j})*V(x{s}_{int(fanins[j, p])})" for s in range(batch))
            avg = f"(({grad_sum})/{{BS}})"
            if use_residual:
                update, residual = pulse_residual_update_expr(
                    avg, f"V(r1_{j}_{p})", "{LR}", gradient_clip, pulse_step, pulse_max_count
                )
                lines.append(f"Bnr1_{j}_{p} nr1_{j}_{p} 0 V = {residual}")
            else:
                update = pulse_update_expr(
                    avg,
                    gradient_mode,
                    gradient_bits,
                    gradient_clip,
                    "{LR}",
                    pulse_step,
                    pulse_max_count,
                    deterministic_phase(1, j, p),
                )
            lines.append(f"Bnw1_{j}_{p} nw1_{j}_{p} 0 V = V(w1_{j}_{p}) + ({update})")
        grad_b1 = " + ".join(f"V(dh{s}_{j})" for s in range(batch))
        avg = f"(({grad_b1})/{{BS}})"
        if use_residual:
            update, residual = pulse_residual_update_expr(
                avg, f"V(rb1_{j})", "{LR}", gradient_clip, pulse_step, pulse_max_count
            )
            lines.append(f"Bnrb1_{j} nrb1_{j} 0 V = {residual}")
        else:
            update = pulse_update_expr(
                avg,
                gradient_mode,
                gradient_bits,
                gradient_clip,
                "{LR}",
                pulse_step,
                pulse_max_count,
                deterministic_phase(2, j),
            )
        lines.append(f"Bnb1_{j} nb1_{j} 0 V = V(b1_{j}) + ({update})")
    for k, edges in enumerate(output_edges):
        for j in edges:
            grad_sum = " + ".join(f"V(d{s}_{k})*V(h{s}_{j})" for s in range(batch))
            avg = f"(({grad_sum})/{{BS}})"
            if use_residual:
                update, residual = pulse_residual_update_expr(
                    avg, f"V(r2_{k}_{j})", "{LR}", gradient_clip, pulse_step, pulse_max_count
                )
                lines.append(f"Bnr2_{k}_{j} nr2_{k}_{j} 0 V = {residual}")
            else:
                update = pulse_update_expr(
                    avg,
                    gradient_mode,
                    gradient_bits,
                    gradient_clip,
                    "{LR}",
                    pulse_step,
                    pulse_max_count,
                    deterministic_phase(3, k, j),
                )
            lines.append(f"Bnw2_{k}_{j} nw2_{k}_{j} 0 V = V(w2_{k}_{j}) + ({update})")
        grad_b2 = " + ".join(f"V(d{s}_{k})" for s in range(batch))
        avg = f"(({grad_b2})/{{BS}})"
        if use_residual:
            update, residual = pulse_residual_update_expr(
                avg, f"V(rb2_{k})", "{LR}", gradient_clip, pulse_step, pulse_max_count
            )
            lines.append(f"Bnrb2_{k} nrb2_{k} 0 V = {residual}")
        else:
            update = pulse_update_expr(
                avg,
                gradient_mode,
                gradient_bits,
                gradient_clip,
                "{LR}",
                pulse_step,
                pulse_max_count,
                deterministic_phase(4, k),
            )
        lines.append(f"Bnb2_{k} nb2_{k} 0 V = V(b2_{k}) + ({update})")
    vectors = [f"V(nw1_{j}_{p})" for j in range(hidden) for p in range(fan_in)]
    vectors += [f"V(nb1_{j})" for j in range(hidden)]
    vectors += [f"V(nw2_{k}_{j})" for k, edges in enumerate(output_edges) for j in edges]
    vectors += [f"V(nb2_{k})" for k in range(n_classes)]
    if use_residual:
        vectors += [f"V(nr1_{j}_{p})" for j in range(hidden) for p in range(fan_in)]
        vectors += [f"V(nrb1_{j})" for j in range(hidden)]
        vectors += [f"V(nr2_{k}_{j})" for k, edges in enumerate(output_edges) for j in edges]
        vectors += [f"V(nrb2_{k})" for k in range(n_classes)]
    lines += ["", ".control", "op", f"wrdata {out_path} " + " ".join(vectors), ".endc", ".end", ""]
    return "\n".join(lines)


def make_eval_netlist(
    x_batch: np.ndarray,
    fanins: np.ndarray,
    output_edges: list[list[int]],
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    out_path: Path,
    activation: str,
    activation_clip: float,
    output_mode: str,
) -> str:
    batch, n_in = x_batch.shape
    hidden, fan_in = fanins.shape
    n_classes = w2.shape[0]
    lines = ["* Random sparse hidden network batch operating-point SPICE inference.", ""]
    for s in range(batch):
        for i in range(n_in):
            lines.append(f"Vx{s}_{i} x{s}_{i} 0 DC {float(x_batch[s, i]):.12g}")
    lines.append("")
    for j in range(hidden):
        for p in range(fan_in):
            lines.append(f"Vw1_{j}_{p} w1_{j}_{p} 0 DC {w1[j, p]:.12g}")
        lines.append(f"Vb1_{j} b1_{j} 0 DC {b1[j]:.12g}")
    for k, edges in enumerate(output_edges):
        for j in edges:
            lines.append(f"Vw2_{k}_{j} w2_{k}_{j} 0 DC {w2[k, j]:.12g}")
        lines.append(f"Vb2_{k} b2_{k} 0 DC {b2[k]:.12g}")
    lines.append("")
    for s in range(batch):
        for j in range(hidden):
            terms = [f"V(w1_{j}_{p})*V(x{s}_{int(fanins[j, p])})" for p in range(fan_in)]
            terms.append(f"V(b1_{j})")
            hidden_activation(lines, s, j, " + ".join(terms), activation, activation_clip)
        for k, edges in enumerate(output_edges):
            terms = [f"V(w2_{k}_{j})*V(h{s}_{j})" for j in edges] + [f"V(b2_{k})"]
            lines.append(f"Bz{s}_{k} z{s}_{k} 0 V = {' + '.join(terms)}")
        if output_mode == "softmax":
            denom = " + ".join(f"exp(V(z{s}_{kk}))" for kk in range(n_classes))
            for k in range(n_classes):
                lines.append(f"By{s}_{k} y{s}_{k} 0 V = exp(V(z{s}_{k}))/({denom})")
        else:
            for k in range(n_classes):
                lines.append(f"By{s}_{k} y{s}_{k} 0 V = 2/(1+exp(-2*V(z{s}_{k})))-1")
    vectors = [f"V(y{s}_{k})" for s in range(batch) for k in range(n_classes)]
    lines += ["", ".control", "op", f"wrdata {out_path} " + " ".join(vectors), ".endc", ".end", ""]
    return "\n".join(lines)


def run_train_batch(
    spice_bin,
    netlist_path,
    data_path,
    x,
    y,
    fanins,
    output_edges,
    feedback,
    w1,
    b1,
    w2,
    b2,
    r1,
    rb1,
    r2,
    rb2,
    lr,
    timeout,
    activation,
    activation_clip,
    output_mode,
    gradient_mode,
    gradient_bits,
    gradient_clip,
    pulse_step,
    pulse_max_count,
    hidden_error_rule,
):
    netlist_path.write_text(
        make_train_netlist(
            x,
            y,
            fanins,
            output_edges,
            feedback,
            w1,
            b1,
            w2,
            b2,
            r1,
            rb1,
            r2,
            rb2,
            lr,
            data_path,
            activation,
            activation_clip,
            output_mode,
            gradient_mode,
            gradient_bits,
            gradient_clip,
            pulse_step,
            pulse_max_count,
            hidden_error_rule,
        )
    )
    proc = subprocess.run([spice_bin, "-b", str(netlist_path)], text=True, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-3000:] or proc.stdout[-3000:])
    hidden, fan_in = fanins.shape
    n_classes = w2.shape[0]
    n_w2 = sum(len(edges) for edges in output_edges)
    use_residual = gradient_mode == "pulse-residual"
    n = hidden * fan_in + hidden + n_w2 + n_classes
    if use_residual:
        n *= 2
    vals = read_wrdata_row(data_path, n)
    offset = 0
    nw1 = vals[offset : offset + hidden * fan_in].reshape(w1.shape)
    offset += hidden * fan_in
    nb1 = vals[offset : offset + hidden]
    offset += hidden
    nw2 = w2.copy()
    flat_w2 = vals[offset : offset + n_w2]
    offset += n_w2
    q = 0
    for k, edges in enumerate(output_edges):
        for j in edges:
            nw2[k, j] = flat_w2[q]
            q += 1
    nb2 = vals[offset : offset + n_classes]
    offset += n_classes
    nr1 = r1.copy()
    nrb1 = rb1.copy()
    nr2 = r2.copy()
    nrb2 = rb2.copy()
    if use_residual:
        nr1 = vals[offset : offset + hidden * fan_in].reshape(r1.shape)
        offset += hidden * fan_in
        nrb1 = vals[offset : offset + hidden]
        offset += hidden
        flat_r2 = vals[offset : offset + n_w2]
        offset += n_w2
        q = 0
        for k, edges in enumerate(output_edges):
            for j in edges:
                nr2[k, j] = flat_r2[q]
                q += 1
        nrb2 = vals[offset : offset + n_classes]
    return nw1, nb1, nw2, nb2, nr1, nrb1, nr2, nrb2


def run_eval(
    spice_bin,
    netlist_path,
    data_path,
    x_eval,
    y_eval,
    fanins,
    output_edges,
    w1,
    b1,
    w2,
    b2,
    batch_size,
    timeout,
    activation,
    activation_clip,
    output_mode,
) -> float:
    correct = 0
    n_classes = w2.shape[0]
    for start in range(0, len(y_eval), batch_size):
        x = x_eval[start : start + batch_size]
        y = y_eval[start : start + batch_size]
        netlist_path.write_text(
            make_eval_netlist(x, fanins, output_edges, w1, b1, w2, b2, data_path, activation, activation_clip, output_mode)
        )
        proc = subprocess.run([spice_bin, "-b", str(netlist_path)], text=True, capture_output=True, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-3000:] or proc.stdout[-3000:])
        vals = read_wrdata_row(data_path, len(y) * n_classes).reshape(len(y), n_classes)
        correct += int((np.argmax(vals, axis=1) == y).sum())
    return correct / max(len(y_eval), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-samples", type=int, default=200)
    ap.add_argument("--test-samples", type=int, default=200)
    ap.add_argument("--image-size", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--fan-in", type=int, default=12)
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--shortcut-fraction", type=float, default=0.1)
    ap.add_argument("--output-density", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--activation", choices=["tanh", "relu", "clipped-relu", "diff-clipped-relu"], default="diff-clipped-relu")
    ap.add_argument("--activation-clip", type=float, default=1.0)
    ap.add_argument("--output-mode", choices=["softmax", "tanh"], default="softmax")
    ap.add_argument("--hidden-error-rule", choices=["backprop", "dfa"], default="backprop")
    ap.add_argument("--feedback-scale", type=float, default=0.3)
    ap.add_argument(
        "--gradient-mode",
        choices=[
            "analog",
            "clipped",
            "quantized",
            "symmetric-quantized",
            "pulse-count",
            "pulse-dithered",
            "pulse-residual",
        ],
        default="analog",
    )
    ap.add_argument("--gradient-bits", type=int, default=8)
    ap.add_argument("--gradient-clip", type=float, default=1.0)
    ap.add_argument("--pulse-step", type=float, default=0.0)
    ap.add_argument("--pulse-max-count", type=int, default=0)
    ap.add_argument("--input-weight-scale", type=float, default=0.05)
    ap.add_argument("--output-weight-scale", type=float, default=0.05)
    ap.add_argument("--hidden-bias-init", type=float, default=0.0)
    ap.add_argument("--init-weights", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--tag", default="sparse_random")
    args = ap.parse_args()

    x_train, y_train, x_test, y_test = load_mnist_sequence(args.train_samples, args.test_samples, args.image_size, args.seed)
    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    generated.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)
    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag)
    stem = f"spice_mnist_sparse_random_{safe_tag}"
    netlist_path = generated / f"{stem}_step.cir"
    eval_netlist = generated / f"{stem}_eval.cir"
    data_path = ROOT / f"spice/results/{stem}_step.dat"

    rng = np.random.default_rng(args.seed)
    centers, fanins = make_sparse_fanins(
        args.image_size, args.hidden, args.fan_in, args.radius, args.shortcut_fraction, rng
    )
    output_edges = make_output_edges(10, args.hidden, args.output_density, rng)
    w1 = rng.normal(0.0, args.input_weight_scale / max(args.fan_in, 1) ** 0.5, size=(args.hidden, args.fan_in))
    b1 = np.full(args.hidden, args.hidden_bias_init)
    w2 = np.zeros((10, args.hidden))
    for k, edges in enumerate(output_edges):
        w2[k, edges] = rng.normal(0.0, args.output_weight_scale / max(len(edges), 1) ** 0.5, size=len(edges))
    b2 = np.zeros(10)
    r1 = np.zeros_like(w1)
    rb1 = np.zeros_like(b1)
    r2 = np.zeros_like(w2)
    rb2 = np.zeros_like(b2)
    feedback_rng = np.random.default_rng(args.seed + 104729)
    feedback = feedback_rng.normal(0.0, args.feedback_scale / max(10, 1) ** 0.5, size=(args.hidden, 10))
    if args.init_weights:
        checkpoint = np.load(args.init_weights, allow_pickle=True)
        centers = np.asarray(checkpoint["centers"], dtype=int)
        fanins = np.asarray(checkpoint["fanins"], dtype=int)
        output_edges = [np.asarray(edges, dtype=int).tolist() for edges in checkpoint["output_edges"]]
        w1 = np.asarray(checkpoint["w1"], dtype=float)
        b1 = np.asarray(checkpoint["b1"], dtype=float)
        w2 = np.asarray(checkpoint["w2"], dtype=float)
        b2 = np.asarray(checkpoint["b2"], dtype=float)
        r1 = np.asarray(checkpoint["r1"], dtype=float) if "r1" in checkpoint.files else np.zeros_like(w1)
        rb1 = np.asarray(checkpoint["rb1"], dtype=float) if "rb1" in checkpoint.files else np.zeros_like(b1)
        r2 = np.asarray(checkpoint["r2"], dtype=float) if "r2" in checkpoint.files else np.zeros_like(w2)
        rb2 = np.asarray(checkpoint["rb2"], dtype=float) if "rb2" in checkpoint.files else np.zeros_like(b2)
        feedback = (
            np.asarray(checkpoint["feedback"], dtype=float)
            if "feedback" in checkpoint.files
            else feedback_rng.normal(0.0, args.feedback_scale / max(10, 1) ** 0.5, size=(w1.shape[0], 10))
        )
        if w1.shape != fanins.shape or b1.shape != (fanins.shape[0],) or w2.shape[0] != 10 or b2.shape != (10,):
            raise ValueError(f"checkpoint has inconsistent shapes: w1={w1.shape}, fanins={fanins.shape}, w2={w2.shape}")
        if r1.shape != w1.shape or rb1.shape != b1.shape or r2.shape != w2.shape or rb2.shape != b2.shape:
            raise ValueError(f"checkpoint has inconsistent residual shapes: r1={r1.shape}, rb1={rb1.shape}, r2={r2.shape}, rb2={rb2.shape}")
        if feedback.shape != (w1.shape[0], 10):
            raise ValueError(f"checkpoint has inconsistent feedback shape: feedback={feedback.shape}, expected={(w1.shape[0], 10)}")

    rows = []
    best_acc = -1.0
    best_state = None
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        order = np.arange(len(y_train))
        rng.shuffle(order)
        epoch_start = time.perf_counter()
        for n, start in enumerate(range(0, len(order), args.batch_size)):
            idx = order[start : start + args.batch_size]
            w1, b1, w2, b2, r1, rb1, r2, rb2 = run_train_batch(
                spice_bin,
                netlist_path,
                data_path,
                x_train[idx],
                y_train[idx],
                fanins,
                output_edges,
                feedback,
                w1,
                b1,
                w2,
                b2,
                r1,
                rb1,
                r2,
                rb2,
                args.lr,
                args.timeout,
                args.activation,
                args.activation_clip,
                args.output_mode,
                args.gradient_mode,
                args.gradient_bits,
                args.gradient_clip,
                args.pulse_step,
                args.pulse_max_count,
                args.hidden_error_rule,
            )
            if (n + 1) % 5 == 0:
                print(f"epoch {epoch + 1} batch {n + 1}", flush=True)
        heldout = run_eval(
            spice_bin,
            eval_netlist,
            data_path,
            x_test,
            y_test,
            fanins,
            output_edges,
            w1,
            b1,
            w2,
            b2,
            args.batch_size,
            args.timeout,
            args.activation,
            args.activation_clip,
            args.output_mode,
        )
        row = {"epoch": epoch + 1, "heldout_accuracy": heldout, "epoch_wall_time_s": time.perf_counter() - epoch_start}
        rows.append(row)
        if heldout > best_acc:
            best_acc = float(heldout)
            best_state = (
                w1.copy(),
                b1.copy(),
                w2.copy(),
                b2.copy(),
                r1.copy(),
                rb1.copy(),
                r2.copy(),
                rb2.copy(),
            )
        print(json.dumps(row), flush=True)

    curve = pd.DataFrame(rows)
    curve_path = ROOT / f"spice/results/{stem}_learning_curve.csv"
    curve.to_csv(curve_path, index=False)
    fig = ROOT / f"spice/results/{stem}_learning_curve.png"
    plot_curve(curve, fig)
    weights_path = ROOT / f"spice/results/{stem}_final_weights.npz"
    np.savez_compressed(
        weights_path,
        centers=centers,
        fanins=fanins,
        output_edges=np.array(output_edges, dtype=object),
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        r1=r1,
        rb1=rb1,
        r2=r2,
        rb2=rb2,
        feedback=feedback,
    )
    best_weights_path = None
    if best_state is not None:
        bw1, bb1, bw2, bb2, br1, brb1, br2, brb2 = best_state
        best_weights_path = ROOT / f"spice/results/{stem}_best_weights.npz"
        np.savez_compressed(
            best_weights_path,
            centers=centers,
            fanins=fanins,
            output_edges=np.array(output_edges, dtype=object),
            w1=bw1,
            b1=bb1,
            w2=bw2,
            b2=bb2,
            r1=br1,
            rb1=brb1,
            r2=br2,
            rb2=brb2,
            feedback=feedback,
        )
    summary = {
        "simulator": version,
        "dataset": "MNIST train/test split, downsampled",
        "architecture": "random_sparse_hidden_dfa" if args.hidden_error_rule == "dfa" else "random_sparse_hidden_backprop",
        "local": bool(args.radius >= 0),
        "image_size": args.image_size,
        "inputs": int(x_train.shape[1]),
        "hidden": int(fanins.shape[0]),
        "fan_in": int(fanins.shape[1]),
        "radius": args.radius,
        "shortcut_fraction": args.shortcut_fraction,
        "output_density": args.output_density,
        "output_edges": int(sum(len(edges) for edges in output_edges)),
        "classes": 10,
        "activation": args.activation,
        "activation_clip": args.activation_clip,
        "output_mode": args.output_mode,
        "hidden_error_rule": args.hidden_error_rule,
        "feedback_scale": args.feedback_scale,
        "gradient_mode": args.gradient_mode,
        "gradient_bits": args.gradient_bits,
        "gradient_clip": args.gradient_clip,
        "pulse_step": args.pulse_step,
        "pulse_max_count": int(args.pulse_max_count) if args.pulse_max_count > 0 else int((1 << max(1, args.gradient_bits)) - 1),
        "input_weight_scale": args.input_weight_scale,
        "output_weight_scale": args.output_weight_scale,
        "hidden_bias_init": args.hidden_bias_init,
        "init_weights": args.init_weights,
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "wall_time_s": time.perf_counter() - t0,
        "learning_curve": str(curve_path),
        "figure": str(fig),
        "final_weights": str(weights_path),
        "best_weights": str(best_weights_path) if best_weights_path else None,
        "heldout_test_accuracy": float(curve.iloc[-1]["heldout_accuracy"]) if len(curve) else None,
        "best_heldout_accuracy": float(curve["heldout_accuracy"].max()) if len(curve) else None,
        "note": (
            "Random sparse hidden all-SPICE trainer: ngspice computes sparse forward pass, "
            "softmax/tanh class errors, hidden backprop or direct feedback-alignment signals, and programmable weight updates. "
            "Gradient precision can be analog, clipped analog, legacy uniform quantized, symmetric signed-magnitude "
            "quantized, deterministic pulse-count, dithered pulse-count, or local residual pulse accumulation."
        ),
    }
    out = ROOT / f"spice/results/{stem}_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
