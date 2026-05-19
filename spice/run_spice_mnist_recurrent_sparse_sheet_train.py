from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

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
from run_spice_mnist_sparse_random_train import (
    deterministic_phase,
    pulse_residual_update_expr,
    pulse_update_expr,
)
from run_spice_mnist_train import load_mnist_sequence
from run_spice_sweep import ROOT, detect_spice, run_tiny_test


def make_sheet_graph(
    image_size: int,
    hidden: int,
    input_fan_in: int,
    input_radius: int,
    recurrent_fan_in: int,
    recurrent_radius: int,
    shortcut_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = int(np.ceil(np.sqrt(hidden)))
    cols = int(np.ceil(hidden / rows))
    hidden_coords = np.array([(j // cols, j % cols) for j in range(hidden)], dtype=int)
    input_coords = np.array([(i // image_size, i % image_size) for i in range(image_size * image_size)], dtype=int)
    input_fanins = np.zeros((hidden, input_fan_in), dtype=int)
    recurrent_fanins = np.zeros((hidden, recurrent_fan_in), dtype=int)

    for j, (hy, hx) in enumerate(hidden_coords):
        cy = int(round((hy + 0.5) * image_size / rows - 0.5))
        cx = int(round((hx + 0.5) * image_size / cols - 0.5))
        dist = np.abs(input_coords[:, 0] - cy) + np.abs(input_coords[:, 1] - cx)
        local_inputs = np.flatnonzero(dist <= input_radius)
        if len(local_inputs) == 0:
            local_inputs = np.arange(image_size * image_size)
        input_fanins[j] = rng.choice(local_inputs, size=input_fan_in, replace=len(local_inputs) < input_fan_in)

        if recurrent_fan_in == 0:
            continue
        hdist = np.abs(hidden_coords[:, 0] - hy) + np.abs(hidden_coords[:, 1] - hx)
        local_hidden = np.flatnonzero((hdist <= recurrent_radius) & (np.arange(hidden) != j))
        if len(local_hidden) == 0:
            local_hidden = np.flatnonzero(np.arange(hidden) != j)
        chosen = rng.choice(local_hidden, size=recurrent_fan_in, replace=len(local_hidden) < recurrent_fan_in)
        n_short = int(round(shortcut_fraction * recurrent_fan_in))
        if n_short > 0:
            global_hidden = np.flatnonzero(np.arange(hidden) != j)
            chosen[:n_short] = rng.choice(global_hidden, size=n_short, replace=len(global_hidden) < n_short)
            rng.shuffle(chosen)
        recurrent_fanins[j] = chosen
    return hidden_coords, input_fanins, recurrent_fanins


def make_output_edges(n_classes: int, hidden: int, density: float, rng: np.random.Generator) -> list[list[int]]:
    density = min(max(float(density), 1.0 / max(hidden, 1)), 1.0)
    n_edges = max(1, int(round(density * hidden)))
    return [sorted(rng.choice(hidden, size=n_edges, replace=False).tolist()) for _ in range(n_classes)]


def activation(
    lines: list[str],
    name: str,
    summed: str,
    mode: str,
    clip: float,
) -> tuple[str, str]:
    a_node = f"a{name}"
    h_node = f"h{name}"
    a_expr = f"V({a_node})"
    lines.append(f"Ba{name} {a_node} 0 V = {summed}")
    if mode == "relu":
        lines.append(f"Bh{name} {h_node} 0 V = {relu_expr(a_expr)}")
        deriv = relu_deriv_expr(a_expr)
    elif mode in {"clipped-relu", "clipped_relu"}:
        c = max(float(clip), 1e-12)
        lines.append(f"Bh{name} {h_node} 0 V = {clipped_relu_expr(a_expr, c)}")
        deriv = clipped_relu_deriv_expr(a_expr, c)
    elif mode in {"diff-clipped-relu", "diff_clipped_relu"}:
        c = max(float(clip), 1e-12)
        neg_a = f"0-({a_expr})"
        lines.append(f"Bh{name} {h_node} 0 V = {clipped_relu_expr(a_expr, c)}-{clipped_relu_expr(neg_a, c)}")
        deriv = f"({clipped_relu_deriv_expr(a_expr, c)}+{clipped_relu_deriv_expr(neg_a, c)})"
    else:
        lines.append(f"Bh{name} {h_node} 0 V = 2/(1+exp(-2*V({a_node})))-1")
        deriv = f"(1-V({h_node})*V({h_node}))"
    return f"V({h_node})", deriv


def add_update(
    lines: list[str],
    new_node: str,
    old_expr: str,
    avg_expr: str,
    residual_node: str,
    new_residual_node: str,
    gradient_mode: str,
    gradient_bits: int,
    gradient_clip: float,
    lr_expr: str,
    pulse_step: float,
    pulse_max_count: int,
    phase: float,
) -> None:
    if gradient_mode == "pulse-residual":
        update, residual = pulse_residual_update_expr(
            avg_expr, f"V({residual_node})", lr_expr, gradient_clip, pulse_step, pulse_max_count
        )
        lines.append(f"B{new_residual_node} {new_residual_node} 0 V = {residual}")
    else:
        update = pulse_update_expr(
            avg_expr,
            gradient_mode,
            gradient_bits,
            gradient_clip,
            lr_expr,
            pulse_step,
            pulse_max_count,
            phase,
        )
    lines.append(f"B{new_node} {new_node} 0 V = {old_expr} + ({update})")


def make_train_netlist(
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    input_fanins: np.ndarray,
    recurrent_fanins: np.ndarray,
    output_edges: list[list[int]],
    feedback: np.ndarray,
    win: np.ndarray,
    brec: np.ndarray,
    wrec: np.ndarray,
    wout: np.ndarray,
    bout: np.ndarray,
    rin: np.ndarray,
    rbrec: np.ndarray,
    rrec: np.ndarray,
    rout: np.ndarray,
    rbout: np.ndarray,
    ticks: int,
    lr: float,
    out_path: Path,
    activation_mode: str,
    activation_clip: float,
    gradient_mode: str,
    gradient_bits: int,
    gradient_clip: float,
    pulse_step: float,
    pulse_max_count: int,
    self_memory: float,
    local_inhibition: float,
    hidden_error_rule: str,
) -> str:
    batch, n_in = x_batch.shape
    hidden, input_fan_in = input_fanins.shape
    recurrent_fan_in = recurrent_fanins.shape[1]
    n_classes = wout.shape[0]
    use_residual = gradient_mode == "pulse-residual"
    hidden_to_classes = [[k for k, edges in enumerate(output_edges) if j in edges] for j in range(hidden)]
    recurrent_posts = [
        [(m, q) for m in range(hidden) for q in range(recurrent_fan_in) if int(recurrent_fanins[m, q]) == j]
        for j in range(hidden)
    ]
    fixed_posts: list[list[tuple[int, float]]] = [[] for _ in range(hidden)]
    if self_memory != 0.0:
        for j in range(hidden):
            fixed_posts[j].append((j, float(self_memory)))
    if local_inhibition != 0.0 and recurrent_fan_in > 0:
        coeff = -float(local_inhibition) / recurrent_fan_in
        for m in range(hidden):
            for q in range(recurrent_fan_in):
                fixed_posts[int(recurrent_fanins[m, q])].append((m, coeff))

    lines = [
        "* Random sparse recurrent sheet batch operating-point SPICE training.",
        "* Parallel local recurrent ticks, unrolled SPICE BPTT, and programmable weight updates.",
        f".param LR={lr:.12g}",
        f".param BS={batch}",
        "",
    ]
    for s in range(batch):
        for i in range(n_in):
            lines.append(f"Vx{s}_{i} x{s}_{i} 0 DC {float(x_batch[s, i]):.12g}")
        target = np.zeros(n_classes)
        target[int(y_batch[s])] = 1.0
        for k in range(n_classes):
            lines.append(f"Vt{s}_{k} t{s}_{k} 0 DC {target[k]:.12g}")
    lines.append("")
    for j in range(hidden):
        for p in range(input_fan_in):
            lines.append(f"Vwin_{j}_{p} win_{j}_{p} 0 DC {win[j, p]:.12g}")
        for q in range(recurrent_fan_in):
            lines.append(f"Vwrec_{j}_{q} wrec_{j}_{q} 0 DC {wrec[j, q]:.12g}")
        lines.append(f"Vbrec_{j} brec_{j} 0 DC {brec[j]:.12g}")
    for k, edges in enumerate(output_edges):
        for j in edges:
            lines.append(f"Vwout_{k}_{j} wout_{k}_{j} 0 DC {wout[k, j]:.12g}")
        lines.append(f"Vbout_{k} bout_{k} 0 DC {bout[k]:.12g}")
    if hidden_error_rule == "dfa":
        for j in range(hidden):
            for k in range(n_classes):
                lines.append(f"Vfb_{j}_{k} fb_{j}_{k} 0 DC {feedback[j, k]:.12g}")
    if use_residual:
        for j in range(hidden):
            for p in range(input_fan_in):
                lines.append(f"Vrin_{j}_{p} rin_{j}_{p} 0 DC {rin[j, p]:.12g}")
            for q in range(recurrent_fan_in):
                lines.append(f"Vrrec_{j}_{q} rrec_{j}_{q} 0 DC {rrec[j, q]:.12g}")
            lines.append(f"Vrbrec_{j} rbrec_{j} 0 DC {rbrec[j]:.12g}")
        for k, edges in enumerate(output_edges):
            for j in edges:
                lines.append(f"Vrout_{k}_{j} rout_{k}_{j} 0 DC {rout[k, j]:.12g}")
            lines.append(f"Vrbout_{k} rbout_{k} 0 DC {rbout[k]:.12g}")
    lines.append("")

    derivs: dict[tuple[int, int, int], str] = {}
    for s in range(batch):
        for t in range(1, ticks + 1):
            for j in range(hidden):
                terms = [f"V(win_{j}_{p})*V(x{s}_{int(input_fanins[j, p])})" for p in range(input_fan_in)]
                for q in range(recurrent_fan_in):
                    src = int(recurrent_fanins[j, q])
                    prev = "0" if t == 1 else f"V(h{s}_{t - 1}_{src})"
                    terms.append(f"V(wrec_{j}_{q})*({prev})")
                if t > 1 and self_memory != 0.0:
                    terms.append(f"{float(self_memory):.12g}*V(h{s}_{t - 1}_{j})")
                if t > 1 and local_inhibition != 0.0 and recurrent_fan_in > 0:
                    inhib = " + ".join(f"V(h{s}_{t - 1}_{int(recurrent_fanins[j, q])})" for q in range(recurrent_fan_in))
                    terms.append(f"-{float(local_inhibition):.12g}*(({inhib})/{recurrent_fan_in})")
                terms.append(f"V(brec_{j})")
                _h, deriv = activation(lines, f"{s}_{t}_{j}", " + ".join(terms), activation_mode, activation_clip)
                derivs[(s, t, j)] = deriv
        for k, edges in enumerate(output_edges):
            terms = [f"V(wout_{k}_{j})*V(h{s}_{ticks}_{j})" for j in edges] + [f"V(bout_{k})"]
            lines.append(f"Bz{s}_{k} z{s}_{k} 0 V = {' + '.join(terms)}")
        denom = " + ".join(f"exp(V(z{s}_{kk}))" for kk in range(n_classes))
        for k in range(n_classes):
            lines.append(f"By{s}_{k} y{s}_{k} 0 V = exp(V(z{s}_{k}))/({denom})")
            lines.append(f"Bd{s}_{k} d{s}_{k} 0 V = V(t{s}_{k})-V(y{s}_{k})")

        for t in range(ticks, 0, -1):
            for j in range(hidden):
                if hidden_error_rule == "dfa":
                    back_terms = [f"V(d{s}_{k})*V(fb_{j}_{k})" for k in range(n_classes)]
                elif t == ticks:
                    back_terms = [f"V(d{s}_{k})*V(wout_{k}_{j})" for k in hidden_to_classes[j]]
                else:
                    back_terms = [f"V(da{s}_{t + 1}_{m})*V(wrec_{m}_{q})" for m, q in recurrent_posts[j]]
                    back_terms += [f"({coeff:.12g})*V(da{s}_{t + 1}_{m})" for m, coeff in fixed_posts[j]]
                back = " + ".join(back_terms) if back_terms else "0"
                lines.append(f"Bdh{s}_{t}_{j} dh{s}_{t}_{j} 0 V = {back}")
                lines.append(f"Bda{s}_{t}_{j} da{s}_{t}_{j} 0 V = V(dh{s}_{t}_{j})*{derivs[(s, t, j)]}")
    lines.append("")

    for j in range(hidden):
        for p in range(input_fan_in):
            grad = " + ".join(
                f"V(da{s}_{t}_{j})*V(x{s}_{int(input_fanins[j, p])})" for s in range(batch) for t in range(1, ticks + 1)
            )
            add_update(
                lines,
                f"nnwin_{j}_{p}",
                f"V(win_{j}_{p})",
                f"(({grad})/{{BS}})",
                f"rin_{j}_{p}",
                f"nrin_{j}_{p}",
                gradient_mode,
                gradient_bits,
                gradient_clip,
                "{LR}",
                pulse_step,
                pulse_max_count,
                deterministic_phase(10, j, p),
            )
        for q in range(recurrent_fan_in):
            src = int(recurrent_fanins[j, q])
            terms = []
            for s in range(batch):
                for t in range(1, ticks + 1):
                    prev = "0" if t == 1 else f"V(h{s}_{t - 1}_{src})"
                    terms.append(f"V(da{s}_{t}_{j})*({prev})")
            grad = " + ".join(terms) if terms else "0"
            add_update(
                lines,
                f"nnwrec_{j}_{q}",
                f"V(wrec_{j}_{q})",
                f"(({grad})/{{BS}})",
                f"rrec_{j}_{q}",
                f"nrrec_{j}_{q}",
                gradient_mode,
                gradient_bits,
                gradient_clip,
                "{LR}",
                pulse_step,
                pulse_max_count,
                deterministic_phase(11, j, q),
            )
        grad_b = " + ".join(f"V(da{s}_{t}_{j})" for s in range(batch) for t in range(1, ticks + 1))
        add_update(
            lines,
            f"nnbrec_{j}",
            f"V(brec_{j})",
            f"(({grad_b})/{{BS}})",
            f"rbrec_{j}",
            f"nrbrec_{j}",
            gradient_mode,
            gradient_bits,
            gradient_clip,
            "{LR}",
            pulse_step,
            pulse_max_count,
            deterministic_phase(12, j),
        )
    for k, edges in enumerate(output_edges):
        for j in edges:
            grad = " + ".join(f"V(d{s}_{k})*V(h{s}_{ticks}_{j})" for s in range(batch))
            add_update(
                lines,
                f"nnwout_{k}_{j}",
                f"V(wout_{k}_{j})",
                f"(({grad})/{{BS}})",
                f"rout_{k}_{j}",
                f"nrout_{k}_{j}",
                gradient_mode,
                gradient_bits,
                gradient_clip,
                "{LR}",
                pulse_step,
                pulse_max_count,
                deterministic_phase(13, k, j),
            )
        grad_b = " + ".join(f"V(d{s}_{k})" for s in range(batch))
        add_update(
            lines,
            f"nnbout_{k}",
            f"V(bout_{k})",
            f"(({grad_b})/{{BS}})",
            f"rbout_{k}",
            f"nrbout_{k}",
            gradient_mode,
            gradient_bits,
            gradient_clip,
            "{LR}",
            pulse_step,
            pulse_max_count,
            deterministic_phase(14, k),
        )

    vectors = [f"V(nnwin_{j}_{p})" for j in range(hidden) for p in range(input_fan_in)]
    vectors += [f"V(nnbrec_{j})" for j in range(hidden)]
    vectors += [f"V(nnwrec_{j}_{q})" for j in range(hidden) for q in range(recurrent_fan_in)]
    vectors += [f"V(nnwout_{k}_{j})" for k, edges in enumerate(output_edges) for j in edges]
    vectors += [f"V(nnbout_{k})" for k in range(n_classes)]
    if use_residual:
        vectors += [f"V(nrin_{j}_{p})" for j in range(hidden) for p in range(input_fan_in)]
        vectors += [f"V(nrbrec_{j})" for j in range(hidden)]
        vectors += [f"V(nrrec_{j}_{q})" for j in range(hidden) for q in range(recurrent_fan_in)]
        vectors += [f"V(nrout_{k}_{j})" for k, edges in enumerate(output_edges) for j in edges]
        vectors += [f"V(nrbout_{k})" for k in range(n_classes)]
    lines += ["", ".control", "op", f"wrdata {out_path} " + " ".join(vectors), ".endc", ".end", ""]
    return "\n".join(lines)


def make_eval_netlist(
    x_batch: np.ndarray,
    input_fanins: np.ndarray,
    recurrent_fanins: np.ndarray,
    output_edges: list[list[int]],
    win: np.ndarray,
    brec: np.ndarray,
    wrec: np.ndarray,
    wout: np.ndarray,
    bout: np.ndarray,
    ticks: int,
    out_path: Path,
    activation_mode: str,
    activation_clip: float,
    self_memory: float,
    local_inhibition: float,
) -> str:
    batch, n_in = x_batch.shape
    hidden, input_fan_in = input_fanins.shape
    recurrent_fan_in = recurrent_fanins.shape[1]
    n_classes = wout.shape[0]
    lines = ["* Random sparse recurrent sheet batch operating-point SPICE inference.", ""]
    for s in range(batch):
        for i in range(n_in):
            lines.append(f"Vx{s}_{i} x{s}_{i} 0 DC {float(x_batch[s, i]):.12g}")
    lines.append("")
    for j in range(hidden):
        for p in range(input_fan_in):
            lines.append(f"Vwin_{j}_{p} win_{j}_{p} 0 DC {win[j, p]:.12g}")
        for q in range(recurrent_fan_in):
            lines.append(f"Vwrec_{j}_{q} wrec_{j}_{q} 0 DC {wrec[j, q]:.12g}")
        lines.append(f"Vbrec_{j} brec_{j} 0 DC {brec[j]:.12g}")
    for k, edges in enumerate(output_edges):
        for j in edges:
            lines.append(f"Vwout_{k}_{j} wout_{k}_{j} 0 DC {wout[k, j]:.12g}")
        lines.append(f"Vbout_{k} bout_{k} 0 DC {bout[k]:.12g}")
    lines.append("")
    for s in range(batch):
        for t in range(1, ticks + 1):
            for j in range(hidden):
                terms = [f"V(win_{j}_{p})*V(x{s}_{int(input_fanins[j, p])})" for p in range(input_fan_in)]
                for q in range(recurrent_fan_in):
                    src = int(recurrent_fanins[j, q])
                    prev = "0" if t == 1 else f"V(h{s}_{t - 1}_{src})"
                    terms.append(f"V(wrec_{j}_{q})*({prev})")
                if t > 1 and self_memory != 0.0:
                    terms.append(f"{float(self_memory):.12g}*V(h{s}_{t - 1}_{j})")
                if t > 1 and local_inhibition != 0.0 and recurrent_fan_in > 0:
                    inhib = " + ".join(f"V(h{s}_{t - 1}_{int(recurrent_fanins[j, q])})" for q in range(recurrent_fan_in))
                    terms.append(f"-{float(local_inhibition):.12g}*(({inhib})/{recurrent_fan_in})")
                terms.append(f"V(brec_{j})")
                activation(lines, f"{s}_{t}_{j}", " + ".join(terms), activation_mode, activation_clip)
        for k, edges in enumerate(output_edges):
            terms = [f"V(wout_{k}_{j})*V(h{s}_{ticks}_{j})" for j in edges] + [f"V(bout_{k})"]
            lines.append(f"Bz{s}_{k} z{s}_{k} 0 V = {' + '.join(terms)}")
        denom = " + ".join(f"exp(V(z{s}_{kk}))" for kk in range(n_classes))
        for k in range(n_classes):
            lines.append(f"By{s}_{k} y{s}_{k} 0 V = exp(V(z{s}_{k}))/({denom})")
    vectors = [f"V(y{s}_{k})" for s in range(batch) for k in range(n_classes)]
    lines += ["", ".control", "op", f"wrdata {out_path} " + " ".join(vectors), ".endc", ".end", ""]
    return "\n".join(lines)


def parse_train_row(vals, input_fanins, recurrent_fanins, output_edges, win, brec, wrec, wout, bout, rin, rbrec, rrec, rout, rbout, use_residual):
    hidden, input_fan_in = input_fanins.shape
    recurrent_fan_in = recurrent_fanins.shape[1]
    n_classes = wout.shape[0]
    n_out = sum(len(edges) for edges in output_edges)
    offset = 0
    nwin = vals[offset : offset + hidden * input_fan_in].reshape(win.shape)
    offset += hidden * input_fan_in
    nbrec = vals[offset : offset + hidden]
    offset += hidden
    nwrec = vals[offset : offset + hidden * recurrent_fan_in].reshape(wrec.shape)
    offset += hidden * recurrent_fan_in
    nwout = wout.copy()
    flat = vals[offset : offset + n_out]
    offset += n_out
    q = 0
    for k, edges in enumerate(output_edges):
        for j in edges:
            nwout[k, j] = flat[q]
            q += 1
    nbout = vals[offset : offset + n_classes]
    offset += n_classes
    if not use_residual:
        return nwin, nbrec, nwrec, nwout, nbout, rin, rbrec, rrec, rout, rbout

    nrin = vals[offset : offset + hidden * input_fan_in].reshape(rin.shape)
    offset += hidden * input_fan_in
    nrbrec = vals[offset : offset + hidden]
    offset += hidden
    nrrec = vals[offset : offset + hidden * recurrent_fan_in].reshape(rrec.shape)
    offset += hidden * recurrent_fan_in
    nrout = rout.copy()
    flat = vals[offset : offset + n_out]
    offset += n_out
    q = 0
    for k, edges in enumerate(output_edges):
        for j in edges:
            nrout[k, j] = flat[q]
            q += 1
    nrbout = vals[offset : offset + n_classes]
    return nwin, nbrec, nwrec, nwout, nbout, nrin, nrbrec, nrrec, nrout, nrbout


def run_train_batch(
    spice_bin,
    netlist_path,
    data_path,
    x,
    y,
    input_fanins,
    recurrent_fanins,
    output_edges,
    feedback,
    win,
    brec,
    wrec,
    wout,
    bout,
    rin,
    rbrec,
    rrec,
    rout,
    rbout,
    ticks,
    lr,
    timeout,
    activation_mode,
    activation_clip,
    gradient_mode,
    gradient_bits,
    gradient_clip,
    pulse_step,
    pulse_max_count,
    self_memory,
    local_inhibition,
    hidden_error_rule,
):
    netlist_path.write_text(
        make_train_netlist(
            x,
            y,
            input_fanins,
            recurrent_fanins,
            output_edges,
            feedback,
            win,
            brec,
            wrec,
            wout,
            bout,
            rin,
            rbrec,
            rrec,
            rout,
            rbout,
            ticks,
            lr,
            data_path,
            activation_mode,
            activation_clip,
            gradient_mode,
            gradient_bits,
            gradient_clip,
            pulse_step,
            pulse_max_count,
            self_memory,
            local_inhibition,
            hidden_error_rule,
        )
    )
    proc = subprocess.run([spice_bin, "-b", str(netlist_path)], text=True, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-3000:] or proc.stdout[-3000:])
    hidden, input_fan_in = input_fanins.shape
    recurrent_fan_in = recurrent_fanins.shape[1]
    n_out = sum(len(edges) for edges in output_edges)
    n = hidden * input_fan_in + hidden + hidden * recurrent_fan_in + n_out + wout.shape[0]
    if gradient_mode == "pulse-residual":
        n *= 2
    vals = read_wrdata_row(data_path, n)
    return parse_train_row(
        vals,
        input_fanins,
        recurrent_fanins,
        output_edges,
        win,
        brec,
        wrec,
        wout,
        bout,
        rin,
        rbrec,
        rrec,
        rout,
        rbout,
        gradient_mode == "pulse-residual",
    )


def run_eval(
    spice_bin,
    netlist_path,
    data_path,
    x_eval,
    y_eval,
    input_fanins,
    recurrent_fanins,
    output_edges,
    win,
    brec,
    wrec,
    wout,
    bout,
    ticks,
    batch_size,
    timeout,
    activation_mode,
    activation_clip,
    self_memory,
    local_inhibition,
) -> float:
    correct = 0
    n_classes = wout.shape[0]
    for start in range(0, len(y_eval), batch_size):
        x = x_eval[start : start + batch_size]
        y = y_eval[start : start + batch_size]
        netlist_path.write_text(
            make_eval_netlist(
                x,
                input_fanins,
                recurrent_fanins,
                output_edges,
                win,
                brec,
                wrec,
                wout,
                bout,
                ticks,
                data_path,
                activation_mode,
                activation_clip,
                self_memory,
                local_inhibition,
            )
        )
        proc = subprocess.run([spice_bin, "-b", str(netlist_path)], text=True, capture_output=True, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-3000:] or proc.stdout[-3000:])
        vals = read_wrdata_row(data_path, len(y) * n_classes).reshape(len(y), n_classes)
        correct += int((np.argmax(vals, axis=1) == y).sum())
    return correct / max(len(y_eval), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-samples", type=int, default=80)
    ap.add_argument("--test-samples", type=int, default=80)
    ap.add_argument("--image-size", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--input-fan-in", type=int, default=8)
    ap.add_argument("--input-radius", type=int, default=3)
    ap.add_argument("--recurrent-fan-in", type=int, default=4)
    ap.add_argument("--recurrent-radius", type=int, default=1)
    ap.add_argument("--shortcut-fraction", type=float, default=0.1)
    ap.add_argument("--output-density", type=float, default=1.0)
    ap.add_argument("--ticks", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--activation", choices=["tanh", "relu", "clipped-relu", "diff-clipped-relu"], default="diff-clipped-relu")
    ap.add_argument("--activation-clip", type=float, default=1.0)
    ap.add_argument("--self-memory", type=float, default=0.0)
    ap.add_argument("--local-inhibition", type=float, default=0.0)
    ap.add_argument("--hidden-error-rule", choices=["backprop", "dfa"], default="backprop")
    ap.add_argument("--feedback-scale", type=float, default=0.3)
    ap.add_argument(
        "--gradient-mode",
        choices=["analog", "clipped", "quantized", "symmetric-quantized", "pulse-count", "pulse-dithered", "pulse-residual"],
        default="analog",
    )
    ap.add_argument("--gradient-bits", type=int, default=8)
    ap.add_argument("--gradient-clip", type=float, default=1.0)
    ap.add_argument("--pulse-step", type=float, default=0.0)
    ap.add_argument("--pulse-max-count", type=int, default=0)
    ap.add_argument("--input-weight-scale", type=float, default=0.15)
    ap.add_argument("--recurrent-weight-scale", type=float, default=0.05)
    ap.add_argument("--output-weight-scale", type=float, default=0.3)
    ap.add_argument("--hidden-bias-init", type=float, default=0.0)
    ap.add_argument("--init-weights", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--tag", default="recurrent_sparse_sheet")
    args = ap.parse_args()

    x_train, y_train, x_test, y_test = load_mnist_sequence(args.train_samples, args.test_samples, args.image_size, args.seed)
    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    generated.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)
    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag)
    stem = f"spice_mnist_recurrent_sparse_sheet_{safe_tag}"
    netlist_path = generated / f"{stem}_step.cir"
    eval_netlist = generated / f"{stem}_eval.cir"
    data_path = ROOT / f"spice/results/{stem}_step.dat"

    rng = np.random.default_rng(args.seed)
    hidden_coords, input_fanins, recurrent_fanins = make_sheet_graph(
        args.image_size,
        args.hidden,
        args.input_fan_in,
        args.input_radius,
        args.recurrent_fan_in,
        args.recurrent_radius,
        args.shortcut_fraction,
        rng,
    )
    output_edges = make_output_edges(10, args.hidden, args.output_density, rng)
    win = rng.normal(0.0, args.input_weight_scale / max(args.input_fan_in, 1) ** 0.5, size=input_fanins.shape)
    brec = np.full(args.hidden, args.hidden_bias_init)
    wrec = rng.normal(
        0.0,
        args.recurrent_weight_scale / max(args.recurrent_fan_in, 1) ** 0.5,
        size=recurrent_fanins.shape,
    )
    wout = np.zeros((10, args.hidden))
    for k, edges in enumerate(output_edges):
        wout[k, edges] = rng.normal(0.0, args.output_weight_scale / max(len(edges), 1) ** 0.5, size=len(edges))
    bout = np.zeros(10)
    rin = np.zeros_like(win)
    rbrec = np.zeros_like(brec)
    rrec = np.zeros_like(wrec)
    rout = np.zeros_like(wout)
    rbout = np.zeros_like(bout)
    feedback_rng = np.random.default_rng(args.seed + 104729)
    feedback = feedback_rng.normal(0.0, args.feedback_scale / max(10, 1) ** 0.5, size=(args.hidden, 10))

    if args.init_weights:
        checkpoint = np.load(args.init_weights, allow_pickle=True)
        hidden_coords = np.asarray(checkpoint["hidden_coords"], dtype=int)
        input_fanins = np.asarray(checkpoint["input_fanins"], dtype=int)
        recurrent_fanins = np.asarray(checkpoint["recurrent_fanins"], dtype=int)
        output_edges = [np.asarray(edges, dtype=int).tolist() for edges in checkpoint["output_edges"]]
        win = np.asarray(checkpoint["win"], dtype=float)
        brec = np.asarray(checkpoint["brec"], dtype=float)
        wrec = np.asarray(checkpoint["wrec"], dtype=float)
        wout = np.asarray(checkpoint["wout"], dtype=float)
        bout = np.asarray(checkpoint["bout"], dtype=float)
        rin = np.asarray(checkpoint["rin"], dtype=float) if "rin" in checkpoint.files else np.zeros_like(win)
        rbrec = np.asarray(checkpoint["rbrec"], dtype=float) if "rbrec" in checkpoint.files else np.zeros_like(brec)
        rrec = np.asarray(checkpoint["rrec"], dtype=float) if "rrec" in checkpoint.files else np.zeros_like(wrec)
        rout = np.asarray(checkpoint["rout"], dtype=float) if "rout" in checkpoint.files else np.zeros_like(wout)
        rbout = np.asarray(checkpoint["rbout"], dtype=float) if "rbout" in checkpoint.files else np.zeros_like(bout)
        feedback = (
            np.asarray(checkpoint["feedback"], dtype=float)
            if "feedback" in checkpoint.files
            else feedback_rng.normal(0.0, args.feedback_scale / max(10, 1) ** 0.5, size=(win.shape[0], 10))
        )
        if feedback.shape != (win.shape[0], 10):
            raise ValueError(f"checkpoint has inconsistent feedback shape: feedback={feedback.shape}, expected={(win.shape[0], 10)}")

    effective_pulse_max = int(args.pulse_max_count) if args.pulse_max_count > 0 else int((1 << max(1, args.gradient_bits)) - 1)
    weights_path = ROOT / f"spice/results/{stem}_final_weights.npz"
    best_weights_path = ROOT / f"spice/results/{stem}_best_weights.npz"

    def save_weights(path: Path) -> None:
        np.savez_compressed(
            path,
            hidden_coords=hidden_coords,
            input_fanins=input_fanins,
            recurrent_fanins=recurrent_fanins,
            output_edges=np.array(output_edges, dtype=object),
            win=win,
            brec=brec,
            wrec=wrec,
            wout=wout,
            bout=bout,
            rin=rin,
            rbrec=rbrec,
            rrec=rrec,
            rout=rout,
            rbout=rbout,
            feedback=feedback,
        )

    rows = []
    best_acc = -1.0
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        order = np.arange(len(y_train))
        rng.shuffle(order)
        epoch_start = time.perf_counter()
        for n, start in enumerate(range(0, len(order), args.batch_size)):
            idx = order[start : start + args.batch_size]
            win, brec, wrec, wout, bout, rin, rbrec, rrec, rout, rbout = run_train_batch(
                spice_bin,
                netlist_path,
                data_path,
                x_train[idx],
                y_train[idx],
                input_fanins,
                recurrent_fanins,
                output_edges,
                feedback,
                win,
                brec,
                wrec,
                wout,
                bout,
                rin,
                rbrec,
                rrec,
                rout,
                rbout,
                args.ticks,
                args.lr,
                args.timeout,
                args.activation,
                args.activation_clip,
                args.gradient_mode,
                args.gradient_bits,
                args.gradient_clip,
                args.pulse_step,
                effective_pulse_max,
                args.self_memory,
                args.local_inhibition,
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
            input_fanins,
            recurrent_fanins,
            output_edges,
            win,
            brec,
            wrec,
            wout,
            bout,
            args.ticks,
            args.batch_size,
            args.timeout,
            args.activation,
            args.activation_clip,
            args.self_memory,
            args.local_inhibition,
        )
        row = {"epoch": epoch + 1, "heldout_accuracy": heldout, "epoch_wall_time_s": time.perf_counter() - epoch_start}
        rows.append(row)
        if heldout > best_acc:
            best_acc = heldout
            save_weights(best_weights_path)
        print(json.dumps(row), flush=True)

    curve = pd.DataFrame(rows)
    curve_path = ROOT / f"spice/results/{stem}_learning_curve.csv"
    curve.to_csv(curve_path, index=False)
    fig = ROOT / f"spice/results/{stem}_learning_curve.png"
    plot_curve(curve, fig)
    save_weights(weights_path)
    summary = {
        "simulator": version,
        "dataset": "MNIST train/test split, downsampled",
        "architecture": "random_sparse_recurrent_sheet_dfa" if args.hidden_error_rule == "dfa" else "random_sparse_recurrent_sheet_bptt",
        "image_size": args.image_size,
        "inputs": int(x_train.shape[1]),
        "hidden": int(args.hidden),
        "input_fan_in": int(args.input_fan_in),
        "input_radius": int(args.input_radius),
        "recurrent_fan_in": int(args.recurrent_fan_in),
        "recurrent_radius": int(args.recurrent_radius),
        "shortcut_fraction": float(args.shortcut_fraction),
        "output_density": float(args.output_density),
        "output_edges": int(sum(len(edges) for edges in output_edges)),
        "ticks": int(args.ticks),
        "activation": args.activation,
        "activation_clip": args.activation_clip,
        "self_memory": args.self_memory,
        "local_inhibition": args.local_inhibition,
        "hidden_error_rule": args.hidden_error_rule,
        "feedback_scale": args.feedback_scale,
        "gradient_mode": args.gradient_mode,
        "gradient_bits": args.gradient_bits,
        "gradient_clip": args.gradient_clip,
        "pulse_step": args.pulse_step,
        "pulse_max_count": effective_pulse_max,
        "input_weight_scale": args.input_weight_scale,
        "recurrent_weight_scale": args.recurrent_weight_scale,
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
        "best_weights": str(best_weights_path),
        "heldout_test_accuracy": float(curve.iloc[-1]["heldout_accuracy"]) if len(curve) else None,
        "best_heldout_accuracy": float(curve["heldout_accuracy"].max()) if len(curve) else None,
        "note": (
            "Random sparse recurrent sheet all-SPICE trainer: ngspice computes parallel recurrent ticks, "
            "softmax class error, unrolled recurrent backprop or direct feedback-alignment signals, "
            "and programmable weight updates."
        ),
    }
    out = ROOT / f"spice/results/{stem}_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
