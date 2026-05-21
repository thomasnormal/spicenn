from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_spice_mnist_batch_op_train import read_wrdata_row
from run_spice_mnist_train import load_mnist_sequence
from run_spice_sweep import ROOT, detect_spice, prepare_netlist_for_simulator, run_tiny_test, run_simulator_netlist


def sample_noise(rng: np.random.Generator, sigma: float, shape: tuple[int, ...]) -> Optional[np.ndarray]:
    if sigma <= 0:
        return None
    return rng.normal(0.0, sigma, size=shape)


def noise_value(noise: Optional[np.ndarray], *idx: int) -> float:
    if noise is None:
        return 0.0
    return float(noise[idx])


def shifted(expr: str, delta: float) -> str:
    if abs(delta) < 1e-15:
        return expr
    return f"({expr} {delta:+.12g})"


def parse_float_list(text: str) -> list[float]:
    vals = []
    for part in text.split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    return vals


def relu_expr(expr: str) -> str:
    return f"0.5*(({expr})+abs({expr}))"


def relu_deriv_expr(expr: str) -> str:
    return f"0.5*(1+({expr})/(abs({expr})+1e-9))"


def clipped_relu_expr(expr: str, clip: float) -> str:
    shifted_expr = f"({expr})-{clip:.12g}"
    return f"({relu_expr(expr)}-{relu_expr(shifted_expr)})"


def clipped_relu_deriv_expr(expr: str, clip: float) -> str:
    shifted_expr = f"({expr})-{clip:.12g}"
    return f"({relu_deriv_expr(expr)}-{relu_deriv_expr(shifted_expr)})"


def add_local_activation(
    lines: list[str],
    sample: int,
    klass: int,
    block: int,
    summed: str,
    local_activation: str,
    relu_clip: float,
) -> tuple[str, str]:
    a_node = f"a{sample}_{klass}_{block}"
    h_node = f"h{sample}_{klass}_{block}"
    lines.append(f"Ba{sample}_{klass}_{block} {a_node} 0 V = {summed}")
    a_expr = f"V({a_node})"
    if local_activation == "relu":
        lines.append(f"Bh{sample}_{klass}_{block} {h_node} 0 V = {relu_expr(a_expr)}")
        deriv = relu_deriv_expr(a_expr)
    elif local_activation in {"clipped-relu", "clipped_relu"}:
        clip = max(float(relu_clip), 1e-12)
        lines.append(f"Bh{sample}_{klass}_{block} {h_node} 0 V = {clipped_relu_expr(a_expr, clip)}")
        deriv = clipped_relu_deriv_expr(a_expr, clip)
    elif local_activation in {"diff-clipped-relu", "differential-clipped-relu", "diff_clipped_relu"}:
        clip = max(float(relu_clip), 1e-12)
        neg_a_expr = f"0-({a_expr})"
        lines.append(
            f"Bh{sample}_{klass}_{block} {h_node} 0 V = "
            f"{clipped_relu_expr(a_expr, clip)}-{clipped_relu_expr(neg_a_expr, clip)}"
        )
        deriv = f"({clipped_relu_deriv_expr(a_expr, clip)}+{clipped_relu_deriv_expr(neg_a_expr, clip)})"
    else:
        lines.append(f"Bh{sample}_{klass}_{block} {h_node} 0 V = 2/(1+exp(-2*V({a_node})))-1")
        deriv = f"(1-V({h_node})*V({h_node}))"
    return f"V({h_node})", deriv


def add_local_activation_deriv(local_activation: str, relu_clip: float, sample: int, klass: int, block: int) -> str:
    a_node = f"a{sample}_{klass}_{block}"
    a_expr = f"V({a_node})"
    h_node = f"h{sample}_{klass}_{block}"
    if local_activation == "relu":
        return relu_deriv_expr(a_expr)
    if local_activation in {"clipped-relu", "clipped_relu"}:
        clip = max(float(relu_clip), 1e-12)
        return clipped_relu_deriv_expr(a_expr, clip)
    if local_activation in {"diff-clipped-relu", "differential-clipped-relu", "diff_clipped_relu"}:
        clip = max(float(relu_clip), 1e-12)
        neg_a_expr = f"0-({a_expr})"
        return f"({clipped_relu_deriv_expr(a_expr, clip)}+{clipped_relu_deriv_expr(neg_a_expr, clip)})"
    return f"(1-V({h_node})*V({h_node}))"


def block_indices(image_size: int, block_size: int, stride: int) -> list[list[int]]:
    if block_size > image_size:
        raise ValueError("block_size must be <= image_size")
    if stride <= 0:
        raise ValueError("stride must be positive")
    blocks: list[list[int]] = []
    for by in range(0, image_size - block_size + 1, stride):
        for bx in range(0, image_size - block_size + 1, stride):
            idx = []
            for dy in range(block_size):
                for dx in range(block_size):
                    y = by + dy
                    x = bx + dx
                    idx.append(y * image_size + x)
            blocks.append(idx)
    return blocks


def append_center_block(blocks: list[list[int]], image_size: int, block_size: int) -> list[list[int]]:
    start = (image_size - block_size) // 2
    idx = []
    for dy in range(block_size):
        for dx in range(block_size):
            y = start + dy
            x = start + dx
            idx.append(y * image_size + x)
    if idx not in blocks:
        blocks = list(blocks) + [idx]
    return blocks


def class_ranges(n_classes: int, class_chunk_size: int) -> list[tuple[int, int]]:
    if class_chunk_size <= 0 or class_chunk_size >= n_classes:
        return [(0, n_classes)]
    return [(start, min(start + class_chunk_size, n_classes)) for start in range(0, n_classes, class_chunk_size)]


def make_batch_train_netlist(
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    weights: np.ndarray,
    local_bias: np.ndarray,
    gains: np.ndarray,
    output_bias: np.ndarray,
    blocks: list[list[int]],
    lr: float,
    out_path: Path,
    train_gains: bool = False,
    input_noise: Optional[np.ndarray] = None,
    weight_mismatch: Optional[np.ndarray] = None,
    local_offset: Optional[np.ndarray] = None,
    output_offset: Optional[np.ndarray] = None,
    linear_output: bool = False,
    softmax_output: bool = False,
    local_activation: str = "tanh",
    relu_clip: float = 1.0,
    class_labels: Optional[np.ndarray] = None,
) -> str:
    batch = x_batch.shape[0]
    n_classes, n_blocks, block_len = weights.shape
    if class_labels is None:
        class_labels = np.arange(n_classes)
    class_labels = np.asarray(class_labels)
    if len(class_labels) != n_classes:
        raise ValueError("class_labels length must match weights.shape[0]")
    lines = [
        "* Local block-evidence batch operating-point SPICE training step.",
        "* Analog/multilevel local evidence: class outputs sum configurable local voltage states.",
        "* Optional fixed/per-sample perturbations emulate circuit noise, offset, and mismatch.",
        f".param LR={lr:.12g}",
        f".param BS={batch}",
        "",
    ]
    for s in range(batch):
        for i, val in enumerate(x_batch[s]):
            vin = float(val) + noise_value(input_noise, s, i)
            lines.append(f"Vx{s}_{i} x{s}_{i} 0 DC {vin:.12g}")
        if softmax_output:
            target = np.zeros(n_classes)
        else:
            target = -np.ones(n_classes)
        match = np.flatnonzero(class_labels == int(y_batch[s]))
        if len(match):
            target[int(match[0])] = 1.0
        for k in range(n_classes):
            lines.append(f"Vt{s}_{k} t{s}_{k} 0 DC {target[k]:.12g}")
    lines.append("")
    for k in range(n_classes):
        for b in range(n_blocks):
            for p in range(block_len):
                lines.append(f"Vw{k}_{b}_{p} w{k}_{b}_{p} 0 DC {weights[k, b, p]:.12g}")
            lines.append(f"Vlb{k}_{b} lb{k}_{b} 0 DC {local_bias[k, b]:.12g}")
            lines.append(f"Vg{k}_{b} g{k}_{b} 0 DC {gains[k, b]:.12g}")
        lines.append(f"Vob{k} ob{k} 0 DC {output_bias[k]:.12g}")
    lines.append("")
    for s in range(batch):
        if softmax_output:
            for k in range(n_classes):
                h_names = []
                for b, idxs in enumerate(blocks):
                    terms = []
                    for p, idx in enumerate(idxs):
                        w_eff = shifted(f"V(w{k}_{b}_{p})", noise_value(weight_mismatch, k, b, p))
                        terms.append(f"{w_eff}*V(x{s}_{idx})")
                    terms.append(shifted(f"V(lb{k}_{b})", noise_value(local_offset, k, b)))
                    summed = " + ".join(terms)
                    h_expr, _deriv = add_local_activation(lines, s, k, b, summed, local_activation, relu_clip)
                    h_names.append(f"V(g{k}_{b})*{h_expr}")
                out_sum = " + ".join(h_names + [shifted(f"V(ob{k})", noise_value(output_offset, k))])
                lines.append(f"Bz{s}_{k} z{s}_{k} 0 V = {out_sum}")
            denom = " + ".join(f"exp(V(z{s}_{j}))" for j in range(n_classes))
            for k in range(n_classes):
                lines.append(f"By{s}_{k} y{s}_{k} 0 V = exp(V(z{s}_{k}))/({denom})")
                lines.append(f"Be{s}_{k} e{s}_{k} 0 V = V(t{s}_{k})-V(y{s}_{k})")
                lines.append(f"Bd{s}_{k} d{s}_{k} 0 V = V(e{s}_{k})")
                for b in range(n_blocks):
                    lines.append(
                        f"Bdh{s}_{k}_{b} dh{s}_{k}_{b} 0 V = "
                        f"V(d{s}_{k})*V(g{k}_{b})*{add_local_activation_deriv(local_activation, relu_clip, s, k, b)}"
                    )
        else:
            for k in range(n_classes):
                h_names = []
                for b, idxs in enumerate(blocks):
                    terms = []
                    for p, idx in enumerate(idxs):
                        w_eff = shifted(f"V(w{k}_{b}_{p})", noise_value(weight_mismatch, k, b, p))
                        terms.append(f"{w_eff}*V(x{s}_{idx})")
                    terms.append(shifted(f"V(lb{k}_{b})", noise_value(local_offset, k, b)))
                    summed = " + ".join(terms)
                    h_expr, _deriv = add_local_activation(lines, s, k, b, summed, local_activation, relu_clip)
                    h_names.append(f"V(g{k}_{b})*{h_expr}")
                out_sum = " + ".join(h_names + [shifted(f"V(ob{k})", noise_value(output_offset, k))])
                if linear_output:
                    lines.append(f"By{s}_{k} y{s}_{k} 0 V = {out_sum}")
                else:
                    lines.append(f"By{s}_{k} y{s}_{k} 0 V = 2/(1+exp(-2*({out_sum})))-1")
                lines.append(f"Be{s}_{k} e{s}_{k} 0 V = V(t{s}_{k})-V(y{s}_{k})")
                if linear_output:
                    lines.append(f"Bd{s}_{k} d{s}_{k} 0 V = V(e{s}_{k})")
                else:
                    lines.append(f"Bd{s}_{k} d{s}_{k} 0 V = V(e{s}_{k})*(1-V(y{s}_{k})*V(y{s}_{k}))")
                for b in range(n_blocks):
                    lines.append(
                        f"Bdh{s}_{k}_{b} dh{s}_{k}_{b} 0 V = "
                        f"V(d{s}_{k})*V(g{k}_{b})*{add_local_activation_deriv(local_activation, relu_clip, s, k, b)}"
                    )
    lines.append("")
    for k in range(n_classes):
        for b, idxs in enumerate(blocks):
            for p, idx in enumerate(idxs):
                grad = " + ".join(f"V(dh{s}_{k}_{b})*V(x{s}_{idx})" for s in range(batch))
                lines.append(f"Bnw{k}_{b}_{p} nw{k}_{b}_{p} 0 V = V(w{k}_{b}_{p}) + {{LR}}*(({grad})/{{BS}})")
            grad_b = " + ".join(f"V(dh{s}_{k}_{b})" for s in range(batch))
            lines.append(f"Bnlb{k}_{b} nlb{k}_{b} 0 V = V(lb{k}_{b}) + {{LR}}*(({grad_b})/{{BS}})")
            if train_gains:
                grad_g = " + ".join(f"V(d{s}_{k})*V(h{s}_{k}_{b})" for s in range(batch))
                lines.append(f"Bng{k}_{b} ng{k}_{b} 0 V = V(g{k}_{b}) + {{LR}}*(({grad_g})/{{BS}})")
        grad_o = " + ".join(f"V(d{s}_{k})" for s in range(batch))
        lines.append(f"Bnob{k} nob{k} 0 V = V(ob{k}) + {{LR}}*(({grad_o})/{{BS}})")
    vectors = [f"V(nw{k}_{b}_{p})" for k in range(n_classes) for b in range(n_blocks) for p in range(block_len)]
    vectors += [f"V(nlb{k}_{b})" for k in range(n_classes) for b in range(n_blocks)]
    if train_gains:
        vectors += [f"V(ng{k}_{b})" for k in range(n_classes) for b in range(n_blocks)]
    vectors += [f"V(nob{k})" for k in range(n_classes)]
    lines += [
        "",
        ".control",
        "op",
        f"wrdata {out_path} " + " ".join(vectors),
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def make_batch_eval_netlist(
    x_batch: np.ndarray,
    weights: np.ndarray,
    local_bias: np.ndarray,
    gains: np.ndarray,
    output_bias: np.ndarray,
    blocks: list[list[int]],
    out_path: Path,
    input_noise: Optional[np.ndarray] = None,
    weight_mismatch: Optional[np.ndarray] = None,
    local_offset: Optional[np.ndarray] = None,
    output_offset: Optional[np.ndarray] = None,
    linear_output: bool = False,
    softmax_output: bool = False,
    local_activation: str = "tanh",
    relu_clip: float = 1.0,
) -> str:
    batch = x_batch.shape[0]
    n_classes, n_blocks, _block_len = weights.shape
    lines = [
        "* Local block-evidence batch operating-point SPICE inference.",
        "* Analog/multilevel local evidence with optional noise, offset, and mismatch.",
        "",
    ]
    for s in range(batch):
        for i, val in enumerate(x_batch[s]):
            vin = float(val) + noise_value(input_noise, s, i)
            lines.append(f"Vx{s}_{i} x{s}_{i} 0 DC {vin:.12g}")
    lines.append("")
    for k in range(n_classes):
        for b, idxs in enumerate(blocks):
            for p, _idx in enumerate(idxs):
                lines.append(f"Vw{k}_{b}_{p} w{k}_{b}_{p} 0 DC {weights[k, b, p]:.12g}")
            lines.append(f"Vlb{k}_{b} lb{k}_{b} 0 DC {local_bias[k, b]:.12g}")
            lines.append(f"Vg{k}_{b} g{k}_{b} 0 DC {gains[k, b]:.12g}")
        lines.append(f"Vob{k} ob{k} 0 DC {output_bias[k]:.12g}")
    lines.append("")
    for s in range(batch):
        if softmax_output:
            for k in range(n_classes):
                h_names = []
                for b, idxs in enumerate(blocks):
                    terms = []
                    for p, idx in enumerate(idxs):
                        w_eff = shifted(f"V(w{k}_{b}_{p})", noise_value(weight_mismatch, k, b, p))
                        terms.append(f"{w_eff}*V(x{s}_{idx})")
                    terms.append(shifted(f"V(lb{k}_{b})", noise_value(local_offset, k, b)))
                    h_expr, _deriv = add_local_activation(lines, s, k, b, " + ".join(terms), local_activation, relu_clip)
                    h_names.append(f"V(g{k}_{b})*{h_expr}")
                out_sum = " + ".join(h_names + [shifted(f"V(ob{k})", noise_value(output_offset, k))])
                lines.append(f"Bz{s}_{k} z{s}_{k} 0 V = {out_sum}")
            denom = " + ".join(f"exp(V(z{s}_{j}))" for j in range(n_classes))
            for k in range(n_classes):
                lines.append(f"By{s}_{k} y{s}_{k} 0 V = exp(V(z{s}_{k}))/({denom})")
        else:
            for k in range(n_classes):
                h_names = []
                for b, idxs in enumerate(blocks):
                    terms = []
                    for p, idx in enumerate(idxs):
                        w_eff = shifted(f"V(w{k}_{b}_{p})", noise_value(weight_mismatch, k, b, p))
                        terms.append(f"{w_eff}*V(x{s}_{idx})")
                    terms.append(shifted(f"V(lb{k}_{b})", noise_value(local_offset, k, b)))
                    h_expr, _deriv = add_local_activation(lines, s, k, b, " + ".join(terms), local_activation, relu_clip)
                    h_names.append(f"V(g{k}_{b})*{h_expr}")
                output_terms = h_names + [shifted(f"V(ob{k})", noise_value(output_offset, k))]
                out_sum = " + ".join(output_terms)
                if linear_output:
                    lines.append(f"By{s}_{k} y{s}_{k} 0 V = {out_sum}")
                else:
                    lines.append(f"By{s}_{k} y{s}_{k} 0 V = 2/(1+exp(-2*({out_sum})))-1")
    vectors = [f"V(y{s}_{k})" for s in range(batch) for k in range(n_classes)]
    lines += [
        "",
        ".control",
        "op",
        f"wrdata {out_path} " + " ".join(vectors),
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def run_train_batch(
    spice_bin,
    netlist_path,
    data_path,
    x,
    y,
    weights,
    local_bias,
    gains,
    output_bias,
    blocks,
    lr,
    timeout,
    train_gains,
    input_noise=None,
    weight_mismatch=None,
    local_offset=None,
    output_offset=None,
    linear_output=False,
    softmax_output=False,
    local_activation="tanh",
    relu_clip=1.0,
    class_labels=None,
):
    netlist_path.write_text(
        prepare_netlist_for_simulator(
            make_batch_train_netlist(
                x,
                y,
                weights,
                local_bias,
                gains,
                output_bias,
                blocks,
                lr,
                data_path,
                train_gains,
                input_noise=input_noise,
                weight_mismatch=weight_mismatch,
                local_offset=local_offset,
                output_offset=output_offset,
                linear_output=linear_output,
                softmax_output=softmax_output,
                local_activation=local_activation,
                relu_clip=relu_clip,
                class_labels=class_labels,
            ),
            spice_bin,
        )
    )
    proc = run_simulator_netlist(spice_bin, netlist_path, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-3000:] or proc.stdout[-3000:])
    n_classes, n_blocks, block_len = weights.shape
    n = n_classes * n_blocks * block_len + n_classes * n_blocks + (n_classes * n_blocks if train_gains else 0) + n_classes
    vals = read_wrdata_row(data_path, n)
    offset = 0
    nw = vals[offset : offset + n_classes * n_blocks * block_len].reshape(weights.shape)
    offset += n_classes * n_blocks * block_len
    nlb = vals[offset : offset + n_classes * n_blocks].reshape(local_bias.shape)
    offset += n_classes * n_blocks
    if train_gains:
        ng = vals[offset : offset + n_classes * n_blocks].reshape(gains.shape)
        offset += n_classes * n_blocks
    else:
        ng = gains
    nob = vals[offset : offset + n_classes]
    return nw, nlb, ng, nob


def run_eval(
    spice_bin,
    netlist_path,
    data_path,
    x_eval,
    y_eval,
    weights,
    local_bias,
    gains,
    output_bias,
    blocks,
    batch_size,
    timeout,
    rng: Optional[np.random.Generator] = None,
    input_noise_sigma: float = 0.0,
    weight_mismatch_sigma: float = 0.0,
    local_offset_sigma: float = 0.0,
    output_offset_sigma: float = 0.0,
    linear_output: bool = False,
    softmax_output: bool = False,
    local_activation: str = "tanh",
    relu_clip: float = 1.0,
    class_chunk_size: int = 0,
):
    correct = 0
    if rng is None:
        rng = np.random.default_rng(0)
    weight_mismatch = sample_noise(rng, weight_mismatch_sigma, weights.shape)
    local_offset = sample_noise(rng, local_offset_sigma, local_bias.shape)
    output_offset = sample_noise(rng, output_offset_sigma, output_bias.shape)
    ranges = class_ranges(weights.shape[0], class_chunk_size)
    if len(ranges) > 1 and softmax_output:
        raise ValueError("class chunking is only valid for independent tanh/linear class outputs, not softmax")
    for start in range(0, len(y_eval), batch_size):
        x = x_eval[start : start + batch_size]
        y = y_eval[start : start + batch_size]
        input_noise = sample_noise(rng, input_noise_sigma, x.shape)
        vals_by_chunk = []
        for cs, ce in ranges:
            wm = None if weight_mismatch is None else weight_mismatch[cs:ce]
            lo = None if local_offset is None else local_offset[cs:ce]
            oo = None if output_offset is None else output_offset[cs:ce]
            netlist_path.write_text(
                prepare_netlist_for_simulator(
                    make_batch_eval_netlist(
                        x,
                        weights[cs:ce],
                        local_bias[cs:ce],
                        gains[cs:ce],
                        output_bias[cs:ce],
                        blocks,
                        data_path,
                        input_noise=input_noise,
                        weight_mismatch=wm,
                        local_offset=lo,
                        output_offset=oo,
                        linear_output=linear_output,
                        softmax_output=softmax_output,
                        local_activation=local_activation,
                        relu_clip=relu_clip,
                    ),
                    spice_bin,
                )
            )
            proc = run_simulator_netlist(spice_bin, netlist_path, timeout=timeout)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr[-3000:] or proc.stdout[-3000:])
            vals_by_chunk.append(read_wrdata_row(data_path, len(y) * (ce - cs)).reshape(len(y), ce - cs))
        vals = vals_by_chunk[0] if len(vals_by_chunk) == 1 else np.concatenate(vals_by_chunk, axis=1)
        correct += int((np.argmax(vals, axis=1) == y).sum())
    return correct / max(len(y_eval), 1)


def run_eval_repeated(
    spice_bin,
    netlist_path,
    data_path,
    x_eval,
    y_eval,
    weights,
    local_bias,
    gains,
    output_bias,
    blocks,
    batch_size,
    timeout,
    rng,
    repeats: int,
    input_noise_sigma: float,
    weight_mismatch_sigma: float,
    local_offset_sigma: float,
    output_offset_sigma: float,
    linear_output: bool,
    softmax_output: bool,
    local_activation: str,
    relu_clip: float,
    class_chunk_size: int,
) -> tuple[float, float, list[float]]:
    accs = []
    for _ in range(max(1, repeats)):
        accs.append(
            run_eval(
                spice_bin,
                netlist_path,
                data_path,
                x_eval,
                y_eval,
                weights,
                local_bias,
                gains,
                output_bias,
                blocks,
                batch_size,
                timeout,
                rng=rng,
                input_noise_sigma=input_noise_sigma,
                weight_mismatch_sigma=weight_mismatch_sigma,
                local_offset_sigma=local_offset_sigma,
                output_offset_sigma=output_offset_sigma,
                linear_output=linear_output,
                softmax_output=softmax_output,
                local_activation=local_activation,
                relu_clip=relu_clip,
                class_chunk_size=class_chunk_size,
            )
        )
    return float(np.mean(accs)), float(np.std(accs)), accs


def plot_curve(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(df["epoch"], df["heldout_accuracy"], marker="o")
    plt.xlabel("epoch")
    plt.ylabel("held-out accuracy")
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-samples", type=int, default=200)
    ap.add_argument("--test-samples", type=int, default=200)
    ap.add_argument("--image-size", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=4)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--add-center-block", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument(
        "--epoch-train-samples",
        type=int,
        default=0,
        help="Optional shuffled training samples per epoch; preserves the loaded train/test split while bounding expensive SPICE epochs.",
    )
    ap.add_argument(
        "--epoch-train-offset",
        type=int,
        default=0,
        help="Offset into the shuffled epoch order when --epoch-train-samples is used.",
    )
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=90)
    ap.add_argument("--train-gains", action="store_true")
    ap.add_argument("--train-input-noise-sigma", type=float, default=0.0)
    ap.add_argument("--train-weight-mismatch-sigma", type=float, default=0.0)
    ap.add_argument("--train-local-offset-sigma", type=float, default=0.0)
    ap.add_argument("--train-output-offset-sigma", type=float, default=0.0)
    ap.add_argument("--eval-input-noise-sigma", type=float, default=0.0)
    ap.add_argument("--eval-weight-mismatch-sigma", type=float, default=0.0)
    ap.add_argument("--eval-local-offset-sigma", type=float, default=0.0)
    ap.add_argument("--eval-output-offset-sigma", type=float, default=0.0)
    ap.add_argument("--eval-repeats", type=int, default=1)
    ap.add_argument("--robustness-sigmas", default="")
    ap.add_argument("--robustness-repeats", type=int, default=3)
    ap.add_argument("--linear-output", action="store_true")
    ap.add_argument("--softmax-output", action="store_true")
    ap.add_argument(
        "--local-activation",
        choices=["tanh", "relu", "clipped-relu", "diff-clipped-relu", "differential-clipped-relu"],
        default="tanh",
    )
    ap.add_argument("--relu-clip", type=float, default=1.0)
    ap.add_argument("--local-bias-init", type=float, default=0.0)
    ap.add_argument("--class-chunk-size", type=int, default=0)
    ap.add_argument("--init-weights", default=None)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--tag", default="local_block")
    args = ap.parse_args()
    if args.linear_output and args.softmax_output:
        raise ValueError("--linear-output and --softmax-output are mutually exclusive")
    if args.class_chunk_size > 0 and args.softmax_output:
        raise ValueError("--class-chunk-size is only valid for independent tanh/linear outputs, not softmax")

    stride = args.block_size if args.stride is None else args.stride
    blocks = block_indices(args.image_size, args.block_size, stride)
    if args.add_center_block:
        blocks = append_center_block(blocks, args.image_size, args.block_size)
    x_train, y_train, x_test, y_test = load_mnist_sequence(args.train_samples, args.test_samples, args.image_size, args.seed)
    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    generated.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)
    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag)
    stem = f"spice_mnist_local_block_{safe_tag}"
    netlist_path = generated / f"{stem}_step.cir"
    eval_netlist = generated / f"{stem}_eval.cir"
    data_path = ROOT / f"spice/results/{stem}_step.dat"
    rng = np.random.default_rng(args.seed)
    weights = rng.normal(0.0, 0.02, size=(10, len(blocks), args.block_size * args.block_size))
    local_bias = np.full((10, len(blocks)), args.local_bias_init)
    gains = np.full((10, len(blocks)), 1.0 / max(len(blocks), 1) if args.train_gains else 1.0)
    output_bias = np.zeros(10)
    if args.init_weights:
        init = np.load(args.init_weights)
        init_weights = init["weights"]
        init_local_bias = init["local_bias"]
        init_gains = init["gains"]
        init_output_bias = init["output_bias"]
        expected_shapes = (
            (10, len(blocks), args.block_size * args.block_size),
            (10, len(blocks)),
            (10, len(blocks)),
            (10,),
        )
        actual_shapes = (init_weights.shape, init_local_bias.shape, init_gains.shape, init_output_bias.shape)
        if actual_shapes == expected_shapes:
            weights = init_weights
            local_bias = init_local_bias
            gains = init_gains
            output_bias = init_output_bias
        elif (
            init_weights.shape[0] == 10
            and init_weights.shape[2] == args.block_size * args.block_size
            and init_local_bias.shape == (10, init_weights.shape[1])
            and init_gains.shape == (10, init_weights.shape[1])
            and init_output_bias.shape == (10,)
            and init_weights.shape[1] < len(blocks)
        ):
            old_blocks = init_weights.shape[1]
            weights[:, :old_blocks, :] = init_weights
            weights[:, old_blocks:, :] = 0.0
            local_bias[:, :old_blocks] = init_local_bias
            local_bias[:, old_blocks:] = args.local_bias_init
            gains[:, :old_blocks] = init_gains
            output_bias = init_output_bias
        elif (
            init_weights.shape[0] == 10
            and init_weights.shape[1] == len(blocks)
            and init_local_bias.shape == (10, len(blocks))
            and init_gains.shape == (10, len(blocks))
            and init_output_bias.shape == (10,)
        ):
            old_block_size = int(round(init_weights.shape[2] ** 0.5))
            if old_block_size * old_block_size != init_weights.shape[2] or old_block_size * 2 != args.block_size:
                raise ValueError(f"cannot 2x upsample initial block length {init_weights.shape[2]} to block_size={args.block_size}")
            reshaped = init_weights.reshape(10, len(blocks), old_block_size, old_block_size)
            expanded = np.repeat(np.repeat(reshaped, 2, axis=2), 2, axis=3) / 4.0
            weights = expanded.reshape(10, len(blocks), args.block_size * args.block_size)
            local_bias = init_local_bias
            gains = init_gains
            output_bias = init_output_bias
        else:
            raise ValueError(f"initial weight shapes {actual_shapes} do not match expected {expected_shapes}")
    train_weight_mismatch = sample_noise(rng, args.train_weight_mismatch_sigma, weights.shape)
    train_local_offset = sample_noise(rng, args.train_local_offset_sigma, local_bias.shape)
    train_output_offset = sample_noise(rng, args.train_output_offset_sigma, output_bias.shape)
    ranges = class_ranges(weights.shape[0], args.class_chunk_size)
    weights_path = ROOT / f"spice/results/{stem}_final_weights.npz"
    best_weights_path = ROOT / f"spice/results/{stem}_best_weights.npz"

    def save_weights(path: Path) -> None:
        np.savez_compressed(path, weights=weights, local_bias=local_bias, gains=gains, output_bias=output_bias)

    rows = []
    best_acc = -1.0
    t0 = time.perf_counter()
    if args.eval_only or args.epochs == 0:
        epoch_start = time.perf_counter()
        heldout, heldout_std, _heldout_repeats = run_eval_repeated(
            spice_bin, eval_netlist, data_path, x_test, y_test,
            weights, local_bias, gains, output_bias, blocks, args.batch_size, args.timeout,
            rng,
            args.eval_repeats,
            args.eval_input_noise_sigma,
            args.eval_weight_mismatch_sigma,
            args.eval_local_offset_sigma,
            args.eval_output_offset_sigma,
            args.linear_output,
            args.softmax_output,
            args.local_activation,
            args.relu_clip,
            args.class_chunk_size,
        )
        row = {
            "epoch": 0,
            "heldout_accuracy": heldout,
            "heldout_accuracy_std": heldout_std,
            "epoch_wall_time_s": time.perf_counter() - epoch_start,
        }
        rows.append(row)
        best_acc = heldout
        save_weights(best_weights_path)
        print(json.dumps(row), flush=True)
    else:
        for epoch in range(args.epochs):
            order = np.arange(len(y_train))
            rng.shuffle(order)
            if args.epoch_train_samples > 0:
                start = min(max(args.epoch_train_offset, 0), len(order))
                stop = min(start + args.epoch_train_samples, len(order))
                order = order[start:stop]
            epoch_start = time.perf_counter()
            for n, start in enumerate(range(0, len(order), args.batch_size)):
                idx = order[start : start + args.batch_size]
                train_input_noise = sample_noise(rng, args.train_input_noise_sigma, x_train[idx].shape)
                for cs, ce in ranges:
                    wm = None if train_weight_mismatch is None else train_weight_mismatch[cs:ce]
                    lo = None if train_local_offset is None else train_local_offset[cs:ce]
                    oo = None if train_output_offset is None else train_output_offset[cs:ce]
                    nw, nlb, ng, nob = run_train_batch(
                        spice_bin, netlist_path, data_path, x_train[idx], y_train[idx],
                        weights[cs:ce], local_bias[cs:ce], gains[cs:ce], output_bias[cs:ce],
                        blocks, args.lr, args.timeout, args.train_gains,
                        input_noise=train_input_noise,
                        weight_mismatch=wm,
                        local_offset=lo,
                        output_offset=oo,
                        linear_output=args.linear_output,
                        softmax_output=args.softmax_output,
                        local_activation=args.local_activation,
                        relu_clip=args.relu_clip,
                        class_labels=np.arange(cs, ce),
                    )
                    weights[cs:ce] = nw
                    local_bias[cs:ce] = nlb
                    gains[cs:ce] = ng
                    output_bias[cs:ce] = nob
                if (n + 1) % 5 == 0:
                    print(f"epoch {epoch + 1} batch {n + 1}", flush=True)
            heldout, heldout_std, _heldout_repeats = run_eval_repeated(
                spice_bin, eval_netlist, data_path, x_test, y_test,
                weights, local_bias, gains, output_bias, blocks, args.batch_size, args.timeout,
                rng,
                args.eval_repeats,
                args.eval_input_noise_sigma,
                args.eval_weight_mismatch_sigma,
                args.eval_local_offset_sigma,
                args.eval_output_offset_sigma,
                args.linear_output,
                args.softmax_output,
                args.local_activation,
                args.relu_clip,
                args.class_chunk_size,
            )
            row = {
                "epoch": epoch + 1,
                "heldout_accuracy": heldout,
                "heldout_accuracy_std": heldout_std,
                "epoch_wall_time_s": time.perf_counter() - epoch_start,
            }
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
    robustness_rows = []
    robustness_sigmas = parse_float_list(args.robustness_sigmas)
    if robustness_sigmas:
        for sigma in robustness_sigmas:
            mean_acc, std_acc, accs = run_eval_repeated(
                spice_bin,
                eval_netlist,
                data_path,
                x_test,
                y_test,
                weights,
                local_bias,
                gains,
                output_bias,
                blocks,
                args.batch_size,
                args.timeout,
                rng,
                args.robustness_repeats,
                sigma,
                sigma,
                sigma,
                sigma,
                args.linear_output,
                args.softmax_output,
                args.local_activation,
                args.relu_clip,
                args.class_chunk_size,
            )
            for repeat, acc in enumerate(accs):
                robustness_rows.append(
                    {
                        "noise_sigma": sigma,
                        "repeat": repeat,
                        "heldout_accuracy": acc,
                        "mean_heldout_accuracy": mean_acc,
                        "std_heldout_accuracy": std_acc,
                    }
                )
            print(json.dumps({"robustness_noise_sigma": sigma, "mean_accuracy": mean_acc, "std_accuracy": std_acc}), flush=True)
    robustness_path = ROOT / f"spice/results/{stem}_robustness.csv"
    if robustness_rows:
        pd.DataFrame(robustness_rows).to_csv(robustness_path, index=False)
    summary = {
        "simulator": version,
        "dataset": "MNIST train/test split, downsampled",
        "architecture": "local_block_class_evidence_trainable_gains" if args.train_gains else "local_block_class_evidence",
        "activation": f"analog_{args.local_activation}_voltage_state",
        "local_activation": args.local_activation,
        "relu_clip": args.relu_clip,
        "local_bias_init": args.local_bias_init,
        "class_chunk_size": args.class_chunk_size,
        "output_mode": "softmax_class_evidence" if args.softmax_output else ("linear_class_evidence" if args.linear_output else "tanh_class_evidence"),
        "image_size": args.image_size,
        "block_size": args.block_size,
        "stride": stride,
        "add_center_block": bool(args.add_center_block),
        "blocks": len(blocks),
        "local": True,
        "train_gains": bool(args.train_gains),
        "linear_output": bool(args.linear_output),
        "softmax_output": bool(args.softmax_output),
        "inputs": int(x_train.shape[1]),
        "classes": 10,
        "train_samples": args.train_samples,
        "epoch_train_samples": int(args.epoch_train_samples) if args.epoch_train_samples > 0 else int(args.train_samples),
        "epoch_train_offset": int(args.epoch_train_offset),
        "test_samples": args.test_samples,
        "epochs": args.epochs,
        "eval_only": bool(args.eval_only),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "init_weights": args.init_weights,
        "train_input_noise_sigma": args.train_input_noise_sigma,
        "train_weight_mismatch_sigma": args.train_weight_mismatch_sigma,
        "train_local_offset_sigma": args.train_local_offset_sigma,
        "train_output_offset_sigma": args.train_output_offset_sigma,
        "eval_input_noise_sigma": args.eval_input_noise_sigma,
        "eval_weight_mismatch_sigma": args.eval_weight_mismatch_sigma,
        "eval_local_offset_sigma": args.eval_local_offset_sigma,
        "eval_output_offset_sigma": args.eval_output_offset_sigma,
        "eval_repeats": args.eval_repeats,
        "robustness_sigmas": robustness_sigmas,
        "robustness_repeats": args.robustness_repeats,
        "robustness_table": str(robustness_path) if robustness_rows else None,
        "wall_time_s": time.perf_counter() - t0,
        "learning_curve": str(curve_path),
        "figure": str(fig),
        "final_weights": str(weights_path),
        "best_weights": str(best_weights_path),
        "heldout_test_accuracy": float(curve.iloc[-1]["heldout_accuracy"]),
        "best_heldout_accuracy": float(curve["heldout_accuracy"].max()),
        "note": (
            "Local block-evidence batch-op all-SPICE training: ngspice computes analog local nonlinear evidence, "
            "class errors, programmable local weight updates, and optional perturbations for noise/offset/mismatch; "
            "Python carries measured state between batches."
        ),
    }
    out = ROOT / f"spice/results/{stem}_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
