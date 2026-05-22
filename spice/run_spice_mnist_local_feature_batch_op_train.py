from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_spice_mnist_batch_op_train import read_wrdata_row
from run_spice_mnist_local_block_batch_op_train import (
    add_local_activation,
    add_local_activation_deriv,
    block_indices,
    readout_feedback_expr,
    synapse_transfer_expr,
)
from local_feature_error import class_centered_expr, mean_centered_expr, softmax_delta_expr, softmax_exp_expr
from run_spice_mnist_train import load_mnist_sequence
from run_spice_sweep import ROOT, detect_spice, is_xyce, prepare_netlist_for_simulator, run_tiny_test, run_simulator_netlist


def feature_node(sample: int, block: int, channel: int, channels: int) -> str:
    return f"h{sample}_0_{block * channels + channel}"


def feature_expr(sample: int, block: int, channel: int, channels: int) -> str:
    return f"V({feature_node(sample, block, channel, channels)})"


def xyce_prn_path(netlist_path: Path) -> Path:
    return Path(str(netlist_path) + ".prn")


def read_xyce_print_row(path: Path, expected_values: int) -> np.ndarray:
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
        return values[:expected_values]
    raise ValueError(f"no Xyce print data row found in {path}")


def read_output_row(spice_bin: str, netlist_path: Path, data_path: Path, expected_values: int) -> np.ndarray:
    if is_xyce(spice_bin):
        return read_xyce_print_row(xyce_prn_path(netlist_path), expected_values)
    return read_wrdata_row(data_path, expected_values)


def append_op_output(lines: list[str], out_path: Path, vectors: list[str]) -> None:
    lines += [
        "",
        ".op",
        ".print DC " + " ".join(vectors),
        ".control",
        "op",
        f"wrdata {out_path} " + " ".join(vectors),
        ".endc",
        ".end",
        "",
    ]


def wrap_xyce_behavioral_rhs(netlist: str) -> str:
    wrapped = []
    for line in netlist.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("B") and " = " in line:
            prefix, rhs = line.split(" = ", 1)
            if not rhs.startswith("{"):
                line = f"{prefix} = {{{rhs}}}"
        wrapped.append(line)
    return "\n".join(wrapped) + ("\n" if netlist.endswith("\n") else "")


def prepare_local_feature_netlist(netlist: str, spice_bin: str) -> str:
    rendered = prepare_netlist_for_simulator(netlist, spice_bin)
    return wrap_xyce_behavioral_rhs(rendered) if is_xyce(spice_bin) else rendered


def make_train_netlist(
    x_batch,
    y_batch,
    w,
    hb,
    v,
    ob,
    blocks,
    lr,
    out_path,
    linear_output,
    softmax_output,
    local_activation,
    relu_clip,
    activation_derivative="exact",
    derivative_floor=0.0,
    derivative_gate_threshold=1e-6,
    readout_feedback_mode="readout",
    readout_feedback_clip=0.05,
    relu_leak=0.01,
    softplus_beta=10.0,
    hidden_synapse_mode="linear",
    readout_synapse_mode="linear",
    synapse_clip=1.0,
    softmax_negative_scale=1.0,
    softmax_error_centering="none",
    softmax_temperature=1.0,
    readout_class_centering="none",
):
    if softmax_negative_scale < 0:
        raise ValueError("softmax_negative_scale must be non-negative")
    if softmax_temperature <= 0:
        raise ValueError("softmax_temperature must be positive")
    if softmax_error_centering not in {"none", "mean"}:
        raise ValueError("softmax_error_centering must be 'none' or 'mean'")
    if readout_class_centering not in {"none", "mean"}:
        raise ValueError("readout_class_centering must be 'none' or 'mean'")
    batch = len(y_batch)
    n_blocks, channels, block_len = w.shape
    n_classes = v.shape[0]
    lines = [
        "* Local feature batch operating-point SPICE training.",
        "* Local learned features feed trainable class evidence; ngspice computes backprop/update.",
        f".param LR={lr:.12g}",
        f".param BS={batch}",
        f".param SOFTMAX_NEGATIVE_SCALE={softmax_negative_scale:.12g}",
        f".param SOFTMAX_TEMPERATURE={softmax_temperature:.12g}",
        "",
    ]
    for s in range(batch):
        for i, val in enumerate(x_batch[s]):
            lines.append(f"Vx{s}_{i} x{s}_{i} 0 DC {float(val):.12g}")
        if softmax_output:
            target = np.zeros(n_classes)
            target[int(y_batch[s])] = 1.0
        else:
            target = -np.ones(n_classes)
            target[int(y_batch[s])] = 1.0
        for k in range(n_classes):
            lines.append(f"Vt{s}_{k} t{s}_{k} 0 DC {target[k]:.12g}")
    lines.append("")
    for b in range(n_blocks):
        for c in range(channels):
            for p in range(block_len):
                lines.append(f"Vw{b}_{c}_{p} w{b}_{c}_{p} 0 DC {w[b, c, p]:.12g}")
            lines.append(f"Vhb{b}_{c} hb{b}_{c} 0 DC {hb[b, c]:.12g}")
    for k in range(n_classes):
        for b in range(n_blocks):
            for c in range(channels):
                lines.append(f"Vv{k}_{b}_{c} v{k}_{b}_{c} 0 DC {v[k, b, c]:.12g}")
        lines.append(f"Vob{k} ob{k} 0 DC {ob[k]:.12g}")
    lines.append("")
    for s in range(batch):
        for b, idxs in enumerate(blocks):
            for c in range(channels):
                terms = [
                    f"{synapse_transfer_expr(f'V(w{b}_{c}_{p})', hidden_synapse_mode, synapse_clip)}*V(x{s}_{idx})"
                    for p, idx in enumerate(idxs)
                ]
                terms.append(f"V(hb{b}_{c})")
                add_local_activation(
                    lines,
                    s,
                    0,
                    b * channels + c,
                    " + ".join(terms),
                    local_activation,
                    relu_clip,
                    relu_leak,
                    softplus_beta,
                )
        for k in range(n_classes):
            readout_exprs_for_class = {}
            for b in range(n_blocks):
                for c in range(channels):
                    class_exprs = [
                        synapse_transfer_expr(f"V(v{kk}_{b}_{c})", readout_synapse_mode, synapse_clip)
                        for kk in range(n_classes)
                    ]
                    readout_exprs_for_class[(b, c)] = class_centered_expr(class_exprs, k, readout_class_centering)
            terms = [
                f"{readout_exprs_for_class[(b, c)]}*{feature_expr(s, b, c, channels)}"
                for b in range(n_blocks)
                for c in range(channels)
            ]
            terms.append(f"V(ob{k})")
            out_sum = " + ".join(terms)
            if softmax_output:
                lines.append(f"Bz{s}_{k} z{s}_{k} 0 V = {out_sum}")
            elif linear_output:
                lines.append(f"By{s}_{k} y{s}_{k} 0 V = {out_sum}")
            else:
                lines.append(f"By{s}_{k} y{s}_{k} 0 V = 2/(1+exp(-2*({out_sum})))-1")
        if softmax_output:
            denom = " + ".join(softmax_exp_expr(f"V(z{s}_{k})") for k in range(n_classes))
            raw_delta_exprs = [softmax_delta_expr(f"V(t{s}_{k})", f"V(y{s}_{k})") for k in range(n_classes)]
            for k in range(n_classes):
                lines.append(f"By{s}_{k} y{s}_{k} 0 V = {softmax_exp_expr(f'V(z{s}_{k})')}/({denom})")
                delta_expr = (
                    mean_centered_expr(raw_delta_exprs, k)
                    if softmax_error_centering == "mean"
                    else raw_delta_exprs[k]
                )
                lines.append(f"Be{s}_{k} e{s}_{k} 0 V = {delta_expr}")
                lines.append(f"Bd{s}_{k} d{s}_{k} 0 V = V(e{s}_{k})")
        else:
            for k in range(n_classes):
                lines.append(f"Be{s}_{k} e{s}_{k} 0 V = V(t{s}_{k})-V(y{s}_{k})")
                if linear_output:
                    lines.append(f"Bd{s}_{k} d{s}_{k} 0 V = V(e{s}_{k})")
                else:
                    lines.append(f"Bd{s}_{k} d{s}_{k} 0 V = V(e{s}_{k})*(1-V(y{s}_{k})*V(y{s}_{k}))")
        for b in range(n_blocks):
            for c in range(channels):
                fb = " + ".join(
                    readout_feedback_expr(
                        class_centered_expr(
                            [
                                synapse_transfer_expr(f"V(v{kk}_{b}_{c})", readout_synapse_mode, synapse_clip)
                                for kk in range(n_classes)
                            ],
                            k,
                            readout_class_centering,
                        ),
                        f"V(d{s}_{k})",
                        readout_feedback_mode,
                        readout_feedback_clip,
                    )
                    for k in range(n_classes)
                )
                deriv = add_local_activation_deriv(
                    local_activation,
                    relu_clip,
                    s,
                    0,
                    b * channels + c,
                    activation_derivative,
                    derivative_floor,
                    derivative_gate_threshold,
                    relu_leak,
                    softplus_beta,
                )
                lines.append(f"Bdh{s}_{b}_{c} dh{s}_{b}_{c} 0 V = ({fb})*{deriv}")
    lines.append("")
    for b, idxs in enumerate(blocks):
        for c in range(channels):
            for p, idx in enumerate(idxs):
                grad = " + ".join(f"V(dh{s}_{b}_{c})*V(x{s}_{idx})" for s in range(batch))
                lines.append(f"Bnw{b}_{c}_{p} nw{b}_{c}_{p} 0 V = V(w{b}_{c}_{p}) + {{LR}}*(({grad})/{{BS}})")
            grad_b = " + ".join(f"V(dh{s}_{b}_{c})" for s in range(batch))
            lines.append(f"Bnhb{b}_{c} nhb{b}_{c} 0 V = V(hb{b}_{c}) + {{LR}}*(({grad_b})/{{BS}})")
    for k in range(n_classes):
        for b in range(n_blocks):
            for c in range(channels):
                grad = " + ".join(f"V(d{s}_{k})*{feature_expr(s, b, c, channels)}" for s in range(batch))
                lines.append(f"Bnv{k}_{b}_{c} nv{k}_{b}_{c} 0 V = V(v{k}_{b}_{c}) + {{LR}}*(({grad})/{{BS}})")
        grad_o = " + ".join(f"V(d{s}_{k})" for s in range(batch))
        lines.append(f"Bnob{k} nob{k} 0 V = V(ob{k}) + {{LR}}*(({grad_o})/{{BS}})")
    vectors = [f"V(nw{b}_{c}_{p})" for b in range(n_blocks) for c in range(channels) for p in range(block_len)]
    vectors += [f"V(nhb{b}_{c})" for b in range(n_blocks) for c in range(channels)]
    vectors += [f"V(nv{k}_{b}_{c})" for k in range(n_classes) for b in range(n_blocks) for c in range(channels)]
    vectors += [f"V(nob{k})" for k in range(n_classes)]
    append_op_output(lines, out_path, vectors)
    return "\n".join(lines)


def make_eval_netlist(
    x_batch,
    w,
    hb,
    v,
    ob,
    blocks,
    out_path,
    linear_output,
    softmax_output,
    local_activation,
    relu_clip,
    relu_leak=0.01,
    softplus_beta=10.0,
    hidden_synapse_mode="linear",
    readout_synapse_mode="linear",
    synapse_clip=1.0,
    softmax_temperature=1.0,
    readout_class_centering="none",
):
    if softmax_temperature <= 0:
        raise ValueError("softmax_temperature must be positive")
    batch = len(x_batch)
    n_blocks, channels, _block_len = w.shape
    n_classes = v.shape[0]
    lines = ["* Local feature batch operating-point SPICE inference.", ""]
    if softmax_output:
        lines += [f".param SOFTMAX_TEMPERATURE={softmax_temperature:.12g}", ""]
    for s in range(batch):
        for i, val in enumerate(x_batch[s]):
            lines.append(f"Vx{s}_{i} x{s}_{i} 0 DC {float(val):.12g}")
    lines.append("")
    for b in range(n_blocks):
        for c in range(channels):
            for p in range(w.shape[2]):
                lines.append(f"Vw{b}_{c}_{p} w{b}_{c}_{p} 0 DC {w[b, c, p]:.12g}")
            lines.append(f"Vhb{b}_{c} hb{b}_{c} 0 DC {hb[b, c]:.12g}")
    for k in range(n_classes):
        for b in range(n_blocks):
            for c in range(channels):
                lines.append(f"Vv{k}_{b}_{c} v{k}_{b}_{c} 0 DC {v[k, b, c]:.12g}")
        lines.append(f"Vob{k} ob{k} 0 DC {ob[k]:.12g}")
    lines.append("")
    for s in range(batch):
        for b, idxs in enumerate(blocks):
            for c in range(channels):
                terms = [
                    f"{synapse_transfer_expr(f'V(w{b}_{c}_{p})', hidden_synapse_mode, synapse_clip)}*V(x{s}_{idx})"
                    for p, idx in enumerate(idxs)
                ]
                terms.append(f"V(hb{b}_{c})")
                add_local_activation(
                    lines,
                    s,
                    0,
                    b * channels + c,
                    " + ".join(terms),
                    local_activation,
                    relu_clip,
                    relu_leak,
                    softplus_beta,
                )
        for k in range(n_classes):
            readout_exprs_for_class = {}
            for b in range(n_blocks):
                for c in range(channels):
                    class_exprs = [
                        synapse_transfer_expr(f"V(v{kk}_{b}_{c})", readout_synapse_mode, synapse_clip)
                        for kk in range(n_classes)
                    ]
                    readout_exprs_for_class[(b, c)] = class_centered_expr(class_exprs, k, readout_class_centering)
            terms = [
                f"{readout_exprs_for_class[(b, c)]}*{feature_expr(s, b, c, channels)}"
                for b in range(n_blocks)
                for c in range(channels)
            ]
            terms.append(f"V(ob{k})")
            out_sum = " + ".join(terms)
            if softmax_output:
                lines.append(f"Bz{s}_{k} z{s}_{k} 0 V = {out_sum}")
            elif linear_output:
                lines.append(f"By{s}_{k} y{s}_{k} 0 V = {out_sum}")
            else:
                lines.append(f"By{s}_{k} y{s}_{k} 0 V = 2/(1+exp(-2*({out_sum})))-1")
        if softmax_output:
            denom = " + ".join(softmax_exp_expr(f"V(z{s}_{k})") for k in range(n_classes))
            for k in range(n_classes):
                lines.append(f"By{s}_{k} y{s}_{k} 0 V = {softmax_exp_expr(f'V(z{s}_{k})')}/({denom})")
    vectors = [f"V(y{s}_{k})" for s in range(batch) for k in range(n_classes)]
    append_op_output(lines, out_path, vectors)
    return "\n".join(lines)


def run_train_batch(
    spice_bin,
    netlist_path,
    data_path,
    x,
    y,
    w,
    hb,
    v,
    ob,
    blocks,
    lr,
    timeout,
    linear_output,
    softmax_output,
    local_activation,
    relu_clip,
    activation_derivative="exact",
    derivative_floor=0.0,
    derivative_gate_threshold=1e-6,
    readout_feedback_mode="readout",
    readout_feedback_clip=0.05,
    relu_leak=0.01,
    softplus_beta=10.0,
    hidden_synapse_mode="linear",
    readout_synapse_mode="linear",
    synapse_clip=1.0,
    softmax_negative_scale=1.0,
    softmax_error_centering="none",
    softmax_temperature=1.0,
    readout_class_centering="none",
):
    netlist_path.write_text(
        prepare_local_feature_netlist(
            make_train_netlist(
                x,
                y,
                w,
                hb,
                v,
                ob,
                blocks,
                lr,
                data_path,
                linear_output,
                softmax_output,
                local_activation,
                relu_clip,
                activation_derivative,
                derivative_floor,
                derivative_gate_threshold,
                readout_feedback_mode,
                readout_feedback_clip,
                relu_leak,
                softplus_beta,
                hidden_synapse_mode,
                readout_synapse_mode,
                synapse_clip,
                softmax_negative_scale,
                softmax_error_centering,
                softmax_temperature,
                readout_class_centering,
            ),
            spice_bin,
        )
    )
    proc = run_simulator_netlist(spice_bin, netlist_path, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-3000:] or proc.stdout[-3000:])
    n_blocks, channels, block_len = w.shape
    n_classes = v.shape[0]
    n = n_blocks * channels * block_len + n_blocks * channels + n_classes * n_blocks * channels + n_classes
    vals = read_output_row(spice_bin, netlist_path, data_path, n)
    offset = 0
    nw = vals[offset : offset + n_blocks * channels * block_len].reshape(w.shape)
    offset += n_blocks * channels * block_len
    nhb = vals[offset : offset + n_blocks * channels].reshape(hb.shape)
    offset += n_blocks * channels
    nv = vals[offset : offset + n_classes * n_blocks * channels].reshape(v.shape)
    offset += n_classes * n_blocks * channels
    nob = vals[offset : offset + n_classes]
    return nw, nhb, nv, nob


def run_eval(
    spice_bin,
    netlist_path,
    data_path,
    x_eval,
    y_eval,
    w,
    hb,
    v,
    ob,
    blocks,
    batch_size,
    timeout,
    linear_output,
    softmax_output,
    local_activation,
    relu_clip,
    relu_leak=0.01,
    softplus_beta=10.0,
    hidden_synapse_mode="linear",
    readout_synapse_mode="linear",
    synapse_clip=1.0,
    softmax_temperature=1.0,
    readout_class_centering="none",
):
    correct = 0
    for start in range(0, len(y_eval), batch_size):
        x = x_eval[start : start + batch_size]
        y = y_eval[start : start + batch_size]
        netlist_path.write_text(
            prepare_local_feature_netlist(
                make_eval_netlist(
                    x,
                    w,
                    hb,
                    v,
                    ob,
                    blocks,
                    data_path,
                    linear_output,
                    softmax_output,
                    local_activation,
                    relu_clip,
                    relu_leak,
                    softplus_beta,
                    hidden_synapse_mode,
                    readout_synapse_mode,
                    synapse_clip,
                    softmax_temperature,
                    readout_class_centering,
                ),
                spice_bin,
            )
        )
        proc = run_simulator_netlist(spice_bin, netlist_path, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-3000:] or proc.stdout[-3000:])
        vals = read_output_row(spice_bin, netlist_path, data_path, len(y) * v.shape[0]).reshape(len(y), v.shape[0])
        correct += int((np.argmax(vals, axis=1) == y).sum())
    return correct / max(len(y_eval), 1)


def epoch_order_slice(order: np.ndarray, epoch_train_samples: int, epoch_train_offset: int, max_train_batches: int, batch_size: int) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if epoch_train_samples > 0:
        start = min(max(epoch_train_offset, 0), len(order))
        stop = min(start + epoch_train_samples, len(order))
        order = order[start:stop]
    elif epoch_train_offset > 0:
        start = min(epoch_train_offset, len(order))
        order = order[start:]
    if max_train_batches > 0:
        order = order[: max_train_batches * batch_size]
    return order


def skip_epoch_shuffles(rng: np.random.Generator, epochs: int, train_count: int) -> None:
    if epochs < 0:
        raise ValueError("epochs must be non-negative")
    for _ in range(epochs):
        order = np.arange(train_count)
        rng.shuffle(order)


def save_weight_checkpoint(path: Path, w: np.ndarray, hb: np.ndarray, v: np.ndarray, ob: np.ndarray, **metadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        local_weights=w,
        local_bias=hb,
        readout=v,
        output_bias=ob,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )


def best_accuracy(rows: list[dict]) -> float | None:
    values = [row.get("heldout_accuracy") for row in rows if row.get("heldout_accuracy") is not None]
    return None if not values else float(max(values))


def final_accuracy(rows: list[dict]) -> float | None:
    for row in reversed(rows):
        if row.get("heldout_accuracy") is not None:
            return float(row["heldout_accuracy"])
    return None


def plot_curve(df, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    if "heldout_accuracy" in df and df["heldout_accuracy"].notna().any():
        plt.plot(df["epoch"], df["heldout_accuracy"], marker="o")
    plt.xlabel("epoch")
    plt.ylabel("held-out accuracy")
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-samples", type=int, default=200)
    ap.add_argument("--test-samples", type=int, default=200)
    ap.add_argument("--image-size", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=4)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--channels", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="Advance the epoch shuffle RNG before training. Use when resuming from an init-weights checkpoint.",
    )
    ap.add_argument("--epoch-train-samples", type=int, default=0)
    ap.add_argument("--epoch-train-offset", type=int, default=0)
    ap.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="Cap train batches per epoch after epoch offset/sample slicing. Use with --init-weights to resume long runs in chunks.",
    )
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--linear-output", action="store_true")
    ap.add_argument("--softmax-output", action="store_true")
    ap.add_argument(
        "--local-activation",
        choices=[
            "tanh",
            "relu",
            "clipped-relu",
            "diff-clipped-relu",
            "differential-clipped-relu",
            "leaky-relu",
            "softplus",
            "softplus-relu",
        ],
        default="tanh",
    )
    ap.add_argument("--relu-clip", type=float, default=1.0)
    ap.add_argument("--relu-leak", type=float, default=0.01)
    ap.add_argument("--softplus-beta", type=float, default=10.0)
    ap.add_argument("--activation-derivative", choices=["exact", "stored-gate", "unity", "floor-exact"], default="exact")
    ap.add_argument("--derivative-floor", type=float, default=0.0)
    ap.add_argument("--derivative-gate-threshold", type=float, default=1e-6)
    ap.add_argument("--readout-feedback-mode", choices=["readout", "full-readout", "exact", "sign-readout", "sign", "clipped-readout", "clipped"], default="readout")
    ap.add_argument("--readout-feedback-clip", type=float, default=0.05)
    ap.add_argument("--softmax-negative-scale", type=float, default=1.0)
    ap.add_argument("--softmax-error-centering", choices=["none", "mean"], default="none")
    ap.add_argument("--softmax-temperature", type=float, default=1.0)
    synapse_modes = ["linear", "full", "ideal", "tanh-clipped", "smooth-clipped", "clipped", "hard-clipped", "bounded", "sign", "binary"]
    ap.add_argument("--hidden-synapse-mode", choices=synapse_modes, default="linear")
    ap.add_argument("--readout-synapse-mode", choices=synapse_modes, default="linear")
    ap.add_argument("--synapse-clip", type=float, default=1.0)
    ap.add_argument("--readout-class-centering", choices=["none", "mean"], default="none")
    ap.add_argument("--init-weights", default="")
    ap.add_argument(
        "--import-extra-readout-scale",
        type=float,
        default=0.0,
        help="Stddev for class readout weights into channels above 10 when importing a class-evidence checkpoint.",
    )
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument(
        "--checkpoint-every-batches",
        type=int,
        default=0,
        help="Write *_latest_weights.npz every N completed train batches. Zero writes only final/best checkpoints.",
    )
    ap.add_argument(
        "--skip-heldout-eval",
        action="store_true",
        help="Train and checkpoint without running held-out SPICE evaluation. Use for fast resumable chunks.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--simulator", default=None)
    ap.add_argument("--timeout", type=float, default=90)
    ap.add_argument("--tag", default="local_feature")
    args = ap.parse_args()
    if args.linear_output and args.softmax_output:
        raise ValueError("--linear-output and --softmax-output are mutually exclusive")
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
    if args.softmax_negative_scale < 0:
        raise ValueError("--softmax-negative-scale must be non-negative")
    if args.softmax_temperature <= 0:
        raise ValueError("--softmax-temperature must be positive")
    if args.synapse_clip <= 0:
        raise ValueError("--synapse-clip must be positive")
    if args.start_epoch < 0:
        raise ValueError("--start-epoch must be non-negative")
    if args.skip_heldout_eval and (args.eval_only or args.epochs == 0):
        raise ValueError("--skip-heldout-eval cannot be used with --eval-only or --epochs 0")

    stride = args.block_size if args.stride is None else args.stride
    blocks = block_indices(args.image_size, args.block_size, stride)
    x_train, y_train, x_test, y_test = load_mnist_sequence(args.train_samples, args.test_samples, args.image_size, args.seed)
    spice_bin, version = detect_spice(args.simulator)
    generated = ROOT / "spice/generated"
    results = ROOT / "spice/results"
    for directory in (generated, results):
        directory.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)
    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag)
    stem = f"spice_mnist_local_feature_{safe_tag}"
    netlist_path = generated / f"{stem}_step.cir"
    eval_netlist = generated / f"{stem}_eval.cir"
    data_path = results / f"{stem}_step.dat"
    latest_weights_path = results / f"{stem}_latest_weights.npz"
    progress_path = results / f"{stem}_progress.json"
    rng = np.random.default_rng(args.seed)
    w = rng.normal(0.0, 0.05, size=(len(blocks), args.channels, args.block_size * args.block_size))
    hb = np.zeros((len(blocks), args.channels))
    v = rng.normal(0.0, 0.05, size=(10, len(blocks), args.channels))
    ob = np.zeros(10)
    if args.init_weights:
        init = np.load(args.init_weights)
        expected_shapes = (
            (len(blocks), args.channels, args.block_size * args.block_size),
            (len(blocks), args.channels),
            (10, len(blocks), args.channels),
            (10,),
        )
        if {"local_weights", "local_bias", "readout", "output_bias"}.issubset(init.files):
            w = init["local_weights"]
            hb = init["local_bias"]
            v = init["readout"]
            ob = init["output_bias"]
            actual_shapes = (w.shape, hb.shape, v.shape, ob.shape)
            if actual_shapes != expected_shapes:
                raise ValueError(f"initial weight shapes {actual_shapes} do not match expected {expected_shapes}")
        elif {"weights", "local_bias", "gains", "output_bias"}.issubset(init.files):
            class_weights = init["weights"]
            class_bias = init["local_bias"]
            class_gains = init["gains"]
            class_ob = init["output_bias"]
            expected_class_shapes = (
                (10, len(blocks), args.block_size * args.block_size),
                (10, len(blocks)),
                (10, len(blocks)),
                (10,),
            )
            actual_class_shapes = (class_weights.shape, class_bias.shape, class_gains.shape, class_ob.shape)
            if actual_class_shapes != expected_class_shapes:
                raise ValueError(f"class-evidence checkpoint shapes {actual_class_shapes} do not match expected {expected_class_shapes}")
            if args.channels < 10:
                raise ValueError("importing a class-evidence checkpoint needs --channels >= 10")
            w[:, :10, :] = np.transpose(class_weights, (1, 0, 2))
            hb[:, :10] = class_bias.T
            v[:, :, :] = 0.0
            for k in range(10):
                v[k, :, k] = class_gains[k]
            if args.channels > 10 and args.import_extra_readout_scale > 0.0:
                v[:, :, 10:] = rng.normal(
                    0.0,
                    args.import_extra_readout_scale,
                    size=(10, len(blocks), args.channels - 10),
                )
            ob = class_ob
        else:
            raise ValueError(f"unrecognized checkpoint keys: {init.files}")
    rows = []
    best_acc = -1.0
    best_state = None
    t0 = time.perf_counter()
    completed_train_batches = 0
    latest_checkpoint_written = False
    if args.eval_only or args.epochs == 0:
        epoch_start = time.perf_counter()
        heldout = run_eval(
            spice_bin, eval_netlist, data_path, x_test, y_test,
            w, hb, v, ob, blocks, args.batch_size, args.timeout,
            args.linear_output,
            args.softmax_output,
            args.local_activation,
            args.relu_clip,
            args.relu_leak,
            args.softplus_beta,
            args.hidden_synapse_mode,
            args.readout_synapse_mode,
            args.synapse_clip,
            args.softmax_temperature,
            args.readout_class_centering,
        )
        rows.append({"epoch": 0, "heldout_accuracy": heldout, "epoch_wall_time_s": time.perf_counter() - epoch_start})
        best_acc = heldout
        best_state = (w.copy(), hb.copy(), v.copy(), ob.copy())
        print(json.dumps(rows[-1]), flush=True)
    else:
        skip_epoch_shuffles(rng, args.start_epoch, len(y_train))
        for local_epoch in range(args.epochs):
            epoch = args.start_epoch + local_epoch
            order = np.arange(len(y_train))
            rng.shuffle(order)
            order = epoch_order_slice(
                order,
                args.epoch_train_samples,
                args.epoch_train_offset,
                args.max_train_batches,
                args.batch_size,
            )
            epoch_start = time.perf_counter()
            for start in range(0, len(order), args.batch_size):
                idx = order[start : start + args.batch_size]
                w, hb, v, ob = run_train_batch(
                    spice_bin, netlist_path, data_path, x_train[idx], y_train[idx],
                    w, hb, v, ob, blocks, args.lr, args.timeout,
                    args.linear_output,
                    args.softmax_output,
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
                    args.softmax_negative_scale,
                    args.softmax_error_centering,
                    args.softmax_temperature,
                    args.readout_class_centering,
                )
                completed_train_batches += 1
                if args.checkpoint_every_batches > 0 and completed_train_batches % args.checkpoint_every_batches == 0:
                    save_weight_checkpoint(
                        latest_weights_path,
                        w,
                        hb,
                        v,
                        ob,
                        epoch=epoch + 1,
                        completed_train_batches=completed_train_batches,
                        epoch_batch_index=start // args.batch_size + 1,
                        epoch_train_offset=args.epoch_train_offset,
                        max_train_batches=args.max_train_batches,
                        train_samples=args.train_samples,
                        test_samples=args.test_samples,
                        image_size=args.image_size,
                        block_size=args.block_size,
                        stride=stride,
                        channels=args.channels,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        seed=args.seed,
                    )
                    latest_checkpoint_written = True
                    progress_path.write_text(
                        json.dumps(
                            {
                                "latest_weights": str(latest_weights_path),
                                "epoch": epoch + 1,
                                "completed_train_batches": completed_train_batches,
                                "epoch_batch_index": start // args.batch_size + 1,
                                "epoch_train_offset": args.epoch_train_offset,
                                "max_train_batches": args.max_train_batches,
                            },
                            indent=2,
                        )
                        + "\n"
                    )
            heldout = None
            if not args.skip_heldout_eval:
                heldout = run_eval(
                    spice_bin, eval_netlist, data_path, x_test, y_test,
                    w, hb, v, ob, blocks, args.batch_size, args.timeout,
                    args.linear_output,
                    args.softmax_output,
                    args.local_activation,
                    args.relu_clip,
                    args.relu_leak,
                    args.softplus_beta,
                    args.hidden_synapse_mode,
                    args.readout_synapse_mode,
                    args.synapse_clip,
                    args.softmax_temperature,
                    args.readout_class_centering,
                )
            row = {"epoch": epoch + 1, "heldout_accuracy": heldout, "epoch_wall_time_s": time.perf_counter() - epoch_start}
            rows.append(row)
            if heldout is not None and heldout > best_acc:
                best_acc = heldout
                best_state = (w.copy(), hb.copy(), v.copy(), ob.copy())
            print(json.dumps(row), flush=True)
    curve = pd.DataFrame(rows)
    curve_path = results / f"{stem}_learning_curve.csv"
    curve.to_csv(curve_path, index=False)
    fig = results / f"{stem}_learning_curve.png"
    plot_curve(curve, fig)
    weights_path = results / f"{stem}_final_weights.npz"
    best_weights_path = results / f"{stem}_best_weights.npz"
    save_weight_checkpoint(
        weights_path,
        w,
        hb,
        v,
        ob,
        completed_train_batches=completed_train_batches,
        checkpoint_kind="final",
    )
    if best_state is not None:
        bw, bhb, bv, bob = best_state
        save_weight_checkpoint(
            best_weights_path,
            bw,
            bhb,
            bv,
            bob,
            completed_train_batches=completed_train_batches,
            checkpoint_kind="best",
        )
    summary = {
        "simulator": version,
        "simulator_selector": args.simulator,
        "dataset": "MNIST train/test split, downsampled",
        "architecture": "local_feature_readout",
        "local_activation": args.local_activation,
        "relu_clip": args.relu_clip,
        "relu_leak": args.relu_leak,
        "softplus_beta": args.softplus_beta,
        "activation_derivative": args.activation_derivative,
        "derivative_floor": args.derivative_floor,
        "derivative_gate_threshold": args.derivative_gate_threshold,
        "readout_feedback_mode": args.readout_feedback_mode,
        "readout_feedback_clip": args.readout_feedback_clip,
        "softmax_negative_scale": args.softmax_negative_scale,
        "softmax_error_centering": args.softmax_error_centering,
        "softmax_temperature": args.softmax_temperature,
        "hidden_synapse_mode": args.hidden_synapse_mode,
        "readout_synapse_mode": args.readout_synapse_mode,
        "synapse_clip": args.synapse_clip,
        "readout_class_centering": args.readout_class_centering,
        "output_mode": "softmax_class_evidence" if args.softmax_output else ("linear_class_evidence" if args.linear_output else "tanh_class_evidence"),
        "image_size": args.image_size,
        "block_size": args.block_size,
        "stride": stride,
        "blocks": len(blocks),
        "channels": args.channels,
        "local": True,
        "inputs": int(x_train.shape[1]),
        "classes": 10,
        "train_samples": args.train_samples,
        "epoch_train_samples": int(args.epoch_train_samples) if args.epoch_train_samples > 0 else int(args.train_samples),
        "epoch_train_offset": int(args.epoch_train_offset),
        "max_train_batches": int(args.max_train_batches),
        "test_samples": args.test_samples,
        "epochs": args.epochs,
        "start_epoch": args.start_epoch,
        "eval_only": bool(args.eval_only),
        "skip_heldout_eval": bool(args.skip_heldout_eval),
        "batch_size": args.batch_size,
        "checkpoint_every_batches": int(args.checkpoint_every_batches),
        "lr": args.lr,
        "init_weights": args.init_weights,
        "import_extra_readout_scale": args.import_extra_readout_scale,
        "linear_output": bool(args.linear_output),
        "softmax_output": bool(args.softmax_output),
        "wall_time_s": time.perf_counter() - t0,
        "learning_curve": str(curve_path),
        "figure": str(fig),
        "final_weights": str(weights_path),
        "best_weights": str(best_weights_path) if best_state is not None else None,
        "latest_weights": str(latest_weights_path) if latest_checkpoint_written else None,
        "completed_train_batches": int(completed_train_batches),
        "heldout_test_accuracy": final_accuracy(rows),
        "best_heldout_accuracy": best_accuracy(rows),
        "note": "Local feature/readout batch-op all-SPICE training with SPICE-computed backprop updates.",
    }
    out = results / f"{stem}_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
