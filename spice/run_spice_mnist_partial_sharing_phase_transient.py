from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_spice_mnist_batch_op_train import read_wrdata_row
from run_spice_mnist_local_block_batch_op_train import block_indices
from run_spice_mnist_local_feature_phase_transient import (
    make_phase_schedule,
    parse_measured_vector,
    phase_pwl,
    sample_source_pwl,
    sanitize_tag,
    tanh_expr,
    target_matrix,
)
from run_spice_mnist_train import load_mnist_sequence
from run_spice_sweep import ROOT, detect_spice, run_tiny_test


PartialState = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def load_partial_checkpoint(path: Path, n_blocks: int, block_len: int) -> PartialState:
    data = np.load(path, allow_pickle=False)
    keys = {
        "shared_local_weights",
        "shared_local_bias",
        "private_local_weights",
        "private_local_bias",
        "readout",
        "output_bias",
    }
    if not keys.issubset(data.files):
        raise ValueError(f"expected partial-sharing checkpoint keys, got {data.files}")
    shared_w = data["shared_local_weights"].astype(float, copy=True)
    shared_bias = data["shared_local_bias"].astype(float, copy=True)
    private_w = data["private_local_weights"].astype(float, copy=True)
    private_bias = data["private_local_bias"].astype(float, copy=True)
    readout = data["readout"].astype(float, copy=True)
    output_bias = data["output_bias"].astype(float, copy=True)
    if shared_w.ndim != 2 or shared_w.shape[1] != block_len:
        raise ValueError(f"shared_local_weights shape {shared_w.shape} incompatible with block_len={block_len}")
    if shared_bias.shape != (shared_w.shape[0],):
        raise ValueError("shared_local_bias shape does not match shared_local_weights")
    if private_w.ndim != 3 or private_w.shape[0] != n_blocks or private_w.shape[2] != block_len:
        raise ValueError(f"private_local_weights shape {private_w.shape} incompatible with blocks/block_len")
    if private_bias.shape != private_w.shape[:2]:
        raise ValueError("private_local_bias shape does not match private_local_weights")
    channels = shared_w.shape[0] + private_w.shape[1]
    if readout.shape != (10, n_blocks, channels):
        raise ValueError(f"readout shape {readout.shape} incompatible with expected {(10, n_blocks, channels)}")
    if output_bias.shape != (10,):
        raise ValueError("output_bias must have shape (10,)")
    return shared_w, shared_bias, private_w, private_bias, readout, output_bias


def expand_local_weights(state: PartialState, n_blocks: int) -> tuple[np.ndarray, np.ndarray]:
    shared_w, shared_bias, private_w, private_bias, _readout, _output_bias = state
    weights = []
    biases = []
    if shared_w.shape[0]:
        weights.append(np.repeat(shared_w[None, :, :], n_blocks, axis=0))
        biases.append(np.repeat(shared_bias[None, :], n_blocks, axis=0))
    if private_w.shape[1]:
        weights.append(private_w)
        biases.append(private_bias)
    return np.concatenate(weights, axis=1), np.concatenate(biases, axis=1)


def save_partial_checkpoint(path: Path, state: PartialState, n_blocks: int) -> None:
    shared_w, shared_bias, private_w, private_bias, readout, output_bias = state
    local_w, local_bias = expand_local_weights(state, n_blocks)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        shared_local_weights=shared_w,
        shared_local_bias=shared_bias,
        private_local_weights=private_w,
        private_local_bias=private_bias,
        local_weights=local_w,
        local_bias=local_bias,
        readout=readout,
        output_bias=output_bias,
        shared_channels=np.array(shared_w.shape[0], dtype=np.int64),
        private_channels=np.array(private_w.shape[1], dtype=np.int64),
        weight_sharing="partial_shared_kernel",
    )


def block_tensor_np(x: np.ndarray, blocks: list[list[int]]) -> np.ndarray:
    return np.stack([x[:, idxs] for idxs in blocks], axis=1)


def fast_forward_np(x: np.ndarray, state: PartialState, blocks: list[list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shared_w, shared_bias, private_w, private_bias, readout, output_bias = state
    xb = block_tensor_np(x, blocks)
    features = []
    if shared_w.shape[0]:
        features.append(np.tanh(np.einsum("nbp,cp->nbc", xb, shared_w) + shared_bias[None, None, :]))
    if private_w.shape[1]:
        features.append(np.tanh(np.einsum("nbp,bcp->nbc", xb, private_w) + private_bias[None, :, :]))
    h = np.concatenate(features, axis=2)
    score = np.einsum("nbc,kbc->nk", h, readout) + output_bias
    y = np.tanh(score)
    return h, score, y


def fast_update_np(x: np.ndarray, labels: np.ndarray, state: PartialState, blocks: list[list[int]], lr: float) -> PartialState:
    shared_w, shared_bias, private_w, private_bias, readout, output_bias = state
    xb = block_tensor_np(x, blocks)
    h, _score, y = fast_forward_np(x, state, blocks)
    targets = -np.ones_like(y)
    targets[np.arange(len(labels)), labels.astype(int)] = 1.0
    d = (targets - y) * (1.0 - y * y)
    dh = np.einsum("nk,kbc->nbc", d, readout) * (1.0 - h * h)
    batch = max(len(labels), 1)
    shared_channels = shared_w.shape[0]
    if shared_channels:
        shared_dh = dh[:, :, :shared_channels]
        shared_w = shared_w + lr * np.einsum("nbc,nbp->cp", shared_dh, xb) / batch
        shared_bias = shared_bias + lr * np.sum(shared_dh, axis=(0, 1)) / batch
    if private_w.shape[1]:
        private_dh = dh[:, :, shared_channels:]
        private_w = private_w + lr * np.einsum("nbc,nbp->bcp", private_dh, xb) / batch
        private_bias = private_bias + lr * np.mean(private_dh, axis=0)
    readout = readout + lr * np.einsum("nk,nbc->kbc", d, h) / batch
    output_bias = output_bias + lr * np.mean(d, axis=0)
    return shared_w, shared_bias, private_w, private_bias, readout, output_bias


def run_fast_reference_chunk(
    x_chunk: np.ndarray,
    y_chunk: np.ndarray,
    state: PartialState,
    blocks: list[list[int]],
    lr: float,
    batch_size: int,
    updates: int,
) -> PartialState:
    next_state = tuple(arr.copy() for arr in state)
    for update in range(updates):
        start = update * batch_size
        stop = start + batch_size
        next_state = fast_update_np(x_chunk[start:stop], y_chunk[start:stop], next_state, blocks, lr)
    return next_state


def state_metrics(ref: PartialState, got: PartialState) -> dict[str, float]:
    names = [
        "shared_local_weights",
        "shared_local_bias",
        "private_local_weights",
        "private_local_bias",
        "readout",
        "output_bias",
    ]
    metrics: dict[str, float] = {}
    diffs = []
    for name, a, b in zip(names, ref, got):
        diff = np.asarray(b) - np.asarray(a)
        diffs.append(diff.ravel())
        metrics[f"{name}_max_abs_diff"] = float(np.max(np.abs(diff))) if diff.size else 0.0
        metrics[f"{name}_mean_abs_diff"] = float(np.mean(np.abs(diff))) if diff.size else 0.0
        metrics[f"{name}_rms_diff"] = float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0
    all_diff = np.concatenate(diffs)
    metrics["state_max_abs_diff"] = float(np.max(np.abs(all_diff)))
    metrics["state_mean_abs_diff"] = float(np.mean(np.abs(all_diff)))
    metrics["state_rms_diff"] = float(np.sqrt(np.mean(all_diff * all_diff)))
    return metrics


def make_partial_phase_netlist(
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    state: PartialState,
    blocks: list[list[int]],
    lr: float,
    out_path: Path,
    batch_size: int,
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
    output_mode: str,
) -> tuple[str, int, float]:
    shared_w, shared_bias, private_w, private_bias, readout, output_bias = state
    total_samples = x_batch.shape[0]
    if total_samples != batch_size * updates:
        raise ValueError("x_batch length must equal batch_size * updates")
    n_blocks = len(blocks)
    shared_channels = shared_w.shape[0]
    private_channels = private_w.shape[1]
    channels = shared_channels + private_channels
    block_len = shared_w.shape[1] if shared_channels else private_w.shape[2]
    n_classes = readout.shape[0]
    phases, sample_starts, t_stop = make_phase_schedule(batch_size, updates, phase, gap)
    tau = phase / settle_ratio
    targets = target_matrix(y_batch, n_classes)
    lines = [
        "* Phase-resolved transient partial-sharing local-feature/readout training deck.",
        "* Shared feature kernels, private feature kernels, readout weights, and biases are capacitor voltages.",
        "* Shared gradient caps sum over block positions before one shared apply pulse.",
        f".param LR={lr:.12g}",
        f".param BS={batch_size}",
        f".param CW={cw:.12g}",
        f".param CSTATE={cstate:.12g}",
        f".param CGRAD={cgrad:.12g}",
        f".param RLEAK={rleak:.12g}",
        f".param TAU={tau:.12g}",
        f".param TPHASE={phase:.12g}",
        "",
        f"Vpact pact 0 {phase_pwl(phases['act'], t_stop, edge)}",
        f"Vpscore pscore 0 {phase_pwl(phases['score'], t_stop, edge)}",
        f"Vperr perr 0 {phase_pwl(phases['err'], t_stop, edge)}",
        f"Vpbwd pbwd 0 {phase_pwl(phases['bwd'], t_stop, edge)}",
        f"Vpacc pacc 0 {phase_pwl(phases['acc'], t_stop, edge)}",
        f"Vpapply papply 0 {phase_pwl(phases['apply'], t_stop, edge)}",
        f"Vpclear pclear 0 {phase_pwl(phases['clear'], t_stop, edge)}",
        "",
    ]
    for i in range(x_batch.shape[1]):
        lines.append(f"Vpix{i} pix{i} 0 {sample_source_pwl(x_batch[:, i], sample_starts, t_stop, edge)}")
    for k in range(n_classes):
        lines.append(f"Vtarget{k} target{k} 0 {sample_source_pwl(targets[:, k], sample_starts, t_stop, edge)}")
    lines.append("")
    for c in range(shared_channels):
        for p in range(block_len):
            lines.append(f"Csw{c}_{p} sw{c}_{p} 0 {{CW}} IC={shared_w[c, p]:.12g}")
            lines.append(f"Rsw{c}_{p} sw{c}_{p} 0 {{RLEAK}}")
            lines.append(f"Cgsw{c}_{p} gsw{c}_{p} 0 {{CGRAD}} IC=0")
        lines.append(f"Cshb{c} shb{c} 0 {{CW}} IC={shared_bias[c]:.12g}")
        lines.append(f"Rshb{c} shb{c} 0 {{RLEAK}}")
        lines.append(f"Cgshb{c} gshb{c} 0 {{CGRAD}} IC=0")
    for b in range(n_blocks):
        for pc in range(private_channels):
            for p in range(block_len):
                lines.append(f"Cpw{b}_{pc}_{p} pw{b}_{pc}_{p} 0 {{CW}} IC={private_w[b, pc, p]:.12g}")
                lines.append(f"Rpw{b}_{pc}_{p} pw{b}_{pc}_{p} 0 {{RLEAK}}")
                lines.append(f"Cgpw{b}_{pc}_{p} gpw{b}_{pc}_{p} 0 {{CGRAD}} IC=0")
            lines.append(f"Cphb{b}_{pc} phb{b}_{pc} 0 {{CW}} IC={private_bias[b, pc]:.12g}")
            lines.append(f"Rphb{b}_{pc} phb{b}_{pc} 0 {{RLEAK}}")
            lines.append(f"Cgphb{b}_{pc} gphb{b}_{pc} 0 {{CGRAD}} IC=0")
        for c in range(channels):
            lines.append(f"Ch{b}_{c} h{b}_{c} 0 {{CSTATE}} IC=0")
            lines.append(f"Cdh{b}_{c} dh{b}_{c} 0 {{CSTATE}} IC=0")
    for k in range(n_classes):
        for b in range(n_blocks):
            for c in range(channels):
                lines.append(f"Cv{k}_{b}_{c} v{k}_{b}_{c} 0 {{CW}} IC={readout[k, b, c]:.12g}")
                lines.append(f"Rv{k}_{b}_{c} v{k}_{b}_{c} 0 {{RLEAK}}")
                lines.append(f"Cgv{k}_{b}_{c} gv{k}_{b}_{c} 0 {{CGRAD}} IC=0")
        lines.append(f"Cob{k} ob{k} 0 {{CW}} IC={output_bias[k]:.12g}")
        lines.append(f"Rob{k} ob{k} 0 {{RLEAK}}")
        lines.append(f"Cgob{k} gob{k} 0 {{CGRAD}} IC=0")
        lines.append(f"Cscore{k} score{k} 0 {{CSTATE}} IC=0")
        lines.append(f"Cd{k} d{k} 0 {{CSTATE}} IC=0")
    lines.append("")
    for b, idxs in enumerate(blocks):
        for c in range(shared_channels):
            terms = [f"V(sw{c}_{p})*V(pix{idx})" for p, idx in enumerate(idxs)]
            terms.append(f"V(shb{c})")
            lines.append(f"Bstore_h{b}_{c} h{b}_{c} 0 I = V(pact)*{{CSTATE}}/{{TAU}}*(V(h{b}_{c})-({tanh_expr(' + '.join(terms))}))")
        for pc in range(private_channels):
            c = shared_channels + pc
            terms = [f"V(pw{b}_{pc}_{p})*V(pix{idx})" for p, idx in enumerate(idxs)]
            terms.append(f"V(phb{b}_{pc})")
            lines.append(f"Bstore_h{b}_{c} h{b}_{c} 0 I = V(pact)*{{CSTATE}}/{{TAU}}*(V(h{b}_{c})-({tanh_expr(' + '.join(terms))}))")
    for k in range(n_classes):
        score_terms = [f"V(v{k}_{b}_{c})*V(h{b}_{c})" for b in range(n_blocks) for c in range(channels)]
        score_terms.append(f"V(ob{k})")
        lines.append(f"Bscore{k} scorecalc{k} 0 V = " + " + ".join(score_terms))
        lines.append(f"Bstore_score{k} score{k} 0 I = V(pscore)*{{CSTATE}}/{{TAU}}*(V(score{k})-V(scorecalc{k}))")
        y_expr = tanh_expr(f"V(score{k})")
        lines.append(f"By{k} y{k} 0 V = {y_expr}")
        delta_expr = f"(V(target{k})-({y_expr}))*(1-({y_expr})*({y_expr}))"
        lines.append(f"Bstore_d{k} d{k} 0 I = V(perr)*{{CSTATE}}/{{TAU}}*(V(d{k})-({delta_expr}))")
    lines.append("")
    for b, idxs in enumerate(blocks):
        for c in range(channels):
            feedback = " + ".join(f"V(v{k}_{b}_{c})*V(d{k})" for k in range(n_classes))
            local_delta = f"({feedback})*(1-V(h{b}_{c})*V(h{b}_{c}))"
            lines.append(f"Bstore_dh{b}_{c} dh{b}_{c} 0 I = V(pbwd)*{{CSTATE}}/{{TAU}}*(V(dh{b}_{c})-({local_delta}))")
    for c in range(shared_channels):
        for p in range(block_len):
            grad = " + ".join(f"V(dh{b}_{c})*V(pix{blocks[b][p]})" for b in range(n_blocks))
            lines.append(f"Bacc_sw{c}_{p} gsw{c}_{p} 0 I = -V(pacc)*{{CGRAD}}/{{TPHASE}}*({grad})")
            lines.append(f"Bupd_sw{c}_{p} sw{c}_{p} 0 I = -V(papply)*{{CW}}*{{LR}}/({{BS}}*{{TPHASE}})*V(gsw{c}_{p})")
            lines.append(f"Bclear_gsw{c}_{p} gsw{c}_{p} 0 I = V(pclear)*{{CGRAD}}/{{TAU}}*V(gsw{c}_{p})")
        grad_b = " + ".join(f"V(dh{b}_{c})" for b in range(n_blocks))
        lines.append(f"Bacc_shb{c} gshb{c} 0 I = -V(pacc)*{{CGRAD}}/{{TPHASE}}*({grad_b})")
        lines.append(f"Bupd_shb{c} shb{c} 0 I = -V(papply)*{{CW}}*{{LR}}/({{BS}}*{{TPHASE}})*V(gshb{c})")
        lines.append(f"Bclear_gshb{c} gshb{c} 0 I = V(pclear)*{{CGRAD}}/{{TAU}}*V(gshb{c})")
    for b, idxs in enumerate(blocks):
        for pc in range(private_channels):
            c = shared_channels + pc
            for p, idx in enumerate(idxs):
                grad = f"V(dh{b}_{c})*V(pix{idx})"
                lines.append(f"Bacc_pw{b}_{pc}_{p} gpw{b}_{pc}_{p} 0 I = -V(pacc)*{{CGRAD}}/{{TPHASE}}*({grad})")
                lines.append(f"Bupd_pw{b}_{pc}_{p} pw{b}_{pc}_{p} 0 I = -V(papply)*{{CW}}*{{LR}}/({{BS}}*{{TPHASE}})*V(gpw{b}_{pc}_{p})")
                lines.append(f"Bclear_gpw{b}_{pc}_{p} gpw{b}_{pc}_{p} 0 I = V(pclear)*{{CGRAD}}/{{TAU}}*V(gpw{b}_{pc}_{p})")
            lines.append(f"Bacc_phb{b}_{pc} gphb{b}_{pc} 0 I = -V(pacc)*{{CGRAD}}/{{TPHASE}}*V(dh{b}_{c})")
            lines.append(f"Bupd_phb{b}_{pc} phb{b}_{pc} 0 I = -V(papply)*{{CW}}*{{LR}}/({{BS}}*{{TPHASE}})*V(gphb{b}_{pc})")
            lines.append(f"Bclear_gphb{b}_{pc} gphb{b}_{pc} 0 I = V(pclear)*{{CGRAD}}/{{TAU}}*V(gphb{b}_{pc})")
    for k in range(n_classes):
        for b in range(n_blocks):
            for c in range(channels):
                grad = f"V(d{k})*V(h{b}_{c})"
                lines.append(f"Bacc_v{k}_{b}_{c} gv{k}_{b}_{c} 0 I = -V(pacc)*{{CGRAD}}/{{TPHASE}}*({grad})")
                lines.append(f"Bupd_v{k}_{b}_{c} v{k}_{b}_{c} 0 I = -V(papply)*{{CW}}*{{LR}}/({{BS}}*{{TPHASE}})*V(gv{k}_{b}_{c})")
                lines.append(f"Bclear_gv{k}_{b}_{c} gv{k}_{b}_{c} 0 I = V(pclear)*{{CGRAD}}/{{TAU}}*V(gv{k}_{b}_{c})")
        lines.append(f"Bacc_ob{k} gob{k} 0 I = -V(pacc)*{{CGRAD}}/{{TPHASE}}*V(d{k})")
        lines.append(f"Bupd_ob{k} ob{k} 0 I = -V(papply)*{{CW}}*{{LR}}/({{BS}}*{{TPHASE}})*V(gob{k})")
        lines.append(f"Bclear_gob{k} gob{k} 0 I = V(pclear)*{{CGRAD}}/{{TAU}}*V(gob{k})")
    vectors = [f"V(sw{c}_{p})" for c in range(shared_channels) for p in range(block_len)]
    vectors += [f"V(shb{c})" for c in range(shared_channels)]
    vectors += [f"V(pw{b}_{pc}_{p})" for b in range(n_blocks) for pc in range(private_channels) for p in range(block_len)]
    vectors += [f"V(phb{b}_{pc})" for b in range(n_blocks) for pc in range(private_channels)]
    vectors += [f"V(v{k}_{b}_{c})" for k in range(n_classes) for b in range(n_blocks) for c in range(channels)]
    vectors += [f"V(ob{k})" for k in range(n_classes)]
    vectors += [f"V(y{k})" for k in range(n_classes)]
    if output_mode not in {"wrdata", "measure"}:
        raise ValueError("output_mode must be 'wrdata' or 'measure'")
    measure_time = max(0.0, t_stop - transient_step)
    lines += ["", ".options method=gear maxord=2", ".control", f"tran {transient_step:.12g} {t_stop:.12g} uic"]
    if output_mode == "wrdata":
        lines.append(f"wrdata {out_path} " + " ".join(vectors))
    else:
        for i, vec in enumerate(vectors):
            lines.append(f"meas tran m{i:05d} FIND {vec} AT={measure_time:.12g}")
    lines += [".endc", ".end", ""]
    return "\n".join(lines), len(vectors), t_stop


def unpack_partial_state(vals: np.ndarray, state: PartialState) -> tuple[PartialState, np.ndarray]:
    shared_w, shared_bias, private_w, private_bias, readout, output_bias = state
    offset = 0
    next_shared_w = vals[offset : offset + shared_w.size].reshape(shared_w.shape)
    offset += shared_w.size
    next_shared_bias = vals[offset : offset + shared_bias.size].reshape(shared_bias.shape)
    offset += shared_bias.size
    next_private_w = vals[offset : offset + private_w.size].reshape(private_w.shape)
    offset += private_w.size
    next_private_bias = vals[offset : offset + private_bias.size].reshape(private_bias.shape)
    offset += private_bias.size
    next_readout = vals[offset : offset + readout.size].reshape(readout.shape)
    offset += readout.size
    next_output_bias = vals[offset : offset + output_bias.size].reshape(output_bias.shape)
    offset += output_bias.size
    y = vals[offset : offset + readout.shape[0]]
    return (next_shared_w, next_shared_bias, next_private_w, next_private_bias, next_readout, next_output_bias), y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-samples", type=int, default=200)
    ap.add_argument("--image-size", type=int, default=7)
    ap.add_argument("--block-size", type=int, default=3)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--updates", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--init-weights", required=True)
    ap.add_argument("--phase", type=float, default=1e-9)
    ap.add_argument("--gap", type=float, default=0.2e-9)
    ap.add_argument("--edge", type=float, default=10e-12)
    ap.add_argument("--settle-ratio", type=float, default=80.0)
    ap.add_argument("--transient-step", type=float, default=25e-12)
    ap.add_argument("--cw", type=float, default=1e-12)
    ap.add_argument("--cstate", type=float, default=1e-12)
    ap.add_argument("--cgrad", type=float, default=1e-12)
    ap.add_argument("--rleak", type=float, default=1e18)
    ap.add_argument("--final-measures", action="store_true")
    ap.add_argument("--tag", default="partial_sharing_phase")
    args = ap.parse_args()

    if args.batch_size <= 0 or args.updates <= 0:
        raise ValueError("--batch-size and --updates must be positive")
    if args.phase <= 0 or args.settle_ratio <= 0:
        raise ValueError("--phase and --settle-ratio must be positive")

    stride = args.block_size if args.stride is None else args.stride
    blocks = block_indices(args.image_size, args.block_size, stride)
    total_samples = args.batch_size * args.updates
    if args.train_samples < total_samples:
        raise ValueError("--train-samples must cover --batch-size * --updates")
    x_train, y_train, _x_test, _y_test = load_mnist_sequence(args.train_samples, 1, args.image_size, args.seed)
    x_batch = x_train[:total_samples]
    y_batch = y_train[:total_samples]
    state = load_partial_checkpoint(Path(args.init_weights), len(blocks), args.block_size * args.block_size)
    shared_channels = state[0].shape[0]
    private_channels = state[2].shape[1]
    channels = shared_channels + private_channels

    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    results = ROOT / "spice/results"
    generated.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)

    stem = f"spice_mnist_partial_sharing_phase_{sanitize_tag(args.tag)}"
    phase_netlist = generated / f"{stem}.cir"
    phase_data = results / f"{stem}.dat"
    netlist, n_vec, t_stop = make_partial_phase_netlist(
        x_batch,
        y_batch,
        state,
        blocks,
        args.lr,
        phase_data,
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
        "measure" if args.final_measures else "wrdata",
    )
    phase_netlist.write_text(netlist)

    t0 = time.perf_counter()
    proc = subprocess.run([spice_bin, "-b", str(phase_netlist)], text=True, capture_output=True, timeout=args.timeout)
    phase_wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-3000:] or proc.stdout[-3000:])
    vals = parse_measured_vector(proc.stdout + "\n" + proc.stderr, n_vec) if args.final_measures else read_wrdata_row(phase_data, n_vec)
    phase_state, phase_y = unpack_partial_state(vals, state)

    ref_state = run_fast_reference_chunk(x_batch, y_batch, state, blocks, args.lr, args.batch_size, args.updates)
    metrics = state_metrics(ref_state, phase_state)
    final_weights_path = results / f"{stem}_final_weights.npz"
    metrics_path = results / f"{stem}_equivalence_metrics.csv"
    save_partial_checkpoint(final_weights_path, phase_state, len(blocks))
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    summary = {
        "simulator": version,
        "architecture": "phase_resolved_transient_partial_sharing_local_feature_readout",
        "status": "one_batch_equivalence_check" if args.updates == 1 else "multi_update_equivalence_check",
        "image_size": args.image_size,
        "block_size": args.block_size,
        "stride": stride,
        "blocks": len(blocks),
        "shared_channels": int(shared_channels),
        "private_channels": int(private_channels),
        "channels": int(channels),
        "classes": 10,
        "train_samples": args.train_samples,
        "batch_size": args.batch_size,
        "updates": args.updates,
        "total_samples": total_samples,
        "lr": args.lr,
        "init_weights": args.init_weights,
        "phase_netlist": str(phase_netlist),
        "phase_data": str(phase_data),
        "final_weights": str(final_weights_path),
        "equivalence_metrics": str(metrics_path),
        "phase_wall_time_s": phase_wall,
        "t_stop_s": t_stop,
        "transient_step_s": args.transient_step,
        "phase_s": args.phase,
        "settle_ratio": args.settle_ratio,
        "output_mode": "measure" if args.final_measures else "wrdata",
        "persistent_state": (
            "shared local kernel weights/biases, private local weights/biases, class readout weights, "
            "and output biases are capacitor voltages with checkpoint ICs"
        ),
        "temporary_state": (
            "feature activations, class scores, class deltas, hidden/backward feature deltas, and "
            "shared/private/readout gradient accumulators are capacitor voltages"
        ),
        "python_role": "Python generates guiding waveforms and compares final state; it does not carry training state during the transient run.",
        "note": (
            "Partial-sharing phase-transient all-SPICE update check. Shared kernel gradient capacitors "
            "sum across block positions before one shared apply pulse."
        ),
        **metrics,
    }
    np.savez_compressed(results / f"{stem}_y.npz", y=phase_y)
    summary_path = results / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
