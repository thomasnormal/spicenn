from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from run_spice_mnist_local_block_batch_op_train import block_indices
from run_spice_mnist_local_feature_phase_transient import load_or_init_weights, parse_probe_update_list, sanitize_tag
from run_spice_mnist_train import load_mnist_sequence
from run_spice_sweep import ROOT


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


def local_activation_deriv_np(
    preactivation: np.ndarray,
    activation: np.ndarray,
    mode: str,
    relu_clip: float,
    derivative_mode: str,
    derivative_floor: float,
    derivative_gate_threshold: float,
    relu_leak: float,
    softplus_beta: float,
) -> np.ndarray:
    if derivative_mode == "unity":
        deriv = np.ones_like(preactivation)
    elif derivative_mode == "stored-gate":
        threshold = max(float(derivative_gate_threshold), 0.0)
        if mode == "tanh":
            deriv = 1.0 - activation * activation
        elif mode in {"diff-clipped-relu", "differential-clipped-relu", "diff_clipped_relu"}:
            deriv = (np.abs(activation) > threshold).astype(float)
        else:
            deriv = (activation > threshold).astype(float)
    else:
        if derivative_mode not in {"exact", "floor-exact"}:
            raise ValueError(f"unknown activation derivative mode {derivative_mode!r}")
        if mode == "tanh":
            deriv = 1.0 - activation * activation
        elif mode == "relu":
            deriv = (preactivation >= 0.0).astype(float)
        elif mode in {"clipped-relu", "clipped_relu"}:
            deriv = ((preactivation >= 0.0) & (preactivation <= relu_clip)).astype(float)
        elif mode in {"diff-clipped-relu", "differential-clipped-relu", "diff_clipped_relu"}:
            deriv = (np.abs(preactivation) <= relu_clip).astype(float)
        elif mode in {"leaky-relu", "leaky_relu"}:
            deriv = np.where(preactivation >= 0.0, 1.0, relu_leak)
        elif mode in {"softplus", "softplus-relu", "softplus_relu"}:
            beta = max(float(softplus_beta), 1e-12)
            deriv = 1.0 / (1.0 + np.exp(-beta * preactivation))
        else:
            raise ValueError(f"unknown local activation {mode!r}")
    floor = derivative_floor if derivative_mode in {"floor-exact", "stored-gate"} else 0.0
    if floor <= 0.0:
        return deriv
    if floor >= 1.0:
        return np.ones_like(deriv)
    return floor + (1.0 - floor) * deriv


def output_activation_np(score: np.ndarray, linear_output: bool, softmax_output: bool) -> np.ndarray:
    if linear_output:
        return score
    if softmax_output:
        shifted = score - np.max(score, axis=1, keepdims=True)
        exp_score = np.exp(shifted)
        return exp_score / np.sum(exp_score, axis=1, keepdims=True)
    return np.tanh(score)


def forward_np(
    x: np.ndarray,
    state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    blocks: list[list[int]],
    *,
    local_activation: str,
    relu_clip: float,
    relu_leak: float,
    softplus_beta: float,
    hidden_synapse_mode: str,
    readout_synapse_mode: str,
    synapse_clip: float,
    linear_output: bool,
    softmax_output: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w, hb, readout, output_bias = state
    xb = block_tensor_np(x, blocks)
    eff_w = synapse_transfer_np(w, hidden_synapse_mode, synapse_clip)
    pre = np.einsum("nbp,bcp->nbc", xb, eff_w) + hb
    h = local_activation_np(pre, local_activation, relu_clip, relu_leak, softplus_beta)
    eff_readout = synapse_transfer_np(readout, readout_synapse_mode, synapse_clip)
    score = np.einsum("nbc,kbc->nk", h, eff_readout) + output_bias
    y = output_activation_np(score, linear_output, softmax_output)
    return xb, pre, h, score, y


def target_matrix(labels: np.ndarray, n_classes: int, softmax_output: bool) -> np.ndarray:
    targets = np.zeros((len(labels), n_classes)) if softmax_output else -np.ones((len(labels), n_classes))
    targets[np.arange(len(labels)), labels.astype(int)] = 1.0
    return targets


def update_np(
    x: np.ndarray,
    labels: np.ndarray,
    state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    blocks: list[list[int]],
    *,
    lr: float,
    linear_output: bool,
    softmax_output: bool,
    local_activation: str,
    relu_clip: float,
    activation_derivative: str,
    derivative_floor: float,
    derivative_gate_threshold: float,
    readout_feedback_mode: str,
    readout_feedback_clip: float,
    relu_leak: float,
    softplus_beta: float,
    hidden_synapse_mode: str,
    readout_synapse_mode: str,
    synapse_clip: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w, hb, readout, output_bias = state
    xb, pre, h, _score, y = forward_np(
        x,
        state,
        blocks,
        local_activation=local_activation,
        relu_clip=relu_clip,
        relu_leak=relu_leak,
        softplus_beta=softplus_beta,
        hidden_synapse_mode=hidden_synapse_mode,
        readout_synapse_mode=readout_synapse_mode,
        synapse_clip=synapse_clip,
        linear_output=linear_output,
        softmax_output=softmax_output,
    )
    targets = target_matrix(labels, readout.shape[0], softmax_output)
    if softmax_output or linear_output:
        d = targets - y
    else:
        d = (targets - y) * (1.0 - y * y)
    eff_readout = synapse_transfer_np(readout, readout_synapse_mode, synapse_clip)
    if readout_feedback_mode in {"sign-readout", "sign"}:
        feedback_readout = eff_readout / (np.abs(eff_readout) + 1e-9)
    elif readout_feedback_mode in {"clipped-readout", "clipped"}:
        clip = max(float(readout_feedback_clip), 1e-12)
        feedback_readout = clip * np.tanh(eff_readout / clip)
    elif readout_feedback_mode in {"readout", "full-readout", "exact"}:
        feedback_readout = eff_readout
    else:
        raise ValueError(f"unknown readout feedback mode {readout_feedback_mode!r}")
    deriv = local_activation_deriv_np(
        pre,
        h,
        local_activation,
        relu_clip,
        activation_derivative,
        derivative_floor,
        derivative_gate_threshold,
        relu_leak,
        softplus_beta,
    )
    dh = np.einsum("nk,kbc->nbc", d, feedback_readout) * deriv
    batch = max(len(labels), 1)
    return (
        w + lr * np.einsum("nbc,nbp->bcp", dh, xb) / batch,
        hb + lr * np.mean(dh, axis=0),
        readout + lr * np.einsum("nk,nbc->kbc", d, h) / batch,
        output_bias + lr * np.mean(d, axis=0),
    )


def accuracy_np(
    x: np.ndarray,
    labels: np.ndarray,
    state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    blocks: list[list[int]],
    batch_size: int,
    **forward_kwargs,
) -> float:
    correct = 0
    for start in range(0, len(labels), batch_size):
        _xb, _pre, _h, _score, y = forward_np(x[start : start + batch_size], state, blocks, **forward_kwargs)
        correct += int(np.sum(np.argmax(y, axis=1) == labels[start : start + batch_size]))
    return correct / max(len(labels), 1)


def run_online(
    x_train: np.ndarray,
    y_train: np.ndarray,
    state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    blocks: list[list[int]],
    probe_updates: Iterable[int],
    **update_kwargs,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    probes = set(probe_updates)
    probe_states = {}
    next_state = tuple(arr.copy() for arr in state)
    for update in range(1, len(y_train) + 1):
        next_state = update_np(x_train[update - 1 : update], y_train[update - 1 : update], next_state, blocks, **update_kwargs)
        if update in probes:
            probe_states[update] = tuple(arr.copy() for arr in next_state)
    return next_state, probe_states


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-samples", type=int, default=1024)
    ap.add_argument("--eval-samples", type=int, default=1000)
    ap.add_argument("--image-size", type=int, default=10)
    ap.add_argument("--block-size", type=int, default=4)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.8)
    ap.add_argument("--linear-output", action="store_true")
    ap.add_argument("--softmax-output", action="store_true", default=True)
    ap.add_argument("--local-activation", default="tanh")
    ap.add_argument("--relu-clip", type=float, default=1.0)
    ap.add_argument("--relu-leak", type=float, default=0.01)
    ap.add_argument("--softplus-beta", type=float, default=10.0)
    ap.add_argument("--activation-derivative", choices=["exact", "stored-gate", "unity", "floor-exact"], default="exact")
    ap.add_argument("--derivative-floor", type=float, default=0.0)
    ap.add_argument("--derivative-gate-threshold", type=float, default=1e-6)
    ap.add_argument("--readout-feedback-mode", default="readout")
    ap.add_argument("--readout-feedback-clip", type=float, default=0.05)
    ap.add_argument("--hidden-synapse-mode", default="linear")
    ap.add_argument("--readout-synapse-mode", default="linear")
    ap.add_argument("--synapse-clip", type=float, default=1.0)
    ap.add_argument("--init-weights", default="")
    ap.add_argument("--probe-updates", default="powers2")
    ap.add_argument("--eval-batch-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="fast_online_reference")
    args = ap.parse_args()

    if args.linear_output and args.softmax_output:
        raise ValueError("--linear-output and --softmax-output are mutually exclusive")
    if args.train_samples <= 0 or args.eval_samples <= 0:
        raise ValueError("sample counts must be positive")
    if args.synapse_clip <= 0:
        raise ValueError("--synapse-clip must be positive")
    stride = args.block_size if args.stride is None else args.stride
    blocks = block_indices(args.image_size, args.block_size, stride)
    probe_updates = parse_probe_update_list(args.probe_updates, args.train_samples)
    x_train, y_train, x_test, y_test = load_mnist_sequence(args.train_samples, args.eval_samples, args.image_size, args.seed)
    rng = np.random.default_rng(args.seed)
    initial_state = load_or_init_weights(args.init_weights, rng, len(blocks), args.channels, args.block_size * args.block_size)

    forward_kwargs = {
        "local_activation": args.local_activation,
        "relu_clip": args.relu_clip,
        "relu_leak": args.relu_leak,
        "softplus_beta": args.softplus_beta,
        "hidden_synapse_mode": args.hidden_synapse_mode,
        "readout_synapse_mode": args.readout_synapse_mode,
        "synapse_clip": args.synapse_clip,
        "linear_output": args.linear_output,
        "softmax_output": args.softmax_output,
    }
    update_kwargs = {
        "lr": args.lr,
        "activation_derivative": args.activation_derivative,
        "derivative_floor": args.derivative_floor,
        "derivative_gate_threshold": args.derivative_gate_threshold,
        "readout_feedback_mode": args.readout_feedback_mode,
        "readout_feedback_clip": args.readout_feedback_clip,
        **forward_kwargs,
    }
    t0 = time.perf_counter()
    initial_eval = accuracy_np(x_test, y_test, initial_state, blocks, args.eval_batch_size, **forward_kwargs)
    final_state, probe_states = run_online(x_train, y_train, initial_state, blocks, probe_updates, **update_kwargs)
    final_eval = accuracy_np(x_test, y_test, final_state, blocks, args.eval_batch_size, **forward_kwargs)
    rows = []
    for update in probe_updates:
        state = probe_states[update]
        rows.append(
            {
                "update": update,
                "eval_accuracy": accuracy_np(x_test, y_test, state, blocks, args.eval_batch_size, **forward_kwargs),
            }
        )
    wall = time.perf_counter() - t0
    results = ROOT / "spice/results"
    tables = ROOT / "results/tables"
    results.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    stem = f"spice_mnist_local_feature_fast_online_{sanitize_tag(args.tag)}"
    curve = pd.DataFrame(rows)
    curve_path = results / f"{stem}_probe_curve.csv"
    table_curve_path = tables / f"{stem}_probe_curve.csv"
    curve.to_csv(curve_path, index=False)
    curve.to_csv(table_curve_path, index=False)
    summary = {
        "architecture": "local_feature_fast_online_reference",
        "dataset": "MNIST train/test split, downsampled",
        "image_size": args.image_size,
        "block_size": args.block_size,
        "stride": stride,
        "blocks": len(blocks),
        "channels": args.channels,
        "classes": 10,
        "train_samples": args.train_samples,
        "eval_samples": args.eval_samples,
        "batch_size": 1,
        "lr": args.lr,
        "linear_output": bool(args.linear_output),
        "softmax_output": bool(args.softmax_output),
        "local_activation": args.local_activation,
        "activation_derivative": args.activation_derivative,
        "readout_feedback_mode": args.readout_feedback_mode,
        "hidden_synapse_mode": args.hidden_synapse_mode,
        "readout_synapse_mode": args.readout_synapse_mode,
        "synapse_clip": args.synapse_clip,
        "initial_eval_accuracy": initial_eval,
        "final_eval_accuracy": final_eval,
        "eval_improvement": final_eval - initial_eval,
        "probe_curve": str(curve_path),
        "table_probe_curve": str(table_curve_path),
        "wall_time_s": wall,
        "python_role": "Fast NumPy reference only; not an on-device/SPICE result.",
    }
    summary_path = results / f"{stem}_summary.json"
    table_summary_path = tables / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    table_summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
