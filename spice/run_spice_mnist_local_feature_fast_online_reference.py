from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from run_spice_mnist_local_block_batch_op_train import block_indices
from run_spice_mnist_local_feature_phase_transient import (
    apply_readout_class_centering_np,
    block_tensor_np,
    load_or_init_weights,
    local_activation_np,
    lr_schedule_values,
    numpy_eval_accuracy as accuracy_np,
    output_activation_np,
    parse_probe_update_list,
    sanitize_tag,
    synapse_transfer_np,
)
from run_spice_mnist_train import load_mnist_sequence
from run_spice_sweep import ROOT


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
    softmax_temperature: float,
    readout_class_centering: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w, hb, readout, output_bias = state
    xb = block_tensor_np(x, blocks)
    eff_w = synapse_transfer_np(w, hidden_synapse_mode, synapse_clip)
    pre = np.einsum("nbp,bcp->nbc", xb, eff_w) + hb
    h = local_activation_np(pre, local_activation, relu_clip, relu_leak, softplus_beta)
    eff_readout = apply_readout_class_centering_np(
        synapse_transfer_np(readout, readout_synapse_mode, synapse_clip),
        readout_class_centering,
    )
    score = np.einsum("nbc,kbc->nk", h, eff_readout) + output_bias
    y = output_activation_np(score, linear_output, softmax_output, softmax_temperature)
    return xb, pre, h, score, y


def target_matrix(labels: np.ndarray, n_classes: int, softmax_output: bool) -> np.ndarray:
    targets = np.zeros((len(labels), n_classes)) if softmax_output else -np.ones((len(labels), n_classes))
    targets[np.arange(len(labels)), labels.astype(int)] = 1.0
    return targets


def fast_reference_objective_fields(
    final_eval_accuracy: float | None,
    eval_samples: int,
    full_objective_eval_samples: int,
    full_objective_accuracy: float,
) -> dict[str, object]:
    if full_objective_eval_samples <= 0:
        raise ValueError("full_objective_eval_samples must be positive")
    if full_objective_accuracy < 0.0 or full_objective_accuracy > 1.0:
        raise ValueError("full_objective_accuracy must be in [0, 1]")
    accuracy = None if final_eval_accuracy is None else float(final_eval_accuracy)
    accuracy_gap = None if accuracy is None else max(0.0, float(full_objective_accuracy) - accuracy)
    full_eval_met = int(eval_samples) >= int(full_objective_eval_samples)
    accuracy_met = accuracy is not None and accuracy >= float(full_objective_accuracy)
    return {
        "fast_reference_full_eval_sample_count_met": full_eval_met,
        "fast_reference_full_objective_accuracy_met": accuracy_met,
        "fast_reference_full_objective_accuracy_gap": accuracy_gap,
        "fast_reference_full_objective_candidate": full_eval_met and accuracy_met,
    }


def softmax_delta_np(
    targets: np.ndarray,
    y: np.ndarray,
    score: np.ndarray,
    *,
    softmax_negative_scale: float,
    softmax_error_centering: str,
    softmax_competition_mode: str,
    softmax_competitor_power: int,
    softmax_error_gate: str,
    softmax_margin: float,
) -> np.ndarray:
    if softmax_negative_scale < 0.0:
        raise ValueError("softmax_negative_scale must be non-negative")
    if softmax_competition_mode == "all":
        d = targets * (1.0 - y) - (1.0 - targets) * softmax_negative_scale * y
    elif softmax_competition_mode == "normalized-power":
        if softmax_competitor_power < 1:
            raise ValueError("softmax_competitor_power must be positive")
        target_error = np.sum(targets * (1.0 - y), axis=1, keepdims=True)
        competitor_weight = (1.0 - targets) * np.power(y, softmax_competitor_power)
        competitor_den = np.sum(competitor_weight, axis=1, keepdims=True)
        d = targets * (1.0 - y) - softmax_negative_scale * target_error * competitor_weight / (competitor_den + 1e-12)
    else:
        raise ValueError("softmax_competition_mode must be 'all' or 'normalized-power'")
    if softmax_error_centering == "mean":
        d = d - np.mean(d, axis=1, keepdims=True)
    elif softmax_error_centering != "none":
        raise ValueError("softmax_error_centering must be 'none' or 'mean'")
    if softmax_error_gate == "target-margin":
        if softmax_margin <= 0.0:
            raise ValueError("softmax_margin must be positive when softmax_error_gate is target-margin")
        target_score = np.sum(targets * score, axis=1, keepdims=True)
        masked_score = np.where(targets > 0.5, -np.inf, score)
        competitor_score = np.max(masked_score, axis=1, keepdims=True)
        deficit = (softmax_margin - (target_score - competitor_score)) / (softmax_margin + 1e-12)
        d = d * np.clip(deficit, 0.0, 1.0)
    elif softmax_error_gate != "none":
        raise ValueError("softmax_error_gate must be 'none' or 'target-margin'")
    return d


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
    softmax_negative_scale: float = 1.0,
    softmax_error_centering: str = "none",
    softmax_temperature: float = 1.0,
    softmax_competition_mode: str = "all",
    softmax_competitor_power: int = 2,
    softmax_error_gate: str = "none",
    softmax_margin: float = 1.0,
    output_bias_update_scale: float = 1.0,
    readout_update_scale: float = 1.0,
    local_update_scale: float = 1.0,
    state_decay: float = 0.0,
    readout_class_centering: str = "none",
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
        softmax_temperature=softmax_temperature,
        readout_class_centering=readout_class_centering,
    )
    targets = target_matrix(labels, readout.shape[0], softmax_output)
    if softmax_output:
        d = softmax_delta_np(
            targets,
            y,
            _score,
            softmax_negative_scale=softmax_negative_scale,
            softmax_error_centering=softmax_error_centering,
            softmax_competition_mode=softmax_competition_mode,
            softmax_competitor_power=softmax_competitor_power,
            softmax_error_gate=softmax_error_gate,
            softmax_margin=softmax_margin,
        )
    elif linear_output:
        d = targets - y
    else:
        d = (targets - y) * (1.0 - y * y)
    eff_readout = apply_readout_class_centering_np(
        synapse_transfer_np(readout, readout_synapse_mode, synapse_clip),
        readout_class_centering,
    )
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
        w * (1.0 - state_decay) + lr * local_update_scale * np.einsum("nbc,nbp->bcp", dh, xb) / batch,
        hb * (1.0 - state_decay) + lr * local_update_scale * np.mean(dh, axis=0),
        readout * (1.0 - state_decay) + lr * readout_update_scale * np.einsum("nk,nbc->kbc", d, h) / batch,
        output_bias * (1.0 - state_decay) + lr * output_bias_update_scale * np.mean(d, axis=0),
    )


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
    update_args = dict(update_kwargs)
    base_lr = float(update_args.pop("lr"))
    lr_schedule = update_args.pop("lr_schedule", "constant")
    lr_final_scale = float(update_args.pop("lr_final_scale", 1.0))
    lr_values = lr_schedule_values(base_lr, len(y_train), lr_schedule, lr_final_scale)
    for update in range(1, len(y_train) + 1):
        next_state = update_np(
            x_train[update - 1 : update],
            y_train[update - 1 : update],
            next_state,
            blocks,
            **update_args,
            lr=float(lr_values[update - 1]),
        )
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
    ap.add_argument("--lr-schedule", choices=["constant", "linear-decay"], default="constant")
    ap.add_argument("--lr-final-scale", type=float, default=1.0)
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
    ap.add_argument("--output-bias-update-scale", type=float, default=1.0)
    ap.add_argument("--readout-update-scale", type=float, default=1.0)
    ap.add_argument("--local-update-scale", type=float, default=1.0)
    ap.add_argument("--state-decay", type=float, default=0.0)
    ap.add_argument("--softmax-negative-scale", type=float, default=1.0)
    ap.add_argument("--softmax-error-centering", choices=["none", "mean"], default="none")
    ap.add_argument("--softmax-temperature", type=float, default=1.0)
    ap.add_argument("--softmax-competition-mode", choices=["all", "normalized-power"], default="all")
    ap.add_argument("--softmax-competitor-power", type=int, default=2)
    ap.add_argument("--softmax-error-gate", choices=["none", "target-margin"], default="none")
    ap.add_argument("--softmax-margin", type=float, default=1.0)
    ap.add_argument("--hidden-synapse-mode", default="linear")
    ap.add_argument("--readout-synapse-mode", default="linear")
    ap.add_argument("--synapse-clip", type=float, default=1.0)
    ap.add_argument("--readout-class-centering", choices=["none", "mean"], default="none")
    ap.add_argument("--init-weights", default="")
    ap.add_argument("--probe-updates", default="powers2")
    ap.add_argument("--eval-batch-size", type=int, default=1024)
    ap.add_argument("--full-objective-eval-samples", type=int, default=10000)
    ap.add_argument("--full-objective-accuracy", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="fast_online_reference")
    args = ap.parse_args()

    if args.linear_output and args.softmax_output:
        raise ValueError("--linear-output and --softmax-output are mutually exclusive")
    if args.train_samples <= 0 or args.eval_samples <= 0:
        raise ValueError("sample counts must be positive")
    if args.lr < 0:
        raise ValueError("--lr must be non-negative")
    if args.lr_final_scale < 0:
        raise ValueError("--lr-final-scale must be non-negative")
    if args.synapse_clip <= 0:
        raise ValueError("--synapse-clip must be positive")
    if args.softmax_negative_scale < 0:
        raise ValueError("--softmax-negative-scale must be non-negative")
    if args.softmax_temperature <= 0:
        raise ValueError("--softmax-temperature must be positive")
    if args.softmax_competitor_power < 1:
        raise ValueError("--softmax-competitor-power must be positive")
    if args.softmax_error_gate == "target-margin" and args.softmax_margin <= 0:
        raise ValueError("--softmax-margin must be positive when --softmax-error-gate is target-margin")
    if args.output_bias_update_scale < 0 or args.readout_update_scale < 0 or args.local_update_scale < 0:
        raise ValueError("update scales must be non-negative")
    if args.state_decay < 0 or args.state_decay >= 1:
        raise ValueError("--state-decay must be in [0, 1)")
    if args.full_objective_eval_samples <= 0:
        raise ValueError("--full-objective-eval-samples must be positive")
    if args.full_objective_accuracy < 0.0 or args.full_objective_accuracy > 1.0:
        raise ValueError("--full-objective-accuracy must be in [0, 1]")
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
        "softmax_temperature": args.softmax_temperature,
        "readout_class_centering": args.readout_class_centering,
    }
    update_kwargs = {
        "lr": args.lr,
        "lr_schedule": args.lr_schedule,
        "lr_final_scale": args.lr_final_scale,
        "activation_derivative": args.activation_derivative,
        "derivative_floor": args.derivative_floor,
        "derivative_gate_threshold": args.derivative_gate_threshold,
        "readout_feedback_mode": args.readout_feedback_mode,
        "readout_feedback_clip": args.readout_feedback_clip,
        "output_bias_update_scale": args.output_bias_update_scale,
        "readout_update_scale": args.readout_update_scale,
        "local_update_scale": args.local_update_scale,
        "state_decay": args.state_decay,
        "softmax_negative_scale": args.softmax_negative_scale,
        "softmax_error_centering": args.softmax_error_centering,
        "softmax_competition_mode": args.softmax_competition_mode,
        "softmax_competitor_power": args.softmax_competitor_power,
        "softmax_error_gate": args.softmax_error_gate,
        "softmax_margin": args.softmax_margin,
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
        "lr_schedule": args.lr_schedule,
        "lr_final_scale": args.lr_final_scale,
        "linear_output": bool(args.linear_output),
        "softmax_output": bool(args.softmax_output),
        "local_activation": args.local_activation,
        "activation_derivative": args.activation_derivative,
        "readout_feedback_mode": args.readout_feedback_mode,
        "output_bias_update_scale": args.output_bias_update_scale,
        "readout_update_scale": args.readout_update_scale,
        "local_update_scale": args.local_update_scale,
        "state_decay": args.state_decay,
        "softmax_negative_scale": args.softmax_negative_scale,
        "softmax_error_centering": args.softmax_error_centering,
        "softmax_temperature": args.softmax_temperature,
        "softmax_competition_mode": args.softmax_competition_mode,
        "softmax_competitor_power": args.softmax_competitor_power,
        "softmax_error_gate": args.softmax_error_gate,
        "softmax_margin": args.softmax_margin,
        "hidden_synapse_mode": args.hidden_synapse_mode,
        "readout_synapse_mode": args.readout_synapse_mode,
        "synapse_clip": args.synapse_clip,
        "readout_class_centering": args.readout_class_centering,
        "initial_eval_accuracy": initial_eval,
        "final_eval_accuracy": final_eval,
        "eval_improvement": final_eval - initial_eval,
        "full_objective_eval_samples": args.full_objective_eval_samples,
        "full_objective_accuracy": args.full_objective_accuracy,
        **fast_reference_objective_fields(
            final_eval,
            args.eval_samples,
            args.full_objective_eval_samples,
            args.full_objective_accuracy,
        ),
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
