from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
import pandas as pd

import run_spice_mnist_local_feature_fast_online_reference as fast_ref
from run_spice_mnist_local_block_batch_op_train import block_indices
from run_spice_mnist_local_feature_phase_variant_sweep import activation_clip_pairs, parse_csv, parse_float_csv, variant_tag
from run_spice_mnist_train import load_mnist_sequence
from run_spice_sweep import ROOT

TrainState = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def variant_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    activations = parse_csv(args.activations)
    relu_clips = parse_float_csv(args.relu_clips)
    derivative_modes = parse_csv(args.derivative_modes)
    feedback_modes = parse_csv(args.feedback_modes)
    hidden_synapse_modes = parse_csv(args.hidden_synapse_modes)
    readout_synapse_modes = parse_csv(args.readout_synapse_modes)
    synapse_clips = parse_float_csv(args.synapse_clips) if args.synapse_clips else [args.synapse_clip]
    if not activations:
        raise ValueError("--activations must not be empty")
    if not relu_clips:
        raise ValueError("--relu-clips must not be empty")
    if not derivative_modes:
        raise ValueError("--derivative-modes must not be empty")
    if not feedback_modes:
        raise ValueError("--feedback-modes must not be empty")
    if not hidden_synapse_modes:
        raise ValueError("--hidden-synapse-modes must not be empty")
    if not readout_synapse_modes:
        raise ValueError("--readout-synapse-modes must not be empty")
    if any(clip <= 0 for clip in synapse_clips):
        raise ValueError("--synapse-clip/--synapse-clips values must be positive")

    rows: list[dict[str, Any]] = []
    for activation, relu_clip in activation_clip_pairs(activations, relu_clips):
        for derivative_mode in derivative_modes:
            for feedback_mode in feedback_modes:
                for hidden_synapse_mode in hidden_synapse_modes:
                    for readout_synapse_mode in readout_synapse_modes:
                        for synapse_clip in synapse_clips:
                            rows.append(
                                {
                                    "local_activation": activation,
                                    "relu_clip": relu_clip,
                                    "activation_derivative": derivative_mode,
                                    "readout_feedback_mode": feedback_mode,
                                    "hidden_synapse_mode": hidden_synapse_mode,
                                    "readout_synapse_mode": readout_synapse_mode,
                                    "synapse_clip": synapse_clip,
                                    "tag": variant_tag(
                                        args.tag,
                                        activation,
                                        relu_clip,
                                        derivative_mode,
                                        feedback_mode,
                                        hidden_synapse_mode,
                                        readout_synapse_mode,
                                        synapse_clip,
                                    ),
                                }
                            )
    return rows


def forward_kwargs(args: argparse.Namespace, variant: dict[str, Any]) -> dict[str, Any]:
    return {
        "local_activation": variant["local_activation"],
        "relu_clip": variant["relu_clip"],
        "relu_leak": args.relu_leak,
        "softplus_beta": args.softplus_beta,
        "hidden_synapse_mode": variant["hidden_synapse_mode"],
        "readout_synapse_mode": variant["readout_synapse_mode"],
        "synapse_clip": variant["synapse_clip"],
        "linear_output": args.linear_output,
        "softmax_output": args.softmax_output,
        "softmax_temperature": args.softmax_temperature,
        "readout_class_centering": args.readout_class_centering,
    }


def update_kwargs(args: argparse.Namespace, variant: dict[str, Any]) -> dict[str, Any]:
    return {
        "lr": args.lr,
        "activation_derivative": variant["activation_derivative"],
        "derivative_floor": args.derivative_floor,
        "derivative_gate_threshold": args.derivative_gate_threshold,
        "readout_feedback_mode": variant["readout_feedback_mode"],
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
        **forward_kwargs(args, variant),
    }


def run_variant(
    args: argparse.Namespace,
    variant: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    initial_state: TrainState,
    blocks: list[list[int]],
    probe_updates: tuple[int, ...],
) -> dict[str, Any]:
    fwd = forward_kwargs(args, variant)
    upd = update_kwargs(args, variant)
    state0 = tuple(arr.copy() for arr in initial_state)
    t0 = time.perf_counter()
    initial_eval = fast_ref.accuracy_np(x_eval, y_eval, state0, blocks, args.eval_batch_size, **fwd)
    final_state, probe_states = fast_ref.run_online(x_train, y_train, state0, blocks, probe_updates, **upd)
    final_eval = fast_ref.accuracy_np(x_eval, y_eval, final_state, blocks, args.eval_batch_size, **fwd)
    probe_rows = [
        {
            "update": update,
            "eval_accuracy": fast_ref.accuracy_np(x_eval, y_eval, probe_states[update], blocks, args.eval_batch_size, **fwd),
        }
        for update in probe_updates
        if update in probe_states
    ]
    best_probe = max(probe_rows, key=lambda row: row["eval_accuracy"]) if probe_rows else None
    wall = time.perf_counter() - t0
    return {
        **variant,
        "initial_eval_accuracy": initial_eval,
        "final_eval_accuracy": final_eval,
        "eval_improvement": final_eval - initial_eval,
        "best_probe_eval_accuracy": best_probe["eval_accuracy"] if best_probe is not None else None,
        "best_probe_update": best_probe["update"] if best_probe is not None else None,
        "wall_time_s": wall,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-samples", type=int, default=256)
    ap.add_argument("--eval-samples", type=int, default=300)
    ap.add_argument("--image-size", type=int, default=10)
    ap.add_argument("--block-size", type=int, default=4)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.8)
    ap.add_argument("--linear-output", action="store_true")
    ap.add_argument("--softmax-output", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--activations",
        "--local-activation",
        dest="activations",
        default="tanh,diff-clipped-relu,softplus,leaky-relu",
    )
    ap.add_argument("--relu-clips", default="1.0")
    ap.add_argument("--relu-leak", type=float, default=0.01)
    ap.add_argument("--softplus-beta", type=float, default=10.0)
    ap.add_argument("--derivative-modes", default="exact")
    ap.add_argument("--derivative-floor", type=float, default=0.0)
    ap.add_argument("--derivative-gate-threshold", type=float, default=1e-6)
    ap.add_argument("--feedback-modes", default="full-readout")
    ap.add_argument("--readout-feedback-clip", type=float, default=0.05)
    ap.add_argument("--output-bias-update-scale", type=float, default=0.0)
    ap.add_argument("--readout-update-scale", type=float, default=0.25)
    ap.add_argument("--local-update-scale", type=float, default=1.0)
    ap.add_argument("--state-decay", type=float, default=0.0)
    ap.add_argument("--softmax-negative-scale", type=float, default=1.0)
    ap.add_argument("--softmax-error-centering", choices=["none", "mean"], default="none")
    ap.add_argument("--softmax-temperature", type=float, default=4.0)
    ap.add_argument("--softmax-competition-mode", choices=["all", "normalized-power"], default="all")
    ap.add_argument("--softmax-competitor-power", type=int, default=2)
    ap.add_argument("--softmax-error-gate", choices=["none", "target-margin"], default="none")
    ap.add_argument("--softmax-margin", type=float, default=1.0)
    ap.add_argument("--hidden-synapse-modes", default="tanh-clipped")
    ap.add_argument("--readout-synapse-modes", default="linear")
    ap.add_argument("--synapse-clip", type=float, default=2.0)
    ap.add_argument("--synapse-clips", default="")
    ap.add_argument("--readout-class-centering", choices=["none", "mean"], default="none")
    ap.add_argument("--init-weights", default="")
    ap.add_argument("--probe-updates", default="powers2")
    ap.add_argument("--eval-batch-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="fast_online_variant_sweep")
    args = ap.parse_args()

    if args.linear_output and args.softmax_output:
        raise ValueError("--linear-output and --softmax-output are mutually exclusive")
    if args.train_samples <= 0 or args.eval_samples <= 0:
        raise ValueError("sample counts must be positive")
    if args.channels <= 0:
        raise ValueError("--channels must be positive")
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

    stride = args.block_size if args.stride is None else args.stride
    blocks = block_indices(args.image_size, args.block_size, stride)
    probe_updates = fast_ref.parse_probe_update_list(args.probe_updates, args.train_samples)
    x_train, y_train, x_eval, y_eval = load_mnist_sequence(args.train_samples, args.eval_samples, args.image_size, args.seed)
    rng = np.random.default_rng(args.seed)
    initial_state = fast_ref.load_or_init_weights(
        args.init_weights,
        rng,
        len(blocks),
        args.channels,
        args.block_size * args.block_size,
    )
    rows = [
        run_variant(args, variant, x_train, y_train, x_eval, y_eval, initial_state, blocks, probe_updates)
        for variant in variant_grid(args)
    ]
    rows.sort(key=lambda row: (row["final_eval_accuracy"], row["eval_improvement"]), reverse=True)

    results = ROOT / "spice/results"
    tables = ROOT / "results/tables"
    results.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    stem = f"spice_mnist_local_feature_fast_online_variant_{fast_ref.sanitize_tag(args.tag)}"
    csv_path = results / f"{stem}.csv"
    table_csv_path = tables / f"{stem}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    pd.DataFrame(rows).to_csv(table_csv_path, index=False)
    summary = {
        "architecture": "local_feature_fast_online_variant_sweep",
        "python_role": "Fast NumPy reference only; not an on-device/SPICE result.",
        "train_samples": args.train_samples,
        "eval_samples": args.eval_samples,
        "image_size": args.image_size,
        "block_size": args.block_size,
        "stride": stride,
        "channels": args.channels,
        "batch_size": 1,
        "variants": len(rows),
        "best_variant": rows[0] if rows else None,
        "csv": str(csv_path),
        "table_csv": str(table_csv_path),
    }
    summary_path = results / f"{stem}_summary.json"
    table_summary_path = tables / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    table_summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
