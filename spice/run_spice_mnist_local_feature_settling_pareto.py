from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_spice_mnist_local_block_batch_op_train import block_indices
from run_spice_mnist_settling_pareto import (
    accuracy_at_time,
    frontier_targets,
    pareto_frontier,
    parse_ns_list,
    parse_pp_list,
    plot_frontier,
    sanitize_tag,
    steady_state_accuracy,
)
from run_spice_mnist_train import load_mnist_sequence
from run_spice_sweep import ROOT


def load_local_feature_checkpoint(
    path: Path,
    n_classes: int,
    n_blocks: int,
    block_len: int,
) -> tuple[np.ndarray, ...]:
    init = np.load(path)
    required = {"local_weights", "local_bias", "readout", "output_bias"}
    missing = sorted(required.difference(init.files))
    if missing:
        raise ValueError(f"checkpoint is missing required arrays: {missing}")

    local_weights = init["local_weights"]
    local_bias = init["local_bias"]
    readout = init["readout"]
    output_bias = init["output_bias"]
    if local_weights.ndim != 3:
        raise ValueError(f"local_weights must have rank 3, got shape {local_weights.shape}")
    if local_weights.shape[0] != n_blocks or local_weights.shape[2] != block_len:
        raise ValueError(
            f"local_weights have shape {local_weights.shape}, expected blocks={n_blocks}, block_len={block_len}"
        )
    channels = local_weights.shape[1]
    if local_bias.shape != (n_blocks, channels):
        raise ValueError(f"local_bias has shape {local_bias.shape}, expected {(n_blocks, channels)}")
    if readout.shape != (n_classes, n_blocks, channels):
        raise ValueError(f"readout has shape {readout.shape}, expected {(n_classes, n_blocks, channels)}")
    if output_bias.shape != (n_classes,):
        raise ValueError(f"output_bias has shape {output_bias.shape}, expected {(n_classes,)}")
    return (
        local_weights.astype(float),
        local_bias.astype(float),
        readout.astype(float),
        output_bias.astype(float),
    )


def local_feature_evidence(
    x: np.ndarray,
    local_weights: np.ndarray,
    local_bias: np.ndarray,
    readout: np.ndarray,
    blocks: list[list[int]],
) -> np.ndarray:
    n = x.shape[0]
    n_blocks, channels, _block_len = local_weights.shape
    hidden = np.empty((n, n_blocks, channels), dtype=float)
    for block, idxs in enumerate(blocks):
        hidden[:, block, :] = np.tanh(x[:, idxs] @ local_weights[block].T + local_bias[block])
    return np.einsum("nbc,kbc->nk", hidden, readout)


def state_counts(n_blocks: int, channels: int, block_len: int, n_classes: int = 10) -> dict[str, int]:
    weight_values = n_blocks * channels * block_len
    local_bias_values = n_blocks * channels
    readout_values = n_classes * n_blocks * channels
    output_bias_values = n_classes
    train_state_values = weight_values + local_bias_values + readout_values + output_bias_values
    gradient_values = train_state_values
    temporary_values = 2 * n_blocks * channels + 2 * n_classes
    return {
        "weight_values": weight_values,
        "local_bias_values": local_bias_values,
        "readout_values": readout_values,
        "output_bias_values": output_bias_values,
        "train_state_values": train_state_values,
        "gradient_state_values": gradient_values,
        "temporary_state_values": temporary_values,
        "phase_state_values": train_state_values + gradient_values + temporary_values,
        "multiply_terms_per_sample": weight_values + readout_values,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-samples", type=int, default=2000)
    ap.add_argument("--test-samples", type=int, default=1000)
    ap.add_argument("--image-size", type=int, default=10)
    ap.add_argument("--block-size", type=int, default=4)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-weights", required=True)
    ap.add_argument("--tau-act-ns", default="0.25,0.35,0.5,0.75,1.0")
    ap.add_argument("--tau-score-ns", default="0.2,0.3,0.5,0.75,1.0")
    ap.add_argument("--time-min-ns", type=float, default=0.0)
    ap.add_argument("--time-max-ns", type=float, default=4.0)
    ap.add_argument("--timepoints", type=int, default=401)
    ap.add_argument(
        "--frontier-margins-pp",
        default="0,0.25,0.5,1,2,5",
        help="Comma-separated absolute percentage-point margins below the best accuracy for fastest-frontier reporting.",
    )
    ap.add_argument("--tag", default="local_feature_settling_pareto")
    args = ap.parse_args()

    if args.timepoints < 2:
        raise ValueError("--timepoints must be at least 2")
    if args.time_max_ns <= args.time_min_ns:
        raise ValueError("--time-max-ns must be greater than --time-min-ns")

    stride = args.block_size if args.stride is None else args.stride
    blocks = block_indices(args.image_size, args.block_size, stride)
    block_len = args.block_size * args.block_size
    local_weights, local_bias, readout, output_bias = load_local_feature_checkpoint(
        Path(args.init_weights),
        10,
        len(blocks),
        block_len,
    )
    channels = int(local_weights.shape[1])

    _x_train, _y_train, x_test, y_test = load_mnist_sequence(
        args.train_samples,
        args.test_samples,
        args.image_size,
        args.seed,
    )
    t0 = time.perf_counter()
    evidence = local_feature_evidence(x_test, local_weights, local_bias, readout, blocks)
    tau_act_values = parse_ns_list(args.tau_act_ns)
    tau_score_values = parse_ns_list(args.tau_score_ns)
    readout_times = np.linspace(args.time_min_ns, args.time_max_ns, args.timepoints)

    rows = []
    for tau_act_ns in tau_act_values:
        for tau_score_ns in tau_score_values:
            tau_act_s = tau_act_ns * 1e-9
            tau_score_s = tau_score_ns * 1e-9
            for readout_time_ns in readout_times:
                acc, correct = accuracy_at_time(
                    evidence,
                    output_bias,
                    y_test,
                    float(readout_time_ns) * 1e-9,
                    tau_act_s,
                    tau_score_s,
                )
                rows.append(
                    {
                        "readout_time_ns": float(readout_time_ns),
                        "tau_act_ns": float(tau_act_ns),
                        "tau_score_ns": float(tau_score_ns),
                        "accuracy": float(acc),
                        "correct": correct,
                        "total": int(len(y_test)),
                        "time_over_tau_act": float(readout_time_ns / tau_act_ns) if tau_act_ns else np.inf,
                        "time_over_tau_score": float(readout_time_ns / tau_score_ns) if tau_score_ns else np.inf,
                    }
                )

    df = pd.DataFrame(rows)
    frontier = pareto_frontier(df)
    steady_acc, steady_correct = steady_state_accuracy(evidence, output_bias, y_test)
    terminal = df[np.isclose(df["readout_time_ns"], readout_times[-1])].copy()
    terminal = terminal.rename(columns={"accuracy": "terminal_accuracy", "correct": "terminal_correct"})
    pair_best = df.loc[df.groupby(["tau_act_ns", "tau_score_ns"])["accuracy"].idxmax()].copy()
    pair_best["steady_accuracy"] = steady_acc
    pair_best["steady_correct"] = steady_correct
    pair_best["peak_minus_steady_accuracy"] = pair_best["accuracy"] - steady_acc
    pair_best["peak_above_steady"] = pair_best["peak_minus_steady_accuracy"] > 0
    terminal_by_pair = terminal.set_index(["tau_act_ns", "tau_score_ns"])
    pair_index = pd.MultiIndex.from_frame(pair_best[["tau_act_ns", "tau_score_ns"]])
    pair_best["terminal_accuracy"] = terminal_by_pair.loc[pair_index, "terminal_accuracy"].to_numpy()
    pair_best["terminal_correct"] = terminal_by_pair.loc[pair_index, "terminal_correct"].to_numpy(dtype=int)
    pair_best["terminal_minus_steady_accuracy"] = pair_best["terminal_accuracy"] - steady_acc
    pair_best = pair_best.sort_values(["accuracy", "readout_time_ns"], ascending=[False, True])

    best = df.loc[df["accuracy"].idxmax()].to_dict()
    terminal_best = terminal.sort_values(["terminal_accuracy", "readout_time_ns"], ascending=[False, True]).iloc[0].to_dict()
    margin_values_pp = sorted(set(parse_pp_list(args.frontier_margins_pp) + [1.0]))
    targets = frontier_targets(frontier, float(best["accuracy"]), margin_values_pp)
    fastest_within_1pp = targets[targets["within_best_pp"] == 1.0].iloc[0].to_dict()

    stem = f"spice_mnist_local_feature_settling_pareto_{sanitize_tag(args.tag)}"
    spice_results = ROOT / "spice/results"
    tables = ROOT / "results/tables"
    figures = ROOT / "results/figures"
    for directory in [spice_results, tables, figures]:
        directory.mkdir(parents=True, exist_ok=True)

    all_csv = spice_results / f"{stem}_all.csv"
    table_all_csv = tables / f"{stem}_all.csv"
    frontier_csv = tables / f"{stem}_frontier.csv"
    pair_best_csv = tables / f"{stem}_pair_best.csv"
    targets_csv = tables / f"{stem}_frontier_targets.csv"
    fig_path = figures / f"{stem}_frontier.png"
    df.to_csv(all_csv, index=False)
    df.to_csv(table_all_csv, index=False)
    frontier.to_csv(frontier_csv, index=False)
    pair_best.to_csv(pair_best_csv, index=False)
    targets.to_csv(targets_csv, index=False)
    plot_frontier(
        df,
        frontier,
        fig_path,
        f"Local-feature settling frontier ({args.image_size}x{args.image_size}, n={args.test_samples})",
        steady_acc,
    )

    summary = {
        "architecture": "local_feature_settling_time_accuracy_surrogate",
        "init_weights": args.init_weights,
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "image_size": args.image_size,
        "block_size": args.block_size,
        "stride": stride,
        "blocks": len(blocks),
        "channels": channels,
        "classes": 10,
        **state_counts(len(blocks), channels, block_len),
        "tau_act_ns": tau_act_values,
        "tau_score_ns": tau_score_values,
        "time_min_ns": args.time_min_ns,
        "time_max_ns": args.time_max_ns,
        "timepoints": args.timepoints,
        "best_accuracy": float(best["accuracy"]),
        "best_readout_time_ns": float(best["readout_time_ns"]),
        "best_tau_act_ns": float(best["tau_act_ns"]),
        "best_tau_score_ns": float(best["tau_score_ns"]),
        "steady_state_accuracy": float(steady_acc),
        "steady_state_correct": int(steady_correct),
        "steady_state_total": int(len(y_test)),
        "best_minus_steady_state_accuracy": float(best["accuracy"] - steady_acc),
        "terminal_accuracy_at_time_max": float(terminal_best["terminal_accuracy"]),
        "terminal_readout_time_ns": float(terminal_best["readout_time_ns"]),
        "terminal_tau_act_ns": float(terminal_best["tau_act_ns"]),
        "terminal_tau_score_ns": float(terminal_best["tau_score_ns"]),
        "fastest_within_1pct_of_best_accuracy": float(fastest_within_1pp["accuracy"]),
        "fastest_within_1pct_of_best_readout_time_ns": float(fastest_within_1pp["readout_time_ns"]),
        "fastest_within_1pct_of_best_tau_act_ns": float(fastest_within_1pp["tau_act_ns"]),
        "fastest_within_1pct_of_best_tau_score_ns": float(fastest_within_1pp["tau_score_ns"]),
        "frontier_targets": targets.to_dict(orient="records"),
        "all_csv": str(all_csv),
        "table_all_csv": str(table_all_csv),
        "frontier_csv": str(frontier_csv),
        "pair_best_csv": str(pair_best_csv),
        "frontier_targets_csv": str(targets_csv),
        "figure": str(fig_path),
        "wall_time_s": time.perf_counter() - t0,
        "note": (
            "Analytical first-order settling surrogate for the local-feature hidden activations "
            "and readout score capacitors. It sweeps readout timing and time constants for a "
            "fixed checkpoint; it is not a training run."
        ),
    }
    summary_path = spice_results / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
