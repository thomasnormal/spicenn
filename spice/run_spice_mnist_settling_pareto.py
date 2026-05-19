from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_spice_mnist_local_block_batch_op_train import block_indices
from run_spice_mnist_train import load_mnist_sequence
from run_spice_sweep import ROOT


def sanitize_tag(tag: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in tag)


def parse_ns_list(text: str) -> list[float]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    if not values:
        raise ValueError("expected at least one comma-separated value")
    return values


def parse_pp_list(text: str) -> list[float]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    if not values:
        raise ValueError("expected at least one comma-separated percentage-point value")
    if any(value < 0 for value in values):
        raise ValueError("percentage-point margins must be nonnegative")
    return values


def load_checkpoint(path: Path, n_classes: int, n_blocks: int, block_len: int) -> tuple[np.ndarray, ...]:
    init = np.load(path)
    weights = init["weights"]
    local_bias = init["local_bias"]
    gains = init["gains"]
    output_bias = init["output_bias"]
    expected = (n_classes, n_blocks, block_len)
    if weights.shape != expected:
        raise ValueError(f"checkpoint weights have shape {weights.shape}, expected {expected}")
    return weights, local_bias, gains, output_bias


def local_evidence(
    x: np.ndarray,
    weights: np.ndarray,
    local_bias: np.ndarray,
    gains: np.ndarray,
    blocks: list[list[int]],
) -> np.ndarray:
    n = x.shape[0]
    n_classes, n_blocks, _block_len = weights.shape
    acts = np.empty((n, n_classes, n_blocks), dtype=float)
    for b, idxs in enumerate(blocks):
        xb = x[:, idxs]
        acts[:, :, b] = np.tanh(xb @ weights[:, b, :].T + local_bias[:, b])
    return np.einsum("nkb,kb->nk", acts, gains)


def cascade_step_factor(t_s: float, tau_act_s: float, tau_score_s: float) -> float:
    if t_s <= 0.0:
        return 0.0
    if abs(tau_act_s - tau_score_s) <= 1e-15 * max(tau_act_s, tau_score_s, 1.0):
        u = t_s / tau_score_s
        return float(1.0 - np.exp(-u) * (1.0 + u))
    return float(
        1.0
        - (tau_act_s / (tau_act_s - tau_score_s)) * np.exp(-t_s / tau_act_s)
        + (tau_score_s / (tau_act_s - tau_score_s)) * np.exp(-t_s / tau_score_s)
    )


def scores_at_time(
    evidence: np.ndarray,
    output_bias: np.ndarray,
    t_s: float,
    tau_act_s: float,
    tau_score_s: float,
) -> np.ndarray:
    bias_factor = 0.0 if t_s <= 0.0 else float(1.0 - np.exp(-t_s / tau_score_s))
    evidence_factor = cascade_step_factor(t_s, tau_act_s, tau_score_s)
    return output_bias[None, :] * bias_factor + evidence * evidence_factor


def accuracy_from_scores(scores: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    pred = np.argmax(scores, axis=1)
    correct = int(np.sum(pred == y))
    return correct / max(len(y), 1), correct


def accuracy_at_time(
    evidence: np.ndarray,
    output_bias: np.ndarray,
    y: np.ndarray,
    t_s: float,
    tau_act_s: float,
    tau_score_s: float,
) -> tuple[float, int]:
    return accuracy_from_scores(scores_at_time(evidence, output_bias, t_s, tau_act_s, tau_score_s), y)


def steady_state_accuracy(
    evidence: np.ndarray,
    output_bias: np.ndarray,
    y: np.ndarray,
) -> tuple[float, int]:
    return accuracy_from_scores(evidence + output_bias[None, :], y)


def pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values(["readout_time_ns", "accuracy"], ascending=[True, False])
    keep = []
    best = -np.inf
    for row in ordered.itertuples(index=False):
        acc = float(row.accuracy)
        if acc > best + 1e-12:
            keep.append(row._asdict())
            best = acc
    return pd.DataFrame(keep)


def frontier_targets(frontier: pd.DataFrame, best_accuracy: float, margins_pp: list[float]) -> pd.DataFrame:
    rows = []
    for margin_pp in sorted(set(margins_pp)):
        target = best_accuracy - margin_pp / 100.0
        candidates = frontier[frontier["accuracy"] >= target - 1e-12]
        if len(candidates) == 0:
            continue
        row = candidates.iloc[0]
        rows.append(
            {
                "within_best_pp": float(margin_pp),
                "target_accuracy": float(target),
                "readout_time_ns": float(row["readout_time_ns"]),
                "accuracy": float(row["accuracy"]),
                "correct": int(row["correct"]),
                "total": int(row["total"]),
                "tau_act_ns": float(row["tau_act_ns"]),
                "tau_score_ns": float(row["tau_score_ns"]),
            }
        )
    return pd.DataFrame(rows)


def validate_against_spice(
    reference_csv: Path,
    evidence: np.ndarray,
    output_bias: np.ndarray,
    y: np.ndarray,
    tau_act_ns: float,
    tau_score_ns: float,
    ignore_before_ns: float,
) -> dict[str, float | str | int]:
    ref = pd.read_csv(reference_csv)
    ref = ref[ref["time_after_load_ns"] >= ignore_before_ns].copy()
    if len(ref) == 0:
        raise ValueError("SPICE reference validation has no rows after --validation-ignore-before-ns")
    analytic = []
    for t_ns in ref["time_after_load_ns"]:
        acc, _correct = accuracy_at_time(
            evidence,
            output_bias,
            y,
            max(float(t_ns), 0.0) * 1e-9,
            tau_act_ns * 1e-9,
            tau_score_ns * 1e-9,
        )
        analytic.append(acc)
    diff = np.asarray(analytic) - ref["accuracy"].to_numpy(dtype=float)
    return {
        "reference_csv": str(reference_csv),
        "rows": int(len(ref)),
        "tau_act_ns": tau_act_ns,
        "tau_score_ns": tau_score_ns,
        "ignore_before_ns": ignore_before_ns,
        "max_abs_accuracy_diff": float(np.max(np.abs(diff))),
        "rms_accuracy_diff": float(np.sqrt(np.mean(diff * diff))),
    }


def plot_frontier(
    df: pd.DataFrame,
    frontier: pd.DataFrame,
    out: Path,
    title: str,
    steady_accuracy: float | None = None,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    plt.scatter(df["readout_time_ns"], df["accuracy"], s=10, alpha=0.18, label="sweep")
    plt.plot(frontier["readout_time_ns"], frontier["accuracy"], linewidth=2.2, label="Pareto frontier")
    if steady_accuracy is not None:
        plt.axhline(
            steady_accuracy,
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.65,
            label="exact steady state",
        )
    plt.xlabel("readout time after load (ns)")
    plt.ylabel("accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.25)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-samples", type=int, default=2000)
    ap.add_argument("--test-samples", type=int, default=1000)
    ap.add_argument("--image-size", type=int, default=14)
    ap.add_argument("--block-size", type=int, default=7)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-weights", required=True)
    ap.add_argument("--tau-act-ns", default="0.5")
    ap.add_argument("--tau-score-ns", default="0.5")
    ap.add_argument("--time-min-ns", type=float, default=0.0)
    ap.add_argument("--time-max-ns", type=float, default=8.0)
    ap.add_argument("--timepoints", type=int, default=321)
    ap.add_argument(
        "--frontier-margins-pp",
        default="0,0.5,1,2,5",
        help="Comma-separated absolute percentage-point margins below the best accuracy for fastest-frontier reporting.",
    )
    ap.add_argument("--spice-reference-csv", default=None)
    ap.add_argument("--spice-reference-test-samples", type=int, default=None)
    ap.add_argument("--reference-tau-act-ns", type=float, default=0.5)
    ap.add_argument("--reference-tau-score-ns", type=float, default=0.5)
    ap.add_argument(
        "--validation-ignore-before-ns",
        type=float,
        default=0.1,
        help="Ignore the earliest SPICE rows where the finite load edge has already partially precharged states.",
    )
    ap.add_argument("--tag", default="settling_pareto")
    args = ap.parse_args()

    if args.timepoints < 2:
        raise ValueError("--timepoints must be at least 2")
    if args.time_max_ns <= args.time_min_ns:
        raise ValueError("--time-max-ns must be greater than --time-min-ns")

    stride = args.block_size if args.stride is None else args.stride
    blocks = block_indices(args.image_size, args.block_size, stride)
    _x_train, _y_train, x_test, y_test = load_mnist_sequence(
        args.train_samples, args.test_samples, args.image_size, args.seed
    )
    weights, local_bias, gains, output_bias = load_checkpoint(
        Path(args.init_weights),
        10,
        len(blocks),
        args.block_size * args.block_size,
    )
    t0 = time.perf_counter()
    evidence = local_evidence(x_test, weights, local_bias, gains, blocks)
    tau_act_values = parse_ns_list(args.tau_act_ns)
    tau_score_values = parse_ns_list(args.tau_score_ns)
    readout_times = np.linspace(args.time_min_ns, args.time_max_ns, args.timepoints)

    rows = []
    for tau_act_ns in tau_act_values:
        for tau_score_ns in tau_score_values:
            tau_act_s = tau_act_ns * 1e-9
            tau_score_s = tau_score_ns * 1e-9
            for t_ns in readout_times:
                acc, correct = accuracy_at_time(
                    evidence,
                    output_bias,
                    y_test,
                    float(t_ns) * 1e-9,
                    tau_act_s,
                    tau_score_s,
                )
                rows.append(
                    {
                        "readout_time_ns": float(t_ns),
                        "tau_act_ns": float(tau_act_ns),
                        "tau_score_ns": float(tau_score_ns),
                        "accuracy": float(acc),
                        "correct": correct,
                        "total": int(len(y_test)),
                        "time_over_tau_act": float(t_ns / tau_act_ns) if tau_act_ns else np.inf,
                        "time_over_tau_score": float(t_ns / tau_score_ns) if tau_score_ns else np.inf,
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
    margin_values_pp = sorted(set(parse_pp_list(args.frontier_margins_pp) + [1.0]))
    targets = frontier_targets(frontier, float(best["accuracy"]), margin_values_pp)
    fastest_final = targets[targets["within_best_pp"] == 1.0].iloc[0].to_dict()
    terminal_best = terminal.sort_values(["terminal_accuracy", "readout_time_ns"], ascending=[False, True]).iloc[0].to_dict()

    validation = None
    if args.spice_reference_csv:
        ref_n = args.spice_reference_test_samples or args.test_samples
        _rx_train, _ry_train, rx_test, ry_test = load_mnist_sequence(
            args.train_samples, ref_n, args.image_size, args.seed
        )
        ref_evidence = local_evidence(rx_test, weights, local_bias, gains, blocks)
        validation = validate_against_spice(
            Path(args.spice_reference_csv),
            ref_evidence,
            output_bias,
            ry_test,
            args.reference_tau_act_ns,
            args.reference_tau_score_ns,
            args.validation_ignore_before_ns,
        )

    stem = f"spice_mnist_settling_pareto_{sanitize_tag(args.tag)}"
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
        f"Settling time/accuracy frontier ({args.image_size}x{args.image_size}, n={args.test_samples})",
        steady_acc,
    )
    summary = {
        "architecture": "local_block_settling_time_accuracy_surrogate",
        "init_weights": args.init_weights,
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "image_size": args.image_size,
        "block_size": args.block_size,
        "stride": stride,
        "blocks": len(blocks),
        "classes": 10,
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
        "fastest_within_1pct_of_best_accuracy": float(fastest_final["accuracy"]),
        "fastest_within_1pct_of_best_readout_time_ns": float(fastest_final["readout_time_ns"]),
        "fastest_within_1pct_of_best_tau_act_ns": float(fastest_final["tau_act_ns"]),
        "fastest_within_1pct_of_best_tau_score_ns": float(fastest_final["tau_score_ns"]),
        "frontier_targets": targets.to_dict(orient="records"),
        "all_csv": str(all_csv),
        "table_all_csv": str(table_all_csv),
        "frontier_csv": str(frontier_csv),
        "pair_best_csv": str(pair_best_csv),
        "frontier_targets_csv": str(targets_csv),
        "figure": str(fig_path),
        "spice_validation": validation,
        "wall_time_s": time.perf_counter() - t0,
        "note": (
            "Analytical first-order settling surrogate for the SPICE local-block activation and score capacitors. "
            "It is a fixed-checkpoint readout-time and time-constant sweep, not a training run."
        ),
    }
    summary_path = spice_results / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
