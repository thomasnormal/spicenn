from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_mnist01_fixed_feature_divider_training as mnist01_fixed
import run_mnist01_live_hidden_divider_training as mnist01_hidden
from run_spice_sweep import ROOT, detect_spice


def _current_best_patch2x2_netlist(
    train: list[dict[str, Any]],
    evals: list[dict[str, Any]],
) -> str:
    hidden_count = mnist01_hidden.patch2x2_hidden_count(len(train[0]["features"]))
    return mnist01_hidden.mnist01_live_hidden_netlist(
        train,
        evals,
        hidden_count=hidden_count,
        hidden_init_mode="patch2x2",
        hidden_connectivity_mode="patch2x2-sparse",
        hidden_inside_positive=0.75,
        hidden_inside_negative=0.15,
        hidden_outside_positive=0.15,
        hidden_outside_negative=0.15,
        hidden_activation_mode="differential-preamp",
        hidden_activation_sense_width_u=4.0,
        readout_activation_mode="pre-differential",
        readout_writer_activation_mode="pre-differential",
        readout_width_u=64.0,
        readout_update_width_u=0.25,
        hidden_credit_gate_mode="dynamic-preamp",
        hidden_error_route_width_u=8.0,
        hidden_writer_topology="pmos-signcharge",
        hidden_write_start_train_index=1,
        hidden_credit_sense_start_ns=5.00,
        hidden_credit_sense_end_ns=5.35,
        hidden_write_start_ns=5.35,
        hidden_write_end_ns=5.45,
        hidden_update_width_u=0.05,
        measure_eval_hidden_states=False,
        tran_step_ps=10.0,
    )


def _write_metric_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "forward_index",
        "cycle_index",
        "phase",
        "phase_index",
        "label",
        "target_signed_v",
        "other_signed_v",
        "margin_v",
        "correct",
        "softplus_loss",
        "cumulative_accuracy",
        "cumulative_mean_loss",
        "phase_cumulative_accuracy",
        "phase_cumulative_mean_loss",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_train_loss(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    train_rows = [row for row in rows if row["phase"] == "train"]
    x = [int(row["phase_index"]) for row in train_rows]
    loss = [float(row["softplus_loss"]) for row in train_rows]
    acc = [float(row["phase_cumulative_accuracy"]) for row in train_rows]

    fig, ax_loss = plt.subplots(figsize=(7, 4))
    ax_loss.plot(x, loss, marker="o", linewidth=1.8, label="train loss")
    ax_loss.set_xlabel("training forward pass")
    ax_loss.set_ylabel("softplus loss from measured margin")
    ax_loss.grid(True, alpha=0.25)
    ax_acc = ax_loss.twinx()
    ax_acc.plot(x, acc, color="C1", marker="s", linewidth=1.4, label="cumulative train accuracy")
    ax_acc.set_ylabel("cumulative train accuracy")
    ax_acc.set_ylim(-0.05, 1.05)
    ax_loss.set_title(title)
    lines, labels = ax_loss.get_legend_handles_labels()
    lines2, labels2 = ax_acc.get_legend_handles_labels()
    ax_loss.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-count-per-digit", type=int, default=3)
    parser.add_argument("--eval-count-per-digit", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=4)
    parser.add_argument("--loss-margin-scale-v", type=float, default=1.0e-3)
    parser.add_argument("--spice", default="ngspice")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--tag",
        default="codex_mnist01_live_hidden_patch2x2_complement_signcharge_3x3",
    )
    parser.add_argument("--table-dir", type=Path, default=ROOT / "results/tables")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "docs/figures")
    args = parser.parse_args()

    spice_bin, spice_version = detect_spice(args.spice)
    train, evals = mnist01_hidden.load_mnist01_records(
        train_count_per_digit=args.train_count_per_digit,
        eval_count_per_digit=args.eval_count_per_digit,
        image_size=args.image_size,
    )
    train = mnist01_fixed.add_complement_features(mnist01_fixed.round_robin_by_label(train), scale=0.5)
    evals = mnist01_fixed.add_complement_features(evals, scale=0.5)

    with tempfile.TemporaryDirectory(prefix=f"{args.tag}_") as tmp:
        measures = mnist01_hidden.run_netlist(
            spice_bin,
            Path(tmp) / f"{args.tag}.cir",
            _current_best_patch2x2_netlist(train, evals),
            timeout=args.timeout,
        )
    rows = mnist01_hidden.forward_metric_rows(
        train,
        evals,
        measures,
        loss_margin_scale_v=args.loss_margin_scale_v,
    )
    final_rows = [row for row in rows if row["phase"] == "final"]
    train_rows = [row for row in rows if row["phase"] == "train"]
    final_accuracy = sum(int(row["correct"]) for row in final_rows) / len(final_rows)
    train_accuracy = sum(int(row["correct"]) for row in train_rows) / len(train_rows)
    final_margins = [float(row["margin_v"]) for row in final_rows]

    csv_path = args.table_dir / f"{args.tag}_forward_metrics.csv"
    summary_path = args.table_dir / f"{args.tag}_summary.json"
    figure_path = args.figure_dir / f"{args.tag}_loss_curve.png"
    _write_metric_csv(csv_path, rows)
    _plot_train_loss(
        figure_path,
        rows,
        f"4x4 MNIST01 local RF signcharge, final acc {final_accuracy:.3f}",
    )
    summary = {
        "tag": args.tag,
        "spice": spice_version,
        "train_count_per_digit": args.train_count_per_digit,
        "eval_count_per_digit": args.eval_count_per_digit,
        "image_size": args.image_size,
        "loss_margin_scale_v": args.loss_margin_scale_v,
        "train_accuracy": train_accuracy,
        "final_accuracy": final_accuracy,
        "final_margin_min_v": min(final_margins),
        "final_margin_mean_v": sum(final_margins) / len(final_margins),
        "metric_csv": str(csv_path),
        "loss_curve_png": str(figure_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
