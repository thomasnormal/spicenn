from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_spice_mnist_local_block_batch_op_train import block_indices
from run_spice_mnist_settling_pareto import load_checkpoint, local_evidence
from run_spice_mnist_train import load_mnist_sequence
from run_spice_sweep import ROOT


def sanitize_tag(tag: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in tag)


def accuracy(evidence: np.ndarray, labels: np.ndarray, output_bias: np.ndarray) -> tuple[float, int]:
    pred = np.argmax(evidence + output_bias[None, :], axis=1)
    correct = int(np.sum(pred == labels))
    return correct / max(len(labels), 1), correct


def coordinate_calibrate_bias(
    evidence: np.ndarray,
    labels: np.ndarray,
    output_bias: np.ndarray,
    radius: float,
    grid_points: int,
    epochs: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    if grid_points < 3:
        raise ValueError("--grid-points must be at least 3")
    bias = output_bias.astype(float).copy()
    rows = []
    base_acc, base_correct = accuracy(evidence, labels, bias)
    rows.append(
        {
            "epoch": -1,
            "class": -1,
            "train_accuracy": base_acc,
            "train_correct": base_correct,
            "bias": np.nan,
            "changed": False,
        }
    )
    for epoch in range(epochs):
        changed_any = False
        for klass in range(len(bias)):
            current_acc, _current_correct = accuracy(evidence, labels, bias)
            best_acc = current_acc
            best_bias = float(bias[klass])
            for candidate in np.linspace(bias[klass] - radius, bias[klass] + radius, grid_points):
                trial = bias.copy()
                trial[klass] = float(candidate)
                acc, _correct = accuracy(evidence, labels, trial)
                if acc > best_acc + 1e-12:
                    best_acc = acc
                    best_bias = float(candidate)
            changed = abs(best_bias - float(bias[klass])) > 1e-15
            if changed:
                bias[klass] = best_bias
                changed_any = True
            train_acc, train_correct = accuracy(evidence, labels, bias)
            rows.append(
                {
                    "epoch": epoch,
                    "class": klass,
                    "train_accuracy": train_acc,
                    "train_correct": train_correct,
                    "bias": float(bias[klass]),
                    "changed": changed,
                }
            )
        if not changed_any:
            break
    return bias, pd.DataFrame(rows)


def plot_bias_delta(original: np.ndarray, calibrated: np.ndarray, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    classes = np.arange(len(original))
    plt.figure(figsize=(7, 4))
    plt.bar(classes - 0.18, original, width=0.36, label="original")
    plt.bar(classes + 0.18, calibrated, width=0.36, label="calibrated")
    plt.xlabel("class")
    plt.ylabel("output bias voltage")
    plt.xticks(classes)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration-samples", type=int, default=2000)
    ap.add_argument("--test-samples", type=int, default=1000)
    ap.add_argument("--image-size", type=int, default=14)
    ap.add_argument("--block-size", type=int, default=7)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-weights", required=True)
    ap.add_argument("--radius", type=float, default=0.3)
    ap.add_argument("--grid-points", type=int, default=121)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--tag", default="output_bias_cal")
    args = ap.parse_args()

    if args.calibration_samples <= 0 or args.test_samples <= 0:
        raise ValueError("sample counts must be positive")
    stride = args.block_size if args.stride is None else args.stride
    blocks = block_indices(args.image_size, args.block_size, stride)
    x_cal, y_cal, x_test, y_test = load_mnist_sequence(
        args.calibration_samples, args.test_samples, args.image_size, args.seed
    )
    weights, local_bias, gains, output_bias = load_checkpoint(
        Path(args.init_weights),
        10,
        len(blocks),
        args.block_size * args.block_size,
    )

    t0 = time.perf_counter()
    cal_evidence = local_evidence(x_cal, weights, local_bias, gains, blocks)
    test_evidence = local_evidence(x_test, weights, local_bias, gains, blocks)
    base_cal_acc, base_cal_correct = accuracy(cal_evidence, y_cal, output_bias)
    base_test_acc, base_test_correct = accuracy(test_evidence, y_test, output_bias)
    calibrated_bias, trace = coordinate_calibrate_bias(
        cal_evidence,
        y_cal,
        output_bias,
        args.radius,
        args.grid_points,
        args.epochs,
    )
    cal_acc, cal_correct = accuracy(cal_evidence, y_cal, calibrated_bias)
    test_acc, test_correct = accuracy(test_evidence, y_test, calibrated_bias)

    stem = f"spice_mnist_output_bias_cal_{sanitize_tag(args.tag)}"
    spice_results = ROOT / "spice/results"
    tables = ROOT / "results/tables"
    figures = ROOT / "results/figures"
    for directory in [spice_results, tables, figures]:
        directory.mkdir(parents=True, exist_ok=True)

    final_weights = spice_results / f"{stem}_final_weights.npz"
    np.savez_compressed(
        final_weights,
        weights=weights,
        local_bias=local_bias,
        gains=gains,
        output_bias=calibrated_bias,
    )
    trace_csv = spice_results / f"{stem}_trace.csv"
    table_trace_csv = tables / f"{stem}_trace.csv"
    trace.to_csv(trace_csv, index=False)
    trace.to_csv(table_trace_csv, index=False)
    fig_path = figures / f"{stem}_biases.png"
    plot_bias_delta(output_bias, calibrated_bias, fig_path)
    summary = {
        "architecture": "local_block_output_bias_capacitor_calibration",
        "init_weights": args.init_weights,
        "image_size": args.image_size,
        "block_size": args.block_size,
        "stride": stride,
        "blocks": len(blocks),
        "classes": 10,
        "calibration_samples": args.calibration_samples,
        "test_samples": args.test_samples,
        "radius": args.radius,
        "grid_points": args.grid_points,
        "epochs": args.epochs,
        "base_calibration_accuracy": base_cal_acc,
        "base_calibration_correct": base_cal_correct,
        "base_test_accuracy": base_test_acc,
        "base_test_correct": base_test_correct,
        "calibrated_calibration_accuracy": cal_acc,
        "calibrated_calibration_correct": cal_correct,
        "calibrated_test_accuracy": test_acc,
        "calibrated_test_correct": test_correct,
        "original_output_bias": output_bias.tolist(),
        "calibrated_output_bias": calibrated_bias.tolist(),
        "output_bias_delta": (calibrated_bias - output_bias).tolist(),
        "final_weights": str(final_weights),
        "trace_csv": str(trace_csv),
        "table_trace_csv": str(table_trace_csv),
        "figure": str(fig_path),
        "wall_time_s": time.perf_counter() - t0,
        "note": (
            "Train-side coordinate calibration of the ten output-bias capacitor initial voltages. "
            "Forward evidence is evaluated by the same local-block equations; this is a cheap meta-calibration "
            "step to test whether the accuracy plateau is partly an output-bias operating point issue."
        ),
    }
    summary_path = spice_results / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
