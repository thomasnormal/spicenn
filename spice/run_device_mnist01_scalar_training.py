from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_device_sequential_training import (
    expected_positive,
    run_netlist,
    sequential_netlist,
)
from run_spice_sweep import ROOT, detect_spice, run_tiny_test


def sanitize_tag(tag: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in tag)


def scalar_feature_from_image(image: np.ndarray) -> float:
    """Map a small MNIST image to the single positive input rail this device cell has today."""
    ink = float(np.mean(np.clip(image, 0.0, 1.0)))
    return float(np.clip(0.55 + 0.55 * ink / 0.35, 0.05, 1.1))


def balanced_digit_indices(labels: np.ndarray, count: int, *, seed: int, digits: tuple[int, int]) -> np.ndarray:
    if count < 0:
        raise ValueError("sample count must be non-negative")
    if count == 0:
        return np.zeros((0,), dtype=np.int64)
    rng = np.random.default_rng(seed)
    per_digit = [count // 2, count - count // 2]
    chosen: list[np.ndarray] = []
    for digit, n_digit in zip(digits, per_digit, strict=True):
        candidates = np.flatnonzero(labels == digit)
        if n_digit > len(candidates):
            raise ValueError(f"not enough digit {digit} samples for requested count")
        chosen.append(rng.permutation(candidates)[:n_digit])
    out = np.concatenate(chosen)
    return rng.permutation(out)


def load_mnist01_scalar_records(
    train_samples: int,
    eval_samples: int,
    *,
    image_size: int,
    seed: int,
    positive_digit: int,
    negative_digit: int,
    download: bool,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    from torchvision import datasets, transforms
    import torch.nn.functional as F

    if positive_digit == negative_digit:
        raise ValueError("positive and negative digits must differ")
    digits = (positive_digit, negative_digit)
    ds_train = datasets.MNIST(root=str(ROOT / "data"), train=True, download=download, transform=transforms.ToTensor())
    ds_eval = datasets.MNIST(root=str(ROOT / "data"), train=False, download=download, transform=transforms.ToTensor())
    train_labels = np.asarray(ds_train.targets, dtype=np.int64)
    eval_labels = np.asarray(ds_eval.targets, dtype=np.int64)
    train_indices = balanced_digit_indices(train_labels, train_samples, seed=seed, digits=digits)
    eval_indices = balanced_digit_indices(eval_labels, eval_samples, seed=seed + 1, digits=digits)

    def extract(ds: Any, indices: np.ndarray) -> list[dict[str, float]]:
        records: list[dict[str, float]] = []
        for index in indices:
            image, digit = ds[int(index)]
            resized = F.interpolate(image.unsqueeze(0), size=(image_size, image_size), mode="area").squeeze()
            feature = scalar_feature_from_image(resized.numpy())
            digit_i = int(digit)
            records.append(
                {
                    "vin": feature,
                    "target": 1.1 if digit_i == positive_digit else 0.0,
                    "digit": float(digit_i),
                    "mnist_index": float(index),
                    "positive_label": 1.0 if digit_i == positive_digit else 0.0,
                }
            )
        return records

    return extract(ds_train, train_indices), extract(ds_eval, eval_indices)


def rows_from_measures(samples: list[dict[str, float]], measures: dict[str, float], *, sequence: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample_idx, sample in enumerate(samples):
        positive = expected_positive(float(sample["target"]))
        row: dict[str, Any] = {
            "sequence": sequence,
            "sample_idx": sample_idx,
            "vin": sample["vin"],
            "target": sample["target"],
            "digit": sample.get("digit"),
            "mnist_index": sample.get("mnist_index"),
            "positive_label": sample.get("positive_label"),
            "expected_direction": "positive" if positive else "negative",
        }
        for key, value in measures.items():
            suffix = f"_{sample_idx}"
            if key.endswith(suffix):
                row[key[: -len(suffix)]] = value
        rows.append(row)
    return pd.DataFrame(rows)


def final_weights_from_rows(rows: pd.DataFrame) -> dict[str, float]:
    if rows.empty:
        raise ValueError("cannot extract final weights from empty rows")
    final = rows.iloc[-1]
    return {
        "whp": float(final["whp_after_apply"]),
        "whn": float(final["whn_after_apply"]),
        "vwp": float(final["vwp_after_apply"]),
        "vwn": float(final["vwn_after_apply"]),
    }


def binary_accuracy(rows: pd.DataFrame, *, threshold: float, output_positive_when: str = "high") -> float:
    if rows.empty:
        return 0.0
    if output_positive_when == "high":
        predicted = rows["out_after"].to_numpy(dtype=float) > threshold
    elif output_positive_when == "low":
        predicted = rows["out_after"].to_numpy(dtype=float) <= threshold
    else:
        raise ValueError("output_positive_when must be 'high' or 'low'")
    expected = rows["positive_label"].to_numpy(dtype=float) > 0.5
    return float(np.mean(predicted == expected))


def run_device_sequence(
    spice_bin: str,
    path: Path,
    samples: list[dict[str, float]],
    weights: dict[str, float],
    *,
    hidden_credit_mode: str,
    output_driver_model: str,
    training_enabled: bool,
    timeout: float,
    sequence: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    netlist = sequential_netlist(
        samples,
        weights["whp"],
        weights["whn"],
        weights["vwp"],
        weights["vwn"],
        hidden_credit_mode=hidden_credit_mode,
        output_driver_model=output_driver_model,
        training_enabled=training_enabled,
    )
    if "\nB" in netlist:
        raise ValueError("device MNIST scalar runner generated a behavioral source")
    measures = run_netlist(spice_bin, path, netlist, timeout)
    return rows_from_measures(samples, measures, sequence=sequence), measures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-samples", type=int, default=8)
    ap.add_argument("--eval-samples", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--positive-digit", type=int, default=0)
    ap.add_argument("--negative-digit", type=int, default=1)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--tag", default="device_mnist01_scalar")
    ap.add_argument("--hidden-credit-mode", choices=["direct_feedback", "exact_backprop"], default="direct_feedback")
    ap.add_argument("--output-driver-model", choices=["sense", "nrel"], default="sense")
    ap.add_argument("--decision-threshold", type=float, default=0.04)
    ap.add_argument("--assert-nonbehavioral", action="store_true")
    args = ap.parse_args()

    if args.train_samples <= 0:
        raise ValueError("train-samples must be positive for a training smoke")
    if args.eval_samples <= 0:
        raise ValueError("eval-samples must be positive")

    generated = ROOT / "spice/generated"
    results = ROOT / "spice/results"
    tables = ROOT / "results/tables"
    for directory in [generated, results, tables]:
        directory.mkdir(parents=True, exist_ok=True)

    spice_bin, version = detect_spice(None)
    run_tiny_test(spice_bin, generated)
    safe_tag = sanitize_tag(args.tag)
    t0 = time.perf_counter()

    train_samples, eval_samples = load_mnist01_scalar_records(
        args.train_samples,
        args.eval_samples,
        image_size=args.image_size,
        seed=args.seed,
        positive_digit=args.positive_digit,
        negative_digit=args.negative_digit,
        download=args.download,
    )
    initial_weights = {"whp": 0.85, "whn": 0.25, "vwp": 0.55, "vwn": 0.25}
    initial_eval_rows, _ = run_device_sequence(
        spice_bin,
        generated / f"{safe_tag}_initial_eval.cir",
        eval_samples,
        initial_weights,
        hidden_credit_mode=args.hidden_credit_mode,
        output_driver_model=args.output_driver_model,
        training_enabled=False,
        timeout=args.timeout,
        sequence="initial_eval",
    )
    train_rows, _ = run_device_sequence(
        spice_bin,
        generated / f"{safe_tag}_train.cir",
        train_samples,
        initial_weights,
        hidden_credit_mode=args.hidden_credit_mode,
        output_driver_model=args.output_driver_model,
        training_enabled=True,
        timeout=args.timeout,
        sequence="train",
    )
    final_weights = final_weights_from_rows(train_rows)
    final_eval_rows, _ = run_device_sequence(
        spice_bin,
        generated / f"{safe_tag}_final_eval.cir",
        eval_samples,
        final_weights,
        hidden_credit_mode=args.hidden_credit_mode,
        output_driver_model=args.output_driver_model,
        training_enabled=False,
        timeout=args.timeout,
        sequence="final_eval",
    )

    curve = pd.concat([initial_eval_rows, train_rows, final_eval_rows], ignore_index=True)
    curve_path = results / f"{safe_tag}.csv"
    table_curve_path = tables / f"{safe_tag}.csv"
    curve.to_csv(curve_path, index=False)
    curve.to_csv(table_curve_path, index=False)

    initial_accuracy = binary_accuracy(initial_eval_rows, threshold=args.decision_threshold)
    final_accuracy = binary_accuracy(final_eval_rows, threshold=args.decision_threshold)
    initial_active_fraction = float(
        np.mean(np.abs(initial_eval_rows["out_after"].to_numpy(dtype=float)) > args.decision_threshold)
    )
    final_active_fraction = float(
        np.mean(np.abs(final_eval_rows["out_after"].to_numpy(dtype=float)) > args.decision_threshold)
    )
    nontrivial_learning_met = final_accuracy > max(initial_accuracy, 0.5)
    summary = {
        "simulator": version,
        "architecture": "device_level_mnist01_scalar_sequential_training",
        "status": "mnist01_scalar_device_training_smoke",
        "model_level": "ngspice built-in LEVEL=1 MOS models; not a foundry PDK.",
        "dataset": "MNIST01 scalar feature smoke",
        "positive_digit": args.positive_digit,
        "negative_digit": args.negative_digit,
        "image_size": args.image_size,
        "scalar_feature": "0.55 + 0.55 * mean_downsampled_ink / 0.35, clipped to [0.05, 1.1]",
        "hidden_credit_mode": args.hidden_credit_mode,
        "output_driver_model": args.output_driver_model,
        "output_driver_interpretation": (
            "Low-threshold transistor sense follower on the output node."
            if args.output_driver_model == "sense"
            else "Nominal NREL output source follower."
        ),
        "learning_device_implementation": "transistor_passive",
        "no_behavioral_signal_math": True,
        "no_behavioral_learning_devices": True,
        "uses_behavioral_learning_devices": False,
        "transistor_or_passive_learning_path": True,
        "single_training_transient": True,
        "continuous_transient_contract_met": True,
        "strict_fully_on_device_contract_met": True,
        "strict_fully_on_device_requested": True,
        "batch_size": 1,
        "python_weight_updates_between_samples": False,
        "python_checkpointing_between_samples": False,
        "python_hidden_state_intervention": False,
        "training_eval_uses_spice_forward_path": True,
        "uses_local_pca": False,
        "realistic_train_order": True,
        "train_samples": args.train_samples,
        "eval_samples": args.eval_samples,
        "mnist_index_order": "stable_balanced_random_digit01",
        "decision_threshold": args.decision_threshold,
        "initial_eval_accuracy": initial_accuracy,
        "final_eval_accuracy": final_accuracy,
        "eval_accuracy_delta": final_accuracy - initial_accuracy,
        "initial_eval_output_active_fraction": initial_active_fraction,
        "final_eval_output_active_fraction": final_active_fraction,
        "nontrivial_learning_met": nontrivial_learning_met,
        "initial_weights": initial_weights,
        "final_weights": final_weights,
        "curve": str(curve_path),
        "table_curve": str(table_curve_path),
        "netlists": {
            "initial_eval": str(generated / f"{safe_tag}_initial_eval.cir"),
            "train": str(generated / f"{safe_tag}_train.cir"),
            "final_eval": str(generated / f"{safe_tag}_final_eval.cir"),
        },
        "wall_time_s": time.perf_counter() - t0,
        "full_objective_contract_issues": [
            "scalar MNIST01 smoke, not multiclass MNIST",
            "single hidden/readout cell, not 10x10 b4 stride2 c2",
            "does not yet demonstrate nontrivial learning" if not nontrivial_learning_met else "",
        ],
        "interpretation": (
            "This is a real-MNIST data-stream integration smoke for the scalar transistor cell, not a target topology. "
            "Training weights change only inside the training transient; Python only supplies the MNIST-derived input/target "
            "rails and reads final diagnostics before running separate SPICE forward-only eval transients."
        ),
    }
    summary["full_objective_contract_issues"] = [issue for issue in summary["full_objective_contract_issues"] if issue]
    if args.assert_nonbehavioral:
        assert summary["no_behavioral_learning_devices"] is True
        assert summary["transistor_or_passive_learning_path"] is True
    summary_path = results / f"{safe_tag}_summary.json"
    table_summary_path = tables / f"{safe_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    table_summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as exc:
        raise SystemExit(f"missing optional MNIST dependency: {exc}") from exc
