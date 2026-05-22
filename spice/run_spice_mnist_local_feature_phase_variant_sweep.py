from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from run_spice_mnist_local_feature_phase_transient import sanitize_tag
from run_spice_sweep import ROOT


CLIPPED_ACTIVATIONS = {"clipped-relu", "clipped_relu", "diff-clipped-relu", "differential-clipped-relu", "diff_clipped_relu"}


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_float_csv(text: str) -> list[float]:
    return [float(part) for part in parse_csv(text)]


def activation_clip_pairs(activations: list[str], relu_clips: list[float]) -> list[tuple[str, float]]:
    pairs: list[tuple[str, float]] = []
    for activation in activations:
        clips = relu_clips if activation in CLIPPED_ACTIVATIONS else [relu_clips[0]]
        for clip in clips:
            pairs.append((activation, clip))
    return pairs


def variant_tag(base_tag: str, activation: str, relu_clip: float) -> str:
    safe_activation = sanitize_tag(activation.replace("-", "_"))
    return sanitize_tag(f"{base_tag}_{safe_activation}_clip{relu_clip:g}")


def build_variant_command(args: argparse.Namespace, activation: str, relu_clip: float) -> list[str]:
    script = ROOT / "spice/run_spice_mnist_local_feature_phase_transient.py"
    command = [
        sys.executable,
        str(script),
        "--simulator",
        args.simulator,
        "--train-samples",
        str(args.train_samples),
        "--eval-samples",
        str(args.eval_samples),
        "--image-size",
        str(args.image_size),
        "--block-size",
        str(args.block_size),
        "--stride",
        str(args.stride),
        "--channels",
        str(args.channels),
        "--batch-size",
        "1",
        "--updates",
        str(args.updates),
        "--lr",
        str(args.lr),
        "--phase",
        str(args.phase),
        "--gap",
        str(args.gap),
        "--edge",
        str(args.edge),
        "--settle-ratio",
        str(args.settle_ratio),
        "--transient-step",
        str(args.transient_step),
        "--timeout",
        str(args.timeout),
        "--probe-updates",
        args.probe_updates,
        "--local-activation",
        activation,
        "--relu-clip",
        str(relu_clip),
        "--tag",
        variant_tag(args.tag, activation, relu_clip),
    ]
    if args.softmax_output:
        command.append("--softmax-output")
    if args.linear_output:
        command.append("--linear-output")
    if args.final_measures:
        command.append("--final-measures")
    return command


def parse_runner_json(stdout: str) -> dict[str, Any]:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"runner did not print a JSON object:\n{stdout[-2000:]}")
    return json.loads(stdout[start : end + 1])


def row_from_summary(activation: str, relu_clip: float, summary: dict[str, Any], command: list[str]) -> dict[str, Any]:
    keys = [
        "continuous_transient_contract_met",
        "direction_matches_batch_op_reference",
        "nontrivial_learning_met",
        "initial_eval_accuracy",
        "phase_eval_accuracy",
        "op_reference_eval_accuracy",
        "phase_eval_improvement",
        "state_update_direction_cosine",
        "state_update_sign_alignment_fraction",
        "state_update_wrong_sign_count",
        "state_max_abs_diff",
        "phase_wall_time_s",
        "op_reference_wall_time_s",
        "eval_wall_time_s",
    ]
    row = {"local_activation": activation, "relu_clip": relu_clip, "tag": summary.get("tag"), "command": " ".join(command)}
    row.update({key: summary.get(key) for key in keys})
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulator", default="Xyce")
    ap.add_argument("--train-samples", type=int, default=20)
    ap.add_argument("--eval-samples", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=10)
    ap.add_argument("--block-size", type=int, default=4)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--updates", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.8)
    ap.add_argument("--phase", type=float, default=1e-9)
    ap.add_argument("--gap", type=float, default=0.1e-9)
    ap.add_argument("--edge", type=float, default=10e-12)
    ap.add_argument("--settle-ratio", type=float, default=80.0)
    ap.add_argument("--transient-step", type=float, default=50e-12)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--probe-updates", default="1,2,4,8,final")
    ap.add_argument("--activations", default="tanh,diff-clipped-relu,relu,clipped-relu")
    ap.add_argument("--relu-clips", default="1.0")
    ap.add_argument("--softmax-output", action="store_true", default=True)
    ap.add_argument("--linear-output", action="store_true")
    ap.add_argument("--final-measures", action="store_true")
    ap.add_argument("--tag", default="phase_variant_sweep")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    activations = parse_csv(args.activations)
    relu_clips = parse_float_csv(args.relu_clips)
    if not activations:
        raise ValueError("--activations must not be empty")
    if not relu_clips:
        raise ValueError("--relu-clips must not be empty")

    rows = []
    for activation, relu_clip in activation_clip_pairs(activations, relu_clips):
        command = build_variant_command(args, activation, relu_clip)
        if args.dry_run:
            rows.append({"local_activation": activation, "relu_clip": relu_clip, "command": " ".join(command)})
            continue
        proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=args.timeout + 60.0)
        if proc.returncode != 0:
            raise RuntimeError(f"variant {activation} clip {relu_clip:g} failed:\nSTDOUT:\n{proc.stdout[-3000:]}\nSTDERR:\n{proc.stderr[-3000:]}")
        summary = parse_runner_json(proc.stdout)
        summary["tag"] = variant_tag(args.tag, activation, relu_clip)
        rows.append(row_from_summary(activation, relu_clip, summary, command))
        print(json.dumps(rows[-1], indent=2))

    results = ROOT / "spice/results"
    results.mkdir(parents=True, exist_ok=True)
    out_csv = results / f"{sanitize_tag(args.tag)}_variant_sweep.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(json.dumps({"variant_sweep_csv": str(out_csv), "variants": len(rows), "dry_run": args.dry_run}, indent=2))


if __name__ == "__main__":
    main()
