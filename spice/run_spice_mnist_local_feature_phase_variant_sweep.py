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
DEFAULT_SAMPLE_EDGE = 0.0
DEFAULT_HIDDEN_PREACTIVATION_MODE = "inline"
DEFAULT_HIDDEN_ACTIVATION_MODE = "stored"
DEFAULT_SCORE_CALCULATION_MODE = "inline"
DEFAULT_OUTPUT_RAIL_MODE = "inline"
DEFAULT_OUTPUT_DELTA_MODE = "node"


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


def variant_tag(
    base_tag: str,
    activation: str,
    relu_clip: float,
    derivative_mode: str = "exact",
    feedback_mode: str = "readout",
    hidden_synapse_mode: str = "linear",
    readout_synapse_mode: str = "linear",
    synapse_clip: float = 1.0,
) -> str:
    safe_activation = sanitize_tag(activation.replace("-", "_"))
    safe_derivative = sanitize_tag(derivative_mode.replace("-", "_"))
    safe_feedback = sanitize_tag(feedback_mode.replace("-", "_"))
    safe_hidden_synapse = sanitize_tag(hidden_synapse_mode.replace("-", "_"))
    safe_readout_synapse = sanitize_tag(readout_synapse_mode.replace("-", "_"))
    return sanitize_tag(
        f"{base_tag}_{safe_activation}_clip{relu_clip:g}_{safe_derivative}_{safe_feedback}"
        f"_hsyn_{safe_hidden_synapse}_rsyn_{safe_readout_synapse}_synclip{synapse_clip:g}"
    )


def build_variant_command(
    args: argparse.Namespace,
    activation: str,
    relu_clip: float,
    derivative_mode: str = "exact",
    feedback_mode: str = "readout",
    hidden_synapse_mode: str = "linear",
    readout_synapse_mode: str = "linear",
    synapse_clip: float | None = None,
) -> list[str]:
    if synapse_clip is None:
        synapse_clip = args.synapse_clip
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
        "--max-transient-points",
        str(getattr(args, "max_transient_points", 0)),
        "--max-source-pwl-points",
        str(getattr(args, "max_source_pwl_points", 0)),
        "--max-sample-sources",
        str(getattr(args, "max_sample_sources", 0)),
        "--max-total-sources",
        str(getattr(args, "max_total_sources", 0)),
        "--max-output-vectors",
        str(getattr(args, "max_output_vectors", 0)),
        "--max-auxiliary-algebraic-sources",
        str(getattr(args, "max_auxiliary_algebraic_sources", 0)),
        "--reference-mode",
        args.reference_mode,
        "--phase-output-mode",
        args.phase_output_mode,
        "--update-mode",
        args.update_mode,
        "--phase-clock-mode",
        args.phase_clock_mode,
        "--target-source-mode",
        args.target_source_mode,
        "--eval-backend",
        args.eval_backend,
        "--probe-updates",
        args.probe_updates,
        "--local-activation",
        activation,
        "--relu-clip",
        str(relu_clip),
        "--relu-leak",
        str(args.relu_leak),
        "--softplus-beta",
        str(args.softplus_beta),
        "--activation-derivative",
        derivative_mode,
        "--derivative-floor",
        str(args.derivative_floor),
        "--derivative-gate-threshold",
        str(args.derivative_gate_threshold),
        "--readout-feedback-mode",
        feedback_mode,
        "--readout-feedback-clip",
        str(args.readout_feedback_clip),
        "--output-bias-update-scale",
        str(args.output_bias_update_scale),
        "--readout-update-scale",
        str(args.readout_update_scale),
        "--local-update-scale",
        str(args.local_update_scale),
        "--state-decay",
        str(args.state_decay),
        "--softmax-negative-scale",
        str(args.softmax_negative_scale),
        "--softmax-error-centering",
        args.softmax_error_centering,
        "--softmax-temperature",
        str(args.softmax_temperature),
        "--softmax-competition-mode",
        args.softmax_competition_mode,
        "--softmax-competitor-power",
        str(args.softmax_competitor_power),
        "--softmax-error-gate",
        args.softmax_error_gate,
        "--softmax-margin",
        str(args.softmax_margin),
        "--hidden-synapse-mode",
        hidden_synapse_mode,
        "--readout-synapse-mode",
        readout_synapse_mode,
        "--synapse-clip",
        str(synapse_clip),
        "--readout-class-centering",
        args.readout_class_centering,
        "--hidden-preactivation-mode",
        args.hidden_preactivation_mode,
        "--hidden-activation-mode",
        getattr(args, "hidden_activation_mode", DEFAULT_HIDDEN_ACTIVATION_MODE),
        "--score-calculation-mode",
        args.score_calculation_mode,
        "--output-rail-mode",
        args.output_rail_mode,
        "--output-delta-mode",
        args.output_delta_mode,
        "--tag",
        variant_tag(
            args.tag,
            activation,
            relu_clip,
            derivative_mode,
            feedback_mode,
            hidden_synapse_mode,
            readout_synapse_mode,
            synapse_clip,
        ),
    ]
    if getattr(args, "sample_edge", None) is not None:
        command.extend(["--sample-edge", str(args.sample_edge)])
    if args.softmax_output:
        command.append("--softmax-output")
    if args.linear_output:
        command.append("--linear-output")
    if args.final_measures:
        command.append("--final-measures")
    if args.eval_probe_updates:
        command.append("--eval-probe-updates")
    if args.strict_fully_on_device:
        command.append("--strict-fully-on-device")
    if args.simulator_extra_args:
        command.extend(["--simulator-extra-args", args.simulator_extra_args])
    return command


def parse_runner_json(stdout: str) -> dict[str, Any]:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"runner did not print a JSON object:\n{stdout[-2000:]}")
    return json.loads(stdout[start : end + 1])


def row_from_summary(
    activation: str,
    relu_clip: float,
    derivative_mode: str,
    feedback_mode: str,
    hidden_synapse_mode: str,
    readout_synapse_mode: str,
    synapse_clip: float,
    summary: dict[str, Any],
    command: list[str],
) -> dict[str, Any]:
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
        "fully_on_device_execution_contract_met",
        "strict_fully_on_device_contract_met",
        "estimated_transient_points",
        "estimated_transient_points_per_update",
        "max_transient_points",
        "transient_budget_met",
        "phase_output_vector_count",
        "max_output_vectors",
        "output_vector_budget_met",
        "max_source_pwl_points",
        "source_pwl_budget_met",
        "max_sample_sources",
        "sample_source_budget_met",
        "max_total_sources",
        "total_source_budget_met",
        "max_auxiliary_algebraic_sources",
        "auxiliary_algebraic_source_budget_met",
        "sample_source_count",
        "sample_source_elided_dc_count",
        "sample_source_pwl_points",
        "pixel_source_count",
        "pixel_source_elided_dc_count",
        "target_source_count",
        "target_behavioral_source_count",
        "phase_clock_source_pwl_points",
        "control_source_pwl_points",
        "total_source_count",
        "total_source_pwl_points",
        "total_source_pwl_points_per_update",
        "output_mode",
        "reference_mode",
        "eval_backend",
        "update_mode",
        "sample_edge_s",
        "hidden_preactivation_mode",
        "hidden_preactivation_source_count",
        "score_calculation_mode",
        "score_calculation_source_count",
        "output_rail_mode",
        "output_rail_source_count",
        "output_delta_mode",
        "output_delta_state_count",
        "auxiliary_algebraic_source_count",
        "target_source_mode",
    ]
    row = {
        "local_activation": activation,
        "relu_clip": relu_clip,
        "activation_derivative": derivative_mode,
        "readout_feedback_mode": feedback_mode,
        "hidden_synapse_mode": hidden_synapse_mode,
        "readout_synapse_mode": readout_synapse_mode,
        "synapse_clip": synapse_clip,
        "tag": summary.get("tag"),
        "command": " ".join(command),
    }
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
    ap.add_argument("--sample-edge", type=float, default=DEFAULT_SAMPLE_EDGE)
    ap.add_argument("--settle-ratio", type=float, default=80.0)
    ap.add_argument("--transient-step", type=float, default=50e-12)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--max-transient-points", type=int, default=0)
    ap.add_argument("--max-source-pwl-points", type=int, default=0)
    ap.add_argument("--max-sample-sources", type=int, default=0)
    ap.add_argument("--max-total-sources", type=int, default=0)
    ap.add_argument("--max-output-vectors", type=int, default=0)
    ap.add_argument("--max-auxiliary-algebraic-sources", type=int, default=0)
    ap.add_argument("--reference-mode", choices=["spice", "none"], default="spice")
    ap.add_argument("--phase-output-mode", choices=["auto", "measure", "print", "control_measure", "wrdata"], default="auto")
    ap.add_argument("--update-mode", choices=["phased", "direct"], default="phased")
    ap.add_argument("--phase-clock-mode", choices=["pwl", "analytic"], default="pwl")
    ap.add_argument("--target-source-mode", choices=["rails", "label"], default="label")
    ap.add_argument("--eval-backend", choices=["spice", "numpy", "both"], default="spice")
    ap.add_argument("--probe-updates", default="1,2,4,8,final")
    ap.add_argument("--activations", default="tanh,diff-clipped-relu,relu,clipped-relu")
    ap.add_argument("--relu-clips", default="1.0")
    ap.add_argument("--relu-leak", type=float, default=0.01)
    ap.add_argument("--softplus-beta", type=float, default=10.0)
    ap.add_argument("--derivative-modes", default="exact")
    ap.add_argument("--derivative-floor", type=float, default=0.0)
    ap.add_argument("--derivative-gate-threshold", type=float, default=1e-6)
    ap.add_argument("--feedback-modes", default="readout")
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
    ap.add_argument("--hidden-synapse-modes", default="linear")
    ap.add_argument("--readout-synapse-modes", default="linear")
    ap.add_argument("--synapse-clip", type=float, default=1.0)
    ap.add_argument("--synapse-clips", default="")
    ap.add_argument("--readout-class-centering", choices=["none", "mean"], default="none")
    ap.add_argument("--hidden-preactivation-mode", choices=["node", "inline"], default=DEFAULT_HIDDEN_PREACTIVATION_MODE)
    ap.add_argument("--hidden-activation-mode", choices=["stored", "inline"], default=DEFAULT_HIDDEN_ACTIVATION_MODE)
    ap.add_argument("--score-calculation-mode", choices=["node", "inline"], default=DEFAULT_SCORE_CALCULATION_MODE)
    ap.add_argument("--output-rail-mode", choices=["node", "inline"], default=DEFAULT_OUTPUT_RAIL_MODE)
    ap.add_argument("--output-delta-mode", choices=["node", "inline"], default=DEFAULT_OUTPUT_DELTA_MODE)
    ap.add_argument("--softmax-output", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--linear-output", action="store_true")
    ap.add_argument("--final-measures", action="store_true")
    ap.add_argument("--eval-probe-updates", action="store_true")
    ap.add_argument("--strict-fully-on-device", action="store_true")
    ap.add_argument("--simulator-extra-args", default="")
    ap.add_argument("--tag", default="phase_variant_sweep")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.sample_edge is not None and args.sample_edge < 0:
        raise ValueError("--sample-edge must be non-negative")
    if args.max_source_pwl_points < 0:
        raise ValueError("--max-source-pwl-points must be non-negative")
    if args.max_sample_sources < 0:
        raise ValueError("--max-sample-sources must be non-negative")
    if args.max_total_sources < 0:
        raise ValueError("--max-total-sources must be non-negative")
    if args.max_output_vectors < 0:
        raise ValueError("--max-output-vectors must be non-negative")
    if args.max_auxiliary_algebraic_sources < 0:
        raise ValueError("--max-auxiliary-algebraic-sources must be non-negative")

    if args.linear_output and args.softmax_output:
        raise ValueError("--linear-output and --softmax-output are mutually exclusive; pass --no-softmax-output for linear variants")

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

    rows = []
    for activation, relu_clip in activation_clip_pairs(activations, relu_clips):
        for derivative_mode in derivative_modes:
            for feedback_mode in feedback_modes:
                for hidden_synapse_mode in hidden_synapse_modes:
                    for readout_synapse_mode in readout_synapse_modes:
                        for synapse_clip in synapse_clips:
                            command = build_variant_command(
                                args,
                                activation,
                                relu_clip,
                                derivative_mode,
                                feedback_mode,
                                hidden_synapse_mode,
                                readout_synapse_mode,
                                synapse_clip,
                            )
                            if args.dry_run:
                                rows.append(
                                    {
                                        "local_activation": activation,
                                        "relu_clip": relu_clip,
                                        "activation_derivative": derivative_mode,
                                        "readout_feedback_mode": feedback_mode,
                                        "hidden_synapse_mode": hidden_synapse_mode,
                                        "readout_synapse_mode": readout_synapse_mode,
                                        "synapse_clip": synapse_clip,
                                        "command": " ".join(command),
                                    }
                                )
                                continue
                            proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=args.timeout + 60.0)
                            if proc.returncode != 0:
                                raise RuntimeError(
                                    f"variant {activation} clip {relu_clip:g} derivative {derivative_mode} feedback {feedback_mode} "
                                    f"hidden_synapse {hidden_synapse_mode} readout_synapse {readout_synapse_mode} "
                                    f"synapse_clip {synapse_clip:g} failed:"
                                    f"\nSTDOUT:\n{proc.stdout[-3000:]}\nSTDERR:\n{proc.stderr[-3000:]}"
                                )
                            summary = parse_runner_json(proc.stdout)
                            summary["tag"] = variant_tag(
                                args.tag,
                                activation,
                                relu_clip,
                                derivative_mode,
                                feedback_mode,
                                hidden_synapse_mode,
                                readout_synapse_mode,
                                synapse_clip,
                            )
                            rows.append(
                                row_from_summary(
                                    activation,
                                    relu_clip,
                                    derivative_mode,
                                    feedback_mode,
                                    hidden_synapse_mode,
                                    readout_synapse_mode,
                                    synapse_clip,
                                    summary,
                                    command,
                                )
                            )
                            print(json.dumps(rows[-1], indent=2))

    results = ROOT / "spice/results"
    results.mkdir(parents=True, exist_ok=True)
    out_csv = results / f"{sanitize_tag(args.tag)}_variant_sweep.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(json.dumps({"variant_sweep_csv": str(out_csv), "variants": len(rows), "dry_run": args.dry_run}, indent=2))


if __name__ == "__main__":
    main()
