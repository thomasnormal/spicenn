#!/usr/bin/env python3
"""Bayesian hyperparameter tuner for device-level SPICE NN experiments.

The tuner treats ``run_device_xor2_random_hidden.py`` as the ground-truth
experiment driver.  Optuna only proposes circuit knobs and launches that driver;
the objective value is read back from the driver's summary JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import optuna
from optuna.trial import Trial

import parameter_theory as theory
from datasets import parse_counted_mnist_dataset


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "spice" / "run_device_xor2_random_hidden.py"
SPICE_RESULTS = ROOT / "spice" / "results"
TABLES = ROOT / "results" / "tables"


Runner = Callable[..., subprocess.CompletedProcess[str]]


def fmt(value: float | int | str) -> str:
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def safe_tag(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)


def driver_arg_name(param_name: str) -> str:
    return "--" + param_name.replace("_", "-")


def params_to_driver_args(params: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for name, value in params.items():
        if value is None:
            continue
        flag = driver_arg_name(name)
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        args.extend([flag, fmt(value)])
    return args


def split_tuner_and_driver_args(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    argv = list(argv)
    if "--" not in argv:
        return argv, []
    index = argv.index("--")
    return argv[:index], argv[index + 1 :]


def last_flag_value(args: Sequence[str], flag: str) -> str | None:
    """Return the last explicit value passed for ``flag`` in a flat argv list."""
    value: str | None = None
    for index, item in enumerate(args):
        if item == flag and index + 1 < len(args):
            value = args[index + 1]
    return value


def effective_driver_arg(profile: "TuningProfile", extra_driver_args: Sequence[str], flag: str) -> str | None:
    """Return the final driver argument after profile args and user overrides."""
    return last_flag_value((*profile.base_args, *extra_driver_args), flag)


def class_count_for_dataset(dataset: str) -> int:
    counted = parse_counted_mnist_dataset(dataset)
    if counted is not None:
        class_count, _frontend, _sample_count = counted
        return class_count
    if dataset == "xor2" or dataset.startswith("moons") or dataset.startswith("mnist01"):
        return 2
    raise ValueError(f"cannot infer class count from dataset {dataset!r}")


@dataclass(frozen=True)
class TuningProfile:
    name: str
    description: str
    base_args: tuple[str, ...]
    sampler: str
    anchor_params: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class ArchitectureContext:
    """Small topology summary needed by theory-derived tuning profiles."""

    class_count: int
    hidden_cells: int
    readout_topology: str
    readout_fan_in: int
    readout_fan_out: int

    @property
    def effective_readout_fan_in(self) -> float:
        if self.readout_topology == "dense":
            return float(self.hidden_cells)
        if self.readout_topology == "random_fanin":
            return float(min(self.readout_fan_in, self.hidden_cells))
        if self.readout_topology in {"random_fanout", "balanced_random_fanout"}:
            fan_out = min(self.readout_fan_out, self.class_count)
            return float(self.hidden_cells * fan_out / self.class_count)
        raise ValueError(f"unknown readout topology {self.readout_topology!r}")


def effective_int_driver_arg(
    profile: TuningProfile,
    extra_driver_args: Sequence[str],
    flag: str,
    *,
    default: int,
) -> int:
    value = effective_driver_arg(profile, extra_driver_args, flag)
    return default if value is None else int(value)


def architecture_context_for_profile(
    profile: TuningProfile,
    extra_driver_args: Sequence[str],
) -> ArchitectureContext:
    dataset = effective_driver_arg(profile, extra_driver_args, "--dataset")
    if dataset is None:
        raise ValueError(f"profile {profile.name} has no effective --dataset")
    return ArchitectureContext(
        class_count=class_count_for_dataset(dataset),
        hidden_cells=effective_int_driver_arg(profile, extra_driver_args, "--hidden-cells", default=8),
        readout_topology=effective_driver_arg(profile, extra_driver_args, "--readout-topology") or "dense",
        readout_fan_in=effective_int_driver_arg(profile, extra_driver_args, "--readout-fan-in", default=3),
        readout_fan_out=effective_int_driver_arg(profile, extra_driver_args, "--readout-fan-out", default=3),
    )


def sample_pmos_charge_discharge(trial: Trial) -> dict[str, Any]:
    """Tune the current balanced PMOS charge/discharge readout writer."""
    low_v = trial.suggest_float("readout_write_low_v", 0.05, 0.30)
    high_v = trial.suggest_float("readout_write_high_v", max(0.72, low_v + 0.25), 1.10)
    center_enabled = trial.suggest_categorical("readout_center_pull_enabled", [False, True])
    bias_enabled = trial.suggest_categorical("output_bias_update_enabled", [False, True])
    return {
        "readout_flow_write_mode": "bounded_pmos_charge_discharge",
        "readout_write_error_exclusion": "diffpair_bleed",
        "readout_write_error_exclusion_width_u": trial.suggest_float(
            "readout_write_error_exclusion_width_u", 2.0, 24.0, log=True
        ),
        "readout_charge_update_width_u": trial.suggest_float(
            "readout_charge_update_width_u", 1e-4, 8e-3, log=True
        ),
        "readout_discharge_update_width_u": trial.suggest_float(
            "readout_discharge_update_width_u", 1e-5, 3e-3, log=True
        ),
        "readout_write_low_v": low_v,
        "readout_write_high_v": high_v,
        "readout_center_pull_width_u": trial.suggest_float(
            "readout_center_pull_width_u", 2e-6, 2e-3, log=True
        )
        if center_enabled
        else 0.0,
        "readout_center_pull_gate": trial.suggest_categorical("readout_center_pull_gate", ["bwd", "apply"]),
        "readout_center_pull_mode": trial.suggest_categorical("readout_center_pull_mode", ["always", "state_high"]),
        "output_bias_update_width_u": trial.suggest_float(
            "output_bias_update_width_u", 1e-5, 2e-3, log=True
        )
        if bias_enabled
        else 0.0,
        "output_bias_offset_v": trial.suggest_float("output_bias_offset_v", -0.45, 0.45),
        "output_bias_forward_width_scale": trial.suggest_float(
            "output_bias_forward_width_scale", 0.35, 2.5, log=True
        ),
        "score_cap_f": trial.suggest_float("score_cap_f", 5.0, 80.0, log=True),
        "score_reset_v": trial.suggest_float("score_reset_v", 0.0, 0.25),
    }


def sample_bounded_discharge(trial: Trial) -> dict[str, Any]:
    """Tune the older bounded-discharge recipe that has produced high MNIST-0/1 anchors."""
    center_enabled = trial.suggest_categorical("readout_center_pull_enabled", [False, True])
    return {
        "readout_flow_write_mode": "bounded_discharge",
        "readout_update_width_u": trial.suggest_float("readout_update_width_u", 16.0, 256.0, log=True),
        "readout_write_low_v": trial.suggest_float("readout_write_low_v", 0.08, 0.30),
        "readout_write_high_v": trial.suggest_float("readout_write_high_v", 0.45, 0.80),
        "readout_center_pull_width_u": trial.suggest_float(
            "readout_center_pull_width_u", 2e-6, 2e-3, log=True
        )
        if center_enabled
        else 0.0,
        "readout_center_pull_gate": trial.suggest_categorical("readout_center_pull_gate", ["bwd", "apply"]),
        "readout_center_pull_mode": trial.suggest_categorical("readout_center_pull_mode", ["always", "state_high"]),
        "output_bias_offset_v": trial.suggest_float("output_bias_offset_v", -0.45, 0.45),
        "output_bias_forward_width_scale": trial.suggest_float(
            "output_bias_forward_width_scale", 0.35, 2.5, log=True
        ),
        "score_cap_f": trial.suggest_float("score_cap_f", 5.0, 80.0, log=True),
    }


def sample_target_mistake_bounded_discharge(trial: Trial) -> dict[str, Any]:
    """Tune around the score/out-senseamp bounded-discharge bridge that scaled best."""
    center_enabled = trial.suggest_categorical("readout_center_pull_enabled", [False, True])
    return {
        "readout_flow_write_mode": "bounded_discharge",
        "readout_update_width_u": trial.suggest_float("readout_update_width_u", 1.5e-4, 8.0e-4, log=True),
        "output_bias_update_width_u": trial.suggest_categorical("output_bias_update_width_u", [0.0, 1.0e-4, 3.5e-4]),
        "readout_write_low_v": trial.suggest_float("readout_write_low_v", 0.10, 0.24),
        "readout_write_high_v": trial.suggest_float("readout_write_high_v", 0.50, 0.70),
        "readout_center_pull_width_u": trial.suggest_float(
            "readout_center_pull_width_u", 2e-6, 2e-3, log=True
        )
        if center_enabled
        else 0.0,
        "readout_center_pull_gate": trial.suggest_categorical("readout_center_pull_gate", ["bwd", "apply"]),
        "readout_center_pull_mode": trial.suggest_categorical("readout_center_pull_mode", ["always", "state_high"]),
        "lead_width_u": trial.suggest_float("lead_width_u", 48.0, 192.0, log=True),
        "backward_gate_width_u": trial.suggest_float("backward_gate_width_u", 32.0, 160.0, log=True),
        "backward_gate_cap_f": trial.suggest_float("backward_gate_cap_f", 1.0, 8.0, log=True),
        "score_cap_f": trial.suggest_float("score_cap_f", 5.0, 40.0, log=True),
        "score_reset_v": trial.suggest_float("score_reset_v", 0.0, 0.20),
        "output_forward_width_scale": trial.suggest_float("output_forward_width_scale", 0.5, 2.5, log=True),
        "output_relu_width_scale": trial.suggest_float("output_relu_width_scale", 0.5, 4.0, log=True),
    }


def sample_multiclass_split_score(trial: Trial) -> dict[str, Any]:
    """Tune a small 3-class split-score/current-competition readout."""
    p = sample_pmos_charge_discharge(trial)
    p.pop("output_bias_offset_v", None)
    p.update(
        {
            "target_high_v": trial.suggest_float("target_high_v", 0.85, 1.2),
            "error_target_source_v": trial.suggest_float("error_target_source_v", 0.45, 1.2),
            "error_nontarget_source_v": trial.suggest_float("error_nontarget_source_v", 0.08, 0.70),
            "score_diode_width_u": trial.suggest_float("score_diode_width_u", 64.0, 2048.0, log=True),
            "score_mirror_cap_f": trial.suggest_float("score_mirror_cap_f", 5.0, 120.0, log=True),
        }
    )
    return p


def sample_multiclass_capstate_bounded_cd(trial: Trial) -> dict[str, Any]:
    """Tune near the reproduced MNIST3 cap-state split-score anchor.

    The historical 90% 3-class run lives in a much lower output-drive regime
    than the generic pmos charge/discharge sampler. Keep this profile separate
    so it remains clear that the run starts from a programmed cap-state
    readout, rather than learning a random readout from scratch.
    """
    return {
        "readout_flow_write_mode": "bounded_charge_discharge",
        "readout_update_width_u": trial.suggest_float("readout_update_width_u", 5e-7, 2e-5, log=True),
        "readout_charge_update_width_u": trial.suggest_float(
            "readout_charge_update_width_u", 5e-5, 8e-4, log=True
        ),
        "readout_discharge_update_width_u": trial.suggest_float(
            "readout_discharge_update_width_u", 5e-7, 2e-5, log=True
        ),
        "readout_write_low_v": trial.suggest_float("readout_write_low_v", 0.08, 0.24),
        "readout_write_high_v": trial.suggest_float("readout_write_high_v", 0.75, 1.10),
        "readout_center_pull_width_u": trial.suggest_categorical(
            "readout_center_pull_width_u", [0.0, 2e-5, 1e-4, 5e-4]
        ),
        "readout_center_pull_gate": trial.suggest_categorical("readout_center_pull_gate", ["bwd", "apply"]),
        "readout_center_pull_mode": trial.suggest_categorical("readout_center_pull_mode", ["always", "state_high"]),
        "output_bias_update_width_u": trial.suggest_categorical("output_bias_update_width_u", [0.0, 1e-5, 1e-4]),
        "target_high_v": trial.suggest_float("target_high_v", 0.90, 1.20),
        "residual_target_width_u": trial.suggest_float("residual_target_width_u", 0.01, 0.60, log=True),
        "residual_output_width_u": trial.suggest_float("residual_output_width_u", 16.0, 128.0, log=True),
        "score_cap_f": trial.suggest_float("score_cap_f", 0.2, 3.0, log=True),
        "score_diode_width_u": trial.suggest_float("score_diode_width_u", 256.0, 2048.0, log=True),
        "score_mirror_cap_f": trial.suggest_float("score_mirror_cap_f", 30.0, 200.0, log=True),
        "output_cap_f": trial.suggest_float("output_cap_f", 50.0, 200.0, log=True),
        "output_forward_width_scale": trial.suggest_float("output_forward_width_scale", 0.01, 0.08, log=True),
    }


def sample_multiclass_random_pmos_charge_only(trial: Trial) -> dict[str, Any]:
    """Tune the best current from-random 3-class readout writer.

    This family is intentionally separate from the cap-state replay profile:
    the readout starts from random capacitor states, uses a simple split-score
    head, and relies on PMOS charge-only writes with a diffpair bleed guard to
    keep common-mode movement from swamping signed learning.
    """
    return {
        "readout_flow_write_mode": "bounded_pmos_charge_only",
        "readout_write_error_exclusion": "diffpair_bleed",
        "readout_write_error_exclusion_width_u": trial.suggest_float(
            "readout_write_error_exclusion_width_u", 2.0, 24.0, log=True
        ),
        "readout_update_width_u": trial.suggest_float("readout_update_width_u", 1e-4, 2e-3, log=True),
        "readout_write_low_v": trial.suggest_float("readout_write_low_v", 0.08, 0.28),
        "readout_write_high_v": trial.suggest_float("readout_write_high_v", 0.72, 1.10),
        "readout_flow_polarity": trial.suggest_categorical("readout_flow_polarity", ["normal", "reversed"]),
        "readout_center_pull_width_u": trial.suggest_categorical(
            "readout_center_pull_width_u", [0.0, 2e-5, 2e-4, 1e-3]
        ),
        "readout_center_pull_gate": trial.suggest_categorical("readout_center_pull_gate", ["bwd", "apply"]),
        "readout_center_pull_mode": trial.suggest_categorical("readout_center_pull_mode", ["always", "state_high"]),
        "output_bias_update_width_u": trial.suggest_categorical("output_bias_update_width_u", [0.0, 1e-5, 1e-4]),
        "target_high_v": trial.suggest_float("target_high_v", 0.90, 1.20),
        "residual_target_width_u": trial.suggest_float("residual_target_width_u", 48.0, 144.0, log=True),
        "residual_output_width_u": trial.suggest_float("residual_output_width_u", 24.0, 128.0, log=True),
        "score_cap_f": trial.suggest_float("score_cap_f", 4.0, 40.0, log=True),
        "output_cap_f": trial.suggest_float("output_cap_f", 8.0, 80.0, log=True),
    }


def derive_multiclass_random_pmos_charge_only_params(
    *,
    class_count: int = 3,
    effective_readout_fan_in: float = 8.0,
    readout_flow_write_mode: str = "bounded_pmos_charge_only",
    target_selector_ratio: float = 1.0,
    nontarget_selector_ratio: float = 1.0,
    output_bias_update_ratio: float | None = None,
    learning_rate_scale: float = 1.0,
    error_drive_scale: float = 1.0,
    score_tau_scale: float = 1.0,
) -> dict[str, Any]:
    """Derive SPICE knobs for the current random multiclass readout family.

    The goal is to make architecture comparisons depend on a small set of
    dimensionless global knobs, not on re-tuning every transistor width after
    each topology change.

    The anchor values come from the best current from-random MNIST3 recipe:
    PMOS charge-only writes, diffpair bleed, split-score caps, and spiking
    pretrace eligibility gates.  The derived rules keep local circuit ratios
    fixed:

    * ``learning_rate_scale`` changes only the effective weight-write mobility.
    * ``error_drive_scale`` changes the target and non-target error currents
      together, preserving their one-vs-rest exposure balance.
    * ``score_tau_scale`` changes score/output capacitor time constants
      together, preserving the 1:2 score/output capacitance ratio.

    The non-target error width is derived from the class count so that, over a
    balanced epoch, one target exposure has the same nominal drive as the
    ``class_count - 1`` non-target exposures for that row.

    The readout fan-in rescales two topology-sensitive quantities.  Score caps
    grow with fan-in so a wider row has roughly the same voltage swing, and the
    write width shrinks with fan-in so the row-level effective learning step
    remains near the anchor stability range.
    """
    if learning_rate_scale <= 0 or error_drive_scale <= 0 or score_tau_scale <= 0:
        raise ValueError("derived global scales must be positive")
    if class_count < 2:
        raise ValueError("class_count must be at least two")
    if effective_readout_fan_in <= 0:
        raise ValueError("effective_readout_fan_in must be positive")
    if readout_flow_write_mode not in {"bounded_pmos_charge_only", "bounded_pmos_charge_discharge"}:
        raise ValueError("derived PMOS write mode must be charge_only or charge_discharge")

    if output_bias_update_ratio is None:
        # The 3-class anchor benefits from bias-capacitor writes, but the first
        # 5-class probes degrade with the same mobility.  Keep the bias circuit
        # available while avoiding a class-count regression until the larger
        # output-bank bias dynamics are characterized.
        output_bias_update_ratio = 1.0 if class_count <= 3 else 0.0

    sizing = theory.derive_multiclass_readout_sizing(
        class_count=class_count,
        effective_readout_fan_in=effective_readout_fan_in,
        learning_rate_scale=learning_rate_scale,
        error_drive_scale=error_drive_scale,
        score_tau_scale=score_tau_scale,
        target_selector_ratio=target_selector_ratio,
        nontarget_selector_ratio=nontarget_selector_ratio,
        output_bias_update_ratio=output_bias_update_ratio,
    )

    return {
        "readout_flow_write_mode": readout_flow_write_mode,
        "readout_write_error_exclusion": "diffpair_bleed",
        "readout_write_error_exclusion_width_u": sizing.readout_write_error_exclusion_width_u,
        "readout_update_width_u": sizing.readout_update_width_u,
        "readout_dp_gate_update_width_u": sizing.readout_dp_gate_update_width_u,
        "readout_dn_gate_update_width_u": sizing.readout_dn_gate_update_width_u,
        "readout_write_low_v": 0.16,
        "readout_write_high_v": 1.0,
        "readout_flow_polarity": "normal",
        "readout_center_pull_width_u": 0.0,
        "readout_center_pull_gate": "bwd",
        "readout_center_pull_mode": "always",
        "output_bias_update_width_u": sizing.output_bias_update_width_u,
        "target_high_v": 1.1,
        "residual_target_width_u": sizing.residual_target_width_u,
        "residual_output_width_u": sizing.residual_output_width_u,
        "score_cap_f": sizing.score_cap_f,
        "output_cap_f": sizing.output_cap_f,
    }


def sample_multiclass_random_pmos_charge_only_derived(trial: Trial) -> dict[str, Any]:
    """Tune only global scales for the current random multiclass readout."""
    return derive_multiclass_random_pmos_charge_only_params(
        learning_rate_scale=trial.suggest_float("learning_rate_scale", 0.35, 2.5, log=True),
        error_drive_scale=trial.suggest_float("error_drive_scale", 0.50, 1.80, log=True),
        score_tau_scale=trial.suggest_float("score_tau_scale", 0.50, 2.50, log=True),
    )


def sample_multiclass_random_pmos_charge_discharge_derived(trial: Trial) -> dict[str, Any]:
    """Tune only global scales for the PMOS charge/discharge multiclass writer."""
    return derive_multiclass_random_pmos_charge_only_params(
        readout_flow_write_mode="bounded_pmos_charge_discharge",
        target_selector_ratio=4.0,
        nontarget_selector_ratio=1.0,
        learning_rate_scale=trial.suggest_float("learning_rate_scale", 0.35, 2.5, log=True),
        error_drive_scale=trial.suggest_float("error_drive_scale", 0.50, 1.80, log=True),
        score_tau_scale=trial.suggest_float("score_tau_scale", 0.50, 2.50, log=True),
    )


def sample_profile_params(
    profile: TuningProfile,
    trial: Trial,
    *,
    extra_driver_args: Sequence[str] = (),
) -> dict[str, Any]:
    """Sample params for a profile, including profile/driver context when needed."""
    if profile.sampler in {
        "multiclass_random_pmos_charge_only_derived",
        "multiclass_random_pmos_charge_discharge_derived",
    }:
        architecture = architecture_context_for_profile(profile, extra_driver_args)
        write_mode = (
            "bounded_pmos_charge_discharge"
            if profile.sampler == "multiclass_random_pmos_charge_discharge_derived"
            else "bounded_pmos_charge_only"
        )
        target_selector_ratio = 4.0 if write_mode == "bounded_pmos_charge_discharge" else 1.0
        return derive_multiclass_random_pmos_charge_only_params(
            class_count=architecture.class_count,
            effective_readout_fan_in=architecture.effective_readout_fan_in,
            readout_flow_write_mode=write_mode,
            target_selector_ratio=target_selector_ratio,
            nontarget_selector_ratio=1.0,
            learning_rate_scale=trial.suggest_float("learning_rate_scale", 0.35, 2.5, log=True),
            error_drive_scale=trial.suggest_float("error_drive_scale", 0.50, 1.80, log=True),
            score_tau_scale=trial.suggest_float("score_tau_scale", 0.50, 2.50, log=True),
        )
    return SAMPLERS[profile.sampler](trial)


SAMPLERS: dict[str, Callable[[Trial], dict[str, Any]]] = {
    "pmos_charge_discharge": sample_pmos_charge_discharge,
    "bounded_discharge": sample_bounded_discharge,
    "target_mistake_bounded_discharge": sample_target_mistake_bounded_discharge,
    "multiclass_split_score": sample_multiclass_split_score,
    "multiclass_capstate_bounded_cd": sample_multiclass_capstate_bounded_cd,
    "multiclass_random_pmos_charge_only": sample_multiclass_random_pmos_charge_only,
    "multiclass_random_pmos_charge_only_derived": sample_multiclass_random_pmos_charge_only_derived,
    "multiclass_random_pmos_charge_discharge_derived": sample_multiclass_random_pmos_charge_discharge_derived,
}


PROFILES: dict[str, TuningProfile] = {
    "xor2_pmos_cd": TuningProfile(
        name="xor2_pmos_cd",
        description="Tiny XOR screen for the balanced PMOS charge/discharge readout writer.",
        sampler="pmos_charge_discharge",
        base_args=(
            "--dataset",
            "xor2",
            "--epochs",
            "2",
            "--learning-mode",
            "flow",
            "--flow-hidden-write",
            "off",
            "--flow-pre-store",
            "shared_node",
            "--error-rule",
            "perceptron",
            "--decision-source",
            "score",
            "--measure-detail",
            "probe",
            "--cycle-ns",
            "12",
            "--skip-train-refire",
            "--tran-step-ps",
            "10",
            "--spice-accuracy-preset",
            "fast",
        ),
    ),
    "mnist01fixed8_readout": TuningProfile(
        name="mnist01fixed8_readout",
        description="MNIST 0-vs-1 fixed8 readout-only screen around the historical bounded-discharge recipe.",
        sampler="bounded_discharge",
        base_args=(
            "--dataset",
            "mnist01fixed8_64",
            "--epochs",
            "2",
            "--order",
            "interleave",
            "--learning-mode",
            "flow",
            "--flow-hidden-write",
            "off",
            "--flow-pre-store",
            "shared_node",
            "--hidden-forward-mode",
            "rail_buffer",
            "--error-rule",
            "out_mistake",
            "--lead-mode",
            "out_senseamp",
            "--decision-source",
            "score",
            "--measure-detail",
            "probe",
            "--cycle-ns",
            "12",
            "--skip-train-refire",
            "--tran-step-ps",
            "10",
            "--spice-accuracy-preset",
            "fast",
        ),
    ),
    "mnist01fixed8_tm_competitive": TuningProfile(
        name="mnist01fixed8_tm_competitive",
        description=(
            "Current score/out-senseamp target-mistake MNIST 0-vs-1 bridge. "
            "This is the high-accuracy bounded-discharge family, not the PMOS charge/discharge smoke."
        ),
        sampler="target_mistake_bounded_discharge",
        base_args=(
            "--dataset",
            "mnist01fixed8_64",
            "--epochs",
            "2",
            "--learning-mode",
            "flow",
            "--flow-hidden-write",
            "off",
            "--flow-pre-store",
            "shared_node",
            "--hidden-forward-mode",
            "rail_buffer",
            "--hidden-init",
            "input_identity",
            "--error-rule",
            "score",
            "--backward-gate-mode",
            "target_mistake",
            "--lead-mode",
            "out_senseamp",
            "--decision-source",
            "out",
            "--cycle-ns",
            "12",
            "--skip-train-refire",
            "--measure-detail",
            "light",
            "--tran-step-ps",
            "10",
            "--spice-accuracy-preset",
            "fast",
        ),
        anchor_params=(
            ("readout_flow_write_mode", "bounded_discharge"),
            ("readout_update_width_u", 3.5e-4),
            ("output_bias_update_width_u", 0.0),
            ("readout_write_low_v", 0.16),
            ("readout_write_high_v", 0.58),
            ("readout_center_pull_width_u", 0.0),
            ("readout_center_pull_gate", "bwd"),
            ("readout_center_pull_mode", "always"),
            ("lead_width_u", 96.0),
            ("backward_gate_width_u", 64.0),
            ("backward_gate_cap_f", 2.0),
            ("score_cap_f", 20.0),
            ("score_reset_v", 0.0),
            ("output_forward_width_scale", 1.0),
            ("output_relu_width_scale", 1.0),
        ),
    ),
    "mnist3fixed8_pmos_cd": TuningProfile(
        name="mnist3fixed8_pmos_cd",
        description="Small 3-class fixed8 screen for split-score multiclass write calibration.",
        sampler="multiclass_split_score",
        base_args=(
            "--dataset",
            "mnist3fixed8_12",
            "--epochs",
            "1",
            "--learning-mode",
            "flow",
            "--flow-hidden-write",
            "off",
            "--flow-pre-store",
            "shared_node",
            "--hidden-forward-mode",
            "rail_buffer",
            "--output-head",
            "split_score_diode_mirror_caps",
            "--error-rule",
            "ce_mirror_compete_limited",
            "--lead-mode",
            "score_direct",
            "--decision-source",
            "score",
            "--measure-detail",
            "probe",
            "--cycle-ns",
            "12",
            "--skip-train-refire",
            "--tran-step-ps",
            "10",
            "--spice-accuracy-preset",
            "fast",
        ),
    ),
    "mnist3fixed8_capstate_replay": TuningProfile(
        name="mnist3fixed8_capstate_replay",
        description=(
            "Reproduced 3-class MNIST fixed8 cap-state anchor: programmed readout caps, "
            "low-drive split-score mirror output, and bounded charge/discharge writes."
        ),
        sampler="multiclass_capstate_bounded_cd",
        base_args=(
            "--dataset",
            "mnist3fixed8_30",
            "--epochs",
            "1",
            "--learning-mode",
            "flow",
            "--flow-hidden-write",
            "off",
            "--flow-pre-store",
            "shared_node",
            "--hidden-forward-mode",
            "rail_buffer",
            "--readout-init",
            "csv_cap_state",
            "--separator-csv",
            "spice/results/device_readout_capfit_sumtransfer_w1024_c100_scale002_caps.csv",
            "--output-head",
            "split_score_diode_mirror_caps",
            "--error-rule",
            "ce_split_limited",
            "--lead-mode",
            "score_direct",
            "--backward-gate-mode",
            "scheduled",
            "--decision-source",
            "out",
            "--measure-detail",
            "light",
            "--cycle-ns",
            "16",
            "--readout-sample-offsets-ns",
            "3.15",
            "--tran-step-ps",
            "10",
            "--spice-accuracy-preset",
            "standard",
        ),
        anchor_params=(
            ("readout_flow_write_mode", "bounded_charge_discharge"),
            ("readout_update_width_u", 2e-6),
            ("readout_charge_update_width_u", 2e-4),
            ("readout_discharge_update_width_u", 2e-6),
            ("readout_write_low_v", 0.16),
            ("readout_write_high_v", 1.0),
            ("readout_center_pull_width_u", 0.0),
            ("readout_center_pull_gate", "bwd"),
            ("readout_center_pull_mode", "always"),
            ("output_bias_update_width_u", 0.0),
            ("target_high_v", 1.1),
            ("residual_target_width_u", 0.06),
            ("residual_output_width_u", 64.0),
            ("score_cap_f", 0.4),
            ("score_diode_width_u", 1024.0),
            ("score_mirror_cap_f", 100.0),
            ("output_cap_f", 100.0),
            ("output_forward_width_scale", 0.02),
        ),
    ),
    "mnist3fixed8_random_pmos_chargeonly": TuningProfile(
        name="mnist3fixed8_random_pmos_chargeonly",
        description=(
            "From-random 3-class MNIST fixed8 readout-learning screen: split-score caps, "
            "onehot-limited error rails, PMOS charge-only writes, and diffpair bleed."
        ),
        sampler="multiclass_random_pmos_charge_only",
        base_args=(
            "--dataset",
            "mnist3fixed8_12",
            "--epochs",
            "2",
            "--learning-mode",
            "flow",
            "--flow-hidden-write",
            "off",
            "--flow-pre-store",
            "synapse_spike",
            "--hidden-forward-mode",
            "rail_buffer",
            "--output-head",
            "split_score_caps",
            "--error-rule",
            "onehot_limited",
            "--lead-mode",
            "score_direct",
            "--backward-gate-mode",
            "scheduled",
            "--decision-source",
            "score",
            "--measure-detail",
            "light",
            "--cycle-ns",
            "16",
            "--tran-step-ps",
            "10",
            "--spice-accuracy-preset",
            "standard",
        ),
        anchor_params=(
            ("readout_flow_write_mode", "bounded_pmos_charge_only"),
            ("readout_write_error_exclusion", "diffpair_bleed"),
            ("readout_write_error_exclusion_width_u", 8.0),
            ("readout_update_width_u", 5e-4),
            ("readout_write_low_v", 0.16),
            ("readout_write_high_v", 1.0),
            ("readout_flow_polarity", "normal"),
            ("readout_center_pull_width_u", 0.0),
            ("readout_center_pull_gate", "bwd"),
            ("readout_center_pull_mode", "always"),
            ("output_bias_update_width_u", 0.0),
            ("target_high_v", 1.1),
            ("residual_target_width_u", 96.0),
            ("residual_output_width_u", 64.0),
            ("score_cap_f", 10.0),
            ("output_cap_f", 20.0),
        ),
    ),
    "mnist3fixed8_random_pmos_chargeonly_derived": TuningProfile(
        name="mnist3fixed8_random_pmos_chargeonly_derived",
        description=(
            "Theory-constrained from-random 3-class MNIST fixed8 profile. "
            "Only global learning-rate, error-drive, and score-time scales are tuned; "
            "local widths/caps are derived from fixed circuit ratios."
        ),
        sampler="multiclass_random_pmos_charge_only_derived",
        base_args=(
            "--dataset",
            "mnist3fixed8_12",
            "--epochs",
            "2",
            "--learning-mode",
            "flow",
            "--flow-hidden-write",
            "off",
            "--flow-pre-store",
            "synapse_spike",
            "--hidden-forward-mode",
            "rail_buffer",
            "--output-head",
            "split_score_caps",
            "--error-rule",
            "onehot_limited",
            "--lead-mode",
            "score_direct",
            "--backward-gate-mode",
            "scheduled",
            "--decision-source",
            "score",
            "--measure-detail",
            "light",
            "--cycle-ns",
            "16",
            "--tran-step-ps",
            "10",
            "--spice-accuracy-preset",
            "standard",
        ),
        anchor_params=(
            ("learning_rate_scale", 1.0),
            ("error_drive_scale", 1.0),
            ("score_tau_scale", 1.0),
        ),
    ),
    "mnist3fixed8_random_pmos_cd_derived": TuningProfile(
        name="mnist3fixed8_random_pmos_cd_derived",
        description=(
            "Theory-constrained from-random multiclass MNIST fixed8 profile using the "
            "PMOS charge/discharge writer.  It keeps the same derived global scales "
            "as the charge-only profile but adds the opposite-branch discharge leg to "
            "test common-mode control."
        ),
        sampler="multiclass_random_pmos_charge_discharge_derived",
        base_args=(
            "--dataset",
            "mnist3fixed8_12",
            "--epochs",
            "2",
            "--learning-mode",
            "flow",
            "--flow-hidden-write",
            "off",
            "--flow-pre-store",
            "synapse_spike",
            "--hidden-forward-mode",
            "rail_buffer",
            "--output-head",
            "split_score_caps",
            "--error-rule",
            "onehot_limited",
            "--lead-mode",
            "score_direct",
            "--backward-gate-mode",
            "scheduled",
            "--decision-source",
            "score",
            "--measure-detail",
            "light",
            "--cycle-ns",
            "16",
            "--tran-step-ps",
            "10",
            "--spice-accuracy-preset",
            "standard",
        ),
        anchor_params=(
            ("learning_rate_scale", 1.0),
            ("error_drive_scale", 1.0),
            ("score_tau_scale", 1.0),
        ),
    ),
}


def profile_names() -> str:
    return ", ".join(sorted(PROFILES))


DERIVED_METRICS = {
    "score_raw_centered_min": (
        "minimum of final_score_accuracy and final_score_column_centered_accuracy; "
        "useful when a centered separator is not enough unless the raw score head also works"
    ),
    "score_raw_centered_mean": "mean of final_score_accuracy and final_score_column_centered_accuracy",
    "score_raw_centered_product": "product of final_score_accuracy and final_score_column_centered_accuracy",
}


def metric_value(summary: dict[str, Any], metric: str) -> float:
    if metric == "auto":
        for key in (
            "best_eval_accuracy",
            "best_final_transient_accuracy",
            "final_score_accuracy",
            "final_eval_accuracy",
            "final_output_accuracy",
        ):
            value = summary.get(key)
            if value is not None:
                return float(value)
        raise KeyError("summary has no known accuracy metric")
    if metric in DERIVED_METRICS:
        raw = float(summary["final_score_accuracy"])
        centered = float(summary["final_score_column_centered_accuracy"])
        if metric == "score_raw_centered_min":
            return min(raw, centered)
        if metric == "score_raw_centered_mean":
            return 0.5 * (raw + centered)
        if metric == "score_raw_centered_product":
            return raw * centered
        raise AssertionError(f"unhandled derived metric {metric}")
    value = summary.get(metric)
    if value is None:
        raise KeyError(f"summary is missing metric {metric!r}")
    return float(value)


def objective_score(
    summary: dict[str, Any],
    *,
    metric: str,
    common_mode_penalty: float,
    hidden_write_penalty: float,
) -> float:
    score = metric_value(summary, metric)
    if common_mode_penalty:
        ratio = summary.get("readout_common_to_signed_delta_ratio")
        if ratio is not None:
            score -= common_mode_penalty * float(ratio)
    if hidden_write_penalty:
        hidden_motion = summary.get("max_abs_total_hidden_signed_delta_v")
        if hidden_motion is not None:
            score -= hidden_write_penalty * float(hidden_motion)
    return float(score)


def build_trial_command(
    *,
    profile: TuningProfile,
    tag: str,
    sampled_params: dict[str, Any],
    driver_timeout_s: float,
    simulator: str | None,
    extra_driver_args: Sequence[str],
) -> list[str]:
    command = [
        sys.executable,
        str(DRIVER),
        "--tag",
        tag,
        "--timeout",
        fmt(driver_timeout_s),
        *profile.base_args,
        *params_to_driver_args(sampled_params),
    ]
    if simulator is not None:
        command.extend(["--simulator", simulator])
    command.extend(extra_driver_args)
    return command


def load_trial_summary(tag: str) -> tuple[dict[str, Any], Path]:
    path = SPICE_RESULTS / f"{safe_tag(tag)}_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"driver did not produce summary JSON: {path}")
    return json.loads(path.read_text()), path


def append_trial_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = sorted(row)
    if exists:
        with path.open(newline="") as f:
            reader = csv.reader(f)
            old_header = next(reader, None)
        if old_header is not None:
            fieldnames = list(dict.fromkeys([*old_header, *fieldnames]))
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_one_trial(
    trial: Trial,
    *,
    args: argparse.Namespace,
    profile: TuningProfile,
    runner: Runner = subprocess.run,
) -> float:
    sampled_params = sample_profile_params(profile, trial, extra_driver_args=args.driver_args)
    tag = safe_tag(f"{args.tag_prefix}_{profile.name}_t{trial.number:04d}")
    command = build_trial_command(
        profile=profile,
        tag=tag,
        sampled_params=sampled_params,
        driver_timeout_s=args.driver_timeout,
        simulator=args.simulator,
        extra_driver_args=args.driver_args,
    )
    command_text = shlex.join(command)
    trial.set_user_attr("tag", tag)
    trial.set_user_attr("command", command_text)
    trial.set_user_attr("sampled_params", sampled_params)
    t0 = time.perf_counter()
    try:
        completed = runner(
            command,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=args.driver_timeout + args.subprocess_grace,
        )
        if completed.returncode != 0:
            trial.set_user_attr("failure", "driver_returncode")
            trial.set_user_attr("stderr_tail", (completed.stderr or completed.stdout)[-2000:])
            value = args.failure_value
            summary: dict[str, Any] = {}
            summary_path = None
        else:
            summary, summary_path = load_trial_summary(tag)
            value = objective_score(
                summary,
                metric=args.metric,
                common_mode_penalty=args.common_mode_penalty,
                hidden_write_penalty=args.hidden_write_penalty,
            )
            trial.set_user_attr("summary_path", str(summary_path))
            trial.set_user_attr("summary", summary)
    except Exception as exc:
        trial.set_user_attr("failure", type(exc).__name__)
        trial.set_user_attr("failure_message", str(exc)[-2000:])
        value = args.failure_value
        summary = {}
        summary_path = None

    row: dict[str, Any] = {
        "trial": trial.number,
        "tag": tag,
        "objective": value,
        "metric": args.metric,
        "status": "failed" if trial.user_attrs.get("failure") else "ok",
        "failure": trial.user_attrs.get("failure", ""),
        "failure_message": trial.user_attrs.get("failure_message", ""),
        "stderr_tail": trial.user_attrs.get("stderr_tail", ""),
        "wall_time_s": time.perf_counter() - t0,
        "summary_path": "" if summary_path is None else str(summary_path),
        "command": command_text,
    }
    for key in (
        "best_eval_accuracy",
        "best_final_transient_accuracy",
        "final_score_accuracy",
        "final_eval_accuracy",
        "final_output_accuracy",
        "final_score_column_centered_accuracy",
        "final_output_column_centered_accuracy",
        "readout_common_to_signed_delta_ratio",
        "max_abs_total_hidden_signed_delta_v",
    ):
        if key in summary:
            row[key] = summary[key]
    row.update({f"param_{k}": v for k, v in sampled_params.items()})
    append_trial_row(TABLES / f"{safe_tag(args.study_name)}_trials.csv", row)
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    tuner_argv, driver_args = split_tuner_and_driver_args(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        description=(
            "Tune SPICE-native neural-network circuit hyperparameters with Optuna. "
            f"Profiles: {profile_names()}."
        )
    )
    ap.add_argument("--profile", choices=sorted(PROFILES), default="xor2_pmos_cd")
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--study-name", default="device_hparam_tuning")
    ap.add_argument("--storage", default=None, help="Optional Optuna storage URL, e.g. sqlite:///results/tables/tune.db")
    ap.add_argument("--sampler-seed", type=int, default=0)
    ap.add_argument("--tag-prefix", default="tune")
    ap.add_argument(
        "--metric",
        default="auto",
        help=(
            "Summary metric to maximize. Built-ins: auto, "
            f"{', '.join(sorted(DERIVED_METRICS))}; any summary JSON key is also accepted."
        ),
    )
    ap.add_argument("--common-mode-penalty", type=float, default=0.0)
    ap.add_argument("--hidden-write-penalty", type=float, default=0.0)
    ap.add_argument("--failure-value", type=float, default=0.0)
    ap.add_argument("--driver-timeout", type=float, default=240.0)
    ap.add_argument("--subprocess-grace", type=float, default=30.0)
    ap.add_argument(
        "--simulator",
        default=None,
        help=(
            "Forwarded to the SPICE driver. Use auto for ngspice-first compatibility, "
            "auto-fast for Xyce/XyceNF-first sweeps, or pass an executable path."
        ),
    )
    ap.add_argument(
        "--enqueue-anchor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate the profile's documented anchor parameters before Bayesian exploration, when available.",
    )
    ap.add_argument("--print-profile", action="store_true")
    args = ap.parse_args(tuner_argv)
    args.driver_args = list(driver_args)
    if args.trials <= 0:
        raise SystemExit("--trials must be positive")
    if args.jobs <= 0:
        raise SystemExit("--jobs must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    profile = PROFILES[args.profile]
    if args.print_profile:
        print(
            json.dumps(
                {
                    "profile": profile.name,
                    "description": profile.description,
                    "base_args": profile.base_args,
                    "anchor_params": dict(profile.anchor_params),
                },
                indent=2,
            )
        )
        return
    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed, multivariate=True)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
    )
    if args.enqueue_anchor and profile.anchor_params:
        study.enqueue_trial(dict(profile.anchor_params), user_attrs={"source": "profile_anchor"}, skip_if_exists=True)
    study.optimize(lambda trial: run_one_trial(trial, args=args, profile=profile), n_trials=args.trials, n_jobs=args.jobs)
    best = {
        "study_name": args.study_name,
        "profile": profile.name,
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "best_params": study.best_trial.params,
        "best_command": study.best_trial.user_attrs.get("command"),
        "trials_csv": str(TABLES / f"{safe_tag(args.study_name)}_trials.csv"),
    }
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
