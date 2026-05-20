#!/usr/bin/env python3
"""Launch small pools of device-level SPICE experiments.

The direct-flow device runs are slow enough that we want reproducible parallel
pools rather than one-off shell commands.  This helper only orchestrates runs;
the actual circuit generation and measurement still lives in
``spice/run_device_xor2_random_hidden.py``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "spice/run_device_xor2_random_hidden.py"


def base_args(timeout_s: str = "1200") -> list[str]:
    return [
        sys.executable,
        "-B",
        str(RUNNER),
        "--timeout",
        timeout_s,
        "--measure-detail",
        "light",
        "--eval-each-epoch",
        "--readout-sample-offsets-ns",
        "2.55,2.75,2.95,3.15,3.35",
    ]


def v557_v560_pool() -> list[tuple[str, list[str]]]:
    base = base_args()
    full_hidden_common = [
        "--hidden-cells",
        "8",
        "--hidden-forward-mode",
        "rail_buffer",
        "--hidden-init",
        "input_identity",
        "--output-head",
        "source_follower",
        "--lead-mode",
        "out_senseamp",
        "--error-rule",
        "score",
        "--backward-gate-mode",
        "target_mistake",
        "--learning-mode",
        "flow",
        "--flow-hidden-write",
        "direct",
        "--flow-pre-store",
        "synapse_gate",
        "--readout-flow-write-mode",
        "bounded_discharge",
        "--readout-update-width-u",
        "0.00035",
        "--hidden-update-width-u",
        "0.0000002",
        "--hidden-flow-write-mode",
        "bounded_discharge",
        "--hidden-delta-internal-leak-ohm",
        "1000000000",
    ]
    onehot_common = [
        "--dataset",
        "mnistfixed8_20",
        "--epochs",
        "4",
        "--hidden-cells",
        "8",
        "--hidden-forward-mode",
        "rail_buffer",
        "--hidden-init",
        "input_identity",
        "--output-head",
        "source_follower",
        "--lead-mode",
        "score_direct",
        "--error-rule",
        "onehot",
        "--backward-gate-mode",
        "scheduled",
        "--learning-mode",
        "flow",
        "--flow-hidden-write",
        "off",
        "--readout-flow-write-mode",
        "bounded_charge_discharge",
        "--readout-update-width-u",
        "0.0002",
        "--residual-target-width-u",
        "96",
        "--output-bias-update-width-u",
        "0",
        "--readout-sample-offsets-ns",
        "2.95",
    ]
    return [
        (
            "device_mnist01fixed8_128_v557_fullhidden_tm_leaky_e1",
            base
            + [
                "--tag",
                "device_mnist01fixed8_128_v557_fullhidden_tm_leaky_e1",
                "--dataset",
                "mnist01fixed8_128",
                "--epochs",
                "1",
                *full_hidden_common,
                "--hidden-delta-internal-cap-f",
                "0.02",
            ],
        ),
        (
            "device_mnist01fixed8_64_v558_fullhidden_tm_cap005_e2",
            base
            + [
                "--tag",
                "device_mnist01fixed8_64_v558_fullhidden_tm_cap005_e2",
                "--dataset",
                "mnist01fixed8_64",
                "--epochs",
                "2",
                *full_hidden_common,
                "--hidden-delta-internal-cap-f",
                "0.05",
            ],
        ),
        (
            "device_mnistfixed8_20_v559_onehot_cd_nt4_e4",
            base
            + [
                "--tag",
                "device_mnistfixed8_20_v559_onehot_cd_nt4_e4",
                *onehot_common,
                "--residual-output-width-u",
                "4",
            ],
        ),
        (
            "device_mnistfixed8_20_v560_onehot_cd_chargeheavy_e4",
            base
            + [
                "--tag",
                "device_mnistfixed8_20_v560_onehot_cd_chargeheavy_e4",
                *onehot_common,
                "--residual-output-width-u",
                "12",
                "--readout-charge-update-width-u",
                "0.0006",
                "--readout-discharge-update-width-u",
                "0.00005",
            ],
        ),
    ]


def v561_v562_pool() -> list[tuple[str, list[str]]]:
    base = base_args()
    onehot_out_common = [
        "--dataset",
        "mnistfixed8_20",
        "--epochs",
        "4",
        "--hidden-cells",
        "8",
        "--hidden-forward-mode",
        "rail_buffer",
        "--hidden-init",
        "input_identity",
        "--output-head",
        "source_follower",
        "--lead-mode",
        "score_direct",
        "--error-rule",
        "onehot_out",
        "--backward-gate-mode",
        "scheduled",
        "--learning-mode",
        "flow",
        "--flow-hidden-write",
        "off",
        "--residual-target-width-u",
        "96",
        "--residual-output-width-u",
        "64",
        "--output-bias-update-width-u",
        "0",
        "--readout-sample-offsets-ns",
        "2.95",
    ]
    return [
        (
            "device_mnistfixed8_20_v561_onehot_out_cd_e4",
            base
            + [
                "--tag",
                "device_mnistfixed8_20_v561_onehot_out_cd_e4",
                *onehot_out_common,
                "--readout-flow-write-mode",
                "bounded_charge_discharge",
                "--readout-update-width-u",
                "0.0002",
            ],
        ),
        (
            "device_mnistfixed8_20_v562_onehot_out_discharge_e4",
            base
            + [
                "--tag",
                "device_mnistfixed8_20_v562_onehot_out_discharge_e4",
                *onehot_out_common,
                "--readout-flow-write-mode",
                "bounded_discharge",
                "--readout-update-width-u",
                "0.00035",
            ],
        ),
    ]


def v563_v565_pool() -> list[tuple[str, list[str]]]:
    base = base_args()
    onehot_out_common = [
        "--dataset",
        "mnistfixed8_20",
        "--epochs",
        "4",
        "--hidden-cells",
        "8",
        "--hidden-forward-mode",
        "rail_buffer",
        "--hidden-init",
        "input_identity",
        "--output-head",
        "source_follower",
        "--lead-mode",
        "score_direct",
        "--error-rule",
        "onehot_out",
        "--backward-gate-mode",
        "scheduled",
        "--learning-mode",
        "flow",
        "--flow-hidden-write",
        "off",
        "--readout-flow-write-mode",
        "bounded_charge_discharge",
        "--readout-update-width-u",
        "0.0002",
        "--output-bias-update-width-u",
        "0",
        "--readout-sample-offsets-ns",
        "2.95",
    ]
    binary_synapse_gate_readout = [
        "--tag",
        "device_mnist01fixed8_128_v565_readout_synapsegate_e2",
        "--dataset",
        "mnist01fixed8_128",
        "--epochs",
        "2",
        "--hidden-cells",
        "8",
        "--hidden-forward-mode",
        "rail_buffer",
        "--hidden-init",
        "input_identity",
        "--output-head",
        "source_follower",
        "--lead-mode",
        "out_senseamp",
        "--error-rule",
        "score",
        "--backward-gate-mode",
        "target_mistake",
        "--learning-mode",
        "flow",
        "--flow-hidden-write",
        "off",
        "--flow-pre-store",
        "synapse_gate",
        "--readout-flow-write-mode",
        "bounded_discharge",
        "--readout-update-width-u",
        "0.00035",
    ]
    return [
        (
            "device_mnistfixed8_20_v563_onehot_out_cd_nt192_e4",
            base
            + [
                "--tag",
                "device_mnistfixed8_20_v563_onehot_out_cd_nt192_e4",
                *onehot_out_common,
                "--residual-target-width-u",
                "96",
                "--residual-output-width-u",
                "192",
            ],
        ),
        (
            "device_mnistfixed8_20_v564_onehot_out_cd_t32_nt64_e4",
            base
            + [
                "--tag",
                "device_mnistfixed8_20_v564_onehot_out_cd_t32_nt64_e4",
                *onehot_out_common,
                "--residual-target-width-u",
                "32",
                "--residual-output-width-u",
                "64",
            ],
        ),
        (
            "device_mnist01fixed8_128_v565_readout_synapsegate_e2",
            base + binary_synapse_gate_readout,
        ),
    ]


def v566_v568_pool() -> list[tuple[str, list[str]]]:
    base = base_args(timeout_s="2400")
    full_hidden_128_common = [
        "--dataset",
        "mnist01fixed8_128",
        "--epochs",
        "2",
        "--hidden-cells",
        "8",
        "--hidden-forward-mode",
        "rail_buffer",
        "--hidden-init",
        "input_identity",
        "--output-head",
        "source_follower",
        "--lead-mode",
        "out_senseamp",
        "--error-rule",
        "score",
        "--backward-gate-mode",
        "target_mistake",
        "--learning-mode",
        "flow",
        "--flow-hidden-write",
        "direct",
        "--flow-pre-store",
        "synapse_gate",
        "--readout-flow-write-mode",
        "bounded_discharge",
        "--readout-update-width-u",
        "0.00035",
        "--hidden-flow-write-mode",
        "bounded_discharge",
        "--hidden-delta-internal-cap-f",
        "0.02",
        "--hidden-delta-internal-leak-ohm",
        "1000000000",
    ]
    return [
        (
            "device_mnist01fixed8_128_v566_fullhidden_tm_leaky_e2",
            base
            + [
                "--tag",
                "device_mnist01fixed8_128_v566_fullhidden_tm_leaky_e2",
                *full_hidden_128_common,
                "--hidden-update-width-u",
                "0.0000002",
            ],
        ),
        (
            "device_mnist01fixed8_128_v567_fullhidden_tm_leaky_reset_e2",
            base
            + [
                "--tag",
                "device_mnist01fixed8_128_v567_fullhidden_tm_leaky_reset_e2",
                *full_hidden_128_common,
                "--hidden-update-width-u",
                "0.0000002",
                "--hidden-delta-internal-reset-width-u",
                "4",
            ],
        ),
        (
            "device_mnist01fixed8_128_v568_fullhidden_tm_reset_weakwrite_e2",
            base
            + [
                "--tag",
                "device_mnist01fixed8_128_v568_fullhidden_tm_reset_weakwrite_e2",
                *full_hidden_128_common,
                "--hidden-update-width-u",
                "0.00000005",
                "--hidden-delta-internal-reset-width-u",
                "4",
            ],
        ),
    ]


def binary_readout_common(dataset: str = "mnist01fixed8_64", epochs: str = "2") -> list[str]:
    return [
        "--dataset",
        dataset,
        "--epochs",
        epochs,
        "--hidden-cells",
        "8",
        "--hidden-forward-mode",
        "rail_buffer",
        "--hidden-init",
        "input_identity",
        "--output-head",
        "source_follower",
        "--lead-mode",
        "out_senseamp",
        "--error-rule",
        "score",
        "--backward-gate-mode",
        "target_mistake",
        "--learning-mode",
        "flow",
        "--flow-hidden-write",
        "off",
        "--readout-flow-write-mode",
        "bounded_discharge",
        "--readout-update-width-u",
        "0.00035",
    ]


def v569_v574_controls_pool() -> list[tuple[str, list[str]]]:
    """Controls for the current binary direct-flow readout anchor.

    Expected interpretation:
    - v569 is the nominal readout-only anchor.
    - v570 reversed-polarity should degrade.
    - v571 label-shuffle should degrade.
    - v572 interleave and v573 seed variation should remain in-family.
    - v574 isolates per-synapse trace loading with hidden writes still off.
    """
    base = base_args(timeout_s="1200")
    common = binary_readout_common()
    return [
        (
            "device_mnist01fixed8_64_v569_readout_anchor_control_e2",
            base
            + [
                "--tag",
                "device_mnist01fixed8_64_v569_readout_anchor_control_e2",
                *common,
                "--flow-pre-store",
                "shared_node",
            ],
        ),
        (
            "device_mnist01fixed8_64_v570_readout_reversed_polarity_e2",
            base
            + [
                "--tag",
                "device_mnist01fixed8_64_v570_readout_reversed_polarity_e2",
                *common,
                "--flow-pre-store",
                "shared_node",
                "--readout-flow-polarity",
                "reversed",
            ],
        ),
        (
            "device_mnist01fixed8_64_v571_readout_labelshuffle_e2",
            base
            + [
                "--tag",
                "device_mnist01fixed8_64_v571_readout_labelshuffle_e2",
                *common,
                "--flow-pre-store",
                "shared_node",
                "--label-shuffle-seed",
                "101",
            ],
        ),
        (
            "device_mnist01fixed8_64_v572_readout_interleave_e2",
            base
            + [
                "--tag",
                "device_mnist01fixed8_64_v572_readout_interleave_e2",
                *common,
                "--flow-pre-store",
                "shared_node",
                "--order",
                "interleave",
            ],
        ),
        (
            "device_mnist01fixed8_64_v573_readout_seed4_e2",
            base
            + [
                "--tag",
                "device_mnist01fixed8_64_v573_readout_seed4_e2",
                *common,
                "--flow-pre-store",
                "shared_node",
                "--seed",
                "4",
            ],
        ),
        (
            "device_mnist01fixed8_64_v574_readout_synapsegate_hiddenoff_e2",
            base
            + [
                "--tag",
                "device_mnist01fixed8_64_v574_readout_synapsegate_hiddenoff_e2",
                *common,
                "--flow-pre-store",
                "synapse_gate",
            ],
        ),
    ]


def v575_v577_exclusive_pool() -> list[tuple[str, list[str]]]:
    """Same fixed8 binary architecture with ambiguous error-rail writes inhibited."""
    base = base_args(timeout_s="1200")
    exclusive = [
        "--write-error-exclusion",
        "pmos_inhibit",
        "--write-error-exclusion-width-u",
        "8",
    ]
    readout_common = binary_readout_common()
    full_hidden_common = [
        "--dataset",
        "mnist01fixed8_64",
        "--epochs",
        "2",
        "--hidden-cells",
        "8",
        "--hidden-forward-mode",
        "rail_buffer",
        "--hidden-init",
        "input_identity",
        "--output-head",
        "source_follower",
        "--lead-mode",
        "out_senseamp",
        "--error-rule",
        "score",
        "--backward-gate-mode",
        "target_mistake",
        "--learning-mode",
        "flow",
        "--flow-hidden-write",
        "direct",
        "--flow-pre-store",
        "synapse_gate",
        "--readout-flow-write-mode",
        "bounded_discharge",
        "--readout-update-width-u",
        "0.00035",
        "--hidden-flow-write-mode",
        "bounded_discharge",
        "--hidden-update-width-u",
        "0.00000005",
        "--hidden-delta-internal-cap-f",
        "0.02",
        "--hidden-delta-internal-leak-ohm",
        "1000000000",
        "--hidden-delta-internal-reset-width-u",
        "4",
    ]
    return [
        (
            "device_mnist01fixed8_64_v575_readout_shared_exclusive_e2",
            base
            + [
                "--tag",
                "device_mnist01fixed8_64_v575_readout_shared_exclusive_e2",
                *readout_common,
                "--flow-pre-store",
                "shared_node",
                *exclusive,
            ],
        ),
        (
            "device_mnist01fixed8_64_v576_readout_synapsegate_exclusive_e2",
            base
            + [
                "--tag",
                "device_mnist01fixed8_64_v576_readout_synapsegate_exclusive_e2",
                *readout_common,
                "--flow-pre-store",
                "synapse_gate",
                *exclusive,
            ],
        ),
        (
            "device_mnist01fixed8_64_v577_fullhidden_exclusive_e2",
            base
            + [
                "--tag",
                "device_mnist01fixed8_64_v577_fullhidden_exclusive_e2",
                *full_hidden_common,
                *exclusive,
            ],
        ),
    ]


def v578_v581_write_split_pool() -> list[tuple[str, list[str]]]:
    """Fair fixed8 write-rail split checks.

    These keep the input/preprocessing fixed and vary only the write-rail
    architecture around the previous 64-sample readout/full-hidden anchors.
    """
    base = base_args(timeout_s="1200")
    readout_common = binary_readout_common()
    fair_full_hidden_common = [
        "--dataset",
        "mnist01fixed8_64",
        "--epochs",
        "2",
        "--hidden-cells",
        "8",
        "--hidden-forward-mode",
        "rail_buffer",
        "--hidden-init",
        "input_identity",
        "--output-head",
        "source_follower",
        "--lead-mode",
        "out_senseamp",
        "--error-rule",
        "score",
        "--backward-gate-mode",
        "target_mistake",
        "--learning-mode",
        "flow",
        "--flow-hidden-write",
        "direct",
        "--flow-pre-store",
        "synapse_gate",
        "--readout-flow-write-mode",
        "bounded_discharge",
        "--readout-update-width-u",
        "0.00035",
        "--hidden-flow-write-mode",
        "bounded_discharge",
        "--hidden-update-width-u",
        "0.0000002",
        "--hidden-delta-internal-cap-f",
        "0.05",
        "--hidden-delta-internal-leak-ohm",
        "1000000000",
        "--hidden-delta-internal-reset-width-u",
        "4",
    ]
    return [
        (
            "device_mnist01fixed8_64_v578_fullhidden_exclusive_fair_e2",
            base
            + [
                "--tag",
                "device_mnist01fixed8_64_v578_fullhidden_exclusive_fair_e2",
                *fair_full_hidden_common,
                "--write-error-exclusion",
                "pmos_inhibit",
                "--write-error-exclusion-width-u",
                "8",
            ],
        ),
        (
            "device_mnist01fixed8_64_v579_fullhidden_hiddenexclusive_e2",
            base
            + [
                "--tag",
                "device_mnist01fixed8_64_v579_fullhidden_hiddenexclusive_e2",
                *fair_full_hidden_common,
                "--readout-write-error-exclusion",
                "none",
                "--hidden-write-error-exclusion",
                "pmos_inhibit",
                "--hidden-write-error-exclusion-width-u",
                "8",
            ],
        ),
        (
            "device_mnist01fixed8_64_v580_readout_synapsegate_excl_w16_e2",
            base
            + [
                "--tag",
                "device_mnist01fixed8_64_v580_readout_synapsegate_excl_w16_e2",
                *readout_common,
                "--flow-pre-store",
                "synapse_gate",
                "--readout-write-error-exclusion",
                "pmos_inhibit",
                "--readout-write-error-exclusion-width-u",
                "16",
            ],
        ),
        (
            "device_mnist01fixed8_64_v581_readout_synapsegate_excl_w2_e2",
            base
            + [
                "--tag",
                "device_mnist01fixed8_64_v581_readout_synapsegate_excl_w2_e2",
                *readout_common,
                "--flow-pre-store",
                "synapse_gate",
                "--readout-write-error-exclusion",
                "pmos_inhibit",
                "--readout-write-error-exclusion-width-u",
                "2",
            ],
        ),
    ]


def v582_v583_write_split_fullhidden_pool() -> list[tuple[str, list[str]]]:
    """Corrected mutually inhibited full-hidden checks after adding NMOS kill devices."""
    base = base_args(timeout_s="1200")
    fair_full_hidden_common = [
        "--dataset",
        "mnist01fixed8_64",
        "--epochs",
        "2",
        "--hidden-cells",
        "8",
        "--hidden-forward-mode",
        "rail_buffer",
        "--hidden-init",
        "input_identity",
        "--output-head",
        "source_follower",
        "--lead-mode",
        "out_senseamp",
        "--error-rule",
        "score",
        "--backward-gate-mode",
        "target_mistake",
        "--learning-mode",
        "flow",
        "--flow-hidden-write",
        "direct",
        "--flow-pre-store",
        "synapse_gate",
        "--readout-flow-write-mode",
        "bounded_discharge",
        "--readout-update-width-u",
        "0.00035",
        "--hidden-flow-write-mode",
        "bounded_discharge",
        "--hidden-update-width-u",
        "0.0000002",
        "--hidden-delta-internal-cap-f",
        "0.05",
        "--hidden-delta-internal-leak-ohm",
        "1000000000",
        "--hidden-delta-internal-reset-width-u",
        "4",
    ]
    return [
        (
            "device_mnist01fixed8_64_v582_fullhidden_exclusive_kill_e2",
            base
            + [
                "--tag",
                "device_mnist01fixed8_64_v582_fullhidden_exclusive_kill_e2",
                *fair_full_hidden_common,
                "--write-error-exclusion",
                "pmos_inhibit",
                "--write-error-exclusion-width-u",
                "8",
            ],
        ),
        (
            "device_mnist01fixed8_64_v583_fullhidden_hiddenexclusive_kill_e2",
            base
            + [
                "--tag",
                "device_mnist01fixed8_64_v583_fullhidden_hiddenexclusive_kill_e2",
                *fair_full_hidden_common,
                "--readout-write-error-exclusion",
                "none",
                "--hidden-write-error-exclusion",
                "pmos_inhibit",
                "--hidden-write-error-exclusion-width-u",
                "8",
            ],
        ),
    ]


def v584_v586_overlap_decay_pool() -> list[tuple[str, list[str]]]:
    """Fixed-input readout overlap-decay checks.

    These keep mnist01fixed8_64 fixed and test whether preserving a symmetric
    ambiguous-error decay path recovers the useful behavior lost by hard
    readout exclusivity.
    """
    base = base_args(timeout_s="1200")
    readout_common = binary_readout_common()
    fair_full_hidden_common = [
        "--dataset",
        "mnist01fixed8_64",
        "--epochs",
        "2",
        "--hidden-cells",
        "8",
        "--hidden-forward-mode",
        "rail_buffer",
        "--hidden-init",
        "input_identity",
        "--output-head",
        "source_follower",
        "--lead-mode",
        "out_senseamp",
        "--error-rule",
        "score",
        "--backward-gate-mode",
        "target_mistake",
        "--learning-mode",
        "flow",
        "--flow-hidden-write",
        "direct",
        "--flow-pre-store",
        "synapse_gate",
        "--readout-flow-write-mode",
        "bounded_discharge",
        "--readout-update-width-u",
        "0.00035",
        "--hidden-flow-write-mode",
        "bounded_discharge",
        "--hidden-update-width-u",
        "0.0000002",
        "--hidden-delta-internal-cap-f",
        "0.05",
        "--hidden-delta-internal-leak-ohm",
        "1000000000",
        "--hidden-delta-internal-reset-width-u",
        "4",
    ]
    return [
        (
            "device_mnist01fixed8_64_v584_readout_exclusive_reset_e2",
            base
            + [
                "--tag",
                "device_mnist01fixed8_64_v584_readout_exclusive_reset_e2",
                *readout_common,
                "--flow-pre-store",
                "synapse_gate",
                "--readout-write-error-exclusion",
                "pmos_inhibit",
                "--readout-write-error-exclusion-width-u",
                "8",
            ],
        ),
        (
            "device_mnist01fixed8_64_v585_readout_overlap_decay_e2",
            base
            + [
                "--tag",
                "device_mnist01fixed8_64_v585_readout_overlap_decay_e2",
                *readout_common,
                "--flow-pre-store",
                "synapse_gate",
                "--readout-write-error-exclusion",
                "pmos_inhibit_decay",
                "--readout-write-error-exclusion-width-u",
                "8",
            ],
        ),
        (
            "device_mnist01fixed8_64_v586_fullhidden_overlap_decay_e2",
            base
            + [
                "--tag",
                "device_mnist01fixed8_64_v586_fullhidden_overlap_decay_e2",
                *fair_full_hidden_common,
                "--readout-write-error-exclusion",
                "pmos_inhibit_decay",
                "--readout-write-error-exclusion-width-u",
                "8",
                "--hidden-write-error-exclusion",
                "pmos_inhibit",
                "--hidden-write-error-exclusion-width-u",
                "8",
            ],
        ),
    ]


def v587_v590_overlap_tune_pool() -> list[tuple[str, list[str]]]:
    """Small readout-only tuning pool for the overlap-decay primitive."""
    base = base_args(timeout_s="1200")

    def common(update_width: str = "0.00035") -> list[str]:
        args = binary_readout_common()
        args = list(args)
        idx = args.index("--readout-update-width-u") + 1
        args[idx] = update_width
        return args + [
            "--flow-pre-store",
            "synapse_gate",
            "--readout-write-error-exclusion",
            "pmos_inhibit_decay",
        ]

    return [
        (
            "device_mnist01fixed8_64_v587_readout_overlap_decay_w16_e2",
            base
            + [
                "--tag",
                "device_mnist01fixed8_64_v587_readout_overlap_decay_w16_e2",
                *common(),
                "--readout-write-error-exclusion-width-u",
                "16",
            ],
        ),
        (
            "device_mnist01fixed8_64_v588_readout_overlap_decay_w4_e2",
            base
            + [
                "--tag",
                "device_mnist01fixed8_64_v588_readout_overlap_decay_w4_e2",
                *common(),
                "--readout-write-error-exclusion-width-u",
                "4",
            ],
        ),
        (
            "device_mnist01fixed8_64_v589_readout_overlap_decay_up45_e2",
            base
            + [
                "--tag",
                "device_mnist01fixed8_64_v589_readout_overlap_decay_up45_e2",
                *common("0.00045"),
                "--readout-write-error-exclusion-width-u",
                "8",
            ],
        ),
        (
            "device_mnist01fixed8_64_v590_readout_overlap_decay_up25_e2",
            base
            + [
                "--tag",
                "device_mnist01fixed8_64_v590_readout_overlap_decay_up25_e2",
                *common("0.00025"),
                "--readout-write-error-exclusion-width-u",
                "8",
            ],
        ),
    ]


def v591_v592_overlap_fullhidden_tune_pool() -> list[tuple[str, list[str]]]:
    """Full-hidden checks using the readout overlap-decay setting that recovered 90.625% readout-only."""
    base = base_args(timeout_s="1200")

    def full_hidden_common(readout_update_width: str) -> list[str]:
        return [
            "--dataset",
            "mnist01fixed8_64",
            "--epochs",
            "2",
            "--hidden-cells",
            "8",
            "--hidden-forward-mode",
            "rail_buffer",
            "--hidden-init",
            "input_identity",
            "--output-head",
            "source_follower",
            "--lead-mode",
            "out_senseamp",
            "--error-rule",
            "score",
            "--backward-gate-mode",
            "target_mistake",
            "--learning-mode",
            "flow",
            "--flow-hidden-write",
            "direct",
            "--flow-pre-store",
            "synapse_gate",
            "--readout-flow-write-mode",
            "bounded_discharge",
            "--readout-update-width-u",
            readout_update_width,
            "--readout-write-error-exclusion",
            "pmos_inhibit_decay",
            "--readout-write-error-exclusion-width-u",
            "8",
            "--hidden-flow-write-mode",
            "bounded_discharge",
            "--hidden-update-width-u",
            "0.0000002",
            "--hidden-write-error-exclusion",
            "pmos_inhibit",
            "--hidden-write-error-exclusion-width-u",
            "8",
            "--hidden-delta-internal-cap-f",
            "0.05",
            "--hidden-delta-internal-leak-ohm",
            "1000000000",
            "--hidden-delta-internal-reset-width-u",
            "4",
        ]

    return [
        (
            "device_mnist01fixed8_64_v591_fullhidden_overlap_decay_up45_e2",
            base
            + [
                "--tag",
                "device_mnist01fixed8_64_v591_fullhidden_overlap_decay_up45_e2",
                *full_hidden_common("0.00045"),
            ],
        ),
        (
            "device_mnist01fixed8_64_v592_fullhidden_overlap_decay_up55_e2",
            base
            + [
                "--tag",
                "device_mnist01fixed8_64_v592_fullhidden_overlap_decay_up55_e2",
                *full_hidden_common("0.00055"),
            ],
        ),
    ]


def v593_v596_overlap_controls_scale_pool() -> list[tuple[str, list[str]]]:
    """Controls and 128-sample scaling for the overlap-decay readout block."""
    base = base_args(timeout_s="1800")

    def readout_common(dataset: str, update_width: str = "0.00055") -> list[str]:
        args = binary_readout_common(dataset=dataset, epochs="2")
        args = list(args)
        idx = args.index("--readout-update-width-u") + 1
        args[idx] = update_width
        return args + [
            "--flow-pre-store",
            "synapse_gate",
            "--readout-write-error-exclusion",
            "pmos_inhibit_decay",
            "--readout-write-error-exclusion-width-u",
            "8",
        ]

    def full_hidden_common(dataset: str, update_width: str = "0.00055") -> list[str]:
        return [
            "--dataset",
            dataset,
            "--epochs",
            "2",
            "--hidden-cells",
            "8",
            "--hidden-forward-mode",
            "rail_buffer",
            "--hidden-init",
            "input_identity",
            "--output-head",
            "source_follower",
            "--lead-mode",
            "out_senseamp",
            "--error-rule",
            "score",
            "--backward-gate-mode",
            "target_mistake",
            "--learning-mode",
            "flow",
            "--flow-hidden-write",
            "direct",
            "--flow-pre-store",
            "synapse_gate",
            "--readout-flow-write-mode",
            "bounded_discharge",
            "--readout-update-width-u",
            update_width,
            "--readout-write-error-exclusion",
            "pmos_inhibit_decay",
            "--readout-write-error-exclusion-width-u",
            "8",
            "--hidden-flow-write-mode",
            "bounded_discharge",
            "--hidden-update-width-u",
            "0.0000002",
            "--hidden-write-error-exclusion",
            "pmos_inhibit",
            "--hidden-write-error-exclusion-width-u",
            "8",
            "--hidden-delta-internal-cap-f",
            "0.05",
            "--hidden-delta-internal-leak-ohm",
            "1000000000",
            "--hidden-delta-internal-reset-width-u",
            "4",
        ]

    return [
        (
            "device_mnist01fixed8_64_v593_readout_overlap_up55_reversed_e2",
            base
            + [
                "--tag",
                "device_mnist01fixed8_64_v593_readout_overlap_up55_reversed_e2",
                *readout_common("mnist01fixed8_64"),
                "--readout-flow-polarity",
                "reversed",
            ],
        ),
        (
            "device_mnist01fixed8_64_v594_readout_overlap_up55_labelshuffle_e2",
            base
            + [
                "--tag",
                "device_mnist01fixed8_64_v594_readout_overlap_up55_labelshuffle_e2",
                *readout_common("mnist01fixed8_64"),
                "--label-shuffle-seed",
                "23",
            ],
        ),
        (
            "device_mnist01fixed8_128_v595_readout_overlap_up55_e2",
            base
            + [
                "--tag",
                "device_mnist01fixed8_128_v595_readout_overlap_up55_e2",
                *readout_common("mnist01fixed8_128"),
            ],
        ),
        (
            "device_mnist01fixed8_128_v596_fullhidden_overlap_up55_e2",
            base
            + [
                "--tag",
                "device_mnist01fixed8_128_v596_fullhidden_overlap_up55_e2",
                *full_hidden_common("mnist01fixed8_128"),
            ],
        ),
    ]


def v597_v600_128_overlap_mobility_pool() -> list[tuple[str, list[str]]]:
    """128-sample readout-only mobility bracket for overlap-decay writes."""
    base = base_args(timeout_s="1800")

    def readout_common(tag: str, update_width: str) -> tuple[str, list[str]]:
        args = binary_readout_common(dataset="mnist01fixed8_128", epochs="2")
        args = list(args)
        idx = args.index("--readout-update-width-u") + 1
        args[idx] = update_width
        return (
            tag,
            base
            + [
                "--tag",
                tag,
                *args,
                "--flow-pre-store",
                "synapse_gate",
                "--readout-write-error-exclusion",
                "pmos_inhibit_decay",
                "--readout-write-error-exclusion-width-u",
                "8",
            ],
        )

    return [
        readout_common("device_mnist01fixed8_128_v597_readout_overlap_up35_e2", "0.00035"),
        readout_common("device_mnist01fixed8_128_v598_readout_overlap_up40_e2", "0.00040"),
        readout_common("device_mnist01fixed8_128_v599_readout_overlap_up45_e2", "0.00045"),
        readout_common("device_mnist01fixed8_128_v600_readout_overlap_up50_e2", "0.00050"),
    ]


POOLS = {
    "v557_v560": v557_v560_pool,
    "v561_v562": v561_v562_pool,
    "v563_v565": v563_v565_pool,
    "v566_v568": v566_v568_pool,
    "v569_v574_controls": v569_v574_controls_pool,
    "v575_v577_exclusive": v575_v577_exclusive_pool,
    "v578_v581_write_split": v578_v581_write_split_pool,
    "v582_v583_write_split_fullhidden": v582_v583_write_split_fullhidden_pool,
    "v584_v586_overlap_decay": v584_v586_overlap_decay_pool,
    "v587_v590_overlap_tune": v587_v590_overlap_tune_pool,
    "v591_v592_overlap_fullhidden_tune": v591_v592_overlap_fullhidden_tune_pool,
    "v593_v596_overlap_controls_scale": v593_v596_overlap_controls_scale_pool,
    "v597_v600_128_overlap_mobility": v597_v600_128_overlap_mobility_pool,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", choices=sorted(POOLS), default="v557_v560")
    parser.add_argument("--max-parallel", type=int, default=0, help="0 means run the full pool at once.")
    parser.add_argument("--poll-s", type=float, default=30.0)
    args = parser.parse_args()

    runs = POOLS[args.pool]()
    max_parallel = len(runs) if args.max_parallel <= 0 else args.max_parallel
    logdir = ROOT / "spice/results/run_logs" / args.pool
    logdir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/private/tmp/mpl_spicenn")

    pending = list(runs)
    running: list[tuple[str, subprocess.Popen[str], object, float]] = []
    print(f"launching {len(runs)} runs into {logdir}", flush=True)

    while pending or running:
        while pending and len(running) < max_parallel:
            tag, cmd = pending.pop(0)
            log = (logdir / f"{tag}.log").open("w")
            proc = subprocess.Popen(
                cmd,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            running.append((tag, proc, log, time.time()))
            print(f"started {tag} pid={proc.pid}", flush=True)

        still_running: list[tuple[str, subprocess.Popen[str], object, float]] = []
        for tag, proc, log, started in running:
            rc = proc.poll()
            if rc is None:
                still_running.append((tag, proc, log, started))
                continue
            log.close()
            elapsed = time.time() - started
            print(f"finished {tag} rc={rc} elapsed_s={elapsed:.1f}", flush=True)
        running = still_running

        if pending or running:
            names = [tag for tag, _, _, _ in running]
            print(f"running={names} pending={len(pending)}", flush=True)
            time.sleep(args.poll_s)

    print("pool_complete", flush=True)


if __name__ == "__main__":
    main()
