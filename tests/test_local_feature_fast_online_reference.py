import argparse
import shlex
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPICE_DIR = ROOT / "spice"
if str(SPICE_DIR) not in sys.path:
    sys.path.insert(0, str(SPICE_DIR))

import run_spice_mnist_local_feature_fast_online_reference as fast_ref  # noqa: E402
import run_spice_mnist_local_feature_fast_online_variant_sweep as fast_sweep  # noqa: E402


def test_synapse_transfer_np_matches_spice_transfer_families() -> None:
    weights = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])

    assert fast_ref.synapse_transfer_np(weights, "linear", 0.25).tolist() == pytest.approx(weights.tolist())
    assert fast_ref.synapse_transfer_np(weights, "hard-clipped", 0.25).tolist() == pytest.approx(
        [-0.25, -0.25, 0.0, 0.25, 0.25]
    )
    assert fast_ref.synapse_transfer_np(np.array([0.25]), "tanh-clipped", 0.25)[0] == pytest.approx(0.25 * np.tanh(1.0))
    assert fast_ref.synapse_transfer_np(np.array([-0.5, 0.5]), "sign", 0.25).tolist() == pytest.approx([-0.25, 0.25])


def test_fast_online_softmax_update_uses_raw_weight_update_with_effective_forward_synapses() -> None:
    x = np.array([[0.5, 1.0]])
    labels = np.array([1])
    blocks = [[0, 1]]
    w = np.array([[[0.2, -0.1]]])
    hb = np.array([[0.0]])
    readout = np.array([[[0.3]], [[-0.2]]])
    output_bias = np.array([0.0, 0.0])
    state = (w.copy(), hb.copy(), readout.copy(), output_bias.copy())

    updated = fast_ref.update_np(
        x,
        labels,
        state,
        blocks,
        lr=0.8,
        linear_output=False,
        softmax_output=True,
        local_activation="tanh",
        relu_clip=1.0,
        activation_derivative="exact",
        derivative_floor=0.0,
        derivative_gate_threshold=1e-6,
        readout_feedback_mode="readout",
        readout_feedback_clip=0.05,
        relu_leak=0.01,
        softplus_beta=10.0,
        hidden_synapse_mode="tanh-clipped",
        readout_synapse_mode="linear",
        synapse_clip=0.25,
    )

    effective_w = 0.25 * np.tanh(w / 0.25)
    pre = np.einsum("nbp,bcp->nbc", fast_ref.block_tensor_np(x, blocks), effective_w) + hb
    h = np.tanh(pre)
    score = np.einsum("nbc,kbc->nk", h, readout) + output_bias
    y = fast_ref.output_activation_np(score, linear_output=False, softmax_output=True)
    d = np.array([[0.0, 1.0]]) - y
    dh = np.einsum("nk,kbc->nbc", d, readout) * (1.0 - h * h)

    assert updated[0] == pytest.approx(w + 0.8 * np.einsum("nbc,nbp->bcp", dh, fast_ref.block_tensor_np(x, blocks)))
    assert updated[1] == pytest.approx(hb + 0.8 * dh[0])
    assert updated[2] == pytest.approx(readout + 0.8 * np.einsum("nk,nbc->kbc", d, h))
    assert updated[3] == pytest.approx(output_bias + 0.8 * d[0])


def test_fast_online_reference_matches_phase_softmax_error_knobs() -> None:
    targets = np.array([[1.0, 0.0, 0.0]])
    y = np.array([[0.2, 0.3, 0.5]])
    score = np.array([[0.1, 0.4, 0.8]])

    default = fast_ref.softmax_delta_np(
        targets,
        y,
        score,
        softmax_negative_scale=1.0,
        softmax_error_centering="none",
        softmax_competition_mode="all",
        softmax_competitor_power=2,
        softmax_error_gate="none",
        softmax_margin=1.0,
    )
    assert default == pytest.approx(np.array([[0.8, -0.3, -0.5]]))

    focused = fast_ref.softmax_delta_np(
        targets,
        y,
        score,
        softmax_negative_scale=1.0,
        softmax_error_centering="none",
        softmax_competition_mode="normalized-power",
        softmax_competitor_power=2,
        softmax_error_gate="none",
        softmax_margin=1.0,
    )
    denom = 0.3 * 0.3 + 0.5 * 0.5
    assert focused == pytest.approx(np.array([[0.8, -0.8 * 0.09 / denom, -0.8 * 0.25 / denom]]))

    gated = fast_ref.softmax_delta_np(
        targets,
        y,
        score,
        softmax_negative_scale=1.0,
        softmax_error_centering="none",
        softmax_competition_mode="all",
        softmax_competitor_power=2,
        softmax_error_gate="target-margin",
        softmax_margin=1.0,
    )
    gate = (1.0 - (0.1 - 0.8)) / (1.0 + 1e-12)
    assert gated == pytest.approx(default * np.clip(gate, 0.0, 1.0))


def test_fast_online_reference_honors_update_scales_and_decay() -> None:
    x = np.array([[0.5, 1.0]])
    labels = np.array([1])
    blocks = [[0, 1]]
    w = np.array([[[0.2, -0.1]]])
    hb = np.array([[0.05]])
    readout = np.array([[[0.3]], [[-0.2]]])
    output_bias = np.array([0.1, -0.1])
    state = (w.copy(), hb.copy(), readout.copy(), output_bias.copy())

    frozen_head = fast_ref.update_np(
        x,
        labels,
        state,
        blocks,
        lr=0.8,
        linear_output=False,
        softmax_output=True,
        local_activation="tanh",
        relu_clip=1.0,
        activation_derivative="exact",
        derivative_floor=0.0,
        derivative_gate_threshold=1e-6,
        readout_feedback_mode="readout",
        readout_feedback_clip=0.05,
        relu_leak=0.01,
        softplus_beta=10.0,
        hidden_synapse_mode="linear",
        readout_synapse_mode="linear",
        synapse_clip=1.0,
        output_bias_update_scale=0.0,
        readout_update_scale=0.0,
        local_update_scale=0.0,
        state_decay=0.1,
    )

    assert frozen_head[0] == pytest.approx(0.9 * w)
    assert frozen_head[1] == pytest.approx(0.9 * hb)
    assert frozen_head[2] == pytest.approx(0.9 * readout)
    assert frozen_head[3] == pytest.approx(0.9 * output_bias)


def test_fast_online_reference_linear_decay_lr_schedule_matches_manual_updates() -> None:
    x = np.array([[0.5, 0.0], [0.0, 0.5]])
    y = np.array([0, 1])
    blocks = [[0, 1]]
    state = (
        np.array([[[0.1, -0.1]]]),
        np.array([[0.0]]),
        np.array([[[0.2]], [[-0.2]]]),
        np.zeros(2),
    )
    kwargs = dict(
        lr=0.8,
        local_activation="tanh",
        relu_clip=1.0,
        relu_leak=0.01,
        softplus_beta=10.0,
        activation_derivative="exact",
        derivative_floor=0.0,
        derivative_gate_threshold=1e-6,
        readout_feedback_mode="full-readout",
        readout_feedback_clip=0.05,
        output_bias_update_scale=0.0,
        readout_update_scale=0.25,
        local_update_scale=1.0,
        state_decay=0.0,
        softmax_negative_scale=1.0,
        softmax_error_centering="none",
        softmax_temperature=4.0,
        softmax_competition_mode="all",
        softmax_competitor_power=2,
        softmax_error_gate="none",
        softmax_margin=1.0,
        hidden_synapse_mode="linear",
        readout_synapse_mode="linear",
        synapse_clip=1.0,
        readout_class_centering="none",
        linear_output=False,
        softmax_output=True,
    )

    scheduled, _probes = fast_ref.run_online(
        x,
        y,
        state,
        blocks,
        (),
        **kwargs,
        lr_schedule="linear-decay",
        lr_final_scale=0.25,
    )
    manual = fast_ref.update_np(x[:1], y[:1], state, blocks, **{**kwargs, "lr": 0.8})
    manual = fast_ref.update_np(x[1:], y[1:], manual, blocks, **{**kwargs, "lr": 0.2})

    for got, expected in zip(scheduled, manual):
        assert got == pytest.approx(expected)


def test_fast_online_variant_sweep_grid_uses_phase_variant_family_axes() -> None:
    args = argparse.Namespace(
        activations="tanh,diff-clipped-relu",
        relu_clips="0.5,1.0",
        derivative_modes="exact,stored-gate",
        feedback_modes="readout,full-readout",
        hidden_synapse_modes="linear,tanh-clipped",
        readout_synapse_modes="linear",
        synapse_clip=2.0,
        synapse_clips="1.0,2.0",
        lr=0.8,
        lrs="0.4,0.8",
        lr_schedule="constant",
        lr_schedules="constant,linear-decay",
        lr_final_scale=1.0,
        lr_final_scales="0.25,0.5",
        output_bias_update_scale=0.0,
        output_bias_update_scales="",
        readout_update_scale=0.25,
        readout_update_scales="0.25,0.5",
        local_update_scale=1.0,
        local_update_scales="",
        state_decay=0.0,
        state_decays="",
        softmax_temperature=4.0,
        softmax_temperatures="2.0,4.0",
        tag="fastscreen",
    )

    rows = fast_sweep.variant_grid(args)

    assert len(rows) == 1152
    assert rows[0]["local_activation"] == "tanh"
    assert rows[0]["relu_clip"] == 0.5
    assert rows[0]["lr"] == 0.4
    assert rows[0]["lr_schedule"] == "constant"
    assert rows[0]["lr_final_scale"] == 1.0
    assert rows[0]["readout_update_scale"] == 0.25
    assert rows[0]["softmax_temperature"] == 2.0
    assert any(row["local_activation"] == "diff-clipped-relu" and row["relu_clip"] == 1.0 for row in rows)
    assert any(row["lr"] == 0.8 and row["readout_update_scale"] == 0.5 for row in rows)
    assert any(row["lr_schedule"] == "linear-decay" and row["lr_final_scale"] == 0.25 for row in rows)
    assert not any(row["lr_schedule"] == "constant" and row["lr_final_scale"] != 1.0 for row in rows)
    assert all("tag" in row for row in rows)
    assert all("_lr" in row["tag"] and "_temp" in row["tag"] for row in rows)


def test_fast_online_variant_sweep_row_reports_best_probe_and_improvement() -> None:
    args = argparse.Namespace(
        lr=0.8,
        linear_output=False,
        softmax_output=True,
        relu_leak=0.01,
        softplus_beta=10.0,
        derivative_floor=0.0,
        derivative_gate_threshold=1e-6,
        readout_feedback_clip=0.05,
        output_bias_update_scale=0.0,
        readout_update_scale=0.25,
        local_update_scale=1.0,
        state_decay=0.0,
        softmax_negative_scale=1.0,
        softmax_error_centering="none",
        softmax_temperature=4.0,
        softmax_competition_mode="all",
        softmax_competitor_power=2,
        softmax_error_gate="none",
        softmax_margin=1.0,
        readout_class_centering="none",
        eval_batch_size=4,
        train_samples=2,
        eval_samples=2,
        image_size=2,
        block_size=2,
        stride=2,
        channels=1,
        promotion_updates=2,
        promotion_simulator="Xyce",
        promotion_phase=0.5e-9,
        promotion_gap=0.05e-9,
        promotion_edge=5e-12,
        promotion_settle_ratio=20.0,
        promotion_transient_step=200e-12,
        promotion_timeout=240.0,
        promotion_max_transient_points=2000,
        promotion_max_source_pwl_points=5000,
        promotion_phase_clock_mode="analytic",
        promotion_probe_updates="",
        promotion_tag_prefix="promote",
        full_objective_eval_samples=2,
        full_objective_accuracy=0.5,
    )
    variant = {
        "local_activation": "tanh",
        "relu_clip": 1.0,
        "activation_derivative": "exact",
        "readout_feedback_mode": "full-readout",
        "hidden_synapse_mode": "linear",
        "readout_synapse_mode": "linear",
        "synapse_clip": 1.0,
        "lr": 0.8,
        "lr_schedule": "linear-decay",
        "lr_final_scale": 0.25,
        "output_bias_update_scale": 0.0,
        "readout_update_scale": 0.25,
        "local_update_scale": 1.0,
        "state_decay": 0.0,
        "softmax_temperature": 4.0,
        "tag": "row",
    }
    x_train = np.array([[0.5, 0.0], [0.0, 0.5]])
    y_train = np.array([0, 1])
    x_eval = x_train.copy()
    y_eval = y_train.copy()
    blocks = [[0, 1]]
    state = (
        np.array([[[0.1, -0.1]]]),
        np.array([[0.0]]),
        np.array([[[0.2]], [[-0.2]]]),
        np.zeros(2),
    )

    row = fast_sweep.run_variant(args, variant, x_train, y_train, x_eval, y_eval, state, blocks, (1, 2))

    assert row["tag"] == "row"
    assert row["lr"] == 0.8
    assert row["lr_schedule"] == "linear-decay"
    assert row["lr_final_scale"] == 0.25
    assert row["readout_update_scale"] == 0.25
    assert row["softmax_temperature"] == 4.0
    command = shlex.split(row["strict_phase_promotion_command"])
    assert "--strict-fully-on-device" in command
    assert command[command.index("--reference-mode") + 1] == "none"
    assert command[command.index("--phase-output-mode") + 1] == "print"
    assert command[command.index("--update-mode") + 1] == "direct"
    assert command[command.index("--phase-clock-mode") + 1] == "analytic"
    assert command[command.index("--target-source-mode") + 1] == "label"
    assert command[command.index("--max-output-vectors") + 1] == "0"
    assert "--phase-output-include-y" not in command
    assert command[command.index("--eval-backend") + 1] == "numpy"
    assert command[command.index("--updates") + 1] == "2"
    assert command[command.index("--lr") + 1] == "0.8"
    assert command[command.index("--lr-schedule") + 1] == "linear-decay"
    assert command[command.index("--lr-final-scale") + 1] == "0.25"
    assert command[command.index("--readout-update-scale") + 1] == "0.25"
    assert command[command.index("--softmax-temperature") + 1] == "4.0"
    assert row["strict_phase_promotion_updates"] == 2
    assert row["strict_phase_promotion_timeout_s"] == 240.0
    assert row["strict_phase_promotion_max_transient_points"] == 2000
    assert row["strict_phase_promotion_max_source_pwl_points"] == 5000
    assert row["strict_phase_promotion_max_output_vectors"] == 0
    assert row["strict_phase_promotion_phase_clock_mode"] == "analytic"
    assert row["strict_phase_promotion_target_source_mode"] == "label"
    assert row["strict_phase_promotion_output_bias_state_frozen"] is True
    assert row["strict_phase_promotion_phase_output_includes_y"] is False
    assert row["strict_phase_promotion_output_vector_count"] == 15
    assert row["strict_phase_promotion_output_vector_budget_met"] is True
    assert row["strict_phase_promotion_estimated_transient_points"] == 34
    assert row["strict_phase_promotion_transient_budget_met"] is True
    assert row["strict_phase_promotion_phase_clock_source_pwl_points"] == 0
    assert row["strict_phase_promotion_control_source_pwl_points"] > 0
    assert row["strict_phase_promotion_total_source_pwl_points"] > row["strict_phase_promotion_sample_source_pwl_points"]
    assert row["promotion_probe_eval_improvement"] == pytest.approx(
        row["promotion_probe_eval_accuracy"] - row["initial_eval_accuracy"]
    )
    assert row["strict_phase_promotion_source_pwl_points_per_update"] == pytest.approx(
        row["strict_phase_promotion_total_source_pwl_points"] / row["strict_phase_promotion_updates"]
    )
    assert row["strict_phase_promotion_eval_improvement_per_1k_source_pwl"] is not None
    assert row["strict_phase_promotion_eval_improvement_per_1k_transient_points"] is not None
    assert row["strict_phase_promotion_source_pwl_budget_met"] is True
    assert row["initial_eval_accuracy"] >= 0.0
    assert row["final_eval_accuracy"] >= 0.0
    assert row["eval_improvement"] == pytest.approx(row["final_eval_accuracy"] - row["initial_eval_accuracy"])
    assert row["fast_reference_full_eval_sample_count_met"] is True
    assert row["fast_reference_full_objective_accuracy_gap"] is not None
    assert row["fast_reference_full_objective_candidate"] == (
        row["fast_reference_full_objective_accuracy_met"] is True
    )
    assert row["best_probe_update"] in {1, 2}
    assert row["probe_eval_accuracy_u1"] >= 0.0
    assert row["probe_eval_accuracy_u2"] >= 0.0
    assert row["promotion_probe_eval_accuracy"] == row["probe_eval_accuracy_u2"]


def test_fast_online_strict_promotion_defaults_to_pwl_phase_clock() -> None:
    args = argparse.Namespace(
        train_samples=2,
        eval_samples=2,
        image_size=2,
        block_size=2,
        stride=2,
        channels=1,
        promotion_updates=2,
        promotion_simulator="Xyce",
        promotion_phase=0.5e-9,
        promotion_gap=0.05e-9,
        promotion_edge=5e-12,
        promotion_settle_ratio=20.0,
        promotion_transient_step=200e-12,
        promotion_timeout=240.0,
        promotion_max_transient_points=2000,
        promotion_max_source_pwl_points=5000,
        promotion_probe_updates="",
        promotion_tag_prefix="promote",
        linear_output=False,
        softmax_output=True,
        relu_leak=0.01,
        softplus_beta=10.0,
        derivative_floor=0.0,
        derivative_gate_threshold=1e-6,
        readout_feedback_clip=0.05,
        softmax_negative_scale=1.0,
        softmax_error_centering="none",
        softmax_competition_mode="all",
        softmax_competitor_power=2,
        softmax_error_gate="none",
        softmax_margin=1.0,
        readout_class_centering="none",
    )
    variant = {
        "local_activation": "tanh",
        "relu_clip": 1.0,
        "activation_derivative": "exact",
        "readout_feedback_mode": "full-readout",
        "hidden_synapse_mode": "linear",
        "readout_synapse_mode": "linear",
        "synapse_clip": 1.0,
        "lr": 0.8,
        "output_bias_update_scale": 0.0,
        "readout_update_scale": 0.25,
        "local_update_scale": 1.0,
        "state_decay": 0.0,
        "softmax_temperature": 4.0,
        "tag": "row",
    }

    command = fast_sweep.strict_phase_promotion_command(args, variant)
    fields = fast_sweep.strict_phase_promotion_cost_fields(
        args,
        variant,
        np.array([[0.5, 0.0], [0.0, 0.5]]),
        np.array([0, 1]),
    )

    assert command[command.index("--phase-clock-mode") + 1] == "pwl"
    assert command[command.index("--target-source-mode") + 1] == "label"
    assert fields["strict_phase_promotion_phase_clock_mode"] == "pwl"
    assert fields["strict_phase_promotion_phase_clock_source_pwl_points"] > 0
    assert fields["strict_phase_promotion_control_source_pwl_points"] == 0


def test_fast_online_strict_promotion_timeout_auto_scales_with_updates() -> None:
    args = argparse.Namespace(
        train_samples=512,
        eval_samples=1000,
        image_size=10,
        block_size=4,
        stride=2,
        channels=2,
        promotion_updates=512,
        promotion_simulator="Xyce",
        promotion_phase=0.5e-9,
        promotion_gap=0.05e-9,
        promotion_edge=5e-12,
        promotion_settle_ratio=20.0,
        promotion_transient_step=200e-12,
        promotion_timeout=0.0,
        promotion_max_transient_points=8000,
        promotion_max_source_pwl_points=80000,
        promotion_probe_updates="",
        promotion_tag_prefix="promote",
        linear_output=False,
        softmax_output=True,
        relu_leak=0.01,
        softplus_beta=10.0,
        derivative_floor=0.0,
        derivative_gate_threshold=1e-6,
        readout_feedback_clip=0.05,
        softmax_negative_scale=1.0,
        softmax_error_centering="none",
        softmax_competition_mode="all",
        softmax_competitor_power=2,
        softmax_error_gate="none",
        softmax_margin=1.0,
        readout_class_centering="none",
    )
    variant = {
        "local_activation": "tanh",
        "relu_clip": 1.0,
        "activation_derivative": "exact",
        "readout_feedback_mode": "full-readout",
        "hidden_synapse_mode": "tanh-clipped",
        "readout_synapse_mode": "linear",
        "synapse_clip": 2.0,
        "lr": 0.5,
        "output_bias_update_scale": 0.0,
        "readout_update_scale": 0.35,
        "local_update_scale": 1.0,
        "state_decay": 0.0,
        "softmax_temperature": 2.5,
        "tag": "row",
    }

    command = fast_sweep.strict_phase_promotion_command(args, variant)

    assert fast_sweep.promotion_timeout_seconds(args, 512) == pytest.approx(640.0)
    assert command[command.index("--timeout") + 1] == "640.0"


def test_fast_online_strict_promotion_cost_fields_respect_pwl_clock_override() -> None:
    args = argparse.Namespace(
        train_samples=2,
        promotion_updates=2,
        promotion_phase=0.5e-9,
        promotion_gap=0.05e-9,
        promotion_edge=5e-12,
        promotion_transient_step=200e-12,
        promotion_max_transient_points=100,
        promotion_max_source_pwl_points=50,
        promotion_phase_clock_mode="pwl",
        softmax_output=True,
    )
    x_train = np.array([[0.5, 0.0], [0.0, 0.5]])
    y_train = np.array([0, 1])
    variant = {
        "lr": 0.8,
        "lr_schedule": "constant",
        "lr_final_scale": 1.0,
    }

    fields = fast_sweep.strict_phase_promotion_cost_fields(args, variant, x_train, y_train)

    assert fields["strict_phase_promotion_phase_clock_mode"] == "pwl"
    assert fields["strict_phase_promotion_target_source_mode"] == "label"
    assert fields["strict_phase_promotion_output_vector_count"] > 0
    assert fields["strict_phase_promotion_output_vector_budget_met"] is True
    assert fields["strict_phase_promotion_estimated_transient_points"] == 34
    assert fields["strict_phase_promotion_phase_clock_source_pwl_points"] == 45
    assert fields["strict_phase_promotion_control_source_pwl_points"] == 0
    assert fields["strict_phase_promotion_total_source_pwl_points"] > fields["strict_phase_promotion_sample_source_pwl_points"]
    assert fields["strict_phase_promotion_source_pwl_budget_met"] is False


def test_fast_online_strict_cost_projection_can_use_a_different_horizon() -> None:
    args = argparse.Namespace(
        train_samples=4,
        promotion_updates=2,
        cost_projection_updates=4,
        promotion_phase=0.5e-9,
        promotion_gap=0.05e-9,
        promotion_edge=5e-12,
        promotion_transient_step=200e-12,
        promotion_max_transient_points=100,
        promotion_max_source_pwl_points=500,
        promotion_max_output_vectors=80,
        promotion_phase_clock_mode="pwl",
        promotion_target_source_mode="label",
        promotion_phase_output_include_y=False,
        image_size=2,
        block_size=2,
        stride=2,
        channels=1,
        softmax_output=True,
    )
    x_train = np.array([[0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [0.0, 0.0]])
    y_train = np.array([0, 1, 0, 1])
    variant = {
        "lr": 0.8,
        "lr_schedule": "constant",
        "lr_final_scale": 1.0,
        "output_bias_update_scale": 0.0,
        "state_decay": 0.0,
    }

    promotion = fast_sweep.strict_phase_promotion_cost_fields(args, variant, x_train, y_train)
    projection = fast_sweep.strict_phase_cost_projection_fields(args, variant, x_train, y_train)

    assert promotion["strict_phase_promotion_updates"] == 2
    assert projection["strict_phase_cost_projection_updates"] == 4
    assert (
        projection["strict_phase_cost_projection_estimated_transient_points"]
        > promotion["strict_phase_promotion_estimated_transient_points"]
    )
    assert (
        projection["strict_phase_cost_projection_total_source_pwl_points"]
        > promotion["strict_phase_promotion_total_source_pwl_points"]
    )
    assert projection["strict_phase_cost_projection_source_pwl_points_per_update"] == pytest.approx(
        projection["strict_phase_cost_projection_total_source_pwl_points"] / 4.0
    )
    assert projection["strict_phase_cost_projection_output_vector_count"] == promotion["strict_phase_promotion_output_vector_count"]


def test_fast_online_variant_sweep_selects_best_promotion_variant() -> None:
    rows = [
        {"tag": "late", "final_eval_accuracy": 0.9, "promotion_probe_eval_accuracy": 0.4, "eval_improvement": 0.8},
        {"tag": "early", "final_eval_accuracy": 0.8, "promotion_probe_eval_accuracy": 0.6, "eval_improvement": 0.7},
    ]

    assert fast_sweep.best_promotion_variant(rows)["tag"] == "early"


def test_fast_online_variant_sweep_selects_best_budget_feasible_efficiency_variant() -> None:
    rows = [
        {
            "tag": "highest_accuracy",
            "promotion_probe_eval_accuracy": 0.8,
            "promotion_probe_eval_improvement": 0.7,
            "eval_improvement": 0.7,
            "strict_phase_promotion_eval_improvement_per_1k_source_pwl": 0.2,
            "strict_phase_promotion_transient_budget_met": True,
            "strict_phase_promotion_source_pwl_budget_met": True,
            "strict_phase_promotion_output_vector_budget_met": True,
        },
        {
            "tag": "most_efficient",
            "promotion_probe_eval_accuracy": 0.7,
            "promotion_probe_eval_improvement": 0.6,
            "eval_improvement": 0.6,
            "strict_phase_promotion_eval_improvement_per_1k_source_pwl": 0.4,
            "strict_phase_promotion_transient_budget_met": True,
            "strict_phase_promotion_source_pwl_budget_met": True,
            "strict_phase_promotion_output_vector_budget_met": True,
        },
        {
            "tag": "efficient_but_too_expensive",
            "promotion_probe_eval_accuracy": 0.6,
            "promotion_probe_eval_improvement": 0.5,
            "eval_improvement": 0.5,
            "strict_phase_promotion_eval_improvement_per_1k_source_pwl": 0.8,
            "strict_phase_promotion_transient_budget_met": True,
            "strict_phase_promotion_source_pwl_budget_met": False,
            "strict_phase_promotion_output_vector_budget_met": True,
        },
    ]

    assert fast_sweep.best_promotion_variant(rows)["tag"] == "highest_accuracy"
    assert fast_sweep.best_promotion_efficiency_variant(rows)["tag"] == "most_efficient"


def test_fast_online_variant_sweep_has_no_efficiency_variant_without_feasible_gain() -> None:
    rows = [
        {
            "tag": "missing_efficiency",
            "promotion_probe_eval_accuracy": 0.8,
            "strict_phase_promotion_eval_improvement_per_1k_source_pwl": None,
            "strict_phase_promotion_transient_budget_met": True,
            "strict_phase_promotion_source_pwl_budget_met": True,
            "strict_phase_promotion_output_vector_budget_met": True,
        },
        {
            "tag": "infeasible",
            "promotion_probe_eval_accuracy": 0.7,
            "strict_phase_promotion_eval_improvement_per_1k_source_pwl": 0.4,
            "strict_phase_promotion_transient_budget_met": False,
            "strict_phase_promotion_source_pwl_budget_met": True,
            "strict_phase_promotion_output_vector_budget_met": True,
        },
    ]

    assert fast_sweep.best_promotion_efficiency_variant(rows) is None


def test_fast_online_variant_sweep_reports_best_probe_horizons_and_threshold_hits() -> None:
    rows = [
        {
            "tag": "fast_start",
            "local_activation": "tanh",
            "relu_clip": 1.0,
            "activation_derivative": "exact",
            "readout_feedback_mode": "full-readout",
            "hidden_synapse_mode": "tanh-clipped",
            "readout_synapse_mode": "linear",
            "synapse_clip": 2.0,
            "lr": 0.8,
            "lr_schedule": "constant",
            "lr_final_scale": 1.0,
            "output_bias_update_scale": 0.0,
            "readout_update_scale": 0.25,
            "local_update_scale": 1.0,
            "state_decay": 0.0,
            "softmax_temperature": 4.0,
            "initial_eval_accuracy": 0.1,
            "final_eval_accuracy": 0.5,
            "eval_improvement": 0.4,
            "best_probe_update": 128,
            "best_probe_eval_accuracy": 0.5,
            "probe_eval_accuracy_u64": 0.3,
            "probe_eval_accuracy_u128": 0.5,
        },
        {
            "tag": "slow_better",
            "local_activation": "tanh",
            "relu_clip": 1.0,
            "activation_derivative": "exact",
            "readout_feedback_mode": "full-readout",
            "hidden_synapse_mode": "tanh-clipped",
            "readout_synapse_mode": "linear",
            "synapse_clip": 2.0,
            "lr": 0.2,
            "lr_schedule": "constant",
            "lr_final_scale": 1.0,
            "output_bias_update_scale": 0.0,
            "readout_update_scale": 0.35,
            "local_update_scale": 1.0,
            "state_decay": 0.0,
            "softmax_temperature": 2.5,
            "initial_eval_accuracy": 0.1,
            "final_eval_accuracy": 0.7,
            "eval_improvement": 0.6,
            "best_probe_update": 128,
            "best_probe_eval_accuracy": 0.7,
            "probe_eval_accuracy_u64": 0.25,
            "probe_eval_accuracy_u128": 0.7,
        },
    ]

    probe_best = fast_sweep.best_probe_variants_by_update(rows)
    hits = fast_sweep.learning_threshold_hits(probe_best, [0.2, 0.5, 0.75])

    assert [row["probe_update"] for row in probe_best] == [64, 128]
    assert probe_best[0]["tag"] == "fast_start"
    assert probe_best[0]["probe_eval_accuracy"] == pytest.approx(0.3)
    assert probe_best[0]["probe_eval_improvement"] == pytest.approx(0.2)
    assert probe_best[1]["tag"] == "slow_better"
    assert probe_best[1]["probe_eval_accuracy"] == pytest.approx(0.7)
    assert hits[0]["met"] is True
    assert hits[0]["probe_update"] == 64
    assert hits[1]["met"] is True
    assert hits[1]["probe_update"] == 128
    assert hits[2] == {
        "threshold": 0.75,
        "met": False,
        "best_available_probe_update": 128,
        "best_available_probe_eval_accuracy": 0.7,
        "best_available_tag": "slow_better",
    }


def test_fast_online_variant_sweep_rejects_invalid_learning_thresholds() -> None:
    assert fast_sweep.parse_accuracy_thresholds("") == []
    assert fast_sweep.parse_accuracy_thresholds("0.2,0.9") == [0.2, 0.9]
    with pytest.raises(ValueError, match="learning-thresholds"):
        fast_sweep.parse_accuracy_thresholds("1.1")


def test_fast_online_variant_sweep_marks_full_objective_candidates_as_fast_reference_only() -> None:
    miss_eval = fast_sweep.fast_reference_objective_fields(
        final_eval_accuracy=0.95,
        eval_samples=300,
        full_objective_eval_samples=10000,
        full_objective_accuracy=0.9,
    )
    miss_accuracy = fast_sweep.fast_reference_objective_fields(
        final_eval_accuracy=0.89,
        eval_samples=10000,
        full_objective_eval_samples=10000,
        full_objective_accuracy=0.9,
    )
    hit = fast_sweep.fast_reference_objective_fields(
        final_eval_accuracy=0.91,
        eval_samples=10000,
        full_objective_eval_samples=10000,
        full_objective_accuracy=0.9,
    )

    assert miss_eval == {
        "fast_reference_full_eval_sample_count_met": False,
        "fast_reference_full_objective_accuracy_met": True,
        "fast_reference_full_objective_accuracy_gap": 0.0,
        "fast_reference_full_objective_candidate": False,
    }
    assert miss_accuracy == {
        "fast_reference_full_eval_sample_count_met": True,
        "fast_reference_full_objective_accuracy_met": False,
        "fast_reference_full_objective_accuracy_gap": 0.010000000000000009,
        "fast_reference_full_objective_candidate": False,
    }
    assert hit["fast_reference_full_objective_candidate"] is True

    rows = [
        {
            "tag": "low",
            "final_eval_accuracy": 0.91,
            "eval_improvement": 0.7,
            "strict_phase_cost_projection_transient_budget_met": True,
            "strict_phase_cost_projection_source_pwl_budget_met": True,
            "strict_phase_cost_projection_output_vector_budget_met": True,
            "strict_phase_cost_projection_updates": 10000,
            "strict_phase_cost_projection_total_source_pwl_points": 1000,
            **hit,
        },
        {
            "tag": "high",
            "final_eval_accuracy": 0.92,
            "eval_improvement": 0.6,
            "strict_phase_cost_projection_transient_budget_met": True,
            "strict_phase_cost_projection_source_pwl_budget_met": False,
            "strict_phase_cost_projection_output_vector_budget_met": True,
            "strict_phase_cost_projection_updates": 10000,
            "strict_phase_cost_projection_total_source_pwl_points": 900,
            **hit,
        },
        {
            "tag": "small_eval",
            "final_eval_accuracy": 0.99,
            "eval_improvement": 0.8,
            "strict_phase_cost_projection_transient_budget_met": True,
            "strict_phase_cost_projection_source_pwl_budget_met": True,
            "strict_phase_cost_projection_output_vector_budget_met": True,
            "strict_phase_cost_projection_updates": 10000,
            "strict_phase_cost_projection_total_source_pwl_points": 100,
            **miss_eval,
        },
    ]

    assert fast_sweep.best_fast_reference_full_objective_variant(rows)["tag"] == "high"
    assert fast_sweep.best_fast_reference_full_objective_cost_feasible_variant(rows)["tag"] == "low"
    assert fast_sweep.cost_projection_summary_fields(rows, cost_projection_updates=10000) == {
        "cost_projection_enabled": True,
        "cost_projection_updates": 10000,
        "cost_projection_rows": 3,
        "cost_projection_budget_feasible_rows": 2,
        "fast_reference_full_objective_candidate_count": 2,
        "fast_reference_full_objective_cost_feasible_candidate_count": 1,
        "fast_reference_full_objective_cost_infeasible_candidate_count": 1,
    }
    assert fast_sweep.cost_projection_summary_fields([], cost_projection_updates=0) == {
        "cost_projection_enabled": False,
        "cost_projection_updates": 0,
        "cost_projection_rows": 0,
        "cost_projection_budget_feasible_rows": 0,
        "fast_reference_full_objective_candidate_count": 0,
        "fast_reference_full_objective_cost_feasible_candidate_count": 0,
        "fast_reference_full_objective_cost_infeasible_candidate_count": 0,
    }


def test_fast_online_promotion_efficiency_fields_normalize_gain_by_cost() -> None:
    fields = fast_sweep.promotion_efficiency_fields(
        initial_eval_accuracy=0.1,
        promotion_probe_eval_accuracy=0.6,
        promotion_updates=10,
        promotion_costs={
            "strict_phase_promotion_total_source_pwl_points": 2000,
            "strict_phase_promotion_estimated_transient_points": 500,
        },
    )

    assert fields["promotion_probe_eval_improvement"] == pytest.approx(0.5)
    assert fields["strict_phase_promotion_source_pwl_points_per_update"] == pytest.approx(200.0)
    assert fields["strict_phase_promotion_eval_improvement_per_1k_source_pwl"] == pytest.approx(0.25)
    assert fields["strict_phase_promotion_eval_improvement_per_1k_transient_points"] == pytest.approx(1.0)


def test_fast_online_promotion_efficiency_fields_handle_missing_probe_accuracy() -> None:
    fields = fast_sweep.promotion_efficiency_fields(
        initial_eval_accuracy=0.1,
        promotion_probe_eval_accuracy=None,
        promotion_updates=10,
        promotion_costs={
            "strict_phase_promotion_total_source_pwl_points": 2000,
            "strict_phase_promotion_estimated_transient_points": 500,
        },
    )

    assert fields["promotion_probe_eval_improvement"] is None
    assert fields["strict_phase_promotion_source_pwl_points_per_update"] == pytest.approx(200.0)
    assert fields["strict_phase_promotion_eval_improvement_per_1k_source_pwl"] is None
    assert fields["strict_phase_promotion_eval_improvement_per_1k_transient_points"] is None


def test_fast_online_variant_sweep_prefers_budget_feasible_promotion_variant() -> None:
    rows = [
        {
            "tag": "too_expensive",
            "final_eval_accuracy": 0.9,
            "promotion_probe_eval_accuracy": 0.8,
            "eval_improvement": 0.8,
            "strict_phase_promotion_transient_budget_met": True,
            "strict_phase_promotion_source_pwl_budget_met": False,
            "strict_phase_promotion_output_vector_budget_met": True,
        },
        {
            "tag": "too_many_vectors",
            "final_eval_accuracy": 0.85,
            "promotion_probe_eval_accuracy": 0.7,
            "eval_improvement": 0.7,
            "strict_phase_promotion_transient_budget_met": True,
            "strict_phase_promotion_source_pwl_budget_met": True,
            "strict_phase_promotion_output_vector_budget_met": False,
        },
        {
            "tag": "feasible",
            "final_eval_accuracy": 0.8,
            "promotion_probe_eval_accuracy": 0.6,
            "eval_improvement": 0.6,
            "strict_phase_promotion_transient_budget_met": True,
            "strict_phase_promotion_source_pwl_budget_met": True,
            "strict_phase_promotion_output_vector_budget_met": True,
        },
    ]

    assert fast_sweep.best_promotion_variant(rows)["tag"] == "feasible"


def test_fast_online_variant_sweep_keeps_best_when_no_promotion_is_budget_feasible() -> None:
    rows = [
        {
            "tag": "expensive_best",
            "final_eval_accuracy": 0.9,
            "promotion_probe_eval_accuracy": 0.8,
            "eval_improvement": 0.8,
            "strict_phase_promotion_transient_budget_met": False,
            "strict_phase_promotion_source_pwl_budget_met": True,
        },
        {
            "tag": "also_expensive",
            "final_eval_accuracy": 0.8,
            "promotion_probe_eval_accuracy": 0.6,
            "eval_improvement": 0.6,
            "strict_phase_promotion_transient_budget_met": True,
            "strict_phase_promotion_source_pwl_budget_met": False,
        },
    ]

    assert fast_sweep.best_promotion_variant(rows)["tag"] == "expensive_best"
