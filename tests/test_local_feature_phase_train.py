import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPICE_DIR = ROOT / "spice"
if str(SPICE_DIR) not in sys.path:
    sys.path.insert(0, str(SPICE_DIR))

import run_spice_mnist_local_feature_phase_train as phase_train  # noqa: E402
import run_spice_mnist_local_feature_phase_transient as phase_transient  # noqa: E402
import run_spice_mnist_local_feature_phase_variant_sweep as phase_variant_sweep  # noqa: E402
import run_spice_mnist_local_feature_batch_op_train as feature_batch_train  # noqa: E402
import run_spice_mnist_train as mnist_train  # noqa: E402


def test_cli_exposes_simulator_selector() -> None:
    proc = subprocess.run(
        [sys.executable, str(SPICE_DIR / "run_spice_mnist_local_feature_phase_train.py"), "--help"],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--simulator" in proc.stdout


def test_phase_transient_cli_exposes_agreement_gates() -> None:
    proc = subprocess.run(
        [sys.executable, str(SPICE_DIR / "run_spice_mnist_local_feature_phase_transient.py"), "--help"],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--simulator" in proc.stdout
    assert "--softmax-output" in proc.stdout
    assert "--direction-cosine-threshold" in proc.stdout
    assert "--sign-alignment-threshold" in proc.stdout
    assert "--eval-accuracy-diff-threshold" in proc.stdout
    assert "--random-accuracy-threshold" in proc.stdout
    assert "--learning-improvement-threshold" in proc.stdout
    assert "--probe-updates" in proc.stdout
    assert "--local-activation" in proc.stdout
    assert "--hidden-synapse-mode" in proc.stdout
    assert "--readout-synapse-mode" in proc.stdout
    assert "--reference-mode" in proc.stdout
    assert "--phase-output-mode" in proc.stdout
    assert "--update-mode" in proc.stdout
    assert "--strict-fully-on-device" in proc.stdout
    assert "--output-bias-update-scale" in proc.stdout
    assert "--readout-update-scale" in proc.stdout
    assert "--local-update-scale" in proc.stdout
    assert "--state-decay" in proc.stdout
    assert "--softmax-negative-scale" in proc.stdout
    assert "--softmax-error-centering" in proc.stdout
    assert "--softmax-temperature" in proc.stdout
    assert "--softmax-competition-mode" in proc.stdout
    assert "--softmax-competitor-power" in proc.stdout
    assert "--softmax-error-gate" in proc.stdout
    assert "--softmax-margin" in proc.stdout
    assert "--readout-class-centering" in proc.stdout
    assert "--eval-backend" in proc.stdout
    assert "--simulator-extra-args" in proc.stdout
    assert "--max-transient-points" in proc.stdout
    assert "--max-source-pwl-points" in proc.stdout
    assert "--preflight-only" in proc.stdout
    assert "--phase-clock-mode" in proc.stdout


def test_phase_transient_x_yce_print_reader_extracts_final_transient_row(tmp_path: Path) -> None:
    path = tmp_path / "deck.cir.prn"
    path.write_text(
        "Index       TIME          V(A)          V(B)\n"
        "0        0.0            1.00000000e+00 -1.00000000e+00\n"
        "1        1.0e-9         1.25000000e+00 -3.50000000e-01\n"
        "End of Xyce(TM) Simulation\n"
    )

    assert phase_transient.read_xyce_print_last_row(path, 2) == pytest.approx([1.25, -0.35])


def test_phase_transient_x_yce_print_rows_support_probe_time_lookup(tmp_path: Path) -> None:
    path = tmp_path / "deck.cir.prn"
    path.write_text(
        "Index       TIME          V(A)          V(B)\n"
        "0        1.0e-9         1.00000000e+00 -1.00000000e+00\n"
        "1        2.0e-9         1.25000000e+00 -3.50000000e-01\n"
        "End of Xyce(TM) Simulation\n"
    )

    rows = phase_transient.read_xyce_print_rows(path, 2)
    parsed = phase_transient.xyce_print_vectors_at_times(rows, {1: 1.0e-9, 2: 2.0e-9})

    assert rows[0][0] == pytest.approx(1.0e-9)
    assert parsed[1] == pytest.approx([1.0, -1.0])
    assert parsed[2] == pytest.approx([1.25, -0.35])


def test_phase_transient_estimates_transient_points_conservatively() -> None:
    assert phase_transient.estimate_transient_points(1.0e-9, 0.25e-9) == 5
    assert phase_transient.estimate_transient_points(1.01e-9, 0.25e-9) == 6

    with pytest.raises(ValueError, match="transient_step"):
        phase_transient.estimate_transient_points(1.0e-9, 0.0)


def test_phase_transient_point_budget_fails_before_large_runs() -> None:
    phase_transient.validate_transient_point_budget(101, 0)
    phase_transient.validate_transient_point_budget(100, 100)

    with pytest.raises(ValueError, match="estimated transient points"):
        phase_transient.validate_transient_point_budget(101, 100)


def test_phase_transient_source_point_budget_fails_before_large_decks() -> None:
    phase_transient.validate_source_point_budget({"sample_source_pwl_points": 101, "phase_clock_source_pwl_points": 50}, 0)
    phase_transient.validate_source_point_budget({"sample_source_pwl_points": 60, "phase_clock_source_pwl_points": 40}, 100)

    with pytest.raises(ValueError, match="estimated source PWL points"):
        phase_transient.validate_source_point_budget({"sample_source_pwl_points": 61, "phase_clock_source_pwl_points": 40}, 100)


def test_phase_transient_total_source_pwl_points_sums_samples_and_clocks() -> None:
    assert phase_transient.total_source_pwl_points({"sample_source_pwl_points": 61, "phase_clock_source_pwl_points": 40}) == 101


def test_phase_transient_final_measure_time_requires_post_update_slack() -> None:
    assert phase_transient.final_state_measure_time(10.0e-9, 2.0e-9, 7.0e-9) == pytest.approx(8.0e-9)

    with pytest.raises(ValueError, match="final-state measurement"):
        phase_transient.final_state_measure_time(10.0e-9, 3.0e-9, 7.0e-9)


def test_eval_accuracy_improved_requires_finite_strict_improvement() -> None:
    assert phase_train.eval_accuracy_improved(0.9, None)
    assert phase_train.eval_accuracy_improved(0.91, 0.9)
    assert not phase_train.eval_accuracy_improved(0.9, 0.9)
    assert not phase_train.eval_accuracy_improved(0.89, 0.9)
    assert not phase_train.eval_accuracy_improved(float("nan"), 0.9)


def test_maybe_save_best_eval_checkpoint_writes_only_on_enabled_improvement(tmp_path: Path) -> None:
    path = tmp_path / "best_weights.npz"
    state = (
        np.arange(8, dtype=float).reshape(1, 2, 4),
        np.array([[0.1, 0.2]]),
        np.arange(20, dtype=float).reshape(10, 1, 2),
        np.linspace(-0.5, 0.5, 10),
    )

    best, fields = phase_train.maybe_save_best_eval_checkpoint(False, path, state, 0.9, None)
    assert best is None
    assert fields == {}
    assert not path.exists()

    best, fields = phase_train.maybe_save_best_eval_checkpoint(True, path, state, 0.9, None)
    assert best == pytest.approx(0.9)
    assert fields == {"best_eval_checkpoint": str(path)}
    with np.load(path) as checkpoint:
        assert checkpoint["local_weights"] == pytest.approx(state[0])
        assert checkpoint["local_bias"] == pytest.approx(state[1])
        assert checkpoint["readout"] == pytest.approx(state[2])
        assert checkpoint["output_bias"] == pytest.approx(state[3])

    path.unlink()
    best, fields = phase_train.maybe_save_best_eval_checkpoint(True, path, state, 0.89, best)
    assert best == pytest.approx(0.9)
    assert fields == {}
    assert not path.exists()


def test_phase_transient_softmax_targets_are_zero_one() -> None:
    labels = np.array([2, 0])

    softmax_targets = phase_transient.target_matrix(labels, 3, softmax_output=True)
    tanh_targets = phase_transient.target_matrix(labels, 3, softmax_output=False)

    assert softmax_targets.tolist() == [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    assert tanh_targets.tolist() == [[-1.0, -1.0, 1.0], [1.0, -1.0, -1.0]]


def test_phase_pulse_area_includes_ramp_edges() -> None:
    assert phase_transient.phase_pulse_area(1.0e-9, 10.0e-12) == pytest.approx(1.01e-9)
    assert phase_transient.phase_pulse_area(1.0e-9, 0.0) == pytest.approx(1.0e-9)
    with pytest.raises(ValueError, match="edge"):
        phase_transient.phase_pulse_area(1.0e-9, -1.0e-12)


def test_phase_transient_sample_source_pwl_skips_unchanged_values() -> None:
    source = phase_transient.sample_source_pwl(
        np.array([0.0, 0.0, 1.0, 1.0, 0.0]),
        [1.0e-9, 2.0e-9, 3.0e-9, 4.0e-9, 5.0e-9],
        6.0e-9,
        0.1e-9,
    )

    assert source == "PWL(0 0 2.9e-09 0 3e-09 1 4.9e-09 1 5e-09 0 6e-09 0)"


def test_phase_transient_sample_source_pwl_uses_dc_for_constant_values() -> None:
    source = phase_transient.sample_source_pwl(
        np.array([-0.125, -0.125, -0.125]),
        [1.0e-9, 2.0e-9, 3.0e-9],
        4.0e-9,
        0.1e-9,
    )

    assert source == "-0.125"


def test_phase_transient_sample_source_pwl_rejects_mismatched_schedule() -> None:
    with pytest.raises(ValueError, match="sample_starts"):
        phase_transient.sample_source_pwl(np.array([1.0, 2.0]), [1.0e-9], 2.0e-9, 0.1e-9)


def test_phase_transient_source_complexity_counts_sample_and_clock_sources() -> None:
    x = np.array(
        [
            [0.0, 0.25],
            [0.0, 0.50],
            [0.0, 0.50],
        ],
        dtype=float,
    )
    y = np.array([0, 1, 1])
    phases, sample_starts, t_stop = phase_transient.make_phase_schedule(1, 3, 1.0e-9, 0.1e-9, True)
    targets = phase_transient.target_matrix(y, 2, softmax_output=True)

    complexity = phase_transient.phase_source_complexity(x, targets, phases, sample_starts, t_stop, 0.1e-9, True)

    assert complexity["sample_source_count"] == 4
    assert complexity["sample_source_dc_count"] == 1
    assert complexity["sample_source_pwl_count"] == 3
    assert complexity["pixel_source_dc_count"] == 1
    assert complexity["target_source_dc_count"] == 0
    assert complexity["sample_source_pwl_points"] == 12
    assert complexity["phase_clock_source_count"] == 5
    assert complexity["phase_clock_source_pwl_count"] == 5
    assert complexity["phase_clock_source_pwl_points"] > 0
    assert complexity["total_source_pwl_points"] == complexity["sample_source_pwl_points"] + complexity["phase_clock_source_pwl_points"]


def test_phase_transient_label_target_source_mode_decodes_one_label_waveform() -> None:
    labels = np.array([0, 1, 1, 0])
    phases, sample_starts, t_stop = phase_transient.make_phase_schedule(1, 4, 1.0e-9, 0.1e-9, True)
    targets = phase_transient.target_matrix(labels, 2, softmax_output=True)

    lines = phase_transient.target_source_lines(
        labels,
        targets,
        sample_starts,
        t_stop,
        0.1e-9,
        target_source_mode="label",
        softmax_output=True,
    )
    complexity = phase_transient.phase_source_complexity(
        np.zeros((4, 1), dtype=float),
        targets,
        phases,
        sample_starts,
        t_stop,
        0.1e-9,
        True,
        labels=labels,
        target_source_mode="label",
    )

    assert lines[0].startswith("Vlabel label 0 PWL(")
    assert lines[1] == "Btarget0 target0 0 V = (0.5*(1+tanh((0.5-abs(V(label)-0))/{TARGET_LABEL_SMOOTH})))"
    assert lines[2] == "Btarget1 target1 0 V = (0.5*(1+tanh((0.5-abs(V(label)-1))/{TARGET_LABEL_SMOOTH})))"
    assert complexity["target_source_count"] == 1
    assert complexity["target_behavioral_source_count"] == 2
    assert complexity["target_source_pwl_points"] < sum(
        phase_transient.pwl_point_count(phase_transient.sample_source_pwl(targets[:, k], sample_starts, t_stop, 0.1e-9))
        for k in range(targets.shape[1])
    )


def test_phase_transient_analytic_phase_clock_expr_is_bounded_direct_clock() -> None:
    phases, _sample_starts, t_stop = phase_transient.make_phase_schedule(1, 2, 1.0e-9, 0.1e-9, True)

    expr = phase_transient.analytic_phase_clock_expr(phases["act"], 1.0e-9, 0.1e-9, 5.0e-12)
    line = phase_transient.phase_clock_source_line(
        "pact",
        "pact",
        phases["act"],
        t_stop,
        1.0e-9,
        0.1e-9,
        5.0e-12,
        "analytic",
        True,
        1,
    )

    assert "floor(" in expr
    assert "5.6e-09" in expr
    assert line.startswith("Bpact pact 0 V = if(")


def test_phase_transient_analytic_phase_clock_complexity_removes_clock_pwl_points() -> None:
    x = np.zeros((2, 2), dtype=float)
    y = np.array([0, 1])
    phases, sample_starts, t_stop = phase_transient.make_phase_schedule(1, 2, 1.0e-9, 0.1e-9, True)
    targets = phase_transient.target_matrix(y, 2, softmax_output=True)

    complexity = phase_transient.phase_source_complexity(
        x,
        targets,
        phases,
        sample_starts,
        t_stop,
        0.1e-9,
        True,
        phase_clock_mode="analytic",
    )

    assert complexity["phase_clock_source_count"] == 5
    assert complexity["phase_clock_source_pwl_count"] == 0
    assert complexity["phase_clock_source_pwl_points"] == 0
    assert complexity["total_source_pwl_points"] == complexity["sample_source_pwl_points"]


def test_phase_transient_lr_schedule_values_support_linear_decay() -> None:
    assert phase_transient.lr_schedule_values(0.8, 1, "linear-decay", 0.25).tolist() == pytest.approx([0.8])
    assert phase_transient.lr_schedule_values(0.8, 3, "linear-decay", 0.25).tolist() == pytest.approx([0.8, 0.5, 0.2])


def test_phase_transient_source_complexity_counts_lr_control_waveform() -> None:
    x = np.zeros((2, 2), dtype=float)
    y = np.array([0, 1])
    phases, sample_starts, t_stop = phase_transient.make_phase_schedule(1, 2, 1.0e-9, 0.1e-9, True)
    targets = phase_transient.target_matrix(y, 2, softmax_output=True)

    complexity = phase_transient.phase_source_complexity(
        x,
        targets,
        phases,
        sample_starts,
        t_stop,
        0.1e-9,
        True,
        phase_clock_mode="analytic",
        lr_values=np.array([0.8, 0.2]),
    )

    assert complexity["control_source_count"] == 1
    assert complexity["control_source_pwl_count"] == 1
    assert complexity["control_source_pwl_points"] > 0
    assert complexity["total_source_pwl_points"] == (
        complexity["sample_source_pwl_points"]
        + complexity["phase_clock_source_pwl_points"]
        + complexity["control_source_pwl_points"]
    )


def test_phase_transient_activation_exprs_cover_relu_families() -> None:
    assert phase_transient.local_activation_expr("x", "relu", 1.0) == "0.5*((x)+abs(x))"
    assert "0.5*(((x)-0.25)+abs((x)-0.25))" in phase_transient.local_activation_expr("x", "clipped-relu", 0.25)
    assert "0.01*0.5*((0-(x))" in phase_transient.local_activation_expr("x", "leaky-relu", 1.0, relu_leak=0.01)
    assert "log(1+exp(5*" in phase_transient.local_activation_expr("x", "softplus", 1.0, softplus_beta=5.0)
    assert "-(" not in phase_transient.local_activation_deriv_expr("x", "h0", "relu", 1.0)
    assert "V(h0)" in phase_transient.local_activation_deriv_expr("x", "h0", "tanh", 1.0)
    assert phase_transient.local_activation_deriv_expr("x", "h0", "relu", 1.0, "unity") == "1"
    assert "V(h0)" in phase_transient.local_activation_deriv_expr("x", "h0", "relu", 1.0, "stored-gate")
    assert "0.05+" in phase_transient.local_activation_deriv_expr("x", "h0", "relu", 1.0, "floor-exact", 0.05)


def test_phase_transient_update_direction_metrics_detect_alignment() -> None:
    initial = (np.array([0.0, 0.0]),)
    ref = (np.array([1.0, -2.0]),)
    aligned = (np.array([0.5, -1.0]),)
    wrong = (np.array([-0.5, -1.0]),)

    aligned_metrics = phase_transient.update_direction_metrics(initial, ref, aligned)
    wrong_metrics = phase_transient.update_direction_metrics(initial, ref, wrong)

    assert aligned_metrics["state_update_direction_cosine"] == pytest.approx(1.0)
    assert aligned_metrics["state_update_sign_alignment_fraction"] == pytest.approx(1.0)
    assert aligned_metrics["state_update_wrong_sign_count"] == pytest.approx(0.0)
    assert wrong_metrics["state_update_sign_alignment_fraction"] == pytest.approx(0.5)
    assert wrong_metrics["state_update_wrong_sign_count"] == pytest.approx(1.0)


def test_phase_transient_empty_reference_metrics_are_json_nulls() -> None:
    metrics = phase_transient.empty_reference_metrics()

    assert metrics["state_update_direction_cosine"] is None
    assert metrics["state_update_sign_alignment_fraction"] is None


def test_phase_transient_phase_only_update_metrics_keep_move_magnitude() -> None:
    initial = (np.array([0.0, 1.0]),)
    moved = (np.array([3.0, 5.0]),)

    metrics = phase_transient.phase_only_update_metrics(initial, moved)

    assert metrics["reference_update_l2"] is None
    assert metrics["phase_update_l2"] == pytest.approx(5.0)
    assert metrics["state_update_direction_cosine"] is None
    assert metrics["state_update_sign_alignment_fraction"] is None


def test_phase_transient_numpy_eval_accuracy_is_post_transient_diagnostic(tmp_path: Path) -> None:
    x = np.eye(4)
    y = np.array([0, 1, 0, 1])
    w = np.ones((1, 1, 4))
    hb = np.zeros((1, 1))
    readout = np.array([[[1.0]], [[-1.0]]])
    output_bias = np.array([0.0, 0.0])
    netlist_path = tmp_path / "unused_eval.cir"
    data_path = tmp_path / "unused_eval.dat"

    acc = phase_transient.diagnostic_eval_accuracy(
        "numpy",
        "unused-spice",
        netlist_path,
        data_path,
        x,
        y,
        (w, hb, readout, output_bias),
        [[0, 1, 2, 3]],
        2,
        1.0,
        linear_output=True,
        softmax_output=False,
        local_activation="tanh",
        relu_clip=1.0,
        relu_leak=0.01,
        softplus_beta=10.0,
        hidden_synapse_mode="linear",
        readout_synapse_mode="linear",
        synapse_clip=1.0,
    )

    assert acc == pytest.approx(0.5)
    assert not netlist_path.exists()
    assert not data_path.exists()

    with pytest.raises(ValueError, match="eval_backend"):
        phase_transient.diagnostic_eval_accuracy(
            "fast",
            "unused-spice",
            netlist_path,
            data_path,
            x,
            y,
            (w, hb, readout, output_bias),
            [[0, 1, 2, 3]],
            2,
            1.0,
            linear_output=True,
            softmax_output=False,
            local_activation="tanh",
            relu_clip=1.0,
            relu_leak=0.01,
            softplus_beta=10.0,
            hidden_synapse_mode="linear",
            readout_synapse_mode="linear",
            synapse_clip=1.0,
        )


def test_phase_transient_numpy_eval_diagnostics_reports_prediction_collapse() -> None:
    x = np.eye(4)
    y = np.array([0, 1, 0, 1])
    w = np.ones((1, 1, 4))
    hb = np.zeros((1, 1))
    readout = np.array([[[1.0]], [[-1.0]]])
    output_bias = np.array([0.0, 0.0])

    stats = phase_transient.numpy_eval_diagnostics(
        x,
        y,
        (w, hb, readout, output_bias),
        [[0, 1, 2, 3]],
        2,
        linear_output=True,
        softmax_output=False,
        local_activation="tanh",
        relu_clip=1.0,
        relu_leak=0.01,
        softplus_beta=10.0,
        hidden_synapse_mode="linear",
        readout_synapse_mode="linear",
        synapse_clip=1.0,
    )

    assert stats["accuracy"] == pytest.approx(0.5)
    assert stats["label_histogram"] == [2, 2]
    assert stats["prediction_histogram"] == [4, 0]
    assert stats["correct_by_label"] == [2, 0]
    assert stats["per_class_accuracy"] == [1.0, 0.0]
    assert stats["dominant_pred_class"] == 0
    assert stats["dominant_pred_fraction"] == pytest.approx(1.0)
    assert stats["unique_predicted_classes"] == 1


def test_phase_transient_both_eval_backend_reports_spice_and_numpy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    x = np.eye(4)
    y = np.array([0, 1, 0, 1])
    w = np.ones((1, 1, 4))
    hb = np.zeros((1, 1))
    readout = np.array([[[1.0]], [[-1.0]]])
    output_bias = np.array([0.0, 0.0])

    def fake_run_eval(*_args, **_kwargs):
        return 0.75

    monkeypatch.setattr(phase_transient, "run_eval", fake_run_eval)
    accuracies = phase_transient.diagnostic_eval_accuracies(
        "both",
        "unused-spice",
        tmp_path / "eval.cir",
        tmp_path / "eval.dat",
        x,
        y,
        (w, hb, readout, output_bias),
        [[0, 1, 2, 3]],
        2,
        1.0,
        linear_output=True,
        softmax_output=False,
        local_activation="tanh",
        relu_clip=1.0,
        relu_leak=0.01,
        softplus_beta=10.0,
        hidden_synapse_mode="linear",
        readout_synapse_mode="linear",
        synapse_clip=1.0,
    )

    assert accuracies["spice"] == pytest.approx(0.75)
    assert accuracies["numpy"] == pytest.approx(0.5)
    assert phase_transient.primary_eval_accuracy("both", accuracies) == pytest.approx(0.75)
    assert phase_transient.eval_backend_abs_diff(accuracies) == pytest.approx(0.25)


def test_phase_transient_selects_sparse_print_for_xyce_without_probes() -> None:
    assert phase_transient.select_phase_output_mode("auto", "Xyce", False, ()) == "print"
    assert phase_transient.select_phase_output_mode("auto", "Xyce", False, (1,)) == "print"
    assert phase_transient.select_phase_output_mode("auto", "Xyce", True, (1,)) == "measure"
    assert phase_transient.select_phase_output_mode("auto", "ngspice", True, ()) == "control_measure"
    assert phase_transient.select_phase_output_mode("auto", "ngspice", False, ()) == "wrdata"
    assert phase_transient.select_phase_output_mode("measure", "Xyce", False, ()) == "measure"
    assert phase_transient.select_phase_output_mode("print", "Xyce", False, (1,)) == "print"


def test_phase_transient_probe_update_parser_supports_ranges_and_powers() -> None:
    assert phase_transient.parse_probe_update_list("", 8) == ()
    assert phase_transient.parse_probe_update_list("1,3-4,final", 8) == (1, 3, 4, 8)
    assert phase_transient.parse_probe_update_list("powers2", 10) == (1, 2, 4, 8, 10)

    with pytest.raises(ValueError, match="probe updates"):
        phase_transient.parse_probe_update_list("0", 8)
    with pytest.raises(ValueError, match="descending"):
        phase_transient.parse_probe_update_list("5-3", 8)


def test_phase_transient_probe_measurement_parser_uses_update_order() -> None:
    text = "\n".join(
        [
            "p000_00000 = 1.0",
            "p000_00001 = 2.0",
            "p001_00000 = 3.0",
            "p001_00001 = 4.0",
        ]
    )

    parsed = phase_transient.parse_probe_measurements(text, (2, 5), 2)

    assert parsed[2] == pytest.approx([1.0, 2.0])
    assert parsed[5] == pytest.approx([3.0, 4.0])


def test_phase_transient_probe_rows_can_be_phase_only_without_reference() -> None:
    w = np.array([[[0.1, -0.2]]])
    hb = np.array([[0.03]])
    readout = np.array([[[0.4]], [[-0.5]]])
    output_bias = np.array([0.01, -0.02])
    probe_vals = {
        1: np.array([0.11, -0.19, 0.04, 0.42, -0.48, 0.03, -0.01, 0.8, 0.2]),
    }

    rows, phase_states = phase_transient.probe_diagnostic_rows(
        (1,),
        probe_vals,
        (w, hb, readout, output_bias),
        {},
    )

    assert rows[0]["update"] == 1
    assert rows[0]["state_update_direction_cosine"] is None
    assert rows[0]["state_max_abs_diff"] is None
    assert rows[0]["phase_update_l2"] > 0.0
    assert rows[0]["reference_update_l2"] is None
    np.testing.assert_allclose(phase_states[1][0], [[[0.11, -0.19]]])
    np.testing.assert_allclose(phase_states[1][3], [0.03, -0.01])


def test_phase_transient_probe_rows_compare_reference_when_available() -> None:
    w = np.array([[[0.0]]])
    hb = np.array([[0.0]])
    readout = np.array([[[0.0]], [[0.0]]])
    output_bias = np.array([0.0, 0.0])
    op_state = (
        np.array([[[1.0]]]),
        np.array([[0.5]]),
        np.array([[[0.25]], [[-0.25]]]),
        np.array([0.1, -0.1]),
    )
    probe_vals = {
        2: np.array([1.0, 0.5, 0.25, -0.25, 0.1, -0.1, 0.9, 0.1]),
    }

    rows, _phase_states = phase_transient.probe_diagnostic_rows(
        (2,),
        probe_vals,
        (w, hb, readout, output_bias),
        {2: op_state},
    )

    assert rows[0]["state_max_abs_diff"] == pytest.approx(0.0)
    assert rows[0]["state_update_direction_cosine"] == pytest.approx(1.0)
    assert rows[0]["state_update_sign_alignment_fraction"] == pytest.approx(1.0)


def test_phase_transient_probe_summary_reports_best_and_final_points() -> None:
    rows = [
        {
            "update": 1,
            "phase_update_l2": 0.2,
            "phase_eval_accuracy": 0.12,
            "phase_eval_improvement": 0.01,
        },
        {
            "update": 4,
            "phase_update_l2": 0.8,
            "phase_eval_accuracy": 0.25,
            "phase_eval_improvement": 0.14,
        },
        {
            "update": 2,
            "phase_update_l2": 0.5,
            "phase_eval_accuracy": 0.18,
            "phase_eval_improvement": 0.07,
        },
    ]

    summary = phase_transient.summarize_probe_rows(rows)

    assert summary["probe_count"] == 3
    assert summary["final_probe_update"] == 4
    assert summary["final_probe_phase_update_l2"] == pytest.approx(0.8)
    assert summary["max_probe_phase_update_l2"] == pytest.approx(0.8)
    assert summary["max_probe_phase_update_l2_update"] == 4
    assert summary["best_probe_phase_eval_accuracy"] == pytest.approx(0.25)
    assert summary["best_probe_phase_eval_update"] == 4
    assert summary["best_probe_phase_eval_improvement"] == pytest.approx(0.14)


def test_phase_transient_probe_summary_handles_missing_eval_fields() -> None:
    summary = phase_transient.summarize_probe_rows(
        [
            {"update": 2, "phase_update_l2": None},
            {"update": 1, "phase_update_l2": 0.3},
        ]
    )

    assert summary["probe_count"] == 2
    assert summary["final_probe_update"] == 2
    assert summary["final_probe_phase_update_l2"] is None
    assert summary["max_probe_phase_update_l2"] == pytest.approx(0.3)
    assert summary["max_probe_phase_update_l2_update"] == 1
    assert summary["best_probe_phase_eval_accuracy"] is None
    assert summary["best_probe_phase_eval_update"] is None


def test_phase_transient_empty_probe_summary_is_json_nulls() -> None:
    summary = phase_transient.summarize_probe_rows([])

    assert summary["probe_count"] == 0
    assert summary["final_probe_update"] is None
    assert summary["max_probe_phase_update_l2"] is None
    assert summary["best_probe_phase_eval_accuracy"] is None


def test_phase_transient_cleanup_simulator_sidecars_removes_only_known_sidecars(tmp_path: Path) -> None:
    netlist = tmp_path / "deck.cir"
    netlist.write_text("* keep\n")
    prn = tmp_path / "deck.cir.prn"
    mt0 = tmp_path / "deck.cir.mt0"
    unrelated = tmp_path / "deck.cir.log"
    prn.write_text("print data\n")
    mt0.write_text("measure data\n")
    unrelated.write_text("log data\n")

    cleaned = phase_transient.cleanup_simulator_sidecars([netlist, netlist])

    assert cleaned == 2
    assert netlist.exists()
    assert unrelated.exists()
    assert not prn.exists()
    assert not mt0.exists()


def test_mnist_index_splits_use_stable_train_and_test_prefixes() -> None:
    train_small, test_small = mnist_train.mnist_index_splits(5, 8, 100, 100, seed=7)
    train_large, test_large = mnist_train.mnist_index_splits(50, 20, 100, 100, seed=7)
    train_again, test_again = mnist_train.mnist_index_splits(5, 8, 100, 100, seed=7)

    assert train_small.tolist() == train_large[:5].tolist()
    assert test_small.tolist() == test_large[:8].tolist()
    assert test_small.tolist() == test_again.tolist()
    assert train_small.tolist() == train_again.tolist()

    with pytest.raises(ValueError, match="exceeds"):
        mnist_train.mnist_index_splits(101, 1, 100, 100, seed=7)


def test_phase_transient_index_prefix_metadata_is_stable_and_compact() -> None:
    indices = np.array([3, 1, 4, 1, 5], dtype=np.int64)

    metadata = phase_transient.index_prefix_metadata(indices, prefix_len=3)

    assert metadata["count"] == 5
    assert metadata["prefix"] == [3, 1, 4]
    assert len(metadata["sha256"]) == 64
    assert metadata == phase_transient.index_prefix_metadata(indices, prefix_len=3)
    assert metadata["sha256"] != phase_transient.index_prefix_metadata(indices[:4], prefix_len=3)["sha256"]


def test_phase_transient_label_sequence_metadata_reports_class_order_and_balance() -> None:
    labels = np.array([3, 1, 3, 2, 3], dtype=np.int64)

    metadata = phase_transient.label_sequence_metadata(labels, n_classes=5, prefix_len=4)

    assert metadata["count"] == 5
    assert metadata["prefix"] == [3, 1, 3, 2]
    assert metadata["histogram"] == [0, 1, 1, 3, 0]
    assert metadata["dominant_label"] == 3
    assert metadata["dominant_label_fraction"] == pytest.approx(0.6)
    assert metadata["unique_labels"] == 3
    assert len(metadata["sha256"]) == 64
    assert metadata == phase_transient.label_sequence_metadata(labels, n_classes=5, prefix_len=4)
    assert metadata["sha256"] != phase_transient.label_sequence_metadata(labels[::-1], n_classes=5, prefix_len=4)["sha256"]


def test_phase_variant_sweep_pairs_only_expand_clipped_activations() -> None:
    pairs = phase_variant_sweep.activation_clip_pairs(
        ["tanh", "relu", "diff-clipped-relu"],
        [0.5, 1.0],
    )

    assert pairs == [("tanh", 0.5), ("relu", 0.5), ("diff-clipped-relu", 0.5), ("diff-clipped-relu", 1.0)]


def test_phase_variant_sweep_dry_command_preserves_online_contract() -> None:
    args = argparse.Namespace(
        simulator="Xyce",
        train_samples=8,
        eval_samples=0,
        image_size=10,
        block_size=4,
        stride=2,
        channels=2,
        updates=8,
        lr=0.8,
        phase=1e-9,
        gap=0.1e-9,
        edge=10e-12,
        settle_ratio=80.0,
        transient_step=50e-12,
        timeout=600.0,
        max_transient_points=500,
        max_source_pwl_points=1200,
        reference_mode="none",
        phase_output_mode="print",
        update_mode="direct",
        phase_clock_mode="analytic",
        eval_backend="numpy",
        probe_updates="1,2,4,8",
        tag="sweep",
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
        hidden_synapse_modes="linear,tanh-clipped",
        readout_synapse_modes="linear,hard-clipped",
        synapse_clip=0.25,
        synapse_clips="0.25,1.0",
        readout_class_centering="none",
        softmax_output=True,
        linear_output=False,
        final_measures=False,
        eval_probe_updates=True,
        strict_fully_on_device=True,
        simulator_extra_args="",
    )

    command = phase_variant_sweep.build_variant_command(
        args,
        "diff-clipped-relu",
        0.5,
        "stored-gate",
        "clipped-readout",
        "tanh-clipped",
        "hard-clipped",
    )

    assert "--batch-size" in command
    assert command[command.index("--batch-size") + 1] == "1"
    assert "--local-activation" in command
    assert command[command.index("--local-activation") + 1] == "diff-clipped-relu"
    assert "--relu-clip" in command
    assert command[command.index("--relu-clip") + 1] == "0.5"
    assert "--activation-derivative" in command
    assert command[command.index("--activation-derivative") + 1] == "stored-gate"
    assert "--readout-feedback-mode" in command
    assert command[command.index("--readout-feedback-mode") + 1] == "clipped-readout"
    assert "--hidden-synapse-mode" in command
    assert command[command.index("--hidden-synapse-mode") + 1] == "tanh-clipped"
    assert "--readout-synapse-mode" in command
    assert command[command.index("--readout-synapse-mode") + 1] == "hard-clipped"
    assert "--synapse-clip" in command
    assert command[command.index("--synapse-clip") + 1] == "0.25"
    assert "--max-transient-points" in command
    assert command[command.index("--max-transient-points") + 1] == "500"
    assert "--max-source-pwl-points" in command
    assert command[command.index("--max-source-pwl-points") + 1] == "1200"
    assert "--reference-mode" in command
    assert command[command.index("--reference-mode") + 1] == "none"
    assert "--phase-output-mode" in command
    assert command[command.index("--phase-output-mode") + 1] == "print"
    assert "--update-mode" in command
    assert command[command.index("--update-mode") + 1] == "direct"
    assert "--phase-clock-mode" in command
    assert command[command.index("--phase-clock-mode") + 1] == "analytic"
    assert "--eval-backend" in command
    assert command[command.index("--eval-backend") + 1] == "numpy"
    assert "--output-bias-update-scale" in command
    assert command[command.index("--output-bias-update-scale") + 1] == "0.0"
    assert "--readout-update-scale" in command
    assert command[command.index("--readout-update-scale") + 1] == "0.25"
    assert "--softmax-temperature" in command
    assert command[command.index("--softmax-temperature") + 1] == "4.0"
    assert "--readout-class-centering" in command
    assert command[command.index("--readout-class-centering") + 1] == "none"
    assert "--strict-fully-on-device" in command
    assert "--eval-probe-updates" in command
    assert "synclip0_25" in command[command.index("--tag") + 1]

    wider_command = phase_variant_sweep.build_variant_command(
        args,
        "diff-clipped-relu",
        0.5,
        "stored-gate",
        "clipped-readout",
        "tanh-clipped",
        "hard-clipped",
        1.0,
    )

    assert wider_command[wider_command.index("--synapse-clip") + 1] == "1.0"
    assert "synclip1" in wider_command[wider_command.index("--tag") + 1]


def test_phase_transient_softmax_deck_is_one_continuous_online_run(tmp_path: Path) -> None:
    x = np.array(
        [
            [0.0, 0.2, 0.4, 0.6],
            [0.1, 0.3, 0.5, 0.7],
            [0.2, 0.4, 0.6, 0.8],
        ],
        dtype=float,
    )
    y = np.array([0, 1, 0])
    w = np.full((1, 1, 4), 0.01)
    hb = np.zeros((1, 1))
    readout = np.zeros((2, 1, 1))
    output_bias = np.zeros(2)

    netlist, _n_vec, _t_stop = phase_transient.make_phase_transient_netlist(
        x,
        y,
        w,
        hb,
        readout,
        output_bias,
        [[0, 1, 2, 3]],
        0.8,
        tmp_path / "out.dat",
        False,
        1,
        len(y),
        1e-9,
        0.1e-9,
        5e-12,
        40.0,
        20e-12,
        1e-12,
        1e-12,
        1e-12,
        1e18,
        True,
    )

    assert netlist.count("Cw0_0_0 w0_0_0 0 {CW}") == 1
    assert "Vw0_0_0" not in netlist
    assert "Vpix0 pix0 0 PWL(" in netlist
    assert "Vtarget0 target0 0 PWL(" in netlist
    assert "Bpre_h0_0 ah0_0 0 V =" in netlist
    assert "By0 y0 0 V = exp((V(score0))/{SOFTMAX_TEMPERATURE})/(exp((V(score0))/{SOFTMAX_TEMPERATURE}) + exp((V(score1))/{SOFTMAX_TEMPERATURE}))" in netlist
    assert "Bstore_d0 d0 0 I = V(perr)*{CSTATE}/{TAU}*(V(d0)-((V(target0))*(1-(V(y0))) - (1-(V(target0)))*{SOFTMAX_NEGATIVE_SCALE}*(V(y0))))" in netlist
    assert ".param SOFTMAX_NEGATIVE_SCALE=1" in netlist
    assert ".param TAREA=1.005e-09" in netlist
    assert "{CGRAD}/{TAREA}" in netlist
    assert "/({BS}*{TAREA})" in netlist
    assert "{CGRAD}/{TPHASE}" not in netlist
    assert netlist.count("Vpapply papply 0 PWL(") == 1
    assert netlist.count("Vpclear pclear 0 PWL(") == 1
    assert netlist.count(" 1 ") >= len(y)


def test_phase_transient_softmax_negative_scale_controls_non_target_error(tmp_path: Path) -> None:
    x = np.zeros((1, 4))
    y = np.array([0])
    w = np.zeros((1, 1, 4))
    hb = np.zeros((1, 1))
    readout = np.zeros((2, 1, 1))
    output_bias = np.zeros(2)

    netlist, _n_vec, _t_stop = phase_transient.make_phase_transient_netlist(
        x,
        y,
        w,
        hb,
        readout,
        output_bias,
        [[0, 1, 2, 3]],
        0.8,
        tmp_path / "out.dat",
        False,
        1,
        1,
        1e-9,
        0.1e-9,
        5e-12,
        40.0,
        20e-12,
        1e-12,
        1e-12,
        1e-12,
        1e18,
        True,
        "measure",
        softmax_negative_scale=0.25,
        softmax_temperature=2.0,
    )

    assert ".param SOFTMAX_NEGATIVE_SCALE=0.25" in netlist
    assert ".param SOFTMAX_TEMPERATURE=2" in netlist
    assert "By0 y0 0 V = exp((V(score0))/{SOFTMAX_TEMPERATURE})/(exp((V(score0))/{SOFTMAX_TEMPERATURE}) + exp((V(score1))/{SOFTMAX_TEMPERATURE}))" in netlist
    assert "(1-(V(target1)))*{SOFTMAX_NEGATIVE_SCALE}*(V(y1))" in netlist

    centered_netlist, _n_vec, _t_stop = phase_transient.make_phase_transient_netlist(
        x,
        y,
        w,
        hb,
        readout,
        output_bias,
        [[0, 1, 2, 3]],
        0.8,
        tmp_path / "centered.dat",
        False,
        1,
        1,
        1e-9,
        0.1e-9,
        5e-12,
        40.0,
        20e-12,
        1e-12,
        1e-12,
        1e-12,
        1e18,
        True,
        "measure",
        softmax_negative_scale=0.25,
        softmax_error_centering="mean",
    )

    assert "Bstore_d1 d1 0 I = V(perr)*{CSTATE}/{TAU}*(V(d1)-(" in centered_netlist
    assert "((V(target1))*(1-(V(y1))) - (1-(V(target1)))*{SOFTMAX_NEGATIVE_SCALE}*(V(y1)))" in centered_netlist
    assert ")/2)))" in centered_netlist

    with pytest.raises(ValueError, match="softmax_negative_scale"):
        phase_transient.make_phase_transient_netlist(
            x,
            y,
            w,
            hb,
            readout,
            output_bias,
            [[0, 1, 2, 3]],
            0.8,
            tmp_path / "bad.dat",
            False,
            1,
            1,
            1e-9,
            0.1e-9,
            5e-12,
            40.0,
            20e-12,
            1e-12,
            1e-12,
            1e-12,
            1e18,
            True,
            "measure",
            softmax_negative_scale=-1.0,
        )

    with pytest.raises(ValueError, match="softmax_error_centering"):
        phase_transient.make_phase_transient_netlist(
            x,
            y,
            w,
            hb,
            readout,
            output_bias,
            [[0, 1, 2, 3]],
            0.8,
            tmp_path / "bad_centering.dat",
            False,
            1,
            1,
            1e-9,
            0.1e-9,
            5e-12,
            40.0,
            20e-12,
            1e-12,
            1e-12,
            1e-12,
            1e18,
            True,
            "measure",
            softmax_error_centering="median",
        )

    with pytest.raises(ValueError, match="softmax_temperature"):
        phase_transient.make_phase_transient_netlist(
            x,
            y,
            w,
            hb,
            readout,
            output_bias,
            [[0, 1, 2, 3]],
            0.8,
            tmp_path / "bad_temperature.dat",
            False,
            1,
            1,
            1e-9,
            0.1e-9,
            5e-12,
            40.0,
            20e-12,
            1e-12,
            1e-12,
            1e-12,
            1e18,
            True,
            "measure",
            softmax_temperature=0.0,
        )


def test_batch_op_softmax_negative_scale_matches_phase_error_expr(tmp_path: Path) -> None:
    x = np.zeros((1, 4))
    y = np.array([0])
    w = np.zeros((1, 1, 4))
    hb = np.zeros((1, 1))
    readout = np.zeros((2, 1, 1))
    output_bias = np.zeros(2)

    netlist = feature_batch_train.make_train_netlist(
        x,
        y,
        w,
        hb,
        readout,
        output_bias,
        [[0, 1, 2, 3]],
        0.8,
        tmp_path / "out.dat",
        False,
        True,
        "tanh",
        1.0,
        softmax_negative_scale=0.25,
        softmax_temperature=2.0,
    )

    assert ".param SOFTMAX_NEGATIVE_SCALE=0.25" in netlist
    assert ".param SOFTMAX_TEMPERATURE=2" in netlist
    assert "By0_0 y0_0 0 V = exp((V(z0_0))/{SOFTMAX_TEMPERATURE})/(exp((V(z0_0))/{SOFTMAX_TEMPERATURE}) + exp((V(z0_1))/{SOFTMAX_TEMPERATURE}))" in netlist
    assert "Be0_1 e0_1 0 V = (V(t0_1))*(1-(V(y0_1))) - (1-(V(t0_1)))*{SOFTMAX_NEGATIVE_SCALE}*(V(y0_1))" in netlist

    centered_netlist = feature_batch_train.make_train_netlist(
        x,
        y,
        w,
        hb,
        readout,
        output_bias,
        [[0, 1, 2, 3]],
        0.8,
        tmp_path / "centered.dat",
        False,
        True,
        "tanh",
        1.0,
        softmax_negative_scale=0.25,
        softmax_error_centering="mean",
    )

    assert "Be0_1 e0_1 0 V = ((V(t0_1))*(1-(V(y0_1)))" in centered_netlist
    assert ")/2)" in centered_netlist

    focused_netlist = feature_batch_train.make_train_netlist(
        x,
        y,
        w,
        hb,
        readout,
        output_bias,
        [[0, 1, 2, 3]],
        0.8,
        tmp_path / "focused.dat",
        False,
        True,
        "tanh",
        1.0,
        softmax_competition_mode="normalized-power",
        softmax_competitor_power=2,
    )

    assert ".param SOFTMAX_COMPETITOR_POWER=2" in focused_netlist
    assert "(V(t0_0))*(1-(V(y0_0))) + (V(t0_1))*(1-(V(y0_1)))" in focused_netlist
    assert "(1-(V(t0_1)))*{SOFTMAX_NEGATIVE_SCALE}" in focused_netlist
    assert "*((V(y0_1))*(V(y0_1)))/" in focused_netlist
    assert "((1-(V(t0_0)))*((V(y0_0))*(V(y0_0))) + (1-(V(t0_1)))*((V(y0_1))*(V(y0_1))))+1e-12" in focused_netlist

    with pytest.raises(ValueError, match="softmax_negative_scale"):
        feature_batch_train.make_train_netlist(
            x,
            y,
            w,
            hb,
            readout,
            output_bias,
            [[0, 1, 2, 3]],
            0.8,
            tmp_path / "bad.dat",
            False,
            True,
            "tanh",
            1.0,
            softmax_negative_scale=-1.0,
        )

    with pytest.raises(ValueError, match="softmax_temperature"):
        feature_batch_train.make_train_netlist(
            x,
            y,
            w,
            hb,
            readout,
            output_bias,
            [[0, 1, 2, 3]],
            0.8,
            tmp_path / "bad_temperature.dat",
            False,
            True,
            "tanh",
            1.0,
            softmax_temperature=0.0,
        )

    with pytest.raises(ValueError, match="softmax_competition_mode"):
        feature_batch_train.make_train_netlist(
            x,
            y,
            w,
            hb,
            readout,
            output_bias,
            [[0, 1, 2, 3]],
            0.8,
            tmp_path / "bad_competition.dat",
            False,
            True,
            "tanh",
            1.0,
            softmax_competition_mode="bad",
        )

    with pytest.raises(ValueError, match="softmax_competitor_power"):
        phase_transient.make_phase_transient_netlist(
            x,
            y,
            w,
            hb,
            readout,
            output_bias,
            [[0, 1, 2, 3]],
            0.8,
            tmp_path / "bad_competition_power.dat",
            False,
            1,
            1,
            1e-9,
            0.1e-9,
            5e-12,
            40.0,
            20e-12,
            1e-12,
            1e-12,
            1e-12,
            1e18,
            True,
            "measure",
            softmax_competitor_power=0,
        )


def test_phase_and_batch_softmax_margin_gate_uses_target_competitor_margin(tmp_path: Path) -> None:
    x = np.zeros((1, 4))
    y = np.array([0])
    w = np.zeros((1, 1, 4))
    hb = np.zeros((1, 1))
    readout = np.zeros((3, 1, 1))
    output_bias = np.zeros(3)

    phase_netlist, _n_vec, _t_stop = phase_transient.make_phase_transient_netlist(
        x,
        y,
        w,
        hb,
        readout,
        output_bias,
        [[0, 1, 2, 3]],
        0.8,
        tmp_path / "phase.dat",
        False,
        1,
        1,
        1e-9,
        0.1e-9,
        5e-12,
        40.0,
        20e-12,
        1e-12,
        1e-12,
        1e-12,
        1e18,
        True,
        "measure",
        softmax_error_gate="target-margin",
        softmax_margin=0.5,
    )

    assert ".param SOFTMAX_MARGIN=0.5" in phase_netlist
    assert "Bgerr gerr 0 V =" in phase_netlist
    assert "V(target0)*V(score0)" in phase_netlist
    assert "Bgerrcmp0_0 gerrcmp0_0 0 V =" in phase_netlist
    assert "0.5*(V(score1)+V(score2)+abs(V(score1)-V(score2)))" in phase_netlist
    assert "V(target0)*(V(gerrcmp0_0))" in phase_netlist
    assert "V(gerr)*((V(target0))*(1-(V(y0)))" in phase_netlist

    batch_netlist = feature_batch_train.make_train_netlist(
        x,
        y,
        w,
        hb,
        readout,
        output_bias,
        [[0, 1, 2, 3]],
        0.8,
        tmp_path / "batch.dat",
        False,
        True,
        "tanh",
        1.0,
        softmax_error_gate="target-margin",
        softmax_margin=0.5,
    )

    assert ".param SOFTMAX_MARGIN=0.5" in batch_netlist
    assert "Bgerr0 gerr0 0 V =" in batch_netlist
    assert "Bgerr0cmp0_0 gerr0cmp0_0 0 V =" in batch_netlist
    assert "V(t0_0)*V(z0_0)" in batch_netlist
    assert "V(gerr0)*((V(t0_0))*(1-(V(y0_0)))" in batch_netlist

    with pytest.raises(ValueError, match="softmax_error_gate"):
        phase_transient.make_phase_transient_netlist(
            x,
            y,
            w,
            hb,
            readout,
            output_bias,
            [[0, 1, 2, 3]],
            0.8,
            tmp_path / "bad_gate.dat",
            False,
            1,
            1,
            1e-9,
            0.1e-9,
            5e-12,
            40.0,
            20e-12,
            1e-12,
            1e-12,
            1e-12,
            1e18,
            True,
            "measure",
            softmax_error_gate="bad",
        )

    with pytest.raises(ValueError, match="softmax_margin"):
        feature_batch_train.make_train_netlist(
            x,
            y,
            w,
            hb,
            readout,
            output_bias,
            [[0, 1, 2, 3]],
            0.8,
            tmp_path / "bad_margin.dat",
            False,
            True,
            "tanh",
            1.0,
            softmax_error_gate="target-margin",
            softmax_margin=0.0,
        )

    ten_class_readout = np.zeros((10, 1, 1))
    ten_class_ob = np.zeros(10)
    ten_class_netlist, _n_vec, _t_stop = phase_transient.make_phase_transient_netlist(
        x,
        y,
        w,
        hb,
        ten_class_readout,
        ten_class_ob,
        [[0, 1, 2, 3]],
        0.8,
        tmp_path / "ten_class.dat",
        False,
        1,
        1,
        1e-9,
        0.1e-9,
        5e-12,
        40.0,
        20e-12,
        1e-12,
        1e-12,
        1e-12,
        1e18,
        True,
        "measure",
        softmax_error_gate="target-margin",
    )
    margin_gate_lines = [
        line
        for line in ten_class_netlist.splitlines()
        if line.startswith("Bgerr") or line.startswith("Bstore_d")
    ]
    assert max(len(line) for line in margin_gate_lines) < 5000


def test_batch_op_state_decay_matches_phase_update_shape(tmp_path: Path) -> None:
    x = np.zeros((1, 4))
    y = np.array([0])
    w = np.zeros((1, 1, 4))
    hb = np.zeros((1, 1))
    readout = np.zeros((2, 1, 1))
    output_bias = np.zeros(2)

    netlist = feature_batch_train.make_train_netlist(
        x,
        y,
        w,
        hb,
        readout,
        output_bias,
        [[0, 1, 2, 3]],
        0.8,
        tmp_path / "out.dat",
        False,
        True,
        "tanh",
        1.0,
        state_decay=0.05,
    )

    assert ".param STATE_DECAY=0.05" in netlist
    assert "Bnw0_0_0 nw0_0_0 0 V = V(w0_0_0)*(1-{STATE_DECAY}) + {LR}*((V(dh0_0_0)*V(x0_0))/{BS})" in netlist
    assert "Bnhb0_0 nhb0_0 0 V = V(hb0_0)*(1-{STATE_DECAY}) + {LR}*((V(dh0_0_0))/{BS})" in netlist
    assert "Bnv0_0_0 nv0_0_0 0 V = V(v0_0_0)*(1-{STATE_DECAY}) + {LR}*((V(d0_0)*V(h0_0_0))/{BS})" in netlist
    assert "Bnob0 nob0 0 V = V(ob0)*(1-{STATE_DECAY}) + {LR}*((V(d0_0))/{BS})" in netlist

    with pytest.raises(ValueError, match="state_decay"):
        feature_batch_train.make_train_netlist(
            x,
            y,
            w,
            hb,
            readout,
            output_bias,
            [[0, 1, 2, 3]],
            0.8,
            tmp_path / "bad.dat",
            False,
            True,
            "tanh",
            1.0,
            state_decay=-0.1,
        )


def test_phase_and_batch_readout_class_centering_use_effective_centered_readout(tmp_path: Path) -> None:
    x = np.zeros((1, 4))
    y = np.array([0])
    w = np.zeros((1, 1, 4))
    hb = np.zeros((1, 1))
    readout = np.zeros((2, 1, 1))
    output_bias = np.zeros(2)

    phase_netlist, _n_vec, _t_stop = phase_transient.make_phase_transient_netlist(
        x,
        y,
        w,
        hb,
        readout,
        output_bias,
        [[0, 1, 2, 3]],
        0.8,
        tmp_path / "phase.dat",
        False,
        1,
        1,
        1e-9,
        0.1e-9,
        5e-12,
        40.0,
        20e-12,
        1e-12,
        1e-12,
        1e-12,
        1e18,
        True,
        "measure",
        readout_class_centering="mean",
    )

    assert "((V(v0_0_0)) - (((V(v0_0_0)) + (V(v1_0_0)))/2))*V(h0_0)" in phase_netlist
    assert "((V(v1_0_0)) - (((V(v0_0_0)) + (V(v1_0_0)))/2))*V(d1)" in phase_netlist

    batch_netlist = feature_batch_train.make_train_netlist(
        x,
        y,
        w,
        hb,
        readout,
        output_bias,
        [[0, 1, 2, 3]],
        0.8,
        tmp_path / "batch.dat",
        False,
        True,
        "tanh",
        1.0,
        readout_class_centering="mean",
    )

    assert "((V(v0_0_0)) - (((V(v0_0_0)) + (V(v1_0_0)))/2))*V(h0_0_0)" in batch_netlist

    with pytest.raises(ValueError, match="readout_class_centering"):
        phase_transient.apply_readout_class_centering_np(readout, "median")


def test_phase_transient_synapse_transfer_modes_affect_forward_and_backward_paths(tmp_path: Path) -> None:
    x = np.zeros((1, 4))
    y = np.array([0])
    w = np.zeros((1, 1, 4))
    hb = np.zeros((1, 1))
    readout = np.zeros((2, 1, 1))
    output_bias = np.zeros(2)

    netlist, _n_vec, _t_stop = phase_transient.make_phase_transient_netlist(
        x,
        y,
        w,
        hb,
        readout,
        output_bias,
        [[0, 1, 2, 3]],
        0.8,
        tmp_path / "out.dat",
        False,
        1,
        1,
        1e-9,
        0.1e-9,
        5e-12,
        40.0,
        20e-12,
        1e-12,
        1e-12,
        1e-12,
        1e18,
        True,
        hidden_synapse_mode="tanh-clipped",
        readout_synapse_mode="sign",
        synapse_clip=0.25,
    )

    assert "(0.25*tanh((V(w0_0_0))/0.25))*V(pix0)" in netlist
    assert "(0.25*(V(v0_0_0))/(abs(V(v0_0_0))+1e-9))*V(h0_0)" in netlist
    assert "(0.25*(V(v0_0_0))/(abs(V(v0_0_0))+1e-9))*V(d0)" in netlist


def test_phase_transient_zero_rleak_omits_state_leak_resistors(tmp_path: Path) -> None:
    x = np.zeros((1, 4))
    y = np.array([0])
    w = np.zeros((1, 1, 4))
    hb = np.zeros((1, 1))
    readout = np.zeros((2, 1, 1))
    output_bias = np.zeros(2)

    netlist, _n_vec, _t_stop = phase_transient.make_phase_transient_netlist(
        x,
        y,
        w,
        hb,
        readout,
        output_bias,
        [[0, 1, 2, 3]],
        0.8,
        tmp_path / "out.dat",
        False,
        1,
        1,
        1e-9,
        0.1e-9,
        5e-12,
        40.0,
        20e-12,
        1e-12,
        1e-12,
        1e-12,
        0.0,
        True,
    )

    assert "Rw0_0_0" not in netlist
    assert "Rhb0_0" not in netlist
    assert "Rv0_0_0" not in netlist
    assert "Rob0" not in netlist


def test_phase_transient_direct_update_mode_omits_gradient_accumulator_family(tmp_path: Path) -> None:
    x = np.zeros((1, 4))
    y = np.array([0])
    w = np.zeros((1, 1, 4))
    hb = np.zeros((1, 1))
    readout = np.zeros((2, 1, 1))
    output_bias = np.zeros(2)

    netlist, _n_vec, _t_stop = phase_transient.make_phase_transient_netlist(
        x,
        y,
        w,
        hb,
        readout,
        output_bias,
        [[0, 1, 2, 3]],
        0.8,
        tmp_path / "out.dat",
        False,
        1,
        1,
        1e-9,
        0.1e-9,
        5e-12,
        40.0,
        20e-12,
        1e-12,
        1e-12,
        1e-12,
        1e18,
        True,
        update_mode="direct",
    )

    assert "Vpapply" not in netlist
    assert "Vpclear" not in netlist
    assert "Cgw0_0_0" not in netlist
    assert "Cghb0_0" not in netlist
    assert "Cgv0_0_0" not in netlist
    assert "Cgob0" not in netlist
    assert "Bacc_w0_0_0" not in netlist
    assert "Bclear_gw0_0_0" not in netlist
    assert "gradient accumulators are capacitor voltages" not in netlist
    assert "weights are updated directly during each per-sample update phase" in netlist
    assert "Bupd_w0_0_0 w0_0_0 0 I = -V(pacc)*{CW}*{LR}*{LOCAL_UPDATE_SCALE}/({BS}*{TAREA})*(V(dh0_0)*V(pix0))" in netlist

    with pytest.raises(ValueError, match="direct update mode"):
        phase_transient.make_phase_transient_netlist(
            np.zeros((2, 4)),
            np.array([0, 1]),
            w,
            hb,
            readout,
            output_bias,
            [[0, 1, 2, 3]],
            0.8,
            tmp_path / "out.dat",
            False,
            2,
            1,
            1e-9,
            0.1e-9,
            5e-12,
            40.0,
            20e-12,
            1e-12,
            1e-12,
            1e-12,
            1e18,
            True,
            update_mode="direct",
        )


def test_phase_transient_linear_decay_lr_schedule_uses_control_waveform(tmp_path: Path) -> None:
    x = np.zeros((2, 4))
    y = np.array([0, 1])
    w = np.zeros((1, 1, 4))
    hb = np.zeros((1, 1))
    readout = np.zeros((2, 1, 1))
    output_bias = np.zeros(2)

    netlist, _n_vec, _t_stop = phase_transient.make_phase_transient_netlist(
        x,
        y,
        w,
        hb,
        readout,
        output_bias,
        [[0, 1, 2, 3]],
        0.8,
        tmp_path / "out.dat",
        False,
        1,
        2,
        1e-9,
        0.1e-9,
        5e-12,
        40.0,
        20e-12,
        1e-12,
        1e-12,
        1e-12,
        1e18,
        True,
        update_mode="direct",
        lr_schedule="linear-decay",
        lr_final_scale=0.25,
    )

    assert "Vlrctrl lrctrl 0 PWL(" in netlist
    assert "0.8" in netlist
    assert "0.2" in netlist
    assert "Bupd_w0_0_0 w0_0_0 0 I = -V(pacc)*{CW}*V(lrctrl)*{LOCAL_UPDATE_SCALE}/({BS}*{TAREA})*(V(dh0_0)*V(pix0))" in netlist


def test_phase_transient_state_descriptions_follow_update_mode() -> None:
    phased = phase_transient.phase_state_descriptions("phased")
    direct = phase_transient.phase_state_descriptions("direct")

    assert "gradient accumulators" in phased["temporary_state"]
    assert "gradient accumulators" not in direct["temporary_state"]
    assert "updated directly during each per-sample update phase" in direct["temporary_state"]
    assert "checkpoint" not in direct["persistent_state"].lower()
    frozen_ob = phase_transient.phase_state_descriptions("direct", output_bias_state_frozen=True)
    assert "output biases are frozen constants" in frozen_ob["persistent_state"]
    assert "output biases are persistent capacitor" not in frozen_ob["persistent_state"]

    with pytest.raises(ValueError, match="update_mode"):
        phase_transient.phase_state_descriptions("online")


def test_phase_transient_execution_contract_is_separate_from_reference_replay() -> None:
    no_reference = phase_transient.phase_execution_contract_fields(1, "none")
    with_reference = phase_transient.phase_execution_contract_fields(1, "spice")
    batched = phase_transient.phase_execution_contract_fields(2, "none")
    checkpoint = phase_transient.phase_execution_contract_fields(1, "none", "weights.npz")
    strict_requested = phase_transient.phase_execution_contract_fields(1, "none", "", True)

    assert no_reference["fully_on_device_execution_contract_met"] is True
    assert no_reference["strict_fully_on_device_contract_met"] is True
    assert no_reference["strict_fully_on_device_requested"] is False
    assert no_reference["random_init_used"] is True
    assert no_reference["initial_weights_source"] == "random_init"
    assert no_reference["single_phase_training_transient"] is True
    assert no_reference["weights_persist_inside_phase_transient"] is True
    assert no_reference["python_weight_updates_between_samples"] is False
    assert no_reference["python_checkpointing_between_samples"] is False
    assert no_reference["reference_replay_used_for_diagnostics"] is False
    assert with_reference["fully_on_device_execution_contract_met"] is True
    assert with_reference["strict_fully_on_device_contract_met"] is False
    assert with_reference["reference_replay_used_for_diagnostics"] is True
    assert batched["fully_on_device_execution_contract_met"] is False
    assert batched["strict_fully_on_device_contract_met"] is False
    assert checkpoint["fully_on_device_execution_contract_met"] is True
    assert checkpoint["strict_fully_on_device_contract_met"] is False
    assert checkpoint["random_init_used"] is False
    assert checkpoint["initial_weights_source"] == "checkpoint"
    assert strict_requested["strict_fully_on_device_requested"] is True

    with pytest.raises(ValueError, match="reference_mode"):
        phase_transient.phase_execution_contract_fields(1, "fast")


def test_phase_transient_strict_fully_on_device_validation() -> None:
    phase_transient.validate_strict_fully_on_device_args(1, "none", "")

    with pytest.raises(ValueError, match="batch-size 1"):
        phase_transient.validate_strict_fully_on_device_args(2, "none", "")
    with pytest.raises(ValueError, match="reference-mode none"):
        phase_transient.validate_strict_fully_on_device_args(1, "spice", "")
    with pytest.raises(ValueError, match="random init"):
        phase_transient.validate_strict_fully_on_device_args(1, "none", "weights.npz")


def test_phase_transient_preflight_summary_has_no_artifact_paths() -> None:
    labels = np.array([2, 0])
    source_complexity = {
        "sample_source_count": 26,
        "sample_source_dc_count": 11,
        "sample_source_pwl_count": 15,
        "sample_source_pwl_points": 60,
        "pixel_source_count": 16,
        "pixel_source_dc_count": 3,
        "pixel_source_pwl_count": 13,
        "pixel_source_pwl_points": 52,
        "target_source_count": 10,
        "target_source_dc_count": 8,
        "target_source_pwl_count": 2,
        "target_source_pwl_points": 8,
        "target_behavioral_source_count": 0,
        "target_source_mode_label": 0,
        "phase_clock_source_count": 5,
        "phase_clock_source_pwl_count": 5,
        "phase_clock_source_pwl_points": 50,
        "total_source_pwl_points": 110,
    }

    summary = phase_transient.phase_preflight_summary(
        simulator_selector="Xyce",
        image_size=4,
        block_size=2,
        stride=2,
        blocks=4,
        channels=1,
        train_samples=2,
        eval_samples=0,
        batch_size=1,
        updates=2,
        total_samples=2,
        train_indices=np.array([5, 7]),
        eval_indices=np.array([11]),
        labels=labels,
        lr=0.8,
        lr_schedule="linear-decay",
        lr_final_scale=0.25,
        update_mode="direct",
        phase_clock_mode="analytic",
        target_source_mode="rails",
        output_bias_state_frozen=True,
        phase_output_vector_count=70,
        phase_output_includes_y=False,
        reference_mode="none",
        init_weights="",
        strict_fully_on_device=True,
        estimated_transient_points=34,
        max_transient_points=100,
        max_source_pwl_points=200,
        t_stop=6.6e-9,
        transient_step=200e-12,
        phase=0.5e-9,
        settle_ratio=20.0,
        source_complexity=source_complexity,
    )

    assert summary["status"] == "phase_preflight_only"
    assert summary["preflight_only"] is True
    assert summary["strict_fully_on_device_contract_met"] is True
    assert summary["lr_schedule"] == "linear-decay"
    assert summary["lr_final_scale"] == 0.25
    assert summary["phase_clock_mode"] == "analytic"
    assert summary["target_source_mode"] == "rails"
    assert summary["output_bias_state_frozen"] is True
    assert summary["phase_output_vector_count"] == 70
    assert summary["phase_output_includes_y"] is False
    assert summary["phase_netlist"] is None
    assert summary["final_weights"] is None
    assert summary["sample_source_pwl_points"] == 60
    assert summary["phase_clock_source_pwl_points"] == 50
    assert summary["total_source_pwl_points"] == 110


def test_phase_transient_relu_deck_matches_forward_and_backward_activation(tmp_path: Path) -> None:
    x = np.zeros((1, 4))
    y = np.array([0])
    w = np.zeros((1, 1, 4))
    hb = np.zeros((1, 1))
    readout = np.zeros((2, 1, 1))
    output_bias = np.zeros(2)

    netlist, _n_vec, _t_stop = phase_transient.make_phase_transient_netlist(
        x,
        y,
        w,
        hb,
        readout,
        output_bias,
        [[0, 1, 2, 3]],
        0.8,
        tmp_path / "out.dat",
        False,
        1,
        1,
        1e-9,
        0.1e-9,
        5e-12,
        40.0,
        20e-12,
        1e-12,
        1e-12,
        1e-12,
        1e18,
        True,
        "measure",
        (),
        "relu",
        1.0,
    )

    assert "0.5*((V(ah0_0))+abs(V(ah0_0)))" in netlist
    assert "0.5*(1+(V(ah0_0))/(abs(V(ah0_0))+1e-9))" in netlist


def test_phase_transient_print_mode_uses_native_tran_print(tmp_path: Path) -> None:
    x = np.zeros((1, 4))
    y = np.array([0])
    w = np.zeros((1, 1, 4))
    hb = np.zeros((1, 1))
    readout = np.zeros((2, 1, 1))
    output_bias = np.zeros(2)

    netlist, _n_vec, _t_stop = phase_transient.make_phase_transient_netlist(
        x,
        y,
        w,
        hb,
        readout,
        output_bias,
        [[0, 1, 2, 3]],
        0.8,
        tmp_path / "out.dat",
        False,
        1,
        1,
        1e-9,
        0.1e-9,
        5e-12,
        40.0,
        20e-12,
        1e-12,
        1e-12,
        1e-12,
        1e18,
        True,
        "print",
    )

    assert ".tran " in netlist
    assert ".options output OUTPUTTIMEPOINTS=" in netlist
    assert ".print TRAN " in netlist
    assert ".control" not in netlist
    assert "wrdata" not in netlist


def test_phase_transient_print_mode_supports_sparse_probe_timepoints(tmp_path: Path) -> None:
    x = np.zeros((2, 4))
    y = np.array([0, 1])
    w = np.zeros((1, 1, 4))
    hb = np.zeros((1, 1))
    readout = np.zeros((2, 1, 1))
    output_bias = np.zeros(2)

    netlist, _n_vec, _t_stop = phase_transient.make_phase_transient_netlist(
        x,
        y,
        w,
        hb,
        readout,
        output_bias,
        [[0, 1, 2, 3]],
        0.8,
        tmp_path / "out.dat",
        False,
        1,
        2,
        1e-9,
        0.1e-9,
        5e-12,
        40.0,
        20e-12,
        1e-12,
        1e-12,
        1e-12,
        1e18,
        True,
        "print",
        (1, 2),
        update_mode="direct",
    )

    output_line = next(line for line in netlist.splitlines() if line.startswith(".options output OUTPUTTIMEPOINTS="))
    assert output_line.count(",") == 2
    assert ".print TRAN " in netlist
    assert ".measure TRAN p000_00000" not in netlist
    assert ".control" not in netlist


def test_phase_transient_measure_mode_uses_top_level_measures(tmp_path: Path) -> None:
    x = np.zeros((1, 4))
    y = np.array([0])
    w = np.zeros((1, 1, 4))
    hb = np.zeros((1, 1))
    readout = np.zeros((2, 1, 1))
    output_bias = np.zeros(2)

    netlist, n_vec, _t_stop = phase_transient.make_phase_transient_netlist(
        x,
        y,
        w,
        hb,
        readout,
        output_bias,
        [[0, 1, 2, 3]],
        0.8,
        tmp_path / "out.dat",
        False,
        1,
        1,
        1e-9,
        0.1e-9,
        5e-12,
        40.0,
        20e-12,
        1e-12,
        1e-12,
        1e-12,
        1e18,
        True,
        "measure",
    )

    assert ".tran " in netlist
    assert ".measure TRAN m00000 FIND V(w0_0_0)" in netlist
    assert f".measure TRAN m{n_vec - 1:05d}" in netlist
    assert ".control" not in netlist
    assert ".print TRAN" not in netlist


def test_phase_transient_probe_measures_are_inside_same_continuous_deck(tmp_path: Path) -> None:
    x = np.zeros((2, 4))
    y = np.array([0, 1])
    w = np.zeros((1, 1, 4))
    hb = np.zeros((1, 1))
    readout = np.zeros((2, 1, 1))
    output_bias = np.zeros(2)

    netlist, _n_vec, _t_stop = phase_transient.make_phase_transient_netlist(
        x,
        y,
        w,
        hb,
        readout,
        output_bias,
        [[0, 1, 2, 3]],
        0.8,
        tmp_path / "out.dat",
        False,
        1,
        2,
        1e-9,
        0.1e-9,
        5e-12,
        40.0,
        20e-12,
        1e-12,
        1e-12,
        1e-12,
        1e18,
        True,
        "measure",
        (1, 2),
    )

    assert netlist.count(".tran ") == 1
    assert ".measure TRAN p000_00000 FIND V(w0_0_0)" in netlist
    assert ".measure TRAN p001_00000 FIND V(w0_0_0)" in netlist


def test_phase_transient_output_bias_update_scale_controls_bias_update(tmp_path: Path) -> None:
    x = np.zeros((1, 4))
    y = np.array([0])
    w = np.zeros((1, 1, 4))
    hb = np.zeros((1, 1))
    readout = np.zeros((2, 1, 1))
    output_bias = np.zeros(2)

    netlist, _n_vec, _t_stop = phase_transient.make_phase_transient_netlist(
        x,
        y,
        w,
        hb,
        readout,
        output_bias,
        [[0, 1, 2, 3]],
        0.8,
        tmp_path / "out.dat",
        False,
        1,
        1,
        1e-9,
        0.1e-9,
        5e-12,
        40.0,
        20e-12,
        1e-12,
        1e-12,
        1e-12,
        1e18,
        True,
        "measure",
        update_mode="direct",
        output_bias_update_scale=0.25,
    )

    assert ".param OB_UPDATE_SCALE=0.25" in netlist
    assert "Bupd_ob0 ob0 0 I = -V(pacc)*{CW}*{LR}*{OB_UPDATE_SCALE}/({BS}*{TAREA})*V(d0)" in netlist

    with pytest.raises(ValueError, match="output_bias_update_scale"):
        phase_transient.make_phase_transient_netlist(
            x,
            y,
            w,
            hb,
            readout,
            output_bias,
            [[0, 1, 2, 3]],
            0.8,
            tmp_path / "out.dat",
            False,
            1,
            1,
            1e-9,
            0.1e-9,
            5e-12,
            40.0,
            20e-12,
            1e-12,
            1e-12,
            1e-12,
            1e18,
            True,
            "measure",
            output_bias_update_scale=-1.0,
        )


def test_phase_transient_readout_update_scale_controls_readout_update(tmp_path: Path) -> None:
    x = np.zeros((1, 4))
    y = np.array([0])
    w = np.zeros((1, 1, 4))
    hb = np.zeros((1, 1))
    readout = np.zeros((2, 1, 1))
    output_bias = np.zeros(2)

    direct_netlist, _n_vec, _t_stop = phase_transient.make_phase_transient_netlist(
        x,
        y,
        w,
        hb,
        readout,
        output_bias,
        [[0, 1, 2, 3]],
        0.8,
        tmp_path / "direct.dat",
        False,
        1,
        1,
        1e-9,
        0.1e-9,
        5e-12,
        40.0,
        20e-12,
        1e-12,
        1e-12,
        1e-12,
        1e18,
        True,
        "measure",
        update_mode="direct",
        readout_update_scale=0.25,
    )

    assert ".param READOUT_UPDATE_SCALE=0.25" in direct_netlist
    assert "Bupd_v0_0_0 v0_0_0 0 I = -V(pacc)*{CW}*{LR}*{READOUT_UPDATE_SCALE}/({BS}*{TAREA})*(V(d0)*V(h0_0))" in direct_netlist
    assert "Bupd_ob0 ob0 0 I = -V(pacc)*{CW}*{LR}*{OB_UPDATE_SCALE}/({BS}*{TAREA})*V(d0)" in direct_netlist

    phased_netlist, _n_vec, _t_stop = phase_transient.make_phase_transient_netlist(
        x,
        y,
        w,
        hb,
        readout,
        output_bias,
        [[0, 1, 2, 3]],
        0.8,
        tmp_path / "phased.dat",
        False,
        1,
        1,
        1e-9,
        0.1e-9,
        5e-12,
        40.0,
        20e-12,
        1e-12,
        1e-12,
        1e-12,
        1e18,
        True,
        "measure",
        readout_update_scale=0.5,
    )

    assert ".param READOUT_UPDATE_SCALE=0.5" in phased_netlist
    assert "Bupd_v0_0_0 v0_0_0 0 I = -V(papply)*{CW}*{LR}*{READOUT_UPDATE_SCALE}/({BS}*{TAREA})*V(gv0_0_0)" in phased_netlist

    with pytest.raises(ValueError, match="readout_update_scale"):
        phase_transient.make_phase_transient_netlist(
            x,
            y,
            w,
            hb,
            readout,
            output_bias,
            [[0, 1, 2, 3]],
            0.8,
            tmp_path / "bad.dat",
            False,
            1,
            1,
            1e-9,
            0.1e-9,
            5e-12,
            40.0,
            20e-12,
            1e-12,
            1e-12,
            1e-12,
            1e18,
            True,
            "measure",
            readout_update_scale=-1.0,
        )


def test_phase_transient_local_update_scale_controls_feature_updates(tmp_path: Path) -> None:
    x = np.zeros((1, 4))
    y = np.array([0])
    w = np.zeros((1, 1, 4))
    hb = np.zeros((1, 1))
    readout = np.zeros((2, 1, 1))
    output_bias = np.zeros(2)

    direct_netlist, _n_vec, _t_stop = phase_transient.make_phase_transient_netlist(
        x,
        y,
        w,
        hb,
        readout,
        output_bias,
        [[0, 1, 2, 3]],
        0.8,
        tmp_path / "direct.dat",
        False,
        1,
        1,
        1e-9,
        0.1e-9,
        5e-12,
        40.0,
        20e-12,
        1e-12,
        1e-12,
        1e-12,
        1e18,
        True,
        "measure",
        update_mode="direct",
        local_update_scale=0.25,
    )

    assert ".param LOCAL_UPDATE_SCALE=0.25" in direct_netlist
    assert "Bupd_w0_0_0 w0_0_0 0 I = -V(pacc)*{CW}*{LR}*{LOCAL_UPDATE_SCALE}/({BS}*{TAREA})*(V(dh0_0)*V(pix0))" in direct_netlist
    assert "Bupd_hb0_0 hb0_0 0 I = -V(pacc)*{CW}*{LR}*{LOCAL_UPDATE_SCALE}/({BS}*{TAREA})*V(dh0_0)" in direct_netlist
    assert "Bupd_v0_0_0 v0_0_0 0 I = -V(pacc)*{CW}*{LR}*{READOUT_UPDATE_SCALE}/({BS}*{TAREA})*(V(d0)*V(h0_0))" in direct_netlist

    phased_netlist, _n_vec, _t_stop = phase_transient.make_phase_transient_netlist(
        x,
        y,
        w,
        hb,
        readout,
        output_bias,
        [[0, 1, 2, 3]],
        0.8,
        tmp_path / "phased.dat",
        False,
        1,
        1,
        1e-9,
        0.1e-9,
        5e-12,
        40.0,
        20e-12,
        1e-12,
        1e-12,
        1e-12,
        1e18,
        True,
        "measure",
        local_update_scale=0.5,
    )

    assert ".param LOCAL_UPDATE_SCALE=0.5" in phased_netlist
    assert "Bupd_w0_0_0 w0_0_0 0 I = -V(papply)*{CW}*{LR}*{LOCAL_UPDATE_SCALE}/({BS}*{TAREA})*V(gw0_0_0)" in phased_netlist
    assert "Bupd_hb0_0 hb0_0 0 I = -V(papply)*{CW}*{LR}*{LOCAL_UPDATE_SCALE}/({BS}*{TAREA})*V(ghb0_0)" in phased_netlist

    with pytest.raises(ValueError, match="local_update_scale"):
        phase_transient.make_phase_transient_netlist(
            x,
            y,
            w,
            hb,
            readout,
            output_bias,
            [[0, 1, 2, 3]],
            0.8,
            tmp_path / "bad.dat",
            False,
            1,
            1,
            1e-9,
            0.1e-9,
            5e-12,
            40.0,
            20e-12,
            1e-12,
            1e-12,
            1e-12,
            1e18,
            True,
            "measure",
            local_update_scale=-1.0,
        )


def test_phase_transient_zero_update_scales_omit_write_sources(tmp_path: Path) -> None:
    x = np.zeros((1, 4))
    y = np.array([0])
    w = np.zeros((1, 1, 4))
    hb = np.zeros((1, 1))
    readout = np.zeros((2, 1, 1))
    output_bias = np.zeros(2)

    direct_netlist, direct_n_vec, _t_stop = phase_transient.make_phase_transient_netlist(
        x,
        y,
        w,
        hb,
        readout,
        output_bias,
        [[0, 1, 2, 3]],
        0.8,
        tmp_path / "direct.dat",
        False,
        1,
        1,
        1e-9,
        0.1e-9,
        5e-12,
        40.0,
        20e-12,
        1e-12,
        1e-12,
        1e-12,
        1e18,
        True,
        "measure",
        update_mode="direct",
        local_update_scale=0.0,
        readout_update_scale=0.0,
        output_bias_update_scale=0.0,
    )

    assert ".param LOCAL_UPDATE_SCALE=0" in direct_netlist
    assert ".param READOUT_UPDATE_SCALE=0" in direct_netlist
    assert ".param OB_UPDATE_SCALE=0" in direct_netlist
    assert "Bupd_w0_0_0" not in direct_netlist
    assert "Bupd_hb0_0" not in direct_netlist
    assert "Bupd_v0_0_0" not in direct_netlist
    assert "Bupd_ob0" not in direct_netlist
    assert "Cob0" not in direct_netlist
    assert "V(ob0)" not in direct_netlist
    assert direct_n_vec == 9

    phased_netlist, phased_n_vec, _t_stop = phase_transient.make_phase_transient_netlist(
        x,
        y,
        w,
        hb,
        readout,
        output_bias,
        [[0, 1, 2, 3]],
        0.8,
        tmp_path / "phased.dat",
        False,
        1,
        1,
        1e-9,
        0.1e-9,
        5e-12,
        40.0,
        20e-12,
        1e-12,
        1e-12,
        1e-12,
        1e18,
        True,
        "measure",
        local_update_scale=0.0,
        readout_update_scale=0.0,
        output_bias_update_scale=0.0,
    )

    assert "Cgw0_0_0" not in phased_netlist
    assert "Cghb0_0" not in phased_netlist
    assert "Cgv0_0_0" not in phased_netlist
    assert "Cgob0" not in phased_netlist
    assert "Bacc_w0_0_0" not in phased_netlist
    assert "Bacc_v0_0_0" not in phased_netlist
    assert "Bacc_ob0" not in phased_netlist
    assert "Cob0" not in phased_netlist
    assert "V(ob0)" not in phased_netlist
    assert phased_n_vec == 9


def test_phase_transient_can_omit_output_y_measurement_vectors(tmp_path: Path) -> None:
    x = np.zeros((1, 4))
    y = np.array([0])
    w = np.zeros((1, 1, 4))
    hb = np.zeros((1, 1))
    readout = np.zeros((2, 1, 1))
    output_bias = np.zeros(2)

    netlist, n_vec, _t_stop = phase_transient.make_phase_transient_netlist(
        x,
        y,
        w,
        hb,
        readout,
        output_bias,
        [[0, 1, 2, 3]],
        0.8,
        tmp_path / "direct.dat",
        False,
        1,
        1,
        1e-9,
        0.1e-9,
        5e-12,
        40.0,
        20e-12,
        1e-12,
        1e-12,
        1e-12,
        1e18,
        True,
        "print",
        update_mode="direct",
        output_bias_update_scale=0.0,
        include_output_y_vectors=False,
    )
    print_line = next(line for line in netlist.splitlines() if line.startswith(".print TRAN "))

    assert n_vec == 7
    assert "V(y0)" in netlist
    assert "V(y0)" not in print_line
    nw, nhb, nv, nob, yy = phase_transient.unpack_state(
        np.zeros(n_vec),
        w,
        hb,
        readout,
        output_bias,
        include_output_bias_vectors=False,
        include_y_vectors=False,
    )
    assert nw.shape == w.shape
    assert nhb.shape == hb.shape
    assert nv.shape == readout.shape
    assert nob.tolist() == output_bias.tolist()
    assert yy.size == 0


def test_phase_transient_state_decay_is_on_device_update_phase_current(tmp_path: Path) -> None:
    x = np.zeros((1, 4))
    y = np.array([0])
    w = np.zeros((1, 1, 4))
    hb = np.zeros((1, 1))
    readout = np.zeros((2, 1, 1))
    output_bias = np.zeros(2)

    direct_netlist, _n_vec, _t_stop = phase_transient.make_phase_transient_netlist(
        x,
        y,
        w,
        hb,
        readout,
        output_bias,
        [[0, 1, 2, 3]],
        0.8,
        tmp_path / "direct.dat",
        False,
        1,
        1,
        1e-9,
        0.1e-9,
        5e-12,
        40.0,
        20e-12,
        1e-12,
        1e-12,
        1e-12,
        1e18,
        True,
        "measure",
        update_mode="direct",
        state_decay=0.05,
    )

    assert ".param STATE_DECAY=0.05" in direct_netlist
    assert "Bdecay_w0_0_0 w0_0_0 0 I = V(pacc)*{CW}*{STATE_DECAY}/{TAREA}*V(w0_0_0)" in direct_netlist
    assert "Bdecay_hb0_0 hb0_0 0 I = V(pacc)*{CW}*{STATE_DECAY}/{TAREA}*V(hb0_0)" in direct_netlist
    assert "Bdecay_v0_0_0 v0_0_0 0 I = V(pacc)*{CW}*{STATE_DECAY}/{TAREA}*V(v0_0_0)" in direct_netlist
    assert "Bdecay_ob0 ob0 0 I = V(pacc)*{CW}*{STATE_DECAY}/{TAREA}*V(ob0)" in direct_netlist
    assert "Cob0 ob0 0 {CW}" in direct_netlist

    phased_netlist, _n_vec, _t_stop = phase_transient.make_phase_transient_netlist(
        x,
        y,
        w,
        hb,
        readout,
        output_bias,
        [[0, 1, 2, 3]],
        0.8,
        tmp_path / "phased.dat",
        False,
        1,
        1,
        1e-9,
        0.1e-9,
        5e-12,
        40.0,
        20e-12,
        1e-12,
        1e-12,
        1e-12,
        1e18,
        True,
        "measure",
        state_decay=0.05,
    )

    assert "Bdecay_w0_0_0 w0_0_0 0 I = V(papply)*{CW}*{STATE_DECAY}/{TAREA}*V(w0_0_0)" in phased_netlist

    with pytest.raises(ValueError, match="state_decay"):
        phase_transient.make_phase_transient_netlist(
            x,
            y,
            w,
            hb,
            readout,
            output_bias,
            [[0, 1, 2, 3]],
            0.8,
            tmp_path / "bad.dat",
            False,
            1,
            1,
            1e-9,
            0.1e-9,
            5e-12,
            40.0,
            20e-12,
            1e-12,
            1e-12,
            1e-12,
            1e18,
            True,
            "measure",
            state_decay=1.0,
        )


def test_phase_transient_control_measure_mode_keeps_measures_inside_control(tmp_path: Path) -> None:
    x = np.zeros((1, 4))
    y = np.array([0])
    w = np.zeros((1, 1, 4))
    hb = np.zeros((1, 1))
    readout = np.zeros((2, 1, 1))
    output_bias = np.zeros(2)

    netlist, _n_vec, _t_stop = phase_transient.make_phase_transient_netlist(
        x,
        y,
        w,
        hb,
        readout,
        output_bias,
        [[0, 1, 2, 3]],
        0.8,
        tmp_path / "out.dat",
        False,
        1,
        1,
        1e-9,
        0.1e-9,
        5e-12,
        40.0,
        20e-12,
        1e-12,
        1e-12,
        1e-12,
        1e18,
        True,
        "control_measure",
    )

    assert ".control" in netlist
    assert "meas tran m00000 FIND V(w0_0_0)" in netlist
    assert ".measure TRAN" not in netlist
    assert "wrdata" not in netlist
