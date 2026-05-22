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
    assert "--eval-backend" in proc.stdout
    assert "--simulator-extra-args" in proc.stdout


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
        probe_updates="1,2,4,8",
        tag="sweep",
        relu_leak=0.01,
        softplus_beta=10.0,
        derivative_floor=0.0,
        derivative_gate_threshold=1e-6,
        readout_feedback_clip=0.05,
        hidden_synapse_modes="linear,tanh-clipped",
        readout_synapse_modes="linear,hard-clipped",
        synapse_clip=0.25,
        synapse_clips="0.25,1.0",
        softmax_output=True,
        linear_output=False,
        final_measures=False,
        eval_probe_updates=True,
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
    assert "By0 y0 0 V = exp(V(score0))/(exp(V(score0)) + exp(V(score1)))" in netlist
    assert "Bstore_d0 d0 0 I = V(perr)*{CSTATE}/{TAU}*(V(d0)-(V(target0)-V(y0)))" in netlist
    assert ".param TAREA=1.005e-09" in netlist
    assert "{CGRAD}/{TAREA}" in netlist
    assert "/({BS}*{TAREA})" in netlist
    assert "{CGRAD}/{TPHASE}" not in netlist
    assert netlist.count("Vpapply papply 0 PWL(") == 1
    assert netlist.count("Vpclear pclear 0 PWL(") == 1
    assert netlist.count(" 1 ") >= len(y)


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


def test_phase_transient_state_descriptions_follow_update_mode() -> None:
    phased = phase_transient.phase_state_descriptions("phased")
    direct = phase_transient.phase_state_descriptions("direct")

    assert "gradient accumulators" in phased["temporary_state"]
    assert "gradient accumulators" not in direct["temporary_state"]
    assert "updated directly during each per-sample update phase" in direct["temporary_state"]
    assert "checkpoint" not in direct["persistent_state"].lower()

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
