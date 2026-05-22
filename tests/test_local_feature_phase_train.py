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


def test_phase_transient_x_yce_print_reader_extracts_final_transient_row(tmp_path: Path) -> None:
    path = tmp_path / "deck.cir.prn"
    path.write_text(
        "Index       TIME          V(A)          V(B)\n"
        "0        0.0            1.00000000e+00 -1.00000000e+00\n"
        "1        1.0e-9         1.25000000e+00 -3.50000000e-01\n"
        "End of Xyce(TM) Simulation\n"
    )

    assert phase_transient.read_xyce_print_last_row(path, 2) == pytest.approx([1.25, -0.35])


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
    assert "By0 y0 0 V = exp(V(score0))/(exp(V(score0)) + exp(V(score1)))" in netlist
    assert "Bstore_d0 d0 0 I = V(perr)*{CSTATE}/{TAU}*(V(d0)-(V(target0)-V(y0)))" in netlist
    assert netlist.count("Vpapply papply 0 PWL(") == 1
    assert netlist.count("Vpclear pclear 0 PWL(") == 1
    assert netlist.count(" 1 ") >= len(y)


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
    assert ".print TRAN " in netlist
    assert ".control" not in netlist
    assert "wrdata" not in netlist


def test_phase_transient_native_measure_mode_uses_top_level_measures(tmp_path: Path) -> None:
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
        "native_measure",
    )

    assert ".tran " in netlist
    assert ".measure TRAN m00000 FIND V(w0_0_0)" in netlist
    assert f".measure TRAN m{n_vec - 1:05d}" in netlist
    assert ".control" not in netlist
    assert ".print TRAN" not in netlist
