import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPICE_DIR = ROOT / "spice"
if str(SPICE_DIR) not in sys.path:
    sys.path.insert(0, str(SPICE_DIR))

import run_spice_mnist_local_feature_batch_op_train as batch_op  # noqa: E402


def test_cli_exposes_simulator_selector() -> None:
    proc = subprocess.run(
        [sys.executable, str(SPICE_DIR / "run_spice_mnist_local_feature_batch_op_train.py"), "--help"],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--simulator" in proc.stdout
    assert "--hidden-synapse-mode" in proc.stdout
    assert "--readout-synapse-mode" in proc.stdout


def test_synapse_transfer_expr_covers_linear_smooth_hard_and_sign_modes() -> None:
    assert batch_op.synapse_transfer_expr("V(w)", "linear", 0.25) == "V(w)"
    assert batch_op.synapse_transfer_expr("V(w)", "tanh-clipped", 0.25) == "(0.25*tanh((V(w))/0.25))"
    assert "abs(V(w))+1e-9" in batch_op.synapse_transfer_expr("V(w)", "sign", 0.25)
    assert "0.5*(((V(w))+0.25)" in batch_op.synapse_transfer_expr("V(w)", "hard-clipped", 0.25)


def test_x_yce_print_reader_extracts_first_operating_point_row(tmp_path: Path) -> None:
    path = tmp_path / "deck.cir.prn"
    path.write_text(
        "Index       V(A)          V(B)\n"
        "0        1.25000000e+00 -3.50000000e-01\n"
        "End of Xyce(TM) Simulation\n"
    )

    assert batch_op.read_xyce_print_row(path, 2) == pytest.approx([1.25, -0.35])


def test_local_feature_decks_include_x_yce_operating_point_print() -> None:
    x = np.zeros((1, 4))
    w = np.zeros((1, 1, 4))
    hb = np.zeros((1, 1))
    v = np.ones((2, 1, 1))
    ob = np.zeros(2)

    netlist = batch_op.make_eval_netlist(
        x,
        w,
        hb,
        v,
        ob,
        [[0, 1, 2, 3]],
        Path("out.dat"),
        True,
        False,
        "tanh",
        1.0,
        hidden_synapse_mode="tanh-clipped",
        readout_synapse_mode="hard-clipped",
        synapse_clip=0.25,
    )

    assert ".op" in netlist
    assert ".print DC V(y0_0) V(y0_1)" in netlist
    assert "wrdata out.dat V(y0_0) V(y0_1)" in netlist
    assert "(0.25*tanh((V(w0_0_0))/0.25))*V(x0_0)" in netlist
    assert "V(v0_0_0))+0.25" in netlist


def test_x_yce_behavioral_rhs_wrapper_preserves_spaced_expressions() -> None:
    netlist = "B1 out 0 V = V(in) + 1 + V(bias)\nR1 out 0 1k\n"

    assert batch_op.wrap_xyce_behavioral_rhs(netlist).splitlines()[0] == "B1 out 0 V = {V(in) + 1 + V(bias)}"
    assert batch_op.wrap_xyce_behavioral_rhs("B1 out 0 V = {V(in) + 1}\n") == "B1 out 0 V = {V(in) + 1}\n"


def test_epoch_order_slice_applies_offset_sample_limit_then_batch_cap() -> None:
    order = np.arange(10)

    assert batch_op.epoch_order_slice(order, 5, 2, 1, 3).tolist() == [2, 3, 4]
    assert batch_op.epoch_order_slice(order, 0, 4, 2, 2).tolist() == [4, 5, 6, 7]
    assert batch_op.epoch_order_slice(order, 0, 0, 0, 4).tolist() == list(range(10))

    with pytest.raises(ValueError, match="batch size"):
        batch_op.epoch_order_slice(order, 0, 0, 1, 0)


def test_skip_epoch_shuffles_advances_resume_order() -> None:
    first_rng = np.random.default_rng(123)
    first_order = np.arange(10)
    first_rng.shuffle(first_order)
    second_order = np.arange(10)
    first_rng.shuffle(second_order)

    resumed_rng = np.random.default_rng(123)
    batch_op.skip_epoch_shuffles(resumed_rng, 1, 10)
    resumed_order = np.arange(10)
    resumed_rng.shuffle(resumed_order)

    assert first_order.tolist() != second_order.tolist()
    assert resumed_order.tolist() == second_order.tolist()


def test_save_weight_checkpoint_is_init_weights_compatible(tmp_path: Path) -> None:
    path = tmp_path / "latest_weights.npz"
    w = np.arange(8, dtype=float).reshape(1, 2, 4)
    hb = np.array([[0.1, 0.2]])
    v = np.arange(20, dtype=float).reshape(10, 1, 2)
    ob = np.linspace(-0.5, 0.5, 10)

    batch_op.save_weight_checkpoint(path, w, hb, v, ob, epoch=3, completed_train_batches=17)

    with np.load(path) as checkpoint:
        assert checkpoint["local_weights"] == pytest.approx(w)
        assert checkpoint["local_bias"] == pytest.approx(hb)
        assert checkpoint["readout"] == pytest.approx(v)
        assert checkpoint["output_bias"] == pytest.approx(ob)
        assert json.loads(str(checkpoint["metadata_json"])) == {
            "completed_train_batches": 17,
            "epoch": 3,
        }


def test_accuracy_helpers_ignore_train_only_rows() -> None:
    rows = [
        {"epoch": 1, "heldout_accuracy": None},
        {"epoch": 2, "heldout_accuracy": 0.4},
        {"epoch": 3, "heldout_accuracy": None},
        {"epoch": 4, "heldout_accuracy": 0.6},
    ]

    assert batch_op.final_accuracy(rows) == pytest.approx(0.6)
    assert batch_op.best_accuracy(rows) == pytest.approx(0.6)
    assert batch_op.final_accuracy([{"heldout_accuracy": None}]) is None
    assert batch_op.best_accuracy([{"heldout_accuracy": None}]) is None
