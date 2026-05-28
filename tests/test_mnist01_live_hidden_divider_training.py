from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPICE_DIR = ROOT / "spice"
sys.path.insert(0, str(SPICE_DIR))

import run_mnist01_live_hidden_divider_training as mnist01_hidden  # noqa: E402


def _require_mnist_raw() -> None:
    raw = ROOT / "data/MNIST/raw"
    required = [
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    ]
    if not all((raw / name).exists() for name in required):
        pytest.skip("raw MNIST IDX files are not available")


def test_mnist01_live_hidden_records_and_quadrant_mapping_are_real_4x4_mnist() -> None:
    _require_mnist_raw()

    train, evals = mnist01_hidden.load_mnist01_records(
        train_count_per_digit=1,
        eval_count_per_digit=1,
        image_size=4,
    )

    assert [sample["label"] for sample in train] == [0, 1]
    assert [sample["label"] for sample in evals] == [0, 1]
    assert mnist01_hidden.hidden_block_for_feature(0, 4) == 0
    assert mnist01_hidden.hidden_block_for_feature(2, 4) == 1
    assert mnist01_hidden.hidden_block_for_feature(8, 4) == 2
    assert mnist01_hidden.hidden_block_for_feature(15, 4) == 3
    for sample in train + evals:
        assert len(sample["features"]) == 16
        assert all(0.0 <= value <= 1.1 for value in sample["features"])
        assert max(sample["features"]) > 0.4


def test_mnist01_live_hidden_netlist_is_live_transistor_path() -> None:
    train = [
        {"features": [1.0] + [0.0] * 15, "label": 0},
        {"features": [0.0, 1.0] + [0.0] * 14, "label": 1},
    ]
    netlist = mnist01_hidden.mnist01_live_hidden_netlist(train, train)

    assert "\nB" not in netlist
    assert "Vapply" not in netlist
    assert "gvp" not in netlist
    assert "ghp" not in netlist
    assert "Mh0f0p_phi px0 featphi h0f0pmid 0 NSENSE" in netlist
    assert "Mhrow1_restore hrow1 hrow1_ctrl vdd vdd PMOS" in netlist
    assert "Mnorm0_score rd0 c0_scorep mir0 0 NSENSE" in netlist
    assert "Cc0_herrp c0_herrp 0 2f IC=0" in netlist
    assert "Mherr_c0p_m vdd b1low herr_c0p_a vdd PMOS" in netlist
    assert "Mh1_c0_cred_pv_e vdd c0_herrp h1_c0_cred_pv_e 0 NSENSE" in netlist
    assert "Mc0_f1_live_pos_up_p c0_vwp1 c0_f1_live_pos_up_ctrl vwhi_ref vdd PMOS" in netlist
    assert "Mh1f6_live_pup_pgate_phi h1f6_live_pup_pgphi errphi 0 0 NSENSE" in netlist
    assert ".meas tran train_hrow_probe_0" in netlist
    assert ".meas tran final_margin_improvement_1" in netlist

    differential_netlist = mnist01_hidden.mnist01_live_hidden_netlist(
        train,
        train,
        hidden_writer_topology="pmos-differential",
    )
    assert (
        "Mh1f6_live_pos_up_ctrl_phi h1f6_live_pos_up_ctrl_phi errphi 0 0 NSENSE"
        in differential_netlist
    )
    assert (
        "Mh1f6_live_neg_up_ctrl_phi h1f6_live_neg_up_ctrl_phi errphi 0 0 NSENSE"
        in differential_netlist
    )
    assert (
        "Mh1f6_live_pos_up_ctrl_latch h1f6_live_pos_up_ctrl h1f6_live_neg_up_ctrl vdd vdd PMOS"
        in differential_netlist
    )
    assert (
        "Mh1f6_live_neg_up_ctrl_latch h1f6_live_neg_up_ctrl h1f6_live_pos_up_ctrl vdd vdd PMOS"
        in differential_netlist
    )
    assert (
        "Mh1f6_live_pos_dn_select h1f6_live_pos_dn h1f6_live_neg_up_ctrl h1f6_live_pos_dn_sel 0 NSENSE"
        in differential_netlist
    )
    assert (
        "Mh1f6_live_neg_dn_select h1f6_live_neg_dn h1f6_live_pos_up_ctrl h1f6_live_neg_dn_sel 0 NSENSE"
        in differential_netlist
    )


def test_mnist01_live_hidden_netlist_validation() -> None:
    sample = {"features": [1.0] * 16, "label": 0}

    with pytest.raises(ValueError, match="empty"):
        mnist01_hidden.mnist01_live_hidden_netlist([], [sample])
    with pytest.raises(ValueError, match="exactly four"):
        mnist01_hidden.mnist01_live_hidden_netlist([sample], [sample], hidden_count=3)
    with pytest.raises(ValueError, match="square"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [{"features": [1.0, 0.0], "label": 0}],
            [{"features": [1.0, 0.0], "label": 0}],
        )
    with pytest.raises(ValueError, match="labels"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [{"features": [1.0] * 16, "label": 2}],
            [sample],
        )
    with pytest.raises(ValueError, match="positive"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [sample],
            [sample],
            hidden_error_route_width_u=0.0,
        )
    with pytest.raises(ValueError, match="hidden_writer_topology"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [sample],
            [sample],
            hidden_writer_topology="BAD",
        )
    with pytest.raises(ValueError, match="even"):
        mnist01_hidden.hidden_block_for_feature(0, 5)
    with pytest.raises(ValueError, match="outside"):
        mnist01_hidden.hidden_block_for_feature(16, 4)


@pytest.mark.ngspice
def test_mnist01_live_hidden_divider_ngspice_bootstraps_readout_and_visible_subthreshold_hidden_credit(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    _require_mnist_raw()
    train, _evals = mnist01_hidden.load_mnist01_records(
        train_count_per_digit=1,
        eval_count_per_digit=1,
        image_size=4,
    )

    parsed = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_live_hidden_divider.cir",
        mnist01_hidden.mnist01_live_hidden_netlist(train, train),
        timeout=120.0,
    )

    assert abs(parsed["initial_margin_0"]) < 1e-6
    assert abs(parsed["initial_margin_1"]) < 1e-6
    assert parsed["final_margin_0"] > 0.10e-3
    assert parsed["final_margin_1"] > 1e-6
    assert parsed["final_margin_improvement_0"] > 0.10e-3
    assert parsed["final_margin_improvement_1"] > 1e-6

    for train_idx in range(2):
        ir_sum = abs(parsed[f"train_ir0_{train_idx}"]) + abs(parsed[f"train_ir1_{train_idx}"])
        assert ir_sum == pytest.approx(1.0e-6, rel=0.08)
        assert parsed[f"train_act_probe_{train_idx}"] > 40e-3
        assert parsed[f"train_hrow_probe_{train_idx}"] > 1.0
        assert parsed[f"train_hrow_ctrl_probe_{train_idx}"] < 10e-3
        assert parsed[f"train_target_errp_{train_idx}"] > parsed[f"train_target_errn_{train_idx}"] + 30e-3
        assert parsed[f"train_other_errn_{train_idx}"] > parsed[f"train_other_errp_{train_idx}"] + 30e-3
        assert parsed[f"train_target_herrp_{train_idx}"] > parsed[f"train_target_errp_{train_idx}"] + 0.20
        assert parsed[f"train_other_herrn_{train_idx}"] > parsed[f"train_other_errn_{train_idx}"] + 0.20
        assert parsed[f"train_target_signed_delta_{train_idx}"] > 3e-3
        assert parsed[f"train_other_signed_delta_{train_idx}"] < -3e-3
        assert abs(parsed[f"train_wh_probe_signed_delta_{train_idx}"]) < 1e-3

    assert abs(parsed["train_hcredit_gate_probe_0"]) < 1e-6
    assert parsed["train_hcredit_gate_probe_1"] < -50e-6
    assert abs(parsed["train_hcredit_gate_probe_1"]) < 1e-3
