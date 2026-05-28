from __future__ import annotations

import sys
from pathlib import Path

import pytest


SPICE_DIR = Path(__file__).resolve().parents[1] / "spice"
sys.path.insert(0, str(SPICE_DIR))

import run_xor_live_hidden_divider_training as live_xor  # noqa: E402


def test_xor_live_hidden_divider_training_is_live_transistor_path() -> None:
    netlist = live_xor.xor_live_hidden_netlist([0, 1, 2, 3])

    assert "\nB" not in netlist
    assert "Vapply" not in netlist
    assert "gvp" not in netlist
    assert "ghp" not in netlist
    assert "Mh00p_w h00pmid wh00p pre0_p 0 NSENSE" in netlist
    assert "Mnorm0_score rd0 c0_scorep mir0 0 NSENSE" in netlist
    assert "Mc0_f0_live_pos_up_p c0_vwp0 c0_f0_live_pos_up_ctrl vwhi_ref vdd PMOS" in netlist
    assert "Mh0_c0_cred_pv_e vdd c0_errp h0_c0_cred_pv_e 0 NSENSE" in netlist
    assert "Mh0_hdp_gate_dn h0_hdp_gate h0_hdn 0 0 NSENSE" in netlist
    assert "Mh0b0_live_pup_pgate_cred h0b0_live_pup_pgmid h0_hdp_gate h0b0_live_pup_pgphi 0 NSENSE" in netlist
    assert "Mh0b0_live_pup_pgate_phi h0b0_live_pup_pgphi errphi 0 0 NSENSE" in netlist
    assert "Mh0b0_live_pup_pmos wh00p h0b0_live_pup_pgate hidden_whi_ref vdd PMOS" in netlist


def test_xor_live_hidden_divider_training_validation() -> None:
    with pytest.raises(ValueError, match="train_order"):
        live_xor.xor_live_hidden_netlist([])
    with pytest.raises(ValueError, match="patterns"):
        live_xor.xor_live_hidden_netlist([0, 4])
    with pytest.raises(ValueError, match="positive"):
        live_xor.xor_live_hidden_netlist([0, 1], hidden_update_width_u=0.0)
    with pytest.raises(ValueError, match="hidden_credit_activation_model"):
        live_xor.xor_live_hidden_netlist([0, 1], hidden_credit_activation_model="BAD")
    with pytest.raises(ValueError, match="hidden_writer_mode"):
        live_xor.xor_live_hidden_netlist([0, 1], hidden_writer_mode="BAD")


def test_xor_live_hidden_divider_ngspice_one_epoch_learns_output_without_hidden_rewrite(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    order = [0, 1, 2, 3]
    measures = live_xor.run_netlist(
        ngspice_path,
        tmp_path / "xor_live_hidden_one_epoch.cir",
        live_xor.xor_live_hidden_netlist(order),
        timeout=90.0,
    )

    for pattern in range(4):
        assert abs(measures[f"initial_margin_{pattern}"]) < 1e-6
        assert measures[f"final_margin_{pattern}"] > 0.30
        assert measures[f"final_margin_improvement_{pattern}"] > 0.30

    for slot, pattern in enumerate(order):
        ir0 = abs(measures[f"train_ir0_{slot}"])
        ir1 = abs(measures[f"train_ir1_{slot}"])
        assert ir0 + ir1 == pytest.approx(1.0e-6, rel=0.08)
        assert measures[f"train_target_errp_{slot}"] > 0.25
        assert measures[f"train_other_errn_{slot}"] > 0.25
        assert measures[f"train_target_signed_delta_{slot}"] > 0.15
        assert measures[f"train_other_signed_delta_{slot}"] < -0.15
        assert abs(measures[f"train_hcredit_gate_active_{slot}"]) < 2e-6
        for bit in range(live_xor.BITS):
            assert abs(measures[f"train_wh{pattern}{bit}_signed_delta_{slot}"]) < 1e-3


def test_xor_live_hidden_divider_ngspice_later_backprop_credit_is_gated_but_bounded(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    order = [0, 1, 2, 3, 0, 1, 2, 3]
    measures = live_xor.run_netlist(
        ngspice_path,
        tmp_path / "xor_live_hidden_two_epoch.cir",
        live_xor.xor_live_hidden_netlist(order),
        timeout=120.0,
    )

    for pattern in range(4):
        assert measures[f"final_margin_{pattern}"] > 0.40

    first_epoch_gates = [abs(measures[f"train_hcredit_gate_active_{slot}"]) for slot in range(4)]
    second_epoch_gates = [measures[f"train_hcredit_gate_active_{slot}"] for slot in range(4, 8)]
    assert max(first_epoch_gates) < 2e-6
    assert min(second_epoch_gates) > 20e-3

    for slot, pattern in enumerate(order[:4]):
        for bit in range(live_xor.BITS):
            assert abs(measures[f"train_wh{pattern}{bit}_signed_delta_{slot}"]) < 1e-3
    for slot, pattern in enumerate(order[4:], start=4):
        for bit in range(live_xor.BITS):
            assert measures[f"train_wh{pattern}{bit}_signed_delta_{slot}"] > 0.10
