from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPICE_DIR = ROOT / "spice"
if str(SPICE_DIR) not in sys.path:
    sys.path.insert(0, str(SPICE_DIR))

import run_device_relu_synapse_sweep as relu_primitives  # noqa: E402
import run_device_signed_learning_cell as signed_primitives  # noqa: E402
import run_device_xor2_random_hidden as direct_flow  # noqa: E402
from run_spice_sweep import detect_spice, run_tiny_test  # noqa: E402


PWL_TIME_RE = re.compile(r"([-+0-9.eE]+)n")


def pwl_times(wave: str) -> list[float]:
    return [float(match) for match in PWL_TIME_RE.findall(wave)]


@pytest.fixture(scope="session")
def spice_bin(tmp_path_factory: pytest.TempPathFactory) -> str:
    try:
        simulator, _version = detect_spice(None)
        run_tiny_test(simulator, tmp_path_factory.mktemp("spice_smoke"))
    except Exception as exc:
        pytest.skip(f"SPICE simulator unavailable: {exc}")
    return simulator


def test_relu_neuron_forward_transfer_is_thresholded(
    spice_bin: str,
    tmp_path: Path,
) -> None:
    low = relu_primitives.run_netlist(
        spice_bin,
        tmp_path / "relu_low.cir",
        relu_primitives.relu_transfer_netlist(0.2),
        timeout=20.0,
    )
    high = relu_primitives.run_netlist(
        spice_bin,
        tmp_path / "relu_high.cir",
        relu_primitives.relu_transfer_netlist(0.8),
        timeout=20.0,
    )

    assert low["vpre_m"] == pytest.approx(0.2, abs=1e-3)
    assert high["vpre_m"] == pytest.approx(0.8, abs=1e-3)
    assert abs(low["vact"]) < 1e-3
    assert high["vact"] > low["vact"] + 0.15
    assert high["eact"] > low["eact"]


def test_synapse_forward_path_sums_conductance_weighted_inputs(
    spice_bin: str,
    tmp_path: Path,
) -> None:
    weak = relu_primitives.run_netlist(
        spice_bin,
        tmp_path / "synapse_weak.cir",
        relu_primitives.synapse_relu_netlist(0.05, 0.2, 0.0, 0.2),
        timeout=20.0,
    )
    strong = relu_primitives.run_netlist(
        spice_bin,
        tmp_path / "synapse_strong.cir",
        relu_primitives.synapse_relu_netlist(1.0, 1.2, 0.0, 0.2),
        timeout=20.0,
    )
    two_inputs = relu_primitives.run_netlist(
        spice_bin,
        tmp_path / "synapse_two_inputs.cir",
        relu_primitives.synapse_relu_netlist(1.0, 1.2, 1.0, 1.2),
        timeout=20.0,
    )

    assert strong["vpre"] > weak["vpre"] + 0.45
    assert strong["vact"] > weak["vact"] + 0.08
    assert two_inputs["vpre"] > strong["vpre"] + 0.03
    assert two_inputs["vact"] > strong["vact"] + 0.02
    assert strong["vw0"] == pytest.approx(1.2, abs=1e-6)
    assert two_inputs["vw1"] == pytest.approx(1.2, abs=1e-6)


def test_update_cell_moves_weight_capacitor_in_both_directions(
    spice_bin: str,
    tmp_path: Path,
) -> None:
    positive = relu_primitives.run_netlist(
        spice_bin,
        tmp_path / "update_positive.cir",
        relu_primitives.update_cell_netlist(w_init=0.5, gplus=1.1, gminus=0.2),
        timeout=20.0,
    )
    negative = relu_primitives.run_netlist(
        spice_bin,
        tmp_path / "update_negative.cir",
        relu_primitives.update_cell_netlist(w_init=0.5, gplus=0.2, gminus=1.1),
        timeout=20.0,
    )

    assert positive["w_before"] == pytest.approx(0.5, abs=1e-6)
    assert negative["w_before"] == pytest.approx(0.5, abs=1e-6)
    assert positive["w_final"] > positive["w_before"] + 0.1
    assert negative["w_final"] < negative["w_before"] - 0.1
    assert positive["gp_final"] > positive["gn_final"]
    assert negative["gn_final"] > negative["gp_final"]


def test_signed_synapse_backward_delta_changes_signed_weight_direction(
    spice_bin: str,
    tmp_path: Path,
) -> None:
    positive_delta = signed_primitives.run_netlist(
        spice_bin,
        tmp_path / "signed_positive_delta.cir",
        signed_primitives.signed_learning_cell_netlist(
            vin=0.8,
            delta_p=1.1,
            delta_n=0.2,
            wp_init=0.95,
            wn_init=0.25,
        ),
        timeout=20.0,
    )
    negative_delta = signed_primitives.run_netlist(
        spice_bin,
        tmp_path / "signed_negative_delta.cir",
        signed_primitives.signed_learning_cell_netlist(
            vin=0.8,
            delta_p=0.2,
            delta_n=1.1,
            wp_init=0.95,
            wn_init=0.25,
        ),
        timeout=20.0,
    )

    assert positive_delta["gp_after_acc"] > positive_delta["gn_after_acc"] + 0.3
    assert positive_delta["d_signed"] > 0.2
    assert positive_delta["d_act"] > 0

    assert negative_delta["gn_after_acc"] > negative_delta["gp_after_acc"] + 0.3
    assert negative_delta["d_signed"] < -0.5
    assert negative_delta["d_act"] < -0.1


def test_direct_flow_generator_wires_backward_path_through_saved_pre_traces_and_neurons() -> None:
    original_hidden = direct_flow.HIDDEN
    original_input_rails = list(direct_flow.INPUT_RAILS)
    try:
        direct_flow.set_hidden_cells(2)
        direct_flow.set_input_rails(["x0", "x1"])
        design = direct_flow.scaled_synapse_design(
            "split_signed_passact_v1",
            hidden_delta_width_scale=1.0,
            hidden_gradient_width_scale=1.0,
            readout_gradient_width_scale=1.0,
        )

        delta = direct_flow.hidden_delta(
            hidden_error_rule="backprop",
            hidden_delta_relu_gate="act_nsense",
            hidden_delta_weight_device="nmos",
            design=design,
            internal_cap_f=0.0,
            internal_leak_ohm=0.0,
        )
        stores = direct_flow.flow_pre_activation_stores(
            mode="synapse_consume",
            cap_f=2.0,
            consume_width_u=0.05,
        )
        readout_updates = direct_flow.readout_flow_updates(
            readout_update_width_u=0.04,
            output_bias_update_width_u=0.04,
            flow_pre_store="synapse_consume",
            readout_flow_polarity="normal",
        )
        reversed_readout_updates = direct_flow.readout_flow_updates(
            readout_update_width_u=0.04,
            output_bias_update_width_u=0.0,
            flow_pre_store="synapse_consume",
            readout_flow_polarity="reversed",
        )
        hidden_updates = direct_flow.hidden_flow_updates(
            update_width_u=0.04,
            flow_pre_store="synapse_consume",
            hidden_delta_output_mode="senseamp",
            hidden_flow_write_mode="discharge",
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_input_rails(original_input_rails)

    assert "vw00p" in delta
    assert "vw00n" in delta
    assert "fb00p" not in delta
    assert "Mhdp_a00_r hdp_a00_1 act0 hdp_a00_2" in delta
    assert "Mhdp_a00_b hdp_a00_2 bwd hdp0" in delta

    assert "Mstore_fpro00 fpro00 fwd act0 0" in stores
    assert "Mstore_fphi0_x0 fphi0_x0 fwd x0 0" in stores
    assert "Mconsume_fphi0_x0 fphi0_x0 bwd 0 0" in stores

    assert "Mvw00n_flow_b vw00n bwd" in readout_updates
    assert "Mvw00n_flow_a vw00n_flow_b fpro00" in readout_updates
    assert "Mvw00n_flow_d vw00n_flow_a dp0 0 0" in readout_updates
    assert "Mvw00p_flow_d vw00p_flow_a dn0 0 0" in readout_updates
    assert "Mvw00n_flow_d vw00n_flow_a dn0 0 0" in reversed_readout_updates
    assert "Mvw00p_flow_d vw00p_flow_a dp0 0 0" in reversed_readout_updates
    assert "Mvbo0n_flow_b" not in reversed_readout_updates

    assert "Mwh0_x0n_flow_b wh0_x0n bwd" in hidden_updates
    assert "Mwh0_x0n_flow_x wh0_x0n_flow_b fphi0_x0" in hidden_updates
    assert "Mwh0_x0n_flow_d wh0_x0n_flow_x hdpg0" in hidden_updates
    assert "Mwh0_x0n_flow_a wh0_x0n_flow_d apply 0 0" in hidden_updates
    assert "Mwh0_x0p_flow_x wh0_x0p_flow_b fphi0_x0" in hidden_updates
    assert "Mwh0_x0p_flow_d wh0_x0p_flow_x hdng0" in hidden_updates


def test_direct_flow_hidden_can_bound_discharge_only() -> None:
    original_hidden = direct_flow.HIDDEN
    original_input_rails = list(direct_flow.INPUT_RAILS)
    try:
        direct_flow.set_hidden_cells(1)
        direct_flow.set_input_rails(["x0"])
        updates = direct_flow.hidden_flow_updates(
            update_width_u=0.02,
            flow_pre_store="synapse_gate",
            hidden_delta_output_mode="raw",
            hidden_flow_write_mode="bounded_discharge",
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_input_rails(original_input_rails)

    assert "Mwh0_x0n_flow_d wh0_x0n_flow_x hdp0 wlow 0 NSENSE" in updates
    assert "Mwh0_x0p_flow_d wh0_x0p_flow_x hdn0 wlow 0 NSENSE" in updates
    assert "_ch_b whigh" not in updates


def test_direct_flow_hidden_can_bound_charge_only_with_latched_delta() -> None:
    original_hidden = direct_flow.HIDDEN
    original_input_rails = list(direct_flow.INPUT_RAILS)
    try:
        direct_flow.set_hidden_cells(1)
        direct_flow.set_input_rails(["x0"])
        updates = direct_flow.hidden_flow_updates(
            update_width_u=0.02,
            flow_pre_store="synapse_gate",
            hidden_delta_output_mode="senseamp",
            hidden_flow_write_mode="bounded_charge_only",
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_input_rails(original_input_rails)

    assert "Mwh0_x0p_ch_b whigh bwd wh0_x0p_ch_b" in updates
    assert "Mwh0_x0p_ch_d wh0_x0p_ch_x hdpg0 wh0_x0p_ch_d" in updates
    assert "Mwh0_x0p_ch_a wh0_x0p_ch_d apply wh0_x0p" in updates
    assert "Mwh0_x0n_ch_d wh0_x0n_ch_x hdng0 wh0_x0n_ch_d" in updates
    assert "_flow_d" not in updates


def test_rail_buffer_hidden_forward_uses_input_pass_gates() -> None:
    original_hidden = direct_flow.HIDDEN
    original_input_rails = list(direct_flow.INPUT_RAILS)
    try:
        direct_flow.set_hidden_cells(3)
        direct_flow.set_input_rails(["x0", "x1"])
        design = direct_flow.scaled_synapse_design(
            "split_signed_passact_v1",
            hidden_delta_width_scale=1.0,
            hidden_gradient_width_scale=1.0,
            readout_gradient_width_scale=1.0,
        )

        forward = direct_flow.hidden_forward(design, "rail_buffer")
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_input_rails(original_input_rails)

    assert "Mhbuf0_act act0 fwd x0 0 NMOS" in forward
    assert "Mhbuf0_pre pre0 fwd x0 0 NMOS" in forward
    assert "Mhbuf1_act act1 fwd x1 0 NMOS" in forward
    assert "General hidden 2" in forward
    assert "Mh2_biasp_x" in forward


def test_synapse_design_scaling_separates_forward_readout_from_update_paths() -> None:
    scaled = direct_flow.scaled_synapse_design(
        "split_signed_v1",
        hidden_delta_width_scale=1.5,
        hidden_gradient_width_scale=2.0,
        readout_gradient_width_scale=0.5,
        output_forward_width_scale=3.0,
        output_forward_pos_width_scale=0.5,
        output_forward_neg_width_scale=2.0,
        output_bias_forward_width_scale=4.0,
        output_relu_width_scale=5.0,
    )

    assert scaled.hidden_delta_width_u == pytest.approx(48.0)
    assert scaled.hidden_gradient_width_u == pytest.approx(80.0)
    assert scaled.readout_gradient_width_u == pytest.approx(12.0)
    assert scaled.output_forward_pos_width_u == pytest.approx(84.0)
    assert scaled.output_forward_neg_width_u == pytest.approx(288.0)
    assert scaled.output_bias_forward_pos_width_u == pytest.approx(160.0)
    assert scaled.output_bias_forward_neg_width_u == pytest.approx(144.0)
    assert scaled.output_relu_width_u == pytest.approx(120.0)


def test_score_caps_can_reset_to_common_mode_for_signed_readout_headroom() -> None:
    caps = direct_flow.temporary_caps(
        gradient_cap_f=4.0,
        hidden_gradient_cap_f=4.0,
        hidden_delta_cap_f=12.0,
        lead_cap_f=2.0,
        include_gradient_caps=False,
        score_reset_v=0.30,
    )
    reset = direct_flow.resets("out_senseamp", include_gradient_resets=False, score_reset_v=0.30)

    assert "Cscore0 score0 0 10f IC=0.3" in caps
    assert "Cscore1 score1 0 10f IC=0.3" in caps
    assert "Mreset_score0 score0 rstf scorecm 0 NMOS" in reset
    assert "Mreset_score1 score1 rstf scorecm 0 NMOS" in reset


def test_score_diff_output_head_rejects_score_common_mode_before_class_output() -> None:
    design = direct_flow.scaled_synapse_design(
        "split_signed_v1",
        hidden_delta_width_scale=1.0,
        hidden_gradient_width_scale=1.0,
        readout_gradient_width_scale=1.0,
    )

    forward = direct_flow.output_forward(design, "score_diff")

    assert "Mrelu_o0" not in forward
    assert "Mout0_diff_pos_s vdd score0 out0_diff_pos" in forward
    assert "Mout0_diff_neg_s out0_diff_neg score1 0" in forward
    assert "Mout1_diff_pos_s vdd score1 out1_diff_pos" in forward
    assert "Mout1_diff_neg_s out1_diff_neg score0 0" in forward


def test_direct_flow_readout_can_charge_and_discharge_signed_weight_branches() -> None:
    updates = direct_flow.readout_flow_updates(
        readout_update_width_u=0.02,
        output_bias_update_width_u=0.0003,
        flow_pre_store="shared_node",
        readout_flow_polarity="normal",
        readout_flow_write_mode="charge_discharge",
    )

    assert "Mvw00n_flow_d vw00n_flow_a dp0 0" in updates
    assert "Mvw00p_flow_d vw00p_flow_a dn0 0" in updates
    assert "Mvw00p_ch_d vw00p_ch_a dp0 vw00p" in updates
    assert "Mvw00n_ch_d vw00n_ch_a dn0 vw00n" in updates
    assert "Mvbo0p_ch_d vbo0p_ch_b dp0 vbo0p" in updates
    assert "Mvbo0n_ch_d vbo0n_ch_b dn0 vbo0n" in updates


def test_direct_flow_readout_bounded_write_uses_local_selected_rails() -> None:
    updates = direct_flow.readout_flow_updates(
        readout_update_width_u=0.02,
        output_bias_update_width_u=0.0003,
        flow_pre_store="shared_node",
        readout_flow_polarity="normal",
        readout_flow_write_mode="bounded_charge_discharge",
    )

    assert "Mvw00n_flow_d vw00n_flow_a dp0 wlow 0 NSENSE" in updates
    assert "Mvw00p_flow_d vw00p_flow_a dn0 wlow 0 NSENSE" in updates
    assert "Mvw00p_ch_b whigh bwd vw00p_ch_b" in updates
    assert "Mvw00n_ch_b whigh bwd vw00n_ch_b" in updates
    assert "Mvbo0n_flow_d vbo0n_flow_b dp0 wlow 0 NSENSE" in updates
    assert "Mvbo0p_ch_b whigh bwd vbo0p_ch_b" in updates


def test_direct_flow_readout_can_bound_discharge_only() -> None:
    updates = direct_flow.readout_flow_updates(
        readout_update_width_u=0.02,
        output_bias_update_width_u=0.0003,
        flow_pre_store="shared_node",
        readout_flow_polarity="normal",
        readout_flow_write_mode="bounded_discharge",
    )

    assert "Mvw00n_flow_d vw00n_flow_a dp0 wlow 0 NSENSE" in updates
    assert "Mvw00p_flow_d vw00p_flow_a dn0 wlow 0 NSENSE" in updates
    assert "_ch_b whigh" not in updates


def test_direct_flow_readout_can_bound_charge_only() -> None:
    updates = direct_flow.readout_flow_updates(
        readout_update_width_u=0.02,
        output_bias_update_width_u=0.0003,
        flow_pre_store="shared_node",
        readout_flow_polarity="normal",
        readout_flow_write_mode="bounded_charge_only",
    )

    assert "Mvw00p_ch_b whigh bwd vw00p_ch_b" in updates
    assert "Mvw00n_ch_b whigh bwd vw00n_ch_b" in updates
    assert "_flow_d" not in updates


def test_direct_flow_readout_bounded_write_accepts_branch_specific_rails() -> None:
    updates = direct_flow.readout_flow_updates(
        readout_update_width_u=0.02,
        output_bias_update_width_u=0.0003,
        flow_pre_store="shared_node",
        readout_flow_polarity="normal",
        readout_flow_write_mode="bounded_charge_discharge",
        readout_pos_write_high_node="pos_hi",
        readout_pos_write_low_node="pos_lo",
        readout_neg_write_high_node="neg_hi",
        readout_neg_write_low_node="neg_lo",
    )

    assert "Mvw00n_flow_d vw00n_flow_a dp0 neg_lo 0 NSENSE" in updates
    assert "Mvw00p_flow_d vw00p_flow_a dn0 pos_lo 0 NSENSE" in updates
    assert "Mvw00p_ch_b pos_hi bwd vw00p_ch_b" in updates
    assert "Mvw00n_ch_b neg_hi bwd vw00n_ch_b" in updates


def test_direct_flow_readout_accepts_branch_specific_update_widths() -> None:
    updates = direct_flow.readout_flow_updates(
        readout_update_width_u=0.02,
        readout_pos_update_width_u=0.003,
        readout_neg_update_width_u=0.007,
        output_bias_update_width_u=0.0003,
        flow_pre_store="shared_node",
        readout_flow_polarity="normal",
        readout_flow_write_mode="bounded_charge_discharge",
    )

    assert "Mvw00n_flow_b vw00n bwd vw00n_flow_b 0 NREL W=0.007u" in updates
    assert "Mvw00p_flow_b vw00p bwd vw00p_flow_b 0 NREL W=0.003u" in updates
    assert "Mvw00p_ch_b whigh bwd vw00p_ch_b 0 NREL W=0.003u" in updates
    assert "Mvw00n_ch_b whigh bwd vw00n_ch_b 0 NREL W=0.007u" in updates


def test_direct_flow_readout_accepts_action_specific_update_widths() -> None:
    updates = direct_flow.readout_flow_updates(
        readout_update_width_u=0.02,
        readout_pos_update_width_u=0.003,
        readout_neg_update_width_u=0.007,
        readout_charge_update_width_u=0.011,
        readout_discharge_update_width_u=0.005,
        output_bias_update_width_u=0.0003,
        flow_pre_store="shared_node",
        readout_flow_polarity="normal",
        readout_flow_write_mode="bounded_charge_discharge",
    )

    assert "Mvw00n_flow_b vw00n bwd vw00n_flow_b 0 NREL W=0.005u" in updates
    assert "Mvw00p_flow_b vw00p bwd vw00p_flow_b 0 NREL W=0.005u" in updates
    assert "Mvw00p_ch_b whigh bwd vw00p_ch_b 0 NREL W=0.011u" in updates
    assert "Mvw00n_ch_b whigh bwd vw00n_ch_b 0 NREL W=0.011u" in updates


def test_direct_flow_readout_write_can_gate_discharge_by_stored_state() -> None:
    updates = direct_flow.readout_flow_updates(
        readout_update_width_u=0.02,
        readout_discharge_update_width_u=0.005,
        output_bias_update_width_u=0.0003,
        flow_pre_store="shared_node",
        readout_flow_polarity="normal",
        readout_flow_write_mode="bounded_discharge",
        readout_write_state_gate_mode="state_high_discharge",
    )

    assert "Mvw00n_flow_s vw00n vw00n vw00n_flow_s 0 NREL W=0.005u" in updates
    assert "Mvw00n_flow_b vw00n_flow_s bwd vw00n_flow_b 0 NREL W=0.005u" in updates
    assert "Mvw00p_flow_s vw00p vw00p vw00p_flow_s 0 NREL W=0.005u" in updates
    assert "Mvbo0n_flow_s vbo0n vbo0n vbo0n_flow_s 0 NREL W=0.0003u" in updates
    assert "_ch_s" not in updates


def test_direct_flow_readout_write_can_gate_charge_by_stored_state_window() -> None:
    updates = direct_flow.readout_flow_updates(
        readout_update_width_u=0.02,
        readout_charge_update_width_u=0.011,
        output_bias_update_width_u=0.0003,
        flow_pre_store="shared_node",
        readout_flow_polarity="normal",
        readout_flow_write_mode="bounded_charge_only",
        readout_write_state_gate_mode="state_window",
        readout_pos_write_high_node="pos_hi",
        readout_neg_write_high_node="neg_hi",
    )

    assert "Mvw00p_ch_s pos_hi vw00p vw00p_ch_s vdd PMOS W=0.011u" in updates
    assert "Mvw00p_ch_b vw00p_ch_s bwd vw00p_ch_b 0 NREL W=0.011u" in updates
    assert "Mvw00n_ch_s neg_hi vw00n vw00n_ch_s vdd PMOS W=0.011u" in updates
    assert "Mvbo0p_ch_s pos_hi vbo0p vbo0p_ch_s vdd PMOS W=0.0003u" in updates
    assert "_flow_s" not in updates


def test_direct_flow_readout_center_pull_is_a_transistor_state_path() -> None:
    updates = direct_flow.readout_flow_updates(
        readout_update_width_u=0.0,
        output_bias_update_width_u=0.0,
        flow_pre_store="shared_node",
        readout_flow_polarity="normal",
        readout_center_pull_width_u=0.0002,
        output_bias_center_pull_width_u=0.0001,
    )

    assert "Mvw00p_center vw00p bwd wcenter 0 NREL W=0.0002u" in updates
    assert "Mvw00n_center vw00n bwd wcenter 0 NREL W=0.0002u" in updates
    assert "Mvbo0p_center vbo0p bwd wcenter 0 NREL W=0.0001u" in updates
    assert "Mvbo0n_center vbo0n bwd wcenter 0 NREL W=0.0001u" in updates


def test_direct_flow_readout_center_pull_can_use_branch_specific_rails() -> None:
    updates = direct_flow.readout_flow_updates(
        readout_update_width_u=0.0,
        output_bias_update_width_u=0.0,
        flow_pre_store="shared_node",
        readout_flow_polarity="normal",
        readout_center_pull_width_u=0.0002,
        readout_pos_center_pull_node="pcenter",
        readout_neg_center_pull_node="ncenter",
    )

    assert "Mvw00p_center vw00p bwd pcenter 0 NREL W=0.0002u" in updates
    assert "Mvw00n_center vw00n bwd ncenter 0 NREL W=0.0002u" in updates


def test_direct_flow_readout_center_pull_can_use_apply_gate() -> None:
    updates = direct_flow.readout_flow_updates(
        readout_update_width_u=0.0,
        output_bias_update_width_u=0.0,
        flow_pre_store="shared_node",
        readout_flow_polarity="normal",
        readout_center_pull_width_u=0.0002,
        output_bias_center_pull_width_u=0.0001,
        readout_center_pull_gate="apply",
        readout_pos_center_pull_node="pcenter",
        readout_neg_center_pull_node="ncenter",
    )

    assert "Mvw00p_center vw00p apply pcenter 0 NREL W=0.0002u" in updates
    assert "Mvw00n_center vw00n apply ncenter 0 NREL W=0.0002u" in updates
    assert "Mvbo0p_center vbo0p apply wcenter 0 NREL W=0.0001u" in updates


def test_direct_flow_readout_center_pull_can_be_state_high_gated() -> None:
    updates = direct_flow.readout_flow_updates(
        readout_update_width_u=0.0,
        output_bias_update_width_u=0.0,
        flow_pre_store="shared_node",
        readout_flow_polarity="normal",
        readout_center_pull_width_u=0.0002,
        output_bias_center_pull_width_u=0.0001,
        readout_center_pull_gate="apply",
        readout_center_pull_mode="state_high",
        readout_pos_center_pull_node="pcenter",
        readout_neg_center_pull_node="ncenter",
    )

    assert "Mvw00p_center_g vw00p apply vw00p_center_g 0 NREL W=0.0002u" in updates
    assert "Mvw00p_center_s vw00p_center_g vw00p pcenter 0 NREL W=0.0002u" in updates
    assert "Mvw00n_center_g vw00n apply vw00n_center_g 0 NREL W=0.0002u" in updates
    assert "Mvw00n_center_s vw00n_center_g vw00n ncenter 0 NREL W=0.0002u" in updates
    assert "Mvbo0p_center_g vbo0p apply vbo0p_center_g 0 NREL W=0.0001u" in updates


def test_target_mistake_latch_stores_compare_result_for_late_backward_window() -> None:
    gate = direct_flow.backward_gate_cells("target_mistake_latch", width_u=64.0, cap_f=2.0)

    assert "Cmerr0 merr0 0 2f IC=0" in gate
    assert "Mmerr0_p vdd lead10 merr0_p vdd PMOS" in gate
    assert "Mmerr0_l merr0_t lead01 merr0_l" in gate
    assert "Mmerr0_c merr0_l cmp merr0" in gate
    assert "Mbwd_merr0_b bwd_merr0_a bwd_src bwd" in gate
    assert "Mmerr1_p vdd lead01 merr1_p vdd PMOS" in gate
    assert "Mmerr1_l merr1_t lead10 merr1_l" in gate
    assert "Mbwd_merr1_b bwd_merr1_a bwd_src bwd" in gate


def test_simple_target_mistake_latch_uses_short_regenerated_lead_stack() -> None:
    gate = direct_flow.backward_gate_cells(
        "target_mistake_latch_simple",
        width_u=64.0,
        cap_f=2.0,
        lead_mode="senseamp_strong",
    )

    assert "Short-stack latched mistake gate" in gate
    assert "Winner gates: class 0 uses lead01; class 1 uses lead10" in gate
    assert "Mmerr0_p" not in gate
    assert "Mmerr0_t vdd t0 merr0_t" in gate
    assert "Mmerr0_l merr0_t lead10 merr0_l" in gate
    assert "Mmerr0_c merr0_l cmp merr0" in gate
    assert "Mmerr1_t vdd t1 merr1_t" in gate
    assert "Mmerr1_l merr1_t lead01 merr1_l" in gate
    assert "Mbwd_merr0_b bwd_merr0_a bwd_src bwd" in gate
    assert "Mbwd_merr1_b bwd_merr1_a bwd_src bwd" in gate


def test_target_output_mistake_latch_samples_output_caps_during_error_window() -> None:
    gate = direct_flow.backward_gate_cells("target_out_mistake_latch", width_u=64.0, cap_f=2.0)

    assert "Cmerr0 merr0 0 2f IC=0" in gate
    assert "Mmerr0_p vdd out0 merr0_p vdd PMOS" in gate
    assert "Mmerr0_o merr0_t out1 merr0_o" in gate
    assert "Mmerr0_e merr0_o err merr0" in gate
    assert "Mmerr1_p vdd out1 merr1_p vdd PMOS" in gate
    assert "Mmerr1_o merr1_t out0 merr1_o" in gate
    assert "Mmerr1_e merr1_o err merr1" in gate
    assert "Mbwd_merr0_b bwd_merr0_a bwd_src bwd" in gate


def test_restored_target_output_mistake_latch_regenerates_backward_rail() -> None:
    gate = direct_flow.backward_gate_cells(
        "target_out_mistake_latch_restore",
        width_u=64.0,
        cap_f=2.0,
    )

    assert "Restored output-capacitor mistake latch" in gate
    assert "Cmerr0_bar merr0_bar 0 2f IC=1.2" in gate
    assert "Mreset_merr0_bar vdd rste merr0_bar 0 NREL W=4u" in gate
    assert "Mmerr0_restore merr0_bar merr0 0 0 NREL W=64u" in gate
    assert "Mbwd_merr0_p bwd_merr0_p merr0_bar vdd vdd PMOS W=64u" in gate
    assert "Mbwd_merr0_b bwd_merr0_p bwd_src bwd" in gate
    assert "Mmerr1_restore merr1_bar merr1 0 0 NREL W=64u" in gate
    assert "Mbwd_merr1_p bwd_merr1_p merr1_bar vdd vdd PMOS W=64u" in gate
    assert "Mbwd_merr1_b bwd_merr1_p bwd_src bwd" in gate


def test_stacked_restored_target_output_mistake_latch_raises_event_threshold() -> None:
    gate = direct_flow.backward_gate_cells(
        "target_out_mistake_latch_restore_stacked",
        width_u=64.0,
        cap_f=2.0,
    )

    assert "Restore discriminator: two-device event stack" in gate
    assert "Mmerr0_restore_a merr0_bar merr0 merr0_bar_a 0 NREL W=64u" in gate
    assert "Mmerr0_restore_b merr0_bar_a merr0 0 0 NREL W=64u" in gate
    assert "Mmerr1_restore_a merr1_bar merr1 merr1_bar_a 0 NREL W=64u" in gate
    assert "Mmerr1_restore_b merr1_bar_a merr1 0 0 NREL W=64u" in gate
    assert "Cmerr0_bar merr0_bar 0 2f IC=1.2" in gate
    assert "Mbwd_merr0_p bwd_merr0_p merr0_bar vdd vdd PMOS W=64u" in gate


def test_timed_stacked_restored_mistake_latch_evaluates_during_backward_window() -> None:
    gate = direct_flow.backward_gate_cells(
        "target_out_mistake_latch_restore_stacked_timed",
        width_u=64.0,
        cap_f=2.0,
    )

    assert "Restore discriminator: two-device event stack gated by bwd_src." in gate
    assert "Mmerr0_restore_t merr0_bar_b bwd_src 0 0 NREL W=64u" in gate
    assert "Mmerr1_restore_t merr1_bar_b bwd_src 0 0 NREL W=64u" in gate
    assert "Mbwd_merr0_b bwd_merr0_p bwd_src bwd" in gate


def test_probe_measurement_keeps_backward_signals_without_full_weight_snapshots() -> None:
    original_hidden = direct_flow.HIDDEN
    original_input_rails = list(direct_flow.INPUT_RAILS)
    try:
        direct_flow.set_hidden_cells(2)
        direct_flow.set_input_rails(["x0", "x1"])
        samples = [
            {"phase": "train", "label": 0, "pattern": 0, "apply_update": True},
        ]

        measures, prints = direct_flow.measure_lines(
            samples=samples,
            hidden_apply_mode="direct",
            learning_mode="flow",
            hidden_delta_output_mode="senseamp",
            measure_detail="probe",
            readout_sample_offsets_ns=[2.95],
            cmp_start_ns=3.25,
            cmp_end_ns=4.10,
            bwd_start_ns=6.75,
            apply_end_ns=11.20,
            backward_gate_mode="scheduled",
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_input_rails(original_input_rails)

    assert "print target_out_0 other_out_0 margin_0" in prints
    assert "output_delta_net_0_0" in measures
    assert "hidden_delta_net_0_0" in measures
    assert "hidden_delta_gate_net_0_0" in measures
    assert "d_vw00_signed_total" in measures
    assert "d_wh0_x0_signed_total" in measures

    assert "vw00p_before_0" not in measures
    assert "wh0_x0p_before_0" not in measures
    assert "hidden_grad_net_0_x0_0" not in measures
    assert "hidden_apply_gate_net_0_x0_0" not in measures


def test_readout_only_probe_measurement_omits_hidden_delta_nodes() -> None:
    original_hidden = direct_flow.HIDDEN
    original_input_rails = list(direct_flow.INPUT_RAILS)
    try:
        direct_flow.set_hidden_cells(2)
        direct_flow.set_input_rails(["x0", "x1"])
        samples = [
            {"phase": "train", "label": 0, "pattern": 0, "apply_update": True},
        ]

        measures, _prints = direct_flow.measure_lines(
            samples=samples,
            hidden_apply_mode="direct",
            learning_mode="flow",
            hidden_delta_output_mode="senseamp",
            measure_detail="full",
            readout_sample_offsets_ns=[2.95],
            cmp_start_ns=3.25,
            cmp_end_ns=4.10,
            bwd_start_ns=6.75,
            apply_end_ns=11.20,
            backward_gate_mode="scheduled",
            hidden_delta_network_enabled=False,
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_input_rails(original_input_rails)

    assert "output_delta_net_0_0" in measures
    assert "d_vw00_signed_0" in measures
    assert "hidden_delta_net_0_0" not in measures
    assert "hidden_delta_gate_net_0_0" not in measures
    assert "hdp0_guard_0" not in measures


def test_backward_probe_time_tracks_shortened_flow_window() -> None:
    original_hidden = direct_flow.HIDDEN
    original_input_rails = list(direct_flow.INPUT_RAILS)
    try:
        direct_flow.set_hidden_cells(1)
        direct_flow.set_input_rails(["x0"])
        samples = [
            {"phase": "train", "label": 0, "pattern": 0, "apply_update": True},
        ]

        measures, _prints = direct_flow.measure_lines(
            samples=samples,
            hidden_apply_mode="direct",
            learning_mode="flow",
            hidden_delta_output_mode="raw",
            measure_detail="probe",
            readout_sample_offsets_ns=[2.95],
            cmp_start_ns=3.25,
            cmp_end_ns=4.10,
            bwd_start_ns=8.90,
            apply_end_ns=9.20,
            backward_gate_mode="scheduled",
            hidden_delta_network_enabled=True,
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_input_rails(original_input_rails)

    assert "FIND V(bwd) AT=9.15n" in measures
    assert "FIND V(dp0) AT=9.15n" in measures
    assert "FIND V(hdp0) AT=9.15n" in measures


def test_mistake_latch_measurement_records_latched_event_caps() -> None:
    samples = [
        {"phase": "train", "label": 0, "pattern": 0, "apply_update": True},
    ]

    measures, _prints = direct_flow.measure_lines(
        samples=samples,
        hidden_apply_mode="direct",
        learning_mode="flow",
        hidden_delta_output_mode="raw",
        measure_detail="light",
        readout_sample_offsets_ns=[2.95],
        cmp_start_ns=3.25,
        cmp_end_ns=4.10,
        bwd_start_ns=6.75,
        apply_end_ns=11.20,
        backward_gate_mode="target_out_mistake_latch",
        hidden_delta_network_enabled=False,
    )

    assert "merr0_0 FIND V(merr0)" in measures
    assert "merr1_0 FIND V(merr1)" in measures


def test_restored_mistake_latch_measurement_records_inverter_nodes() -> None:
    samples = [
        {"phase": "train", "label": 0, "pattern": 0, "apply_update": True},
    ]

    measures, _prints = direct_flow.measure_lines(
        samples=samples,
        hidden_apply_mode="direct",
        learning_mode="flow",
        hidden_delta_output_mode="raw",
        measure_detail="light",
        readout_sample_offsets_ns=[2.95],
        cmp_start_ns=3.25,
        cmp_end_ns=4.10,
        bwd_start_ns=6.75,
        apply_end_ns=11.20,
        backward_gate_mode="target_out_mistake_latch_restore_stacked",
        hidden_delta_network_enabled=False,
    )

    assert "merr0_bar_0 FIND V(merr0_bar)" in measures
    assert "merr1_bar_0 FIND V(merr1_bar)" in measures


def test_error_and_target_mistake_backward_gate_are_transistor_generated() -> None:
    error = direct_flow.error_cells("out_competitive", latch_boost_width_u=0.0)
    gate = direct_flow.backward_gate_cells("target_mistake", width_u=64.0, cap_f=2.0)
    phases = direct_flow.phases(
        samples=[{"phase": "train", "apply_update": True}],
        bwd_start_ns=6.75,
        apply_start_ns=9.25,
        apply_end_ns=11.20,
        cmp_start_ns=3.25,
        cmp_end_ns=4.10,
        learning_mode="flow",
        backward_gate_mode="target_mistake",
    )

    assert "Mdp0_t0 vdd t0 dp0_t" in error
    assert "Mdp0_o0 dp0_t out1 dp0_o" in error
    assert "Mdp0_e0 dp0_o err dp0" in error
    assert "Mdn0_t0 vdd t1 dn0_t" in error
    assert "Mdn0_s0 dn0_t out0 dn0_s" in error
    assert "Mdn0_e0 dn0_s err dn0" in error

    assert "Cbwd_gate bwd 0 2f" in gate
    assert "Mbwd_t0_a bwd_t0_p t0 bwd_t0_a" in gate
    assert "Mbwd_t0_l bwd_t0_a lead01 bwd_t0_l" in gate
    assert "Mbwd_t0_b bwd_t0_l bwd_src bwd" in gate
    assert "Mbwd_t1_a bwd_t1_p t1 bwd_t1_a" in gate
    assert "Mbwd_t1_l bwd_t1_a lead10 bwd_t1_l" in gate
    assert "Mbwd_t1_b bwd_t1_l bwd_src bwd" in gate

    assert "Vbwd_src bwd_src 0" in phases
    assert "Vbwd bwd 0" not in phases


def test_target_mistake_gate_can_use_score_senseamp_polarity() -> None:
    gate = direct_flow.backward_gate_cells(
        "target_mistake",
        width_u=64.0,
        cap_f=2.0,
        lead_mode="senseamp",
    )

    assert "class 0 uses lead01; class 1 uses lead10" in gate
    assert "Mbwd_t0_p vdd lead01 bwd_t0_p" in gate
    assert "Mbwd_t0_l bwd_t0_a lead10 bwd_t0_l" in gate
    assert "Mbwd_t1_p vdd lead10 bwd_t1_p" in gate
    assert "Mbwd_t1_l bwd_t1_a lead01 bwd_t1_l" in gate


def test_target_mistake_gate_can_use_scores_directly_without_lead_latch() -> None:
    lead_cells = direct_flow.score_lead_gate_cells(lead_width_u=64.0, lead_mode="score_direct")
    gate = direct_flow.backward_gate_cells(
        "target_mistake",
        width_u=64.0,
        cap_f=2.0,
        lead_mode="score_direct",
    )

    assert "without an intermediate lead latch" in lead_cells
    assert "class 0 uses score0; class 1 uses score1" in gate
    assert "Mbwd_t0_p vdd score0 bwd_t0_p" in gate
    assert "Mbwd_t0_l bwd_t0_a score1 bwd_t0_l" in gate
    assert "Mbwd_t1_p vdd score1 bwd_t1_p" in gate
    assert "Mbwd_t1_l bwd_t1_a score0 bwd_t1_l" in gate


def test_score_charge_lead_amplifies_scores_without_discharge_path() -> None:
    lead_cells = direct_flow.score_lead_gate_cells(lead_width_u=96.0, lead_mode="score_charge")
    gate = direct_flow.backward_gate_cells(
        "target_mistake",
        width_u=64.0,
        cap_f=2.0,
        lead_mode="score_charge",
    )

    assert "Charge-only score lead" in lead_cells
    assert "Mlead01_up_s vdd score0 lead01_up" in lead_cells
    assert "Mlead10_up_s vdd score1 lead10_up" in lead_cells
    assert "lead01_dn" not in lead_cells
    assert "class 0 uses lead01; class 1 uses lead10" in gate


def test_strong_score_senseamp_adds_cross_coupled_nmos_regeneration() -> None:
    lead_cells = direct_flow.score_lead_gate_cells(lead_width_u=128.0, lead_mode="senseamp_strong")
    gate = direct_flow.backward_gate_cells(
        "target_mistake",
        width_u=64.0,
        cap_f=2.0,
        lead_mode="senseamp_strong",
    )

    assert "cross-coupled NMOS pull-downs" in lead_cells
    assert "Mlead01_nkeep lead01 lead10 0 0 NMOS" in lead_cells
    assert "Mlead10_nkeep lead10 lead01 0 0 NMOS" in lead_cells
    assert "class 0 uses lead01; class 1 uses lead10" in gate


def test_target_mistake_gate_stats_count_false_opens_and_misses() -> None:
    train = pd.DataFrame(
        [
            {"label": 0, "score0_cmp": 0.10, "score1_cmp": 0.20, "bwd_signal": 1.05},
            {"label": 1, "score0_cmp": 0.10, "score1_cmp": 0.20, "bwd_signal": 0.02},
            {"label": 0, "score0_cmp": 0.20, "score1_cmp": 0.10, "bwd_signal": 1.05},
            {"label": 1, "score0_cmp": 0.20, "score1_cmp": 0.10, "bwd_signal": 0.02},
        ]
    )

    stats = direct_flow.target_mistake_gate_stats(train, bwd_threshold_v=0.5)

    assert stats["target_mistake_score_loses_count"] == 2
    assert stats["target_mistake_bwd_open_count"] == 2
    assert stats["target_mistake_bwd_false_positive_count"] == 1
    assert stats["target_mistake_bwd_false_negative_count"] == 1
    assert stats["target_mistake_bwd_match_fraction"] == pytest.approx(0.5)
    assert stats["target_mistake_bwd_target_loses_mean_v"] == pytest.approx(0.535)
    assert stats["target_mistake_bwd_target_wins_mean_v"] == pytest.approx(0.535)
    assert stats["target_mistake_bwd_voltage_separation_v"] == pytest.approx(-1.03)
    assert stats["target_mistake_bwd_best_threshold_match_fraction"] == pytest.approx(0.5)


def test_target_mistake_latch_stats_count_channel_specific_events() -> None:
    train = pd.DataFrame(
        [
            {"label": 0, "score0_cmp": 0.10, "score1_cmp": 0.20, "merr0": 1.05, "merr1": 0.02},
            {"label": 1, "score0_cmp": 0.10, "score1_cmp": 0.20, "merr0": 0.02, "merr1": 0.02},
            {"label": 0, "score0_cmp": 0.20, "score1_cmp": 0.10, "merr0": 1.05, "merr1": 0.02},
            {"label": 1, "score0_cmp": 0.20, "score1_cmp": 0.10, "merr0": 0.02, "merr1": 0.02},
        ]
    )

    stats = direct_flow.target_mistake_latch_stats(train, latch_threshold_v=0.5)

    assert stats["target_mistake_latch_open_count"] == 2
    assert stats["target_mistake_latch_false_positive_count"] == 1
    assert stats["target_mistake_latch_false_negative_count"] == 1
    assert stats["target_mistake_latch_match_fraction"] == pytest.approx(0.5)
    assert stats["target_mistake_latch_best_threshold_match_fraction"] == pytest.approx(0.5)


def test_out_residual_error_uses_each_outputs_own_stored_score() -> None:
    error = direct_flow.error_cells(
        "out_residual",
        latch_boost_width_u=0.0,
        residual_target_width_u=120.0,
        residual_output_width_u=80.0,
    )

    assert "Mdp0_t0 vdd t0 dp0_t" in error
    assert "Mdp0_y1 dp0_y out0 0" in error
    assert "Mdn0_y0 vdd out0 dn0_y" in error
    assert "Mdn0_t1 dn0_t t0 0" in error
    assert "Mdp1_t0 vdd t1 dp1_t" in error
    assert "Mdp1_y1 dp1_y out1 0" in error
    assert "Mdn1_y0 vdd out1 dn1_y" in error
    assert "Mdn1_t1 dn1_t t1 0" in error
    assert "Mdp0_t0 vdd t0 dp0_t 0 NSENSE W=120u" in error
    assert "Mdp0_y0 dp0 err dp0_y 0 NSENSE W=80u" in error
    assert "Mdp0_o0 dp0_t out1" not in error
    assert "Mdn1_s0 dn1_t out1" not in error


def test_mnist_fixed_sensory_frontends_are_python_preprocessors() -> None:
    image = np.arange(64, dtype=np.float64).reshape(8, 8) / 63.0

    fixed32, fixed_desc = direct_flow.mnist01_frontend(image, "fixed32")
    sensory48, sensory_desc = direct_flow.mnist01_frontend(image, "sensory48")
    sensory64, _ = direct_flow.mnist01_frontend(image, "sensory64")

    assert fixed32.shape == (32,)
    assert sensory48.shape == (48,)
    assert sensory64.shape == (64,)
    assert "dct" in fixed_desc
    assert "random_local_relu" in sensory_desc


def test_random_readout_init_can_be_centered_in_measured_mobility_window() -> None:
    original_hidden = direct_flow.HIDDEN
    try:
        direct_flow.set_hidden_cells(3)
        init = direct_flow.readout_init(
            seed=7,
            mode="random",
            separator_scale=0.0,
            separator_offset_v=0.0,
            readout_center_v=0.62,
            random_center_v=0.34,
            random_span_v=0.20,
            random_pos_center_v=None,
            random_neg_center_v=None,
            random_pos_span_v=None,
            random_neg_span_v=None,
            separator_csv=None,
            separator_phase="final_eval",
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)

    assert set(init) == {
        "vbo0p",
        "vbo0n",
        "vbo1p",
        "vbo1n",
        "vw00p",
        "vw00n",
        "vw01p",
        "vw01n",
        "vw02p",
        "vw02n",
        "vw10p",
        "vw10n",
        "vw11p",
        "vw11n",
        "vw12p",
        "vw12n",
    }
    assert min(init.values()) >= 0.24 - 1e-12
    assert max(init.values()) <= 0.44 + 1e-12


def test_random_readout_init_can_center_positive_and_negative_branches_separately() -> None:
    original_hidden = direct_flow.HIDDEN
    try:
        direct_flow.set_hidden_cells(2)
        init = direct_flow.readout_init(
            seed=2,
            mode="random",
            separator_scale=0.0,
            separator_offset_v=0.0,
            readout_center_v=0.62,
            random_center_v=None,
            random_span_v=0.20,
            random_pos_center_v=0.44,
            random_neg_center_v=0.13,
            random_pos_span_v=0.08,
            random_neg_span_v=0.04,
            separator_csv=None,
            separator_phase="final_eval",
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)

    positive_values = [value for key, value in init.items() if key.endswith("p")]
    negative_values = [value for key, value in init.items() if key.endswith("n")]
    assert min(positive_values) >= 0.40 - 1e-12
    assert max(positive_values) <= 0.48 + 1e-12
    assert min(negative_values) >= 0.11 - 1e-12
    assert max(negative_values) <= 0.15 + 1e-12


def test_lead_score_winner_metric_uses_out_senseamp_polarity() -> None:
    lead01 = np.array([0.1, 1.0])
    lead10 = np.array([1.0, 0.1])

    assert direct_flow.lead_class0_wins("score", lead01, lead10).tolist() == [False, True]
    assert direct_flow.lead_class0_wins("senseamp", lead01, lead10).tolist() == [False, True]
    assert direct_flow.lead_class0_wins("out_senseamp", lead01, lead10).tolist() == [True, False]


def test_input_and_target_pwl_sources_have_strictly_increasing_times() -> None:
    samples = [
        {"phase": "train", "pattern": 0, "label": 0, "apply_update": True},
        {"phase": "train", "pattern": 1, "label": 1, "apply_update": True},
        {"phase": "train", "pattern": 2, "label": 1, "apply_update": True},
    ]
    stop_ns = len(samples) * direct_flow.CYCLE_NS

    input_times = pwl_times(direct_flow.sample_wave(samples, "x0", stop_ns))
    target_times = pwl_times(direct_flow.target_wave(samples, 0, stop_ns))

    assert all(right > left for left, right in zip(input_times, input_times[1:]))
    assert all(right > left for left, right in zip(target_times, target_times[1:]))
