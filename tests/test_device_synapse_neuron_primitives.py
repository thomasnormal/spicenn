from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPICE_DIR = ROOT / "spice"
if str(SPICE_DIR) not in sys.path:
    sys.path.insert(0, str(SPICE_DIR))

import run_device_relu_synapse_sweep as relu_primitives  # noqa: E402
import run_device_signed_learning_cell as signed_primitives  # noqa: E402
import run_device_xor2_random_hidden as direct_flow  # noqa: E402
from run_spice_sweep import detect_spice, run_tiny_test  # noqa: E402


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
        )
        hidden_updates = direct_flow.hidden_flow_updates(
            update_width_u=0.04,
            flow_pre_store="synapse_consume",
            hidden_delta_output_mode="senseamp",
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

    assert "Mwh0_x0n_flow_b wh0_x0n bwd" in hidden_updates
    assert "Mwh0_x0n_flow_x wh0_x0n_flow_b fphi0_x0" in hidden_updates
    assert "Mwh0_x0n_flow_d wh0_x0n_flow_x hdpg0" in hidden_updates
    assert "Mwh0_x0n_flow_a wh0_x0n_flow_d apply 0 0" in hidden_updates
    assert "Mwh0_x0p_flow_x wh0_x0p_flow_b fphi0_x0" in hidden_updates
    assert "Mwh0_x0p_flow_d wh0_x0p_flow_x hdng0" in hidden_updates


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
