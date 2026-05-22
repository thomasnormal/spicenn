from __future__ import annotations

import re
import subprocess
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
import run_device_readout_transfer as readout_transfer  # noqa: E402
import run_device_readout_branch_surface as readout_branch_surface  # noqa: E402
import run_device_readout_capfit as readout_capfit  # noqa: E402
import run_device_readout_error_loop as readout_error_loop  # noqa: E402
import run_device_output_head_transfer as output_head_transfer  # noqa: E402
import run_device_readout_array_eval as readout_array_eval  # noqa: E402
import run_device_readout_array_capfit as readout_array_capfit  # noqa: E402
import run_device_readout_write_selectivity as readout_write_selectivity  # noqa: E402
import run_device_softmax_current_competition as softmax_primitives  # noqa: E402
import run_device_write_rail_exclusion_sweep as write_rail_primitives  # noqa: E402
import run_device_xor2_two_hidden as two_hidden_probe  # noqa: E402
import run_device_xor2_random_hidden as direct_flow  # noqa: E402
import spice_adapter  # noqa: E402
import run_spice_sweep  # noqa: E402
from run_spice_sweep import canonical_circuit_netlist, detect_spice, prepare_netlist_for_simulator, run_tiny_test  # noqa: E402
from spice_adapter import raw_netlist_to_spice_deck, render_spice_deck, simulator_kind  # noqa: E402


PWL_TIME_RE = re.compile(r"([-+0-9.eE]+)n")


def pwl_times(wave: str) -> list[float]:
    return [float(match) for match in PWL_TIME_RE.findall(wave)]


def test_spice_accuracy_presets_expose_fast_screening_mode() -> None:
    assert "reltol" not in direct_flow.spice_options_for_preset("standard")
    fast = direct_flow.spice_options_for_preset("fast")
    loose = direct_flow.spice_options_for_preset("loose")

    assert "reltol=3e-3" in fast
    assert "rshunt=1e11" in fast
    assert "reltol=1e-2" in loose
    with pytest.raises(ValueError, match="unknown SPICE accuracy preset"):
        direct_flow.spice_options_for_preset("invalid")


def test_pmos_charge_write_defaults_to_true_high_side_rail() -> None:
    assert direct_flow.default_readout_write_high_v("bounded_cmos_charge_discharge") == pytest.approx(1.0)
    assert direct_flow.default_readout_write_high_v("bounded_pmos_charge_only") == pytest.approx(1.0)
    assert direct_flow.default_readout_write_high_v("bounded_pmos_charge_discharge") == pytest.approx(1.0)
    assert direct_flow.default_readout_write_high_v("bounded_charge_discharge") == pytest.approx(0.58)


def test_direct_flow_balanced_random_fanout_balances_readout_rows() -> None:
    old_hidden = direct_flow.HIDDEN
    old_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(16)
        direct_flow.set_output_count(5)
        fanins = direct_flow.build_readout_fanins(
            "balanced_random_fanout",
            fan_in=3,
            fan_out=3,
            seed=7,
        )
        summary = direct_flow.readout_topology_summary(fanins)

        assert summary["readout_edge_count"] == 48
        assert sorted(summary["readout_fanin_counts"]) == [9, 9, 10, 10, 10]
        assert summary["readout_fanout_counts"] == [3] * 16
    finally:
        direct_flow.set_hidden_cells(old_hidden)
        direct_flow.set_output_count(old_outputs)


def test_selected_decision_can_use_hardware_score_rails_instead_of_output_caps() -> None:
    out_decision = direct_flow.selected_decision(
        decision_source="out",
        output_target=0.30,
        output_other=0.70,
        output_predicted_label=2,
        score_target=0.90,
        score_other=0.40,
        score_predicted_label=1,
    )
    score_decision = direct_flow.selected_decision(
        decision_source="score",
        output_target=0.30,
        output_other=0.70,
        output_predicted_label=2,
        score_target=0.90,
        score_other=0.40,
        score_predicted_label=1,
    )

    assert out_decision == pytest.approx((0.30, 0.70, -0.40, 2))
    assert score_decision == pytest.approx((0.90, 0.40, 0.50, 1))
    with pytest.raises(ValueError, match="unknown decision source"):
        direct_flow.selected_decision(
            decision_source="python",
            output_target=0.0,
            output_other=0.0,
            output_predicted_label=0,
            score_target=0.0,
            score_other=0.0,
            score_predicted_label=0,
        )


def test_column_centering_metrics_exposes_common_mode_separator() -> None:
    source = np.array(
        [
            [1.10, 1.00],
            [1.20, 1.25],
            [1.05, 0.95],
            [1.15, 1.30],
        ]
    )
    labels = np.array([0, 1, 0, 1])

    metrics = direct_flow.column_centering_metrics("score", source, labels)

    assert metrics["score_mean_by_output_v"] == pytest.approx([1.125, 1.125])
    assert metrics["score_column_centered_accuracy"] == pytest.approx(1.0)
    assert metrics["score_column_centered_min_margin_v"] > 0.0


def test_output_decision_matrix_reconstructs_binary_target_other_rows() -> None:
    rows = pd.DataFrame(
        [
            {"label": 0, "output_target": 0.8, "output_other": 0.2},
            {"label": 1, "output_target": 0.7, "output_other": 0.1},
        ]
    )

    matrix = direct_flow.output_decision_matrix(rows, 2, low_true_output=False)

    assert matrix == pytest.approx(np.array([[0.8, 0.2], [0.1, 0.7]]))


def test_output_delta_alignment_metrics_detect_correct_sample_overwrite() -> None:
    train = pd.DataFrame(
        [
            {
                "label": 0,
                "score_correct": True,
                "output_delta_net_0": 0.20,
                "output_delta_net_1": -0.01,
                "output_delta_net_2": -0.02,
            },
            {
                "label": 2,
                "score_correct": False,
                "output_delta_net_0": -0.03,
                "output_delta_net_1": -0.04,
                "output_delta_net_2": 0.50,
            },
        ]
    )

    metrics = direct_flow.output_delta_alignment_metrics(train, output_count=3)

    assert metrics["train_output_delta_sign_alignment_fraction"] == pytest.approx(1.0)
    assert metrics["train_output_delta_target_gt_all_others_fraction"] == pytest.approx(1.0)
    assert metrics["train_output_delta_target_positive_fraction"] == pytest.approx(1.0)
    assert metrics["train_output_delta_other_negative_fraction"] == pytest.approx(1.0)
    assert metrics["train_output_delta_all_other_negative_fraction"] == pytest.approx(1.0)
    assert metrics["train_output_delta_target_minus_max_other_mean_v"] == pytest.approx(0.37)
    assert metrics["train_output_delta_correct_target_mean_v"] == pytest.approx(0.20)
    assert metrics["train_output_delta_wrong_target_mean_v"] == pytest.approx(0.50)
    assert metrics["train_output_delta_wrong_to_correct_target_ratio"] == pytest.approx(2.5)
    assert metrics["train_output_delta_target_mistake_gain_v"] == pytest.approx(0.30)

    assert direct_flow.output_delta_sums_by_out(train, output_count=3) == pytest.approx([0.17, -0.05, 0.48])
    assert direct_flow.signed_alignment_fraction([0.17, -0.05, 0.48], [1.0, -1.0, 2.0]) == pytest.approx(1.0)
    assert direct_flow.signed_alignment_fraction([0.17, -0.05, 0.48], [-1.0, -1.0, 2.0]) == pytest.approx(2 / 3)
    assert direct_flow.cosine_similarity([1.0, 0.0, 0.0], [2.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_xyce_netlist_runner_strips_ngspice_control_block(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, str] = {}

    class Result:
        returncode = 0
        stdout = "foo = 1.0\n"
        stderr = ""

    def fake_run(spice_bin: str, netlist: Path, *, timeout: float) -> Result:
        cmd = run_spice_sweep.spice_batch_command(spice_bin, netlist)
        captured["cmd"] = " ".join(cmd)
        captured["netlist"] = Path(cmd[-1]).read_text()
        return Result()

    monkeypatch.setattr("spice_adapter._run_simulator_process", fake_run)

    parsed = direct_flow.run_netlist(
        "Xyce",
        tmp_path / "deck.cir",
        ".tran 1p 1n\n.control\nrun\nprint v(x)\n.endc\n.end\n",
        timeout=1.0,
    )

    assert parsed == {"foo": 1.0}
    assert ".control" not in captured["netlist"]
    assert ".endc" not in captured["netlist"]
    assert captured["cmd"].startswith("Xyce ")


def test_tiny_spice_smoke_uses_simulator_specific_netlist_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    written: dict[str, str] = {}

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(spice_bin: str, netlist: Path, *, timeout: float) -> Result:
        cmd = run_spice_sweep.spice_batch_command(spice_bin, netlist)
        path = Path(cmd[-1])
        written[path.name] = path.read_text()
        return Result()

    monkeypatch.setattr("spice_adapter._run_simulator_process", fake_run)
    run_tiny_test("ngspice", tmp_path)
    run_tiny_test("Xyce", tmp_path)

    assert any(name.startswith("tiny_test_ngspice_") for name in written)
    assert any(name.startswith("tiny_test_xyce_") for name in written)
    assert any(".control" in text for name, text in written.items() if name.startswith("tiny_test_ngspice_"))
    assert all(".control" not in text for name, text in written.items() if name.startswith("tiny_test_xyce_"))


def test_simulator_deck_preparation_preserves_canonical_circuit_body() -> None:
    netlist = "* deck\nV1 in 0 DC 1\nR1 in 0 1k\n.tran 1p 1n\n.control\nrun\nprint v(in)\n.endc\n.end\n"
    ng = prepare_netlist_for_simulator(netlist, "ngspice")
    xy = prepare_netlist_for_simulator(netlist, "Xyce")

    assert ng.startswith(".title spicenn generated deck")
    assert xy.startswith(".title spicenn generated deck")
    assert ".control" in ng
    assert ".control" not in xy
    assert canonical_circuit_netlist(ng) == canonical_circuit_netlist(xy)


def test_spice_raw_deck_adapter_keeps_control_block_separate() -> None:
    netlist = "* deck\nV1 in 0 DC 1\nR1 in 0 1k\n.op\n.control\nrun\nprint v(in)\n.endc\n.end\n"
    deck = raw_netlist_to_spice_deck(netlist)
    no_control = render_spice_deck(deck, include_control=False)
    with_control = render_spice_deck(deck, include_control=True)

    assert simulator_kind("/opt/bin/ngspice") == "ngspice"
    assert simulator_kind("/opt/bin/Xyce") == "xyce"
    assert simulator_kind("/opt/bin/XyceNF") == "xyce"
    assert no_control.startswith(".title spicenn generated deck")
    assert ".control" not in no_control
    assert ".control" in with_control
    assert canonical_circuit_netlist(no_control) == canonical_circuit_netlist(with_control)


def test_spicelib_resolves_executable_without_forcing_raw_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    netlist = tmp_path / "deck.cir"
    netlist.write_text(".title deck\n.end\n")

    class Simulator:
        spice_exe = ["/sim/Xyce"]

    monkeypatch.setattr(spice_adapter, "spicelib_simulator", lambda spice_bin: Simulator)

    cmd = spice_adapter.spice_batch_command("Xyce", netlist)

    assert cmd == ["/sim/Xyce", netlist.as_posix()]
    assert "-r" not in cmd
    assert "-l" not in cmd


def test_simulator_auto_modes_make_fast_xyce_choice_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    versions = {
        "/sim/ngspice": "ngspice-46 : Circuit level simulation program\n",
        "/sim/Xyce": "Xyce Release 7.10.0\n",
    }

    def fake_which(name: str) -> str | None:
        return {"ngspice": "/sim/ngspice", "Xyce": "/sim/Xyce"}.get(name)

    class Result:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout
            self.stderr = ""

    def fake_run(cmd: list[str], text: bool, capture_output: bool, timeout: float) -> Result:
        return Result(versions[cmd[0]])

    monkeypatch.setattr(run_spice_sweep.shutil, "which", fake_which)
    monkeypatch.setattr(run_spice_sweep.subprocess, "run", fake_run)
    monkeypatch.delenv(run_spice_sweep.SPICE_SIMULATOR_ENV, raising=False)

    compat_path, compat_version = run_spice_sweep.detect_spice("auto")
    fast_path, fast_version = run_spice_sweep.detect_spice("auto-fast")

    assert compat_path == "/sim/ngspice"
    assert "ngspice" in compat_version
    assert fast_path == "/sim/Xyce"
    assert "Xyce" in fast_version


def test_simulator_env_can_select_fast_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_which(name: str) -> str | None:
        return {"ngspice": "/sim/ngspice", "Xyce": "/sim/Xyce"}.get(name)

    class Result:
        stdout = "Xyce Release 7.10.0\n"
        stderr = ""

    monkeypatch.setattr(run_spice_sweep.shutil, "which", fake_which)
    monkeypatch.setattr(run_spice_sweep.subprocess, "run", lambda *args, **kwargs: Result())
    monkeypatch.setenv(run_spice_sweep.SPICE_SIMULATOR_ENV, "auto-fast")

    path, _version = run_spice_sweep.detect_spice(None)

    assert path == "/sim/Xyce"


def test_two_hidden_direct_script_help_can_import_spicenn() -> None:
    proc = subprocess.run(
        [sys.executable, "spice/run_device_xor2_two_hidden.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert proc.returncode == 0, proc.stderr
    assert "--simulator" in proc.stdout


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


def test_signed_learning_cell_exclusive_delta_gate_blocks_conflicting_error_rails(
    spice_bin: str,
    tmp_path: Path,
) -> None:
    ungated = signed_primitives.run_netlist(
        spice_bin,
        tmp_path / "signed_conflict_ungated.cir",
        signed_primitives.signed_learning_cell_netlist(
            vin=0.8,
            delta_p=1.1,
            delta_n=1.1,
            wp_init=0.95,
            wn_init=0.25,
        ),
        timeout=20.0,
    )
    gated = signed_primitives.run_netlist(
        spice_bin,
        tmp_path / "signed_conflict_gated.cir",
        signed_primitives.signed_learning_cell_netlist(
            vin=0.8,
            delta_p=1.1,
            delta_n=1.1,
            wp_init=0.95,
            wn_init=0.25,
            exclusive_delta_gate=True,
        ),
        timeout=20.0,
    )

    assert ungated["d_signed"] < -0.25
    assert abs(gated["d_signed"]) < 1e-3
    assert gated["gp_after_acc"] < 1e-3
    assert gated["gn_after_acc"] < 1e-3


def test_write_rail_exclusion_cell_selects_one_signed_error_rail(
    spice_bin: str,
    tmp_path: Path,
) -> None:
    positive = write_rail_primitives.run_netlist(
        spice_bin,
        tmp_path / "write_rail_positive.cir",
        write_rail_primitives.write_rail_exclusion_netlist(
            dp_v=1.1,
            dn_v=0.2,
            width_u=8.0,
        ),
        timeout=20.0,
    )
    conflict = write_rail_primitives.run_netlist(
        spice_bin,
        tmp_path / "write_rail_conflict.cir",
        write_rail_primitives.write_rail_exclusion_netlist(
            dp_v=1.1,
            dn_v=1.1,
            width_u=8.0,
        ),
        timeout=20.0,
    )

    assert positive["pos_late"] > 0.8
    assert positive["neg_late"] < 0.05
    assert conflict["pos_late"] < 0.05
    assert conflict["neg_late"] < 0.05


def test_write_rail_overlap_decay_cell_detects_ambiguous_error_overlap(
    spice_bin: str,
    tmp_path: Path,
) -> None:
    selected = write_rail_primitives.run_netlist(
        spice_bin,
        tmp_path / "write_rail_overlap_selected.cir",
        write_rail_primitives.write_rail_exclusion_netlist(
            dp_v=1.1,
            dn_v=0.2,
            width_u=8.0,
            overlap_decay=True,
        ),
        timeout=20.0,
    )
    conflict = write_rail_primitives.run_netlist(
        spice_bin,
        tmp_path / "write_rail_overlap_conflict.cir",
        write_rail_primitives.write_rail_exclusion_netlist(
            dp_v=1.1,
            dn_v=1.1,
            width_u=8.0,
            overlap_decay=True,
        ),
        timeout=20.0,
    )

    assert selected["pos_late"] > 0.8
    assert selected["neg_late"] < 0.05
    assert selected["ov_late"] < 0.05
    assert conflict["pos_late"] < 0.05
    assert conflict["neg_late"] < 0.05
    assert conflict["ov_late"] > 0.55


def test_diffpair_bleed_write_rail_cell_suppresses_common_mode_error(
    spice_bin: str,
    tmp_path: Path,
) -> None:
    selected = write_rail_primitives.run_netlist(
        spice_bin,
        tmp_path / "write_rail_diffpair_selected.cir",
        write_rail_primitives.write_rail_exclusion_netlist(
            dp_v=1.03,
            dn_v=0.0,
            width_u=8.0,
            selector_mode="diffpair_bleed",
        ),
        timeout=20.0,
    )
    conflict = write_rail_primitives.run_netlist(
        spice_bin,
        tmp_path / "write_rail_diffpair_conflict.cir",
        write_rail_primitives.write_rail_exclusion_netlist(
            dp_v=1.03,
            dn_v=1.03,
            width_u=8.0,
            selector_mode="diffpair_bleed",
        ),
        timeout=20.0,
    )

    assert selected["pos_late"] > 0.45
    assert selected["neg_late"] < 0.05
    assert conflict["pos_late"] < 0.05
    assert conflict["neg_late"] < 0.05


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
            internal_reset_width_u=0.0,
        )
        reset_delta = direct_flow.hidden_delta(
            hidden_error_rule="backprop",
            hidden_delta_relu_gate="act_nsense",
            hidden_delta_weight_device="nmos",
            design=design,
            internal_cap_f=0.02,
            internal_leak_ohm=1e9,
            internal_reset_width_u=4.0,
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
    assert "Mreset_hdp_a00_0 hdp_a00_0 rste 0 0 NMOS W=4u" in reset_delta
    assert "Mreset_hdp_a00_1 hdp_a00_1 rste 0 0 NMOS W=4u" in reset_delta
    assert "Mreset_hdp_a00_2 hdp_a00_2 rste 0 0 NMOS W=4u" in reset_delta

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


def test_label_shuffle_preserves_inputs_and_label_multiset() -> None:
    records = [
        {"pattern": 0, "label": 0, "inputs": {"x0": 0.1}},
        {"pattern": 1, "label": 0, "inputs": {"x0": 0.2}},
        {"pattern": 2, "label": 1, "inputs": {"x0": 0.8}},
        {"pattern": 3, "label": 1, "inputs": {"x0": 0.9}},
    ]

    shuffled = direct_flow.label_shuffled_records(records, seed=3)

    assert [record["inputs"] for record in shuffled] == [record["inputs"] for record in records]
    assert sorted(record["label"] for record in shuffled) == [0, 0, 1, 1]
    assert [record["true_label"] for record in shuffled] == [0, 0, 1, 1]
    assert [record["label"] for record in shuffled] != [0, 0, 1, 1]
    assert all(record["label_shuffle_seed"] == 3 for record in shuffled)


def test_counted_mnist_dataset_parser_supports_small_multiclass_runs() -> None:
    assert direct_flow.parse_counted_mnist_dataset("mnist3fixed8_30") == (3, "fixed8", 30)
    assert direct_flow.parse_counted_mnist_dataset("mnist5sensory64_50") == (5, "sensory64", 50)
    assert direct_flow.parse_counted_mnist_dataset("mnist10_100") == (10, "fixed8", 100)
    assert direct_flow.parse_counted_mnist_dataset("mnistfixed8_100") == (10, "fixed8", 100)
    assert direct_flow.parse_counted_mnist_dataset("mnist01fixed8_16") is None


def test_composed_flow_updates_can_use_exclusive_error_write_rails() -> None:
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    original_input_rails = list(direct_flow.INPUT_RAILS)
    try:
        direct_flow.set_hidden_cells(1)
        direct_flow.set_output_count(2)
        direct_flow.set_input_rails(["x0"])
        readout_updates = direct_flow.readout_flow_updates(
            readout_update_width_u=0.04,
            output_bias_update_width_u=0.04,
            flow_pre_store="synapse_gate",
            readout_flow_polarity="normal",
            readout_flow_write_mode="bounded_charge_discharge",
            write_error_exclusion="pmos_inhibit",
            write_error_exclusion_width_u=7.0,
        )
        hidden_updates = direct_flow.hidden_flow_updates(
            update_width_u=0.04,
            flow_pre_store="synapse_gate",
            hidden_delta_output_mode="raw",
            hidden_flow_write_mode="bounded_charge_discharge",
            write_error_exclusion="pmos_inhibit",
            write_error_exclusion_width_u=5.0,
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)
        direct_flow.set_input_rails(original_input_rails)

    assert "Mrwpos0_inh vdd dn0 rwpos0_src vdd PMOS W=7u" in readout_updates
    assert "Mrwpos0_gate rwpos0_src dp0 rwpos0 0 NSENSE W=7u" in readout_updates
    assert "Mrwneg0_inh vdd dp0 rwneg0_src vdd PMOS W=7u" in readout_updates
    assert "Mrwpos0_kill rwpos0 dn0 0 0 NMOS W=7u" in readout_updates
    assert "Mrwneg0_kill rwneg0 dp0 0 0 NMOS W=7u" in readout_updates
    assert "Mvw00n_flow_d vw00n_flow_a rwpos0 wlow" in readout_updates
    assert "Mvw00p_flow_d vw00p_flow_a rwneg0 wlow" in readout_updates
    assert "Mvw00p_ch_d vw00p_ch_a rwpos0 vw00p" in readout_updates
    assert "Mvw00n_ch_d vw00n_ch_a rwneg0 vw00n" in readout_updates

    assert "Mhwpos0_inh vdd hdn0 hwpos0_src vdd PMOS W=5u" in hidden_updates
    assert "Mhwpos0_gate hwpos0_src hdp0 hwpos0 0 NSENSE W=5u" in hidden_updates
    assert "Mhwneg0_inh vdd hdp0 hwneg0_src vdd PMOS W=5u" in hidden_updates
    assert "Mhwpos0_kill hwpos0 hdn0 0 0 NMOS W=5u" in hidden_updates
    assert "Mhwneg0_kill hwneg0 hdp0 0 0 NMOS W=5u" in hidden_updates
    assert "Mwh0_x0n_flow_d wh0_x0n_flow_x hwpos0 wlow" in hidden_updates
    assert "Mwh0_x0p_flow_d wh0_x0p_flow_x hwneg0 wlow" in hidden_updates
    assert "Mwh0_x0p_ch_d wh0_x0p_ch_x hwpos0 wh0_x0p" in hidden_updates
    assert "Mwh0_x0n_ch_d wh0_x0n_ch_x hwneg0 wh0_x0n" in hidden_updates


def test_composed_readout_can_use_exclusive_signed_rails_with_overlap_decay() -> None:
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    original_input_rails = list(direct_flow.INPUT_RAILS)
    try:
        direct_flow.set_hidden_cells(1)
        direct_flow.set_output_count(2)
        direct_flow.set_input_rails(["x0"])
        updates = direct_flow.readout_flow_updates(
            readout_update_width_u=0.04,
            output_bias_update_width_u=0.04,
            flow_pre_store="synapse_gate",
            readout_flow_polarity="normal",
            readout_flow_write_mode="bounded_discharge",
            write_error_exclusion="pmos_inhibit_decay",
            write_error_exclusion_width_u=9.0,
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)
        direct_flow.set_input_rails(original_input_rails)

    assert "Mrwpos0_kill rwpos0 dn0 0 0 NMOS W=9u" in updates
    assert "Mrwov0_p vdd dp0 rwov0_mid 0 NMOS W=9u" in updates
    assert "Mrwov0_n rwov0_mid dn0 rwov0 0 NMOS W=9u" in updates
    assert "Mvw00n_flow_d vw00n_flow_a rwpos0 wlow" in updates
    assert "Mvw00p_flow_d vw00p_flow_a rwneg0 wlow" in updates
    assert "Mvw00p_ov_d vw00p_ov_a rwov0 wlow" in updates
    assert "Mvw00n_ov_d vw00n_ov_a rwov0 wlow" in updates
    assert "Mvbo0p_ov_d vbo0p_ov_b rwov0 wlow" in updates
    assert "Mvbo0n_ov_d vbo0n_ov_b rwov0 wlow" in updates


def test_readout_and_hidden_exclusive_write_rails_can_be_selected_independently() -> None:
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    original_input_rails = list(direct_flow.INPUT_RAILS)
    try:
        direct_flow.set_hidden_cells(1)
        direct_flow.set_output_count(2)
        direct_flow.set_input_rails(["x0"])
        readout_updates = direct_flow.readout_flow_updates(
            readout_update_width_u=0.04,
            output_bias_update_width_u=0.0,
            flow_pre_store="synapse_gate",
            readout_flow_polarity="normal",
            write_error_exclusion="none",
        )
        hidden_updates = direct_flow.hidden_flow_updates(
            update_width_u=0.04,
            flow_pre_store="synapse_gate",
            hidden_delta_output_mode="raw",
            hidden_flow_write_mode="bounded_discharge",
            write_error_exclusion="pmos_inhibit",
            write_error_exclusion_width_u=11.0,
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)
        direct_flow.set_input_rails(original_input_rails)

    assert "rwpos" not in readout_updates
    assert "rwneg" not in readout_updates
    assert "Mvw00n_flow_d vw00n_flow_a dp0 0" in readout_updates

    assert "Mhwpos0_inh vdd hdn0 hwpos0_src vdd PMOS W=11u" in hidden_updates
    assert "Mhwneg0_inh vdd hdp0 hwneg0_src vdd PMOS W=11u" in hidden_updates
    assert "Mwh0_x0n_flow_d wh0_x0n_flow_x hwpos0 wlow" in hidden_updates


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


def test_direct_flow_synapse_boost_creates_mos_generated_write_gates() -> None:
    original_hidden = direct_flow.HIDDEN
    original_input_rails = list(direct_flow.INPUT_RAILS)
    try:
        direct_flow.set_hidden_cells(1)
        direct_flow.set_input_rails(["x0"])
        stores = direct_flow.flow_pre_activation_stores(
            mode="synapse_boost",
            cap_f=2.0,
            consume_width_u=0.05,
            boost_width_u=3.5,
        )
        readout_updates = direct_flow.readout_flow_updates(
            readout_update_width_u=0.02,
            output_bias_update_width_u=0.0,
            flow_pre_store="synapse_boost",
            readout_flow_polarity="normal",
        )
        hidden_updates = direct_flow.hidden_flow_updates(
            update_width_u=0.02,
            flow_pre_store="synapse_boost",
            hidden_delta_output_mode="senseamp",
            hidden_flow_write_mode="discharge",
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_input_rails(original_input_rails)

    assert "Mstore_fpro00 fpro00 fwd act0 0 NREL" in stores
    assert "Cfprb00 fprb00 0 2f IC=0" in stores
    assert "Cboost_fprb00 preboost fprb00 2f" in stores
    assert "Mstore_fprb00 fprb00 fwd act0 0 NREL W=3.5u" in stores
    assert "Cboost_fphib0_x0 preboost fphib0_x0 2f" in stores
    assert "Mstore_fphib0_x0 fphib0_x0 fwd x0 0 NREL W=3.5u" in stores
    assert "Mconsume_fpro00" not in stores
    assert "Mvw00n_flow_a vw00n_flow_b fprb00" in readout_updates
    assert "Mwh0_x0n_flow_x wh0_x0n_flow_b fphib0_x0" in hidden_updates


def test_direct_flow_synapse_spike_creates_full_swing_write_gates() -> None:
    original_hidden = direct_flow.HIDDEN
    original_input_rails = list(direct_flow.INPUT_RAILS)
    try:
        direct_flow.set_hidden_cells(1)
        direct_flow.set_input_rails(["x0"])
        stores = direct_flow.flow_pre_activation_stores(
            mode="synapse_spike",
            cap_f=2.0,
            consume_width_u=0.05,
            boost_width_u=3.5,
        )
        readout_updates = direct_flow.readout_flow_updates(
            readout_update_width_u=0.02,
            output_bias_update_width_u=0.0,
            flow_pre_store="synapse_spike",
            readout_flow_polarity="normal",
            readout_flow_write_mode="bounded_charge_only",
        )
        hidden_updates = direct_flow.hidden_flow_updates(
            update_width_u=0.02,
            flow_pre_store="synapse_spike",
            hidden_delta_output_mode="senseamp",
            hidden_flow_write_mode="bounded_discharge",
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_input_rails(original_input_rails)

    assert "Cfprg00 fprg00 0 2f IC=0" in stores
    assert "Cfprbar00 fprbar00 0 2f IC=1.2" in stores
    assert "Mreset_fprg00 fprg00 rstf 0 0 NMOS W=4u" in stores
    assert "Mreset_fprbar00 vdd rstf fprbar00 0 NSENSE W=4u" in stores
    assert "Mspike_fprbar00_fwd fprbar00 fwd fprm00 0 NREL W=3.5u" in stores
    assert "Mspike_fprbar00_act fprm00 act0 spikeref 0 NSENSE W=3.5u" in stores
    assert "Mspike_fprg00_p vdd fprbar00 fprg00 vdd PMOS W=3.5u" in stores
    assert "Mspike_fprg00_n fprg00 fprbar00 0 0 NMOS W=3.5u" in stores
    assert "Cfphig0_x0 fphig0_x0 0 2f IC=0" in stores
    assert "Mspike_fphibar0_x0_act fphim0_x0 x0 spikeref 0 NSENSE W=3.5u" in stores
    assert "Mvw00p_ch_a vw00p_ch_b fprg00" in readout_updates
    assert "Mwh0_x0n_flow_x wh0_x0n_flow_b fphig0_x0" in hidden_updates


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


def test_direct_flow_hidden_can_use_diffpair_bleed_write_selector() -> None:
    original_hidden = direct_flow.HIDDEN
    original_input_rails = list(direct_flow.INPUT_RAILS)
    try:
        direct_flow.set_hidden_cells(1)
        direct_flow.set_input_rails(["x0"])
        updates = direct_flow.hidden_flow_updates(
            update_width_u=0.02,
            flow_pre_store="synapse_gate",
            hidden_delta_output_mode="raw",
            hidden_flow_write_mode="bounded_charge_only",
            write_error_exclusion="diffpair_bleed",
            write_error_exclusion_width_u=8.0,
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_input_rails(original_input_rails)

    assert "Chwsel0_pos hwpos0 0 0.1f IC=0" in updates
    assert "Mhwsel0_pos_sel hwsel0_posbar hdp0 hwsel0_src 0 NSENSE W=8u" in updates
    assert "Mhwsel0_neg_sel hwsel0_negbar hdn0 hwsel0_src 0 NSENSE W=8u" in updates
    assert "Mhwsel0_tail hwsel0_src bwd 0 0 NMOS W=4u" in updates
    assert "Mwh0_x0p_ch_d wh0_x0p_ch_x hwpos0 wh0_x0p 0 NSENSE" in updates
    assert "Mwh0_x0n_ch_d wh0_x0n_ch_x hwneg0 wh0_x0n 0 NSENSE" in updates


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


def test_weighted_relu_pass_input_hidden_forward_uses_input_as_positive_source() -> None:
    original_hidden = direct_flow.HIDDEN
    original_input_rails = list(direct_flow.INPUT_RAILS)
    try:
        direct_flow.set_hidden_cells(1)
        direct_flow.set_input_rails(["x0"])
        design = direct_flow.scaled_synapse_design(
            "split_signed_passact_v1",
            hidden_delta_width_scale=1.0,
            hidden_gradient_width_scale=1.0,
            readout_gradient_width_scale=1.0,
        )

        forward = direct_flow.hidden_forward(design, "weighted_relu_pass_input")
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_input_rails(original_input_rails)

    assert "Mh0_x0p_w x0 wh0_x0p h0_x0p1 0 NREL" in forward
    assert "Mh0_x0p_f h0_x0p1 fwd pre0 0 NREL" in forward
    assert "Mh0_x0p_x vdd x0" not in forward
    assert "Mh0_x0n_f pre0 fwd h0_x0n0 0 NREL" in forward
    assert "Mh0_x0n_x h0_x0n0 x0 h0_x0n1 0 NREL" in forward
    assert "Mrelu_h0 vdd pre0 act0 0 NREL" in forward


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
    assert "Cscorep0 scorep0 0 10f IC=0.3" in caps
    assert "Cscoren0 scoren0 0 10f IC=0.3" in caps
    assert "Cybar0 ybar0 0 20f IC=1.2" in caps
    assert "Mreset_score0 score0 rstf scorecm 0 NMOS" in reset
    assert "Mreset_score1 score1 rstf scorecm 0 NMOS" in reset
    assert "Mreset_ybar0_high vdd rste ybar0 0 NSENSE" in reset

    larger_score_caps = direct_flow.temporary_caps(
        gradient_cap_f=4.0,
        hidden_gradient_cap_f=4.0,
        hidden_delta_cap_f=12.0,
        lead_cap_f=2.0,
        include_gradient_caps=False,
        score_reset_v=0.30,
        score_cap_f=100.0,
    )
    assert "Cscore0 score0 0 100f IC=0.3" in larger_score_caps
    assert "Cscorep0 scorep0 0 100f IC=0.3" in larger_score_caps
    assert "Cscoren0 scoren0 0 100f IC=0.3" in larger_score_caps
    assert "Cout0 out0 0 20f IC=0" in larger_score_caps
    larger_output_caps = direct_flow.temporary_caps(
        gradient_cap_f=4.0,
        hidden_gradient_cap_f=4.0,
        hidden_delta_cap_f=12.0,
        lead_cap_f=2.0,
        include_gradient_caps=False,
        score_reset_v=0.30,
        score_cap_f=100.0,
        output_cap_f=100.0,
    )
    assert "Cout0 out0 0 100f IC=0" in larger_output_caps
    assert "Mreset_scorep0 scorep0 rstf scorecm 0 NMOS" in reset
    assert "Mreset_scoren0 scoren0 rstf scorecm 0 NMOS" in reset
    split_reset = direct_flow.resets(
        "score_direct",
        include_gradient_resets=False,
        score_reset_v=0.30,
        output_head="split_score_caps",
    )
    assert "Mreset_out0 out0 rstf scorecm 0 NMOS" in split_reset

    low_true_caps = direct_flow.temporary_caps(
        gradient_cap_f=4.0,
        hidden_gradient_cap_f=4.0,
        hidden_delta_cap_f=12.0,
        lead_cap_f=2.0,
        include_gradient_caps=False,
        score_reset_v=0.0,
        output_cap_f=100.0,
        output_head="split_score_diode_mirror_caps",
    )
    assert "Cout0 out0 0 100f IC=1.2" in low_true_caps
    assert "Rout0 out0 vdd 1G" in low_true_caps


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


def test_split_score_caps_output_head_stores_positive_and_negative_evidence_separately() -> None:
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(2)
        direct_flow.set_output_count(3)
        design = direct_flow.scaled_synapse_design(
            "split_signed_v1",
            hidden_delta_width_scale=1.0,
            hidden_gradient_width_scale=1.0,
            readout_gradient_width_scale=1.0,
        )

        forward = direct_flow.output_forward(design, "split_score_caps")
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    assert "Mo00pos_f o00p1 fwd scorep0" in forward
    assert "Mo00neg_f o00n1 fwd scoren0" in forward
    assert "Mo2bpos_f o2bp1 fwd scorep2" in forward
    assert "Mo2bneg_f o2bn1 fwd scoren2" in forward
    assert "Mout0_split_pos_s vdd scorep0 out0_split_pos" in forward
    assert "Mout0_split_neg_s out0_split_neg scoren0 0" in forward
    assert "fwd score0" not in forward


def test_split_score_none_keeps_score_rails_unloaded_for_direct_decision() -> None:
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(2)
        direct_flow.set_output_count(3)
        design = direct_flow.scaled_synapse_design(
            "split_signed_v1",
            hidden_delta_width_scale=1.0,
            hidden_gradient_width_scale=1.0,
            readout_gradient_width_scale=1.0,
        )

        forward = direct_flow.output_forward(design, "split_score_none")
        caps = direct_flow.temporary_caps(
            gradient_cap_f=4.0,
            hidden_gradient_cap_f=4.0,
            hidden_delta_cap_f=12.0,
            lead_cap_f=2.0,
            include_gradient_caps=False,
            score_reset_v=0.0,
            output_head="split_score_none",
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    assert "Mo00pos_f o00p1 fwd scorep0" in forward
    assert "Mo00neg_f o00n1 fwd scoren0" in forward
    assert "Mo2bpos_f o2bp1 fwd scorep2" in forward
    assert "Mo2bneg_f o2bn1 fwd scoren2" in forward
    assert "Score-rail-only split output" in forward
    assert "Mout0_" not in forward
    assert "fwd score0" not in forward
    assert "Cscorep0 scorep0 0 10f IC=0" in caps
    assert "Cscoren0 scoren0 0 10f IC=0" in caps
    assert "Cout0 out0 0 20f IC=0" in caps


def test_split_score_diffgate_output_head_rejects_score_common_mode_locally() -> None:
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(2)
        direct_flow.set_output_count(3)
        design = direct_flow.scaled_synapse_design(
            "split_signed_v1",
            hidden_delta_width_scale=1.0,
            hidden_gradient_width_scale=1.0,
            readout_gradient_width_scale=1.0,
        )

        forward = direct_flow.output_forward(design, "split_score_diffgate")
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    assert "Mo00pos_f o00p1 fwd scorep0" in forward
    assert "Mo00neg_f o00n1 fwd scoren0" in forward
    assert "Mo2bpos_f o2bp1 fwd scorep2" in forward
    assert "Mo2bneg_f o2bn1 fwd scoren2" in forward
    assert "Mout0_dg_pos_inhibit vdd scoren0 out0_diffgate_pos_low vdd PMOS" in forward
    assert "Mout0_dg_pos_score out0_diffgate_pos_low scorep0 out0_diffgate_pos_mid" in forward
    assert "Mout0_dg_neg_inhibit out0 scorep0 out0_diffgate_neg_low vdd PMOS" in forward
    assert "Mout0_dg_neg_score out0_diffgate_neg_low scoren0 out0_diffgate_neg_mid" in forward
    assert "fwd score0" not in forward


def test_split_score_chargegate_output_head_has_no_discharge_leg() -> None:
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(2)
        direct_flow.set_output_count(3)
        design = direct_flow.scaled_synapse_design(
            "split_signed_v1",
            hidden_delta_width_scale=1.0,
            hidden_gradient_width_scale=1.0,
            readout_gradient_width_scale=1.0,
        )

        forward = direct_flow.output_forward(design, "split_score_chargegate")
        reset = direct_flow.resets(
            "score_direct",
            include_gradient_resets=False,
            score_reset_v=0.30,
            output_head="split_score_chargegate",
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    assert "Mo00pos_f o00p1 fwd scorep0" in forward
    assert "Mo00neg_f o00n1 fwd scoren0" in forward
    assert "Mout0_cg_inhibit vdd scoren0 out0_chargegate_pos_low vdd PMOS" in forward
    assert "Mout0_cg_score out0_chargegate_pos_low scorep0 out0_chargegate_pos_mid" in forward
    assert "Mout0_cg_f out0_chargegate_pos_mid fwd out0" in forward
    assert "Mout0_cg_neg" not in forward
    assert "Mreset_out0 out0 rstf 0 0 NMOS" in reset
    assert "fwd score0" not in forward


def test_split_score_diffpair_output_head_uses_local_score_pair_before_voltage_output() -> None:
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(2)
        direct_flow.set_output_count(3)
        design = direct_flow.scaled_synapse_design(
            "split_signed_v1",
            hidden_delta_width_scale=1.0,
            hidden_gradient_width_scale=1.0,
            readout_gradient_width_scale=1.0,
        )

        forward = direct_flow.output_forward(design, "split_score_diffpair")
        reset = direct_flow.resets(
            "score_direct",
            include_gradient_resets=False,
            score_reset_v=0.30,
            output_head="split_score_diffpair",
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    assert "Mo00pos_f o00p1 fwd scorep0" in forward
    assert "Mo00neg_f o00n1 fwd scoren0" in forward
    assert "Rout0_dpair_pull out0 vdd 1e12" in forward
    assert "Mout0_dpair_pos out0 scorep0 out0_dpair_src" in forward
    assert "Mout0_dpair_neg vdd scoren0 out0_dpair_src" in forward
    assert "Mout0_dpair_tail out0_dpair_src fwd 0" in forward
    assert "Mout0_dpair_inv" not in forward
    assert "Mreset_out0 out0 rstf vdd 0 NMOS" in reset
    assert "fwd score0" not in forward


def test_split_score_diode_diffpair_output_head_loads_scores_with_mos_diodes() -> None:
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(2)
        direct_flow.set_output_count(3)
        design = direct_flow.scaled_synapse_design(
            "split_signed_v1",
            hidden_delta_width_scale=1.0,
            hidden_gradient_width_scale=1.0,
            readout_gradient_width_scale=1.0,
        )

        forward = direct_flow.output_forward(
            design,
            "split_score_diode_diffpair",
            score_diode_width_u=2048.0,
        )
        reset = direct_flow.resets(
            "score_direct",
            include_gradient_resets=False,
            score_reset_v=0.0,
            output_head="split_score_diode_diffpair",
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    assert "Mo00pos_f o00p1 fwd scorep0" in forward
    assert "Mo00neg_f o00n1 fwd scoren0" in forward
    assert "Mscorep0_diode scorep0 scorep0 0 0 NSENSE W=2048u" in forward
    assert "Mscoren0_diode scoren0 scoren0 0 0 NSENSE W=2048u" in forward
    assert "Rout0_ddpair_pull out0 vdd 1e12" in forward
    assert "Mout0_ddpair_pos out0 scorep0 out0_ddpair_src" in forward
    assert "Mout0_ddpair_neg vdd scoren0 out0_ddpair_src" in forward
    assert "Mout0_ddpair_tail out0_ddpair_src fwd 0" in forward
    assert "Mreset_out0 out0 rstf vdd 0 NMOS" in reset
    assert "fwd score0" not in forward


def test_split_score_compete_tail_output_head_shares_class_tail_current() -> None:
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(2)
        direct_flow.set_output_count(3)
        design = direct_flow.scaled_synapse_design(
            "split_signed_v1",
            hidden_delta_width_scale=1.0,
            hidden_gradient_width_scale=1.0,
            readout_gradient_width_scale=1.0,
        )

        forward = direct_flow.output_forward(design, "split_score_compete_tail")
        reset = direct_flow.resets(
            "score_direct",
            include_gradient_resets=False,
            score_reset_v=0.0,
            output_head="split_score_compete_tail",
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    assert "Mo00pos_f o00p1 fwd scorep0" in forward
    assert "Mo00neg_f o00n1 fwd scoren0" in forward
    assert "Rout0_ctail_pull out0 vdd 1e12" in forward
    assert "Mout0_ctail_inhibit out0_ctail_mid scoren0 out0 vdd PMOS" in forward
    assert "Mout0_ctail_score out0_ctail_mid scorep0 out_compete_src 0 NREL" in forward
    assert "Mout1_ctail_score out1_ctail_mid scorep1 out_compete_src 0 NREL" in forward
    assert "Mout2_ctail_score out2_ctail_mid scorep2 out_compete_src 0 NREL" in forward
    assert forward.count("Mout_ctail_tail out_compete_src outg 0 0 NMOS") == 1
    assert "Rpar_out_compete_src out_compete_src 0 1e9" in forward
    assert "Mreset_out0 out0 rstf vdd 0 NMOS" in reset
    assert "fwd score0" not in forward


def test_split_score_diode_mirror_caps_output_head_uses_current_derived_score_caps() -> None:
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(2)
        direct_flow.set_output_count(3)
        design = direct_flow.scaled_synapse_design(
            "split_signed_v1",
            hidden_delta_width_scale=1.0,
            hidden_gradient_width_scale=1.0,
            readout_gradient_width_scale=1.0,
        )

        forward = direct_flow.output_forward(
            design,
            "split_score_diode_mirror_caps",
            score_diode_width_u=64.0,
            score_mirror_cap_f=7.0,
        )
        reset = direct_flow.resets(
            "score_direct",
            include_gradient_resets=False,
            score_reset_v=0.0,
            output_head="split_score_diode_mirror_caps",
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    assert "Mo00pos_f o00p1 fwd scorep0" in forward
    assert "Mo00neg_f o00n1 fwd scoren0" in forward
    assert "Mscorep0_diode scorep0 scorep0 0 0 NSENSE W=64u" in forward
    assert "Cscorepm0 scorepm0 0 7f IC=1.2" in forward
    assert "Cscorenm0 scorenm0 0 7f IC=1.2" in forward
    assert "Mscorep0_mirror scorepm0 scorep0 0 0 NSENSE W=64u" in forward
    assert "Mscoren0_mirror scorenm0 scoren0 0 0 NSENSE W=64u" in forward
    assert "Mout0_dmcap_pos out0 scorenm0 out0_dmcap_src" in forward
    assert "Mout0_dmcap_neg vdd scorepm0 out0_dmcap_src" in forward
    assert "Mout0_dmcap_tail out0_dmcap_src outg 0" in forward
    assert "Mreset_out0 out0 rstf vdd 0 NMOS" in reset
    assert "fwd score0" not in forward


def test_split_score_diode_mirror_gate_caps_output_head_uses_conditional_discharge() -> None:
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(2)
        direct_flow.set_output_count(3)
        design = direct_flow.scaled_synapse_design(
            "split_signed_v1",
            hidden_delta_width_scale=1.0,
            hidden_gradient_width_scale=1.0,
            readout_gradient_width_scale=1.0,
        )

        forward = direct_flow.output_forward(
            design,
            "split_score_diode_mirror_gate_caps",
            score_diode_width_u=64.0,
            score_mirror_cap_f=7.0,
        )
        reset = direct_flow.resets(
            "score_direct",
            include_gradient_resets=False,
            score_reset_v=0.0,
            output_head="split_score_diode_mirror_gate_caps",
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    assert "Mscorep0_diode scorep0 scorep0 0 0 NSENSE W=64u" in forward
    assert "Cscorepm0 scorepm0 0 7f IC=1.2" in forward
    assert "Mscorep0_mirror scorepm0 scorep0 0 0 NSENSE W=64u" in forward
    assert "Mout0_dmgate_inhibit out0 scorepm0 out0_dmgate_mid vdd PMOS" in forward
    assert "Mout0_dmgate_score out0_dmgate_mid scorenm0 out0_dmgate_src" in forward
    assert "Mout0_dmgate_tail out0_dmgate_src outg 0" in forward
    assert "Mout0_dmcap_neg" not in forward
    assert "Mreset_out0 out0 rstf vdd 0 NMOS" in reset
    assert "fwd score0" not in forward


def test_output_head_transfer_harness_uses_production_score_to_output_fragment() -> None:
    netlist = output_head_transfer.output_head_netlist(
        score_pairs=((0.46, 0.35), (0.40, 0.35), (0.38, 0.35)),
        output_head="split_score_diffpair",
        design_name="split_signed_v1",
        output_width_scale=1.0,
        output_cap_f=10.0,
        score_reset_v=0.30,
        tstop_ns=0.30,
    )
    circuit_body = netlist.split(".control", maxsplit=1)[0]

    assert "Vscorep0 scorep0 0 0.46" in netlist
    assert "Vscoren0 scoren0 0 0.35" in netlist
    assert "Vfwd fwd 0 1.0" in netlist
    assert "Voutg outg 0 1.0" in netlist
    assert "Cout0 out0 0 10f IC=1.2" in netlist
    assert "Mout0_dpair_pos out0 scorep0 out0_dpair_src 0 NREL" in netlist
    assert "Mout0_dpair_neg vdd scoren0 out0_dpair_src 0 NREL" in netlist
    assert "Mout0_dpair_tail out0_dpair_src fwd 0 0 NMOS" in netlist
    assert "B" not in "\n".join(line[:1] for line in circuit_body.splitlines() if line)

    ng = prepare_netlist_for_simulator(netlist, "ngspice")
    xy = prepare_netlist_for_simulator(netlist, "Xyce")
    assert ".control" in ng
    assert ".control" not in xy
    assert canonical_circuit_netlist(ng) == canonical_circuit_netlist(xy)


def test_output_head_transfer_harness_can_use_diode_loaded_diffpair() -> None:
    netlist = output_head_transfer.output_head_netlist(
        score_pairs=((0.46, 0.35), (0.40, 0.35), (0.38, 0.35)),
        output_head="split_score_diode_diffpair",
        design_name="split_signed_v1",
        output_width_scale=1.0,
        output_cap_f=10.0,
        score_reset_v=0.0,
        tstop_ns=0.30,
        score_diode_width_u=1024.0,
    )
    circuit_body = netlist.split(".control", maxsplit=1)[0]

    assert "Vscorep0 scorep0 0 0.46" in netlist
    assert "Mscorep0_diode scorep0 scorep0 0 0 NSENSE W=1024u" in netlist
    assert "Mout0_ddpair_pos out0 scorep0 out0_ddpair_src 0 NREL" in netlist
    assert "Mout0_ddpair_neg vdd scoren0 out0_ddpair_src 0 NREL" in netlist
    assert "Cout0 out0 0 10f IC=1.2" in netlist
    assert "B" not in "\n".join(line[:1] for line in circuit_body.splitlines() if line)


def test_output_head_transfer_harness_can_use_shared_tail_competition() -> None:
    netlist = output_head_transfer.output_head_netlist(
        score_pairs=((0.46, 0.35), (0.40, 0.35), (0.38, 0.35)),
        output_head="split_score_compete_tail",
        design_name="split_signed_v1",
        output_width_scale=1.0,
        output_cap_f=10.0,
        score_reset_v=0.0,
        tstop_ns=0.30,
    )
    circuit_body = netlist.split(".control", maxsplit=1)[0]

    assert "Vscorep0 scorep0 0 0.46" in netlist
    assert "Cout0 out0 0 10f IC=1.2" in netlist
    assert "Mout0_ctail_inhibit out0_ctail_mid scoren0 out0 vdd PMOS" in netlist
    assert "Mout0_ctail_score out0_ctail_mid scorep0 out_compete_src 0 NREL" in netlist
    assert netlist.count("Mout_ctail_tail out_compete_src outg 0 0 NMOS") == 1
    assert "B" not in "\n".join(line[:1] for line in circuit_body.splitlines() if line)


def test_output_head_transfer_harness_can_use_diode_mirror_cap_head() -> None:
    netlist = output_head_transfer.output_head_netlist(
        score_pairs=((0.46, 0.35), (0.40, 0.35), (0.38, 0.35)),
        output_head="split_score_diode_mirror_caps",
        design_name="split_signed_v1",
        output_width_scale=1.0,
        output_cap_f=10.0,
        score_reset_v=0.0,
        tstop_ns=0.30,
        score_diode_width_u=64.0,
    )
    circuit_body = netlist.split(".control", maxsplit=1)[0]

    assert "Vscorep0 scorep0 0 0.46" in netlist
    assert "Mscorep0_diode scorep0 scorep0 0 0 NSENSE W=64u" in netlist
    assert "Mscorep0_mirror scorepm0 scorep0 0 0 NSENSE W=64u" in netlist
    assert "Mout0_dmcap_pos out0 scorenm0 out0_dmcap_src 0 NREL" in netlist
    assert "Cout0 out0 0 10f IC=1.2" in netlist
    assert "B" not in "\n".join(line[:1] for line in circuit_body.splitlines() if line)


def test_output_head_transfer_harness_can_use_diode_mirror_gate_cap_head() -> None:
    netlist = output_head_transfer.output_head_netlist(
        score_pairs=((0.46, 0.35), (0.40, 0.35), (0.38, 0.35)),
        output_head="split_score_diode_mirror_gate_caps",
        design_name="split_signed_v1",
        output_width_scale=1.0,
        output_cap_f=10.0,
        score_reset_v=0.0,
        tstop_ns=0.30,
        score_diode_width_u=64.0,
        score_mirror_cap_f=7.0,
    )
    circuit_body = netlist.split(".control", maxsplit=1)[0]

    assert "Vscorep0 scorep0 0 0.46" in netlist
    assert "Mscorep0_diode scorep0 scorep0 0 0 NSENSE W=64u" in netlist
    assert "Cscorepm0 scorepm0 0 7f IC=1.2" in netlist
    assert "Mout0_dmgate_inhibit out0 scorepm0 out0_dmgate_mid vdd PMOS" in netlist
    assert "Mout0_dmgate_score out0_dmgate_mid scorenm0 out0_dmgate_src 0 NREL" in netlist
    assert "Cout0 out0 0 10f IC=1.2" in netlist
    assert "B" not in "\n".join(line[:1] for line in circuit_body.splitlines() if line)


def test_buffered_passact_readout_does_not_use_stored_activation_as_source() -> None:
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(2)
        direct_flow.set_output_count(3)
        design = direct_flow.scaled_synapse_design(
            "split_signed_passact_buffered_v1",
            hidden_delta_width_scale=1.0,
            hidden_gradient_width_scale=1.0,
            readout_gradient_width_scale=1.0,
        )

        forward = direct_flow.output_forward(design, "source_follower")
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    assert "Mactbuf0_src vdd act0 actbuf0" in forward
    assert "Mactbuf0_rst actbuf0 rstf 0" in forward
    assert "Mo00pos_w actbuf0 vw00p" in forward
    assert "Mo00pos_w act0 vw00p" not in forward
    assert "Mo20pos_w actbuf0 vw20p" in forward


def test_passact_split_readout_bias_uses_same_source_style_as_readout_branch() -> None:
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(1)
        direct_flow.set_output_count(2)
        design = direct_flow.scaled_synapse_design(
            "split_signed_passact_v1",
            hidden_delta_width_scale=1.0,
            hidden_gradient_width_scale=1.0,
            readout_gradient_width_scale=1.0,
        )

        forward = direct_flow.output_forward(design, "split_score_none")
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    assert "Mo0bpos_w bias vbo0p o0bp1 0 NREL" in forward
    assert "Mo0bneg_w bias vbo0n o0bn1 0 NREL" in forward
    assert "Mo0bpos_src vdd vbo0p" not in forward
    assert "Mo0bpos_gate" not in forward


def test_direct_flow_sparse_readout_forward_omits_absent_hidden_edges() -> None:
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(3)
        direct_flow.set_output_count(2)
        design = direct_flow.scaled_synapse_design(
            "split_signed_passact_v1",
            hidden_delta_width_scale=1.0,
            hidden_gradient_width_scale=1.0,
            readout_gradient_width_scale=1.0,
        )

        forward = direct_flow.output_forward(
            design,
            "split_score_none",
            readout_fanins={0: (0, 2), 1: (1,)},
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    assert "Output 0: signed readout from 2 general hidden activations" in forward
    assert "Mo00pos_w act0 vw00p" in forward
    assert "Mo02pos_w act2 vw02p" in forward
    assert "Mo01pos_w" not in forward
    assert "Mo11pos_w act1 vw11p" in forward
    assert "Mo10pos_w" not in forward
    assert "Mo12pos_w" not in forward
    assert "Mo0bpos_w bias vbo0p" in forward
    assert "Mo1bpos_w bias vbo1p" in forward


def test_random_readout_fanout_topology_limits_outputs_per_hidden() -> None:
    fanins = direct_flow.random_readout_fanins(8, 5, seed=13, fan_out=3)
    fanouts = direct_flow.readout_fanouts_from_fanins(fanins, 8)

    assert sum(len(srcs) for srcs in fanins.values()) == 24
    assert all(len(outs) == 3 for outs in fanouts.values())
    assert all(len(set(outs)) == 3 for outs in fanouts.values())


def test_random_hidden_input_topology_keeps_bias_and_limits_data_fanin() -> None:
    fanins = direct_flow.random_hidden_fanins(5, ["x0", "x1", "x2", "x3"], seed=17, fan_in=3)
    fanouts = direct_flow.hidden_fanouts_from_fanins(fanins)

    assert all(rails[0] == "bias" for rails in fanins.values())
    assert all(len([rail for rail in rails if rail != "bias"]) == 3 for rails in fanins.values())
    assert set(fanouts) >= {"bias", "x0", "x1", "x2", "x3"}


def test_direct_flow_sparse_hidden_input_omits_absent_synapse_edges() -> None:
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    original_input_rails = list(direct_flow.INPUT_RAILS)
    try:
        direct_flow.set_hidden_cells(2)
        direct_flow.set_output_count(2)
        direct_flow.set_input_rails(["x0", "x1", "x2"])
        fanins = {0: ("bias", "x0"), 1: ("bias", "x2")}
        hidden = direct_flow.hidden_init(3, "random")
        readout = {"vbo0p": 0.65, "vbo0n": 0.62, "vbo1p": 0.63, "vbo1n": 0.61}
        for out in range(2):
            for h in range(2):
                readout[f"vw{out}{h}p"] = 0.66
                readout[f"vw{out}{h}n"] = 0.60
        design = direct_flow.scaled_synapse_design(
            "split_signed_v1",
            hidden_delta_width_scale=1.0,
            hidden_gradient_width_scale=1.0,
            readout_gradient_width_scale=1.0,
        )

        caps = direct_flow.persistent_caps(hidden, readout, 4.0, hidden_fanins=fanins)
        temp = direct_flow.temporary_caps(4.0, 4.0, 12.0, 2.0, True, 0.2, hidden_fanins=fanins)
        reset = direct_flow.resets("score_direct", True, 0.2, hidden_fanins=fanins)
        forward = direct_flow.hidden_forward(design, "weighted_relu", fanins)
        stores = direct_flow.flow_pre_activation_stores(
            "synapse_spike",
            2.0,
            0.05,
            hidden_fanins=fanins,
        )
        grads = direct_flow.hidden_gradients_and_updates(
            0.04,
            "act_nrel",
            "direct",
            128.0,
            2.0,
            design,
            fanins,
        )
        flow_updates = direct_flow.hidden_flow_updates(
            0.04,
            "synapse_spike",
            "raw",
            "discharge",
            hidden_fanins=fanins,
        )
        pairs = direct_flow.signed_weight_pairs("hidden", hidden_fanins=fanins)
        measures, _prints = direct_flow.measure_lines(
            samples=[{"phase": "train", "label": 0, "pattern": 0, "apply_update": True}],
            hidden_apply_mode="direct",
            learning_mode="flow",
            hidden_delta_output_mode="raw",
            measure_detail="probe",
            readout_sample_offsets_ns=[2.95],
            activation_sample_offsets_ns=[],
            cmp_start_ns=3.25,
            cmp_end_ns=4.10,
            bwd_start_ns=6.75,
            apply_end_ns=11.20,
            backward_gate_mode="scheduled",
            hidden_fanins=fanins,
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)
        direct_flow.set_input_rails(original_input_rails)

    assert "Cwh0_x0p wh0_x0p" in caps
    assert "Cwh0_x1p" not in caps
    assert "Cghp0_x0 ghp0_x0" in temp
    assert "Cghp0_x1" not in temp
    assert "Mreset_ghp0_x0" in reset
    assert "Mreset_ghp0_x1" not in reset
    assert "Mh0_x0p_x" in forward
    assert "Mh0_x1p_x" not in forward
    assert "Cfphi0_x0 fphi0_x0" in stores
    assert "Cfphi0_x1" not in stores
    assert "Mghp0_x0_x" in grads
    assert "Mghp0_x1_x" not in grads
    assert "Mwh0_x0n_flow" in flow_updates
    assert "Mwh0_x1n_flow" not in flow_updates
    assert ("wh0_x0", "wh0_x0p", "wh0_x0n") in pairs
    assert ("wh0_x1", "wh0_x1p", "wh0_x1n") not in pairs
    assert "d_wh0_x0_signed_total" in measures
    assert "d_wh0_x1_signed_total" not in measures


def test_direct_flow_sparse_readout_omits_absent_backward_trace_and_write_edges() -> None:
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(3)
        direct_flow.set_output_count(2)
        fanins = {0: (0, 2), 1: (1,)}
        hidden = direct_flow.hidden_init(3, "random")
        readout = {"vbo0p": 0.65, "vbo0n": 0.62, "vbo1p": 0.63, "vbo1n": 0.61}
        for out in range(2):
            for h in range(3):
                readout[f"vw{out}{h}p"] = 0.66
                readout[f"vw{out}{h}n"] = 0.60
        design = direct_flow.scaled_synapse_design(
            "split_signed_passact_v1",
            hidden_delta_width_scale=1.0,
            hidden_gradient_width_scale=1.0,
            readout_gradient_width_scale=1.0,
        )

        caps = direct_flow.persistent_caps(hidden, readout, 4.0, fanins)
        temp = direct_flow.temporary_caps(4.0, 4.0, 12.0, 2.0, True, 0.2, readout_fanins=fanins)
        reset = direct_flow.resets("score_direct", True, 0.2, readout_fanins=fanins)
        stores = direct_flow.flow_pre_activation_stores("synapse_spike", 2.0, 0.05, readout_fanins=fanins)
        delta = direct_flow.hidden_delta(
            "backprop",
            "act_nrel",
            "nmos",
            design,
            internal_cap_f=0.0,
            internal_leak_ohm=0.0,
            internal_reset_width_u=0.0,
            readout_fanins=fanins,
        )
        grads = direct_flow.readout_gradients_and_updates(0.04, 0.04, design, fanins)
        flow_updates = direct_flow.readout_flow_updates(
            0.04,
            0.04,
            "synapse_spike",
            "normal",
            readout_fanins=fanins,
        )
        pairs = direct_flow.signed_weight_pairs("readout", fanins)
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    for present in ["vw00p", "vw02p", "vw11p"]:
        assert present in caps
    for absent in ["Cvw01p", "Cvw10p", "Cvw12p"]:
        assert absent not in caps
    assert "Cgvp00 gvp00" in temp
    assert "Cgvp01 gvp01" not in temp
    assert "Mreset_gvp00 gvp00" in reset
    assert "Mreset_gvp01 gvp01" not in reset
    assert "Cfprg00 fprg00" in stores
    assert "Cfprg01 fprg01" not in stores
    assert "vw00p" in delta
    assert "vw01p" not in delta
    assert "Mgvp00_a" in grads
    assert "Mgvp01_a" not in grads
    assert "Mvw00n_flow" in flow_updates
    assert "Mvw01n_flow" not in flow_updates
    assert ("vw00", "vw00p", "vw00n") in pairs
    assert ("vw01", "vw01p", "vw01n") not in pairs


def test_sparse_readout_measurements_only_reference_existing_edges() -> None:
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(3)
        direct_flow.set_output_count(2)
        measures, _prints = direct_flow.measure_lines(
            samples=[{"phase": "train", "label": 0, "pattern": 0, "apply_update": True}],
            hidden_apply_mode="direct",
            learning_mode="flow",
            hidden_delta_output_mode="raw",
            measure_detail="probe",
            readout_sample_offsets_ns=[2.95],
            activation_sample_offsets_ns=[],
            cmp_start_ns=3.25,
            cmp_end_ns=4.10,
            bwd_start_ns=6.75,
            apply_end_ns=11.20,
            backward_gate_mode="scheduled",
            flow_pre_store="synapse_spike",
            readout_fanins={0: (0, 2), 1: (1,)},
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    assert "fprg00_0" in measures
    assert "fprg01_0" not in measures
    assert "d_vw00_signed_total" in measures
    assert "d_vw01_signed_total" not in measures


def test_readout_array_eval_sparse_topology_omits_caps_and_forward_devices() -> None:
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(3)
        direct_flow.set_output_count(2)
        state = {
            "vbo0p": 0.2,
            "vbo0n": 0.1,
            "vbo1p": 0.3,
            "vbo1n": 0.2,
        }
        for out in range(2):
            for h in range(3):
                state[f"vw{out}{h}p"] = 0.4
                state[f"vw{out}{h}n"] = 0.2
        activations = pd.DataFrame(
            {
                "label": [0],
                "act0": [0.10],
                "act1": [0.20],
                "act2": [0.30],
            }
        )
        fanins = {0: (0, 2), 1: (1,)}

        netlist, _samples = readout_array_eval.readout_array_netlist(
            activations=activations,
            readout_state=state,
            design_name="split_signed_passact_v1",
            output_head="split_score_none",
            hidden_cells=3,
            outputs=2,
            hidden_cap_f=4.0,
            score_reset_v=0.2,
            score_cap_f=10.0,
            output_cap_f=20.0,
            sample_ns=2.95,
            cycle_ns=16.0,
            activation_drive="held",
            activation_settle_ns=2.95,
            activation_sample_offsets_ns=[],
            readout_load_mode="forward_only",
            score_sense_mode="clamped_current",
            tran_step_ps=10.0,
            spice_accuracy_preset="fast",
            readout_fanins=fanins,
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    assert "Cvw00p vw00p 0 4f IC=0.4" in netlist
    assert "Cvw02p vw02p 0 4f IC=0.4" in netlist
    assert "Cvw01p" not in netlist
    assert "Mo00pos_w act0 vw00p" in netlist
    assert "Mo02pos_w act2 vw02p" in netlist
    assert "Mo01pos_w" not in netlist
    assert "Mo11pos_w act1 vw11p" in netlist
    assert "Mo10pos_w" not in netlist


def test_readout_transfer_deck_programs_separator_and_measures_rows(tmp_path: Path) -> None:
    separator = tmp_path / "readout.csv"
    separator.write_text(
        "out,bias,w0,w1\n"
        "0,0.0,1.0,-1.0\n"
        "1,0.0,-1.0,1.0\n"
        "2,0.1,0.5,0.5\n"
    )
    activations = pd.DataFrame(
        {
            "label": [0, 1],
            "act0": [0.8, 0.2],
            "act1": [0.2, 0.8],
        }
    )
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(2)
        direct_flow.set_output_count(3)
        netlist, rows = readout_transfer.build_readout_transfer_netlist(
            activations=activations,
            separator_csv=separator,
            synapse_design="split_signed_passact_buffered_v1",
            separator_scale=0.04,
            readout_center_v=0.64,
            score_reset_v=0.0,
            output_forward_width_scale=1.0,
            output_forward_pos_width_scale=1.0,
            output_forward_neg_width_scale=1.0,
            output_bias_forward_width_scale=1.0,
            output_relu_width_scale=1.0,
            output_head="source_follower",
            transfer_topology="main",
            cycle_ns=4.0,
            sample_offset_ns=2.95,
            tran_step_ps=10.0,
            cap_f=4.0,
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    assert "Mactbuf0_src vdd act0 actbuf0" in netlist
    assert "Vact0 act0 0 PWL(" in netlist
    assert "Cvw00p vw00p 0 4f IC=0.66" in netlist
    assert "Cvw00n vw00n 0 4f IC=0.62" in netlist
    assert ".meas tran score2_1 FIND V(score2)" in netlist
    assert rows["ideal_predicted_label"].tolist() == [0, 1]
    assert rows["ideal_correct"].tolist() == [True, True]


def test_split_score_caps_transfer_topology_uses_differential_score_storage(tmp_path: Path) -> None:
    separator = tmp_path / "readout.csv"
    separator.write_text(
        "out,bias,w0,w1\n"
        "0,0.0,1.0,-1.0\n"
        "1,0.0,-1.0,1.0\n"
        "2,0.1,0.5,0.5\n"
    )
    activations = pd.DataFrame(
        {
            "label": [0, 1],
            "act0": [0.8, 0.2],
            "act1": [0.2, 0.8],
        }
    )
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(2)
        direct_flow.set_output_count(3)
        netlist, _rows = readout_transfer.build_readout_transfer_netlist(
            activations=activations,
            separator_csv=separator,
            synapse_design="split_signed_v1",
            separator_scale=0.04,
            readout_center_v=0.64,
            score_reset_v=0.0,
            output_forward_width_scale=1.0,
            output_forward_pos_width_scale=1.0,
            output_forward_neg_width_scale=1.0,
            output_bias_forward_width_scale=1.0,
            output_relu_width_scale=1.0,
            output_head="source_follower",
            transfer_topology="split_score_caps",
            cycle_ns=4.0,
            sample_offset_ns=2.95,
            tran_step_ps=10.0,
            cap_f=4.0,
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    assert "Cscorep0 scorep0 0 10f IC=0" in netlist
    assert "Cscoren0 scoren0 0 10f IC=0" in netlist
    assert "Msp00pos_f sp00p1 fwd scorep0" in netlist
    assert "Msn00neg_f sn00n1 fwd scoren0" in netlist
    assert ".meas tran scorep2_1 FIND V(scorep2)" in netlist
    assert ".meas tran scoren2_1 FIND V(scoren2)" in netlist


def test_passact_split_transfer_bias_matches_branch_surface_model(tmp_path: Path) -> None:
    separator = tmp_path / "readout.csv"
    separator.write_text(
        "out,bias,w0\n"
        "0,0.1,1.0\n"
        "1,-0.1,-1.0\n"
    )
    activations = pd.DataFrame({"label": [0], "act0": [0.8]})
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(1)
        direct_flow.set_output_count(2)
        netlist, _rows = readout_transfer.build_readout_transfer_netlist(
            activations=activations,
            separator_csv=separator,
            synapse_design="split_signed_passact_v1",
            separator_scale=0.04,
            readout_center_v=0.64,
            score_reset_v=0.2,
            output_forward_width_scale=1.0,
            output_forward_pos_width_scale=1.0,
            output_forward_neg_width_scale=1.0,
            output_bias_forward_width_scale=1.0,
            output_relu_width_scale=1.0,
            output_head="source_follower",
            transfer_topology="split_score_clamped_current",
            cycle_ns=4.0,
            sample_offset_ns=2.95,
            tran_step_ps=10.0,
            cap_f=4.0,
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    assert "Msp0bpos_w bias vbo0p sp0bp1 0 NREL" in netlist
    assert "Msn0bneg_w bias vbo0n sn0bn1 0 NREL" in netlist
    assert "Msp0bpos_src vdd vbo0p" not in netlist
    assert "Msp0bpos_gate" not in netlist


def test_clamped_current_transfer_topology_measures_absorbed_branch_currents(tmp_path: Path) -> None:
    separator = tmp_path / "readout.csv"
    separator.write_text(
        "out,bias,w0,w1\n"
        "0,0.0,1.0,-1.0\n"
        "1,0.0,-1.0,1.0\n"
        "2,0.1,0.5,0.5\n"
    )
    activations = pd.DataFrame(
        {
            "label": [0, 1],
            "act0": [0.8, 0.2],
            "act1": [0.2, 0.8],
        }
    )
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(2)
        direct_flow.set_output_count(3)
        netlist, _rows = readout_transfer.build_readout_transfer_netlist(
            activations=activations,
            separator_csv=separator,
            synapse_design="split_signed_v1",
            separator_scale=0.04,
            readout_center_v=0.64,
            score_reset_v=0.2,
            output_forward_width_scale=1.0,
            output_forward_pos_width_scale=1.0,
            output_forward_neg_width_scale=1.0,
            output_bias_forward_width_scale=1.0,
            output_relu_width_scale=1.0,
            output_head="source_follower",
            transfer_topology="split_score_clamped_current",
            cycle_ns=4.0,
            sample_offset_ns=2.95,
            tran_step_ps=10.0,
            cap_f=4.0,
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    assert "Vscorep0_clamp scorep0 0 0.2" in netlist
    assert "Vscoren0_clamp scoren0 0 0.2" in netlist
    assert "Cscorep0 scorep0" not in netlist
    assert "Mreset_scorep0 scorep0" not in netlist
    assert "Msp00pos_f sp00p1 fwd scorep0" in netlist
    assert ".meas tran scorep2_1 FIND I(Vscorep2_clamp)" in netlist
    assert ".meas tran scoren2_1 FIND I(Vscoren2_clamp)" in netlist


def test_diode_clamp_transfer_topology_uses_transistor_score_loads(tmp_path: Path) -> None:
    separator = tmp_path / "readout.csv"
    separator.write_text(
        "out,bias,w0,w1\n"
        "0,0.0,1.0,-1.0\n"
        "1,0.0,-1.0,1.0\n"
        "2,0.1,0.5,0.5\n"
    )
    activations = pd.DataFrame(
        {
            "label": [0, 1],
            "act0": [0.8, 0.2],
            "act1": [0.2, 0.8],
        }
    )
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(2)
        direct_flow.set_output_count(3)
        netlist, _rows = readout_transfer.build_readout_transfer_netlist(
            activations=activations,
            separator_csv=separator,
            synapse_design="split_signed_v1",
            separator_scale=0.04,
            readout_center_v=0.64,
            score_reset_v=0.0,
            output_forward_width_scale=1.0,
            output_forward_pos_width_scale=1.0,
            output_forward_neg_width_scale=1.0,
            output_bias_forward_width_scale=1.0,
            output_relu_width_scale=1.0,
            output_head="source_follower",
            transfer_topology="split_score_diode_clamp",
            cycle_ns=4.0,
            sample_offset_ns=2.95,
            tran_step_ps=10.0,
            cap_f=4.0,
            score_cap_f=0.4,
            score_diode_width_u=128.0,
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    assert "Mscorep0_diode scorep0 scorep0 0 0 NSENSE W=128u" in netlist
    assert "Mscoren0_diode scoren0 scoren0 0 0 NSENSE W=128u" in netlist
    assert "Vscorep0_clamp" not in netlist
    assert "Cscorep0 scorep0 0 0.4f IC=0" in netlist
    assert ".meas tran scorep2_1 FIND V(scorep2)" in netlist


def test_diode_current_transfer_topology_measures_transistor_score_load_current(tmp_path: Path) -> None:
    separator = tmp_path / "readout.csv"
    separator.write_text(
        "out,bias,w0,w1\n"
        "0,0.0,1.0,-1.0\n"
        "1,0.0,-1.0,1.0\n"
        "2,0.1,0.5,0.5\n"
    )
    activations = pd.DataFrame(
        {
            "label": [0, 1],
            "act0": [0.8, 0.2],
            "act1": [0.2, 0.8],
        }
    )
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(2)
        direct_flow.set_output_count(3)
        netlist, _rows = readout_transfer.build_readout_transfer_netlist(
            activations=activations,
            separator_csv=separator,
            synapse_design="split_signed_v1",
            separator_scale=0.04,
            readout_center_v=0.64,
            score_reset_v=0.0,
            output_forward_width_scale=1.0,
            output_forward_pos_width_scale=1.0,
            output_forward_neg_width_scale=1.0,
            output_bias_forward_width_scale=1.0,
            output_relu_width_scale=1.0,
            output_head="source_follower",
            transfer_topology="split_score_diode_current",
            cycle_ns=4.0,
            sample_offset_ns=2.95,
            tran_step_ps=10.0,
            cap_f=4.0,
            score_cap_f=0.4,
            score_diode_width_u=128.0,
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    assert "Vscorep0_sense scorep0_sense 0 0" in netlist
    assert "Mscorep0_diode scorep0 scorep0 scorep0_sense 0 NSENSE W=128u" in netlist
    assert "Vscorep0_clamp" not in netlist
    assert ".meas tran scorep2_1 FIND I(Vscorep2_sense)" in netlist


def test_readout_transfer_summary_reports_affine_recovery_controls() -> None:
    rows = pd.DataFrame(
        {
            "label": [0, 1, 2],
            "ideal0": [2.0, 0.0, 0.0],
            "ideal1": [0.0, 2.0, 0.0],
            "ideal2": [0.0, 0.0, 2.0],
            "ideal_correct": [True, True, True],
            "score0": [10.2, 10.0, 10.0],
            "score1": [0.0, 2.0, 0.0],
            "score2": [0.0, 0.0, 2.0],
            "score_correct": [True, False, False],
            "out0": [10.2, 10.0, 10.0],
            "out1": [0.0, 2.0, 0.0],
            "out2": [0.0, 0.0, 2.0],
            "out_correct": [True, False, False],
        }
    )

    summary = readout_transfer.transfer_summary(rows, "toy", "split_signed_v1", 1.0, 0.0)

    assert summary["score_accuracy"] == pytest.approx(1 / 3)
    assert summary["score_column_centered_accuracy"] == pytest.approx(1.0)
    assert summary["score_mean_by_output_v"]["0"] == pytest.approx(10.066666666666666)
    assert summary["score_diag_idealfit_accuracy"] == pytest.approx(1.0)
    assert summary["score_full_idealfit_accuracy"] == pytest.approx(1.0)
    assert summary["out_diag_idealfit_accuracy"] == pytest.approx(1.0)


def test_readout_transfer_can_program_direct_cap_csv(tmp_path: Path) -> None:
    cap_csv = tmp_path / "caps.csv"
    rows = []
    for out in range(2):
        rows += [
            {"cap": f"vbo{out}p", "value": 0.21 + out * 0.01},
            {"cap": f"vbo{out}n", "value": 0.31 + out * 0.01},
        ]
        for h in range(2):
            rows += [
                {"cap": f"vw{out}{h}p", "value": 0.40 + out * 0.10 + h * 0.01},
                {"cap": f"vw{out}{h}n", "value": 0.50 + out * 0.10 + h * 0.01},
            ]
    pd.DataFrame(rows).to_csv(cap_csv, index=False)

    init = readout_transfer.load_readout_cap_init(cap_csv, hidden=2, outputs=2)

    assert init["vw10p"] == pytest.approx(0.50)
    assert init["vw11n"] == pytest.approx(0.61)
    assert init["vbo1p"] == pytest.approx(0.22)


def test_readout_transfer_output_bias_offset_programs_bias_caps(tmp_path: Path) -> None:
    path = tmp_path / "readout.csv"
    path.write_text("out,bias,w0\n0,0.0,0.0\n1,0.0,0.0\n")
    activations = pd.DataFrame([{"label": 1, "act0": 0.2}])
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(1)
        direct_flow.set_output_count(2)
        netlist, _rows = readout_transfer.build_readout_transfer_netlist(
            activations=activations,
            separator_csv=path,
            synapse_design="split_signed_v1",
            separator_scale=1.0,
            readout_center_v=0.64,
            score_reset_v=0.0,
            output_forward_width_scale=1.0,
            output_forward_pos_width_scale=1.0,
            output_forward_neg_width_scale=1.0,
            output_bias_forward_width_scale=1.0,
            output_relu_width_scale=1.0,
            output_head="source_follower",
            transfer_topology="main",
            cycle_ns=4.0,
            sample_offset_ns=2.95,
            tran_step_ps=10.0,
            cap_f=4.0,
            output_bias_offset_v=-0.20,
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    assert "Cvbo0p vbo0p 0 4f IC=0.54" in netlist
    assert "Cvbo0n vbo0n 0 4f IC=0.74" in netlist
    assert "Cvbo1p vbo1p 0 4f IC=0.74" in netlist
    assert "Cvbo1n vbo1n 0 4f IC=0.54" in netlist


def test_readout_capfit_greedy_selects_class_specific_mos_basis_caps() -> None:
    cap_values = [0.16, 0.90]
    labels = np.asarray([0, 1])
    pos_tensor = np.asarray(
        [
            [[0.0, 0.0], [2.0, 0.0]],
            [[0.0, 0.0], [0.0, 2.0]],
        ],
        dtype=float,
    )
    neg_tensor = np.zeros_like(pos_tensor)

    fit = readout_capfit.greedy_cap_fit(
        pos_tensor=pos_tensor,
        neg_tensor=neg_tensor,
        labels=labels,
        outputs=2,
        cap_values=cap_values,
        sweeps=4,
        seed=0,
    )
    init = readout_capfit.cap_init_from_indices(fit["pos_idx"], fit["neg_idx"], cap_values, hidden=1)

    assert fit["accuracy"] == pytest.approx(1.0)
    assert init["vw00p"] == pytest.approx(0.90)
    assert init["vbo1p"] == pytest.approx(0.90)
    assert init["vw10p"] == pytest.approx(0.16)
    assert init["vbo0p"] == pytest.approx(0.16)


def test_readout_capfit_branch_tensor_interpolates_weight_cap_candidates() -> None:
    rows = []
    for act in [0.0, 1.0]:
        for weight in [0.0, 1.0]:
            rows.append(
                {
                    "design": "split_signed_v1",
                    "style": "gate_stack",
                    "branch": "pos",
                    "act_v": act,
                    "weight_v": weight,
                    "score_delta_v": act * weight,
                }
            )
    tensor = readout_capfit.branch_candidate_tensor(
        pd.DataFrame(rows),
        branch="pos",
        source_values=np.asarray([[0.25], [0.75]], dtype=float),
        cap_values=[0.0, 0.5, 1.0],
    )

    assert tensor.shape == (1, 3, 2)
    assert tensor[0, 1, 0] == pytest.approx(0.125)
    assert tensor[0, 1, 1] == pytest.approx(0.375)


def test_readout_capfit_summed_transfer_sums_currents_before_output_nonlinearity() -> None:
    cap_values = [0.16, 0.90]
    labels = np.asarray([0, 1])
    pos_current_tensor = np.asarray(
        [
            [[0.0, 0.0], [2.0, 0.0]],
            [[0.0, 0.0], [0.0, 2.0]],
        ],
        dtype=float,
    )
    neg_current_tensor = np.zeros_like(pos_current_tensor)

    fit = readout_capfit.greedy_cap_fit_summed_transfer(
        pos_current_tensor=pos_current_tensor,
        neg_current_tensor=neg_current_tensor,
        labels=labels,
        outputs=2,
        cap_values=cap_values,
        sweeps=4,
        seed=0,
        transfer_curve={"current": [0.0, 1.0, 2.0], "drop": [0.0, 0.8, 1.2]},
    )
    init = readout_capfit.cap_init_from_indices(fit["pos_idx"], fit["neg_idx"], cap_values, hidden=1)

    assert fit["accuracy"] == pytest.approx(1.0)
    assert init["vw00p"] == pytest.approx(0.90)
    assert init["vbo1p"] == pytest.approx(0.90)
    assert init["vw10p"] == pytest.approx(0.16)
    assert init["vbo0p"] == pytest.approx(0.16)


def test_readout_capfit_transfer_curve_is_monotone_from_current_to_mirror_drop() -> None:
    rows = []
    for branch, scale in [("pos", 1.0), ("neg", 0.8)]:
        for idx, current in enumerate([0.0, 0.1, 0.2]):
            rows.append(
                {
                    "design": "split_signed_v1",
                    "style": "gate_stack",
                    "branch": branch,
                    "act_v": float(idx),
                    "weight_v": 0.5,
                    "score_delta_v": current * scale,
                }
            )
    current_surface = pd.DataFrame(rows)
    mirror_surface = current_surface.copy()
    mirror_surface["score_delta_v"] = [0.0, 0.3, 0.2, 0.0, 0.25, 0.8]

    curve = readout_capfit.transfer_curve_from_surfaces(current_surface, mirror_surface)

    assert curve["current"][0] == pytest.approx(0.0)
    assert curve["drop"][0] == pytest.approx(0.0)
    assert np.all(np.diff(curve["drop"]) >= -1e-12)
    assert readout_capfit.apply_transfer(np.asarray([0.15]), curve)[0] >= 0.3


def test_readout_write_selectivity_deck_uses_production_readout_write_fragment() -> None:
    netlist = readout_write_selectivity.build_readout_write_selectivity_netlist(
        label=1,
        hidden_values=[0.8, 0.45],
        outputs=3,
        center_v=0.64,
        readout_write_high_v=1.0,
        readout_write_low_v=0.16,
    )

    assert "Vact0 act0 0 PWL(0n 0.8" in netlist
    assert "Vdp1 dp1 0 PWL(0n 1.1" in netlist
    assert "Vdn0 dn0 0 PWL(0n 1.1" in netlist
    assert "Vdn2 dn2 0 PWL(0n 1.1" in netlist
    assert "Mvw10p_ch_b whigh bwd vw10p_ch_b" in netlist
    assert "Mvw10n_flow_d vw10n_flow_a dp1 wlow" in netlist
    assert ".meas tran bias1_signed_delta PARAM='(vbo1p_after-vbo1n_after)-(vbo1p_before-vbo1n_before)'" in netlist
    assert ".meas tran row1_signed_delta PARAM='d_w10_signed+d_w11_signed'" in netlist
    assert ".meas tran row1_common_delta PARAM='row1_pos_delta+row1_neg_delta'" in netlist
    assert "print row0_signed_delta row0_pos_delta row0_neg_delta row0_common_delta" in netlist
    assert "bias0_signed_delta bias0_common_delta" in netlist
    assert "row2_signed_delta row2_pos_delta row2_neg_delta row2_common_delta" in netlist
    assert "readout_flow_updates" not in netlist


def test_readout_write_selectivity_requires_write_rails_to_straddle_center() -> None:
    with pytest.raises(ValueError, match="bounded write rails must straddle"):
        readout_write_selectivity.build_readout_write_selectivity_netlist(
            label=0,
            hidden_values=[0.8],
            outputs=3,
            center_v=0.64,
            readout_write_high_v=0.58,
            readout_write_low_v=0.16,
        )


def test_readout_write_selectivity_can_probe_reversed_flow_polarity() -> None:
    netlist = readout_write_selectivity.build_readout_write_selectivity_netlist(
        label=1,
        hidden_values=[0.8, 0.45],
        outputs=3,
        center_v=0.64,
        readout_write_high_v=1.0,
        readout_write_low_v=0.16,
        readout_flow_polarity="reversed",
    )

    assert "Mvw10n_flow_d vw10n_flow_a dn1 wlow" in netlist
    assert "Mvw10p_ch_d vw10p_ch_a dn1 vw10p" in netlist
    assert "Mvw00n_flow_d vw00n_flow_a dn0 wlow" in netlist


def test_readout_write_selectivity_can_probe_write_gate_device() -> None:
    netlist = readout_write_selectivity.build_readout_write_selectivity_netlist(
        label=1,
        hidden_values=[0.8, 0.45],
        outputs=3,
        center_v=0.64,
        readout_write_high_v=1.0,
        readout_write_low_v=0.16,
        readout_flow_polarity="reversed",
        readout_flow_write_mode="bounded_charge_only",
        readout_write_gate_device="NREL",
    )

    assert "Mvw10p_ch_d vw10p_ch_a dn1 vw10p 0 NREL" in netlist
    assert "Mvw10n_ch_d vw10n_ch_a dp1 vw10n 0 NREL" in netlist


def test_readout_write_selectivity_can_probe_boosted_pre_trace_mode() -> None:
    netlist = readout_write_selectivity.build_readout_write_selectivity_netlist(
        label=1,
        hidden_values=[0.8, 0.45],
        outputs=3,
        center_v=0.64,
        readout_write_high_v=1.0,
        readout_write_low_v=0.16,
        flow_pre_store="synapse_boost",
        boosted_pre_offset_v=0.75,
    )

    assert "Vfprb10 fprb10 0 PWL(0n 1.2 8n 1.2)" in netlist
    assert "Vfprb11 fprb11 0 PWL(0n 1.2 8n 1.2)" in netlist
    assert "Mvw10p_ch_a vw10p_ch_b fprb10 vw10p_ch_a 0 NREL" in netlist
    assert "Mvw11p_ch_a vw11p_ch_b fprb11 vw11p_ch_a 0 NREL" in netlist


def test_readout_write_selectivity_can_probe_spike_pre_trace_mode() -> None:
    netlist = readout_write_selectivity.build_readout_write_selectivity_netlist(
        label=1,
        hidden_values=[0.8, 0.1],
        outputs=3,
        center_v=0.64,
        readout_write_high_v=1.0,
        readout_write_low_v=0.16,
        flow_pre_store="synapse_spike",
        spike_pre_threshold_v=0.2,
    )

    assert "Vfprg10 fprg10 0 PWL(0n 1.2 8n 1.2)" in netlist
    assert "Vfprg11 fprg11 0 PWL(0n 0 8n 0)" in netlist
    assert "Mvw10p_ch_a vw10p_ch_b fprg10 vw10p_ch_a 0 NREL" in netlist
    assert "Mvw11p_ch_a vw11p_ch_b fprg11 vw11p_ch_a 0 NREL" in netlist


def test_readout_write_selectivity_can_use_dynamic_spike_pre_store() -> None:
    netlist = readout_write_selectivity.build_readout_write_selectivity_netlist(
        label=1,
        hidden_values=[0.8, 0.1],
        outputs=3,
        center_v=0.64,
        readout_write_high_v=1.0,
        readout_write_low_v=0.16,
        flow_pre_store="synapse_spike",
        dynamic_pre_store=True,
        flow_pre_spike_ref_v=0.3,
        readout_flow_write_mode="bounded_pmos_charge_only",
        write_error_exclusion="diffpair_bleed",
    )

    assert "Vfprg10" not in netlist
    assert "Vspikeref spikeref 0 0.3" in netlist
    assert "Vrstf rstf 0 PWL" in netlist
    assert "Vfwd fwd 0 PWL" in netlist
    assert "Cfprg10 fprg10 0 2f IC=0" in netlist
    assert "Mspike_fprbar10_act fprm10 act0 spikeref 0 NSENSE" in netlist
    assert "Mvw10p_pch_a vw10p_pch_g fprg10 vw10p 0 NREL" in netlist
    assert ".meas tran fprg10_write FIND V(fprg10)" in netlist
    assert "print row0_signed_delta" in netlist
    assert "fprg10_write fprbar10_write" in netlist


def test_readout_write_selectivity_can_probe_state_bias_and_center_topologies() -> None:
    netlist = readout_write_selectivity.build_readout_write_selectivity_netlist(
        label=1,
        hidden_values=[0.8, 0.45],
        outputs=3,
        center_v=0.64,
        output_bias_update_width_u=0.00002,
        output_bias_write_pre_gate="bias",
        readout_write_state_gate_mode="state_window",
        readout_center_pull_width_u=0.000005,
        output_bias_center_pull_width_u=0.000005,
        readout_center_pull_mode="state_high",
    )

    assert "Mvbo1p_ch_a vbo1p_ch_b bias vbo1p_ch_a" in netlist
    assert "Mvbo1p_ch_s whigh vbo1p vbo1p_ch_s" in netlist
    assert "Mvbo1p_center_s vbo1p_center_g vbo1p wcenter" in netlist
    assert "Mvw10p_ch_s whigh vw10p vw10p_ch_s" in netlist
    assert "Mvw10p_center_s vw10p_center_g vw10p wcenter" in netlist
    assert ".meas tran bias1_common_delta" in netlist


def test_readout_write_selectivity_can_decouple_row_and_bias_polarity() -> None:
    netlist = readout_write_selectivity.build_readout_write_selectivity_netlist(
        label=1,
        hidden_values=[0.8, 0.45],
        outputs=3,
        output_bias_update_width_u=0.00002,
        output_bias_write_pre_gate="bias",
        readout_flow_polarity="reversed",
        output_bias_flow_polarity="normal",
    )

    assert "Mvw10n_flow_d vw10n_flow_a dn1 wlow" in netlist
    assert "Mvw10p_ch_d vw10p_ch_a dn1 vw10p" in netlist
    assert "Mvbo1p_ch_d vbo1p_ch_a dn1 vbo1p" in netlist
    assert "Mvbo1n_ch_d vbo1n_ch_a dp1 vbo1n" in netlist


def test_composed_readout_error_loop_uses_production_forward_error_and_write_fragments() -> None:
    netlist = readout_error_loop.build_readout_error_loop_netlist(
        label=1,
        hidden_values=[0.8, 0.45],
        weights=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        biases=[0.0, 0.0, 0.0],
        readout_flow_polarity="reversed",
        readout_flow_write_mode="bounded_charge_discharge",
        readout_write_high_v=1.0,
        readout_write_low_v=0.16,
    )

    assert "Mo10pos_f o10p1 fwd scorep1" in netlist
    assert "Vwphigh wphigh 0 1" in netlist
    assert "Voutg outg 0 PWL" in netlist
    assert "Mdp1_t0 vdd t1 dp1_t 0 NSENSE W=96u" in netlist
    assert "Mdn0_nt0 vdd nt0 dn0_nt 0 NSENSE W=64u" in netlist
    assert "Mvw10n_flow_d vw10n_flow_a dn1 wnlow" in netlist
    assert "Mvw10p_ch_b wphigh bwd vw10p_ch_b" in netlist
    assert "Mvw10p_ch_d vw10p_ch_a dn1 vw10p" in netlist
    assert "Mvbo1p_ch_d vbo1p_ch_b dp1 vbo1p" in netlist
    assert "Mvbo1n_ch_d vbo1n_ch_b dn1 vbo1n" in netlist
    assert ".meas tran score1_fwd PARAM='scorep1_fwd-scoren1_fwd'" in netlist
    assert ".meas tran row1_signed_delta PARAM='d_w10_signed+d_w11_signed'" in netlist
    assert ".meas tran row1_common_delta PARAM='row1_pos_delta+row1_neg_delta'" in netlist
    assert ".meas tran row1_fwd_common_delta PARAM='row1_fwd_pos_delta+row1_fwd_neg_delta'" in netlist
    assert ".meas tran bias1_common_delta" in netlist
    assert canonical_circuit_netlist(prepare_netlist_for_simulator(netlist, "ngspice")) == canonical_circuit_netlist(
        prepare_netlist_for_simulator(netlist, "Xyce")
    )


def test_composed_readout_error_loop_can_match_synapse_spike_pretrace_writes() -> None:
    netlist = readout_error_loop.build_readout_error_loop_netlist(
        label=1,
        hidden_values=[0.8, 0.45],
        weights=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        biases=[0.0, 0.0, 0.0],
        flow_pre_store="synapse_spike",
        readout_flow_polarity="reversed",
        readout_flow_write_mode="bounded_charge_only",
        readout_write_error_exclusion="diffpair_bleed",
        readout_update_width_u=0.0005,
        readout_dp_gate_update_width_u=0.002,
        readout_dn_gate_update_width_u=0.00001,
        output_bias_update_width_u=0.0,
    )

    assert "Vspikeref spikeref 0 0.3" in netlist
    assert "Cfprg10 fprg10 0 2f IC=0" in netlist
    assert "Mspike_fprg10_p vdd fprbar10 fprg10 vdd PMOS W=4u" in netlist
    assert "Mvw10p_ch_a vw10p_ch_b fprg10 vw10p_ch_a" in netlist
    assert "Mvw10p_ch_d vw10p_ch_a rwpos1 vw10p 0 NSENSE W=1e-05u" in netlist
    assert "Mvw10n_ch_a vw10n_ch_b fprg10 vw10n_ch_a" in netlist
    assert "Mvw10n_ch_d vw10n_ch_a rwneg1 vw10n 0 NSENSE W=0.002u" in netlist


def test_composed_readout_error_loop_can_use_diffpair_pmos_write_selector() -> None:
    netlist = readout_error_loop.build_readout_error_loop_netlist(
        label=1,
        hidden_values=[0.8, 0.45],
        weights=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        biases=[0.0, 0.0, 0.0],
        readout_flow_polarity="normal",
        readout_flow_write_mode="bounded_pmos_charge_only",
        readout_write_error_exclusion="diffpair_bleed",
        readout_write_error_exclusion_width_u=8.0,
        output_bias_update_width_u=0.0,
    )

    assert "Mrwsel1_pos_sel rwsel1_posbar dp1 rwsel1_src 0 NSENSE W=8u" in netlist
    assert "Mrwsel1_neg_sel rwsel1_negbar dn1 rwsel1_src 0 NSENSE W=8u" in netlist
    assert "Mvw10p_pch_s vw10p_pch_b rwsel1_posbar wphigh vdd PMOS" in netlist
    assert "Mvw10n_pch_s vw10n_pch_b rwsel1_negbar wnhigh vdd PMOS" in netlist
    assert ".meas tran row1_common_delta PARAM='row1_pos_delta+row1_neg_delta'" in netlist


def test_composed_readout_error_loop_can_add_center_pull_damping() -> None:
    netlist = readout_error_loop.build_readout_error_loop_netlist(
        label=1,
        hidden_values=[0.8, 0.45],
        weights=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        biases=[0.0, 0.0, 0.0],
        readout_flow_write_mode="bounded_charge_discharge",
        readout_center_pull_width_u=0.000005,
        output_bias_center_pull_width_u=0.000003,
        readout_center_pull_mode="state_high",
        readout_pos_center_pull_v=0.62,
        readout_neg_center_pull_v=0.66,
        output_bias_pos_center_pull_v=0.61,
        output_bias_neg_center_pull_v=0.67,
    )

    assert "Vwcenterp wcenterp 0 0.62" in netlist
    assert "Vwcentern wcentern 0 0.66" in netlist
    assert "Vwbocenterp wbocenterp 0 0.61" in netlist
    assert "Vwbocentern wbocentern 0 0.67" in netlist
    assert "Mvw10p_center_s vw10p_center_g vw10p wcenterp" in netlist
    assert "Mvw10n_center_s vw10n_center_g vw10n wcentern" in netlist
    assert "Mvbo1p_center_s vbo1p_center_g vbo1p wbocenterp" in netlist
    assert "Mvbo1n_center_s vbo1n_center_g vbo1n wbocentern" in netlist


def test_composed_readout_error_loop_initializes_mirror_outputs_low_true() -> None:
    netlist = readout_error_loop.build_readout_error_loop_netlist(
        label=1,
        hidden_values=[0.8, 0.45],
        weights=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        biases=[0.0, 0.0, 0.0],
        output_head="split_score_diode_mirror_caps",
        error_rule="ce_mirror_winner_limited",
        readout_flow_polarity="reversed",
        readout_flow_write_mode="bounded_charge_discharge",
        readout_write_high_v=1.0,
        readout_write_low_v=0.16,
    )

    assert "Cout0 out0 0 20f IC=1.2" in netlist
    assert "Rout0 out0 vdd 1G" in netlist
    assert "Cscorepm0 scorepm0 0 20f IC=1.2" in netlist
    assert "Mdp1_out0 dp1_t out1 dp1_out 0 NREL" in netlist
    assert "Mdn1_pred0 cesrc out1 dn1_pred vdd PMOS" in netlist


def test_readout_transfer_split_score_harness_does_not_duplicate_production_score_caps(tmp_path: Path) -> None:
    old_hidden = direct_flow.HIDDEN
    old_outputs = direct_flow.OUTPUTS
    direct_flow.set_hidden_cells(2)
    direct_flow.set_output_count(3)
    try:
        separator = tmp_path / "separator.csv"
        separator.write_text(
            "out,bias,w0,w1\n"
            "0,0.01,0.02,-0.01\n"
            "1,-0.02,-0.01,0.03\n"
            "2,0.00,0.01,0.01\n"
        )
        activations = pd.DataFrame(
            [
                {"label": 0, "act0": 0.2, "act1": 0.7},
                {"label": 1, "act0": 0.6, "act1": 0.1},
            ]
        )
        netlist, _rows = readout_transfer.build_readout_transfer_netlist(
            activations=activations,
            separator_csv=separator,
            synapse_design="split_signed_v1",
            separator_scale=1.0,
            readout_center_v=0.64,
            score_reset_v=0.0,
            output_forward_width_scale=1.0,
            output_forward_pos_width_scale=1.0,
            output_forward_neg_width_scale=1.0,
            output_bias_forward_width_scale=1.0,
            output_relu_width_scale=1.0,
            output_head="source_follower",
            transfer_topology="split_score_caps",
            cycle_ns=4.0,
            sample_offset_ns=2.95,
            tran_step_ps=10.0,
            cap_f=4.0,
            readout_init_mode="csv_readout",
        )
    finally:
        direct_flow.set_hidden_cells(old_hidden)
        direct_flow.set_output_count(old_outputs)

    assert netlist.count("Cscorep0 scorep0") == 1
    assert netlist.count("Cscoren0 scoren0") == 1
    assert netlist.count("Mreset_scorep0 scorep0") == 1
    assert canonical_circuit_netlist(prepare_netlist_for_simulator(netlist, "ngspice")) == canonical_circuit_netlist(
        prepare_netlist_for_simulator(netlist, "Xyce")
    )


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
    assert "Mvbo0p_ch_d vbo0p_ch_b dn0 vbo0p" in updates
    assert "Mvbo0n_ch_d vbo0n_ch_b dp0 vbo0n" in updates


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
    assert "Mvbo0n_flow_d vbo0n_flow_b dn0 wlow 0 NSENSE" in updates
    assert "Mvbo0p_ch_b whigh bwd vbo0p_ch_b" in updates


def test_direct_flow_readout_can_tune_dp_dn_write_gate_widths() -> None:
    updates = direct_flow.readout_flow_updates(
        readout_update_width_u=0.02,
        output_bias_update_width_u=0.0,
        flow_pre_store="shared_node",
        readout_flow_polarity="reversed",
        readout_flow_write_mode="bounded_charge_discharge",
        readout_dp_gate_update_width_u=0.05,
        readout_dn_gate_update_width_u=0.007,
    )

    assert "Mvw00n_flow_d vw00n_flow_a dn0 wlow 0 NSENSE W=0.007u" in updates
    assert "Mvw00p_ch_d vw00p_ch_a dn0 vw00p 0 NSENSE W=0.007u" in updates
    assert "Mvw00p_flow_d vw00p_flow_a dp0 wlow 0 NSENSE W=0.05u" in updates
    assert "Mvw00n_ch_d vw00n_ch_a dp0 vw00n 0 NSENSE W=0.05u" in updates


def test_direct_flow_readout_gate_width_overrides_survive_diffpair_selector() -> None:
    updates = direct_flow.readout_flow_updates(
        readout_update_width_u=0.02,
        output_bias_update_width_u=0.0,
        flow_pre_store="shared_node",
        readout_flow_polarity="normal",
        readout_flow_write_mode="bounded_pmos_charge_discharge",
        write_error_exclusion="diffpair_bleed",
        readout_dp_gate_update_width_u=0.05,
        readout_dn_gate_update_width_u=0.007,
    )

    assert "Mvw00n_flow_d vw00n_flow_a rwpos0 wlow 0 NSENSE W=0.05u" in updates
    assert "Mvw00p_flow_d vw00p_flow_a rwneg0 wlow 0 NSENSE W=0.007u" in updates
    assert "Mvw00p_pch_s vw00p_pch_b rwsel0_posbar whigh vdd PMOS W=0.05u" in updates
    assert "Mvw00n_pch_s vw00n_pch_b rwsel0_negbar whigh vdd PMOS W=0.007u" in updates


def test_direct_flow_readout_can_size_error_gate_widths_by_action() -> None:
    updates = direct_flow.readout_flow_updates(
        readout_update_width_u=0.02,
        output_bias_update_width_u=0.0,
        flow_pre_store="synapse_spike",
        readout_flow_polarity="normal",
        readout_flow_write_mode="bounded_cmos_charge_discharge",
        write_error_exclusion="diffpair_bleed",
        readout_charge_update_width_u=0.5,
        readout_discharge_update_width_u=0.003,
        readout_dp_discharge_gate_update_width_u=0.004,
        readout_dp_charge_gate_update_width_u=0.05,
        readout_dn_discharge_gate_update_width_u=0.006,
        readout_dn_charge_gate_update_width_u=0.007,
    )

    assert "Mvw00n_flow_d vw00n_flow_a rwpos0 wlow 0 NSENSE W=0.004u" in updates
    assert "Mvw00p_flow_d vw00p_flow_a rwneg0 wlow 0 NSENSE W=0.006u" in updates
    assert "Mvw00p_cch_w vw00p_cch_w rwsel0_posbar whigh vdd PMOS W=0.05u" in updates
    assert "Mvw00n_cch_w vw00n_cch_w rwsel0_negbar whigh vdd PMOS W=0.007u" in updates


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


def test_direct_flow_readout_can_use_high_threshold_write_gate_device() -> None:
    updates = direct_flow.readout_flow_updates(
        readout_update_width_u=0.02,
        output_bias_update_width_u=0.0003,
        flow_pre_store="shared_node",
        readout_flow_polarity="normal",
        readout_flow_write_mode="bounded_charge_only",
        readout_write_gate_device="NMOS",
    )

    assert "Mvw00p_ch_b whigh bwd vw00p_ch_b 0 NREL W=0.02u" in updates
    assert "Mvw00p_ch_a vw00p_ch_b act0 vw00p_ch_a 0 NREL W=0.02u" in updates
    assert "Mvw00p_ch_d vw00p_ch_a dp0 vw00p 0 NMOS W=0.02u" in updates
    assert "Mvw00n_ch_d vw00n_ch_a dn0 vw00n 0 NMOS W=0.02u" in updates
    assert "Mvbo0p_ch_d vbo0p_ch_b dn0 vbo0p 0 NMOS W=0.0003u" in updates


def test_direct_flow_readout_can_use_diffpair_bleed_write_selector() -> None:
    updates = direct_flow.readout_flow_updates(
        readout_update_width_u=0.02,
        output_bias_update_width_u=0.0003,
        flow_pre_store="shared_node",
        readout_flow_polarity="normal",
        readout_flow_write_mode="bounded_charge_only",
        write_error_exclusion="diffpair_bleed",
        write_error_exclusion_width_u=8.0,
    )

    assert "Crwsel0_pos rwpos0 0 0.1f IC=0" in updates
    assert "Mrwsel0_pos_sel rwsel0_posbar dp0 rwsel0_src 0 NSENSE W=8u" in updates
    assert "Mrwsel0_neg_sel rwsel0_negbar dn0 rwsel0_src 0 NSENSE W=8u" in updates
    assert "Mrwsel0_tail rwsel0_src bwd 0 0 NMOS W=4u" in updates
    assert "Mrwsel0_pos_bleed rwpos0 bwd 0 0 NMOS W=0.2u" in updates
    assert "Mvw00p_ch_d vw00p_ch_a rwpos0 vw00p 0 NSENSE W=0.02u" in updates
    assert "Mvw00n_ch_d vw00n_ch_a rwneg0 vw00n 0 NSENSE W=0.02u" in updates


def test_direct_flow_readout_can_use_low_true_diffpair_bars_for_pmos_charge() -> None:
    updates = direct_flow.readout_flow_updates(
        readout_update_width_u=0.02,
        output_bias_update_width_u=0.0003,
        flow_pre_store="shared_node",
        readout_flow_polarity="normal",
        readout_flow_write_mode="bounded_pmos_charge_discharge",
        write_error_exclusion="diffpair_bleed",
        write_error_exclusion_width_u=8.0,
    )

    assert "Mrwsel0_pos_sel rwsel0_posbar dp0 rwsel0_src 0 NSENSE W=8u" in updates
    assert "Mrwsel0_neg_sel rwsel0_negbar dn0 rwsel0_src 0 NSENSE W=8u" in updates
    assert "Mvw00n_flow_d vw00n_flow_a rwpos0 wlow 0 NSENSE W=0.02u" in updates
    assert "Mvw00p_flow_d vw00p_flow_a rwneg0 wlow 0 NSENSE W=0.02u" in updates
    assert "Mvw00p_pch_s vw00p_pch_b rwsel0_posbar whigh vdd PMOS W=0.02u" in updates
    assert "Mvw00p_pch_a vw00p_pch_g act0 vw00p 0 NREL W=0.02u" in updates
    assert "Mvw00n_pch_s vw00n_pch_b rwsel0_negbar whigh vdd PMOS W=0.02u" in updates
    assert "Mvw00p_ch_d" not in updates
    assert "Mvbo0p_pch_s vbo0p_pch_b rwsel0_negbar whigh vdd PMOS W=0.0003u" in updates


def test_direct_flow_readout_can_use_pmos_charge_without_discharge_leg() -> None:
    updates = direct_flow.readout_flow_updates(
        readout_update_width_u=0.02,
        output_bias_update_width_u=0.0003,
        flow_pre_store="synapse_spike",
        readout_flow_polarity="normal",
        readout_flow_write_mode="bounded_pmos_charge_only",
        write_error_exclusion="diffpair_bleed",
        write_error_exclusion_width_u=8.0,
    )

    assert "Mvw00p_pch_s vw00p_pch_b rwsel0_posbar whigh vdd PMOS W=0.02u" in updates
    assert "Mvw00p_pch_a vw00p_pch_g fprg00 vw00p 0 NREL W=0.02u" in updates
    assert "Mvw00n_pch_s vw00n_pch_b rwsel0_negbar whigh vdd PMOS W=0.02u" in updates
    assert "Mvw00n_flow_d" not in updates
    assert "Mvw00p_flow_d" not in updates
    assert "Mvw00p_ch_d" not in updates


def test_direct_flow_readout_can_use_cmos_complementary_spike_pretrace_writer() -> None:
    updates = direct_flow.readout_flow_updates(
        readout_update_width_u=0.02,
        output_bias_update_width_u=0.0,
        flow_pre_store="synapse_spike",
        readout_flow_polarity="normal",
        readout_flow_write_mode="bounded_cmos_charge_discharge",
        write_error_exclusion="diffpair_bleed",
        write_error_exclusion_width_u=8.0,
    )

    assert "Mvw00n_flow_d vw00n_flow_a rwpos0 wlow 0 NSENSE W=0.02u" in updates
    assert "Mvw00p_flow_d vw00p_flow_a rwneg0 wlow 0 NSENSE W=0.02u" in updates
    assert "Mvw00p_cch_w vw00p_cch_w rwsel0_posbar whigh vdd PMOS W=0.02u" in updates
    assert "Mvw00p_cch_a vw00p fprbar00 vw00p_cch_w vdd PMOS W=0.02u" in updates
    assert "Mvw00n_cch_w vw00n_cch_w rwsel0_negbar whigh vdd PMOS W=0.02u" in updates
    assert "Mvw00n_cch_a vw00n fprbar00 vw00n_cch_w vdd PMOS W=0.02u" in updates
    assert "Mvw00p_ch_d" not in updates
    assert "Mvw00p_pch_s" not in updates


def test_direct_flow_readout_cmos_complementary_requires_spike_pretrace_and_diffpair() -> None:
    with pytest.raises(ValueError, match="requires write_error_exclusion='diffpair_bleed'"):
        direct_flow.readout_flow_updates(
            readout_update_width_u=0.02,
            output_bias_update_width_u=0.0,
            flow_pre_store="synapse_spike",
            readout_flow_polarity="normal",
            readout_flow_write_mode="bounded_cmos_charge_discharge",
        )
    with pytest.raises(ValueError, match="requires flow_pre_store='synapse_spike'"):
        direct_flow.readout_flow_updates(
            readout_update_width_u=0.02,
            output_bias_update_width_u=0.0,
            flow_pre_store="shared_node",
            readout_flow_polarity="normal",
            readout_flow_write_mode="bounded_cmos_charge_discharge",
            write_error_exclusion="diffpair_bleed",
        )


def test_direct_flow_readout_pmos_bias_can_use_explicit_polarity() -> None:
    updates = direct_flow.readout_flow_updates(
        readout_update_width_u=0.02,
        output_bias_update_width_u=0.0003,
        flow_pre_store="synapse_spike",
        readout_flow_polarity="normal",
        readout_flow_write_mode="bounded_pmos_charge_only",
        output_bias_flow_polarity="normal",
        write_error_exclusion="diffpair_bleed",
        write_error_exclusion_width_u=8.0,
    )

    assert "Mvbo0p_pch_s vbo0p_pch_b rwsel0_negbar whigh vdd PMOS W=0.0003u" in updates
    assert "Mvbo0n_pch_s vbo0n_pch_b rwsel0_posbar whigh vdd PMOS W=0.0003u" in updates


def test_direct_flow_readout_pmos_charge_requires_diffpair_selector() -> None:
    with pytest.raises(ValueError, match="requires write_error_exclusion='diffpair_bleed'"):
        direct_flow.readout_flow_updates(
            readout_update_width_u=0.02,
            output_bias_update_width_u=0.0,
            flow_pre_store="shared_node",
            readout_flow_polarity="normal",
            readout_flow_write_mode="bounded_pmos_charge_discharge",
        )
    with pytest.raises(ValueError, match="requires write_error_exclusion='diffpair_bleed'"):
        direct_flow.readout_flow_updates(
            readout_update_width_u=0.02,
            output_bias_update_width_u=0.0,
            flow_pre_store="shared_node",
            readout_flow_polarity="normal",
            readout_flow_write_mode="bounded_pmos_charge_only",
        )


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


def test_direct_flow_readout_action_widths_also_bound_output_bias_writes() -> None:
    discharge_width, charge_width = direct_flow.output_bias_flow_action_widths(
        output_bias_update_width_u=120.0,
        readout_charge_update_width_u=0.0002,
        readout_discharge_update_width_u=0.000002,
    )
    updates = direct_flow.readout_flow_updates(
        readout_update_width_u=120.0,
        readout_charge_update_width_u=0.0002,
        readout_discharge_update_width_u=0.000002,
        output_bias_update_width_u=120.0,
        flow_pre_store="shared_node",
        readout_flow_polarity="reversed",
        readout_flow_write_mode="bounded_charge_discharge",
    )

    assert discharge_width == pytest.approx(0.000002)
    assert charge_width == pytest.approx(0.0002)
    assert "Mvw00n_flow_b vw00n bwd vw00n_flow_b 0 NREL W=2e-06u" in updates
    assert "Mvw00p_ch_b whigh bwd vw00p_ch_b 0 NREL W=0.0002u" in updates
    assert "Mvbo0n_flow_b vbo0n bwd vbo0n_flow_b 0 NREL W=2e-06u" in updates
    assert "Mvbo0p_ch_b whigh bwd vbo0p_ch_b 0 NREL W=0.0002u" in updates
    assert not re.search(r"^Mvbo\S+ .* W=120u", updates, flags=re.MULTILINE)


def test_direct_flow_output_bias_write_can_match_synapse_pre_gate_topology() -> None:
    updates = direct_flow.readout_flow_updates(
        readout_update_width_u=0.02,
        readout_charge_update_width_u=0.011,
        readout_discharge_update_width_u=0.005,
        output_bias_update_width_u=0.02,
        flow_pre_store="shared_node",
        readout_flow_polarity="normal",
        readout_flow_write_mode="bounded_charge_discharge",
        output_bias_write_pre_gate="bias",
    )

    assert "Mvbo0n_flow_b vbo0n bwd vbo0n_flow_b 0 NREL W=0.005u" in updates
    assert "Mvbo0n_flow_a vbo0n_flow_b bias vbo0n_flow_a 0 NREL W=0.005u" in updates
    assert "Mvbo0n_flow_d vbo0n_flow_a dn0 wlow 0 NSENSE W=0.005u" in updates
    assert "Mvbo0p_ch_b whigh bwd vbo0p_ch_b 0 NREL W=0.011u" in updates
    assert "Mvbo0p_ch_a vbo0p_ch_b bias vbo0p_ch_a 0 NREL W=0.011u" in updates
    assert "Mvbo0p_ch_d vbo0p_ch_a dn0 vbo0p 0 NSENSE W=0.011u" in updates


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
            activation_sample_offsets_ns=[],
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
            activation_sample_offsets_ns=[],
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


def test_probe_measurement_can_expose_diffpair_selector_and_spike_trace_nodes() -> None:
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    original_input_rails = list(direct_flow.INPUT_RAILS)
    try:
        direct_flow.set_hidden_cells(2)
        direct_flow.set_output_count(3)
        direct_flow.set_input_rails(["x0", "x1"])
        measures, _prints = direct_flow.measure_lines(
            samples=[{"phase": "train", "label": 2, "pattern": 0, "apply_update": True}],
            hidden_apply_mode="direct",
            learning_mode="flow",
            hidden_delta_output_mode="raw",
            measure_detail="probe",
            readout_sample_offsets_ns=[2.95],
            activation_sample_offsets_ns=[],
            cmp_start_ns=3.25,
            cmp_end_ns=4.10,
            bwd_start_ns=6.75,
            apply_end_ns=11.20,
            backward_gate_mode="scheduled",
            hidden_delta_network_enabled=False,
            flow_pre_store="synapse_spike",
            readout_write_error_exclusion="diffpair_bleed",
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)
        direct_flow.set_input_rails(original_input_rails)

    assert ".meas tran rwpos0_0 FIND V(rwpos0)" in measures
    assert ".meas tran rwneg2_0 FIND V(rwneg2)" in measures
    assert ".meas tran rwsel1_posbar_0 FIND V(rwsel1_posbar)" in measures
    assert ".meas tran readout_write_select_net_2_0 PARAM='rwpos2_0-rwneg2_0'" in measures
    assert ".meas tran readout_write_select_bar_net_2_0 PARAM='rwsel2_negbar_0-rwsel2_posbar_0'" in measures
    assert ".meas tran fprg00_0 FIND V(fprg00)" in measures
    assert ".meas tran fprbar21_0 FIND V(fprbar21)" in measures
    assert "hidden_delta_net_0_0" not in measures


def test_split_score_caps_measurement_reports_differential_score() -> None:
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_output_count(3)
        samples = [
            {"phase": "initial_eval", "label": 2, "pattern": 0},
        ]
        measures, _prints = direct_flow.measure_lines(
            samples=samples,
            hidden_apply_mode="direct",
            learning_mode="flow",
            hidden_delta_output_mode="raw",
            measure_detail="light",
            readout_sample_offsets_ns=[2.95],
            activation_sample_offsets_ns=[],
            cmp_start_ns=3.25,
            cmp_end_ns=4.10,
            bwd_start_ns=6.75,
            apply_end_ns=11.20,
            backward_gate_mode="scheduled",
            hidden_delta_network_enabled=False,
            output_head="split_score_caps",
        )
    finally:
        direct_flow.set_output_count(original_outputs)

    assert "scorep2_0 FIND V(scorep2)" in measures
    assert "scoren2_0 FIND V(scoren2)" in measures
    assert "score2_0 PARAM='scorep2_0-scoren2_0'" in measures
    assert "score2_cmp_0 PARAM='scorep2_cmp_0-scoren2_cmp_0'" in measures


def test_measurement_can_record_hidden_activation_waveform_offsets() -> None:
    original_hidden = direct_flow.HIDDEN
    try:
        direct_flow.set_hidden_cells(2)
        measures, _prints = direct_flow.measure_lines(
            samples=[{"phase": "initial_eval", "label": 0, "pattern": 0}],
            hidden_apply_mode="direct",
            learning_mode="flow",
            hidden_delta_output_mode="raw",
            measure_detail="light",
            readout_sample_offsets_ns=[2.95],
            activation_sample_offsets_ns=[1.5, 2.5, 2.95],
            cmp_start_ns=3.25,
            cmp_end_ns=4.10,
            bwd_start_ns=6.75,
            apply_end_ns=11.20,
            backward_gate_mode="scheduled",
            hidden_delta_network_enabled=False,
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)

    assert ".meas tran act0_0 FIND V(act0) AT=2.95n" in measures
    assert ".meas tran act0_1500ps_0 FIND V(act0) AT=1.50n" in measures
    assert ".meas tran act1_2500ps_0 FIND V(act1) AT=2.50n" in measures
    assert ".meas tran act1_2950ps_0 FIND V(act1) AT=2.95n" in measures


def test_split_score_diffgate_measurement_reports_differential_score() -> None:
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_output_count(3)
        samples = [
            {"phase": "initial_eval", "label": 2, "pattern": 0},
        ]
        measures, _prints = direct_flow.measure_lines(
            samples=samples,
            hidden_apply_mode="direct",
            learning_mode="flow",
            hidden_delta_output_mode="raw",
            measure_detail="light",
            readout_sample_offsets_ns=[2.95],
            activation_sample_offsets_ns=[],
            cmp_start_ns=3.25,
            cmp_end_ns=4.10,
            bwd_start_ns=6.75,
            apply_end_ns=11.20,
            backward_gate_mode="scheduled",
            hidden_delta_network_enabled=False,
            output_head="split_score_diffgate",
        )
    finally:
        direct_flow.set_output_count(original_outputs)

    assert "scorep2_0 FIND V(scorep2)" in measures
    assert "scoren2_0 FIND V(scoren2)" in measures
    assert "score2_0 PARAM='scorep2_0-scoren2_0'" in measures
    assert "score2_cmp_0 PARAM='scorep2_cmp_0-scoren2_cmp_0'" in measures


def test_split_score_chargegate_measurement_reports_differential_score() -> None:
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_output_count(3)
        measures, _prints = direct_flow.measure_lines(
            samples=[{"phase": "initial_eval", "label": 2, "pattern": 0}],
            hidden_apply_mode="direct",
            learning_mode="flow",
            hidden_delta_output_mode="raw",
            measure_detail="light",
            readout_sample_offsets_ns=[2.95],
            activation_sample_offsets_ns=[],
            cmp_start_ns=3.25,
            cmp_end_ns=4.10,
            bwd_start_ns=6.75,
            apply_end_ns=11.20,
            backward_gate_mode="scheduled",
            hidden_delta_network_enabled=False,
            output_head="split_score_chargegate",
        )
    finally:
        direct_flow.set_output_count(original_outputs)

    assert "scorep2_0 FIND V(scorep2)" in measures
    assert "scoren2_0 FIND V(scoren2)" in measures
    assert "score2_0 PARAM='scorep2_0-scoren2_0'" in measures


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
            activation_sample_offsets_ns=[],
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
            activation_sample_offsets_ns=[],
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
            activation_sample_offsets_ns=[],
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


def test_lead_mistake_error_uses_configured_winner_polarity() -> None:
    score_error = direct_flow.error_cells("lead_mistake", latch_boost_width_u=0.0, lead_mode="senseamp")
    out_error = direct_flow.error_cells("lead_mistake", latch_boost_width_u=0.0, lead_mode="out_senseamp")

    assert "Full-swing lead-mistake rails" in score_error
    assert "Mdp0_l0 dp0_t lead10 dp0_l" in score_error
    assert "Mdn0_l0 dn0_t lead01 dn0_l" in score_error
    assert "Mdp1_l0 dp1_t lead01 dp1_l" in score_error
    assert "Mdn1_l0 dn1_t lead10 dn1_l" in score_error

    assert "Mdp0_l0 dp0_t lead01 dp0_l" in out_error
    assert "Mdn0_l0 dn0_t lead10 dn0_l" in out_error
    assert "Mdp1_l0 dp1_t lead10 dp1_l" in out_error
    assert "Mdn1_l0 dn1_t lead01 dn1_l" in out_error


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


def test_target_mistake_gate_stats_can_use_out_senseamp_polarity() -> None:
    train = pd.DataFrame(
        [
            {"label": 0, "score0_cmp": 0.90, "score1_cmp": 0.10, "lead01": 1.05, "lead10": 0.02, "bwd_signal": 1.05},
            {"label": 1, "score0_cmp": 0.10, "score1_cmp": 0.90, "lead01": 0.02, "lead10": 1.05, "bwd_signal": 1.05},
            {"label": 0, "score0_cmp": 0.10, "score1_cmp": 0.90, "lead01": 0.02, "lead10": 1.05, "bwd_signal": 0.02},
            {"label": 1, "score0_cmp": 0.90, "score1_cmp": 0.10, "lead01": 1.05, "lead10": 0.02, "bwd_signal": 0.02},
        ]
    )

    stats = direct_flow.target_mistake_gate_stats(train, lead_mode="out_senseamp", bwd_threshold_v=0.5)

    assert stats["target_mistake_reference"] == "out_senseamp"
    assert stats["target_mistake_score_loses_count"] == 2
    assert stats["target_mistake_bwd_match_fraction"] == pytest.approx(1.0)
    assert stats["target_mistake_bwd_false_positive_count"] == 0
    assert stats["target_mistake_bwd_false_negative_count"] == 0


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


def test_output_error_rail_stats_tracks_target_loss_actions() -> None:
    train = pd.DataFrame(
        [
            {
                "label": 0,
                "score0_cmp": 0.10,
                "score1_cmp": 0.20,
                "dp0": 1.05,
                "dn0": 0.02,
                "dp1": 0.02,
                "dn1": 1.05,
            },
            {
                "label": 1,
                "score0_cmp": 0.20,
                "score1_cmp": 0.10,
                "dp0": 0.02,
                "dn0": 1.05,
                "dp1": 1.05,
                "dn1": 0.02,
            },
            {
                "label": 0,
                "score0_cmp": 0.20,
                "score1_cmp": 0.10,
                "dp0": 1.05,
                "dn0": 0.02,
                "dp1": 0.02,
                "dn1": 0.02,
            },
        ]
    )

    stats = direct_flow.output_error_rail_stats(train, lead_mode="score_direct", rail_threshold_v=0.5)

    assert stats["output_error_rail_target_loses_count"] == 2
    assert stats["output_error_rail_open_count"] == 5
    assert stats["output_error_rail_false_positive_count"] == 1
    assert stats["output_error_rail_false_negative_count"] == 0
    assert stats["output_error_rail_match_fraction"] == pytest.approx(2 / 3)


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


def test_skip_train_refire_removes_late_diagnostic_forward_window() -> None:
    samples = [{"phase": "train", "apply_update": True}]

    with_refire = direct_flow.phases(
        samples,
        bwd_start_ns=6.75,
        apply_start_ns=9.25,
        apply_end_ns=11.20,
        cmp_start_ns=3.25,
        cmp_end_ns=4.10,
        learning_mode="flow",
        backward_gate_mode="scheduled",
        train_refire=True,
    )
    without_refire = direct_flow.phases(
        samples,
        bwd_start_ns=6.75,
        apply_start_ns=9.25,
        apply_end_ns=11.20,
        cmp_start_ns=3.25,
        cmp_end_ns=4.10,
        learning_mode="flow",
        backward_gate_mode="scheduled",
        train_refire=False,
    )

    assert "12.8n" in with_refire
    assert "15.6n" in with_refire
    assert "12.8n" not in without_refire
    assert "15.6n" not in without_refire


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


def test_target_wave_supports_tunable_complement_rails() -> None:
    samples = [
        {"phase": "train", "pattern": 0, "label": 0, "apply_update": True},
        {"phase": "train", "pattern": 1, "label": 1, "apply_update": True},
    ]
    stop_ns = len(samples) * direct_flow.CYCLE_NS

    target = direct_flow.target_wave(samples, 0, stop_ns, high_v=0.9, low_v=0.2)
    complement = direct_flow.target_wave(samples, 0, stop_ns, high_v=0.9, low_v=0.2, complement=True)

    assert "0n 0.9" in target
    assert "0n 0.2" in complement
    assert f"{direct_flow.CYCLE_NS:g}n 0.2" in target
    assert f"{direct_flow.CYCLE_NS:g}n 0.9" in complement


def test_onehot_error_cell_uses_complement_target_rails_for_multiclass_dn() -> None:
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_output_count(3)
        netlist = direct_flow.error_cells(
            "onehot",
            latch_boost_width_u=0.0,
            residual_target_width_u=96.0,
            residual_output_width_u=12.0,
            lead_mode="score_direct",
        )
    finally:
        direct_flow.set_output_count(original_outputs)

    assert "Mdp2_t0 vdd t2 dp2_t" in netlist
    assert "Mdn2_nt0 vdd nt2 dn2_nt" in netlist
    assert "W=96u" in netlist
    assert "W=12u" in netlist


def test_onehot_out_error_cell_gates_non_target_dn_by_output_activity() -> None:
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_output_count(3)
        netlist = direct_flow.error_cells(
            "onehot_out",
            latch_boost_width_u=0.0,
            residual_target_width_u=80.0,
            residual_output_width_u=10.0,
            lead_mode="score_direct",
        )
    finally:
        direct_flow.set_output_count(original_outputs)

    assert "Mdp1_t0 vdd t1 dp1_t" in netlist
    assert "Mdn1_nt0 vdd nt1 dn1_nt" in netlist
    assert "Mdn1_out0 dn1_nt out1 dn1_out" in netlist
    assert "Mdn1_err0 dn1_out err dn1" in netlist
    assert "W=80u" in netlist
    assert "W=10u" in netlist


def test_ce_out_error_cell_uses_target_low_and_non_target_output_gates() -> None:
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_output_count(3)
        netlist = direct_flow.error_cells(
            "ce_out",
            latch_boost_width_u=0.0,
            residual_target_width_u=80.0,
            residual_output_width_u=10.0,
            lead_mode="score_direct",
        )
    finally:
        direct_flow.set_output_count(original_outputs)

    assert "Mdp1_low0 vdd out1 dp1_low vdd PMOS" in netlist
    assert "Mdp1_t0 dp1_low t1 dp1_t" in netlist
    assert "Mdp1_err0 dp1_t err dp1" in netlist
    assert "Mdn1_nt0 vdd nt1 dn1_nt" in netlist
    assert "Mdn1_out0 dn1_nt out1 dn1_out" in netlist
    assert "Mdn1_err0 dn1_out err dn1" in netlist
    assert "W=80u" in netlist
    assert "W=10u" in netlist


def test_ce_split_score_error_cell_uses_split_score_caps_not_output_cap() -> None:
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_output_count(3)
        netlist = direct_flow.error_cells(
            "ce_split_score",
            latch_boost_width_u=0.0,
            residual_target_width_u=80.0,
            residual_output_width_u=10.0,
            lead_mode="score_direct",
        )
    finally:
        direct_flow.set_output_count(original_outputs)

    assert "Mdp1_low0 vdd scorep1 dp1_low vdd PMOS" in netlist
    assert "Mdp1_t0 dp1_low t1 dp1_t" in netlist
    assert "Mdp1_err0 dp1_t err dp1" in netlist
    assert "Mdn1_nt0 vdd nt1 dn1_nt" in netlist
    assert "Mdn1_score0 dn1_nt scorep1 dn1_score" in netlist
    assert "Mdn1_err0 dn1_score err dn1" in netlist
    assert " out1 " not in netlist
    assert "W=80u" in netlist
    assert "W=10u" in netlist


def test_ce_split_diffgate_error_cell_rejects_split_score_common_mode_locally() -> None:
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_output_count(3)
        netlist = direct_flow.error_cells(
            "ce_split_diffgate",
            latch_boost_width_u=0.0,
            residual_target_width_u=80.0,
            residual_output_width_u=10.0,
            lead_mode="score_direct",
        )
    finally:
        direct_flow.set_output_count(original_outputs)

    assert "Mdp1_low0 vdd scorep1 dp1_low vdd PMOS W=80u" in netlist
    assert "Mdp1_neg0 dp1_t scoren1 dp1_neg 0 NREL W=80u" in netlist
    assert "Mdp1_err0 dp1_neg err dp1" in netlist
    assert "Mdn1_low0 vdd scoren1 dn1_low vdd PMOS W=10u" in netlist
    assert "Mdn1_pos0 dn1_nt scorep1 dn1_pos 0 NREL W=10u" in netlist
    assert "Mdn1_err0 dn1_pos err dn1" in netlist
    assert " out1 " not in netlist


def test_ce_split_dpair_error_cell_uses_differential_pair_prediction_rail() -> None:
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_output_count(3)
        netlist = direct_flow.error_cells(
            "ce_split_dpair",
            latch_boost_width_u=0.0,
            residual_target_width_u=80.0,
            residual_output_width_u=10.0,
            lead_mode="score_direct",
        )
    finally:
        direct_flow.set_output_count(original_outputs)

    assert "Myp1_pos ybar1 scorep1 ysrc1 0 NREL W=10u" in netlist
    assert "Myp1_neg vdd scoren1 ysrc1 0 NREL W=10u" in netlist
    assert "Myp1_tail ysrc1 err 0 0 NMOS W=10u" in netlist
    assert "Mdp1_yp0 dp1_t ybar1 dp1_yp 0 NSENSE W=80u" in netlist
    assert "Mdp1_err0 dp1_yp err dp1" in netlist
    assert "Mdn1_pred0 vdd ybar1 dn1_pred vdd PMOS W=10u" in netlist
    assert "Mdn1_err0 dn1_nt err dn1" in netlist
    assert " out1 " not in netlist


def test_ce_split_compete_error_cell_uses_shared_tail_current_competition() -> None:
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_output_count(3)
        netlist = direct_flow.error_cells(
            "ce_split_compete",
            latch_boost_width_u=0.0,
            residual_target_width_u=80.0,
            residual_output_width_u=10.0,
            lead_mode="score_direct",
        )
    finally:
        direct_flow.set_output_count(original_outputs)

    assert "Mcc0_branch cc0_mid scorep0 ccsrc 0 NREL W=10u" in netlist
    assert "Mcc1_branch cc1_mid scorep1 ccsrc 0 NREL W=10u" in netlist
    assert "Mcc2_branch cc2_mid scorep2 ccsrc 0 NREL W=10u" in netlist
    assert "Mcc1_inh cc1_mid scoren1 ybar1 vdd PMOS W=10u" in netlist
    assert "Mcc_tail ccsrc err 0 0 NMOS W=10u" in netlist
    assert netlist.count("Mcc_tail") == 1
    assert "Mdp1_yp0 dp1_t ybar1 dp1_yp 0 NSENSE W=80u" in netlist
    assert "Mdn1_pred0 vdd ybar1 dn1_pred vdd PMOS W=10u" in netlist
    assert " out1 " not in netlist


def test_ce_split_current_error_cell_charges_dn_from_shared_source_without_ybar() -> None:
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_output_count(3)
        netlist = direct_flow.error_cells(
            "ce_split_current",
            latch_boost_width_u=0.0,
            residual_target_width_u=80.0,
            residual_output_width_u=10.0,
            lead_mode="score_direct",
        )
    finally:
        direct_flow.set_output_count(original_outputs)

    assert "Mdp1_t0 vdd t1 dp1_t 0 NSENSE W=80u" in netlist
    assert "Mdn0_nt0 cesrc nt0 dn0_nt 0 NSENSE W=10u" in netlist
    assert "Mdn1_inh0 dn1_nt scoren1 dn1_inh vdd PMOS W=10u" in netlist
    assert "Mdn2_score0 dn2_inh scorep2 dn2_score 0 NREL W=10u" in netlist
    assert "Mdn1_err0 dn1_score err dn1 0 NSENSE W=10u" in netlist
    assert "Ccesrc cesrc 0 0.2f IC=0" in netlist
    assert "Mreset_cesrc cesrc rste 0 0 NMOS W=4u" in netlist
    assert "Mcesrc vdd err cesrc 0 NSENSE W=10u" in netlist
    assert "ybar1" not in netlist
    assert " out1 " not in netlist


def test_ce_split_hybrid_suppresses_target_with_ybar_and_charges_dn_from_current() -> None:
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_output_count(3)
        netlist = direct_flow.error_cells(
            "ce_split_hybrid",
            latch_boost_width_u=0.0,
            residual_target_width_u=80.0,
            residual_output_width_u=10.0,
            lead_mode="score_direct",
        )
    finally:
        direct_flow.set_output_count(original_outputs)

    assert "Mcc0_branch cc0_mid scorep0 ccsrc 0 NREL W=10u" in netlist
    assert "Mcc_tail ccsrc err 0 0 NMOS W=10u" in netlist
    assert "Mdp1_yp0 dp1_t ybar1 dp1_yp 0 NSENSE W=80u" in netlist
    assert "Mdn1_nt0 cesrc nt1 dn1_nt 0 NSENSE W=10u" in netlist
    assert "Mdn1_score0 dn1_inh scorep1 dn1_score 0 NREL W=10u" in netlist
    assert "Mcesrc vdd err cesrc 0 NSENSE W=10u" in netlist
    assert netlist.count("Mcc_tail") == 1
    assert " out1 " not in netlist


def test_ce_split_limited_current_stars_target_and_nontarget_sources() -> None:
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_output_count(3)
        netlist = direct_flow.error_cells(
            "ce_split_limited",
            latch_boost_width_u=0.0,
            residual_target_width_u=2.0,
            residual_output_width_u=10.0,
            lead_mode="score_direct",
        )
    finally:
        direct_flow.set_output_count(original_outputs)

    assert "Mcc0_branch cc0_mid scorep0 ccsrc 0 NREL W=10u" in netlist
    assert "Mcc_tail ccsrc err 0 0 NMOS W=10u" in netlist
    assert "Mdp1_t0 ctsrc t1 dp1_t 0 NSENSE W=2u" in netlist
    assert "Mdp1_yp0 dp1_t ybar1 dp1_yp 0 NSENSE W=2u" in netlist
    assert "Mdn1_nt0 cesrc nt1 dn1_nt 0 NSENSE W=10u" in netlist
    assert "Cctsrc ctsrc 0 2f IC=0" in netlist
    assert "Mreset_ctsrc ctsrc rste 0 0 NMOS W=4u" in netlist
    assert "Mctsrc vdd err ctsrc 0 NSENSE W=2u" in netlist
    assert "Mcesrc vdd err cesrc 0 NSENSE W=10u" in netlist
    assert netlist.count("Mcc_tail") == 1
    assert " out1 " not in netlist


def test_onehot_limited_uses_balanced_current_limited_target_and_nontarget_sources() -> None:
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_output_count(3)
        netlist = direct_flow.error_cells(
            "onehot_limited",
            latch_boost_width_u=0.0,
            residual_target_width_u=2.0,
            residual_output_width_u=1.0,
            lead_mode="score_direct",
        )
    finally:
        direct_flow.set_output_count(original_outputs)

    assert "Mdp1_t0 ctsrc t1 dp1_t 0 NSENSE W=2u" in netlist
    assert "Mdp1_err0 dp1_t err dp1 0 NSENSE W=2u" in netlist
    assert "Mdn1_nt0 cesrc nt1 dn1_nt 0 NSENSE W=1u" in netlist
    assert "Mdn1_err0 dn1_nt err dn1 0 NSENSE W=1u" in netlist
    assert "Mctsrc vdd err ctsrc 0 NSENSE W=2u" in netlist
    assert "Mcesrc vdd err cesrc 0 NSENSE W=1u" in netlist
    assert "scorep1" not in netlist
    assert "out1" not in netlist


def test_onehot_limited_can_tune_error_source_rails_below_vdd() -> None:
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_output_count(3)
        netlist = direct_flow.error_cells(
            "onehot_limited",
            latch_boost_width_u=0.0,
            residual_target_width_u=2.0,
            residual_output_width_u=1.0,
            lead_mode="score_direct",
            error_target_source_v=1.0,
            error_nontarget_source_v=0.45,
        )
    finally:
        direct_flow.set_output_count(original_outputs)

    assert "Vctsrch ctsrch 0 1" in netlist
    assert "Vcesrch cesrch 0 0.45" in netlist
    assert "Mctsrc ctsrch err ctsrc 0 NSENSE W=2u" in netlist
    assert "Mcesrc cesrch err cesrc 0 NSENSE W=1u" in netlist


def test_onehot_limited_average_balance_resolves_source_rails() -> None:
    target, nontarget = direct_flow.resolve_error_source_rails(
        error_rule="onehot_limited",
        output_count=3,
        target_high_v=1.1,
        error_target_source_v=None,
        error_nontarget_source_v=None,
        error_source_balance="onehot_average",
        error_nontarget_balance_scale=1.0,
    )

    assert target == pytest.approx(1.1)
    assert nontarget == pytest.approx(0.55)


def test_onehot_limited_average_balance_respects_overrides_and_scale() -> None:
    target, nontarget = direct_flow.resolve_error_source_rails(
        error_rule="onehot_limited",
        output_count=5,
        target_high_v=1.1,
        error_target_source_v=1.0,
        error_nontarget_source_v=None,
        error_source_balance="onehot_average",
        error_nontarget_balance_scale=0.8,
    )

    assert target == pytest.approx(1.0)
    assert nontarget == pytest.approx(0.2)


def test_ce_mirror_limited_uses_mirror_caps_for_nontarget_error_rails() -> None:
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_output_count(3)
        netlist = direct_flow.error_cells(
            "ce_mirror_limited",
            latch_boost_width_u=0.0,
            residual_target_width_u=2.0,
            residual_output_width_u=10.0,
            lead_mode="score_direct",
        )
    finally:
        direct_flow.set_output_count(original_outputs)

    assert "Mdp1_t0 ctsrc t1 dp1_t 0 NSENSE W=2u" in netlist
    assert "Mdn1_nt0 cesrc nt1 dn1_nt 0 NSENSE W=10u" in netlist
    assert "Mdn1_pm0 dn1_nt scorepm1 dn1_pm vdd PMOS W=10u" in netlist
    assert "Mdn1_nm0 dn1_pm scorenm1 dn1_nm 0 NREL W=10u" in netlist
    assert "Mdn1_err0 dn1_nm err dn1 0 NSENSE W=10u" in netlist
    assert "Mcesrc vdd err cesrc 0 NSENSE W=10u" in netlist
    assert "Mctsrc vdd err ctsrc 0 NSENSE W=2u" in netlist
    assert "scorep1 dn1" not in netlist


def test_ce_mirror_winner_limited_uses_active_low_output_winner_gate() -> None:
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_output_count(3)
        netlist = direct_flow.error_cells(
            "ce_mirror_winner_limited",
            latch_boost_width_u=0.0,
            residual_target_width_u=2.0,
            residual_output_width_u=10.0,
            lead_mode="score_direct",
        )
    finally:
        direct_flow.set_output_count(original_outputs)

    assert "Mdp1_t0 ctsrc t1 dp1_t 0 NSENSE W=2u" in netlist
    assert "Mdp1_out0 dp1_t out1 dp1_out 0 NREL W=2u" in netlist
    assert "Mdp1_err0 dp1_out err dp1 0 NSENSE W=2u" in netlist
    assert "Mdn1_pred0 cesrc out1 dn1_pred vdd PMOS W=10u" in netlist
    assert "Mdn1_nt0 dn1_pred nt1 dn1_nt 0 NSENSE W=10u" in netlist
    assert "Mdn1_err0 dn1_nt err dn1 0 NSENSE W=10u" in netlist
    assert "Mcesrc vdd err cesrc 0 NSENSE W=10u" in netlist
    assert "Mctsrc vdd err ctsrc 0 NSENSE W=2u" in netlist


def test_ce_mirror_hybrid_limited_uses_hard_target_gate_and_soft_nontarget_mirror() -> None:
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_output_count(3)
        netlist = direct_flow.error_cells(
            "ce_mirror_hybrid_limited",
            latch_boost_width_u=0.0,
            residual_target_width_u=2.0,
            residual_output_width_u=10.0,
            lead_mode="score_direct",
        )
    finally:
        direct_flow.set_output_count(original_outputs)

    assert "Mdp1_t0 ctsrc t1 dp1_t 0 NSENSE W=2u" in netlist
    assert "Mdp1_out0 dp1_t out1 dp1_out 0 NREL W=2u" in netlist
    assert "Mdp1_err0 dp1_out err dp1 0 NSENSE W=2u" in netlist
    assert "Mdn1_nt0 cesrc nt1 dn1_nt 0 NSENSE W=10u" in netlist
    assert "Mdn1_pm0 dn1_nt scorepm1 dn1_pm vdd PMOS W=10u" in netlist
    assert "Mdn1_nm0 dn1_pm scorenm1 dn1_nm 0 NREL W=10u" in netlist
    assert "Mdn1_err0 dn1_nm err dn1 0 NSENSE W=10u" in netlist
    assert "Mdn1_pred0" not in netlist
    assert "Mmc_tail" not in netlist
    assert "Mcesrc vdd err cesrc 0 NSENSE W=10u" in netlist
    assert "Mctsrc vdd err ctsrc 0 NSENSE W=2u" in netlist


def test_ce_mirror_compete_limited_builds_shared_tail_mirror_ybar() -> None:
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_output_count(3)
        netlist = direct_flow.error_cells(
            "ce_mirror_compete_limited",
            latch_boost_width_u=0.0,
            residual_target_width_u=2.0,
            residual_output_width_u=10.0,
            lead_mode="score_direct",
        )
    finally:
        direct_flow.set_output_count(original_outputs)

    assert "Mmc1_inh mc1_mid scorepm1 ybar1 vdd PMOS W=10u" in netlist
    assert "Mmc1_branch mc1_mid scorenm1 mcsrc 0 NREL W=10u" in netlist
    assert "Mmc_tail mcsrc err 0 0 NMOS W=10u" in netlist
    assert netlist.count("Mmc_tail") == 1
    assert "Mdp1_t0 ctsrc t1 dp1_t 0 NSENSE W=2u" in netlist
    assert "Mdp1_yp0 dp1_t ybar1 dp1_yp 0 NSENSE W=2u" in netlist
    assert "Mdn1_pred0 cesrc ybar1 dn1_pred vdd PMOS W=10u" in netlist
    assert "Mdn1_nt0 dn1_pred nt1 dn1_nt 0 NSENSE W=10u" in netlist
    assert "Mcesrc vdd err cesrc 0 NSENSE W=10u" in netlist
    assert "Mctsrc vdd err ctsrc 0 NSENSE W=2u" in netlist


def test_multiclass_device_measurements_expose_all_outputs_for_python_argmax() -> None:
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(1)
        direct_flow.set_output_count(3)
        lines, prints = direct_flow.measure_lines(
            samples=[{"phase": "initial_eval", "pattern": 0, "label": 2, "epoch": 0}],
            hidden_apply_mode="direct",
            learning_mode="flow",
            hidden_delta_output_mode="raw",
            measure_detail="light",
            readout_sample_offsets_ns=[2.95, 3.25],
            activation_sample_offsets_ns=[],
            cmp_start_ns=3.25,
            cmp_end_ns=4.10,
            bwd_start_ns=6.75,
            apply_end_ns=11.20,
            backward_gate_mode="scheduled",
            hidden_delta_network_enabled=False,
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    assert ".meas tran target_out_0 FIND V(out2)" in lines
    assert ".meas tran out0_0 FIND V(out0)" in lines
    assert ".meas tran out1_0 FIND V(out1)" in lines
    assert ".meas tran out2_0 FIND V(out2)" in lines
    assert ".meas tran out2_3250ps_0 FIND V(out2)" in lines
    assert "other_out_0" not in lines
    assert "margin_0" not in lines
    assert prints == "print target_out_0"


def test_multiclass_perceptron_separability_reports_training_accuracy() -> None:
    x = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, -1.0],
        ]
    )
    labels = np.asarray([0, 1, 2])

    stats = direct_flow.perceptron_separable_array(x, labels)

    assert stats["perceptron_type"] == "multiclass"
    assert stats["classes"] == [0, 1, 2]
    assert stats["linearly_separable"] is True
    assert stats["training_accuracy"] == pytest.approx(1.0)


def test_prediction_histogram_and_class_accuracy_cover_collapsed_multiclass_output() -> None:
    df = pd.DataFrame(
        [
            {
                "label": 0,
                "predicted_label": 2,
                "correct": False,
                "score_predicted_label": 0,
                "score_correct": True,
            },
            {
                "label": 1,
                "predicted_label": 2,
                "correct": False,
                "score_predicted_label": 1,
                "score_correct": True,
            },
            {
                "label": 2,
                "predicted_label": 2,
                "correct": True,
                "score_predicted_label": 1,
                "score_correct": False,
            },
            {
                "label": 2,
                "predicted_label": 0,
                "correct": False,
                "score_predicted_label": 2,
                "score_correct": True,
            },
        ]
    )

    assert direct_flow.prediction_histogram(df, 3) == {"0": 1, "1": 0, "2": 3}
    assert direct_flow.prediction_histogram_for(df, 3, "score_predicted_label") == {
        "0": 1,
        "1": 2,
        "2": 1,
    }

    per_class = direct_flow.per_class_accuracy(df, 4)
    assert per_class["0"] == pytest.approx(0.0)
    assert per_class["1"] == pytest.approx(0.0)
    assert per_class["2"] == pytest.approx(0.5)
    assert per_class["3"] is None

    score_per_class = direct_flow.per_class_accuracy_for(df, 4, "score_correct")
    assert score_per_class["0"] == pytest.approx(1.0)
    assert score_per_class["1"] == pytest.approx(1.0)
    assert score_per_class["2"] == pytest.approx(0.5)
    assert score_per_class["3"] is None


def test_csv_readout_initializes_multiclass_capacitor_matrix(tmp_path: Path) -> None:
    path = tmp_path / "readout.csv"
    path.write_text(
        "out,bias,w0,w1\n"
        "0,1.0,2.0,-3.0\n"
        "1,-2.0,0.5,4.0\n"
        "2,0.0,-1.0,1.5\n"
    )
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(2)
        direct_flow.set_output_count(3)
        init = direct_flow.readout_init(
            seed=0,
            mode="csv_readout",
            separator_scale=0.04,
            separator_offset_v=0.0,
            readout_center_v=0.64,
            random_center_v=None,
            random_span_v=0.0,
            random_pos_center_v=None,
            random_neg_center_v=None,
            random_pos_span_v=None,
            random_neg_span_v=None,
            separator_csv=path,
            separator_phase="initial_eval",
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    assert init["vbo0p"] == pytest.approx(0.66)
    assert init["vbo0n"] == pytest.approx(0.62)
    assert init["vw00p"] == pytest.approx(0.68)
    assert init["vw00n"] == pytest.approx(0.60)
    assert init["vw01p"] == pytest.approx(0.58)
    assert init["vw01n"] == pytest.approx(0.70)
    assert init["vbo1p"] == pytest.approx(0.60)
    assert init["vbo1n"] == pytest.approx(0.68)
    assert init["vw11p"] == pytest.approx(0.72)
    assert init["vw11n"] == pytest.approx(0.56)
    assert set(k for k in init if k.startswith("vbo")) == {"vbo0p", "vbo0n", "vbo1p", "vbo1n", "vbo2p", "vbo2n"}


def test_csv_readout_rectified_initializes_one_branch_per_signed_weight(tmp_path: Path) -> None:
    path = tmp_path / "readout.csv"
    path.write_text(
        "out,bias,w0,w1\n"
        "0,1.0,2.0,-3.0\n"
        "1,-2.0,0.5,4.0\n"
        "2,0.0,-1.0,1.5\n"
    )
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(2)
        direct_flow.set_output_count(3)
        init = direct_flow.readout_init(
            seed=0,
            mode="csv_readout_rectified",
            separator_scale=0.04,
            separator_offset_v=0.0,
            readout_center_v=0.64,
            random_center_v=None,
            random_span_v=0.0,
            random_pos_center_v=None,
            random_neg_center_v=None,
            random_pos_span_v=None,
            random_neg_span_v=None,
            separator_csv=path,
            separator_phase="initial_eval",
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    assert init["vbo0p"] == pytest.approx(0.68)
    assert init["vbo0n"] == pytest.approx(0.64)
    assert init["vw00p"] == pytest.approx(0.72)
    assert init["vw00n"] == pytest.approx(0.64)
    assert init["vw01p"] == pytest.approx(0.64)
    assert init["vw01n"] == pytest.approx(0.76)
    assert init["vbo1p"] == pytest.approx(0.64)
    assert init["vbo1n"] == pytest.approx(0.72)
    assert init["vw11p"] == pytest.approx(0.80)
    assert init["vw11n"] == pytest.approx(0.64)


def test_csv_readout_sparse_rectified_turns_inactive_signed_branches_near_off(tmp_path: Path) -> None:
    path = tmp_path / "readout.csv"
    path.write_text(
        "out,bias,w0,w1\n"
        "0,1.0,2.0,-3.0\n"
        "1,-2.0,0.5,4.0\n"
        "2,0.0,-1.0,1.5\n"
    )
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(2)
        direct_flow.set_output_count(3)
        init = direct_flow.readout_init(
            seed=0,
            mode="csv_readout_sparse_rectified",
            separator_scale=0.04,
            separator_offset_v=0.0,
            readout_center_v=0.64,
            random_center_v=None,
            random_span_v=0.0,
            random_pos_center_v=None,
            random_neg_center_v=None,
            random_pos_span_v=None,
            random_neg_span_v=None,
            separator_csv=path,
            separator_phase="initial_eval",
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    assert init["vbo0p"] == pytest.approx(0.68)
    assert init["vbo0n"] == pytest.approx(direct_flow.SPARSE_READOUT_INACTIVE_V)
    assert init["vw00p"] == pytest.approx(0.72)
    assert init["vw00n"] == pytest.approx(direct_flow.SPARSE_READOUT_INACTIVE_V)
    assert init["vw01p"] == pytest.approx(direct_flow.SPARSE_READOUT_INACTIVE_V)
    assert init["vw01n"] == pytest.approx(0.76)
    assert init["vbo1p"] == pytest.approx(direct_flow.SPARSE_READOUT_INACTIVE_V)
    assert init["vbo1n"] == pytest.approx(0.72)
    assert init["vbo2p"] == pytest.approx(direct_flow.SPARSE_READOUT_INACTIVE_V)
    assert init["vbo2n"] == pytest.approx(direct_flow.SPARSE_READOUT_INACTIVE_V)


def test_csv_cap_state_readout_loads_exact_capacitor_voltages(tmp_path: Path) -> None:
    path = tmp_path / "readout_caps.csv"
    path.write_text(
        "cap,value\n"
        "vbo0p,0.11\n"
        "vbo0n,0.12\n"
        "vbo1p,0.21\n"
        "vbo1n,0.22\n"
        "vw00p,0.31\n"
        "vw00n,0.32\n"
        "vw01p,0.33\n"
        "vw01n,0.34\n"
        "vw10p,0.41\n"
        "vw10n,0.42\n"
        "vw11p,0.43\n"
        "vw11n,0.44\n"
    )
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(2)
        direct_flow.set_output_count(2)
        init = direct_flow.readout_init(
            seed=0,
            mode="csv_cap_state",
            separator_scale=99.0,
            separator_offset_v=0.0,
            readout_center_v=0.64,
            random_center_v=None,
            random_span_v=0.0,
            random_pos_center_v=None,
            random_neg_center_v=None,
            random_pos_span_v=None,
            random_neg_span_v=None,
            separator_csv=path,
            separator_phase="initial_eval",
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    assert init == {
        "vbo0p": pytest.approx(0.11),
        "vbo0n": pytest.approx(0.12),
        "vbo1p": pytest.approx(0.21),
        "vbo1n": pytest.approx(0.22),
        "vw00p": pytest.approx(0.31),
        "vw00n": pytest.approx(0.32),
        "vw01p": pytest.approx(0.33),
        "vw01n": pytest.approx(0.34),
        "vw10p": pytest.approx(0.41),
        "vw10n": pytest.approx(0.42),
        "vw11p": pytest.approx(0.43),
        "vw11n": pytest.approx(0.44),
    }


def test_csv_cap_state_readout_rejects_missing_capacitors(tmp_path: Path) -> None:
    path = tmp_path / "bad_readout_caps.csv"
    path.write_text(
        "cap,value\n"
        "vbo0p,0.11\n"
        "vbo0n,0.12\n"
        "vw00p,0.31\n"
        "vw00n,0.32\n"
    )
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(1)
        direct_flow.set_output_count(2)
        with pytest.raises(ValueError, match="missing capacitor states"):
            direct_flow.readout_init(
                seed=0,
                mode="csv_cap_state",
                separator_scale=1.0,
                separator_offset_v=0.0,
                readout_center_v=0.64,
                random_center_v=None,
                random_span_v=0.0,
                random_pos_center_v=None,
                random_neg_center_v=None,
                random_pos_span_v=None,
                random_neg_span_v=None,
                separator_csv=path,
                separator_phase="initial_eval",
            )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)


def test_two_hidden_probe_can_use_low_threshold_output_head() -> None:
    netlist = two_hidden_probe.output_forward(output_device="NSENSE", output_width_u=48.0)

    assert "Mrelu_o0 vdd score0 out0 0 NSENSE W=48u L=180n" in netlist
    assert "Mrelu_o1 vdd score1 out1 0 NSENSE W=48u L=180n" in netlist
    assert "signed readout from hidden layer 2" in netlist


def test_two_hidden_random_middle_topology_has_exact_three_in_and_out_edges() -> None:
    hidden = two_hidden_probe.HIDDEN
    fanins = two_hidden_probe.random_fanins(hidden, hidden, fan_in=3, fan_out=3, seed=7)
    fanouts = two_hidden_probe.fanouts_from_fanins(fanins, hidden)

    assert all(len(srcs) == 3 for srcs in fanins.values())
    assert all(len(dsts) == 3 for dsts in fanouts.values())
    assert all(len(set(srcs)) == 3 for srcs in fanins.values())


def test_two_hidden_sparse_topology_omits_absent_transistor_edges() -> None:
    hidden = two_hidden_probe.HIDDEN
    outputs = two_hidden_probe.OUTPUTS
    middle = two_hidden_probe.random_fanins(hidden, hidden, fan_in=3, fan_out=3, seed=11)
    readout = two_hidden_probe.random_fanins(hidden, outputs, fan_in=3, seed=19)
    present_middle_dst = 0
    present_middle_src = middle[present_middle_dst][0]
    absent_middle_dst, absent_middle_src = next(
        (dst, src)
        for dst in range(hidden)
        for src in range(hidden)
        if src not in middle[dst]
    )
    present_out = 0
    present_readout_h = readout[present_out][0]
    absent_out, absent_readout_h = next(
        (out, h)
        for out in range(outputs)
        for h in range(hidden)
        if h not in readout[out]
    )

    netlist = "\n".join(
        [
            two_hidden_probe.middle_caps(0.82, 0.08, 0.04, 20.0, middle),
            two_hidden_probe.readout_caps(0.72, 0.14, 20.0, readout),
            two_hidden_probe.temporary_caps(8.0, middle, readout),
            two_hidden_probe.resets(middle, readout),
            two_hidden_probe.hidden2_forward(middle),
            two_hidden_probe.output_forward("NREL", 24.0, readout),
            two_hidden_probe.hidden2_delta(readout),
            two_hidden_probe.hidden1_delta(middle),
            two_hidden_probe.update_cells(7.0, 4.0, middle, readout),
        ]
    )

    assert f"Cwm{present_middle_dst}{present_middle_src}p" in netlist
    assert f"Mm{present_middle_dst}{present_middle_src}pos_a" in netlist
    assert f"Cwm{absent_middle_dst}{absent_middle_src}p" not in netlist
    assert f"Mm{absent_middle_dst}{absent_middle_src}pos_a" not in netlist
    assert f"Mh1dp{absent_middle_src}{absent_middle_dst}a0" not in netlist
    assert f"Mgmp{absent_middle_dst}{absent_middle_src}_a" not in netlist

    assert f"Cvw{present_out}{present_readout_h}p" in netlist
    assert f"Mo{present_out}{present_readout_h}pos_a" in netlist
    assert f"Cvw{absent_out}{absent_readout_h}p" not in netlist
    assert f"Mo{absent_out}{absent_readout_h}pos_a" not in netlist
    assert f"Mh2dp{absent_readout_h}{absent_out}a0" not in netlist
    assert f"Mgvp{absent_out}{absent_readout_h}_a" not in netlist


def test_softmax_current_competition_netlist_uses_shared_tail_not_behavioral_divide() -> None:
    netlist = softmax_primitives.netlist(
        scores=(0.38, 0.35, 0.35),
        branch_model="NREL",
        branch_width_u=12.0,
        tail_width_u=16.0,
        tail_gate_v=0.55,
    )

    assert "Mbranch0 drain0 score0 src 0 NREL W=12u L=180n" in netlist
    assert "Mbranch1 drain1 score1 src 0 NREL W=12u L=180n" in netlist
    assert "Mtail src tail 0 0 NMOS W=16u L=180n" in netlist
    assert "exp(" not in netlist.lower()
    assert "/" not in netlist.split(".control", maxsplit=1)[0]


def test_split_score_current_competition_consumes_positive_and_negative_score_caps() -> None:
    netlist = softmax_primitives.split_score_netlist(
        score_pairs=((0.46, 0.35), (0.40, 0.35), (0.38, 0.35)),
        branch_model="NREL",
        branch_width_u=12.0,
        inhibit_width_u=9.0,
        tail_width_u=16.0,
        tail_gate_v=0.55,
    )

    assert "Vscorep0 scorep0 0 0.46" in netlist
    assert "Vscoren0 scoren0 0 0.35" in netlist
    assert "Minhibit0 mid0 scoren0 psrc0 vdd PMOS W=9u L=180n" in netlist
    assert "Mbranch0 mid0 scorep0 src 0 NREL W=12u L=180n" in netlist
    assert "Mtail src tail 0 0 NMOS W=16u L=180n" in netlist
    assert "exp(" not in netlist.lower()
    assert "/" not in netlist.split(".control", maxsplit=1)[0]

    ng = prepare_netlist_for_simulator(netlist, "ngspice")
    xy = prepare_netlist_for_simulator(netlist, "Xyce")
    assert ng.startswith(".title spicenn generated deck")
    assert xy.startswith(".title spicenn generated deck")
    assert ".control" in ng
    assert ".control" not in xy
    assert canonical_circuit_netlist(ng) == canonical_circuit_netlist(xy)


def test_split_differential_pair_current_competition_rejects_common_mode_before_class_share() -> None:
    netlist = softmax_primitives.split_differential_pair_netlist(
        score_pairs=((0.35, 0.25), (0.55, 0.48), (0.75, 0.70)),
        branch_model="NREL",
        branch_width_u=12.0,
        tail_width_u=16.0,
        tail_gate_v=0.55,
    )
    circuit_body = netlist.split(".control", maxsplit=1)[0]

    assert "Vscorep0 scorep0 0 0.35" in netlist
    assert "Vscoren0 scoren0 0 0.25" in netlist
    assert "Mpos0 posdrain0 scorep0 pairsrc0 0 NREL W=12u L=180n" in netlist
    assert "Mneg0 negdrain0 scoren0 pairsrc0 0 NREL W=12u L=180n" in netlist
    assert "Mtail0 pairsrc0 tail 0 0 NMOS W=16u L=180n" in netlist
    assert "Vdump0 vdd negdrain0 0" in netlist
    assert "exp(" not in circuit_body.lower()
    assert "/" not in circuit_body

    ng = prepare_netlist_for_simulator(netlist, "ngspice")
    xy = prepare_netlist_for_simulator(netlist, "Xyce")
    assert ".control" in ng
    assert ".control" not in xy
    assert canonical_circuit_netlist(ng) == canonical_circuit_netlist(xy)


def test_readout_branch_surface_gate_stack_uses_production_mos_stack() -> None:
    netlist = readout_branch_surface.readout_branch_netlist(
        design_name="split_signed_v1",
        branch="pos",
        act_v=0.8,
        weight_v=0.75,
    )
    circuit_body = netlist.split(".control", maxsplit=1)[0]

    assert "Mbranch_a vdd act rb0 0 NSENSE W=56u L=180n" in netlist
    assert "Mbranch_w rb0 weight rb1 0 NREL W=56u L=180n" in netlist
    assert "Mbranch_f rb1 fwd score 0 NREL W=56u L=180n" in netlist
    assert "Cscore score 0 10f IC=0" in netlist
    assert "B" not in "\n".join(line[:1] for line in circuit_body.splitlines() if line)
    assert "exp(" not in circuit_body.lower()

    ng = prepare_netlist_for_simulator(netlist, "ngspice")
    xy = prepare_netlist_for_simulator(netlist, "Xyce")
    assert ".control" in ng
    assert ".control" not in xy
    assert canonical_circuit_netlist(ng) == canonical_circuit_netlist(xy)


def test_readout_branch_surface_pass_act_style_uses_activation_as_source() -> None:
    netlist = readout_branch_surface.readout_branch_netlist(
        design_name="split_signed_passact_v1",
        branch="neg",
        act_v=0.6,
        weight_v=0.9,
        width_scale=0.5,
    )

    assert "Mbranch_w act weight rb1 0 NREL W=24u L=180n" in netlist
    assert "Mbranch_f rb1 fwd score 0 NREL W=24u L=180n" in netlist
    assert "Mbranch_a" not in netlist


def test_readout_branch_surface_can_measure_clamped_branch_current() -> None:
    netlist = readout_branch_surface.readout_branch_netlist(
        design_name="split_signed_v1",
        branch="pos",
        act_v=0.6,
        weight_v=0.7,
        score_reset_v=0.2,
        surface_mode="clamped_current",
    )

    assert "Vscore_clamp score 0 0.2" in netlist
    assert "Cscore score" not in netlist
    assert "Mscore_rst" not in netlist
    assert ".meas tran score_delta FIND I(Vscore_clamp)" in netlist
    assert "surface_mode=clamped_current" in netlist


def test_readout_branch_surface_can_measure_diode_clamped_voltage() -> None:
    netlist = readout_branch_surface.readout_branch_netlist(
        design_name="split_signed_v1",
        branch="pos",
        act_v=0.6,
        weight_v=0.7,
        surface_mode="diode_voltage",
        cap_f=0.4,
        diode_width_u=128.0,
    )

    assert "Mscore_diode score score 0 0 NSENSE W=128u" in netlist
    assert "Vscore_clamp" not in netlist
    assert "Cscore score 0 0.4f IC=0" in netlist
    assert ".meas tran score_after FIND V(score)" in netlist
    assert "surface_mode=diode_voltage" in netlist


def test_readout_branch_surface_can_measure_diode_current() -> None:
    netlist = readout_branch_surface.readout_branch_netlist(
        design_name="split_signed_v1",
        branch="pos",
        act_v=0.6,
        weight_v=0.7,
        surface_mode="diode_current",
        cap_f=0.4,
        diode_width_u=128.0,
    )

    assert "Vscore_sense score_sense 0 0" in netlist
    assert "Mscore_diode score score score_sense 0 NSENSE W=128u" in netlist
    assert "Vscore_clamp" not in netlist
    assert ".meas tran score_delta FIND I(Vscore_sense)" in netlist
    assert "surface_mode=diode_current" in netlist


def test_readout_branch_surface_can_measure_diode_mirror_voltage() -> None:
    netlist = readout_branch_surface.readout_branch_netlist(
        design_name="split_signed_v1",
        branch="pos",
        act_v=0.6,
        weight_v=0.7,
        surface_mode="diode_mirror_voltage",
        cap_f=0.4,
        diode_width_u=64.0,
        mirror_cap_f=7.0,
    )

    assert "Mscore_diode score score 0 0 NSENSE W=64u" in netlist
    assert "Cmirror mirror 0 7f IC=1.2" in netlist
    assert "Mmirror_rst vdd rstf mirror 0 NSENSE W=16u" in netlist
    assert "Mmirror_sink mirror score 0 0 NSENSE W=64u" in netlist
    assert ".meas tran mirror_before FIND V(mirror)" in netlist
    assert ".meas tran score_delta PARAM='mirror_before-mirror_after'" in netlist
    assert "surface_mode=diode_mirror_voltage" in netlist


def test_readout_branch_surface_summary_reports_monotone_product_quality() -> None:
    rows = []
    for act in [0.0, 0.5, 1.0]:
        for weight in [0.3, 0.6, 0.9]:
            rows.append(
                {
                    "design": "toy",
                    "style": "gate_stack",
                    "branch": "pos",
                    "act_v": act,
                    "weight_v": weight,
                    "score_delta_v": act * weight,
                }
            )
    summary = readout_branch_surface.summarize(pd.DataFrame(rows), tolerance=1e-12)
    group = summary["groups"][0]

    assert group["monotone_vs_weight_fraction"] == pytest.approx(1.0)
    assert group["monotone_vs_act_fraction"] == pytest.approx(1.0)
    assert group["zero_act_abs_delta_max_v"] == pytest.approx(0.0)
    assert group["bilinear_fit"]["r2"] == pytest.approx(1.0)


def test_readout_capfit_candidate_selection_respects_requested_metric() -> None:
    losses = [0.40, 0.30, 0.20]
    accuracies = [1.0, 0.5, 0.5]

    assert readout_capfit.select_candidate_index(losses, accuracies, "loss") == 2
    assert readout_capfit.select_candidate_index(losses, accuracies, "accuracy") == 0


def test_readout_capfit_uses_full_deck_bias_voltage_as_source() -> None:
    activations = pd.DataFrame(
        {
            "act0": [0.1, 0.2],
            "act1": [0.3, 0.4],
        }
    )

    sources = readout_capfit.source_matrix_from_activations(activations, 2, bias_source_v=1.2)

    assert sources.shape == (2, 3)
    assert np.allclose(sources, [[0.1, 0.3, 1.2], [0.2, 0.4, 1.2]])


def test_readout_array_eval_uses_production_readout_fragment_and_score_caps() -> None:
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(2)
        direct_flow.set_output_count(3)
        activations = pd.DataFrame(
            {
                "label": [1, 2],
                "act0": [0.20, 0.40],
                "act1": [0.30, 0.10],
            }
        )
        state = {
            "vbo0p": 0.2,
            "vbo0n": 0.1,
            "vbo1p": 0.7,
            "vbo1n": 0.1,
            "vbo2p": 0.2,
            "vbo2n": 0.1,
        }
        for out in range(3):
            for h in range(2):
                state[f"vw{out}{h}p"] = 0.4 + 0.1 * out
                state[f"vw{out}{h}n"] = 0.2 + 0.05 * h

        netlist, samples = readout_array_eval.readout_array_netlist(
            activations=activations,
            readout_state=state,
            design_name="split_signed_v1",
            output_head="split_score_none",
            hidden_cells=2,
            outputs=3,
            hidden_cap_f=4.0,
            score_reset_v=0.0,
            score_cap_f=10.0,
            output_cap_f=20.0,
            sample_ns=2.95,
            cycle_ns=16.0,
            activation_drive="ramp",
            activation_settle_ns=2.95,
            activation_sample_offsets_ns=[],
            readout_load_mode="forward_only",
            tran_step_ps=10.0,
            spice_accuracy_preset="fast",
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    assert len(samples) == 2
    assert "Vact0 act0 0 PWL" in netlist
    assert "Vact1 act1 0 PWL" in netlist
    assert "0.75n 0" in netlist
    assert "2.95n 0.2" in netlist
    assert "Cvbo1p vbo1p 0 4f IC=0.7" in netlist
    assert "Cscorep0 scorep0 0 10f IC=0" in netlist
    assert "Mreset_scorep0 scorep0 rstf scorecm 0 NMOS" in netlist
    assert "Mo00pos_f o00p1 fwd scorep0" in netlist
    assert "Mo2bpos_f o2bp1 fwd scorep2" in netlist
    assert "Score-rail-only split output" in netlist
    assert ".meas tran score2_1 PARAM='scorep2_1-scoren2_1'" in netlist
    assert "print scorep0_0 scoren0_0 score0_0 out0_0" in netlist


def test_readout_array_eval_can_measure_clamped_score_currents() -> None:
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(1)
        direct_flow.set_output_count(2)
        activations = pd.DataFrame({"label": [1], "act0": [0.55]})
        state = {
            "vbo0p": 0.2,
            "vbo0n": 0.1,
            "vbo1p": 0.3,
            "vbo1n": 0.2,
            "vw00p": 0.4,
            "vw00n": 0.2,
            "vw10p": 0.5,
            "vw10n": 0.3,
        }
        netlist, _samples = readout_array_eval.readout_array_netlist(
            activations=activations,
            readout_state=state,
            design_name="split_signed_v1",
            output_head="split_score_none",
            hidden_cells=1,
            outputs=2,
            hidden_cap_f=4.0,
            score_reset_v=0.2,
            score_cap_f=10.0,
            output_cap_f=20.0,
            sample_ns=2.95,
            cycle_ns=16.0,
            activation_drive="held",
            activation_settle_ns=2.95,
            activation_sample_offsets_ns=[],
            readout_load_mode="forward_only",
            score_sense_mode="clamped_current",
            tran_step_ps=10.0,
            spice_accuracy_preset="fast",
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    assert "Vscorep0_clamp scorep0 0 0.2" in netlist
    assert "Vscoren0_clamp scoren0 0 0.2" in netlist
    assert "Cscorep0 scorep0" not in netlist
    assert "Mreset_scorep0 scorep0" not in netlist
    assert "Mo00pos_f o00p1 fwd scorep0" in netlist
    assert ".meas tran scorep1_0 FIND I(Vscorep1_clamp)" in netlist
    assert ".meas tran scoren1_0 FIND I(Vscoren1_clamp)" in netlist
    assert ".meas tran out1_0 FIND V(out1)" not in netlist

    rows = readout_array_eval.rows_from_measures(
        {
            "score0_0": -0.20,
            "score1_0": -0.10,
            "scorep0_0": -0.30,
            "scoren0_0": -0.10,
            "scorep1_0": -0.25,
            "scoren1_0": -0.15,
        },
        activations,
        2,
    )
    assert rows.loc[0, "predicted_label"] == 1
    assert rows.loc[0, "out1"] == pytest.approx(-0.10)


def test_readout_array_eval_reports_column_centering_diagnostics() -> None:
    df = pd.DataFrame(
        {
            "label": [0, 1, 2],
            "score0": [10.0, 9.0, 8.0],
            "score1": [1.0, 3.0, 1.0],
            "score2": [0.0, 0.0, 2.0],
        }
    )

    metrics = readout_array_eval.score_diagnostics(df, 3)

    assert metrics["column_centered_accuracy"] == pytest.approx(1.0)
    assert metrics["inverted_accuracy"] == pytest.approx(0.0)
    assert metrics["score_mean_by_output_v"]["0"] == pytest.approx(9.0)
    assert metrics["score_span_by_output_v"]["1"] == pytest.approx(2.0)
    assert metrics["column_centered_min_margin_v"] > 0.0
    assert metrics["confusion_matrix"] == {
        "0": {"0": 1, "1": 0, "2": 0},
        "1": {"0": 1, "1": 0, "2": 0},
        "2": {"0": 1, "1": 0, "2": 0},
    }
    assert metrics["accuracy_by_label"] == {"0": pytest.approx(1.0), "1": pytest.approx(0.0), "2": pytest.approx(0.0)}


def test_readout_array_eval_can_attach_inactive_direct_flow_write_loads() -> None:
    original_hidden = direct_flow.HIDDEN
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_hidden_cells(1)
        direct_flow.set_output_count(2)
        activations = pd.DataFrame({"label": [1], "act0": [0.55]})
        state = {
            "vbo0p": 0.2,
            "vbo0n": 0.1,
            "vbo1p": 0.3,
            "vbo1n": 0.2,
            "vw00p": 0.4,
            "vw00n": 0.2,
            "vw10p": 0.5,
            "vw10n": 0.3,
        }
        netlist, _samples = readout_array_eval.readout_array_netlist(
            activations=activations,
            readout_state=state,
            design_name="split_signed_v1",
            output_head="split_score_none",
            hidden_cells=1,
            outputs=2,
            hidden_cap_f=4.0,
            score_reset_v=0.0,
            score_cap_f=10.0,
            output_cap_f=20.0,
            sample_ns=2.95,
            cycle_ns=16.0,
            activation_drive="held",
            activation_settle_ns=2.95,
            activation_sample_offsets_ns=[],
            readout_load_mode="flow_offstate",
            tran_step_ps=10.0,
            spice_accuracy_preset="fast",
        )
    finally:
        direct_flow.set_hidden_cells(original_hidden)
        direct_flow.set_output_count(original_outputs)

    assert "Vdp0 dp0 0 0" in netlist
    assert "Vdn1 dn1 0 0" in netlist
    assert "Cfpsrg" not in netlist
    assert "Cfprg00 fprg00 0 2f IC=0" in netlist
    assert "Mspike_fprbar00_act fprm00 act0 spikeref 0 NSENSE W=4u" in netlist
    assert "Mvw00p_pch_s vw00p_pch_b rwsel0_posbar wphigh vdd PMOS W=120u" in netlist
    assert "Mvw00p_pch_a vw00p_pch_g fprg00 vw00p 0 NREL W=120u" in netlist


def test_readout_array_eval_can_replay_measured_activation_waveform_columns() -> None:
    activations = pd.DataFrame(
        {
            "label": [0, 1],
            "act0": [0.30, 0.60],
            "act0_1500ps": [0.10, 0.20],
            "act0_2500ps": [0.25, 0.50],
        }
    )

    wave = readout_array_eval.activation_source_wave(
        activations,
        0,
        stop_ns=8.0,
        cycle_ns=4.0,
        mode="measured",
        settle_ns=2.95,
        sample_offsets_ns=[1.5, 2.5],
    )

    assert "1.5n 0.1" in wave
    assert "2.5n 0.25" in wave
    assert "5.5n 0.2" in wave
    assert "6.5n 0.5" in wave
    assert "8n 0.5" in wave


def test_readout_array_capfit_selects_requested_cap_variable_scope() -> None:
    state = {
        "vbo0p": 0.4,
        "vbo0n": 0.2,
        "vw00p": 0.7,
        "vw00n": 0.1,
    }

    assert readout_array_capfit.cap_variable_names(state, "bias") == ["vbo0n", "vbo0p"]
    assert readout_array_capfit.cap_variable_names(state, "readout") == ["vw00n", "vw00p"]
    assert readout_array_capfit.cap_variable_names(state, "all") == ["vbo0n", "vbo0p", "vw00n", "vw00p"]


def test_readout_array_capfit_variable_scope_respects_sparse_edges() -> None:
    state = {
        "vbo0p": 0.4,
        "vbo0n": 0.2,
        "vw00p": 0.7,
        "vw00n": 0.1,
        "vw01p": 0.8,
        "vw01n": 0.2,
    }
    fanins = {0: (1,)}

    assert readout_array_capfit.cap_variable_names(state, "readout", fanins) == ["vw01n", "vw01p"]
    assert readout_array_capfit.cap_variable_names(state, "all", fanins) == [
        "vbo0n",
        "vbo0p",
        "vw01n",
        "vw01p",
    ]


def test_readout_array_capfit_parses_cap_roles_with_multidigit_hidden_index() -> None:
    assert readout_array_capfit.cap_role("vbo2n", hidden_cells=12, outputs=3) == ("bias", 2, None, "n")
    assert readout_array_capfit.cap_role("vw210p", hidden_cells=12, outputs=3) == ("readout", 2, 10, "p")
    assert readout_array_capfit.cap_role("vw910p", hidden_cells=12, outputs=3) is None


def test_readout_array_capfit_orders_readout_variables_by_current_errors_and_activations() -> None:
    activations = pd.DataFrame(
        {
            "label": [1, 1],
            "act0": [0.05, 0.9],
            "act1": [0.95, 0.1],
            "act2": [0.20, 0.2],
        }
    )
    scores = np.array(
        [
            [0.9, 0.2, 0.0],
            [0.1, 0.4, 0.0],
        ]
    )
    labels = activations["label"].to_numpy(dtype=int)
    variables = ["vw12p", "vw20p", "vw10p", "vw01p", "vw11p", "vw00p"]

    ranked = readout_array_capfit.order_cap_variables(
        variables,
        activations,
        scores,
        labels,
        hidden_cells=3,
        outputs=3,
        variable_order="error_activation",
    )

    assert ranked[:2] == ["vw01p", "vw11p"]
    assert ranked.index("vw12p") < ranked.index("vw20p")
    assert readout_array_capfit.order_cap_variables(
        variables,
        activations,
        scores,
        labels,
        hidden_cells=3,
        outputs=3,
        variable_order="alphabetical",
    ) == sorted(variables)


def test_readout_array_capfit_can_group_readout_pn_pairs_for_mirror_search() -> None:
    variables = ["vw20n", "vw20p", "vw21n", "vw00p"]
    groups = readout_array_capfit.variable_groups(
        variables,
        hidden_cells=2,
        outputs=3,
        group_mode="readout_pn_mirror",
    )

    assert groups == [("vw20n", "vw20p"), ("vw21n",), ("vw00p",)]
    candidates = readout_array_capfit.group_candidate_states(
        {"vw20n": 0.1, "vw20p": 0.9},
        ("vw20n", "vw20p"),
        [0.1, 0.5, 0.9],
        group_mode="readout_pn_mirror",
    )

    assert candidates == [
        {"vw20p": pytest.approx(0.1), "vw20n": pytest.approx(0.9)},
        {"vw20p": pytest.approx(0.5), "vw20n": pytest.approx(0.5)},
    ]


def test_readout_array_capfit_prioritizes_measured_activation_offsets() -> None:
    activations = pd.DataFrame(
        {
            "label": [1],
            "act0": [0.01],
            "act0_1500ps": [0.02],
            "act0_2950ps": [0.90],
            "act1": [0.40],
        }
    )
    scores = np.array([[0.8, 0.1]])
    labels = activations["label"].to_numpy(dtype=int)

    ranked = readout_array_capfit.order_cap_variables(
        ["vw10p", "vw11p"],
        activations,
        scores,
        labels,
        hidden_cells=2,
        outputs=2,
        variable_order="error_activation",
    )

    assert readout_array_capfit.activation_priority_value(activations, 0, 0) == pytest.approx(0.90)
    assert ranked[0] == "vw10p"


def test_readout_array_capfit_metric_tuple_prioritizes_accuracy_then_margin() -> None:
    labels = np.array([0, 1], dtype=int)
    higher_accuracy = np.array([[2.0, 0.0], [0.5, 0.4]])
    lower_accuracy = np.array([[2.0, 0.0], [0.6, 0.4]])
    higher_margin = np.array([[2.0, 0.0], [0.0, 0.8]])

    assert readout_array_capfit.metric_tuple(higher_accuracy, labels, "accuracy") > readout_array_capfit.metric_tuple(
        lower_accuracy,
        labels,
        "accuracy",
    )
    assert readout_array_capfit.metric_tuple(higher_margin, labels, "margin") > readout_array_capfit.metric_tuple(
        higher_accuracy,
        labels,
        "margin",
    )
