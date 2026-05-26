from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPICE_DIR = ROOT / "spice"
if str(SPICE_DIR) not in sys.path:
    sys.path.insert(0, str(SPICE_DIR))

import run_device_spicenn_sparse_forward as sparse_forward  # noqa: E402
import run_device_spicenn_hidden_writer_probe as hidden_writer_probe  # noqa: E402


def test_spicenn_sparse_forward_netlist_uses_two_sparse_hidden_layers_and_signed_readout() -> None:
    text, topology = sparse_forward.netlist(
        x0=1.0,
        x1=0.0,
        bias=0.35,
        hidden_count=4,
        output_count=2,
        hidden1_fan_in=1,
        hidden2_fan_in=3,
        readout_fan_in=3,
        seed=7,
    )

    assert ".tran 5p 2.4n uic" in text
    assert "Vfwd fwd 0 PULSE" in text
    assert "Mrelu_h1_0_sense" in text
    assert "Mrelu_h2_0_sense" in text
    assert "Cscore0p score0p 0 10f IC=0" in text
    assert "Cscore0n score0n 0 10f IC=0" in text
    assert "Mro0_" in text
    assert "_p_a vdd h2_" in text
    assert "_p_w h2_" not in text
    assert ".meas tran score0_p FIND V(score0p) AT=2.2n" in text
    assert ".meas tran score0_n FIND V(score0n) AT=2.2n" in text
    assert all(len(srcs) == 2 for srcs in topology.hidden1.as_fanins().values())
    assert all(len(srcs) == 3 for srcs in topology.hidden2.as_fanins().values())
    assert all(len(srcs) == 3 for srcs in topology.readout.as_fanins().values())


def test_spicenn_sparse_forward_can_still_render_pass_activation_source_readout() -> None:
    text, _topology = sparse_forward.netlist(
        x0=1.0,
        x1=0.0,
        bias=0.35,
        hidden_count=4,
        output_count=2,
        hidden1_fan_in=1,
        hidden2_fan_in=3,
        readout_fan_in=3,
        seed=7,
        readout_branch_style="pass_act_source",
    )

    assert "_p_w h2_" in text
    assert "_p_a vdd h2_" not in text


def test_spicenn_sparse_forward_readout_bias_uses_full_swing_source() -> None:
    topology = sparse_forward.build_topology(
        hidden_count=4,
        output_count=2,
        hidden1_fan_in=2,
        hidden2_fan_in=3,
        readout_fan_in=3,
        seed=7,
        readout_bias=True,
    )

    source_nodes = sparse_forward.readout_source_nodes(topology.readout)

    assert source_nodes["bias"] == "vdd"
    assert source_nodes[0] == "h2_0"


def test_spicenn_sparse_forward_topology_can_use_balanced_fanout() -> None:
    topology = sparse_forward.build_topology(
        hidden_count=8,
        output_count=5,
        hidden1_fan_in=2,
        hidden2_fan_in=3,
        readout_fan_in=3,
        seed=7,
        readout_bias=True,
        hidden2_topology_mode="balanced_fanout",
        readout_topology_mode="balanced_fanout",
    )

    assert all(count == 3 for count in topology.hidden2.fanout_counts())
    assert max(topology.hidden2.fanin_counts()) - min(topology.hidden2.fanin_counts()) <= 1
    assert all(count == 3 for count in topology.readout.fanout_counts(sources=tuple(range(8))))
    readout_feature_fanin_counts = topology.readout.fanin_counts(exclude_sources=("bias",))
    assert max(readout_feature_fanin_counts) - min(readout_feature_fanin_counts) <= 1
    assert all(sources[0] == "bias" for sources in topology.readout.as_fanins().values())


def test_spicenn_sparse_forward_topology_can_use_ring_fanout() -> None:
    topology = sparse_forward.build_topology(
        hidden_count=8,
        output_count=5,
        hidden1_fan_in=2,
        hidden2_fan_in=3,
        readout_fan_in=3,
        seed=7,
        hidden2_topology_mode="ring_fanout",
        readout_topology_mode="ring_fanout",
    )

    assert all(count == 3 for count in topology.hidden2.fanout_counts())
    assert max(topology.hidden2.fanin_counts()) - min(topology.hidden2.fanin_counts()) <= 1
    assert all(count == 3 for count in topology.readout.fanout_counts(sources=tuple(range(8))))
    assert max(topology.readout.fanin_counts()) - min(topology.readout.fanin_counts()) <= 1
    for source, sinks in topology.readout.fanouts().items():
        assert len(sinks) == 3
        assert set(sinks) == {
            (int(source) * 3 + 7 + 1009 + offset) % topology.readout.sink_count for offset in range(3)
        }


def test_spicenn_sparse_forward_topology_accepts_many_input_rails() -> None:
    input_sources = tuple(f"x{i}" for i in range(8))

    topology = sparse_forward.build_topology(
        input_sources=input_sources,
        hidden_count=6,
        output_count=3,
        hidden1_fan_in=3,
        hidden2_fan_in=2,
        readout_fan_in=2,
        seed=11,
        readout_bias=True,
    )

    assert topology.hidden1.sources == ("bias", *input_sources)
    assert all(len(sources) == 4 for sources in topology.hidden1.as_fanins().values())
    assert all(sources[0] == "bias" for sources in topology.hidden1.as_fanins().values())
    assert set(source for sources in topology.hidden1.as_fanins().values() for source in sources) <= set(
        topology.hidden1.sources
    )


def test_spicenn_sparse_forward_netlist_renders_many_input_voltage_sources() -> None:
    input_values = {f"x{i}": 0.08 + 0.02 * i for i in range(8)}

    text, topology = sparse_forward.netlist(
        x0=None,
        x1=None,
        input_values=input_values,
        input_sources=tuple(input_values),
        bias=0.20,
        hidden_count=6,
        output_count=3,
        hidden1_fan_in=3,
        hidden2_fan_in=2,
        readout_fan_in=2,
        seed=11,
    )

    assert "Vin_x0 x0 0 DC 0.08" in text
    assert "Vin_x7 x7 0 DC 0.22" in text
    assert "Vx0 x0 0" not in text
    assert "Vx1 x1 0" not in text
    assert "Cw10_x" in text
    assert topology.hidden1.sources == ("bias", *tuple(input_values))


def test_spicenn_sparse_forward_extracts_mnist_style_sample_inputs() -> None:
    sample = {
        "label": 2,
        "input_rails": ["x0", "x1", "x2"],
        "inputs": {"x0": 0.10, "x1": 0.20, "x2": 0.30},
        "source_digit": 2,
    }

    assert sparse_forward.sample_input_values(sample) == pytest.approx({"x0": 0.10, "x1": 0.20, "x2": 0.30})
    assert sparse_forward.input_sources_from_samples([sample]) == ("x0", "x1", "x2")
    assert sparse_forward.sample_display_fields(sample)["source_digit"] == 2


def test_spicenn_sparse_forward_measures_readout_hidden_coverage() -> None:
    topology = sparse_forward.build_topology(
        hidden_count=6,
        output_count=3,
        hidden1_fan_in=2,
        hidden2_fan_in=2,
        readout_fan_in=2,
        seed=1,
        readout_topology_mode="ring_fanout",
    )

    coverage = sparse_forward.readout_hidden_coverage([0.0, 0.2, 0.0, 0.4, 0.0, 0.1], topology.readout)

    assert [row["fanin"] for row in coverage] == [4, 4, 4]
    assert coverage[0]["active_count"] == 2
    assert coverage[1]["active_count"] == 2
    assert coverage[2]["active_count"] == 2
    assert coverage[0]["active_sum"] == pytest.approx(0.5)


def test_spicenn_sparse_forward_rejects_unknown_topology_mode() -> None:
    with pytest.raises(ValueError, match="unknown hidden2 topology mode"):
        sparse_forward.build_topology(
            hidden_count=4,
            output_count=2,
            hidden1_fan_in=2,
            hidden2_fan_in=3,
            readout_fan_in=3,
            seed=7,
            hidden2_topology_mode="diagonal",
        )


def test_spicenn_hidden_writer_probe_renders_cmos_probe() -> None:
    text = hidden_writer_probe.netlist(
        write_mode="cmos_complementary_charge_discharge",
        delta_p_v=0.18,
        delta_n_v=0.12,
        pre_v=0.50,
        weight_p_v=0.40,
        weight_n_v=0.40,
        update_width_u=0.004,
        charge_width_u=0.004,
        discharge_width_u=0.004,
        selector_width_u=6.0,
        stop_ns=4.0,
    )

    assert "Mhwsel0_pos_sel hwsel0_posbar hd0p hwsel0_src 0 NSENSE W=6u L=180n" in text
    assert "Mwh0_0p_cch_w wh0_0p_cch_w hwsel0_posbar whigh vdd PMOS W=0.004u L=180n" in text
    assert "Vfwd fwd 0 PWL(" in text
    assert ".meas tran pretrace_gate_mid FIND V(fhpg0_0) AT=2.20n" in text
    assert ".meas tran hwpos_bar_mid FIND V(hwsel0_posbar) AT=2.20n" in text


def test_spicenn_hidden_writer_probe_derived_sign_metrics() -> None:
    measures = hidden_writer_probe.add_derived(
        {
            "p_before": 0.40,
            "n_before": 0.40,
            "p_after": 0.43,
            "n_after": 0.39,
        },
        delta_p_v=0.18,
        delta_n_v=0.12,
        pre_v=0.50,
    )

    assert measures["signed_delta"] == pytest.approx(0.04)
    assert measures["common_delta"] == pytest.approx(0.02)
    assert measures["expected_direction"] == pytest.approx(0.03)
    assert measures["sign_correct"] == 1.0


def test_spicenn_sparse_forward_rejects_empty_topology() -> None:
    with pytest.raises(ValueError, match="hidden_count and output_count"):
        sparse_forward.build_topology(
            hidden_count=0,
            output_count=2,
            hidden1_fan_in=1,
            hidden2_fan_in=3,
            readout_fan_in=3,
            seed=7,
        )


def test_spicenn_sparse_forward_adds_derived_score_differences() -> None:
    measures = sparse_forward.add_derived_measures(
        {
            "score0_p": 0.70,
            "score0_n": 0.20,
            "score1_p": 0.30,
            "score1_n": 0.60,
        },
        2,
    )

    assert measures["score0_diff"] == pytest.approx(0.50)
    assert measures["score1_diff"] == pytest.approx(-0.30)
    assert measures["score_margin_0_1"] == pytest.approx(0.80)


def test_spicenn_sparse_forward_vector_helpers_measure_feature_distance() -> None:
    measures = {"h2_0": 0.10, "h2_1": 0.40, "h2_2": 0.0}

    assert sparse_forward.activation_vector(measures, "h2", 3) == pytest.approx([0.10, 0.40, 0.0])
    assert sparse_forward.vector_l1([0.0, 1.0], [0.5, 0.25]) == pytest.approx(1.25)
    assert sparse_forward.vector_l2([0.0, 1.0], [0.3, 0.6]) == pytest.approx(0.5)


def test_spicenn_sparse_forward_class_balanced_replay_interleaves_labels() -> None:
    samples = sparse_forward.named_sample_stream("or_vs_zero", epochs=1)

    balanced = sparse_forward.class_balanced_replay(samples)

    labels = [int(sample["label"]) for sample in balanced]
    assert labels == [0, 1, 0, 1, 0, 1]
    assert sum(1 for sample in balanced if int(sample["label"]) == 0) == 3
    assert sum(1 for sample in balanced if int(sample["label"]) == 1) == 3
    assert (balanced[1]["x0"], balanced[1]["x1"]) == pytest.approx((0.0, 0.0))
    assert (balanced[3]["x0"], balanced[3]["x1"]) == pytest.approx((0.0, 0.0))


def test_spicenn_sparse_forward_readout_offset_initials_program_signed_rows() -> None:
    topology = sparse_forward.build_topology(
        hidden_count=3,
        output_count=2,
        hidden1_fan_in=2,
        hidden2_fan_in=2,
        readout_fan_in=2,
        seed=2,
    )

    weights = sparse_forward.signed_offset_readout_initials(
        topology.readout,
        target_output=0,
        signed_delta_v=0.08,
        center_v=0.55,
        mirror_other_outputs=True,
    )
    summary = sparse_forward.summarize_readout_signed_weights(weights, topology)

    assert all(weights[(0, int(source))] == pytest.approx((0.59, 0.51)) for source in topology.readout.as_fanins()[0])
    assert all(weights[(1, int(source))] == pytest.approx((0.51, 0.59)) for source in topology.readout.as_fanins()[1])
    assert summary["row0_signed_sum"] == pytest.approx(0.08 * len(topology.readout.as_fanins()[0]))
    assert summary["row1_signed_sum"] == pytest.approx(-0.08 * len(topology.readout.as_fanins()[1]))

    with pytest.raises(ValueError, match="outside the 0..1.2 V"):
        sparse_forward.signed_offset_readout_initials(
            topology.readout,
            target_output=0,
            signed_delta_v=1.4,
            center_v=0.55,
        )


def test_spicenn_sparse_forward_jittered_readout_initials_break_symmetry() -> None:
    topology = sparse_forward.build_topology(
        hidden_count=4,
        output_count=2,
        hidden1_fan_in=2,
        hidden2_fan_in=3,
        readout_fan_in=3,
        seed=7,
    )

    weights = sparse_forward.random_jittered_readout_initials(
        topology.readout,
        seed=2010,
        center_v=0.30,
        signed_span_v=0.02,
    )
    summary = sparse_forward.summarize_readout_signed_weights(weights, topology)

    assert any(abs(pos - neg) > 1e-6 for pos, neg in weights.values())
    assert all(0.29 <= pos <= 0.31 and 0.29 <= neg <= 0.31 for pos, neg in weights.values())
    assert summary["row0_signed_sum"] != pytest.approx(0.0)

    with pytest.raises(ValueError, match="outside the 0..1.2 V"):
        sparse_forward.random_jittered_readout_initials(
            topology.readout,
            seed=2010,
            center_v=0.01,
            signed_span_v=0.05,
        )


def test_spicenn_sparse_forward_random_hidden_initials_keep_bias_positive() -> None:
    topology = sparse_forward.build_topology(
        hidden_count=6,
        output_count=2,
        hidden1_fan_in=1,
        hidden2_fan_in=3,
        readout_fan_in=2,
        seed=1,
    )

    weights = sparse_forward.random_signed_hidden_initials(topology.hidden1, seed=18)

    assert all(weights[(sink, "bias")] == (0.88, 0.04) for sink in topology.hidden1.as_fanins())
    input_edges = [
        weights[(sink, source)]
        for sink, sources in topology.hidden1.as_fanins().items()
        for source in sources
        if source in {"x0", "x1"}
    ]
    assert (0.88, 0.04) in input_edges
    assert (0.04, 0.88) in input_edges


def test_spicenn_sparse_forward_weak_hidden_mode_keeps_random_signs_near_threshold() -> None:
    topology = sparse_forward.build_topology(
        hidden_count=6,
        output_count=2,
        hidden1_fan_in=1,
        hidden2_fan_in=3,
        readout_fan_in=2,
        seed=1,
    )

    weights = sparse_forward.hidden1_initials_for_mode(
        topology.hidden1,
        hidden_weight_mode="weak_signed_hidden1_random_bias",
        seed=18,
    )

    assert weights is not None
    values = set(weights.values())
    assert values <= {(0.68, 0.22), (0.22, 0.68)}
    assert (0.68, 0.22) in values
    assert (0.22, 0.68) in values


def test_spicenn_sparse_forward_readout_sensitivity_netlist_uses_programmed_signed_delta() -> None:
    text, topology, weights = sparse_forward.readout_sensitivity_netlist(
        x0=1.0,
        x1=1.0,
        bias=0.6,
        hidden_count=4,
        output_count=2,
        hidden1_fan_in=2,
        hidden2_fan_in=3,
        readout_fan_in=3,
        seed=7,
        signed_delta_v=0.10,
        target_output=0,
        center_v=0.55,
        readout_width_u=16.0,
        score_cap_f=6.0,
        readout_branch_style="pass_act_buffered",
        measure_time_ns=0.8,
    )

    first_source = int(topology.readout.as_fanins()[0][0])
    assert weights[(0, first_source)] == pytest.approx((0.60, 0.50))
    assert "* spicenn readout signed-weight sensitivity primitive" in text
    assert "Cscore0p score0p 0 6f IC=0" in text
    assert f"Cvw0_{first_source}p vw0_{first_source}p 0 4f IC=0.6" in text
    assert f"Cvw0_{first_source}n vw0_{first_source}n 0 4f IC=0.5" in text
    assert ".meas tran score0_p FIND V(score0p) AT=0.8n" in text
    assert "_actbuf_src" in text
    assert "W=16u L=180n" in text


def test_spicenn_sparse_forward_train_netlist_composes_error_and_update_cells() -> None:
    text, topology = sparse_forward.train_netlist(
        x0=1.0,
        x1=1.0,
        bias=0.6,
        label=0,
        hidden_count=4,
        output_count=1,
        hidden1_fan_in=2,
        hidden2_fan_in=3,
        readout_fan_in=3,
        seed=7,
        update_width_u=0.004,
        charge_width_u=None,
        discharge_width_u=None,
        update_write_mode="simple_charge_discharge",
        spike_ref_v=0.1,
    )

    assert ".tran 5p 5.8n uic" in text
    assert "Verr err 0 PULSE" in text
    assert "Vbwd bwd 0 PULSE" in text
    assert "Merr0_dp_t0 vdd t0 err0_dp_t 0 NSENSE" in text
    assert "Cbt" not in text
    assert "Chd" in text
    assert "Mbt0_" in text
    assert "_wp_ep_e vdd dp0" in text
    assert "_wp_ep_w bt0_" in text
    assert "_wn_ep_b" in text
    assert "Mvw0_" in text
    assert "_ch_d" in text
    assert "_flow_d" in text
    assert ".meas tran dp0_err FIND V(dp0) AT=3.15n" in text
    assert ".meas tran hd" in text
    assert ".meas tran vw0_" in text
    assert topology.readout.sink_count == 1
    assert all(len(srcs) == 3 for srcs in topology.readout.as_fanins().values())


def test_spicenn_sparse_forward_train_netlist_can_size_readout_bias_writer() -> None:
    text, topology = sparse_forward.train_netlist(
        x0=1.0,
        x1=1.0,
        bias=0.05,
        label=0,
        hidden_count=4,
        output_count=1,
        hidden1_fan_in=2,
        hidden2_fan_in=3,
        readout_fan_in=3,
        seed=7,
        update_width_u=0.004,
        charge_width_u=None,
        discharge_width_u=None,
        update_write_mode="simple_charge_discharge",
        spike_ref_v=0.1,
        readout_bias=True,
        readout_bias_update_scale=0.25,
    )

    assert "bias" in topology.readout.sources
    first_nonbias = next(source for source in topology.readout.as_fanins()[0] if source != "bias")
    assert "Mvw0_biasp_ch_d vw0_biasp_ch_a dp0 vw0_biasp 0 NSENSE W=0.001u L=180n" in text
    assert "Mvw0_biasn_flow_d vw0_biasn_flow_a dp0 wlow 0 NSENSE W=0.001u L=180n" in text
    assert (
        f"Mvw0_{first_nonbias}p_ch_d vw0_{first_nonbias}p_ch_a dp0 "
        f"vw0_{first_nonbias}p 0 NSENSE W=0.004u L=180n"
    ) in text


def test_spicenn_sparse_forward_train_netlist_can_update_hidden2_weights() -> None:
    text, topology = sparse_forward.train_netlist(
        x0=1.0,
        x1=1.0,
        bias=0.6,
        label=0,
        hidden_count=4,
        output_count=1,
        hidden1_fan_in=2,
        hidden2_fan_in=3,
        readout_fan_in=3,
        seed=7,
        update_width_u=0.004,
        hidden2_update_width_u=0.002,
        hidden2_delta_mode="relu_gate",
    )

    first_sink = 0
    first_source = topology.hidden2.as_fanins()[first_sink][0]
    assert "Cgdh2_0p gdh2_0p 0 6f IC=0" in text
    assert "Mh2_delta_gate_0_p_gate h2_delta_gate_0_p_in h2_0 h2_delta_gate_0_p_gate 0 NREL" in text
    assert "Mhwsel0_pos_inh vdd gdh2_0n hwsel0_possrc vdd PMOS" in text
    assert "Mhwsel0_pos_gate hwsel0_possrc gdh2_0p hwpos0 0 NSENSE" in text
    assert f"Mw2{first_sink}_{first_source}n_flow_d" in text
    assert f"hwpos{first_sink} 0 0 NSENSE W=0.002u" in text
    assert f".meas tran w2{first_sink}_{first_source}p_before FIND V(w2{first_sink}_{first_source}p) AT=3.35n" in text
    assert ".meas tran gdh2_0_p FIND V(gdh2_0p) AT=4.35n" in text


def test_spicenn_sparse_forward_hidden2_senseamp_cmos_transient_moves_signed_weights(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    topology = sparse_forward.build_topology(
        hidden_count=4,
        output_count=1,
        hidden1_fan_in=2,
        hidden2_fan_in=3,
        readout_fan_in=3,
        seed=7,
    )
    readout_weights = sparse_forward.signed_offset_readout_initials(
        topology.readout,
        target_output=0,
        signed_delta_v=0.20,
        center_v=0.55,
        mirror_other_outputs=False,
    )
    text, step_topology = sparse_forward.train_netlist(
        x0=1.0,
        x1=1.0,
        bias=0.6,
        label=0,
        hidden_count=4,
        output_count=1,
        hidden1_fan_in=2,
        hidden2_fan_in=3,
        readout_fan_in=3,
        seed=7,
        update_width_u=0.0005,
        charge_width_u=0.0005,
        discharge_width_u=0.000005,
        update_write_mode="cmos_complementary_charge_discharge",
        readout_weight_initials=readout_weights,
        hidden_weight_mode="centered_hidden2",
        hidden2_update_width_u=0.0002,
        hidden2_delta_mode="raw",
        hidden2_update_write_mode="senseamp_cmos_complementary_charge_discharge",
    )
    measures = sparse_forward.add_train_derived_measures(
        sparse_forward.run_netlist(
            ngspice_path,
            tmp_path / "spicenn_hidden2_senseamp_cmos_train.cir",
            text,
            timeout=20,
        ),
        step_topology,
    )

    assert measures["hd0_p"] > measures["hd0_n"] + 0.05
    assert measures["hwpos0_gate"] > 0.8
    assert measures["hwneg0_gate"] < 0.01
    assert measures["hidden2_row0_signed_delta"] > 0.0005
    assert 0.0 <= measures["hidden2_row0_common_delta"] < 0.002
    assert measures["hidden2_row2_signed_delta"] == pytest.approx(0.0, abs=1e-6)


def test_spicenn_sparse_forward_train_derived_measures_tracks_row_weight_motion() -> None:
    topology = sparse_forward.build_topology(
        hidden_count=3,
        output_count=1,
        hidden1_fan_in=2,
        hidden2_fan_in=2,
        readout_fan_in=2,
        seed=2,
    )
    sources = topology.readout.as_fanins()[0]
    measures: dict[str, float] = {"score0_p": 0.7, "score0_n": 0.2}
    for source in sources:
        measures[f"vw0_{source}p_before"] = 0.5
        measures[f"vw0_{source}n_before"] = 0.5
        measures[f"vw0_{source}p_after"] = 0.6
        measures[f"vw0_{source}n_after"] = 0.45

    derived = sparse_forward.add_train_derived_measures(measures, topology)

    assert derived["score0_diff"] == pytest.approx(0.5)
    assert derived["row0_signed_delta"] == pytest.approx(0.15 * len(sources))
    assert derived["row0_common_delta"] == pytest.approx(0.05 * len(sources))


def test_spicenn_sparse_forward_train_derived_measures_tracks_hidden2_weight_motion() -> None:
    topology = sparse_forward.build_topology(
        hidden_count=3,
        output_count=1,
        hidden1_fan_in=2,
        hidden2_fan_in=2,
        readout_fan_in=2,
        seed=2,
    )
    sources = topology.hidden2.as_fanins()[0]
    measures: dict[str, float] = {"score0_p": 0.7, "score0_n": 0.2}
    for source in topology.readout.as_fanins()[0]:
        measures[f"vw0_{source}p_before"] = 0.5
        measures[f"vw0_{source}n_before"] = 0.5
        measures[f"vw0_{source}p_after"] = 0.5
        measures[f"vw0_{source}n_after"] = 0.5
    for source in sources:
        measures[f"w20_{source}p_before"] = 0.5
        measures[f"w20_{source}n_before"] = 0.5
        measures[f"w20_{source}p_after"] = 0.55
        measures[f"w20_{source}n_after"] = 0.48

    derived = sparse_forward.add_train_derived_measures(measures, topology)

    assert derived["hidden2_row0_signed_delta"] == pytest.approx(0.07 * len(sources))
    assert derived["hidden2_row0_common_delta"] == pytest.approx(0.03 * len(sources))


def test_spicenn_sparse_forward_extracts_after_weights_for_next_step() -> None:
    topology = sparse_forward.build_topology(
        hidden_count=3,
        output_count=2,
        hidden1_fan_in=2,
        hidden2_fan_in=2,
        readout_fan_in=2,
        seed=2,
    )
    measures: dict[str, float] = {}
    for out, sources in topology.readout.as_fanins().items():
        for source in sources:
            measures[f"vw{out}_{source}p_after"] = 0.5 + out * 0.1 + int(source) * 0.01
            measures[f"vw{out}_{source}n_after"] = 0.4 + out * 0.1 + int(source) * 0.01

    weights = sparse_forward.extract_readout_initials_after(measures, topology)
    summary = sparse_forward.summarize_readout_signed_weights(weights, topology)

    assert set(weights) == {
        (out, int(source))
        for out, sources in topology.readout.as_fanins().items()
        for source in sources
    }
    assert all(pos > neg for pos, neg in weights.values())
    assert summary["row0_signed_sum"] == pytest.approx(0.2)


def test_spicenn_sparse_forward_train_step_transient_moves_readout_row_positive(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    text, topology = sparse_forward.train_netlist(
        x0=1.0,
        x1=1.0,
        bias=0.6,
        label=0,
        hidden_count=4,
        output_count=1,
        hidden1_fan_in=2,
        hidden2_fan_in=3,
        readout_fan_in=3,
        seed=7,
        update_width_u=0.004,
        charge_width_u=None,
        discharge_width_u=None,
        update_write_mode="simple_charge_discharge",
        spike_ref_v=0.1,
    )
    measures = sparse_forward.add_train_derived_measures(
        sparse_forward.run_netlist(ngspice_path, tmp_path / "spicenn_sparse_train_step.cir", text, timeout=20),
        topology,
    )

    assert measures["h2_0"] > 0.2
    assert measures["dp0_err"] > measures["dn0_err"]
    assert measures["row0_signed_delta"] > 0.005


def test_spicenn_sparse_forward_train_netlist_can_use_cmos_complementary_writer() -> None:
    text, _topology = sparse_forward.train_netlist(
        x0=1.0,
        x1=1.0,
        bias=0.6,
        label=0,
        hidden_count=4,
        output_count=1,
        hidden1_fan_in=2,
        hidden2_fan_in=3,
        readout_fan_in=3,
        seed=7,
        update_width_u=0.004,
        charge_width_u=0.0005,
        discharge_width_u=0.000005,
        update_write_mode="cmos_complementary_charge_discharge",
        spike_ref_v=0.1,
    )

    assert "Vspikeref spikeref 0 DC 0.1" in text
    assert "Mrwsel0_pos_sel rwsel0_posbar dp0 rwsel0_src 0 NSENSE" in text
    assert "Cfprg0_" in text
    assert "_cch_w" in text
    assert "_cch_a" in text
    assert "_pch_s" not in text
    assert ".meas tran rwpos0_gate FIND V(rwpos0) AT=4.35n" in text
    assert ".meas tran rwsel0_posbar_gate FIND V(rwsel0_posbar) AT=4.35n" in text
    assert ".meas tran fprg0_" in text
    assert ".meas tran fprbar0_" in text


def test_spicenn_sparse_forward_hidden_writes_auto_center_hidden2_weights() -> None:
    text, _topology = sparse_forward.train_netlist(
        x0=1.0,
        x1=1.0,
        bias=0.6,
        label=0,
        hidden_count=4,
        output_count=1,
        hidden1_fan_in=2,
        hidden2_fan_in=3,
        readout_fan_in=3,
        seed=7,
        update_width_u=0.0005,
        hidden_weight_mode="positive",
        hidden2_update_width_u=0.0002,
    )
    fixed_text, _topology = sparse_forward.train_netlist(
        x0=1.0,
        x1=1.0,
        bias=0.6,
        label=0,
        hidden_count=4,
        output_count=1,
        hidden1_fan_in=2,
        hidden2_fan_in=3,
        readout_fan_in=3,
        seed=7,
        update_width_u=0.0005,
        hidden_weight_mode="positive",
        hidden2_update_width_u=0.0,
    )

    assert sparse_forward.effective_hidden_weight_mode_for_updates("positive", 0.0002) == "centered_jittered_hidden2"
    hidden2_weights = sparse_forward.hidden2_initials_for_mode(
        _topology.hidden2,
        hidden_weight_mode="centered_jittered_hidden2",
        seed=7,
    )
    first_source = _topology.hidden2.as_fanins()[0][0]
    first_pos, first_neg = hidden2_weights[(0, first_source)]
    assert 0.32 <= first_pos <= 0.48
    assert 0.32 <= first_neg <= 0.48
    assert first_pos != pytest.approx(first_neg)
    assert f"Cw20_{first_source}p w20_{first_source}p 0 4f IC={first_pos:.12g}" in text
    assert f"Cw20_{first_source}n w20_{first_source}n 0 4f IC={first_neg:.12g}" in text
    assert "Cw20_0p w20_0p 0 4f IC=0.84" in fixed_text
    assert "Cw20_0n w20_0n 0 4f IC=0.04" in fixed_text


def test_spicenn_sparse_forward_train_netlist_can_use_hybrid_trace_spike_writer() -> None:
    text, _topology = sparse_forward.train_netlist(
        x0=1.0,
        x1=1.0,
        bias=0.6,
        label=0,
        hidden_count=4,
        output_count=1,
        hidden1_fan_in=2,
        hidden2_fan_in=3,
        readout_fan_in=3,
        seed=7,
        update_width_u=0.004,
        charge_width_u=0.0005,
        discharge_width_u=0.000005,
        update_write_mode="hybrid_trace_spike_charge_discharge",
        spike_ref_v=0.1,
    )

    assert "Vspikeref spikeref 0 DC 0.1" in text
    assert "Mrwsel0_pos_sel rwsel0_posbar dp0 rwsel0_src 0 NSENSE" in text
    assert "Cfprg0_" in text
    assert "_cch_w" in text
    assert "_pch_a" in text
    assert " fpr0_" in text
    assert ".meas tran rwpos0_gate FIND V(rwpos0) AT=4.35n" in text
    assert ".meas tran fprg0_" in text
    assert ".meas tran fprbar0_" in text


def test_spicenn_sparse_forward_train_netlist_defaults_to_low_pretrace_reference() -> None:
    text, _topology = sparse_forward.train_netlist(
        x0=1.0,
        x1=1.0,
        bias=0.6,
        label=0,
        hidden_count=4,
        output_count=1,
        hidden1_fan_in=2,
        hidden2_fan_in=3,
        readout_fan_in=3,
        seed=7,
        update_width_u=0.004,
        update_write_mode="hybrid_trace_spike_charge_discharge",
    )

    assert sparse_forward.DEFAULT_SPIKE_REF_V == pytest.approx(0.02)
    assert "Vspikeref spikeref 0 DC 0.02" in text


def test_spicenn_sparse_forward_summarizes_pretrace_gate_coverage() -> None:
    summary = sparse_forward.pretrace_gate_summary(
        {
            "0_0": {"gate": 1.2, "bar": 0.02},
            "0_1": {"gate": 0.04, "bar": 0.6},
            "0_2": {"gate": 0.0, "bar": 1.2},
            "0_3": {"gate": None, "bar": None},
        }
    )

    assert summary == {"total": 4, "active": 1, "weak": 1, "inactive": 1, "missing": 1}


def test_spicenn_sparse_forward_train_netlist_can_use_stored_analog_trace_writer() -> None:
    text, _topology = sparse_forward.train_netlist(
        x0=1.0,
        x1=1.0,
        bias=0.6,
        label=0,
        hidden_count=4,
        output_count=1,
        hidden1_fan_in=2,
        hidden2_fan_in=3,
        readout_fan_in=3,
        seed=7,
        update_width_u=0.004,
        update_write_mode="analog_trace_charge_discharge",
    )

    assert "Cfpr0_" in text
    assert "Mstore_fpr0_" in text
    assert "_ch_a" in text
    assert "Cfprg0_" not in text
    assert "rwsel0" not in text


def test_spicenn_sparse_forward_train_netlist_can_use_differential_targets() -> None:
    text, _topology = sparse_forward.train_netlist(
        x0=1.0,
        x1=1.0,
        bias=0.6,
        label=0,
        hidden_count=4,
        output_count=2,
        hidden1_fan_in=2,
        hidden2_fan_in=3,
        readout_fan_in=3,
        seed=7,
        update_width_u=0.004,
        charge_width_u=0.0005,
        discharge_width_u=0.000005,
        update_write_mode="cmos_complementary_charge_discharge",
        spike_ref_v=0.1,
        false_negative_target_v=0.4,
    )

    assert "Vtn0 tn0 0 DC 0" in text
    assert "Vtn1 tn1 0 DC 0.4" in text
    assert "Merr1_dn_tn0 vdd tn1 err1_dn_tn 0 NSENSE" in text
    assert "Merr1_dn_tn1 err1_dn_tn err dn1 0 NSENSE" in text


def test_spicenn_sparse_forward_train_netlist_can_quiet_nonlabel_error_rows() -> None:
    text, _topology = sparse_forward.train_netlist(
        x0=1.0,
        x1=1.0,
        bias=0.6,
        label=1,
        hidden_count=4,
        output_count=3,
        hidden1_fan_in=2,
        hidden2_fan_in=3,
        readout_fan_in=3,
        seed=7,
        update_width_u=0.004,
        update_write_mode="cmos_complementary_charge_discharge",
        false_negative_target_v=0.4,
        error_rule="target_only",
    )

    assert "Cdp0 dp0 0 6f IC=0" in text
    assert "Cdn0 dn0 0 6f IC=0" in text
    assert "Merr0_" not in text
    assert "Merr1_dp_t0 vdd t1 err1_dp_t 0 NSENSE W=32u L=180n" in text
    assert "Merr1_dn_sp0 vdd score1p err1_dn_sp 0 NSENSE W=24u L=180n" in text
    assert "Merr2_" not in text


def test_spicenn_sparse_forward_train_netlist_can_use_split_score_ce_limited_error_bank() -> None:
    text, _topology = sparse_forward.train_netlist(
        x0=1.0,
        x1=0.0,
        bias=0.6,
        label=2,
        hidden_count=4,
        output_count=3,
        hidden1_fan_in=2,
        hidden2_fan_in=3,
        readout_fan_in=3,
        seed=7,
        update_width_u=0.004,
        error_rule="ce_split_limited",
    )

    assert "Vt2 t2 0 DC 1.1" in text
    assert "Vnt2 nt2 0 DC 0" in text
    assert "Vnt0 nt0 0 DC 1.1" in text
    assert "Mceerr_ctsrc vdd err ctsrc 0 NSENSE W=32u L=180n" in text
    assert "Mceerr_cesrc vdd err cesrc 0 NSENSE W=24u L=180n" in text
    assert "Mceerr_cc_tail ccsrc err 0 0 NMOS W=24u L=180n" in text
    assert "Mceerr_cc2_branch ceerr_cc2_mid score2p ccsrc 0 NREL W=24u L=180n" in text
    assert "Mceerr_dp2_yp0 ceerr_dp2_t ybar2 ceerr_dp2_yp 0 NSENSE W=32u L=180n" in text
    assert "Mceerr_dn0_score0 ceerr_dn0_inh score0p ceerr_dn0_score 0 NREL W=24u L=180n" in text
    assert ".meas tran ybar2_err FIND V(ybar2) AT=3.15n" in text
    assert ".meas tran ctsrc_err FIND V(ctsrc) AT=3.15n" in text
    assert ".meas tran cesrc_err FIND V(cesrc) AT=3.15n" in text
    assert "Merr0_" not in text


def test_spicenn_sparse_forward_train_netlist_threads_readout_sizing() -> None:
    text, _topology = sparse_forward.train_netlist(
        x0=1.0,
        x1=1.0,
        bias=0.6,
        label=0,
        hidden_count=4,
        output_count=2,
        hidden1_fan_in=2,
        hidden2_fan_in=3,
        readout_fan_in=3,
        seed=7,
        update_width_u=0.004,
        score_cap_f=6.0,
        readout_width_u=18.0,
    )

    assert "Cscore0p score0p 0 6f IC=0" in text
    assert "Mro0_" in text
    assert "W=18u L=180n" in text


def test_spicenn_sparse_forward_train_netlist_can_use_signed_hidden_features() -> None:
    text, _topology = sparse_forward.train_netlist(
        x0=1.0,
        x1=0.0,
        bias=0.6,
        label=0,
        hidden_count=6,
        output_count=2,
        hidden1_fan_in=1,
        hidden2_fan_in=3,
        readout_fan_in=3,
        seed=1,
        update_width_u=0.004,
        update_write_mode="simple_charge_discharge",
        hidden_weight_mode="signed_hidden1",
    )

    assert "Cw10_x0p w10_x0p 0 4f IC=0.04" in text
    assert "Cw10_x0n w10_x0n 0 4f IC=0.88" in text
    assert "Cw12_x1p w12_x1p 0 4f IC=0.88" in text
    assert "Cw12_x1n w12_x1n 0 4f IC=0.04" in text


def test_spicenn_sparse_forward_feature_probe_summarizes_hidden_separation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_measures = iter(
        [
            {
                "h1_0": 0.1,
                "h1_1": 0.2,
                "h2_0": 0.1,
                "h2_1": 0.2,
                "score0_p": 0.5,
                "score0_n": 0.4,
                "score1_p": 0.4,
                "score1_n": 0.5,
            },
            {
                "h1_0": 0.1,
                "h1_1": 0.3,
                "h2_0": 0.4,
                "h2_1": 0.2,
                "score0_p": 0.4,
                "score0_n": 0.5,
                "score1_p": 0.5,
                "score1_n": 0.4,
            },
        ]
    )

    def fake_run_netlist(_spice_bin: str, _path: Path, text: str, _timeout: float) -> dict[str, float]:
        assert ".meas tran h2_1 FIND V(h2_1) AT=1.8n" in text
        return next(fake_measures)

    monkeypatch.setattr(sparse_forward, "run_netlist", fake_run_netlist)
    result = sparse_forward.run_feature_separation_probe(
        spice_bin="fake-spice",
        generated_dir=tmp_path,
        tag="feature",
        samples=sparse_forward.named_sample_stream("x0_identity", epochs=1),
        bias_values=[0.25],
        hidden_count=2,
        output_count=2,
        hidden1_fan_in=1,
        hidden2_fan_in=1,
        readout_fan_in=1,
        seed=1,
        center_v=0.30,
        hidden_weight_mode="signed_hidden1",
        score_cap_f=10.0,
        readout_width_u=10.0,
        readout_branch_style="pass_act_source",
        measure_time_ns=1.8,
        timeout=1.0,
    )

    row = result["rows"][0]
    assert row["bias"] == pytest.approx(0.25)
    assert row["min_inter_label_h2_l1"] == pytest.approx(0.3)
    assert row["min_inter_label_h2_l2"] == pytest.approx(0.3)
    assert row["min_h2_active_count"] == 2
    assert result["hidden_weight_mode"] == "signed_hidden1"


def test_spicenn_sparse_forward_cmos_train_step_transient_has_low_common_mode(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    text, topology = sparse_forward.train_netlist(
        x0=1.0,
        x1=1.0,
        bias=0.6,
        label=0,
        hidden_count=4,
        output_count=1,
        hidden1_fan_in=2,
        hidden2_fan_in=3,
        readout_fan_in=3,
        seed=7,
        update_width_u=0.0005,
        charge_width_u=0.0005,
        discharge_width_u=0.000005,
        update_write_mode="cmos_complementary_charge_discharge",
        spike_ref_v=0.1,
    )
    measures = sparse_forward.add_train_derived_measures(
        sparse_forward.run_netlist(ngspice_path, tmp_path / "spicenn_sparse_cmos_train_step.cir", text, timeout=20),
        topology,
    )

    assert measures["h2_0"] > 0.2
    assert measures["dp0_err"] > measures["dn0_err"]
    assert measures["row0_signed_delta"] > 0.003
    assert abs(measures["row0_common_delta"]) < abs(measures["row0_signed_delta"])


def test_spicenn_sparse_forward_repeated_readout_training_carries_cap_states(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    result = sparse_forward.run_repeated_readout_training(
        spice_bin=ngspice_path,
        generated_dir=tmp_path,
        tag="repeat",
        samples=[
            {"x0": 1.0, "x1": 1.0, "label": 0},
            {"x0": 1.0, "x1": 1.0, "label": 1},
        ],
        hidden_count=4,
        output_count=2,
        hidden1_fan_in=2,
        hidden2_fan_in=3,
        readout_fan_in=3,
        seed=7,
        bias=0.6,
        update_width_u=0.0005,
        charge_width_u=0.0005,
        discharge_width_u=0.000005,
        update_write_mode="cmos_complementary_charge_discharge",
        spike_ref_v=0.1,
        readout_center_v=0.30,
        false_negative_target_v=0.0,
        hidden_weight_mode="positive",
        timeout=20,
    )

    assert len(result["steps"]) == 2
    assert result["steps"][0]["label_row_signed_delta"] > 0.003
    assert result["steps"][1]["label_row_signed_delta"] > 0.003
    assert len(result["steps"][0]["row_signed_deltas"]) == 2
    assert len(result["steps"][0]["error_diffs"]) == 2
    assert "final_evaluation" in result
    assert result["final_weight_summary"]["row0_signed_sum"] > result["initial_weight_summary"]["row0_signed_sum"]
    assert result["final_weight_summary"]["row1_signed_sum"] > result["initial_weight_summary"]["row1_signed_sum"]


def test_spicenn_sparse_forward_repeated_training_carries_hidden2_cap_states(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    topology = sparse_forward.build_topology(
        hidden_count=2,
        output_count=1,
        hidden1_fan_in=1,
        hidden2_fan_in=1,
        readout_fan_in=1,
        seed=3,
    )
    first_sink = 0
    first_source = topology.hidden2.as_fanins()[first_sink][0]
    rendered_texts: list[str] = []

    def fake_train_measures(step: int) -> dict[str, float]:
        hidden_before = 0.40 + 0.01 * step
        hidden_after = 0.41 + 0.01 * step
        measures: dict[str, float] = {
            "score0_p": 0.6,
            "score0_n": 0.3,
            "dp0_err": 0.4,
            "dn0_err": 0.1,
        }
        for out, sources in topology.readout.as_fanins().items():
            for source in sources:
                measures[f"vw{out}_{source}p_before"] = 0.30 + 0.01 * step
                measures[f"vw{out}_{source}n_before"] = 0.30 - 0.01 * step
                measures[f"vw{out}_{source}p_after"] = 0.31 + 0.01 * step
                measures[f"vw{out}_{source}n_after"] = 0.29 - 0.01 * step
        for sink, sources in topology.hidden2.as_fanins().items():
            for source in sources:
                measures[f"w2{sink}_{source}p_before"] = hidden_before
                measures[f"w2{sink}_{source}n_before"] = 0.40 - 0.01 * step
                measures[f"w2{sink}_{source}p_after"] = hidden_after
                measures[f"w2{sink}_{source}n_after"] = 0.39 - 0.01 * step
        return measures

    def fake_run_netlist(_spice_bin: str, _path: Path, text: str, _timeout: float) -> dict[str, float]:
        step = len(rendered_texts)
        rendered_texts.append(text)
        if step == 1:
            assert f"Cw2{first_sink}_{first_source}p w2{first_sink}_{first_source}p 0 4f IC=0.41" in text
            assert f"Cw2{first_sink}_{first_source}n w2{first_sink}_{first_source}n 0 4f IC=0.39" in text
        return fake_train_measures(step)

    monkeypatch.setattr(sparse_forward, "run_netlist", fake_run_netlist)
    result = sparse_forward.run_repeated_readout_training(
        spice_bin="fake-spice",
        generated_dir=tmp_path,
        tag="repeat_hidden",
        samples=[
            {"x0": 1.0, "x1": 0.0, "label": 0},
            {"x0": 0.0, "x1": 1.0, "label": 0},
        ],
        hidden_count=2,
        output_count=1,
        hidden1_fan_in=1,
        hidden2_fan_in=1,
        readout_fan_in=1,
        seed=3,
        bias=0.6,
        update_width_u=0.0005,
        charge_width_u=0.0005,
        discharge_width_u=0.000005,
        update_write_mode="cmos_complementary_charge_discharge",
        spike_ref_v=0.1,
        readout_center_v=0.30,
        false_negative_target_v=0.0,
        hidden_weight_mode="centered_hidden2",
        timeout=1.0,
        evaluation_samples=[],
        hidden2_update_width_u=0.0002,
        hidden2_delta_mode="raw",
        hidden2_update_write_mode="senseamp_cmos_complementary_charge_discharge",
        hidden2_update_selector_width_u=64.0,
    )

    assert len(rendered_texts) == 2
    assert result["steps"][0]["hidden2_row_signed_deltas"][first_sink] == pytest.approx(0.02)
    assert result["steps"][1]["hidden2_weights_after"][f"{first_sink}_{first_source}"] == pytest.approx([0.42, 0.38])
    assert result["final_hidden2_weight_summary"]["hidden2_total_signed_sum"] > result[
        "initial_hidden2_weight_summary"
    ]["hidden2_total_signed_sum"]
