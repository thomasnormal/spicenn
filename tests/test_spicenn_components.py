from __future__ import annotations

import re
import subprocess

import pytest

from spicenn import (
    CapState,
    CapStateArray,
    CapStateProgram,
    Component,
    DiffPairBleedWriteSelector,
    DifferentialCapState,
    DifferentialCapStateArray,
    DifferentialReLUNeuron,
    DifferentialSignalGate,
    DifferentialToDifferentialSynapse,
    DirectFlowWeightCell,
    FanInTopology,
    Layer,
    NetlistBuilder,
    MutuallyInhibitedWriteSelector,
    Neuron,
    NonnegativeToDifferentialSynapse,
    NodeParasitics,
    NonnegativeSignal,
    Port,
    PreTraceArray,
    PreTraceCell,
    ReadoutBranch,
    RegenerativeDifferentialWriteSelector,
    ReLUNeuron,
    SplitScoreCELimitedErrorBank,
    SignedScoreErrorCell,
    SignedSignal,
    SignedMosSynapse,
    Synapse,
    expected_readout_cap_names,
    load_readout_cap_state_csv,
    make_sparse_differential_relu_layer,
    make_sparse_differential_error_transport_layer,
    make_sparse_hidden_update_layer,
    make_sparse_readout_update_layer,
    make_sparse_relu_delta_gate_layer,
    make_sparse_signed_readout_layer,
    require_nonnegative_signal,
    require_signed_signal,
)


MEAS_RE = re.compile(r"(?im)^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([-+0-9.eE]+)")


def render(component: Component) -> str:
    deck = NetlistBuilder()
    deck.render_component(component)
    return deck.render_body()


def parse_measures(text: str) -> dict[str, float]:
    return {name.lower(): float(value) for name, value in MEAS_RE.findall(text)}


def test_signal_types_separate_relu_wires_from_signed_rails() -> None:
    activation = NonnegativeSignal.at("act0")
    score_error = SignedSignal.from_base("dp0", positive_suffix="p", negative_suffix="n")

    assert activation.nodes() == ("act0",)
    assert str(activation) == "act0"
    assert activation.as_signed_against("0").nodes() == ("act0", "0")
    assert score_error.nodes() == ("dp0p", "dp0n")
    assert score_error.differential_expr() == "dp0p-dp0n"
    assert require_nonnegative_signal(activation, context="hidden activation") is activation
    assert require_signed_signal(score_error, context="output error") is score_error

    with pytest.raises(ValueError, match="hidden activation must be a nonnegative"):
        require_nonnegative_signal(score_error, context="hidden activation")
    with pytest.raises(ValueError, match="output error must be signed"):
        require_signed_signal(activation, context="output error")


def test_component_hierarchy_matches_physical_blocks() -> None:
    weight_p = CapState("w0p", "w0p", 4.0)
    weight_n = CapState("w0n", "w0n", 4.0)
    synapse = SignedMosSynapse("s0", "pre0", "post0", weight_p, weight_n)
    neuron = ReLUNeuron("h0", CapState("pre0", "pre0", 2.0), "act0")
    topology = FanInTopology.random_fanin(("pre0", "pre1"), 1, seed=4, fan_in=1)
    layer = Layer(
        "hidden",
        _children=[synapse, neuron],
        inputs=(Port.at("in", "pre0"),),
        outputs=(Port.at("out", "act0"),),
        topology=topology,
    )

    assert isinstance(weight_p, Component)
    assert isinstance(synapse, Synapse)
    assert isinstance(neuron, Neuron)
    assert isinstance(layer, Component)
    assert synapse.input_nodes() == ("pre0",)
    assert synapse.output_nodes() == ("post0",)
    assert synapse.state_nodes() == ("w0p", "w0n")
    assert neuron.input_nodes() == ("pre0",)
    assert neuron.output_nodes() == ("act0",)
    assert neuron.state_nodes() == ("pre0",)
    assert layer.input_nodes() == ("pre0",)
    assert layer.output_nodes() == ("act0",)
    assert layer.fanins() == topology.as_fanins()
    assert layer.fanouts() == topology.fanouts()

    text = render(layer)
    assert "Cw0p w0p 0 4f IC=0" in text
    assert "Cw0n w0n 0 4f IC=0" in text
    assert "Mrelu_h0 vdd pre0 act0 0 NREL W=24u L=180n" in text
    assert layer.render_lines() == text.splitlines()


def test_cap_state_renders_resettable_leaky_physical_state() -> None:
    state = CapState(
        name="fprbar00",
        node="fprbar00",
        cap_f=2.0,
        ic_v=1.2,
        leak_to="vdd",
        leak_ohm="1G",
        reset_gate="rstf",
        reset_to="vdd",
        reset_model="NSENSE",
    )

    text = render(state)

    assert "Cfprbar00 fprbar00 0 2f IC=1.2" in text
    assert "Rfprbar00 fprbar00 vdd 1G" in text
    assert "Mreset_fprbar00 vdd rstf fprbar00 0 NSENSE W=4u L=180n" in text


def test_cap_state_array_renders_activation_and_error_caps() -> None:
    states = CapStateArray.from_nodes(
        "hidden_dynamic",
        ("pre0", "act0", "hdp0"),
        cap_f=10.0,
        ic_v=0.0,
        leak_to="0",
        leak_ohm="1G",
    )

    text = render(states)

    assert states.state_nodes() == ("pre0", "act0", "hdp0")
    assert "Cpre0 pre0 0 10f IC=0" in text
    assert "Rpre0 pre0 0 1G" in text
    assert "Cact0 act0 0 10f IC=0" in text
    assert "Chdp0 hdp0 0 10f IC=0" in text


def test_cap_state_program_renders_sorted_programmed_physical_state() -> None:
    program = CapStateProgram("readout_caps", {"vw10n": 0.62, "vw10p": 0.70}, 4.0)

    text = render(program)

    assert program.state_nodes() == ("vw10n", "vw10p")
    assert "Cvw10n vw10n 0 4f IC=0.62" in text
    assert "Rvw10n vw10n 0 1e15" in text
    assert "Cvw10p vw10p 0 4f IC=0.7" in text


def test_differential_cap_state_array_renders_signed_weight_caps() -> None:
    states = DifferentialCapStateArray(
        name="readout",
        bases=("vw00", "vbo0"),
        initial={
            "vw00p": 0.61,
            "vw00n": 0.52,
            "vbo0p": 0.66,
            "vbo0n": 0.54,
        },
        cap_f=4.0,
    )

    text = render(states)

    assert states.state_nodes() == ("vw00p", "vw00n", "vbo0p", "vbo0n")
    assert "Cvw00p vw00p 0 4f IC=0.61" in text
    assert "Cvw00n vw00n 0 4f IC=0.52" in text
    assert "Rvw00p vw00p 0 1e15" in text
    assert "Cvbo0p vbo0p 0 4f IC=0.66" in text
    assert "Cvbo0n vbo0n 0 4f IC=0.54" in text


def test_differential_relu_hidden_cell_renders_signed_sum_to_nonnegative_activation() -> None:
    weight = DifferentialCapState.from_base(
        "wh0_x0",
        cap_f=4.0,
        pos_ic_v=0.72,
        neg_ic_v=0.48,
    )
    preactivation = DifferentialCapState.from_base(
        "uh0",
        cap_f=10.0,
        pos_ic_v=0.0,
        neg_ic_v=0.0,
        leak_to="0",
        leak_ohm="1G",
    )
    activation = CapState("acth0", "acth0", 8.0, leak_to="0", leak_ohm="1G")
    synapse = NonnegativeToDifferentialSynapse(
        "sh0_x0",
        activation_node="x0",
        pos_weight_node=weight.pos_node,
        neg_weight_node=weight.neg_node,
        post_pos_node=preactivation.pos_node,
        post_neg_node=preactivation.neg_node,
        width_u=3.0,
    )
    neuron = DifferentialReLUNeuron("h0", preactivation=preactivation, activation=activation, width_u=12.0)
    layer = Layer(
        "hidden_signed",
        _children=[weight, synapse, neuron],
        inputs=(Port.at("in", "x0"),),
        outputs=(Port.at("out", "acth0"),),
    )

    text = render(layer)

    assert isinstance(synapse, Synapse)
    assert isinstance(neuron, Neuron)
    assert synapse.input_nodes() == ("x0", "wh0_x0p", "wh0_x0n", "fwd")
    assert synapse.output_nodes() == ("uh0p", "uh0n")
    assert neuron.input_nodes() == ("uh0p", "uh0n")
    assert neuron.output_nodes() == ("acth0",)
    assert "Cwh0_x0p wh0_x0p 0 4f IC=0.72" in text
    assert "Cwh0_x0n wh0_x0n 0 4f IC=0.48" in text
    assert "Msh0_x0_pos_w x0 wh0_x0p sh0_x0_pos_mid 0 NREL W=3u L=180n" in text
    assert "Msh0_x0_pos_f sh0_x0_pos_mid fwd uh0p 0 NREL W=3u L=180n" in text
    assert "Msh0_x0_neg_w x0 wh0_x0n sh0_x0_neg_mid 0 NREL W=3u L=180n" in text
    assert "Msh0_x0_neg_f sh0_x0_neg_mid fwd uh0n 0 NREL W=3u L=180n" in text
    assert "Mrelu_h0_inhibit vdd uh0n h0_relu_pos_low vdd PMOS W=12u L=180n" in text
    assert "Mrelu_h0_sense h0_relu_pos_low uh0p h0_relu_pos_mid 0 NSENSE W=12u L=180n" in text
    assert "Mrelu_h0_fwd h0_relu_pos_mid fwd acth0 0 NREL W=12u L=180n" in text
    assert "Mrelu_h0_neg_bleed acth0 uh0n 0 0 NREL W=12u L=180n" in text


def test_differential_to_differential_synapse_routes_backward_sign_products() -> None:
    synapse = DifferentialToDifferentialSynapse(
        "bt0_3",
        source_pos_node="dp0",
        source_neg_node="dn0",
        pos_weight_node="vw0_3p",
        neg_weight_node="vw0_3n",
        post_pos_node="hd3p",
        post_neg_node="hd3n",
        width_u=5.0,
    )

    text = render(synapse)

    assert synapse.input_nodes() == ("dp0", "dn0", "vw0_3p", "vw0_3n", "bwd")
    assert synapse.output_nodes() == ("hd3p", "hd3n")
    assert synapse.state_nodes() == (
        "bt0_3_wp_ep_mid",
        "bt0_3_wp_en_mid",
        "bt0_3_wn_ep_mid",
        "bt0_3_wn_en_mid",
    )
    assert "Mbt0_3_wp_ep_w dp0 vw0_3p bt0_3_wp_ep_mid 0 NREL W=5u L=180n" in text
    assert "Mbt0_3_wp_ep_b bt0_3_wp_ep_mid bwd hd3p 0 NREL W=5u L=180n" in text
    assert "Mbt0_3_wp_en_w dn0 vw0_3p bt0_3_wp_en_mid 0 NREL W=5u L=180n" in text
    assert "Mbt0_3_wp_en_b bt0_3_wp_en_mid bwd hd3n 0 NREL W=5u L=180n" in text
    assert "Mbt0_3_wn_ep_w dp0 vw0_3n bt0_3_wn_ep_mid 0 NREL W=5u L=180n" in text
    assert "Mbt0_3_wn_ep_b bt0_3_wn_ep_mid bwd hd3n 0 NREL W=5u L=180n" in text
    assert "Mbt0_3_wn_en_w dn0 vw0_3n bt0_3_wn_en_mid 0 NREL W=5u L=180n" in text
    assert "Mbt0_3_wn_en_b bt0_3_wn_en_mid bwd hd3p 0 NREL W=5u L=180n" in text


def test_differential_signal_gate_renders_relu_masked_delta_caps() -> None:
    gate = DifferentialSignalGate(
        "gd2",
        positive_input_node="hd2p",
        negative_input_node="hd2n",
        gate_node="h2_2",
        positive_output=CapState("gdh2p", "gdh2p", 6.0, leak_to="0", leak_ohm="1G"),
        negative_output=CapState("gdh2n", "gdh2n", 6.0, leak_to="0", leak_ohm="1G"),
        width_u=7.0,
    )

    text = render(gate)

    assert gate.input_nodes() == ("hd2p", "hd2n", "h2_2", "bwd")
    assert gate.output_nodes() == ("gdh2p", "gdh2n")
    assert "Cgdh2p gdh2p 0 6f IC=0" in text
    assert "Mgd2_p_in vdd hd2p gd2_p_in 0 NSENSE W=7u L=180n" in text
    assert "Mgd2_p_gate gd2_p_in h2_2 gd2_p_gate 0 NREL W=7u L=180n" in text
    assert "Mgd2_p_bwd gd2_p_gate bwd gdh2p 0 NREL W=7u L=180n" in text
    assert "Mgd2_n_in vdd hd2n gd2_n_in 0 NSENSE W=7u L=180n" in text
    assert "Mgd2_n_bwd gd2_n_gate bwd gdh2n 0 NREL W=7u L=180n" in text


def test_sparse_differential_relu_layer_builder_uses_explicit_signed_preactivations() -> None:
    topology = FanInTopology.from_fanins(
        ("bias", "x0", "x1"),
        2,
        {
            0: ("bias", "x0"),
            1: ("bias", "x1"),
        },
    )

    layer = make_sparse_differential_relu_layer(
        "hidden",
        topology=topology,
        source_nodes={"bias": "xbias", "x0": "x0", "x1": "x1"},
        activation_prefix="h",
        preactivation_prefix="u",
        weight_prefix="wh",
        synapse_prefix="sh",
        weight_pos_ic_v=0.62,
        weight_neg_ic_v=0.58,
        weight_initials={(1, "x1"): (0.81, 0.22)},
    )

    text = render(layer)

    assert layer.input_nodes() == ("xbias", "x0", "x1")
    assert layer.output_nodes() == ("h0", "h1")
    assert layer.fanins() == {0: ("bias", "x0"), 1: ("bias", "x1")}
    assert layer.fanouts() == {"bias": (0, 1), "x0": (0,), "x1": (1,)}
    assert "Cwh0_biasp wh0_biasp 0 4f IC=0.62" in text
    assert "Cwh0_biasn wh0_biasn 0 4f IC=0.58" in text
    assert "Cwh1_x1p wh1_x1p 0 4f IC=0.81" in text
    assert "Cwh1_x1n wh1_x1n 0 4f IC=0.22" in text
    assert "Msh0_bias_pos_f sh0_bias_pos_mid fwd u0p 0 NREL W=3u L=180n" in text
    assert "Msh0_x0_neg_f sh0_x0_neg_mid fwd u0n 0 NREL W=3u L=180n" in text
    assert "Msh1_x1_pos_f sh1_x1_pos_mid fwd u1p 0 NREL W=3u L=180n" in text
    assert "Mrelu_hidden_0_sense hidden_0_relu_pos_low u0p hidden_0_relu_pos_mid 0 NSENSE W=12u L=180n" in text
    assert "Mrelu_hidden_0_neg_bleed h0 u0n 0 0 NREL W=12u L=180n" in text
    assert "Mrelu_hidden_1_inhibit vdd u1n hidden_1_relu_pos_low vdd PMOS W=12u L=180n" in text


def test_sparse_signed_readout_layer_builder_uses_explicit_score_rails() -> None:
    topology = FanInTopology.from_fanins(
        (0, 1, 2),
        2,
        {
            0: (0, 2),
            1: (1,),
        },
    )

    layer = make_sparse_signed_readout_layer(
        "readout",
        topology=topology,
        source_nodes={0: "h0", 1: "h1", 2: "h2"},
        score_prefix="score",
        weight_prefix="vw",
        branch_prefix="rb",
        weight_pos_ic_v=0.70,
        weight_neg_ic_v=0.52,
        weight_initials={(1, 1): (0.25, 0.85)},
        branch_width_u=9.0,
    )

    text = render(layer)

    assert layer.input_nodes() == ("h0", "h1", "h2")
    assert layer.output_nodes() == ("score0p", "score0n", "score1p", "score1n")
    assert layer.fanins() == {0: (0, 2), 1: (1,)}
    assert "Cscore0p score0p 0 10f IC=0" in text
    assert "Cscore0n score0n 0 10f IC=0" in text
    assert "Cvw0_0p vw0_0p 0 4f IC=0.7" in text
    assert "Cvw0_0n vw0_0n 0 4f IC=0.52" in text
    assert "Cvw1_1p vw1_1p 0 4f IC=0.25" in text
    assert "Cvw1_1n vw1_1n 0 4f IC=0.85" in text
    assert "Mrb0_0_p_a vdd h0 rb0_0_p_0 0 NSENSE W=9u L=180n" in text
    assert "Mrb0_0_p_w rb0_0_p_0 vw0_0p rb0_0_p_1 0 NREL W=9u L=180n" in text
    assert "Mrb0_0_p_f rb0_0_p_1 fwd score0p 0 NREL W=9u L=180n" in text
    assert "Mrb0_0_n_a vdd h0 rb0_0_n_0 0 NSENSE W=9u L=180n" in text
    assert "Mrb0_0_n_w rb0_0_n_0 vw0_0n rb0_0_n_1 0 NREL W=9u L=180n" in text
    assert "Mrb0_0_n_f rb0_0_n_1 fwd score0n 0 NREL W=9u L=180n" in text
    assert "Cvw0_1p" not in text
    assert "Mrb0_1_p" not in text


def test_sparse_error_transport_layer_reuses_readout_weights_for_signed_deltas() -> None:
    topology = FanInTopology.from_fanins(
        ("bias", 0, 1, 2),
        2,
        {
            0: ("bias", 0, 2),
            1: ("bias", 1),
        },
    )

    layer = make_sparse_differential_error_transport_layer(
        "backprop_readout",
        topology=topology,
        weight_prefix="vw",
        delta_prefix="hd",
        transport_prefix="bt",
        transport_width_u=6.0,
    )

    text = render(layer)

    assert layer.input_nodes() == ("dp0", "dn0", "dp1", "dn1")
    assert layer.output_nodes() == ("hd0p", "hd0n", "hd1p", "hd1n", "hd2p", "hd2n")
    assert "Chdbiasp" not in text
    assert "Cbias" not in text
    assert "Chd0p hd0p 0 10f IC=0" in text
    assert "Chd2n hd2n 0 10f IC=0" in text
    assert "Mbt0_0_wp_ep_e vdd dp0 bt0_0_wp_ep_e_mid 0 NREL W=6u L=180n" in text
    assert "Mbt0_0_wp_ep_w bt0_0_wp_ep_e_mid vw0_0p bt0_0_wp_ep_w_mid 0 NREL W=6u L=180n" in text
    assert "Mbt0_0_wn_ep_b bt0_0_wn_ep_w_mid bwd hd0n 0 NREL W=6u L=180n" in text
    assert "Mbt0_2_wn_en_b bt0_2_wn_en_w_mid bwd hd2p 0 NREL W=6u L=180n" in text
    assert "Mbt1_1_wp_en_b bt1_1_wp_en_w_mid bwd hd1n 0 NREL W=6u L=180n" in text
    assert "Mbt0_bias" not in text


def test_sparse_relu_delta_gate_layer_builds_one_gated_delta_pair_per_neuron() -> None:
    layer = make_sparse_relu_delta_gate_layer(
        "relu_delta",
        sink_count=2,
        activation_nodes={0: "h2_0", 1: "h2_1"},
        input_delta_prefix="hd",
        output_delta_prefix="gdh",
        gate_width_u=9.0,
    )

    text = render(layer)

    assert layer.input_nodes() == ("hd0p", "hd0n", "h2_0", "hd1p", "hd1n", "h2_1")
    assert layer.output_nodes() == ("gdh0p", "gdh0n", "gdh1p", "gdh1n")
    assert "Cgdh0p gdh0p 0 6f IC=0" in text
    assert "Mrelu_delta_0_p_in vdd hd0p relu_delta_0_p_in 0 NSENSE W=9u L=180n" in text
    assert "Mrelu_delta_0_p_gate relu_delta_0_p_in h2_0 relu_delta_0_p_gate 0 NREL W=9u L=180n" in text
    assert "Mrelu_delta_1_n_bwd relu_delta_1_n_gate bwd gdh1n 0 NREL W=9u L=180n" in text


def test_sparse_hidden_update_layer_writes_hidden_synapse_weights_from_signed_deltas() -> None:
    topology = FanInTopology.from_fanins(
        ("bias", 0, 1),
        2,
        {
            0: ("bias", 0),
            1: (1,),
        },
    )
    layer = make_sparse_hidden_update_layer(
        "hidden_update",
        topology=topology,
        source_nodes={"bias": "bias", 0: "h1_0", 1: "h1_1"},
        weight_prefix="w2",
        update_prefix="uh2",
        delta_prefix="gdh",
        update_width_u=0.003,
    )

    text = render(layer)

    assert layer.input_nodes() == ("bias", "h1_0", "h1_1", "gdh0p", "gdh0n", "gdh1p", "gdh1n")
    assert "Mw20_biasn_flow_d w20_biasn_flow_a gdh0p wlow 0 NSENSE W=0.003u L=180n" in text
    assert "Mw20_biasp_ch_d w20_biasp_ch_a gdh0p w20_biasp 0 NSENSE W=0.003u L=180n" in text
    assert "Mw20_0p_flow_d w20_0p_flow_a gdh0n wlow 0 NSENSE W=0.003u L=180n" in text
    assert "Mw21_1n_ch_d w21_1n_ch_a gdh1n w21_1n 0 NSENSE W=0.003u L=180n" in text
    assert "Mw20_1p" not in text


def test_sparse_hidden_update_layer_can_select_delta_polarity_before_writing() -> None:
    topology = FanInTopology.from_fanins((0, 1), 1, {0: (0,)})
    layer = make_sparse_hidden_update_layer(
        "hidden_update",
        topology=topology,
        source_nodes={0: "h1_0", 1: "h1_1"},
        weight_prefix="w2",
        update_prefix="uh2",
        delta_prefix="gdh",
        update_width_u=0.003,
        write_mode="diffpair_charge_discharge",
    )

    text = render(layer)

    assert "Mhwsel0_pos_sel hwsel0_posbar gdh0p hwsel0_src 0 NSENSE W=8u L=180n" in text
    assert "Mhwsel0_neg_sel hwsel0_negbar gdh0n hwsel0_src 0 NSENSE W=8u L=180n" in text
    assert "Mw20_0n_flow_d w20_0n_flow_a hwpos0 wlow 0 NSENSE W=0.003u L=180n" in text
    assert "Mw20_0p_flow_d w20_0p_flow_a hwneg0 wlow 0 NSENSE W=0.003u L=180n" in text


def test_signed_score_error_cell_renders_positive_target_plus_negative_score_minus_positive_score() -> None:
    cell = SignedScoreErrorCell(
        "err0",
        target_node="t0",
        score_pos_node="score0p",
        score_neg_node="score0n",
        positive_error=CapState("dp0", "dp0", 6.0, leak_to="0", leak_ohm="1G"),
        negative_error=CapState("dn0", "dn0", 6.0, leak_to="0", leak_ohm="1G"),
        target_width_u=40.0,
        score_width_u=24.0,
    )

    text = render(cell)

    assert cell.input_nodes() == ("t0", "score0p", "score0n", "err")
    assert cell.output_nodes() == ("dp0", "dn0")
    assert "Cdp0 dp0 0 6f IC=0" in text
    assert "Cdn0 dn0 0 6f IC=0" in text
    assert "Merr0_dp_t0 vdd t0 err0_dp_t 0 NSENSE W=40u L=180n" in text
    assert "Merr0_dp_t1 err0_dp_t err dp0 0 NSENSE W=40u L=180n" in text
    assert "Merr0_dp_sn0 vdd score0n err0_dp_sn 0 NSENSE W=24u L=180n" in text
    assert "Merr0_dp_sn1 err0_dp_sn err dp0 0 NSENSE W=24u L=180n" in text
    assert "Merr0_dn_sp0 vdd score0p err0_dn_sp 0 NSENSE W=24u L=180n" in text
    assert "Merr0_dn_sp1 err0_dn_sp err dn0 0 NSENSE W=24u L=180n" in text
    assert "Merr0_dn_tn0" not in text


def test_signed_score_error_cell_can_render_negative_target_rail() -> None:
    cell = SignedScoreErrorCell(
        "err1",
        target_node="tp1",
        negative_target_node="tn1",
        score_pos_node="score1p",
        score_neg_node="score1n",
        positive_error=CapState("dp1", "dp1", 6.0, leak_to="0", leak_ohm="1G"),
        negative_error=CapState("dn1", "dn1", 6.0, leak_to="0", leak_ohm="1G"),
        target_width_u=40.0,
        negative_target_width_u=36.0,
        score_width_u=24.0,
    )

    text = render(cell)

    assert cell.input_nodes() == ("tp1", "tn1", "score1p", "score1n", "err")
    assert "Merr1_dp_t0 vdd tp1 err1_dp_t 0 NSENSE W=40u L=180n" in text
    assert "Merr1_dn_tn0 vdd tn1 err1_dn_tn 0 NSENSE W=36u L=180n" in text
    assert "Merr1_dn_tn1 err1_dn_tn err dn1 0 NSENSE W=36u L=180n" in text
    assert "Rpar_err1_dn_tn err1_dn_tn 0 1e9" in text


def test_signed_score_error_cell_zero_widths_render_quiet_error_caps_only() -> None:
    cell = SignedScoreErrorCell(
        "errq",
        target_node="tq",
        negative_target_node="tnq",
        score_pos_node="scoreqp",
        score_neg_node="scoreqn",
        positive_error=CapState("dpq", "dpq", 6.0, leak_to="0", leak_ohm="1G"),
        negative_error=CapState("dnq", "dnq", 6.0, leak_to="0", leak_ohm="1G"),
        target_width_u=0.0,
        negative_target_width_u=0.0,
        score_width_u=0.0,
    )

    text = render(cell)

    assert "Cdpq dpq 0 6f IC=0" in text
    assert "Cdnq dnq 0 6f IC=0" in text
    assert "Merrq_" not in text
    assert "Rpar_errq_" not in text


def test_split_score_ce_limited_error_bank_renders_shared_competition() -> None:
    bank = SplitScoreCELimitedErrorBank(
        "ceerr",
        output_count=3,
        error_cap_f=7.0,
        target_width_u=40.0,
        nontarget_width_u=25.0,
    )

    text = render(bank)

    assert bank.input_nodes() == (
        "err",
        "rste",
        "vdd",
        "vdd",
        "t0",
        "nt0",
        "score0p",
        "score0n",
        "t1",
        "nt1",
        "score1p",
        "score1n",
        "t2",
        "nt2",
        "score2p",
        "score2n",
    )
    assert bank.output_nodes() == ("dp0", "dn0", "dp1", "dn1", "dp2", "dn2")
    assert "Cdp0 dp0 0 7f IC=0" in text
    assert "Cybar0 ybar0 0 20f IC=1.2" in text
    assert "Mceerr_ctsrc vdd err ctsrc 0 NSENSE W=40u L=180n" in text
    assert "Mceerr_cesrc vdd err cesrc 0 NSENSE W=25u L=180n" in text
    assert "Mceerr_cc_tail ccsrc err 0 0 NMOS W=25u L=180n" in text
    assert "Mceerr_cc1_inh ceerr_cc1_mid score1n ybar1 vdd PMOS W=25u L=180n" in text
    assert "Mceerr_cc1_branch ceerr_cc1_mid score1p ccsrc 0 NREL W=25u L=180n" in text
    assert "Mceerr_dp1_t0 ctsrc t1 ceerr_dp1_t 0 NSENSE W=40u L=180n" in text
    assert "Mceerr_dp1_yp0 ceerr_dp1_t ybar1 ceerr_dp1_yp 0 NSENSE W=40u L=180n" in text
    assert "Mceerr_dn1_nt0 cesrc nt1 ceerr_dn1_nt 0 NSENSE W=25u L=180n" in text
    assert "Mceerr_dn1_score0 ceerr_dn1_inh score1p ceerr_dn1_score 0 NREL W=25u L=180n" in text


def test_sparse_readout_update_layer_builds_local_writers_for_existing_edges_only() -> None:
    topology = FanInTopology.from_fanins(
        (0, 1, 2),
        2,
        {
            0: (0, 2),
            1: (1,),
        },
    )
    layer = make_sparse_readout_update_layer(
        "readout_update",
        topology=topology,
        source_nodes={0: "h0", 1: "h1", 2: "h2"},
        weight_prefix="vw",
        update_prefix="uw",
        update_width_u=0.003,
    )

    text = render(layer)

    assert layer.input_nodes() == ("h0", "h1", "h2", "dp0", "dn0", "dp1", "dn1")
    assert "Mvw0_0n_flow_d vw0_0n_flow_a dp0 wlow 0 NSENSE W=0.003u L=180n" in text
    assert "Mvw0_0p_flow_d vw0_0p_flow_a dn0 wlow 0 NSENSE W=0.003u L=180n" in text
    assert "Mvw0_0p_ch_d vw0_0p_ch_a dp0 vw0_0p 0 NSENSE W=0.003u L=180n" in text
    assert "Mvw0_0n_ch_d vw0_0n_ch_a dn0 vw0_0n 0 NSENSE W=0.003u L=180n" in text
    assert "Mvw0_1p_ch_d" not in text
    assert "Mvw0_1n_flow_d" not in text


def test_sparse_readout_update_layer_can_scale_specific_source_writer_widths() -> None:
    topology = FanInTopology.from_fanins(("bias", 0), 1, {0: ("bias", 0)})
    layer = make_sparse_readout_update_layer(
        "readout_update",
        topology=topology,
        source_nodes={"bias": "vdd", 0: "h0"},
        weight_prefix="vw",
        update_prefix="uw",
        update_width_u=0.004,
        source_update_scales={"bias": 0.25},
    )

    text = render(layer)

    assert "Mvw0_biasp_ch_d vw0_biasp_ch_a dp0 vw0_biasp 0 NSENSE W=0.001u L=180n" in text
    assert "Mvw0_biasn_flow_d vw0_biasn_flow_a dp0 wlow 0 NSENSE W=0.001u L=180n" in text
    assert "Mvw0_0p_ch_d vw0_0p_ch_a dp0 vw0_0p 0 NSENSE W=0.004u L=180n" in text
    assert "Mvw0_0n_flow_d vw0_0n_flow_a dp0 wlow 0 NSENSE W=0.004u L=180n" in text


def test_sparse_readout_update_layer_can_use_cmos_complementary_writer() -> None:
    topology = FanInTopology.from_fanins((0, 1), 1, {0: (0,)})
    layer = make_sparse_readout_update_layer(
        "readout_update",
        topology=topology,
        source_nodes={0: "h0", 1: "h1"},
        weight_prefix="vw",
        update_prefix="uw",
        update_width_u=0.004,
        write_mode="cmos_complementary_charge_discharge",
        selector_width_u=6.0,
    )

    text = render(layer)

    assert "Mrwsel0_pos_sel rwsel0_posbar dp0 rwsel0_src 0 NSENSE W=6u L=180n" in text
    assert "Cfprg0_0 fprg0_0 0 2f IC=0" in text
    assert "Cfprbar0_0 fprbar0_0 0 2f IC=1.2" in text
    assert "Mvw0_0n_flow_d vw0_0n_flow_a rwpos0 wlow 0 NSENSE W=0.004u L=180n" in text
    assert "Mvw0_0p_flow_d vw0_0p_flow_a rwneg0 wlow 0 NSENSE W=0.004u L=180n" in text
    assert "Mvw0_0p_cch_w vw0_0p_cch_w rwsel0_posbar whigh vdd PMOS W=0.004u L=180n" in text
    assert "Mvw0_0p_cch_a vw0_0p fprbar0_0 vw0_0p_cch_w vdd PMOS W=0.004u L=180n" in text
    assert "Mvw0_0n_cch_w vw0_0n_cch_w rwsel0_negbar whigh vdd PMOS W=0.004u L=180n" in text
    assert "Mvw0_0n_cch_a vw0_0n fprbar0_0 vw0_0n_cch_w vdd PMOS W=0.004u L=180n" in text
    assert "Mvw0_0p_cch_b" not in text
    assert "Mvw0_0p_ch_d" not in text


def test_sparse_hidden_update_layer_can_use_cmos_complementary_writer() -> None:
    topology = FanInTopology.from_fanins((0, 1), 1, {0: (0,)})
    layer = make_sparse_hidden_update_layer(
        "hidden_update",
        topology=topology,
        source_nodes={0: "h0", 1: "h1"},
        weight_prefix="wh",
        update_prefix="uh",
        update_width_u=0.004,
        write_mode="cmos_complementary_charge_discharge",
        selector_width_u=6.0,
    )

    text = render(layer)

    assert "Mhwsel0_pos_sel hwsel0_posbar hd0p hwsel0_src 0 NSENSE W=6u L=180n" in text
    assert "Cfhpg0_0 fhpg0_0 0 2f IC=0" in text
    assert "Cfhpbar0_0 fhpbar0_0 0 2f IC=1.2" in text
    assert "Mwh0_0n_flow_d wh0_0n_flow_a hwpos0 wlow 0 NSENSE W=0.004u L=180n" in text
    assert "Mwh0_0p_flow_d wh0_0p_flow_a hwneg0 wlow 0 NSENSE W=0.004u L=180n" in text
    assert "Mwh0_0p_cch_w wh0_0p_cch_w hwsel0_posbar whigh vdd PMOS W=0.004u L=180n" in text
    assert "Mwh0_0p_cch_a wh0_0p fhpbar0_0 wh0_0p_cch_w vdd PMOS W=0.004u L=180n" in text
    assert "Mwh0_0n_cch_w wh0_0n_cch_w hwsel0_negbar whigh vdd PMOS W=0.004u L=180n" in text
    assert "Mwh0_0n_cch_a wh0_0n fhpbar0_0 wh0_0n_cch_w vdd PMOS W=0.004u L=180n" in text
    assert "Mwh0_0p_cch_b" not in text
    assert "Mwh0_0p_ch_d" not in text


def test_regenerative_differential_write_selector_renders_dynamic_sense_amp() -> None:
    selector = RegenerativeDifferentialWriteSelector(
        "hsense0",
        positive_error_gate="hd0p",
        negative_error_gate="hd0n",
        positive_write_gate="hwpos0",
        negative_write_gate="hwneg0",
        width_u=12.0,
        label="hidden delta 0",
        nkeeper_width_u=0.5,
    )

    text = render(selector)

    assert "Chsense0_pos hwpos0 0 0.1f IC=1.2" in text
    assert "Mhsense0_pos_dis_s hwpos0 hd0n hsense0_posdn 0 NSENSE W=12u L=180n" in text
    assert "Mhsense0_neg_dis_s hwneg0 hd0p hsense0_negdn 0 NSENSE W=12u L=180n" in text
    assert "Chsense0_active hsense0_active 0 0.1f IC=0" in text
    assert "Mhsense0_actp_low hsense0_actp hwneg0 vdd vdd PMOS W=1u L=180n" in text
    assert "Mhsense0_actp_high hsense0_actp hwpos0 hsense0_actp_sense 0 NSENSE W=12u L=180n" in text
    assert "Mhsense0_actp_bwd hsense0_actp_sense bwd hsense0_active 0 NSENSE W=12u L=180n" in text
    assert "Mhsense0_pos_keep hwpos0 hwneg0 vdd vdd PMOS W=1u L=180n" in text
    assert "Mhsense0_neg_nkeep hwneg0 hwpos0 0 0 NMOS W=0.5u L=180n" in text


def test_sparse_hidden_update_layer_can_use_senseamp_cmos_writer() -> None:
    topology = FanInTopology.from_fanins((0,), 1, {0: (0,)})
    layer = make_sparse_hidden_update_layer(
        "hidden_update",
        topology=topology,
        source_nodes={0: "h0"},
        weight_prefix="wh",
        update_prefix="uh",
        update_width_u=0.004,
        write_mode="senseamp_cmos_complementary_charge_discharge",
        selector_width_u=12.0,
    )

    text = render(layer)

    assert "Mhwsel0_pos_dis_s hwpos0 hd0n hwsel0_posdn 0 NSENSE W=12u L=180n" in text
    assert "Cfhpg0_0 fhpg0_0 0 2f IC=0" in text
    assert "Mwh0_0n_flow_b" not in text
    assert "Mwh0_0p_cch_w wh0_0p_cch_w hwneg0 whigh vdd PMOS W=0.004u L=180n" in text
    assert "Mwh0_0p_cch_b wh0_0p_cch_b hwsel0_active wh0_0p_cch_w 0 NREL W=0.004u L=180n" in text
    assert "Mwh0_0n_cch_w wh0_0n_cch_w hwpos0 whigh vdd PMOS W=0.004u L=180n" in text
    assert "Mwh0_0n_cch_b wh0_0n_cch_b hwsel0_active wh0_0n_cch_w 0 NREL W=0.004u L=180n" in text
    assert "hwsel0_posbar" not in text


def test_sparse_readout_update_layer_can_use_stored_analog_pretrace_writer() -> None:
    topology = FanInTopology.from_fanins((0, 1), 1, {0: (0,)})
    layer = make_sparse_readout_update_layer(
        "readout_update",
        topology=topology,
        source_nodes={0: "h0", 1: "h1"},
        weight_prefix="vw",
        update_prefix="uw",
        update_width_u=0.004,
        write_mode="analog_trace_charge_discharge",
    )

    text = render(layer)

    assert "Cfpr0_0 fpr0_0 0 2f IC=0" in text
    assert "Mstore_fpr0_0 fpr0_0 fwd h0 0 NREL W=4u L=180n" in text
    assert "Mvw0_0p_ch_a vw0_0p_ch_b fpr0_0 vw0_0p_ch_a 0 NREL W=0.004u L=180n" in text
    assert "Mvw0_0p_ch_d vw0_0p_ch_a dp0 vw0_0p 0 NSENSE W=0.004u L=180n" in text
    assert "Mvw0_0n_flow_a vw0_0n_flow_b fpr0_0 vw0_0n_flow_a 0 NREL W=0.004u L=180n" in text
    assert "Cfprg0_0" not in text
    assert "rwsel0" not in text


def test_sparse_readout_update_layer_can_use_hybrid_trace_spike_writer() -> None:
    topology = FanInTopology.from_fanins((0,), 1, {0: (0,)})
    layer = make_sparse_readout_update_layer(
        "readout_update",
        topology=topology,
        source_nodes={0: "h0"},
        weight_prefix="vw",
        update_prefix="uw",
        update_width_u=0.004,
        charge_width_u=0.0008,
        discharge_width_u=0.00002,
        write_mode="hybrid_trace_spike_charge_discharge",
    )

    text = render(layer)

    assert "Mrwsel0_pos_sel rwsel0_posbar dp0 rwsel0_src 0 NSENSE W=8u L=180n" in text
    assert "Cfpr0_0 fpr0_0 0 2f IC=0" in text
    assert "Cfprg0_0 fprg0_0 0 2f IC=0" in text
    assert "Cfprbar0_0 fprbar0_0 0 2f IC=1.2" in text
    assert "Mvw0_0p_cch_w vw0_0p_cch_w rwsel0_posbar whigh vdd PMOS W=0.0008u" in text
    assert "Mvw0_0n_flow_d vw0_0n_flow_a rwpos0 wlow 0 NSENSE W=2e-05u" in text
    assert "Mvw0_0p_pch_s vw0_0p_pch_b rwsel0_posbar whigh vdd PMOS W=0.0002u" in text
    assert "Mvw0_0p_pch_x vw0_0p_pch_g fprbar0_0 vw0_0p_pch_x 0 NREL W=0.0002u" in text
    assert "Mvw0_0p_pch_a vw0_0p_pch_x fpr0_0 vw0_0p 0 NREL W=0.0002u" in text
    assert "Mvw0_0n_pch_s vw0_0n_pch_b rwsel0_negbar whigh vdd PMOS W=0.0002u" in text


def test_sparse_hidden_update_layer_can_use_hybrid_trace_spike_writer() -> None:
    topology = FanInTopology.from_fanins((0,), 1, {0: (0,)})
    layer = make_sparse_hidden_update_layer(
        "hidden_update",
        topology=topology,
        source_nodes={0: "h0"},
        weight_prefix="wh",
        update_prefix="uh",
        update_width_u=0.004,
        write_mode="hybrid_trace_spike_charge_discharge",
        selector_width_u=12.0,
    )

    text = render(layer)

    assert "Mhwsel0_pos_dis_s hwpos0 hd0n hwsel0_posdn 0 NSENSE W=12u L=180n" in text
    assert "Cfhp0_0 fhp0_0 0 2f IC=0" in text
    assert "Cfhpg0_0 fhpg0_0 0 2f IC=0" in text
    assert "Mwh0_0p_cch_w wh0_0p_cch_w hwneg0 whigh vdd PMOS W=0.004u" in text
    assert "Mwh0_0p_cch_b wh0_0p_cch_b hwsel0_active wh0_0p_cch_w 0 NREL W=0.004u" in text
    assert "Mwh0_0n_flow_d" not in text
    assert "Mwh0_0p_pch_s wh0_0p_pch_b hwneg0 whigh vdd PMOS W=0.001u" in text
    assert "Mwh0_0p_pch_x wh0_0p_pch_g fhpbar0_0 wh0_0p_pch_x 0 NREL W=0.001u" in text
    assert "Mwh0_0p_pch_a wh0_0p_pch_x fhp0_0 wh0_0p 0 NREL W=0.001u" in text


def test_differential_relu_hidden_cell_transient_routes_signed_synapse_rails(
    tmp_path,
    ngspice_path: str,
) -> None:
    def render_case(prefix: str, weight_pos_v: float, weight_neg_v: float) -> str:
        deck = NetlistBuilder()
        weight = DifferentialCapState.from_base(
            f"w_{prefix}",
            cap_f=4.0,
            pos_ic_v=weight_pos_v,
            neg_ic_v=weight_neg_v,
        )
        preactivation = DifferentialCapState.from_base(
            f"u_{prefix}",
            cap_f=10.0,
            pos_ic_v=0.0,
            neg_ic_v=0.0,
            leak_to="0",
            leak_ohm="1G",
        )
        activation = CapState(
            f"act_{prefix}",
            f"act_{prefix}",
            8.0,
            0.0,
            leak_to="0",
            leak_ohm="1G",
        )
        synapse = NonnegativeToDifferentialSynapse(
            f"s_{prefix}",
            activation_node="x",
            pos_weight_node=weight.pos_node,
            neg_weight_node=weight.neg_node,
            post_pos_node=preactivation.pos_node,
            post_neg_node=preactivation.neg_node,
            width_u=8.0,
        )
        neuron = DifferentialReLUNeuron(
            prefix,
            preactivation=preactivation,
            activation=activation,
            width_u=24.0,
        )
        for component in (weight, synapse, neuron):
            deck.render_component(component)
        return deck.render_body()

    netlist = f"""
* differential ReLU hidden-cell transient smoke
.param VDD=1.2
.model NREL NMOS LEVEL=1 VTO=0.12 KP=260u LAMBDA=0.03 GAMMA=0.10 PHI=0.60
.model NSENSE NMOS LEVEL=1 VTO=0.03 KP=260u LAMBDA=0.03 GAMMA=0.05 PHI=0.60
.model PMOS PMOS LEVEL=1 VTO=-0.35 KP=90u LAMBDA=0.03 GAMMA=0.20 PHI=0.60
Vdd vdd 0 {{VDD}}
Vx x 0 DC 1.0
Vfwd fwd 0 PULSE(0 {{VDD}} 0.1n 10p 10p 2n 4n)
{render_case("pos", 1.1, 0.0)}
{render_case("neg", 0.0, 1.1)}
.options method=gear maxord=2 reltol=1e-3 abstol=1e-12 vntol=1e-6 rshunt=1e12
.tran 5p 2.2n uic
.meas tran up_pos FIND V(u_posp) AT=2.0n
.meas tran un_pos FIND V(u_posn) AT=2.0n
.meas tran act_pos FIND V(act_pos) AT=2.0n
.meas tran up_neg FIND V(u_negp) AT=2.0n
.meas tran un_neg FIND V(u_negn) AT=2.0n
.meas tran act_neg FIND V(act_neg) AT=2.0n
.end
""".lstrip()
    path = tmp_path / "differential_relu_hidden.cir"
    path.write_text(netlist)

    proc = subprocess.run([ngspice_path, "-b", str(path)], text=True, capture_output=True, timeout=20)

    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-2000:]
    measures = parse_measures(proc.stdout + "\n" + proc.stderr)
    assert measures["up_pos"] > 0.8
    assert measures["un_pos"] < 0.05
    assert measures["act_pos"] > 0.5
    assert measures["up_neg"] < 0.05
    assert measures["un_neg"] > 0.8
    assert measures["act_neg"] < 0.05


def test_differential_relu_hidden_cell_transient_stays_quiet_at_zero_preactivation(
    tmp_path,
    ngspice_path: str,
) -> None:
    deck = NetlistBuilder()
    preactivation = DifferentialCapState.from_base(
        "u_zero",
        cap_f=10.0,
        pos_ic_v=0.0,
        neg_ic_v=0.0,
        leak_to="0",
        leak_ohm="1G",
    )
    activation = CapState(
        "act_zero",
        "act_zero",
        8.0,
        0.0,
        leak_to="0",
        leak_ohm="1G",
    )
    neuron = DifferentialReLUNeuron(
        "zero",
        preactivation=preactivation,
        activation=activation,
        width_u=24.0,
    )
    deck.render_component(neuron)

    netlist = f"""
* zero-input differential ReLU transient regression
.param VDD=1.2
.model NREL NMOS LEVEL=1 VTO=0.12 KP=260u LAMBDA=0.03 GAMMA=0.10 PHI=0.60
.model NSENSE NMOS LEVEL=1 VTO=0.03 KP=260u LAMBDA=0.03 GAMMA=0.05 PHI=0.60
.model PMOS PMOS LEVEL=1 VTO=-0.35 KP=90u LAMBDA=0.03 GAMMA=0.20 PHI=0.60
Vdd vdd 0 {{VDD}}
Vfwd fwd 0 PULSE(0 {{VDD}} 0.1n 10p 10p 2n 4n)
{deck.render_body()}
.options method=gear maxord=2 reltol=1e-3 abstol=1e-12 vntol=1e-6 rshunt=1e12
.tran 5p 2.2n uic
.meas tran up_zero FIND V(u_zerop) AT=2.0n
.meas tran un_zero FIND V(u_zeron) AT=2.0n
.meas tran act_zero FIND V(act_zero) AT=2.0n
.end
""".lstrip()
    path = tmp_path / "differential_relu_zero.cir"
    path.write_text(netlist)

    proc = subprocess.run([ngspice_path, "-b", str(path)], text=True, capture_output=True, timeout=20)

    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-2000:]
    measures = parse_measures(proc.stdout + "\n" + proc.stderr)
    assert abs(measures["up_zero"]) < 0.01
    assert abs(measures["un_zero"]) < 0.01
    assert measures["act_zero"] < 0.05


def test_differential_relu_hidden_cell_transient_bleeds_modest_negative_rail(
    tmp_path,
    ngspice_path: str,
) -> None:
    deck = NetlistBuilder()
    for name, pos_v, neg_v in (
        ("positive", 0.22, 0.05),
        ("negative", 0.00, 0.16),
    ):
        preactivation = DifferentialCapState.from_base(
            f"u_{name}",
            cap_f=10.0,
            pos_ic_v=pos_v,
            neg_ic_v=neg_v,
            leak_to="0",
            leak_ohm="1G",
        )
        activation = CapState(
            f"act_{name}",
            f"act_{name}",
            8.0,
            0.0,
            leak_to="0",
            leak_ohm="1G",
        )
        deck.render_component(
            DifferentialReLUNeuron(
                name,
                preactivation=preactivation,
                activation=activation,
                width_u=24.0,
            )
        )

    netlist = f"""
* modest-negative differential ReLU transient regression
.param VDD=1.2
.model NREL NMOS LEVEL=1 VTO=0.12 KP=260u LAMBDA=0.03 GAMMA=0.10 PHI=0.60
.model NSENSE NMOS LEVEL=1 VTO=0.03 KP=260u LAMBDA=0.03 GAMMA=0.05 PHI=0.60
.model PMOS PMOS LEVEL=1 VTO=-0.35 KP=90u LAMBDA=0.03 GAMMA=0.20 PHI=0.60
Vdd vdd 0 {{VDD}}
Vfwd fwd 0 PULSE(0 {{VDD}} 0.1n 10p 10p 2n 4n)
{deck.render_body()}
.options method=gear maxord=2 reltol=1e-3 abstol=1e-12 vntol=1e-6 rshunt=1e12
.tran 5p 2.2n uic
.meas tran act_positive FIND V(act_positive) AT=2.0n
.meas tran act_negative FIND V(act_negative) AT=2.0n
.end
""".lstrip()
    path = tmp_path / "differential_relu_modest_negative.cir"
    path.write_text(netlist)

    proc = subprocess.run([ngspice_path, "-b", str(path)], text=True, capture_output=True, timeout=20)

    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-2000:]
    measures = parse_measures(proc.stdout + "\n" + proc.stderr)
    assert measures["act_positive"] > 0.10
    assert measures["act_negative"] < 0.05


def test_differential_error_transport_transient_routes_signed_products(
    tmp_path,
    ngspice_path: str,
) -> None:
    def render_case(prefix: str, err_pos_v: float, err_neg_v: float, weight_pos_v: float, weight_neg_v: float) -> str:
        deck = NetlistBuilder()
        weight = DifferentialCapState.from_base(
            f"w_{prefix}",
            cap_f=4.0,
            pos_ic_v=weight_pos_v,
            neg_ic_v=weight_neg_v,
        )
        delta = DifferentialCapState.from_base(
            f"d_{prefix}",
            cap_f=10.0,
            pos_ic_v=0.0,
            neg_ic_v=0.0,
            leak_to="0",
            leak_ohm="1G",
        )
        synapse = DifferentialToDifferentialSynapse(
            f"bt_{prefix}",
            source_pos_node=f"ep_{prefix}",
            source_neg_node=f"en_{prefix}",
            pos_weight_node=weight.pos_node,
            neg_weight_node=weight.neg_node,
            post_pos_node=delta.pos_node,
            post_neg_node=delta.neg_node,
            width_u=10.0,
        )
        deck.raw(f"Vep_{prefix} ep_{prefix} 0 DC {err_pos_v:.12g}")
        deck.raw(f"Ven_{prefix} en_{prefix} 0 DC {err_neg_v:.12g}")
        for component in (weight, delta, synapse):
            deck.render_component(component)
        return deck.render_body()

    netlist = f"""
* differential backward-transport transient smoke
.param VDD=1.2
.model NREL NMOS LEVEL=1 VTO=0.12 KP=260u LAMBDA=0.03 GAMMA=0.10 PHI=0.60
Vbwd bwd 0 PULSE(0 {{VDD}} 0.1n 10p 10p 2n 4n)
{render_case("posw_pose", 1.0, 0.0, 1.1, 0.0)}
{render_case("negw_pose", 1.0, 0.0, 0.0, 1.1)}
{render_case("posw_nege", 0.0, 1.0, 1.1, 0.0)}
.options method=gear maxord=2 reltol=1e-3 abstol=1e-12 vntol=1e-6 rshunt=1e12
.tran 5p 2.2n uic
.meas tran dp_posw_pose FIND V(d_posw_posep) AT=2.0n
.meas tran dn_posw_pose FIND V(d_posw_posen) AT=2.0n
.meas tran dp_negw_pose FIND V(d_negw_posep) AT=2.0n
.meas tran dn_negw_pose FIND V(d_negw_posen) AT=2.0n
.meas tran dp_posw_nege FIND V(d_posw_negep) AT=2.0n
.meas tran dn_posw_nege FIND V(d_posw_negen) AT=2.0n
.end
""".lstrip()
    path = tmp_path / "differential_error_transport.cir"
    path.write_text(netlist)

    proc = subprocess.run([ngspice_path, "-b", str(path)], text=True, capture_output=True, timeout=20)

    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-2000:]
    measures = parse_measures(proc.stdout + "\n" + proc.stderr)
    assert measures["dp_posw_pose"] > 0.5
    assert measures["dn_posw_pose"] < 0.05
    assert measures["dp_negw_pose"] < 0.05
    assert measures["dn_negw_pose"] > 0.5
    assert measures["dp_posw_nege"] < 0.05
    assert measures["dn_posw_nege"] > 0.5


def test_pretrace_cell_renders_current_synapse_spike_contract() -> None:
    cell = PreTraceCell(
        name="fpro00",
        source_node="act0",
        mode="synapse_spike",
        cap_f=2.0,
        consume_width_u=0.05,
        boost_width_u=3.5,
        spike_gate_name="fprg00",
        spike_bar_name="fprbar00",
        spike_mid_name="fprm00",
    )

    text = render(cell)

    assert cell.input_nodes() == ("act0",)
    assert cell.state_nodes() == ("fpro00", "fprg00", "fprbar00")
    assert "Cfpro00 fpro00 0 2f IC=0" in text
    assert "Mstore_fpro00 fpro00 fwd act0 0 NREL W=4u L=180n" in text
    assert "Cfprg00 fprg00 0 2f IC=0" in text
    assert "Cfprbar00 fprbar00 0 2f IC=1.2" in text
    assert "Mspike_fprbar00_fwd fprbar00 fwd fprm00 0 NREL W=3.5u L=180n" in text
    assert "Mspike_fprbar00_act fprm00 act0 spikeref 0 NSENSE W=3.5u L=180n" in text
    assert "Mspike_fprg00_p vdd fprbar00 fprg00 vdd PMOS W=3.5u L=180n" in text
    assert "Mspike_fprg00_n fprg00 fprbar00 0 0 NMOS W=3.5u L=180n" in text


def test_pretrace_spike_detector_transient_fires_on_small_relu_activation(
    tmp_path,
    ngspice_path: str,
) -> None:
    cell = PreTraceCell(
        name="fpr",
        source_node="act",
        mode="synapse_spike",
        cap_f=2.0,
        consume_width_u=0.05,
        boost_width_u=4.0,
        spike_ref_node="spikeref",
    )
    body = render(cell)
    netlist = f"""
* pretrace small-activation transient
.param VDD=1.2
.model NREL NMOS LEVEL=1 VTO=0.12 KP=260u LAMBDA=0.03 GAMMA=0.10 PHI=0.60
.model NSENSE NMOS LEVEL=1 VTO=0.03 KP=260u LAMBDA=0.03 GAMMA=0.05 PHI=0.60
.model NMOS NMOS LEVEL=1 VTO=0.45 KP=220u LAMBDA=0.04
.model PMOS PMOS LEVEL=1 VTO=-0.45 KP=90u LAMBDA=0.05
Vdd vdd 0 {{VDD}}
Vact act 0 DC 0.076
Vspikeref spikeref 0 DC 0
Vfwd fwd 0 PULSE(0 {{VDD}} 0.1n 10p 10p 2.0n 4n)
Vrstf rstf 0 DC 0
{body}
.options method=gear maxord=2 reltol=1e-3 abstol=1e-12 vntol=1e-6 rshunt=1e12
.tran 5p 2.4n uic
.meas tran gate FIND V(fprg) AT=2.2n
.meas tran bar FIND V(fprbar) AT=2.2n
.end
""".lstrip()
    path = tmp_path / "pretrace_low_activation.cir"
    path.write_text(netlist)

    proc = subprocess.run([ngspice_path, "-b", str(path)], text=True, capture_output=True, timeout=20)

    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-2000:]
    measures = parse_measures(proc.stdout + "\n" + proc.stderr)
    assert measures["gate"] > 0.8
    assert measures["bar"] < 0.2


def test_pretrace_array_builds_readout_and_hidden_trace_names_from_topology() -> None:
    readout = PreTraceArray.from_readout_topology(
        "readout_traces",
        FanInTopology.from_fanins((0, 1, 2), 2, {0: (0, 2), 1: (1,)}),
        mode="synapse_spike",
        cap_f=2.0,
        consume_width_u=0.05,
        boost_width_u=3.5,
    )
    hidden = PreTraceArray.from_hidden_topology(
        "hidden_traces",
        FanInTopology.from_fanins(("bias", "x0", "x1"), 2, {0: ("bias", "x0"), 1: ("x1",)}),
        mode="synapse_spike",
        cap_f=2.0,
        consume_width_u=0.05,
        boost_width_u=3.5,
    )

    deck = NetlistBuilder()
    deck.render_component(readout)
    deck.render_component(hidden)
    text = deck.render_body()

    assert readout.input_nodes() == ("act0", "act2", "act1")
    assert "fprg02" in readout.state_nodes()
    assert "fphig0_x0" in hidden.state_nodes()
    assert "Cfprg00 fprg00 0 2f IC=0" in text
    assert "Cfprg02 fprg02 0 2f IC=0" in text
    assert "Cfprg11 fprg11 0 2f IC=0" in text
    assert "Cfprg01" not in text
    assert "Cfphig0_x0 fphig0_x0 0 2f IC=0" in text
    assert "Cfphig1_x1 fphig1_x1 0 2f IC=0" in text
    assert "Cfphig0_x1" not in text
    assert readout.spike_mid_nodes() == ("fprm00", "fprm02", "fprm11")
    assert hidden.spike_mid_nodes() == ("fphim0_bias", "fphim0_x0", "fphim1_x1")


def test_pretrace_validation_rejects_nonphysical_widths() -> None:
    with pytest.raises(ValueError, match="spike width"):
        render(
            PreTraceCell(
                name="fpro00",
                source_node="act0",
                mode="synapse_spike",
                cap_f=2.0,
                consume_width_u=0.05,
                boost_width_u=3.5,
                spike_width_u=0.0,
            )
        )


def test_diffpair_bleed_write_selector_renders_current_selector_contract() -> None:
    selector = DiffPairBleedWriteSelector(
        "rwsel0",
        positive_error_gate="dp0",
        negative_error_gate="dn0",
        positive_write_gate="rwpos0",
        negative_write_gate="rwneg0",
        width_u=8.0,
        label="readout output 0",
    )

    text = render(selector)

    assert (
        "* Differential-pair write selector for readout output 0: shared-tail dp/dn comparison "
        "with weak bwd bleed."
    ) in text
    assert "Crwsel0_pos rwpos0 0 0.1f IC=0" in text
    assert "Mrwsel0_pos_bleed rwpos0 bwd 0 0 NMOS W=0.2u L=180n" in text
    assert "Crwsel0_posbar rwsel0_posbar 0 0.05f IC=1.2" in text
    assert "Mrwsel0_pos_sel rwsel0_posbar dp0 rwsel0_src 0 NSENSE W=8u L=180n" in text
    assert "Mrwsel0_tail rwsel0_src bwd 0 0 NMOS W=4u L=180n" in text
    assert "Mrwsel0_pos_n rwsel0_posmid rwsel0_negbar rwpos0 0 NMOS W=8u L=180n" in text
    assert selector.input_nodes() == ("dp0", "dn0")
    assert selector.output_nodes() == ("rwpos0", "rwneg0")
    assert selector.positive_bar_node == "rwsel0_posbar"
    assert selector.negative_bar_node == "rwsel0_negbar"


def test_mutually_inhibited_write_selector_renders_cross_inhibited_rails() -> None:
    selector = MutuallyInhibitedWriteSelector(
        "hwsel0",
        positive_error_gate="gdh0p",
        negative_error_gate="gdh0n",
        positive_write_gate="hwpos0",
        negative_write_gate="hwneg0",
        width_u=9.0,
        label="hidden neuron 0",
    )

    text = render(selector)

    assert selector.input_nodes() == ("gdh0p", "gdh0n")
    assert selector.output_nodes() == ("hwpos0", "hwneg0")
    assert "Chwsel0_pos hwpos0 0 0.1f IC=0" in text
    assert "Mhwsel0_pos_inh vdd gdh0n hwsel0_possrc vdd PMOS W=9u L=180n" in text
    assert "Mhwsel0_pos_gate hwsel0_possrc gdh0p hwpos0 0 NSENSE W=9u L=180n" in text
    assert "Mhwsel0_neg_inh vdd gdh0p hwsel0_negsrc vdd PMOS W=9u L=180n" in text
    assert "Mhwsel0_neg_gate hwsel0_negsrc gdh0n hwneg0 0 NSENSE W=9u L=180n" in text
    assert "Mhwsel0_pos_kill hwpos0 gdh0n 0 0 NMOS W=9u L=180n" in text
    assert "Mhwsel0_neg_kill hwneg0 gdh0p 0 0 NMOS W=9u L=180n" in text


def test_direct_flow_weight_cell_renders_bounded_charge_discharge_contract() -> None:
    cell = DirectFlowWeightCell(
        "w00",
        pos_weight_node="vw00p",
        neg_weight_node="vw00n",
        pre_gate="act0",
        positive_write_gate="dp0",
        negative_write_gate="dn0",
        pos_discharge_width_u=0.005,
        neg_discharge_width_u=0.005,
        pos_charge_width_u=0.011,
        neg_charge_width_u=0.011,
        charge_enabled=True,
        discharge_enabled=True,
        pos_high_node="pos_hi",
        neg_high_node="neg_hi",
        pos_low_node="pos_lo",
        neg_low_node="neg_lo",
    )

    text = render(cell)

    assert cell.input_nodes() == ("act0", "dp0", "dn0", "bwd")
    assert "Mvw00n_flow_b vw00n bwd vw00n_flow_b 0 NREL W=0.005u L=180n" in text
    assert "Mvw00n_flow_a vw00n_flow_b act0 vw00n_flow_a 0 NREL W=0.005u L=180n" in text
    assert "Mvw00n_flow_d vw00n_flow_a dp0 neg_lo 0 NSENSE W=0.005u L=180n" in text
    assert "Mvw00p_flow_d vw00p_flow_a dn0 pos_lo 0 NSENSE W=0.005u L=180n" in text
    assert "Mvw00p_ch_b pos_hi bwd vw00p_ch_b 0 NREL W=0.011u L=180n" in text
    assert "Mvw00p_ch_d vw00p_ch_a dp0 vw00p 0 NSENSE W=0.011u L=180n" in text
    assert "Mvw00n_ch_b neg_hi bwd vw00n_ch_b 0 NREL W=0.011u L=180n" in text
    assert "Mvw00n_ch_d vw00n_ch_a dn0 vw00n 0 NSENSE W=0.011u L=180n" in text
    assert "Rpar_vw00p_ch_a vw00p_ch_a 0 1e9" in text
    assert "vw00p" in cell.state_nodes()
    assert "vw00n_ch_a" in cell.state_nodes()


def test_direct_flow_weight_cell_renders_pmos_charge_and_center_contract() -> None:
    cell = DirectFlowWeightCell(
        "w00",
        pos_weight_node="vw00p",
        neg_weight_node="vw00n",
        pre_gate="fprg00",
        positive_write_gate="rwpos0",
        negative_write_gate="rwneg0",
        positive_write_gate_low_true="rwsel0_posbar",
        negative_write_gate_low_true="rwsel0_negbar",
        pos_charge_width_u=0.02,
        neg_charge_width_u=0.02,
        pmos_charge_write=True,
        discharge_enabled=False,
        center_pull_width_u=0.0002,
        center_pull_mode="state_high",
        center_pull_gate="apply",
        pos_center_pull_node="pcenter",
        neg_center_pull_node="ncenter",
    )

    text = render(cell)

    assert "Mvw00p_pch_s vw00p_pch_b rwsel0_posbar whigh vdd PMOS W=0.02u L=180n" in text
    assert "Mvw00p_pch_g vw00p_pch_b bwd vw00p_pch_g 0 NREL W=0.02u L=180n" in text
    assert "Mvw00p_pch_a vw00p_pch_g fprg00 vw00p 0 NREL W=0.02u L=180n" in text
    assert "Mvw00n_pch_s vw00n_pch_b rwsel0_negbar whigh vdd PMOS W=0.02u L=180n" in text
    assert "Mvw00p_center_g vw00p apply vw00p_center_g 0 NREL W=0.0002u L=180n" in text
    assert "Mvw00n_center_s vw00n_center_g vw00n ncenter 0 NREL W=0.0002u L=180n" in text
    assert "Mvw00p_flow_d" not in text


def test_direct_flow_weight_cell_uses_gate_width_overrides_for_pmos_charge_selectors() -> None:
    cell = DirectFlowWeightCell(
        "w00",
        pos_weight_node="vw00p",
        neg_weight_node="vw00n",
        pre_gate="fprg00",
        positive_write_gate="rwpos0",
        negative_write_gate="rwneg0",
        positive_write_gate_low_true="rwsel0_posbar",
        negative_write_gate_low_true="rwsel0_negbar",
        pos_charge_width_u=0.02,
        neg_charge_width_u=0.02,
        positive_gate_pos_charge_width_u=0.05,
        negative_gate_neg_charge_width_u=0.007,
        pmos_charge_write=True,
        discharge_enabled=False,
    )

    text = render(cell)

    assert "Mvw00p_pch_s vw00p_pch_b rwsel0_posbar whigh vdd PMOS W=0.05u" in text
    assert "Mvw00p_pch_g vw00p_pch_b bwd vw00p_pch_g 0 NREL W=0.02u" in text
    assert "Mvw00n_pch_s vw00n_pch_b rwsel0_negbar whigh vdd PMOS W=0.007u" in text
    assert "Mvw00n_pch_g vw00n_pch_b bwd vw00n_pch_g 0 NREL W=0.02u" in text


def test_direct_flow_weight_cell_rejects_pmos_without_low_true_gates() -> None:
    with pytest.raises(ValueError, match="low-true write gates"):
        render(
            DirectFlowWeightCell(
                "w00",
                pos_weight_node="vw00p",
                neg_weight_node="vw00n",
                pre_gate="act0",
                positive_write_gate="rwpos0",
                negative_write_gate="rwneg0",
                pos_charge_width_u=0.02,
                neg_charge_width_u=0.02,
                pmos_charge_write=True,
            )
        )


def test_node_parasitics_render_spice_anchor_contract() -> None:
    anchors = NodeParasitics("anchors", ("mid0", "src0"))
    text = render(anchors)

    assert anchors.state_nodes() == ("mid0", "src0")
    assert "Rpar_mid0 mid0 0 1e9" in text
    assert "Cpar_mid0 mid0 0 0.02f IC=0" in text
    assert "Rpar_src0 src0 0 1e9" in text
    assert "Cpar_src0 src0 0 0.02f IC=0" in text


def test_readout_branch_renders_gate_stack_read_kernel_contract() -> None:
    branch = ReadoutBranch(
        "rb",
        style="gate_stack",
        branch="pos",
        activation_node="act0",
        weight_node="vw00p",
        score_node="score0",
        width_u=56.0,
    )

    text = render(branch)

    assert branch.input_nodes() == ("act0", "vw00p")
    assert branch.output_nodes() == ("score0",)
    assert branch.state_nodes() == ("rb_0", "rb_1")
    assert "Mrb_a vdd act0 rb_0 0 NSENSE W=56u L=180n" in text
    assert "Mrb_w rb_0 vw00p rb_1 0 NREL W=56u L=180n" in text
    assert "Mrb_f rb_1 fwd score0 0 NREL W=56u L=180n" in text
    assert "Rpar_rb_0 rb_0 0 1e9" in text


def test_readout_branch_renders_buffered_activation_contract() -> None:
    branch = ReadoutBranch(
        "rb",
        style="pass_act_buffered",
        branch="neg",
        activation_node="act0",
        weight_node="vw00n",
        score_node="scoren0",
        width_u=48.0,
    )

    text = render(branch)

    assert branch.state_nodes() == ("rb_actbuf", "rb_1")
    assert "Mrb_actbuf_src vdd act0 rb_actbuf 0 NSENSE W=48u L=180n" in text
    assert "Mrb_w rb_actbuf vw00n rb_1 0 NREL W=48u L=180n" in text
    assert "Mrb_f rb_1 fwd scoren0 0 NREL W=48u L=180n" in text


def test_readout_cap_state_loader_matches_topology_contract(tmp_path) -> None:
    path = tmp_path / "readout_caps.csv"
    path.write_text(
        "name,value\n"
        "vbo0p,0.11\n"
        "vbo0n,0.12\n"
        "vbo1p,0.21\n"
        "vbo1n,0.22\n"
        "vw00p,0.31\n"
        "vw00n,0.32\n"
        "vw11p,0.43\n"
        "vw11n,0.44\n"
    )

    init = load_readout_cap_state_csv(
        path,
        hidden_count=2,
        output_count=2,
        readout_fanins={0: (0,), 1: (1,)},
    )

    assert expected_readout_cap_names(
        hidden_count=2,
        output_count=2,
        readout_fanins={0: (0,), 1: (1,)},
    ) == set(init)
    assert init["vbo1n"] == 0.22
    assert init["vw11p"] == 0.43


def test_readout_cap_state_loader_rejects_missing_caps(tmp_path) -> None:
    path = tmp_path / "bad_caps.csv"
    path.write_text("cap,value\nvbo0p,0.11\n")

    with pytest.raises(ValueError, match="missing capacitor states"):
        load_readout_cap_state_csv(path, hidden_count=1, output_count=1)


def test_fanin_topology_supports_sparse_fanin_and_fanout_contracts() -> None:
    readout = FanInTopology.random_fanout(tuple(range(8)), 5, seed=13, fan_out=3)
    readout_fanouts = readout.fanouts()

    assert readout.edge_count() == 24
    assert all(len(outs) == 3 for outs in readout_fanouts.values())
    assert all(len(set(outs)) == 3 for outs in readout_fanouts.values())

    hidden = FanInTopology.random_fanin(
        ("x0", "x1", "x2", "x3"),
        5,
        seed=17,
        fan_in=3,
        always_sources=("bias",),
    )

    assert all(rails[0] == "bias" for rails in hidden.as_fanins().values())
    assert hidden.fanin_counts(exclude_sources=("bias",)) == [3, 3, 3, 3, 3]
    assert hidden.edge_count(exclude_sources=("bias",)) == 15
    assert set(hidden.fanouts()) >= {"bias", "x0", "x1", "x2", "x3"}


def test_fanin_topology_supports_balanced_sparse_fanout() -> None:
    readout = FanInTopology.balanced_random_fanout(tuple(range(16)), 5, seed=7, fan_out=3)
    readout_fanouts = readout.fanouts()
    fanin_counts = readout.fanin_counts()

    assert readout.edge_count() == 48
    assert all(len(outs) == 3 for outs in readout_fanouts.values())
    assert all(len(set(outs)) == 3 for outs in readout_fanouts.values())
    assert max(fanin_counts) - min(fanin_counts) <= 1
    assert sorted(fanin_counts) == [9, 9, 10, 10, 10]
