from __future__ import annotations

from typing import Mapping

from .components import (
    CapState,
    DiffPairBleedWriteSelector,
    DifferentialCapState,
    DifferentialReLUNeuron,
    DifferentialSignalGate,
    DifferentialToDifferentialSynapse,
    DirectFlowWeightCell,
    MutuallyInhibitedWriteSelector,
    NonnegativeToDifferentialSynapse,
    PreTraceCell,
    ReadoutBranch,
    RegenerativeDifferentialWriteSelector,
)
from .core import Component, Layer, Port
from .topology import FanInTopology, SourceId


def _source_slug(source: SourceId) -> str:
    text = str(source)
    slug = "".join(ch if ch.isalnum() else "_" for ch in text)
    return slug or "src"


def make_sparse_differential_relu_layer(
    name: str,
    *,
    topology: FanInTopology,
    source_nodes: Mapping[SourceId, str],
    activation_prefix: str = "act",
    preactivation_prefix: str = "u",
    weight_prefix: str = "w",
    synapse_prefix: str = "s",
    weight_cap_f: float = 4.0,
    preactivation_cap_f: float = 10.0,
    activation_cap_f: float = 8.0,
    weight_pos_ic_v: float = 0.60,
    weight_neg_ic_v: float = 0.60,
    weight_initials: Mapping[tuple[int, SourceId], tuple[float, float]] | None = None,
    preactivation_leak_ohm: str | float = "1G",
    activation_leak_ohm: str | float = "1G",
    synapse_width_u: float = 3.0,
    neuron_width_u: float = 12.0,
    neuron_sense_model: str = "NSENSE",
    neuron_fwd_model: str = "NREL",
    fwd_gate: str = "fwd",
) -> Layer:
    """Build a sparse hidden layer with signed accumulation and ReLU outputs.

    The layer contract is:

    ``nonnegative source -> signed synapse branches -> differential u+/u- -> ReLU -> nonnegative activation``.

    Weight capacitors are per edge.  Preactivation capacitors are per neuron.
    """

    if not name:
        raise ValueError("layer name must be nonempty")
    if topology.sink_count <= 0:
        raise ValueError("topology must have at least one sink")
    if weight_cap_f <= 0 or preactivation_cap_f <= 0 or activation_cap_f <= 0:
        raise ValueError("layer capacitances must be positive")
    if synapse_width_u <= 0 or neuron_width_u <= 0:
        raise ValueError("layer transistor widths must be positive")
    missing = [source for source in topology.sources if source not in source_nodes]
    if missing:
        raise ValueError(f"missing source node mappings: {', '.join(str(source) for source in missing[:6])}")

    children: list[Component] = []
    used_weight_bases: set[str] = set()
    outputs: list[Port] = []

    for sink in range(topology.sink_count):
        pre = DifferentialCapState.from_base(
            f"{preactivation_prefix}{sink}",
            cap_f=preactivation_cap_f,
            pos_ic_v=0.0,
            neg_ic_v=0.0,
            leak_to="0",
            leak_ohm=preactivation_leak_ohm,
        )
        act = CapState(
            name=f"{activation_prefix}{sink}",
            node=f"{activation_prefix}{sink}",
            cap_f=activation_cap_f,
            ic_v=0.0,
            leak_to="0",
            leak_ohm=activation_leak_ohm,
        )
        for source in topology.as_fanins()[sink]:
            if source not in source_nodes:
                raise ValueError(f"missing source node mapping for fan-in {source!r}")
            source_slug = _source_slug(source)
            weight_base = f"{weight_prefix}{sink}_{source_slug}"
            if weight_base in used_weight_bases:
                raise ValueError(f"duplicate weight base generated: {weight_base}")
            used_weight_bases.add(weight_base)
            pos_ic_v, neg_ic_v = (
                weight_initials[(sink, source)]
                if weight_initials is not None and (sink, source) in weight_initials
                else (weight_pos_ic_v, weight_neg_ic_v)
            )
            weight = DifferentialCapState.from_base(
                weight_base,
                cap_f=weight_cap_f,
                pos_ic_v=pos_ic_v,
                neg_ic_v=neg_ic_v,
            )
            children.append(weight)
            children.append(
                NonnegativeToDifferentialSynapse(
                    name=f"{synapse_prefix}{sink}_{source_slug}",
                    activation_node=source_nodes[source],
                    pos_weight_node=weight.pos_node,
                    neg_weight_node=weight.neg_node,
                    post_pos_node=pre.pos_node,
                    post_neg_node=pre.neg_node,
                    width_u=synapse_width_u,
                    fwd_gate=fwd_gate,
                )
            )
        children.append(
            DifferentialReLUNeuron(
                name=f"{name}_{sink}",
                preactivation=pre,
                activation=act,
                width_u=neuron_width_u,
                sense_model=neuron_sense_model,
                fwd_model=neuron_fwd_model,
                fwd_gate=fwd_gate,
            )
        )
        outputs.append(Port.at(f"{activation_prefix}{sink}", act.node))

    inputs = tuple(Port.at(str(source), source_nodes[source]) for source in topology.sources)
    return Layer(name=name, _children=children, inputs=inputs, outputs=tuple(outputs), topology=topology)


def make_sparse_signed_readout_layer(
    name: str,
    *,
    topology: FanInTopology,
    source_nodes: Mapping[SourceId, str],
    score_prefix: str = "score",
    weight_prefix: str = "vw",
    branch_prefix: str = "ro",
    weight_cap_f: float = 4.0,
    score_cap_f: float = 10.0,
    weight_pos_ic_v: float = 0.60,
    weight_neg_ic_v: float = 0.60,
    weight_initials: Mapping[tuple[int, SourceId], tuple[float, float]] | None = None,
    score_leak_ohm: str | float = "1G",
    branch_style: str = "gate_stack",
    branch_width_u: float = 8.0,
    fwd_gate: str = "fwd",
) -> Layer:
    """Build a sparse readout with explicit positive/negative score rails."""

    if not name:
        raise ValueError("readout layer name must be nonempty")
    if topology.sink_count <= 0:
        raise ValueError("readout topology must have at least one sink")
    if weight_cap_f <= 0 or score_cap_f <= 0:
        raise ValueError("readout capacitances must be positive")
    if branch_width_u <= 0:
        raise ValueError("readout branch width must be positive")
    missing = [source for source in topology.sources if source not in source_nodes]
    if missing:
        raise ValueError(f"missing readout source node mappings: {', '.join(str(source) for source in missing[:6])}")

    children: list[Component] = []
    used_weight_bases: set[str] = set()
    outputs: list[Port] = []

    for sink in range(topology.sink_count):
        score = DifferentialCapState.from_base(
            f"{score_prefix}{sink}",
            cap_f=score_cap_f,
            pos_ic_v=0.0,
            neg_ic_v=0.0,
            leak_to="0",
            leak_ohm=score_leak_ohm,
        )
        children.append(score)
        outputs.extend(
            [
                Port.at(f"{score_prefix}{sink}p", score.pos_node),
                Port.at(f"{score_prefix}{sink}n", score.neg_node),
            ]
        )
        for source in topology.as_fanins()[sink]:
            if source not in source_nodes:
                raise ValueError(f"missing readout source node mapping for fan-in {source!r}")
            source_slug = _source_slug(source)
            weight_base = f"{weight_prefix}{sink}_{source_slug}"
            if weight_base in used_weight_bases:
                raise ValueError(f"duplicate readout weight base generated: {weight_base}")
            used_weight_bases.add(weight_base)
            pos_ic_v, neg_ic_v = (
                weight_initials[(sink, source)]
                if weight_initials is not None and (sink, source) in weight_initials
                else (weight_pos_ic_v, weight_neg_ic_v)
            )
            weight = DifferentialCapState.from_base(
                weight_base,
                cap_f=weight_cap_f,
                pos_ic_v=pos_ic_v,
                neg_ic_v=neg_ic_v,
            )
            children.append(weight)
            children.extend(
                [
                    ReadoutBranch(
                        name=f"{branch_prefix}{sink}_{source_slug}_p",
                        style=branch_style,
                        branch="pos",
                        activation_node=source_nodes[source],
                        weight_node=weight.pos_node,
                        score_node=score.pos_node,
                        width_u=branch_width_u,
                        fwd_gate=fwd_gate,
                    ),
                    ReadoutBranch(
                        name=f"{branch_prefix}{sink}_{source_slug}_n",
                        style=branch_style,
                        branch="neg",
                        activation_node=source_nodes[source],
                        weight_node=weight.neg_node,
                        score_node=score.neg_node,
                        width_u=branch_width_u,
                        fwd_gate=fwd_gate,
                    ),
                ]
            )

    inputs = tuple(Port.at(str(source), source_nodes[source]) for source in topology.sources)
    return Layer(name=name, _children=children, inputs=inputs, outputs=tuple(outputs), topology=topology)


def make_sparse_readout_update_layer(
    name: str,
    *,
    topology: FanInTopology,
    source_nodes: Mapping[SourceId, str],
    weight_prefix: str = "vw",
    update_prefix: str = "uw",
    positive_error_prefix: str = "dp",
    negative_error_prefix: str = "dn",
    update_width_u: float = 0.002,
    charge_width_u: float | None = None,
    discharge_width_u: float | None = None,
    source_update_scales: Mapping[SourceId, float] | None = None,
    pos_high_node: str = "whigh",
    pos_low_node: str = "wlow",
    neg_high_node: str = "whigh",
    neg_low_node: str = "wlow",
    backward_gate: str = "bwd",
    write_gate_device: str = "NSENSE",
    write_mode: str = "simple_charge_discharge",
    selector_width_u: float = 8.0,
    pretrace_cap_f: float = 2.0,
    pretrace_consume_width_u: float = 0.05,
    pretrace_boost_width_u: float = 4.0,
    spike_ref_node: str = "spikeref",
    hybrid_trace_scale: float = 0.25,
) -> Layer:
    """Build local readout writers for every sparse readout edge."""

    if not name:
        raise ValueError("readout update layer name must be nonempty")
    if write_mode not in {
        "simple_charge_discharge",
        "analog_trace_charge_discharge",
        "cmos_complementary_charge_discharge",
        "hybrid_trace_spike_charge_discharge",
    }:
        raise ValueError(f"unknown readout update write mode: {write_mode}")
    if update_width_u < 0:
        raise ValueError("readout update width must be nonnegative")
    if selector_width_u < 0:
        raise ValueError("readout selector width must be nonnegative")
    if pretrace_cap_f <= 0 or pretrace_consume_width_u <= 0 or pretrace_boost_width_u <= 0:
        raise ValueError("readout pretrace parameters must be positive")
    charge = update_width_u if charge_width_u is None else charge_width_u
    discharge = update_width_u if discharge_width_u is None else discharge_width_u
    if charge < 0 or discharge < 0:
        raise ValueError("readout update action widths must be nonnegative")
    if hybrid_trace_scale < 0:
        raise ValueError("readout hybrid trace scale must be nonnegative")
    source_update_scales = {} if source_update_scales is None else dict(source_update_scales)
    negative_scales = [source for source, scale in source_update_scales.items() if scale < 0]
    if negative_scales:
        raise ValueError(f"readout source update scales must be nonnegative: {negative_scales[:6]}")
    missing = [source for source in topology.sources if source not in source_nodes]
    if missing:
        raise ValueError(f"missing update source node mappings: {', '.join(str(source) for source in missing[:6])}")

    children: list[Component] = []
    inputs: list[Port] = [Port.at(str(source), source_nodes[source]) for source in topology.sources]
    selector_nodes: dict[int, tuple[str, str, str | None, str | None]] = {}
    for sink in range(topology.sink_count):
        if write_mode in {"cmos_complementary_charge_discharge", "hybrid_trace_spike_charge_discharge"}:
            selector = DiffPairBleedWriteSelector(
                name=f"rwsel{sink}",
                positive_error_gate=f"{positive_error_prefix}{sink}",
                negative_error_gate=f"{negative_error_prefix}{sink}",
                positive_write_gate=f"rwpos{sink}",
                negative_write_gate=f"rwneg{sink}",
                width_u=selector_width_u,
                label=f"{name} output {sink}",
                backward_gate=backward_gate,
            )
            children.append(selector)
            selector_nodes[sink] = (
                selector.positive_write_gate,
                selector.negative_write_gate,
                selector.positive_bar_node,
                selector.negative_bar_node,
            )
        else:
            selector_nodes[sink] = (
                f"{positive_error_prefix}{sink}",
                f"{negative_error_prefix}{sink}",
                None,
                None,
            )
        inputs.extend(
            [
                Port.at(f"{positive_error_prefix}{sink}", f"{positive_error_prefix}{sink}"),
                Port.at(f"{negative_error_prefix}{sink}", f"{negative_error_prefix}{sink}"),
            ]
        )
        for source in topology.as_fanins()[sink]:
            if source not in source_nodes:
                raise ValueError(f"missing update source node mapping for fan-in {source!r}")
            source_slug = _source_slug(source)
            weight_base = f"{weight_prefix}{sink}_{source_slug}"
            positive_write_gate, negative_write_gate, positive_bar, negative_bar = selector_nodes[sink]
            source_scale = source_update_scales.get(source, 1.0)
            source_charge = charge * source_scale
            source_discharge = discharge * source_scale
            pretrace: PreTraceCell | None = None
            if write_mode in {"cmos_complementary_charge_discharge", "hybrid_trace_spike_charge_discharge"}:
                pretrace = PreTraceCell(
                    name=f"fpr{sink}_{source_slug}",
                    source_node=source_nodes[source],
                    mode="synapse_spike",
                    cap_f=pretrace_cap_f,
                    consume_width_u=pretrace_consume_width_u,
                    boost_width_u=pretrace_boost_width_u,
                    spike_gate_name=f"fprg{sink}_{source_slug}",
                    spike_bar_name=f"fprbar{sink}_{source_slug}",
                    spike_mid_name=f"fprm{sink}_{source_slug}",
                    spike_ref_node=spike_ref_node,
                )
                children.append(pretrace)
                pre_gate = pretrace.spike_gate_node
                pre_gate_low_true = pretrace.spike_bar_node
            elif write_mode == "analog_trace_charge_discharge":
                pretrace = PreTraceCell(
                    name=f"fpr{sink}_{source_slug}",
                    source_node=source_nodes[source],
                    mode="synapse_gate",
                    cap_f=pretrace_cap_f,
                    consume_width_u=pretrace_consume_width_u,
                    boost_width_u=pretrace_boost_width_u,
                    spike_ref_node=spike_ref_node,
                )
                children.append(pretrace)
                pre_gate = pretrace.trace_node
                pre_gate_low_true = None
            else:
                pre_gate = source_nodes[source]
                pre_gate_low_true = None
            children.append(
                DirectFlowWeightCell(
                    name=f"{update_prefix}{sink}_{source_slug}",
                    pos_weight_node=f"{weight_base}p",
                    neg_weight_node=f"{weight_base}n",
                    pre_gate=pre_gate,
                    positive_write_gate=positive_write_gate,
                    negative_write_gate=negative_write_gate,
                    pos_discharge_width_u=source_discharge,
                    neg_discharge_width_u=source_discharge,
                    pos_charge_width_u=source_charge,
                    neg_charge_width_u=source_charge,
                    pos_high_node=pos_high_node,
                    pos_low_node=pos_low_node,
                    neg_high_node=neg_high_node,
                    neg_low_node=neg_low_node,
                    backward_gate=backward_gate,
                    write_gate_device=write_gate_device,
                    charge_enabled=write_mode in {"simple_charge_discharge", "analog_trace_charge_discharge"},
                    discharge_enabled=True,
                    cmos_complementary_charge=write_mode
                    in {"cmos_complementary_charge_discharge", "hybrid_trace_spike_charge_discharge"},
                    positive_write_gate_low_true=positive_bar,
                    negative_write_gate_low_true=negative_bar,
                    pre_gate_low_true=pre_gate_low_true,
                )
            )
            if write_mode == "hybrid_trace_spike_charge_discharge" and hybrid_trace_scale > 0:
                assert pretrace is not None
                children.append(
                    DirectFlowWeightCell(
                        name=f"{update_prefix}{sink}_{source_slug}_trace",
                        pos_weight_node=f"{weight_base}p",
                        neg_weight_node=f"{weight_base}n",
                        pre_gate=pretrace.trace_node,
                        positive_write_gate=positive_write_gate,
                        negative_write_gate=negative_write_gate,
                        pos_discharge_width_u=0.0,
                        neg_discharge_width_u=0.0,
                        pos_charge_width_u=source_charge * hybrid_trace_scale,
                        neg_charge_width_u=source_charge * hybrid_trace_scale,
                        pos_high_node=pos_high_node,
                        pos_low_node=pos_low_node,
                        neg_high_node=neg_high_node,
                        neg_low_node=neg_low_node,
                        backward_gate=backward_gate,
                        write_gate_device=write_gate_device,
                        charge_enabled=False,
                        discharge_enabled=False,
                        pmos_charge_write=True,
                        positive_write_gate_low_true=positive_bar,
                        negative_write_gate_low_true=negative_bar,
                        charge_extra_gate=pretrace.spike_bar_node,
                    )
                )

    return Layer(name=name, _children=children, inputs=tuple(inputs), outputs=(), topology=topology)


def make_sparse_differential_error_transport_layer(
    name: str,
    *,
    topology: FanInTopology,
    positive_error_prefix: str = "dp",
    negative_error_prefix: str = "dn",
    delta_prefix: str = "hd",
    weight_prefix: str = "vw",
    transport_prefix: str = "bt",
    delta_cap_f: float = 10.0,
    delta_leak_ohm: str | float = "1G",
    transport_width_u: float = 4.0,
    transport_style: str = "gate_stack",
    backward_gate: str = "bwd",
    skip_sources: tuple[SourceId, ...] = ("bias",),
) -> Layer:
    """Build reverse signed error transport through an existing sparse weight array.

    For every readout edge ``source -> sink`` the layer reuses the same
    differential weight capacitor names as the forward readout and routes the
    signed output error rails into per-source delta rails.  It implements the
    signed product routing needed for ``W^T e`` without collapsing errors into a
    one-wire signal.
    """

    if not name:
        raise ValueError("error transport layer name must be nonempty")
    if topology.sink_count <= 0:
        raise ValueError("error transport topology must have at least one sink")
    if delta_cap_f <= 0:
        raise ValueError("delta capacitance must be positive")
    if transport_width_u <= 0:
        raise ValueError("error transport width must be positive")
    if transport_style not in {"pass_error_source", "gate_stack"}:
        raise ValueError(f"unknown error transport style: {transport_style}")

    children: list[Component] = []
    outputs: list[Port] = []
    skipped = set(skip_sources)
    delta_states: dict[SourceId, DifferentialCapState] = {}
    for source in topology.sources:
        if source in skipped:
            continue
        source_slug = _source_slug(source)
        delta = DifferentialCapState.from_base(
            f"{delta_prefix}{source_slug}",
            cap_f=delta_cap_f,
            pos_ic_v=0.0,
            neg_ic_v=0.0,
            leak_to="0",
            leak_ohm=delta_leak_ohm,
        )
        delta_states[source] = delta
        children.append(delta)
        outputs.extend(
            [
                Port.at(f"{delta_prefix}{source_slug}p", delta.pos_node),
                Port.at(f"{delta_prefix}{source_slug}n", delta.neg_node),
            ]
        )

    used_transport_names: set[str] = set()
    for sink, sources in topology.as_fanins().items():
        for source in sources:
            if source in skipped:
                continue
            source_slug = _source_slug(source)
            transport_name = f"{transport_prefix}{sink}_{source_slug}"
            if transport_name in used_transport_names:
                raise ValueError(f"duplicate error transport name generated: {transport_name}")
            used_transport_names.add(transport_name)
            weight_base = f"{weight_prefix}{sink}_{source_slug}"
            delta = delta_states[source]
            children.append(
                DifferentialToDifferentialSynapse(
                    name=transport_name,
                    source_pos_node=f"{positive_error_prefix}{sink}",
                    source_neg_node=f"{negative_error_prefix}{sink}",
                    pos_weight_node=f"{weight_base}p",
                    neg_weight_node=f"{weight_base}n",
                    post_pos_node=delta.pos_node,
                    post_neg_node=delta.neg_node,
                    width_u=transport_width_u,
                    flow_gate=backward_gate,
                    style=transport_style,
                )
            )

    inputs = tuple(
        port
        for sink in range(topology.sink_count)
        for port in (
            Port.at(f"{positive_error_prefix}{sink}", f"{positive_error_prefix}{sink}"),
            Port.at(f"{negative_error_prefix}{sink}", f"{negative_error_prefix}{sink}"),
        )
    )
    return Layer(name=name, _children=children, inputs=inputs, outputs=tuple(outputs), topology=topology)


def make_sparse_relu_delta_gate_layer(
    name: str,
    *,
    sink_count: int,
    activation_nodes: Mapping[int, str],
    input_delta_prefix: str = "hd",
    output_delta_prefix: str = "gdh",
    delta_cap_f: float = 6.0,
    delta_leak_ohm: str | float = "1G",
    gate_width_u: float = 8.0,
    backward_gate: str = "bwd",
) -> Layer:
    """Gate per-neuron signed deltas by stored ReLU activations."""

    if not name:
        raise ValueError("delta gate layer name must be nonempty")
    if sink_count <= 0:
        raise ValueError("delta gate sink_count must be positive")
    if delta_cap_f <= 0:
        raise ValueError("delta gate capacitance must be positive")
    if gate_width_u <= 0:
        raise ValueError("delta gate width must be positive")
    missing = [sink for sink in range(sink_count) if sink not in activation_nodes]
    if missing:
        raise ValueError(f"missing activation nodes for delta gates: {', '.join(str(sink) for sink in missing[:6])}")

    children: list[Component] = []
    inputs: list[Port] = []
    outputs: list[Port] = []
    for sink in range(sink_count):
        pos = CapState(
            name=f"{output_delta_prefix}{sink}p",
            node=f"{output_delta_prefix}{sink}p",
            cap_f=delta_cap_f,
            ic_v=0.0,
            leak_to="0",
            leak_ohm=delta_leak_ohm,
        )
        neg = CapState(
            name=f"{output_delta_prefix}{sink}n",
            node=f"{output_delta_prefix}{sink}n",
            cap_f=delta_cap_f,
            ic_v=0.0,
            leak_to="0",
            leak_ohm=delta_leak_ohm,
        )
        children.append(
            DifferentialSignalGate(
                name=f"{name}_{sink}",
                positive_input_node=f"{input_delta_prefix}{sink}p",
                negative_input_node=f"{input_delta_prefix}{sink}n",
                gate_node=activation_nodes[sink],
                positive_output=pos,
                negative_output=neg,
                width_u=gate_width_u,
                flow_gate=backward_gate,
            )
        )
        inputs.extend(
            [
                Port.at(f"{input_delta_prefix}{sink}p", f"{input_delta_prefix}{sink}p"),
                Port.at(f"{input_delta_prefix}{sink}n", f"{input_delta_prefix}{sink}n"),
                Port.at(f"act{sink}", activation_nodes[sink]),
            ]
        )
        outputs.extend(
            [
                Port.at(f"{output_delta_prefix}{sink}p", pos.node),
                Port.at(f"{output_delta_prefix}{sink}n", neg.node),
            ]
        )

    return Layer(name=name, _children=children, inputs=tuple(inputs), outputs=tuple(outputs))


def make_sparse_hidden_update_layer(
    name: str,
    *,
    topology: FanInTopology,
    source_nodes: Mapping[SourceId, str],
    weight_prefix: str = "w",
    update_prefix: str = "uh",
    delta_prefix: str = "hd",
    update_width_u: float = 0.002,
    charge_width_u: float | None = None,
    discharge_width_u: float | None = None,
    pos_high_node: str = "whigh",
    pos_low_node: str = "wlow",
    neg_high_node: str = "whigh",
    neg_low_node: str = "wlow",
    backward_gate: str = "bwd",
    write_gate_device: str = "NSENSE",
    write_mode: str = "simple_charge_discharge",
    selector_width_u: float = 8.0,
    pretrace_cap_f: float = 2.0,
    pretrace_consume_width_u: float = 0.05,
    pretrace_boost_width_u: float = 4.0,
    spike_ref_node: str = "spikeref",
    hybrid_trace_scale: float = 0.25,
) -> Layer:
    """Build local hidden-synapse writers driven by signed post-neuron deltas."""

    if not name:
        raise ValueError("hidden update layer name must be nonempty")
    if write_mode not in {
        "simple_charge_discharge",
        "analog_trace_charge_discharge",
        "diffpair_charge_discharge",
        "inhibit_charge_discharge",
        "cmos_complementary_charge_discharge",
        "hybrid_trace_spike_charge_discharge",
        "senseamp_charge_discharge",
        "senseamp_cmos_complementary_charge_discharge",
    }:
        raise ValueError(f"unknown hidden update write mode: {write_mode}")
    if update_width_u < 0:
        raise ValueError("hidden update width must be nonnegative")
    if selector_width_u < 0:
        raise ValueError("hidden update selector width must be nonnegative")
    if pretrace_cap_f <= 0 or pretrace_consume_width_u <= 0 or pretrace_boost_width_u <= 0:
        raise ValueError("hidden pretrace parameters must be positive")
    charge = update_width_u if charge_width_u is None else charge_width_u
    discharge = update_width_u if discharge_width_u is None else discharge_width_u
    if charge < 0 or discharge < 0:
        raise ValueError("hidden update action widths must be nonnegative")
    if hybrid_trace_scale < 0:
        raise ValueError("hidden hybrid trace scale must be nonnegative")
    missing = [source for source in topology.sources if source not in source_nodes]
    if missing:
        raise ValueError(f"missing hidden update source mappings: {', '.join(str(source) for source in missing[:6])}")

    children: list[Component] = []
    inputs: list[Port] = [Port.at(str(source), source_nodes[source]) for source in topology.sources]
    selector_nodes: dict[int, tuple[str, str, str | None, str | None]] = {}
    for sink in range(topology.sink_count):
        positive_delta = f"{delta_prefix}{sink}p"
        negative_delta = f"{delta_prefix}{sink}n"
        inputs.extend([Port.at(positive_delta, positive_delta), Port.at(negative_delta, negative_delta)])
        if write_mode in {
            "diffpair_charge_discharge",
            "cmos_complementary_charge_discharge",
        }:
            selector = DiffPairBleedWriteSelector(
                name=f"hwsel{sink}",
                positive_error_gate=positive_delta,
                negative_error_gate=negative_delta,
                positive_write_gate=f"hwpos{sink}",
                negative_write_gate=f"hwneg{sink}",
                width_u=selector_width_u,
                label=f"{name} hidden neuron {sink}",
                backward_gate=backward_gate,
            )
            children.append(selector)
            selector_nodes[sink] = (
                selector.positive_write_gate,
                selector.negative_write_gate,
                selector.positive_bar_node,
                selector.negative_bar_node,
            )
        elif write_mode == "hybrid_trace_spike_charge_discharge":
            selector = RegenerativeDifferentialWriteSelector(
                name=f"hwsel{sink}",
                positive_error_gate=positive_delta,
                negative_error_gate=negative_delta,
                positive_write_gate=f"hwpos{sink}",
                negative_write_gate=f"hwneg{sink}",
                width_u=selector_width_u,
                label=f"{name} hidden neuron {sink}",
                sense_gate=backward_gate,
                nkeeper_width_u=0.0,
            )
            children.append(selector)
            selector_nodes[sink] = (
                selector.positive_write_gate,
                selector.negative_write_gate,
                selector.negative_write_gate,
                selector.positive_write_gate,
            )
        elif write_mode == "inhibit_charge_discharge":
            selector = MutuallyInhibitedWriteSelector(
                name=f"hwsel{sink}",
                positive_error_gate=positive_delta,
                negative_error_gate=negative_delta,
                positive_write_gate=f"hwpos{sink}",
                negative_write_gate=f"hwneg{sink}",
                width_u=selector_width_u,
                label=f"{name} hidden neuron {sink}",
            )
            children.append(selector)
            selector_nodes[sink] = (selector.positive_write_gate, selector.negative_write_gate, None, None)
        elif write_mode in {"senseamp_charge_discharge", "senseamp_cmos_complementary_charge_discharge"}:
            selector = RegenerativeDifferentialWriteSelector(
                name=f"hwsel{sink}",
                positive_error_gate=positive_delta,
                negative_error_gate=negative_delta,
                positive_write_gate=f"hwpos{sink}",
                negative_write_gate=f"hwneg{sink}",
                width_u=selector_width_u,
                label=f"{name} hidden neuron {sink}",
                sense_gate=backward_gate,
                nkeeper_width_u=0.0,
            )
            children.append(selector)
            selector_nodes[sink] = (
                selector.positive_write_gate,
                selector.negative_write_gate,
                selector.negative_write_gate,
                selector.positive_write_gate,
            )
        else:
            selector_nodes[sink] = (positive_delta, negative_delta, None, None)
        for source in topology.as_fanins()[sink]:
            source_slug = _source_slug(source)
            weight_base = f"{weight_prefix}{sink}_{source_slug}"
            pretrace: PreTraceCell | None = None
            if write_mode in {
                "cmos_complementary_charge_discharge",
                "hybrid_trace_spike_charge_discharge",
                "senseamp_cmos_complementary_charge_discharge",
            }:
                pretrace = PreTraceCell(
                    name=f"fhp{sink}_{source_slug}",
                    source_node=source_nodes[source],
                    mode="synapse_spike",
                    cap_f=pretrace_cap_f,
                    consume_width_u=pretrace_consume_width_u,
                    boost_width_u=pretrace_boost_width_u,
                    spike_gate_name=f"fhpg{sink}_{source_slug}",
                    spike_bar_name=f"fhpbar{sink}_{source_slug}",
                    spike_mid_name=f"fhpm{sink}_{source_slug}",
                    spike_ref_node=spike_ref_node,
                )
                children.append(pretrace)
                pre_gate = pretrace.spike_gate_node
                pre_gate_low_true = pretrace.spike_bar_node
            elif write_mode == "analog_trace_charge_discharge":
                pretrace = PreTraceCell(
                    name=f"fhp{sink}_{source_slug}",
                    source_node=source_nodes[source],
                    mode="synapse_gate",
                    cap_f=pretrace_cap_f,
                    consume_width_u=pretrace_consume_width_u,
                    boost_width_u=pretrace_boost_width_u,
                )
                children.append(pretrace)
                pre_gate = pretrace.trace_node
                pre_gate_low_true = None
            else:
                pre_gate = source_nodes[source]
                pre_gate_low_true = None
            positive_write_gate, negative_write_gate, positive_bar, negative_bar = selector_nodes[sink]
            cell_backward_gate = (
                f"hwsel{sink}_active"
                if write_mode
                in {
                    "hybrid_trace_spike_charge_discharge",
                    "senseamp_charge_discharge",
                    "senseamp_cmos_complementary_charge_discharge",
                }
                else backward_gate
            )
            children.append(
                DirectFlowWeightCell(
                    name=f"{update_prefix}{sink}_{source_slug}",
                    pos_weight_node=f"{weight_base}p",
                    neg_weight_node=f"{weight_base}n",
                    pre_gate=pre_gate,
                    positive_write_gate=positive_write_gate,
                    negative_write_gate=negative_write_gate,
                    pos_discharge_width_u=discharge,
                    neg_discharge_width_u=discharge,
                    pos_charge_width_u=charge,
                    neg_charge_width_u=charge,
                    pos_high_node=pos_high_node,
                    pos_low_node=pos_low_node,
                    neg_high_node=neg_high_node,
                    neg_low_node=neg_low_node,
                    backward_gate=cell_backward_gate,
                    write_gate_device=write_gate_device,
                    charge_enabled=write_mode in {
                        "simple_charge_discharge",
                        "analog_trace_charge_discharge",
                        "diffpair_charge_discharge",
                        "inhibit_charge_discharge",
                        "senseamp_charge_discharge",
                    },
                    discharge_enabled=write_mode
                    not in {
                        "hybrid_trace_spike_charge_discharge",
                        "senseamp_cmos_complementary_charge_discharge",
                    },
                    cmos_complementary_charge=write_mode in {
                        "cmos_complementary_charge_discharge",
                        "hybrid_trace_spike_charge_discharge",
                        "senseamp_cmos_complementary_charge_discharge",
                    },
                    gate_cmos_charge_with_backward=write_mode
                    in {"hybrid_trace_spike_charge_discharge", "senseamp_cmos_complementary_charge_discharge"},
                    positive_write_gate_low_true=positive_bar,
                    negative_write_gate_low_true=negative_bar,
                    pre_gate_low_true=pre_gate_low_true,
                )
            )
            if write_mode == "hybrid_trace_spike_charge_discharge" and hybrid_trace_scale > 0:
                assert pretrace is not None
                children.append(
                    DirectFlowWeightCell(
                        name=f"{update_prefix}{sink}_{source_slug}_trace",
                        pos_weight_node=f"{weight_base}p",
                        neg_weight_node=f"{weight_base}n",
                        pre_gate=pretrace.trace_node,
                        positive_write_gate=positive_write_gate,
                        negative_write_gate=negative_write_gate,
                        pos_discharge_width_u=0.0,
                        neg_discharge_width_u=0.0,
                        pos_charge_width_u=charge * hybrid_trace_scale,
                        neg_charge_width_u=charge * hybrid_trace_scale,
                        pos_high_node=pos_high_node,
                        pos_low_node=pos_low_node,
                        neg_high_node=neg_high_node,
                        neg_low_node=neg_low_node,
                        backward_gate=cell_backward_gate,
                        write_gate_device=write_gate_device,
                        charge_enabled=False,
                        discharge_enabled=False,
                        pmos_charge_write=True,
                        positive_write_gate_low_true=positive_bar,
                        negative_write_gate_low_true=negative_bar,
                        charge_extra_gate=pretrace.spike_bar_node,
                    )
                )

    return Layer(name=name, _children=children, inputs=tuple(inputs), outputs=(), topology=topology)


__all__ = [
    "make_sparse_differential_error_transport_layer",
    "make_sparse_differential_relu_layer",
    "make_sparse_hidden_update_layer",
    "make_sparse_readout_update_layer",
    "make_sparse_relu_delta_gate_layer",
    "make_sparse_signed_readout_layer",
]
