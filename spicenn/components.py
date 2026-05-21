from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .core import Component, NetlistBuilder, Neuron, Synapse
from .topology import FanInTopology


@dataclass
class CapState(Component):
    """Physical capacitor-backed state node.

    This is intentionally SPICE-first: a CapState owns the capacitor, optional
    leakage, and optional reset switch that make the state observable in the
    transient deck.
    """

    name: str
    node: str
    cap_f: float
    ic_v: float = 0.0
    leak_to: str | None = None
    leak_ohm: str | float | None = None
    reset_gate: str | None = None
    reset_to: str = "0"
    reset_model: str = "NMOS"
    reset_width_u: float = 4.0
    reset_body: str = "0"

    def children(self) -> list[Component]:
        return []

    def state_nodes(self) -> tuple[str, ...]:
        return (self.node,)

    def validate(self) -> None:
        if not self.name or not self.node:
            raise ValueError("CapState requires nonempty name and node")
        if self.cap_f <= 0:
            raise ValueError("CapState capacitance must be positive")
        if self.leak_to is not None and self.leak_ohm is None:
            raise ValueError("CapState leak_to requires leak_ohm")
        if self.reset_gate is not None and self.reset_width_u <= 0:
            raise ValueError("CapState reset width must be positive")

    def render(self, deck: NetlistBuilder) -> None:
        deck.capacitor(self.name, self.node, "0", f"{self.cap_f:.12g}f", ic=self.ic_v)
        if self.leak_to is not None:
            deck.resistor(self.name, self.node, self.leak_to, self.leak_ohm or "1G")
        if self.reset_gate is not None:
            if self.reset_to == "0":
                deck.mos(
                    f"reset_{self.name}",
                    self.node,
                    self.reset_gate,
                    "0",
                    self.reset_body,
                    self.reset_model,
                    width_u=self.reset_width_u,
                )
            else:
                deck.mos(
                    f"reset_{self.name}",
                    self.reset_to,
                    self.reset_gate,
                    self.node,
                    self.reset_body,
                    self.reset_model,
                    width_u=self.reset_width_u,
                )


@dataclass
class CapStateArray(Component):
    """Collection of ordinary capacitor-backed state nodes."""

    name: str
    states: tuple[CapState, ...]

    @classmethod
    def from_nodes(
        cls,
        name: str,
        nodes: tuple[str, ...],
        *,
        cap_f: float,
        ic_v: float = 0.0,
        leak_to: str = "0",
        leak_ohm: str | float = "1G",
    ) -> "CapStateArray":
        return cls(
            name=name,
            states=tuple(
                CapState(
                    name=node,
                    node=node,
                    cap_f=cap_f,
                    ic_v=ic_v,
                    leak_to=leak_to,
                    leak_ohm=leak_ohm,
                )
                for node in nodes
            ),
        )

    def children(self) -> list[Component]:
        return list(self.states)

    def state_nodes(self) -> tuple[str, ...]:
        return tuple(node for state in self.states for node in state.state_nodes())

    def validate(self) -> None:
        if not self.name:
            raise ValueError("CapStateArray requires a nonempty name")
        for state in self.states:
            state.validate()

    def render(self, deck: NetlistBuilder) -> None:
        for state in self.states:
            deck.render_component(state)


@dataclass
class CapStateProgram(Component):
    """Program a named set of capacitor-backed state nodes from initial voltages."""

    name: str
    initial: Mapping[str, float]
    cap_f: float
    leak_to: str = "0"
    leak_ohm: str | float = "1e15"

    def children(self) -> list[Component]:
        return [
            CapState(
                name=node,
                node=node,
                cap_f=self.cap_f,
                ic_v=float(value),
                leak_to=self.leak_to,
                leak_ohm=self.leak_ohm,
            )
            for node, value in sorted(self.initial.items())
        ]

    def state_nodes(self) -> tuple[str, ...]:
        return tuple(node for node, _value in sorted(self.initial.items()))

    def validate(self) -> None:
        if not self.name:
            raise ValueError("CapStateProgram requires a nonempty name")
        if self.cap_f <= 0:
            raise ValueError("CapStateProgram capacitance must be positive")
        if not self.initial:
            raise ValueError("CapStateProgram requires at least one initial state")
        for node in self.initial:
            if not node:
                raise ValueError("CapStateProgram requires nonempty state names")

    def render(self, deck: NetlistBuilder) -> None:
        for child in self.children():
            deck.render_component(child)


@dataclass
class DifferentialCapState(Component):
    """Signed state represented by matched positive/negative capacitor rails."""

    name: str
    pos_node: str
    neg_node: str
    cap_f: float
    pos_ic_v: float
    neg_ic_v: float
    leak_to: str = "0"
    leak_ohm: str | float = "1e15"

    @classmethod
    def from_base(
        cls,
        base: str,
        *,
        cap_f: float,
        pos_ic_v: float,
        neg_ic_v: float,
        leak_to: str = "0",
        leak_ohm: str | float = "1e15",
    ) -> "DifferentialCapState":
        return cls(
            name=base,
            pos_node=f"{base}p",
            neg_node=f"{base}n",
            cap_f=cap_f,
            pos_ic_v=pos_ic_v,
            neg_ic_v=neg_ic_v,
            leak_to=leak_to,
            leak_ohm=leak_ohm,
        )

    def children(self) -> list[Component]:
        return [
            CapState(f"{self.name}p", self.pos_node, self.cap_f, self.pos_ic_v, self.leak_to, self.leak_ohm),
            CapState(f"{self.name}n", self.neg_node, self.cap_f, self.neg_ic_v, self.leak_to, self.leak_ohm),
        ]

    def state_nodes(self) -> tuple[str, ...]:
        return (self.pos_node, self.neg_node)

    def validate(self) -> None:
        if not self.name or not self.pos_node or not self.neg_node:
            raise ValueError("DifferentialCapState requires nonempty name and nodes")
        if self.cap_f <= 0:
            raise ValueError("DifferentialCapState capacitance must be positive")

    def render(self, deck: NetlistBuilder) -> None:
        for child in self.children():
            deck.render_component(child)


@dataclass
class DifferentialCapStateArray(Component):
    """Collection of signed capacitor states keyed by their base node name."""

    name: str
    bases: tuple[str, ...]
    initial: Mapping[str, float]
    cap_f: float
    leak_to: str = "0"
    leak_ohm: str | float = "1e15"

    def children(self) -> list[Component]:
        return [
            DifferentialCapState.from_base(
                base,
                cap_f=self.cap_f,
                pos_ic_v=float(self.initial[f"{base}p"]),
                neg_ic_v=float(self.initial[f"{base}n"]),
                leak_to=self.leak_to,
                leak_ohm=self.leak_ohm,
            )
            for base in self.bases
        ]

    def state_nodes(self) -> tuple[str, ...]:
        return tuple(node for child in self.children() for node in child.state_nodes())

    def validate(self) -> None:
        if not self.name:
            raise ValueError("DifferentialCapStateArray requires a nonempty name")
        if self.cap_f <= 0:
            raise ValueError("DifferentialCapStateArray capacitance must be positive")
        missing = [node for base in self.bases for node in (f"{base}p", f"{base}n") if node not in self.initial]
        if missing:
            raise ValueError(f"missing initial capacitor states: {', '.join(missing[:6])}")

    def render(self, deck: NetlistBuilder) -> None:
        for child in self.children():
            deck.render_component(child)


@dataclass
class NonnegativeToDifferentialSynapse(Synapse):
    """Signed synapse from one nonnegative activation into differential sum rails.

    The activation can be a single ReLU output wire because it is nonnegative.
    The synapse contribution is signed by routing the positive weight branch to
    the positive preactivation rail and the negative weight branch to the
    negative preactivation rail.
    """

    name: str
    activation_node: str
    pos_weight_node: str
    neg_weight_node: str
    post_pos_node: str
    post_neg_node: str
    width_u: float
    fwd_gate: str = "fwd"
    model: str = "NREL"

    @property
    def pos_mid_node(self) -> str:
        return f"{self.name}_pos_mid"

    @property
    def neg_mid_node(self) -> str:
        return f"{self.name}_neg_mid"

    def input_nodes(self) -> tuple[str, ...]:
        return (self.activation_node, self.pos_weight_node, self.neg_weight_node, self.fwd_gate)

    def output_nodes(self) -> tuple[str, ...]:
        return (self.post_pos_node, self.post_neg_node)

    def state_nodes(self) -> tuple[str, ...]:
        return (self.pos_mid_node, self.neg_mid_node)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("NonnegativeToDifferentialSynapse requires a nonempty name")
        for node in (
            self.activation_node,
            self.pos_weight_node,
            self.neg_weight_node,
            self.post_pos_node,
            self.post_neg_node,
            self.fwd_gate,
            self.model,
        ):
            if not node:
                raise ValueError("NonnegativeToDifferentialSynapse requires nonempty nodes")
        if self.width_u <= 0:
            raise ValueError("NonnegativeToDifferentialSynapse width must be positive")

    def render(self, deck: NetlistBuilder) -> None:
        deck.extend(
            [
                (
                    f"* {self.name}: nonnegative activation {self.activation_node} drives "
                    f"signed contribution {self.post_pos_node}-{self.post_neg_node}."
                ),
                f"M{self.name}_pos_w {self.activation_node} {self.pos_weight_node} {self.pos_mid_node} 0 {self.model} W={self.width_u:.12g}u L=180n",
                f"M{self.name}_pos_f {self.pos_mid_node} {self.fwd_gate} {self.post_pos_node} 0 {self.model} W={self.width_u:.12g}u L=180n",
                f"M{self.name}_neg_w {self.activation_node} {self.neg_weight_node} {self.neg_mid_node} 0 {self.model} W={self.width_u:.12g}u L=180n",
                f"M{self.name}_neg_f {self.neg_mid_node} {self.fwd_gate} {self.post_neg_node} 0 {self.model} W={self.width_u:.12g}u L=180n",
            ]
        )
        deck.render_component(NodeParasitics(f"{self.name}_parasitics", self.state_nodes()))


@dataclass
class DifferentialToDifferentialSynapse(Synapse):
    """Signed reverse-flow synapse from signed source rails into signed sum rails.

    This is the backward/error-flow counterpart to
    ``NonnegativeToDifferentialSynapse``.  A signed source value ``e = ep-en``
    passing through a signed stored weight ``w = wp-wn`` contributes:

    ``wp*ep + wn*en`` to the downstream positive rail, and
    ``wp*en + wn*ep`` to the downstream negative rail.

    The cell therefore keeps the sign accounting in the circuit instead of
    asking Python to collapse the error into a scalar.
    """

    name: str
    source_pos_node: str
    source_neg_node: str
    pos_weight_node: str
    neg_weight_node: str
    post_pos_node: str
    post_neg_node: str
    width_u: float
    flow_gate: str = "bwd"
    model: str = "NREL"
    style: str = "pass_error_source"

    def _mid_node(self, label: str) -> str:
        return f"{self.name}_{label}_mid"

    def _gate_mid_nodes(self, label: str) -> tuple[str, str]:
        return (f"{self.name}_{label}_e_mid", f"{self.name}_{label}_w_mid")

    def input_nodes(self) -> tuple[str, ...]:
        return (
            self.source_pos_node,
            self.source_neg_node,
            self.pos_weight_node,
            self.neg_weight_node,
            self.flow_gate,
        )

    def output_nodes(self) -> tuple[str, ...]:
        return (self.post_pos_node, self.post_neg_node)

    def state_nodes(self) -> tuple[str, ...]:
        if self.style == "gate_stack":
            return tuple(node for label in ("wp_ep", "wp_en", "wn_ep", "wn_en") for node in self._gate_mid_nodes(label))
        return (
            self._mid_node("wp_ep"),
            self._mid_node("wp_en"),
            self._mid_node("wn_ep"),
            self._mid_node("wn_en"),
        )

    def validate(self) -> None:
        if not self.name:
            raise ValueError("DifferentialToDifferentialSynapse requires a nonempty name")
        for node in (
            self.source_pos_node,
            self.source_neg_node,
            self.pos_weight_node,
            self.neg_weight_node,
            self.post_pos_node,
            self.post_neg_node,
            self.flow_gate,
            self.model,
        ):
            if not node:
                raise ValueError("DifferentialToDifferentialSynapse requires nonempty nodes")
        if self.width_u <= 0:
            raise ValueError("DifferentialToDifferentialSynapse width must be positive")
        if self.style not in {"pass_error_source", "gate_stack"}:
            raise ValueError(f"unknown DifferentialToDifferentialSynapse style: {self.style}")

    def _render_branch(
        self,
        deck: NetlistBuilder,
        *,
        label: str,
        source_node: str,
        weight_node: str,
        post_node: str,
    ) -> None:
        if self.style == "gate_stack":
            err_mid, weight_mid = self._gate_mid_nodes(label)
            deck.extend(
                [
                    f"M{self.name}_{label}_e vdd {source_node} {err_mid} 0 {self.model} W={self.width_u:.12g}u L=180n",
                    f"M{self.name}_{label}_w {err_mid} {weight_node} {weight_mid} 0 {self.model} W={self.width_u:.12g}u L=180n",
                    f"M{self.name}_{label}_b {weight_mid} {self.flow_gate} {post_node} 0 {self.model} W={self.width_u:.12g}u L=180n",
                ]
            )
        else:
            mid = self._mid_node(label)
            deck.extend(
                [
                    f"M{self.name}_{label}_w {source_node} {weight_node} {mid} 0 {self.model} W={self.width_u:.12g}u L=180n",
                    f"M{self.name}_{label}_b {mid} {self.flow_gate} {post_node} 0 {self.model} W={self.width_u:.12g}u L=180n",
                ]
            )

    def render(self, deck: NetlistBuilder) -> None:
        deck.extend(
            [
                (
                    f"* {self.name}: signed reverse flow "
                    f"({self.source_pos_node}-{self.source_neg_node}) through "
                    f"({self.pos_weight_node}-{self.neg_weight_node}) into "
                    f"{self.post_pos_node}-{self.post_neg_node}."
                )
            ]
        )
        self._render_branch(
            deck,
            label="wp_ep",
            source_node=self.source_pos_node,
            weight_node=self.pos_weight_node,
            post_node=self.post_pos_node,
        )
        self._render_branch(
            deck,
            label="wp_en",
            source_node=self.source_neg_node,
            weight_node=self.pos_weight_node,
            post_node=self.post_neg_node,
        )
        self._render_branch(
            deck,
            label="wn_ep",
            source_node=self.source_pos_node,
            weight_node=self.neg_weight_node,
            post_node=self.post_neg_node,
        )
        self._render_branch(
            deck,
            label="wn_en",
            source_node=self.source_neg_node,
            weight_node=self.neg_weight_node,
            post_node=self.post_pos_node,
        )
        deck.render_component(NodeParasitics(f"{self.name}_parasitics", self.state_nodes()))


@dataclass
class DifferentialSignalGate(Component):
    """Gate signed rails with a stored nonnegative activation/mask.

    ReLU backward propagation needs ``delta = mask * raw_delta``.  This block
    renders that as two local MOS gate stacks, one for the positive rail and
    one for the negative rail.  The input rails are used as gates, so the raw
    delta capacitors are not directly charge-shared into the gated delta state.
    """

    name: str
    positive_input_node: str
    negative_input_node: str
    gate_node: str
    positive_output: CapState
    negative_output: CapState
    width_u: float = 8.0
    flow_gate: str = "bwd"
    model: str = "NSENSE"

    @property
    def positive_input_mid_node(self) -> str:
        return f"{self.name}_p_in"

    @property
    def positive_gate_mid_node(self) -> str:
        return f"{self.name}_p_gate"

    @property
    def negative_input_mid_node(self) -> str:
        return f"{self.name}_n_in"

    @property
    def negative_gate_mid_node(self) -> str:
        return f"{self.name}_n_gate"

    def children(self) -> list[Component]:
        return [self.positive_output, self.negative_output]

    def input_nodes(self) -> tuple[str, ...]:
        return (self.positive_input_node, self.negative_input_node, self.gate_node, self.flow_gate)

    def output_nodes(self) -> tuple[str, ...]:
        return (self.positive_output.node, self.negative_output.node)

    def state_nodes(self) -> tuple[str, ...]:
        return (
            *self.positive_output.state_nodes(),
            *self.negative_output.state_nodes(),
            self.positive_input_mid_node,
            self.positive_gate_mid_node,
            self.negative_input_mid_node,
            self.negative_gate_mid_node,
        )

    def validate(self) -> None:
        if not self.name:
            raise ValueError("DifferentialSignalGate requires a nonempty name")
        for node in (
            self.positive_input_node,
            self.negative_input_node,
            self.gate_node,
            self.flow_gate,
            self.model,
        ):
            if not node:
                raise ValueError("DifferentialSignalGate requires nonempty nodes")
        if self.width_u <= 0:
            raise ValueError("DifferentialSignalGate width must be positive")
        self.positive_output.validate()
        self.negative_output.validate()

    def render(self, deck: NetlistBuilder) -> None:
        self.positive_output.render(deck)
        self.negative_output.render(deck)
        deck.extend(
            [
                f"* {self.name}: gate signed delta rails with stored activation {self.gate_node}.",
                f"M{self.name}_p_in vdd {self.positive_input_node} {self.positive_input_mid_node} 0 {self.model} W={self.width_u:.12g}u L=180n",
                f"M{self.name}_p_gate {self.positive_input_mid_node} {self.gate_node} {self.positive_gate_mid_node} 0 NREL W={self.width_u:.12g}u L=180n",
                f"M{self.name}_p_bwd {self.positive_gate_mid_node} {self.flow_gate} {self.positive_output.node} 0 NREL W={self.width_u:.12g}u L=180n",
                f"M{self.name}_n_in vdd {self.negative_input_node} {self.negative_input_mid_node} 0 {self.model} W={self.width_u:.12g}u L=180n",
                f"M{self.name}_n_gate {self.negative_input_mid_node} {self.gate_node} {self.negative_gate_mid_node} 0 NREL W={self.width_u:.12g}u L=180n",
                f"M{self.name}_n_bwd {self.negative_gate_mid_node} {self.flow_gate} {self.negative_output.node} 0 NREL W={self.width_u:.12g}u L=180n",
            ]
        )
        deck.render_component(
            NodeParasitics(
                f"{self.name}_parasitics",
                (
                    self.positive_input_mid_node,
                    self.positive_gate_mid_node,
                    self.negative_input_mid_node,
                    self.negative_gate_mid_node,
                ),
            )
        )


@dataclass
class DifferentialReLUNeuron(Neuron):
    """ReLU neuron with signed differential preactivation and one output wire."""

    name: str
    preactivation: DifferentialCapState
    activation: CapState
    width_u: float = 24.0
    fwd_gate: str = "fwd"
    sense_model: str = "NSENSE"
    fwd_model: str = "NREL"
    negative_bleed_model: str = "NREL"

    @property
    def pos_low_node(self) -> str:
        return f"{self.name}_relu_pos_low"

    @property
    def pos_mid_node(self) -> str:
        return f"{self.name}_relu_pos_mid"

    def children(self) -> list[Component]:
        return [self.preactivation, self.activation]

    def input_nodes(self) -> tuple[str, ...]:
        return self.preactivation.state_nodes()

    def output_nodes(self) -> tuple[str, ...]:
        return (self.activation.node,)

    def state_nodes(self) -> tuple[str, ...]:
        return (
            *self.preactivation.state_nodes(),
            *self.activation.state_nodes(),
            self.pos_low_node,
            self.pos_mid_node,
        )

    def validate(self) -> None:
        if not self.name:
            raise ValueError("DifferentialReLUNeuron requires a nonempty name")
        if not self.fwd_gate:
            raise ValueError("DifferentialReLUNeuron requires a nonempty forward gate")
        if not self.sense_model or not self.fwd_model or not self.negative_bleed_model:
            raise ValueError("DifferentialReLUNeuron requires nonempty MOS model names")
        if self.width_u <= 0:
            raise ValueError("DifferentialReLUNeuron width must be positive")
        self.preactivation.validate()
        self.activation.validate()

    def render(self, deck: NetlistBuilder) -> None:
        self.preactivation.render(deck)
        self.activation.render(deck)
        pre_pos, pre_neg = self.preactivation.state_nodes()
        deck.extend(
            [
                (
                    f"* {self.name}: differential ReLU, activation charges only when "
                    f"{pre_pos} wins over {pre_neg}."
                ),
                f"Mrelu_{self.name}_inhibit vdd {pre_neg} {self.pos_low_node} vdd PMOS W={self.width_u:.12g}u L=180n",
                f"Mrelu_{self.name}_sense {self.pos_low_node} {pre_pos} {self.pos_mid_node} 0 {self.sense_model} W={self.width_u:.12g}u L=180n",
                f"Mrelu_{self.name}_fwd {self.pos_mid_node} {self.fwd_gate} {self.activation.node} 0 {self.fwd_model} W={self.width_u:.12g}u L=180n",
                f"Mrelu_{self.name}_neg_bleed {self.activation.node} {pre_neg} 0 0 {self.negative_bleed_model} W={self.width_u:.12g}u L=180n",
            ]
        )
        deck.render_component(NodeParasitics(f"{self.name}_relu_parasitics", (self.pos_low_node, self.pos_mid_node)))


@dataclass
class SignedScoreErrorCell(Component):
    """Hardware-local signed error rails from target and signed score pairs.

    The small-signal target is:

    ``dp - dn ~= target_pos - target_neg + score_neg - score_pos``.

    This is a hardware-shaped surrogate for the class-local error used by the
    readout writer.  It intentionally produces rails, not a Python-computed
    scalar residual.
    """

    name: str
    target_node: str
    score_pos_node: str
    score_neg_node: str
    positive_error: CapState
    negative_error: CapState
    negative_target_node: str | None = None
    err_gate: str = "err"
    target_width_u: float = 32.0
    negative_target_width_u: float | None = None
    score_width_u: float = 24.0
    model: str = "NSENSE"

    @property
    def target_mid_node(self) -> str:
        return f"{self.name}_dp_t"

    @property
    def score_neg_mid_node(self) -> str:
        return f"{self.name}_dp_sn"

    @property
    def score_pos_mid_node(self) -> str:
        return f"{self.name}_dn_sp"

    @property
    def negative_target_mid_node(self) -> str:
        return f"{self.name}_dn_tn"

    def children(self) -> list[Component]:
        return [self.positive_error, self.negative_error]

    def input_nodes(self) -> tuple[str, ...]:
        nodes = [self.target_node]
        if self.negative_target_node is not None:
            nodes.append(self.negative_target_node)
        nodes.extend([self.score_pos_node, self.score_neg_node, self.err_gate])
        return tuple(nodes)

    def output_nodes(self) -> tuple[str, ...]:
        return (self.positive_error.node, self.negative_error.node)

    def state_nodes(self) -> tuple[str, ...]:
        return (
            *self.positive_error.state_nodes(),
            *self.negative_error.state_nodes(),
            self.target_mid_node,
            self.score_neg_mid_node,
            self.score_pos_mid_node,
            *((self.negative_target_mid_node,) if self.negative_target_node is not None else ()),
        )

    def validate(self) -> None:
        if not self.name:
            raise ValueError("SignedScoreErrorCell requires a nonempty name")
        for node in (
            self.target_node,
            self.score_pos_node,
            self.score_neg_node,
            self.err_gate,
            self.model,
        ):
            if not node:
                raise ValueError("SignedScoreErrorCell requires nonempty nodes")
        if self.negative_target_node is not None and not self.negative_target_node:
            raise ValueError("SignedScoreErrorCell requires nonempty negative target node")
        if self.target_width_u < 0 or self.score_width_u < 0:
            raise ValueError("SignedScoreErrorCell widths must be nonnegative")
        if self.negative_target_width_u is not None and self.negative_target_width_u < 0:
            raise ValueError("SignedScoreErrorCell negative target width must be nonnegative")
        self.positive_error.validate()
        self.negative_error.validate()

    def render(self, deck: NetlistBuilder) -> None:
        self.positive_error.render(deck)
        self.negative_error.render(deck)
        dp = self.positive_error.node
        dn = self.negative_error.node
        negative_target_width = (
            self.target_width_u if self.negative_target_width_u is None else self.negative_target_width_u
        )
        deck.extend(
            [
                f"* {self.name}: signed error dp-dn ~= target_pos - target_neg + score_neg - score_pos.",
            ]
        )
        parasitic_nodes = []
        if self.target_width_u > 0:
            deck.extend(
                [
                    f"M{self.name}_dp_t0 vdd {self.target_node} {self.target_mid_node} 0 {self.model} W={self.target_width_u:.12g}u L=180n",
                    f"M{self.name}_dp_t1 {self.target_mid_node} {self.err_gate} {dp} 0 {self.model} W={self.target_width_u:.12g}u L=180n",
                ]
            )
            parasitic_nodes.append(self.target_mid_node)
        if self.score_width_u > 0:
            deck.extend(
                [
                    f"M{self.name}_dp_sn0 vdd {self.score_neg_node} {self.score_neg_mid_node} 0 {self.model} W={self.score_width_u:.12g}u L=180n",
                    f"M{self.name}_dp_sn1 {self.score_neg_mid_node} {self.err_gate} {dp} 0 {self.model} W={self.score_width_u:.12g}u L=180n",
                    f"M{self.name}_dn_sp0 vdd {self.score_pos_node} {self.score_pos_mid_node} 0 {self.model} W={self.score_width_u:.12g}u L=180n",
                    f"M{self.name}_dn_sp1 {self.score_pos_mid_node} {self.err_gate} {dn} 0 {self.model} W={self.score_width_u:.12g}u L=180n",
                ]
            )
            parasitic_nodes.extend([self.score_neg_mid_node, self.score_pos_mid_node])
        if self.negative_target_node is not None and negative_target_width > 0:
            deck.extend(
                [
                    f"M{self.name}_dn_tn0 vdd {self.negative_target_node} {self.negative_target_mid_node} 0 {self.model} W={negative_target_width:.12g}u L=180n",
                    f"M{self.name}_dn_tn1 {self.negative_target_mid_node} {self.err_gate} {dn} 0 {self.model} W={negative_target_width:.12g}u L=180n",
                ]
            )
            parasitic_nodes.append(self.negative_target_mid_node)
        if parasitic_nodes:
            deck.render_component(
                NodeParasitics(
                    f"{self.name}_parasitics",
                    tuple(parasitic_nodes),
                )
            )


@dataclass
class SplitScoreCELimitedErrorBank(Component):
    """Multiclass split-score error bank with shared current competition.

    This is the reusable SPICENN version of the most useful hardware-native
    softmax surrogate from the older direct-flow decks.  It does not compute a
    textbook exponential softmax.  Instead each class score pair discharges an
    active-low ``ybar`` rail through a shared tail, then the local error rails
    approximate ``target - positive_average(score)``:

    * target rows charge ``dp`` while their class is not already winning;
    * non-target rows charge ``dn`` in proportion to positive score evidence;
    * target and non-target source rails are current-starved independently so
      one target pulse can be balanced against several non-target pulses.
    """

    name: str
    output_count: int
    target_node_prefix: str = "t"
    nontarget_node_prefix: str = "nt"
    score_pos_prefix: str = "score"
    score_neg_prefix: str = "score"
    score_pos_suffix: str = "p"
    score_neg_suffix: str = "n"
    positive_error_prefix: str = "dp"
    negative_error_prefix: str = "dn"
    err_gate: str = "err"
    reset_gate: str = "rste"
    target_source_node: str = "ctsrc"
    nontarget_source_node: str = "cesrc"
    compete_tail_node: str = "ccsrc"
    target_supply_node: str = "vdd"
    nontarget_supply_node: str = "vdd"
    error_cap_f: float = 6.0
    ybar_cap_f: float = 20.0
    target_source_cap_f: float = 2.0
    nontarget_source_cap_f: float = 0.2
    target_width_u: float = 32.0
    nontarget_width_u: float = 24.0
    source_width_u: float | None = None
    model: str = "NSENSE"
    pass_model: str = "NREL"

    def children(self) -> list[Component]:
        children: list[Component] = [
            CapState(
                self.target_source_node,
                self.target_source_node,
                self.target_source_cap_f,
                ic_v=0.0,
                leak_to="0",
                leak_ohm="1G",
                reset_gate=self.reset_gate,
            ),
            CapState(
                self.nontarget_source_node,
                self.nontarget_source_node,
                self.nontarget_source_cap_f,
                ic_v=0.0,
                leak_to="0",
                leak_ohm="1G",
                reset_gate=self.reset_gate,
            ),
        ]
        for out in range(self.output_count):
            children.extend(
                [
                    CapState(
                        f"{self.positive_error_prefix}{out}",
                        f"{self.positive_error_prefix}{out}",
                        self.error_cap_f,
                        ic_v=0.0,
                        leak_to="0",
                        leak_ohm="1G",
                    ),
                    CapState(
                        f"{self.negative_error_prefix}{out}",
                        f"{self.negative_error_prefix}{out}",
                        self.error_cap_f,
                        ic_v=0.0,
                        leak_to="0",
                        leak_ohm="1G",
                    ),
                    CapState(
                        f"ybar{out}",
                        f"ybar{out}",
                        self.ybar_cap_f,
                        ic_v=1.2,
                        leak_to="0",
                        leak_ohm="1G",
                        reset_gate=self.reset_gate,
                        reset_to="vdd",
                        reset_model=self.model,
                        reset_width_u=32.0,
                    ),
                ]
            )
        return children

    def input_nodes(self) -> tuple[str, ...]:
        nodes = [self.err_gate, self.reset_gate, self.target_supply_node, self.nontarget_supply_node]
        for out in range(self.output_count):
            nodes.extend(
                [
                    f"{self.target_node_prefix}{out}",
                    f"{self.nontarget_node_prefix}{out}",
                    f"{self.score_pos_prefix}{out}{self.score_pos_suffix}",
                    f"{self.score_neg_prefix}{out}{self.score_neg_suffix}",
                ]
            )
        return tuple(nodes)

    def output_nodes(self) -> tuple[str, ...]:
        return tuple(
            node
            for out in range(self.output_count)
            for node in (f"{self.positive_error_prefix}{out}", f"{self.negative_error_prefix}{out}")
        )

    def state_nodes(self) -> tuple[str, ...]:
        nodes = [self.target_source_node, self.nontarget_source_node, self.compete_tail_node]
        for out in range(self.output_count):
            nodes.extend(
                [
                    f"{self.positive_error_prefix}{out}",
                    f"{self.negative_error_prefix}{out}",
                    f"ybar{out}",
                    f"{self.name}_cc{out}_mid",
                    f"{self.name}_dp{out}_t",
                    f"{self.name}_dp{out}_yp",
                    f"{self.name}_dn{out}_nt",
                    f"{self.name}_dn{out}_inh",
                    f"{self.name}_dn{out}_score",
                ]
            )
        return tuple(nodes)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("SplitScoreCELimitedErrorBank requires a nonempty name")
        if self.output_count <= 0:
            raise ValueError("SplitScoreCELimitedErrorBank output_count must be positive")
        for node in (
            self.err_gate,
            self.reset_gate,
            self.target_source_node,
            self.nontarget_source_node,
            self.compete_tail_node,
            self.target_supply_node,
            self.nontarget_supply_node,
            self.model,
            self.pass_model,
        ):
            if not node:
                raise ValueError("SplitScoreCELimitedErrorBank requires nonempty nodes/models")
        for value_name, value in (
            ("error_cap_f", self.error_cap_f),
            ("ybar_cap_f", self.ybar_cap_f),
            ("target_source_cap_f", self.target_source_cap_f),
            ("nontarget_source_cap_f", self.nontarget_source_cap_f),
        ):
            if value <= 0:
                raise ValueError(f"SplitScoreCELimitedErrorBank {value_name} must be positive")
        if self.target_width_u <= 0 or self.nontarget_width_u <= 0:
            raise ValueError("SplitScoreCELimitedErrorBank widths must be positive")
        if self.source_width_u is not None and self.source_width_u <= 0:
            raise ValueError("SplitScoreCELimitedErrorBank source_width_u must be positive")

    def render(self, deck: NetlistBuilder) -> None:
        target_source_width = self.target_width_u if self.source_width_u is None else self.source_width_u
        nontarget_source_width = self.nontarget_width_u if self.source_width_u is None else self.source_width_u
        deck.extend(
            [
                (
                    f"* {self.name}: split-score current-limited CE surrogate. "
                    "Shared ybar competition approximates positive-average softmax probability."
                ),
            ]
        )
        for child in self.children():
            deck.render_component(child)
        deck.extend(
            [
                f"M{self.name}_ctsrc {self.target_supply_node} {self.err_gate} {self.target_source_node} 0 {self.model} W={target_source_width:.12g}u L=180n",
                f"M{self.name}_cesrc {self.nontarget_supply_node} {self.err_gate} {self.nontarget_source_node} 0 {self.model} W={nontarget_source_width:.12g}u L=180n",
                f"R{self.name}_{self.compete_tail_node} {self.compete_tail_node} 0 1e12",
                f"M{self.name}_cc_tail {self.compete_tail_node} {self.err_gate} 0 0 NMOS W={self.nontarget_width_u:.12g}u L=180n",
            ]
        )
        parasitic_nodes = [self.compete_tail_node]
        for out in range(self.output_count):
            score_pos = f"{self.score_pos_prefix}{out}{self.score_pos_suffix}"
            score_neg = f"{self.score_neg_prefix}{out}{self.score_neg_suffix}"
            target = f"{self.target_node_prefix}{out}"
            nontarget = f"{self.nontarget_node_prefix}{out}"
            dp = f"{self.positive_error_prefix}{out}"
            dn = f"{self.negative_error_prefix}{out}"
            ybar = f"ybar{out}"
            cc_mid = f"{self.name}_cc{out}_mid"
            dp_t = f"{self.name}_dp{out}_t"
            dp_yp = f"{self.name}_dp{out}_yp"
            dn_nt = f"{self.name}_dn{out}_nt"
            dn_inh = f"{self.name}_dn{out}_inh"
            dn_score = f"{self.name}_dn{out}_score"
            deck.extend(
                [
                    f"* {self.name} output {out}: target dp and score-competition non-target dn.",
                    f"M{self.name}_cc{out}_inh {cc_mid} {score_neg} {ybar} vdd PMOS W={self.nontarget_width_u:.12g}u L=180n",
                    f"M{self.name}_cc{out}_branch {cc_mid} {score_pos} {self.compete_tail_node} 0 {self.pass_model} W={self.nontarget_width_u:.12g}u L=180n",
                    f"M{self.name}_dp{out}_t0 {self.target_source_node} {target} {dp_t} 0 {self.model} W={self.target_width_u:.12g}u L=180n",
                    f"M{self.name}_dp{out}_yp0 {dp_t} {ybar} {dp_yp} 0 {self.model} W={self.target_width_u:.12g}u L=180n",
                    f"M{self.name}_dp{out}_err0 {dp_yp} {self.err_gate} {dp} 0 {self.model} W={self.target_width_u:.12g}u L=180n",
                    f"M{self.name}_dn{out}_nt0 {self.nontarget_source_node} {nontarget} {dn_nt} 0 {self.model} W={self.nontarget_width_u:.12g}u L=180n",
                    f"M{self.name}_dn{out}_inh0 {dn_nt} {score_neg} {dn_inh} vdd PMOS W={self.nontarget_width_u:.12g}u L=180n",
                    f"M{self.name}_dn{out}_score0 {dn_inh} {score_pos} {dn_score} 0 {self.pass_model} W={self.nontarget_width_u:.12g}u L=180n",
                    f"M{self.name}_dn{out}_err0 {dn_score} {self.err_gate} {dn} 0 {self.model} W={self.nontarget_width_u:.12g}u L=180n",
                ]
            )
            parasitic_nodes.extend([cc_mid, dp_t, dp_yp, dn_nt, dn_inh, dn_score])
        deck.render_component(NodeParasitics(f"{self.name}_parasitics", tuple(parasitic_nodes)))


@dataclass
class PreTraceCell(Component):
    """Per-synapse activation trace and optional full-swing eligibility gate."""

    name: str
    source_node: str
    mode: str
    cap_f: float
    consume_width_u: float
    store_width_u: float = 4.0
    boost_width_u: float = 4.0
    spike_width_u: float | None = None
    boosted_name: str | None = None
    spike_gate_name: str | None = None
    spike_bar_name: str | None = None
    spike_mid_name: str | None = None
    spike_ref_node: str = "spikeref"
    spike_model: str = "NSENSE"
    fwd_gate: str = "fwd"
    reset_gate: str = "rstf"
    backward_gate: str = "bwd"
    boost_node: str = "preboost"

    @property
    def trace_node(self) -> str:
        return self.name

    @property
    def boosted_node(self) -> str:
        return self.boosted_name or f"{self.name}b"

    @property
    def spike_gate_node(self) -> str:
        return self.spike_gate_name or f"{self.name}g"

    @property
    def spike_bar_node(self) -> str:
        return self.spike_bar_name or f"{self.name}bar"

    @property
    def spike_mid_node(self) -> str:
        return self.spike_mid_name or f"{self.name}m"

    def children(self) -> list[Component]:
        return []

    def input_nodes(self) -> tuple[str, ...]:
        return (self.source_node,)

    def state_nodes(self) -> tuple[str, ...]:
        nodes = [self.trace_node]
        if self.mode == "synapse_boost":
            nodes.append(self.boosted_node)
        if self.mode == "synapse_spike":
            nodes.extend([self.spike_gate_node, self.spike_bar_node])
        return tuple(nodes)

    def validate(self) -> None:
        if self.mode not in {"synapse_gate", "synapse_consume", "synapse_boost", "synapse_spike"}:
            raise ValueError(f"unknown pretrace mode: {self.mode}")
        if self.cap_f <= 0:
            raise ValueError("pretrace capacitance must be positive")
        if self.consume_width_u <= 0:
            raise ValueError("pretrace consume width must be positive")
        if self.store_width_u <= 0 or self.boost_width_u <= 0:
            raise ValueError("pretrace store/boost widths must be positive")
        if self.spike_width_u is not None and self.spike_width_u <= 0:
            raise ValueError("pretrace spike width must be positive")
        if not self.spike_model:
            raise ValueError("pretrace spike model must be nonempty")

    def render(self, deck: NetlistBuilder) -> None:
        CapState(
            name=self.trace_node,
            node=self.trace_node,
            cap_f=self.cap_f,
            ic_v=0,
            leak_to="0",
            leak_ohm="1G",
            reset_gate=self.reset_gate,
        ).render(deck)
        deck.mos(
            f"store_{self.trace_node}",
            self.trace_node,
            self.fwd_gate,
            self.source_node,
            "0",
            "NREL",
            width_u=self.store_width_u,
        )
        if self.mode == "synapse_consume":
            deck.mos(
                f"consume_{self.trace_node}",
                self.trace_node,
                self.backward_gate,
                "0",
                "0",
                "NREL",
                width_u=self.consume_width_u,
            )
        if self.mode == "synapse_boost":
            boosted = self.boosted_node
            CapState(
                name=boosted,
                node=boosted,
                cap_f=self.cap_f,
                ic_v=0,
                leak_to="0",
                leak_ohm="1G",
                reset_gate=self.reset_gate,
            ).render(deck)
            deck.capacitor(f"boost_{boosted}", self.boost_node, boosted, f"{self.cap_f:.12g}f")
            deck.mos(
                f"store_{boosted}",
                boosted,
                self.fwd_gate,
                self.source_node,
                "0",
                "NREL",
                width_u=self.boost_width_u,
            )
        if self.mode == "synapse_spike":
            width_u = self.boost_width_u if self.spike_width_u is None else self.spike_width_u
            gate = self.spike_gate_node
            bar = self.spike_bar_node
            mid = self.spike_mid_node
            CapState(
                name=gate,
                node=gate,
                cap_f=self.cap_f,
                ic_v=0,
                leak_to="0",
                leak_ohm="1G",
                reset_gate=self.reset_gate,
            ).render(deck)
            CapState(
                name=bar,
                node=bar,
                cap_f=self.cap_f,
                ic_v=1.2,
                leak_to="vdd",
                leak_ohm="1G",
                reset_gate=self.reset_gate,
                reset_to="vdd",
                reset_model="NSENSE",
            ).render(deck)
            deck.mos(f"spike_{bar}_fwd", bar, self.fwd_gate, mid, "0", "NREL", width_u=width_u)
            deck.mos(
                f"spike_{bar}_act",
                mid,
                self.source_node,
                self.spike_ref_node,
                "0",
                self.spike_model,
                width_u=width_u,
            )
            deck.mos(f"spike_{gate}_p", "vdd", bar, gate, "vdd", "PMOS", width_u=width_u)
            deck.mos(f"spike_{gate}_n", gate, bar, "0", "0", "NMOS", width_u=width_u)


@dataclass
class PreTraceArray(Component):
    """Topology-indexed collection of per-synapse activation traces."""

    name: str
    cells: tuple[PreTraceCell, ...]
    comment: str | None = None

    @classmethod
    def from_readout_topology(
        cls,
        name: str,
        topology: FanInTopology,
        *,
        mode: str,
        cap_f: float,
        consume_width_u: float,
        boost_width_u: float = 4.0,
        spike_ref_node: str = "spikeref",
    ) -> "PreTraceArray":
        cells: list[PreTraceCell] = []
        for out, hidden_ids in topology.as_fanins().items():
            for hidden in hidden_ids:
                h = int(hidden)
                cells.append(
                    PreTraceCell(
                        name=f"fpro{out}{h}",
                        source_node=f"act{h}",
                        mode=mode,
                        cap_f=cap_f,
                        consume_width_u=consume_width_u,
                        boost_width_u=boost_width_u,
                        boosted_name=f"fprb{out}{h}",
                        spike_gate_name=f"fprg{out}{h}",
                        spike_bar_name=f"fprbar{out}{h}",
                        spike_mid_name=f"fprm{out}{h}",
                        spike_ref_node=spike_ref_node,
                    )
                )
        return cls(name=name, cells=tuple(cells))

    @classmethod
    def from_hidden_topology(
        cls,
        name: str,
        topology: FanInTopology,
        *,
        mode: str,
        cap_f: float,
        consume_width_u: float,
        boost_width_u: float = 4.0,
        spike_ref_node: str = "spikeref",
    ) -> "PreTraceArray":
        cells: list[PreTraceCell] = []
        for hidden, rails in topology.as_fanins().items():
            for rail in rails:
                rail_name = str(rail)
                cells.append(
                    PreTraceCell(
                        name=f"fphi{hidden}_{rail_name}",
                        source_node=rail_name,
                        mode=mode,
                        cap_f=cap_f,
                        consume_width_u=consume_width_u,
                        boost_width_u=boost_width_u,
                        boosted_name=f"fphib{hidden}_{rail_name}",
                        spike_gate_name=f"fphig{hidden}_{rail_name}",
                        spike_bar_name=f"fphibar{hidden}_{rail_name}",
                        spike_mid_name=f"fphim{hidden}_{rail_name}",
                        spike_ref_node=spike_ref_node,
                    )
                )
        return cls(name=name, cells=tuple(cells))

    def children(self) -> list[Component]:
        return list(self.cells)

    def input_nodes(self) -> tuple[str, ...]:
        return tuple(cell.source_node for cell in self.cells)

    def state_nodes(self) -> tuple[str, ...]:
        return tuple(node for cell in self.cells for node in cell.state_nodes())

    def spike_mid_nodes(self) -> tuple[str, ...]:
        return tuple(cell.spike_mid_node for cell in self.cells if cell.mode == "synapse_spike")

    def validate(self) -> None:
        if not self.name:
            raise ValueError("PreTraceArray requires a nonempty name")
        for cell in self.cells:
            cell.validate()

    def render(self, deck: NetlistBuilder) -> None:
        if self.comment is not None:
            deck.comment(self.comment)
        for cell in self.cells:
            deck.render_component(cell)


@dataclass
class NodeParasitics(Component):
    """Weak RC anchors for otherwise-floating internal MOS stack nodes."""

    name: str
    nodes: tuple[str, ...]
    resistor_ohm: str | float = "1e9"
    cap_f: float = 0.02
    ic_v: float = 0.0
    anchor_node: str = "0"

    def children(self) -> list[Component]:
        return []

    def state_nodes(self) -> tuple[str, ...]:
        return self.nodes

    def validate(self) -> None:
        if not self.name:
            raise ValueError("NodeParasitics requires a nonempty name")
        if any(not node for node in self.nodes):
            raise ValueError("NodeParasitics requires nonempty node names")
        if not self.anchor_node:
            raise ValueError("NodeParasitics requires a nonempty anchor node")
        if self.cap_f <= 0:
            raise ValueError("NodeParasitics capacitance must be positive")

    def render(self, deck: NetlistBuilder) -> None:
        for node in self.nodes:
            deck.resistor(f"par_{node}", node, self.anchor_node, self.resistor_ohm)
            deck.capacitor(f"par_{node}", node, "0", f"{self.cap_f:.12g}f", ic=self.ic_v)


@dataclass
class DiffPairBleedWriteSelector(Component):
    """High-true write rails generated by a differential error comparison.

    This is the reusable version of the direct-flow ``diffpair_bleed`` selector:
    common-mode error drives both comparison nodes similarly, and the bwd-gated
    bleeds keep both write rails low unless one side wins strongly.
    """

    name: str
    positive_error_gate: str
    negative_error_gate: str
    positive_write_gate: str
    negative_write_gate: str
    width_u: float
    label: str
    reset_gate: str = "rste"
    backward_gate: str = "bwd"

    @property
    def positive_bar_node(self) -> str:
        return f"{self.name}_posbar"

    @property
    def negative_bar_node(self) -> str:
        return f"{self.name}_negbar"

    @property
    def positive_mid_node(self) -> str:
        return f"{self.name}_posmid"

    @property
    def negative_mid_node(self) -> str:
        return f"{self.name}_negmid"

    @property
    def source_node(self) -> str:
        return f"{self.name}_src"

    def children(self) -> list[Component]:
        return []

    def input_nodes(self) -> tuple[str, ...]:
        return (self.positive_error_gate, self.negative_error_gate)

    def output_nodes(self) -> tuple[str, ...]:
        return (self.positive_write_gate, self.negative_write_gate)

    def state_nodes(self) -> tuple[str, ...]:
        return (
            self.positive_write_gate,
            self.negative_write_gate,
            self.positive_bar_node,
            self.negative_bar_node,
            self.positive_mid_node,
            self.negative_mid_node,
            self.source_node,
        )

    def validate(self) -> None:
        if not self.name:
            raise ValueError("DiffPairBleedWriteSelector requires a nonempty name")
        for node in (
            self.positive_error_gate,
            self.negative_error_gate,
            self.positive_write_gate,
            self.negative_write_gate,
            self.reset_gate,
            self.backward_gate,
        ):
            if not node:
                raise ValueError("DiffPairBleedWriteSelector requires nonempty nodes")
        if self.width_u < 0:
            raise ValueError("DiffPairBleedWriteSelector width must be nonnegative")

    def render(self, deck: NetlistBuilder) -> None:
        tail_width_u = max(self.width_u * 0.5, 1e-9)
        bleed_width_u = max(self.width_u * 0.025, 1e-9)
        pos_bar = self.positive_bar_node
        neg_bar = self.negative_bar_node
        pos_mid = self.positive_mid_node
        neg_mid = self.negative_mid_node
        src = self.source_node
        prefix = self.name
        deck.extend(
            [
                f"* Differential-pair write selector for {self.label}: shared-tail dp/dn comparison with weak bwd bleed.",
                f"C{prefix}_pos {self.positive_write_gate} 0 0.1f IC=0",
                f"C{prefix}_neg {self.negative_write_gate} 0 0.1f IC=0",
                f"R{prefix}_pos {self.positive_write_gate} 0 1G",
                f"R{prefix}_neg {self.negative_write_gate} 0 1G",
                f"Mreset_{prefix}_pos {self.positive_write_gate} {self.reset_gate} 0 0 NMOS W=4u L=180n",
                f"Mreset_{prefix}_neg {self.negative_write_gate} {self.reset_gate} 0 0 NMOS W=4u L=180n",
                f"M{prefix}_pos_bleed {self.positive_write_gate} {self.backward_gate} 0 0 NMOS W={bleed_width_u:.12g}u L=180n",
                f"M{prefix}_neg_bleed {self.negative_write_gate} {self.backward_gate} 0 0 NMOS W={bleed_width_u:.12g}u L=180n",
                f"C{prefix}_posbar {pos_bar} 0 0.05f IC=1.2",
                f"C{prefix}_negbar {neg_bar} 0 0.05f IC=1.2",
                f"R{prefix}_posbar {pos_bar} vdd 1G",
                f"R{prefix}_negbar {neg_bar} vdd 1G",
                f"M{prefix}_pos_load {pos_bar} {pos_bar} vdd vdd PMOS W={self.width_u:.12g}u L=180n",
                f"M{prefix}_neg_load {neg_bar} {neg_bar} vdd vdd PMOS W={self.width_u:.12g}u L=180n",
                f"M{prefix}_pos_sel {pos_bar} {self.positive_error_gate} {src} 0 NSENSE W={self.width_u:.12g}u L=180n",
                f"M{prefix}_neg_sel {neg_bar} {self.negative_error_gate} {src} 0 NSENSE W={self.width_u:.12g}u L=180n",
                f"M{prefix}_tail {src} {self.backward_gate} 0 0 NMOS W={tail_width_u:.12g}u L=180n",
                f"M{prefix}_pos_p vdd {pos_bar} {pos_mid} vdd PMOS W={self.width_u:.12g}u L=180n",
                f"M{prefix}_pos_n {pos_mid} {neg_bar} {self.positive_write_gate} 0 NMOS W={self.width_u:.12g}u L=180n",
                f"M{prefix}_neg_p vdd {neg_bar} {neg_mid} vdd PMOS W={self.width_u:.12g}u L=180n",
                f"M{prefix}_neg_n {neg_mid} {pos_bar} {self.negative_write_gate} 0 NMOS W={self.width_u:.12g}u L=180n",
                f"C{prefix}_posmid {pos_mid} 0 0.02f IC=0",
                f"C{prefix}_negmid {neg_mid} 0 0.02f IC=0",
                f"R{prefix}_posmid {pos_mid} 0 1G",
                f"R{prefix}_negmid {neg_mid} 0 1G",
                f"C{prefix}_src {src} 0 0.02f IC=0",
                f"R{prefix}_src {src} 0 1G",
            ]
        )


@dataclass
class MutuallyInhibitedWriteSelector(Component):
    """High-true write rails with explicit opposite-rail inhibition.

    ``positive_write_gate`` approximates ``positive_error AND NOT negative_error``;
    ``negative_write_gate`` approximates ``negative_error AND NOT positive_error``.
    This costs less than the differential-pair selector and can produce more
    separated write rails when both input rails have substantial common-mode.
    """

    name: str
    positive_error_gate: str
    negative_error_gate: str
    positive_write_gate: str
    negative_write_gate: str
    width_u: float
    label: str
    reset_gate: str = "rste"
    gate_model: str = "NSENSE"
    kill_model: str = "NMOS"

    @property
    def positive_source_node(self) -> str:
        return f"{self.name}_possrc"

    @property
    def negative_source_node(self) -> str:
        return f"{self.name}_negsrc"

    def children(self) -> list[Component]:
        return []

    def input_nodes(self) -> tuple[str, ...]:
        return (self.positive_error_gate, self.negative_error_gate)

    def output_nodes(self) -> tuple[str, ...]:
        return (self.positive_write_gate, self.negative_write_gate)

    def state_nodes(self) -> tuple[str, ...]:
        return (
            self.positive_write_gate,
            self.negative_write_gate,
            self.positive_source_node,
            self.negative_source_node,
        )

    def validate(self) -> None:
        if not self.name:
            raise ValueError("MutuallyInhibitedWriteSelector requires a nonempty name")
        for node in (
            self.positive_error_gate,
            self.negative_error_gate,
            self.positive_write_gate,
            self.negative_write_gate,
            self.reset_gate,
            self.gate_model,
            self.kill_model,
        ):
            if not node:
                raise ValueError("MutuallyInhibitedWriteSelector requires nonempty nodes")
        if self.width_u < 0:
            raise ValueError("MutuallyInhibitedWriteSelector width must be nonnegative")

    def render(self, deck: NetlistBuilder) -> None:
        psrc = self.positive_source_node
        nsrc = self.negative_source_node
        prefix = self.name
        deck.extend(
            [
                f"* Mutually inhibited write selector for {self.label}: pos~=dp&~dn, neg~=dn&~dp.",
                f"C{prefix}_pos {self.positive_write_gate} 0 0.1f IC=0",
                f"C{prefix}_neg {self.negative_write_gate} 0 0.1f IC=0",
                f"R{prefix}_pos {self.positive_write_gate} 0 1G",
                f"R{prefix}_neg {self.negative_write_gate} 0 1G",
                f"Mreset_{prefix}_pos {self.positive_write_gate} {self.reset_gate} 0 0 NMOS W=4u L=180n",
                f"Mreset_{prefix}_neg {self.negative_write_gate} {self.reset_gate} 0 0 NMOS W=4u L=180n",
                f"M{prefix}_pos_inh vdd {self.negative_error_gate} {psrc} vdd PMOS W={self.width_u:.12g}u L=180n",
                f"M{prefix}_pos_gate {psrc} {self.positive_error_gate} {self.positive_write_gate} 0 {self.gate_model} W={self.width_u:.12g}u L=180n",
                f"M{prefix}_neg_inh vdd {self.positive_error_gate} {nsrc} vdd PMOS W={self.width_u:.12g}u L=180n",
                f"M{prefix}_neg_gate {nsrc} {self.negative_error_gate} {self.negative_write_gate} 0 {self.gate_model} W={self.width_u:.12g}u L=180n",
                f"M{prefix}_pos_kill {self.positive_write_gate} {self.negative_error_gate} 0 0 {self.kill_model} W={self.width_u:.12g}u L=180n",
                f"M{prefix}_neg_kill {self.negative_write_gate} {self.positive_error_gate} 0 0 {self.kill_model} W={self.width_u:.12g}u L=180n",
            ]
        )
        deck.render_component(NodeParasitics(f"{self.name}_parasitics", (psrc, nsrc)))


@dataclass
class RegenerativeDifferentialWriteSelector(Component):
    """Dynamic differential selector for tiny signed delta rails.

    The outputs are precharged high and then discharged competitively during
    the backward/write window.  ``positive_write_gate`` remains high when the
    positive delta rail beats the negative rail; ``negative_write_gate`` remains
    high for the opposite sign.  The two outputs can also be used as each
    other's low-true rails by CMOS complementary write cells.
    """

    name: str
    positive_error_gate: str
    negative_error_gate: str
    positive_write_gate: str
    negative_write_gate: str
    width_u: float
    label: str
    sense_gate: str = "bwd"
    reset_gate: str = "rste"
    cap_f: float = 0.1
    keeper_width_u: float | None = None
    nkeeper_width_u: float = 0.0
    model: str = "NSENSE"

    @property
    def positive_discharge_node(self) -> str:
        return f"{self.name}_posdn"

    @property
    def negative_discharge_node(self) -> str:
        return f"{self.name}_negdn"

    @property
    def activity_gate_node(self) -> str:
        return f"{self.name}_active"

    @property
    def positive_activity_mid_node(self) -> str:
        return f"{self.name}_actp"

    @property
    def positive_activity_sense_mid_node(self) -> str:
        return f"{self.name}_actp_sense"

    @property
    def negative_activity_mid_node(self) -> str:
        return f"{self.name}_actn"

    @property
    def negative_activity_sense_mid_node(self) -> str:
        return f"{self.name}_actn_sense"

    def children(self) -> list[Component]:
        return []

    def input_nodes(self) -> tuple[str, ...]:
        return (self.positive_error_gate, self.negative_error_gate, self.sense_gate)

    def output_nodes(self) -> tuple[str, ...]:
        return (self.positive_write_gate, self.negative_write_gate)

    def state_nodes(self) -> tuple[str, ...]:
        return (
            self.positive_write_gate,
            self.negative_write_gate,
            self.positive_discharge_node,
            self.negative_discharge_node,
            self.activity_gate_node,
            self.positive_activity_mid_node,
            self.positive_activity_sense_mid_node,
            self.negative_activity_mid_node,
            self.negative_activity_sense_mid_node,
        )

    def validate(self) -> None:
        if not self.name:
            raise ValueError("RegenerativeDifferentialWriteSelector requires a nonempty name")
        for node in (
            self.positive_error_gate,
            self.negative_error_gate,
            self.positive_write_gate,
            self.negative_write_gate,
            self.sense_gate,
            self.reset_gate,
            self.model,
        ):
            if not node:
                raise ValueError("RegenerativeDifferentialWriteSelector requires nonempty nodes")
        if self.width_u <= 0:
            raise ValueError("RegenerativeDifferentialWriteSelector width must be positive")
        if self.cap_f <= 0:
            raise ValueError("RegenerativeDifferentialWriteSelector capacitance must be positive")
        if self.keeper_width_u is not None and self.keeper_width_u < 0:
            raise ValueError("RegenerativeDifferentialWriteSelector keeper width must be nonnegative")
        if self.nkeeper_width_u < 0:
            raise ValueError("RegenerativeDifferentialWriteSelector n-keeper width must be nonnegative")

    def render(self, deck: NetlistBuilder) -> None:
        pos = self.positive_write_gate
        neg = self.negative_write_gate
        posdn = self.positive_discharge_node
        negdn = self.negative_discharge_node
        keeper_width = max(1.0, self.width_u / 64.0) if self.keeper_width_u is None else self.keeper_width_u
        prefix = self.name
        deck.extend(
            [
                f"* Regenerative differential write selector for {self.label}: high rail wins the signed delta comparison.",
                f"C{prefix}_pos {pos} 0 {self.cap_f:.12g}f IC=1.2",
                f"C{prefix}_neg {neg} 0 {self.cap_f:.12g}f IC=1.2",
                f"R{prefix}_pos {pos} vdd 1G",
                f"R{prefix}_neg {neg} vdd 1G",
                f"Mreset_{prefix}_pos vdd {self.reset_gate} {pos} 0 {self.model} W=32u L=180n",
                f"Mreset_{prefix}_neg vdd {self.reset_gate} {neg} 0 {self.model} W=32u L=180n",
                f"M{prefix}_pos_dis_s {pos} {self.negative_error_gate} {posdn} 0 {self.model} W={self.width_u:.12g}u L=180n",
                f"M{prefix}_pos_dis_e {posdn} {self.sense_gate} 0 0 {self.model} W={self.width_u:.12g}u L=180n",
                f"M{prefix}_neg_dis_s {neg} {self.positive_error_gate} {negdn} 0 {self.model} W={self.width_u:.12g}u L=180n",
                f"M{prefix}_neg_dis_e {negdn} {self.sense_gate} 0 0 {self.model} W={self.width_u:.12g}u L=180n",
                f"M{prefix}_pos_keep {pos} {neg} vdd vdd PMOS W={keeper_width:.12g}u L=180n",
                f"M{prefix}_neg_keep {neg} {pos} vdd vdd PMOS W={keeper_width:.12g}u L=180n",
                f"C{prefix}_active {self.activity_gate_node} 0 {self.cap_f:.12g}f IC=0",
                f"R{prefix}_active {self.activity_gate_node} 0 1G",
                f"Mreset_{prefix}_active {self.activity_gate_node} {self.reset_gate} 0 0 NMOS W=4u L=180n",
                f"M{prefix}_actp_low {self.positive_activity_mid_node} {neg} vdd vdd PMOS W={keeper_width:.12g}u L=180n",
                f"M{prefix}_actp_high {self.positive_activity_mid_node} {pos} {self.positive_activity_sense_mid_node} 0 {self.model} W={self.width_u:.12g}u L=180n",
                f"M{prefix}_actp_bwd {self.positive_activity_sense_mid_node} {self.sense_gate} {self.activity_gate_node} 0 {self.model} W={self.width_u:.12g}u L=180n",
                f"M{prefix}_actn_low {self.negative_activity_mid_node} {pos} vdd vdd PMOS W={keeper_width:.12g}u L=180n",
                f"M{prefix}_actn_high {self.negative_activity_mid_node} {neg} {self.negative_activity_sense_mid_node} 0 {self.model} W={self.width_u:.12g}u L=180n",
                f"M{prefix}_actn_bwd {self.negative_activity_sense_mid_node} {self.sense_gate} {self.activity_gate_node} 0 {self.model} W={self.width_u:.12g}u L=180n",
            ]
        )
        if self.nkeeper_width_u > 0:
            deck.extend(
                [
                    f"M{prefix}_pos_nkeep {pos} {neg} 0 0 NMOS W={self.nkeeper_width_u:.12g}u L=180n",
                    f"M{prefix}_neg_nkeep {neg} {pos} 0 0 NMOS W={self.nkeeper_width_u:.12g}u L=180n",
                ]
            )
        deck.render_component(
            NodeParasitics(
                f"{self.name}_parasitics",
                (
                    posdn,
                    negdn,
                    self.positive_activity_mid_node,
                    self.positive_activity_sense_mid_node,
                    self.negative_activity_mid_node,
                    self.negative_activity_sense_mid_node,
                ),
            )
        )


@dataclass
class DirectFlowWeightCell(Component):
    """Local direct-flow writer for one signed capacitor-backed weight.

    The cell is intentionally close to the transistor-level update primitive:
    a stored pre-synaptic trace gates backward/write current, while separate
    positive and negative error rails choose which signed branch moves.
    """

    name: str
    pos_weight_node: str
    neg_weight_node: str
    pre_gate: str
    positive_write_gate: str
    negative_write_gate: str
    pos_discharge_width_u: float = 0.0
    neg_discharge_width_u: float = 0.0
    pos_charge_width_u: float = 0.0
    neg_charge_width_u: float = 0.0
    positive_gate_neg_discharge_width_u: float | None = None
    negative_gate_pos_discharge_width_u: float | None = None
    positive_gate_pos_charge_width_u: float | None = None
    negative_gate_neg_charge_width_u: float | None = None
    pos_high_node: str = "whigh"
    pos_low_node: str = "wlow"
    neg_high_node: str = "whigh"
    neg_low_node: str = "wlow"
    backward_gate: str = "bwd"
    write_gate_device: str = "NSENSE"
    discharge_enabled: bool = True
    charge_enabled: bool = False
    state_gate_discharge: bool = False
    state_gate_charge: bool = False
    pmos_charge_write: bool = False
    cmos_complementary_charge: bool = False
    gate_cmos_charge_with_backward: bool = False
    positive_write_gate_low_true: str | None = None
    negative_write_gate_low_true: str | None = None
    pre_gate_low_true: str | None = None
    charge_extra_gate: str | None = None
    overlap_write_gate: str | None = None
    center_pull_width_u: float = 0.0
    center_pull_gate: str = "bwd"
    center_pull_mode: str = "always"
    pos_center_pull_node: str = "wcenter"
    neg_center_pull_node: str = "wcenter"
    vdd_node: str = "vdd"

    def children(self) -> list[Component]:
        return []

    def input_nodes(self) -> tuple[str, ...]:
        nodes = [
            self.pre_gate,
            self.positive_write_gate,
            self.negative_write_gate,
            self.backward_gate,
        ]
        if self.positive_write_gate_low_true is not None:
            nodes.append(self.positive_write_gate_low_true)
        if self.negative_write_gate_low_true is not None:
            nodes.append(self.negative_write_gate_low_true)
        if self.pre_gate_low_true is not None:
            nodes.append(self.pre_gate_low_true)
        if self.charge_extra_gate is not None:
            nodes.append(self.charge_extra_gate)
        if self.overlap_write_gate is not None:
            nodes.append(self.overlap_write_gate)
        if self.center_pull_width_u > 0:
            nodes.append(self.center_pull_gate)
        return tuple(nodes)

    def state_nodes(self) -> tuple[str, ...]:
        nodes: list[str] = [self.pos_weight_node, self.neg_weight_node]
        nodes.extend(self._discharge_internal_nodes())
        nodes.extend(self._charge_internal_nodes())
        nodes.extend(self._pmos_charge_internal_nodes())
        nodes.extend(self._cmos_complementary_charge_internal_nodes())
        nodes.extend(self._overlap_internal_nodes())
        nodes.extend(self._center_internal_nodes())
        return tuple(dict.fromkeys(nodes))

    def validate(self) -> None:
        if not self.name:
            raise ValueError("DirectFlowWeightCell requires a nonempty name")
        for node in (
            self.pos_weight_node,
            self.neg_weight_node,
            self.pre_gate,
            self.positive_write_gate,
            self.negative_write_gate,
            self.backward_gate,
            self.write_gate_device,
        ):
            if not node:
                raise ValueError("DirectFlowWeightCell requires nonempty nodes")
        if self.center_pull_mode not in {"always", "state_high"}:
            raise ValueError(f"unknown center pull mode: {self.center_pull_mode}")
        for value in (
            self.pos_discharge_width_u,
            self.neg_discharge_width_u,
            self.pos_charge_width_u,
            self.neg_charge_width_u,
            self.center_pull_width_u,
        ):
            if value < 0:
                raise ValueError("DirectFlowWeightCell widths must be nonnegative")
        for value in (
            self.positive_gate_neg_discharge_width_u,
            self.negative_gate_pos_discharge_width_u,
            self.positive_gate_pos_charge_width_u,
            self.negative_gate_neg_charge_width_u,
        ):
            if value is not None and value < 0:
                raise ValueError("DirectFlowWeightCell gate widths must be nonnegative")
        if self.pmos_charge_write and (
            self.positive_write_gate_low_true is None or self.negative_write_gate_low_true is None
        ):
            raise ValueError("PMOS charge write requires low-true write gates")
        if self.cmos_complementary_charge and (
            self.positive_write_gate_low_true is None
            or self.negative_write_gate_low_true is None
            or self.pre_gate_low_true is None
        ):
            raise ValueError("CMOS complementary charge write requires low-true write and pretrace gates")

    def render(self, deck: NetlistBuilder) -> None:
        if self.discharge_enabled and (self.pos_discharge_width_u > 0 or self.neg_discharge_width_u > 0):
            self._render_discharge(deck)
        if self.charge_enabled and (self.pos_charge_width_u > 0 or self.neg_charge_width_u > 0):
            self._render_charge(deck)
        if self.pmos_charge_write and (self.pos_charge_width_u > 0 or self.neg_charge_width_u > 0):
            self._render_pmos_charge(deck)
        if self.cmos_complementary_charge and (self.pos_charge_width_u > 0 or self.neg_charge_width_u > 0):
            self._render_cmos_complementary_charge(deck)
        if (
            self.overlap_write_gate is not None
            and self.discharge_enabled
            and (self.pos_discharge_width_u > 0 or self.neg_discharge_width_u > 0)
        ):
            self._render_overlap(deck)
        if self.center_pull_width_u > 0:
            self._render_center_pull(deck)

    def _gate_neg_discharge_width(self) -> float:
        return (
            self.neg_discharge_width_u
            if self.positive_gate_neg_discharge_width_u is None
            else self.positive_gate_neg_discharge_width_u
        )

    def _gate_pos_discharge_width(self) -> float:
        return (
            self.pos_discharge_width_u
            if self.negative_gate_pos_discharge_width_u is None
            else self.negative_gate_pos_discharge_width_u
        )

    def _gate_pos_charge_width(self) -> float:
        return (
            self.pos_charge_width_u
            if self.positive_gate_pos_charge_width_u is None
            else self.positive_gate_pos_charge_width_u
        )

    def _gate_neg_charge_width(self) -> float:
        return (
            self.neg_charge_width_u
            if self.negative_gate_neg_charge_width_u is None
            else self.negative_gate_neg_charge_width_u
        )

    def _discharge_internal_nodes(self) -> list[str]:
        if not self.discharge_enabled or (self.pos_discharge_width_u <= 0 and self.neg_discharge_width_u <= 0):
            return []
        if self.state_gate_discharge:
            return [
                f"{self.neg_weight_node}_flow_s",
                f"{self.neg_weight_node}_flow_b",
                f"{self.neg_weight_node}_flow_a",
                f"{self.pos_weight_node}_flow_s",
                f"{self.pos_weight_node}_flow_b",
                f"{self.pos_weight_node}_flow_a",
            ]
        return [
            f"{self.neg_weight_node}_flow_b",
            f"{self.neg_weight_node}_flow_a",
            f"{self.pos_weight_node}_flow_b",
            f"{self.pos_weight_node}_flow_a",
        ]

    def _charge_internal_nodes(self) -> list[str]:
        if not self.charge_enabled or (self.pos_charge_width_u <= 0 and self.neg_charge_width_u <= 0):
            return []
        if self.state_gate_charge:
            return [
                f"{self.pos_weight_node}_ch_s",
                f"{self.pos_weight_node}_ch_b",
                f"{self.pos_weight_node}_ch_a",
                f"{self.neg_weight_node}_ch_s",
                f"{self.neg_weight_node}_ch_b",
                f"{self.neg_weight_node}_ch_a",
            ]
        return [
            f"{self.pos_weight_node}_ch_b",
            f"{self.pos_weight_node}_ch_a",
            f"{self.neg_weight_node}_ch_b",
            f"{self.neg_weight_node}_ch_a",
        ]

    def _pmos_charge_internal_nodes(self) -> list[str]:
        if not self.pmos_charge_write or (self.pos_charge_width_u <= 0 and self.neg_charge_width_u <= 0):
            return []
        return [
            f"{self.pos_weight_node}_pch_b",
            f"{self.pos_weight_node}_pch_g",
            f"{self.neg_weight_node}_pch_b",
            f"{self.neg_weight_node}_pch_g",
        ] + (
            [f"{self.pos_weight_node}_pch_x", f"{self.neg_weight_node}_pch_x"]
            if self.charge_extra_gate is not None
            else []
        )

    def _cmos_complementary_charge_internal_nodes(self) -> list[str]:
        if (
            not self.cmos_complementary_charge
            or (self.pos_charge_width_u <= 0 and self.neg_charge_width_u <= 0)
        ):
            return []
        return [
            f"{self.pos_weight_node}_cch_w",
            f"{self.neg_weight_node}_cch_w",
        ] + (
            [
                f"{self.pos_weight_node}_cch_b",
                f"{self.neg_weight_node}_cch_b",
            ]
            if self.gate_cmos_charge_with_backward
            else []
        )

    def _overlap_internal_nodes(self) -> list[str]:
        if (
            self.overlap_write_gate is None
            or not self.discharge_enabled
            or (self.pos_discharge_width_u <= 0 and self.neg_discharge_width_u <= 0)
        ):
            return []
        if self.state_gate_discharge:
            return [
                f"{self.pos_weight_node}_ov_s",
                f"{self.pos_weight_node}_ov_b",
                f"{self.pos_weight_node}_ov_a",
                f"{self.neg_weight_node}_ov_s",
                f"{self.neg_weight_node}_ov_b",
                f"{self.neg_weight_node}_ov_a",
            ]
        return [
            f"{self.pos_weight_node}_ov_b",
            f"{self.pos_weight_node}_ov_a",
            f"{self.neg_weight_node}_ov_b",
            f"{self.neg_weight_node}_ov_a",
        ]

    def _center_internal_nodes(self) -> list[str]:
        if self.center_pull_width_u <= 0 or self.center_pull_mode == "always":
            return []
        return [f"{self.pos_weight_node}_center_g", f"{self.neg_weight_node}_center_g"]

    def _render_discharge(self, deck: NetlistBuilder) -> None:
        p = self.pos_weight_node
        n = self.neg_weight_node
        if self.state_gate_discharge:
            deck.extend(
                [
                    f"M{n}_flow_s {n} {n} {n}_flow_s 0 NREL W={self.neg_discharge_width_u:.12g}u L=180n",
                    f"M{n}_flow_b {n}_flow_s {self.backward_gate} {n}_flow_b 0 NREL W={self.neg_discharge_width_u:.12g}u L=180n",
                    f"M{n}_flow_a {n}_flow_b {self.pre_gate} {n}_flow_a 0 NREL W={self.neg_discharge_width_u:.12g}u L=180n",
                    f"M{n}_flow_d {n}_flow_a {self.positive_write_gate} {self.neg_low_node} 0 {self.write_gate_device} W={self._gate_neg_discharge_width():.12g}u L=180n",
                    f"M{p}_flow_s {p} {p} {p}_flow_s 0 NREL W={self.pos_discharge_width_u:.12g}u L=180n",
                    f"M{p}_flow_b {p}_flow_s {self.backward_gate} {p}_flow_b 0 NREL W={self.pos_discharge_width_u:.12g}u L=180n",
                    f"M{p}_flow_a {p}_flow_b {self.pre_gate} {p}_flow_a 0 NREL W={self.pos_discharge_width_u:.12g}u L=180n",
                    f"M{p}_flow_d {p}_flow_a {self.negative_write_gate} {self.pos_low_node} 0 {self.write_gate_device} W={self._gate_pos_discharge_width():.12g}u L=180n",
                ]
            )
        else:
            deck.extend(
                [
                    f"M{n}_flow_b {n} {self.backward_gate} {n}_flow_b 0 NREL W={self.neg_discharge_width_u:.12g}u L=180n",
                    f"M{n}_flow_a {n}_flow_b {self.pre_gate} {n}_flow_a 0 NREL W={self.neg_discharge_width_u:.12g}u L=180n",
                    f"M{n}_flow_d {n}_flow_a {self.positive_write_gate} {self.neg_low_node} 0 {self.write_gate_device} W={self._gate_neg_discharge_width():.12g}u L=180n",
                    f"M{p}_flow_b {p} {self.backward_gate} {p}_flow_b 0 NREL W={self.pos_discharge_width_u:.12g}u L=180n",
                    f"M{p}_flow_a {p}_flow_b {self.pre_gate} {p}_flow_a 0 NREL W={self.pos_discharge_width_u:.12g}u L=180n",
                    f"M{p}_flow_d {p}_flow_a {self.negative_write_gate} {self.pos_low_node} 0 {self.write_gate_device} W={self._gate_pos_discharge_width():.12g}u L=180n",
                ]
            )
        deck.render_component(NodeParasitics(f"{self.name}_flow_parasitics", tuple(self._discharge_internal_nodes())))

    def _render_charge(self, deck: NetlistBuilder) -> None:
        p = self.pos_weight_node
        n = self.neg_weight_node
        if self.state_gate_charge:
            deck.extend(
                [
                    f"M{p}_ch_s {self.pos_high_node} {p} {p}_ch_s {self.vdd_node} PMOS W={self.pos_charge_width_u:.12g}u L=180n",
                    f"M{p}_ch_b {p}_ch_s {self.backward_gate} {p}_ch_b 0 NREL W={self.pos_charge_width_u:.12g}u L=180n",
                    f"M{p}_ch_a {p}_ch_b {self.pre_gate} {p}_ch_a 0 NREL W={self.pos_charge_width_u:.12g}u L=180n",
                    f"M{p}_ch_d {p}_ch_a {self.positive_write_gate} {p} 0 {self.write_gate_device} W={self._gate_pos_charge_width():.12g}u L=180n",
                    f"M{n}_ch_s {self.neg_high_node} {n} {n}_ch_s {self.vdd_node} PMOS W={self.neg_charge_width_u:.12g}u L=180n",
                    f"M{n}_ch_b {n}_ch_s {self.backward_gate} {n}_ch_b 0 NREL W={self.neg_charge_width_u:.12g}u L=180n",
                    f"M{n}_ch_a {n}_ch_b {self.pre_gate} {n}_ch_a 0 NREL W={self.neg_charge_width_u:.12g}u L=180n",
                    f"M{n}_ch_d {n}_ch_a {self.negative_write_gate} {n} 0 {self.write_gate_device} W={self._gate_neg_charge_width():.12g}u L=180n",
                ]
            )
        else:
            deck.extend(
                [
                    f"M{p}_ch_b {self.pos_high_node} {self.backward_gate} {p}_ch_b 0 NREL W={self.pos_charge_width_u:.12g}u L=180n",
                    f"M{p}_ch_a {p}_ch_b {self.pre_gate} {p}_ch_a 0 NREL W={self.pos_charge_width_u:.12g}u L=180n",
                    f"M{p}_ch_d {p}_ch_a {self.positive_write_gate} {p} 0 {self.write_gate_device} W={self._gate_pos_charge_width():.12g}u L=180n",
                    f"M{n}_ch_b {self.neg_high_node} {self.backward_gate} {n}_ch_b 0 NREL W={self.neg_charge_width_u:.12g}u L=180n",
                    f"M{n}_ch_a {n}_ch_b {self.pre_gate} {n}_ch_a 0 NREL W={self.neg_charge_width_u:.12g}u L=180n",
                    f"M{n}_ch_d {n}_ch_a {self.negative_write_gate} {n} 0 {self.write_gate_device} W={self._gate_neg_charge_width():.12g}u L=180n",
                ]
            )
        deck.render_component(NodeParasitics(f"{self.name}_charge_parasitics", tuple(self._charge_internal_nodes())))

    def _render_pmos_charge(self, deck: NetlistBuilder) -> None:
        p = self.pos_weight_node
        n = self.neg_weight_node
        assert self.positive_write_gate_low_true is not None
        assert self.negative_write_gate_low_true is not None
        if self.charge_extra_gate is None:
            deck.extend(
                [
                    f"M{p}_pch_s {p}_pch_b {self.positive_write_gate_low_true} {self.pos_high_node} {self.vdd_node} PMOS W={self._gate_pos_charge_width():.12g}u L=180n",
                    f"M{p}_pch_g {p}_pch_b {self.backward_gate} {p}_pch_g 0 NREL W={self.pos_charge_width_u:.12g}u L=180n",
                    f"M{p}_pch_a {p}_pch_g {self.pre_gate} {p} 0 NREL W={self.pos_charge_width_u:.12g}u L=180n",
                    f"M{n}_pch_s {n}_pch_b {self.negative_write_gate_low_true} {self.neg_high_node} {self.vdd_node} PMOS W={self._gate_neg_charge_width():.12g}u L=180n",
                    f"M{n}_pch_g {n}_pch_b {self.backward_gate} {n}_pch_g 0 NREL W={self.neg_charge_width_u:.12g}u L=180n",
                    f"M{n}_pch_a {n}_pch_g {self.pre_gate} {n} 0 NREL W={self.neg_charge_width_u:.12g}u L=180n",
                ]
            )
        else:
            deck.extend(
                [
                    f"M{p}_pch_s {p}_pch_b {self.positive_write_gate_low_true} {self.pos_high_node} {self.vdd_node} PMOS W={self._gate_pos_charge_width():.12g}u L=180n",
                    f"M{p}_pch_g {p}_pch_b {self.backward_gate} {p}_pch_g 0 NREL W={self.pos_charge_width_u:.12g}u L=180n",
                    f"M{p}_pch_x {p}_pch_g {self.charge_extra_gate} {p}_pch_x 0 NREL W={self.pos_charge_width_u:.12g}u L=180n",
                    f"M{p}_pch_a {p}_pch_x {self.pre_gate} {p} 0 NREL W={self.pos_charge_width_u:.12g}u L=180n",
                    f"M{n}_pch_s {n}_pch_b {self.negative_write_gate_low_true} {self.neg_high_node} {self.vdd_node} PMOS W={self._gate_neg_charge_width():.12g}u L=180n",
                    f"M{n}_pch_g {n}_pch_b {self.backward_gate} {n}_pch_g 0 NREL W={self.neg_charge_width_u:.12g}u L=180n",
                    f"M{n}_pch_x {n}_pch_g {self.charge_extra_gate} {n}_pch_x 0 NREL W={self.neg_charge_width_u:.12g}u L=180n",
                    f"M{n}_pch_a {n}_pch_x {self.pre_gate} {n} 0 NREL W={self.neg_charge_width_u:.12g}u L=180n",
                ]
            )
        deck.render_component(
            NodeParasitics(
                f"{self.name}_pmos_charge_parasitics",
                tuple(self._pmos_charge_internal_nodes()),
            )
        )

    def _render_cmos_complementary_charge(self, deck: NetlistBuilder) -> None:
        p = self.pos_weight_node
        n = self.neg_weight_node
        assert self.positive_write_gate_low_true is not None
        assert self.negative_write_gate_low_true is not None
        assert self.pre_gate_low_true is not None
        if self.gate_cmos_charge_with_backward:
            deck.extend(
                [
                    f"M{p}_cch_w {p}_cch_w {self.positive_write_gate_low_true} {self.pos_high_node} {self.vdd_node} PMOS W={self._gate_pos_charge_width():.12g}u L=180n",
                    f"M{p}_cch_b {p}_cch_b {self.backward_gate} {p}_cch_w 0 NREL W={self.pos_charge_width_u:.12g}u L=180n",
                    f"M{p}_cch_a {p} {self.pre_gate_low_true} {p}_cch_b {self.vdd_node} PMOS W={self.pos_charge_width_u:.12g}u L=180n",
                    f"M{n}_cch_w {n}_cch_w {self.negative_write_gate_low_true} {self.neg_high_node} {self.vdd_node} PMOS W={self._gate_neg_charge_width():.12g}u L=180n",
                    f"M{n}_cch_b {n}_cch_b {self.backward_gate} {n}_cch_w 0 NREL W={self.neg_charge_width_u:.12g}u L=180n",
                    f"M{n}_cch_a {n} {self.pre_gate_low_true} {n}_cch_b {self.vdd_node} PMOS W={self.neg_charge_width_u:.12g}u L=180n",
                ]
            )
        else:
            deck.extend(
                [
                    f"M{p}_cch_w {p}_cch_w {self.positive_write_gate_low_true} {self.pos_high_node} {self.vdd_node} PMOS W={self._gate_pos_charge_width():.12g}u L=180n",
                    f"M{p}_cch_a {p} {self.pre_gate_low_true} {p}_cch_w {self.vdd_node} PMOS W={self.pos_charge_width_u:.12g}u L=180n",
                    f"M{n}_cch_w {n}_cch_w {self.negative_write_gate_low_true} {self.neg_high_node} {self.vdd_node} PMOS W={self._gate_neg_charge_width():.12g}u L=180n",
                    f"M{n}_cch_a {n} {self.pre_gate_low_true} {n}_cch_w {self.vdd_node} PMOS W={self.neg_charge_width_u:.12g}u L=180n",
                ]
            )
        deck.render_component(
            NodeParasitics(
                f"{self.name}_cmos_complementary_charge_parasitics",
                tuple(self._cmos_complementary_charge_internal_nodes()),
            )
        )

    def _render_overlap(self, deck: NetlistBuilder) -> None:
        p = self.pos_weight_node
        n = self.neg_weight_node
        assert self.overlap_write_gate is not None
        if self.state_gate_discharge:
            deck.extend(
                [
                    f"M{p}_ov_s {p} {p} {p}_ov_s 0 NREL W={self.pos_discharge_width_u:.12g}u L=180n",
                    f"M{p}_ov_b {p}_ov_s {self.backward_gate} {p}_ov_b 0 NREL W={self.pos_discharge_width_u:.12g}u L=180n",
                    f"M{p}_ov_a {p}_ov_b {self.pre_gate} {p}_ov_a 0 NREL W={self.pos_discharge_width_u:.12g}u L=180n",
                    f"M{p}_ov_d {p}_ov_a {self.overlap_write_gate} {self.pos_low_node} 0 {self.write_gate_device} W={self.pos_discharge_width_u:.12g}u L=180n",
                    f"M{n}_ov_s {n} {n} {n}_ov_s 0 NREL W={self.neg_discharge_width_u:.12g}u L=180n",
                    f"M{n}_ov_b {n}_ov_s {self.backward_gate} {n}_ov_b 0 NREL W={self.neg_discharge_width_u:.12g}u L=180n",
                    f"M{n}_ov_a {n}_ov_b {self.pre_gate} {n}_ov_a 0 NREL W={self.neg_discharge_width_u:.12g}u L=180n",
                    f"M{n}_ov_d {n}_ov_a {self.overlap_write_gate} {self.neg_low_node} 0 {self.write_gate_device} W={self.neg_discharge_width_u:.12g}u L=180n",
                ]
            )
        else:
            deck.extend(
                [
                    f"M{p}_ov_b {p} {self.backward_gate} {p}_ov_b 0 NREL W={self.pos_discharge_width_u:.12g}u L=180n",
                    f"M{p}_ov_a {p}_ov_b {self.pre_gate} {p}_ov_a 0 NREL W={self.pos_discharge_width_u:.12g}u L=180n",
                    f"M{p}_ov_d {p}_ov_a {self.overlap_write_gate} {self.pos_low_node} 0 {self.write_gate_device} W={self.pos_discharge_width_u:.12g}u L=180n",
                    f"M{n}_ov_b {n} {self.backward_gate} {n}_ov_b 0 NREL W={self.neg_discharge_width_u:.12g}u L=180n",
                    f"M{n}_ov_a {n}_ov_b {self.pre_gate} {n}_ov_a 0 NREL W={self.neg_discharge_width_u:.12g}u L=180n",
                    f"M{n}_ov_d {n}_ov_a {self.overlap_write_gate} {self.neg_low_node} 0 {self.write_gate_device} W={self.neg_discharge_width_u:.12g}u L=180n",
                ]
            )
        deck.render_component(NodeParasitics(f"{self.name}_overlap_parasitics", tuple(self._overlap_internal_nodes())))

    def _render_center_pull(self, deck: NetlistBuilder) -> None:
        p = self.pos_weight_node
        n = self.neg_weight_node
        if self.center_pull_mode == "always":
            deck.extend(
                [
                    f"M{p}_center {p} {self.center_pull_gate} {self.pos_center_pull_node} 0 NREL W={self.center_pull_width_u:.12g}u L=180n",
                    f"M{n}_center {n} {self.center_pull_gate} {self.neg_center_pull_node} 0 NREL W={self.center_pull_width_u:.12g}u L=180n",
                ]
            )
        else:
            deck.extend(
                [
                    f"M{p}_center_g {p} {self.center_pull_gate} {p}_center_g 0 NREL W={self.center_pull_width_u:.12g}u L=180n",
                    f"M{p}_center_s {p}_center_g {p} {self.pos_center_pull_node} 0 NREL W={self.center_pull_width_u:.12g}u L=180n",
                    f"M{n}_center_g {n} {self.center_pull_gate} {n}_center_g 0 NREL W={self.center_pull_width_u:.12g}u L=180n",
                    f"M{n}_center_s {n}_center_g {n} {self.neg_center_pull_node} 0 NREL W={self.center_pull_width_u:.12g}u L=180n",
                ]
            )
            deck.render_component(NodeParasitics(f"{self.name}_center_parasitics", tuple(self._center_internal_nodes())))


@dataclass
class ReadoutBranch(Component):
    """One MOS readout branch from an activation/weight pair into a score node."""

    name: str
    style: str
    branch: str
    activation_node: str
    weight_node: str
    score_node: str
    width_u: float
    fwd_gate: str = "fwd"
    vdd_node: str = "vdd"
    reset_gate: str = "rstf"
    internal_prefix: str | None = None
    buffered_activation_name: str | None = None

    @property
    def buffered_activation_node(self) -> str:
        return self.buffered_activation_name or f"{self.name}_actbuf"

    @property
    def internal_node_prefix(self) -> str:
        return self.internal_prefix if self.internal_prefix is not None else f"{self.name}_"

    @property
    def internal_nodes(self) -> tuple[str, ...]:
        if self.style == "gate_stack":
            prefix = self.internal_node_prefix
            return (f"{prefix}0", f"{prefix}1")
        if self.style == "pass_act_source":
            return (f"{self.internal_node_prefix}1",)
        if self.style == "pass_act_buffered":
            return (self.buffered_activation_node, f"{self.internal_node_prefix}1")
        return ()

    def children(self) -> list[Component]:
        return []

    def input_nodes(self) -> tuple[str, ...]:
        return (self.activation_node, self.weight_node)

    def output_nodes(self) -> tuple[str, ...]:
        return (self.score_node,)

    def state_nodes(self) -> tuple[str, ...]:
        return self.internal_nodes

    def validate(self) -> None:
        if not self.name:
            raise ValueError("ReadoutBranch requires a nonempty name")
        if self.style not in {"gate_stack", "pass_act_source", "pass_act_buffered"}:
            raise ValueError(f"unknown readout branch style: {self.style}")
        if self.branch not in {"pos", "neg"}:
            raise ValueError(f"unknown readout branch polarity: {self.branch}")
        for node in (self.activation_node, self.weight_node, self.score_node, self.fwd_gate):
            if not node:
                raise ValueError("ReadoutBranch requires nonempty node names")
        if self.width_u <= 0:
            raise ValueError("ReadoutBranch width must be positive")

    def render(self, deck: NetlistBuilder) -> None:
        if self.style == "gate_stack":
            n0, n1 = self.internal_nodes
            deck.mos(
                f"{self.name}_a",
                self.vdd_node,
                self.activation_node,
                n0,
                "0",
                "NSENSE",
                width_u=self.width_u,
            )
            deck.mos(f"{self.name}_w", n0, self.weight_node, n1, "0", "NREL", width_u=self.width_u)
            deck.mos(f"{self.name}_f", n1, self.fwd_gate, self.score_node, "0", "NREL", width_u=self.width_u)
        elif self.style == "pass_act_source":
            (n1,) = self.internal_nodes
            deck.mos(f"{self.name}_w", self.activation_node, self.weight_node, n1, "0", "NREL", width_u=self.width_u)
            deck.mos(f"{self.name}_f", n1, self.fwd_gate, self.score_node, "0", "NREL", width_u=self.width_u)
        else:
            actbuf, n1 = self.internal_nodes
            deck.mos(
                f"{self.name}_actbuf_src",
                self.vdd_node,
                self.activation_node,
                actbuf,
                "0",
                "NSENSE",
                width_u=self.width_u,
            )
            deck.mos(f"{self.name}_actbuf_rst", actbuf, self.reset_gate, "0", "0", "NREL", width_u=4)
            deck.capacitor(f"{self.name}_actbuf", actbuf, "0", "0.02f", ic=0)
            deck.resistor(f"{self.name}_actbuf", actbuf, "0", "1e13")
            deck.mos(f"{self.name}_w", actbuf, self.weight_node, n1, "0", "NREL", width_u=self.width_u)
            deck.mos(f"{self.name}_f", n1, self.fwd_gate, self.score_node, "0", "NREL", width_u=self.width_u)
        deck.render_component(NodeParasitics(f"{self.name}_parasitics", self.internal_nodes))


@dataclass
class SignedMosSynapse(Synapse):
    name: str
    pre: str
    post: str
    weight_p: CapState
    weight_n: CapState

    def children(self) -> list[Component]:
        return [self.weight_p, self.weight_n]

    def input_nodes(self) -> tuple[str, ...]:
        return (self.pre,)

    def output_nodes(self) -> tuple[str, ...]:
        return (self.post,)

    def state_nodes(self) -> tuple[str, ...]:
        return (*self.weight_p.state_nodes(), *self.weight_n.state_nodes())

    def validate(self) -> None:
        if not self.name or not self.pre or not self.post:
            raise ValueError("SignedMosSynapse requires nonempty name/pre/post")
        self.weight_p.validate()
        self.weight_n.validate()

    def render(self, deck: NetlistBuilder) -> None:
        self.weight_p.render(deck)
        self.weight_n.render(deck)


@dataclass
class ReLUNeuron(Neuron):
    name: str
    preactivation: CapState
    activation_node: str
    width_u: float = 24.0

    def children(self) -> list[Component]:
        return [self.preactivation]

    def input_nodes(self) -> tuple[str, ...]:
        return (self.preactivation.node,)

    def output_nodes(self) -> tuple[str, ...]:
        return (self.activation_node,)

    def state_nodes(self) -> tuple[str, ...]:
        return self.preactivation.state_nodes()

    def validate(self) -> None:
        if not self.name or not self.activation_node:
            raise ValueError("ReLUNeuron requires nonempty name and activation node")
        if self.width_u <= 0:
            raise ValueError("ReLUNeuron width must be positive")
        self.preactivation.validate()

    def render(self, deck: NetlistBuilder) -> None:
        self.preactivation.render(deck)
        deck.mos(
            f"relu_{self.name}",
            "vdd",
            self.preactivation.node,
            self.activation_node,
            "0",
            "NREL",
            width_u=self.width_u,
        )
