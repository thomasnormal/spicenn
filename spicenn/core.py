from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Union

from .topology import FanInTopology, SourceId


def fmt(value: str | int | float) -> str:
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


@dataclass(frozen=True)
class Node:
    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("node name must be nonempty")

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Port:
    name: str
    node: Node

    @classmethod
    def at(cls, name: str, node: str | Node) -> "Port":
        return cls(name=name, node=node if isinstance(node, Node) else Node(node))


@dataclass(frozen=True)
class NonnegativeSignal:
    """Single-ended signal whose represented value is constrained to be >= 0.

    ReLU activations and one-hot input rails can use this compact representation.
    Signed quantities should not be silently squeezed into this form; use
    ``SignedSignal`` so both rails are visible to topology and netlist checks.
    """

    node: Node

    @classmethod
    def at(cls, node: str | Node) -> "NonnegativeSignal":
        return cls(node=node if isinstance(node, Node) else Node(node))

    def nodes(self) -> tuple[str, ...]:
        return (str(self.node),)

    def as_signed_against(self, negative_node: str | Node = "0") -> "SignedSignal":
        return SignedSignal(
            positive=self.node,
            negative=negative_node if isinstance(negative_node, Node) else Node(negative_node),
        )

    def __str__(self) -> str:
        return str(self.node)


@dataclass(frozen=True)
class SignedSignal:
    """Differential positive/negative rail pair for a signed analog value."""

    positive: Node
    negative: Node

    @classmethod
    def at(cls, positive: str | Node, negative: str | Node) -> "SignedSignal":
        return cls(
            positive=positive if isinstance(positive, Node) else Node(positive),
            negative=negative if isinstance(negative, Node) else Node(negative),
        )

    @classmethod
    def from_base(cls, base: str, *, positive_suffix: str = "p", negative_suffix: str = "n") -> "SignedSignal":
        if not base:
            raise ValueError("signed signal base must be nonempty")
        return cls.at(f"{base}{positive_suffix}", f"{base}{negative_suffix}")

    def nodes(self) -> tuple[str, ...]:
        return (str(self.positive), str(self.negative))

    def differential_expr(self) -> str:
        return f"{self.positive}-{self.negative}"


Signal = Union[NonnegativeSignal, SignedSignal]


def require_nonnegative_signal(signal: Signal, *, context: str = "signal") -> NonnegativeSignal:
    if isinstance(signal, NonnegativeSignal):
        return signal
    raise ValueError(f"{context} must be a nonnegative single-ended signal, got signed rails {signal.nodes()}")


def require_signed_signal(signal: Signal, *, context: str = "signal") -> SignedSignal:
    if isinstance(signal, SignedSignal):
        return signal
    raise ValueError(f"{context} must be signed differential rails, got nonnegative node {signal.nodes()[0]}")


class Component:
    name: str

    def children(self) -> list["Component"]:
        return []

    def input_nodes(self) -> tuple[str, ...]:
        return ()

    def output_nodes(self) -> tuple[str, ...]:
        return ()

    def state_nodes(self) -> tuple[str, ...]:
        return ()

    def render(self, deck: "NetlistBuilder") -> None:
        raise NotImplementedError

    def render_spice(self) -> str:
        deck = NetlistBuilder()
        deck.render_component(self)
        return deck.render_body()

    def render_lines(self) -> list[str]:
        body = self.render_spice()
        return [] if not body else body.splitlines()

    def validate(self) -> None:
        if not getattr(self, "name", ""):
            raise ValueError("component name must be nonempty")


@dataclass
class NetlistBuilder:
    title: str = ""
    lines: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.title:
            self.comment(self.title)

    def comment(self, text: str = "") -> None:
        if not text:
            self.lines.append("*")
            return
        for line in text.splitlines():
            self.lines.append(f"* {line}")

    def blank(self) -> None:
        self.lines.append("")

    def raw(self, line: str = "") -> None:
        self.lines.append(line)

    def extend(self, lines: Iterable[str]) -> None:
        self.lines.extend(lines)

    def capacitor(
        self,
        name: str,
        p: str | Node,
        n: str | Node,
        value: str | int | float,
        *,
        ic: str | int | float | None = None,
    ) -> None:
        line = f"C{name} {p} {n} {fmt(value)}"
        if ic is not None:
            line += f" IC={fmt(ic)}"
        self.lines.append(line)

    def resistor(self, name: str, p: str | Node, n: str | Node, value: str | int | float) -> None:
        self.lines.append(f"R{name} {p} {n} {fmt(value)}")

    def mos(
        self,
        name: str,
        drain: str | Node,
        gate: str | Node,
        source: str | Node,
        body: str | Node,
        model: str,
        *,
        width_u: str | int | float,
        length: str = "180n",
    ) -> None:
        self.lines.append(
            f"M{name} {drain} {gate} {source} {body} {model} W={fmt(width_u)}u L={length}"
        )

    def render_component(self, component: Component) -> None:
        component.validate()
        component.render(self)

    def render(self) -> str:
        text = "\n".join(self.lines)
        return text if text.endswith("\n") else text + "\n"

    def render_body(self) -> str:
        return "\n".join(self.lines)


@dataclass
class CompositeComponent(Component):
    name: str
    _children: list[Component] = field(default_factory=list)

    def add(self, child: Component) -> Component:
        self._children.append(child)
        return child

    def children(self) -> list[Component]:
        return list(self._children)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("component name must be nonempty")
        for child in self._children:
            child.validate()

    def render(self, deck: NetlistBuilder) -> None:
        for child in self._children:
            deck.render_component(child)


@dataclass
class Layer(CompositeComponent):
    inputs: tuple[Port, ...] = ()
    outputs: tuple[Port, ...] = ()
    topology: FanInTopology | None = None

    def input_nodes(self) -> tuple[str, ...]:
        return tuple(str(port.node) for port in self.inputs)

    def output_nodes(self) -> tuple[str, ...]:
        return tuple(str(port.node) for port in self.outputs)

    def fanins(self) -> dict[int, tuple[SourceId, ...]]:
        return {} if self.topology is None else self.topology.as_fanins()

    def fanouts(self) -> dict[SourceId, tuple[int, ...]]:
        return {} if self.topology is None else self.topology.fanouts()


class Synapse(Component):
    """Base class for SPICE-renderable synapse cells."""


class Neuron(Component):
    """Base class for SPICE-renderable neuron cells."""
