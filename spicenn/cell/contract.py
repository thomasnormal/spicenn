from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from .access import StateAccess


class RailDomain(Enum):
    VOLTAGE = "voltage"
    CURRENT = "current"
    SPIKE = "spike"
    LOGIC = "logic"


class Signedness(Enum):
    UNSIGNED = "unsigned"
    SIGNED_SINGLE = "signed_single"
    DIFFERENTIAL = "differential"


class Quantity(Enum):
    ACTIVATION = "activation"
    INPUT = "input"
    PREACTIVATION = "preactivation"
    GRADIENT = "gradient"
    ERROR = "error"
    NUDGE = "nudge"
    LEARNING_SIGNAL = "learning_signal"
    EVENT = "event"
    PHASE = "phase"
    CONTROL = "control"


class CellRole(Enum):
    SCALAR_NEURON = "scalar_neuron"
    LOCAL_FEATURE_CELL = "local_feature_cell"
    SYNAPSE_WRITER = "synapse_writer"
    CROSSBAR_TILE = "crossbar_tile"
    EQPROP_TILE = "eqprop_tile"
    RESERVOIR_TILE = "reservoir_tile"
    SPIKING_CELL = "spiking_cell"


class StateRole(Enum):
    TRAINABLE = "trainable"
    FORWARD_DYNAMIC = "forward_dynamic"
    BACKWARD_DYNAMIC = "backward_dynamic"
    ELIGIBILITY = "eligibility"
    SAMPLE_FREE = "sample_free"
    SAMPLE_NUDGED = "sample_nudged"
    AUXILIARY = "auxiliary"
    FIXED_RANDOM = "fixed_random"


@dataclass(frozen=True)
class RailBundle:
    name: str
    width: int
    domain: RailDomain
    signed: Signedness
    quantity: Quantity
    nodes: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("rail bundle name must be nonempty")
        if self.width <= 0:
            raise ValueError(f"rail bundle {self.name!r} width must be positive")
        if self.nodes is None:
            return
        expected = self.width * 2 if self.signed is Signedness.DIFFERENTIAL else self.width
        if len(self.nodes) != expected:
            raise ValueError(
                f"rail bundle {self.name!r} has {len(self.nodes)} nodes, expected {expected}"
            )

    def node_set(self) -> set[str]:
        return set(self.nodes or ())


@dataclass(frozen=True)
class CellPorts:
    inputs: Mapping[str, RailBundle] = field(default_factory=dict)
    outputs: Mapping[str, RailBundle] = field(default_factory=dict)
    learning_inputs: Mapping[str, RailBundle] = field(default_factory=dict)
    learning_outputs: Mapping[str, RailBundle] = field(default_factory=dict)
    phases: Mapping[str, RailBundle] = field(default_factory=dict)
    controls: Mapping[str, RailBundle] = field(default_factory=dict)

    def all(self) -> dict[str, RailBundle]:
        merged: dict[str, RailBundle] = {}
        for group in (
            self.inputs,
            self.outputs,
            self.learning_inputs,
            self.learning_outputs,
            self.phases,
            self.controls,
        ):
            overlap = set(merged).intersection(group)
            if overlap:
                raise ValueError(f"duplicate cell port names: {sorted(overlap)}")
            merged.update(group)
        return merged

    def public_nodes(self) -> set[str]:
        nodes: set[str] = set()
        for bundle in self.all().values():
            nodes.update(bundle.node_set())
        return nodes

    def names(self) -> set[str]:
        return set(self.all())


@dataclass(frozen=True)
class StateSpec:
    name: str
    role: StateRole
    nodes: tuple[str, ...]
    access: StateAccess
    units: str = "V"
    expected_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("state spec name must be nonempty")
        if not self.nodes:
            raise ValueError(f"state spec {self.name!r} must contain at least one node")
        if self.expected_range is not None and self.expected_range[0] > self.expected_range[1]:
            raise ValueError(f"state spec {self.name!r} has an invalid expected range")


@dataclass(frozen=True)
class ParamSpec:
    name: str
    default: float | int | str | None = None
    units: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("parameter name must be nonempty")


@dataclass(frozen=True)
class CellCapabilities:
    role: CellRole
    supports_forward: bool
    supports_backward: bool
    supports_local_update: bool
    emits_learning_outputs: bool
    stores_forward_state: bool
    stores_backward_state: bool
    has_trainable_state: bool
    supports_spiking_io: bool = False
    supports_contrastive_update: bool = False
    supports_noise_injection: bool = False


@dataclass(frozen=True)
class CellContract:
    ports: CellPorts
    state: tuple[StateSpec, ...]
    params: Mapping[str, ParamSpec]
    protocol: object
    capabilities: CellCapabilities

    def __post_init__(self) -> None:
        # Force duplicate-port validation early.
        self.ports.all()
        state_names = [spec.name for spec in self.state]
        duplicate_state_names = sorted({name for name in state_names if state_names.count(name) > 1})
        if duplicate_state_names:
            raise ValueError(f"duplicate state names: {duplicate_state_names}")
        state_nodes = [node for spec in self.state for node in spec.nodes]
        duplicate_state_nodes = sorted({node for node in state_nodes if state_nodes.count(node) > 1})
        if duplicate_state_nodes:
            raise ValueError(f"duplicate state nodes: {duplicate_state_nodes}")

    def state_by_node(self) -> dict[str, StateSpec]:
        return {node: spec for spec in self.state for node in spec.nodes}

    def state_roles(self) -> set[StateRole]:
        return {spec.role for spec in self.state}
