from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from .access import StateAccess
from .contract import CellContract, StateRole


class ProtocolFamily(Enum):
    BACKPROP_LOCAL = "forward_store_backward_update"
    DFA = "forward_store_direct_feedback_update"
    CROSSBAR_SGD = "forward_backward_outer_product"
    TIKI_TAKA = "forward_backward_accumulate_transfer"
    EQPROP = "free_nudged_contrastive_update"
    SPIKING_ELIGIBILITY = "spike_trace_error_modulated_update"


class IncompatibleCellError(ValueError):
    pass


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    supported_protocols: frozenset[ProtocolFamily]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("experiment spec name must be nonempty")
        if not self.supported_protocols:
            raise ValueError(f"experiment {self.name!r} must support at least one protocol family")

    def supports(self, family: ProtocolFamily) -> bool:
        return family in self.supported_protocols

    def validate_contract(self, contract: CellContract) -> None:
        protocol = contract.protocol
        family = getattr(protocol, "family", None)
        if not self.supports(family):
            supported = sorted(entry.value for entry in self.supported_protocols)
            raise IncompatibleCellError(
                f"experiment {self.name!r} does not support protocol {getattr(family, 'value', family)!r}; "
                f"supported: {supported}"
            )
        protocol.validate_contract(contract)


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    required: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("phase name must be nonempty")


@dataclass(frozen=True)
class LearningProtocol:
    family: ProtocolFamily
    phases: Mapping[str, PhaseSpec]
    required_ports: frozenset[str]
    optional_ports: frozenset[str] = frozenset()
    required_state_roles: frozenset[StateRole] = frozenset()
    allowed_experiment_state_access: Mapping[StateRole, frozenset[StateAccess]] = field(default_factory=dict)
    default_characterization_tests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        missing_phase_names = [name for name, spec in self.phases.items() if name != spec.name]
        if missing_phase_names:
            raise ValueError(f"phase mapping keys must match PhaseSpec names: {missing_phase_names}")
        overlap = set(self.required_ports).intersection(self.optional_ports)
        if overlap:
            raise ValueError(f"ports cannot be both required and optional: {sorted(overlap)}")

    def validate_contract(self, contract: CellContract) -> None:
        port_names = contract.ports.names()
        missing_ports = sorted(self.required_ports.difference(port_names))
        if missing_ports:
            raise IncompatibleCellError(
                f"{contract.capabilities.role.value} cell missing required "
                f"{self.family.value} ports: {missing_ports}"
            )

        missing_phases = sorted(
            name for name, spec in self.phases.items() if spec.required and name not in contract.ports.phases
        )
        if missing_phases:
            raise IncompatibleCellError(
                f"{contract.capabilities.role.value} cell missing required phases: {missing_phases}"
            )

        missing_roles = sorted(role.value for role in self.required_state_roles.difference(contract.state_roles()))
        if missing_roles:
            raise IncompatibleCellError(
                f"{contract.capabilities.role.value} cell missing required state roles: {missing_roles}"
            )

        for spec in contract.state:
            allowed = self.allowed_experiment_state_access.get(spec.role)
            if allowed is not None and spec.access not in allowed:
                raise IncompatibleCellError(
                    f"state {spec.name!r} role {spec.role.value} has access {spec.access.value}, "
                    f"allowed: {[entry.value for entry in allowed]}"
                )


def _phase_names(*names: str) -> dict[str, PhaseSpec]:
    return {name: PhaseSpec(name=name) for name in names}


def backprop_local_protocol() -> LearningProtocol:
    return LearningProtocol(
        family=ProtocolFamily.BACKPROP_LOCAL,
        phases=_phase_names("pact", "pbwd", "pacc"),
        required_ports=frozenset({"x", "h", "learning_in", "pact", "pbwd", "pacc", "eta"}),
        optional_ports=frozenset({"learning_out", "noise_en", "clamp"}),
        required_state_roles=frozenset(
            {StateRole.TRAINABLE, StateRole.FORWARD_DYNAMIC, StateRole.BACKWARD_DYNAMIC}
        ),
        default_characterization_tests=(
            "forward_transfer",
            "forward_storage",
            "backward_alignment",
            "update_alignment",
            "hold_drift",
            "read_disturb",
        ),
    )


def dfa_protocol() -> LearningProtocol:
    return LearningProtocol(
        family=ProtocolFamily.DFA,
        phases=_phase_names("pact", "pbwd", "pacc"),
        required_ports=frozenset({"x", "h", "output_error", "pact", "pbwd", "pacc", "eta"}),
        optional_ports=frozenset({"learning_out", "noise_en", "clamp"}),
        required_state_roles=frozenset(
            {
                StateRole.TRAINABLE,
                StateRole.FORWARD_DYNAMIC,
                StateRole.BACKWARD_DYNAMIC,
                StateRole.FIXED_RANDOM,
            }
        ),
        default_characterization_tests=(
            "forward_transfer",
            "dfa_feedback_gain",
            "update_alignment",
            "hold_drift",
            "read_disturb",
        ),
    )


def eqprop_protocol() -> LearningProtocol:
    return LearningProtocol(
        family=ProtocolFamily.EQPROP,
        phases=_phase_names("free", "sample_free", "nudged", "sample_nudged", "update"),
        required_ports=frozenset(
            {"input_clamp", "output_nodes", "target_nudge", "free", "sample_free", "nudged", "sample_nudged", "update", "eta", "beta"}
        ),
        optional_ports=frozenset({"noise_en"}),
        required_state_roles=frozenset({StateRole.TRAINABLE, StateRole.SAMPLE_FREE, StateRole.SAMPLE_NUDGED}),
        default_characterization_tests=(
            "free_phase_convergence",
            "nudged_phase_convergence",
            "contrastive_update_alignment",
            "sample_retention",
        ),
    )


def spiking_eligibility_protocol() -> LearningProtocol:
    return LearningProtocol(
        family=ProtocolFamily.SPIKING_ELIGIBILITY,
        phases=_phase_names("run", "update"),
        required_ports=frozenset({"event_in", "learning_signal", "run", "update", "eta"}),
        optional_ports=frozenset({"event_out", "membrane", "noise_en"}),
        required_state_roles=frozenset({StateRole.TRAINABLE, StateRole.FORWARD_DYNAMIC, StateRole.ELIGIBILITY}),
        default_characterization_tests=(
            "spike_threshold",
            "trace_decay",
            "eligibility_update_alignment",
            "noise_robust_alignment",
        ),
    )
