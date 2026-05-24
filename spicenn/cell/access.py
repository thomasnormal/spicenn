from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EmitMode(Enum):
    """Context in which a cell is emitted into a deck."""

    EXPERIMENT = "experiment"
    CHARACTERIZATION = "characterization"
    GOLDEN_REFERENCE = "golden_reference"


class StateAccess(Enum):
    """Allowed access policy for a declared cell state node."""

    PUBLIC_INITIAL_CONDITION = "public_initial_condition"
    PASSIVE_PROBE_ALLOWED = "passive_probe_allowed"
    CHARACTERIZATION_FORCE_ALLOWED = "characterization_force_allowed"
    EXPERIMENT_INTERNAL = "experiment_internal"


class ElementTag(Enum):
    """Deck-element ownership tag used by experiment/characterization linters."""

    CELL_CORE = "cell_core"
    CELL_WRITE = "cell_write"
    EXPERIMENT_INPUT = "experiment_input"
    PHASE_SOURCE = "phase_source"
    CONTROL_SOURCE = "control_source"
    INITIAL_CONDITION = "initial_condition"
    PASSIVE_PROBE = "passive_probe"
    DEBUG_FORCE = "debug_force"
    CHARACTERIZATION_ONLY = "characterization_only"


@dataclass(frozen=True)
class TaggedElement:
    """Minimal manifest entry for an emitted netlist element.

    The linter intentionally works from a small sidecar manifest rather than
    parsing every simulator dialect.  Emitters can tag elements as they create
    them; characterization harnesses can add debug-only elements explicitly.
    """

    name: str
    nodes: tuple[str, ...]
    tag: ElementTag
    passive: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tagged element name must be nonempty")
        if not self.nodes:
            raise ValueError(f"tagged element {self.name!r} must touch at least one node")
