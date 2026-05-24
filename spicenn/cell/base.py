from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .access import EmitMode, TaggedElement
from .characterization import CharacterizationExpectation
from .contract import CellContract


class ReferenceModel(Protocol):
    """Numerical model used by characterization and alignment tests."""

    def forward(self, inputs: Mapping[str, Any], state: Mapping[str, Any]) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class CellInstance:
    name: str
    contract: CellContract
    elements: tuple[TaggedElement, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("cell instance name must be nonempty")


class TrainableDynamicalCell(Protocol):
    """SPICE-emittable trainable dynamical block.

    A cell may be a scalar neuron, local-feature cell, writer, crossbar tile,
    EqProp tile, reservoir, or spiking eligibility block.  The stable API is
    the declared contract plus tagged emitted elements.
    """

    name: str

    def contract(self) -> CellContract:
        ...

    def emit(self, deck: Any, inst: str, bindings: Mapping[str, str], mode: EmitMode) -> CellInstance:
        ...

    def reference_model(self) -> ReferenceModel:
        ...

    def characterization_suite(self) -> list[CharacterizationExpectation]:
        ...
