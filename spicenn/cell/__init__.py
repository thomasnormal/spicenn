"""Contracts for swappable trainable dynamical SPICE cells."""

from .access import EmitMode, ElementTag, StateAccess, TaggedElement
from .base import CellInstance, ReferenceModel, TrainableDynamicalCell
from .characterization import CharacterizationExpectation, CharacterizationResult, PromotionLevel
from .contract import (
    CellCapabilities,
    CellContract,
    CellPorts,
    CellRole,
    ParamSpec,
    Quantity,
    RailBundle,
    RailDomain,
    Signedness,
    StateRole,
    StateSpec,
)
from .lint import ExperimentLintError, lint_experiment_elements
from .local_feature import LOCAL_FEATURE_CELLS, LocalFeatureCell, local_feature_cell_by_name
from .local_feature_characterization import characterize_local_feature_cell
from .protocol import (
    ExperimentSpec,
    IncompatibleCellError,
    LearningProtocol,
    PhaseSpec,
    ProtocolFamily,
    backprop_local_protocol,
    dfa_protocol,
    eqprop_protocol,
    spiking_eligibility_protocol,
)
from .registry import CellRegistry

__all__ = [
    "CellCapabilities",
    "CellContract",
    "CellInstance",
    "CellPorts",
    "CellRegistry",
    "CellRole",
    "CharacterizationExpectation",
    "CharacterizationResult",
    "characterize_local_feature_cell",
    "EmitMode",
    "ElementTag",
    "ExperimentLintError",
    "ExperimentSpec",
    "IncompatibleCellError",
    "LearningProtocol",
    "LOCAL_FEATURE_CELLS",
    "LocalFeatureCell",
    "ParamSpec",
    "PhaseSpec",
    "PromotionLevel",
    "ProtocolFamily",
    "Quantity",
    "RailBundle",
    "RailDomain",
    "ReferenceModel",
    "Signedness",
    "StateAccess",
    "StateRole",
    "StateSpec",
    "TaggedElement",
    "TrainableDynamicalCell",
    "backprop_local_protocol",
    "dfa_protocol",
    "eqprop_protocol",
    "lint_experiment_elements",
    "local_feature_cell_by_name",
    "spiking_eligibility_protocol",
]
