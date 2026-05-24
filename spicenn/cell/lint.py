from __future__ import annotations

from .access import ElementTag, StateAccess, TaggedElement
from .contract import CellContract


class ExperimentLintError(ValueError):
    pass


_CELL_OWNED_TAGS = {ElementTag.CELL_CORE, ElementTag.CELL_WRITE}
_EXTERNAL_ACTIVE_TAGS = {ElementTag.EXPERIMENT_INPUT, ElementTag.PHASE_SOURCE, ElementTag.CONTROL_SOURCE}
_DEBUG_ONLY_TAGS = {ElementTag.DEBUG_FORCE, ElementTag.CHARACTERIZATION_ONLY}


def lint_experiment_elements(contract: CellContract, elements: list[TaggedElement] | tuple[TaggedElement, ...]) -> None:
    """Reject active characterization/debug access in experiment-mode decks."""

    state_by_node = contract.state_by_node()
    public_nodes = contract.ports.public_nodes()

    for element in elements:
        if element.tag in _DEBUG_ONLY_TAGS:
            raise ExperimentLintError(
                f"{element.name}: {element.tag.value} element is not allowed in experiment mode"
            )

        touched_state = [node for node in element.nodes if node in state_by_node]

        if element.tag is ElementTag.PASSIVE_PROBE:
            if not element.passive:
                raise ExperimentLintError(f"{element.name}: passive probe tag must set passive=True")
            continue

        if element.tag in _CELL_OWNED_TAGS:
            continue

        if element.tag is ElementTag.INITIAL_CONDITION:
            for node in touched_state:
                access = state_by_node[node].access
                if access not in {
                    StateAccess.PUBLIC_INITIAL_CONDITION,
                    StateAccess.CHARACTERIZATION_FORCE_ALLOWED,
                }:
                    raise ExperimentLintError(
                        f"{element.name}: initial condition on internal state node {node!r} "
                        f"with access {access.value} is not allowed"
                    )
            continue

        if element.tag in _EXTERNAL_ACTIVE_TAGS:
            illegal_state_nodes = [node for node in touched_state if node not in public_nodes]
            if illegal_state_nodes:
                raise ExperimentLintError(
                    f"{element.name}: external active source touches internal state nodes "
                    f"{illegal_state_nodes}"
                )
            continue

        # Unknown future tags should fail closed unless they are explicitly
        # passive.  This prevents characterization helpers from silently leaking
        # into headline runs.
        if touched_state:
            raise ExperimentLintError(
                f"{element.name}: unrecognized active access to state nodes {touched_state}"
            )
