import pytest

from spicenn.cell import (
    CellCapabilities,
    CellContract,
    CellPorts,
    CellRole,
    ElementTag,
    ExperimentLintError,
    ExperimentSpec,
    IncompatibleCellError,
    ParamSpec,
    ProtocolFamily,
    Quantity,
    RailBundle,
    RailDomain,
    Signedness,
    StateAccess,
    StateRole,
    StateSpec,
    TaggedElement,
    backprop_local_protocol,
    eqprop_protocol,
    lint_experiment_elements,
)


def rail(name, quantity, nodes=None, *, width=1, signed=Signedness.UNSIGNED):
    return RailBundle(
        name=name,
        width=width,
        domain=RailDomain.VOLTAGE,
        signed=signed,
        quantity=quantity,
        nodes=nodes,
    )


def backprop_contract(*, include_pacc=True):
    phases = {
        "pact": rail("pact", Quantity.PHASE, ("pact",)),
        "pbwd": rail("pbwd", Quantity.PHASE, ("pbwd",)),
    }
    if include_pacc:
        phases["pacc"] = rail("pacc", Quantity.PHASE, ("pacc",))
    ports = CellPorts(
        inputs={"x": rail("x", Quantity.INPUT, ("x0", "x1"), width=2)},
        outputs={"h": rail("h", Quantity.ACTIVATION, ("h",))},
        learning_inputs={
            "learning_in": rail(
                "learning_in",
                Quantity.ERROR,
                ("errp", "errn"),
                signed=Signedness.DIFFERENTIAL,
            )
        },
        phases=phases,
        controls={"eta": rail("eta", Quantity.CONTROL, ("eta",))},
    )
    return CellContract(
        ports=ports,
        state=(
            StateSpec(
                name="w",
                role=StateRole.TRAINABLE,
                nodes=("w0", "w1"),
                access=StateAccess.PUBLIC_INITIAL_CONDITION,
            ),
            StateSpec(
                name="h_cap",
                role=StateRole.FORWARD_DYNAMIC,
                nodes=("hcap",),
                access=StateAccess.PASSIVE_PROBE_ALLOWED,
            ),
            StateSpec(
                name="dh_cap",
                role=StateRole.BACKWARD_DYNAMIC,
                nodes=("dhp", "dhn"),
                access=StateAccess.PASSIVE_PROBE_ALLOWED,
            ),
        ),
        params={"eta0": ParamSpec("eta0", default=0.1)},
        protocol=backprop_local_protocol(),
        capabilities=CellCapabilities(
            role=CellRole.LOCAL_FEATURE_CELL,
            supports_forward=True,
            supports_backward=True,
            supports_local_update=True,
            emits_learning_outputs=False,
            stores_forward_state=True,
            stores_backward_state=True,
            has_trainable_state=True,
        ),
    )


def test_backprop_protocol_accepts_complete_local_feature_contract():
    contract = backprop_contract()
    contract.protocol.validate_contract(contract)


def test_backprop_protocol_rejects_missing_required_phase():
    contract = backprop_contract(include_pacc=False)
    with pytest.raises(IncompatibleCellError, match="pacc"):
        contract.protocol.validate_contract(contract)


def test_eqprop_protocol_rejects_backprop_cell():
    contract = backprop_contract()
    with pytest.raises(IncompatibleCellError, match="input_clamp"):
        eqprop_protocol().validate_contract(contract)


def test_experiment_spec_checks_protocol_family_before_contract_details():
    contract = backprop_contract()
    local_experiment = ExperimentSpec(
        name="mnist_local_feature",
        supported_protocols=frozenset({ProtocolFamily.BACKPROP_LOCAL, ProtocolFamily.DFA}),
    )
    local_experiment.validate_contract(contract)

    eqprop_only = ExperimentSpec(
        name="mnist_eqprop_tile",
        supported_protocols=frozenset({ProtocolFamily.EQPROP}),
    )
    with pytest.raises(IncompatibleCellError, match="does not support"):
        eqprop_only.validate_contract(contract)


def test_differential_rail_requires_two_nodes_per_value():
    with pytest.raises(ValueError, match="expected 4"):
        rail(
            "bad_diff",
            Quantity.ERROR,
            ("p0", "n0", "p1"),
            width=2,
            signed=Signedness.DIFFERENTIAL,
        )


def test_experiment_linter_allows_cell_writes_initial_conditions_and_passive_probes():
    contract = backprop_contract()
    lint_experiment_elements(
        contract,
        (
            TaggedElement("Bupd_w", ("w0", "0"), ElementTag.CELL_WRITE),
            TaggedElement("IC_w", ("w0",), ElementTag.INITIAL_CONDITION),
            TaggedElement("P_hcap", ("hcap",), ElementTag.PASSIVE_PROBE, passive=True),
            TaggedElement("Vx0", ("x0", "0"), ElementTag.EXPERIMENT_INPUT),
        ),
    )


def test_experiment_linter_rejects_debug_force_in_headline_run():
    contract = backprop_contract()
    with pytest.raises(ExperimentLintError, match="debug_force"):
        lint_experiment_elements(
            contract,
            (TaggedElement("Vforce_h", ("hcap", "0"), ElementTag.DEBUG_FORCE),),
        )


def test_experiment_linter_rejects_external_source_on_internal_state():
    contract = backprop_contract()
    with pytest.raises(ExperimentLintError, match="internal state nodes"):
        lint_experiment_elements(
            contract,
            (TaggedElement("Vbad_h", ("hcap", "0"), ElementTag.EXPERIMENT_INPUT),),
        )


def test_experiment_linter_requires_passive_probe_flag():
    contract = backprop_contract()
    with pytest.raises(ExperimentLintError, match="passive=True"):
        lint_experiment_elements(
            contract,
            (TaggedElement("P_hcap", ("hcap",), ElementTag.PASSIVE_PROBE),),
        )
