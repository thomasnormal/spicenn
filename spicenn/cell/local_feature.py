from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .access import EmitMode, StateAccess
from .base import CellInstance
from .characterization import CharacterizationExpectation
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
from .protocol import LearningProtocol, ProtocolFamily, backprop_local_protocol, dfa_protocol


def _rail(name: str, quantity: Quantity, *, width: int = 1, signed: Signedness = Signedness.UNSIGNED) -> RailBundle:
    return RailBundle(name=name, width=width, domain=RailDomain.VOLTAGE, signed=signed, quantity=quantity)


SIGNED_ACTIVATIONS = frozenset(
    {
        "tanh",
        "diff-clipped-relu",
        "leaky-relu",
        "leaky-hardtanh",
        "centered-softplus",
        "softsign",
        "linear",
    }
)


def activation_signedness(mode: str) -> Signedness:
    return Signedness.SIGNED_SINGLE if mode in SIGNED_ACTIVATIONS else Signedness.UNSIGNED


def local_feature_contract(
    protocol: LearningProtocol,
    *,
    activation_mode: str,
    includes_fixed_feedback: bool = False,
) -> CellContract:
    learning_inputs = (
        {"output_error": _rail("output_error", Quantity.ERROR, width=10, signed=Signedness.SIGNED_SINGLE)}
        if protocol.family is ProtocolFamily.DFA
        else {"learning_in": _rail("learning_in", Quantity.ERROR, width=1, signed=Signedness.DIFFERENTIAL)}
    )
    state = [
        StateSpec("w", StateRole.TRAINABLE, ("w",), StateAccess.PUBLIC_INITIAL_CONDITION),
        StateSpec("b", StateRole.TRAINABLE, ("b",), StateAccess.PUBLIC_INITIAL_CONDITION),
        StateSpec("h_cap", StateRole.FORWARD_DYNAMIC, ("h",), StateAccess.PASSIVE_PROBE_ALLOWED),
        StateSpec("dh_cap", StateRole.BACKWARD_DYNAMIC, ("dhp", "dhn"), StateAccess.PASSIVE_PROBE_ALLOWED),
    ]
    if includes_fixed_feedback:
        state.append(StateSpec("feedback", StateRole.FIXED_RANDOM, ("fb",), StateAccess.EXPERIMENT_INTERNAL))
    return CellContract(
        ports=CellPorts(
            inputs={"x": _rail("x", Quantity.INPUT, width=1, signed=Signedness.SIGNED_SINGLE)},
            outputs={"h": _rail("h", Quantity.ACTIVATION, signed=activation_signedness(activation_mode))},
            learning_inputs=learning_inputs,
            phases={
                "pact": _rail("pact", Quantity.PHASE),
                "pbwd": _rail("pbwd", Quantity.PHASE),
                "pacc": _rail("pacc", Quantity.PHASE),
            },
            controls={"eta": _rail("eta", Quantity.CONTROL)},
        ),
        state=tuple(state),
        params={
            "relu_clip": ParamSpec("relu_clip", default=1.0),
            "relu_leak": ParamSpec("relu_leak", default=0.05),
            "softplus_beta": ParamSpec("softplus_beta", default=5.0),
            "synapse_clip": ParamSpec("synapse_clip", default=1.0),
        },
        protocol=protocol,
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


@dataclass(frozen=True)
class LocalFeatureCell:
    name: str
    description: str
    local_activation: str
    hidden_synapse_mode: str
    readout_synapse_mode: str = "linear"
    activation_derivative: str = "exact"
    readout_feedback_mode: str = "readout"
    relu_clip: float = 1.0
    relu_leak: float = 0.05
    softplus_beta: float = 5.0
    derivative_floor: float = 0.0
    derivative_gate_threshold: float = 1e-6
    synapse_clip: float = 1.0
    readout_feedback_clip: float = 0.1
    readout_update_scale: float = 1.0
    local_update_scale: float = 1.0
    output_bias_update_scale: float = 0.0
    softmax_temperature: float = 2.0
    protocol_family: ProtocolFamily = ProtocolFamily.BACKPROP_LOCAL

    def protocol(self) -> LearningProtocol:
        return dfa_protocol() if self.protocol_family is ProtocolFamily.DFA else backprop_local_protocol()

    def contract(self) -> CellContract:
        return local_feature_contract(
            self.protocol(),
            activation_mode=self.local_activation,
            includes_fixed_feedback=self.protocol_family is ProtocolFamily.DFA,
        )

    def emit(self, deck: Any, inst: str, bindings: Mapping[str, str], mode: EmitMode) -> CellInstance:
        # These first cells are reference/behavioral candidates.  They are
        # benchmarked by the NumPy MNIST runner and later mapped to SPICE
        # emitters one by one.
        return CellInstance(name=inst, contract=self.contract(), elements=())

    def reference_model(self) -> "LocalFeatureCell":
        return self

    def forward(self, inputs: Mapping[str, Any], state: Mapping[str, Any]) -> Mapping[str, Any]:
        z = np.asarray(inputs["z"])
        return {"h": activation_np(z, self.local_activation, self.relu_clip, self.relu_leak, self.softplus_beta)}

    def characterization_suite(self) -> list[CharacterizationExpectation]:
        return [
            CharacterizationExpectation("forward_monotone", "forward_monotonicity_violations", maximum=0),
            CharacterizationExpectation("update_sign", "update_sign_alignment", minimum=1.0),
            CharacterizationExpectation("update_cosine", "update_cosine", minimum=0.0),
        ]

    def update_kwargs(self) -> dict[str, object]:
        return {
            "local_activation": self.local_activation,
            "hidden_synapse_mode": self.hidden_synapse_mode,
            "readout_synapse_mode": self.readout_synapse_mode,
            "activation_derivative": self.activation_derivative,
            "readout_feedback_mode": self.readout_feedback_mode,
            "relu_clip": self.relu_clip,
            "relu_leak": self.relu_leak,
            "softplus_beta": self.softplus_beta,
            "derivative_floor": self.derivative_floor,
            "derivative_gate_threshold": self.derivative_gate_threshold,
            "synapse_clip": self.synapse_clip,
            "readout_feedback_clip": self.readout_feedback_clip,
            "readout_update_scale": self.readout_update_scale,
            "local_update_scale": self.local_update_scale,
            "output_bias_update_scale": self.output_bias_update_scale,
            "softmax_temperature": self.softmax_temperature,
        }


def activation_np(x: np.ndarray, mode: str, relu_clip: float, relu_leak: float, softplus_beta: float) -> np.ndarray:
    if mode == "tanh":
        return np.tanh(x)
    if mode == "relu":
        return np.maximum(x, 0.0)
    if mode == "clipped-relu":
        return np.clip(x, 0.0, relu_clip)
    if mode == "diff-clipped-relu":
        return np.clip(x, -relu_clip, relu_clip)
    if mode == "leaky-relu":
        return np.where(x >= 0.0, x, relu_leak * x)
    if mode == "leaky-hardtanh":
        return np.where(np.abs(x) <= relu_clip, x, np.sign(x) * (relu_clip + relu_leak * (np.abs(x) - relu_clip)))
    if mode == "softplus":
        beta = max(float(softplus_beta), 1e-12)
        bx = beta * x
        return (np.maximum(bx, 0.0) + np.log1p(np.exp(-np.abs(bx)))) / beta
    if mode == "centered-softplus":
        beta = max(float(softplus_beta), 1e-12)
        bx = beta * x
        return (np.maximum(bx, 0.0) + np.log1p(np.exp(-np.abs(bx))) - np.log(2.0)) / beta
    if mode == "softsign":
        return x / (1.0 + np.abs(x))
    if mode == "linear":
        return x
    raise ValueError(f"unknown activation mode {mode!r}")


def activation_derivative_np(
    pre: np.ndarray,
    h: np.ndarray,
    mode: str,
    relu_clip: float,
    relu_leak: float,
    softplus_beta: float,
    derivative_mode: str,
    derivative_floor: float,
    derivative_gate_threshold: float,
) -> np.ndarray:
    if derivative_mode == "unity":
        deriv = np.ones_like(pre)
    elif derivative_mode == "stored-gate":
        if mode == "tanh":
            deriv = 1.0 - h * h
        else:
            deriv = (np.abs(h) > max(float(derivative_gate_threshold), 0.0)).astype(float)
    else:
        if derivative_mode not in {"exact", "floor-exact"}:
            raise ValueError(f"unknown derivative mode {derivative_mode!r}")
        if mode == "tanh":
            deriv = 1.0 - h * h
        elif mode == "relu":
            deriv = (pre >= 0.0).astype(float)
        elif mode == "clipped-relu":
            deriv = ((pre >= 0.0) & (pre <= relu_clip)).astype(float)
        elif mode == "diff-clipped-relu":
            deriv = (np.abs(pre) <= relu_clip).astype(float)
        elif mode == "leaky-relu":
            deriv = np.where(pre >= 0.0, 1.0, relu_leak)
        elif mode == "leaky-hardtanh":
            deriv = np.where(np.abs(pre) <= relu_clip, 1.0, relu_leak)
        elif mode in {"softplus", "centered-softplus"}:
            beta = max(float(softplus_beta), 1e-12)
            deriv = 1.0 / (1.0 + np.exp(-beta * pre))
        elif mode == "softsign":
            deriv = 1.0 / np.square(1.0 + np.abs(pre))
        elif mode == "linear":
            deriv = np.ones_like(pre)
        else:
            raise ValueError(f"unknown activation mode {mode!r}")
    if derivative_mode in {"floor-exact", "stored-gate"} and derivative_floor > 0.0:
        floor = min(float(derivative_floor), 1.0)
        deriv = floor + (1.0 - floor) * deriv
    return deriv


def synapse_transfer_np(weight: np.ndarray, mode: str, clip: float) -> np.ndarray:
    if mode in {"linear", "full", "ideal"}:
        return weight
    if mode in {"tanh-clipped", "smooth-clipped", "clipped"}:
        return clip * np.tanh(weight / max(float(clip), 1e-12))
    if mode in {"hard-clipped", "bounded"}:
        return np.clip(weight, -clip, clip)
    if mode in {"sign", "binary"}:
        return clip * np.sign(weight)
    raise ValueError(f"unknown synapse mode {mode!r}")


LOCAL_FEATURE_CELLS: tuple[LocalFeatureCell, ...] = (
    LocalFeatureCell(
        name="stored_tanh_tanhclip",
        description="Current behavioral winner shape: stored tanh activation with tanh-clipped hidden synapses.",
        local_activation="tanh",
        hidden_synapse_mode="tanh-clipped",
        synapse_clip=1.0,
        softmax_temperature=2.0,
        readout_update_scale=0.5,
    ),
    LocalFeatureCell(
        name="stored_tanh_linear",
        description="Stored tanh activation with linear hidden/readout synapses.",
        local_activation="tanh",
        hidden_synapse_mode="linear",
    ),
    LocalFeatureCell(
        name="diffpair_tanh_softclip",
        description="Physical differential-pair hypothesis: tanh activation, soft-clipped weights, clipped feedback.",
        local_activation="tanh",
        hidden_synapse_mode="tanh-clipped",
        readout_feedback_mode="clipped-readout",
        readout_feedback_clip=0.25,
        synapse_clip=0.75,
    ),
    LocalFeatureCell(
        name="signed_hardtanh",
        description="Signed bounded hard-tanh proxy for differential clipped transport.",
        local_activation="diff-clipped-relu",
        hidden_synapse_mode="hard-clipped",
        synapse_clip=1.0,
        relu_clip=1.0,
    ),
    LocalFeatureCell(
        name="leaky_hardtanh",
        description="Bounded ReLU-family branch with nonzero saturated derivative.",
        local_activation="leaky-hardtanh",
        hidden_synapse_mode="hard-clipped",
        relu_clip=1.0,
        relu_leak=0.05,
        derivative_floor=0.02,
        activation_derivative="floor-exact",
    ),
    LocalFeatureCell(
        name="leaky_relu",
        description="Sparse-ish ReLU branch with nonzero negative-side transport.",
        local_activation="leaky-relu",
        hidden_synapse_mode="tanh-clipped",
        relu_leak=0.05,
    ),
    LocalFeatureCell(
        name="clipped_relu",
        description="Single-ended bounded ReLU proxy.",
        local_activation="clipped-relu",
        hidden_synapse_mode="tanh-clipped",
        relu_clip=1.0,
        activation_derivative="floor-exact",
        derivative_floor=0.02,
    ),
    LocalFeatureCell(
        name="softplus",
        description="Smooth noisy-ReLU proxy with sigmoid derivative.",
        local_activation="softplus",
        hidden_synapse_mode="tanh-clipped",
        softplus_beta=4.0,
    ),
    LocalFeatureCell(
        name="centered_softplus",
        description="Zero-centered softplus branch for smoother signed transport.",
        local_activation="centered-softplus",
        hidden_synapse_mode="tanh-clipped",
        softplus_beta=4.0,
    ),
    LocalFeatureCell(
        name="dfa_tanh_fixed_feedback",
        description="Direct feedback alignment branch with fixed random hidden error projection.",
        local_activation="tanh",
        hidden_synapse_mode="tanh-clipped",
        protocol_family=ProtocolFamily.DFA,
        readout_feedback_mode="fixed-random",
        synapse_clip=1.0,
    ),
)


def local_feature_cell_by_name(name: str) -> LocalFeatureCell:
    for cell in LOCAL_FEATURE_CELLS:
        if cell.name == name:
            return cell
    raise KeyError(f"unknown local feature cell {name!r}")
