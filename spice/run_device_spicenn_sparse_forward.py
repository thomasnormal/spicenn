from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    from spicenn import (
        CapState,
        FanInTopology,
        NetlistBuilder,
        SignedScoreErrorCell,
        SplitScoreCELimitedErrorBank,
        SourceId,
        make_sparse_differential_error_transport_layer,
        make_sparse_differential_relu_layer,
        make_sparse_hidden_update_layer,
        make_sparse_readout_update_layer,
        make_sparse_relu_delta_gate_layer,
        make_sparse_signed_readout_layer,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from spicenn import (
        CapState,
        FanInTopology,
        NetlistBuilder,
        SignedScoreErrorCell,
        SplitScoreCELimitedErrorBank,
        SourceId,
        make_sparse_differential_error_transport_layer,
        make_sparse_differential_relu_layer,
        make_sparse_hidden_update_layer,
        make_sparse_readout_update_layer,
        make_sparse_relu_delta_gate_layer,
        make_sparse_signed_readout_layer,
    )

from _util import parse_measures
from run_device_multicell_classifier import mos_models
from run_spice_sweep import ROOT, detect_spice, run_text_netlist, run_tiny_test


@dataclass(frozen=True)
class SparseForwardTopology:
    hidden1: FanInTopology
    hidden2: FanInTopology
    readout: FanInTopology


TOPOLOGY_MODES = ("fanin", "balanced_fanout", "ring_fanout")
ERROR_RULES = ("residual", "target_only", "ce_split_limited")
DEFAULT_SPIKE_REF_V = 0.02


def _source_slug(source: SourceId) -> str:
    text = str(source)
    slug = "".join(ch if ch.isalnum() else "_" for ch in text)
    return slug or "src"


def _validate_topology_mode(name: str, mode: str) -> None:
    if mode not in TOPOLOGY_MODES:
        choices = ", ".join(TOPOLOGY_MODES)
        raise ValueError(f"unknown {name} topology mode {mode!r}; choices: {choices}")


def _add_always_source(topology: FanInTopology, source: SourceId) -> FanInTopology:
    return FanInTopology.from_fanins(
        (source, *topology.sources),
        topology.sink_count,
        {sink: (source, *sources) for sink, sources in topology.as_fanins().items()},
    )


def build_topology(
    *,
    input_sources: tuple[SourceId, ...] = ("x0", "x1"),
    hidden_count: int,
    output_count: int,
    hidden1_fan_in: int,
    hidden2_fan_in: int,
    readout_fan_in: int,
    seed: int,
    readout_bias: bool = False,
    hidden2_topology_mode: str = "fanin",
    readout_topology_mode: str = "fanin",
) -> SparseForwardTopology:
    """Topology for a small two-hidden-layer sparse forward smoke."""

    if hidden_count <= 0 or output_count <= 0:
        raise ValueError("hidden_count and output_count must be positive")
    if not input_sources:
        raise ValueError("input_sources must be nonempty")
    if "bias" in input_sources:
        raise ValueError("input_sources must not include the reserved bias source")
    _validate_topology_mode("hidden2", hidden2_topology_mode)
    _validate_topology_mode("readout", readout_topology_mode)
    hidden1_sources = tuple(input_sources)
    hidden1 = FanInTopology.random_fanin(
        hidden1_sources,
        hidden_count,
        seed=seed,
        fan_in=min(hidden1_fan_in, len(hidden1_sources)),
        always_sources=("bias",),
    )
    hidden2_sources = tuple(range(hidden_count))
    if hidden2_topology_mode == "balanced_fanout":
        hidden2 = FanInTopology.balanced_random_fanout(
            hidden2_sources,
            hidden_count,
            seed=seed + 101,
            fan_out=min(hidden2_fan_in, hidden_count),
        )
    elif hidden2_topology_mode == "ring_fanout":
        hidden2 = FanInTopology.ring_fanout(
            hidden2_sources,
            hidden_count,
            fan_out=min(hidden2_fan_in, hidden_count),
            offset=seed + 101,
        )
    else:
        hidden2 = FanInTopology.random_fanin(
            hidden2_sources,
            hidden_count,
            seed=seed + 101,
            fan_in=min(hidden2_fan_in, hidden_count),
        )
    readout_sources = tuple(range(hidden_count))
    if readout_topology_mode == "balanced_fanout":
        readout = FanInTopology.balanced_random_fanout(
            readout_sources,
            output_count,
            seed=seed + 1009,
            fan_out=min(readout_fan_in, output_count),
        )
        if readout_bias:
            readout = _add_always_source(readout, "bias")
    elif readout_topology_mode == "ring_fanout":
        readout = FanInTopology.ring_fanout(
            readout_sources,
            output_count,
            fan_out=min(readout_fan_in, output_count),
            offset=seed + 1009,
        )
        if readout_bias:
            readout = _add_always_source(readout, "bias")
    else:
        readout_always_sources: tuple[SourceId, ...] = ("bias",) if readout_bias else ()
        readout = FanInTopology.random_fanin(
            readout_sources,
            output_count,
            seed=seed + 1009,
            fan_in=min(readout_fan_in, hidden_count),
            always_sources=readout_always_sources,
        )
    return SparseForwardTopology(hidden1=hidden1, hidden2=hidden2, readout=readout)


def readout_initials(topology: FanInTopology) -> dict[tuple[int, SourceId], tuple[float, float]]:
    """Program a deterministic signed readout separator for transient smoke tests."""

    initials: dict[tuple[int, SourceId], tuple[float, float]] = {}
    for out, sources in topology.as_fanins().items():
        for source in sources:
            if out == 0:
                initials[(out, source)] = (0.92, 0.08)
            else:
                initials[(out, source)] = (0.08, 0.92)
    return initials


def centered_readout_initials(
    topology: FanInTopology,
    center_v: float = 0.30,
) -> dict[tuple[int, SourceId], tuple[float, float]]:
    return {
        (out, source): (center_v, center_v)
        for out, sources in topology.as_fanins().items()
        for source in sources
    }


def random_jittered_readout_initials(
    topology: FanInTopology,
    *,
    seed: int,
    center_v: float = 0.30,
    signed_span_v: float = 0.02,
) -> dict[tuple[int, SourceId], tuple[float, float]]:
    """Small deterministic signed offsets for symmetry breaking.

    A perfectly centered readout has zero signed reverse gain, so hidden
    backprop starts as common-mode.  This programs a tiny capacitor mismatch
    directly into the SPICE netlist, representing ordinary device mismatch or
    deliberate startup dither.
    """

    if signed_span_v < 0:
        raise ValueError("readout signed jitter span must be nonnegative")
    half_span = signed_span_v / 2.0
    if center_v - half_span < 0.0 or center_v + half_span > 1.2:
        raise ValueError("readout jitter and center put caps outside the 0..1.2 V range")
    rng = random.Random(seed)
    initials: dict[tuple[int, SourceId], tuple[float, float]] = {}
    for out, sources in topology.as_fanins().items():
        for source in sources:
            offset = rng.uniform(-half_span, half_span)
            initials[(out, source)] = (center_v + offset, center_v - offset)
    return initials


def signed_offset_readout_initials(
    topology: FanInTopology,
    *,
    target_output: int,
    signed_delta_v: float,
    center_v: float = 0.55,
    mirror_other_outputs: bool = True,
) -> dict[tuple[int, SourceId], tuple[float, float]]:
    """Program known differential readout offsets for readout sensitivity checks.

    ``signed_delta_v`` is the per-edge capacitor difference ``vw+ - vw-`` on the
    target row.  Other rows either stay centered or receive the mirrored
    negative offset so a two-class margin check has a useful opposite row.
    """

    if not 0 <= target_output < topology.sink_count:
        raise ValueError("target_output must be a valid readout row")
    half = signed_delta_v / 2.0
    high = center_v + abs(half)
    low = center_v - abs(half)
    if low < 0.0 or high > 1.2:
        raise ValueError("signed_delta_v and center_v put readout caps outside the 0..1.2 V range")

    initials: dict[tuple[int, SourceId], tuple[float, float]] = {}
    for out, sources in topology.as_fanins().items():
        row_sign = 1.0 if out == target_output else (-1.0 if mirror_other_outputs else 0.0)
        for source in sources:
            offset = row_sign * half
            initials[(out, source)] = (center_v + offset, center_v - offset)
    return initials


def random_signed_hidden_initials(
    topology: FanInTopology,
    *,
    seed: int,
    positive_sources: tuple[str | int, ...] = ("bias",),
    high_v: float = 0.88,
    low_v: float = 0.04,
) -> dict[tuple[int, str | int], tuple[float, float]]:
    rng = random.Random(seed)
    positive_set = set(positive_sources)
    initials: dict[tuple[int, str | int], tuple[float, float]] = {}
    for sink, sources in topology.as_fanins().items():
        for source in sources:
            if source in positive_set or rng.choice((False, True)):
                initials[(sink, source)] = (high_v, low_v)
            else:
                initials[(sink, source)] = (low_v, high_v)
    return initials


def centered_hidden_initials(
    topology: FanInTopology,
    *,
    center_v: float = 0.40,
) -> dict[tuple[int, SourceId], tuple[float, float]]:
    return {
        (sink, source): (center_v, center_v)
        for sink, sources in topology.as_fanins().items()
        for source in sources
    }


def random_centered_hidden_initials(
    topology: FanInTopology,
    *,
    seed: int,
    center_v: float = 0.40,
    signed_span_v: float = 0.16,
) -> dict[tuple[int, SourceId], tuple[float, float]]:
    """Centered hidden weights with small signed startup mismatch.

    Exact centering keeps trainable hidden weights in the useful mobility
    window, but it also makes the hidden2 layer nearly featureless at startup.
    This keeps the same common-mode state while injecting a deterministic
    signed offset, matching the device-mismatch/dither we expect in hardware.
    """

    if signed_span_v < 0:
        raise ValueError("hidden signed jitter span must be nonnegative")
    half_span = signed_span_v / 2.0
    if center_v - half_span < 0.0 or center_v + half_span > 1.2:
        raise ValueError("hidden jitter and center put caps outside the 0..1.2 V range")
    rng = random.Random(seed)
    initials: dict[tuple[int, SourceId], tuple[float, float]] = {}
    for sink, sources in topology.as_fanins().items():
        for source in sources:
            offset = rng.uniform(-half_span, half_span)
            initials[(sink, source)] = (center_v + offset, center_v - offset)
    return initials


def positive_hidden_initials(
    topology: FanInTopology,
    *,
    pos_v: float = 0.84,
    neg_v: float = 0.04,
) -> dict[tuple[int, SourceId], tuple[float, float]]:
    return {
        (sink, source): (pos_v, neg_v)
        for sink, sources in topology.as_fanins().items()
        for source in sources
    }


def hidden1_initials_for_mode(
    topology: FanInTopology,
    *,
    hidden_weight_mode: str,
    seed: int,
) -> dict[tuple[int, SourceId], tuple[float, float]] | None:
    random_bias = hidden_weight_mode.endswith("_random_bias")
    if hidden_weight_mode == "centered_all":
        return centered_hidden_initials(topology)
    if hidden_weight_mode == "centered_jittered_all":
        return random_centered_hidden_initials(topology, seed=seed + 17)
    if hidden_weight_mode in {"weak_signed_all_random_bias", "weak_signed_hidden1_random_bias"}:
        return random_signed_hidden_initials(
            topology,
            seed=seed + 17,
            positive_sources=(),
            high_v=0.68,
            low_v=0.22,
        )
    if hidden_weight_mode in {
        "signed_hidden1",
        "signed_all",
        "signed_hidden1_random_bias",
        "signed_all_random_bias",
    }:
        return random_signed_hidden_initials(
            topology,
            seed=seed + 17,
            positive_sources=() if random_bias else ("bias",),
        )
    return None


def hidden2_initials_for_mode(
    topology: FanInTopology,
    *,
    hidden_weight_mode: str,
    seed: int,
) -> dict[tuple[int, SourceId], tuple[float, float]]:
    random_bias = hidden_weight_mode.endswith("_random_bias")
    if hidden_weight_mode in {"centered_hidden2", "centered_all"}:
        return centered_hidden_initials(topology)
    if hidden_weight_mode in {"centered_jittered_hidden2", "centered_jittered_all"}:
        return random_centered_hidden_initials(topology, seed=seed + 37)
    if hidden_weight_mode == "weak_signed_all_random_bias":
        return random_signed_hidden_initials(
            topology,
            seed=seed + 37,
            positive_sources=(),
            high_v=0.68,
            low_v=0.22,
        )
    if hidden_weight_mode in {"signed_all", "signed_all_random_bias"}:
        return random_signed_hidden_initials(
            topology,
            seed=seed + 37,
            positive_sources=() if random_bias else ("bias",),
        )
    return positive_hidden_initials(topology)


def effective_hidden_weight_mode_for_updates(hidden_weight_mode: str, hidden2_update_width_u: float) -> str:
    """Keep trainable hidden weights in the centered mobility window by default."""

    if hidden2_update_width_u > 0 and hidden_weight_mode == "positive":
        return "centered_jittered_hidden2"
    return hidden_weight_mode


def readout_source_nodes(topology: FanInTopology) -> dict[SourceId, str]:
    return {source: ("vdd" if source == "bias" else f"h2_{source}") for source in topology.sources}


def hidden1_source_nodes(topology: FanInTopology) -> dict[SourceId, str]:
    return {source: ("bias" if source == "bias" else str(source)) for source in topology.sources}


def default_input_values(x0: float | None, x1: float | None) -> dict[SourceId, float]:
    if x0 is None or x1 is None:
        raise ValueError("x0 and x1 are required when input_values is not supplied")
    return {"x0": float(x0), "x1": float(x1)}


def input_sources_from_values(input_values: Mapping[SourceId, float]) -> tuple[SourceId, ...]:
    if not input_values:
        raise ValueError("input_values must be nonempty")
    return tuple(input_values.keys())


def sample_input_values(sample: dict[str, Any]) -> dict[SourceId, float]:
    if "inputs" in sample:
        inputs = sample["inputs"]
        if not isinstance(inputs, dict):
            raise ValueError("sample inputs must be a mapping")
        if "input_rails" in sample:
            return {rail: float(inputs[rail]) for rail in sample["input_rails"]}
        return {source: float(value) for source, value in inputs.items()}
    return default_input_values(float(sample["x0"]), float(sample["x1"]))


def input_sources_from_samples(samples: list[dict[str, Any]]) -> tuple[SourceId, ...]:
    if not samples:
        return ("x0", "x1")
    first = samples[0]
    if "input_rails" in first:
        return tuple(first["input_rails"])
    return input_sources_from_values(sample_input_values(first))


def sample_display_fields(sample: dict[str, Any]) -> dict[str, Any]:
    inputs = sample_input_values(sample)
    fields: dict[str, Any] = {
        "inputs": {str(source): value for source, value in inputs.items()},
    }
    if "x0" in inputs:
        fields["x0"] = inputs["x0"]
    if "x1" in inputs:
        fields["x1"] = inputs["x1"]
    for key in ("pattern", "source_index", "source_digit", "input_frontend_key"):
        if key in sample:
            fields[key] = sample[key]
    return fields


def input_voltage_sources(input_values: Mapping[SourceId, float]) -> str:
    lines = []
    for source, value in input_values.items():
        slug = _source_slug(source)
        lines.append(f"Vin_{slug} {source} 0 DC {float(value):.12g}")
    return "\n".join(lines)


def render_network(
    topology: SparseForwardTopology,
    *,
    seed: int = 0,
    hidden_weight_mode: str = "positive",
    weight_cap_f: float = 4.0,
    hidden_cap_f: float = 10.0,
    activation_cap_f: float = 8.0,
    score_cap_f: float = 10.0,
    hidden_width_u: float = 8.0,
    readout_width_u: float = 10.0,
    readout_branch_style: str = "gate_stack",
    readout_weight_initials: dict[tuple[int, SourceId], tuple[float, float]] | None = None,
    hidden1_weight_initials: dict[tuple[int, SourceId], tuple[float, float]] | None = None,
    hidden2_weight_initials: dict[tuple[int, SourceId], tuple[float, float]] | None = None,
) -> str:
    hidden_weight_modes = {
        "positive",
        "signed_hidden1",
        "signed_all",
        "signed_hidden1_random_bias",
        "signed_all_random_bias",
        "weak_signed_hidden1_random_bias",
        "weak_signed_all_random_bias",
        "centered_hidden2",
        "centered_jittered_hidden2",
        "centered_all",
        "centered_jittered_all",
    }
    if hidden_weight_mode not in hidden_weight_modes:
        raise ValueError(f"unknown hidden_weight_mode: {hidden_weight_mode}")
    hidden1_initials = (
        hidden1_weight_initials
        if hidden1_weight_initials is not None
        else hidden1_initials_for_mode(topology.hidden1, hidden_weight_mode=hidden_weight_mode, seed=seed)
    )
    hidden2_initials = (
        hidden2_weight_initials
        if hidden2_weight_initials is not None
        else hidden2_initials_for_mode(topology.hidden2, hidden_weight_mode=hidden_weight_mode, seed=seed)
    )
    hidden1 = make_sparse_differential_relu_layer(
        "h1",
        topology=topology.hidden1,
        source_nodes=hidden1_source_nodes(topology.hidden1),
        activation_prefix="h1_",
        preactivation_prefix="u1_",
        weight_prefix="w1",
        synapse_prefix="s1",
        weight_cap_f=weight_cap_f,
        preactivation_cap_f=hidden_cap_f,
        activation_cap_f=activation_cap_f,
        weight_pos_ic_v=0.88,
        weight_neg_ic_v=0.04,
        weight_initials=hidden1_initials,
        synapse_width_u=hidden_width_u,
        neuron_width_u=24.0,
    )
    hidden2 = make_sparse_differential_relu_layer(
        "h2",
        topology=topology.hidden2,
        source_nodes={source: f"h1_{source}" for source in topology.hidden2.sources},
        activation_prefix="h2_",
        preactivation_prefix="u2_",
        weight_prefix="w2",
        synapse_prefix="s2",
        weight_cap_f=weight_cap_f,
        preactivation_cap_f=hidden_cap_f,
        activation_cap_f=activation_cap_f,
        weight_pos_ic_v=0.84,
        weight_neg_ic_v=0.04,
        weight_initials=hidden2_initials,
        synapse_width_u=hidden_width_u,
        neuron_width_u=24.0,
    )
    readout = make_sparse_signed_readout_layer(
        "readout",
        topology=topology.readout,
        source_nodes=readout_source_nodes(topology.readout),
        score_prefix="score",
        weight_prefix="vw",
        branch_prefix="ro",
        weight_cap_f=weight_cap_f,
        score_cap_f=score_cap_f,
        weight_initials=readout_initials(topology.readout) if readout_weight_initials is None else readout_weight_initials,
        branch_style=readout_branch_style,
        branch_width_u=readout_width_u,
    )

    deck = NetlistBuilder()
    deck.render_component(hidden1)
    deck.render_component(hidden2)
    deck.render_component(readout)
    return deck.render_body()


def measurement_lines(hidden_count: int, output_count: int, *, measure_time_ns: float = 2.2) -> str:
    at = f"{measure_time_ns:.12g}n"
    lines: list[str] = []
    for h in range(hidden_count):
        lines += [
            f".meas tran h1_{h} FIND V(h1_{h}) AT={at}",
            f".meas tran h2_{h} FIND V(h2_{h}) AT={at}",
        ]
    for out in range(output_count):
        lines += [
            f".meas tran score{out}_p FIND V(score{out}p) AT={at}",
            f".meas tran score{out}_n FIND V(score{out}n) AT={at}",
        ]
    return "\n".join(lines)


def add_derived_measures(measures: dict[str, float], output_count: int) -> dict[str, float]:
    enriched = dict(measures)
    for out in range(output_count):
        p = enriched.get(f"score{out}_p")
        n = enriched.get(f"score{out}_n")
        if p is not None and n is not None:
            enriched[f"score{out}_diff"] = p - n
    if output_count >= 2 and "score0_diff" in enriched and "score1_diff" in enriched:
        enriched["score_margin_0_1"] = enriched["score0_diff"] - enriched["score1_diff"]
    return enriched


def add_train_derived_measures(
    measures: dict[str, float],
    topology: SparseForwardTopology,
) -> dict[str, float]:
    enriched = add_derived_measures(measures, topology.readout.sink_count)
    for out, sources in topology.readout.as_fanins().items():
        signed_delta = 0.0
        common_delta = 0.0
        for source in sources:
            base = f"vw{out}_{source}"
            p_before = enriched[f"{base}p_before"]
            n_before = enriched[f"{base}n_before"]
            p_after = enriched[f"{base}p_after"]
            n_after = enriched[f"{base}n_after"]
            signed_delta += (p_after - n_after) - (p_before - n_before)
            common_delta += (p_after - p_before) + (n_after - n_before)
        enriched[f"row{out}_signed_delta"] = signed_delta
        enriched[f"row{out}_common_delta"] = common_delta
    for sink, sources in topology.hidden2.as_fanins().items():
        signed_delta = 0.0
        common_delta = 0.0
        seen = False
        for source in sources:
            base = f"w2{sink}_{source}"
            before_pos = enriched.get(f"{base}p_before")
            before_neg = enriched.get(f"{base}n_before")
            after_pos = enriched.get(f"{base}p_after")
            after_neg = enriched.get(f"{base}n_after")
            if before_pos is None or before_neg is None or after_pos is None or after_neg is None:
                continue
            signed_delta += (after_pos - after_neg) - (before_pos - before_neg)
            common_delta += (after_pos - before_pos) + (after_neg - before_neg)
            seen = True
        if seen:
            enriched[f"hidden2_row{sink}_signed_delta"] = signed_delta
            enriched[f"hidden2_row{sink}_common_delta"] = common_delta
    return enriched


def activation_vector(measures: dict[str, float], prefix: str, count: int) -> list[float]:
    return [float(measures.get(f"{prefix}_{index}", 0.0)) for index in range(count)]


def pretrace_gate_summary(
    pretrace_gates: Mapping[str, Mapping[str, float | None]],
    *,
    active_threshold: float = 0.8,
    weak_threshold: float = 1e-3,
) -> dict[str, int]:
    """Summarize hybrid pretrace coverage from per-edge gate measurements."""

    total = 0
    active = 0
    weak = 0
    missing = 0
    for gates in pretrace_gates.values():
        total += 1
        gate = gates.get("gate")
        if gate is None:
            missing += 1
            continue
        if gate > active_threshold:
            active += 1
        elif gate > weak_threshold:
            weak += 1
    return {
        "total": total,
        "active": active,
        "weak": weak,
        "inactive": total - active - weak - missing,
        "missing": missing,
    }


def vector_l1(a: list[float], b: list[float]) -> float:
    return sum(abs(left - right) for left, right in zip(a, b))


def vector_l2(a: list[float], b: list[float]) -> float:
    return sum((left - right) ** 2 for left, right in zip(a, b)) ** 0.5


def readout_hidden_coverage(
    h2: list[float],
    topology: FanInTopology,
    *,
    active_threshold: float = 1e-3,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for out, sources in topology.as_fanins().items():
        hidden_sources = [source for source in sources if isinstance(source, int)]
        active_values = [float(h2[source]) for source in hidden_sources if source < len(h2)]
        rows.append(
            {
                "output": out,
                "fanin": len(hidden_sources),
                "active_count": sum(1 for value in active_values if value > active_threshold),
                "active_sum": sum(max(0.0, value) for value in active_values),
            }
        )
    return rows


def extract_readout_initials_after(
    measures: dict[str, float],
    topology: SparseForwardTopology,
) -> dict[tuple[int, SourceId], tuple[float, float]]:
    return {
        (out, source): (
            measures[f"vw{out}_{source}p_after"],
            measures[f"vw{out}_{source}n_after"],
        )
        for out, sources in topology.readout.as_fanins().items()
        for source in sources
    }


def extract_hidden2_initials_after(
    measures: dict[str, float],
    topology: SparseForwardTopology,
) -> dict[tuple[int, SourceId], tuple[float, float]]:
    return {
        (sink, source): (
            measures[f"w2{sink}_{source}p_after"],
            measures[f"w2{sink}_{source}n_after"],
        )
        for sink, sources in topology.hidden2.as_fanins().items()
        for source in sources
    }


def summarize_readout_signed_weights(
    weights: dict[tuple[int, SourceId], tuple[float, float]],
    topology: SparseForwardTopology,
) -> dict[str, float]:
    summary: dict[str, float] = {}
    for out, sources in topology.readout.as_fanins().items():
        signed_sum = 0.0
        common_sum = 0.0
        for source in sources:
            pos, neg = weights[(out, source)]
            signed_sum += pos - neg
            common_sum += pos + neg
        summary[f"row{out}_signed_sum"] = signed_sum
        summary[f"row{out}_common_sum"] = common_sum
    return summary


def summarize_hidden2_signed_weights(
    weights: dict[tuple[int, SourceId], tuple[float, float]],
    topology: SparseForwardTopology,
) -> dict[str, float]:
    summary: dict[str, float] = {}
    total_signed = 0.0
    total_common = 0.0
    for sink, sources in topology.hidden2.as_fanins().items():
        signed_sum = 0.0
        common_sum = 0.0
        for source in sources:
            pos, neg = weights[(sink, source)]
            signed_sum += pos - neg
            common_sum += pos + neg
        summary[f"hidden2_row{sink}_signed_sum"] = signed_sum
        summary[f"hidden2_row{sink}_common_sum"] = common_sum
        total_signed += signed_sum
        total_common += common_sum
    summary["hidden2_total_signed_sum"] = total_signed
    summary["hidden2_total_common_sum"] = total_common
    return summary


def readout_train_extras(
    topology: SparseForwardTopology,
    *,
    label: int,
    target_high_v: float,
    target_low_v: float,
    false_negative_target_v: float,
    error_cap_f: float,
    error_rule: str = "residual",
    update_width_u: float,
    charge_width_u: float | None,
    discharge_width_u: float | None,
    update_write_mode: str,
    readout_bias_update_scale: float = 1.0,
    hidden2_update_width_u: float = 0.0,
    hidden2_delta_mode: str = "relu_gate",
    hidden2_update_write_mode: str = "inhibit_charge_discharge",
    hidden2_update_selector_width_u: float = 2.0,
) -> str:
    output_count = topology.readout.sink_count
    if error_rule not in ERROR_RULES:
        raise ValueError(f"unknown error_rule: {error_rule}")
    if readout_bias_update_scale < 0:
        raise ValueError("readout_bias_update_scale must be nonnegative")
    deck = NetlistBuilder()
    if error_rule == "ce_split_limited":
        deck.render_component(
            SplitScoreCELimitedErrorBank(
                "ceerr",
                output_count=output_count,
                target_node_prefix="t",
                nontarget_node_prefix="nt",
                positive_error_prefix="dp",
                negative_error_prefix="dn",
                error_cap_f=error_cap_f,
                target_width_u=32.0,
                nontarget_width_u=24.0,
            )
        )
    else:
        for out in range(output_count):
            quiet_non_label = error_rule == "target_only" and out != label
            error = SignedScoreErrorCell(
                f"err{out}",
                target_node=f"t{out}",
                negative_target_node=(f"tn{out}" if false_negative_target_v > 0.0 else None),
                score_pos_node=f"score{out}p",
                score_neg_node=f"score{out}n",
                positive_error=CapState(f"dp{out}", f"dp{out}", error_cap_f, leak_to="0", leak_ohm="1G"),
                negative_error=CapState(f"dn{out}", f"dn{out}", error_cap_f, leak_to="0", leak_ohm="1G"),
                target_width_u=0.0 if quiet_non_label else 32.0,
                negative_target_width_u=0.0 if quiet_non_label else None,
                score_width_u=0.0 if quiet_non_label else 24.0,
            )
            deck.render_component(error)
    transport = make_sparse_differential_error_transport_layer(
        "readout_backprop",
        topology=topology.readout,
        positive_error_prefix="dp",
        negative_error_prefix="dn",
        delta_prefix="hd",
        weight_prefix="vw",
        transport_prefix="bt",
        transport_width_u=4.0,
        backward_gate="bwd",
        skip_sources=("bias",),
    )
    deck.render_component(transport)
    hidden2_delta_prefix = "hd"
    if hidden2_update_width_u > 0:
        if hidden2_delta_mode not in {"raw", "relu_gate"}:
            raise ValueError(f"unknown hidden2_delta_mode: {hidden2_delta_mode}")
        if hidden2_delta_mode == "relu_gate":
            hidden2_delta_prefix = "gdh2_"
            delta_gate = make_sparse_relu_delta_gate_layer(
                "h2_delta_gate",
                sink_count=topology.hidden2.sink_count,
                activation_nodes={sink: f"h2_{sink}" for sink in range(topology.hidden2.sink_count)},
                input_delta_prefix="hd",
                output_delta_prefix=hidden2_delta_prefix,
                gate_width_u=8.0,
                backward_gate="bwd",
            )
            deck.render_component(delta_gate)
        hidden2_update = make_sparse_hidden_update_layer(
            "hidden2_update",
            topology=topology.hidden2,
            source_nodes={source: f"h1_{source}" for source in topology.hidden2.sources},
            weight_prefix="w2",
            update_prefix="uh2",
            delta_prefix=hidden2_delta_prefix,
            update_width_u=hidden2_update_width_u,
            charge_width_u=hidden2_update_width_u,
            discharge_width_u=hidden2_update_width_u,
            pos_low_node="0",
            neg_low_node="0",
            write_mode=hidden2_update_write_mode,
            selector_width_u=hidden2_update_selector_width_u,
        )
        deck.render_component(hidden2_update)
    update = make_sparse_readout_update_layer(
        "readout_update",
        topology=topology.readout,
        source_nodes=readout_source_nodes(topology.readout),
        weight_prefix="vw",
        update_prefix="uw",
        update_width_u=update_width_u,
        charge_width_u=charge_width_u,
        discharge_width_u=discharge_width_u,
        source_update_scales={"bias": readout_bias_update_scale} if "bias" in topology.readout.sources else None,
        write_mode=update_write_mode,
    )
    deck.render_component(update)
    target_sources = "\n".join(
        f"Vt{out} t{out} 0 DC {(target_high_v if out == label else target_low_v):.12g}"
        for out in range(output_count)
    )
    nontarget_sources = ""
    if error_rule == "ce_split_limited":
        nontarget_sources = "\n".join(
            f"Vnt{out} nt{out} 0 DC {(target_low_v if out == label else target_high_v):.12g}"
            for out in range(output_count)
        )
    negative_target_sources = ""
    if false_negative_target_v > 0.0:
        negative_target_sources = "\n".join(
            f"Vtn{out} tn{out} 0 DC {(0.0 if out == label else false_negative_target_v):.12g}"
            for out in range(output_count)
        )
    source_blocks = [target_sources]
    if nontarget_sources:
        source_blocks.append(nontarget_sources)
    if negative_target_sources:
        source_blocks.append(negative_target_sources)
    source_blocks.append(deck.render_body())
    return "\n".join(source_blocks)


def train_measurement_lines(
    topology: SparseForwardTopology,
    output_count: int,
    *,
    include_selectors: bool = False,
    include_competition_error: bool = False,
    include_hidden2_updates: bool = False,
    include_hidden2_gated_delta: bool = False,
    include_hidden2_selectors: bool = False,
    include_hidden2_selector_bars: bool = False,
    include_hidden2_activity_gate: bool = False,
) -> str:
    lines = [
        measurement_lines(topology.hidden1.sink_count, output_count),
    ]
    for out in range(output_count):
        lines += [
            f".meas tran dp{out}_err FIND V(dp{out}) AT=3.15n",
            f".meas tran dn{out}_err FIND V(dn{out}) AT=3.15n",
        ]
        if include_selectors:
            lines += [
                f".meas tran rwpos{out}_gate FIND V(rwpos{out}) AT=4.35n",
                f".meas tran rwneg{out}_gate FIND V(rwneg{out}) AT=4.35n",
                f".meas tran rwsel{out}_posbar_gate FIND V(rwsel{out}_posbar) AT=4.35n",
                f".meas tran rwsel{out}_negbar_gate FIND V(rwsel{out}_negbar) AT=4.35n",
            ]
        if include_competition_error:
            lines.append(f".meas tran ybar{out}_err FIND V(ybar{out}) AT=3.15n")
        for source in topology.readout.as_fanins()[out]:
            base = f"vw{out}_{source}"
            lines += [
                f".meas tran {base}p_before FIND V({base}p) AT=3.35n",
                f".meas tran {base}n_before FIND V({base}n) AT=3.35n",
                f".meas tran {base}p_after FIND V({base}p) AT=5.55n",
                f".meas tran {base}n_after FIND V({base}n) AT=5.55n",
            ]
            if include_selectors:
                lines += [
                    f".meas tran fprg{out}_{source}_gate FIND V(fprg{out}_{source}) AT=3.35n",
                    f".meas tran fprbar{out}_{source}_gate FIND V(fprbar{out}_{source}) AT=3.35n",
                ]
    for source in topology.readout.sources:
        if source == "bias":
            continue
        lines += [
            f".meas tran hd{source}_p FIND V(hd{source}p) AT=4.35n",
            f".meas tran hd{source}_n FIND V(hd{source}n) AT=4.35n",
        ]
        if include_hidden2_gated_delta:
            lines += [
                f".meas tran gdh2_{source}_p FIND V(gdh2_{source}p) AT=4.35n",
                f".meas tran gdh2_{source}_n FIND V(gdh2_{source}n) AT=4.35n",
            ]
        if include_hidden2_selectors:
            lines += [
                f".meas tran hwpos{source}_gate FIND V(hwpos{source}) AT=4.35n",
                f".meas tran hwneg{source}_gate FIND V(hwneg{source}) AT=4.35n",
            ]
        if include_hidden2_selector_bars:
            lines += [
                f".meas tran hwsel{source}_posbar_gate FIND V(hwsel{source}_posbar) AT=4.35n",
                f".meas tran hwsel{source}_negbar_gate FIND V(hwsel{source}_negbar) AT=4.35n",
            ]
        if include_hidden2_activity_gate:
            lines.append(f".meas tran hwsel{source}_active_gate FIND V(hwsel{source}_active) AT=4.35n")
    if include_competition_error:
        lines += [
            ".meas tran ctsrc_err FIND V(ctsrc) AT=3.15n",
            ".meas tran cesrc_err FIND V(cesrc) AT=3.15n",
            ".meas tran ccsrc_err FIND V(ccsrc) AT=3.15n",
        ]
    if include_hidden2_updates:
        for sink, sources in topology.hidden2.as_fanins().items():
            for source in sources:
                base = f"w2{sink}_{source}"
                lines += [
                    f".meas tran {base}p_before FIND V({base}p) AT=3.35n",
                    f".meas tran {base}n_before FIND V({base}n) AT=3.35n",
                    f".meas tran {base}p_after FIND V({base}p) AT=5.55n",
                    f".meas tran {base}n_after FIND V({base}n) AT=5.55n",
                ]
    return "\n".join(lines)


def readout_sensitivity_netlist(
    *,
    x0: float | None,
    x1: float | None,
    input_values: Mapping[SourceId, float] | None = None,
    input_sources: tuple[SourceId, ...] | None = None,
    bias: float,
    hidden_count: int,
    output_count: int,
    hidden1_fan_in: int,
    hidden2_fan_in: int,
    readout_fan_in: int,
    seed: int,
    signed_delta_v: float,
    hidden2_topology_mode: str = "fanin",
    readout_topology_mode: str = "fanin",
    target_output: int = 0,
    center_v: float = 0.55,
    mirror_other_outputs: bool = True,
    readout_width_u: float = 10.0,
    score_cap_f: float = 10.0,
    readout_branch_style: str = "gate_stack",
    measure_time_ns: float = 2.2,
) -> tuple[str, SparseForwardTopology, dict[tuple[int, SourceId], tuple[float, float]]]:
    resolved_inputs = default_input_values(x0, x1) if input_values is None else dict(input_values)
    resolved_sources = tuple(input_sources) if input_sources is not None else input_sources_from_values(resolved_inputs)
    topology = build_topology(
        input_sources=resolved_sources,
        hidden_count=hidden_count,
        output_count=output_count,
        hidden1_fan_in=hidden1_fan_in,
        hidden2_fan_in=hidden2_fan_in,
        readout_fan_in=readout_fan_in,
        seed=seed,
        hidden2_topology_mode=hidden2_topology_mode,
        readout_topology_mode=readout_topology_mode,
    )
    weights = signed_offset_readout_initials(
        topology.readout,
        target_output=target_output,
        signed_delta_v=signed_delta_v,
        center_v=center_v,
        mirror_other_outputs=mirror_other_outputs,
    )
    circuit = render_network(
        topology,
        score_cap_f=score_cap_f,
        readout_width_u=readout_width_u,
        readout_branch_style=readout_branch_style,
        readout_weight_initials=weights,
    )
    text = f"""
* spicenn readout signed-weight sensitivity primitive
.param VDD=1.2
{mos_models()}
Vdd vdd 0 {{VDD}}
Vbias bias 0 DC {bias:.12g}
{input_voltage_sources(resolved_inputs)}
Vfwd fwd 0 PULSE(0 {{VDD}} 0.1n 10p 10p 2n 4n)
{circuit}
.options method=gear maxord=2 reltol=1e-3 abstol=1e-12 vntol=1e-6 rshunt=1e12
.tran 5p 2.4n uic
{measurement_lines(hidden_count, output_count, measure_time_ns=measure_time_ns)}
.control
run
.endc
.end
""".lstrip()
    return text, topology, weights


def parse_float_list(text: str) -> list[float]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one comma-separated float")
    return [float(value) for value in values]


def run_readout_sensitivity_sweep(
    *,
    spice_bin: str,
    generated_dir: Path,
    tag: str,
    signed_deltas_v: list[float],
    hidden_count: int,
    output_count: int,
    hidden1_fan_in: int,
    hidden2_fan_in: int,
    readout_fan_in: int,
    hidden2_topology_mode: str,
    readout_topology_mode: str,
    seed: int,
    x0: float,
    x1: float,
    bias: float,
    target_output: int,
    center_v: float,
    readout_width_u: float,
    score_cap_f: float,
    readout_branch_style: str,
    measure_time_ns: float,
    timeout: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, delta in enumerate(signed_deltas_v):
        text, topology, weights = readout_sensitivity_netlist(
            x0=x0,
            x1=x1,
            bias=bias,
            hidden_count=hidden_count,
            output_count=output_count,
            hidden1_fan_in=hidden1_fan_in,
            hidden2_fan_in=hidden2_fan_in,
            readout_fan_in=readout_fan_in,
            seed=seed,
            hidden2_topology_mode=hidden2_topology_mode,
            readout_topology_mode=readout_topology_mode,
            signed_delta_v=delta,
            target_output=target_output,
            center_v=center_v,
            readout_width_u=readout_width_u,
            score_cap_f=score_cap_f,
            readout_branch_style=readout_branch_style,
            measure_time_ns=measure_time_ns,
        )
        measures = add_derived_measures(
            run_netlist(spice_bin, generated_dir / f"{tag}_sens{index:03d}.cir", text, timeout),
            output_count,
        )
        target_diff = measures.get(f"score{target_output}_diff", 0.0)
        rows.append(
            {
                "signed_delta_v": delta,
                "target_score_diff_v": target_diff,
                "target_score_gain_v_per_v": None if abs(delta) < 1e-15 else target_diff / delta,
                "score_diffs_v": [measures.get(f"score{out}_diff", 0.0) for out in range(output_count)],
                "weight_summary": summarize_readout_signed_weights(weights, topology),
                "measures": measures,
            }
        )
    return {
        "architecture": "spicenn_sparse_two_hidden_readout_sensitivity",
        "hidden_count": hidden_count,
        "output_count": output_count,
        "hidden2_topology_mode": hidden2_topology_mode,
        "readout_topology_mode": readout_topology_mode,
        "readout_branch_style": readout_branch_style,
        "readout_width_u": readout_width_u,
        "score_cap_f": score_cap_f,
        "measure_time_ns": measure_time_ns,
        "rows": rows,
    }


def netlist(
    *,
    x0: float | None,
    x1: float | None,
    input_values: Mapping[SourceId, float] | None = None,
    input_sources: tuple[SourceId, ...] | None = None,
    bias: float,
    hidden_count: int,
    output_count: int,
    hidden1_fan_in: int,
    hidden2_fan_in: int,
    readout_fan_in: int,
    seed: int,
    readout_weight_initials: dict[tuple[int, SourceId], tuple[float, float]] | None = None,
    hidden2_weight_initials: dict[tuple[int, SourceId], tuple[float, float]] | None = None,
    score_cap_f: float = 10.0,
    readout_width_u: float = 10.0,
    readout_branch_style: str = "gate_stack",
    hidden_weight_mode: str = "positive",
    measure_time_ns: float = 2.2,
    readout_bias: bool = False,
    hidden2_topology_mode: str = "fanin",
    readout_topology_mode: str = "fanin",
) -> tuple[str, SparseForwardTopology]:
    resolved_inputs = default_input_values(x0, x1) if input_values is None else dict(input_values)
    resolved_sources = tuple(input_sources) if input_sources is not None else input_sources_from_values(resolved_inputs)
    topology = build_topology(
        input_sources=resolved_sources,
        hidden_count=hidden_count,
        output_count=output_count,
        hidden1_fan_in=hidden1_fan_in,
        hidden2_fan_in=hidden2_fan_in,
        readout_fan_in=readout_fan_in,
        seed=seed,
        readout_bias=readout_bias,
        hidden2_topology_mode=hidden2_topology_mode,
        readout_topology_mode=readout_topology_mode,
    )
    circuit = render_network(
        topology,
        seed=seed,
        hidden_weight_mode=hidden_weight_mode,
        score_cap_f=score_cap_f,
        readout_width_u=readout_width_u,
        readout_branch_style=readout_branch_style,
        readout_weight_initials=readout_weight_initials,
        hidden2_weight_initials=hidden2_weight_initials,
    )
    text = f"""
* spicenn sparse two-hidden-layer forward primitive
.param VDD=1.2
{mos_models()}
Vdd vdd 0 {{VDD}}
Vbias bias 0 DC {bias:.12g}
{input_voltage_sources(resolved_inputs)}
Vfwd fwd 0 PULSE(0 {{VDD}} 0.1n 10p 10p 2n 4n)
{circuit}
.options method=gear maxord=2 reltol=1e-3 abstol=1e-12 vntol=1e-6 rshunt=1e12
.tran 5p 2.4n uic
{measurement_lines(hidden_count, output_count, measure_time_ns=measure_time_ns)}
.control
run
.endc
.end
""".lstrip()
    return text, topology


def train_netlist(
    *,
    x0: float | None,
    x1: float | None,
    input_values: Mapping[SourceId, float] | None = None,
    input_sources: tuple[SourceId, ...] | None = None,
    bias: float,
    label: int,
    hidden_count: int,
    output_count: int,
    hidden1_fan_in: int,
    hidden2_fan_in: int,
    readout_fan_in: int,
    seed: int,
    update_width_u: float,
    charge_width_u: float | None = None,
    discharge_width_u: float | None = None,
    update_write_mode: str = "simple_charge_discharge",
    spike_ref_v: float = DEFAULT_SPIKE_REF_V,
    readout_center_v: float = 0.30,
    false_negative_target_v: float = 0.0,
    hidden_weight_mode: str = "positive",
    readout_weight_initials: dict[tuple[int, SourceId], tuple[float, float]] | None = None,
    hidden2_weight_initials: dict[tuple[int, SourceId], tuple[float, float]] | None = None,
    readout_bias: bool = False,
    readout_bias_update_scale: float = 1.0,
    hidden2_topology_mode: str = "fanin",
    readout_topology_mode: str = "fanin",
    readout_branch_style: str = "gate_stack",
    score_cap_f: float = 10.0,
    readout_width_u: float = 10.0,
    error_rule: str = "residual",
    hidden2_update_width_u: float = 0.0,
    hidden2_delta_mode: str = "relu_gate",
    hidden2_update_write_mode: str = "inhibit_charge_discharge",
    hidden2_update_selector_width_u: float = 2.0,
) -> tuple[str, SparseForwardTopology]:
    resolved_inputs = default_input_values(x0, x1) if input_values is None else dict(input_values)
    resolved_sources = tuple(input_sources) if input_sources is not None else input_sources_from_values(resolved_inputs)
    effective_hidden_weight_mode = effective_hidden_weight_mode_for_updates(hidden_weight_mode, hidden2_update_width_u)
    topology = build_topology(
        input_sources=resolved_sources,
        hidden_count=hidden_count,
        output_count=output_count,
        hidden1_fan_in=hidden1_fan_in,
        hidden2_fan_in=hidden2_fan_in,
        readout_fan_in=readout_fan_in,
        seed=seed,
        readout_bias=readout_bias,
        hidden2_topology_mode=hidden2_topology_mode,
        readout_topology_mode=readout_topology_mode,
    )
    circuit = render_network(
        topology,
        seed=seed,
        hidden_weight_mode=effective_hidden_weight_mode,
        readout_branch_style=readout_branch_style,
        score_cap_f=score_cap_f,
        readout_width_u=readout_width_u,
        readout_weight_initials=(
            centered_readout_initials(topology.readout, center_v=readout_center_v)
            if readout_weight_initials is None
            else readout_weight_initials
        ),
        hidden2_weight_initials=hidden2_weight_initials,
    )
    extras = readout_train_extras(
        topology,
        label=label,
        target_high_v=1.1,
        target_low_v=0.0,
        false_negative_target_v=false_negative_target_v,
        error_cap_f=6.0,
        error_rule=error_rule,
        update_width_u=update_width_u,
        charge_width_u=charge_width_u,
        discharge_width_u=discharge_width_u,
        update_write_mode=update_write_mode,
        readout_bias_update_scale=readout_bias_update_scale,
        hidden2_update_width_u=hidden2_update_width_u,
        hidden2_delta_mode=hidden2_delta_mode,
        hidden2_update_write_mode=hidden2_update_write_mode,
        hidden2_update_selector_width_u=hidden2_update_selector_width_u,
    )
    measurements = train_measurement_lines(
        topology,
        output_count,
        include_selectors=update_write_mode
        in {"cmos_complementary_charge_discharge", "hybrid_trace_spike_charge_discharge"},
        include_competition_error=error_rule == "ce_split_limited",
        include_hidden2_updates=hidden2_update_width_u > 0,
        include_hidden2_gated_delta=hidden2_update_width_u > 0 and hidden2_delta_mode == "relu_gate",
        include_hidden2_selectors=hidden2_update_width_u > 0,
        include_hidden2_selector_bars=hidden2_update_width_u > 0
        and hidden2_update_write_mode in {
            "diffpair_charge_discharge",
            "cmos_complementary_charge_discharge",
        },
        include_hidden2_activity_gate=hidden2_update_width_u > 0
        and hidden2_update_write_mode in {
            "hybrid_trace_spike_charge_discharge",
            "senseamp_charge_discharge",
            "senseamp_cmos_complementary_charge_discharge",
        },
    )
    text = f"""
* spicenn sparse two-hidden-layer readout train-step primitive
.param VDD=1.2
{mos_models()}
Vdd vdd 0 {{VDD}}
Vwhigh whigh 0 DC 1.0
Vwlow wlow 0 DC 0.1
Vrstf rstf 0 DC 0
Vrste rste 0 DC 0
Vspikeref spikeref 0 DC {spike_ref_v:.12g}
Vbias bias 0 DC {bias:.12g}
{input_voltage_sources(resolved_inputs)}
Vfwd fwd 0 PULSE(0 {{VDD}} 0.10n 10p 10p 2.20n 8n)
Verr err 0 PULSE(0 {{VDD}} 2.55n 10p 10p 0.75n 8n)
Vbwd bwd 0 PULSE(0 {{VDD}} 3.60n 10p 10p 1.80n 8n)
{circuit}
{extras}
.options method=gear maxord=2 reltol=1e-3 abstol=1e-12 vntol=1e-6 rshunt=1e12
.tran 5p 5.8n uic
{measurements}
.control
run
.endc
.end
""".lstrip()
    return text, topology


def run_netlist(spice_bin: str, path: Path, text: str, timeout: float) -> dict[str, float]:
    proc = run_text_netlist(spice_bin, path, text, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-3000:])
    measures = parse_measures(proc.stdout + "\n" + proc.stderr)
    if not measures:
        raise RuntimeError("SPICE produced no parseable measurements:\n" + (proc.stdout + proc.stderr)[-3000:])
    return measures


def named_sample_stream(name: str, *, epochs: int) -> list[dict[str, float | int]]:
    """Small fixed streams for transistor-level learning smoke tests.

    ``sum_extremes`` is intentionally simple and representable by the current
    positive sparse ReLU front end.  ``conflicting_same_input`` is a negative
    control: no readout should be expected to solve it because the same input is
    assigned both labels.
    """

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    base_streams: dict[str, list[dict[str, float | int]]] = {
        "sum_extremes": [
            {"x0": 1.0, "x1": 1.0, "label": 0},
            {"x0": 0.0, "x1": 0.0, "label": 1},
        ],
        "or_vs_zero": [
            {"x0": 1.0, "x1": 0.0, "label": 0},
            {"x0": 0.0, "x1": 1.0, "label": 0},
            {"x0": 1.0, "x1": 1.0, "label": 0},
            {"x0": 0.0, "x1": 0.0, "label": 1},
        ],
        "x0_identity": [
            {"x0": 1.0, "x1": 0.0, "label": 0},
            {"x0": 0.0, "x1": 1.0, "label": 1},
        ],
        "conflicting_same_input": [
            {"x0": 1.0, "x1": 1.0, "label": 0},
            {"x0": 1.0, "x1": 1.0, "label": 1},
        ],
    }
    if name not in base_streams:
        choices = ", ".join(sorted(base_streams))
        raise ValueError(f"unknown sample stream {name!r}; choices: {choices}")
    return [dict(sample) for _epoch in range(epochs) for sample in base_streams[name]]


def class_balanced_replay(samples: list[dict[str, float | int]]) -> list[dict[str, float | int]]:
    """Repeat minority-label samples so every class has equal presentation count.

    This is an experiment scheduling control, not a Python-side gradient change:
    every returned sample is still trained by a separate SPICE forward/error/
    backward/write transient.
    """

    if not samples:
        return []
    by_label: dict[int, list[dict[str, float | int]]] = {}
    for sample in samples:
        by_label.setdefault(int(sample["label"]), []).append(dict(sample))
    target_count = max(len(group) for group in by_label.values())
    balanced_by_label: dict[int, list[dict[str, float | int]]] = {}
    for label, group in by_label.items():
        balanced_by_label[label] = [dict(group[index % len(group)]) for index in range(target_count)]

    labels = sorted(balanced_by_label)
    balanced: list[dict[str, float | int]] = []
    for index in range(target_count):
        for label in labels:
            balanced.append(dict(balanced_by_label[label][index]))
    return balanced


def repeat_samples(samples: list[dict[str, Any]], *, epochs: int) -> list[dict[str, Any]]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    return [dict(sample) for _epoch in range(epochs) for sample in samples]


def inferred_output_count(samples: list[dict[str, Any]]) -> int:
    if not samples:
        return 0
    return max(int(sample["label"]) for sample in samples) + 1


def default_repeated_samples() -> list[dict[str, float | int]]:
    return named_sample_stream("sum_extremes", epochs=2)


def unique_samples(samples: list[dict[str, float | int]]) -> list[dict[str, float | int]]:
    seen: set[tuple[tuple[tuple[str, float], ...], int]] = set()
    unique: list[dict[str, float | int]] = []
    for sample in samples:
        input_key = tuple((str(source), value) for source, value in sample_input_values(sample).items())
        key = (input_key, int(sample["label"]))
        if key not in seen:
            seen.add(key)
            unique.append(dict(sample))
    return unique


def evaluate_readout(
    *,
    spice_bin: str,
    generated_dir: Path,
    tag: str,
    samples: list[dict[str, float | int]],
    readout_weight_initials: dict[tuple[int, SourceId], tuple[float, float]],
    hidden2_weight_initials: dict[tuple[int, SourceId], tuple[float, float]] | None,
    hidden_count: int,
    output_count: int,
    hidden1_fan_in: int,
    hidden2_fan_in: int,
    readout_fan_in: int,
    seed: int,
    bias: float,
    hidden_weight_mode: str,
    timeout: float,
    measure_time_ns: float = 2.2,
    readout_bias: bool = False,
    hidden2_topology_mode: str = "fanin",
    readout_topology_mode: str = "fanin",
    readout_branch_style: str = "gate_stack",
    score_cap_f: float = 10.0,
    readout_width_u: float = 10.0,
    input_sources: tuple[SourceId, ...] | None = None,
) -> dict[str, Any]:
    resolved_sources = tuple(input_sources) if input_sources is not None else input_sources_from_samples(samples)
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        sample_inputs = sample_input_values(sample)
        label = int(sample["label"])
        if not 0 <= label < output_count:
            raise ValueError(f"sample label {label} is outside output_count={output_count}")
        text, _topology = netlist(
            x0=None,
            x1=None,
            input_values=sample_inputs,
            input_sources=resolved_sources,
            bias=bias,
            hidden_count=hidden_count,
            output_count=output_count,
            hidden1_fan_in=hidden1_fan_in,
            hidden2_fan_in=hidden2_fan_in,
            readout_fan_in=readout_fan_in,
            seed=seed,
            readout_weight_initials=readout_weight_initials,
            hidden2_weight_initials=hidden2_weight_initials,
            hidden_weight_mode=hidden_weight_mode,
            measure_time_ns=measure_time_ns,
            score_cap_f=score_cap_f,
            readout_width_u=readout_width_u,
            readout_bias=readout_bias,
            hidden2_topology_mode=hidden2_topology_mode,
            readout_topology_mode=readout_topology_mode,
            readout_branch_style=readout_branch_style,
        )
        measures = add_derived_measures(
            run_netlist(spice_bin, generated_dir / f"{tag}_eval{index:03d}.cir", text, timeout),
            output_count,
        )
        score_diffs = [measures.get(f"score{out}_diff", 0.0) for out in range(output_count)]
        h2 = activation_vector(measures, "h2", hidden_count)
        predicted = int(max(range(output_count), key=lambda out: score_diffs[out]))
        rows.append(
            {
                "index": index,
                **sample_display_fields(sample),
                "label": label,
                "predicted": predicted,
                "correct": predicted == label,
                "score_diffs": score_diffs,
                "h2": h2,
                "h2_mean": sum(h2) / len(h2) if h2 else 0.0,
                "h2_active_count": sum(1 for value in h2 if value > 1e-3),
            }
        )
    correct = sum(1 for row in rows if row["correct"])
    return {
        "samples": rows,
        "accuracy": 0.0 if not rows else correct / len(rows),
        "correct": correct,
        "count": len(rows),
    }


def run_feature_separation_probe(
    *,
    spice_bin: str,
    generated_dir: Path,
    tag: str,
    samples: list[dict[str, float | int]],
    bias_values: list[float],
    hidden_count: int,
    output_count: int,
    hidden1_fan_in: int,
    hidden2_fan_in: int,
    readout_fan_in: int,
    seed: int,
    center_v: float,
    hidden_weight_mode: str,
    score_cap_f: float,
    readout_width_u: float,
    readout_branch_style: str,
    measure_time_ns: float,
    timeout: float,
    readout_bias: bool = False,
    hidden2_topology_mode: str = "fanin",
    readout_topology_mode: str = "fanin",
    input_sources: tuple[SourceId, ...] | None = None,
) -> dict[str, Any]:
    """Measure whether the sparse hidden circuit produces separable features."""

    resolved_sources = tuple(input_sources) if input_sources is not None else input_sources_from_samples(samples)
    topology = build_topology(
        input_sources=resolved_sources,
        hidden_count=hidden_count,
        output_count=output_count,
        hidden1_fan_in=hidden1_fan_in,
        hidden2_fan_in=hidden2_fan_in,
        readout_fan_in=readout_fan_in,
        seed=seed,
        readout_bias=readout_bias,
        hidden2_topology_mode=hidden2_topology_mode,
        readout_topology_mode=readout_topology_mode,
    )
    centered_weights = centered_readout_initials(topology.readout, center_v=center_v)
    eval_samples = unique_samples(samples)
    rows: list[dict[str, Any]] = []
    for bias_index, bias_value in enumerate(bias_values):
        sample_rows: list[dict[str, Any]] = []
        for sample_index, sample in enumerate(eval_samples):
            sample_inputs = sample_input_values(sample)
            text, _topology = netlist(
                x0=None,
                x1=None,
                input_values=sample_inputs,
                input_sources=resolved_sources,
                bias=bias_value,
                hidden_count=hidden_count,
                output_count=output_count,
                hidden1_fan_in=hidden1_fan_in,
                hidden2_fan_in=hidden2_fan_in,
                readout_fan_in=readout_fan_in,
                seed=seed,
                readout_weight_initials=centered_weights,
                score_cap_f=score_cap_f,
                readout_width_u=readout_width_u,
                readout_branch_style=readout_branch_style,
                hidden_weight_mode=hidden_weight_mode,
                measure_time_ns=measure_time_ns,
                readout_bias=readout_bias,
                hidden2_topology_mode=hidden2_topology_mode,
                readout_topology_mode=readout_topology_mode,
            )
            measures = add_derived_measures(
                run_netlist(
                    spice_bin,
                    generated_dir / f"{tag}_feature_b{bias_index:03d}_s{sample_index:03d}.cir",
                    text,
                    timeout,
                ),
                output_count,
            )
            h1 = activation_vector(measures, "h1", hidden_count)
            h2 = activation_vector(measures, "h2", hidden_count)
            coverage = readout_hidden_coverage(h2, topology.readout)
            label = int(sample["label"])
            sample_rows.append(
                {
                    **sample_display_fields(sample),
                    "label": label,
                    "h1": h1,
                    "h2": h2,
                    "h2_mean": sum(h2) / len(h2) if h2 else 0.0,
                    "h2_max": max(h2) if h2 else 0.0,
                    "h2_active_count": sum(1 for value in h2 if value > 1e-3),
                    "readout_hidden_coverage": coverage,
                    "label_readout_active_count": coverage[label]["active_count"] if 0 <= label < len(coverage) else None,
                    "label_readout_active_sum": coverage[label]["active_sum"] if 0 <= label < len(coverage) else None,
                    "score_diffs": [measures.get(f"score{out}_diff", 0.0) for out in range(output_count)],
                }
            )

        inter_l1: list[float] = []
        inter_l2: list[float] = []
        same_l2: list[float] = []
        for left_index, left in enumerate(sample_rows):
            for right in sample_rows[left_index + 1 :]:
                l1 = vector_l1(left["h2"], right["h2"])
                l2 = vector_l2(left["h2"], right["h2"])
                if left["label"] == right["label"]:
                    same_l2.append(l2)
                else:
                    inter_l1.append(l1)
                    inter_l2.append(l2)
        rows.append(
            {
                "bias": bias_value,
                "samples": sample_rows,
                "min_inter_label_h2_l1": min(inter_l1) if inter_l1 else None,
                "mean_inter_label_h2_l1": (sum(inter_l1) / len(inter_l1)) if inter_l1 else None,
                "min_inter_label_h2_l2": min(inter_l2) if inter_l2 else None,
                "mean_inter_label_h2_l2": (sum(inter_l2) / len(inter_l2)) if inter_l2 else None,
                "max_same_label_h2_l2": max(same_l2) if same_l2 else None,
                "min_h2_mean": min((sample["h2_mean"] for sample in sample_rows), default=0.0),
                "min_h2_active_count": min((sample["h2_active_count"] for sample in sample_rows), default=0),
                "min_label_readout_active_count": min(
                    (sample["label_readout_active_count"] for sample in sample_rows if sample["label_readout_active_count"] is not None),
                    default=None,
                ),
                "min_label_readout_active_sum": min(
                    (sample["label_readout_active_sum"] for sample in sample_rows if sample["label_readout_active_sum"] is not None),
                    default=None,
                ),
            }
        )

    return {
        "architecture": "spicenn_sparse_two_hidden_feature_separation",
        "hidden_count": hidden_count,
        "output_count": output_count,
        "input_sources": [str(source) for source in resolved_sources],
        "sample_count": len(eval_samples),
        "hidden_weight_mode": hidden_weight_mode,
        "hidden2_topology_mode": hidden2_topology_mode,
        "readout_topology_mode": readout_topology_mode,
        "measure_time_ns": measure_time_ns,
        "topology": {
            "hidden1_fanins": {str(k): list(v) for k, v in topology.hidden1.as_fanins().items()},
            "hidden2_fanins": {str(k): list(v) for k, v in topology.hidden2.as_fanins().items()},
            "readout_fanins": {str(k): list(v) for k, v in topology.readout.as_fanins().items()},
        },
        "rows": rows,
    }


def run_repeated_readout_training(
    *,
    spice_bin: str,
    generated_dir: Path,
    tag: str,
    samples: list[dict[str, float | int]],
    hidden_count: int,
    output_count: int,
    hidden1_fan_in: int,
    hidden2_fan_in: int,
    readout_fan_in: int,
    seed: int,
    bias: float,
    update_width_u: float,
    charge_width_u: float | None,
    discharge_width_u: float | None,
    update_write_mode: str,
    spike_ref_v: float,
    readout_center_v: float,
    false_negative_target_v: float,
    hidden_weight_mode: str,
    timeout: float,
    evaluation_samples: list[dict[str, float | int]] | None = None,
    readout_bias: bool = False,
    readout_bias_update_scale: float = 1.0,
    hidden2_topology_mode: str = "fanin",
    readout_topology_mode: str = "fanin",
    readout_branch_style: str = "gate_stack",
    score_cap_f: float = 10.0,
    readout_width_u: float = 10.0,
    hidden2_update_width_u: float = 0.0,
    hidden2_delta_mode: str = "relu_gate",
    hidden2_update_write_mode: str = "inhibit_charge_discharge",
    hidden2_update_selector_width_u: float = 2.0,
    readout_init_mode: str = "centered",
    readout_jitter_v: float = 0.02,
    input_sources: tuple[SourceId, ...] | None = None,
    error_rule: str = "residual",
    measure_time_ns: float = 2.2,
) -> dict[str, Any]:
    resolved_sources = tuple(input_sources) if input_sources is not None else input_sources_from_samples(samples)
    effective_hidden_weight_mode = effective_hidden_weight_mode_for_updates(hidden_weight_mode, hidden2_update_width_u)
    topology = build_topology(
        input_sources=resolved_sources,
        hidden_count=hidden_count,
        output_count=output_count,
        hidden1_fan_in=hidden1_fan_in,
        hidden2_fan_in=hidden2_fan_in,
        readout_fan_in=readout_fan_in,
        seed=seed,
        readout_bias=readout_bias,
        hidden2_topology_mode=hidden2_topology_mode,
        readout_topology_mode=readout_topology_mode,
    )
    if readout_init_mode == "centered":
        weights = centered_readout_initials(topology.readout, center_v=readout_center_v)
    elif readout_init_mode == "jittered":
        weights = random_jittered_readout_initials(
            topology.readout,
            seed=seed + 2003,
            center_v=readout_center_v,
            signed_span_v=readout_jitter_v,
        )
    else:
        raise ValueError(f"unknown readout_init_mode: {readout_init_mode}")
    hidden2_weights = hidden2_initials_for_mode(
        topology.hidden2,
        hidden_weight_mode=effective_hidden_weight_mode,
        seed=seed,
    )
    initial_weight_summary = summarize_readout_signed_weights(weights, topology)
    initial_hidden2_weight_summary = summarize_hidden2_signed_weights(hidden2_weights, topology)
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        sample_inputs = sample_input_values(sample)
        label = int(sample["label"])
        text, step_topology = train_netlist(
            x0=None,
            x1=None,
            input_values=sample_inputs,
            input_sources=resolved_sources,
            bias=bias,
            label=label,
            hidden_count=hidden_count,
            output_count=output_count,
            hidden1_fan_in=hidden1_fan_in,
            hidden2_fan_in=hidden2_fan_in,
            readout_fan_in=readout_fan_in,
            seed=seed,
            update_width_u=update_width_u,
            charge_width_u=charge_width_u,
            discharge_width_u=discharge_width_u,
            update_write_mode=update_write_mode,
            spike_ref_v=spike_ref_v,
            readout_center_v=readout_center_v,
            false_negative_target_v=false_negative_target_v,
            hidden_weight_mode=effective_hidden_weight_mode,
            readout_weight_initials=weights,
            hidden2_weight_initials=hidden2_weights,
            readout_bias=readout_bias,
            readout_bias_update_scale=readout_bias_update_scale,
            hidden2_topology_mode=hidden2_topology_mode,
            readout_topology_mode=readout_topology_mode,
            readout_branch_style=readout_branch_style,
            score_cap_f=score_cap_f,
            readout_width_u=readout_width_u,
            error_rule=error_rule,
            hidden2_update_width_u=hidden2_update_width_u,
            hidden2_delta_mode=hidden2_delta_mode,
            hidden2_update_write_mode=hidden2_update_write_mode,
            hidden2_update_selector_width_u=hidden2_update_selector_width_u,
        )
        measures = add_train_derived_measures(
            run_netlist(spice_bin, generated_dir / f"{tag}_step{index:03d}.cir", text, timeout),
            step_topology,
        )
        weights = extract_readout_initials_after(measures, step_topology)
        if hidden2_update_width_u > 0:
            hidden2_weights = extract_hidden2_initials_after(measures, step_topology)
        score_diffs = [measures.get(f"score{out}_diff", 0.0) for out in range(output_count)]
        predicted = int(max(range(output_count), key=lambda out: score_diffs[out]))
        row_signed_deltas = [measures.get(f"row{out}_signed_delta", 0.0) for out in range(output_count)]
        row_common_deltas = [measures.get(f"row{out}_common_delta", 0.0) for out in range(output_count)]
        error_diffs = [
            measures.get(f"dp{out}_err", 0.0) - measures.get(f"dn{out}_err", 0.0)
            for out in range(output_count)
        ]
        selector_gates = [
            {
                "rwpos": measures.get(f"rwpos{out}_gate"),
                "rwneg": measures.get(f"rwneg{out}_gate"),
                "posbar": measures.get(f"rwsel{out}_posbar_gate"),
                "negbar": measures.get(f"rwsel{out}_negbar_gate"),
            }
            for out in range(output_count)
        ]
        competition_error = None
        if error_rule == "ce_split_limited":
            competition_error = {
                "ctsrc": measures.get("ctsrc_err"),
                "cesrc": measures.get("cesrc_err"),
                "ccsrc": measures.get("ccsrc_err"),
                "ybar": [measures.get(f"ybar{out}_err") for out in range(output_count)],
            }
        pretrace_gates = {
            f"{out}_{source}": {
                "gate": measures.get(f"fprg{out}_{source}_gate"),
                "bar": measures.get(f"fprbar{out}_{source}_gate"),
            }
            for out, sources in step_topology.readout.as_fanins().items()
            for source in sources
        }
        hidden2_row_signed_deltas = [
            measures.get(f"hidden2_row{sink}_signed_delta", 0.0) for sink in range(step_topology.hidden2.sink_count)
        ]
        hidden2_row_common_deltas = [
            measures.get(f"hidden2_row{sink}_common_delta", 0.0) for sink in range(step_topology.hidden2.sink_count)
        ]
        hidden2_selector_gates = [
            {
                "hwpos": measures.get(f"hwpos{sink}_gate"),
                "hwneg": measures.get(f"hwneg{sink}_gate"),
                "posbar": measures.get(f"hwsel{sink}_posbar_gate"),
                "negbar": measures.get(f"hwsel{sink}_negbar_gate"),
                "active": measures.get(f"hwsel{sink}_active_gate"),
            }
            for sink in range(step_topology.hidden2.sink_count)
        ]
        rows.append(
            {
                "step": index,
                **sample_display_fields(sample),
                "label": label,
                "predicted_before_update": predicted,
                "correct_before_update": predicted == label,
                "score_diffs_before_update": score_diffs,
                "error_diffs": error_diffs,
                "row_signed_deltas": row_signed_deltas,
                "row_common_deltas": row_common_deltas,
                "label_row_signed_delta": measures.get(f"row{label}_signed_delta", 0.0),
                "label_row_common_delta": measures.get(f"row{label}_common_delta", 0.0),
                "selector_gates": selector_gates,
                "competition_error": competition_error,
                "pretrace_gates": pretrace_gates,
                "pretrace_gate_summary": pretrace_gate_summary(pretrace_gates),
                "hidden2_row_signed_deltas": hidden2_row_signed_deltas,
                "hidden2_row_common_deltas": hidden2_row_common_deltas,
                "hidden2_selector_gates": hidden2_selector_gates,
                "hidden2_weights_after": {
                    f"{sink}_{source}": [pos, neg]
                    for (sink, source), (pos, neg) in sorted(
                        hidden2_weights.items(),
                        key=lambda item: (item[0][0], str(item[0][1])),
                    )
                },
                "weights_after": {
                    f"{out}_{source}": [pos, neg]
                    for (out, source), (pos, neg) in sorted(weights.items(), key=lambda item: (item[0][0], str(item[0][1])))
                },
            }
        )
    final_weight_summary = summarize_readout_signed_weights(weights, topology)
    final_hidden2_weight_summary = summarize_hidden2_signed_weights(hidden2_weights, topology)
    final_evaluation = evaluate_readout(
        spice_bin=spice_bin,
        generated_dir=generated_dir,
        tag=f"{tag}_final",
        samples=unique_samples(samples) if evaluation_samples is None else evaluation_samples,
        readout_weight_initials=weights,
        hidden2_weight_initials=hidden2_weights,
        hidden_count=hidden_count,
        output_count=output_count,
        hidden1_fan_in=hidden1_fan_in,
        hidden2_fan_in=hidden2_fan_in,
        readout_fan_in=readout_fan_in,
        seed=seed,
        bias=bias,
        hidden_weight_mode=effective_hidden_weight_mode,
        timeout=timeout,
        readout_bias=readout_bias,
        hidden2_topology_mode=hidden2_topology_mode,
        readout_topology_mode=readout_topology_mode,
        readout_branch_style=readout_branch_style,
        score_cap_f=score_cap_f,
        readout_width_u=readout_width_u,
        measure_time_ns=measure_time_ns,
        input_sources=resolved_sources,
    )
    train_correct = sum(1 for row in rows if row["correct_before_update"])
    return {
        "topology": {
            "hidden1_fanins": {str(k): list(v) for k, v in topology.hidden1.as_fanins().items()},
            "hidden2_fanins": {str(k): list(v) for k, v in topology.hidden2.as_fanins().items()},
            "readout_fanins": {str(k): list(v) for k, v in topology.readout.as_fanins().items()},
        },
        "steps": rows,
        "input_sources": [str(source) for source in resolved_sources],
        "training_before_update_accuracy": 0.0 if not rows else train_correct / len(rows),
        "readout_init_mode": readout_init_mode,
        "readout_jitter_v": readout_jitter_v,
        "requested_hidden_weight_mode": hidden_weight_mode,
        "effective_hidden_weight_mode": effective_hidden_weight_mode,
        "error_rule": error_rule,
        "score_cap_f": score_cap_f,
        "readout_width_u": readout_width_u,
        "measure_time_ns": measure_time_ns,
        "readout_bias_update_scale": readout_bias_update_scale,
        "hidden2_topology_mode": hidden2_topology_mode,
        "readout_topology_mode": readout_topology_mode,
        "initial_weight_summary": initial_weight_summary,
        "final_weight_summary": final_weight_summary,
        "initial_hidden2_weight_summary": initial_hidden2_weight_summary,
        "final_hidden2_weight_summary": final_hidden2_weight_summary,
        "final_evaluation": final_evaluation,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulator", default=None)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--tag", default="spicenn_sparse_forward")
    ap.add_argument("--x0", type=float, default=1.0)
    ap.add_argument("--x1", type=float, default=1.0)
    ap.add_argument("--bias", type=float, default=0.60)
    ap.add_argument("--hidden-count", type=int, default=4)
    ap.add_argument("--output-count", type=int, default=2)
    ap.add_argument("--hidden1-fan-in", type=int, default=2)
    ap.add_argument("--hidden2-fan-in", type=int, default=3)
    ap.add_argument("--readout-fan-in", type=int, default=3)
    ap.add_argument("--hidden2-topology-mode", choices=TOPOLOGY_MODES, default="fanin")
    ap.add_argument("--readout-topology-mode", choices=TOPOLOGY_MODES, default="fanin")
    ap.add_argument("--readout-bias", action="store_true")
    ap.add_argument("--readout-bias-update-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--hidden-weight-mode",
        choices=[
            "positive",
            "signed_hidden1",
            "signed_all",
            "signed_hidden1_random_bias",
            "signed_all_random_bias",
            "weak_signed_hidden1_random_bias",
            "weak_signed_all_random_bias",
            "centered_hidden2",
            "centered_jittered_hidden2",
            "centered_all",
            "centered_jittered_all",
        ],
        default="positive",
    )
    ap.add_argument("--train-readout", action="store_true")
    ap.add_argument("--repeat-readout-training", action="store_true")
    ap.add_argument("--readout-sensitivity-sweep", action="store_true")
    ap.add_argument("--feature-separation-probe", action="store_true")
    ap.add_argument(
        "--sample-set",
        choices=["sum_extremes", "or_vs_zero", "x0_identity", "conflicting_same_input"],
        default="sum_extremes",
    )
    ap.add_argument(
        "--dataset",
        default=None,
        help=(
            "Optional dataset name from spice/datasets.py, e.g. mnist3fixed8_6 or mnistfixed8_20. "
            "When set, it replaces --sample-set for repeated training and feature probes."
        ),
    )
    ap.add_argument(
        "--download-dataset",
        action="store_true",
        help="Allow torchvision-backed datasets such as MNIST to be downloaded into ./data when missing.",
    )
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--class-balanced-replay", action="store_true")
    ap.add_argument("--label", type=int, default=0)
    ap.add_argument("--target-output", type=int, default=0)
    ap.add_argument("--update-width-u", type=float, default=0.004)
    ap.add_argument("--hidden2-update-width-u", type=float, default=0.0)
    ap.add_argument("--hidden2-update-selector-width-u", type=float, default=2.0)
    ap.add_argument("--hidden2-delta-mode", choices=["raw", "relu_gate"], default="relu_gate")
    ap.add_argument(
        "--hidden2-update-write-mode",
        choices=[
            "simple_charge_discharge",
            "diffpair_charge_discharge",
            "inhibit_charge_discharge",
            "cmos_complementary_charge_discharge",
            "hybrid_trace_spike_charge_discharge",
            "senseamp_charge_discharge",
            "senseamp_cmos_complementary_charge_discharge",
        ],
        default="inhibit_charge_discharge",
    )
    ap.add_argument("--charge-width-u", type=float, default=None)
    ap.add_argument("--discharge-width-u", type=float, default=None)
    ap.add_argument("--spike-ref-v", type=float, default=DEFAULT_SPIKE_REF_V)
    ap.add_argument("--false-negative-target-v", type=float, default=0.0)
    ap.add_argument("--error-rule", choices=ERROR_RULES, default="residual")
    ap.add_argument("--sensitivity-deltas", default="0,0.01,0.02,0.05,0.1")
    ap.add_argument("--bias-sweep", default="0.15,0.25,0.30,0.40,0.60")
    ap.add_argument("--center-v", type=float, default=0.30)
    ap.add_argument("--readout-init-mode", choices=["centered", "jittered"], default="centered")
    ap.add_argument("--readout-jitter-v", type=float, default=0.02)
    ap.add_argument("--readout-width-u", type=float, default=10.0)
    ap.add_argument("--score-cap-f", type=float, default=10.0)
    ap.add_argument("--measure-time-ns", type=float, default=2.2)
    ap.add_argument(
        "--readout-branch-style",
        choices=["gate_stack", "pass_act_source", "pass_act_buffered"],
        default="gate_stack",
    )
    ap.add_argument(
        "--update-write-mode",
        choices=[
            "simple_charge_discharge",
            "analog_trace_charge_discharge",
            "cmos_complementary_charge_discharge",
            "hybrid_trace_spike_charge_discharge",
        ],
        default="simple_charge_discharge",
    )
    args = ap.parse_args()

    spice_bin, version = detect_spice(args.simulator)
    generated = ROOT / "spice/generated"
    generated.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)
    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag)

    def load_cli_samples(*, epochs: int) -> list[dict[str, Any]]:
        if args.dataset is None:
            return named_sample_stream(args.sample_set, epochs=epochs)
        from datasets import dataset_records

        return repeat_samples(
            dataset_records(args.dataset, args.seed, root=ROOT, download=args.download_dataset),
            epochs=epochs,
        )

    if args.feature_separation_probe:
        t0 = time.perf_counter()
        samples = load_cli_samples(epochs=1)
        output_count = max(args.output_count, inferred_output_count(samples))
        input_sources = input_sources_from_samples(samples)
        result = run_feature_separation_probe(
            spice_bin=spice_bin,
            generated_dir=generated,
            tag=safe_tag,
            samples=samples,
            bias_values=parse_float_list(args.bias_sweep),
            hidden_count=args.hidden_count,
            output_count=output_count,
            hidden1_fan_in=args.hidden1_fan_in,
            hidden2_fan_in=args.hidden2_fan_in,
            readout_fan_in=args.readout_fan_in,
            seed=args.seed,
            center_v=args.center_v,
            hidden_weight_mode=args.hidden_weight_mode,
            score_cap_f=args.score_cap_f,
            readout_width_u=args.readout_width_u,
            readout_branch_style=args.readout_branch_style,
            measure_time_ns=args.measure_time_ns,
            timeout=args.timeout,
            readout_bias=args.readout_bias,
            hidden2_topology_mode=args.hidden2_topology_mode,
            readout_topology_mode=args.readout_topology_mode,
            input_sources=input_sources,
        )
        print(
            json.dumps(
                {
                    "simulator": version,
                    "status": "feature_separation_probe",
                    "sample_set": args.dataset or args.sample_set,
                    **result,
                    "wall_time_s": time.perf_counter() - t0,
                },
                indent=2,
            )
        )
        return

    if args.readout_sensitivity_sweep:
        t0 = time.perf_counter()
        result = run_readout_sensitivity_sweep(
            spice_bin=spice_bin,
            generated_dir=generated,
            tag=safe_tag,
            signed_deltas_v=parse_float_list(args.sensitivity_deltas),
            hidden_count=args.hidden_count,
            output_count=args.output_count,
            hidden1_fan_in=args.hidden1_fan_in,
            hidden2_fan_in=args.hidden2_fan_in,
            readout_fan_in=args.readout_fan_in,
            hidden2_topology_mode=args.hidden2_topology_mode,
            readout_topology_mode=args.readout_topology_mode,
            seed=args.seed,
            x0=args.x0,
            x1=args.x1,
            bias=args.bias,
            target_output=args.target_output,
            center_v=args.center_v,
            readout_width_u=args.readout_width_u,
            score_cap_f=args.score_cap_f,
            readout_branch_style=args.readout_branch_style,
            measure_time_ns=args.measure_time_ns,
            timeout=args.timeout,
        )
        print(
            json.dumps(
                {
                    "simulator": version,
                    "status": "readout_signed_weight_sensitivity_sweep",
                    **result,
                    "wall_time_s": time.perf_counter() - t0,
                },
                indent=2,
            )
        )
        return

    if args.repeat_readout_training:
        t0 = time.perf_counter()
        samples = load_cli_samples(epochs=args.epochs)
        if args.class_balanced_replay:
            samples = class_balanced_replay(samples)
        output_count = max(args.output_count, inferred_output_count(samples))
        input_sources = input_sources_from_samples(samples)
        repeated = run_repeated_readout_training(
            spice_bin=spice_bin,
            generated_dir=generated,
            tag=safe_tag,
            samples=samples,
            hidden_count=args.hidden_count,
            output_count=output_count,
            hidden1_fan_in=args.hidden1_fan_in,
            hidden2_fan_in=args.hidden2_fan_in,
            readout_fan_in=args.readout_fan_in,
            seed=args.seed,
            bias=args.bias,
            update_width_u=args.update_width_u,
            charge_width_u=args.charge_width_u,
            discharge_width_u=args.discharge_width_u,
            update_write_mode=args.update_write_mode,
            spike_ref_v=args.spike_ref_v,
            readout_center_v=args.center_v,
            false_negative_target_v=args.false_negative_target_v,
            hidden_weight_mode=args.hidden_weight_mode,
            evaluation_samples=unique_samples(samples),
            timeout=args.timeout,
            readout_bias=args.readout_bias,
            readout_bias_update_scale=args.readout_bias_update_scale,
            hidden2_topology_mode=args.hidden2_topology_mode,
            readout_topology_mode=args.readout_topology_mode,
            readout_branch_style=args.readout_branch_style,
            hidden2_update_width_u=args.hidden2_update_width_u,
            hidden2_delta_mode=args.hidden2_delta_mode,
            hidden2_update_write_mode=args.hidden2_update_write_mode,
            hidden2_update_selector_width_u=args.hidden2_update_selector_width_u,
            readout_init_mode=args.readout_init_mode,
            readout_jitter_v=args.readout_jitter_v,
            input_sources=input_sources,
            error_rule=args.error_rule,
            score_cap_f=args.score_cap_f,
            readout_width_u=args.readout_width_u,
            measure_time_ns=args.measure_time_ns,
        )
        summary = {
            "simulator": version,
            "architecture": "spicenn_sparse_two_hidden_repeated_readout_train",
            "status": "repeated_readout_training_smoke",
            "sample_set": args.dataset or args.sample_set,
            "epochs": args.epochs,
            "class_balanced_replay": args.class_balanced_replay,
            "training_sample_count": len(samples),
            "hidden_count": args.hidden_count,
            "output_count": output_count,
            "hidden_weight_mode": args.hidden_weight_mode,
            "effective_hidden_weight_mode": effective_hidden_weight_mode_for_updates(
                args.hidden_weight_mode,
                args.hidden2_update_width_u,
            ),
            "hidden2_topology_mode": args.hidden2_topology_mode,
            "readout_topology_mode": args.readout_topology_mode,
            "readout_bias": args.readout_bias,
            "readout_branch_style": args.readout_branch_style,
            "readout_bias_update_scale": args.readout_bias_update_scale,
            "update_write_mode": args.update_write_mode,
            "hidden2_update_width_u": args.hidden2_update_width_u,
            "hidden2_update_selector_width_u": args.hidden2_update_selector_width_u,
            "hidden2_delta_mode": args.hidden2_delta_mode,
            "hidden2_update_write_mode": args.hidden2_update_write_mode,
            "spike_ref_v": args.spike_ref_v,
            "readout_center_v": args.center_v,
            "readout_init_mode": args.readout_init_mode,
            "readout_jitter_v": args.readout_jitter_v,
            "score_cap_f": args.score_cap_f,
            "readout_width_u": args.readout_width_u,
            "measure_time_ns": args.measure_time_ns,
            "false_negative_target_v": args.false_negative_target_v,
            "error_rule": args.error_rule,
            **repeated,
            "wall_time_s": time.perf_counter() - t0,
        }
        print(json.dumps(summary, indent=2))
        return

    if args.train_readout:
        text, topology = train_netlist(
            x0=args.x0,
            x1=args.x1,
            bias=args.bias,
            label=args.label,
            hidden_count=args.hidden_count,
            output_count=args.output_count,
            hidden1_fan_in=args.hidden1_fan_in,
            hidden2_fan_in=args.hidden2_fan_in,
            readout_fan_in=args.readout_fan_in,
            seed=args.seed,
            update_width_u=args.update_width_u,
            charge_width_u=args.charge_width_u,
            discharge_width_u=args.discharge_width_u,
            update_write_mode=args.update_write_mode,
            spike_ref_v=args.spike_ref_v,
            readout_center_v=args.center_v,
            false_negative_target_v=args.false_negative_target_v,
            hidden_weight_mode=args.hidden_weight_mode,
            readout_bias=args.readout_bias,
            readout_bias_update_scale=args.readout_bias_update_scale,
            hidden2_topology_mode=args.hidden2_topology_mode,
            readout_topology_mode=args.readout_topology_mode,
            readout_branch_style=args.readout_branch_style,
            hidden2_update_width_u=args.hidden2_update_width_u,
            hidden2_delta_mode=args.hidden2_delta_mode,
            hidden2_update_write_mode=args.hidden2_update_write_mode,
            hidden2_update_selector_width_u=args.hidden2_update_selector_width_u,
            error_rule=args.error_rule,
            score_cap_f=args.score_cap_f,
            readout_width_u=args.readout_width_u,
        )
    else:
        text, topology = netlist(
            x0=args.x0,
            x1=args.x1,
            bias=args.bias,
            hidden_count=args.hidden_count,
            output_count=args.output_count,
            hidden1_fan_in=args.hidden1_fan_in,
            hidden2_fan_in=args.hidden2_fan_in,
            readout_fan_in=args.readout_fan_in,
            seed=args.seed,
            hidden_weight_mode=args.hidden_weight_mode,
            readout_bias=args.readout_bias,
            hidden2_topology_mode=args.hidden2_topology_mode,
            readout_topology_mode=args.readout_topology_mode,
            readout_branch_style=args.readout_branch_style,
            score_cap_f=args.score_cap_f,
            readout_width_u=args.readout_width_u,
        )
    t0 = time.perf_counter()
    raw_measures = run_netlist(spice_bin, generated / f"{safe_tag}.cir", text, args.timeout)
    measures = (
        add_train_derived_measures(raw_measures, topology)
        if args.train_readout
        else add_derived_measures(raw_measures, args.output_count)
    )
    summary: dict[str, Any] = {
        "simulator": version,
        "architecture": "spicenn_sparse_two_hidden_readout_train" if args.train_readout else "spicenn_sparse_two_hidden_forward",
        "status": "readout_train_step_smoke" if args.train_readout else "forward_primitive_smoke",
        "hidden_count": args.hidden_count,
        "output_count": args.output_count,
        "hidden_weight_mode": args.hidden_weight_mode,
        "effective_hidden_weight_mode": effective_hidden_weight_mode_for_updates(
            args.hidden_weight_mode,
            args.hidden2_update_width_u if args.train_readout else 0.0,
        ),
        "hidden2_topology_mode": args.hidden2_topology_mode,
        "readout_topology_mode": args.readout_topology_mode,
        "readout_bias": args.readout_bias,
        "readout_branch_style": args.readout_branch_style,
        "readout_bias_update_scale": args.readout_bias_update_scale,
        "hidden2_update_width_u": args.hidden2_update_width_u,
        "hidden2_update_selector_width_u": args.hidden2_update_selector_width_u,
        "hidden2_delta_mode": args.hidden2_delta_mode,
        "hidden2_update_write_mode": args.hidden2_update_write_mode,
        "hidden1_fanins": {str(k): list(v) for k, v in topology.hidden1.as_fanins().items()},
        "hidden2_fanins": {str(k): list(v) for k, v in topology.hidden2.as_fanins().items()},
        "readout_fanins": {str(k): list(v) for k, v in topology.readout.as_fanins().items()},
        "measures": measures,
        "wall_time_s": time.perf_counter() - t0,
    }
    if args.train_readout:
        summary["label"] = args.label
        summary["update_write_mode"] = args.update_write_mode
        summary["spike_ref_v"] = args.spike_ref_v
        summary["readout_center_v"] = args.center_v
        summary["false_negative_target_v"] = args.false_negative_target_v
        summary["error_rule"] = args.error_rule
        summary["label_row_signed_delta"] = measures.get(f"row{args.label}_signed_delta")
        summary["label_row_common_delta"] = measures.get(f"row{args.label}_common_delta")
        summary["hidden2_row_signed_deltas"] = [
            measures.get(f"hidden2_row{sink}_signed_delta") for sink in range(topology.hidden2.sink_count)
        ]
        summary["hidden2_row_common_deltas"] = [
            measures.get(f"hidden2_row{sink}_common_delta") for sink in range(topology.hidden2.sink_count)
        ]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
