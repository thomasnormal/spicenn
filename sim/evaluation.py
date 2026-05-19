from __future__ import annotations

from dataclasses import asdict

import torch

from .energy_model import estimate_inference_energy
from .topology import WireMetrics


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def architecture_record(
    name: str,
    model: torch.nn.Module,
    accuracy: float,
    wire: WireMetrics,
    neuron_count: int,
    stochastic_cycles: int = 1,
) -> dict:
    energy = estimate_inference_energy(
        synapse_count=wire.synapse_count,
        active_synapse_count=wire.synapse_count,
        total_wire_length_um=wire.total_wire_length_um,
        neuron_count=neuron_count,
        stochastic_cycles=stochastic_cycles,
    )
    return {
        "architecture": name,
        "accuracy": accuracy,
        "parameter_count": count_parameters(model),
        **asdict(wire),
        **energy.to_dict(),
    }

