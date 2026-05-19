from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .energy_model import EnergyConfig, estimate_inference_energy


@dataclass(frozen=True)
class LayerHardware:
    name: str
    layer_type: str
    out_h: int
    out_w: int
    in_channels: int
    out_channels: int
    kernel: int
    stride: int = 1
    pitch_um: float = 10.0

    @property
    def output_neurons(self) -> int:
        return self.out_h * self.out_w * self.out_channels

    @property
    def synapse_instances(self) -> int:
        return self.output_neurons * self.in_channels * self.kernel * self.kernel

    @property
    def parameter_count(self) -> int:
        if self.layer_type == "shared_conv":
            return self.in_channels * self.out_channels * self.kernel * self.kernel
        return self.synapse_instances

    @property
    def total_wire_length_um(self) -> float:
        radius = self.kernel // 2
        per_kernel = 0.0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                per_kernel += (abs(dx) + abs(dy)) * self.pitch_um
        return self.out_h * self.out_w * self.out_channels * self.in_channels * per_kernel

    @property
    def max_wire_length_um(self) -> float:
        radius = self.kernel // 2
        return 2.0 * radius * self.pitch_um

    @property
    def average_wire_length_um(self) -> float:
        if self.synapse_instances == 0:
            return 0.0
        return self.total_wire_length_um / self.synapse_instances

    def to_record(self) -> dict:
        return {
            **asdict(self),
            "output_neurons": self.output_neurons,
            "synapse_instances": self.synapse_instances,
            "parameter_count": self.parameter_count,
            "total_wire_length_um": self.total_wire_length_um,
            "max_wire_length_um": self.max_wire_length_um,
            "average_wire_length_um": self.average_wire_length_um,
        }


def local_evidence_net_layers(
    image_size: int = 28,
    channels: tuple[int, int, int] = (24, 48, 64),
    pitch_um: float = 10.0,
    coord_channels: bool = True,
) -> list[LayerHardware]:
    input_channels = 3 if coord_channels else 1
    return [
        LayerHardware("conv1_5x5", "shared_conv", image_size, image_size, input_channels, channels[0], 5, pitch_um=pitch_um),
        LayerHardware("conv2_3x3", "shared_conv", image_size // 2, image_size // 2, channels[0], channels[1], 3, pitch_um=pitch_um),
        LayerHardware("conv3_3x3", "shared_conv", image_size // 4, image_size // 4, channels[1], channels[2], 3, pitch_um=pitch_um),
        LayerHardware("class_1x1", "shared_conv", image_size // 4, image_size // 4, channels[2], 10, 1, pitch_um=pitch_um),
    ]


def summarize_layers(
    layers: list[LayerHardware],
    stochastic_cycles: int,
    comparator_cycles: int | None = None,
    wire_read_cycles: int | None = None,
    activity: float = 0.25,
    cfg: EnergyConfig | None = None,
) -> dict:
    synapses = sum(layer.synapse_instances for layer in layers)
    neurons = sum(layer.output_neurons for layer in layers)
    total_wire = sum(layer.total_wire_length_um for layer in layers)
    max_wire = max((layer.max_wire_length_um for layer in layers), default=0.0)
    params = sum(layer.parameter_count for layer in layers)
    energy = estimate_inference_energy(
        synapse_count=synapses,
        active_synapse_count=int(activity * synapses),
        total_wire_length_um=total_wire,
        neuron_count=neurons,
        stochastic_cycles=stochastic_cycles,
        wire_cycles=wire_read_cycles,
        read_cycles=wire_read_cycles,
        integrator_cycles=wire_read_cycles,
        comparator_cycles=comparator_cycles,
        activity=activity,
        cfg=cfg,
    )
    return {
        "synapse_instances": synapses,
        "active_synapse_instances": int(activity * synapses),
        "hardware_parameter_count": params,
        "neuron_decisions_per_cycle": neurons,
        "total_wire_length_um": total_wire,
        "max_wire_length_um": max_wire,
        "average_wire_length_um": total_wire / max(synapses, 1),
        **energy.to_dict(),
    }


def layer_table(layers: list[LayerHardware]) -> pd.DataFrame:
    return pd.DataFrame([layer.to_record() for layer in layers])
