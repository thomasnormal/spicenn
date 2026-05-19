from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EnergyConfig:
    vdd: float = 0.8
    v_read: float = 0.1
    c_wire_per_um: float = 0.2e-15
    e_comp_per_decision: float = 5e-15
    e_program_pulse: float = 100e-15
    c_integrator: float = 10e-15
    v_swing: float = 0.1
    tau: float = 3e-9
    g_mean: float = 100e-9


@dataclass(frozen=True)
class EnergyBreakdown:
    wire_j: float
    synapse_read_j: float
    integrator_j: float
    comparator_j: float
    inference_j: float
    update_j: float
    stochastic_cycles: int

    def to_dict(self):
        return asdict(self)


def estimate_inference_energy(
    synapse_count: int,
    active_synapse_count: int,
    total_wire_length_um: float,
    neuron_count: int,
    stochastic_cycles: int = 1,
    wire_cycles: int | None = None,
    read_cycles: int | None = None,
    integrator_cycles: int | None = None,
    comparator_cycles: int | None = None,
    activity: float = 0.2,
    updates: int = 0,
    cfg: EnergyConfig | None = None,
) -> EnergyBreakdown:
    cfg = cfg or EnergyConfig()
    cycles = max(int(stochastic_cycles), 1)
    wire_n = max(int(wire_cycles if wire_cycles is not None else cycles), 1)
    read_n = max(int(read_cycles if read_cycles is not None else cycles), 1)
    int_n = max(int(integrator_cycles if integrator_cycles is not None else cycles), 1)
    comp_n = max(int(comparator_cycles if comparator_cycles is not None else cycles), 1)
    active = max(float(active_synapse_count), 0.0)
    wire_j = activity * cfg.c_wire_per_um * total_wire_length_um * cfg.vdd * cfg.vdd
    syn_read_j = active * cfg.tau * cfg.v_read * cfg.v_read * cfg.g_mean
    integrator_j = neuron_count * 0.5 * cfg.c_integrator * cfg.v_swing * cfg.v_swing
    comparator_j = neuron_count * cfg.e_comp_per_decision
    total_wire_j = wire_j * wire_n
    total_read_j = syn_read_j * read_n
    total_integrator_j = integrator_j * int_n
    total_comparator_j = comparator_j * comp_n
    update_j = updates * cfg.e_program_pulse
    return EnergyBreakdown(
        wire_j=total_wire_j,
        synapse_read_j=total_read_j,
        integrator_j=total_integrator_j,
        comparator_j=total_comparator_j,
        inference_j=total_wire_j + total_read_j + total_integrator_j + total_comparator_j,
        update_j=update_j,
        stochastic_cycles=cycles,
    )
