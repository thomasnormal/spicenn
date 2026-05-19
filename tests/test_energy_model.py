from sim.energy_model import estimate_inference_energy


def test_energy_scales_with_cycles():
    e1 = estimate_inference_energy(10, 10, 100.0, 5, stochastic_cycles=1)
    e8 = estimate_inference_energy(10, 10, 100.0, 5, stochastic_cycles=8)
    assert e8.inference_j == 8 * e1.inference_j
    assert e1.wire_j > 0

