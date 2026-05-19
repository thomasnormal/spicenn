import torch

from sim.neuron_models import ActivationCurve, SpiceChargeADCLUT, activation_probability
from sim.stochastic_ops import bitstream_average


def test_probit_midpoint_and_monotonicity():
    curve = ActivationCurve(theta=0.0, sigma=1.0)
    x = torch.tensor([-2.0, 0.0, 2.0])
    p = curve.probability(x)
    assert torch.isclose(p[1], torch.tensor(0.5), atol=1e-6)
    assert p[0] < p[1] < p[2]


def test_activation_modes_return_expected_shapes():
    x = torch.randn(4, 3)
    for mode in [
        "clean_relu",
        "hard_threshold",
        "expected_probit_from_sigma",
        "expected_logistic_from_sigma",
        "expected_probit_bipolar",
        "quantized_relu6",
        "time_to_threshold_code",
        "quantized_time_to_threshold",
        "thermometer_probit",
        "thermometer_probit_bipolar",
        "quantized_tanh",
        "quantized_charge",
    ]:
        y = activation_probability(x, mode=mode, sigma=0.5)
        assert y.shape == x.shape


def test_stratified_bitstream_has_bounds():
    p = torch.full((10,), 0.3)
    y = bitstream_average(p, cycles=8, mode="stratified")
    assert torch.all(y >= 0)
    assert torch.all(y <= 1)


def test_spice_charge_adc_lut_tmp(tmp_path):
    p = tmp_path / "charge.csv"
    p.write_text(
        "bits,cint_f,sigma_v,vin,mean_code_0_1\n"
        "4,1e-14,0.001,-1,0\n"
        "4,1e-14,0.001,0,0.5\n"
        "4,1e-14,0.001,1,1\n"
    )
    lut = SpiceChargeADCLUT(p, bits=4, cint_f=1e-14, sigma_v=0.001)
    y = lut(torch.tensor([-0.5, 0.0, 0.5]))
    assert torch.all(y >= 0)
    assert torch.all(y <= 1)

    rectified = SpiceChargeADCLUT(p, bits=4, cint_f=1e-14, sigma_v=0.001, output_mode="rectified")
    yr = rectified(torch.tensor([-0.5, 0.0, 0.5]))
    assert torch.isclose(yr[1], torch.tensor(0.0))
    assert yr[2] > yr[1]
