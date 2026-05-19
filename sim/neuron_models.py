from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ActivationCurve:
    theta: float = 0.0
    sigma: float = 1.0
    kind: str = "probit"
    slope: Optional[float] = None
    source: str = "analytic"

    def probability(self, x: torch.Tensor) -> torch.Tensor:
        sigma = max(float(self.sigma), 1e-12)
        z = (x - float(self.theta)) / sigma
        if self.kind == "logistic":
            return torch.sigmoid(z)
        if self.kind == "hard":
            return (z > 0).to(x.dtype)
        return 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))


class SpiceLUTActivation:
    """Interpolated SPICE-derived P(out=1 | signal) lookup table."""

    def __init__(self, csv_path: str | Path):
        df = pd.read_csv(csv_path).sort_values("signal")
        if "p_high" not in df.columns:
            raise ValueError("LUT CSV must contain columns signal,p_high")
        self.signal = torch.tensor(df["signal"].to_numpy(dtype=np.float32))
        self.p_high = torch.tensor(df["p_high"].to_numpy(dtype=np.float32))
        self.source = str(csv_path)

    def probability(self, x: torch.Tensor) -> torch.Tensor:
        signal = self.signal.to(x.device)
        p_high = self.p_high.to(x.device)
        flat = x.reshape(-1)
        idx = torch.searchsorted(signal, flat).clamp(1, len(signal) - 1)
        lo = idx - 1
        hi = idx
        denom = (signal[hi] - signal[lo]).clamp_min(1e-12)
        w = (flat - signal[lo]) / denom
        out = p_high[lo] * (1.0 - w) + p_high[hi] * w
        return out.reshape_as(x).clamp(0.0, 1.0)


class SpiceChargeADCLUT:
    """Interpolated SPICE charge-ADC code curve with STE-style gradients."""

    def __init__(
        self,
        csv_path: str | Path,
        bits: int = 4,
        cint_f: float = 10e-15,
        sigma_v: float | None = None,
        input_scale: float = 1.0,
        output_mode: str = "unipolar",
    ):
        df = pd.read_csv(csv_path)
        df = df[(df["bits"] == bits) & np.isclose(df["cint_f"], cint_f)]
        if sigma_v is not None:
            df = df[np.isclose(df["sigma_v"], sigma_v)]
        if df.empty:
            raise ValueError(f"No SPICE charge ADC rows for bits={bits}, C={cint_f}, sigma={sigma_v}")
        value_col = "mean_code_bipolar" if output_mode == "bipolar" else "mean_code_0_1"
        grouped = df.groupby("vin", as_index=False)[value_col].mean().sort_values("vin")
        values = grouped[value_col].to_numpy(dtype=np.float32)
        if output_mode == "rectified":
            vin_vals = grouped["vin"].to_numpy(dtype=np.float32)
            zero_code = np.interp(0.0, vin_vals, values)
            denom = max(float(values.max() - zero_code), 1e-12)
            values = np.clip((values - zero_code) / denom, 0.0, 1.0).astype(np.float32)
        elif output_mode not in {"unipolar", "bipolar"}:
            raise ValueError("output_mode must be one of: unipolar, bipolar, rectified")
        self.vin = torch.tensor(grouped["vin"].to_numpy(dtype=np.float32))
        self.code = torch.tensor(values)
        self.bits = bits
        self.cint_f = cint_f
        self.input_scale = float(input_scale)
        self.output_mode = output_mode
        self.source = str(csv_path)
        diffs = np.diff(grouped["vin"].to_numpy(dtype=np.float32))
        self._uniform_step = float(diffs[0]) if len(diffs) and np.allclose(diffs, diffs[0]) else None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        vin = self.vin.to(x.device)
        code = self.code.to(x.device)
        xv = x * self.input_scale
        flat = xv.reshape(-1)
        if self._uniform_step is not None:
            pos = ((flat - float(vin[0])) / self._uniform_step).clamp(0.0, len(vin) - 1.0)
            lo = torch.floor(pos).to(torch.long).clamp(0, len(vin) - 2)
            hi = lo + 1
            w = (pos - lo.to(pos.dtype)).clamp(0.0, 1.0)
        else:
            idx = torch.searchsorted(vin, flat).clamp(1, len(vin) - 1)
            lo = idx - 1
            hi = idx
            denom = (vin[hi] - vin[lo]).clamp_min(1e-12)
            w = (flat - vin[lo]) / denom
        y = code[lo] * (1.0 - w) + code[hi] * w
        lo_clip, hi_clip = (-1.0, 1.0) if self.output_mode == "bipolar" else (0.0, 1.0)
        y = y.reshape_as(x).clamp(lo_clip, hi_clip)
        # STE gradient through a clipped linear voltage range.
        surrogate = ((xv - float(vin.min())) / float(vin.max() - vin.min())).clamp(0.0, 1.0)
        if self.output_mode == "bipolar":
            surrogate = 2.0 * surrogate - 1.0
        return surrogate + (y - surrogate).detach()


def activation_probability(
    x: torch.Tensor,
    mode: str = "expected_probit_from_sigma",
    sigma: float = 1.0,
    theta: float = 0.0,
    lut: Optional[SpiceLUTActivation] = None,
    bits: int = 4,
    value_min: float = -1.0,
    value_max: float = 1.0,
) -> torch.Tensor:
    if mode in {"clean_relu", "relu"}:
        return F.relu(x)
    if mode in {"clean_silu_or_tanh", "silu"}:
        return F.silu(x)
    if mode in {"relu6", "clipped_relu"}:
        return F.relu6(x)
    if mode == "tanh":
        return torch.tanh(x)
    if mode in {"quantized_relu6", "multi_bit_relu6", "quantized_rectified_charge"}:
        return quantize_ste(F.relu6(x), bits=bits, value_min=0.0, value_max=6.0)
    if mode in {"time_to_threshold_code", "time_code"}:
        drive = F.softplus(x)
        threshold = max(float(sigma), 1e-6)
        return torch.clamp(1.0 - threshold / (drive + 1e-6), 0.0, 1.0)
    if mode in {"quantized_time_to_threshold", "quantized_time_code"}:
        drive = F.softplus(x)
        threshold = max(float(sigma), 1e-6)
        y = torch.clamp(1.0 - threshold / (drive + 1e-6), 0.0, 1.0)
        return quantize_ste(y, bits=bits, value_min=0.0, value_max=1.0)
    if mode in {"quantized_tanh", "multi_bit_tanh"}:
        return quantize_ste(torch.tanh(x), bits=bits, value_min=value_min, value_max=value_max)
    if mode in {"quantized_charge", "multi_bit_charge"}:
        # A clipped charge/voltage value digitized to a few local levels.
        y = torch.clamp((x - theta) / (6.0 * max(float(sigma), 1e-12)) + 0.5, 0.0, 1.0)
        return quantize_ste(y, bits=bits, value_min=0.0, value_max=1.0)
    if mode in {"thermometer_probit", "multi_comparator_probit"}:
        return thermometer_probit(x, sigma=sigma, theta=theta, bits=bits, bipolar=False)
    if mode in {"thermometer_probit_bipolar", "multi_comparator_bipolar"}:
        return thermometer_probit(x, sigma=sigma, theta=theta, bits=bits, bipolar=True)
    if mode in {"hard_threshold", "hard"}:
        return (x > theta).to(x.dtype)
    if mode in {"expected_probit_bipolar", "bipolar_probit"}:
        p = ActivationCurve(theta=theta, sigma=sigma, kind="probit").probability(x)
        return 2.0 * p - 1.0
    if mode in {"expected_logistic_from_sigma", "logistic"}:
        return ActivationCurve(theta=theta, sigma=sigma, kind="logistic").probability(x)
    if mode in {"sampled_noisy_threshold_from_spice_lut", "expected_lut"}:
        if lut is None:
            raise ValueError("SPICE LUT activation requested without a LUT")
        return lut.probability(x)
    return ActivationCurve(theta=theta, sigma=sigma, kind="probit").probability(x)


def quantize_ste(x: torch.Tensor, bits: int = 4, value_min: float = -1.0, value_max: float = 1.0) -> torch.Tensor:
    levels = max(2, 2 ** int(bits))
    lo = float(value_min)
    hi = float(value_max)
    y = x.clamp(lo, hi)
    q = torch.round((y - lo) * (levels - 1) / (hi - lo)) * (hi - lo) / (levels - 1) + lo
    return y + (q - y).detach()


def thermometer_probit(
    x: torch.Tensor,
    sigma: float = 1.0,
    theta: float = 0.0,
    bits: int = 4,
    bipolar: bool = False,
) -> torch.Tensor:
    levels = max(2, 2 ** int(bits))
    # Multi-comparator ADC-like code: thresholds span the useful transition band.
    thresholds = torch.linspace(-3.0, 3.0, levels - 1, device=x.device, dtype=x.dtype)
    thresholds = theta + max(float(sigma), 1e-12) * thresholds
    z = (x.unsqueeze(0) - thresholds.reshape((-1,) + (1,) * x.ndim)) / max(float(sigma), 1e-12)
    p = 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))
    y = p.mean(dim=0)
    if bipolar:
        return 2.0 * y - 1.0
    return y


class StochasticThresholdSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, sigma: float, theta: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.sigma = sigma
        ctx.theta = theta
        p = activation_probability(x, "expected_probit_from_sigma", sigma=sigma, theta=theta)
        return torch.bernoulli(p)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        sigma = max(float(ctx.sigma), 1e-12)
        z = (x - float(ctx.theta)) / sigma
        pdf = torch.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))
        return grad_output * pdf, None, None


def stochastic_threshold_ste(x: torch.Tensor, sigma: float = 1.0, theta: float = 0.0) -> torch.Tensor:
    return StochasticThresholdSTE.apply(x, sigma, theta)
