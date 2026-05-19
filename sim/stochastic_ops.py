from __future__ import annotations

import torch


def bitstream_average(
    probability: torch.Tensor,
    cycles: int = 1,
    mode: str = "iid",
    generator: torch.Generator | None = None,
    shared_shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Return the finite-L stochastic bitstream average for Bernoulli probabilities."""
    if cycles <= 0:
        raise ValueError("cycles must be positive")
    p = probability.clamp(0.0, 1.0)
    if mode in {"expected", "mean"}:
        return p
    if mode in {"sampled_dithered_threshold_ramp", "ramp"}:
        thresholds = (torch.arange(cycles, device=p.device, dtype=p.dtype) + 0.5) / cycles
        return (p.unsqueeze(0) > thresholds.reshape((cycles,) + (1,) * p.ndim)).to(p.dtype).mean(0)
    if mode in {"sampled_stratified_threshold", "stratified"}:
        base = (torch.arange(cycles, device=p.device, dtype=p.dtype) + torch.rand(
            (cycles,), device=p.device, dtype=p.dtype, generator=generator
        )) / cycles
        return (p.unsqueeze(0) > base.reshape((cycles,) + (1,) * p.ndim)).to(p.dtype).mean(0)
    if mode in {"shared-global", "shared_global"}:
        r = torch.rand((cycles,), device=p.device, dtype=p.dtype, generator=generator)
        return (p.unsqueeze(0) > r.reshape((cycles,) + (1,) * p.ndim)).to(p.dtype).mean(0)
    if mode in {"shared-per-tile", "shared_tile"} and shared_shape is not None:
        r = torch.rand((cycles,) + shared_shape, device=p.device, dtype=p.dtype, generator=generator)
        while r.ndim < p.ndim + 1:
            r = r.unsqueeze(-1)
        return (p.unsqueeze(0) > r).to(p.dtype).mean(0)
    if mode in {"sampled_noisy_threshold_correlated", "correlated"}:
        common = torch.rand((cycles,) + (1,) * p.ndim, device=p.device, dtype=p.dtype, generator=generator)
        local = torch.rand((cycles,) + tuple(p.shape), device=p.device, dtype=p.dtype, generator=generator)
        r = 0.7 * common + 0.3 * local
        return (p.unsqueeze(0) > r).to(p.dtype).mean(0)
    r = torch.rand((cycles,) + tuple(p.shape), device=p.device, dtype=p.dtype, generator=generator)
    return (p.unsqueeze(0) > r).to(p.dtype).mean(0)


def low_bit_uniform(shape, bits: int = 4, device=None, dtype=torch.float32, generator=None) -> torch.Tensor:
    levels = 2 ** bits
    values = torch.randint(0, levels, shape, device=device, generator=generator)
    return (values.to(dtype) + 0.5) / levels


def slow_drift(shape, steps: int, sigma: float, rho: float = 0.995, device=None) -> torch.Tensor:
    out = torch.zeros((steps,) + tuple(shape), device=device)
    noise_scale = sigma * (1.0 - rho * rho) ** 0.5
    for t in range(1, steps):
        out[t] = rho * out[t - 1] + noise_scale * torch.randn(shape, device=device)
    return out

