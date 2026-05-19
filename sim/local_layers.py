from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .neuron_models import SpiceChargeADCLUT, activation_probability
from .stochastic_ops import bitstream_average


class StochasticActivation(nn.Module):
    def __init__(
        self,
        mode: str = "expected_probit_from_sigma",
        sigma: float = 1.0,
        theta: float = 0.0,
        cycles: int = 1,
        sample_mode: str = "expected",
        bits: int = 4,
        lut: SpiceChargeADCLUT | None = None,
    ):
        super().__init__()
        self.mode = mode
        self.sigma = sigma
        self.theta = theta
        self.cycles = cycles
        self.sample_mode = sample_mode
        self.bits = bits
        self.lut = lut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        direct_modes = {
            "clean_relu",
            "relu",
            "clean_silu_or_tanh",
            "silu",
            "relu6",
            "clipped_relu",
            "quantized_relu6",
            "multi_bit_relu6",
            "quantized_rectified_charge",
            "spice_charge_adc_lut",
            "time_to_threshold_code",
            "time_code",
            "quantized_time_to_threshold",
            "quantized_time_code",
            "tanh",
            "quantized_tanh",
            "multi_bit_tanh",
            "quantized_charge",
            "multi_bit_charge",
            "thermometer_probit",
            "multi_comparator_probit",
            "thermometer_probit_bipolar",
            "multi_comparator_bipolar",
            "expected_probit_bipolar",
            "bipolar_probit",
        }
        if self.mode in {"sampled_noisy_threshold_bipolar", "sampled_bipolar"}:
            p = activation_probability(x, "expected_probit_from_sigma", sigma=self.sigma, theta=self.theta, bits=self.bits)
            return 2.0 * bitstream_average(p, cycles=self.cycles, mode=self.sample_mode) - 1.0
        if self.mode == "spice_charge_adc_lut":
            if self.lut is None:
                raise ValueError("spice_charge_adc_lut mode requires a SpiceChargeADCLUT")
            return self.lut(x)
        p = activation_probability(x, self.mode, sigma=self.sigma, theta=self.theta, bits=self.bits)
        if self.mode in direct_modes:
            return p
        if self.mode.startswith("clean") or self.mode in {"relu", "silu", "tanh"}:
            return p
        return bitstream_average(p, cycles=self.cycles, mode=self.sample_mode)


def set_stochastic_mode(module: nn.Module, cycles: int = 1, sample_mode: str = "expected") -> None:
    for child in module.modules():
        if isinstance(child, StochasticActivation):
            child.cycles = cycles
            child.sample_mode = sample_mode


def set_activation_mode(
    module: nn.Module,
    mode: str,
    sigma: float | None = None,
    theta: float | None = None,
    bits: int | None = None,
    lut: SpiceChargeADCLUT | None = None,
) -> None:
    for child in module.modules():
        if isinstance(child, StochasticActivation):
            child.mode = mode
            if sigma is not None:
                child.sigma = sigma
            if theta is not None:
                child.theta = theta
            if bits is not None:
                child.bits = bits
            if lut is not None:
                child.lut = lut


def quantize_weight_ste(weight: torch.Tensor, bits: int | None) -> torch.Tensor:
    if bits is None or bits <= 0:
        return weight
    levels = 2 ** (int(bits) - 1) - 1
    if levels <= 0:
        return weight
    max_abs = weight.detach().abs().amax().clamp_min(1e-12)
    clipped = weight.clamp(-max_abs, max_abs)
    q = torch.round(clipped / max_abs * levels).clamp(-levels, levels) / levels * max_abs
    return clipped + (q - clipped).detach()


class DenseStochasticMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: tuple[int, ...] = (64,),
        num_classes: int = 10,
        activation: StochasticActivation | None = None,
    ):
        super().__init__()
        self.activation = activation or StochasticActivation()
        dims = (input_dim,) + tuple(hidden)
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.readout = nn.Linear(dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.readout(x)


class SharedConvNet(nn.Module):
    def __init__(self, in_channels: int = 1, channels: tuple[int, int] = (8, 16), num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, padding=1)
        self.readout = nn.Conv2d(channels[1], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            side = int(math.sqrt(x.shape[1]))
            x = x.reshape(x.shape[0], 1, side, side)
        x = F.avg_pool2d(F.relu(self.conv1(x)), 2)
        x = F.avg_pool2d(F.relu(self.conv2(x)), 2)
        x = self.readout(x).mean(dim=(2, 3))
        return x


class HardwareLocalEvidenceNet(nn.Module):
    """Local-only MNIST network with class evidence maps and global summation.

    The final 1x1 layer produces class evidence at each spatial tile; there is
    no dense readout. BatchNorm parameters can be folded into preceding
    conductance/bias settings for inference.
    """

    def __init__(
        self,
        channels: tuple[int, int, int] = (24, 48, 64),
        activation_sigma: float = 0.75,
        activation_bits: int = 4,
        num_classes: int = 10,
        clean_train: bool = False,
        coord_channels: bool = True,
        activation_mode: str | None = None,
        weight_bits: int | None = None,
        activation_lut: SpiceChargeADCLUT | None = None,
    ):
        super().__init__()
        self.coord_channels = coord_channels
        self.weight_bits = weight_bits
        in_channels = 3 if coord_channels else 1
        mode = activation_mode or ("clean_silu_or_tanh" if clean_train else "expected_probit_from_sigma")
        self.act1 = StochasticActivation(mode=mode, sigma=activation_sigma, bits=activation_bits, lut=activation_lut)
        self.act2 = StochasticActivation(mode=mode, sigma=activation_sigma, bits=activation_bits, lut=activation_lut)
        self.act3 = StochasticActivation(mode=mode, sigma=activation_sigma, bits=activation_bits, lut=activation_lut)
        self.conv1 = nn.Conv2d(in_channels, channels[0], 5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels[2])
        self.readout = nn.Conv2d(channels[2], num_classes, 1)

    def _conv(self, x: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
        return F.conv2d(
            x,
            quantize_weight_ste(conv.weight, self.weight_bits),
            conv.bias,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.reshape(x.shape[0], 1, 28, 28)
        if self.coord_channels:
            yy, xx = torch.meshgrid(
                torch.linspace(-1.0, 1.0, x.shape[-2], device=x.device, dtype=x.dtype),
                torch.linspace(-1.0, 1.0, x.shape[-1], device=x.device, dtype=x.dtype),
                indexing="ij",
            )
            coords = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            x = torch.cat([x, coords], dim=1)
        x = self.act1(self.bn1(self._conv(x, self.conv1)))
        x = F.avg_pool2d(x, 2)
        x = self.act2(self.bn2(self._conv(x, self.conv2)))
        x = F.avg_pool2d(x, 2)
        x = self.act3(self.bn3(self._conv(x, self.conv3)))
        return self._conv(x, self.readout).mean(dim=(2, 3))


class UnsharedLocal2d(nn.Module):
    """Locally connected 2D layer with no weight sharing."""

    def __init__(self, in_channels: int, out_channels: int, height: int, width: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.height = height
        self.width = width
        self.kernel_size = kernel_size
        fan_in = in_channels * kernel_size * kernel_size
        self.weight = nn.Parameter(torch.randn(height * width, out_channels, fan_in) / math.sqrt(fan_in))
        self.bias = nn.Parameter(torch.zeros(height * width, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = F.unfold(x, self.kernel_size, padding=self.kernel_size // 2)
        patches = patches.transpose(1, 2)
        out = torch.einsum("blf,lof->blo", patches, self.weight) + self.bias
        return out.transpose(1, 2).reshape(x.shape[0], self.out_channels, self.height, self.width)


class UnsharedLocalNet(nn.Module):
    def __init__(self, image_size: int = 8, channels: tuple[int, int] = (4, 8), num_classes: int = 10):
        super().__init__()
        self.local1 = UnsharedLocal2d(1, channels[0], image_size, image_size, 3)
        h2 = image_size // 2
        self.local2 = UnsharedLocal2d(channels[0], channels[1], h2, h2, 3)
        self.readout = nn.Conv2d(channels[1], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            side = int(math.sqrt(x.shape[1]))
            x = x.reshape(x.shape[0], 1, side, side)
        x = F.avg_pool2d(F.relu(self.local1(x)), 2)
        x = F.avg_pool2d(F.relu(self.local2(x)), 2)
        return self.readout(x).mean(dim=(2, 3))


class HierarchicalLocalNet(nn.Module):
    def __init__(self, image_size: int = 8, channels: tuple[int, int, int] = (4, 8, 12), num_classes: int = 10):
        super().__init__()
        self.l1 = UnsharedLocal2d(1, channels[0], image_size, image_size, 3)
        self.l2 = UnsharedLocal2d(channels[0], channels[1], image_size // 2, image_size // 2, 3)
        self.l3 = nn.Conv2d(channels[1], channels[2], 3, padding=1)
        self.readout = nn.Conv2d(channels[2], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            side = int(math.sqrt(x.shape[1]))
            x = x.reshape(x.shape[0], 1, side, side)
        x = F.avg_pool2d(F.relu(self.l1(x)), 2)
        x = F.avg_pool2d(F.relu(self.l2(x)), 2)
        x = F.relu(self.l3(x))
        return self.readout(x).mean(dim=(2, 3))


class MaskedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, mask: torch.Tensor, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer("mask", mask.to(torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)


class SmallWorldMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, shortcut_fraction: float = 0.01, num_classes: int = 10):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        mask = torch.zeros(hidden, input_dim)
        radius = max(1, input_dim // hidden)
        for j in range(hidden):
            center = int((j + 0.5) * input_dim / hidden)
            lo = max(0, center - radius)
            hi = min(input_dim, center + radius + 1)
            mask[j, lo:hi] = 1.0
        shortcuts = torch.rand(mask.shape, generator=g) < shortcut_fraction
        mask = torch.maximum(mask, shortcuts.to(mask.dtype))
        self.local = MaskedLinear(input_dim, hidden, mask)
        self.readout = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        return self.readout(F.relu(self.local(x)))


class RelayChainNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 96, chain_length: int = 2, num_classes: int = 10):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden), nn.ReLU()]
        for _ in range(max(chain_length - 1, 0)):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))


class LayerlessRecurrentSheetNet(nn.Module):
    """Parallel local relaxation sheet.

    All cells update in parallel for a fixed number of ticks using shared local
    recurrent kernels. This is a high-level model for a clocked or self-timed
    cellular array rather than a feed-forward layer stack.
    """

    def __init__(
        self,
        channels: int = 64,
        ticks: int = 6,
        activation_mode: str = "quantized_relu6",
        activation_bits: int = 4,
        activation_sigma: float = 0.5,
        coord_channels: bool = True,
        residual: float = 0.5,
        num_classes: int = 10,
    ):
        super().__init__()
        self.ticks = ticks
        self.coord_channels = coord_channels
        self.residual = residual
        in_channels = 3 if coord_channels else 1
        self.input_proj = nn.Conv2d(in_channels, channels, 1)
        self.local = nn.Conv2d(channels, channels, 3, padding=1, groups=1, bias=False)
        self.inhibit = nn.Conv2d(channels, channels, 1, groups=1, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.activation = StochasticActivation(
            mode=activation_mode,
            sigma=activation_sigma,
            bits=activation_bits,
        )
        self.readout = nn.Conv2d(channels, num_classes, 1)

    def _with_coords(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.reshape(x.shape[0], 1, 28, 28)
        if not self.coord_channels:
            return x
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, x.shape[-2], device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, x.shape[-1], device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        return torch.cat([x, coords], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        drive = self.input_proj(self._with_coords(x))
        state = self.activation(self.norm(drive))
        for _ in range(self.ticks):
            update = drive + self.local(state) - 0.1 * self.inhibit(state)
            new_state = self.activation(self.norm(update))
            state = self.residual * state + (1.0 - self.residual) * new_state
        return self.readout(state).mean(dim=(2, 3))
