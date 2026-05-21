"""Dataset loaders shared by the SPICENN run scripts.

These are the input-side helpers that prepare a list of training/eval
``records`` (one per sample) ready to be fed into a netlist generator.  Each
record is a plain ``dict`` so the run scripts can stay agnostic to dataset
provenance.

The functions previously lived inside ``run_device_xor2_random_hidden.py`` —
moving them here keeps that file focused on the XOR/MNIST direct-flow netlist
and lets other run scripts share the same encoders without copy-paste.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np


DATASET_EXAMPLES = [
    "xor2",
    "moons8",
    "moons12",
    "mnist01_8",
    "mnist01_12",
    "mnist01pool16_12",
    "mnist01fixed16_12",
    "mnist01fixed32_12",
    "mnist01sensory64_12",
    "mnist01rand16_12",
    "mnist3fixed8_30",
    "mnist5fixed8_50",
    "mnistfixed8_20",
    "mnistsensory64_20",
]


def _bit_value(pattern: int, bit: int) -> int:
    return (pattern >> bit) & 1


def xor_label(pattern: int) -> int:
    """Two-bit XOR truth-table label for a binary input ``pattern`` in 0..3."""
    return _bit_value(pattern, 0) ^ _bit_value(pattern, 1)


# ---------------------------------------------------------------------------
# Synthetic 2-D datasets
# ---------------------------------------------------------------------------


def two_moons_records(sample_count: int, seed: int) -> list[dict[str, Any]]:
    """Two-moons 2-D dataset, encoded as ``(x0, nx0, x1, nx1)`` rail inputs."""
    if sample_count % 2 != 0:
        raise ValueError("two-moons sample count must be even.")
    rng = np.random.default_rng(seed)
    half = sample_count // 2
    angles = np.linspace(0.15 * np.pi, 0.95 * np.pi, half)
    x0 = np.c_[np.cos(angles), np.sin(angles)]
    x1 = np.c_[1.0 - np.cos(angles), 1.0 - np.sin(angles) - 0.5]
    xy = np.vstack([x0, x1])
    labels = np.array([0] * half + [1] * half, dtype=int)
    xy += rng.normal(0.0, 0.035, size=xy.shape)
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    scaled = 0.08 + 0.84 * (xy - lo) / np.maximum(hi - lo, 1e-9)
    records: list[dict[str, Any]] = []
    for idx, ((x, y), label) in enumerate(zip(scaled, labels)):
        records.append(
            {
                "pattern": idx,
                "label": int(label),
                "inputs": {
                    "x0": float(x),
                    "nx0": float(1.0 - x),
                    "x1": float(y),
                    "nx1": float(1.0 - y),
                },
            }
        )
    return records


# ---------------------------------------------------------------------------
# MNIST front-ends
# ---------------------------------------------------------------------------


def dct2_lowfreq(image: np.ndarray, side: int) -> np.ndarray:
    """Low-frequency 2-D DCT block (top-left ``side`` x ``side`` coefficients)."""
    n = image.shape[0]
    coords = np.arange(n, dtype=np.float64)
    basis = []
    for k in range(side):
        alpha = np.sqrt(1.0 / n) if k == 0 else np.sqrt(2.0 / n)
        basis.append(alpha * np.cos(np.pi * (coords + 0.5) * k / n))
    mat = np.stack(basis)
    return mat @ image @ mat.T


def random_local_relu_features(image: np.ndarray, feature_count: int) -> np.ndarray:
    """Fixed (seeded) bank of random local ReLU features for retina-style input encodings."""
    rng = np.random.default_rng(17_003 + feature_count)
    features: list[float] = []
    for feature_idx in range(feature_count):
        patch_side = int(rng.choice([3, 4, 5]))
        row0 = int(rng.integers(0, image.shape[0] - patch_side + 1))
        col0 = int(rng.integers(0, image.shape[1] - patch_side + 1))
        patch = image[row0 : row0 + patch_side, col0 : col0 + patch_side]
        weights = rng.choice([-1.0, 0.0, 1.0], size=patch.shape, p=[0.35, 0.30, 0.35])
        if not np.any(weights):
            weights[patch_side // 2, patch_side // 2] = 1.0
        response = float(np.sum(weights * patch) / max(1, np.count_nonzero(weights)))
        bias = float(rng.uniform(-0.08, 0.08))
        if feature_idx % 2:
            response = -response
        features.append(max(0.0, response - bias))
    return np.asarray(features, dtype=np.float64)


def mnist01_frontend(image: np.ndarray, frontend: str) -> tuple[np.ndarray, str]:
    """Apply one of the named MNIST sensory front-ends to an 8x8 image."""
    if frontend == "pool2":
        return image.reshape(2, 4, 2, 4).mean(axis=(1, 3)).reshape(-1), "2x2_area_downsample"
    if frontend == "pool16":
        return image.reshape(4, 2, 4, 2).mean(axis=(1, 3)).reshape(-1), "4x4_area_downsample"
    if frontend == "fixed8":
        pooled = image.reshape(2, 4, 2, 4).mean(axis=(1, 3)).reshape(-1)
        lr = abs(float(image[:, :4].mean() - image[:, 4:].mean()))
        tb = abs(float(image[:4, :].mean() - image[4:, :].mean()))
        diag = abs(float(np.trace(image) / 8.0 - np.trace(np.fliplr(image)) / 8.0))
        center = float(image[2:6, 2:6].mean())
        return np.r_[pooled, [lr, tb, diag, center]], "2x2_pool_plus_global_haar_energy"
    if frontend == "fixed16":
        pooled = image.reshape(2, 4, 2, 4).mean(axis=(1, 3)).reshape(-1)
        features = [*pooled]
        for br in range(2):
            for bc in range(2):
                block = image[br * 4 : (br + 1) * 4, bc * 4 : (bc + 1) * 4]
                features.extend(
                    [
                        abs(float(block[:, :2].mean() - block[:, 2:].mean())),
                        abs(float(block[:2, :].mean() - block[2:, :].mean())),
                        abs(float(np.trace(block) / 4.0 - np.trace(np.fliplr(block)) / 4.0)),
                    ]
                )
        return np.asarray(features, dtype=np.float64), "2x2_pool_plus_local_haar_energy"
    if frontend == "haar16":
        features = []
        for br in range(2):
            for bc in range(2):
                block = image[br * 4 : (br + 1) * 4, bc * 4 : (bc + 1) * 4]
                features.extend(
                    [
                        float(block.mean()),
                        abs(float(block[:, :2].mean() - block[:, 2:].mean())),
                        abs(float(block[:2, :].mean() - block[2:, :].mean())),
                        abs(float(np.trace(block) / 4.0 - np.trace(np.fliplr(block)) / 4.0)),
                    ]
                )
        return np.asarray(features, dtype=np.float64), "local_haar_dc_and_energy"
    if frontend == "dct16":
        coeff = dct2_lowfreq(image, 4).reshape(-1)
        coeff[1:] = np.abs(coeff[1:])
        return coeff, "low_frequency_4x4_dct_abs_ac"
    if frontend == "fixed32":
        fixed, _fixed_desc = mnist01_frontend(image, "fixed16")
        dct, _dct_desc = mnist01_frontend(image, "dct16")
        return np.r_[fixed, dct], "local_haar_pool_plus_low_frequency_dct"
    sensory_match = re.fullmatch(r"sensory(\d+)", frontend)
    if sensory_match:
        feature_count = int(sensory_match.group(1))
        if not 32 <= feature_count <= 96:
            raise ValueError("sensory frontend feature count must be in 32..96.")
        fixed32, _fixed_desc = mnist01_frontend(image, "fixed32")
        random_count = feature_count - 32
        if random_count:
            random_features = random_local_relu_features(image, random_count)
            features = np.r_[fixed32, random_features]
            desc = f"local_haar_pool_low_frequency_dct_plus_{random_count}_random_local_relu"
        else:
            features = fixed32
            desc = "local_haar_pool_plus_low_frequency_dct"
        return features, desc
    random_match = re.fullmatch(r"rand(\d+)", frontend)
    if random_match:
        feature_count = int(random_match.group(1))
        if not 1 <= feature_count <= 64:
            raise ValueError("random local ReLU frontend feature count must be in 1..64.")
        return (
            random_local_relu_features(image, feature_count),
            f"{feature_count}_fixed_sparse_random_local_relu_filters",
        )
    raise ValueError(
        f"unknown MNIST01 frontend: {frontend}. "
        "Expected pool2, pool16, fixed8, fixed16, fixed32, haar16, dct16, sensoryN, or randN."
    )


def _load_mnist(root: Path, *, download: bool = False):
    """Return ``(dataset, labels_np, F)`` so the per-sample loop can pull tensors out."""
    from torch.nn import functional as F  # local import keeps torch optional
    from torchvision import datasets, transforms

    ds = datasets.MNIST(root=str(root / "data"), train=True, download=download, transform=transforms.ToTensor())
    labels_np = np.asarray(ds.targets)
    return ds, labels_np, F


def mnist01_records(
    sample_count: int,
    seed: int,
    frontend: str = "pool2",
    *,
    root: Path,
    download: bool = False,
) -> list[dict[str, Any]]:
    """Balanced binary subset of MNIST (digits 0 and 1) fed through ``frontend``."""
    if sample_count % 2 != 0:
        raise ValueError("MNIST 0/1 sample count must be even.")
    ds, labels_np, F = _load_mnist(root, download=download)
    rng = np.random.default_rng(seed)
    half = sample_count // 2
    selected: list[tuple[int, int]] = []
    for digit in [0, 1]:
        candidates = np.flatnonzero(labels_np == digit)
        chosen = rng.choice(candidates, size=half, replace=False)
        selected.extend((int(idx), digit) for idx in chosen)
    raw_features: list[np.ndarray] = []
    frontend_description = ""
    for idx, _label in selected:
        x, _digit = ds[idx]
        x8 = F.interpolate(x.unsqueeze(0), size=(8, 8), mode="area").squeeze().numpy().astype(np.float64)
        features, frontend_description = mnist01_frontend(x8, frontend)
        raw_features.append(features)
    raw = np.stack(raw_features)
    lo = raw.min(axis=0)
    hi = raw.max(axis=0)
    scaled = 0.08 + 0.84 * (raw - lo) / np.maximum(hi - lo, 1e-9)
    if frontend == "pool2":
        rail_names = ["x0", "nx0", "x1", "nx1"]
    else:
        rail_names = [f"x{i}" for i in range(scaled.shape[1])]
    records: list[dict[str, Any]] = []
    for pattern, ((idx, label), features) in enumerate(zip(selected, scaled)):
        records.append(
            {
                "pattern": pattern,
                "label": int(label),
                "source_index": idx,
                "source_digit": int(label),
                "input_frontend": f"{frontend_description}_per_feature_selected_subset_minmax_to_0p08_0p92",
                "input_frontend_key": frontend,
                "input_rails": rail_names,
                "inputs": {rail: float(value) for rail, value in zip(rail_names, features)},
            }
        )
    return records


def parse_counted_mnist_dataset(name: str) -> tuple[int, str, int] | None:
    """Split names like ``mnist3fixed8_30`` into ``(class_count, frontend, sample_count)``."""
    if name.startswith("mnist01"):
        return None
    counted = re.fullmatch(r"mnist(10|[2-9])([a-z][a-z0-9]*)?_(\d+)", name)
    if counted:
        return int(counted.group(1)), counted.group(2) or "fixed8", int(counted.group(3))
    ten_way = re.fullmatch(r"mnist([a-z0-9]*)_(\d+)", name)
    if ten_way:
        return 10, ten_way.group(1) or "fixed8", int(ten_way.group(2))
    return None


def mnist_records(
    sample_count: int,
    seed: int,
    frontend: str = "fixed8",
    class_count: int = 10,
    *,
    root: Path,
    download: bool = False,
) -> list[dict[str, Any]]:
    """Balanced ``class_count``-way MNIST subset fed through ``frontend``."""
    if not 2 <= class_count <= 10:
        raise ValueError("MNIST class count must be in 2..10.")
    if sample_count % class_count != 0:
        raise ValueError(f"{class_count}-way MNIST sample count must be divisible by {class_count}.")
    ds, labels_np, F = _load_mnist(root, download=download)
    rng = np.random.default_rng(seed)
    per_digit = sample_count // class_count
    selected: list[tuple[int, int]] = []
    for digit in range(class_count):
        candidates = np.flatnonzero(labels_np == digit)
        chosen = rng.choice(candidates, size=per_digit, replace=False)
        selected.extend((int(idx), digit) for idx in chosen)
    rng.shuffle(selected)

    raw_features: list[np.ndarray] = []
    frontend_description = ""
    for idx, _label in selected:
        x, _digit = ds[idx]
        x8 = F.interpolate(x.unsqueeze(0), size=(8, 8), mode="area").squeeze().numpy().astype(np.float64)
        features, frontend_description = mnist01_frontend(x8, frontend)
        raw_features.append(features)
    raw = np.stack(raw_features)
    lo = raw.min(axis=0)
    hi = raw.max(axis=0)
    scaled = 0.08 + 0.84 * (raw - lo) / np.maximum(hi - lo, 1e-9)
    rail_names = [f"x{i}" for i in range(scaled.shape[1])]
    records: list[dict[str, Any]] = []
    for pattern, ((idx, label), features) in enumerate(zip(selected, scaled)):
        records.append(
            {
                "pattern": pattern,
                "label": int(label),
                "source_index": idx,
                "source_digit": int(label),
                "input_frontend": f"{frontend_description}_per_feature_selected_subset_minmax_to_0p08_0p92",
                "input_frontend_key": frontend,
                "input_rails": rail_names,
                "inputs": {rail: float(value) for rail, value in zip(rail_names, features)},
            }
        )
    return records


def dataset_records(name: str, seed: int, *, root: Path, download: bool = False) -> list[dict[str, Any]]:
    """Top-level dataset dispatcher used by the run scripts."""
    if name == "xor2":
        return [{"pattern": p, "label": xor_label(p)} for p in range(4)]
    if name.startswith("moons"):
        suffix = name.removeprefix("moons").removeprefix("_")
        if suffix.isdigit():
            return two_moons_records(int(suffix), seed)
    mnist_match = re.fullmatch(r"mnist01([a-z0-9]*)_(\d+)", name)
    if mnist_match:
        frontend = mnist_match.group(1) or "pool2"
        return mnist01_records(int(mnist_match.group(2)), seed, frontend, root=root, download=download)
    counted_mnist = parse_counted_mnist_dataset(name)
    if counted_mnist:
        class_count, frontend, sample_count = counted_mnist
        return mnist_records(sample_count, seed, frontend, class_count=class_count, root=root, download=download)
    examples = ", ".join(
        DATASET_EXAMPLES
        + [
            "moons16",
            "mnist01_16",
            "mnist01fixed8_16",
            "mnist01fixed32_16",
            "mnist01rand8_16",
            "mnist3fixed8_60",
            "mnist5fixed8_100",
            "mnistfixed8_100",
        ]
    )
    raise ValueError(f"unknown dataset: {name}. Expected one of {examples} or another even-sized counted variant.")


__all__ = [
    "DATASET_EXAMPLES",
    "xor_label",
    "two_moons_records",
    "dct2_lowfreq",
    "random_local_relu_features",
    "mnist01_frontend",
    "mnist01_records",
    "parse_counted_mnist_dataset",
    "mnist_records",
    "dataset_records",
]
