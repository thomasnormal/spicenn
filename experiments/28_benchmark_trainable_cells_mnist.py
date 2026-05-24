#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import struct
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spicenn.cell import LOCAL_FEATURE_CELLS, LocalFeatureCell, characterize_local_feature_cell
from spicenn.cell.local_feature import (
    activation_derivative_np,
    activation_np,
    local_feature_cell_by_name,
    synapse_transfer_np,
)


MNIST_FILES = {
    "train_images": (
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
        ("train-images-idx3-ubyte.gz", "train_images.gz"),
    ),
    "train_labels": (
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
        ("train-labels-idx1-ubyte.gz", "train_labels.gz"),
    ),
    "test_images": (
        "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
        ("t10k-images-idx3-ubyte.gz", "test_images.gz"),
    ),
    "test_labels": (
        "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
        ("t10k-labels-idx1-ubyte.gz", "test_labels.gz"),
    ),
}


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=60) as response:
        tmp.write_bytes(response.read())
    tmp.replace(path)


def _idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path} is not an IDX image file")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(count, rows, cols)


def _idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path} is not an IDX label file")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(count)


def load_mnist_idx(root: Path, *, download: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    files: dict[str, Path] = {}
    for name, (url, candidates) in MNIST_FILES.items():
        existing = next((root / candidate for candidate in candidates if (root / candidate).exists()), None)
        files[name] = existing if existing is not None else root / candidates[0]
        if not files[name].exists():
            if not download:
                raise FileNotFoundError(f"missing {files[name]}; rerun with --download")
            _download(url, files[name])
    return (
        _idx_images(files["train_images"]),
        _idx_labels(files["train_labels"]),
        _idx_images(files["test_images"]),
        _idx_labels(files["test_labels"]),
    )


def resize_to_square(images: np.ndarray, size: int) -> np.ndarray:
    if images.shape[1:] == (size, size):
        out = images.astype(np.float64) / 255.0
        return 2.0 * out - 1.0
    src_h, src_w = images.shape[1:]
    ys = np.linspace(0, src_h - 1, size)
    xs = np.linspace(0, src_w - 1, size)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.clip(y0 + 1, 0, src_h - 1)
    x1 = np.clip(x0 + 1, 0, src_w - 1)
    wy = (ys - y0).reshape(1, size, 1)
    wx = (xs - x0).reshape(1, 1, size)
    src = images.astype(np.float64) / 255.0
    top = src[:, y0][:, :, x0] * (1.0 - wx) + src[:, y0][:, :, x1] * wx
    bot = src[:, y1][:, :, x0] * (1.0 - wx) + src[:, y1][:, :, x1] * wx
    out = top * (1.0 - wy) + bot * wy
    return 2.0 * out - 1.0


def quantize_inputs(x: np.ndarray, levels: int) -> np.ndarray:
    if levels <= 0:
        return x
    if levels < 2:
        raise ValueError("input quantization levels must be 0 or at least 2")
    y = np.clip((x + 1.0) * 0.5, 0.0, 1.0)
    q = np.round(y * (levels - 1)) / (levels - 1)
    return 2.0 * q - 1.0


def block_indices(image_size: int, block_size: int, stride: int) -> list[list[int]]:
    blocks = []
    for r in range(0, image_size - block_size + 1, stride):
        for c in range(0, image_size - block_size + 1, stride):
            blocks.append([(r + dr) * image_size + (c + dc) for dr in range(block_size) for dc in range(block_size)])
    return blocks


def block_tensor(x: np.ndarray, blocks: list[list[int]]) -> np.ndarray:
    flat = x.reshape((x.shape[0], -1))
    return np.stack([flat[:, block] for block in blocks], axis=1)


def softmax(score: np.ndarray, temperature: float) -> np.ndarray:
    z = score / max(float(temperature), 1e-12)
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def init_state(rng: np.random.Generator, n_blocks: int, channels: int, block_pixels: int, classes: int = 10):
    scale = 0.05
    return (
        rng.normal(0.0, scale, size=(n_blocks, channels, block_pixels)),
        rng.normal(0.0, scale, size=(n_blocks, channels)),
        rng.normal(0.0, scale, size=(classes, n_blocks, channels)),
        np.zeros(classes),
    )


def forward(
    x: np.ndarray,
    state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    blocks: list[list[int]],
    cell: LocalFeatureCell,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w, hb, readout, output_bias = state
    xb = block_tensor(x, blocks)
    eff_w = synapse_transfer_np(w, cell.hidden_synapse_mode, cell.synapse_clip)
    pre = np.einsum("nbp,bcp->nbc", xb, eff_w) + hb
    h = activation_np(pre, cell.local_activation, cell.relu_clip, cell.relu_leak, cell.softplus_beta)
    eff_readout = synapse_transfer_np(readout, cell.readout_synapse_mode, cell.synapse_clip)
    score = np.einsum("nbc,kbc->nk", h, eff_readout) + output_bias
    y = softmax(score, cell.softmax_temperature)
    return xb, pre, h, score, y


def accuracy(
    x: np.ndarray,
    labels: np.ndarray,
    state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    blocks: list[list[int]],
    cell: LocalFeatureCell,
    batch_size: int,
) -> float:
    correct = 0
    for start in range(0, len(labels), batch_size):
        _xb, _pre, _h, _score, y = forward(x[start : start + batch_size], state, blocks, cell)
        correct += int(np.sum(np.argmax(y, axis=1) == labels[start : start + batch_size]))
    return correct / max(len(labels), 1)


def update_one(
    x: np.ndarray,
    label: int,
    state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    blocks: list[list[int]],
    cell: LocalFeatureCell,
    lr: float,
    fixed_feedback: np.ndarray | None,
):
    w, hb, readout, output_bias = state
    xb, pre, h, _score, y = forward(x[None, ...], state, blocks, cell)
    d = -y
    d[0, int(label)] += 1.0
    eff_readout = synapse_transfer_np(readout, cell.readout_synapse_mode, cell.synapse_clip)
    if cell.readout_feedback_mode == "fixed-random":
        if fixed_feedback is None:
            raise ValueError("fixed feedback requested without fixed_feedback matrix")
        feedback = fixed_feedback
    elif cell.readout_feedback_mode in {"sign-readout", "sign"}:
        feedback = eff_readout / (np.abs(eff_readout) + 1e-9)
    elif cell.readout_feedback_mode in {"clipped-readout", "clipped"}:
        clip = max(float(cell.readout_feedback_clip), 1e-12)
        feedback = clip * np.tanh(eff_readout / clip)
    elif cell.readout_feedback_mode in {"readout", "full-readout", "exact"}:
        feedback = eff_readout
    else:
        raise ValueError(f"unknown feedback mode {cell.readout_feedback_mode!r}")
    deriv = activation_derivative_np(
        pre,
        h,
        cell.local_activation,
        cell.relu_clip,
        cell.relu_leak,
        cell.softplus_beta,
        cell.activation_derivative,
        cell.derivative_floor,
        cell.derivative_gate_threshold,
    )
    dh = np.einsum("nk,kbc->nbc", d, feedback) * deriv
    return (
        w + lr * cell.local_update_scale * np.einsum("nbc,nbp->bcp", dh, xb),
        hb + lr * cell.local_update_scale * dh[0],
        readout + lr * cell.readout_update_scale * np.einsum("nk,nbc->kbc", d, h),
        output_bias + lr * cell.output_bias_update_scale * d[0],
    )


def benchmark_cell(
    cell: LocalFeatureCell,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    blocks: list[list[int]],
    initial_state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    lr: float,
    eval_batch_size: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed + 7919)
    fixed_feedback = None
    if cell.readout_feedback_mode == "fixed-random":
        fixed_feedback = rng.normal(0.0, 0.1, size=initial_state[2].shape)
    state = tuple(arr.copy() for arr in initial_state)
    initial_acc = accuracy(x_eval, y_eval, state, blocks, cell, eval_batch_size)
    t0 = time.perf_counter()
    for x, label in zip(x_train, y_train):
        state = update_one(x, int(label), state, blocks, cell, lr, fixed_feedback)
    wall = time.perf_counter() - t0
    final_acc = accuracy(x_eval, y_eval, state, blocks, cell, eval_batch_size)
    char = characterize_local_feature_cell(cell, seed=seed)
    return {
        "cell": cell.name,
        "description": cell.description,
        "protocol_family": cell.protocol_family.value,
        "characterization_passed": char.passed,
        "characterization_update_cosine": char.update_cosine,
        "characterization_update_sign_alignment": char.update_sign_alignment,
        "local_activation": cell.local_activation,
        "hidden_synapse_mode": cell.hidden_synapse_mode,
        "readout_feedback_mode": cell.readout_feedback_mode,
        "initial_eval_accuracy": initial_acc,
        "final_eval_accuracy": final_acc,
        "eval_improvement": final_acc - initial_acc,
        "wall_time_s": wall,
    }


def selected_cells(names: str) -> tuple[LocalFeatureCell, ...]:
    if names == "all":
        return LOCAL_FEATURE_CELLS
    return tuple(local_feature_cell_by_name(name.strip()) for name in names.split(",") if name.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="all")
    ap.add_argument("--train-samples", type=int, default=256)
    ap.add_argument("--eval-samples", type=int, default=500)
    ap.add_argument("--image-size", type=int, default=10)
    ap.add_argument("--block-size", type=int, default=4)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--input-quantization-levels", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--mnist-root", type=Path, default=Path("data/MNIST/raw"))
    ap.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--tag", default="trainable_cells_mnist")
    args = ap.parse_args()

    if args.train_samples <= 0 or args.eval_samples <= 0:
        raise ValueError("sample counts must be positive")
    if args.lr < 0.0:
        raise ValueError("--lr must be non-negative")
    if args.input_quantization_levels < 0 or args.input_quantization_levels == 1:
        raise ValueError("--input-quantization-levels must be 0 or at least 2")

    train_images, train_labels, test_images, test_labels = load_mnist_idx(args.mnist_root, download=args.download)
    rng = np.random.default_rng(args.seed)
    train_idx = rng.permutation(len(train_labels))[: args.train_samples]
    eval_idx = rng.permutation(len(test_labels))[: args.eval_samples]
    x_train = resize_to_square(train_images[train_idx], args.image_size)
    x_eval = resize_to_square(test_images[eval_idx], args.image_size)
    x_train = quantize_inputs(x_train, args.input_quantization_levels)
    x_eval = quantize_inputs(x_eval, args.input_quantization_levels)
    y_train = train_labels[train_idx].astype(int)
    y_eval = test_labels[eval_idx].astype(int)

    blocks = block_indices(args.image_size, args.block_size, args.stride)
    initial_state = init_state(
        np.random.default_rng(args.seed),
        len(blocks),
        args.channels,
        args.block_size * args.block_size,
    )
    rows = [
        benchmark_cell(
            cell,
            x_train,
            y_train,
            x_eval,
            y_eval,
            blocks,
            initial_state,
            lr=args.lr,
            eval_batch_size=args.eval_batch_size,
            seed=args.seed,
        )
        for cell in selected_cells(args.cells)
    ]
    rows.sort(key=lambda row: row["final_eval_accuracy"], reverse=True)

    tables = Path("results/tables")
    tables.mkdir(parents=True, exist_ok=True)
    safe_tag = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in args.tag)
    csv_path = tables / f"{safe_tag}.csv"
    json_path = tables / f"{safe_tag}_summary.json"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "benchmark": "trainable_dynamical_cells_mnist_numpy",
        "dataset": "MNIST IDX train/test",
        "train_samples": args.train_samples,
        "eval_samples": args.eval_samples,
        "image_size": args.image_size,
        "block_size": args.block_size,
        "stride": args.stride,
        "channels": args.channels,
        "input_quantization_levels": args.input_quantization_levels,
        "lr": args.lr,
        "seed": args.seed,
        "train_order": "seeded_random",
        "local_pca_used": False,
        "rows_csv": str(csv_path),
        "best_cell": rows[0],
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
