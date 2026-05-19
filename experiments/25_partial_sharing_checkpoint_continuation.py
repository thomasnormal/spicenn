from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spice"))

from run_spice_mnist_local_block_batch_op_train import block_indices  # noqa: E402
from run_spice_mnist_local_feature_phase_transient import sanitize_tag  # noqa: E402
from run_spice_mnist_local_feature_torch_train import make_block_tensor  # noqa: E402
from run_spice_mnist_train import load_mnist_sequence  # noqa: E402


def load_partial_module() -> Any:
    path = ROOT / "experiments/24_local_feature_partial_sharing_frontier.py"
    spec = importlib.util.spec_from_file_location("partial_sharing_frontier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PARTIAL = load_partial_module()


def load_partial_checkpoint(path: Path, blocks: int, block_len: int) -> tuple[torch.Tensor, ...]:
    data = np.load(path, allow_pickle=False)
    required = [
        "shared_local_weights",
        "shared_local_bias",
        "private_local_weights",
        "private_local_bias",
        "readout",
        "output_bias",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path} is missing partial-sharing arrays: {missing}")
    shared_w = torch.tensor(data["shared_local_weights"], dtype=torch.float32)
    shared_bias = torch.tensor(data["shared_local_bias"], dtype=torch.float32)
    private_w = torch.tensor(data["private_local_weights"], dtype=torch.float32)
    private_bias = torch.tensor(data["private_local_bias"], dtype=torch.float32)
    readout = torch.tensor(data["readout"], dtype=torch.float32)
    output_bias = torch.tensor(data["output_bias"], dtype=torch.float32)

    if shared_w.ndim != 2 or shared_w.shape[1] != block_len:
        raise ValueError(f"shared_local_weights has shape {tuple(shared_w.shape)}, expected (*, {block_len})")
    if shared_bias.shape != (shared_w.shape[0],):
        raise ValueError("shared_local_bias shape does not match shared_local_weights")
    if private_w.ndim != 3 or private_w.shape[0] != blocks or private_w.shape[2] != block_len:
        raise ValueError(
            f"private_local_weights has shape {tuple(private_w.shape)}, expected ({blocks}, *, {block_len})"
        )
    if private_bias.shape != private_w.shape[:2]:
        raise ValueError("private_local_bias shape does not match private_local_weights")
    channels = shared_w.shape[0] + private_w.shape[1]
    if readout.shape != (10, blocks, channels):
        raise ValueError(f"readout has shape {tuple(readout.shape)}, expected (10, {blocks}, {channels})")
    if output_bias.shape != (10,):
        raise ValueError("output_bias shape must be (10,)")
    return shared_w, shared_bias, private_w, private_bias, readout, output_bias


def save_partial_checkpoint(path: Path, state: tuple[torch.Tensor, ...], blocks: int) -> None:
    PARTIAL.save_checkpoint(path, state, blocks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-samples", type=int, default=5000)
    ap.add_argument("--test-samples", type=int, default=2000)
    ap.add_argument("--image-size", type=int, default=7)
    ap.add_argument("--block-size", type=int, default=3)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--updates-per-chunk", type=int, default=2)
    ap.add_argument("--start-chunk", type=int, default=0)
    ap.add_argument("--chunks", type=int, default=53)
    ap.add_argument("--lr-spice", type=float, default=0.02)
    ap.add_argument("--init-weights", required=True)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-batch-size", type=int, default=1024)
    ap.add_argument("--checkpoint-every", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="partial_sharing_checkpoint_continuation")
    args = ap.parse_args()

    if args.batch_size <= 0 or args.updates_per_chunk <= 0 or args.chunks <= 0:
        raise ValueError("--batch-size, --updates-per-chunk, and --chunks must be positive")
    if args.start_chunk < 0 or args.eval_every < 0 or args.checkpoint_every < 0:
        raise ValueError("--start-chunk, --eval-every, and --checkpoint-every must be non-negative")

    stride = args.block_size if args.stride is None else args.stride
    blocks = block_indices(args.image_size, args.block_size, stride)
    block_len = args.block_size * args.block_size
    samples_per_chunk = args.batch_size * args.updates_per_chunk
    start_sample = args.start_chunk * samples_per_chunk
    end_sample = start_sample + args.chunks * samples_per_chunk
    if args.train_samples < end_sample:
        raise ValueError("--train-samples must cover start chunk plus requested chunks")

    x_train_np, y_train_np, x_test_np, y_test_np = load_mnist_sequence(
        args.train_samples,
        args.test_samples,
        args.image_size,
        args.seed,
    )
    x_train = make_block_tensor(x_train_np, blocks)
    x_test = make_block_tensor(x_test_np, blocks)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    y_test = torch.tensor(y_test_np, dtype=torch.long)

    state = load_partial_checkpoint(Path(args.init_weights), len(blocks), block_len)
    shared_channels = int(state[0].shape[0])
    private_channels = int(state[2].shape[1])
    channels = shared_channels + private_channels

    safe_tag = sanitize_tag(args.tag)
    results = ROOT / "spice/results"
    tables = ROOT / "results/tables"
    checkpoint_dir = results / f"partial_sharing_checkpoint_continuation_{safe_tag}"
    for directory in [results, tables, checkpoint_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    gen = torch.Generator()
    gen.manual_seed(args.seed + int(1e6 * args.lr_spice) + 549755813)
    rows: list[dict[str, float | int | str]] = []
    t0 = time.perf_counter()

    def maybe_checkpoint(chunk: int) -> str:
        if args.checkpoint_every == 0:
            return ""
        if chunk % args.checkpoint_every != 0 and chunk != args.start_chunk + args.chunks:
            return ""
        path = checkpoint_dir / f"{safe_tag}_chunk{chunk:04d}_weights.npz"
        save_partial_checkpoint(path, state, len(blocks))
        return str(path)

    def append_row(chunk: int, samples_seen: int, note: str, checkpoint: str = "") -> None:
        acc, correct = PARTIAL.accuracy(x_test, y_test, *state, args.eval_batch_size)
        rows.append(
            {
                "chunk": chunk,
                "samples_seen": samples_seen,
                "test_accuracy": acc,
                "test_correct": correct,
                "note": note,
                "checkpoint": checkpoint,
                "wall_time_s": time.perf_counter() - t0,
            }
        )

    append_row(args.start_chunk, start_sample, "initial")
    for local_chunk in range(1, args.chunks + 1):
        chunk = args.start_chunk + local_chunk
        start = start_sample + (local_chunk - 1) * samples_per_chunk
        stop = start + samples_per_chunk
        for batch_start in range(start, stop, args.batch_size):
            batch_stop = batch_start + args.batch_size
            state = PARTIAL.hardware_update(
                x_train[batch_start:batch_stop],
                y_train[batch_start:batch_stop],
                *state,
                args.lr_spice,
                0.0,
                gen,
            )
        checkpoint = maybe_checkpoint(chunk)
        if args.eval_every and (chunk % args.eval_every == 0 or local_chunk == args.chunks):
            append_row(chunk, stop, "post_chunk_eval", checkpoint)

    final_weights = results / f"fast_mnist_{safe_tag}_final_weights.npz"
    save_partial_checkpoint(final_weights, state, len(blocks))

    curve = pd.DataFrame(rows)
    curve_path = results / f"fast_mnist_{safe_tag}_curve.csv"
    table_curve_path = tables / f"fast_mnist_{safe_tag}_curve.csv"
    curve.to_csv(curve_path, index=False)
    curve.to_csv(table_curve_path, index=False)
    best_row = curve.loc[curve["test_accuracy"].idxmax()].to_dict()
    final_row = curve.iloc[-1].to_dict()
    summary = {
        "architecture": "fast_partial_sharing_checkpoint_continuation",
        "dataset": "MNIST train/test split, downsampled",
        "image_size": args.image_size,
        "block_size": args.block_size,
        "stride": stride,
        "blocks": len(blocks),
        "shared_channels": shared_channels,
        "private_channels": private_channels,
        "channels": channels,
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "batch_size": args.batch_size,
        "updates_per_chunk": args.updates_per_chunk,
        "samples_per_chunk": samples_per_chunk,
        "start_chunk": args.start_chunk,
        "end_chunk": args.start_chunk + args.chunks,
        "start_sample": start_sample,
        "end_sample": end_sample,
        "chunks": args.chunks,
        "lr_spice": args.lr_spice,
        "init_weights": args.init_weights,
        "best_eval": best_row,
        "final_eval": final_row,
        "final_weights": str(final_weights),
        "curve": str(curve_path),
        "table_curve": str(table_curve_path),
        "wall_time_s": time.perf_counter() - t0,
        "note": (
            "Fast continuation that preserves partial-sharing capacitor semantics: shared channels "
            "receive summed block-position gradients, private channels remain per-block."
        ),
    }
    summary_path = results / f"fast_mnist_{safe_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
