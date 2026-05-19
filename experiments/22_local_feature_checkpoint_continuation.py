from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spice"))

from run_spice_mnist_local_block_batch_op_train import block_indices  # noqa: E402
from run_spice_mnist_local_feature_phase_train import (  # noqa: E402
    fast_accuracy_np,
    load_or_init_weights,
    run_fast_reference_chunk,
    save_checkpoint,
    state_metrics,
)
from run_spice_mnist_local_feature_phase_transient import sanitize_tag  # noqa: E402
from run_spice_mnist_train import load_mnist_sequence  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-samples", type=int, default=2000)
    ap.add_argument("--test-samples", type=int, default=1000)
    ap.add_argument("--image-size", type=int, default=12)
    ap.add_argument("--block-size", type=int, default=6)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--updates-per-chunk", type=int, default=2)
    ap.add_argument("--start-chunk", type=int, default=0)
    ap.add_argument("--chunks", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--init-weights", required=True)
    ap.add_argument("--reference-weights", default="")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-batch-size", type=int, default=1024)
    ap.add_argument("--checkpoint-every", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="local_feature_checkpoint_continuation")
    args = ap.parse_args()

    if args.batch_size <= 0 or args.updates_per_chunk <= 0 or args.chunks <= 0:
        raise ValueError("--batch-size, --updates-per-chunk, and --chunks must be positive")
    if args.start_chunk < 0 or args.eval_every < 0 or args.checkpoint_every < 0:
        raise ValueError("--start-chunk, --eval-every, and --checkpoint-every must be non-negative")

    stride = args.block_size if args.stride is None else args.stride
    blocks = block_indices(args.image_size, args.block_size, stride)
    samples_per_chunk = args.batch_size * args.updates_per_chunk
    start_sample = args.start_chunk * samples_per_chunk
    end_sample = start_sample + args.chunks * samples_per_chunk
    if args.train_samples < end_sample:
        raise ValueError("--train-samples must cover start chunk plus requested chunks")

    x_train, y_train, x_test, y_test = load_mnist_sequence(
        args.train_samples,
        args.test_samples,
        args.image_size,
        args.seed,
    )
    rng = np.random.default_rng(args.seed)
    state = load_or_init_weights(
        args.init_weights,
        rng,
        len(blocks),
        args.channels,
        args.block_size * args.block_size,
    )
    reference_state = None
    if args.reference_weights:
        reference_state = load_or_init_weights(
            args.reference_weights,
            rng,
            len(blocks),
            args.channels,
            args.block_size * args.block_size,
        )

    safe_tag = sanitize_tag(args.tag)
    results = ROOT / "spice/results"
    tables = ROOT / "results/tables"
    checkpoint_dir = results / f"fast_checkpoint_continuation_{safe_tag}"
    for directory in [results, tables, checkpoint_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    t0 = time.perf_counter()

    def maybe_checkpoint(chunk: int) -> str:
        if args.checkpoint_every == 0:
            return ""
        if chunk % args.checkpoint_every != 0 and chunk != args.start_chunk + args.chunks:
            return ""
        path = checkpoint_dir / f"{safe_tag}_chunk{chunk:04d}_weights.npz"
        save_checkpoint(path, state)
        return str(path)

    def append_row(chunk: int, samples_seen: int, note: str, checkpoint: str = "") -> None:
        acc, correct = fast_accuracy_np(x_test, y_test, state, blocks, args.eval_batch_size)
        row: dict[str, float | int | str] = {
            "chunk": chunk,
            "samples_seen": samples_seen,
            "test_accuracy": acc,
            "test_correct": correct,
            "note": note,
            "checkpoint": checkpoint,
            "wall_time_s": time.perf_counter() - t0,
        }
        if reference_state is not None:
            row.update({f"reference_{k}": v for k, v in state_metrics(reference_state, state).items()})
        rows.append(row)

    append_row(args.start_chunk, start_sample, "initial")
    for local_chunk in range(1, args.chunks + 1):
        chunk = args.start_chunk + local_chunk
        start = start_sample + (local_chunk - 1) * samples_per_chunk
        stop = start + samples_per_chunk
        state = run_fast_reference_chunk(
            x_train[start:stop],
            y_train[start:stop],
            state,
            blocks,
            args.lr,
            args.batch_size,
            args.updates_per_chunk,
        )
        checkpoint = maybe_checkpoint(chunk)
        if args.eval_every and (chunk % args.eval_every == 0 or local_chunk == args.chunks):
            append_row(chunk, stop, "post_chunk_eval", checkpoint)

    final_weights = results / f"fast_mnist_{safe_tag}_final_weights.npz"
    save_checkpoint(final_weights, state)

    curve = pd.DataFrame(rows)
    curve_path = results / f"fast_mnist_{safe_tag}_curve.csv"
    table_curve_path = tables / f"fast_mnist_{safe_tag}_curve.csv"
    curve.to_csv(curve_path, index=False)
    curve.to_csv(table_curve_path, index=False)
    best_row = curve.loc[curve["test_accuracy"].idxmax()].to_dict()
    final_row = curve.iloc[-1].to_dict()
    summary = {
        "architecture": "fast_checkpoint_continuation_for_phase_portable_local_feature_rule",
        "dataset": "MNIST train/test split, downsampled",
        "image_size": args.image_size,
        "block_size": args.block_size,
        "stride": stride,
        "blocks": len(blocks),
        "channels": args.channels,
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
        "lr": args.lr,
        "init_weights": args.init_weights,
        "reference_weights": args.reference_weights,
        "best_eval": best_row,
        "final_eval": final_row,
        "final_weights": str(final_weights),
        "curve": str(curve_path),
        "table_curve": str(table_curve_path),
        "wall_time_s": time.perf_counter() - t0,
        "note": (
            "Fast reference continuation from a local-feature checkpoint. This predicts which "
            "phase-training window is worth spending ngspice transient time on."
        ),
    }
    summary_path = results / f"fast_mnist_{safe_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
