from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_spice_mnist_batch_op_train import read_wrdata_row
from run_spice_mnist_local_block_batch_op_train import block_indices
from run_spice_mnist_local_feature_batch_op_train import run_eval
from run_spice_mnist_local_feature_phase_transient import parse_measured_vector, sanitize_tag
from run_spice_mnist_partial_sharing_phase_transient import (
    PartialState,
    expand_local_weights,
    fast_forward_np,
    load_partial_checkpoint,
    make_partial_phase_netlist,
    run_fast_reference_chunk,
    save_partial_checkpoint,
    state_metrics,
    unpack_partial_state,
)
from run_spice_mnist_train import load_mnist_sequence
from run_spice_sweep import ROOT, detect_spice, run_tiny_test


def run_phase_chunk(
    spice_bin: str,
    netlist_path: Path,
    data_path: Path,
    x_chunk: np.ndarray,
    y_chunk: np.ndarray,
    state: PartialState,
    blocks: list[list[int]],
    lr: float,
    batch_size: int,
    updates_per_chunk: int,
    phase: float,
    gap: float,
    edge: float,
    settle_ratio: float,
    transient_step: float,
    cw: float,
    cstate: float,
    cgrad: float,
    rleak: float,
    timeout: float,
    final_measures: bool,
) -> tuple[PartialState, dict[str, float | str]]:
    netlist, n_vec, t_stop = make_partial_phase_netlist(
        x_chunk,
        y_chunk,
        state,
        blocks,
        lr,
        data_path,
        batch_size,
        updates_per_chunk,
        phase,
        gap,
        edge,
        settle_ratio,
        transient_step,
        cw,
        cstate,
        cgrad,
        rleak,
        "measure" if final_measures else "wrdata",
    )
    netlist_path.write_text(netlist)
    t0 = time.perf_counter()
    proc = subprocess.run([spice_bin, "-b", str(netlist_path)], text=True, capture_output=True, timeout=timeout)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-3000:] or proc.stdout[-3000:])
    vals = parse_measured_vector(proc.stdout + "\n" + proc.stderr, n_vec) if final_measures else read_wrdata_row(data_path, n_vec)
    next_state, _y = unpack_partial_state(vals, state)
    return next_state, {"phase_wall_time_s": wall, "t_stop_s": t_stop, "netlist": str(netlist_path), "data": str(data_path)}


def fast_accuracy_np(x: np.ndarray, labels: np.ndarray, state: PartialState, blocks: list[list[int]], batch_size: int) -> tuple[float, int]:
    preds = []
    for start in range(0, len(labels), batch_size):
        _h, _score, y = fast_forward_np(x[start : start + batch_size], state, blocks)
        preds.append(np.argmax(y, axis=1))
    pred = np.concatenate(preds) if preds else np.array([], dtype=int)
    correct = int(np.sum(pred == labels))
    return correct / max(len(labels), 1), correct


def run_spice_eval(
    spice_bin: str,
    netlist_path: Path,
    data_path: Path,
    x_test: np.ndarray,
    y_test: np.ndarray,
    state: PartialState,
    blocks: list[list[int]],
    batch_size: int,
    timeout: float,
) -> float:
    w, hb = expand_local_weights(state, len(blocks))
    _shared_w, _shared_bias, _private_w, _private_bias, readout, output_bias = state
    return run_eval(
        spice_bin,
        netlist_path,
        data_path,
        x_test,
        y_test,
        w,
        hb,
        readout,
        output_bias,
        blocks,
        batch_size,
        timeout,
        False,
        False,
        "tanh",
        1.0,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-samples", type=int, default=5000)
    ap.add_argument("--test-samples", type=int, default=1000)
    ap.add_argument("--image-size", type=int, default=7)
    ap.add_argument("--block-size", type=int, default=3)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--updates-per-chunk", type=int, default=2)
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--start-chunk", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--init-weights", required=True)
    ap.add_argument("--reference-init-weights", default="")
    ap.add_argument("--track-fast-reference", action="store_true")
    ap.add_argument("--checkpoint-every", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--fast-eval-batch-size", type=int, default=1024)
    ap.add_argument("--skip-initial-eval", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--phase", type=float, default=1e-9)
    ap.add_argument("--gap", type=float, default=0.2e-9)
    ap.add_argument("--edge", type=float, default=10e-12)
    ap.add_argument("--settle-ratio", type=float, default=80.0)
    ap.add_argument("--transient-step", type=float, default=25e-12)
    ap.add_argument("--cw", type=float, default=1e-12)
    ap.add_argument("--cstate", type=float, default=1e-12)
    ap.add_argument("--cgrad", type=float, default=1e-12)
    ap.add_argument("--rleak", type=float, default=1e18)
    ap.add_argument("--final-measures", action="store_true")
    ap.add_argument("--tag", default="partial_sharing_phase_train")
    args = ap.parse_args()

    if args.batch_size <= 0 or args.updates_per_chunk <= 0 or args.chunks <= 0:
        raise ValueError("--batch-size, --updates-per-chunk, and --chunks must be positive")
    if args.start_chunk < 0 or args.checkpoint_every < 0 or args.eval_every < 0:
        raise ValueError("--start-chunk, --checkpoint-every, and --eval-every must be non-negative")
    if args.phase <= 0 or args.settle_ratio <= 0:
        raise ValueError("--phase and --settle-ratio must be positive")

    stride = args.block_size if args.stride is None else args.stride
    blocks = block_indices(args.image_size, args.block_size, stride)
    samples_per_chunk = args.batch_size * args.updates_per_chunk
    phase_train_samples_this_run = samples_per_chunk * args.chunks
    start_sample = args.start_chunk * samples_per_chunk
    end_sample = start_sample + phase_train_samples_this_run
    if args.train_samples < end_sample:
        raise ValueError("--train-samples must cover requested chunk window")

    x_train, y_train, x_test, y_test = load_mnist_sequence(
        args.train_samples,
        args.test_samples,
        args.image_size,
        args.seed,
    )
    state = load_partial_checkpoint(Path(args.init_weights), len(blocks), args.block_size * args.block_size)
    reference_state = None
    if args.track_fast_reference:
        reference_state = (
            load_partial_checkpoint(Path(args.reference_init_weights), len(blocks), args.block_size * args.block_size)
            if args.reference_init_weights
            else tuple(arr.copy() for arr in state)
        )

    shared_channels = state[0].shape[0]
    private_channels = state[2].shape[1]
    channels = shared_channels + private_channels

    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    results = ROOT / "spice/results"
    tables = ROOT / "results/tables"
    for directory in [generated, results, tables]:
        directory.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)

    safe_tag = sanitize_tag(args.tag)
    stem = f"spice_mnist_partial_sharing_phase_train_{safe_tag}"
    eval_netlist = generated / f"{stem}_eval.cir"
    eval_data = results / f"{stem}_eval.dat"
    checkpoint_dir = results / f"{stem}_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    curve_rows: list[dict[str, float | int | str]] = []
    phase_wall_total = 0.0
    eval_wall_total = 0.0

    def tracking_fields() -> dict[str, float | int]:
        if reference_state is None:
            return {}
        metrics = state_metrics(reference_state, state)
        phase_acc, phase_correct = fast_accuracy_np(x_test, y_test, state, blocks, args.fast_eval_batch_size)
        ref_acc, ref_correct = fast_accuracy_np(x_test, y_test, reference_state, blocks, args.fast_eval_batch_size)
        return {
            "phase_fast_accuracy": phase_acc,
            "phase_fast_correct": phase_correct,
            "fast_reference_accuracy": ref_acc,
            "fast_reference_correct": ref_correct,
            **{f"fast_reference_{k}": v for k, v in metrics.items()},
        }

    def save_checkpoint_pair(chunk: int) -> dict[str, str]:
        fields: dict[str, str] = {}
        phase_path = checkpoint_dir / f"{stem}_chunk{chunk:04d}_weights.npz"
        save_partial_checkpoint(phase_path, state, len(blocks))
        fields["chunk_checkpoint"] = str(phase_path)
        if reference_state is not None:
            ref_path = checkpoint_dir / f"{stem}_chunk{chunk:04d}_fast_reference_weights.npz"
            save_partial_checkpoint(ref_path, reference_state, len(blocks))
            fields["fast_reference_checkpoint"] = str(ref_path)
        return fields

    def append_curve_row(
        chunk: int,
        samples_seen: int,
        phase_wall: float,
        eval_wall: float,
        heldout_accuracy: float,
        note: str,
        extra_fields: dict[str, str] | None = None,
    ) -> None:
        row: dict[str, float | int | str] = {
            "chunk": chunk,
            "samples_seen": samples_seen,
            "heldout_accuracy": heldout_accuracy,
            "phase_wall_time_s": phase_wall,
            "eval_wall_time_s": eval_wall,
            "note": note,
        }
        if extra_fields:
            row.update(extra_fields)
        row.update(tracking_fields())
        curve_rows.append(row)

    def append_eval_row(chunk: int, samples_seen: int, phase_wall: float, note: str, extra_fields: dict[str, str] | None = None) -> None:
        nonlocal eval_wall_total
        t_eval = time.perf_counter()
        acc = run_spice_eval(
            spice_bin,
            eval_netlist,
            eval_data,
            x_test,
            y_test,
            state,
            blocks,
            args.eval_batch_size,
            args.timeout,
        )
        eval_wall = time.perf_counter() - t_eval
        eval_wall_total += eval_wall
        append_curve_row(chunk, samples_seen, phase_wall, eval_wall, acc, note, extra_fields)

    if not args.skip_initial_eval and args.eval_every != 0:
        append_eval_row(args.start_chunk, start_sample, 0.0, "initial_spice_eval")

    for local_chunk in range(1, args.chunks + 1):
        global_chunk = args.start_chunk + local_chunk
        start = start_sample + (local_chunk - 1) * samples_per_chunk
        stop = start + samples_per_chunk
        chunk_netlist = generated / f"{stem}_chunk{global_chunk:04d}.cir"
        chunk_data = results / f"{stem}_chunk{global_chunk:04d}.dat"
        state, meta = run_phase_chunk(
            spice_bin,
            chunk_netlist,
            chunk_data,
            x_train[start:stop],
            y_train[start:stop],
            state,
            blocks,
            args.lr,
            args.batch_size,
            args.updates_per_chunk,
            args.phase,
            args.gap,
            args.edge,
            args.settle_ratio,
            args.transient_step,
            args.cw,
            args.cstate,
            args.cgrad,
            args.rleak,
            args.timeout,
            args.final_measures,
        )
        if reference_state is not None:
            reference_state = run_fast_reference_chunk(
                x_train[start:stop],
                y_train[start:stop],
                reference_state,
                blocks,
                args.lr,
                args.batch_size,
                args.updates_per_chunk,
            )
        phase_wall_total += float(meta["phase_wall_time_s"])
        checkpoint_fields = {}
        if args.checkpoint_every and (global_chunk % args.checkpoint_every == 0 or local_chunk == args.chunks):
            checkpoint_fields = save_checkpoint_pair(global_chunk)
        if args.eval_every and (global_chunk % args.eval_every == 0 or local_chunk == args.chunks):
            append_eval_row(
                global_chunk,
                stop,
                float(meta["phase_wall_time_s"]),
                "post_phase_chunk_eval",
                checkpoint_fields,
            )
        else:
            append_curve_row(
                global_chunk,
                stop,
                float(meta["phase_wall_time_s"]),
                0.0,
                np.nan,
                "phase_chunk_no_eval",
                checkpoint_fields,
            )

    final_weights_path = results / f"{stem}_final_weights.npz"
    save_partial_checkpoint(final_weights_path, state, len(blocks))
    curve = pd.DataFrame(curve_rows)
    curve_path = results / f"{stem}_learning_curve.csv"
    table_curve_path = tables / f"{stem}_learning_curve.csv"
    curve.to_csv(curve_path, index=False)
    curve.to_csv(table_curve_path, index=False)
    eval_rows = curve.dropna(subset=["heldout_accuracy"])
    best_row = None if eval_rows.empty else eval_rows.loc[eval_rows["heldout_accuracy"].idxmax()].to_dict()
    final_acc = None if eval_rows.empty else float(eval_rows.iloc[-1]["heldout_accuracy"])
    summary = {
        "simulator": version,
        "architecture": "phase_resolved_transient_partial_sharing_local_feature_readout_training_harness",
        "status": "phase_training_smoke" if phase_train_samples_this_run < 100 else "phase_training_run",
        "dataset": "MNIST train/test split, downsampled",
        "image_size": args.image_size,
        "block_size": args.block_size,
        "stride": stride,
        "blocks": len(blocks),
        "shared_channels": int(shared_channels),
        "private_channels": int(private_channels),
        "channels": int(channels),
        "classes": 10,
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "phase_train_samples": phase_train_samples_this_run,
        "phase_train_start_chunk": args.start_chunk,
        "phase_train_end_chunk": args.start_chunk + args.chunks,
        "phase_train_start_sample": start_sample,
        "phase_train_end_sample": end_sample,
        "batch_size": args.batch_size,
        "updates_per_chunk": args.updates_per_chunk,
        "chunks": args.chunks,
        "samples_per_chunk": samples_per_chunk,
        "lr": args.lr,
        "checkpoint_every": args.checkpoint_every,
        "checkpoint_dir": str(checkpoint_dir),
        "phase_s": args.phase,
        "settle_ratio": args.settle_ratio,
        "transient_step_s": args.transient_step,
        "init_weights": args.init_weights,
        "reference_init_weights": args.reference_init_weights,
        "final_weights": str(final_weights_path),
        "learning_curve": str(curve_path),
        "table_learning_curve": str(table_curve_path),
        "final_heldout_accuracy": final_acc,
        "best_eval": best_row,
        "phase_wall_time_s": phase_wall_total,
        "eval_wall_time_s": eval_wall_total,
        "phase_output_mode": "measure" if args.final_measures else "wrdata",
        "track_fast_reference": bool(args.track_fast_reference),
        "python_role": (
            "Python generates guiding waveforms, launches phase-transient chunks, and restarts the next chunk "
            "from the previous capacitor voltages. Within each chunk, shared/private weights, activations, "
            "deltas, and gradient accumulators are SPICE capacitor states."
        ),
        "note": (
            "Repeated partial-sharing phase-transient training harness. Shared kernel gradient capacitors "
            "sum across block positions before one shared apply pulse; chunk boundaries serialize through checkpoint ICs."
        ),
    }
    if args.track_fast_reference:
        final_tracking = tracking_fields()
        summary.update({f"final_{k}": v for k, v in final_tracking.items()})
    summary_path = results / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
