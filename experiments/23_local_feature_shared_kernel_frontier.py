from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spice"))

from run_spice_mnist_local_block_batch_op_train import block_indices  # noqa: E402
from run_spice_mnist_local_feature_torch_train import make_block_tensor  # noqa: E402
from run_spice_mnist_train import load_mnist_sequence  # noqa: E402


@dataclass(frozen=True)
class TopologySpec:
    image_size: int
    block_size: int
    channels: int
    stride: int


@dataclass(frozen=True)
class Trial:
    topology: TopologySpec
    train_samples: int
    lr_spice: float
    update_noise_std: float
    seed: int


def sanitize_tag(tag: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in tag)


def parse_int_list(text: str) -> list[int]:
    vals = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not vals:
        raise ValueError("expected at least one integer")
    return vals


def parse_float_list(text: str) -> list[float]:
    vals = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not vals:
        raise ValueError("expected at least one float")
    return vals


def parse_topology_specs(text: str) -> list[TopologySpec]:
    specs: list[TopologySpec] = []
    for raw in [part.strip() for part in text.split(",") if part.strip()]:
        parts = raw.split(":")
        if len(parts) not in {3, 4}:
            raise ValueError("topology specs must be image:block:channels or image:block:channels:stride")
        image_size, block_size, channels = (int(parts[0]), int(parts[1]), int(parts[2]))
        stride = int(parts[3]) if len(parts) == 4 else block_size
        if min(image_size, block_size, channels, stride) <= 0:
            raise ValueError(f"topology values must be positive: {raw}")
        if block_size > image_size:
            raise ValueError(f"block size must fit inside image: {raw}")
        specs.append(TopologySpec(image_size, block_size, channels, stride))
    if not specs:
        raise ValueError("expected at least one topology spec")
    return specs


def pm1_targets(labels: torch.Tensor, n_classes: int = 10) -> torch.Tensor:
    target = -torch.ones((len(labels), n_classes), dtype=torch.float32)
    target[torch.arange(len(labels)), labels] = 1.0
    return target


def init_state(blocks: int, channels: int, block_len: int, seed: int) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator()
    gen.manual_seed(seed)
    shared_w = torch.randn(channels, block_len, generator=gen) * 0.05
    shared_bias = torch.zeros(channels)
    readout = torch.randn(10, blocks, channels, generator=gen) * 0.05
    output_bias = torch.zeros(10)
    return shared_w, shared_bias, readout, output_bias


def forward_scores(
    x_blocks: torch.Tensor,
    shared_w: torch.Tensor,
    shared_bias: torch.Tensor,
    readout: torch.Tensor,
    output_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h = torch.tanh(torch.einsum("nbp,cp->nbc", x_blocks, shared_w) + shared_bias[None, None, :])
    score = torch.einsum("nbc,kbc->nk", h, readout) + output_bias
    y = torch.tanh(score)
    return h, score, y


def accuracy(
    x_blocks: torch.Tensor,
    labels: torch.Tensor,
    shared_w: torch.Tensor,
    shared_bias: torch.Tensor,
    readout: torch.Tensor,
    output_bias: torch.Tensor,
    batch_size: int,
) -> tuple[float, int]:
    preds = []
    for start in range(0, len(labels), batch_size):
        _h, _score, y = forward_scores(
            x_blocks[start : start + batch_size],
            shared_w,
            shared_bias,
            readout,
            output_bias,
        )
        preds.append(torch.argmax(y, dim=1))
    pred = torch.cat(preds)
    correct = int(torch.sum(pred == labels).item())
    return correct / max(len(labels), 1), correct


def hardware_update(
    x_blocks: torch.Tensor,
    labels: torch.Tensor,
    shared_w: torch.Tensor,
    shared_bias: torch.Tensor,
    readout: torch.Tensor,
    output_bias: torch.Tensor,
    lr_spice: float,
    update_noise_std: float,
    gen: torch.Generator,
) -> tuple[torch.Tensor, ...]:
    batch = len(labels)
    h, _score, y = forward_scores(x_blocks, shared_w, shared_bias, readout, output_bias)
    d = (pm1_targets(labels, y.shape[1]) - y) * (1.0 - y * y)
    dh = torch.einsum("nk,kbc->nbc", d, readout) * (1.0 - h * h)

    shared_w = shared_w + lr_spice * torch.einsum("nbc,nbp->cp", dh, x_blocks) / batch
    shared_bias = shared_bias + lr_spice * dh.sum(dim=(0, 1)) / batch
    readout = readout + lr_spice * torch.einsum("nk,nbc->kbc", d, h) / batch
    output_bias = output_bias + lr_spice * d.mean(dim=0)

    if update_noise_std > 0.0:
        shared_w = shared_w + torch.randn(shared_w.shape, generator=gen) * update_noise_std
        shared_bias = shared_bias + torch.randn(shared_bias.shape, generator=gen) * update_noise_std
        readout = readout + torch.randn(readout.shape, generator=gen) * update_noise_std
        output_bias = output_bias + torch.randn(output_bias.shape, generator=gen) * update_noise_std
    return shared_w, shared_bias, readout, output_bias


def state_counts(blocks: int, channels: int, block_len: int, n_classes: int = 10) -> dict[str, int]:
    shared_weight_values = channels * block_len
    shared_bias_values = channels
    readout_values = n_classes * blocks * channels
    output_bias_values = n_classes
    train_state_values = shared_weight_values + shared_bias_values + readout_values + output_bias_values
    gradient_values = train_state_values
    temporary_values = 2 * blocks * channels + 2 * n_classes
    phase_state_values = train_state_values + gradient_values + temporary_values
    expanded_weight_equivalent_values = blocks * channels * block_len
    return {
        "shared_weight_values": shared_weight_values,
        "shared_bias_values": shared_bias_values,
        "readout_values": readout_values,
        "output_bias_values": output_bias_values,
        "train_state_values": train_state_values,
        "gradient_state_values": gradient_values,
        "temporary_state_values": temporary_values,
        "phase_state_values": phase_state_values,
        "expanded_weight_equivalent_values": expanded_weight_equivalent_values,
        "multiply_terms_per_sample": expanded_weight_equivalent_values + readout_values,
    }


def save_checkpoint(path: Path, state: tuple[torch.Tensor, ...], blocks: int) -> None:
    shared_w, shared_bias, readout, output_bias = state
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        shared_local_weights=shared_w.numpy(),
        shared_local_bias=shared_bias.numpy(),
        local_weights=shared_w.unsqueeze(0).repeat(blocks, 1, 1).numpy(),
        local_bias=shared_bias.unsqueeze(0).repeat(blocks, 1).numpy(),
        readout=readout.numpy(),
        output_bias=output_bias.numpy(),
        weight_sharing="shared_kernel",
    )


def pareto_front(df: pd.DataFrame, cost_col: str) -> pd.DataFrame:
    ordered = df.sort_values([cost_col, "test_accuracy"], ascending=[True, False])
    rows = []
    best_acc = -1.0
    for _, row in ordered.iterrows():
        acc = float(row["test_accuracy"])
        if acc > best_acc + 1e-12:
            rows.append(row)
            best_acc = acc
    return pd.DataFrame(rows)


def run_trial(
    trial: Trial,
    blocks: list[list[int]],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    checkpoint_dir: Path,
    tag: str,
) -> tuple[list[dict[str, float | int | str]], Path]:
    block_len = trial.topology.block_size * trial.topology.block_size
    counts = state_counts(len(blocks), trial.topology.channels, block_len)
    init_seed = (
        trial.seed
        + 1009 * trial.topology.image_size
        + 9173 * trial.topology.block_size
        + 7919 * trial.topology.channels
        + 271 * trial.topology.stride
        + 15485863
    )
    state = init_state(len(blocks), trial.topology.channels, block_len, init_seed)
    gen = torch.Generator()
    gen.manual_seed(init_seed + int(1e6 * trial.lr_spice) + int(1e9 * trial.update_noise_std))
    rows: list[dict[str, float | int | str]] = []
    best_acc = -1.0
    best_path = checkpoint_dir / (
        f"{tag}_i{trial.topology.image_size}_b{trial.topology.block_size}"
        f"_s{trial.topology.stride}_c{trial.topology.channels}_n{trial.train_samples}"
        f"_lrspice{trial.lr_spice:g}_noise{trial.update_noise_std:g}_seed{trial.seed}_best_weights.npz"
    )
    t0 = time.perf_counter()
    for epoch in range(epochs):
        perm = torch.randperm(len(y_train), generator=gen)
        for start in range(0, len(y_train), batch_size):
            idx = perm[start : start + batch_size]
            state = hardware_update(
                x_train[idx],
                y_train[idx],
                *state,
                trial.lr_spice,
                trial.update_noise_std,
                gen,
            )
        acc, correct = accuracy(x_test, y_test, *state, eval_batch_size)
        if acc > best_acc:
            best_acc = acc
            save_checkpoint(best_path, state, len(blocks))
        rows.append(
            {
                "image_size": trial.topology.image_size,
                "block_size": trial.topology.block_size,
                "stride": trial.topology.stride,
                "blocks": len(blocks),
                "channels": trial.topology.channels,
                "local_features": len(blocks) * trial.topology.channels,
                "train_samples": trial.train_samples,
                "test_samples": len(y_test),
                "lr_spice": trial.lr_spice,
                "equivalent_lr_torch_mse_tanh": 5.0 * trial.lr_spice,
                "update_noise_std": trial.update_noise_std,
                "seed": trial.seed,
                "epoch": epoch + 1,
                "test_accuracy": acc,
                "test_correct": correct,
                "trial_wall_time_s": time.perf_counter() - t0,
                "weight_sharing": "shared_kernel",
                **counts,
            }
        )
    return rows, best_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topologies", default="7:5:1:1,10:4:1:2,12:4:1:2,14:7:1:2")
    ap.add_argument("--train-samples-list", default="2000")
    ap.add_argument("--test-samples", type=int, default=2000)
    ap.add_argument("--lr-spice-list", default="0.005,0.01,0.02,0.03,0.05")
    ap.add_argument("--update-noise-std-list", default="0")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--eval-batch-size", type=int, default=1024)
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--target-accuracy", type=float, default=0.90)
    ap.add_argument("--tag", default="local_feature_shared_kernel_frontier")
    args = ap.parse_args()

    torch.set_num_threads(max(args.torch_threads, 1))
    np.random.seed(0)

    topologies = parse_topology_specs(args.topologies)
    train_samples_list = parse_int_list(args.train_samples_list)
    lr_spice_list = parse_float_list(args.lr_spice_list)
    update_noise_std_list = parse_float_list(args.update_noise_std_list)
    seeds = parse_int_list(args.seeds)
    if args.test_samples <= 0 or args.epochs <= 0:
        raise ValueError("--test-samples and --epochs must be positive")
    if min(train_samples_list) <= 0:
        raise ValueError("--train-samples-list values must be positive")

    safe_tag = sanitize_tag(args.tag)
    results = ROOT / "spice/results"
    tables = ROOT / "results/tables"
    checkpoints = results / f"shared_kernel_frontier_{safe_tag}"
    for directory in [results, tables, checkpoints]:
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    best_paths: dict[str, str] = {}
    dataset_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    t0 = time.perf_counter()
    trials_done = 0
    trial_specs = [
        Trial(topology, train_samples, lr_spice, noise_std, seed)
        for topology in topologies
        for train_samples in train_samples_list
        for lr_spice in lr_spice_list
        for noise_std in update_noise_std_list
        for seed in seeds
    ]
    for trial in trial_specs:
        data_key = (trial.topology.image_size, trial.seed)
        if data_key not in dataset_cache:
            dataset_cache[data_key] = load_mnist_sequence(
                max(train_samples_list),
                args.test_samples,
                trial.topology.image_size,
                trial.seed,
            )
        x_train_np, y_train_np, x_test_np, y_test_np = dataset_cache[data_key]
        blocks = block_indices(trial.topology.image_size, trial.topology.block_size, trial.topology.stride)
        x_train_blocks = make_block_tensor(x_train_np[: trial.train_samples], blocks)
        x_test_blocks = make_block_tensor(x_test_np, blocks)
        y_train = torch.tensor(y_train_np[: trial.train_samples], dtype=torch.long)
        y_test = torch.tensor(y_test_np, dtype=torch.long)

        trials_done += 1
        print(
            json.dumps(
                {
                    "trial": trials_done,
                    "trials": len(trial_specs),
                    "image_size": trial.topology.image_size,
                    "block_size": trial.topology.block_size,
                    "stride": trial.topology.stride,
                    "channels": trial.topology.channels,
                    "train_samples": trial.train_samples,
                    "lr_spice": trial.lr_spice,
                    "update_noise_std": trial.update_noise_std,
                    "seed": trial.seed,
                    "weight_sharing": "shared_kernel",
                }
            ),
            flush=True,
        )
        trial_rows, best_path = run_trial(
            trial,
            blocks,
            x_train_blocks,
            y_train,
            x_test_blocks,
            y_test,
            args.epochs,
            args.batch_size,
            args.eval_batch_size,
            checkpoints,
            safe_tag,
        )
        rows.extend(trial_rows)
        key = (
            f"shared_i{trial.topology.image_size}_b{trial.topology.block_size}_s{trial.topology.stride}"
            f"_c{trial.topology.channels}_n{trial.train_samples}_lrspice{trial.lr_spice:g}"
            f"_noise{trial.update_noise_std:g}_seed{trial.seed}"
        )
        best_paths[key] = str(best_path)
        print(json.dumps(trial_rows[-1]), flush=True)

    curve = pd.DataFrame(rows)
    curve_path = results / f"spice_mnist_{safe_tag}_curve.csv"
    table_curve_path = tables / f"spice_mnist_{safe_tag}_curve.csv"
    curve.to_csv(curve_path, index=False)
    curve.to_csv(table_curve_path, index=False)

    group_cols = [
        "image_size",
        "block_size",
        "stride",
        "channels",
        "train_samples",
        "lr_spice",
        "update_noise_std",
        "seed",
    ]
    best_by_trial = curve.loc[curve.groupby(group_cols)["test_accuracy"].idxmax()].copy()
    best_by_trial = best_by_trial.sort_values(
        ["test_accuracy", "phase_state_values", "trial_wall_time_s"],
        ascending=[False, True, True],
    )
    best_path = tables / f"spice_mnist_{safe_tag}_best_by_trial.csv"
    best_by_trial.to_csv(best_path, index=False)

    pareto_state = pareto_front(best_by_trial, "phase_state_values")
    pareto_wall = pareto_front(best_by_trial, "trial_wall_time_s")
    pareto_state_path = tables / f"spice_mnist_{safe_tag}_pareto_by_state.csv"
    pareto_wall_path = tables / f"spice_mnist_{safe_tag}_pareto_by_wall_time.csv"
    pareto_state.to_csv(pareto_state_path, index=False)
    pareto_wall.to_csv(pareto_wall_path, index=False)

    target_hits = best_by_trial[best_by_trial["test_accuracy"] >= args.target_accuracy]
    smallest_target_hit = None
    if len(target_hits):
        smallest_target_hit = target_hits.sort_values(
            ["phase_state_values", "trial_wall_time_s", "test_accuracy"],
            ascending=[True, True, False],
        ).iloc[0].to_dict()

    summary = {
        "architecture": "local_feature_manual_phase_update_shared_kernel_frontier",
        "dataset": "MNIST train/test split, downsampled",
        "weight_sharing": "shared_kernel",
        "topologies": [spec.__dict__ for spec in topologies],
        "train_samples_list": train_samples_list,
        "test_samples": args.test_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr_spice_list": lr_spice_list,
        "update_noise_std_list": update_noise_std_list,
        "seeds": seeds,
        "target_accuracy": args.target_accuracy,
        "trials": len(trial_specs),
        "best_overall": best_by_trial.iloc[0].to_dict(),
        "smallest_target_hit": smallest_target_hit,
        "curve": str(curve_path),
        "table_curve": str(table_curve_path),
        "best_by_trial_csv": str(best_path),
        "pareto_by_state_csv": str(pareto_state_path),
        "pareto_by_wall_time_csv": str(pareto_wall_path),
        "best_weight_paths": best_paths,
        "wall_time_s": time.perf_counter() - t0,
        "note": (
            "Fast small-network search using shared local feature kernels. "
            "A shared kernel capacitor is reused across block positions while readout capacitors remain position-specific. "
            "This is a fast surrogate for evaluating whether weight sharing can reduce capacitor state before building a SPICE deck."
        ),
    }
    summary_path = results / f"spice_mnist_{safe_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
