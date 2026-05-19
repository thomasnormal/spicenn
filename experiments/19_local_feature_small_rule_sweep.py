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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spice"))

from run_spice_mnist_local_block_batch_op_train import block_indices  # noqa: E402
from run_spice_mnist_local_feature_torch_train import LocalFeatureReadout, eval_accuracy, make_block_tensor  # noqa: E402
from run_spice_mnist_train import load_mnist_sequence  # noqa: E402


@dataclass(frozen=True)
class Trial:
    channels: int
    loss: str
    optimizer: str
    lr: float


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


def parse_str_list(text: str) -> list[str]:
    vals = [part.strip() for part in text.split(",") if part.strip()]
    if not vals:
        raise ValueError("expected at least one value")
    return vals


def targets_pm1(y: torch.Tensor, n_classes: int = 10) -> torch.Tensor:
    target = -torch.ones((len(y), n_classes), dtype=torch.float32)
    target[torch.arange(len(y)), y] = 1.0
    return target


def trial_loss(logits: torch.Tensor, y: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "cross_entropy":
        return F.cross_entropy(logits, y)
    target = targets_pm1(y, logits.shape[1]).to(logits.device)
    if loss_name == "mse_linear":
        return F.mse_loss(logits, target)
    if loss_name == "mse_tanh":
        return F.mse_loss(torch.tanh(logits), target)
    raise ValueError(f"unknown loss {loss_name}")


def make_optimizer(model: torch.nn.Module, name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name}")


def is_phase_spice_portable(trial: Trial) -> bool:
    return trial.optimizer == "sgd" and trial.loss in {"mse_linear", "mse_tanh"}


def save_checkpoint(path: Path, model: LocalFeatureReadout) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        local_weights=model.local_weights.detach().cpu().numpy(),
        local_bias=model.local_bias.detach().cpu().numpy(),
        readout=model.readout.detach().cpu().numpy(),
        output_bias=model.output_bias.detach().cpu().numpy(),
    )


def run_trial(
    trial: Trial,
    blocks: list[list[int]],
    x_train_blocks: torch.Tensor,
    y_train: torch.Tensor,
    x_test_blocks: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    weight_decay: float,
    seed: int,
    checkpoint_dir: Path,
    tag: str,
) -> tuple[list[dict[str, float | int | str | bool]], Path]:
    model = LocalFeatureReadout(len(blocks), trial.channels, x_train_blocks.shape[2], seed)
    optimizer = make_optimizer(model, trial.optimizer, trial.lr, weight_decay)
    rows: list[dict[str, float | int | str | bool]] = []
    best_acc = -1.0
    best_path = checkpoint_dir / (
        f"{tag}_c{trial.channels}_{trial.loss}_{trial.optimizer}_lr{trial.lr:g}_best_weights.npz"
    )
    t0 = time.perf_counter()
    gen = torch.Generator()
    gen.manual_seed(seed + 1009 * trial.channels + int(1e6 * trial.lr))
    for epoch in range(epochs):
        perm = torch.randperm(len(y_train), generator=gen)
        total_loss = 0.0
        for start in range(0, len(y_train), batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            logits = model(x_train_blocks[idx])
            loss = trial_loss(logits, y_train[idx], trial.loss)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item()) * len(idx)
        acc, correct = eval_accuracy(model, x_test_blocks, y_test, eval_batch_size)
        if acc > best_acc:
            best_acc = acc
            save_checkpoint(best_path, model)
        rows.append(
            {
                "channels": trial.channels,
                "loss": trial.loss,
                "optimizer": trial.optimizer,
                "lr": trial.lr,
                "phase_spice_portable": is_phase_spice_portable(trial),
                "epoch": epoch + 1,
                "test_accuracy": acc,
                "test_correct": correct,
                "train_loss": total_loss / len(y_train),
                "trial_wall_time_s": time.perf_counter() - t0,
            }
        )
    return rows, best_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-samples", type=int, default=2000)
    ap.add_argument("--test-samples", type=int, default=1000)
    ap.add_argument("--image-size", type=int, default=14)
    ap.add_argument("--block-size", type=int, default=7)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--channels-list", default="4,8,16")
    ap.add_argument("--losses", default="cross_entropy,mse_linear,mse_tanh")
    ap.add_argument("--optimizers", default="sgd")
    ap.add_argument("--lrs", default="0.05,0.2")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--eval-batch-size", type=int, default=1024)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="local_feature_small_rule_sweep")
    args = ap.parse_args()

    torch.set_num_threads(max(args.torch_threads, 1))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    stride = args.block_size if args.stride is None else args.stride
    blocks = block_indices(args.image_size, args.block_size, stride)
    x_train, y_train_np, x_test, y_test_np = load_mnist_sequence(
        args.train_samples,
        args.test_samples,
        args.image_size,
        args.seed,
    )
    x_train_blocks = make_block_tensor(x_train, blocks)
    x_test_blocks = make_block_tensor(x_test, blocks)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    y_test = torch.tensor(y_test_np, dtype=torch.long)

    channels = parse_int_list(args.channels_list)
    losses = parse_str_list(args.losses)
    optimizers = parse_str_list(args.optimizers)
    lrs = parse_float_list(args.lrs)
    trials = [Trial(c, loss, opt, lr) for c in channels for loss in losses for opt in optimizers for lr in lrs]

    safe_tag = sanitize_tag(args.tag)
    results = ROOT / "spice/results"
    tables = ROOT / "results/tables"
    checkpoints = results / f"small_rule_sweep_{safe_tag}"
    for directory in [results, tables, checkpoints]:
        directory.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    rows: list[dict[str, float | int | str | bool]] = []
    best_paths: dict[str, str] = {}
    for i, trial in enumerate(trials, start=1):
        print(
            json.dumps(
                {
                    "trial": i,
                    "trials": len(trials),
                    "channels": trial.channels,
                    "loss": trial.loss,
                    "optimizer": trial.optimizer,
                    "lr": trial.lr,
                    "phase_spice_portable": is_phase_spice_portable(trial),
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
            args.weight_decay,
            args.seed,
            checkpoints,
            safe_tag,
        )
        rows.extend(trial_rows)
        best_paths[f"c{trial.channels}_{trial.loss}_{trial.optimizer}_lr{trial.lr:g}"] = str(best_path)
        print(json.dumps(trial_rows[-1]), flush=True)

    curve = pd.DataFrame(rows)
    curve_path = results / f"spice_mnist_{safe_tag}_curve.csv"
    table_curve_path = tables / f"spice_mnist_{safe_tag}_curve.csv"
    curve.to_csv(curve_path, index=False)
    curve.to_csv(table_curve_path, index=False)
    final_rows = curve.loc[curve.groupby(["channels", "loss", "optimizer", "lr"])["epoch"].idxmax()].copy()
    best_by_trial = curve.loc[curve.groupby(["channels", "loss", "optimizer", "lr"])["test_accuracy"].idxmax()].copy()
    best_by_trial = best_by_trial.sort_values(["test_accuracy", "channels"], ascending=[False, True])
    best_path = tables / f"spice_mnist_{safe_tag}_best_by_trial.csv"
    best_by_trial.to_csv(best_path, index=False)
    portable = best_by_trial[best_by_trial["phase_spice_portable"]]
    best_overall = best_by_trial.iloc[0].to_dict()
    best_portable = portable.iloc[0].to_dict() if len(portable) else None
    summary = {
        "architecture": "local_feature_readout_small_rule_sweep",
        "dataset": "MNIST train/test split, downsampled",
        "image_size": args.image_size,
        "block_size": args.block_size,
        "stride": stride,
        "blocks": len(blocks),
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "channels_list": channels,
        "losses": losses,
        "optimizers": optimizers,
        "lrs": lrs,
        "trials": len(trials),
        "best_overall": best_overall,
        "best_phase_spice_portable": best_portable,
        "curve": str(curve_path),
        "table_curve": str(table_curve_path),
        "best_by_trial_csv": str(best_path),
        "best_weight_paths": best_paths,
        "wall_time_s": time.perf_counter() - t0,
        "note": (
            "Fast small-network sweep for the same local-feature equations used by SPICE. "
            "Rows marked phase_spice_portable use plain SGD with MSE-style deltas and are the easiest candidates "
            "to port to the phase-transient capacitor update deck."
        ),
    }
    summary_path = results / f"spice_mnist_{safe_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
