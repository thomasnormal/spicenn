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
class Trial:
    channels: int
    lr_spice: float
    update_noise_std: float


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


def pm1_targets(labels: torch.Tensor, n_classes: int = 10) -> torch.Tensor:
    target = -torch.ones((len(labels), n_classes), dtype=torch.float32)
    target[torch.arange(len(labels)), labels] = 1.0
    return target


def init_state(blocks: int, channels: int, block_len: int, seed: int) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator()
    gen.manual_seed(seed)
    w = torch.randn(blocks, channels, block_len, generator=gen) * 0.05
    hb = torch.zeros(blocks, channels)
    readout = torch.randn(10, blocks, channels, generator=gen) * 0.05
    ob = torch.zeros(10)
    return w, hb, readout, ob


def forward_scores(
    x_blocks: torch.Tensor,
    w: torch.Tensor,
    hb: torch.Tensor,
    readout: torch.Tensor,
    ob: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h = torch.tanh(torch.einsum("nbp,bcp->nbc", x_blocks, w) + hb)
    score = torch.einsum("nbc,kbc->nk", h, readout) + ob
    y = torch.tanh(score)
    return h, score, y


def accuracy(
    x_blocks: torch.Tensor,
    labels: torch.Tensor,
    w: torch.Tensor,
    hb: torch.Tensor,
    readout: torch.Tensor,
    ob: torch.Tensor,
    batch_size: int,
) -> tuple[float, int]:
    preds = []
    for start in range(0, len(labels), batch_size):
        _h, _score, y = forward_scores(x_blocks[start : start + batch_size], w, hb, readout, ob)
        preds.append(torch.argmax(y, dim=1))
    pred = torch.cat(preds)
    correct = int(torch.sum(pred == labels).item())
    return correct / max(len(labels), 1), correct


def hardware_update(
    x_blocks: torch.Tensor,
    labels: torch.Tensor,
    w: torch.Tensor,
    hb: torch.Tensor,
    readout: torch.Tensor,
    ob: torch.Tensor,
    lr_spice: float,
    update_noise_std: float,
    gen: torch.Generator,
) -> tuple[torch.Tensor, ...]:
    batch = len(labels)
    h, _score, y = forward_scores(x_blocks, w, hb, readout, ob)
    d = (pm1_targets(labels, y.shape[1]) - y) * (1.0 - y * y)
    dh = torch.einsum("nk,kbc->nbc", d, readout) * (1.0 - h * h)

    w = w + lr_spice * torch.einsum("nbc,nbp->bcp", dh, x_blocks) / batch
    hb = hb + lr_spice * dh.mean(dim=0)
    readout = readout + lr_spice * torch.einsum("nk,nbc->kbc", d, h) / batch
    ob = ob + lr_spice * d.mean(dim=0)

    if update_noise_std > 0.0:
        w = w + torch.randn(w.shape, generator=gen) * update_noise_std
        hb = hb + torch.randn(hb.shape, generator=gen) * update_noise_std
        readout = readout + torch.randn(readout.shape, generator=gen) * update_noise_std
        ob = ob + torch.randn(ob.shape, generator=gen) * update_noise_std
    return w, hb, readout, ob


def save_checkpoint(path: Path, state: tuple[torch.Tensor, ...]) -> None:
    w, hb, readout, ob = state
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        local_weights=w.numpy(),
        local_bias=hb.numpy(),
        readout=readout.numpy(),
        output_bias=ob.numpy(),
    )


def run_trial(
    trial: Trial,
    blocks: int,
    block_len: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    seed: int,
    checkpoint_dir: Path,
    tag: str,
) -> tuple[list[dict[str, float | int]], Path]:
    state = init_state(blocks, trial.channels, block_len, seed)
    gen = torch.Generator()
    gen.manual_seed(seed + 7919 * trial.channels + int(1e6 * trial.lr_spice) + int(1e9 * trial.update_noise_std))
    rows: list[dict[str, float | int]] = []
    best_acc = -1.0
    best_path = checkpoint_dir / (
        f"{tag}_c{trial.channels}_lrspice{trial.lr_spice:g}_noise{trial.update_noise_std:g}_best_weights.npz"
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
            save_checkpoint(best_path, state)
        rows.append(
            {
                "channels": trial.channels,
                "lr_spice": trial.lr_spice,
                "equivalent_lr_torch_mse_tanh": 5.0 * trial.lr_spice,
                "update_noise_std": trial.update_noise_std,
                "epoch": epoch + 1,
                "test_accuracy": acc,
                "test_correct": correct,
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
    ap.add_argument("--channels-list", default="8,16")
    ap.add_argument("--lr-spice-list", default="0.3,0.8")
    ap.add_argument("--update-noise-std-list", default="0,0.0001,0.0003,0.001,0.003")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--eval-batch-size", type=int, default=1024)
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="local_feature_update_noise")
    args = ap.parse_args()

    torch.set_num_threads(max(args.torch_threads, 1))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    stride = args.block_size if args.stride is None else args.stride
    blocks = block_indices(args.image_size, args.block_size, stride)
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

    trials = [
        Trial(channels, lr_spice, noise_std)
        for channels in parse_int_list(args.channels_list)
        for lr_spice in parse_float_list(args.lr_spice_list)
        for noise_std in parse_float_list(args.update_noise_std_list)
    ]
    safe_tag = sanitize_tag(args.tag)
    results = ROOT / "spice/results"
    tables = ROOT / "results/tables"
    checkpoints = results / f"update_noise_sweep_{safe_tag}"
    for directory in [results, tables, checkpoints]:
        directory.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    rows: list[dict[str, float | int]] = []
    best_paths: dict[str, str] = {}
    for i, trial in enumerate(trials, start=1):
        print(
            json.dumps(
                {
                    "trial": i,
                    "trials": len(trials),
                    "channels": trial.channels,
                    "lr_spice": trial.lr_spice,
                    "update_noise_std": trial.update_noise_std,
                }
            ),
            flush=True,
        )
        trial_rows, best_path = run_trial(
            trial,
            len(blocks),
            args.block_size * args.block_size,
            x_train,
            y_train,
            x_test,
            y_test,
            args.epochs,
            args.batch_size,
            args.eval_batch_size,
            args.seed,
            checkpoints,
            safe_tag,
        )
        rows.extend(trial_rows)
        key = f"c{trial.channels}_lrspice{trial.lr_spice:g}_noise{trial.update_noise_std:g}"
        best_paths[key] = str(best_path)
        print(json.dumps(trial_rows[-1]), flush=True)

    curve = pd.DataFrame(rows)
    curve_path = results / f"spice_mnist_{safe_tag}_curve.csv"
    table_curve_path = tables / f"spice_mnist_{safe_tag}_curve.csv"
    curve.to_csv(curve_path, index=False)
    curve.to_csv(table_curve_path, index=False)
    best_by_trial = curve.loc[curve.groupby(["channels", "lr_spice", "update_noise_std"])["test_accuracy"].idxmax()]
    best_by_trial = best_by_trial.sort_values(["test_accuracy", "channels"], ascending=[False, True])
    best_path = tables / f"spice_mnist_{safe_tag}_best_by_trial.csv"
    best_by_trial.to_csv(best_path, index=False)

    summary = {
        "architecture": "local_feature_manual_phase_update_noise_surrogate",
        "dataset": "MNIST train/test split, downsampled",
        "image_size": args.image_size,
        "block_size": args.block_size,
        "stride": stride,
        "blocks": len(blocks),
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "channels_list": parse_int_list(args.channels_list),
        "lr_spice_list": parse_float_list(args.lr_spice_list),
        "update_noise_std_list": parse_float_list(args.update_noise_std_list),
        "trials": len(trials),
        "best_overall": best_by_trial.iloc[0].to_dict(),
        "curve": str(curve_path),
        "table_curve": str(table_curve_path),
        "best_by_trial_csv": str(best_path),
        "best_weight_paths": best_paths,
        "wall_time_s": time.perf_counter() - t0,
        "note": (
            "Fast manual implementation of the same tanh-output delta rule used by the SPICE local-feature deck. "
            "update_noise_std injects absolute parameter noise after each update to approximate accumulated phase/integration error."
        ),
    }
    summary_path = results / f"spice_mnist_{safe_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
