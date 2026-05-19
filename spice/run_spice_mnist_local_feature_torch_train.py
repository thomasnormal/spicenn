from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from run_spice_mnist_local_block_batch_op_train import block_indices
from run_spice_mnist_train import load_mnist_sequence
from run_spice_sweep import ROOT


def sanitize_tag(tag: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in tag)


def make_block_tensor(x: np.ndarray, blocks: list[list[int]]) -> torch.Tensor:
    return torch.tensor(np.stack([x[:, idxs] for idxs in blocks], axis=1), dtype=torch.float32)


class LocalFeatureReadout(torch.nn.Module):
    def __init__(self, blocks: int, channels: int, block_len: int, seed: int):
        super().__init__()
        gen = torch.Generator()
        gen.manual_seed(seed)
        self.local_weights = torch.nn.Parameter(torch.randn(blocks, channels, block_len, generator=gen) * 0.05)
        self.local_bias = torch.nn.Parameter(torch.zeros(blocks, channels))
        self.readout = torch.nn.Parameter(torch.randn(10, blocks, channels, generator=gen) * 0.05)
        self.output_bias = torch.nn.Parameter(torch.zeros(10))

    def forward(self, x_blocks: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(torch.einsum("nbp,bcp->nbc", x_blocks, self.local_weights) + self.local_bias)
        return torch.einsum("nbc,kbc->nk", hidden, self.readout) + self.output_bias


def eval_accuracy(model: LocalFeatureReadout, x_blocks: torch.Tensor, y: torch.Tensor, batch_size: int) -> tuple[float, int]:
    preds = []
    with torch.no_grad():
        for start in range(0, len(y), batch_size):
            logits = model(x_blocks[start : start + batch_size])
            preds.append(torch.argmax(logits, dim=1))
    pred = torch.cat(preds)
    correct = int(torch.sum(pred == y).item())
    return correct / max(len(y), 1), correct


def plot_curve(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(df["epoch"], df["test_accuracy"], marker="o", label="test")
    plt.plot(df["epoch"], df["train_probe_accuracy"], marker=".", label="train probe")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def save_checkpoint(path: Path, model: LocalFeatureReadout) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        local_weights=model.local_weights.detach().cpu().numpy(),
        local_bias=model.local_bias.detach().cpu().numpy(),
        readout=model.readout.detach().cpu().numpy(),
        output_bias=model.output_bias.detach().cpu().numpy(),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-samples", type=int, default=60000)
    ap.add_argument("--test-samples", type=int, default=10000)
    ap.add_argument("--image-size", type=int, default=14)
    ap.add_argument("--block-size", type=int, default=7)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--eval-batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--train-probe-samples", type=int, default=10000)
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="local_feature_torch")
    args = ap.parse_args()

    if args.train_samples <= 0 or args.test_samples <= 0:
        raise ValueError("sample counts must be positive")
    if args.channels <= 0 or args.epochs <= 0:
        raise ValueError("--channels and --epochs must be positive")
    torch.set_num_threads(max(args.torch_threads, 1))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    stride = args.block_size if args.stride is None else args.stride
    blocks = block_indices(args.image_size, args.block_size, stride)
    x_train, y_train, x_test, y_test = load_mnist_sequence(
        args.train_samples, args.test_samples, args.image_size, args.seed
    )
    x_train_blocks = make_block_tensor(x_train, blocks)
    x_test_blocks = make_block_tensor(x_test, blocks)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    model = LocalFeatureReadout(len(blocks), args.channels, args.block_size * args.block_size, args.seed)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stem = f"spice_mnist_local_feature_torch_{sanitize_tag(args.tag)}"
    results = ROOT / "spice/results"
    figures = ROOT / "results/figures"
    tables = ROOT / "results/tables"
    for directory in [results, figures, tables]:
        directory.mkdir(parents=True, exist_ok=True)

    rows = []
    best_acc = -1.0
    best_path = results / f"{stem}_best_weights.npz"
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(y_train_t))
        total_loss = 0.0
        for start in range(0, len(y_train_t), args.batch_size):
            idx = perm[start : start + args.batch_size]
            opt.zero_grad()
            logits = model(x_train_blocks[idx])
            loss = F.cross_entropy(logits, y_train_t[idx])
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().item()) * len(idx)

        model.eval()
        test_acc, test_correct = eval_accuracy(model, x_test_blocks, y_test_t, args.eval_batch_size)
        probe_n = min(args.train_probe_samples, len(y_train_t))
        train_probe_acc, train_probe_correct = eval_accuracy(
            model,
            x_train_blocks[:probe_n],
            y_train_t[:probe_n],
            args.eval_batch_size,
        )
        row = {
            "epoch": epoch + 1,
            "loss": total_loss / len(y_train_t),
            "test_accuracy": test_acc,
            "test_correct": test_correct,
            "train_probe_accuracy": train_probe_acc,
            "train_probe_correct": train_probe_correct,
            "wall_time_s": time.perf_counter() - t0,
        }
        rows.append(row)
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(best_path, model)
        print(json.dumps(row), flush=True)

    final_path = results / f"{stem}_final_weights.npz"
    save_checkpoint(final_path, model)
    curve = pd.DataFrame(rows)
    curve_path = results / f"{stem}_learning_curve.csv"
    table_curve_path = tables / f"{stem}_learning_curve.csv"
    curve.to_csv(curve_path, index=False)
    curve.to_csv(table_curve_path, index=False)
    fig_path = figures / f"{stem}_learning_curve.png"
    plot_curve(curve, fig_path)
    best_row = curve.loc[curve["test_accuracy"].idxmax()].to_dict()
    summary = {
        "architecture": "local_feature_readout_torch_export_for_spice",
        "dataset": "MNIST train/test split, downsampled",
        "image_size": args.image_size,
        "block_size": args.block_size,
        "stride": stride,
        "blocks": len(blocks),
        "channels": args.channels,
        "local": True,
        "activation": "tanh",
        "output_mode": "linear_class_logits",
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "best_test_accuracy": float(best_row["test_accuracy"]),
        "best_test_correct": int(best_row["test_correct"]),
        "best_epoch": int(best_row["epoch"]),
        "final_test_accuracy": float(curve.iloc[-1]["test_accuracy"]),
        "learning_curve": str(curve_path),
        "table_learning_curve": str(table_curve_path),
        "figure": str(fig_path),
        "best_weights": str(best_path),
        "final_weights": str(final_path),
        "wall_time_s": time.perf_counter() - t0,
        "note": (
            "High-level PyTorch training of the same local-feature/readout topology used by the SPICE local-feature deck. "
            "The exported checkpoint is SPICE-evaluable but this run is not itself an all-SPICE training result."
        ),
    }
    summary_path = results / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
