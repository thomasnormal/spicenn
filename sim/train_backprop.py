from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TrainResult:
    train_loss: list[float]
    test_accuracy: list[float]
    best_accuracy: float = 0.0


def accuracy(model: nn.Module, loader, device: str = "cpu") -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += int((pred == y).sum())
            total += int(y.numel())
    return correct / max(total, 1)


def train_classifier(
    model: nn.Module,
    train_loader,
    test_loader,
    epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: str = "cpu",
) -> TrainResult:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    losses: list[float] = []
    accs: list[float] = []
    best = 0.0
    for _ in range(epochs):
        model.train()
        running = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += float(loss.detach()) * y.numel()
            n += int(y.numel())
        losses.append(running / max(n, 1))
        acc = accuracy(model, test_loader, device=device)
        accs.append(acc)
        best = max(best, acc)
    return TrainResult(losses, accs, best)
