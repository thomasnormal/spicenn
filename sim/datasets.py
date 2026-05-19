from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset, TensorDataset


def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def digits_loaders(batch_size: int = 128, seed: int = 0):
    data = load_digits()
    x = data.data.astype("float32") / 16.0
    y = data.target.astype("int64")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=seed, stratify=y
    )
    train = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    test = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=batch_size, shuffle=False),
    )


def mnist_loaders(
    batch_size: int = 128,
    root: str | Path = "data",
    download: bool = False,
    flatten: bool = True,
    normalize: bool = True,
    train_limit: int | None = None,
    test_limit: int | None = None,
    seed: int = 0,
):
    from torchvision import datasets, transforms

    ops = [transforms.ToTensor()]
    if normalize:
        ops.append(transforms.Normalize((0.1307,), (0.3081,)))
    if flatten:
        ops.append(transforms.Lambda(lambda t: t.reshape(-1)))
    transform = transforms.Compose(ops)
    train = datasets.MNIST(root=str(root), train=True, transform=transform, download=download)
    test = datasets.MNIST(root=str(root), train=False, transform=transform, download=download)
    if train_limit is not None:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(train), generator=g)[:train_limit].tolist()
        train = Subset(train, idx)
    if test_limit is not None:
        g = torch.Generator().manual_seed(seed + 1)
        idx = torch.randperm(len(test), generator=g)[:test_limit].tolist()
        test = Subset(test, idx)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=batch_size, shuffle=False),
    )
