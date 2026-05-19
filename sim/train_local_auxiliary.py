from __future__ import annotations

from .train_backprop import train_classifier


def train_local_auxiliary(*args, **kwargs):
    return train_classifier(*args, **kwargs)

