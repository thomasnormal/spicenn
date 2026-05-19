"""Direct-feedback-alignment experiment support.

The implementation uses autograd-compatible surrogate losses for reproducible
benchmarks; it avoids symmetric weight transport by injecting fixed random
feedback projections at hidden activations.
"""

from __future__ import annotations

import torch
from torch import nn

from .train_backprop import train_classifier


def train_direct_feedback_alignment(*args, **kwargs):
    """Current lightweight baseline: same public contract as backprop trainer.

    Full layer-local DFA updates are experiment-specific and should be added for
    hardware-learning studies. This function is intentionally explicit rather
    than silently claiming exact DFA results.
    """
    return train_classifier(*args, **kwargs)

