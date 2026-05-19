from __future__ import annotations

import torch


class EligibilityTrace:
    def __init__(self, shape, decay: float = 0.95, device: str = "cpu"):
        self.decay = decay
        self.trace = torch.zeros(shape, device=device)

    def update(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        self.trace.mul_(self.decay).add_(torch.einsum("bi,bj->ij", pre, post) / max(pre.shape[0], 1))
        return self.trace

    def delta(self, modulatory: torch.Tensor, eta: float = 1e-3) -> torch.Tensor:
        return eta * self.trace * modulatory.mean()

