from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from .protocol import ProtocolFamily


class PromotionLevel(IntEnum):
    CONTRACT_VALID = 0
    SINGLE_CELL_CHARACTERIZED = 1
    LEARNING_ALIGNED = 2
    TINY_INTEGRATION = 3
    MNIST_SMOKE = 4
    FULL_EXPERIMENT = 5


@dataclass(frozen=True)
class CharacterizationExpectation:
    name: str
    metric: str
    minimum: float | None = None
    maximum: float | None = None
    required: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("characterization expectation name must be nonempty")
        if not self.metric:
            raise ValueError("characterization expectation metric must be nonempty")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError(f"invalid expectation bounds for {self.name!r}")


@dataclass
class CharacterizationResult:
    passed: bool
    protocol_family: ProtocolFamily
    promotion_level: PromotionLevel = PromotionLevel.CONTRACT_VALID
    forward_error: float | None = None
    forward_monotonicity_violations: int | None = None
    settling_time: float | None = None
    backward_cosine: float | None = None
    update_cosine: float | None = None
    update_sign_alignment: float | None = None
    hold_drift: float | None = None
    read_disturb: float | None = None
    phase_overlap_sensitivity: float | None = None
    noise_robust_alignment: float | None = None
    notes: list[str] = field(default_factory=list)
