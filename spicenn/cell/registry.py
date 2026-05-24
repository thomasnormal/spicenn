from __future__ import annotations

from dataclasses import dataclass, field

from .base import TrainableDynamicalCell


@dataclass
class CellRegistry:
    _cells: dict[str, TrainableDynamicalCell] = field(default_factory=dict)

    def register(self, cell: TrainableDynamicalCell) -> None:
        name = getattr(cell, "name", "")
        if not name:
            raise ValueError("registered cell must have a nonempty name")
        if name in self._cells:
            raise ValueError(f"cell {name!r} is already registered")
        self._cells[name] = cell

    def get(self, name: str) -> TrainableDynamicalCell:
        try:
            return self._cells[name]
        except KeyError as exc:
            raise KeyError(f"unknown cell {name!r}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._cells))
