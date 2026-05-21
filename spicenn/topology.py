from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Union

import numpy as np


SourceId = Union[int, str]


@dataclass(frozen=True)
class FanInTopology:
    """Output-indexed connectivity for sparse SPICE layer generation."""

    sources: tuple[SourceId, ...]
    sink_count: int
    fanins: dict[int, tuple[SourceId, ...]]

    @classmethod
    def dense(cls, sources: Iterable[SourceId], sink_count: int) -> "FanInTopology":
        source_tuple = tuple(sources)
        if sink_count <= 0:
            raise ValueError("sink_count must be positive")
        return cls(
            sources=source_tuple,
            sink_count=sink_count,
            fanins={sink: source_tuple for sink in range(sink_count)},
        )

    @classmethod
    def random_fanin(
        cls,
        sources: Iterable[SourceId],
        sink_count: int,
        *,
        seed: int,
        fan_in: int,
        always_sources: Iterable[SourceId] = (),
    ) -> "FanInTopology":
        source_tuple = tuple(sources)
        always_tuple = tuple(always_sources)
        if sink_count <= 0:
            raise ValueError("sink_count must be positive")
        if fan_in <= 0:
            raise ValueError("fan_in must be positive")
        if fan_in > len(source_tuple):
            raise ValueError("fan_in cannot exceed source count without duplicate synapses")
        rng = np.random.default_rng(seed)
        fanins: dict[int, tuple[SourceId, ...]] = {}
        for sink in range(sink_count):
            indices = rng.choice(len(source_tuple), size=fan_in, replace=False)
            selected = tuple(source_tuple[int(index)] for index in indices)
            fanins[sink] = (*always_tuple, *tuple(sorted(selected)))
        return cls(sources=(*always_tuple, *source_tuple), sink_count=sink_count, fanins=fanins)

    @classmethod
    def random_fanout(
        cls,
        sources: Iterable[SourceId],
        sink_count: int,
        *,
        seed: int,
        fan_out: int,
    ) -> "FanInTopology":
        source_tuple = tuple(sources)
        if sink_count <= 0:
            raise ValueError("sink_count must be positive")
        if fan_out <= 0:
            raise ValueError("fan_out must be positive")
        if fan_out > sink_count:
            raise ValueError("fan_out cannot exceed sink_count without duplicate synapses")
        rng = np.random.default_rng(seed)
        fanin_lists: dict[int, list[SourceId]] = {sink: [] for sink in range(sink_count)}
        for source in source_tuple:
            for sink in rng.choice(sink_count, size=fan_out, replace=False):
                fanin_lists[int(sink)].append(source)
        return cls(
            sources=source_tuple,
            sink_count=sink_count,
            fanins={sink: tuple(sorted(srcs)) for sink, srcs in fanin_lists.items()},
        )

    @classmethod
    def balanced_random_fanout(
        cls,
        sources: Iterable[SourceId],
        sink_count: int,
        *,
        seed: int,
        fan_out: int,
    ) -> "FanInTopology":
        """Random fan-out with nearly equal fan-in counts across sinks.

        This preserves the hardware constraint that each source/neuron drives a
        fixed number of output synapses, while avoiding unlucky class rows with
        much smaller fan-in.
        """
        source_tuple = tuple(sources)
        if sink_count <= 0:
            raise ValueError("sink_count must be positive")
        if fan_out <= 0:
            raise ValueError("fan_out must be positive")
        if fan_out > sink_count:
            raise ValueError("fan_out cannot exceed sink_count without duplicate synapses")
        rng = np.random.default_rng(seed)
        total_edges = len(source_tuple) * fan_out
        base_capacity, extra = divmod(total_edges, sink_count)
        extra_sinks = set(int(sink) for sink in rng.choice(sink_count, size=extra, replace=False))
        remaining = {
            sink: base_capacity + (1 if sink in extra_sinks else 0)
            for sink in range(sink_count)
        }
        fanin_lists: dict[int, list[SourceId]] = {sink: [] for sink in range(sink_count)}
        for source_index in rng.permutation(len(source_tuple)):
            source = source_tuple[int(source_index)]
            selected: list[int] = []
            for _edge in range(fan_out):
                candidates = [
                    sink
                    for sink, capacity in remaining.items()
                    if capacity > 0 and sink not in selected
                ]
                if not candidates:
                    raise RuntimeError("balanced fanout assignment failed; check fan_out and sink_count")
                max_remaining = max(remaining[sink] for sink in candidates)
                best = [sink for sink in candidates if remaining[sink] == max_remaining]
                sink = int(rng.choice(best))
                selected.append(sink)
                remaining[sink] -= 1
                fanin_lists[sink].append(source)
        if any(remaining.values()):
            raise RuntimeError("balanced fanout assignment left unfilled sink capacity")
        return cls(
            sources=source_tuple,
            sink_count=sink_count,
            fanins={sink: tuple(sorted(srcs)) for sink, srcs in fanin_lists.items()},
        )

    @classmethod
    def ring_fanout(
        cls,
        sources: Iterable[SourceId],
        sink_count: int,
        *,
        fan_out: int,
        offset: int = 0,
    ) -> "FanInTopology":
        """Deterministic sparse fan-out with low-clump multiclass coverage.

        Source ``i`` connects to ``fan_out`` consecutive sinks on a ring.  Start
        positions advance by ``fan_out`` so edge load stays close to balanced
        even when ``sink_count`` and ``fan_out`` are not divisors.  This
        keeps every source at the requested out-degree while distributing nearby
        hidden features across class rows more predictably than random fan-out.
        """
        source_tuple = tuple(sources)
        if sink_count <= 0:
            raise ValueError("sink_count must be positive")
        if fan_out <= 0:
            raise ValueError("fan_out must be positive")
        if fan_out > sink_count:
            raise ValueError("fan_out cannot exceed sink_count without duplicate synapses")
        fanin_lists: dict[int, list[SourceId]] = {sink: [] for sink in range(sink_count)}
        for source_index, source in enumerate(source_tuple):
            start = (source_index * fan_out + offset) % sink_count
            for edge in range(fan_out):
                fanin_lists[(start + edge) % sink_count].append(source)
        return cls(
            sources=source_tuple,
            sink_count=sink_count,
            fanins={sink: tuple(srcs) for sink, srcs in fanin_lists.items()},
        )

    @classmethod
    def from_fanins(
        cls,
        sources: Iterable[SourceId],
        sink_count: int,
        fanins: dict[int, tuple[SourceId, ...]],
    ) -> "FanInTopology":
        if sink_count <= 0:
            raise ValueError("sink_count must be positive")
        return cls(
            sources=tuple(sources),
            sink_count=sink_count,
            fanins={sink: tuple(fanins.get(sink, ())) for sink in range(sink_count)},
        )

    def as_fanins(self) -> dict[int, tuple[SourceId, ...]]:
        return {sink: tuple(self.fanins.get(sink, ())) for sink in range(self.sink_count)}

    def fanouts(self) -> dict[SourceId, tuple[int, ...]]:
        fanout_lists: dict[SourceId, list[int]] = {source: [] for source in self.sources}
        for sink, srcs in self.fanins.items():
            for source in srcs:
                fanout_lists.setdefault(source, []).append(sink)
        return {source: tuple(sinks) for source, sinks in fanout_lists.items()}

    def fanin_counts(self, *, exclude_sources: Iterable[SourceId] = ()) -> list[int]:
        excluded = set(exclude_sources)
        return [
            len(tuple(source for source in self.fanins.get(sink, ()) if source not in excluded))
            for sink in range(self.sink_count)
        ]

    def fanout_counts(self, *, sources: Iterable[SourceId] | None = None) -> list[int]:
        selected_sources = self.sources if sources is None else tuple(sources)
        fanouts = self.fanouts()
        return [len(fanouts.get(source, ())) for source in selected_sources]

    def edge_count(self, *, exclude_sources: Iterable[SourceId] = ()) -> int:
        return int(sum(self.fanin_counts(exclude_sources=exclude_sources)))

    def summary(self, *, prefix: str, fanout_sources: Iterable[SourceId] | None = None) -> dict[str, Any]:
        selected_sources = self.sources if fanout_sources is None else tuple(fanout_sources)
        fanouts = self.fanouts()
        return {
            f"{prefix}_edge_count": self.edge_count(),
            f"{prefix}_fanin_counts": self.fanin_counts(),
            f"{prefix}_fanout_counts": self.fanout_counts(sources=selected_sources),
            f"{prefix}_fanins": {str(sink): list(self.fanins.get(sink, ())) for sink in range(self.sink_count)},
            f"{prefix}_fanouts": {str(source): list(fanouts.get(source, ())) for source in selected_sources},
        }
