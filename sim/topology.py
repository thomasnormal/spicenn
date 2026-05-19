from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class WireMetrics:
    synapse_count: int
    average_wire_length_um: float
    max_wire_length_um: float
    total_wire_length_um: float


def grid_coordinates(width: int, height: int, pitch_um: float = 10.0) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    return np.stack([xs.ravel() * pitch_um, ys.ravel() * pitch_um], axis=1).astype(float)


def dense_edges(n_src: int, n_dst: int, offset_dst: int = 0) -> list[tuple[int, int]]:
    return [(i, offset_dst + j) for i in range(n_src) for j in range(n_dst)]


def local_grid_edges(width: int, height: int, radius: int = 1, include_self: bool = True) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for y in range(height):
        for x in range(width):
            dst = y * width + x
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if not include_self and dx == 0 and dy == 0:
                        continue
                    sx, sy = x + dx, y + dy
                    if 0 <= sx < width and 0 <= sy < height:
                        edges.append((sy * width + sx, dst))
    return edges


def small_world_edges(
    width: int,
    height: int,
    radius: int = 1,
    shortcut_fraction: float = 0.01,
    seed: int = 0,
) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    edges = local_grid_edges(width, height, radius)
    n_nodes = width * height
    n_shortcuts = int(round(len(edges) * shortcut_fraction))
    for _ in range(n_shortcuts):
        src = int(rng.integers(0, n_nodes))
        dst = int(rng.integers(0, n_nodes))
        if src != dst:
            edges.append((src, dst))
    return edges


def relay_chain_edges(src: int, dst: int, first_relay: int, chain_length: int) -> list[tuple[int, int]]:
    if chain_length <= 0:
        return [(src, dst)]
    relays = list(range(first_relay, first_relay + chain_length))
    return list(zip([src] + relays[:-1], relays)) + [(relays[-1], dst)]


def wire_metrics(coords: np.ndarray, edges: Iterable[tuple[int, int]], metric: str = "manhattan") -> WireMetrics:
    lengths = []
    for src, dst in edges:
        delta = np.abs(coords[dst] - coords[src])
        if metric == "euclidean":
            lengths.append(float(np.linalg.norm(delta)))
        else:
            lengths.append(float(delta.sum()))
    if not lengths:
        return WireMetrics(0, 0.0, 0.0, 0.0)
    arr = np.asarray(lengths, dtype=float)
    return WireMetrics(len(lengths), float(arr.mean()), float(arr.max()), float(arr.sum()))


def graph_from_edges(n_nodes: int, edges: Iterable[tuple[int, int]]) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(range(n_nodes))
    g.add_edges_from(edges)
    return g

