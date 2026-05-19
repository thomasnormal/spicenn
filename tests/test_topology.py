import numpy as np

from sim.topology import grid_coordinates, local_grid_edges, relay_chain_edges, small_world_edges, wire_metrics


def test_grid_coordinates_and_local_edges():
    coords = grid_coordinates(3, 3, pitch_um=2.0)
    assert coords.shape == (9, 2)
    edges = local_grid_edges(3, 3, radius=1)
    assert len(edges) > 9
    metrics = wire_metrics(coords, edges)
    assert metrics.synapse_count == len(edges)
    assert metrics.total_wire_length_um > 0


def test_small_world_adds_shortcuts():
    local = local_grid_edges(5, 5, radius=1)
    sw = small_world_edges(5, 5, radius=1, shortcut_fraction=0.2, seed=1)
    assert len(sw) >= len(local)


def test_relay_chain_edges():
    assert relay_chain_edges(0, 9, 3, 0) == [(0, 9)]
    assert relay_chain_edges(0, 9, 3, 2) == [(0, 3), (3, 4), (4, 9)]

