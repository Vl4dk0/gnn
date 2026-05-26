"""Tests for 2-switch / 3-switch enumeration and application.

Verifies the core invariant: applying any enumerated swap preserves
k-regularity and produces a simple graph.
"""

from __future__ import annotations

import networkx as nx

from ai.cage.refine.swaps import (
    apply_2_switch,
    apply_3_switch,
    enumerate_2_switches,
    enumerate_3_switches,
)


def _petersen():
    graph = nx.petersen_graph()
    return nx.convert_node_labels_to_integers(graph)


class TestEnumerate2Switches:
    def test_petersen_yields_some_swaps(self) -> None:
        graph = _petersen()
        swaps = list(enumerate_2_switches(graph))
        assert len(swaps) > 0, "Petersen should admit at least one 2-switch"

    def test_yielded_endpoints_distinct(self) -> None:
        graph = _petersen()
        for swap in enumerate_2_switches(graph):
            assert len({swap.u, swap.v, swap.x, swap.y}) == 4


class TestApply2Switch:
    def test_preserves_regularity(self) -> None:
        graph = _petersen()
        for swap in list(enumerate_2_switches(graph))[:20]:
            result = apply_2_switch(graph, swap)
            assert all(d == 3 for _, d in result.degree())

    def test_no_multi_edges_or_self_loops(self) -> None:
        graph = _petersen()
        for swap in list(enumerate_2_switches(graph))[:20]:
            result = apply_2_switch(graph, swap)
            edges = list(result.edges())
            assert not any(a == b for a, b in edges), "self-loop introduced"
            assert len(edges) == len(set((min(a, b), max(a, b)) for a, b in edges))


class TestApply3Switch:
    def test_preserves_regularity_when_yielded(self) -> None:
        graph = _petersen()
        count = 0
        for swap in enumerate_3_switches(graph, sample_size=20):
            result = apply_3_switch(graph, swap)
            assert all(d == 3 for _, d in result.degree())
            count += 1
            if count >= 10:
                break
