"""Tests for classical TabuRefiner (no ML).

Verifies plumbing: the loop runs end-to-end, returns a graph and stats,
and the final cost is no worse than the initial cost.
"""

from __future__ import annotations

import random

import networkx as nx

from ai.cage.refine.cost import short_cycle_cost
from ai.cage.refine.tabu import TabuRefiner


def _broken_3_regular(n: int = 14, seed: int = 0) -> nx.Graph[int]:
    """A small 3-regular graph likely to contain short cycles."""
    random.seed(seed)
    graph = nx.random_regular_graph(3, n, seed=seed)
    return nx.convert_node_labels_to_integers(graph)


class TestTabuClassical:
    def test_returns_graph_and_stats(self) -> None:
        graph = _broken_3_regular()
        refiner = TabuRefiner(g_target=5, max_iter=20, sample_size=50)
        result, stats = refiner.refine(graph)
        assert result is not None
        assert "initial_cost" in stats
        assert "final_cost" in stats
        assert "iterations" in stats

    def test_final_cost_not_worse_than_initial(self) -> None:
        graph = _broken_3_regular()
        initial = short_cycle_cost(graph, g_target=5)
        refiner = TabuRefiner(g_target=5, max_iter=30, sample_size=80)
        _, stats = refiner.refine(graph)
        assert stats["final_cost"] <= initial

    def test_preserves_regularity(self) -> None:
        graph = _broken_3_regular()
        refiner = TabuRefiner(g_target=5, max_iter=20, sample_size=50)
        result, _ = refiner.refine(graph)
        assert all(d == 3 for _, d in result.degree())

    def test_zero_iter_returns_input_cost(self) -> None:
        graph = _broken_3_regular()
        refiner = TabuRefiner(g_target=5, max_iter=0)
        _, stats = refiner.refine(graph)
        assert stats["final_cost"] == stats["initial_cost"]
