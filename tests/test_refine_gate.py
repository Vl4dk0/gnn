"""Tests for the size-relative refine-routing gate metric.

``short_cycle_edge_fraction`` measures the share of edges lying on at least one
simple cycle shorter than ``g_target``; the forge ``defect_density`` gate uses
it to route near-miss lifts to refine.
"""

from __future__ import annotations

import networkx as nx

from ai.cage.refine.cost import short_cycle_edge_fraction


class TestSingleTriangle:
    def test_one_triangle_plus_long_path(self) -> None:
        # A single triangle (0-1-2) plus a disjoint long path. With a large
        # g_target the only short cycle is the triangle, so exactly its 3 edges
        # are defective.
        graph: nx.Graph[int] = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2), (2, 0)])
        graph.add_edges_from([(3, 4), (4, 5), (5, 6), (6, 7)])
        m = graph.number_of_edges()
        assert short_cycle_edge_fraction(graph, g_target=7) == 3 / m


class TestNoShortCycles:
    def test_girth_at_least_target_returns_zero(self) -> None:
        # Petersen has girth 5; with g_target=5 there are no cycles below 5.
        graph = nx.convert_node_labels_to_integers(nx.petersen_graph())
        assert short_cycle_edge_fraction(graph, g_target=5) == 0.0

    def test_no_edges_returns_zero(self) -> None:
        graph: nx.Graph[int] = nx.Graph()
        graph.add_nodes_from([0, 1, 2])
        assert short_cycle_edge_fraction(graph, g_target=7) == 0.0

    def test_g_target_le_3_returns_zero(self) -> None:
        graph: nx.Graph[int] = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2), (2, 0)])
        assert short_cycle_edge_fraction(graph, g_target=3) == 0.0


class TestEarlyExit:
    def test_max_fraction_early_exit(self) -> None:
        # K5 is highly defective: every edge lies on a triangle, so the true
        # fraction is 1.0. With a small budget the generator must stop early and
        # return a value strictly greater than max_fraction.
        graph: nx.Graph[int] = nx.convert_node_labels_to_integers(nx.complete_graph(5))
        result = short_cycle_edge_fraction(graph, g_target=4, max_fraction=0.1)
        assert result > 0.1

    def test_within_budget_returns_true_fraction(self) -> None:
        # A single triangle: 3 bad edges out of 3 total = 1.0, which is within a
        # generous budget, so the exact fraction is returned.
        graph: nx.Graph[int] = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2), (2, 0)])
        assert short_cycle_edge_fraction(graph, g_target=7, max_fraction=1.0) == 1.0
