"""Tests for short-cycle cost function on known graphs."""

from __future__ import annotations

import networkx as nx

from ai.cage.refine.cost import count_cycles_up_to, short_cycle_cost


class TestCountCyclesPetersen:
    def test_petersen_no_cycles_below_5(self) -> None:
        graph = nx.convert_node_labels_to_integers(nx.petersen_graph())
        counts = count_cycles_up_to(graph, max_len=4)
        assert counts[3] == 0
        assert counts[4] == 0

    def test_petersen_has_5_cycles(self) -> None:
        graph = nx.convert_node_labels_to_integers(nx.petersen_graph())
        counts = count_cycles_up_to(graph, max_len=5)
        # Petersen's girth is 5; it has exactly 12 5-cycles.
        assert counts[5] == 12


class TestCountCyclesK4:
    def test_k4_has_4_triangles(self) -> None:
        graph: nx.Graph[int] = nx.convert_node_labels_to_integers(nx.complete_graph(4))
        counts = count_cycles_up_to(graph, max_len=3)
        # K4 has C(4,3) = 4 triangles.
        assert counts[3] == 4


class TestShortCycleCost:
    def test_petersen_zero_cost_for_g5(self) -> None:
        graph = nx.convert_node_labels_to_integers(nx.petersen_graph())
        assert short_cycle_cost(graph, g_target=5) == 0.0

    def test_k4_positive_cost_for_g4(self) -> None:
        graph: nx.Graph[int] = nx.convert_node_labels_to_integers(nx.complete_graph(4))
        # g_target=4 penalises triangles (length 3); weight = 2**(4-3) = 2
        # 4 triangles * weight 2 = 8.
        assert short_cycle_cost(graph, g_target=4) == 8.0

    def test_g_target_3_returns_zero(self) -> None:
        graph: nx.Graph[int] = nx.convert_node_labels_to_integers(nx.complete_graph(4))
        assert short_cycle_cost(graph, g_target=3) == 0.0

    def test_custom_weights_respected(self) -> None:
        graph: nx.Graph[int] = nx.convert_node_labels_to_integers(nx.complete_graph(4))
        cost = short_cycle_cost(graph, g_target=4, weights={3: 1.0})
        assert cost == 4.0
