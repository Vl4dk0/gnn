"""Tests for the (k,g)-graph forge cascade and its excision shrink step.

These are fast smoke tests: small group orders and short search budgets, no
training.  They exercise the classical fallbacks of every stage (the excision
repair policy is trained on PERUN and absent locally), which is the intended
graceful-degradation path.
"""

from __future__ import annotations

import networkx as nx

from ai.cage.excision.repair_search import excise_and_repair
from ai.cage.forge import forge_graph
from backend.utils.graph_utils import compute_girth, is_k_regular


def _girth(G: nx.Graph[int]) -> int | float:
    return compute_girth(G)


class TestForgeGraph:
    """forge_graph(3, 6) must produce a valid (3,6)-graph end to end."""

    def test_forge_3_6_is_valid(self) -> None:
        G = forge_graph(
            3,
            6,
            predictor=None,
            max_voltage_attempts=15,
            max_group_order=20,
            tabu_iterations=800,
            time_budget=120,
            verbose=False,
        )
        assert G is not None, "forge_graph(3,6) returned None"
        assert is_k_regular(G, 3), "forged graph is not 3-regular"
        girth = _girth(G)
        assert isinstance(girth, int) and girth >= 6, f"girth {girth} < 6"
        # The (3,6)-cage (Heawood) has 14 vertices; a valid forged graph is at
        # least that big.
        assert G.number_of_nodes() >= 14


class TestExciseAndRepair:
    """excise_and_repair must shrink when feasible and return None cleanly otherwise."""

    def _pappus(self) -> nx.Graph[int]:
        # Pappus graph: 3-regular, girth 6, 18 vertices.
        g = nx.LCF_graph(18, [5, 7, -7, 7, -7, -5], 3)
        return nx.convert_node_labels_to_integers(g)

    def test_shrinks_pappus_g4(self) -> None:
        """Depth-1 (g=4) excision of Pappus(18) repairs into a smaller (3,4)-graph."""
        G = self._pappus()
        smaller = excise_and_repair(G, 3, 4, None)
        assert smaller is not None, "expected a successful shrink"
        assert smaller.number_of_nodes() < G.number_of_nodes()
        assert is_k_regular(smaller, 3)
        girth = _girth(smaller)
        assert isinstance(girth, int) and girth >= 4

    def test_heawood_cage_returns_none(self) -> None:
        """The (3,6)-cage cannot be shrunk; excise_and_repair returns None."""
        G = nx.convert_node_labels_to_integers(nx.heawood_graph())
        result = excise_and_repair(G, 3, 6, None)
        assert result is None

    def test_no_crash_when_policy_absent(self) -> None:
        """Classical fallback path runs without raising for any root."""
        G = self._pappus()
        # g=6 depth-2 excision on 18 vertices is infeasible to repair small;
        # it must not crash and must return None cleanly.
        result = excise_and_repair(G, 3, 6, None)
        assert result is None or is_k_regular(result, 3)
