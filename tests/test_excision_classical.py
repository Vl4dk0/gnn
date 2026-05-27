"""Tests for classical baseline repair algorithms.

Tests backtracking_repair and greedy_match_repair on Petersen/Heawood
depth-1 excisions.

Key constraint: the Petersen residue (6 vertices after depth-1 excision)
has diameter 3.  For g_target=5, legality requires dist >= 4, which is
impossible on 6 vertices with diameter 3.  Therefore we test:
  - g_target=4 on Petersen residue (requires dist >= 3, achievable — diameter is 3).
  - g_target=6 on Heawood residue (10 vertices, may have dist >= 5 pairs).

This also models realistic usage: tree excision on large graphs (>> 10 vertices)
will produce residues with larger diameter where g_target-repairs are feasible.
"""

from __future__ import annotations

import networkx as nx

from ai.cage.excision.baseline_repair import backtracking_repair, greedy_match_repair
from ai.cage.excision.excise import excise_tree


def petersen() -> nx.Graph[int]:
    g = nx.petersen_graph()
    return nx.convert_node_labels_to_integers(g)


def heawood() -> nx.Graph[int]:
    g = nx.heawood_graph()
    return nx.convert_node_labels_to_integers(g)


def is_regular(G: nx.Graph[int], k: int) -> bool:
    return all(d == k for _, d in G.degree())


def has_girth_at_least(G: nx.Graph[int], g: int) -> bool:
    try:
        return nx.girth(G) >= g
    except nx.NetworkXError:
        return False


class TestBacktrackingRepairPetersenG4:
    """g_target=4 on Petersen depth-1 residue.

    The 6-vertex residue has diameter 3 >= g-1=3, so some pairs are legal.
    The repair adds 3 edges to make all 6 vertices 3-regular.
    """

    def test_succeeds_and_returns_graph(self) -> None:
        G = petersen()
        reduced, deficient, _ = excise_tree(G, 0, depth=1)
        result = backtracking_repair(reduced, deficient, g_target=4, k=3)
        assert result is not None, (
            "backtracking_repair returned None on Petersen depth-1 g_target=4"
        )

    def test_result_is_3_regular(self) -> None:
        G = petersen()
        reduced, deficient, _ = excise_tree(G, 0, depth=1)
        result = backtracking_repair(reduced, deficient, g_target=4, k=3)
        assert result is not None
        degs = [d for _, d in result.degree()]
        assert all(d == 3 for d in degs), f"Degrees: {degs}"

    def test_result_girth_at_least_4(self) -> None:
        G = petersen()
        reduced, deficient, _ = excise_tree(G, 0, depth=1)
        result = backtracking_repair(reduced, deficient, g_target=4, k=3)
        assert result is not None
        assert has_girth_at_least(result, 4), f"Girth {nx.girth(result)} < 4"

    def test_result_is_smaller_than_original(self) -> None:
        G = petersen()
        reduced, deficient, _ = excise_tree(G, 0, depth=1)
        result = backtracking_repair(reduced, deficient, g_target=4, k=3)
        assert result is not None
        assert result.number_of_nodes() < G.number_of_nodes()

    def test_result_is_simple(self) -> None:
        G = petersen()
        reduced, deficient, _ = excise_tree(G, 0, depth=1)
        result = backtracking_repair(reduced, deficient, g_target=4, k=3)
        assert result is not None
        assert isinstance(result, nx.Graph)
        assert all(u != v for u, v in result.edges())

    def test_all_roots_can_be_repaired(self) -> None:
        """By Petersen's vertex-transitivity, all roots should give a valid repair."""
        G = petersen()
        for root in G.nodes():
            reduced, deficient, _ = excise_tree(G, root, depth=1)
            if not deficient:
                continue
            result = backtracking_repair(reduced, deficient, g_target=4, k=3)
            assert result is not None, f"Repair failed for root={root}"
            assert is_regular(result, k=3), f"Not 3-regular for root={root}"
            assert has_girth_at_least(result, 4), f"Girth < 4 for root={root}"


class TestPetersenDepth1G5Infeasible:
    """Document that g_target=5 repair on the Petersen depth-1 residue is infeasible.

    The 6-vertex residue has diameter 3 < g-1=4, so backtracking_repair returns
    None.  This is not a bug; it reflects the mathematical constraint: the Petersen
    graph is already the (3,5)-cage (smallest possible), so any strict sub-graph
    cannot be repaired to a valid (3,5)-graph.
    """

    def test_g5_repair_returns_none(self) -> None:
        G = petersen()
        reduced, deficient, _ = excise_tree(G, 0, depth=1)
        result = backtracking_repair(reduced, deficient, g_target=5, k=3)
        assert result is None, (
            "Expected None: Petersen depth-1 residue cannot be repaired to girth 5"
        )


class TestGreedyRepairPetersenG4:
    def test_greedy_succeeds_on_some_root(self) -> None:
        G = petersen()
        successes = 0
        for root in G.nodes():
            reduced, deficient, _ = excise_tree(G, root, depth=1)
            if not deficient:
                continue
            result = greedy_match_repair(reduced, deficient, g_target=4, k=3)
            if result is not None:
                successes += 1
        assert successes > 0, (
            "Greedy repair never succeeded on any Petersen depth-1 g_target=4 instance"
        )

    def test_greedy_result_valid_when_succeeds(self) -> None:
        G = petersen()
        for root in G.nodes():
            reduced, deficient, _ = excise_tree(G, root, depth=1)
            if not deficient:
                continue
            result = greedy_match_repair(reduced, deficient, g_target=4, k=3)
            if result is not None:
                assert is_regular(result, k=3), f"Not 3-regular for root={root}"
                assert has_girth_at_least(result, 4), f"Girth < 4 for root={root}"


class TestBacktrackingRepairHeawood:
    def test_heawood_depth1_g4_repair(self) -> None:
        """Heawood depth-1 residue has 10 vertices; g_target=4 repair should succeed."""
        G = heawood()
        successes = 0
        for root in list(G.nodes())[:5]:
            reduced, deficient, _ = excise_tree(G, root, depth=1)
            if not deficient:
                continue
            result = backtracking_repair(reduced, deficient, g_target=4, k=3)
            if result is not None:
                assert is_regular(result, k=3), f"Not 3-regular for root={root}"
                assert has_girth_at_least(result, 4), f"Girth < 4 for root={root}"
                successes += 1
        assert successes > 0, "No Heawood depth-1 repair succeeded for g_target=4"

    def test_heawood_depth1_no_crash(self) -> None:
        """Repair does not crash regardless of g_target; it may return None."""
        G = heawood()
        for root in list(G.nodes())[:3]:
            reduced, deficient, _ = excise_tree(G, root, depth=1)
            if not deficient:
                continue
            # Just verify it runs without exception
            _ = backtracking_repair(reduced, deficient, g_target=6, k=3)
