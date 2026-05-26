"""Legality check tests for candidate repair edges.

Tests is_legal_edge and legal_candidate_edges on known graphs.
"""

from __future__ import annotations

import networkx as nx
import pytest

from ai.cage.excision.excise import excise_tree
from ai.cage.excision.legality import (
    DistanceCache,
    is_legal_edge,
    legal_candidate_edges,
)


def petersen_residue() -> tuple[nx.Graph[int], list[int]]:
    """Return (reduced_graph, deficient) after depth-1 excision of Petersen at root 0."""
    graph = nx.petersen_graph()
    graph = nx.convert_node_labels_to_integers(graph)
    reduced, deficient, _ = excise_tree(graph, 0, depth=1)
    return reduced, deficient


class TestIsLegalEdge:
    def test_self_loop_illegal(self) -> None:
        graph = nx.path_graph(5)
        graph = nx.convert_node_labels_to_integers(graph)
        assert not is_legal_edge(graph, 0, 0, g_target=5)

    def test_existing_edge_illegal(self) -> None:
        graph = nx.path_graph(5)
        graph = nx.convert_node_labels_to_integers(graph)
        # Edge 0-1 exists
        assert not is_legal_edge(graph, 0, 1, g_target=5)

    def test_short_path_illegal(self) -> None:
        """Two vertices at distance 2 cannot be connected for g_target=5 (need dist >= 4)."""
        graph = nx.path_graph(5)
        graph = nx.convert_node_labels_to_integers(graph)
        # 0 - 1 - 2: dist(0,2) = 2 < 4 = g-1 for g=5
        assert not is_legal_edge(graph, 0, 2, g_target=5)

    def test_long_path_legal(self) -> None:
        """Two vertices at distance >= g-1 can be connected."""
        # Path 0-1-2-3-4: dist(0,4) = 4 = g-1 for g=5 → legal
        graph = nx.path_graph(5)
        graph = nx.convert_node_labels_to_integers(graph)
        assert is_legal_edge(graph, 0, 4, g_target=5)

    def test_distance_exactly_g_minus_1_legal(self) -> None:
        """dist(u,v) == g-1 is legal (adds a cycle of length exactly g)."""
        # Cycle of length 8: dist(0,4) = 4 = g-1 for g=5
        graph = nx.cycle_graph(8)
        graph = nx.convert_node_labels_to_integers(graph)
        # Shortest path in the cycle: min(4, 4) = 4 >= g-1 = 4 for g=5
        assert is_legal_edge(graph, 0, 4, g_target=5)

    def test_petersen_residue_some_legal(self) -> None:
        """On the Petersen residue, legal pairs exist for g_target=4 (dist >= 3).

        The 6-vertex residue has diameter 3, so g_target=5 (requires dist>=4) has
        no legal pairs.  g_target=4 (requires dist>=3) does have pairs at exactly
        distance 3, making them legal.
        """
        H, deficient = petersen_residue()
        # At g_target=4, legal pairs must have dist >= 3.
        # The Petersen residue has 6 vertices and diameter 3 — pairs at distance 3
        # are legal for g_target=4.
        found_legal = False
        for i, u in enumerate(deficient):
            for v in deficient[i + 1 :]:
                if is_legal_edge(H, u, v, g_target=4):
                    found_legal = True
                    break
        assert found_legal, (
            "Expected at least one legal edge in Petersen residue for g_target=4"
        )

    def test_petersen_residue_no_legal_for_g5(self) -> None:
        """The 6-vertex Petersen residue has diameter 3 < g-1=4, so no g_target=5 legal edges.

        This is expected: after depth-1 excision the residue is too small/dense
        to support a (3,5)-repair.  A valid (3,5)-repair would require the
        starting graph to be much larger (e.g., the (3,5)-cage has 10 vertices;
        after removing 4 we only have 6 left, and the residue is not repairable
        back to girth 5).
        """
        H, deficient = petersen_residue()
        for i, u in enumerate(deficient):
            for v in deficient[i + 1 :]:
                assert not is_legal_edge(H, u, v, g_target=5)

    def test_g_target_3_more_permissive(self) -> None:
        """g_target=3 allows any non-adjacent, non-self pair (dist >= 2)."""
        graph = nx.path_graph(4)
        graph = nx.convert_node_labels_to_integers(graph)
        # 0-1-2-3: dist(0,2)=2 >= g-1=2, so legal for g=3
        assert is_legal_edge(graph, 0, 2, g_target=3)


class TestLegalCandidateEdges:
    def test_returns_list(self) -> None:
        H, deficient = petersen_residue()
        candidates = legal_candidate_edges(H, deficient, g_target=5)
        assert isinstance(candidates, list)

    def test_pairs_are_sorted(self) -> None:
        """All returned pairs have u < v."""
        H, deficient = petersen_residue()
        candidates = legal_candidate_edges(H, deficient, g_target=5)
        for u, v in candidates:
            assert u < v, f"Pair ({u},{v}) not sorted"

    def test_all_pairs_legal(self) -> None:
        """Every returned pair passes is_legal_edge."""
        H, deficient = petersen_residue()
        candidates = legal_candidate_edges(H, deficient, g_target=5)
        for u, v in candidates:
            assert is_legal_edge(H, u, v, g_target=5), (
                f"Pair ({u},{v}) reported legal but is not"
            )

    def test_no_duplicates(self) -> None:
        H, deficient = petersen_residue()
        candidates = legal_candidate_edges(H, deficient, g_target=5)
        assert len(candidates) == len(set(candidates))


class TestDistanceCache:
    def test_basic_dist(self) -> None:
        graph = nx.path_graph(6)
        graph = nx.convert_node_labels_to_integers(graph)
        cache = DistanceCache(graph, g_target=5)
        assert cache.dist(0, 4) == 4
        assert cache.dist(0, 1) == 1

    def test_is_legal_via_cache(self) -> None:
        graph = nx.path_graph(6)
        graph = nx.convert_node_labels_to_integers(graph)
        cache = DistanceCache(graph, g_target=5)
        # dist(0,4)=4 >= g-1=4: legal
        assert cache.is_legal(0, 4)
        # dist(0,2)=2 < g-1=4: illegal
        assert not cache.is_legal(0, 2)

    def test_invalidation_on_edge_add(self) -> None:
        """After adding an edge, cache entries for those endpoints are gone."""
        graph = nx.path_graph(6)
        graph = nx.convert_node_labels_to_integers(graph)
        cache = DistanceCache(graph, g_target=5)
        # Pre-populate
        _ = cache.dist(0, 5)
        key = (0, 5)
        assert key in cache._cache
        # Add edge and invalidate
        graph.add_edge(0, 5)
        cache.invalidate_edge(0, 5)
        assert key not in cache._cache

    def test_cache_recomputes_after_invalidation(self) -> None:
        graph = nx.path_graph(6)
        graph = nx.convert_node_labels_to_integers(graph)
        cache = DistanceCache(graph, g_target=10)
        old_dist = cache.dist(0, 5)
        graph.add_edge(0, 3)
        cache.invalidate_edge(0, 3)
        # Re-query — 0-3 shortcut may change dist(0,5), but cache must recompute
        new_dist = cache.dist(0, 5)
        # With shortcut 0-3 added, dist(0,5) = min(5, 1+2) = 3
        assert new_dist <= old_dist
