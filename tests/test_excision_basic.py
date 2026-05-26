"""Basic excision tests.

Petersen graph: 3-regular, girth 5, 10 vertices.
BFS excision at depth d = floor((5-1)/2) = 2 from any vertex:
  - Depth 2 on Petersen: root + 3 neighbours + 6 next-level = 10 vertices.
    That removes the whole graph — not useful for testing.

Depth 1 on Petersen:
  - Tree = root (1 vertex) + 3 neighbours (3 vertices) = 4 vertices removed.
  - Remaining: 10 - 4 = 6 vertices.
  - Each remaining vertex was adjacent to exactly one of the 3 removed neighbours
    (Petersen is 3-regular, neighbours of root are connected to 2 non-root
    non-tree vertices each). Let's verify the counts empirically.
"""

from __future__ import annotations

import networkx as nx
import pytest

from ai.cage.excision.excise import excise_tree


def petersen() -> nx.Graph[int]:
    g = nx.petersen_graph()
    return nx.convert_node_labels_to_integers(g)


class TestExcisePetersenDepth1:
    def test_reduced_vertex_count(self) -> None:
        """After depth-1 excision, reduced graph has 10 - 4 = 6 vertices."""
        graph = petersen()
        root = 0
        reduced, deficient, levels = excise_tree(graph, root, depth=1)
        assert reduced.number_of_nodes() == 6

    def test_deficient_count(self) -> None:
        """Exactly 3 boundary vertices become deficient (one per removed neighbour)."""
        graph = petersen()
        root = 0
        reduced, deficient, levels = excise_tree(graph, root, depth=1)
        # Each of root's 3 neighbours had 2 edges leaving the tree (one to another
        # tree-member neighbour of root via Petersen structure, and one outward).
        # Actually: each removed neighbour had degree 3 in graph.  It was connected to
        # root (1 edge) + 2 non-tree vertices.  So each removed neighbour exposes
        # 2 deficient boundary vertices.  But each non-tree vertex was adjacent to
        # AT MOST ONE removed neighbour (Petersen has girth 5, so no triangles).
        # Empirically: 3 removed neighbours × 2 boundary connections = 6 deficient
        # vertices — but there are only 6 remaining vertices and each loses exactly
        # 1 neighbour, so all 6 are deficient with deficiency 1.
        assert len(deficient) == 6
        for v in deficient:
            assert levels[v] == 1

    def test_deficient_vertices_have_degree_2(self) -> None:
        """All deficient vertices have degree 2 in the reduced graph (k-1 = 2)."""
        graph = petersen()
        root = 0
        reduced, deficient, _ = excise_tree(graph, root, depth=1)
        for v in deficient:
            assert reduced.degree(v) == 2

    def test_by_symmetry_all_roots_give_same_counts(self) -> None:
        """Petersen is vertex-transitive: all roots give the same excision shape."""
        graph = petersen()
        for root in graph.nodes():
            reduced, deficient, levels = excise_tree(graph, root, depth=1)
            assert reduced.number_of_nodes() == 6
            assert len(deficient) == 6

    def test_original_graph_not_modified(self) -> None:
        """excise_tree must not modify the input graph."""
        graph = petersen()
        n_before = graph.number_of_nodes()
        m_before = graph.number_of_edges()
        _ = excise_tree(graph, 0, depth=1)
        assert graph.number_of_nodes() == n_before
        assert graph.number_of_edges() == m_before

    def test_reduced_graph_is_simple(self) -> None:
        """Reduced graph must be a simple graph (no self-loops, no multi-edges)."""
        graph = petersen()
        reduced, _, _ = excise_tree(graph, 0, depth=1)
        assert not any(u == v for u, v in reduced.edges())
        # NetworkX Graph is always simple by construction — just double-check type
        assert isinstance(reduced, nx.Graph)


class TestExcisePetersenDepth2:
    def test_removes_whole_graph(self) -> None:
        """BFS depth 2 on Petersen covers all 10 vertices (diameter = 2)."""
        graph = petersen()
        reduced, deficient, levels = excise_tree(graph, 0, depth=2)
        # All nodes reachable within 2 hops from any vertex in Petersen (diameter=2)
        assert reduced.number_of_nodes() == 0
        assert deficient == []


class TestExciseHeawoodDepth1:
    def test_heawood_depth1(self) -> None:
        """Heawood: 3-regular, girth 6, 14 vertices. Depth-1 removes 4 vertices."""
        graph = nx.heawood_graph()
        graph = nx.convert_node_labels_to_integers(graph)
        reduced, deficient, levels = excise_tree(graph, 0, depth=1)
        assert reduced.number_of_nodes() == 14 - 4  # root + 3 neighbours
        # Deficient vertices have degree 2 (lost one neighbour each)
        for v in deficient:
            assert reduced.degree(v) == 2
            assert levels[v] == 1
