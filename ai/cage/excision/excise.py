"""Tree excision for (k,g)-cage graphs.

Given a k-regular graph G with girth g, excising a BFS tree of depth d =
floor((g-1)/2) rooted at vertex v removes |T| vertices from G and leaves some
boundary vertices with degree k-1 or lower (the "deficient" set).

The depth d = floor((g-1)/2) ensures that the removed tree spans no cycles in G
(any cycle through the tree root has length >= g > 2d+1 for odd g, >= 2d+2 for
even g), so the excision does not destroy girth information outside the tree.
A successful repair of the deficient set yields a smaller (k,g)-graph.

Example (Petersen graph, g=5, d=1):
  - Remove root v + its 3 neighbours (tree has 4 vertices total)
  - Actually: Petersen is 3-regular, so the BFS tree at depth 1 has root + 3
    leaves = 4 vertices.  But Petersen only has 10 vertices; after removal only
    6 remain and the 6 boundary vertices each lose exactly 1 edge.
  - Wait: the 3 neighbours ARE the depth-1 layer; they are removed WITH the
    tree, not made deficient.  The deficient vertices are those outside the
    tree that were adjacent to TREE nodes.
"""

from __future__ import annotations

from collections import deque

import networkx as nx


def excise_tree(
    G: nx.Graph[int],
    root: int,
    depth: int,
) -> tuple[nx.Graph[int], list[int], dict[int, int]]:
    """BFS-remove a depth-d tree from G and return the reduced graph.

    BFS from `root` up to `depth` levels builds the tree T.  All vertices in T
    are deleted from G.  Vertices NOT in T that were adjacent to at least one T
    node lose edges; their degree drops below k = max degree of G.  These are
    the "deficient" vertices.

    The original graph G is not modified.

    Args:
        G:     A simple, k-regular NetworkX graph with integer node labels.
        root:  Root of the BFS tree to excise.
        depth: BFS depth.  For a (k,g)-graph use depth = (g-1) // 2.

    Returns:
        reduced_graph:      G with all tree nodes removed (copy).
        deficient_vertices: Sorted list of nodes in reduced_graph whose degree
                            is < max_degree(reduced_graph).  (max_degree is
                            k from the original graph, since all non-tree nodes
                            that had NO tree neighbours retain degree k.)
        deficiency_levels:  {v: k - deg_in_reduced(v)} for each deficient v.
                            Value is always >= 1.

    Notes on BFS:
        - We perform a standard BFS from `root` collecting all nodes at
          distance <= depth.  Each such node belongs to T.
        - T forms a *tree* in G only if depth <= (g-1)/2; for larger depth a
          "tree" could include vertices at the same BFS level that are adjacent
          in G.  The caller is responsible for choosing depth appropriately.
        - After deletion, deficient vertices are those with at least one deleted
          neighbour (equivalently, those whose degree in reduced_graph is less
          than their degree in G).
    """
    # BFS to collect tree nodes
    tree_nodes: set[int] = set()
    visited: set[int] = set()
    queue: deque[tuple[int, int]] = deque()
    queue.append((root, 0))
    visited.add(root)
    while queue:
        node, dist = queue.popleft()
        tree_nodes.add(node)
        if dist < depth:
            for nb in G.neighbors(node):
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))

    # k = degree of any node in G (assumed k-regular; use max as proxy)
    k = max(d for _, d in G.degree()) if G.number_of_nodes() > 0 else 0

    # Build reduced graph: remove all tree nodes
    reduced: nx.Graph[int] = G.copy()
    reduced.remove_nodes_from(tree_nodes)

    # Find deficient vertices: nodes in reduced whose degree < k
    deficient: list[int] = []
    deficiency_levels: dict[int, int] = {}
    for v in reduced.nodes():
        deg = reduced.degree(v)
        if deg < k:
            deficient.append(v)
            deficiency_levels[v] = k - deg

    deficient.sort()
    return reduced, deficient, deficiency_levels
