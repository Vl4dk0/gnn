"""Base graph representations for voltage graph constructions.

A base graph is a directed multigraph (allowing loops and parallel arcs).
Each undirected edge becomes two opposing directed arcs.  A voltage
assignment maps each directed arc to a group element, with the constraint
that reverse arcs carry inverse voltages.

We therefore only store voltages for *undirected* edges (one per pair of
opposing arcs).  The voltage for the reverse arc is computed automatically.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass(slots=True)
class Arc:
    """A directed arc in the base graph."""

    src: int
    dst: int
    arc_id: int
    reverse_id: int  # arc_id of the opposing arc


@dataclass(slots=True)
class BaseGraph:
    """Directed multigraph used as the base for a voltage graph lift.

    Attributes:
        num_nodes: Number of vertices.
        arcs: All directed arcs (two per undirected edge).
        undirected_edge_ids: List of arc_id values for one representative
            direction of each undirected edge.  Voltage assignments are
            indexed by position in this list.
    """

    num_nodes: int
    arcs: list[Arc] = field(default_factory=list)
    undirected_edge_ids: list[int] = field(default_factory=list)

    def num_undirected_edges(self) -> int:
        return len(self.undirected_edge_ids)

    def add_edge(self, u: int, v: int) -> tuple[int, int]:
        """Add an undirected edge (u, v) as two opposing arcs.

        Returns the arc_ids of the forward and reverse arcs.
        """
        fwd_id = len(self.arcs)
        rev_id = fwd_id + 1
        self.arcs.append(Arc(src=u, dst=v, arc_id=fwd_id, reverse_id=rev_id))
        self.arcs.append(Arc(src=v, dst=u, arc_id=rev_id, reverse_id=fwd_id))
        self.undirected_edge_ids.append(fwd_id)
        return fwd_id, rev_id

    def add_loop(self, v: int) -> tuple[int, int]:
        """Add a loop at vertex v (two arcs, one in each 'direction').

        For loops the two arcs are (v,v) and (v,v) but they are still
        treated as forward/reverse for voltage purposes.
        """
        return self.add_edge(v, v)

    def degree(self, v: int) -> int:
        """Out-degree of vertex v (equals in-degree for undirected base)."""
        return sum(1 for a in self.arcs if a.src == v)

    def is_regular(self, k: int) -> bool:
        return all(self.degree(v) == k for v in range(self.num_nodes))

    def spanning_tree_edge_indices(self) -> list[int]:
        """Return indices into undirected_edge_ids that form a spanning tree.

        Uses BFS from node 0.  Returns positions in undirected_edge_ids,
        NOT arc_ids.  The remaining edges are the "free" edges whose
        voltages actually matter (tree edges can always be set to 0).
        """
        if self.num_nodes <= 1:
            return []

        visited = {0}
        queue: deque[int] = deque([0])
        tree_indices: list[int] = []

        # Build lookup: for each undirected edge index, what nodes does it connect?
        edge_endpoints: list[tuple[int, int]] = []
        for fwd_id in self.undirected_edge_ids:
            arc = self.arcs[fwd_id]
            edge_endpoints.append((arc.src, arc.dst))

        while queue and len(visited) < self.num_nodes:
            node = queue.popleft()
            for idx, (u, v) in enumerate(edge_endpoints):
                if idx in tree_indices:
                    continue
                other = -1
                if u == node and v not in visited:
                    other = v
                elif v == node and u not in visited:
                    other = u
                if other >= 0:
                    visited.add(other)
                    queue.append(other)
                    tree_indices.append(idx)

        return tree_indices

    def free_edge_indices(self) -> list[int]:
        """Return indices of non-tree edges (the ones that need voltages).

        For a connected base graph with n nodes and m edges, there are
        m - (n-1) free edges.  Fixing tree-edge voltages to 0 and only
        searching over free edges gives an equivalent but smaller search.
        """
        tree = set(self.spanning_tree_edge_indices())
        return [i for i in range(self.num_undirected_edges()) if i not in tree]


# ── Known productive base graph topologies ───────────────────────────────


def dumbbell(k: int = 3) -> BaseGraph:
    """Dumbbell / theta graph on 2 nodes with k parallel edges.

    This is the simplest base graph for k-regular voltage lifts.
    The Heawood graph (3,6)-cage is the lift of dumbbell(3) with Z_7
    voltages (0,1,3).

    Note: the Petersen graph (3,5)-cage CANNOT be constructed as a
    dumbbell(3) voltage lift — every 3-element subset of Z_5 contains
    an arithmetic progression, forcing 4-cycles in the lift.
    """
    bg = BaseGraph(num_nodes=2)
    for _ in range(k):
        _ = bg.add_edge(0, 1)
    return bg


def bouquet(k: int) -> BaseGraph:
    """Bouquet (rose) graph: single vertex with k loops.

    Lifts of bouquets are Cayley graphs of the voltage group
    with generating set = {voltage(loop_i), voltage(loop_i)^{-1}}.
    The lift is 2k-regular (each loop contributes degree 2).

    For k-regular lifts with k odd, this doesn't work directly.
    """
    bg = BaseGraph(num_nodes=1)
    for _ in range(k):
        _ = bg.add_loop(0)
    return bg


def dipole(k: int) -> BaseGraph:
    """Dipole graph: 2 nodes with k parallel edges (alias for dumbbell)."""
    return dumbbell(k)


def theta(a: int, b: int, c: int) -> BaseGraph:
    """Theta graph: 2 nodes connected by three paths of lengths a, b, c.

    For a voltage graph lift to be k-regular, we need the base to be
    k-regular.  theta(1,1,1) is the same as dumbbell(3).

    For paths of length > 1, we insert intermediate nodes.
    """
    bg = BaseGraph(num_nodes=2)
    next_node = 2

    for path_len in (a, b, c):
        if path_len == 1:
            _ = bg.add_edge(0, 1)
        else:
            # Path from 0 to 1 through (path_len - 1) intermediate nodes
            prev = 0
            for _step in range(path_len - 1):
                new_node = next_node
                next_node += 1
                bg.num_nodes = max(bg.num_nodes, new_node + 1)
                _ = bg.add_edge(prev, new_node)
                prev = new_node
            _ = bg.add_edge(prev, 1)

    return bg


def heawood_base() -> BaseGraph:
    """The base graph whose Z_7 lift gives the Heawood graph.

    This is dumbbell(3) — two nodes with 3 parallel edges.
    Voltages: 0, 1, 3 in Z_7.
    """
    return dumbbell(3)


def cubic_multigraph_4nodes() -> BaseGraph:
    """A 3-regular multigraph on 4 nodes (K4 minus perfect matching + parallel edges).

    Useful as base graph for trivalent cage constructions with larger groups.
    """
    bg = BaseGraph(num_nodes=4)
    # Form a cycle 0-1-2-3-0
    _ = bg.add_edge(0, 1)
    _ = bg.add_edge(1, 2)
    _ = bg.add_edge(2, 3)
    _ = bg.add_edge(3, 0)
    # Add two crossing edges to make it 3-regular
    _ = bg.add_edge(0, 2)
    _ = bg.add_edge(1, 3)
    return bg


def complete_graph_base(n: int) -> BaseGraph:
    """Complete graph K_n as a base graph.

    The lift is (n-1)*|Gamma|-regular... not directly useful for cages,
    but the incidence graph construction uses substructures of K_n.
    """
    bg = BaseGraph(num_nodes=n)
    for i in range(n):
        for j in range(i + 1, n):
            _ = bg.add_edge(i, j)
    return bg


def prism_base() -> BaseGraph:
    """Prism graph (K3 x K2) on 6 nodes — 3-regular.

    Two triangles connected by a perfect matching.
    """
    bg = BaseGraph(num_nodes=6)
    # Triangle 0-1-2
    _ = bg.add_edge(0, 1)
    _ = bg.add_edge(1, 2)
    _ = bg.add_edge(2, 0)
    # Triangle 3-4-5
    _ = bg.add_edge(3, 4)
    _ = bg.add_edge(4, 5)
    _ = bg.add_edge(5, 3)
    # Matching
    _ = bg.add_edge(0, 3)
    _ = bg.add_edge(1, 4)
    _ = bg.add_edge(2, 5)
    return bg


def moebius_kantor_base() -> BaseGraph:
    """Möbius-Kantor base: the Petersen graph as a base graph (10 nodes, 3-regular).

    The rec(3,16)=960 graph is a lift of the Petersen graph with Z_2 x Z_48
    voltages. This provides a 10-node cubic base for large trivalent lifts.
    """
    bg = BaseGraph(num_nodes=10)
    # Outer 5-cycle: 0-1-2-3-4
    for i in range(5):
        _ = bg.add_edge(i, (i + 1) % 5)
    # Inner pentagram: 5-7-9-6-8 (i.e., 5+0, 5+2, 5+4, 5+1, 5+3)
    inner_order = [5, 7, 9, 6, 8]
    for i in range(5):
        _ = bg.add_edge(inner_order[i], inner_order[(i + 1) % 5])
    # Spokes
    for i in range(5):
        _ = bg.add_edge(i, i + 5)
    return bg
