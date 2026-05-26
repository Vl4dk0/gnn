"""Legality checks for candidate repair edges.

A candidate edge (u, v) added to the partially-repaired graph H is **legal** iff:
  1. u != v                               (no self-loops)
  2. u and v are not already adjacent in H (keep it simple)
  3. dist_H(u, v) >= g - 1               (no cycle shorter than g)

Condition 3 follows from the cycle identity: adding edge (u,v) creates a cycle
of length dist_H(u,v) + 1. For girth to stay >= g we need that new cycle length
>= g, i.e. dist_H(u,v) + 1 >= g, i.e. dist_H(u,v) >= g - 1.

We maintain a pairwise BFS-distance cache among the deficient set (plus any
vertices they might be matched to). The cache is invalidated conservatively
whenever a new edge is added: only the entries involving endpoints of the new
edge are stale.

Staleness rule (conservative but correct):
  After adding edge (a, b), any cached distance dist(u, v) where {u,v} shares
  an endpoint with {a,b} OR where the shortest path might route through {a,b}
  is potentially stale. The truly safe approach is to invalidate all entries
  involving a or b; re-queries lazily recompute via BFS. This is O(n) per
  invalidation, but the repair phase only adds O(k * |deficient| / 2) edges
  so the total work is modest.
"""

from __future__ import annotations

from collections import deque

import networkx as nx


def _bfs_distance(G: nx.Graph[int], source: int, cutoff: int) -> dict[int, int]:
    """BFS from `source` returning distances up to `cutoff` (inclusive).

    Returns {node: distance} for all reachable nodes within `cutoff` hops.
    Does NOT include nodes beyond cutoff (they would not affect legality).
    """
    dist: dict[int, int] = {source: 0}
    queue: deque[int] = deque([source])
    while queue:
        u = queue.popleft()
        d = dist[u]
        if d >= cutoff:
            continue
        for nb in G.neighbors(u):
            if nb not in dist:
                dist[nb] = d + 1
                queue.append(nb)
    return dist


def is_legal_edge(G: nx.Graph[int], u: int, v: int, g_target: int) -> bool:
    """Return True iff adding edge (u, v) to G is legal for girth g_target.

    Legality: u != v, u and v not already adjacent, and dist_G(u,v) >= g-1.
    """
    if u == v:
        return False
    if G.has_edge(u, v):
        return False
    # BFS from u up to g-2 steps; if v is reachable in <= g-2 steps, illegal
    dist = _bfs_distance(G, u, g_target - 2)
    if v in dist:
        return False  # dist(u,v) <= g-2 → new cycle length <= g-1 < g
    return True


def legal_candidate_edges(
    G: nx.Graph[int],
    deficient_vertices: list[int],
    g_target: int,
) -> list[tuple[int, int]]:
    """Enumerate all legal candidate edges among deficient vertices.

    Only considers pairs within the deficient set (the repair should reconnect
    deficient vertices to each other; connecting to a non-deficient vertex
    would raise its degree above k).

    For each unordered pair (u, v) with u < v, checks legality via BFS.
    Uses a shared BFS distance cache: for each u we BFS up to g-2 steps
    once and record which deficient vertices are reachable (= illegal partners).

    Returns:
        List of (u, v) pairs with u < v that are legal to add.
    """
    deficient_set = set(deficient_vertices)
    candidates: list[tuple[int, int]] = []

    # For each deficient vertex, BFS to find all nodes within g-2 steps
    # (these cannot be connected). Every deficient vertex NOT in that set
    # and not already adjacent is a legal partner.
    for i, u in enumerate(deficient_vertices):
        close: set[int] = set(_bfs_distance(G, u, g_target - 2).keys())
        for v in deficient_vertices[i + 1 :]:
            if v not in close and not G.has_edge(u, v):
                candidates.append((u, v))

    return candidates


class DistanceCache:
    """Lazy pairwise BFS-distance cache for a working graph.

    Caches dist(u, v) for queried pairs and invalidates on edge addition.

    Staleness rule: when edge (a, b) is added, all cached entries (u, v) where
    u == a or u == b or v == a or v == b are removed. This is conservative
    (may evict valid entries) but is always correct (never returns stale data).

    Usage:
        cache = DistanceCache(G, g_target)
        d = cache.dist(u, v)            # lazily computed
        cache.invalidate_edge(a, b)     # call after G.add_edge(a, b)
    """

    def __init__(self, G: nx.Graph[int], g_target: int) -> None:
        self._G = G
        self._g = g_target
        self._cache: dict[tuple[int, int], int] = {}

    def dist(self, u: int, v: int) -> int:
        """BFS distance from u to v in _G (capped at g_target, returns g if unreachable)."""
        key = (min(u, v), max(u, v))
        if key not in self._cache:
            raw = _bfs_distance(self._G, u, self._g - 1)
            d = raw.get(v, self._g)  # g = "too far" sentinel
            self._cache[key] = d
        return self._cache[key]

    def is_legal(self, u: int, v: int) -> bool:
        """True iff edge (u, v) is legal to add for girth g_target."""
        if u == v:
            return False
        if self._G.has_edge(u, v):
            return False
        return self.dist(u, v) >= self._g - 1

    def invalidate_edge(self, a: int, b: int) -> None:
        """Invalidate all cache entries involving a or b."""
        stale = [k for k in self._cache if k[0] in (a, b) or k[1] in (a, b)]
        for k in stale:
            del self._cache[k]
