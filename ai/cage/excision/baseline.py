"""Classical baseline repair algorithms for tree excision.

After BFS-excision a subset of boundary vertices is deficient (degree < k).
These algorithms try to restore full k-regularity by adding edges among the
deficient set without introducing cycles shorter than g_target.

Two variants:
  greedy_match_repair  — one-pass greedy with least-constrained-partner heuristic;
                         no backtracking (fast, may fail).
  backtracking_repair  — DFS with the same heuristic and a backtrack cap
                         (slower, higher success rate; this is the classical
                          baseline the GNN must beat in success rate).

Heuristic (minimum remaining values):
  At each step, pick the deficient vertex u with the fewest legal partners
  (most constrained). Among u's legal partners, pick the one with the fewest
  legal partners excluding u (least constrained partner — breaks ties, avoids
  corner-casing the next step).

Both functions operate on a COPY of the input graph and do not modify it.
"""

from __future__ import annotations

import copy

import networkx as nx

from ai.cage.excision.legality import DistanceCache


def _count_legal_partners(
    G: nx.Graph[int],
    v: int,
    deficient_set: set[int],
    cache: DistanceCache,
) -> int:
    """Count legal partners for v within deficient_set (excluding v itself)."""
    return sum(1 for u in deficient_set if u != v and cache.is_legal(u, v))


def greedy_match_repair(
    G: nx.Graph[int],
    deficient: list[int],
    g_target: int,
    k: int,
) -> nx.Graph[int] | None:
    """Greedy one-pass repair: pick most-constrained vertex, match to best partner.

    `k` is the target regularity (degree) of the final graph — the k of the
    original (k, g)-graph being excised, not inferred from the reduced graph.

    No backtracking. Returns the repaired graph (copy) or None on failure.

    The graph returned is always simple (no multi-edges, no self-loops).
    """
    H: nx.Graph[int] = G.copy()
    deficiency: dict[int, int] = {}
    for v in deficient:
        deficiency[v] = k - H.degree(v)

    cache = DistanceCache(H, g_target)

    while any(d > 0 for d in deficiency.values()):
        deficient_now = [v for v, d in deficiency.items() if d > 0]
        if not deficient_now:
            break

        # Pick most-constrained vertex
        best_u: int | None = None
        best_u_count = -1
        for u in deficient_now:
            cnt = _count_legal_partners(H, u, set(deficient_now), cache)
            if best_u is None or cnt < best_u_count:
                best_u = u
                best_u_count = cnt

        if best_u is None or best_u_count == 0:
            return None  # stuck

        u = best_u
        partners = [v for v in deficient_now if v != u and cache.is_legal(u, v)]
        if not partners:
            return None

        # Least-constrained partner
        partners.sort(
            key=lambda v: _count_legal_partners(H, v, set(deficient_now) - {u}, cache),
            reverse=True,
        )
        chosen = partners[0]

        H.add_edge(u, chosen)
        cache.invalidate_edge(u, chosen)
        deficiency[u] -= 1
        deficiency[chosen] -= 1

    return H


def backtracking_repair(
    G: nx.Graph[int],
    deficient: list[int],
    g_target: int,
    k: int,
    max_backtracks: int = 10000,
) -> nx.Graph[int] | None:
    """DFS repair with backtracking and MRV heuristic.

    `k` is the target regularity (degree) of the final graph — the k of the
    original (k, g)-graph being excised, not inferred from the reduced graph.

    Maintains a stack of (graph_snapshot, deficiency_snapshot, cache_snapshot)
    triples so that failed branches can be undone.

    Returns the repaired graph (copy) or None if all branches exhausted or
    max_backtracks exceeded.

    The graph returned is always simple (no multi-edges, no self-loops).
    """

    def _initial_deficiency(H: nx.Graph[int]) -> dict[int, int]:
        return {v: k - H.degree(v) for v in deficient if k - H.degree(v) > 0}

    # Stack entry: (H_copy, deficiency_dict)
    # We avoid pickling the DistanceCache; just rebuild it from H on each visit.
    stack: list[tuple[nx.Graph[int], dict[int, int], list[tuple[int, int]]]] = [
        (G.copy(), _initial_deficiency(G), [])
    ]
    backtracks = 0

    while stack:
        H, deficiency, tried = stack.pop()

        if not deficiency:
            return H

        # Rebuild cache from current graph state
        cache = DistanceCache(H, g_target)

        deficient_now = [v for v, d in deficiency.items() if d > 0]

        # Most-constrained vertex
        best_u: int | None = None
        best_u_count = -1
        for u in deficient_now:
            cnt = _count_legal_partners(H, u, set(deficient_now), cache)
            if best_u is None or cnt < best_u_count:
                best_u = u
                best_u_count = cnt

        if best_u is None or best_u_count == 0:
            backtracks += 1
            if backtracks > max_backtracks:
                return None
            continue

        u = best_u
        partners = [v for v in deficient_now if v != u and cache.is_legal(u, v)]
        # Sort: least-constrained partner first (best to try first)
        partners.sort(
            key=lambda v: _count_legal_partners(H, v, set(deficient_now) - {u}, cache),
            reverse=True,
        )

        for partner in partners:
            if (u, partner) in tried or (partner, u) in tried:
                continue

            H2: nx.Graph[int] = H.copy()
            H2.add_edge(u, partner)
            def2 = copy.copy(deficiency)
            def2[u] -= 1
            if def2[u] == 0:
                del def2[u]
            def2[partner] -= 1
            if def2[partner] == 0:
                del def2[partner]

            stack.append((H2, def2, []))

        backtracks += 1
        if backtracks > max_backtracks:
            return None

    return None
