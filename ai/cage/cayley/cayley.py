"""Cayley graph construction and girth analysis.

Given a finite group G and a symmetric generating set S = S^{-1} not
containing the identity, the Cayley graph Gamma(G, S) has |G| vertices and
is automatically |S|-regular. Vertex-transitivity means the girth equals
the length of the shortest closed non-backtracking walk at any single
vertex — we use the identity. This is the same as the length of the
shortest non-trivial relation in the generators.
"""

from __future__ import annotations

from collections import deque

import networkx as nx

from ai.cage.cayley.groups import FiniteGroup


def inverse_index_in_set(group: FiniteGroup, generators: list[int]) -> list[int]:
    """For each s_i in S, return the index j with S[j] = s_i^{-1}, or -1.

    For symmetric sets the result has no -1 entries.
    """
    out: list[int] = []
    for s in generators:
        sinv = group.inv(s)
        idx = -1
        for j, t in enumerate(generators):
            if t == sinv:
                idx = j
                break
        out.append(idx)
    return out


def is_symmetric(group: FiniteGroup, generators: list[int]) -> bool:
    """True iff S contains the inverse of every element it contains."""
    s_set = set(generators)
    return all(group.inv(s) in s_set for s in generators)


def build_cayley(group: FiniteGroup, generators: list[int]) -> nx.Graph[int]:
    """Construct the Cayley graph Gamma(G, S) as an undirected simple graph.

    Edges are placed between v and v * s for every v in G and s in S. Loops
    (s == identity) are skipped; parallel-edge collapses are absorbed by
    nx.Graph. The caller is responsible for ensuring S is a valid symmetric
    generating set if a connected k-regular graph is required.
    """
    if any(s == 0 for s in generators):
        raise ValueError("generating set must not contain the identity")
    g: nx.Graph[int] = nx.Graph()
    for v in range(group.order):
        _ = g.add_node(v)
    for v in range(group.order):
        for s in generators:
            u = group.mult(v, s)
            if u != v:
                _ = g.add_edge(v, u)
    return g


def cayley_girth(
    group: FiniteGroup,
    generators: list[int],
    max_girth: int = 24,
) -> int | float:
    """Girth of Gamma(G, S) via BFS from identity in the non-backtracking
    cover.

    Returns float('inf') if no closed non-backtracking walk of length <=
    max_girth is found (so the girth is at least max_girth + 1).
    """
    if any(s == 0 for s in generators):
        raise ValueError("generating set must not contain the identity")

    k = len(generators)
    s_inv_idx = inverse_index_in_set(group, generators)

    # State: (current_vertex, last_generator_index). last = -1 at the start.
    visited: dict[tuple[int, int], int] = {(0, -1): 0}
    queue: deque[tuple[int, int, int]] = deque([(0, -1, 0)])

    while queue:
        v, last, d = queue.popleft()
        if d >= max_girth:
            continue
        for i in range(k):
            if last >= 0 and s_inv_idx[last] == i:
                continue  # backtracking on the same undirected edge
            new_v = group.mult(v, generators[i])
            new_d = d + 1
            if new_v == 0:
                # First close-back to identity in BFS order = shortest relation.
                # new_d cannot be 1 (S has no identity) or 2 (immediate
                # backtrack would have been skipped above), so new_d >= 3.
                return new_d
            key = (new_v, i)
            if key not in visited:
                visited[key] = new_d
                queue.append((new_v, i, new_d))

    return float("inf")


def count_short_relations(
    group: FiniteGroup,
    generators: list[int],
    max_len: int,
) -> int:
    """Count closed non-backtracking walks at the identity of length
    in [3, max_len].

    Used as a continuous-ish cost function for tabu search: reaching 0
    means the Cayley graph has girth >= max_len + 1.
    """
    if max_len < 3:
        return 0

    k = len(generators)
    s_inv_idx = inverse_index_in_set(group, generators)

    # dp maps (vertex, last_gen_idx) -> number of non-backtracking walks
    # of the current length ending there.
    dp: dict[tuple[int, int], int] = {(0, -1): 1}
    total = 0

    for step in range(1, max_len + 1):
        new_dp: dict[tuple[int, int], int] = {}
        for (v, last), cnt in dp.items():
            for i in range(k):
                if last >= 0 and s_inv_idx[last] == i:
                    continue
                new_v = group.mult(v, generators[i])
                key = (new_v, i)
                new_dp[key] = new_dp.get(key, 0) + cnt
        dp = new_dp
        if step >= 3:
            for (v, _last), cnt in dp.items():
                if v == 0:
                    total += cnt

    return total


def verify_cayley_kg(
    group: FiniteGroup,
    generators: list[int],
    k: int,
    g_target: int,
) -> dict[str, object]:
    """Sanity-check a Cayley graph candidate against the (k,g) requirements.

    Checks degree, connectedness, simplicity, and computed girth.
    """
    if len(set(generators)) != len(generators):
        return {
            "is_valid_kg": False,
            "reason": "duplicate generators",
            "degree": len(set(generators)),
            "girth": 0,
            "is_connected": False,
        }
    if not is_symmetric(group, generators):
        return {
            "is_valid_kg": False,
            "reason": "asymmetric generating set",
            "degree": len(generators),
            "girth": 0,
            "is_connected": False,
        }
    if any(s == 0 for s in generators):
        return {
            "is_valid_kg": False,
            "reason": "identity in generating set",
            "degree": len(generators),
            "girth": 0,
            "is_connected": False,
        }

    cay = build_cayley(group, generators)
    # All vertices have the same degree by construction; spot-check.
    degree = cay.degree(0)
    if degree != k:
        return {
            "is_valid_kg": False,
            "reason": f"degree {degree} != k {k}",
            "degree": degree,
            "girth": 0,
            "is_connected": False,
        }
    connected = nx.is_connected(cay)
    girth_val = cayley_girth(group, generators, max_girth=max(2 * g_target, 24))
    girth_ok = isinstance(girth_val, int) and girth_val >= g_target

    return {
        "is_valid_kg": bool(connected and girth_ok),
        "reason": "" if (connected and girth_ok) else "girth or connectivity",
        "degree": degree,
        "girth": girth_val,
        "is_connected": bool(connected),
        "order": group.order,
    }


__all__ = [
    "build_cayley",
    "cayley_girth",
    "count_short_relations",
    "inverse_index_in_set",
    "is_symmetric",
    "verify_cayley_kg",
]
