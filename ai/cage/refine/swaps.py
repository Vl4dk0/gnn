"""2-switch and 3-switch operations on k-regular graphs.

A **2-switch** is a degree-preserving edge swap:
    Given independent edges (u,v) and (x,y) with {u,v,x,y} pairwise distinct,
    replace them with either (u,x),(v,y) [mode "uxvy"]
                          or (u,y),(v,x) [mode "uyvx"].
    The graph remains k-regular because each vertex loses exactly one edge
    and gains exactly one edge.

A **3-switch** involves three independent edges (u,v), (x,y), (p,q) with
    all six endpoints distinct.  Remove all three; add the three edges
    forming the other perfect matching on {u,v,x,y,p,q} (there are exactly
    three non-trivial perfect matchings; we try both alternatives).
    Degree is preserved: each vertex loses exactly one edge and gains one.

Complexity notes:
  enumerate_2_switches — O(|E|^2) worst-case, O(sample_size) when capped.
  enumerate_3_switches — O(|E|^3) worst-case, O(sample_size) when capped.
"""

from __future__ import annotations

import random
from typing import Iterator, Literal, NamedTuple

import networkx as nx


class Swap2(NamedTuple):
    """A 2-switch: swap edges (u,v) and (x,y)."""

    u: int
    v: int
    x: int
    y: int
    mode: Literal["uxvy", "uyvx"]


class Swap3(NamedTuple):
    """A 3-switch: rotate three independent edges.

    Edges removed: (u,v), (x,y), (p,q).
    mode "abc": add (u,x),(v,p),(y,q)
    mode "acb": add (u,x),(v,q),(y,p)
    """

    u: int
    v: int
    x: int
    y: int
    p: int
    q: int
    mode: Literal["abc", "acb"]


def _edge_key(u: int, v: int) -> tuple[int, int]:
    return (min(u, v), max(u, v))


def _edges_list(G: nx.Graph[int]) -> list[tuple[int, int]]:
    """Return a canonical list of undirected edges with u < v."""
    return [_edge_key(u, v) for u, v in G.edges()]


def enumerate_2_switches(
    G: nx.Graph[int],
    sample_size: int | None = None,
) -> Iterator[Swap2]:
    """Yield valid 2-switch candidates for G.

    A candidate (u,v,x,y) is valid when:
      - All four endpoints are distinct.
      - The replacement edges do not already exist in G.

    Both modes ("uxvy" and "uyvx") are yielded independently.

    If *sample_size* is given, only a random subset of edge-pairs is examined
    (best-effort), keeping the function O(sample_size).
    """
    edges = _edges_list(G)
    n_edges = len(edges)
    edge_set: set[tuple[int, int]] = set(edges)

    if sample_size is not None:
        tried: set[tuple[int, int]] = set()
        attempts = 0
        max_attempts = min(sample_size * 6, n_edges * (n_edges - 1))
        while len(tried) < sample_size and attempts < max_attempts:
            attempts += 1
            i = random.randrange(n_edges)
            j = random.randrange(n_edges)
            if i == j:
                continue
            key = (min(i, j), max(i, j))
            if key in tried:
                continue
            tried.add(key)
            u, v = edges[i]
            x, y = edges[j]
            if len({u, v, x, y}) < 4:
                continue
            if _edge_key(u, x) not in edge_set and _edge_key(v, y) not in edge_set:
                yield Swap2(u, v, x, y, "uxvy")
            if _edge_key(u, y) not in edge_set and _edge_key(v, x) not in edge_set:
                yield Swap2(u, v, x, y, "uyvx")
        return

    # Full enumeration
    for i in range(n_edges):
        u, v = edges[i]
        for j in range(i + 1, n_edges):
            x, y = edges[j]
            if len({u, v, x, y}) < 4:
                continue
            if _edge_key(u, x) not in edge_set and _edge_key(v, y) not in edge_set:
                yield Swap2(u, v, x, y, "uxvy")
            if _edge_key(u, y) not in edge_set and _edge_key(v, x) not in edge_set:
                yield Swap2(u, v, x, y, "uyvx")


def apply_2_switch(G: nx.Graph[int], swap: Swap2) -> nx.Graph[int]:
    """Return a new graph obtained by applying *swap* to G.

    The original graph is not modified.  Preserves k-regularity: each of the
    four endpoints loses one edge and gains one edge.
    """
    H: nx.Graph[int] = G.copy()
    u, v, x, y, mode = swap
    H.remove_edge(u, v)
    H.remove_edge(x, y)
    if mode == "uxvy":
        H.add_edge(u, x)
        H.add_edge(v, y)
    else:
        H.add_edge(u, y)
        H.add_edge(v, x)
    return H


def enumerate_3_switches(
    G: nx.Graph[int],
    sample_size: int | None = None,
) -> Iterator[Swap3]:
    """Yield valid 3-switch candidates for G.

    A 3-switch selects three independent edges with all 6 endpoints distinct.
    We remove all three and add the edges of one of the two alternative perfect
    matchings on those 6 vertices.

    Given edges (u,v),(x,y),(p,q), the two alternatives to the original
    matching are:
        "abc": (u,x),(v,p),(y,q)
        "acb": (u,x),(v,q),(y,p)

    If *sample_size* is given, only a random subset of edge-triples is examined.
    """
    edges = _edges_list(G)
    n_edges = len(edges)
    edge_set: set[tuple[int, int]] = set(edges)

    def _try_triple(u: int, v: int, x: int, y: int, p: int, q: int) -> Iterator[Swap3]:
        # mode "abc": add (u,x),(v,p),(y,q)
        if (
            _edge_key(u, x) not in edge_set
            and _edge_key(v, p) not in edge_set
            and _edge_key(y, q) not in edge_set
        ):
            yield Swap3(u, v, x, y, p, q, "abc")
        # mode "acb": add (u,x),(v,q),(y,p)
        if (
            _edge_key(u, x) not in edge_set
            and _edge_key(v, q) not in edge_set
            and _edge_key(y, p) not in edge_set
        ):
            yield Swap3(u, v, x, y, p, q, "acb")

    if sample_size is not None:
        count = 0
        attempts = 0
        max_attempts = sample_size * 10
        while count < sample_size and attempts < max_attempts:
            attempts += 1
            indices = random.sample(range(n_edges), min(3, n_edges))
            if len(indices) < 3:
                break
            i, j, k_ = indices
            u, v = edges[i]
            x, y = edges[j]
            p, q = edges[k_]
            if len({u, v, x, y, p, q}) < 6:
                continue
            for swap3 in _try_triple(u, v, x, y, p, q):
                yield swap3
                count += 1
        return

    # Full enumeration
    for i in range(n_edges):
        u, v = edges[i]
        for j in range(i + 1, n_edges):
            x, y = edges[j]
            if len({u, v, x, y}) < 4:
                continue
            for k_ in range(j + 1, n_edges):
                p, q = edges[k_]
                if len({u, v, x, y, p, q}) < 6:
                    continue
                yield from _try_triple(u, v, x, y, p, q)


def apply_3_switch(G: nx.Graph[int], swap: Swap3) -> nx.Graph[int]:
    """Return a new graph obtained by applying *swap* to G.

    Removes edges (u,v), (x,y), (p,q) and adds the three replacement edges
    determined by *swap.mode*.  The original graph is not modified.

    Degree is preserved: each of the 6 endpoints loses exactly one edge and
    gains exactly one edge.
    """
    H: nx.Graph[int] = G.copy()
    u, v, x, y, p, q, mode = swap
    H.remove_edge(u, v)
    H.remove_edge(x, y)
    H.remove_edge(p, q)
    if mode == "abc":
        H.add_edge(u, x)
        H.add_edge(v, p)
        H.add_edge(y, q)
    else:  # "acb"
        H.add_edge(u, x)
        H.add_edge(v, q)
        H.add_edge(y, p)
    return H
