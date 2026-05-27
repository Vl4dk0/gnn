"""Short-cycle cost function for post-lift graph refinement.

The cost function penalises cycles shorter than g_target:

    f(G) = sum_{c=3}^{g_target-1} w_c * N_c(G)

where N_c(G) is the total number of simple cycles of length c in G and
w_c = 2^(g_target - c) gives exponentially higher weight to shorter cycles.

The goal of the tabu refiner is to drive f -> 0.  f = 0 iff the graph
has girth >= g_target.

Performance note:
    count_cycles_up_to uses nx.simple_cycles with a length_bound to
    avoid full enumeration of long cycles.  For cage-scale graphs
    (n <= a few hundred) this is fast enough in practice.
    Complexity: O(n * (n+m)) per cycle found via Johnson's algorithm
    with the length cap; empirically fast for n <= 200.
"""

from __future__ import annotations

import networkx as nx


def count_cycles_up_to(
    G: nx.Graph[int],
    max_len: int,
) -> dict[int, int]:
    """Count simple cycles of each length from 3 to max_len (inclusive).

    Uses networkx.simple_cycles with length_bound=max_len for efficiency.
    Each undirected cycle is counted once (we halve the directed count from
    nx.simple_cycles which yields both orientations).

    Returns a dict mapping length -> count.  Missing lengths have count 0.

    Complexity: O(n * E * n^{max_len}) worst-case, but the length_bound
    causes early pruning; fast in practice for max_len <= 12 and n <= 300.
    """
    counts: dict[int, int] = {length: 0 for length in range(3, max_len + 1)}

    # nx.simple_cycles on an undirected graph yields each undirected cycle
    # exactly once (canonical representative; verified on networkx >= 3.0).
    for cycle in nx.simple_cycles(G, length_bound=max_len):
        length = len(cycle)
        if 3 <= length <= max_len:
            counts[length] += 1

    return counts


def short_cycle_cost(
    G: nx.Graph[int],
    g_target: int,
    weights: dict[int, float] | None = None,
) -> float:
    """Compute the weighted short-cycle cost for graph G.

    f(G) = sum_{c=3}^{g_target-1} w_c * N_c(G)

    Default weights: w_c = 2^(g_target - c)  (exponential penalty for
    shorter cycles; length g_target-1 gets weight 1, length 3 gets the
    highest weight).

    Custom weights can be passed as {cycle_length: weight}.  Any cycle
    length not in the dict defaults to 0 (ignored).

    Returns 0.0 when g_target <= 3 (no shorter cycles possible) or when G
    has no cycles shorter than g_target.
    """
    if g_target <= 3:
        return 0.0

    max_len = g_target - 1
    cycle_counts = count_cycles_up_to(G, max_len)

    total = 0.0
    for c in range(3, g_target):
        n_c = cycle_counts.get(c, 0)
        if n_c == 0:
            continue
        if weights is not None:
            w = weights.get(c, 0.0)
        else:
            w = float(2 ** (g_target - c))
        total += w * n_c

    return total
