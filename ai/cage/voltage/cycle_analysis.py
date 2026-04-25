"""Fast girth computation for voltage graph lifts via net voltage analysis.

The girth of a voltage graph lift equals the length of the shortest closed
walk in the base graph whose *net voltage* (product of edge voltages along
the walk) equals the group identity.

This avoids constructing the full lift (which can have thousands of nodes)
and instead only enumerates short closed walks in the tiny base graph.
"""

from __future__ import annotations

import math

from ai.cage.voltage.base_graphs import BaseGraph
from ai.cage.voltage.groups import FiniteGroup


def compute_lift_girth(
    base: BaseGraph,
    group: FiniteGroup,
    voltages: list[int],
    max_girth: int = 50,
) -> int | float:
    """Compute the girth of the voltage graph lift without building it.

    Enumerates all closed walks in the base graph up to length *max_girth*
    and checks if any has net voltage equal to the identity.

    Args:
        base: The base (multi)graph.
        group: The voltage group Gamma.
        voltages: Voltage assignment (one group element per undirected edge).
        max_girth: Maximum walk length to check.  If no identity-voltage
            walk is found up to this length, returns ``max_girth + 1``
            (a lower bound on the girth).

    Returns:
        The girth of the lift, or ``math.inf`` if no closed walk with
        identity net voltage exists up to *max_girth*.
    """
    # Build arc-level voltage lookup
    arc_voltage: dict[int, int] = {}
    for edge_pos, fwd_id in enumerate(base.undirected_edge_ids):
        v = voltages[edge_pos]
        rev_id = base.arcs[fwd_id].reverse_id
        arc_voltage[fwd_id] = v
        arc_voltage[rev_id] = group.inv(v)

    # Build adjacency: node -> list of (neighbor, arc_id)
    adj: dict[int, list[tuple[int, int]]] = {v: [] for v in range(base.num_nodes)}
    for arc in base.arcs:
        adj[arc.src].append((arc.dst, arc.arc_id))

    # Precompute which arcs produce the same lift edge.  Two arcs map to
    # the same set of lift edges when they share (src, dst, voltage).
    # After traversing arc a, we must skip any arc a' that would retrace
    # the same lift edge — not just the direct reverse of a.
    # The "reverse lift edge" of arc (u->v, voltage α) is any arc
    # (v->u, voltage inv(α)), i.e. same (src=v, dst=u, voltage=inv(α)).
    arc_signature: dict[int, tuple[int, int, int]] = {}
    for arc in base.arcs:
        arc_signature[arc.arc_id] = (arc.src, arc.dst, arc_voltage[arc.arc_id])

    def _would_backtrack(prev_arc_id: int, next_arc_id: int) -> bool:
        """True if next_arc retraces the same lift edge as prev_arc."""
        # The reverse lift-edge of prev has signature (dst, src, inv(voltage))
        p = base.arcs[prev_arc_id]
        prev_v = arc_voltage[prev_arc_id]
        reverse_sig = (p.dst, p.src, group.inv(prev_v))
        return arc_signature[next_arc_id] == reverse_sig

    identity = group.identity()
    best_girth: int | float = math.inf

    for start in range(base.num_nodes):
        frontier: list[tuple[int, int, int, int]] = []

        for neighbor, arc_id in adj[start]:
            v = arc_voltage[arc_id]
            frontier.append((neighbor, v, 1, arc_id))

        for _depth in range(1, max_girth + 1):
            if not frontier:
                break

            next_frontier: list[tuple[int, int, int, int]] = []
            for node, net_v, length, prev_arc in frontier:
                if length >= best_girth:
                    continue

                if node == start and net_v == identity and length >= 3:
                    best_girth = min(best_girth, length)
                    continue

                if length + 1 >= best_girth:
                    continue

                for next_node, arc_id in adj[node]:
                    if _would_backtrack(prev_arc, arc_id):
                        continue
                    new_v = group.mult(net_v, arc_voltage[arc_id])
                    next_frontier.append((next_node, new_v, length + 1, arc_id))

            frontier = next_frontier

    return best_girth


def has_girth_at_least(
    base: BaseGraph,
    group: FiniteGroup,
    voltages: list[int],
    g_target: int,
) -> bool:
    """Check whether the lift has girth >= g_target.

    This is more efficient than computing the exact girth because we can
    stop as soon as we find any closed walk shorter than g_target with
    identity net voltage.
    """
    girth = compute_lift_girth(base, group, voltages, max_girth=g_target)
    if isinstance(girth, float):
        return True  # inf means no short cycles found
    return girth >= g_target


def count_short_identity_walks(
    base: BaseGraph,
    group: FiniteGroup,
    voltages: list[int],
    g_target: int,
) -> int:
    """Count closed walks shorter than g_target with identity net voltage.

    This is the cost function for tabu search: the goal is to reach 0.
    A cost of 0 means girth >= g_target.
    """
    arc_voltage: dict[int, int] = {}
    for edge_pos, fwd_id in enumerate(base.undirected_edge_ids):
        v = voltages[edge_pos]
        rev_id = base.arcs[fwd_id].reverse_id
        arc_voltage[fwd_id] = v
        arc_voltage[rev_id] = group.inv(v)

    adj: dict[int, list[tuple[int, int]]] = {v: [] for v in range(base.num_nodes)}
    for arc in base.arcs:
        adj[arc.src].append((arc.dst, arc.arc_id))

    arc_signature: dict[int, tuple[int, int, int]] = {}
    for arc in base.arcs:
        arc_signature[arc.arc_id] = (arc.src, arc.dst, arc_voltage[arc.arc_id])

    def _would_backtrack(prev_arc_id: int, next_arc_id: int) -> bool:
        p = base.arcs[prev_arc_id]
        prev_v = arc_voltage[prev_arc_id]
        reverse_sig = (p.dst, p.src, group.inv(prev_v))
        return arc_signature[next_arc_id] == reverse_sig

    identity = group.identity()
    count = 0

    for start in range(base.num_nodes):
        frontier: list[tuple[int, int, int, int]] = []
        for neighbor, arc_id in adj[start]:
            v = arc_voltage[arc_id]
            frontier.append((neighbor, v, 1, arc_id))

        for _depth in range(1, g_target):
            if not frontier:
                break
            next_frontier: list[tuple[int, int, int, int]] = []
            for node, net_v, length, prev_arc in frontier:
                if node == start and net_v == identity and length >= 3:
                    count += 1
                    continue
                for next_node, arc_id in adj[node]:
                    if _would_backtrack(prev_arc, arc_id):
                        continue
                    new_v = group.mult(net_v, arc_voltage[arc_id])
                    next_frontier.append((next_node, new_v, length + 1, arc_id))
            frontier = next_frontier

    return count
