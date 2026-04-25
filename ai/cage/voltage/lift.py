"""Voltage graph lift construction.

Given a base graph B, a finite group Gamma, and a voltage assignment
alpha mapping each undirected edge index to a group element, the *lift*
(also called derived graph or derived cover) is a graph with vertex set
V(B) x Gamma and edges determined mechanically by the base graph topology
plus the voltage labels.

The lifted graph is automatically k-regular if B is k-regular, and its
order is |V(B)| * |Gamma|.
"""

from __future__ import annotations

import networkx as nx

from ai.cage.voltage.base_graphs import BaseGraph
from ai.cage.voltage.groups import FiniteGroup
from backend.utils.graph_utils import compute_girth, is_k_regular


def build_lift(
    base: BaseGraph,
    group: FiniteGroup,
    voltages: list[int],
) -> nx.Graph[int]:
    """Construct the voltage graph lift.

    Args:
        base: The base (multi)graph.
        group: The voltage group Gamma.
        voltages: List of group elements, one per undirected edge of *base*
            (indexed by position in ``base.undirected_edge_ids``).
            The voltage for the reverse arc is automatically ``group.inv(v)``.

    Returns:
        A NetworkX simple graph on |V(B)| * |Gamma| vertices.
        Vertex (v, g) is encoded as ``v * group.order + g``.
    """
    n_base = base.num_nodes
    n_group = group.order
    n_total = n_base * n_group

    G: nx.Graph[int] = nx.Graph()
    G.add_nodes_from(range(n_total))

    # Build a mapping: arc_id -> voltage (group element)
    arc_voltage: dict[int, int] = {}
    for edge_pos, fwd_id in enumerate(base.undirected_edge_ids):
        v = voltages[edge_pos]
        rev_id = base.arcs[fwd_id].reverse_id
        arc_voltage[fwd_id] = v
        arc_voltage[rev_id] = group.inv(v)

    # For each arc (u -> w) with voltage alpha, add edges:
    # (u, g) -- (w, g * alpha) for all g in Gamma
    for arc in base.arcs:
        alpha = arc_voltage[arc.arc_id]
        u = arc.src
        w = arc.dst
        for g in group.elements():
            h = group.mult(g, alpha)
            node_ug = u * n_group + g
            node_wh = w * n_group + h
            if node_ug != node_wh:  # avoid self-loops from degenerate cases
                _ = G.add_edge(node_ug, node_wh)

    return G


def lift_order(base: BaseGraph, group: FiniteGroup) -> int:
    """Number of vertices in the lift."""
    return base.num_nodes * group.order


def verify_lift(
    G: nx.Graph[int],
    k: int,
    g: int,
) -> dict[str, bool | int | float]:
    """Check properties of a lifted graph.

    Returns a dict with:
        - num_nodes, num_edges
        - is_k_regular: True if all nodes have degree k
        - girth: shortest cycle length (inf if acyclic)
        - is_valid_kg: True if k-regular with girth >= g
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    k_reg = is_k_regular(G, k)
    connected = nx.is_connected(G) if num_nodes > 0 else False
    girth = compute_girth(G)

    girth_ok: bool
    if isinstance(girth, float):
        girth_ok = True  # infinite girth (tree) satisfies any girth requirement
    else:
        girth_ok = girth >= g

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "is_k_regular": k_reg,
        "is_connected": connected,
        "girth": girth,
        "is_valid_kg": k_reg and girth_ok and connected,
    }
