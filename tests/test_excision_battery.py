"""Feasibility tests for the excision benchmark battery.

The excision benchmark previously reported 0% success because its source
graphs sat at or near the Moore bound, so a depth-(g-1)//2 BFS-tree excision
removed nearly the whole graph and left nothing repairable.  These tests guard
the invariant the battery now relies on: at least one instance must be a valid
(k,g)-graph whose depth-(g-1)//2 excision from some root leaves a non-empty
reduced graph that the classical repair can rebuild into a k-regular,
girth>=g graph.
"""

from __future__ import annotations

import networkx as nx

from ai.cage.excision.baseline import backtracking_repair, greedy_match_repair
from ai.cage.excision.excise import excise_tree
from backend.utils.graph_utils import compute_girth, is_k_regular
from results.battery import excision_instances


def _classical_repair(
    reduced: nx.Graph[int], deficient: list[int], g: int, k: int
) -> nx.Graph[int] | None:
    out = greedy_match_repair(reduced, deficient, g, k)
    if out is None:
        out = backtracking_repair(reduced, deficient, g, k, max_backtracks=10000)
    return out


def test_dodecahedral_instance_is_repairable() -> None:
    """The (3,5) anchor: dodecahedral, depth-2 excision, classical repair restores it."""
    instances = excision_instances(quick=True)
    by_name = {name: (k, g, graph) for name, k, g, graph in instances}
    assert "dodecahedral" in by_name, "dodecahedral must be in the quick battery"

    k, g, graph = by_name["dodecahedral"]

    # It is a genuine (k,g)-graph.
    assert is_k_regular(graph, k)
    assert compute_girth(graph) >= g

    depth = (g - 1) // 2
    reduced, deficient, _ = excise_tree(graph, 0, depth)

    # Depth-(g-1)//2 excision leaves a non-trivial boundary, not the whole graph.
    assert reduced.number_of_nodes() > 0
    assert len(deficient) > 0

    repaired = _classical_repair(reduced, deficient, g, k)
    assert repaired is not None
    assert is_k_regular(repaired, k)
    assert compute_girth(repaired) >= g


def test_some_instance_repairable_from_some_root() -> None:
    """At least one battery instance is repairable from at least one root."""
    for _name, k, g, graph in excision_instances(quick=True):
        assert is_k_regular(graph, k)
        assert compute_girth(graph) >= g
        depth = (g - 1) // 2
        for root in sorted(graph.nodes()):
            reduced, deficient, _ = excise_tree(graph, root, depth)
            if reduced.number_of_nodes() == 0 or not deficient:
                continue
            repaired = _classical_repair(reduced, deficient, g, k)
            if (
                repaired is not None
                and is_k_regular(repaired, k)
                and compute_girth(repaired) >= g
            ):
                return  # found a feasible (instance, root): invariant holds
    raise AssertionError("no battery instance was repairable from any root")
