import random
from typing import cast

import networkx as nx
import pytest

from ai.cage.cayley.cayley import (
    build_cayley,
    cayley_girth,
    count_short_relations,
    inverse_index_in_set,
    is_symmetric,
    verify_cayley_kg,
)
from ai.cage.cayley.data_gen import cayley_ball_to_pyg
from ai.cage.cayley.groups import (
    cyclic_group,
    dihedral_group,
    element_order,
    involutions,
    random_generating_set,
    symmetric_closure,
)
from ai.cage.cayley.search import candidate_swaps


def test_cayley_group_helpers_detect_orders_and_symmetric_closure() -> None:
    group = cyclic_group(6)

    assert element_order(group, 0) == 1
    assert element_order(group, 2) == 3
    assert involutions(group) == [3]
    assert symmetric_closure(group, [1, 2, 3]) == [1, 5, 2, 4, 3]


def test_random_generating_set_returns_symmetric_degree_k_set() -> None:
    random.seed(1)
    group = dihedral_group(4)

    gens = random_generating_set(group, 3)

    assert gens is not None
    assert len(gens) == 3
    assert len(set(gens)) == 3
    assert is_symmetric(group, gens)
    assert all(gen != group.identity() for gen in gens)


def test_cyclic_cayley_graph_is_cycle_with_expected_girth() -> None:
    group = cyclic_group(5)
    gens = [1, 4]

    graph = build_cayley(group, gens)

    assert graph.number_of_nodes() == 5
    assert graph.number_of_edges() == 5
    assert nx.is_connected(graph)
    assert graph.degree(0) == 2
    assert inverse_index_in_set(group, gens) == [1, 0]
    assert cayley_girth(group, gens, max_girth=8) == 5
    assert count_short_relations(group, gens, max_len=4) == 0
    assert count_short_relations(group, gens, max_len=5) == 2


def test_cayley_graph_rejects_identity_generator() -> None:
    group = cyclic_group(5)

    with pytest.raises(ValueError, match="must not contain the identity"):
        _ = build_cayley(group, [0, 1, 4])

    with pytest.raises(ValueError, match="must not contain the identity"):
        _ = cayley_girth(group, [0, 1, 4])


def test_verify_cayley_kg_reports_valid_and_invalid_generating_sets() -> None:
    group = cyclic_group(5)

    valid = verify_cayley_kg(group, [1, 4], k=2, g_target=5)
    duplicate = verify_cayley_kg(group, [1, 1], k=2, g_target=5)
    asymmetric = verify_cayley_kg(group, [1, 2], k=2, g_target=5)

    assert valid["is_valid_kg"] is True
    assert valid["degree"] == 2
    assert valid["girth"] == 5
    assert duplicate["reason"] == "duplicate generators"
    assert asymmetric["reason"] == "asymmetric generating set"


def test_candidate_swaps_preserve_symmetric_set_size() -> None:
    group = cyclic_group(7)
    current = [1, 6]

    swaps = candidate_swaps(group, current)

    assert swaps
    assert all(len(new_gens) == len(current) for _, _, new_gens in swaps)
    assert all(is_symmetric(group, new_gens) for _, _, new_gens in swaps)
    assert all(0 not in new_gens for _, _, new_gens in swaps)


def test_cayley_ball_to_pyg_has_expected_features_and_context() -> None:
    group = cyclic_group(5)
    data = cayley_ball_to_pyg(group, [1, 4], k=2, g_target=5, girth=5, radius=2)

    assert data.x is not None
    assert data.edge_index is not None
    assert data.edge_attr is not None
    assert data.x.shape[1] == 1
    assert data.edge_index.shape[0] == 2
    assert data.edge_attr.shape[1] == 2
    assert cast(int, data.k) == 2
    assert cast(int, data.g_target) == 5
    assert cast(int, data.group_order) == 5
    assert cast(int, data.num_involutions) == 0
    assert cast(int, data.girth) == 5
    assert cast(int, data.girth) >= cast(int, data.g_target)


