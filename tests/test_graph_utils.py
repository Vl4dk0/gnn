import networkx as nx
import pytest

from backend.utils.graph_utils import (
    can_add_edge_preserving_girth,
    compute_girth,
    compute_regularity_score,
    get_nodes_by_degree,
    graph_to_edge_list,
    is_k_regular,
    is_valid_cage,
    moore_bound,
    moore_hoffman_upper_bound,
    parse_edge_list,
    score_graph_quality,
)


def _path_graph(n: int) -> nx.Graph[int]:
    graph: nx.Graph[int] = nx.Graph()
    graph.add_nodes_from(range(n))
    for node in range(n - 1):
        _ = graph.add_edge(node, node + 1)
    return graph


def _cycle_graph(n: int) -> nx.Graph[int]:
    graph = _path_graph(n)
    if n > 2:
        _ = graph.add_edge(n - 1, 0)
    return graph


def _petersen_graph() -> nx.Graph[int]:
    graph: nx.Graph[int] = nx.Graph()
    graph.add_nodes_from(range(10))
    graph.add_edges_from(
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 0),
            (5, 7),
            (7, 9),
            (9, 6),
            (6, 8),
            (8, 5),
            (0, 5),
            (1, 6),
            (2, 7),
            (3, 8),
            (4, 9),
        ]
    )
    return graph


def test_parse_and_serialize_edge_list_preserves_isolated_vertices_and_self_loops() -> None:
    graph = parse_edge_list("3\n2 0\n1 1\n0 3")

    assert sorted(graph.nodes()) == [0, 1, 2, 3]
    assert graph.has_edge(0, 2)
    assert graph.has_edge(0, 3)
    assert graph.has_edge(1, 1)
    assert graph_to_edge_list(graph) == "0 2\n0 3\n1 1"


def test_parse_edge_list_reports_line_number_for_bad_input() -> None:
    with pytest.raises(ValueError, match="Line 2"):
        _ = parse_edge_list("0 1\nnot-a-node")


def test_compute_girth_handles_empty_graph() -> None:
    graph: nx.Graph[int] = nx.Graph()

    assert compute_girth(graph) == float("inf")


def test_compute_girth_handles_acyclic_graph() -> None:
    assert compute_girth(_path_graph(4)) == float("inf")


def test_compute_girth_finds_shortest_cycle() -> None:
    assert compute_girth(_cycle_graph(5)) == 5


def test_compute_girth_counts_self_loop_as_cycle_length_one() -> None:
    graph: nx.Graph[int] = nx.Graph()
    _ = graph.add_edge(0, 0)

    assert compute_girth(graph) == 1


@pytest.mark.parametrize(
    ("k", "g", "expected"),
    [
        (3, 5, 10),
        (3, 6, 14),
        (4, 5, 17),
        (5, 6, 42),
    ],
)
def test_moore_bound_known_small_values(k: int, g: int, expected: int) -> None:
    assert moore_bound(k, g) == expected


@pytest.mark.parametrize(
    ("k", "g", "expected"),
        [
            (3, 5, 16),
            (3, 6, 32),
            (4, 5, 54),
    ],
)
def test_moore_hoffman_upper_bound_known_small_values(
    k: int, g: int, expected: int
) -> None:
    assert moore_hoffman_upper_bound(k, g) == expected


def test_petersen_graph_validates_as_3_5_graph() -> None:
    petersen = _petersen_graph()

    assert is_k_regular(petersen, 3)
    assert is_valid_cage(petersen, 3, 5)
    assert not is_valid_cage(petersen, 3, 6)


def test_can_add_edge_preserving_girth_accepts_exact_target_cycle() -> None:
    path = _path_graph(4)

    assert not can_add_edge_preserving_girth(path, 0, 2, 4)
    assert can_add_edge_preserving_girth(path, 0, 3, 4)
    assert not can_add_edge_preserving_girth(path, 0, 0, 3)


def test_degree_bucket_helpers_classify_partial_graph() -> None:
    graph = _path_graph(4)

    assert compute_regularity_score(graph, 2) == 0.5
    assert get_nodes_by_degree(graph, 2) == {
        "low": [0, 3],
        "correct": [1, 2],
        "high": [],
    }


def test_score_graph_quality_rewards_valid_graph_more_than_partial_graph() -> None:
    partial = _path_graph(10)
    petersen = _petersen_graph()

    assert score_graph_quality(petersen, 3, 5) == 1.0
    assert score_graph_quality(petersen, 3, 5) > score_graph_quality(partial, 3, 5)
