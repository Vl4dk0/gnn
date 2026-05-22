import networkx as nx
import pytest

from ai.degree.functions.graph_service import get_true_degree
from ai.min_cycle.functions.graph_service import get_min_cycle


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


def test_true_degree_counts_self_loop_twice() -> None:
    graph: nx.Graph[int] = nx.Graph()
    _ = graph.add_edge(0, 0)
    _ = graph.add_edge(0, 1)

    assert get_true_degree(graph, 0) == 3
    assert get_true_degree(graph, 1) == 1


def test_true_degree_rejects_missing_vertex() -> None:
    with pytest.raises(ValueError, match="Vertex 99 not found"):
        _ = get_true_degree(_path_graph(3), 99)


@pytest.mark.parametrize(
    ("graph", "vertex", "expected"),
    [
        (_cycle_graph(4), 0, 4),
        (_path_graph(4), 0, 0),
    ],
)
def test_min_cycle_labels_cycles_and_trees(
    graph: nx.Graph[int], vertex: int, expected: int
) -> None:
    assert get_min_cycle(graph, vertex) == expected


def test_min_cycle_labels_self_loop_as_one() -> None:
    graph: nx.Graph[int] = nx.Graph()
    _ = graph.add_edge(7, 7)

    assert get_min_cycle(graph, 7) == 1


def test_min_cycle_does_not_mutate_input_graph() -> None:
    graph = _cycle_graph(5)
    edges_before: list[tuple[int, int]] = sorted(graph.edges())

    assert get_min_cycle(graph, 0) == 5
    assert sorted(graph.edges()) == edges_before


def test_min_cycle_rejects_missing_vertex() -> None:
    with pytest.raises(ValueError, match="Vertex 99 not found"):
        _ = get_min_cycle(_path_graph(3), 99)
