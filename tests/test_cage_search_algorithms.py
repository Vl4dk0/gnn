import networkx as nx

from ai.cage.registry.astar import graph_hash


def _path_graph(n: int) -> nx.Graph[int]:
    graph: nx.Graph[int] = nx.Graph()
    graph.add_nodes_from(range(n))
    for node in range(n - 1):
        _ = graph.add_edge(node, node + 1)
    return graph


def test_graph_hash_is_stable_for_same_labeled_graph() -> None:
    graph = _path_graph(4)

    assert graph_hash(graph) == graph_hash(graph.copy())
