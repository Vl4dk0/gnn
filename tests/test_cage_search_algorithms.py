import networkx as nx

from ai.cage.functions.astar import AStarGenerator, graph_hash
from ai.cage.functions.bruteforce import BruteforceGenerator
from ai.cage.functions.random_walk import RandomWalkGenerator
from backend.utils.graph_utils import compute_girth, moore_bound


def _path_graph(n: int) -> nx.Graph[int]:
    graph: nx.Graph[int] = nx.Graph()
    graph.add_nodes_from(range(n))
    for node in range(n - 1):
        _ = graph.add_edge(node, node + 1)
    return graph


def test_random_walk_initializes_at_moore_bound_and_smoke_steps() -> None:
    generator = RandomWalkGenerator(k=3, g=5)

    assert generator.graph.number_of_nodes() == moore_bound(3, 5)

    for _ in range(5):
        generator.step()

    assert generator.step_count == 5
    assert generator.graph.number_of_nodes() <= generator.upper_bound
    assert compute_girth(generator.graph) in {1, 3, 4, 5, float("inf")}


def test_bruteforce_generator_smoke_steps_keep_bounds() -> None:
    generator = BruteforceGenerator(k=3, g=5)

    for _ in range(5):
        generator.step()

    assert generator.step_count == 5
    assert generator.graph.number_of_nodes() <= generator.upper_bound
    assert compute_girth(generator.graph) in {5, float("inf")}


def test_astar_generator_smoke_steps_keep_bounds() -> None:
    generator = AStarGenerator(k=3, g=5)

    for _ in range(2):
        generator.step()

    assert generator.step_count == 2
    assert generator.graph.number_of_nodes() <= generator.upper_bound
    assert compute_girth(generator.graph) in {5, float("inf")}


def test_graph_hash_is_stable_for_same_labeled_graph() -> None:
    graph = _path_graph(4)

    assert graph_hash(graph) == graph_hash(graph.copy())
