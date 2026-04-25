"""Shared graph generation utilities."""

import os
import random
from typing import cast

import networkx as nx


def parse_edge_list(edge_list_str: str) -> nx.Graph[int]:
    """
    Parse edge list string into a NetworkX graph.

    Args:
        edge_list_str: String containing edges, one per line (e.g., "0 1\n0 2")
                      Single numbers represent isolated vertices (e.g., "3")

    Returns:
        NetworkX Graph object (allows self-loops but NOT multiple edges)

    Raises:
        ValueError: If edge format is invalid
    """
    G: nx.Graph[int] = nx.Graph()

    lines = edge_list_str.strip().split("\n")
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()

        if len(parts) == 1:
            # Single number - isolated vertex
            try:
                vertex = int(parts[0])
                G.add_node(vertex)
            except ValueError:
                raise ValueError(
                    f"Line {line_num}: Invalid vertex format: '{line}'. Vertex must be an integer."
                )
        elif len(parts) == 2:
            # Two numbers - edge
            try:
                v1, v2 = int(parts[0]), int(parts[1])
                _ = G.add_edge(v1, v2)
            except ValueError:
                raise ValueError(
                    f"Line {line_num}: Invalid vertex format: '{line}'. Vertices must be integers."
                )
        else:
            raise ValueError(
                f"Line {line_num}: Invalid format: '{line}'. Expected one vertex (isolated) or two vertices (edge) per line."
            )

    return G


def generate_random_graph(
    num_nodes_range: tuple[int, int] | None = None,
    p_range: tuple[float, float] | None = None,
    self_loop_prob: float = 0.1,
) -> nx.Graph[int]:
    """
    Generate a random graph with configurable parameters.

    Args:
        num_nodes_range: Tuple of (min, max) number of nodes.
                        If None, reads from environment variables.
        p_range: Tuple of (min, max) edge probability for Erdős-Rényi model.
                If None, reads from environment variables.
        self_loop_prob: Probability of adding a self-loop to each node (default: 0.1)

    Returns:
        NetworkX Graph object (allows self-loops but NOT multiple edges)
    """
    # Get defaults from environment variables if not provided
    if num_nodes_range is None:
        num_nodes_min: int = int(os.getenv("GRAPH_NUM_NODES_MIN", 5))
        num_nodes_max: int = int(os.getenv("GRAPH_NUM_NODES_MAX", 12))
        num_nodes_range = (num_nodes_min, num_nodes_max)

    if p_range is None:
        edge_prob_min: float = float(os.getenv("GRAPH_EDGE_PROB_MIN", 0.15))
        edge_prob_max: float = float(os.getenv("GRAPH_EDGE_PROB_MAX", 0.6))
        p_range = (edge_prob_min, edge_prob_max)

    # Random graph parameters
    num_nodes: int = random.randint(*num_nodes_range)
    p: float = random.uniform(*p_range)

    # Generate random Erdős-Rényi graph (no multiple edges)
    G: nx.Graph[int] = nx.erdos_renyi_graph(num_nodes, p)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    # Add random self-loops
    i: int
    for i in range(num_nodes):
        if random.random() < self_loop_prob:
            _ = G.add_edge(i, i)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    return G  # pyright: ignore[reportUnknownVariableType]


def graph_to_edge_list(G: nx.Graph[int]) -> str:
    """
    Convert a NetworkX graph to an edge list string format.

    Args:
        G: NetworkX Graph object

    Returns:
        String with edge list format (one edge per line, isolated vertices listed separately)
        Example: "0 1\n0 2\n3\n4 5" where 3 is an isolated vertex
    """
    edge_list: list[str] = []

    # Add isolated vertices first (sorted)
    isolated: list[int] = sorted([node for node in G.nodes() if G.degree(node) == 0])
    node: int
    for node in isolated:
        edge_list.append(str(node))

    # Add edges (sorted)
    edges_sorted: list[tuple[int, int]] = sorted(
        [(min(u, v), max(u, v)) for u, v in G.edges()]
    )
    u: int
    v: int
    for u, v in edges_sorted:
        edge_list.append(f"{u} {v}")

    return "\n".join(edge_list)


def compute_girth(G: nx.Graph[int]) -> int | float:
    """
    Compute the girth (shortest cycle length) of a graph.

    Args:
        G: NetworkX Graph object

    Returns:
        Girth as an integer, or float('inf') if the graph is acyclic
    """
    if len(G.nodes()) == 0:
        return float("inf")

    if len(G.edges()) == 0:
        return float("inf")

    if any(G.has_edge(node, node) for node in G.nodes()):
        return 1

    # BFS-based girth computation
    return _compute_girth_bfs(G)


def _compute_girth_bfs(G: nx.Graph[int]) -> int | float:
    """
    Compute girth using BFS from each node.

    For undirected graphs, girth is the length of the shortest cycle.
    """
    if len(G.nodes()) < 3:
        return float("inf")

    min_cycle: int | float = float("inf")

    start_node: int
    for start_node in G.nodes():
        # BFS to find shortest cycle containing start_node
        visited: dict[int, int] = {start_node: 0}
        queue: list[tuple[int, int, int]] = [
            (start_node, -1, 0)
        ]  # (node, parent, distance)

        while queue:
            node: int
            parent: int
            dist: int
            node, parent, dist = queue.pop(0)

            neighbor: int
            for neighbor in G.neighbors(node):
                if neighbor == parent:
                    # Skip the edge we came from (undirected)
                    continue

                if neighbor in visited:
                    # Found a cycle
                    cycle_length: int = dist + visited[neighbor] + 1
                    min_cycle = min(min_cycle, cycle_length)
                else:
                    visited[neighbor] = dist + 1
                    queue.append((neighbor, node, dist + 1))

    return min_cycle


def is_k_regular(G: nx.Graph[int], k: int) -> bool:
    """
    Check if graph G is k-regular (all nodes have degree k).

    Args:
        G: NetworkX Graph object
        k: Target degree

    Returns:
        True if all nodes have degree k, False otherwise
    """
    if len(G.nodes()) == 0:
        return False

    return all(G.degree(node) == k for node in G.nodes())


def moore_bound(k: int, g: int) -> int:
    """
    Compute the Moore lower bound for the minimum number of vertices
    in a (k,g)-cage graph.

    Args:
        k: Degree (must be >= 2)
        g: Girth (must be >= 3)

    Returns:
        Minimum number of vertices (Moore bound)
    """
    if k < 2 or g < 3:
        raise ValueError("k must be >= 2 and g must be >= 3")

    sum_term: int
    if g % 2 == 1:  # Odd girth
        # N = 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
        sum_term = sum((k - 1) ** i for i in range((g - 3) // 2 + 1))
        return 1 + k * sum_term
    else:  # Even girth
        # N = 2 * sum_{i=0}^{(g/2) - 1} (k-1)^i
        sum_term = sum((k - 1) ** i for i in range(g // 2))
        return 2 * sum_term


def moore_hoffman_upper_bound(k: int, g: int) -> int:
    """
    Compute the Moore-Hoffman upper bound for the maximum number of vertices
    in a (k,g)-cage graph.

    This is a theoretical upper limit - no (k,g)-cage can have more vertices
    than this bound.

    Formula:
        N(k,g) ≤ 2(k-1)^(g-2)    if g is odd
        N(k,g) ≤ 4(k-1)^(g-3)    if g is even

    Args:
        k: Degree (must be >= 2)
        g: Girth (must be >= 3)

    Returns:
        Maximum number of vertices (Moore-Hoffman upper bound)
    """
    if k < 2 or g < 3:
        raise ValueError("k must be >= 2 and g must be >= 3")

    if g % 2 == 1:  # Odd girth
        return cast(int, 2 * (k - 1) ** (g - 2))
    else:  # Even girth
        return cast(int, 4 * (k - 1) ** (g - 3))


def is_valid_cage(G: nx.Graph[int], k: int, g: int) -> bool:
    """
    Check if graph G is a valid (k,g)-cage.

    A cage is k-regular and has girth exactly g.

    Args:
        G: NetworkX Graph object
        k: Target degree
        g: Target girth

    Returns:
        True if G is a valid (k,g)-cage, False otherwise
    """
    if not is_k_regular(G, k):
        return False

    computed_girth: int | float = compute_girth(G)
    return computed_girth == g


def can_add_edge_preserving_girth(
    G: nx.Graph[int], u: int, v: int, target_girth: int
) -> bool:
    """
    Check if adding edge (u,v) would preserve girth >= target_girth.

    Smart incremental approach: Only check shortest path through new edge.
    Uses the insight that the smallest cycle can only change around the newly added edge.

    Args:
        G: NetworkX Graph object
        u, v: Nodes to connect
        target_girth: Minimum acceptable girth

    Returns:
        True if edge can be added without creating cycle < target_girth
    """
    if u == v:
        # Self-loop creates cycle of length 1
        return target_girth <= 1

    try:
        # Find shortest path from u to v (excluding the direct edge)
        path_length: float = nx.shortest_path_length(G, u, v)  # pyright: ignore[reportUnknownMemberType]
        # Adding edge would create cycle of length path_length + 1
        cycle_length: float = path_length + 1
        return cycle_length >= target_girth
    except nx.NetworkXNoPath:
        # No path exists, adding edge won't create a cycle
        return True


def compute_regularity_score(G: nx.Graph[int], k: int) -> float:
    """
    Compute how close the graph is to being k-regular.

    Args:
        G: NetworkX Graph object
        k: Target degree

    Returns:
        Score in [0, 1] where 1 = perfectly regular
    """
    if len(G.nodes()) == 0:
        return 0.0

    nodes_at_correct_degree: int = sum(1 for node in G.nodes() if G.degree(node) == k)
    return nodes_at_correct_degree / len(G.nodes())


def get_nodes_by_degree(G: nx.Graph[int], k: int) -> dict[str, list[int]]:
    """
    Categorize nodes by their degree relative to k.

    Args:
        G: NetworkX Graph object
        k: Target degree

    Returns:
        dict with keys 'low', 'correct', 'high' containing node lists
    """
    low: list[int] = []
    correct: list[int] = []
    high: list[int] = []

    node: int
    for node in G.nodes():
        deg: int = G.degree(node)
        if deg < k:
            low.append(node)
        elif deg == k:
            correct.append(node)
        else:
            high.append(node)

    return {"low": low, "correct": correct, "high": high}


def score_graph_quality(G: nx.Graph[int], k: int, g: int) -> float:
    """
    Score how close a graph is to being a valid (k,g)-cage.

    Used for beam search and other algorithms that need to rank partial graphs.

    Args:
        G: NetworkX Graph object
        k: Target degree
        g: Target girth

    Returns:
        Score in [0, 1] where 1 = valid cage
    """
    if len(G.nodes()) == 0:
        return 0.0

    # Component 1: Regularity (40% weight)
    regularity: float = compute_regularity_score(G, k)

    # Component 2: Size relative to Moore bound (30% weight)
    num_nodes: int = len(G.nodes())
    mb: int = moore_bound(k, g)
    size_score: float = 1.0 - abs(num_nodes - mb) / max(mb, num_nodes)

    # Component 3: Girth (30% weight)
    current_girth: int | float = compute_girth(G)
    girth_score: float
    if current_girth == float("inf"):
        girth_score = 0.5  # No cycles yet, neutral
    elif current_girth < g:
        girth_score = 0.0  # Violated constraint
    elif current_girth == g:
        girth_score = 1.0  # Perfect
    else:
        girth_score = 0.8  # Larger than needed, okay but not optimal

    return 0.4 * regularity + 0.3 * size_score + 0.3 * girth_score
