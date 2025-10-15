"""Shared graph generation utilities."""
import os
import random

import networkx as nx


def generate_random_graph(num_nodes_range=None, p_range=None, self_loop_prob=0.1):
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
        num_nodes_min = int(os.getenv('GRAPH_NUM_NODES_MIN', 5))
        num_nodes_max = int(os.getenv('GRAPH_NUM_NODES_MAX', 12))
        num_nodes_range = (num_nodes_min, num_nodes_max)
    
    if p_range is None:
        edge_prob_min = float(os.getenv('GRAPH_EDGE_PROB_MIN', 0.15))
        edge_prob_max = float(os.getenv('GRAPH_EDGE_PROB_MAX', 0.6))
        p_range = (edge_prob_min, edge_prob_max)
    
    # Random graph parameters
    num_nodes = random.randint(*num_nodes_range)
    p = random.uniform(*p_range)
    
    # Generate random Erdős-Rényi graph (no multiple edges)
    G = nx.erdos_renyi_graph(num_nodes, p)
    
    # Add random self-loops
    for i in range(num_nodes):
        if random.random() < self_loop_prob:
            G.add_edge(i, i)
    
    return G


def graph_to_edge_list(G):
    """
    Convert a NetworkX graph to an edge list string format.
    
    Args:
        G: NetworkX Graph object
    
    Returns:
        String with edge list format (one edge per line, isolated vertices listed separately)
        Example: "0 1\n0 2\n3\n4 5" where 3 is an isolated vertex
    """
    edge_list = []
    
    # Add isolated vertices first (sorted)
    isolated = sorted([node for node in G.nodes() if G.degree(node) == 0])
    for node in isolated:
        edge_list.append(str(node))
    
    # Add edges (sorted)
    edges_sorted = sorted([(min(u, v), max(u, v)) for u, v in G.edges()])
    for u, v in edges_sorted:
        edge_list.append(f"{u} {v}")
    
    return "\n".join(edge_list)


# ============================================================================
# Cage Graph Utilities
# ============================================================================

def compute_girth(G):
    """
    Compute the girth (shortest cycle length) of a graph.
    
    Args:
        G: NetworkX Graph object
    
    Returns:
        Girth as an integer, or float('inf') if the graph is acyclic
    """
    if len(G.nodes()) == 0:
        return float('inf')
    
    if len(G.edges()) == 0:
        return float('inf')
    
    # BFS-based girth computation
    return _compute_girth_bfs(G)


def _compute_girth_bfs(G):
    """
    Compute girth using BFS from each node.
    
    For undirected graphs, girth is the length of the shortest cycle.
    """
    if len(G.nodes()) < 3:
        return float('inf')
    
    min_cycle = float('inf')
    
    for start_node in G.nodes():
        # BFS to find shortest cycle containing start_node
        visited = {start_node: 0}
        queue = [(start_node, -1, 0)]  # (node, parent, distance)
        
        while queue:
            node, parent, dist = queue.pop(0)
            
            for neighbor in G.neighbors(node):
                if neighbor == parent:
                    # Skip the edge we came from (undirected)
                    continue
                
                if neighbor in visited:
                    # Found a cycle
                    cycle_length = dist + visited[neighbor] + 1
                    min_cycle = min(min_cycle, cycle_length)
                else:
                    visited[neighbor] = dist + 1
                    queue.append((neighbor, node, dist + 1))
    
    return min_cycle


def is_k_regular(G, k):
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


def moore_bound(k, g):
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
    
    if g % 2 == 1:  # Odd girth
        # N = 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
        sum_term = sum((k - 1) ** i for i in range((g - 3) // 2 + 1))
        return 1 + k * sum_term
    else:  # Even girth
        # N = 2 * sum_{i=0}^{(g/2) - 1} (k-1)^i
        sum_term = sum((k - 1) ** i for i in range(g // 2))
        return 2 * sum_term


def is_valid_cage(G, k, g):
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
    
    computed_girth = compute_girth(G)
    return computed_girth == g


def can_add_edge_preserving_girth(G, u, v, target_girth):
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
        path_length = nx.shortest_path_length(G, u, v)
        # Adding edge would create cycle of length path_length + 1
        cycle_length = path_length + 1
        return cycle_length >= target_girth
    except nx.NetworkXNoPath:
        # No path exists, adding edge won't create a cycle
        return True


def compute_regularity_score(G, k):
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
    
    nodes_at_correct_degree = sum(1 for node in G.nodes() if G.degree(node) == k)
    return nodes_at_correct_degree / len(G.nodes())


def get_nodes_by_degree(G, k):
    """
    Categorize nodes by their degree relative to k.
    
    Args:
        G: NetworkX Graph object
        k: Target degree
    
    Returns:
        dict with keys 'low', 'correct', 'high' containing node lists
    """
    low = []
    correct = []
    high = []
    
    for node in G.nodes():
        deg = G.degree(node)
        if deg < k:
            low.append(node)
        elif deg == k:
            correct.append(node)
        else:
            high.append(node)
    
    return {'low': low, 'correct': correct, 'high': high}


def score_graph_quality(G, k, g):
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
    regularity = compute_regularity_score(G, k)
    
    # Component 2: Size relative to Moore bound (30% weight)
    num_nodes = len(G.nodes())
    mb = moore_bound(k, g)
    size_score = 1.0 - abs(num_nodes - mb) / max(mb, num_nodes)
    
    # Component 3: Girth (30% weight)
    current_girth = compute_girth(G)
    if current_girth == float('inf'):
        girth_score = 0.5  # No cycles yet, neutral
    elif current_girth < g:
        girth_score = 0.0  # Violated constraint
    elif current_girth == g:
        girth_score = 1.0  # Perfect
    else:
        girth_score = 0.8  # Larger than needed, okay but not optimal
    
    return 0.4 * regularity + 0.3 * size_score + 0.3 * girth_score
