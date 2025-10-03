"""Graph analysis service."""
import networkx as nx


def get_true_degree(G: nx.MultiGraph, vertex: int) -> int:
    """
    Get the true degree of a vertex in the graph.
    For multigraphs, this counts all edges including self-loops (counted twice).
    
    Args:
        G: NetworkX MultiGraph object
        vertex: The vertex to get the degree for
    
    Returns:
        Degree of the vertex
    
    Raises:
        ValueError: If vertex not found in graph
    """
    if vertex not in G.nodes():
        raise ValueError(f"Vertex {vertex} not found in graph")
    
    return G.degree(vertex)


def predict_degree_with_gnn(G: nx.MultiGraph, vertex: int) -> float:
    """
    Predict the degree of a vertex using a GNN.
    This is a placeholder for now - returns the true degree.
    
    Args:
        G: NetworkX MultiGraph object
        vertex: The vertex to predict the degree for
    
    Returns:
        Predicted degree (placeholder - currently returns true degree)
    """
    # TODO: Implement actual GNN prediction
    # For now, return the true degree as a placeholder
    true_degree = get_true_degree(G, vertex)
    
    # Add some small random variation to show it's a "prediction"
    # This will be replaced with actual GNN prediction later
    return float(true_degree)
