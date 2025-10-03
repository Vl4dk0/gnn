"""Graph parsing utilities."""
import networkx as nx


def parse_edge_list(edge_list_str: str) -> nx.MultiGraph:
    """
    Parse edge list string into a NetworkX multigraph.
    
    Args:
        edge_list_str: String containing edges, one per line (e.g., "0 1\n0 2")
    
    Returns:
        NetworkX MultiGraph object (allows self-loops and multiple edges)
    
    Raises:
        ValueError: If edge format is invalid
    """
    G = nx.MultiGraph()
    
    lines = edge_list_str.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(
                f"Invalid edge format: '{line}'. Expected two vertices per line."
            )
        
        try:
            v1, v2 = int(parts[0]), int(parts[1])
            G.add_edge(v1, v2)
        except ValueError:
            raise ValueError(
                f"Invalid vertex format: '{line}'. Vertices must be integers."
            )
    
    return G
