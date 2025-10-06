"""Graph parsing utilities."""
import networkx as nx


def parse_edge_list(edge_list_str: str) -> nx.MultiGraph:
    """
    Parse edge list string into a NetworkX multigraph.

    Args:
        edge_list_str: String containing edges, one per line (e.g., "0 1\n0 2")
                      Single numbers represent isolated vertices (e.g., "3")

    Returns:
        NetworkX MultiGraph object (allows self-loops and multiple edges)

    Raises:
        ValueError: If edge format is invalid
    """
    G = nx.MultiGraph()

    lines = edge_list_str.strip().split('\n')
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
                G.add_edge(v1, v2)
            except ValueError:
                raise ValueError(
                    f"Line {line_num}: Invalid vertex format: '{line}'. Vertices must be integers."
                )
        else:
            raise ValueError(
                f"Line {line_num}: Invalid format: '{line}'. Expected one vertex (isolated) or two vertices (edge) per line."
            )

    return G
