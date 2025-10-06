"""Graph visualization service."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import io
import base64


def create_graph_visualization(G: nx.MultiGraph, target_vertex: int) -> str:
    """
    Create a visualization of the graph with the target vertex highlighted.
    Handles multigraphs with self-loops and multiple edges.

    Args:
        G: NetworkX MultiGraph object
        target_vertex: The vertex to highlight

    Returns:
        Base64 encoded PNG image
    """
    plt.figure(figsize=(10, 8))

    # Use spring layout for better visualization
    pos = nx.spring_layout(G, seed=42, k=1, iterations=50)

    # Create node colors - highlight target vertex
    node_colors = [
        '#888888' if node == target_vertex else '#666666'
        for node in G.nodes()
    ]
    node_sizes = [1200 if node == target_vertex else 800 for node in G.nodes()]

    # Create edge colors for nodes - green border for target vertex
    edge_colors = [
        '#00ff00' if node == target_vertex else '#222222'
        for node in G.nodes()
    ]
    edge_widths = [3 if node == target_vertex else 2 for node in G.nodes()]

    # Draw edges (including multiple edges and self-loops)
    # For multigraphs, we need to draw each edge
    ax = plt.gca()

    # Draw regular edges
    for u, v, key in G.edges(keys=True):
        if u == v:  # Self-loop
            # Draw self-loop as a circle above the node
            x, y = pos[u]
            radius = 0.1
            circle = plt.Circle((x, y + radius),
                                radius,
                                fill=False,
                                edgecolor='#444444',
                                linewidth=2,
                                alpha=0.5)
            ax.add_patch(circle)
        else:
            # Draw regular edge
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ax.plot([x0, x1], [y0, y1],
                    color='#444444',
                    linewidth=2,
                    alpha=0.5,
                    zorder=1)

    # Draw nodes
    nx.draw_networkx_nodes(G,
                           pos,
                           node_color=node_colors,
                           node_size=node_sizes,
                           alpha=0.9,
                           edgecolors=edge_colors,
                           linewidths=edge_widths)

    # Draw labels
    nx.draw_networkx_labels(G,
                            pos,
                            font_size=12,
                            font_weight='bold',
                            font_color='white')

    plt.axis('off')
    plt.tight_layout()

    # Convert plot to base64 encoded image
    buffer = io.BytesIO()
    plt.savefig(buffer,
                format='png',
                dpi=100,
                bbox_inches='tight',
                facecolor='#1a1a1a')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return image_base64
