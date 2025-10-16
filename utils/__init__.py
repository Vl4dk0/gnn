"""Graph utilities package."""

from .graph_utils import (
    # General graph utilities
    generate_random_graph,
    graph_to_edge_list,
    
    # Cage graph utilities
    compute_girth,
    is_k_regular,
    moore_bound,
    is_valid_cage,
    can_add_edge_preserving_girth,
    compute_regularity_score,
    get_nodes_by_degree,
    score_graph_quality,
)

__all__ = [
    'generate_random_graph',
    'graph_to_edge_list',
    'compute_girth',
    'is_k_regular',
    'moore_bound',
    'is_valid_cage',
    'can_add_edge_preserving_girth',
    'compute_regularity_score',
    'get_nodes_by_degree',
    'score_graph_quality',
]
