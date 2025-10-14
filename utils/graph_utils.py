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
    
    # Add edges
    for u, v in G.edges():
        edge_list.append(f"{u} {v}")
    
    # Add isolated vertices (nodes with degree 0)
    for node in G.nodes():
        if G.degree(node) == 0:
            edge_list.append(str(node))
    
    return "\n".join(edge_list)
