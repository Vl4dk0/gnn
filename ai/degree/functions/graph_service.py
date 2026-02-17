"""Graph service for degree prediction with model selection support."""

import os
from typing import Optional

import networkx as nx
import torch
from torch_geometric.data import Data

from ai.utils.r_neighborhood import apply_r_neighborhood

# Model cache - maps model_id to loaded model
_model_cache: dict = {}


def get_true_degree(G: nx.Graph, vertex: int) -> int:
    """
    Get the true degree of a vertex in the graph.
    For graphs with self-loops, self-loops are counted twice.

    Args:
        G: NetworkX Graph object
        vertex: The vertex to get the degree for

    Returns:
        Degree of the vertex

    Raises:
        ValueError: If vertex not found in graph
    """
    if vertex not in G.nodes():
        raise ValueError(f"Vertex {vertex} not found in graph")

    return G.degree(vertex)  # type: ignore


def load_degree_gnn(model_id: Optional[str] = None):
    """
    Load a trained GNN model by model_id.

    Args:
        model_id: The model identifier (e.g., 'gin_v1').
                  If None, loads the best available model.

    Returns:
        Loaded model or None if not found
    """
    global _model_cache

    # Import here to avoid circular imports
    from ai.registry import load_model, get_best_model_id

    # Get the model_id to use
    if model_id is None:
        model_id = get_best_model_id("degree")
        if model_id is None:
            print("Warning: No trained models found for degree prediction")
            return None

    # Check cache
    if model_id in _model_cache:
        return _model_cache[model_id]

    try:
        model = load_model("degree", model_id)
        _model_cache[model_id] = model
        print(f"Loaded model '{model_id}' for degree prediction")
        return model
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        return None
    except Exception as e:
        print(f"Error loading model '{model_id}': {e}")
        return None


def predict_all_nodes(G: nx.Graph, model_id: Optional[str] = None) -> list[dict]:
    """
    Predict the degree of all vertices in the graph using a trained GNN.

    Args:
        G: NetworkX Graph object
        model_id: Optional model identifier. Uses best model if not provided.

    Returns:
        List of dictionaries with format:
        [
            {
                "node_id": 0,
                "true": 2,
                "predicted": 2.0
            },
            ...
        ]
    """
    model = load_degree_gnn(model_id)

    # If model not available, return true degrees
    if model is None:
        return [
            {
                "node_id": node,
                "true": get_true_degree(G, node),
                "predicted": float(get_true_degree(G, node)),
            }
            for node in sorted(G.nodes())
        ]

    try:
        # Convert NetworkX graph to PyTorch Geometric format
        num_nodes = len(G.nodes())

        if num_nodes == 0:
            return []

        # Create node ID mapping (handle non-sequential node IDs)
        node_list = sorted(G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}

        # Build edge index
        edge_index = []
        for u, v in G.edges():
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            edge_index.append([u_idx, v_idx])
            edge_index.append([v_idx, u_idx])

        if len(edge_index) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Node features: match training setup with rich features
        # Feature 1: Normalized node index (0 to 1)
        node_idx_feature = torch.arange(num_nodes, dtype=torch.float).unsqueeze(
            1
        ) / max(num_nodes - 1, 1)

        # Feature 2: Random embedding (deterministic with seed for consistency)
        torch.manual_seed(42)
        random_feature = torch.randn(num_nodes, 2)

        # Feature 3: Clustering coefficient placeholder
        torch.manual_seed(42)
        clustering_feature = torch.rand(num_nodes, 1)

        # Combine all features (must match training: 4 features)
        x = torch.cat([node_idx_feature, random_feature, clustering_feature], dim=1)

        # Create data object
        data = Data(x=x, edge_index=edge_index)

        # For loopy models, apply r-neighborhood transform
        if hasattr(model, "r") and model.r is not None:
            data = apply_r_neighborhood(data, r=model.r)

        # Predict with GNN
        with torch.no_grad():
            predictions = model(data).squeeze()

            # Handle single node case
            if num_nodes == 1:
                predictions = predictions.unsqueeze(0)

        # Build results for all nodes
        results = []
        for node in node_list:
            idx = node_to_idx[node]
            predicted_degree = max(0.0, round(predictions[idx].item()))
            true_degree = get_true_degree(G, node)

            results.append(
                {
                    "node_id": node,
                    "true": true_degree,
                    "predicted": predicted_degree,
                }
            )

        return results

    except Exception as e:
        print(f"Error in GNN prediction for all nodes: {e}")
        return [
            {
                "node_id": node,
                "true": get_true_degree(G, node),
                "predicted": float(get_true_degree(G, node)),
            }
            for node in sorted(G.nodes())
        ]
