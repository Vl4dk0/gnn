"""Graph service for degree prediction with model selection support."""

from typing import cast

import networkx as nx
import torch
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]

from ai.models.base import BaseGNN
from ai.utils.device import configure_torch_device
from ai.utils.r_neighborhood import apply_r_neighborhood

# Model cache - maps model_id to loaded model
_model_cache: dict[str, BaseGNN] = {}


def _build_node_features(num_nodes: int, feature_mode: str) -> torch.Tensor:
    """Build the node-feature tensor for *feature_mode*.

    "const1" -> a constant 1-dim feature; "feat4" -> the 4-dim vector
    (normalized node index + two random coords + one random scalar). These
    mirror the input_dim 1 vs 4 schemes used during training.
    """
    if feature_mode == "const1":
        return torch.ones(num_nodes, 1)
    if feature_mode == "feat4":
        node_idx_feature = torch.arange(num_nodes, dtype=torch.float).unsqueeze(
            1
        ) / max(num_nodes - 1, 1)
        _ = torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]
        random_feature = torch.randn(num_nodes, 2)
        _ = torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]
        clustering_feature = torch.rand(num_nodes, 1)
        return torch.cat([node_idx_feature, random_feature, clustering_feature], dim=1)
    raise ValueError(f"Unknown feature_mode '{feature_mode}'")


def get_true_degree(G: nx.Graph[int], vertex: int) -> int:
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

    return int(G.degree(vertex))


def load_degree_gnn(model_id: str | None = None) -> BaseGNN | None:
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


def predict_all_nodes(
    G: nx.Graph[int],
    model_id: str | None = None,
    feature_mode: str = "feat4",
) -> list[dict[str, int | float]]:
    """
    Predict the degree of all vertices in the graph using a trained GNN.

    Args:
        G: NetworkX Graph object
        model_id: Optional model identifier. Uses best model if not provided.
        feature_mode: Input node-feature scheme. "feat4" (default) uses the
            4-dim vector (normalized node index + two random coords + one random
            scalar) matching training; "const1" uses a constant 1-dim feature.
            Must match the loaded model's ``input_dim`` or a ValueError is
            raised.

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
    _ = configure_torch_device()
    model = load_degree_gnn(model_id)

    if model is None:
        raise RuntimeError("No trained model available for degree prediction")

    # Convert NetworkX graph to PyTorch Geometric format
    num_nodes = len(G.nodes())

    if num_nodes == 0:
        return []

    # Create node ID mapping (handle non-sequential node IDs)
    node_list: list[int] = sorted(G.nodes())
    node_to_idx: dict[int, int] = {node: idx for idx, node in enumerate(node_list)}

    # Build edge index
    edge_index_list: list[list[int]] = []
    for u, v in G.edges():
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        edge_index_list.append([u_idx, v_idx])
        edge_index_list.append([v_idx, u_idx])

    edge_index: torch.Tensor
    if len(edge_index_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

    # Node features depend on feature_mode (matches the training feature
    # schemes in ai/degree/train.py: input_dim 1 vs 4).
    x = _build_node_features(num_nodes, feature_mode)

    # Guard: the loaded model was trained for a specific input_dim. Feeding a
    # mismatched feature width would either error deep in the conv or silently
    # misbehave, so fail fast with a clear message the caller can catch.
    expected_dim = int(getattr(model, "input_dim", x.size(1)))
    if x.size(1) != expected_dim:
        raise ValueError(
            f"feature_mode '{feature_mode}' produces {x.size(1)}-dim features but"
            + f" model '{model_id}' expects input_dim={expected_dim}"
        )

    # Create data object
    data = Data(x=x, edge_index=edge_index)

    # For loopy models, apply r-neighborhood transform
    r_attr = cast(int | None, getattr(model, "r", None))
    if r_attr is not None:
        data = apply_r_neighborhood(data, r=r_attr)

    # Predict with GNN
    with torch.no_grad():
        predictions = cast(torch.Tensor, model(data)).squeeze()

        # Handle single node case
        if num_nodes == 1:
            predictions = predictions.unsqueeze(0)

    # Build results for all nodes
    results: list[dict[str, int | float]] = []
    for node in node_list:
        idx = node_to_idx[node]
        predicted_degree = max(0.0, round(float(predictions[idx].item())))
        true_degree = get_true_degree(G, node)

        results.append(
            {
                "node_id": node,
                "true": true_degree,
                "predicted": predicted_degree,
            }
        )

    return results
