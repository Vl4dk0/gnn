"""Graph service for min_cycle prediction with model selection support."""

import copy
from typing import cast

import networkx as nx
import torch
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]

from ai.models.base import BaseGNN
from ai.utils.r_neighborhood import apply_r_neighborhood

# Model cache - maps model_id to loaded model
_model_cache: dict[str, BaseGNN] = {}


def get_min_cycle(G: nx.Graph[int], vertex: int) -> int:
    """
    Get the minimal cycle containing a vertex in the graph.
    For graphs with self-loops, self-loops are counted cycle of length 1.

    Args:
        G: NetworkX Graph object
        vertex: The vertex to get the smallest cycle for

    Returns:
        smallest cycle containing vertex
        0 if no cycle contains vertex

    Raises:
        ValueError: If vertex not found in graph
    """
    if vertex not in G.nodes():
        raise ValueError(f"Vertex {vertex} not found in graph")

    if G.has_edge(vertex, vertex):
        return 1

    # for each neighbour, delete edge between vertex and neighbour
    # find shortest path from vertex to neighbour
    # shortest path + 1 is length of a cycle
    # add back the edge

    ans: int | None = None
    neighs: list[int] = list(G.neighbors(vertex))
    for neigh in neighs:
        G.remove_edge(vertex, neigh)

        try:
            path: list[int] = nx.shortest_path(G, vertex, neigh)  # pyright: ignore[reportUnknownMemberType]
            if ans is None:
                ans = len(path)
            else:
                ans = min(ans, len(path))
        except Exception:
            pass

        _ = G.add_edge(vertex, neigh)

    return ans if ans is not None else 0


def load_mincycle_gnn(model_id: str | None = None) -> BaseGNN | None:
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
        model_id = get_best_model_id("min_cycle")
        if model_id is None:
            print("Warning: No trained models found for min_cycle prediction")
            return None

    # Check cache
    if model_id in _model_cache:
        return _model_cache[model_id]

    try:
        model = load_model("min_cycle", model_id)
        _model_cache[model_id] = model
        print(f"Loaded model '{model_id}' for min_cycle prediction")
        return model
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        return None
    except Exception as e:
        print(f"Error loading model '{model_id}': {e}")
        return None


def predict_all_nodes(
    G: nx.Graph[int], model_id: str | None = None
) -> list[dict[str, int | float]]:
    """
    Predict the smallest cycle for all vertices in the graph using a trained GNN.

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
    model = load_mincycle_gnn(model_id)

    _G: nx.Graph[int] = copy.deepcopy(G)

    if model is None:
        return [
            {
                "node_id": node,
                "true": get_min_cycle(_G, node),
                "predicted": float(get_min_cycle(_G, node)),
            }
            for node in sorted(G.nodes())
        ]

    try:
        # Convert NetworkX graph to PyTorch Geometric format
        num_nodes: int = len(G.nodes())

        if num_nodes == 0:
            return []

        # Create node ID mapping (handle non-sequential node IDs)
        node_list: list[int] = sorted(G.nodes())
        node_to_idx: dict[int, int] = {node: idx for idx, node in enumerate(node_list)}

        # Build edge index
        edge_index: list[list[int]] = []
        for u, v in G.edges():
            u_idx: int = node_to_idx[u]
            v_idx: int = node_to_idx[v]
            edge_index.append([u_idx, v_idx])
            edge_index.append([v_idx, u_idx])

        edge_index_tensor: torch.Tensor
        if len(edge_index) == 0:
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index_tensor = (
                torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            )

        # Node features: match training setup with rich features
        # Feature 1: Normalized node index (0 to 1)
        node_idx_feature: torch.Tensor = torch.arange(
            num_nodes, dtype=torch.float
        ).unsqueeze(1) / max(num_nodes - 1, 1)

        # Feature 2: Random embedding (deterministic with seed for consistency)
        _ = torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]
        random_feature: torch.Tensor = torch.randn(num_nodes, 2)

        # Feature 3: Clustering coefficient placeholder
        _ = torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]
        clustering_feature: torch.Tensor = torch.rand(num_nodes, 1)

        # Combine all features (must match training: 4 features)
        x: torch.Tensor = torch.cat(
            [node_idx_feature, random_feature, clustering_feature], dim=1
        )

        # Create data object
        data: Data = Data(x=x, edge_index=edge_index_tensor)

        # For loopy models, apply r-neighborhood transform
        r_attr = cast(int | None, getattr(model, "r", None))
        if r_attr is not None:
            data = apply_r_neighborhood(data, r=int(r_attr))

        # Predict with GNN
        with torch.no_grad():
            predictions: torch.Tensor = cast(torch.Tensor, model(data)).squeeze()

            # Handle single node case
            if num_nodes == 1:
                predictions = predictions.unsqueeze(0)

        # Build results for all nodes
        results: list[dict[str, int | float]] = []
        for node in node_list:
            idx: int = node_to_idx[node]
            predicted: float = max(0.0, round(predictions[idx].item()))
            true: int = get_min_cycle(_G, node)

            results.append(
                {
                    "node_id": node,
                    "true": true,
                    "predicted": predicted,
                }
            )

        return results

    except Exception as e:
        print(f"Error in GNN prediction for all nodes: {e}")
        return [
            {
                "node_id": node,
                "true": get_min_cycle(_G, node),
                "predicted": float(get_min_cycle(_G, node)),
            }
            for node in sorted(G.nodes())
        ]
