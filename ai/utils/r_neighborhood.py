"""
Compute r-neighborhoods for Loopy GNN.
Based on "Weisfeiler and Leman Go Loopy" (NeurIPS 2024).

The r-neighborhood of a node v is the collection of all simple cycles
of length <= r+2 that pass through v, represented as paths starting at v.
"""

import networkx as nx
import torch
from typing import cast
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]
from torch_geometric.utils import to_networkx  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]


def compute_r_neighborhood(G: nx.Graph[int], r: int = 3) -> dict[str, torch.Tensor]:
    """
    Compute r-neighborhood for all nodes in graph G.

    For each node v, finds all simple cycles of length <= r+2 that contain v,
    and represents them as paths starting and ending at v.

    Args:
        G: NetworkX graph
        r: Neighborhood radius (default 3, detects cycles up to length 5)

    Returns:
        Dictionary with:
        - loopyN{L}: tensor of paths of length L+2, shape (L+2, num_paths)
        - loopyA{L}: tensor of distances from center, shape (L+2, num_paths)

        where L ranges from 0 to r.
    """
    # Find all simple cycles up to length r+2
    cycles: list[list[int]] = list(nx.simple_cycles(G, length_bound=r + 2))

    # Compute all pairwise distances (for atomic type - distance from center)
    distances = cast(
        dict[int, dict[int, int]], dict(nx.all_pairs_shortest_path_length(G))
    )

    result: dict[str, torch.Tensor] = {}

    for L in range(r + 1):
        path_length = L + 2  # Cycles of this length

        # Filter cycles of this length and generate all rotations
        L_paths: list[list[int]] = []
        L_atomic: list[list[int]] = []

        for cycle in cycles:
            if len(cycle) == path_length:
                # Generate all cyclic rotations (one for each starting node)
                for start_idx in range(len(cycle)):
                    # Rotate so cycle starts at different node each time
                    rotated = cycle[start_idx:] + cycle[:start_idx]
                    center = rotated[0]

                    # Compute distances from center for each node in path
                    atomic = [distances[center][node] for node in rotated]

                    L_paths.append(rotated)
                    L_atomic.append(atomic)

        if L_paths:
            # Convert to tensors, transpose to (path_length, num_paths)
            result[f"loopyN{L}"] = torch.tensor(L_paths, dtype=torch.long).T
            result[f"loopyA{L}"] = torch.tensor(L_atomic, dtype=torch.long).T
        else:
            # Empty tensors with correct shape
            result[f"loopyN{L}"] = torch.zeros((path_length, 0), dtype=torch.long)
            result[f"loopyA{L}"] = torch.zeros((path_length, 0), dtype=torch.long)

    return result


def apply_r_neighborhood(data: Data, r: int = 3) -> Data:
    """
    Apply r-neighborhood transform to PyG Data object.

    Adds loopyN{L} and loopyA{L} attributes for L in 0..r.
    These are used by the Loopy GNN model for cycle-aware message passing.

    Args:
        data: PyTorch Geometric Data object with edge_index
        r: Neighborhood radius (default 3)

    Returns:
        Modified Data object with loopy attributes added
    """
    # Convert to NetworkX (undirected for cycle detection)
    G = cast(nx.Graph[int], to_networkx(data, to_undirected=True))

    # Compute r-neighborhoods
    r_neigh = compute_r_neighborhood(G, r=r)

    # Add to data object
    for key, value in r_neigh.items():
        data[key] = value

    return data
