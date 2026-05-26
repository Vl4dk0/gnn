"""Structural node features for GNN inputs.

Implements:
- Cycle count features: number of closed simple cycles of each length through
  each vertex, computed via brute-force enumeration with networkx.simple_cycles.
  Hard cap at length 16; for larger lengths a warning is emitted.
- Random-Walk Positional Encoding (RWPE) following Dwivedi et al. 2022:
  the diagonal of M, M^2, ..., M^K where M = D^{-1} A.

These features break the Chen-Song-Caramanis 2023 symmetry barrier that causes
standard message-passing GNNs to assign identical embeddings on regular graphs.
"""

from __future__ import annotations

import warnings
from typing import cast

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]

# Hard cap on brute-force cycle enumeration.
# nx.simple_cycles enumerates all simple cycles; we filter by length.
# For l > this cap a UserWarning is emitted and counts are capped.
_BRUTE_FORCE_MAX_LEN = 16


def _cycle_counts_brute_force(
    G: nx.Graph[int],
    n: int,
    lengths: list[int],
) -> dict[int, np.ndarray[tuple[int], np.dtype[np.float64]]]:
    """Compute per-vertex simple-cycle counts via networkx.simple_cycles.

    nx.simple_cycles yields oriented cycles (each undirected cycle appears in
    both orientations). We halve the count to get undirected cycles through v.
    """
    max_len = max(lengths)
    counts: dict[int, np.ndarray[tuple[int], np.dtype[np.float64]]] = {
        length: np.zeros(n, dtype=np.float64) for length in lengths
    }

    for cycle in nx.simple_cycles(G, length_bound=max_len):
        length = len(cycle)
        if length not in counts:
            continue
        for v in cycle:
            counts[length][v] += 0.5  # two orientations per undirected cycle

    return counts


def _compute_cycle_counts(
    G: nx.Graph[int],
    n: int,
    lengths: list[int],
) -> dict[int, np.ndarray[tuple[int], np.dtype[np.float64]]]:
    """Return per-vertex simple-cycle counts for each requested length.

    Uses brute-force enumeration via networkx. Cage-scale graphs (n <= a few
    thousand) are small enough for this to be fast in practice.

    For any length > _BRUTE_FORCE_MAX_LEN a UserWarning is emitted and the
    count is capped at _BRUTE_FORCE_MAX_LEN.
    """
    if any(length > _BRUTE_FORCE_MAX_LEN for length in lengths):
        warnings.warn(
            f"cycle_lengths contains values > {_BRUTE_FORCE_MAX_LEN}; "
            f"cycle counts are capped at length {_BRUTE_FORCE_MAX_LEN}.",
            UserWarning,
            stacklevel=3,
        )

    effective_lengths = [min(length, _BRUTE_FORCE_MAX_LEN) for length in lengths]
    return _cycle_counts_brute_force(G, n, effective_lengths)


def _compute_rwpe(
    adj_mat: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    n: int,
    k: int,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Compute Random-Walk Positional Encoding of dimension k.

    rw_mat = D^{-1} adj_mat  (random-walk transition matrix)
    RWPE[v, i] = (rw_mat^i)[v, v]  for i = 1, ..., k.

    Following Dwivedi et al. 2022. Sign-invariant by construction since we
    only use diagonal entries.

    Isolated vertices (degree 0) get a zero row in rw_mat; their RWPE is all zeros.
    """
    deg: np.ndarray[tuple[int], np.dtype[np.float64]] = adj_mat.sum(axis=1)
    deg_safe: np.ndarray[tuple[int], np.dtype[np.float64]] = np.where(deg > 0, deg, 1.0)
    rw_mat: np.ndarray[tuple[int, int], np.dtype[np.float64]] = (
        adj_mat / deg_safe[:, np.newaxis]
    )

    rwpe: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.zeros(
        (n, k), dtype=np.float64
    )
    rw_pow: np.ndarray[tuple[int, int], np.dtype[np.float64]] = rw_mat.copy()
    for i in range(k):
        rwpe[:, i] = np.diag(rw_pow)
        if i < k - 1:
            rw_pow = rw_pow @ rw_mat

    return rwpe


def structural_feature_dim(
    *,
    cycle_lengths: list[int] | None,
    rwpe_dim: int,
) -> int:
    """Return the number of columns that add_structural_features appends.

    One column per entry in cycle_lengths, plus rwpe_dim columns for RWPE.
    """
    n_cycle = len(cycle_lengths) if cycle_lengths else 0
    return n_cycle + rwpe_dim


def add_structural_features(
    data: Data,
    *,
    cycle_lengths: list[int] | None = None,
    rwpe_dim: int = 0,
) -> Data:
    """Return a new Data with cycle counts and/or RWPE concatenated onto data.x.

    cycle_lengths: list like [3, 4, 5, 6]. For each l in the list, computes
        N_l(v) = number of simple cycles of length l through vertex v.
        Computed via brute-force enumeration using networkx.simple_cycles for
        l <= 16; for l > 16 the count is capped and a UserWarning is emitted.

    rwpe_dim: K. Compute RWPE of dimension K following Dwivedi et al. 2022
        (the diagonal of M, M^2, ..., M^K where M = D^{-1} A).
        Sign-invariant by construction. 0 = no RWPE.

    Returns Data with x updated. If data.x is None, creates a leading column
    of ones as a default node feature before appending structural features.

    If cycle_lengths is empty/None and rwpe_dim == 0, returns data unchanged.
    """
    n_cycle = len(cycle_lengths) if cycle_lengths else 0
    if n_cycle == 0 and rwpe_dim == 0:
        return data

    edge_index = cast(torch.Tensor, data.edge_index)
    num_nodes: int = int(cast(int, data.num_nodes))

    # Build NetworkX graph for cycle counting (if needed)
    graph: nx.Graph[int] | None = None
    if n_cycle > 0:
        graph = nx.Graph()
        graph.add_nodes_from(range(num_nodes))
        if edge_index.numel() > 0:
            graph.add_edges_from(edge_index.t().tolist())

    # Build adjacency matrix for RWPE (if needed)
    adj: np.ndarray[tuple[int, int], np.dtype[np.float64]] | None = None
    if rwpe_dim > 0:
        adj_dense: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.zeros(
            (num_nodes, num_nodes), dtype=np.float64
        )
        if edge_index.numel() > 0:
            rows = edge_index[0].cpu().numpy()
            cols = edge_index[1].cpu().numpy()
            for row_i, col_i in zip(rows, cols):
                adj_dense[int(row_i), int(col_i)] += 1.0
        adj = adj_dense

    feature_parts: list[torch.Tensor] = []

    # Existing node features (or ones column if absent)
    if data.x is not None:
        feature_parts.append(cast(torch.Tensor, data.x).float())
    else:
        feature_parts.append(torch.ones((num_nodes, 1), dtype=torch.float))

    # Cycle count features
    if n_cycle > 0 and cycle_lengths is not None and graph is not None:
        counts = _compute_cycle_counts(graph, num_nodes, cycle_lengths)
        for length in cycle_lengths:
            col = torch.tensor(counts[length], dtype=torch.float).unsqueeze(-1)
            feature_parts.append(col)

    # RWPE features
    if rwpe_dim > 0 and adj is not None:
        rwpe = _compute_rwpe(adj, num_nodes, rwpe_dim)
        feature_parts.append(torch.tensor(rwpe, dtype=torch.float))

    new_x = torch.cat(feature_parts, dim=-1)

    # Return a copy of data with x replaced. We copy attribute-by-attribute to
    # preserve all custom fields (k, g_target, group_order, girth, etc.)
    new_data: Data = data.__class__()
    for key in data.keys():
        setattr(new_data, str(key), getattr(data, str(key)))
    new_data.x = new_x
    new_data.num_nodes = num_nodes
    return new_data
