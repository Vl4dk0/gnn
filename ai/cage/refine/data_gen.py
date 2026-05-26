"""Supervised dataset generation for MoveOracle training.

Each training sample is a tuple (graph, swap, Δcost) where:
  - graph: a k-regular graph with some short cycles (a "broken" graph)
  - swap: a random valid 2-switch
  - Δcost: short_cycle_cost(apply_2_switch(graph, swap)) - short_cycle_cost(graph)

The dataset is returned as a list of PyG Data objects, each carrying:
  - x: node features (ones + structural features)
  - edge_index: graph connectivity
  - swap_idx: LongTensor [4] with node indices (u, v, x, y)
  - delta: FloatTensor scalar (target Δcost)

Source graphs are sampled from:
  1. Random k-regular graphs via nx.random_regular_graph.
  2. Voltage lift outputs are not included at data-gen time (used at
     inference time by the tabu pipeline).
"""

from __future__ import annotations

import random

import networkx as nx
import torch
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]

from ai.cage.refine.cost import short_cycle_cost
from ai.cage.refine.swaps import apply_2_switch, enumerate_2_switches
from ai.utils.structural_features import add_structural_features


def _random_k_regular(
    k: int,
    n: int,
    seed: int | None = None,
) -> nx.Graph[int]:
    """Return a random k-regular graph on n vertices.

    nx.random_regular_graph requires k*n to be even.  We increment n by 1
    if necessary.
    """
    rng = random.Random(seed)
    if (k * n) % 2 != 0:
        n += 1
    # nx.random_regular_graph uses its own RNG seed
    seed_val = rng.randint(0, 2**31 - 1)
    return nx.random_regular_graph(k, n, seed=seed_val)  # type: ignore[return-value]


def _graph_to_pyg(
    G: nx.Graph[int],
    cycle_lengths: list[int],
    rwpe_dim: int,
) -> Data:
    """Convert a NetworkX graph to a PyG Data with structural features."""
    nodes = sorted(G.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)

    edges = list(G.edges())
    if edges:
        src = [node_idx[u] for u, _ in edges] + [node_idx[v] for _, v in edges]
        dst = [node_idx[v] for _, v in edges] + [node_idx[u] for u, _ in edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    data = Data(
        x=torch.ones((n, 1), dtype=torch.float),
        edge_index=edge_index,
        num_nodes=n,
    )
    return add_structural_features(
        data,
        cycle_lengths=cycle_lengths,
        rwpe_dim=rwpe_dim,
    )


def generate_dataset(
    num_samples: int,
    k_range: tuple[int, int] = (3, 5),
    n_range: tuple[int, int] = (20, 100),
    g_target_range: tuple[int, int] = (5, 8),
    cycle_lengths: list[int] | None = None,
    rwpe_dim: int = 8,
    seed: int = 42,
) -> list[Data]:
    """Generate a supervised dataset of (graph, swap, Δcost) tuples.

    Parameters
    ----------
    num_samples:
        Number of (graph, swap, Δcost) samples to generate.
    k_range:
        (min_k, max_k) inclusive for random k-regular graph degree.
    n_range:
        (min_n, max_n) inclusive for graph order.
    g_target_range:
        (min_g, max_g) inclusive for g_target used in cost evaluation.
    cycle_lengths:
        Cycle lengths for structural features.  Default: [3,4,5,6,7,8].
    rwpe_dim:
        RWPE dimension for structural features.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    List of PyG Data objects with fields:
        x           — node features [n, feat_dim]
        edge_index  — [2, 2m]
        swap_idx    — LongTensor [4]  (u, v, x, y node indices)
        delta       — FloatTensor scalar (Δcost target)
        g_target    — int (used for cost evaluation)
    """
    if cycle_lengths is None:
        cycle_lengths = [3, 4, 5, 6, 7, 8]

    rng = random.Random(seed)
    dataset: list[Data] = []
    attempts = 0
    max_attempts = num_samples * 20

    while len(dataset) < num_samples and attempts < max_attempts:
        attempts += 1

        k = rng.randint(k_range[0], k_range[1])
        n = rng.randint(n_range[0], n_range[1])
        g_target = rng.randint(g_target_range[0], g_target_range[1])
        graph_seed = rng.randint(0, 2**31 - 1)

        try:
            G = _random_k_regular(k, n, seed=graph_seed)
        except nx.NetworkXError:
            continue

        base_cost = short_cycle_cost(G, g_target)

        # Pick a random valid 2-switch
        swaps = list(enumerate_2_switches(G, sample_size=20))
        if not swaps:
            continue

        swap = rng.choice(swaps)
        try:
            G2 = apply_2_switch(G, swap)
            new_cost = short_cycle_cost(G2, g_target)
        except Exception:
            continue

        delta = new_cost - base_cost

        # Build PyG data for the original graph
        nodes = sorted(G.nodes())
        node_idx = {v: i for i, v in enumerate(nodes)}

        data = _graph_to_pyg(G, cycle_lengths, rwpe_dim)
        data.swap_idx = torch.tensor(
            [node_idx[swap.u], node_idx[swap.v], node_idx[swap.x], node_idx[swap.y]],
            dtype=torch.long,
        )
        data.delta = torch.tensor(delta, dtype=torch.float)
        data.g_target = g_target

        dataset.append(data)

    return dataset
