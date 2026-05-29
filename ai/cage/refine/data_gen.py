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

import os
import random
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from typing import Any, cast

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]

from ai.cage.refine.cost import short_cycle_cost
from ai.cage.refine.swaps import apply_2_switch, enumerate_2_switches
from ai.utils.structural_features import add_structural_features

# Worker spec: primitive types only, no torch tensors across IPC.
# See ai/cage/voltage/supervised/data_gen.py for the failure mode being avoided.


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


def _data_to_spec(
    data: Data, swap_idx: list[int], delta: float, g_target: int
) -> dict[str, Any]:
    x = cast(torch.Tensor, data.x)
    edge_index = cast(torch.Tensor, data.edge_index)
    return {
        "x": x.cpu().numpy(),
        "edge_index": edge_index.cpu().numpy(),
        "num_nodes": int(cast(int, data.num_nodes)),
        "swap_idx": swap_idx,
        "delta": delta,
        "g_target": g_target,
    }


def _spec_to_data(spec: dict[str, Any]) -> Data:
    data = Data(
        x=torch.from_numpy(cast(np.ndarray, spec["x"])).float(),
        edge_index=torch.from_numpy(cast(np.ndarray, spec["edge_index"])).long(),
        num_nodes=spec["num_nodes"],
    )
    data.swap_idx = torch.tensor(spec["swap_idx"], dtype=torch.long)
    data.delta = torch.tensor(spec["delta"], dtype=torch.float)
    data.g_target = spec["g_target"]
    return data


def _generate_one_sample(
    sample_seed: int,
    k_range: tuple[int, int],
    n_range: tuple[int, int],
    g_target_range: tuple[int, int],
    cycle_lengths: list[int],
    rwpe_dim: int,
    max_inner_attempts: int = 20,
) -> dict[str, Any] | None:
    """Worker: produce one labeled spec dict, or None on failure."""
    rng = random.Random(sample_seed)

    for _ in range(max_inner_attempts):
        k = rng.randint(k_range[0], k_range[1])
        n = rng.randint(n_range[0], n_range[1])
        g_target = rng.randint(g_target_range[0], g_target_range[1])
        graph_seed = rng.randint(0, 2**31 - 1)

        try:
            G = _random_k_regular(k, n, seed=graph_seed)
        except nx.NetworkXError:
            continue

        base_cost = short_cycle_cost(G, g_target)

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
        nodes = sorted(G.nodes())
        node_idx = {v: i for i, v in enumerate(nodes)}

        data = _graph_to_pyg(G, cycle_lengths, rwpe_dim)
        swap_idx = [
            node_idx[swap.u],
            node_idx[swap.v],
            node_idx[swap.x],
            node_idx[swap.y],
        ]
        return _data_to_spec(data, swap_idx, float(delta), g_target)

    return None


def generate_dataset(
    num_samples: int,
    k_range: tuple[int, int] = (3, 5),
    n_range: tuple[int, int] = (20, 100),
    g_target_range: tuple[int, int] = (5, 8),
    cycle_lengths: list[int] | None = None,
    rwpe_dim: int = 8,
    seed: int = 42,
    workers: int | None = None,
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

    if workers is None:
        workers = max(os.cpu_count() or 1, 1)

    rng = random.Random(seed)
    dataset: list[Data] = []

    if workers <= 1:
        attempts = 0
        max_attempts = num_samples * 2
        while len(dataset) < num_samples and attempts < max_attempts:
            attempts += 1
            sample = _generate_one_sample(
                rng.randint(0, 2**31 - 1),
                k_range,
                n_range,
                g_target_range,
                cycle_lengths,
                rwpe_dim,
            )
            if sample is not None:
                dataset.append(_spec_to_data(sample))
        return dataset

    # Streamed submission: keep ~workers*4 futures in flight at any time,
    # refill as they complete. Submitting all 2*num_samples up-front
    # exhausts memory and file descriptors for large num_samples.
    max_in_flight = workers * 4
    max_attempts = num_samples * 2
    attempts_submitted = 0
    in_flight: set[Future[dict[str, Any] | None]] = set()

    def _submit() -> None:
        nonlocal attempts_submitted
        fut = pool.submit(
            _generate_one_sample,
            rng.randint(0, 2**31 - 1),
            k_range,
            n_range,
            g_target_range,
            cycle_lengths,
            rwpe_dim,
        )
        in_flight.add(fut)
        attempts_submitted += 1

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for _ in range(min(max_in_flight, max_attempts)):
            _submit()

        while in_flight and len(dataset) < num_samples:
            done = next(as_completed(in_flight))
            in_flight.discard(done)
            sample = done.result()
            if sample is not None:
                dataset.append(_spec_to_data(sample))
            if len(dataset) < num_samples and attempts_submitted < max_attempts:
                _submit()

        for fut in in_flight:
            _ = fut.cancel()

    return dataset[:num_samples]
