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
  1. Voltage lift near-misses: lifts whose girth is g_target-1 or g_target-2
     (exactly the kind of graph TabuRefiner sees during deployment).
  2. Random k-regular graphs via nx.random_regular_graph (diversity minority).

The lift-to-random ratio is controlled by ``lift_fraction`` (default 0.7).
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
from ai.cage.voltage.base_graphs import BaseGraph, dumbbell
from ai.cage.voltage.groups import FiniteGroup, cyclic_group, dihedral_group
from ai.cage.voltage.lift import build_lift
from ai.utils.r_neighborhood import apply_r_neighborhood
from ai.utils.structural_features import add_structural_features
from backend.utils.graph_utils import compute_girth

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


def _random_lift_near_miss(
    k: int,
    g_target: int,
    rng: random.Random,
    max_attempts: int = 30,
) -> nx.Graph[int] | None:
    """Try to produce a voltage lift near-miss for degree k, target girth g_target.

    A near-miss is a lift whose girth is in [g_target-2, g_target-1] (i.e. it
    has short cycles, which is exactly the situation TabuRefiner faces).

    Strategy: dumbbell base with cyclic or dihedral group, random voltages.
    We try several random assignments and return the first near-miss found.
    Returns None if no near-miss is found within max_attempts.
    """
    base: BaseGraph = dumbbell(k)
    n_free = base.num_undirected_edges()

    min_girth = max(3, g_target - 2)
    max_girth = g_target - 1  # strictly below target = still broken

    for _ in range(max_attempts):
        # Vary group order; prefer sizes that yield tractable lifts
        group_order = rng.choice([6, 7, 8, 10, 12, 14, 16, 18, 20])
        # Alternate between cyclic and dihedral for diversity
        group: FiniteGroup
        if rng.random() < 0.5:
            group = cyclic_group(group_order)
        else:
            # dihedral has order 2*n; pick half-orders
            half = max(3, group_order // 2)
            group = dihedral_group(half)

        voltages = [rng.randint(0, group.order - 1) for _ in range(n_free)]
        try:
            lift = build_lift(base, group, voltages)
        except Exception:
            continue

        girth = compute_girth(lift)
        if isinstance(girth, float):
            continue  # infinite girth (tree) or acyclic — not useful
        if min_girth <= girth <= max_girth:
            return lift

    return None


def graph_to_pyg(
    G: nx.Graph[int],
    cycle_lengths: list[int],
    rwpe_dim: int,
    r: int | None = None,
) -> Data:
    """Convert a NetworkX graph to a PyG Data with structural features.

    Parameters
    ----------
    G:
        Input NetworkX graph.
    cycle_lengths:
        Cycle lengths for structural features.
    rwpe_dim:
        RWPE dimension for structural features.
    r:
        If not None, attach Loopy r-neighborhood tensors (loopyN{L} /
        loopyA{L}) required by the loopy backbone.
    """
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
    data = add_structural_features(
        data,
        cycle_lengths=cycle_lengths,
        rwpe_dim=rwpe_dim,
    )
    if r is not None:
        data = apply_r_neighborhood(data, r=r)
    return data


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


def _make_sample_from_graph(
    G: nx.Graph[int],
    g_target: int,
    rng: random.Random,
    cycle_lengths: list[int],
    rwpe_dim: int,
    r: int | None = None,
) -> dict[str, Any] | None:
    """Given a graph G (already selected), produce one labeled spec dict."""
    swaps = list(enumerate_2_switches(G, sample_size=20))
    if not swaps:
        return None
    swap = rng.choice(swaps)
    try:
        G2 = apply_2_switch(G, swap)
        base_cost = short_cycle_cost(G, g_target)
        new_cost = short_cycle_cost(G2, g_target)
    except Exception:
        return None

    delta = new_cost - base_cost
    nodes = sorted(G.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}
    data = graph_to_pyg(G, cycle_lengths, rwpe_dim, r=r)
    swap_idx = [
        node_idx[swap.u],
        node_idx[swap.v],
        node_idx[swap.x],
        node_idx[swap.y],
    ]
    return _data_to_spec(data, swap_idx, float(delta), g_target)


def _generate_one_sample(
    sample_seed: int,
    k_range: tuple[int, int],
    n_range: tuple[int, int],
    g_target_range: tuple[int, int],
    cycle_lengths: list[int],
    rwpe_dim: int,
    lift_fraction: float,
    r: int | None = None,
    max_inner_attempts: int = 20,
) -> dict[str, Any] | None:
    """Worker: produce one labeled spec dict, or None on failure.

    With probability ``lift_fraction`` the graph is sourced from a voltage
    lift near-miss (girth = g_target-1 or g_target-2); otherwise a random
    k-regular graph is used for diversity.
    """
    rng = random.Random(sample_seed)

    for _ in range(max_inner_attempts):
        k = rng.randint(k_range[0], k_range[1])
        g_target = rng.randint(g_target_range[0], g_target_range[1])

        graph: nx.Graph[int] | None = None

        if rng.random() < lift_fraction:
            # Attempt lift near-miss
            graph = _random_lift_near_miss(k, g_target, rng)

        if graph is None:
            # Fallback to random k-regular
            n = rng.randint(n_range[0], n_range[1])
            graph_seed = rng.randint(0, 2**31 - 1)
            try:
                graph = _random_k_regular(k, n, seed=graph_seed)
            except nx.NetworkXError:
                continue

        spec = _make_sample_from_graph(
            graph, g_target, rng, cycle_lengths, rwpe_dim, r=r
        )
        if spec is not None:
            return spec

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
    lift_fraction: float = 0.7,
    r: int | None = None,
) -> list[Data]:
    """Generate a supervised dataset of (graph, swap, Δcost) tuples.

    Parameters
    ----------
    num_samples:
        Number of (graph, swap, Δcost) samples to generate.
    k_range:
        (min_k, max_k) inclusive for random k-regular graph degree.
    n_range:
        (min_n, max_n) inclusive for graph order (used for random-regular
        fallback only; lift sizes are determined by group order).
    g_target_range:
        (min_g, max_g) inclusive for g_target used in cost evaluation.
    cycle_lengths:
        Cycle lengths for structural features.  Default: [3,4,5,6,7,8].
    rwpe_dim:
        RWPE dimension for structural features.
    seed:
        Random seed for reproducibility.
    lift_fraction:
        Fraction of samples to source from voltage lift near-misses (the
        rest come from random k-regular graphs for diversity).  Default 0.7.
    r:
        If not None, attach Loopy r-neighborhood tensors (loopyN{L} /
        loopyA{L}) to each sample.  Required when training with
        backbone="loopy".

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
        max_attempts = num_samples * 3
        while len(dataset) < num_samples and attempts < max_attempts:
            attempts += 1
            sample = _generate_one_sample(
                rng.randint(0, 2**31 - 1),
                k_range,
                n_range,
                g_target_range,
                cycle_lengths,
                rwpe_dim,
                lift_fraction,
                r=r,
            )
            if sample is not None:
                dataset.append(_spec_to_data(sample))
        return dataset

    # Streamed submission: keep ~workers*4 futures in flight at any time,
    # refill as they complete. Submitting all 3*num_samples up-front
    # exhausts memory and file descriptors for large num_samples.
    max_in_flight = workers * 4
    max_attempts = num_samples * 3
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
            lift_fraction,
            r,
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
