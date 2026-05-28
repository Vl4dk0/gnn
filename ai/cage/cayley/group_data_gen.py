"""Data generation for the group-promise predictor.

Unlike the (misconceived) Cayley *girth* predictor — which predicted the
girth of one specific Cayley graph, a quantity a single BFS already
computes exactly — this module's label is genuinely expensive: the best
girth achievable by ANY degree-k symmetric generating set of a group,
obtained by running the full classical inner search. Prediction is then
cheap. That asymmetry is exactly what makes the model a real search
heuristic — it ranks which GROUPS are worth searching before paying for
the inner search.

A sample = (group G, k, g_target) -> PyG Data:
  * graph: the Cayley graph of G w.r.t. a canonical minimal generating
    set, giving the GNN the group's multiplication structure.
  * node features: (element_order, conjugacy_class_size, is_involution),
    each normalised by |G|.
  * edge features: (generator_order / |G|, generator_is_involution).
  * context: (k, g_target, |G|, log|G|, #involutions, #conjugacy_classes).
  * label: best achievable girth (regression) and whether it reaches
    g_target (binary classification).
"""

from __future__ import annotations

import math
import multiprocessing
import random

import networkx as nx
import torch
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]

from ai.cage.cayley.cayley import build_cayley
from ai.cage.cayley.groups import FiniteGroup, available_groups, element_order
from ai.cage.cayley.search import random_search, tabu_search


def _generated_closure(group: FiniteGroup, gens: list[int]) -> set[int]:
    """All elements reachable from the identity by products of gens."""
    seen: set[int] = {0}
    frontier: list[int] = [0]
    while frontier:
        v = frontier.pop()
        for s in gens:
            u = group.mult(v, s)
            if u not in seen:
                _ = seen.add(u)
                frontier.append(u)
    return seen


def canonical_generating_set(group: FiniteGroup) -> list[int]:
    """Deterministic minimal-greedy symmetric generating set of the group.

    Used only to wire the group-graph fed to the GNN — it is not the
    degree-k search variable. Elements are added in index order until the
    generated subgroup is all of G.
    """
    gens: list[int] = []
    closure: set[int] = {0}
    for a in range(1, group.order):
        if a in closure:
            continue
        gens.append(a)
        a_inv = group.inv(a)
        if a_inv != a:
            gens.append(a_inv)
        closure = _generated_closure(group, gens)
        if len(closure) == group.order:
            break
    return gens


def _element_orders(group: FiniteGroup) -> list[int]:
    """Order of every element of the group."""
    return [element_order(group, a) for a in range(group.order)]


def _conjugacy_data(group: FiniteGroup) -> tuple[list[int], int]:
    """Return (per-element conjugacy-class size, number of conjugacy classes)."""
    order = group.order
    sizes: list[int] = [0] * order
    assigned: list[bool] = [False] * order
    num_classes = 0
    for a in range(order):
        if assigned[a]:
            continue
        members: set[int] = set()
        for g in range(order):
            _ = members.add(group.mult(group.mult(g, a), group.inv(g)))
        size = len(members)
        for m in members:
            sizes[m] = size
            assigned[m] = True
        num_classes += 1
    return sizes, num_classes


def best_achievable_girth(
    group: FiniteGroup,
    k: int,
    g_target: int,
    num_random_trials: int,
    num_tabu_iters: int,
) -> int:
    """Best verified girth over degree-k symmetric generating sets of G.

    This is the expensive label: it runs the classical inner search
    (random + tabu). Returns 0 if no valid degree-k connected Cayley graph
    was found at all.
    """
    best = 0

    _gens_r, girth_r = random_search(
        group, k, g_target, num_trials=num_random_trials, verbose=False
    )
    if isinstance(girth_r, int) or girth_r == float('inf'):
        effective = int(2 * g_target + 1) if girth_r == float('inf') else int(girth_r)
        if effective > best:
            best = effective

    gens_t, girth_t = tabu_search(
        group, k, g_target, num_iterations=num_tabu_iters, verbose=False
    )
    if gens_t is not None:
        effective_t = int(2 * g_target + 1) if girth_t == float('inf') else (int(girth_t) if isinstance(girth_t, int) else 0)
        if effective_t > best:
            cay = build_cayley(group, gens_t)
            if nx.is_connected(cay) and cay.degree(0) == k:
                best = effective_t

    return best


def group_to_pyg(
    group: FiniteGroup,
    k: int,
    g_target: int,
    best_girth: int,
    *,
    gen_set: list[int] | None = None,
    elem_orders: list[int] | None = None,
    conj_sizes: list[int] | None = None,
    num_conj: int | None = None,
) -> Data:
    """Build the PyG Data object for one (group, k, g_target) sample.

    The precomputed gen_set / elem_orders / conj_sizes / num_conj are
    optional; they let a worker share per-group work across targets.
    """
    if gen_set is None:
        gen_set = canonical_generating_set(group)
    if elem_orders is None:
        elem_orders = _element_orders(group)
    if conj_sizes is None or num_conj is None:
        conj_sizes, num_conj = _conjugacy_data(group)

    order = group.order
    inv_order = 1.0 / max(order, 1)

    src: list[int] = []
    dst: list[int] = []
    edge_feats: list[list[float]] = []
    gen_order = {s: elem_orders[s] for s in gen_set}
    for v in range(order):
        for s in gen_set:
            u = group.mult(v, s)
            if u != v:
                src.append(v)
                dst.append(u)
                is_invol = 1.0 if gen_order[s] == 2 else 0.0
                edge_feats.append([gen_order[s] * inv_order, is_invol])

    x = torch.tensor(
        [
            [
                elem_orders[v] * inv_order,
                conj_sizes[v] * inv_order,
                1.0 if elem_orders[v] == 2 else 0.0,
            ]
            for v in range(order)
        ],
        dtype=torch.float,
    )
    edge_index = (
        torch.tensor([src, dst], dtype=torch.long)
        if src
        else torch.zeros((2, 0), dtype=torch.long)
    )
    edge_attr = (
        torch.tensor(edge_feats, dtype=torch.float)
        if edge_feats
        else torch.zeros((0, 2), dtype=torch.float)
    )

    n_invols = sum(1 for o in elem_orders if o == 2)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=order)
    data.k = k
    data.g_target = g_target
    data.group_order = order
    data.log_group_order = math.log(max(order, 2))
    data.num_involutions = n_invols
    data.num_conj = num_conj
    data.girth = int(best_girth)
    data.group_name = group.name
    return data


_GroupResult = tuple[
    int, list[int], list[int], list[int], int, list[tuple[int, int, int]]
]


def _gen_one_group(
    args: tuple[int, FiniteGroup, list[tuple[int, int]], int, int],
) -> _GroupResult:
    """Process-safe worker: compute labels and group invariants for one group.

    Returns only plain Python ints/lists (no torch tensors) — the main
    process builds the PyG Data objects from these. Returning tensors
    from workers triggers Unix file-descriptor passing in multiprocessing
    which fails on PERUN with "received 0 items of ancdata".
    """
    group_idx, group, targets, num_random_trials, num_tabu_iters = args
    elem_orders = _element_orders(group)
    conj_sizes, num_conj = _conjugacy_data(group)
    gen_set = canonical_generating_set(group)

    labels: list[tuple[int, int, int]] = []
    for k, g_target in targets:
        best_girth = best_achievable_girth(
            group, k, g_target, num_random_trials, num_tabu_iters
        )
        labels.append((k, g_target, best_girth))
    return group_idx, gen_set, elem_orders, conj_sizes, num_conj, labels


def generate_group_dataset(
    targets: list[tuple[int, int]],
    max_group_order: int = 600,
    num_random_trials: int = 800,
    num_tabu_iters: int = 800,
    num_workers: int = 1,
    seed: int | None = None,
) -> tuple[list[Data], dict[str, object]]:
    """Generate one labelled sample per (group, target) over the catalogue.

    The label for each sample is produced by the classical inner search,
    so this is CPU-heavy — it is meant to run multi-process on PERUN.
    """
    if not targets:
        raise ValueError("targets must be non-empty")
    if seed is not None:
        random.seed(seed)

    groups = available_groups(max_group_order)
    configs: list[tuple[int, FiniteGroup, list[tuple[int, int]], int, int]] = [
        (i, group, targets, num_random_trials, num_tabu_iters)
        for i, group in enumerate(groups)
    ]

    results: list[_GroupResult] = []
    if num_workers > 1:
        with multiprocessing.Pool(num_workers) as pool:
            for r in pool.imap_unordered(_gen_one_group, configs):
                results.append(r)
    else:
        for cfg in configs:
            results.append(_gen_one_group(cfg))

    # Build PyG Data objects in the main process from the plain-Python
    # invariants returned by the workers.  Keeps torch tensors entirely
    # out of inter-process communication.
    dataset: list[Data] = []
    for group_idx, gen_set, elem_orders, conj_sizes, num_conj, labels in results:
        group = groups[group_idx]
        for k_label, g_label, best_girth in labels:
            dataset.append(
                group_to_pyg(
                    group,
                    k_label,
                    g_label,
                    best_girth,
                    gen_set=gen_set,
                    elem_orders=elem_orders,
                    conj_sizes=conj_sizes,
                    num_conj=num_conj,
                )
            )

    per_target_count: dict[str, int] = {}
    per_target_pos: dict[str, int] = {}
    for d in dataset:
        k_d = int(d.k)  # pyright: ignore[reportAny]
        g_d = int(d.g_target)  # pyright: ignore[reportAny]
        key = f"{k_d}_{g_d}"
        per_target_count[key] = per_target_count.get(key, 0) + 1
        if int(d.girth) >= g_d:  # pyright: ignore[reportAny]
            per_target_pos[key] = per_target_pos.get(key, 0) + 1

    per_target_summary: dict[str, dict[str, float]] = {}
    for key, count in per_target_count.items():
        per_target_summary[key] = {
            "produced": count,
            "pos_rate": per_target_pos.get(key, 0) / count if count > 0 else 0.0,
        }

    stats: dict[str, object] = {
        "produced": len(dataset),
        "num_groups": len(groups),
        "per_target": per_target_summary,
    }
    return dataset, stats


__all__ = [
    "best_achievable_girth",
    "canonical_generating_set",
    "generate_group_dataset",
    "group_to_pyg",
]
