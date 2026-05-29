"""Training data generation for the Cayley girth predictor.

A sample = (group G, symmetric generating set S, k, g_target, true girth).

The graph fed to the GNN is the *truncated Cayley ball* of radius
floor(g_target / 2) + 1 around the identity in Gamma(G, S). For most
useful (k, g) targets the full Cayley graph is small enough to fit
anyway, but truncation keeps cost predictable for larger groups.

Node features: normalized distance from identity.
Edge features: (generator_order / group_order, is_involution).
Graph context: k, g_target, group_order, num_involutions_in_S.
"""

from __future__ import annotations

import math
import random
from collections import deque

import torch
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]

from ai.cage.cayley.cayley import cayley_girth
from ai.cage.cayley.groups import (
    FiniteGroup,
    available_groups,
    element_order,
    random_generating_set,
)
from backend.utils.graph_utils import moore_bound


def cayley_ball_to_pyg(
    group: FiniteGroup,
    generators: list[int],
    k: int,
    g_target: int,
    girth: int | float,
    radius: int | None = None,
) -> Data:
    """Build a PyG Data object representing the truncated Cayley ball at identity."""
    if radius is None:
        radius = max(2, g_target // 2 + 1)

    dist: dict[int, int] = {0: 0}
    queue: deque[int] = deque([0])
    while queue:
        v = queue.popleft()
        if dist[v] >= radius:
            continue
        for s in generators:
            u = group.mult(v, s)
            if u not in dist:
                dist[u] = dist[v] + 1
                queue.append(u)

    vertices = list(dist.keys())
    v_to_idx = {v: i for i, v in enumerate(vertices)}

    # Per-generator features (constant for the whole graph but stored on edges).
    gen_order = [element_order(group, s) for s in generators]
    gen_is_invol = [1.0 if group.mult(s, s) == 0 else 0.0 for s in generators]

    src: list[int] = []
    dst: list[int] = []
    edge_feats: list[list[float]] = []
    for v in vertices:
        for i, s in enumerate(generators):
            u = group.mult(v, s)
            if u in v_to_idx and u != v:
                src.append(v_to_idx[v])
                dst.append(v_to_idx[u])
                edge_feats.append([gen_order[i] / max(group.order, 1), gen_is_invol[i]])

    x = torch.tensor([[dist[v] / max(radius, 1)] for v in vertices], dtype=torch.float)
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

    if isinstance(girth, float):
        girth_int = 2 * g_target + 2  # lower-bound estimate of actual girth
    else:
        girth_int = int(girth)

    n_invols = int(sum(gen_is_invol))

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(vertices),
    )
    data.k = k
    data.g_target = g_target
    data.group_order = group.order
    data.log_group_order = math.log(max(group.order, 2))
    data.num_involutions = n_invols
    data.girth = girth_int
    return data


def generate_dataset(
    targets: list[tuple[int, int]],
    num_samples: int = 50000,
    max_group_order: int = 200,
    seed: int | None = None,
    max_attempts_multiplier: int = 5,
) -> tuple[list[Data], dict[str, object]]:
    """Generate (G, S, k, g_target, girth) samples as PyG Data objects.

    Deduplicates by (k, g_target, group_name, sorted-tuple-of-S) to prevent
    leakage between train/val/test splits.
    """
    if not targets:
        raise ValueError("targets must be non-empty")

    rng = random.Random(seed)
    if seed is not None:
        random.seed(seed)

    groups = available_groups(max_group_order)

    per_target_count: dict[tuple[int, int], int] = {t: 0 for t in targets}
    per_target_pos: dict[tuple[int, int], int] = {t: 0 for t in targets}
    per_target_dups: dict[tuple[int, int], int] = {t: 0 for t in targets}

    seen: set[tuple[int, int, str, tuple[int, ...]]] = set()
    dataset: list[Data] = []
    attempts = 0
    max_attempts = num_samples * max_attempts_multiplier

    while len(dataset) < num_samples and attempts < max_attempts:
        attempts += 1
        k, g_target = rng.choice(targets)

        # Bias toward groups whose order is in the cage-plausible range
        # for this target. Falls back to all groups if none match.
        mb = moore_bound(k, g_target)
        plausible = [g for g in groups if mb <= g.order <= max(4 * mb, mb + 200)]
        if not plausible:
            plausible = groups

        group = rng.choice(plausible)
        gens = random_generating_set(group, k)
        if gens is None or len(gens) != k:
            continue

        key = (k, g_target, group.name, tuple(sorted(gens)))
        if key in seen:
            per_target_dups[(k, g_target)] += 1
            continue
        _ = seen.add(key)

        girth = cayley_girth(group, gens, max_girth=2 * g_target + 2)
        data = cayley_ball_to_pyg(group, gens, k, g_target, girth)
        data.base_name = group.name
        data.gen_tuple = list(gens)

        dataset.append(data)
        per_target_count[(k, g_target)] += 1
        if int(data.girth) >= g_target:
            per_target_pos[(k, g_target)] += 1

    per_target_summary: dict[str, dict[str, float]] = {}
    for (k, g), count in per_target_count.items():
        per_target_summary[f"{k}_{g}"] = {
            "produced": count,
            "pos_rate": (per_target_pos[(k, g)] / count) if count > 0 else 0.0,
            "duplicates_skipped": per_target_dups[(k, g)],
        }

    stats: dict[str, object] = {
        "requested": num_samples,
        "produced": len(dataset),
        "attempts": attempts,
        "duplicates_skipped": sum(per_target_dups.values()),
        "per_target": per_target_summary,
    }
    return dataset, stats


__all__ = ["cayley_ball_to_pyg", "generate_dataset"]
