"""Training data generation for the girth predictor.

Generates (base_graph, voltage_assignment, girth) triples as PyG Data
objects.  Each sample encodes a voltage-labeled base graph together with
the girth of the corresponding lift.
"""

from __future__ import annotations

import random
from typing import cast

import torch
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]

from ai.cage.voltage.base_graphs import (
    BaseGraph,
    bouquet,
    cubic_multigraph_4nodes,
    dumbbell,
    prism_base,
)
from ai.cage.voltage.cycle_analysis import compute_lift_girth
from ai.cage.voltage.groups import (
    FiniteGroup,
    cyclic_group,
    dihedral_group,
    direct_product,
)
from backend.utils.graph_utils import moore_bound


def base_graph_to_pyg(
    base: BaseGraph,
    voltages: list[int],
    group: FiniteGroup,
    k: int,
    g_target: int,
    girth: int | float,
) -> Data:
    """Convert a voltage-labeled base graph to a PyG Data object."""
    n = base.num_nodes

    # Node features: [normalized_degree, normalized_index]
    max_deg = max((base.degree(v) for v in range(n)), default=1)
    x = torch.zeros((n, 2), dtype=torch.float)
    for v in range(n):
        x[v, 0] = base.degree(v) / max(max_deg, 1)
        x[v, 1] = v / max(n - 1, 1)

    # Build arc -> voltage mapping
    arc_volt: dict[int, int] = {}
    for edge_pos, fwd_id in enumerate(base.undirected_edge_ids):
        v = voltages[edge_pos]
        rev_id = base.arcs[fwd_id].reverse_id
        arc_volt[fwd_id] = v
        arc_volt[rev_id] = group.inv(v)

    src_list: list[int] = []
    dst_list: list[int] = []
    edge_voltage: list[int] = []
    for arc in base.arcs:
        src_list.append(arc.src)
        dst_list.append(arc.dst)
        edge_voltage.append(arc_volt[arc.arc_id])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(edge_voltage, dtype=torch.long).unsqueeze(-1)

    # Label — infinite girth means degenerate lift
    if isinstance(girth, float):
        girth_int = 0
        girth_class = 0
    else:
        girth_int = int(girth)
        girth_class = 1 if girth_int >= g_target else 0

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=n,
    )
    data.k = k
    data.g_target = g_target
    data.group_order = group.order
    data.girth = girth_int
    data.girth_class = girth_class

    return data


def _candidate_groups(max_order: int) -> list[FiniteGroup]:
    """Generate a list of candidate groups up to a given order, deduplicated by name."""
    groups: list[FiniteGroup] = []
    seen_names: set[str] = set()

    def _add(g: FiniteGroup) -> None:
        if g.name not in seen_names:
            seen_names.add(g.name)
            groups.append(g)

    for n in range(2, max_order + 1):
        _add(cyclic_group(n))

    for n in range(3, max_order // 2 + 1):
        if 2 * n <= max_order:
            _add(dihedral_group(n))

    for a in range(2, min(10, max_order)):
        for b in range(a, min(10, max_order)):
            if a * b <= max_order:
                _add(direct_product(cyclic_group(a), cyclic_group(b)))

    return groups


def _candidate_base_graphs(k: int) -> list[tuple[str, BaseGraph]]:
    """Generate candidate base graphs for degree k, paired with stable names."""
    bases: list[tuple[str, BaseGraph]] = []

    if k == 3:
        bases.append(("dumbbell(3)", dumbbell(3)))
        bases.append(("cubic_4nodes", cubic_multigraph_4nodes()))
        bases.append(("prism", prism_base()))

    bases.append((f"dumbbell({k})", dumbbell(k)))

    if k % 2 == 0:
        bases.append((f"bouquet({k // 2})", bouquet(k // 2)))

    # Deduplicate by name (e.g., dumbbell(3) added twice for k=3)
    seen: set[str] = set()
    unique: list[tuple[str, BaseGraph]] = []
    for name, base in bases:
        if name not in seen:
            seen.add(name)
            unique.append((name, base))
    return unique


def generate_dataset(
    targets: list[tuple[int, int]],
    num_samples: int = 10000,
    max_group_order: int = 60,
    seed: int | None = None,
    max_attempts_multiplier: int = 5,
) -> tuple[list[Data], dict[str, object]]:
    """Generate deduplicated training data for the girth predictor.

    `targets` is a list of (k, g_target) pairs. Each attempt picks a target
    uniformly at random; the produced sample carries that target in its
    Data fields, so a single dataset can train one model across many (k, g).

    Returns (dataset, stats). Stats reports overall counts plus a per-target
    breakdown keyed by "k_g".

    Each sample is keyed by (k, base_name, group_name, voltage_tuple) to
    prevent train/val/test leakage across splits.
    """
    if not targets:
        raise ValueError("targets must be non-empty")

    rng = random.Random(seed)

    base_cache: dict[int, list[tuple[str, BaseGraph]]] = {}
    groups = _candidate_groups(max_group_order)

    moore_cache: dict[tuple[int, int], int] = {
        (k, g): moore_bound(k, g) for (k, g) in targets
    }

    dataset: list[Data] = []
    seen_keys: set[tuple[int, int, str, str, tuple[int, ...]]] = set()
    duplicates_skipped = 0
    attempts = 0
    max_attempts = num_samples * max_attempts_multiplier

    per_target: dict[str, dict[str, int]] = {
        f"{k}_{g}": {"produced": 0, "duplicates_skipped": 0, "positives": 0}
        for (k, g) in targets
    }

    while len(dataset) < num_samples and attempts < max_attempts:
        attempts += 1
        k, g_target = rng.choice(targets)
        target_key = f"{k}_{g_target}"

        if k not in base_cache:
            base_cache[k] = _candidate_base_graphs(k)
        bases = base_cache[k]

        base_name, base = rng.choice(bases)

        mb = moore_cache[(k, g_target)]
        valid_groups = [
            g for g in groups if mb <= base.num_nodes * g.order <= max(4 * mb, mb + 200)
        ]
        if not valid_groups:
            valid_groups = groups

        group = rng.choice(valid_groups)
        n_edges = base.num_undirected_edges()
        volt = [rng.randint(0, group.order - 1) for _ in range(n_edges)]

        key = (k, g_target, base_name, group.name, tuple(volt))
        if key in seen_keys:
            duplicates_skipped += 1
            per_target[target_key]["duplicates_skipped"] += 1
            continue
        seen_keys.add(key)

        girth = compute_lift_girth(base, group, volt, max_girth=2 * g_target)
        data = base_graph_to_pyg(base, volt, group, k, g_target, girth)
        data.base_name = base_name
        data.group_name = group.name

        dataset.append(data)
        per_target[target_key]["produced"] += 1
        if int(cast(int, data.girth_class)) == 1:
            per_target[target_key]["positives"] += 1

    # Add pos_rate to per_target stats
    per_target_out: dict[str, dict[str, float]] = {}
    for tkey, stats_t in per_target.items():
        produced = stats_t["produced"]
        per_target_out[tkey] = {
            "produced": float(produced),
            "duplicates_skipped": float(stats_t["duplicates_skipped"]),
            "pos_rate": round(stats_t["positives"] / max(produced, 1), 4),
        }

    stats: dict[str, object] = {
        "requested": num_samples,
        "produced": len(dataset),
        "duplicates_skipped": duplicates_skipped,
        "attempts": attempts,
        "per_target": per_target_out,
    }
    return dataset, stats
