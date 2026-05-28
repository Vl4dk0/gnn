"""Training data generation for the girth predictor.

Generates (base_graph, voltage_assignment, girth) triples as PyG Data
objects.  Each sample encodes a voltage-labeled base graph together with
the girth of the corresponding lift.
"""

from __future__ import annotations

import os
import random
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from typing import Any, cast

import numpy as np
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
from ai.cage.voltage.lift import build_lift
from ai.cage.voltage.groups import (
    FiniteGroup,
    cyclic_group,
    dihedral_group,
    direct_product,
)
from ai.utils.structural_features import add_structural_features
from backend.utils.graph_utils import is_k_regular, moore_bound

# Worker return type: a dedup key plus a "spec" dict of native python /
# numpy values that reconstructs a Data object. We deliberately avoid
# sending torch.Tensors across the IPC boundary because torch registers
# shared-memory reducers on import, and at scale that exhausts FDs and
# mmap mappings (RuntimeError: Cannot allocate memory).
_DedupKey = tuple[int, int, str, str, tuple[int, ...]]


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

    # Label — infinite girth means girth > 2*g_target, treat as lower bound
    girth_int = 2 * g_target if isinstance(girth, float) else int(girth)

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
    if isinstance(girth, float):
        # girth=inf means girth > 2*g_target >= g_target: positive
        data.girth_class = 1
    elif girth_int >= g_target:
        # finite high girth: only positive if lift is actually k-regular
        lift_graph = build_lift(base, group, voltages)
        data.girth_class = 1 if is_k_regular(lift_graph, k) else 0
    else:
        data.girth_class = 0

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


def _data_to_spec(data: Data, base_name: str, group_name: str) -> dict[str, Any]:
    """Convert a Data object into a primitive dict (numpy arrays + python scalars).

    This is what crosses the IPC boundary instead of the Data object itself.
    """
    x = cast(torch.Tensor, data.x)
    edge_index = cast(torch.Tensor, data.edge_index)
    edge_attr = data.edge_attr
    return {
        "x": x.cpu().numpy(),
        "edge_index": edge_index.cpu().numpy(),
        "edge_attr": (
            cast(torch.Tensor, edge_attr).cpu().numpy()
            if edge_attr is not None
            else None
        ),
        "num_nodes": int(cast(int, data.num_nodes)),
        "k": int(cast(int, data.k)),
        "g_target": int(cast(int, data.g_target)),
        "group_order": int(cast(int, data.group_order)),
        "girth": int(cast(int, data.girth)),
        "base_name": base_name,
        "group_name": group_name,
    }


def _spec_to_data(spec: dict[str, Any]) -> Data:
    """Inverse of _data_to_spec; runs in the main process."""
    edge_attr_np = spec["edge_attr"]
    data = Data(
        x=torch.from_numpy(cast(np.ndarray, spec["x"])).float(),
        edge_index=torch.from_numpy(cast(np.ndarray, spec["edge_index"])).long(),
        edge_attr=(
            torch.from_numpy(cast(np.ndarray, edge_attr_np)).long()
            if edge_attr_np is not None
            else None
        ),
        num_nodes=spec["num_nodes"],
    )
    data.k = spec["k"]
    data.g_target = spec["g_target"]
    data.group_order = spec["group_order"]
    data.girth = spec["girth"]
    data.base_name = spec["base_name"]
    data.group_name = spec["group_name"]
    return data


def _generate_chunk(
    chunk_seed: int,
    chunk_size: int,
    targets: list[tuple[int, int]],
    max_group_order: int,
    cycle_lengths: list[int] | None,
    rwpe_dim: int,
) -> list[tuple[_DedupKey, dict[str, Any]]]:
    """Worker: produce `chunk_size` candidate (dedup_key, spec_dict) pairs.

    Workers do the expensive work (girth detection + structural features) and
    convert the resulting Data to a primitive spec dict so IPC pickling does
    not involve torch tensors.
    """
    rng = random.Random(chunk_seed)
    base_cache: dict[int, list[tuple[str, BaseGraph]]] = {}
    groups = _candidate_groups(max_group_order)
    moore_cache: dict[tuple[int, int], int] = {
        (k, g): moore_bound(k, g) for (k, g) in targets
    }

    out: list[tuple[_DedupKey, dict[str, Any]]] = []
    for _ in range(chunk_size):
        k, g_target = rng.choice(targets)
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

        lift_graph = build_lift(base, group, volt)
        if not is_k_regular(lift_graph, k):
            continue
        girth = compute_lift_girth(base, group, volt, max_girth=2 * g_target)
        data = base_graph_to_pyg(base, volt, group, k, g_target, girth)
        if cycle_lengths or rwpe_dim > 0:
            data = add_structural_features(
                data, cycle_lengths=cycle_lengths, rwpe_dim=rwpe_dim
            )

        out.append((key, _data_to_spec(data, base_name, group.name)))

    return out


def generate_dataset(
    targets: list[tuple[int, int]],
    num_samples: int = 10000,
    max_group_order: int = 60,
    seed: int | None = None,
    max_attempts_multiplier: int = 5,
    cycle_lengths: list[int] | None = None,
    rwpe_dim: int = 0,
    workers: int | None = None,
) -> tuple[list[Data], dict[str, object]]:
    """Generate deduplicated training data for the girth predictor.

    `targets` is a list of (k, g_target) pairs. Each attempt picks a target
    uniformly at random; the produced sample carries that target in its
    Data fields, so a single dataset can train one model across many (k, g).

    Returns (dataset, stats). Stats reports overall counts plus a per-target
    breakdown keyed by "k_g".

    Each sample is keyed by (k, g_target, base_name, group_name, voltage_tuple)
    to prevent train/val/test leakage across splits without dropping the same
    lift evaluated under a different girth target.
    """
    if not targets:
        raise ValueError("targets must be non-empty")

    if workers is None:
        workers = max(os.cpu_count() or 1, 1)

    rng = random.Random(seed)

    dataset: list[Data] = []
    seen_keys: set[tuple[int, int, str, str, tuple[int, ...]]] = set()
    duplicates_skipped = 0
    attempts = 0
    max_attempts = num_samples * max_attempts_multiplier

    per_target: dict[str, dict[str, int]] = {
        f"{k}_{g}": {"produced": 0, "duplicates_skipped": 0, "positives": 0}
        for (k, g) in targets
    }

    # Submit chunks of work; each chunk produces `chunk_size` candidates.
    # Cap chunk size at 256 to bound the number of spec dicts buffered.
    chunk_size = min(max(64, num_samples // max(workers * 16, 1)), 256)

    def _ingest(candidates: list[tuple[_DedupKey, dict[str, Any]]]) -> None:
        nonlocal duplicates_skipped
        for key, spec in candidates:
            if len(dataset) >= num_samples:
                return
            attempts_local_target = f"{key[0]}_{key[1]}"
            if key in seen_keys:
                duplicates_skipped += 1
                per_target[attempts_local_target]["duplicates_skipped"] += 1
                continue
            seen_keys.add(key)
            data = _spec_to_data(spec)
            dataset.append(data)
            per_target[attempts_local_target]["produced"] += 1
            if int(spec["girth"]) >= int(spec["g_target"]):
                per_target[attempts_local_target]["positives"] += 1

    if workers <= 1:
        # Serial fallback (also exercised by tests).
        while len(dataset) < num_samples and attempts < max_attempts:
            batch = _generate_chunk(
                rng.randint(0, 2**31 - 1),
                chunk_size,
                targets,
                max_group_order,
                cycle_lengths,
                rwpe_dim,
            )
            attempts += len(batch)
            _ingest(batch)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            in_flight: set[Future[list[tuple[_DedupKey, dict[str, Any]]]]] = set()

            def _submit() -> None:
                fut = pool.submit(
                    _generate_chunk,
                    rng.randint(0, 2**31 - 1),
                    chunk_size,
                    targets,
                    max_group_order,
                    cycle_lengths,
                    rwpe_dim,
                )
                in_flight.add(fut)

            for _ in range(workers * 2):
                _submit()

            while in_flight and len(dataset) < num_samples and attempts < max_attempts:
                done = next(as_completed(in_flight))
                in_flight.discard(done)
                batch = done.result()
                attempts += len(batch)
                _ingest(batch)
                if len(dataset) < num_samples and attempts < max_attempts:
                    _submit()

            for fut in in_flight:
                _ = fut.cancel()

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
