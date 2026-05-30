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

import networkx as nx
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
from ai.cage.voltage.cycle_analysis import (
    compute_lift_girth,
    count_short_identity_walks,
)
from ai.cage.voltage.lift import build_lift
from ai.cage.voltage.groups import (
    FiniteGroup,
    cyclic_group,
    dihedral_group,
    direct_product,
    semidirect_product_cyclic,
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

    # Label — infinite girth means girth > 2*g_target, treat as lower bound.
    # The label is stored in `data.girth` regardless of target kind so the GNN
    # code is unchanged; for tabu_cost callers pass the walk count as `girth`.
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

    # Non-abelian semidirect products Z_n ⋊ Z_m. Abelian groups (cyclic,
    # dihedral, direct products) cannot host high-girth lifts for hard targets
    # like (5, 7) — the trajectory stalls at girth 6 — so we add the same
    # non-abelian family the cage search relies on for those targets.
    for n in range(3, min(30, max_order + 1)):
        for m in range(2, min(10, max_order + 1)):
            if n * m > max_order:
                continue
            for phi in range(2, n):
                if pow(phi, m, n) == 1:
                    _add(semidirect_product_cyclic(n, m, phi))
                    break  # one nontrivial action per (n, m) is enough

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


def _tabu_trajectory(
    base: BaseGraph,
    group: FiniteGroup,
    g_target: int,
    rng: random.Random,
    num_iterations: int,
    tabu_tenure: int = 10,
    candidate_cap: int = 8,
) -> list[list[int]]:
    """Run a tabu descent on the short-walk cost, logging every visited assignment.

    Mirrors the tabu search in supervised/search.py (spanning-tree
    normalization, single-edge moves, aspiration) but instead of returning only
    the best, it records each accepted assignment along the trajectory. These
    concentrate near high girth (cost 0 means girth >= g_target), giving the
    predictor the positive examples that uniform-random sampling almost never
    produces for hard targets.

    Per move, at most ``candidate_cap`` new values are tried per free edge
    (sampled when the group is larger). This bounds the per-iteration cost
    independently of group order, which is essential for large groups (e.g. the
    order-100 groups needed for (5, 7)); a full scan there is intractable at
    data-generation scale.

    Returns a list of full-length voltage vectors (tree edges fixed to 0).
    """
    free_indices = base.free_edge_indices()
    m = base.num_undirected_edges()
    order = group.order

    volts = [0] * m
    for idx in free_indices:
        volts[idx] = rng.randint(0, order - 1)

    if not free_indices:
        return [volts[:]]

    visited: list[list[int]] = [volts[:]]
    current_cost = count_short_identity_walks(base, group, volts, g_target)
    best_cost = current_cost

    tabu: dict[tuple[int, int], int] = {}

    for iteration in range(num_iterations):
        best_move_cost = current_cost + 1
        best_move_edge = -1
        best_move_val = -1

        # Cost 0 here may be a genuine high-girth assignment OR a degenerate
        # disconnected lift (also walk-free). Either way we keep moving: real
        # positives are still harvested via _make_sample's exact-girth label,
        # and we escape degenerate basins instead of stalling on them.
        if current_cost > 0:
            for edge_idx in free_indices:
                old_val = volts[edge_idx]
                if order - 1 <= candidate_cap:
                    candidate_vals = [v for v in range(order) if v != old_val]
                else:
                    candidate_vals = rng.sample(
                        [v for v in range(order) if v != old_val], candidate_cap
                    )
                for new_val in candidate_vals:
                    is_tabu = tabu.get((edge_idx, old_val), -1) > iteration
                    volts[edge_idx] = new_val
                    new_cost = count_short_identity_walks(base, group, volts, g_target)
                    volts[edge_idx] = old_val
                    if new_cost < best_cost or (
                        not is_tabu and new_cost < best_move_cost
                    ):
                        best_move_cost = new_cost
                        best_move_edge = edge_idx
                        best_move_val = new_val

        if best_move_edge < 0:
            # Local optimum (or cost-0 basin): perturb with a random move so the
            # trajectory keeps harvesting diverse near-feasible assignments.
            best_move_edge = rng.choice(free_indices)
            best_move_val = rng.randint(0, order - 1)

        old_val = volts[best_move_edge]
        volts[best_move_edge] = best_move_val
        tabu[(best_move_edge, old_val)] = iteration + tabu_tenure
        current_cost = count_short_identity_walks(base, group, volts, g_target)
        best_cost = min(best_cost, current_cost)
        visited.append(volts[:])

    return visited


def _make_sample(
    base: BaseGraph,
    base_name: str,
    group: FiniteGroup,
    volt: list[int],
    k: int,
    g_target: int,
    cycle_lengths: list[int] | None,
    rwpe_dim: int,
    kind: str = "girth",
) -> tuple[_DedupKey, dict[str, Any]] | None:
    """Label one voltage assignment and pack it for IPC.

    The label depends on ``kind``:
      - "girth":     the exact girth of the lift (higher = better).
      - "tabu_cost": count_short_identity_walks (the dense cost the tabu search
                     minimizes; 0 iff girth >= g_target, lower = better).
    Either way it is stored in the ``girth`` field so the model code is unchanged.

    Returns None for degenerate lifts (not k-regular, or disconnected). A
    disconnected lift has no short identity walk and would otherwise be labeled
    as high-girth / zero-cost, teaching the model that disconnection is good. The
    search itself only accepts connected k-regular lifts, so we mirror that.
    """
    key = (k, g_target, base_name, group.name, tuple(volt))
    lift_graph = build_lift(base, group, volt)
    if not is_k_regular(lift_graph, k):
        return None
    if lift_graph.number_of_nodes() == 0 or not nx.is_connected(lift_graph):
        return None
    if kind == "tabu_cost":
        label: int | float = count_short_identity_walks(base, group, volt, g_target)
    else:
        label = compute_lift_girth(base, group, volt, max_girth=2 * g_target)
    data = base_graph_to_pyg(base, volt, group, k, g_target, label)
    if cycle_lengths or rwpe_dim > 0:
        data = add_structural_features(
            data, cycle_lengths=cycle_lengths, rwpe_dim=rwpe_dim
        )
    return key, _data_to_spec(data, base_name, group.name)


def _generate_chunk(
    chunk_seed: int,
    chunk_size: int,
    targets: list[tuple[int, int]],
    max_group_order: int,
    cycle_lengths: list[int] | None,
    rwpe_dim: int,
    traj_fraction: float = 0.6,
    kind: str = "girth",
) -> list[tuple[_DedupKey, dict[str, Any]]]:
    """Worker: produce roughly `chunk_size` candidate (dedup_key, spec_dict) pairs.

    A `traj_fraction` portion of the budget comes from tabu-search trajectories
    (near-feasible assignments that the search actually visits), the rest from
    uniform-random sampling for coverage. Mixing fixes the train/serve
    distribution skew: random assignments almost never reach high girth, so the
    trajectory source supplies the positive examples for hard targets.

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

    def _pick_config() -> tuple[int, int, str, BaseGraph, FiniteGroup]:
        k, g_target = rng.choice(targets)
        if k not in base_cache:
            base_cache[k] = _candidate_base_graphs(k)
        base_name, base = rng.choice(base_cache[k])
        mb = moore_cache[(k, g_target)]
        valid_groups = [
            g for g in groups if mb <= base.num_nodes * g.order <= max(4 * mb, mb + 200)
        ]
        if not valid_groups:
            valid_groups = groups
        group = rng.choice(valid_groups)
        return k, g_target, base_name, base, group

    out: list[tuple[_DedupKey, dict[str, Any]]] = []
    n_traj_target = int(round(chunk_size * traj_fraction))

    # Trajectory-sourced samples: each tabu run yields several near-feasible
    # assignments, so we keep launching runs until the trajectory budget is met.
    traj_produced = 0
    while traj_produced < n_traj_target:
        k, g_target, base_name, base, group = _pick_config()
        trajectory = _tabu_trajectory(base, group, g_target, rng, num_iterations=60)
        added = 0
        for volt in trajectory:
            sample = _make_sample(
                base,
                base_name,
                group,
                volt,
                k,
                g_target,
                cycle_lengths,
                rwpe_dim,
                kind,
            )
            if sample is not None:
                out.append(sample)
                added += 1
        # Always count a run as progress (>= 1) so a config that yields nothing
        # usable — every visited lift degenerate — cannot spin the loop forever.
        traj_produced += max(added, 1)

    # Uniform-random samples for coverage.
    for _ in range(chunk_size - n_traj_target):
        k, g_target, base_name, base, group = _pick_config()
        n_edges = base.num_undirected_edges()
        volt = [rng.randint(0, group.order - 1) for _ in range(n_edges)]
        sample = _make_sample(
            base,
            base_name,
            group,
            volt,
            k,
            g_target,
            cycle_lengths,
            rwpe_dim,
            kind,
        )
        if sample is not None:
            out.append(sample)

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
    traj_fraction: float = 0.6,
    kind: str = "girth",
) -> tuple[list[Data], dict[str, object]]:
    """Generate deduplicated training data for the predictor.

    ``kind`` selects the label stored in each sample's ``girth`` field:
    "girth" (the lift's exact girth, higher = better) or "tabu_cost" (the count
    of short identity walks the tabu search minimizes, lower = better).

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
            # "positive" = achieves target. For girth that is girth >= g_target;
            # for tabu_cost it is count == 0 (equivalent feasibility condition).
            label = int(spec["girth"])
            is_pos = (
                label == 0
                if kind == "tabu_cost"
                else label >= int(spec["g_target"])
            )
            if is_pos:
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
                traj_fraction,
                kind,
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
                    traj_fraction,
                    kind,
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
