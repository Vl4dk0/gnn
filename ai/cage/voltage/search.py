"""Beam search over voltage assignments guided by the girth predictor.

Also includes a brute-force exact search for small groups and a
meta-search that iterates over base graphs and groups.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing
import random
import time
from typing import cast

import torch

from ai.cage.voltage.base_graphs import (
    BaseGraph,
    dumbbell,
    bouquet,
    cubic_multigraph_4nodes,
    prism_base,
    moebius_kantor_base,
)
from ai.cage.voltage.cycle_analysis import (
    compute_lift_girth,
    count_short_identity_walks,
)
from ai.cage.voltage.data_gen import base_graph_to_pyg
from ai.utils.structural_features import add_structural_features
from ai.cage.voltage.groups import (
    FiniteGroup,
    cyclic_group,
    dihedral_group,
    direct_product,
    semidirect_product_cyclic,
)
from ai.cage.voltage.lift import build_lift, verify_lift
from ai.cage.voltage.model import GirthPredictor, load_girth_predictor
from backend.utils.graph_utils import moore_bound


# ── Exact brute-force search (small groups) ──────────────────────────────


def exhaustive_search(
    base: BaseGraph,
    group: FiniteGroup,
    k: int,
    g_target: int,
    verbose: bool = False,
) -> tuple[list[int] | None, int | float]:
    """Try every possible voltage assignment and return the one with highest
    **verified** girth (connected + k-regular lift only).

    Only practical for small groups and few edges (|Gamma|^m manageable).
    """
    m = base.num_undirected_edges()
    order = group.order
    total = order**m

    if verbose:
        print(f"Exhaustive search: {order}^{m} = {total} assignments")

    best_voltages: list[int] | None = None
    best_girth: int | float = 0

    for idx in range(total):
        volts: list[int] = []
        rem = idx
        for _ in range(m):
            rem, v = divmod(rem, order)
            volts.append(v)

        # Fast pre-filter
        girth = compute_lift_girth(base, group, volts, max_girth=2 * g_target)
        if isinstance(girth, float) or girth <= best_girth:
            continue

        # Verify the lift is actually valid
        props = verify_lift(build_lift(base, group, volts), k, g_target)
        if not props["is_k_regular"] or not props["is_connected"]:
            continue
        verified_girth = props["girth"]
        if not isinstance(verified_girth, int):
            continue

        if verified_girth > best_girth:
            best_girth = verified_girth
            best_voltages = volts[:]
            if verbose:
                print(f"  New best: girth={verified_girth}, voltages={volts}")

    return best_voltages, best_girth


# ── Random search ────────────────────────────────────────────────────────


def random_search(
    base: BaseGraph,
    group: FiniteGroup,
    k: int,
    g_target: int,
    num_trials: int = 10000,
    verbose: bool = False,
) -> tuple[list[int] | None, int | float]:
    """Random sampling of voltage assignments, keeping the best verified girth."""
    m = base.num_undirected_edges()
    order = group.order

    best_voltages: list[int] | None = None
    best_girth: int | float = 0
    hits = 0

    for trial in range(num_trials):
        volts = [random.randint(0, order - 1) for _ in range(m)]
        girth = compute_lift_girth(base, group, volts, max_girth=2 * g_target)

        if isinstance(girth, float) or girth <= best_girth:
            continue

        # Verify the lift is valid before accepting
        props = verify_lift(build_lift(base, group, volts), k, g_target)
        if not props["is_k_regular"] or not props["is_connected"]:
            continue
        verified_girth = props["girth"]
        if not isinstance(verified_girth, int):
            continue

        if verified_girth >= g_target:
            hits += 1

        if verified_girth > best_girth:
            best_girth = verified_girth
            best_voltages = volts[:]
            if verbose and (trial < 100 or trial % 1000 == 0):
                lift_order = base.num_nodes * group.order
                print(
                    f"  Trial {trial}: girth={verified_girth}, order={lift_order}, "
                    f"voltages={volts}"
                )

    if verbose:
        print(
            f"  {hits}/{num_trials} assignments achieved girth >= {g_target} "
            f"({100 * hits / num_trials:.1f}%)"
        )

    return best_voltages, best_girth


# ── Tabu search (2025 paper approach) ────────────────────────────────────


def tabu_search(
    base: BaseGraph,
    group: FiniteGroup,
    _k: int,
    g_target: int,
    num_iterations: int = 5000,
    tabu_tenure: int = 10,
    verbose: bool = False,
) -> tuple[list[int] | None, int | float]:
    """Tabu search over voltage assignments using short-walk cost.

    Based on the approach from "New small regular graphs of given girth"
    (2025).  Uses spanning-tree normalization to reduce the search space:
    tree-edge voltages are fixed to 0, only non-tree edges are modified.

    The cost function counts closed walks shorter than g_target with
    identity net voltage.  Cost 0 means girth >= g_target.
    """

    free_indices = base.free_edge_indices()
    tree_indices = base.spanning_tree_edge_indices()
    m = base.num_undirected_edges()
    order = group.order

    if not free_indices:
        # All edges are tree edges; only one assignment (all zeros)
        volts = [0] * m
        girth = compute_lift_girth(base, group, volts, max_girth=2 * g_target)
        return volts, girth

    # Initialize: random voltages on free edges, 0 on tree edges
    volts = [0] * m
    for idx in free_indices:
        volts[idx] = random.randint(0, order - 1)

    best_cost = count_short_identity_walks(base, group, volts, g_target)
    best_volts = volts[:]
    best_girth: int | float = 0

    if best_cost == 0:
        best_girth = compute_lift_girth(base, group, volts, max_girth=2 * g_target)
        # Reject disconnected/degenerate lifts (infinite girth)
        if isinstance(best_girth, int) and best_girth >= g_target:
            if verbose:
                print(f"  Tabu: found girth={best_girth} immediately")
            return best_volts, best_girth
        # Not actually valid — reset cost to force search
        best_cost = 1

    # Tabu list: (edge_index, old_value) -> expiry iteration
    tabu: dict[tuple[int, int], int] = {}
    current_cost = best_cost

    for iteration in range(num_iterations):
        # Try all single-edge voltage changes on free edges
        best_move_cost = current_cost + 1  # worse than current
        best_move_edge = -1
        best_move_val = -1

        for edge_idx in free_indices:
            old_val = volts[edge_idx]
            for new_val in range(order):
                if new_val == old_val:
                    continue

                # Check tabu
                move_key = (edge_idx, old_val)
                is_tabu = tabu.get(move_key, -1) > iteration

                # Apply move temporarily
                volts[edge_idx] = new_val
                new_cost = count_short_identity_walks(base, group, volts, g_target)
                volts[edge_idx] = old_val

                # Aspiration: accept if it beats the global best, even if tabu
                if new_cost < best_cost or (not is_tabu and new_cost < best_move_cost):
                    best_move_cost = new_cost
                    best_move_edge = edge_idx
                    best_move_val = new_val

        if best_move_edge < 0:
            break  # no improving move found

        # Apply the best move
        old_val = volts[best_move_edge]
        volts[best_move_edge] = best_move_val
        tabu[(best_move_edge, old_val)] = iteration + tabu_tenure
        current_cost = best_move_cost

        if current_cost < best_cost:
            best_cost = current_cost
            best_volts = volts[:]

            if best_cost == 0:
                best_girth = compute_lift_girth(
                    base, group, best_volts, max_girth=2 * g_target
                )
                if isinstance(best_girth, int) and best_girth >= g_target:
                    if verbose:
                        print(f"  Tabu: cost=0 at iter {iteration}, girth={best_girth}")
                    return best_volts, best_girth
                # Degenerate (disconnected) — keep searching
                best_cost = 1

        if verbose and (iteration + 1) % 500 == 0:
            print(
                f"  Tabu iter {iteration + 1}: cost={current_cost}, best_cost={best_cost}"
            )

    # Compute girth for best found
    best_girth = compute_lift_girth(base, group, best_volts, max_girth=2 * g_target)
    return best_volts, best_girth


# ── GNN-guided beam search ───────────────────────────────────────────────


def beam_search(
    base: BaseGraph,
    group: FiniteGroup,
    k: int,
    g_target: int,
    model: GirthPredictor,
    beam_width: int = 100,
    verbose: bool = False,
    feat_cycle_lengths: list[int] | None = None,
    feat_rwpe_dim: int = 0,
) -> tuple[list[int] | None, int | float]:
    """Beam search over voltage assignments guided by the girth predictor.

    Assigns voltages one edge at a time, keeping the top-B partial
    assignments according to the model's predicted girth.

    feat_cycle_lengths / feat_rwpe_dim: structural feature config that was used
    during training. Must match the saved feature_config in the model's info.json.
    """
    m = base.num_undirected_edges()
    order = group.order

    _ = model.eval()
    device = next(model.parameters()).device

    # Each candidate is a partial voltage list (length increases each step)
    candidates: list[tuple[list[int], float]] = [([], 0.0)]

    with torch.no_grad():
        for edge_idx in range(m):
            new_candidates: list[tuple[list[int], float]] = []

            for partial, _prev_score in candidates:
                for g_elem in range(order):
                    extended = partial + [g_elem]

                    # Pad remaining edges with 0 for model input
                    full_volts = extended + [0] * (m - len(extended))

                    data = base_graph_to_pyg(
                        base, full_volts, group, k, g_target, girth=0
                    )
                    if feat_cycle_lengths or feat_rwpe_dim > 0:
                        data = add_structural_features(
                            data,
                            cycle_lengths=feat_cycle_lengths,
                            rwpe_dim=feat_rwpe_dim,
                        )
                    data = data.to(device)
                    girth_pred = model(data)
                    score = float(girth_pred.item())

                    new_candidates.append((extended, score))

            # Keep top beam_width
            new_candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = new_candidates[:beam_width]

            if verbose:
                best_score = candidates[0][1] if candidates else 0
                print(
                    f"  Edge {edge_idx + 1}/{m}: {len(new_candidates)} candidates "
                    f"-> top {len(candidates)}, best score={best_score:.2f}"
                )

    # Verify top candidates with exact girth computation
    best_voltages: list[int] | None = None
    best_girth: int | float = 0

    for volts, score in candidates:
        girth = compute_lift_girth(base, group, volts, max_girth=2 * g_target)
        if isinstance(girth, float):
            continue
        if girth > best_girth:
            best_girth = girth
            best_voltages = volts[:]

    if verbose and best_voltages is not None:
        lift_order = base.num_nodes * group.order
        print(
            f"  Best: girth={best_girth}, order={lift_order}, voltages={best_voltages}"
        )

    return best_voltages, best_girth


# ── Meta-search over base graphs and groups ──────────────────────────────


def _candidate_groups_for_target(
    k: int, g_target: int, base: BaseGraph, max_order: int = 100
) -> list[FiniteGroup]:
    """Select groups whose lift size is in a reasonable range."""

    mb = moore_bound(k, g_target)
    n_base = base.num_nodes
    groups: list[FiniteGroup] = []

    for n in range(max(2, mb // n_base - 5), min(max_order, 4 * mb // n_base + 5)):
        if n < 2:
            continue
        groups.append(cyclic_group(n))
        if n >= 6 and n % 2 == 0:
            groups.append(dihedral_group(n // 2))

    # Direct products of small cyclic groups
    for a in range(2, min(15, max_order)):
        for b in range(a, min(15, max_order)):
            order = a * b
            if order < 2 or order > max_order:
                continue
            lift_size = n_base * order
            if mb <= lift_size <= 4 * mb:
                groups.append(direct_product(cyclic_group(a), cyclic_group(b)))

    # Some semidirect products
    for n in range(3, min(30, max_order)):
        for m in range(2, min(10, max_order)):
            if n * m > max_order:
                continue
            for phi in range(2, n):
                if pow(phi, m, n) == 1:
                    lift_size = n_base * n * m
                    if mb <= lift_size <= 4 * mb:
                        groups.append(semidirect_product_cyclic(n, m, phi))
                    break  # one phi per (n, m) is enough

    return groups


def _candidate_bases(k: int) -> list[tuple[str, BaseGraph]]:
    """Enumerate candidate base graphs for degree k."""
    bases: list[tuple[str, BaseGraph]] = []

    if k == 3:
        bases.append(("dumbbell(3)", dumbbell(3)))
        bases.append(("cubic_4nodes", cubic_multigraph_4nodes()))
        bases.append(("prism", prism_base()))
        bases.append(("petersen_base", moebius_kantor_base()))

    bases.append((f"dumbbell({k})", dumbbell(k)))

    if k % 2 == 0:
        bases.append((f"bouquet({k // 2})", bouquet(k // 2)))

    return bases


def _search_one_config(
    args: tuple[str, BaseGraph, FiniteGroup, int, int, int],
) -> dict[str, object] | None:
    """Run random + tabu search on one (base, group) pair. Process-safe worker."""
    base_name, base, group, k, g_target, num_random_trials = args
    lift_size = base.num_nodes * group.order
    best: dict[str, object] | None = None

    def _make_result(
        volts: list[int], girth_val: object, method: str
    ) -> dict[str, object]:
        return dict(
            girth=girth_val,
            order=lift_size,
            voltages=volts,
            base_name=base_name,
            group_name=group.name,
            base=base,
            group=group,
            method=method,
        )

    # Random search
    volts, girth = random_search(
        base, group, k, g_target, num_trials=num_random_trials, verbose=False
    )
    if isinstance(girth, int) and girth >= g_target and volts is not None:
        lifted = build_lift(base, group, volts)
        props = verify_lift(lifted, k, g_target)
        if props["is_valid_kg"]:
            best = _make_result(volts, props["girth"], "random")

    # Tabu search
    volts_t, girth_t = tabu_search(
        base, group, k, g_target, num_iterations=2000, verbose=False
    )
    if isinstance(girth_t, int) and girth_t >= g_target and volts_t is not None:
        lifted = build_lift(base, group, volts_t)
        props = verify_lift(lifted, k, g_target)
        if props["is_valid_kg"]:
            candidate = _make_result(volts_t, props["girth"], "tabu")
            if best is None or lift_size < int(str(best.get("order", 999999))):
                best = candidate

    return best


def meta_search(
    k: int,
    g_target: int,
    model: GirthPredictor | None = None,
    num_random_trials: int = 5000,
    beam_width: int = 50,
    max_group_order: int = 100,
    verbose: bool = True,
    num_workers: int = 1,
    feat_cycle_lengths: list[int] | None = None,
    feat_rwpe_dim: int = 0,
) -> dict[str, object]:
    """Search over base graphs and groups for small (k, g)-graphs.

    Uses random + tabu search in parallel across (base, group) configs.
    Beam search runs if a model is provided (sequential, requires GPU).
    """
    bases = _candidate_bases(k)

    # Build flat list of all configs
    configs: list[tuple[str, BaseGraph, FiniteGroup, int, int, int]] = []
    for base_name, base in bases:
        groups = _candidate_groups_for_target(k, g_target, base, max_group_order)
        for group in groups:
            configs.append((base_name, base, group, k, g_target, num_random_trials))

    if verbose:
        print(f"Searching for ({k},{g_target})-graphs via voltage graph lifts...")
        print(f"  Total configs: {len(configs)} across {len(bases)} bases")
        print(f"  Workers: {num_workers}")
        print(f"  Max group order: {max_group_order}")

    best_result: dict[str, object] = {
        "girth": 0,
        "order": math.inf,
        "voltages": None,
        "base_name": None,
        "group_name": None,
    }
    best_order: float = math.inf
    start_time = time.time()

    def _update_best(result: dict[str, object] | None) -> None:
        nonlocal best_order, best_result
        if result is None:
            return
        order = int(str(result["order"]))
        if order < best_order:
            best_order = float(order)
            best_result = result
            if verbose:
                print(
                    f"  ** NEW BEST: {result['group_name']}, order={order}, "
                    f"girth={result['girth']}, method={result['method']}, "
                    f"voltages={result['voltages']}"
                )

    if num_workers > 1:
        with multiprocessing.Pool(num_workers) as pool:
            for result in pool.imap_unordered(_search_one_config, configs):
                _update_best(result)
    else:
        for cfg in configs:
            result = _search_one_config(cfg)
            _update_best(result)

    # Beam search pass — runs sequentially, model in main process only
    if model is not None:
        if verbose:
            print(f"\n  --- ML-guided beam search (beam_width={beam_width}) ---")
        for base_name, base in bases:
            groups = _candidate_groups_for_target(k, g_target, base, max_group_order)
            for group in groups:
                lift_size = base.num_nodes * group.order
                if lift_size >= best_order:
                    continue
                volts_b, girth_b = beam_search(
                    base,
                    group,
                    k,
                    g_target,
                    model=model,
                    beam_width=beam_width,
                    verbose=False,
                    feat_cycle_lengths=feat_cycle_lengths,
                    feat_rwpe_dim=feat_rwpe_dim,
                )
                if (
                    isinstance(girth_b, int)
                    and girth_b >= g_target
                    and volts_b is not None
                ):
                    lifted = build_lift(base, group, volts_b)
                    props = verify_lift(lifted, k, g_target)
                    if props["is_valid_kg"]:
                        _update_best(
                            dict(
                                girth=props["girth"],
                                order=lift_size,
                                voltages=volts_b,
                                base_name=base_name,
                                group_name=group.name,
                                base=base,
                                group=group,
                                method="beam",
                            )
                        )

    elapsed = time.time() - start_time

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Search complete: {len(configs)} configs in {elapsed:.1f}s")
        if best_result["voltages"] is not None:
            print(f"Best ({k},{g_target})-graph found:")
            print(f"  Order: {best_result['order']}")
            print(f"  Girth: {best_result['girth']}")
            print(f"  Base: {best_result['base_name']}")
            print(f"  Group: {best_result['group_name']}")
            print(f"  Voltages: {best_result['voltages']}")
            print(f"  Method: {best_result.get('method', 'unknown')}")

            # Verify the best result. The base/group objects are carried in
            # best_result directly (workers echo them back), so no fragile
            # name re-parsing.
            volts_final = best_result["voltages"]
            base_final = cast(BaseGraph | None, best_result.get("base"))
            group_final = cast(FiniteGroup | None, best_result.get("group"))
            if (
                base_final is not None
                and group_final is not None
                and isinstance(volts_final, list)
            ):
                lifted = build_lift(base_final, group_final, volts_final)
                props = verify_lift(lifted, k, g_target)
                print(f"  Verified: {props}")
        else:
            print(f"No ({k},{g_target})-graph found in search.")
        print(f"{'=' * 60}")

    return best_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voltage graph cage search")
    _ = parser.add_argument("k", type=int, nargs="?", default=3, help="Degree k")
    _ = parser.add_argument("g", type=int, nargs="?", default=5, help="Girth g")
    _ = parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    _ = parser.add_argument(
        "--max-group-order", type=int, default=100, help="Max group order"
    )
    _ = parser.add_argument(
        "--trials", type=int, default=5000, help="Random trials per config"
    )
    _ = parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Girth predictor model_id (e.g. girth_predictor_k3_g7) — enables beam search",
    )
    _ = parser.add_argument(
        "--beam-width", type=int, default=50, help="Beam width when --model-id is set"
    )
    args = parser.parse_args()

    loaded_model: GirthPredictor | None = None
    loaded_feat_cl: list[int] | None = None
    loaded_feat_rwpe: int = 0
    if args.model_id is not None:
        loaded_model, loaded_feat_cl, loaded_feat_rwpe = load_girth_predictor(
            str(args.model_id)
        )
        print(f"Loaded girth predictor: {args.model_id}")

    _ = meta_search(
        args.k,
        args.g,
        model=loaded_model,
        num_random_trials=args.trials,
        max_group_order=args.max_group_order,
        verbose=True,
        num_workers=args.workers,
        beam_width=args.beam_width,
        feat_cycle_lengths=loaded_feat_cl,
        feat_rwpe_dim=loaded_feat_rwpe,
    )
