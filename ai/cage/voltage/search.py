"""Beam search over voltage assignments guided by the girth predictor.

Also includes a brute-force exact search for small groups and a
meta-search that iterates over base graphs and groups.
"""

from __future__ import annotations

import math
import random
import sys
import time

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
from ai.cage.voltage.groups import (
    FiniteGroup,
    cyclic_group,
    dihedral_group,
    direct_product,
    semidirect_product_cyclic,
)
from ai.cage.voltage.lift import build_lift, verify_lift
from ai.cage.voltage.model import GirthPredictor
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
) -> tuple[list[int] | None, int | float]:
    """Beam search over voltage assignments guided by the girth predictor.

    Assigns voltages one edge at a time, keeping the top-B partial
    assignments according to the model's predicted girth.
    """
    m = base.num_undirected_edges()
    order = group.order

    _ = model.eval()

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
                    girth_pred, class_logit = model(data)
                    score = float(girth_pred.item()) + float(
                        torch.sigmoid(class_logit).item()
                    )

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


def meta_search(
    k: int,
    g_target: int,
    model: GirthPredictor | None = None,
    num_random_trials: int = 5000,
    beam_width: int = 50,
    max_group_order: int = 100,
    verbose: bool = True,
) -> dict[str, object]:
    """Search over base graphs and groups for small (k, g)-graphs.

    Uses random search (always) and beam search (if model provided).
    Returns information about the best graph found.
    """
    bases = _candidate_bases(k)
    best_result: dict[str, object] = {
        "girth": 0,
        "order": math.inf,
        "voltages": None,
        "base_name": None,
        "group_name": None,
    }
    best_order: float = math.inf

    start_time = time.time()
    configs_tried = 0

    for base_name, base in bases:
        groups = _candidate_groups_for_target(k, g_target, base, max_group_order)

        if verbose:
            print(
                f"\nBase: {base_name} ({base.num_nodes} nodes, {base.num_undirected_edges()} edges)"
            )
            print(f"  Groups to try: {len(groups)}")

        for group in groups:
            configs_tried += 1
            lift_size = base.num_nodes * group.order

            # Random search
            volts, girth = random_search(
                base,
                group,
                k,
                g_target,
                num_trials=num_random_trials,
                verbose=False,
            )

            if isinstance(girth, int) and girth >= g_target and volts is not None:
                if lift_size < best_order:
                    lifted = build_lift(base, group, volts)
                    props = verify_lift(lifted, k, g_target)
                    if props["is_valid_kg"]:
                        best_order = float(lift_size)
                        best_result = {
                            "girth": props["girth"],
                            "order": lift_size,
                            "voltages": volts,
                            "base_name": base_name,
                            "group_name": group.name,
                            "method": "random",
                        }
                        if verbose:
                            print(
                                f"  ** NEW BEST: {group.name}, order={lift_size}, "
                                f"girth={props['girth']}, voltages={volts}"
                            )

            # Tabu search
            volts_t, girth_t = tabu_search(
                base,
                group,
                k,
                g_target,
                num_iterations=2000,
                verbose=False,
            )

            if isinstance(girth_t, int) and girth_t >= g_target and volts_t is not None:
                if lift_size < best_order:
                    lifted = build_lift(base, group, volts_t)
                    props = verify_lift(lifted, k, g_target)
                    if props["is_valid_kg"]:
                        best_order = float(lift_size)
                        best_result = {
                            "girth": props["girth"],
                            "order": lift_size,
                            "voltages": volts_t,
                            "base_name": base_name,
                            "group_name": group.name,
                            "method": "tabu",
                        }
                        if verbose:
                            print(
                                f"  ** NEW BEST (tabu): {group.name}, order={lift_size}, "
                                f"girth={props['girth']}, voltages={volts_t}"
                            )

            # Beam search (if model available)
            if model is not None:
                volts_b, girth_b = beam_search(
                    base,
                    group,
                    k,
                    g_target,
                    model=model,
                    beam_width=beam_width,
                    verbose=False,
                )

                if (
                    isinstance(girth_b, int)
                    and girth_b >= g_target
                    and volts_b is not None
                ):
                    if lift_size < best_order:
                        lifted = build_lift(base, group, volts_b)
                        props = verify_lift(lifted, k, g_target)
                        if props["is_valid_kg"]:
                            best_order = float(lift_size)
                            best_result = {
                                "girth": props["girth"],
                                "order": lift_size,
                                "voltages": volts_b,
                                "base_name": base_name,
                                "group_name": group.name,
                                "method": "beam",
                            }
                            if verbose:
                                print(
                                    f"  ** NEW BEST (beam): {group.name}, order={lift_size}, "
                                    f"girth={props['girth']}, voltages={volts_b}"
                                )

    elapsed = time.time() - start_time

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Search complete: {configs_tried} configs in {elapsed:.1f}s")
        if best_result["voltages"] is not None:
            print(f"Best ({k},{g_target})-graph found:")
            print(f"  Order: {best_result['order']}")
            print(f"  Girth: {best_result['girth']}")
            print(f"  Base: {best_result['base_name']}")
            print(f"  Group: {best_result['group_name']}")
            print(f"  Voltages: {best_result['voltages']}")
            print(f"  Method: {best_result.get('method', 'unknown')}")

            # Build and verify the actual lift
            volts_final = best_result["voltages"]
            base_final = None
            for bname, b in bases:
                if bname == best_result["base_name"]:
                    base_final = b
                    break
            if base_final is not None and isinstance(volts_final, list):
                group_final = _find_group_by_name(
                    str(best_result["group_name"]), max_group_order
                )
                if group_final is not None:
                    lifted = build_lift(base_final, group_final, volts_final)
                    props = verify_lift(lifted, k, g_target)
                    print(f"  Verified: {props}")
        else:
            print(f"No ({k},{g_target})-graph found in search.")
        print(f"{'=' * 60}")

    return best_result


def _find_group_by_name(name: str, max_order: int) -> FiniteGroup | None:
    """Reconstruct a group from its name string."""
    if name.startswith("Z_"):
        n = int(name[2:])
        return cyclic_group(n)
    if name.startswith("D_"):
        n = int(name[2:])
        return dihedral_group(n)
    if "x" in name and name.count("x") == 1:
        parts = name.split("x")
        g1 = _find_group_by_name(parts[0], max_order)
        g2 = _find_group_by_name(parts[1], max_order)
        if g1 is not None and g2 is not None:
            return direct_product(g1, g2)
    return None


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    g = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    print(f"Searching for ({k},{g})-graphs via voltage graph lifts...")
    _ = meta_search(k, g, model=None, num_random_trials=5000, verbose=True)
