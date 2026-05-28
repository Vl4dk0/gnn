"""Cayley graph search for (k, g)-cage candidates.

The core search variable here is the symmetric generating set S \\subset G,
not a voltage assignment. We provide three search modes:

  * random_search  — uniform sampling baseline
  * tabu_search    — local search swapping one generator at a time, with
                     a tabu list to avoid cycling
  * meta_search    — orchestrates the above across a catalogue of groups

When a CayleyGirthPredictor is provided to meta_search, it is used inside
the tabu inner loop to score candidate swaps cheaply, with classical
verification of the top-K before committing — exactly the role the voltage
girth predictor plays in beam search.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing
import random
import time
from typing import Protocol, cast

import networkx as nx

from ai.cage.cayley.cayley import (
    build_cayley,
    cayley_girth,
    count_short_relations,
    is_symmetric,
)
from ai.cage.cayley.groups import (
    FiniteGroup,
    available_groups,
    element_order,
    involutions,
    non_identity_elements,
    random_generating_set,
)
from ai.cage.cayley.model import (
    CayleyGirthPredictor,
    load_cayley_girth_predictor,
    make_score_fn,
)
from backend.utils.graph_utils import moore_bound


# ── Random search ───────────────────────────────────────────────────────


def random_search(
    group: FiniteGroup,
    k: int,
    g_target: int,
    num_trials: int = 5000,
    verbose: bool = False,
) -> tuple[list[int] | None, int | float]:
    """Uniformly sample symmetric generating sets and keep the best verified girth."""
    best_gens: list[int] | None = None
    best_girth: int | float = 0

    for trial in range(num_trials):
        gens = random_generating_set(group, k)
        if gens is None or len(gens) != k:
            continue
        girth = cayley_girth(group, gens, max_girth=2 * g_target)
        if girth <= best_girth:
            continue

        cay = build_cayley(group, gens)
        if not nx.is_connected(cay):
            continue
        if cay.degree(0) != k:
            continue

        if girth > best_girth:
            best_girth = girth
            best_gens = gens[:]
            if verbose and (trial < 50 or trial % 500 == 0):
                print(f"  Trial {trial}: girth={girth} order={group.order} S={gens}")

    return best_gens, best_girth


# ── Tabu search ─────────────────────────────────────────────────────────


def candidate_swaps(
    group: FiniteGroup, current: list[int]
) -> list[tuple[int, int, list[int]]]:
    """Enumerate single-slot swaps that keep S symmetric and of fixed size.

    Returns a list of (slot_kind, replaced_element, new_gens) tuples, where
    slot_kind is 0 for an involution swap and 1 for an inverse-pair swap.
    """
    out: list[tuple[int, int, list[int]]] = []
    current_set = set(current)

    invols_all = set(involutions(group))
    # Identify involution slots in current S.
    current_invols = [s for s in current if s in invols_all]
    # Identify non-involution pair representatives in current S.
    visited_pair: set[int] = set()
    current_pairs: list[int] = []
    for s in current:
        if s in invols_all or s in visited_pair:
            continue
        _ = visited_pair.add(s)
        _ = visited_pair.add(group.inv(s))
        current_pairs.append(s)

    # Swap one involution for another (not already in S).
    for s_old in current_invols:
        for s_new in invols_all:
            if s_new in current_set or s_new == s_old:
                continue
            new_gens = [x for x in current if x != s_old] + [s_new]
            if len(new_gens) == len(current):
                out.append((0, s_old, new_gens))

    # Swap one inverse-pair for another. Need pair reps not in S.
    all_pair_reps: list[int] = []
    seen: set[int] = set()
    for a in non_identity_elements(group):
        if a in invols_all or a in seen:
            continue
        _ = seen.add(a)
        _ = seen.add(group.inv(a))
        all_pair_reps.append(a)

    for s_old in current_pairs:
        s_old_inv = group.inv(s_old)
        for s_new in all_pair_reps:
            if s_new in current_set or group.inv(s_new) in current_set:
                continue
            new_gens = [x for x in current if x not in (s_old, s_old_inv)]
            new_gens = new_gens + [s_new, group.inv(s_new)]
            if len(new_gens) == len(current):
                out.append((1, s_old, new_gens))

    return out


def tabu_search(
    group: FiniteGroup,
    k: int,
    g_target: int,
    num_iterations: int = 2000,
    tabu_tenure: int = 10,
    max_neighbors_per_step: int = 200,
    verbose: bool = False,
    score_fn: "ScoreFn | None" = None,
) -> tuple[list[int] | None, int | float]:
    """Tabu search over symmetric generating sets.

    Cost = number of closed non-backtracking walks at identity of length
    < g_target. Cost 0 means girth >= g_target (still must verify
    connectedness).

    score_fn, if provided, is a callable (group, gens, k, g_target) -> float
    used to rank candidates *before* expensive verification. It plays the
    same role as the GNN beam-search heuristic in the voltage pipeline:
    we pre-rank, then verify the top-K classically.
    """
    initial = random_generating_set(group, k)
    if initial is None or len(initial) != k:
        return None, 0
    gens: list[int] = initial

    current_cost = count_short_relations(group, gens, g_target - 1)
    best_gens = gens[:]
    best_cost = current_cost

    tabu: dict[int, int] = {}

    for it in range(num_iterations):
        if current_cost == 0:
            girth = cayley_girth(group, gens, max_girth=2 * g_target)
            cay = build_cayley(group, gens)
            if (
                isinstance(girth, int)
                and girth >= g_target
                and nx.is_connected(cay)
                and cay.degree(0) == k
            ):
                if verbose:
                    print(f"  Tabu: cost=0 at iter {it}, girth={girth}")
                return gens, girth
            # Disconnected or wrong degree — bump cost to keep searching.
            current_cost = 1

        # Enumerate candidate moves and rank.
        candidates = candidate_swaps(group, gens)
        if not candidates:
            break
        random.shuffle(candidates)
        candidates = candidates[:max_neighbors_per_step]

        scored: list[tuple[float, int, list[int], int]] = []
        for kind, s_old, new_gens in candidates:
            if score_fn is not None:
                # Use the ML heuristic; lower predicted girth gap is better.
                hint = score_fn(group, new_gens, k, g_target)
                scored.append((hint, kind, new_gens, s_old))
            else:
                cost = count_short_relations(group, new_gens, g_target - 1)
                scored.append((float(cost), kind, new_gens, s_old))

        # If a heuristic was used, verify the top-K cheaply via the true cost.
        if score_fn is not None:
            scored.sort(key=lambda x: x[0])
            top = scored[: max(5, min(20, len(scored)))]
            scored = [
                (
                    float(count_short_relations(group, new_gens, g_target - 1)),
                    kind,
                    new_gens,
                    s_old,
                )
                for (_, kind, new_gens, s_old) in top
            ]

        # Find best non-tabu move (with aspiration: tabu is overridden if it
        # would set a new global best).
        scored.sort(key=lambda x: x[0])
        chosen: tuple[float, int, list[int], int] | None = None
        for entry in scored:
            new_cost, _kind, new_gens, s_old = entry
            is_tabu = tabu.get(s_old, -1) > it
            if not is_tabu or new_cost < best_cost:
                chosen = entry
                break

        if chosen is None:
            # All neighbours are tabu and none aspirational — accept the best.
            chosen = scored[0]

        new_cost, _kind, new_gens, s_old = chosen
        tabu[s_old] = it + tabu_tenure
        gens = new_gens
        current_cost = int(new_cost)

        if current_cost < best_cost:
            best_cost = current_cost
            best_gens = gens[:]

        if verbose and (it + 1) % 200 == 0:
            print(f"  Tabu iter {it + 1}: cost={current_cost} best={best_cost}")

    best_girth = cayley_girth(group, best_gens, max_girth=2 * g_target)
    return best_gens, best_girth


# ── Heuristic interface ─────────────────────────────────────────────────


class ScoreFn(Protocol):
    """Protocol: (group, generators, k, g_target) -> heuristic score (lower better)."""

    def __call__(
        self,
        group: FiniteGroup,
        generators: list[int],
        k: int,
        g_target: int,
    ) -> float: ...


# ── Meta-search across groups ───────────────────────────────────────────


def search_one_group(
    args: tuple[FiniteGroup, int, int, int, int],
) -> dict[str, object] | None:
    """Process-safe worker: run random + tabu on one group, return best hit.

    Returns None if no verified (k,g) Cayley graph was found in this group.
    No ML score_fn is used here because the predictor (PyTorch model with
    its CUDA / MPS tensors) is hard to pickle across worker processes.
    """
    group, k, g_target, num_random_trials, num_tabu_iters = args

    def _make(gens: list[int], girth_val: int, method: str) -> dict[str, object]:
        return {
            "girth": girth_val,
            "order": group.order,
            "generators": gens,
            "group_name": group.name,
            "method": method,
        }

    found: dict[str, object] | None = None

    gens_r, girth_r = random_search(
        group, k, g_target, num_trials=num_random_trials, verbose=False
    )
    if gens_r is not None and isinstance(girth_r, int) and girth_r >= g_target:
        cay = build_cayley(group, gens_r)
        if nx.is_connected(cay) and cay.degree(0) == k:
            found = _make(gens_r, girth_r, "random")

    gens_t, girth_t = tabu_search(
        group,
        k,
        g_target,
        num_iterations=num_tabu_iters,
        verbose=False,
        score_fn=None,
    )
    if gens_t is not None and isinstance(girth_t, int) and girth_t >= g_target:
        cay = build_cayley(group, gens_t)
        if nx.is_connected(cay) and cay.degree(0) == k:
            candidate = _make(gens_t, girth_t, "tabu")
            if found is None or int(cast(int, candidate["girth"])) > int(
                cast(int, found["girth"])
            ):
                found = candidate

    return found


def meta_search(
    k: int,
    g_target: int,
    max_group_order: int = 500,
    num_random_trials: int = 2000,
    num_tabu_iters: int = 1500,
    score_fn: ScoreFn | None = None,
    num_workers: int = 1,
    verbose: bool = True,
) -> dict[str, object]:
    """Search the catalogue of groups for the smallest verified (k,g) Cayley graph.

    When num_workers > 1, the per-group random + tabu pass is parallelised
    across processes. The ML-guided refinement (score_fn != None) runs
    sequentially in the main process, mirroring the voltage meta-search.
    """
    groups = available_groups(max_group_order)
    mb = moore_bound(k, g_target)
    groups = [g for g in groups if mb <= g.order <= max(4 * mb, mb + 500)]
    if not groups:
        groups = available_groups(max_group_order)
    groups.sort(key=lambda g: g.order)

    if verbose:
        print(f"Cayley meta-search for ({k},{g_target}). Moore bound = {mb}.")
        print(f"  Candidate groups: {len(groups)} | Workers: {num_workers}")

    best: dict[str, object] = {
        "girth": 0,
        "order": math.inf,
        "generators": None,
        "group_name": None,
    }
    best_order: float = math.inf
    start = time.time()

    def _update(result: dict[str, object] | None) -> None:
        nonlocal best_order, best
        if result is None:
            return
        order = int(cast(int, result["order"]))
        if order < best_order:
            best_order = float(order)
            best = result
            if verbose:
                print(
                    f"  ** {result['group_name']}: girth={result['girth']} "
                    + f"order={order} method={result['method']} "
                    + f"S={result['generators']}"
                )

    configs: list[tuple[FiniteGroup, int, int, int, int]] = [
        (group, k, g_target, num_random_trials, num_tabu_iters) for group in groups
    ]

    if num_workers > 1:
        with multiprocessing.Pool(num_workers) as pool:
            for result in pool.imap_unordered(search_one_group, configs):
                _update(result)
    else:
        for cfg in configs:
            _update(search_one_group(cfg))

    # Optional ML-guided refinement pass: sequential because the model
    # carries device-resident tensors that don't pickle well across workers.
    if score_fn is not None:
        if verbose:
            print("\n  --- ML-guided tabu refinement ---")
        for group in groups:
            if group.order >= best_order:
                continue
            gens_t, girth_t = tabu_search(
                group,
                k,
                g_target,
                num_iterations=num_tabu_iters,
                verbose=False,
                score_fn=score_fn,
            )
            if gens_t is None or not isinstance(girth_t, int) or girth_t < g_target:
                continue
            cay = build_cayley(group, gens_t)
            if not (nx.is_connected(cay) and cay.degree(0) == k):
                continue
            _update(
                {
                    "girth": girth_t,
                    "order": group.order,
                    "generators": gens_t,
                    "group_name": group.name,
                    "method": "tabu-ml",
                }
            )

    elapsed = time.time() - start
    best["elapsed"] = elapsed
    best["num_groups"] = len(groups)
    if verbose:
        print(f"\nCayley meta-search done in {elapsed:.1f}s.")
        if best["generators"] is not None:
            print(f"  Best: order={best['order']} girth={best['girth']}")
            print(f"        group={best['group_name']}  S={best['generators']}")
        else:
            print(f"  No ({k},{g_target})-Cayley graph found in catalogue.")

    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cayley graph cage search")
    _ = parser.add_argument("k", type=int, nargs="?", default=3)
    _ = parser.add_argument("g", type=int, nargs="?", default=6)
    _ = parser.add_argument("--max-order", type=int, default=500)
    _ = parser.add_argument("--trials", type=int, default=2000)
    _ = parser.add_argument("--tabu-iters", type=int, default=1500)
    _ = parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    _ = parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="CayleyGirthPredictor model_id (enables ML-guided ranking inside tabu)",
    )
    args = parser.parse_args()

    score_fn_arg: ScoreFn | None = None
    model_id_arg = cast("str | None", args.model_id)
    if model_id_arg is not None:
        model: CayleyGirthPredictor = load_cayley_girth_predictor(model_id_arg)
        score_fn_arg = make_score_fn(model)
        print(f"Loaded Cayley girth predictor: {model_id_arg}")

    _ = meta_search(
        cast(int, args.k),
        cast(int, args.g),
        max_group_order=cast(int, args.max_order),
        num_random_trials=cast(int, args.trials),
        num_tabu_iters=cast(int, args.tabu_iters),
        num_workers=cast(int, args.workers),
        score_fn=score_fn_arg,
        verbose=True,
    )

    # Smoke-call so element_order remains importable (small fixture); avoids
    # an unused-import diagnostic and exercises the helper.
    _ = element_order
    _ = is_symmetric
