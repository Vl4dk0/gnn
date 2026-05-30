"""Forge a (k,g)-graph by cascading the three cage-construction methods.

``forge_graph(k, g, ...)`` chains the existing methods into one adaptive
pipeline:

  1. Voltage stage  — search base/group voltage lifts for a k-regular graph
     with girth as high as possible (ideally >= g).  Reuses the supervised
     search (tabu, plus GNN-guided beam search when a girth/tabu predictor is
     on disk).  The lift is k-regular by construction; its girth may be < g.

  2. Refine stage   — if the lift's girth is below g, run the TabuRefiner to
     push the girth up to g via degree-preserving 2-/3-switches.  Uses the
     trained move oracle as the swap scorer when present, else classical.
     If girth is still < g, the (base, group) attempt failed and the next one
     is tried.

  3. Excision stage — once a valid (k,g)-graph is in hand, repeatedly excise a
     tree and repair it into a strictly smaller (k,g)-graph (policy-guided when
     a trained repair policy is on disk, classical backtracking otherwise).
     The loop stops when no root yields a valid smaller graph.

Every stage degrades gracefully when no trained model is available, so the
whole cascade runs with a purely classical backend.  The returned graph is
verified k-regular with girth >= g before being handed back.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable, TypeAlias, cast

import networkx as nx
import torch

from ai.cage.excision.repair_search import (
    RepairPolicy,
    excise_and_repair,
    load_repair_policy,
)
from ai.cage.refine.move_oracle import build_score_fn, load_move_oracle
from ai.cage.refine.swaps import Swap2
from ai.cage.refine.tabu import TabuRefiner
from ai.cage.voltage.lift import build_lift, verify_lift
from ai.cage.voltage.supervised.model import GirthPredictor, load_girth_predictor
from ai.cage.voltage.supervised.search import (
    _candidate_bases,  # pyright: ignore[reportPrivateUsage]
    _candidate_groups_for_target,  # pyright: ignore[reportPrivateUsage]
    beam_search,
    tabu_search,
)
from backend.utils.graph_utils import compute_girth, is_k_regular

# Default on-disk location of the trained move oracle (refine scorer).
_MOVE_ORACLE_DIR = "move_oracle/move_oracle"

# Refine only realistically closes a small girth gap (the move oracle is trained
# on g-1 / g-2 near-misses), so a near-miss lift is only worth refining when its
# girth is within this many of the target.  Keep in sync with
# ``ai.cage.registry.forge.REFINE_MARGIN``.
REFINE_MARGIN = 2

# Score function signature accepted by TabuRefiner.
_ScoreFn: TypeAlias = "Callable[[nx.Graph[int], list[Swap2]], torch.Tensor]"


def _girth_value(G: nx.Graph[int]) -> int:
    """Integer girth of G (a large sentinel if acyclic)."""
    girth = compute_girth(G)
    return 10**9 if isinstance(girth, float) else girth


def _load_predictor(
    predictor: str | None, verbose: bool
) -> tuple[GirthPredictor | None, list[int] | None, int, str]:
    """Load the voltage predictor if requested and present; else (None, ...)."""
    if predictor is None:
        return None, None, 0, "girth"
    try:
        model, feat_cl, feat_rwpe, kind = load_girth_predictor(predictor)
    except FileNotFoundError:
        if verbose:
            print(f"[voltage] predictor {predictor!r} not found; classical tabu only")
        return None, None, 0, "girth"
    if verbose:
        print(f"[voltage] loaded {kind} predictor {predictor!r}")
    return model, feat_cl, feat_rwpe, kind


def _load_refine_score_fn(verbose: bool) -> _ScoreFn | None:
    """Load the move oracle as a TabuRefiner score_fn, or None if absent."""
    model_dir = Path(__file__).resolve().parents[1] / "trained" / _MOVE_ORACLE_DIR
    try:
        oracle, cycle_lengths, rwpe_dim = load_move_oracle(model_dir)
    except FileNotFoundError:
        if verbose:
            print("[refine] move oracle not found; classical refine")
        return None
    if verbose:
        print("[refine] loaded move oracle scorer")
    return build_score_fn(oracle, cycle_lengths, rwpe_dim)


def _find_valid_lift(
    k: int,
    g: int,
    *,
    predictor_model: GirthPredictor | None,
    feat_cl: list[int] | None,
    feat_rwpe: int,
    kind: str,
    refine_score_fn: _ScoreFn | None,
    max_voltage_attempts: int,
    max_group_order: int,
    tabu_iterations: int,
    refine_max_iter: int,
    refine_margin: int,
    beam_width: int,
    time_budget: float | None,
    start_time: float,
    verbose: bool,
) -> nx.Graph[int] | None:
    """Voltage + refine stages: return the first valid (k,g)-graph found.

    Iterates over (base, group) configs.  For each, builds the best lift via
    tabu search (or beam search with a predictor); if the lift girth is below
    g, the refiner is run to push it up.  Returns the lift as soon as it is a
    valid (k,g)-graph, or None if the attempt budget is exhausted.
    """
    bases = _candidate_bases(k)
    attempts = 0

    for base_name, base in bases:
        groups = _candidate_groups_for_target(k, g, base, max_group_order)
        for group in groups:
            if attempts >= max_voltage_attempts:
                return None
            if time_budget is not None and time.time() - start_time > time_budget:
                if verbose:
                    print("[voltage] time budget exceeded")
                return None
            attempts += 1

            if predictor_model is not None:
                volts, _girth = beam_search(
                    base,
                    group,
                    k,
                    g,
                    model=predictor_model,
                    beam_width=beam_width,
                    feat_cycle_lengths=feat_cl,
                    feat_rwpe_dim=feat_rwpe,
                    kind=kind,
                )
            else:
                volts, _girth = tabu_search(
                    base, group, k, g, num_iterations=tabu_iterations
                )

            if volts is None:
                continue

            lift = build_lift(base, group, volts)
            props = verify_lift(lift, k, g)
            if not props["is_k_regular"] or not props["is_connected"]:
                continue

            if props["is_valid_kg"]:
                if verbose:
                    print(
                        f"[voltage] {base_name}/{group.name}: "
                        f"order={lift.number_of_nodes()}, girth>={g} (no refine)"
                    )
                return lift

            # Lift is k-regular but girth < g: try to refine it up, but only when
            # it is a near-miss within refine_margin of g (refine cannot close a
            # large gap).  Lifts further away are skipped, matching ForgeGenerator.
            lift_girth = _girth_value(lift)
            if lift_girth < g - refine_margin:
                if verbose:
                    print(
                        f"[refine] {base_name}/{group.name}: girth={lift_girth} too "
                        f"far below {g}; skipping refine"
                    )
                continue
            if verbose:
                print(
                    f"[refine] {base_name}/{group.name}: girth={lift_girth} < {g}, "
                    f"refining order={lift.number_of_nodes()}"
                )
            refiner = TabuRefiner(
                g_target=g,
                score_fn=refine_score_fn,
                max_iter=refine_max_iter,
            )
            refined, _stats = refiner.refine(lift)
            if is_k_regular(refined, k) and _girth_value(refined) >= g:
                if verbose:
                    print(
                        f"[refine] success: order={refined.number_of_nodes()}, "
                        f"girth>={g}"
                    )
                return refined
            if verbose:
                print("[refine] failed to reach target girth; next config")

    return None


def forge_graph(
    k: int,
    g: int,
    *,
    predictor: str | None = "girth_predictor",
    max_voltage_attempts: int = 40,
    max_group_order: int = 60,
    tabu_iterations: int = 2000,
    refine_max_iter: int = 300,
    refine_margin: int = REFINE_MARGIN,
    beam_width: int = 30,
    excision_max_roots: int | None = None,
    excision_backtracks: int = 10000,
    time_budget: float | None = None,
    verbose: bool = False,
) -> nx.Graph[int] | None:
    """Forge a (k,g)-graph by cascading voltage -> refine -> excision.

    Returns the smallest valid (k,g)-graph found (k-regular, girth >= g), or
    None if the voltage+refine stage could not produce any valid graph.

    Args:
        k, g:                  Target degree and girth.
        predictor:             Voltage predictor model_id ("girth_predictor" /
                               "tabu_predictor") or None for classical tabu.
        max_voltage_attempts:  Cap on (base, group) configs tried.
        time_budget:           Optional wall-clock cap (seconds) for the whole
                               forge.
        verbose:               Print per-stage progress.
    """
    start_time = time.time()

    predictor_model, feat_cl, feat_rwpe, kind = _load_predictor(predictor, verbose)
    refine_score_fn = _load_refine_score_fn(verbose)
    repair_policy: RepairPolicy | None = load_repair_policy()
    if verbose:
        policy_status = (
            "loaded" if repair_policy is not None else "absent; classical backtrack"
        )
        print(f"[excision] repair policy {policy_status}")

    base_graph = _find_valid_lift(
        k,
        g,
        predictor_model=predictor_model,
        feat_cl=feat_cl,
        feat_rwpe=feat_rwpe,
        kind=kind,
        refine_score_fn=refine_score_fn,
        max_voltage_attempts=max_voltage_attempts,
        max_group_order=max_group_order,
        tabu_iterations=tabu_iterations,
        refine_max_iter=refine_max_iter,
        refine_margin=refine_margin,
        beam_width=beam_width,
        time_budget=time_budget,
        start_time=start_time,
        verbose=verbose,
    )
    if base_graph is None:
        if verbose:
            print("[forge] no valid (k,g)-graph from voltage+refine")
        return None

    # Excision shrink loop: repeatedly excise + repair into a smaller graph.
    best = base_graph
    while True:
        if time_budget is not None and time.time() - start_time > time_budget:
            if verbose:
                print("[excision] time budget exceeded; stopping shrink loop")
            break
        smaller = excise_and_repair(
            best,
            k,
            g,
            repair_policy,
            max_roots=excision_max_roots,
            max_backtracks=excision_backtracks,
            verbose=verbose,
        )
        if smaller is None:
            break
        if verbose:
            print(
                f"[excision] shrunk {best.number_of_nodes()} -> "
                f"{smaller.number_of_nodes()}"
            )
        best = smaller

    # Final verification before returning.
    if not is_k_regular(best, k) or _girth_value(best) < g:
        if verbose:
            print("[forge] final graph failed verification")
        return None
    return best


def main() -> None:
    """CLI: forge a (k,g)-graph and print its order/girth/regularity."""
    parser = argparse.ArgumentParser(description="Forge a (k,g)-graph by cascade")
    _ = parser.add_argument("k", type=int, help="Degree k")
    _ = parser.add_argument("g", type=int, help="Girth g")
    _ = parser.add_argument(
        "--predictor",
        type=str,
        default="girth_predictor",
        help="Voltage predictor model_id, or 'none' for classical tabu",
    )
    _ = parser.add_argument(
        "--max-voltage-attempts", type=int, default=40, help="Max (base,group) configs"
    )
    _ = parser.add_argument(
        "--max-group-order", type=int, default=60, help="Max group order"
    )
    _ = parser.add_argument(
        "--time-budget",
        type=float,
        default=None,
        help="Optional wall-clock cap in seconds",
    )
    _ = parser.add_argument("--quiet", action="store_true", help="Suppress progress")
    args = parser.parse_args()

    predictor_arg = cast(str, args.predictor)
    predictor: str | None = None if predictor_arg.lower() == "none" else predictor_arg

    graph = forge_graph(
        cast(int, args.k),
        cast(int, args.g),
        predictor=predictor,
        max_voltage_attempts=cast(int, args.max_voltage_attempts),
        max_group_order=cast(int, args.max_group_order),
        time_budget=cast("float | None", args.time_budget),
        verbose=not cast(bool, args.quiet),
    )

    k = cast(int, args.k)
    g = cast(int, args.g)
    if graph is None:
        print(f"No ({k},{g})-graph forged.")
        return

    girth = compute_girth(graph)
    print("=" * 50)
    print(f"Forged ({k},{g})-graph:")
    print(f"  order:      {graph.number_of_nodes()}")
    print(f"  edges:      {graph.number_of_edges()}")
    print(f"  k-regular:  {is_k_regular(graph, k)}")
    print(f"  girth:      {girth}")
    print("=" * 50)


if __name__ == "__main__":
    main()
