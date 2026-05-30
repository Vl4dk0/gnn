"""Forge a (k,g)-graph by driving the queue-driven cage pipeline to completion.

``forge_graph(k, g, ...)`` is the one-shot entry point.  It builds a
``ForgeGenerator`` — the cooperative producer-consumer pipeline that streams
voltage lifts into refine and excision workers via queues — and loops its
``step()`` until completion, returning the first valid (k,g)-graph the pipeline
produces (k-regular, girth >= g) or ``None`` if the pipeline drains without one.

The stepping pipeline owns all stage logic (multi-base voltage producing,
refine on near-misses, greedy excision shrink, round-robin scheduling and
termination); this module is a thin synchronous wrapper plus a CLI.
"""

from __future__ import annotations

import argparse
import time
from typing import cast

import networkx as nx

from ai.cage.registry.forge import (
    DEFAULT_MAX_CANDIDATES,
    REFINE_MARGIN,
    ForgeGenerator,
)
from backend.utils.graph_utils import compute_girth, is_k_regular


def _girth_value(G: nx.Graph[int]) -> int:
    """Integer girth of G (a large sentinel if acyclic)."""
    girth = compute_girth(G)
    return 10**9 if isinstance(girth, float) else girth


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
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    excision_max_roots: int | None = None,
    excision_backtracks: int = 10000,
    time_budget: float | None = None,
    verbose: bool = False,
) -> nx.Graph[int] | None:
    """Forge a (k,g)-graph by driving the queue-driven pipeline to completion.

    Returns the first valid (k,g)-graph (k-regular, girth >= g) the pipeline
    produces, or None if it drains without one.

    Args:
        k, g:                  Target degree and girth.
        predictor:             Voltage predictor model_id ("girth_predictor" /
                               "tabu_predictor") or None for classical tabu.
        max_candidates:        Hard cap on lifts the producer pushes downstream.
        refine_max_iter:       Per-near-miss refine step budget.
        refine_margin:         Max girth gap below g a near-miss may have to be
                               worth refining.
        time_budget:           Optional wall-clock cap (seconds) for the forge.
        verbose:               Print pipeline progress.

    The ``max_voltage_attempts``, ``max_group_order``, ``tabu_iterations``,
    ``beam_width``, ``excision_max_roots`` and ``excision_backtracks`` arguments
    are retained for signature compatibility with the previous cascade; the
    pipeline manages those budgets internally.
    """
    _ = (
        max_voltage_attempts,
        max_group_order,
        tabu_iterations,
        beam_width,
        excision_max_roots,
        excision_backtracks,
    )

    gen = ForgeGenerator(
        k,
        g,
        model_id=predictor,
        refine_max_iter=refine_max_iter,
        refine_margin=refine_margin,
        max_candidates=max_candidates,
    )

    start_time = time.time()
    while not gen.is_complete:
        gen.step()
        if verbose and gen.last_event is not None:
            print(f"[forge] {gen.last_event['action']}")
        if time_budget is not None and time.time() - start_time > time_budget:
            if verbose:
                print("[forge] time budget exceeded")
            break

    if not gen.success:
        if verbose:
            print("[forge] pipeline produced no valid (k,g)-graph")
        return None

    result = gen.graph
    if not is_k_regular(result, k) or _girth_value(result) < g:
        if verbose:
            print("[forge] final graph failed verification")
        return None
    return result


def main() -> None:
    """CLI: forge a (k,g)-graph and print its order/girth/regularity."""
    parser = argparse.ArgumentParser(description="Forge a (k,g)-graph by pipeline")
    _ = parser.add_argument("k", type=int, help="Degree k")
    _ = parser.add_argument("g", type=int, help="Girth g")
    _ = parser.add_argument(
        "--predictor",
        type=str,
        default="girth_predictor",
        help="Voltage predictor model_id, or 'none' for classical tabu",
    )
    _ = parser.add_argument(
        "--max-candidates",
        type=int,
        default=DEFAULT_MAX_CANDIDATES,
        help="Hard cap on lifts pushed downstream by the producer",
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
        max_candidates=cast(int, args.max_candidates),
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
