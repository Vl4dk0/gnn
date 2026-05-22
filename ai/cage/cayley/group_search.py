"""Group-guided Cayley cage search — the ML-assisted meta-search.

This is where the group-promise predictor earns (or fails to earn) its
keep. Classical meta-search runs the expensive inner search on every
candidate group. Here, the trained predictor first scores every group's
best achievable girth in one cheap forward pass; groups predicted to fall
short of the target are pruned, and the classical inner search runs only
on the survivors.

The honest test is the comparison against ai.cage.cayley.baseline: does
group-guided search reach the same smallest order while touching fewer
groups? compare_to_baseline() runs both and reports the gap.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing
import time
from typing import cast

from ai.cage.cayley.group_model import (
    GroupPromisePredictor,
    load_group_promise_predictor,
    make_group_filter,
)
from ai.cage.cayley.groups import FiniteGroup, available_groups
from ai.cage.cayley.search import meta_search, search_one_group
from backend.utils.graph_utils import moore_bound


def group_guided_meta_search(
    k: int,
    g_target: int,
    model: GroupPromisePredictor,
    max_group_order: int = 2000,
    num_random_trials: int = 2000,
    num_tabu_iters: int = 1500,
    prune_slack: float = 1.0,
    num_workers: int = 1,
    verbose: bool = True,
) -> dict[str, object]:
    """Meta-search that prunes the catalogue with the group-promise predictor.

    A group is kept only if its predicted best achievable girth is at
    least g_target - prune_slack. The classical inner search then runs on
    the kept groups exactly as in the baseline.
    """
    groups = available_groups(max_group_order)
    mb = moore_bound(k, g_target)
    groups = [g for g in groups if mb <= g.order <= max(4 * mb, mb + 500)]
    if not groups:
        groups = available_groups(max_group_order)
    groups.sort(key=lambda g: g.order)

    predict = make_group_filter(model)

    start = time.time()
    kept: list[FiniteGroup] = []
    predictions: dict[str, float] = {}
    for group in groups:
        pred = predict(group, k, g_target)
        predictions[group.name] = pred
        if pred >= g_target - prune_slack:
            kept.append(group)

    if verbose:
        print(f"Group-guided search for ({k},{g_target}). Moore bound = {mb}.")
        print(
            f"  Catalogue: {len(groups)} groups | "
            + f"kept after pruning: {len(kept)} | workers: {num_workers}"
        )

    best: dict[str, object] = {
        "girth": 0,
        "order": math.inf,
        "generators": None,
        "group_name": None,
    }
    best_order: float = math.inf

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
                    + f"order={order} S={result['generators']}"
                )

    configs: list[tuple[FiniteGroup, int, int, int, int]] = [
        (group, k, g_target, num_random_trials, num_tabu_iters) for group in kept
    ]

    if num_workers > 1 and configs:
        with multiprocessing.Pool(num_workers) as pool:
            for result in pool.imap_unordered(search_one_group, configs):
                _update(result)
    else:
        for cfg in configs:
            _update(search_one_group(cfg))

    elapsed = time.time() - start
    best["elapsed"] = elapsed
    best["num_groups"] = len(groups)
    best["groups_kept"] = len(kept)
    best["groups_pruned"] = len(groups) - len(kept)
    if verbose:
        print(f"\nGroup-guided search done in {elapsed:.1f}s.")
        if best["generators"] is not None:
            print(f"  Best: order={best['order']} girth={best['girth']}")
        else:
            print(f"  No ({k},{g_target})-Cayley graph found in kept groups.")
    return best


def compare_to_baseline(
    k: int,
    g_target: int,
    model: GroupPromisePredictor,
    max_group_order: int = 2000,
    num_random_trials: int = 2000,
    num_tabu_iters: int = 1500,
    prune_slack: float = 1.0,
    num_workers: int = 1,
    verbose: bool = True,
) -> dict[str, object]:
    """Run both the classical baseline and group-guided search; report the gap."""
    if verbose:
        print("\n--- classical baseline ---")
    base = meta_search(
        k,
        g_target,
        max_group_order=max_group_order,
        num_random_trials=num_random_trials,
        num_tabu_iters=num_tabu_iters,
        score_fn=None,
        num_workers=num_workers,
        verbose=verbose,
    )
    if verbose:
        print("\n--- group-guided search ---")
    guided = group_guided_meta_search(
        k,
        g_target,
        model,
        max_group_order=max_group_order,
        num_random_trials=num_random_trials,
        num_tabu_iters=num_tabu_iters,
        prune_slack=prune_slack,
        num_workers=num_workers,
        verbose=verbose,
    )

    def _order(d: dict[str, object]) -> float:
        return float(cast(float, d["order"]))

    summary: dict[str, object] = {
        "k": k,
        "g": g_target,
        "baseline_order": _order(base),
        "guided_order": _order(guided),
        "baseline_time": base.get("elapsed"),
        "guided_time": guided.get("elapsed"),
        "groups_total": guided.get("num_groups"),
        "groups_kept": guided.get("groups_kept"),
        "groups_pruned": guided.get("groups_pruned"),
        "same_result": _order(base) == _order(guided),
    }
    if verbose:
        print("\n=== comparison ===")
        print(
            f"  baseline order={summary['baseline_order']} "
            + f"time={summary['baseline_time']}"
        )
        print(
            f"  guided   order={summary['guided_order']} "
            + f"time={summary['guided_time']} "
            + f"(kept {summary['groups_kept']}/{summary['groups_total']})"
        )
        print(f"  same result: {summary['same_result']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Group-guided Cayley cage search")
    _ = parser.add_argument("k", type=int)
    _ = parser.add_argument("g", type=int)
    _ = parser.add_argument(
        "--model-id", type=str, default="group_promise_predictor_multi"
    )
    _ = parser.add_argument("--max-order", type=int, default=2000)
    _ = parser.add_argument("--trials", type=int, default=2000)
    _ = parser.add_argument("--tabu-iters", type=int, default=1500)
    _ = parser.add_argument("--prune-slack", type=float, default=1.0)
    _ = parser.add_argument("--workers", type=int, default=1)
    _ = parser.add_argument(
        "--compare",
        action="store_true",
        help="Also run the classical baseline and report the gap",
    )
    args = parser.parse_args()

    model = load_group_promise_predictor(cast(str, args.model_id))
    print(f"Loaded group-promise predictor: {args.model_id}")

    if cast(bool, args.compare):
        _ = compare_to_baseline(
            cast(int, args.k),
            cast(int, args.g),
            model,
            max_group_order=cast(int, args.max_order),
            num_random_trials=cast(int, args.trials),
            num_tabu_iters=cast(int, args.tabu_iters),
            prune_slack=cast(float, args.prune_slack),
            num_workers=cast(int, args.workers),
            verbose=True,
        )
    else:
        _ = group_guided_meta_search(
            cast(int, args.k),
            cast(int, args.g),
            model,
            max_group_order=cast(int, args.max_order),
            num_random_trials=cast(int, args.trials),
            num_tabu_iters=cast(int, args.tabu_iters),
            prune_slack=cast(float, args.prune_slack),
            num_workers=cast(int, args.workers),
            verbose=True,
        )


if __name__ == "__main__":
    main()


__all__ = ["compare_to_baseline", "group_guided_meta_search"]
