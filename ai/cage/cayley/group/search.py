"""Group-guided Cayley cage search — the ML-assisted meta-search.

This is where the group-promise predictor earns (or fails to earn) its
keep. Classical meta-search runs the expensive inner search on every
candidate group. Here, the trained predictor first scores every group's
best achievable girth in one cheap forward pass; groups predicted to fall
short of the target are pruned, and the classical inner search runs only
on the survivors.

The honest test is the comparison against ai.cage.cayley.baseline:
run_comparison() loads the cached classical baseline (so we don't repay
its 11-hour wall-clock), then for each target reports:

  * the predictor's predicted girth for the baseline-winning group and its
    rank among all catalogue groups — a direct test of predictor quality;
  * the guided search at several prune_slack values, with order found,
    wall-clock, and groups examined.

The interesting question is not "does ML beat classical?" — exact girth
eval is cheap so that's the wrong question — but "how aggressively can
the predictor prune the catalogue before the optimum is lost?"
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import random
import time
from collections.abc import Callable
from typing import cast

from ai.cage.cayley.generators.search import meta_search, search_one_group
from ai.cage.cayley.group.model import (
    GroupPromisePredictor,
    load_group_promise_predictor,
    make_group_filter,
)
from ai.cage.cayley.groups import FiniteGroup, available_groups
from ai.cage.train_utils import parse_targets
from backend.utils.graph_utils import moore_bound

PredictFn = Callable[[FiniteGroup, int, int], float]


def _filtered_catalogue(
    k: int, g_target: int, max_group_order: int
) -> list[FiniteGroup]:
    """The (moore-filtered, order-sorted) group catalogue both search modes share."""
    groups = available_groups(max_group_order)
    mb = moore_bound(k, g_target)
    filtered = [g for g in groups if mb <= g.order <= max(4 * mb, mb + 500)]
    if not filtered:
        filtered = groups
    filtered.sort(key=lambda g: g.order)
    return filtered


def group_guided_meta_search(
    k: int,
    g_target: int,
    *,
    model: GroupPromisePredictor | None = None,
    predict_fn: PredictFn | None = None,
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

    Pass either a model or a predict_fn (the experiment harness caches
    predictions across slack values via predict_fn).
    """
    if predict_fn is None:
        if model is None:
            raise ValueError("provide either model or predict_fn")
        predict_fn = make_group_filter(model)

    groups = _filtered_catalogue(k, g_target, max_group_order)

    start = time.time()
    kept: list[FiniteGroup] = []
    for group in groups:
        if predict_fn(group, k, g_target) >= g_target - prune_slack - 1e-6:
            kept.append(group)

    if verbose:
        mb = moore_bound(k, g_target)
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


def load_baseline_json(
    path: str,
) -> dict[tuple[int, int], dict[str, object]]:
    """Load cached classical-baseline results keyed by (k, g)."""
    with open(path) as f:
        payload = cast(dict[str, object], json.load(f))
    results = cast(list[dict[str, object]], payload.get("results", []))
    out: dict[tuple[int, int], dict[str, object]] = {}
    for row in results:
        k = int(cast(int, row["k"]))
        g = int(cast(int, row["g"]))
        out[(k, g)] = row
    return out


def run_comparison(
    targets: list[tuple[int, int]],
    model: GroupPromisePredictor,
    *,
    baseline_results: dict[tuple[int, int], dict[str, object]] | None = None,
    max_group_order: int = 2000,
    num_random_trials: int = 2000,
    num_tabu_iters: int = 1500,
    slack_values: tuple[float, ...] = (0.0, 1.0, 2.0),
    num_workers: int = 1,
    seed: int = 42,
    verbose: bool = True,
) -> list[dict[str, object]]:
    """Per (k, g): predictor diagnostics + guided search at each slack value.

    For each target we report:
      * the predictor's predicted girth for the baseline-winning group
        and its rank among all catalogue groups (predictor quality);
      * for each slack: order found, wall-clock, groups examined,
        whether the same order as the classical baseline was found.
    """
    rows: list[dict[str, object]] = []
    base_predict = make_group_filter(model)

    for k, g in targets:
        if verbose:
            print(f"\n=== ({k},{g}) ===")

        if baseline_results and (k, g) in baseline_results:
            base = baseline_results[(k, g)]
            baseline_order = base.get("found_order")
            winner_name = cast("str | None", base.get("group_name"))
            baseline_time = base.get("elapsed")
            if verbose:
                print(
                    f"  baseline (cached): order={baseline_order} "
                    + f"group={winner_name} time={baseline_time}"
                )
        else:
            if verbose:
                print("  no cached baseline — running classical meta_search")
            random.seed(seed)
            base_run = meta_search(
                k,
                g,
                max_group_order=max_group_order,
                num_random_trials=num_random_trials,
                num_tabu_iters=num_tabu_iters,
                score_fn=None,
                num_workers=num_workers,
                verbose=False,
            )
            order_val = base_run.get("order")
            baseline_order = (
                int(cast(int, order_val))
                if base_run.get("generators") is not None
                else None
            )
            winner_name = cast("str | None", base_run.get("group_name"))
            baseline_time = base_run.get("elapsed")

        # Score every group in the same filtered catalogue the guided
        # search uses. Cache the predictions so the slack sweep reuses them.
        groups = _filtered_catalogue(k, g, max_group_order)
        predictions: dict[str, float] = {
            group.name: base_predict(group, k, g) for group in groups
        }

        # Rank of the baseline-winning group in the predictor's ordering.
        ranked = sorted(predictions.items(), key=lambda kv: -kv[1])
        winner_pred: float | None = None
        winner_rank: int | None = None
        if winner_name is not None:
            for rank, (name, pred) in enumerate(ranked, start=1):
                if name == winner_name:
                    winner_pred = pred
                    winner_rank = rank
                    break
        if verbose:
            print(
                f"  predictor: winner_pred={winner_pred} "
                + f"rank={winner_rank}/{len(ranked)}"
            )

        def _cached(
            group: FiniteGroup,
            _k: int,
            _g: int,
            cache: dict[str, float] = predictions,
        ) -> float:
            return cache.get(group.name, 0.0)

        slack_results: list[dict[str, object]] = []
        for slack in slack_values:
            random.seed(seed)
            guided = group_guided_meta_search(
                k,
                g,
                predict_fn=_cached,
                max_group_order=max_group_order,
                num_random_trials=num_random_trials,
                num_tabu_iters=num_tabu_iters,
                prune_slack=slack,
                num_workers=num_workers,
                verbose=False,
            )
            found_order = (
                int(cast(int, guided["order"]))
                if guided["generators"] is not None
                else None
            )
            same = baseline_order is not None and found_order == baseline_order
            entry: dict[str, object] = {
                "slack": slack,
                "found_order": found_order,
                "found_girth": guided.get("girth"),
                "group_name": guided.get("group_name"),
                "elapsed": guided.get("elapsed"),
                "groups_kept": guided.get("groups_kept"),
                "groups_total": guided.get("num_groups"),
                "same_as_baseline": bool(same),
            }
            slack_results.append(entry)
            if verbose:
                print(
                    f"  slack={slack}: order={found_order} "
                    + f"kept={guided.get('groups_kept')}/{guided.get('num_groups')} "
                    + f"time={guided.get('elapsed', 0.0):.1f}s "
                    + f"same={same}"
                )

        rows.append(
            {
                "k": k,
                "g": g,
                "baseline_order": baseline_order,
                "baseline_group": winner_name,
                "baseline_time": baseline_time,
                "winner_pred": winner_pred,
                "winner_rank": winner_rank,
                "catalogue_size": len(ranked),
                "slacks": slack_results,
            }
        )

    return rows


def format_comparison_table(rows: list[dict[str, object]]) -> str:
    """Render run_comparison rows as an aligned table."""
    out: list[str] = []
    out.append("=" * 100)
    out.append("CAYLEY CAGE SEARCH — classical vs group-guided")
    out.append("=" * 100)
    for r in rows:
        kg = f"({r['k']},{r['g']})"
        base = r["baseline_order"] if r["baseline_order"] is not None else "-"
        group = cast("str | None", r["baseline_group"]) or "-"
        wp = r["winner_pred"]
        wp_s = f"{wp:.2f}" if isinstance(wp, float) else "-"
        rank = r["winner_rank"]
        rank_s = f"{rank}/{r['catalogue_size']}" if rank is not None else "-"
        out.append(
            f"\n{kg}  baseline order={base}  group={group}  "
            + f"predictor: pred={wp_s}  rank={rank_s}"
        )
        for s in cast(list[dict[str, object]], r["slacks"]):
            order = s["found_order"]
            order_s = str(order) if order is not None else "NOT FOUND"
            kept = s["groups_kept"]
            total = s["groups_total"]
            elapsed = s["elapsed"]
            elapsed_s = f"{elapsed:.1f}s" if isinstance(elapsed, float) else "-"
            mark = "OK" if s["same_as_baseline"] else "MISS"
            out.append(
                f"    slack {cast(float, s['slack']):>4.1f}  "
                + f"order={order_s:<10}  "
                + f"kept={kept}/{total:<5}  "
                + f"time={elapsed_s:<8}  {mark}"
            )
    return "\n".join(out)


def _parse_slacks(spec: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in spec.split(",") if x.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Group-guided Cayley cage search (single target or comparison)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp_search = sub.add_parser("search", help="Single-target group-guided search")
    _ = sp_search.add_argument("k", type=int)
    _ = sp_search.add_argument("g", type=int)
    _ = sp_search.add_argument(
        "--model-id", type=str, default="group_promise_predictor_multi"
    )
    _ = sp_search.add_argument("--max-order", type=int, default=2000)
    _ = sp_search.add_argument("--trials", type=int, default=2000)
    _ = sp_search.add_argument("--tabu-iters", type=int, default=1500)
    _ = sp_search.add_argument("--prune-slack", type=float, default=1.0)
    _ = sp_search.add_argument("--workers", type=int, default=1)

    sp_cmp = sub.add_parser(
        "compare",
        help="Multi-target comparison: predictor diagnostics + slack sweep",
    )
    _ = sp_cmp.add_argument(
        "--targets",
        type=str,
        required=True,
        help='Semicolon-separated, e.g. "3,5;3,6;3,7"',
    )
    _ = sp_cmp.add_argument(
        "--model-id", type=str, default="group_promise_predictor_multi"
    )
    _ = sp_cmp.add_argument(
        "--baseline-json",
        type=str,
        default=None,
        help="Path to cached baseline JSON (from ai.cage.cayley.baseline --json)",
    )
    _ = sp_cmp.add_argument(
        "--slacks", type=str, default="0.0,1.0,2.0", help="Comma-separated"
    )
    _ = sp_cmp.add_argument("--max-order", type=int, default=2000)
    _ = sp_cmp.add_argument("--trials", type=int, default=2000)
    _ = sp_cmp.add_argument("--tabu-iters", type=int, default=1500)
    _ = sp_cmp.add_argument("--workers", type=int, default=1)
    _ = sp_cmp.add_argument("--seed", type=int, default=42)
    _ = sp_cmp.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Where to write the full comparison JSON",
    )

    args = parser.parse_args()
    cmd = cast(str, args.cmd)
    model_id = cast(str, args.model_id)
    model = load_group_promise_predictor(model_id)
    print(f"Loaded group-promise predictor: {model_id}")

    if cmd == "search":
        _ = group_guided_meta_search(
            cast(int, args.k),
            cast(int, args.g),
            model=model,
            max_group_order=cast(int, args.max_order),
            num_random_trials=cast(int, args.trials),
            num_tabu_iters=cast(int, args.tabu_iters),
            prune_slack=cast(float, args.prune_slack),
            num_workers=cast(int, args.workers),
            verbose=True,
        )
        return

    # compare
    targets = parse_targets(cast(str, args.targets))
    slack_values = _parse_slacks(cast(str, args.slacks))
    baseline_path = cast("str | None", args.baseline_json)
    baseline_results = load_baseline_json(baseline_path) if baseline_path else None
    if baseline_results is not None:
        print(f"Loaded cached baseline for {len(baseline_results)} targets")

    start = time.time()
    rows = run_comparison(
        targets,
        model,
        baseline_results=baseline_results,
        max_group_order=cast(int, args.max_order),
        num_random_trials=cast(int, args.trials),
        num_tabu_iters=cast(int, args.tabu_iters),
        slack_values=slack_values,
        num_workers=cast(int, args.workers),
        seed=cast(int, args.seed),
        verbose=True,
    )
    elapsed = time.time() - start
    print("\n" + format_comparison_table(rows))
    print(f"\nTotal comparison wall-clock: {elapsed:.1f}s")

    output_path = cast("str | None", args.output_json)
    if output_path is not None:
        payload: dict[str, object] = {
            "targets": [list(t) for t in targets],
            "slacks": list(slack_values),
            "max_order": cast(int, args.max_order),
            "trials": cast(int, args.trials),
            "tabu_iters": cast(int, args.tabu_iters),
            "seed": cast(int, args.seed),
            "total_seconds": elapsed,
            "results": rows,
        }

        def _fallback(o: object) -> object:
            return None if o == math.inf else o

        with open(output_path, "w") as fh:
            json.dump(payload, fh, indent=2, default=_fallback)
        print(f"Wrote comparison JSON to {output_path}")


if __name__ == "__main__":
    main()


__all__ = [
    "format_comparison_table",
    "group_guided_meta_search",
    "load_baseline_json",
    "run_comparison",
]
