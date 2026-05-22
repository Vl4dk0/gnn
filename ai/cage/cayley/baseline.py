"""Classical (no-ML) Cayley graph cage finder — the baseline.

This is the bottom line every ML-assisted result must beat. For each
(k, g) target it searches a catalogue of finite groups for the smallest
group admitting a degree-k symmetric generating set whose Cayley graph has
girth >= g, using only classical random + tabu search. No neural network is
involved anywhere.

The output table reports, per target, the smallest Cayley (k,g)-graph found
next to the order of the known (k,g)-cage, so the gap is visible at a
glance. The known cage need not itself be a Cayley graph, so a positive gap
is expected and is not, on its own, a failure of the search.

Run:

    uv run python -m ai.cage.cayley.baseline \\
        --targets "3,6;3,7;3,8" --max-order 2000 --workers 8
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from typing import cast

from ai.cage.cayley.search import meta_search

# Orders of the known (k, g)-cages. The known cage is not necessarily a
# Cayley graph, so these are reference values, not search targets the
# Cayley-restricted search is guaranteed to reach.
KNOWN_CAGES: dict[tuple[int, int], int] = {
    (3, 3): 4,
    (3, 4): 6,
    (3, 5): 10,
    (3, 6): 14,
    (3, 7): 24,
    (3, 8): 30,
    (3, 9): 58,
    (3, 10): 70,
    (3, 11): 112,
    (3, 12): 126,
    (4, 3): 5,
    (4, 4): 8,
    (4, 5): 19,
    (4, 6): 26,
    (4, 7): 67,
    (4, 8): 80,
    (5, 3): 6,
    (5, 4): 10,
    (5, 5): 30,
    (5, 6): 42,
    (6, 3): 7,
    (6, 4): 12,
    (6, 5): 40,
    (6, 6): 62,
    (7, 5): 50,
}


def parse_targets(spec: str) -> list[tuple[int, int]]:
    """Parse a "k,g;k,g;..." target specification into (k, g) tuples."""
    targets: list[tuple[int, int]] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(",")
        if len(parts) != 2:
            raise ValueError(f"bad target {chunk!r}: expected 'k,g'")
        k, g = int(parts[0]), int(parts[1])
        if k < 2 or g < 3:
            raise ValueError(f"bad target {chunk!r}: need k >= 2 and g >= 3")
        targets.append((k, g))
    if not targets:
        raise ValueError("no targets parsed")
    return targets


def run_baseline(
    targets: list[tuple[int, int]],
    max_group_order: int = 2000,
    num_random_trials: int = 2000,
    num_tabu_iters: int = 1500,
    num_workers: int = 1,
    seed: int | None = None,
    verbose: bool = True,
) -> list[dict[str, object]]:
    """Run the classical Cayley search for every target; return result rows."""
    rows: list[dict[str, object]] = []

    for k, g in targets:
        if seed is not None:
            random.seed(seed)
        if verbose:
            print(f"\n=== Target ({k},{g}) ===")

        best = meta_search(
            k,
            g,
            max_group_order=max_group_order,
            num_random_trials=num_random_trials,
            num_tabu_iters=num_tabu_iters,
            score_fn=None,
            num_workers=num_workers,
            verbose=verbose,
        )

        found = best.get("generators") is not None
        order = best.get("order")
        found_order = int(cast(int, order)) if found else None
        cage_order = KNOWN_CAGES.get((k, g))
        gap = (
            found_order - cage_order
            if (found_order is not None and cage_order is not None)
            else None
        )

        rows.append(
            {
                "k": k,
                "g": g,
                "cage_order": cage_order,
                "found_order": found_order,
                "found_girth": best.get("girth") if found else None,
                "gap": gap,
                "group_name": best.get("group_name") if found else None,
                "generators": best.get("generators") if found else None,
                "method": best.get("method") if found else None,
                "num_groups": best.get("num_groups"),
                "elapsed": best.get("elapsed"),
            }
        )

    return rows


def format_table(rows: list[dict[str, object]]) -> str:
    """Render result rows as an aligned monospace table."""
    header = (
        f"{'(k,g)':>8}  {'cage':>6}  {'Cayley':>7}  {'gap':>5}  "
        + f"{'girth':>5}  {'group':<22}  {'method':<8}  {'groups':>7}  {'time(s)':>8}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        kg = f"({r['k']},{r['g']})"
        cage = r["cage_order"]
        cage_s = str(cage) if cage is not None else "?"
        found = r["found_order"]
        found_s = str(found) if found is not None else "NOT FOUND"
        gap = r["gap"]
        gap_s = (
            (f"+{gap}" if cast(int, gap) >= 0 else str(gap)) if gap is not None else "-"
        )
        girth = r["found_girth"]
        girth_s = str(girth) if girth is not None else "-"
        group = cast("str | None", r["group_name"]) or "-"
        method = cast("str | None", r["method"]) or "-"
        ng = r["num_groups"]
        ng_s = str(ng) if ng is not None else "-"
        elapsed = r["elapsed"]
        elapsed_s = f"{cast(float, elapsed):.1f}" if elapsed is not None else "-"
        lines.append(
            f"{kg:>8}  {cage_s:>6}  {found_s:>7}  {gap_s:>5}  "
            + f"{girth_s:>5}  {group:<22}  {method:<8}  {ng_s:>7}  {elapsed_s:>8}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classical (no-ML) Cayley graph cage finder — the baseline"
    )
    _ = parser.add_argument(
        "--targets",
        type=str,
        default="3,6;3,7;3,8",
        help="Semicolon-separated (k,g) targets, e.g. '3,6;3,7;4,6'",
    )
    _ = parser.add_argument("--max-order", type=int, default=2000)
    _ = parser.add_argument("--trials", type=int, default=2000)
    _ = parser.add_argument("--tabu-iters", type=int, default=1500)
    _ = parser.add_argument("--workers", type=int, default=1)
    _ = parser.add_argument("--seed", type=int, default=None)
    _ = parser.add_argument(
        "--json", type=str, default=None, help="Optional path to write results JSON"
    )
    args = parser.parse_args()

    targets = parse_targets(cast(str, args.targets))
    start = time.time()
    rows = run_baseline(
        targets,
        max_group_order=cast(int, args.max_order),
        num_random_trials=cast(int, args.trials),
        num_tabu_iters=cast(int, args.tabu_iters),
        num_workers=cast(int, args.workers),
        seed=cast("int | None", args.seed),
        verbose=True,
    )
    total = time.time() - start

    print("\n" + "=" * 78)
    print("CLASSICAL CAYLEY BASELINE — results")
    print("=" * 78)
    print(format_table(rows))
    print(f"\nTotal wall-clock: {total:.1f}s over {len(rows)} target(s).")

    json_path = cast("str | None", args.json)
    if json_path is not None:
        payload: dict[str, object] = {
            "max_order": cast(int, args.max_order),
            "trials": cast(int, args.trials),
            "tabu_iters": cast(int, args.tabu_iters),
            "seed": cast("int | None", args.seed),
            "total_seconds": total,
            "results": rows,
        }

        def _fallback(o: object) -> object:
            return None if o == math.inf else o

        with open(json_path, "w") as fh:
            json.dump(payload, fh, indent=2, default=_fallback)
        print(f"Wrote results JSON to {json_path}")


if __name__ == "__main__":
    main()


__all__ = ["KNOWN_CAGES", "format_table", "parse_targets", "run_baseline"]
