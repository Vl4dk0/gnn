"""Query benchmark run results — an agent-friendly read API over results/runs/.

This is the tool to reach for when you need to *report* on a benchmark: pull
one approach's numbers, compare variants, or extract a single (k,g) cell,
without hand-parsing CSVs or jsonl. It reads a run's ``results.jsonl`` (the
source of truth), filters, aggregates per (benchmark, approach, variant,
target), and prints JSON by default so the output drops straight into prose or
further processing.

Run ``python -m results.query --help`` for the full flag list and examples.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
from pathlib import Path

from results.metrics import TrialResult

# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------


def list_runs(out_root: Path) -> list[Path]:
    """Return run directories under *out_root*, newest last (sorted by name).

    Run dirs are timestamped ``YYYY-MM-DD_HH-MM-SS`` so lexical sort == time
    order. Only dirs containing a ``results.jsonl`` count as runs.
    """
    if not out_root.is_dir():
        return []
    return sorted(p for p in out_root.iterdir() if (p / "results.jsonl").is_file())


def resolve_run(run: str | None, out_root: Path) -> Path:
    """Resolve the run to query.

    *run* may be a full path, a bare timestamp under *out_root*, or None (use
    the newest run). Raises FileNotFoundError with a helpful message otherwise.
    """
    if run is not None:
        cand = Path(run)
        if not (cand / "results.jsonl").is_file():
            cand = out_root / run
        if not (cand / "results.jsonl").is_file():
            raise FileNotFoundError(
                f"No results.jsonl found for run {run!r} (looked at {cand})."
            )
        return cand

    runs = list_runs(out_root)
    if not runs:
        raise FileNotFoundError(
            f"No runs found under {out_root}. Run the benchmark first "
            + "(see the 'benchmark' skill), or pass --run / --out-root."
        )
    return runs[-1]


def load_records(run_dir: Path) -> list[TrialResult]:
    """Load every trial from a run's results.jsonl as a typed TrialResult."""
    text = (run_dir / "results.jsonl").read_text()
    return [
        TrialResult(**json.loads(line)) for line in text.splitlines() if line.strip()
    ]


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def target_str(rec: TrialResult) -> str:
    """Human-readable target tag: '(k,g)' for cage-like, else '' (node tasks)."""
    t = rec.target
    if "k" in t and "g" in t:
        return f"({t['k']},{t['g']})"
    return ""


def filter_records(
    records: list[TrialResult],
    *,
    benchmark: str | None,
    approach: str | None,
    variant: str | None,
    target: tuple[int, int] | None,
) -> list[TrialResult]:
    """Keep records matching every supplied filter (None = no constraint)."""
    out: list[TrialResult] = []
    for r in records:
        if benchmark is not None and r.benchmark != benchmark:
            continue
        if approach is not None and r.approach != approach:
            continue
        if variant is not None and r.variant != variant:
            continue
        if target is not None and (r.target.get("k"), r.target.get("g")) != target:
            continue
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def aggregate(records: list[TrialResult]) -> list[dict[str, object]]:
    """Group records by (benchmark, approach, variant, target) and summarise.

    Counts use all trials in the group; timing/size/metric means use only the
    SOLVED trials (so a reported time is a real time-to-solve, not polluted by
    budget-timeout failures). Groups are sorted by Moore bound then target.
    """
    groups: dict[tuple[str, str, str, str], list[TrialResult]] = {}
    for r in records:
        key = (r.benchmark, r.approach, r.variant, target_str(r))
        groups.setdefault(key, []).append(r)

    rows: list[dict[str, object]] = []
    for (benchmark, approach, variant, target), recs in groups.items():
        solved = [r for r in recs if r.success]

        metric_keys: list[str] = []
        for r in recs:
            for mk in r.metrics:
                if mk not in metric_keys:
                    metric_keys.append(mk)
        metric_means: dict[str, float | None] = {
            mk: _mean([r.metrics[mk] for r in solved if mk in r.metrics])
            for mk in metric_keys
        }

        moore: int | None = next(
            (int(r.metrics["moore_bound"]) for r in recs if "moore_bound" in r.metrics),
            None,
        )
        rows.append(
            {
                "benchmark": benchmark,
                "approach": approach,
                "variant": variant,
                "target": target,
                "moore_bound": moore,
                "n_trials": len(recs),
                "n_solved": len(solved),
                "success_rate": len(solved) / len(recs) if recs else 0.0,
                "mean_time_s": _mean([r.elapsed_s for r in solved]),
                "mean_nodes": _mean(
                    [float(r.n_nodes) for r in solved if r.n_nodes is not None]
                ),
                "mean_edges": _mean(
                    [float(r.n_edges) for r in solved if r.n_edges is not None]
                ),
                "metrics": metric_means,
            }
        )

    def _sort_key(row: dict[str, object]) -> tuple[bool, int, str]:
        mb = row["moore_bound"]
        return (mb is None, mb if isinstance(mb, int) else 0, str(row["target"]))

    rows.sort(key=_sort_key)
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _fmt(v: object) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def render_table(rows: list[dict[str, object]], raw: bool) -> str:
    """Render rows as a fixed-width text table (for a human glance)."""
    if not rows:
        return "(no matching records)"
    if raw:
        cols = [
            "benchmark",
            "approach",
            "variant",
            "target",
            "success",
            "elapsed_s",
            "n_nodes",
            "n_edges",
        ]
    else:
        cols = [
            "benchmark",
            "approach",
            "variant",
            "target",
            "moore_bound",
            "n_solved",
            "n_trials",
            "success_rate",
            "mean_time_s",
            "mean_nodes",
            "mean_edges",
        ]
    widths = {c: max([len(c), *(len(_fmt(r.get(c))) for r in rows)]) for c in cols}
    head = "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)
    body = "\n".join(
        "  ".join(_fmt(r.get(c)).ljust(widths[c]) for c in cols) for r in rows
    )
    return f"{head}\n{sep}\n{body}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_EPILOG = """\
Examples
--------
  # Everything about forge in the latest run, as JSON (default)
  python -m results.query --approach forge

  # Compare the voltage variants on one target
  python -m results.query --benchmark cage --approach voltage --target 4-6

  # A single forge ablation, human-readable table
  python -m results.query --approach forge --variant no_excision --format table

  # Per-trial rows (not aggregated), e.g. to see each seed
  python -m results.query --approach forge --target 4-5 --raw

  # Pick a specific run, or list what's available
  python -m results.query --run 2026-05-31_01-32-27 --approach forge
  python -m results.query --list-runs

Notes
-----
  * Default run is the NEWEST under results/runs/ (no need to know the stamp).
  * Aggregated output groups by (benchmark, approach, variant, target). Counts
    use all trials; mean time/size/metrics use only SOLVED trials, so a reported
    time is a true time-to-solve. success_rate = n_solved / n_trials.
  * For cage, compare mean_nodes against moore_bound (ratio 1.0 = the cage).
  * Names match the benchmark exactly. Cage approaches: randomwalk, astar,
    bruteforce, rl, voltage, voltage_rl, forge. voltage variants:
    Algebraic/GirthPredictor/TabuPredictor. forge variants: '' (full),
    no_refine, no_excision. Node tasks (degree/min_cycle): the architecture is
    the approach (gcn/gin/sage/gps/loopy).
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m results.query",
        description="Query benchmark run results (reads results/runs/<stamp>/results.jsonl).",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _ = parser.add_argument(
        "--run", default=None, help="Run timestamp or path (default: newest run)"
    )
    _ = parser.add_argument(
        "--out-root",
        default="results/runs",
        help="Where runs live (default: results/runs)",
    )
    _ = parser.add_argument(
        "--list-runs", action="store_true", help="List available runs and exit"
    )
    _ = parser.add_argument(
        "--benchmark",
        default=None,
        help="Filter: degree|min_cycle|cage|refine|excision",
    )
    _ = parser.add_argument(
        "--approach",
        default=None,
        help="Filter by approach (e.g. forge, voltage, astar)",
    )
    _ = parser.add_argument(
        "--variant",
        default=None,
        help="Filter by variant (e.g. no_excision, GirthPredictor)",
    )
    _ = parser.add_argument(
        "--target", default=None, help="Filter cage target as 'k-g' (e.g. 4-6)"
    )
    _ = parser.add_argument(
        "--raw", action="store_true", help="Emit per-trial rows instead of aggregating"
    )
    _ = parser.add_argument(
        "--format",
        default="json",
        choices=["json", "table", "md"],
        help="Output format (default: json)",
    )
    args = parser.parse_args()

    out_root = Path(str(args.out_root))
    fmt: str = str(args.format)

    if bool(args.list_runs):
        run_names = [p.name for p in list_runs(out_root)]
        print(
            json.dumps(run_names, indent=2)
            if fmt == "json"
            else ("\n".join(run_names) or "(no runs)")
        )
        return

    run_arg: str | None = None if args.run is None else str(args.run)
    run_dir = resolve_run(run_arg, out_root)

    target: tuple[int, int] | None = None
    if args.target is not None:
        k_str, g_str = str(args.target).split("-")
        target = (int(k_str), int(g_str))

    records = filter_records(
        load_records(run_dir),
        benchmark=None if args.benchmark is None else str(args.benchmark),
        approach=None if args.approach is None else str(args.approach),
        variant=None if args.variant is None else str(args.variant),
        target=target,
    )

    payload: dict[str, object] = {
        "run": str(run_dir),
        "filters": {
            "benchmark": args.benchmark,
            "approach": args.approach,
            "variant": args.variant,
            "target": args.target,
        },
        "n_records": len(records),
    }
    raw: bool = bool(args.raw)
    if raw:
        table_rows = [dataclasses.asdict(r) for r in records]
        payload["trials"] = table_rows
    else:
        table_rows = aggregate(records)
        payload["groups"] = table_rows

    if fmt == "json":
        print(json.dumps(payload, indent=2))
    elif fmt == "md":
        print(f"# query: {run_dir.name}  (n={len(records)})\n")
        print(render_table(table_rows, raw).replace("  ", " | "))
    else:
        print(f"run: {run_dir}   n_records: {len(records)}")
        print(render_table(table_rows, raw))


if __name__ == "__main__":
    main()
