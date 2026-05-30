"""Parallel benchmark runner and CLI.

Run with:
    uv run python -m results.runner --quick
    uv run python -m results.runner --benchmarks example --quick
    uv run python -m results.runner --benchmarks cage,refine --seeds 3
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import torch

import ai.utils.device as _dev
import results.benchmarks  # noqa: F401 — populates REGISTRY via each module's register() call
import results.report
from results.metrics import TrialResult
from results.registry import REGISTRY, RunConfig, Task


def _init_worker() -> None:
    """Worker initializer: force CPU-only torch, disable MPS/CUDA.

    Sets CUDA_VISIBLE_DEVICES before any CUDA context is created, then
    restricts torch to one thread per worker and patches the device selector
    so all model loads and inference use CPU only.  This avoids MPS +
    multiprocessing incompatibilities on macOS.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    _dev.get_preferred_device = lambda: "cpu"
    torch.set_default_device("cpu")


def _execute(task: Task) -> list[TrialResult]:
    """Module-level worker entrypoint: look up benchmark and run the task."""
    try:
        return REGISTRY[task.benchmark].execute(task)
    except Exception as exc:
        return [
            TrialResult(
                benchmark=task.benchmark,
                approach="",
                variant="",
                label=task.label,
                instance="",
                target={},
                success=False,
                elapsed_s=0.0,
                steps=None,
                n_nodes=None,
                n_edges=None,
                error=repr(exc),
            )
        ]


def run(config: RunConfig) -> list[TrialResult]:
    """Gather tasks and run them in parallel; return all TrialResults."""
    if config.benchmarks == ["all"]:
        names = list(REGISTRY.keys())
    else:
        names = config.benchmarks

    tasks: list[Task] = []
    for name in names:
        if name not in REGISTRY:
            print(f"[warn] Unknown benchmark '{name}' — skipping", file=sys.stderr)
            continue
        tasks.extend(REGISTRY[name].make_tasks(config))

    total = len(tasks)
    workers = config.workers
    if workers is None:
        workers = min(8, max(1, (os.cpu_count() or 4) - 2))

    print(f"Running {total} tasks across {workers} workers …")

    trial_results: list[TrialResult] = []
    done = 0

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        futures = {pool.submit(_execute, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            done += 1
            try:
                batch = future.result()
            except Exception as exc:
                batch = [
                    TrialResult(
                        benchmark=task.benchmark,
                        approach="",
                        variant="",
                        label=task.label,
                        instance="",
                        target={},
                        success=False,
                        elapsed_s=0.0,
                        steps=None,
                        n_nodes=None,
                        n_edges=None,
                        error=repr(exc),
                    )
                ]
            print(f"[{done}/{total}] {task.label}")
            trial_results.extend(batch)

    return trial_results


def main() -> None:
    parser = argparse.ArgumentParser(description="GNN benchmark runner")
    parser.add_argument(
        "--benchmarks",
        default="all",
        help='Comma-separated benchmark names or "all" (default: all)',
    )
    parser.add_argument("--quick", action="store_true", help="Use small battery")
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds")
    parser.add_argument(
        "--cage-budget",
        type=float,
        default=20.0,
        dest="cage_budget",
        help="Cage time budget in seconds",
    )
    parser.add_argument(
        "--cage-max-steps",
        type=int,
        default=200_000,
        dest="cage_max_steps",
        help="Cage max search steps",
    )
    parser.add_argument("--workers", type=int, default=None, help="Worker processes")
    parser.add_argument(
        "--out-root",
        default="results/runs",
        dest="out_root",
        help="Root directory for run outputs",
    )
    args = parser.parse_args()

    # argparse Namespace fields are dynamically typed; extract with explicit casts
    benchmarks_str: str = str(args.benchmarks)
    quick: bool = bool(args.quick)
    seeds: int = int(args.seeds)
    cage_budget: float = float(args.cage_budget)
    cage_max_steps: int = int(args.cage_max_steps)
    workers: int | None = int(args.workers) if args.workers is not None else None
    out_root: str = str(args.out_root)

    benchmark_names = [b.strip() for b in benchmarks_str.split(",")]

    config = RunConfig(
        benchmarks=benchmark_names,
        quick=quick,
        seeds=seeds,
        cage_time_budget_s=cage_budget,
        cage_max_steps=cage_max_steps,
        workers=workers,
    )

    t0 = time.perf_counter()
    run_results = run(config)
    wall = time.perf_counter() - t0

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(out_root) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "raw.json"
    _ = raw_path.write_text(
        json.dumps([dataclasses.asdict(r) for r in run_results], indent=2)
    )

    results.report.write_reports(out_dir, run_results)

    success_count = sum(1 for r in run_results if r.success)
    print(
        f"\nDone in {wall:.1f}s — {success_count}/{len(run_results)} succeeded.  "
        + f"Outputs: {out_dir}"
    )


if __name__ == "__main__":
    main()
