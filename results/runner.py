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
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from types import FrameType

import torch

import ai.utils.device as _dev
import results.benchmarks  # noqa: F401 — populates REGISTRY via each module's register() call
import results.report
from results.metrics import TrialResult
from results.registry import REGISTRY, RunConfig, Task

# Hard per-task wall-clock cap, set per worker via the pool initializer. A task
# that exceeds it (a runaway search step, an O(|E|^2) refine, …) is interrupted
# by SIGALRM and recorded as a timeout, rather than hanging the whole run.
_task_timeout_s: int = 120


def _timeout_handler(_signum: int, _frame: FrameType | None) -> None:
    raise TimeoutError(f"task exceeded {_task_timeout_s}s hard cap")


def _init_worker(task_timeout_s: float) -> None:
    """Worker initializer: force CPU-only torch and arm the per-task timeout.

    Sets CUDA_VISIBLE_DEVICES before any CUDA context is created, restricts
    torch to one thread per worker, patches the device selector so all model
    loads and inference use CPU only (avoids MPS/CUDA + multiprocessing
    issues), and installs a SIGALRM handler used to bound each task.
    """
    global _task_timeout_s
    _task_timeout_s = max(1, int(task_timeout_s))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    _dev.get_preferred_device = lambda: "cpu"
    torch.set_default_device("cpu")
    _ = signal.signal(signal.SIGALRM, _timeout_handler)


def _error_result(task: Task, message: str) -> TrialResult:
    """Build a failure stub for a timeout/crash.

    Carries (k,g), approach and variant from the task payload so timeouts and
    crashes still show up in the summary tables and the (k,g) solvability
    matrix as explicit failures, instead of dropping out with empty keys.
    """
    payload = task.payload
    k = payload.get("k")
    g = payload.get("g")
    target: dict[str, int] = (
        {"k": k, "g": g} if isinstance(k, int) and isinstance(g, int) else {}
    )
    approach = payload.get("approach", "")
    variant = payload.get("variant", "")
    name = payload.get("name")
    return TrialResult(
        benchmark=task.benchmark,
        approach=approach if isinstance(approach, str) else "",
        variant=variant if isinstance(variant, str) else "",
        label=task.label,
        instance=name if isinstance(name, str) else "",
        target=target,
        success=False,
        elapsed_s=0.0,
        steps=None,
        n_nodes=None,
        n_edges=None,
        error=message,
    )


def _execute(task: Task) -> list[TrialResult]:
    """Module-level worker entrypoint: run the task under a hard SIGALRM cap."""
    signal.alarm(_task_timeout_s)
    try:
        return REGISTRY[task.benchmark].execute(task)
    except TimeoutError as exc:
        return [_error_result(task, f"timeout: {exc}")]
    except Exception as exc:
        return [_error_result(task, repr(exc))]
    finally:
        signal.alarm(0)


def run(config: RunConfig, out_dir: Path) -> list[TrialResult]:
    """Gather tasks, run them in parallel, and stream results to disk.

    Each completed task's records are appended to ``out_dir/results.jsonl`` as
    they arrive, so partial results survive even if the process is later killed
    (OOM, time limit). Returns all collected TrialResults.
    """
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
    jsonl_path = out_dir / "results.jsonl"

    with open(jsonl_path, "w") as jsonl:
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(config.task_timeout_s,),
            ) as pool:
                futures = {pool.submit(_execute, task): task for task in tasks}
                for future in as_completed(futures):
                    task = futures[future]
                    done += 1
                    try:
                        batch = future.result()
                    except Exception as exc:
                        batch = [_error_result(task, repr(exc))]
                    for record in batch:
                        _ = jsonl.write(json.dumps(dataclasses.asdict(record)) + "\n")
                    jsonl.flush()
                    trial_results.extend(batch)
                    ok = sum(1 for r in batch if r.success)
                    print(f"[{done}/{total}] {task.label}  ({ok}/{len(batch)} ok)")
        except Exception as exc:  # pool broke (e.g. worker OOM-killed)
            print(f"[error] pool failed after {done}/{total}: {exc!r}", file=sys.stderr)

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
        "--seed-base",
        type=int,
        default=0,
        dest="seed_base",
        help="First seed value; seeds run over range(seed_base, seed_base+seeds)",
    )
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
        "--task-timeout",
        type=float,
        default=120.0,
        dest="task_timeout",
        help="Hard per-task wall-clock cap in seconds (SIGALRM)",
    )
    parser.add_argument(
        "--approaches",
        default=None,
        help=(
            "Comma-separated approaches to include (e.g. 'forge' or "
            "'voltage,forge'; for node tasks these are architectures like 'sage'). "
            "Default: all."
        ),
    )
    parser.add_argument(
        "--targets",
        default=None,
        help=(
            "Restrict cage (k,g) targets, as k-g pairs, e.g. '3-6' or '3-6,4-6'. "
            "Default: the full grid."
        ),
    )
    parser.add_argument(
        "--throughput-seconds",
        type=float,
        default=None,
        dest="throughput_seconds",
        help="Wall-clock budget per throughput trial (s)",
    )
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
    seed_base: int = int(args.seed_base)
    cage_budget: float = float(args.cage_budget)
    cage_max_steps: int = int(args.cage_max_steps)
    workers: int | None = int(args.workers) if args.workers is not None else None
    task_timeout: float = float(args.task_timeout)
    throughput_seconds: float | None = (
        float(args.throughput_seconds) if args.throughput_seconds is not None else None
    )
    out_root: str = str(args.out_root)

    benchmark_names = [b.strip() for b in benchmarks_str.split(",")]

    approaches: list[str] | None = None
    if args.approaches is not None:
        approaches = [a.strip() for a in str(args.approaches).split(",") if a.strip()]

    targets: list[tuple[int, int]] | None = None
    if args.targets is not None:
        targets = []
        for tok in str(args.targets).split(","):
            tok = tok.strip()
            if not tok:
                continue
            k_str, g_str = tok.split("-")
            targets.append((int(k_str), int(g_str)))

    extra: dict[str, object] = {}
    if throughput_seconds is not None:
        extra["throughput_seconds"] = throughput_seconds
    config = RunConfig(
        benchmarks=benchmark_names,
        quick=quick,
        seeds=seeds,
        seed_base=seed_base,
        cage_time_budget_s=cage_budget,
        cage_max_steps=cage_max_steps,
        workers=workers,
        task_timeout_s=task_timeout,
        approaches=approaches,
        targets=targets,
        extra=extra,
    )

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(out_root) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    t0 = time.perf_counter()
    run_results = run(config, out_dir)
    wall = time.perf_counter() - t0

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
