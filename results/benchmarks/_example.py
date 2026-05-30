"""Example benchmark — reference implementation and copy-me template.

HOW TO WRITE A NEW BENCHMARK MODULE
=====================================
1. Define ``make_tasks(config: RunConfig) -> list[Task]``.
   - Build one Task per (instance × approach × variant) combination.
   - Task.payload must contain ONLY picklable primitives (str, int, float, bool,
     list, dict).  Never put live nx.Graph or torch.Tensor objects in payload;
     reconstruct them inside execute() using seeds / names from the battery.

2. Define ``execute(task: Task) -> list[TrialResult]``.
   - Receives a single Task; returns one or more TrialResult objects.
   - This function runs in a worker process (spawned on macOS), so it must be
     a module-level callable (picklable).
   - Import heavy dependencies (torch, nx, model loaders) at the top of the
     module — workers re-import the module, so top-level imports are fine.

3. At the bottom of the module, register with:
       register(Benchmark("my_benchmark_name", make_tasks, execute))

4. Add an import line in results/benchmarks/__init__.py:
       from . import my_module  # noqa: F401

That's it.  The runner will discover the benchmark via REGISTRY and schedule
all tasks across worker processes automatically.
"""

from __future__ import annotations

import time
from typing import cast

from ..metrics import TrialResult
from ..registry import Benchmark, RunConfig, Task, register

# Number of dummy tasks emitted in full mode
_FULL_TASK_COUNT = 8
_QUICK_TASK_COUNT = 4

# Simulated per-task work duration (seconds)
_SLEEP_S = 0.2


def make_tasks(config: RunConfig) -> list[Task]:
    """Return a small set of dummy tasks to prove the parallel contract."""
    n = _QUICK_TASK_COUNT if config.quick else _FULL_TASK_COUNT
    return [
        Task(
            benchmark="example",
            label=f"example/task_{i}",
            payload={"task_index": i, "sleep_s": _SLEEP_S},
        )
        for i in range(n)
    ]


def execute(task: Task) -> list[TrialResult]:
    """Sleep briefly and return a fake TrialResult — proves parallelism works."""
    # payload values are typed `object`; cast to the primitive you stored.
    idx: int = cast(int, task.payload["task_index"])
    sleep_s: float = cast(float, task.payload["sleep_s"])

    t0 = time.perf_counter()
    time.sleep(sleep_s)
    elapsed = time.perf_counter() - t0

    return [
        TrialResult(
            benchmark="example",
            approach="dummy",
            variant="sleep",
            label=task.label,
            instance=f"task_{idx}",
            target={},
            success=True,
            elapsed_s=elapsed,
            steps=None,
            n_nodes=None,
            n_edges=None,
            metrics={"fake_score": float(idx) * 0.1},
        )
    ]


register(Benchmark("example", make_tasks, execute))
