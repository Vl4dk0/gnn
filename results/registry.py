"""Benchmark registry: Task/Benchmark descriptors and the global REGISTRY."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from results.metrics import TrialResult


@dataclass
class RunConfig:
    benchmarks: list[str]  # which benchmark names to run; ["all"] means all registered
    quick: bool = False
    seeds: int = 1
    cage_time_budget_s: float = 20.0
    cage_max_steps: int = 200_000
    workers: int | None = None  # default in runner = min(8, cpu-2)
    task_timeout_s: float = 120.0  # hard per-task wall-clock cap (SIGALRM in worker)
    # Optional scoping: restrict which approaches / (k,g) targets to run. None
    # means "all". Lets you benchmark one thing (e.g. just forge on (3,6)).
    approaches: list[str] | None = None
    targets: list[tuple[int, int]] | None = None
    extra: dict[str, object] = field(default_factory=dict)


@dataclass
class Task:
    benchmark: str
    label: str  # for progress display
    payload: dict[str, object]  # picklable primitives only (no live graphs/tensors)


@dataclass
class Benchmark:
    name: str
    make_tasks: Callable[[RunConfig], list[Task]]
    execute: Callable[
        [Task], list[TrialResult]
    ]  # runs one Task, returns >=1 TrialResult


REGISTRY: dict[str, Benchmark] = {}


def register(bm: Benchmark) -> None:
    """Register a Benchmark under its name."""
    REGISTRY[bm.name] = bm
