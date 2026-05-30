"""Node-prediction benchmarks for degree and min_cycle tasks.

Registers two benchmarks ("degree" and "min_cycle") that evaluate every
trained model for that task against the shared node_battery.
"""

from __future__ import annotations

import math
import random
import time
from typing import cast

import networkx as nx
import numpy
import torch

from ai.degree.functions.graph_service import predict_all_nodes as predict_degree
from ai.min_cycle.functions.graph_service import (
    predict_all_nodes as predict_min_cycle,
)
from ai.registry import list_trained_models

from ..battery import node_battery
from ..metrics import TrialResult, model_meta
from ..registry import Benchmark, RunConfig, Task, register

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PREDICT_FNS = {
    "degree": predict_degree,
    "min_cycle": predict_min_cycle,
}


def _make_tasks(task_name: str, config: RunConfig) -> list[Task]:
    """Emit one Task per trained model for *task_name*."""
    tasks: list[Task] = []
    for m in list_trained_models(task_name):
        mid: str = m["model_id"]
        tasks.append(
            Task(
                benchmark=task_name,
                label=f"{task_name}:{mid}",
                payload={
                    "task": task_name,
                    "model_id": mid,
                    "quick": config.quick,
                },
            )
        )
    return tasks


def _execute(task: Task) -> list[TrialResult]:
    """Run one model over the full node battery and return a single TrialResult."""
    # Determinism
    random.seed(0)
    numpy.random.seed(0)
    _ = torch.manual_seed(0)

    task_name: str = cast(str, task.payload["task"])
    model_id: str = cast(str, task.payload["model_id"])
    quick: bool = cast(bool, task.payload["quick"])

    # Look up model_type from the registry
    model_type: str = "unknown"
    for m in list_trained_models(task_name):
        if m["model_id"] == model_id:
            model_type = m.get("model_type", "unknown") or "unknown"
            break

    predict_fn = _PREDICT_FNS[task_name]
    graphs: list[tuple[str, nx.Graph[int]]] = node_battery(quick)

    all_true: list[int] = []
    all_pred: list[float] = []
    first_error: str | None = None
    n_failed = 0

    t0 = time.perf_counter()
    for _name, G in graphs:
        try:
            results = predict_fn(G, model_id)
        except Exception as exc:
            n_failed += 1
            if first_error is None:
                first_error = repr(exc)
            continue
        for row in results:
            all_true.append(cast(int, row["true"]))
            all_pred.append(float(cast(float, row["predicted"])))
    elapsed = time.perf_counter() - t0

    n_total = len(all_true)
    n_graphs = len(graphs)

    if n_total == 0:
        accuracy = 0.0
        mae = 0.0
        rmse = 0.0
        off_by_one = 0.0
    else:
        correct = sum(1 for t, p in zip(all_true, all_pred) if round(p) == t)
        accuracy = correct / n_total
        diffs = [abs(p - t) for t, p in zip(all_true, all_pred)]
        mae = sum(diffs) / n_total
        rmse = math.sqrt(sum(d * d for d in diffs) / n_total)
        obo = sum(1 for d in diffs if round(d) == 1)
        off_by_one = obo / n_total

    params, size_mb, hparams = model_meta(task_name, model_id)

    # A model that produced no predictions (errored on every graph, e.g. a
    # weights/architecture mismatch) is a FAILURE, not a 0%-accuracy success.
    success = n_total > 0

    return [
        TrialResult(
            benchmark=task_name,
            approach=model_type,
            variant="",
            label=f"{task_name}:{model_id}",
            instance="battery",
            target={},
            success=success,
            elapsed_s=elapsed,
            steps=None,
            n_nodes=None,
            n_edges=None,
            metrics={
                "accuracy": accuracy,
                "mae": mae,
                "rmse": rmse,
                "off_by_one": off_by_one,
                "n_graphs": float(n_graphs),
                "n_nodes": float(n_total),
                "n_failed_graphs": float(n_failed),
            },
            model_id=model_id,
            model_params=params,
            model_size_mb=size_mb,
            model_hparams=hparams,
            error=first_error,
        )
    ]


# ---------------------------------------------------------------------------
# Per-benchmark make_tasks wrappers (one closure each so each benchmark only
# emits tasks for its own task name).
# ---------------------------------------------------------------------------


def _make_degree_tasks(config: RunConfig) -> list[Task]:
    return _make_tasks("degree", config)


def _make_min_cycle_tasks(config: RunConfig) -> list[Task]:
    return _make_tasks("min_cycle", config)


# ---------------------------------------------------------------------------
# Public convenience — used by the smoke test
# ---------------------------------------------------------------------------


def make_tasks_for(task_name: str, config: RunConfig) -> list[Task]:
    """Return tasks for *task_name* using the shared helper (for testing)."""
    return _make_tasks(task_name, config)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register(Benchmark("degree", _make_degree_tasks, _execute))
register(Benchmark("min_cycle", _make_min_cycle_tasks, _execute))
