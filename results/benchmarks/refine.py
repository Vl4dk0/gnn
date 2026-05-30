"""Refine benchmark — tabu refinement WITHOUT vs WITH the MoveOracle GNN.

Benchmarks the tabu search refiner on shared k-regular input graphs drawn
from the ``refine_instances`` battery.  Two variants are compared:

* ``no_gnn``: exact Δcost evaluation for every candidate swap.
* ``gnn``: MoveOracle ranks candidates; only the top-ranked one is evaluated.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch

from ..battery import refine_instances
from ..metrics import TrialResult, model_meta_from_dir
from ..registry import Benchmark, RunConfig, Task, register

_ORACLE_DIR = Path("ai/trained/move_oracle/move_oracle")


def make_tasks(config: RunConfig) -> list[Task]:
    """Emit one Task per (instance × variant × seed) combination."""
    tasks: list[Task] = []
    for idx, (name, k, g, _) in enumerate(refine_instances(config.quick)):
        for variant in ("no_gnn", "gnn"):
            for seed in range(config.seeds):
                tasks.append(
                    Task(
                        benchmark="refine",
                        label=f"{name} refine[{variant}] s{seed}",
                        payload={
                            "idx": idx,
                            "name": name,
                            "k": k,
                            "g": g,
                            "variant": variant,
                            "seed": seed,
                            "quick": config.quick,
                        },
                    )
                )
    return tasks


def execute(task: Task) -> list[TrialResult]:
    """Run one refine trial and return a single TrialResult."""
    # Cast payload primitives
    idx: int = cast(int, task.payload["idx"])
    name: str = cast(str, task.payload["name"])
    k: int = cast(int, task.payload["k"])
    g: int = cast(int, task.payload["g"])
    variant: str = cast(str, task.payload["variant"])
    seed: int = cast(int, task.payload["seed"])
    quick: bool = cast(bool, task.payload["quick"])

    # Seed all RNGs
    random.seed(seed)
    np.random.seed(seed)
    _ = torch.manual_seed(seed)

    # Reconstruct the deterministic input graph from the battery
    G = refine_instances(quick)[idx][3]

    # Lazy imports of heavy dependencies (fine here — module is re-imported per worker)
    from ai.cage.refine.tabu import TabuRefiner

    model_id: str | None = None
    model_params: int | None = None
    model_size_mb: float | None = None
    model_hparams: dict[str, object] | None = None

    if variant == "gnn":
        from ai.cage.refine.move_oracle import build_score_fn, load_move_oracle

        model, cycle_lengths, rwpe_dim = load_move_oracle(_ORACLE_DIR)
        score_fn = build_score_fn(
            model=model,
            cycle_lengths=cycle_lengths,
            rwpe_dim=rwpe_dim,
            device="cpu",
        )
        refiner = TabuRefiner(g_target=g, score_fn=score_fn)
        params, size_mb, hparams = model_meta_from_dir(_ORACLE_DIR)
        model_id = "move_oracle"
        model_params = params
        model_size_mb = size_mb
        model_hparams = hparams
    else:
        refiner = TabuRefiner(g_target=g)

    t0 = time.perf_counter()
    best, stats = refiner.refine(G)
    elapsed = time.perf_counter() - t0

    success: bool = float(stats["final_cost"]) == 0.0

    return [
        TrialResult(
            benchmark="refine",
            approach="tabu_refine",
            variant=variant,
            label=task.label,
            instance=name,
            target={"k": k, "g": g},
            success=success,
            elapsed_s=elapsed,
            steps=int(stats["iterations"]),
            n_nodes=best.number_of_nodes(),
            n_edges=best.number_of_edges(),
            metrics={
                "cost_initial": float(stats["initial_cost"]),
                "cost_final": float(stats["final_cost"]),
                "escalations": float(stats["escalations"]),
                "improved": float(bool(stats["improved"])),
            },
            model_id=model_id,
            model_params=model_params,
            model_size_mb=model_size_mb,
            model_hparams=model_hparams,
        )
    ]


register(Benchmark("refine", make_tasks, execute))
