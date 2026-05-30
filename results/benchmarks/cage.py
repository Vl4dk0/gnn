"""Cage generator benchmark — all approaches on shared (k,g) targets.

Covers every in-process cage generator and provides a first-class
no-GNN vs GNN comparison for the voltage approach.

Generator specs:
  randomwalk  —  RandomWalkGenerator (seeded via random)
  astar       —  AStarGenerator (deterministic)
  bruteforce  —  BruteforceGenerator (deterministic)
  rl          —  RLGenerator with best actor_critic model (torch)
  voltage/no_gnn  —  VoltageSearchGenerator, model_id=None (tabu only)
  voltage/gnn     —  VoltageSearchGenerator, model_id="girth_predictor"
  voltage_rl  —  VoltageRLGenerator with best voltage_actor_critic model
"""

from __future__ import annotations

import random
import time
from typing import cast

import networkx as nx
import numpy as np
import torch

from ai.cage.forge import forge_graph
from ai.cage.registry.astar import AStarGenerator
from ai.cage.registry.bruteforce import BruteforceGenerator
from ai.cage.registry.direct_rl import RLGenerator
from ai.cage.registry.random_walk import RandomWalkGenerator
from ai.cage.registry.voltage import VoltageSearchGenerator
from ai.cage.registry.voltage_rl import VoltageRLGenerator
from ai.registry import list_trained_models
from backend.utils.graph_utils import compute_girth, is_k_regular, moore_bound

from ..battery import cage_targets
from ..metrics import TrialResult, model_meta
from ..registry import Benchmark, RunConfig, Task, register

# ---------------------------------------------------------------------------
# Model-id helpers — resolved once at import time so workers share the same ids
# ---------------------------------------------------------------------------


def _best_actor_critic() -> str | None:
    """Return the model_id of the highest-reward direct RL model, or None."""
    models = [
        m for m in list_trained_models("cage") if m.get("model_type") == "actor_critic"
    ]
    if not models:
        return None
    best = max(
        models, key=lambda m: m.get("metrics", {}).get("avg_reward", float("-inf"))
    )
    return best["model_id"]


def _best_voltage_actor_critic() -> str | None:
    """Return the model_id of the highest-reward voltage RL model, or None."""
    models = [
        m
        for m in list_trained_models("cage")
        if m.get("model_type") == "voltage_actor_critic"
    ]
    if not models:
        return None
    best = max(
        models, key=lambda m: m.get("metrics", {}).get("best_avg_reward", float("-inf"))
    )
    return best["model_id"]


def _girth_predictor_exists() -> bool:
    """Return True if girth_predictor exists under voltage_girth task."""
    return any(
        m["model_id"] == "girth_predictor" for m in list_trained_models("voltage_girth")
    )


_ACTOR_CRITIC_ID: str | None = _best_actor_critic()
_VOLTAGE_AC_ID: str | None = _best_voltage_actor_critic()
_GIRTH_PREDICTOR_OK: bool = _girth_predictor_exists()

# ---------------------------------------------------------------------------
# Spec table: (approach, variant, model_id_or_none)
# ---------------------------------------------------------------------------

_SPECS: list[tuple[str, str, str | None]] = [
    ("randomwalk", "", None),
    ("astar", "", None),
    ("bruteforce", "", None),
]

if _ACTOR_CRITIC_ID is not None:
    _SPECS.append(("rl", "", _ACTOR_CRITIC_ID))

_SPECS.append(("voltage", "no_gnn", None))

if _GIRTH_PREDICTOR_OK:
    _SPECS.append(("voltage", "gnn", "girth_predictor"))

if _VOLTAGE_AC_ID is not None:
    _SPECS.append(("voltage_rl", "", _VOLTAGE_AC_ID))

# forge: the full voltage -> refine -> excision cascade. Unlike the other
# approaches it shrinks the result toward the cage via the excision loop, so
# its graph size (and Moore ratio) is the interesting signal. Uses the girth
# predictor when present (graceful classical fallback otherwise).
_SPECS.append(("forge", "", "girth_predictor" if _GIRTH_PREDICTOR_OK else None))

# ---------------------------------------------------------------------------
# make_tasks
# ---------------------------------------------------------------------------


def make_tasks(config: RunConfig) -> list[Task]:
    """Emit one Task per (k,g) × generator-spec × seed."""
    tasks: list[Task] = []
    for k, g in cage_targets(config.quick):
        for approach, variant, model_id in _SPECS:
            for seed in range(config.seeds):
                if variant:
                    label = f"({k},{g}) {approach}[{variant}] s{seed}"
                else:
                    label = f"({k},{g}) {approach} s{seed}"
                tasks.append(
                    Task(
                        benchmark="cage",
                        label=label,
                        payload={
                            "k": k,
                            "g": g,
                            "approach": approach,
                            "variant": variant,
                            "model_id": model_id if model_id is not None else "",
                            "seed": seed,
                            "time_budget_s": config.cage_time_budget_s,
                            "max_steps": config.cage_max_steps,
                        },
                    )
                )
    return tasks


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------


def execute(task: Task) -> list[TrialResult]:
    """Run one generator trial and return a single TrialResult."""
    k: int = cast(int, task.payload["k"])
    g: int = cast(int, task.payload["g"])
    approach: str = cast(str, task.payload["approach"])
    variant: str = cast(str, task.payload["variant"])
    raw_model_id: str = cast(str, task.payload["model_id"])
    seed: int = cast(int, task.payload["seed"])
    time_budget_s: float = cast(float, task.payload["time_budget_s"])
    max_steps: int = cast(int, task.payload["max_steps"])

    model_id: str | None = raw_model_id if raw_model_id else None

    # Seed all RNG sources before constructing any generator
    random.seed(seed)
    np.random.seed(seed)
    _ = torch.manual_seed(seed)

    # forge is a one-shot cascade (voltage -> refine -> excision), not a step
    # generator, so it runs outside the step loop and self-limits via its own
    # time_budget. Every other approach follows the step protocol.
    graph: nx.Graph[int]
    success: bool
    steps: int | None
    if approach == "forge":
        t0 = time.perf_counter()
        forged = forge_graph(
            k, g, predictor=model_id, time_budget=time_budget_s, verbose=False
        )
        elapsed = time.perf_counter() - t0
        graph = forged if forged is not None else nx.Graph()
        success = forged is not None
        steps = None
    else:
        gen: (
            RandomWalkGenerator
            | AStarGenerator
            | BruteforceGenerator
            | RLGenerator
            | VoltageSearchGenerator
            | VoltageRLGenerator
        )
        if approach == "randomwalk":
            gen = RandomWalkGenerator(k, g)
        elif approach == "astar":
            gen = AStarGenerator(k, g)
        elif approach == "bruteforce":
            gen = BruteforceGenerator(k, g)
        elif approach == "rl":
            gen = RLGenerator(k, g, model_id=model_id)
        elif approach == "voltage":
            gen = VoltageSearchGenerator(k, g, model_id=model_id)
        elif approach == "voltage_rl":
            gen = VoltageRLGenerator(k, g, model_id=model_id)
        else:
            raise ValueError(f"Unknown approach: {approach!r}")

        # Run the step loop with both budget guards
        t0 = time.perf_counter()
        while not gen.is_complete:
            gen.step()
            if gen.step_count >= max_steps:
                break
            if gen.elapsed_time() > time_budget_s:
                break
        elapsed = time.perf_counter() - t0
        graph = gen.graph
        success = gen.success
        steps = gen.step_count

    # Collect graph metrics
    n_nodes: int = graph.number_of_nodes()
    n_edges: int = graph.number_of_edges()

    raw_girth = compute_girth(graph) if n_nodes > 0 else float("inf")
    girth_metric: float = float(raw_girth) if raw_girth != float("inf") else -1.0
    kreg: bool = is_k_regular(graph, k) if n_nodes > 0 else False
    mb: int = moore_bound(k, g)

    # Model metadata
    m_params: int | None = None
    m_size: float | None = None
    m_hparams: dict[str, object] | None = None

    if approach == "rl" and model_id is not None:
        m_params, m_size, m_hparams = model_meta("cage", model_id)
    elif approach == "voltage_rl" and model_id is not None:
        m_params, m_size, m_hparams = model_meta("cage", model_id)
    elif approach == "voltage" and variant == "gnn" and model_id is not None:
        m_params, m_size, m_hparams = model_meta("voltage_girth", model_id)
    elif approach == "forge" and model_id is not None:
        m_params, m_size, m_hparams = model_meta("voltage_girth", model_id)

    if variant:
        instance_label = f"({k},{g}) {approach}[{variant}] s{seed}"
    else:
        instance_label = f"({k},{g}) {approach} s{seed}"

    return [
        TrialResult(
            benchmark="cage",
            approach=approach,
            variant=variant,
            label=instance_label,
            instance=f"k{k}_g{g}_s{seed}",
            target={"k": k, "g": g},
            success=success,
            elapsed_s=elapsed,
            steps=steps,
            n_nodes=n_nodes,
            n_edges=n_edges,
            metrics={
                "girth": girth_metric,
                "moore_bound": float(mb),
                "moore_ratio": float(n_nodes) / mb if mb > 0 else 0.0,
                "is_k_regular": float(kreg),
            },
            model_id=model_id,
            model_params=m_params,
            model_size_mb=m_size,
            model_hparams=m_hparams,
        )
    ]


register(Benchmark("cage", make_tasks, execute))
