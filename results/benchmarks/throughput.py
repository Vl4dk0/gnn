"""Search-throughput benchmark: candidate graphs evaluated per second.

Research question: why do voltage_rl (better reach) and A* (smaller graphs)
beat forge, when forge looks like a superior pipeline? Hypothesis: voltage_rl
and A* evaluate MORE candidate graphs per unit time, and forge's refine and
excision stages drag its candidate throughput down.

This benchmark gives a focused set of approaches a hard target, runs each for a
fixed wall-clock budget, and reports:
  - candidates_per_sec  — candidate graphs whose fitness is computed per second
  - steps_per_sec       — scheduler iterations (gen.step() calls) per second
  - candidates_per_step — how many graphs each step evaluates on average

A *step* is one gen.step() call (one scheduler iteration).  A *candidate* is
one graph whose fitness (girth / regularity) is actually computed; a voltage
step that internally tries many voltage assignments counts fairly against
approaches that evaluate one graph per step.

Forge-specific per-stage breakdown is also collected when available:
  producer_candidates/steps — voltage backbone stage
  refine_candidates/steps   — tabu-swap refinement stage
  excision_candidates/steps — excision+repair stage

Note: excision only fires on targets forge can actually solve (it requires the
producer to yield a valid initial graph to shrink). At a hard target such as
(3,12) the excision_candidates count is 0 — the producer never succeeds, so
there is nothing to shrink. This is expected and informative.
"""

from __future__ import annotations

import random
import time
from typing import cast

import numpy as np
import torch

from ..metrics import TrialResult
from ..registry import Benchmark, RunConfig, Task, register
from .cage import SPECS, build_generator, variant_params

# Default hard target: the (3,12) Tutte 12-cage (126 vertices). No in-process
# approach finds it in 60s, so the whole budget is spent evaluating candidates.
DEFAULT_TARGET: tuple[int, int] = (3, 12)


def _focus_specs() -> list[tuple[str, str, str | None, dict[str, object]]]:
    """Focused approaches for the throughput study.

    Reuse the trained-model rows from cage's SPECS for astar + voltage_rl so
    model ids and graceful fallback are inherited.  Build the forge
    stage-ablation set by hand on the default producer.
    """
    default_producer = variant_params("forge", "")["forge_producer"]
    specs: list[tuple[str, str, str | None, dict[str, object]]] = []
    for approach, variant, model_id in SPECS:
        if approach in ("astar", "voltage_rl"):
            specs.append((approach, variant, model_id, {}))
    forge_ablations: list[tuple[str, dict[str, object]]] = [
        ("all-stages", {"forge_use_refine": True, "forge_use_excision": True}),
        ("no-refine", {"forge_use_refine": False, "forge_use_excision": True}),
        ("no-excision", {"forge_use_refine": True, "forge_use_excision": False}),
        ("producer-only", {"forge_use_refine": False, "forge_use_excision": False}),
    ]
    for label, flags in forge_ablations:
        params: dict[str, object] = {"forge_producer": default_producer}
        params.update(flags)
        specs.append(("forge", label, None, params))
    return specs


def make_tasks(config: RunConfig) -> list[Task]:
    focus = _focus_specs()
    if config.approaches is not None:
        focus = [s for s in focus if s[0] in config.approaches]
    targets = config.targets if config.targets is not None else [DEFAULT_TARGET]

    raw_budget = config.extra.get("throughput_seconds", config.cage_time_budget_s)
    budget = float(cast(float, raw_budget))

    tasks: list[Task] = []
    for k, g in targets:
        for approach, variant, model_id, params in focus:
            for seed in range(config.seed_base, config.seed_base + config.seeds):
                label = (
                    f"({k},{g}) {approach}[{variant}] s{seed}"
                    if variant
                    else f"({k},{g}) {approach} s{seed}"
                )
                payload: dict[str, object] = {
                    "k": k,
                    "g": g,
                    "approach": approach,
                    "variant": variant,
                    "model_id": model_id if model_id is not None else "",
                    "seed": seed,
                    "time_budget_s": budget,
                    "max_steps": config.cage_max_steps,
                }
                if approach in ("astar", "voltage_rl"):
                    payload.update(variant_params(approach, variant))
                else:
                    # forge: use per-spec params dict directly; our hyphen
                    # variant labels are not recognized by variant_params
                    payload.update(params)
                tasks.append(Task(benchmark="throughput", label=label, payload=payload))
    return tasks


def execute(task: Task) -> list[TrialResult]:
    k = cast(int, task.payload["k"])
    g = cast(int, task.payload["g"])
    approach = cast(str, task.payload["approach"])
    variant = cast(str, task.payload["variant"])
    raw_model_id = cast(str, task.payload["model_id"])
    seed = cast(int, task.payload["seed"])
    time_budget_s = cast(float, task.payload["time_budget_s"])
    max_steps = cast(int, task.payload["max_steps"])
    model_id = raw_model_id if raw_model_id else None

    random.seed(seed)
    np.random.seed(seed)
    _ = torch.manual_seed(seed)

    gen = build_generator(approach, k, g, model_id, task.payload)

    t0 = time.perf_counter()
    while not gen.is_complete:
        gen.step()
        if gen.step_count >= max_steps:
            break
        if gen.elapsed_time() > time_budget_s:
            break
    elapsed = time.perf_counter() - t0

    candidates: int = getattr(gen, "candidates_evaluated", gen.step_count)
    steps = gen.step_count
    cps = candidates / elapsed if elapsed > 0 else 0.0
    sps = steps / elapsed if elapsed > 0 else 0.0
    cpstep = candidates / steps if steps > 0 else 0.0

    metrics: dict[str, float] = {
        "candidates_per_sec": float(cps),
        "steps_per_sec": float(sps),
        "candidates_per_step": float(cpstep),
        "candidates_total": float(candidates),
    }

    if approach == "forge":
        for stage in ("producer", "refine", "excision"):
            sc = getattr(gen, f"{stage}_candidates", 0)
            ss = getattr(gen, f"{stage}_steps", 0)
            metrics[f"{stage}_candidates"] = float(sc)
            metrics[f"{stage}_steps"] = float(ss)

    graph = gen.graph
    instance_label = (
        f"({k},{g}) {approach}[{variant}] s{seed}"
        if variant
        else f"({k},{g}) {approach} s{seed}"
    )
    return [
        TrialResult(
            benchmark="throughput",
            approach=approach,
            variant=variant,
            label=instance_label,
            instance=f"k{k}_g{g}_s{seed}",
            target={"k": k, "g": g},
            success=gen.success,
            elapsed_s=elapsed,
            steps=steps,
            n_nodes=graph.number_of_nodes(),
            n_edges=graph.number_of_edges(),
            metrics=metrics,
        )
    ]


register(Benchmark("throughput", make_tasks, execute))
