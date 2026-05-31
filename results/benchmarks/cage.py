"""Cage generator benchmark — all approaches on shared (k,g) targets.

Covers every in-process cage generator. The voltage approach is split into
three variants so the algebraic baseline and both trained predictors are all
tested head to head.

Generator specs:
  randomwalk              —  RandomWalkGenerator (seeded via random)
  astar                   —  AStarGenerator (deterministic)
  bruteforce              —  BruteforceGenerator (deterministic)
  rl                      —  RLGenerator with best actor_critic model (torch)
  voltage/Algebraic       —  VoltageSearchGenerator, model_id=None (tabu only)
  voltage/GirthPredictor  —  VoltageSearchGenerator, model_id="girth_predictor"
  voltage/TabuPredictor   —  VoltageSearchGenerator, model_id="tabu_predictor"
  voltage_rl              —  VoltageRLGenerator with best voltage_actor_critic
  forge                   —  ForgeGenerator (voltage -> refine -> excision)
"""

from __future__ import annotations

import random
import time
from typing import cast

import numpy as np
import torch

from ai.cage.registry.astar import AStarGenerator
from ai.cage.registry.bruteforce import BruteforceGenerator
from ai.cage.registry.direct_rl import RLGenerator
from ai.cage.registry.forge import (
    DEFAULT_FORGE_PRODUCER,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_REFINE_DEFECT_TAU,
    DEFAULT_REFINE_GATE,
    FORGE_PRODUCERS,
    REFINE_MARGIN,
    ForgeGenerator,
)
from ai.cage.registry.random_walk import RandomWalkGenerator
from ai.cage.registry.voltage import (
    BEAM_PROBE_INTERVAL,
    DEFAULT_BEAM_WIDTH,
    DEFAULT_TABU_TENURE,
    VoltageSearchGenerator,
)
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


def _voltage_predictor_exists(model_id: str) -> bool:
    """Return True if a voltage predictor with *model_id* is on disk."""
    return any(m["model_id"] == model_id for m in list_trained_models("voltage_girth"))


_ACTOR_CRITIC_ID: str | None = _best_actor_critic()
_VOLTAGE_AC_ID: str | None = _best_voltage_actor_critic()
_GIRTH_PREDICTOR_OK: bool = _voltage_predictor_exists("girth_predictor")
_TABU_PREDICTOR_OK: bool = _voltage_predictor_exists("tabu_predictor")

# ---------------------------------------------------------------------------
# Spec table: (approach, variant, model_id_or_none)
#
# The voltage approach has three variants, all running the same tabu lift
# search; they differ only in what (if anything) guides the beam-search probe:
#   Algebraic       — no GNN, pure tabu search over voltage assignments
#   GirthPredictor  — GNN trained to predict lift girth (kind="girth")
#   TabuPredictor   — GNN trained to predict tabu cost (kind="tabu_cost")
# ---------------------------------------------------------------------------

_SPECS: list[tuple[str, str, str | None]] = [
    ("randomwalk", "", None),
    ("astar", "", None),
    ("bruteforce", "", None),
]

if _ACTOR_CRITIC_ID is not None:
    _SPECS.append(("rl", "", _ACTOR_CRITIC_ID))

_SPECS.append(("voltage", "Algebraic", None))

if _GIRTH_PREDICTOR_OK:
    _SPECS.append(("voltage", "GirthPredictor", "girth_predictor"))

if _TABU_PREDICTOR_OK:
    _SPECS.append(("voltage", "TabuPredictor", "tabu_predictor"))

if _VOLTAGE_AC_ID is not None:
    # voltage_rl decoding mode: the policy can pick actions stochastically
    # (sample from the action distribution, the historical default) or
    # deterministically (argmax).  Both run the same weights; the payload flag
    # ``vrl_deterministic`` selects the decode.
    _SPECS.append(("voltage_rl", "Stochastic", _VOLTAGE_AC_ID))
    _SPECS.append(("voltage_rl", "Deterministic", _VOLTAGE_AC_ID))

# forge: the full voltage -> refine -> excision cascade. Unlike the other
# approaches it shrinks the result toward the cage via the excision loop, so
# its graph size (and Moore ratio) is the interesting signal.
#
# Producer comparison: forge's voltage (production) stage can be driven by any
# of four producers. One full-pipeline variant per producer pits them against
# each other (the variant IS the producer name; refine + excision both on):
#   voltage_rl         — voltage RL policy (the default)
#   voltage_girth      — voltage tabu search w/ girth predictor
#   voltage_tabu       — voltage tabu search w/ tabu-cost predictor
#   voltage_algebraic  — plain algebraic voltage tabu search (no GNN)
for _producer in FORGE_PRODUCERS:
    _SPECS.append(("forge", _producer, None))

# forge ablation variants on the default producer: isolate each stage.
#   no_refine   — voltage + excision (near-misses dropped; only direct hits shrunk)
#   no_excision — voltage + refine, no shrink (first valid graph returned unshrunk)
_SPECS.append(("forge", "no_refine", None))
_SPECS.append(("forge", "no_excision", None))

# A3 — forge ablations across the other (non-default) producers.  The default
# producer's ablations live above as the bare "no_refine"/"no_excision" rows; to
# avoid duplicating them we ablate only the three NON-default producers here.
# Composite variant name "<producer>+<ablation>" (e.g. "voltage_tabu+no_refine")
# is parsed back into (producer, use_refine, use_excision) in execute().
FORGE_ABLATION_PRODUCERS = tuple(
    p for p in FORGE_PRODUCERS if p != DEFAULT_FORGE_PRODUCER
)
for _producer in FORGE_ABLATION_PRODUCERS:
    _SPECS.append(("forge", f"{_producer}+no_refine", None))
    _SPECS.append(("forge", f"{_producer}+no_excision", None))

# C — forge hyperparameter sweep on the default producer (full pipeline).  Swept
# as two small one-dimensional families (not a cross-product): max_candidates
# controls how many lifts the producer pushes downstream; refine_margin controls
# how far below girth g a near-miss may sit and still be queued for refine.  The
# swept value is threaded into ForgeGenerator via the payload.
FORGE_MAX_CANDIDATES_SWEEP = (10, 20, 40)
FORGE_REFINE_MARGIN_SWEEP = (1, 2, 3)
for _mc in FORGE_MAX_CANDIDATES_SWEEP:
    _SPECS.append(("forge", f"max_candidates={_mc}", None))
for _rm in FORGE_REFINE_MARGIN_SWEEP:
    _SPECS.append(("forge", f"refine_margin={_rm}", None))

# C — forge refine-routing gate sweep on the default producer (full pipeline).
# The "defect_density" gate routes a near-miss to refine by the fraction of
# edges lying on a short cycle (size-relative) rather than by girth window, so a
# lift whose girth is low only because of one short cycle is still refined.
# Sweep the defect threshold tau; the value rides in via the payload.
FORGE_DEFECT_TAU_SWEEP = (0.1, 0.2, 0.3)
for _tau in FORGE_DEFECT_TAU_SWEEP:
    _SPECS.append(("forge", f"defect_gate_tau={_tau}", None))

# C — voltage hyperparameter sweep on the algebraic (no-GNN) tabu search, so no
# model dependency is needed.  Three small one-dimensional families: beam width
# (only meaningful when a girth predictor is loaded, but kept here so the sweep
# is uniform), beam-probe interval, and tabu tenure.  Each swept value rides in
# via the payload and is threaded into VoltageSearchGenerator in execute().
VOLTAGE_BEAM_WIDTH_SWEEP = (10, 20, 40)
VOLTAGE_BEAM_PROBE_INTERVAL_SWEEP = (5, 10, 20)
VOLTAGE_TABU_TENURE_SWEEP = (5, 10, 20)
for _bw in VOLTAGE_BEAM_WIDTH_SWEEP:
    _SPECS.append(("voltage", f"beam_width={_bw}", None))
for _bpi in VOLTAGE_BEAM_PROBE_INTERVAL_SWEEP:
    _SPECS.append(("voltage", f"beam_probe_interval={_bpi}", None))
for _tt in VOLTAGE_TABU_TENURE_SWEEP:
    _SPECS.append(("voltage", f"tabu_tenure={_tt}", None))

# ---------------------------------------------------------------------------
# Variant -> tunable payload
#
# Every tunable a variant introduces is threaded through task.payload so the
# same generator code serves every variant.  Producer-named and ablation
# variants for forge, decode mode for voltage_rl, and the hyperparameter sweeps
# all decode here from the (approach, variant) pair into explicit payload keys.
# ---------------------------------------------------------------------------


def _variant_params(approach: str, variant: str) -> dict[str, object]:
    """Decode a spec's variant string into explicit tunable payload entries.

    Keys (all optional; absent => the generator's own default applies):
      forge_producer       str   — which voltage producer to drive (forge)
      forge_use_refine     bool  — enable the refine stage (forge)
      forge_use_excision   bool  — enable the excision stage (forge)
      forge_max_candidates int   — producer push cap (forge sweep)
      forge_refine_margin  int   — near-miss girth window (forge sweep)
      forge_refine_gate    str   — refine-routing gate mode (forge sweep)
      forge_refine_defect_tau float — defect-density gate threshold (forge sweep)
      vrl_deterministic    bool  — argmax vs sampled decode (voltage_rl)
      beam_width           int   — beam-probe width (voltage sweep)
      beam_probe_interval  int   — steps between beam probes (voltage sweep)
      tabu_tenure          int   — tabu move tenure (voltage sweep)
    """
    params: dict[str, object] = {}

    if approach == "forge":
        # Producer + stage flags.  A bare ablation variant ("no_refine" /
        # "no_excision") applies to the default producer; a composite
        # "<producer>+<ablation>" applies to that producer.  A producer-named
        # variant runs the full pipeline.  Anything else (the sweep variants)
        # keeps the default producer with both stages on.
        producer = DEFAULT_FORGE_PRODUCER
        ablation = ""
        if variant in FORGE_PRODUCERS:
            producer = variant
        elif variant in ("no_refine", "no_excision"):
            ablation = variant
        elif "+" in variant:
            prod_part, _, abl_part = variant.partition("+")
            if prod_part in FORGE_PRODUCERS:
                producer = prod_part
                ablation = abl_part
        params["forge_producer"] = producer
        params["forge_use_refine"] = ablation != "no_refine"
        params["forge_use_excision"] = ablation != "no_excision"

        if variant.startswith("max_candidates="):
            params["forge_max_candidates"] = int(variant.split("=", 1)[1])
        elif variant.startswith("refine_margin="):
            params["forge_refine_margin"] = int(variant.split("=", 1)[1])
        elif variant.startswith("defect_gate_tau="):
            params["forge_refine_gate"] = "defect_density"
            params["forge_refine_defect_tau"] = float(variant.split("=", 1)[1])

    elif approach == "voltage_rl":
        params["vrl_deterministic"] = variant == "Deterministic"

    elif approach == "voltage":
        if variant.startswith("beam_width="):
            params["beam_width"] = int(variant.split("=", 1)[1])
        elif variant.startswith("beam_probe_interval="):
            params["beam_probe_interval"] = int(variant.split("=", 1)[1])
        elif variant.startswith("tabu_tenure="):
            params["tabu_tenure"] = int(variant.split("=", 1)[1])

    return params


# ---------------------------------------------------------------------------
# make_tasks
# ---------------------------------------------------------------------------


def make_tasks(config: RunConfig) -> list[Task]:
    """Emit one Task per (k,g) × generator-spec × seed.

    Honours config.approaches (restrict to named approaches, e.g. ["forge"])
    and config.targets (restrict the (k,g) grid), so a single approach can be
    benchmarked on a couple of targets without running everything.
    """
    specs = _SPECS
    if config.approaches is not None:
        specs = [s for s in _SPECS if s[0] in config.approaches]
    targets = (
        config.targets if config.targets is not None else cage_targets(config.quick)
    )

    tasks: list[Task] = []
    for k, g in targets:
        for approach, variant, model_id in specs:
            for seed in range(config.seed_base, config.seed_base + config.seeds):
                if variant:
                    label = f"({k},{g}) {approach}[{variant}] s{seed}"
                else:
                    label = f"({k},{g}) {approach} s{seed}"
                payload: dict[str, object] = {
                    "k": k,
                    "g": g,
                    "approach": approach,
                    "variant": variant,
                    "model_id": model_id if model_id is not None else "",
                    "seed": seed,
                    "time_budget_s": config.cage_time_budget_s,
                    "max_steps": config.cage_max_steps,
                }
                payload.update(_variant_params(approach, variant))
                tasks.append(Task(benchmark="cage", label=label, payload=payload))
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

    # Build generator. forge follows the same step protocol as the others
    # (its embedded voltage->refine->excision cascade advances one move per
    # step), so it runs through the shared step loop and honours the same
    # max_steps / time_budget guards.
    gen: (
        RandomWalkGenerator
        | AStarGenerator
        | BruteforceGenerator
        | RLGenerator
        | VoltageSearchGenerator
        | VoltageRLGenerator
        | ForgeGenerator
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
        # Hyperparameter-sweep variants thread beam_width / beam_probe_interval /
        # tabu_tenure through the payload; absent keys fall back to the
        # generator's own defaults so the plain Algebraic/predictor rows are
        # unaffected.
        beam_width = cast(int, task.payload.get("beam_width", DEFAULT_BEAM_WIDTH))
        beam_probe_interval = cast(
            int, task.payload.get("beam_probe_interval", BEAM_PROBE_INTERVAL)
        )
        tabu_tenure = cast(int, task.payload.get("tabu_tenure", DEFAULT_TABU_TENURE))
        gen = VoltageSearchGenerator(
            k,
            g,
            model_id=model_id,
            beam_width=beam_width,
            beam_probe_interval=beam_probe_interval,
            tabu_tenure=tabu_tenure,
        )
    elif approach == "voltage_rl":
        deterministic = cast(bool, task.payload.get("vrl_deterministic", False))
        gen = VoltageRLGenerator(k, g, model_id=model_id, deterministic=deterministic)
    elif approach == "forge":
        # Producer + stage flags + sweep values all ride in via the payload
        # (decoded from the variant by _variant_params); absent keys fall back to
        # ForgeGenerator's own defaults.
        producer = cast(str, task.payload.get("forge_producer", DEFAULT_FORGE_PRODUCER))
        use_refine = cast(bool, task.payload.get("forge_use_refine", True))
        use_excision = cast(bool, task.payload.get("forge_use_excision", True))
        max_candidates = cast(
            int, task.payload.get("forge_max_candidates", DEFAULT_MAX_CANDIDATES)
        )
        refine_margin = cast(
            int, task.payload.get("forge_refine_margin", REFINE_MARGIN)
        )
        refine_gate = cast(
            str, task.payload.get("forge_refine_gate", DEFAULT_REFINE_GATE)
        )
        refine_defect_tau = cast(
            float,
            task.payload.get("forge_refine_defect_tau", DEFAULT_REFINE_DEFECT_TAU),
        )
        gen = ForgeGenerator(
            k,
            g,
            producer=producer,
            use_refine=use_refine,
            use_excision=use_excision,
            max_candidates=max_candidates,
            refine_margin=refine_margin,
            refine_gate=refine_gate,
            refine_defect_tau=refine_defect_tau,
        )
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

    # Collect graph metrics
    graph = gen.graph
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
    elif approach == "voltage" and model_id is not None:
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
            success=gen.success,
            elapsed_s=elapsed,
            steps=gen.step_count,
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
