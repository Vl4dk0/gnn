"""Excision benchmark — tree-excision repair: classical vs RL policy.

For each source (k,g)-graph the benchmark:
  1. Excises a BFS tree of depth (g-1)//2.  It sweeps candidate roots and
     accepts the first root whose excision + repair yields a valid (k,g)-graph
     (the chosen root and how many were tried are recorded in the metrics).
  2. Repairs the deficient boundary set using one of:
     - greedy:       greedy_match_repair only (no backtracking fallback).
     - backtracking: backtracking_repair only.
     - classical:    greedy_match_repair → backtracking_repair fallback.
     - rl:           GNN RepairActorCritic rollout (deterministic).
  3. Verifies the result is k-regular with girth >= g.

The trained RL policy at ai/trained/excision/excision is a full 300k-episode run
(g_target=7, match_size=6).  Re-run this benchmark to measure it; earlier results
predate the trained policy being loadable.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
import torch

from ai.cage.excision.baseline import backtracking_repair, greedy_match_repair
from ai.cage.excision.excise import excise_tree
from ai.cage.excision.rl.env import RepairEnv
from ai.cage.excision.rl.model import RepairActorCritic
from backend.utils.graph_utils import compute_girth, is_k_regular

from ..battery import excision_instances
from ..metrics import TrialResult, model_meta_from_dir
from ..registry import Benchmark, RunConfig, Task, register

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_POLICY_DIR = Path("ai/trained/excision/excision")

# ---------------------------------------------------------------------------
# RL helpers
# ---------------------------------------------------------------------------


def load_repair_policy(
    model_dir: Path,
    device: str = "cpu",
) -> tuple[RepairActorCritic, list[int] | None, int]:
    """Load RepairActorCritic from a model directory.

    Returns:
        (policy, cycle_lengths, rwpe_dim)
    """
    info_path = model_dir / "info.json"
    weights_path = model_dir / "weights.pt"

    training: dict[str, object] = {}
    if info_path.exists():
        raw = json.loads(info_path.read_text())
        training = dict(raw.get("training", {}))

    input_dim: int = cast(int, training.get("input_dim", 19))
    hidden_dim: int = cast(int, training.get("hidden_dim", 64))
    num_layers: int = cast(int, training.get("num_layers", 3))
    dropout: float = float(cast(float, training.get("dropout", 0.1)))

    feature_config = cast(dict[str, object], training.get("feature_config", {}))
    cycle_lengths_raw = feature_config.get("cycle_lengths", None)
    cycle_lengths: list[int] | None = (
        [int(x) for x in cast(list[int], cycle_lengths_raw)]
        if cycle_lengths_raw is not None
        else None
    )
    rwpe_dim: int = int(cast(int, feature_config.get("rwpe_dim", 8)))

    policy = RepairActorCritic(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    _ = policy.to(device)

    if weights_path.exists():
        state_dict: dict[str, torch.Tensor] = torch.load(
            weights_path, map_location=device
        )
        _ = policy.load_state_dict(state_dict)

    _ = policy.eval()
    return policy, cycle_lengths, rwpe_dim


def rl_repair_rollout(
    reduced: nx.Graph[int],
    deficient: list[int],
    g: int,
    policy: RepairActorCritic,
    cycle_lengths: list[int] | None,
    rwpe_dim: int,
    max_steps: int = 500,
) -> tuple[bool, nx.Graph[int], int]:
    """Run a single deterministic rollout using the RL repair policy.

    Args:
        reduced:       Excised graph (deficient vertices present, tree removed).
        deficient:     List of deficient vertex IDs.
        g:             Target girth.
        policy:        Loaded RepairActorCritic.
        cycle_lengths: Cycle-length features to pass to RepairEnv.
        rwpe_dim:      RWPE dimension to pass to RepairEnv.
        max_steps:     Safety cap to prevent infinite loops.

    Returns:
        (success, final_graph, n_steps)
    """
    env = RepairEnv(
        starting_graph=reduced,
        deficient=deficient,
        g_target=g,
        cycle_lengths=cycle_lengths,
        rwpe_dim=rwpe_dim,
    )

    obs = env.reset()
    # data.node_ids is a list[int] set dynamically by RepairEnv._build_obs
    node_ids: list[int] = list(cast(list[int], getattr(obs, "node_ids")))

    # Track the graph ourselves by replaying accepted edges onto a copy
    current_graph: nx.Graph[int] = reduced.copy()
    success = False
    n_steps = 0

    with torch.no_grad():
        for _ in range(max_steps):
            legal = env.legal_actions()
            if not legal:
                break

            action, _logp, _value = policy.get_action(
                obs, legal, node_ids, deterministic=True
            )
            obs, _reward, done, info = env.step(action)
            node_ids = list(cast(list[int], getattr(obs, "node_ids")))
            _ = current_graph.add_edge(action[0], action[1])
            n_steps += 1

            if done:
                success = cast(bool, info["success"])
                break

    return success, current_graph, n_steps


# ---------------------------------------------------------------------------
# Task factory
# ---------------------------------------------------------------------------


def make_tasks(config: RunConfig) -> list[Task]:
    """Emit one Task per (instance × variant) combination."""
    tasks: list[Task] = []
    for idx, (name, k, g, _) in enumerate(excision_instances(config.quick)):
        for variant in ("greedy", "backtracking", "classical", "rl"):
            tasks.append(
                Task(
                    benchmark="excision",
                    label=f"{name} excise[{variant}]",
                    payload={
                        "idx": idx,
                        "name": name,
                        "k": k,
                        "g": g,
                        "approach": "excision",
                        "variant": variant,
                        "quick": config.quick,
                        "seed": 0,
                    },
                )
            )
    return tasks


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------


def execute(task: Task) -> list[TrialResult]:
    """Run one excision-repair trial and return a single TrialResult."""
    # -- Unpack payload (all primitives, cast explicitly) --------------------
    idx: int = cast(int, task.payload["idx"])
    name: str = cast(str, task.payload["name"])
    k: int = cast(int, task.payload["k"])
    g: int = cast(int, task.payload["g"])
    variant: str = cast(str, task.payload["variant"])
    quick: bool = cast(bool, task.payload["quick"])
    seed: int = cast(int, task.payload["seed"])

    # -- Seed RNGs -----------------------------------------------------------
    random.seed(seed)
    np.random.seed(seed)
    _ = torch.manual_seed(seed)

    # -- Rebuild source graph ------------------------------------------------
    source_graph: nx.Graph[int] = excision_instances(quick)[idx][3]

    depth = (g - 1) // 2
    orig_n = float(source_graph.number_of_nodes())

    # -- Repair-attempt closure ----------------------------------------------
    # Each variant repairs ONE reduced graph; the root sweep below calls it once
    # per candidate root and accepts the first valid (k,g)-graph.
    error: str | None = None
    steps: int | None = None

    # Model metadata (only filled for rl variant)
    model_id: str | None = None
    model_params: int | None = None
    model_size_mb: float | None = None
    model_hparams: dict[str, object] | None = None

    # The rl variant loads the policy once before the sweep.
    policy: RepairActorCritic | None = None
    cycle_lengths: list[int] | None = None
    rwpe_dim: int = 8
    if variant == "rl":
        try:
            policy, cycle_lengths, rwpe_dim = load_repair_policy(_POLICY_DIR)
            params, size_mb, hparams = model_meta_from_dir(_POLICY_DIR)
            model_id = "excision"
            model_params = params
            model_size_mb = size_mb
            model_hparams = hparams
        except Exception as exc:
            error = str(exc)

    def _repair(
        reduced: nx.Graph[int], deficient: list[int]
    ) -> tuple[nx.Graph[int] | None, int | None]:
        """Repair one reduced graph per the active variant. Returns (graph, steps)."""
        if variant == "greedy":
            return greedy_match_repair(reduced, deficient, g, k), None
        if variant == "backtracking":
            return backtracking_repair(
                reduced, deficient, g, k, max_backtracks=10000
            ), None
        if variant == "classical":
            out = greedy_match_repair(reduced, deficient, g, k)
            if out is None:
                out = backtracking_repair(
                    reduced, deficient, g, k, max_backtracks=10000
                )
            return out, None
        # variant == "rl"
        if policy is None:
            return None, None
        rl_success, rl_graph, rl_steps = rl_repair_rollout(
            reduced, deficient, g, policy, cycle_lengths, rwpe_dim
        )
        return (rl_graph if rl_success else None), rl_steps

    # -- Root sweep ----------------------------------------------------------
    # Excise a depth-d tree from each candidate root and repair; accept the
    # first root that yields a valid (k,g)-graph. Track how many roots we tried
    # and which one succeeded for reporting.
    repaired: nx.Graph[int] | None = None
    reduced: nx.Graph[int] = source_graph.copy()
    n_deficient = 0.0
    roots_tried = 0
    chosen_root = -1.0

    t0 = time.perf_counter()
    if error is None:
        for root in sorted(source_graph.nodes()):
            roots_tried += 1
            try:
                cand_reduced, cand_deficient, _ = excise_tree(source_graph, root, depth)
            except Exception as exc:
                error = str(exc)
                break

            if cand_reduced.number_of_nodes() == 0 or not cand_deficient:
                continue

            # Keep the first non-trivial reduced graph as the reporting fallback.
            if repaired is None and chosen_root < 0 and reduced is source_graph:
                reduced = cand_reduced
                n_deficient = float(len(cand_deficient))

            try:
                cand_repaired, cand_steps = _repair(cand_reduced, cand_deficient)
            except Exception as exc:
                error = str(exc)
                break

            if cand_repaired is None:
                continue

            try:
                ok = (
                    is_k_regular(cand_repaired, k) and compute_girth(cand_repaired) >= g
                )
            except Exception:
                ok = False

            if ok:
                repaired = cand_repaired
                reduced = cand_reduced
                n_deficient = float(len(cand_deficient))
                steps = cand_steps
                chosen_root = float(root)
                break

    elapsed = time.perf_counter() - t0

    success = repaired is not None
    reduced_n = float(reduced.number_of_nodes())

    # -- Result graph for reporting ------------------------------------------
    result_graph = repaired if repaired is not None else reduced

    # -- Girth of result -------------------------------------------------------
    try:
        girth_val = compute_girth(result_graph)
        girth_metric = float(girth_val) if math.isfinite(float(girth_val)) else -1.0
    except Exception:
        girth_metric = -1.0

    return [
        TrialResult(
            benchmark="excision",
            approach="excision",
            variant=variant,
            label=task.label,
            instance=name,
            target={"k": k, "g": g},
            success=success,
            elapsed_s=elapsed,
            steps=steps,
            n_nodes=result_graph.number_of_nodes(),
            n_edges=result_graph.number_of_edges(),
            metrics={
                "orig_n": orig_n,
                "reduced_n": reduced_n,
                "deficient": n_deficient,
                "girth": girth_metric,
                "roots_tried": float(roots_tried),
                "chosen_root": chosen_root,
            },
            model_id=model_id,
            model_params=model_params,
            model_size_mb=model_size_mb,
            model_hparams=model_hparams,
            error=error,
        )
    ]


register(Benchmark("excision", make_tasks, execute))
