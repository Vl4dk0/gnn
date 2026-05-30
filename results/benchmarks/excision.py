"""Excision benchmark — tree-excision repair: classical vs RL policy.

For each source (k,g)-cage graph the benchmark:
  1. Excises a BFS tree of depth (g-1)//2 rooted at vertex 0.
  2. Repairs the deficient boundary set using either:
     - classical: greedy_match_repair → backtracking_repair fallback.
     - rl:        GNN RepairActorCritic rollout (deterministic).
  3. Verifies the result is k-regular with girth >= g.

The trained RL policy at ai/trained/excision_repair/excision_repair_policy was
trained for only ~2 episodes and is essentially random — near-random results are
expected.  The benchmark is designed to be re-run after proper training.
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

_POLICY_DIR = Path("ai/trained/excision_repair/excision_repair_policy")

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
        for variant in ("classical", "rl"):
            tasks.append(
                Task(
                    benchmark="excision",
                    label=f"{name} excise[{variant}]",
                    payload={
                        "idx": idx,
                        "name": name,
                        "k": k,
                        "g": g,
                        "approach": "excision_repair",
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

    # -- Excise --------------------------------------------------------------
    depth = (g - 1) // 2
    reduced, deficient, _ = excise_tree(source_graph, 0, depth)

    # Shared metrics (graph-size context)
    orig_n = float(source_graph.number_of_nodes())
    reduced_n = float(reduced.number_of_nodes())
    n_deficient = float(len(deficient))

    # -- Repair --------------------------------------------------------------
    error: str | None = None
    repaired: nx.Graph[int] | None = None
    steps: int | None = None

    # Model metadata (only filled for rl variant)
    model_id: str | None = None
    model_params: int | None = None
    model_size_mb: float | None = None
    model_hparams: dict[str, object] | None = None

    t0 = time.perf_counter()

    if variant == "classical":
        try:
            repaired = greedy_match_repair(reduced, deficient, g, k)
            if repaired is None:
                repaired = backtracking_repair(
                    reduced, deficient, g, k, max_backtracks=10000
                )
        except Exception as exc:
            error = str(exc)
            repaired = None

    else:  # variant == "rl"
        try:
            policy, cycle_lengths, rwpe_dim = load_repair_policy(_POLICY_DIR)
            params, size_mb, hparams = model_meta_from_dir(_POLICY_DIR)
            model_id = "excision_repair_policy"
            model_params = params
            model_size_mb = size_mb
            model_hparams = hparams

            rl_success, rl_graph, rl_steps = rl_repair_rollout(
                reduced, deficient, g, policy, cycle_lengths, rwpe_dim
            )
            repaired = rl_graph if rl_success else None
            steps = rl_steps
        except Exception as exc:
            error = str(exc)
            repaired = None

    elapsed = time.perf_counter() - t0

    # -- Verify success -------------------------------------------------------
    if repaired is not None:
        try:
            success = is_k_regular(repaired, k) and compute_girth(repaired) >= g
        except Exception:
            success = False
    else:
        success = False

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
            approach="excision_repair",
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
            },
            model_id=model_id,
            model_params=model_params,
            model_size_mb=model_size_mb,
            model_hparams=model_hparams,
            error=error,
        )
    ]


register(Benchmark("excision", make_tasks, execute))
