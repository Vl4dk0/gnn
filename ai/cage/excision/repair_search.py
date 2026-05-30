"""Inference-time excision-and-repair for shrinking a valid (k,g)-graph.

Tree excision removes a BFS tree of depth (g-1)//2 from a k-regular,
girth->=g graph, leaving a smaller reduced graph with a set of deficient
vertices (degree < k).  Repairing that reduced graph back to k-regularity,
without introducing any cycle shorter than g, yields a *smaller* (k,g)-graph.

This module provides the inference counterpart to the excision RL trainer.
``excise_and_repair`` tries several roots; for each it excises a tree and
attempts a repair.  Repairs are driven by the trained ``RepairActorCritic``
policy when one is on disk (run deterministically), and fall back to the
classical ``backtracking_repair`` when no policy is present or the policy
gets stuck.  The returned graph is always verified to be k-regular, have
girth >= g, and be strictly smaller than the input.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import networkx as nx
import torch

from ai.cage.excision.baseline import backtracking_repair
from ai.cage.excision.excise import excise_tree
from ai.cage.excision.rl.env import RepairEnv
from ai.cage.excision.rl.model import RepairActorCritic
from backend.utils.graph_utils import compute_girth, is_k_regular, moore_bound


# Default location of the trained excision-repair policy on disk.
DEFAULT_POLICY_DIR = (
    Path(__file__).resolve().parents[3]
    / "trained"
    / "excision_repair"
    / "excision_repair_policy"
)


class RepairPolicy:
    """A loaded RepairActorCritic plus its feature config, ready for inference."""

    model: RepairActorCritic
    cycle_lengths: list[int]
    rwpe_dim: int

    def __init__(
        self,
        model: RepairActorCritic,
        cycle_lengths: list[int],
        rwpe_dim: int,
    ) -> None:
        self.model = model
        self.cycle_lengths = cycle_lengths
        self.rwpe_dim = rwpe_dim


def load_repair_policy(model_dir: Path = DEFAULT_POLICY_DIR) -> RepairPolicy | None:
    """Load the trained repair policy, or return None if it is not on disk.

    Returns None (rather than raising) when the weights/info are missing, so
    callers degrade gracefully to the classical backtracking repair.
    """
    info_path = model_dir / "info.json"
    weights_path = model_dir / "weights.pt"
    if not info_path.exists() or not weights_path.exists():
        return None

    with open(info_path) as f:
        info = cast(dict[str, object], json.load(f))
    training = cast(dict[str, object], info.get("training", {}))
    feat_cfg = cast(dict[str, object], training.get("feature_config", {}))

    cycle_lengths_raw = feat_cfg.get("cycle_lengths", [3, 4, 5, 6, 7, 8])
    cycle_lengths = [int(cast(int, v)) for v in cast(list[object], cycle_lengths_raw)]
    rwpe_dim = int(cast(int, feat_cfg.get("rwpe_dim", 8)))
    input_dim = int(cast(int, training.get("input_dim", 0)))
    hidden_dim = int(cast(int, training.get("hidden_dim", 64)))
    num_layers = int(cast(int, training.get("num_layers", 3)))

    model = RepairActorCritic(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
    state = torch.load(weights_path, map_location="cpu")  # pyright: ignore[reportAny]
    _ = model.load_state_dict(state)  # pyright: ignore[reportAny]
    _ = model.eval()
    return RepairPolicy(model, cycle_lengths, rwpe_dim)


def _repair_with_policy(
    reduced: nx.Graph[int],
    deficient: list[int],
    g_target: int,
    policy: RepairPolicy,
) -> nx.Graph[int] | None:
    """Greedily repair *reduced* by running the policy deterministically.

    Returns the repaired graph (all deficiencies resolved) or None if the
    policy runs into a dead end (no legal action left while still deficient).
    """
    env = RepairEnv(
        reduced,
        deficient,
        g_target,
        cycle_lengths=policy.cycle_lengths,
        rwpe_dim=policy.rwpe_dim,
    )
    obs = env.reset()

    # Rebuild the repaired graph from the edges the policy adds, so we do not
    # depend on the env's private working graph.
    repaired: nx.Graph[int] = reduced.copy()

    _ = policy.model.eval()
    with torch.no_grad():
        while True:
            legal = env.legal_actions()
            if not legal:
                return None  # stuck while still deficient
            node_ids: list[int] = list(cast(list[int], obs.node_ids))
            action, _logp, _val = policy.model.get_action(
                obs, legal, node_ids, deterministic=True
            )
            _ = repaired.add_edge(action[0], action[1])
            obs, _reward, done, info = env.step(action)
            if done:
                if bool(info.get("success", False)):
                    return repaired
                return None


def repair_reduced(
    reduced: nx.Graph[int],
    deficient: list[int],
    g_target: int,
    k: int,
    policy: RepairPolicy | None,
    max_backtracks: int = 10000,
) -> nx.Graph[int] | None:
    """Repair a reduced graph to k-regularity with girth >= g_target.

    Uses the trained policy first (deterministic) when one is provided; on a
    policy dead end, or when no policy is provided, falls back to the classical
    backtracking repair.  Returns the repaired graph or None on failure.
    """
    if policy is not None:
        repaired = _repair_with_policy(reduced, deficient, g_target, policy)
        if repaired is not None:
            return repaired
    return backtracking_repair(
        reduced, deficient, g_target, k, max_backtracks=max_backtracks
    )


def try_excise_root(
    G: nx.Graph[int],
    k: int,
    g: int,
    root: int,
    policy: RepairPolicy | None = None,
    *,
    depth: int | None = None,
    max_backtracks: int = 10000,
) -> nx.Graph[int] | None:
    """Try ONE root: excise a depth-d tree and repair it.

    ``depth`` defaults to the maximal girth-safe depth ``(g - 1) // 2``.  A
    *smaller* depth excises a smaller tree, which is the right choice when the
    full-depth excision would shrink the graph below the Moore bound (and thus
    be unrepairable) or leave a boundary the classical repair cannot close;
    callers that want incremental shrink can sweep depths from 1 upward.

    Returns the repaired graph iff it is a strictly smaller valid (k,g)-graph,
    else None.  This is the single-root core of ``excise_and_repair`` so callers
    that step through roots one at a time (e.g. the forge generator) can reuse
    the exact same excision + repair + validation logic.
    """
    if depth is None:
        depth = (g - 1) // 2
    reduced, deficient, _levels = excise_tree(G, root, depth)
    # A (k,g)-graph needs at least moore_bound(k, g) vertices, so a reduced graph
    # below that bound can never be repaired into a valid (k,g)-graph.
    if (
        not deficient
        or reduced.number_of_nodes() == 0
        or reduced.number_of_nodes() < moore_bound(k, g)
    ):
        return None

    repaired = repair_reduced(
        reduced, deficient, g, k, policy, max_backtracks=max_backtracks
    )
    if repaired is None:
        return None

    if (
        repaired.number_of_nodes() < G.number_of_nodes()
        and is_k_regular(repaired, k)
        and _girth_ok(repaired, g)
    ):
        return repaired
    return None


def excise_and_repair(
    G: nx.Graph[int],
    k: int,
    g: int,
    policy: RepairPolicy | None = None,
    *,
    roots: list[int] | None = None,
    max_roots: int | None = None,
    max_backtracks: int = 10000,
    verbose: bool = False,
) -> nx.Graph[int] | None:
    """Excise a depth-(g-1)//2 tree from G and repair it into a smaller (k,g)-graph.

    Tries roots one by one.  For each root the tree is excised and the reduced
    graph is repaired (policy first, classical fallback).  The first repair that
    is k-regular, has girth >= g, and is strictly smaller than G is returned.
    Returns None if no root yields such a graph.

    Args:
        G:        A valid (k,g)-graph (k-regular, girth >= g).
        k, g:     Target degree and girth.
        policy:   Optional trained repair policy.  None -> classical only.
        roots:    Roots to try (default: all vertices of G, sorted).
        max_roots: Cap on the number of roots attempted (default: all).
        verbose:  Print per-root progress.
    """
    candidate_roots = roots if roots is not None else sorted(G.nodes())
    if max_roots is not None:
        candidate_roots = candidate_roots[:max_roots]

    for root in candidate_roots:
        repaired = try_excise_root(G, k, g, root, policy, max_backtracks=max_backtracks)
        if repaired is not None:
            if verbose:
                print(
                    f"  root {root}: shrunk {G.number_of_nodes()} -> "
                    f"{repaired.number_of_nodes()}"
                )
            return repaired
        if verbose:
            print(f"  root {root}: no valid smaller (k,g)-graph")

    return None


def _girth_ok(G: nx.Graph[int], g: int) -> bool:
    """True iff G has girth >= g (infinite girth counts as satisfying)."""
    girth = compute_girth(G)
    if isinstance(girth, float):
        return True
    return girth >= g
