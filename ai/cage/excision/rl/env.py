"""RL environment for GNN-guided repair of excised (k,g)-graphs.

After tree excision, a set of "deficient" vertices (degree < k) must be
reconnected by adding edges that do not create short cycles.  This environment
models that sequential edge-addition process as an MDP.

State  : the partially-repaired graph as a PyG Data, with node features that
         include degree-deficiency, deficiency-set membership, and rich
         structural features (cycle counts + RWPE) to break the symmetry
         barrier on regular graphs (Chen et al. NeurIPS 2023).

Action : an unordered pair (u, v) from the set of currently-legal edges.
         The action space shrinks monotonically; the environment exposes
         legal_actions() for masking.

Reward :
  +1   per deficient vertex whose degree is fully restored by the step
  +10  on full repair success (all deficiencies resolved)
  -1   if no legal action remains but deficiencies persist (episode failure)

  In addition to those terminal/restoration rewards, a small potential-based
  shaping term is added so the agent gets signal *before* the episode ends:
    + SHAPE_RESOLVE * (deficiency units resolved this step)
    - DEADEND_PENALTY * (number of still-deficient vertices left with zero
                         legal partners after this step)
  Shaping never changes the success/failure terminal semantics (a stuck step
  still terminates with FAILURE_REWARD, a full repair still gets SUCCESS_REWARD).
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import torch
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]

from ai.cage.excision.legality import DistanceCache
from ai.utils.structural_features import add_structural_features


class RepairEnv:
    """Gym-like RL environment for graph repair after tree excision.

    The environment works on a single excision instance: a fixed starting
    graph, a fixed deficient set, and a fixed g_target.  Episodes reset back
    to the same starting state.

    Args:
        starting_graph:  Reduced graph (tree removed, some vertices deficient).
        deficient:       List of deficient vertex IDs (degree < k in starting_graph).
        g_target:        Target girth.
        cycle_lengths:   Lengths for cycle-count structural features.
        rwpe_dim:        Dimension for RWPE structural features.
    """

    SUCCESS_REWARD: float = 10.0
    RESOLVE_REWARD: float = 1.0
    FAILURE_REWARD: float = -1.0
    # Potential-based shaping (intermediate signal; does not alter terminals).
    SHAPE_RESOLVE: float = 0.5
    DEADEND_PENALTY: float = 0.5

    def __init__(
        self,
        starting_graph: nx.Graph[int],
        deficient: list[int],
        g_target: int,
        cycle_lengths: list[int] | None = None,
        rwpe_dim: int = 8,
    ) -> None:
        self._starting_graph: nx.Graph[int] = starting_graph.copy()
        self._deficient_init: list[int] = list(deficient)
        self.g_target: int = g_target
        self.cycle_lengths: list[int] | None = cycle_lengths
        self.rwpe_dim: int = rwpe_dim

        self._k: int = (
            max(d for _, d in starting_graph.degree())
            if starting_graph.number_of_nodes() > 0
            else 0
        )

        # Current episode state
        self._graph: nx.Graph[int] = starting_graph.copy()
        self._deficiency: dict[int, int] = self._compute_deficiency(self._graph)
        # Incremental BFS-distance cache over the working graph; invalidated
        # per added edge so legal_actions() avoids recomputing all-pairs BFS.
        self._dist: DistanceCache = DistanceCache(self._graph, g_target)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Data:
        """Reset to the initial excised state."""
        self._graph = self._starting_graph.copy()
        self._deficiency = self._compute_deficiency(self._graph)
        self._dist = DistanceCache(self._graph, self.g_target)
        return self._build_obs()

    def step(
        self,
        action: tuple[int, int],
    ) -> tuple[Data, float, bool, dict[str, Any]]:
        """Add edge `action` to the current graph.

        Args:
            action: (u, v) with u < v.  Must be a legal action.

        Returns:
            (observation, reward, done, info)
        """
        u, v = action
        assert u != v, "Self-loop action"
        assert not self._graph.has_edge(u, v), "Edge already exists"

        prev_deficient_count = sum(1 for d in self._deficiency.values() if d > 0)
        prev_units = sum(d for d in self._deficiency.values() if d > 0)

        self._graph.add_edge(u, v)
        self._dist.invalidate_edge(u, v)

        # Update deficiency
        if u in self._deficiency:
            self._deficiency[u] -= 1
        if v in self._deficiency:
            self._deficiency[v] -= 1

        now_deficient_count = sum(1 for d in self._deficiency.values() if d > 0)
        now_units = sum(d for d in self._deficiency.values() if d > 0)
        resolved = prev_deficient_count - now_deficient_count
        units_resolved = prev_units - now_units  # always 2 (one per endpoint)

        # Check termination
        fully_repaired = now_deficient_count == 0
        legal = self.legal_actions()
        stuck = (not fully_repaired) and len(legal) == 0

        done = fully_repaired or stuck

        # Potential-based shaping: reward deficiency units resolved this step and
        # penalise leaving any still-deficient vertex with no legal partner
        # (a dead end). Applied only on non-terminal steps so success/failure
        # terminal semantics stay unchanged.
        if fully_repaired:
            reward = self.RESOLVE_REWARD * resolved + self.SUCCESS_REWARD
        elif stuck:
            reward = self.FAILURE_REWARD
        else:
            partners = self._legal_partner_counts(legal)
            dead_ends = sum(
                1
                for w, d in self._deficiency.items()
                if d > 0 and partners.get(w, 0) == 0
            )
            reward = (
                self.RESOLVE_REWARD * resolved
                + self.SHAPE_RESOLVE * units_resolved
                - self.DEADEND_PENALTY * dead_ends
            )

        info: dict[str, Any] = {
            "action": action,
            "resolved": resolved,
            "deficient_remaining": now_deficient_count,
            "success": fully_repaired,
            "done_reason": "success"
            if fully_repaired
            else ("stuck" if stuck else "step"),
        }

        obs = self._build_obs()
        return obs, reward, done, info

    def legal_actions(self) -> list[tuple[int, int]]:
        """Return all currently legal (u, v) actions with u < v.

        An action is legal iff:
          - u and v are both deficient (degree < k)
          - adding (u, v) would not create a cycle shorter than g_target
        """
        deficient_now = [v for v, d in self._deficiency.items() if d > 0]
        actions: list[tuple[int, int]] = []
        for i, u in enumerate(deficient_now):
            for v in deficient_now[i + 1 :]:
                if self._dist.is_legal(u, v):
                    actions.append((u, v))
        return actions

    @staticmethod
    def _legal_partner_counts(
        legal: list[tuple[int, int]],
    ) -> dict[int, int]:
        """Count, per vertex, how many legal actions currently touch it."""
        counts: dict[int, int] = {}
        for a, b in legal:
            counts[a] = counts.get(a, 0) + 1
            counts[b] = counts.get(b, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_deficiency(self, H: nx.Graph[int]) -> dict[int, int]:
        """Return {v: k - deg(v)} for all deficient vertices."""
        deficiency: dict[int, int] = {}
        for v in self._deficient_init:
            if v in H:
                deficit = self._k - H.degree(v)
                if deficit > 0:
                    deficiency[v] = deficit
        return deficiency

    def _build_obs(self) -> Data:
        """Build PyG observation from current graph state.

        Node features (per vertex):
          [0] degree / k                  (normalised current degree)
          [1] deficiency / k              (0 if not deficient)
          [2] is_deficient (0/1)          (binary membership)
          [3] k / 10.0                    (context: regularity)
          [4] g_target / 14.0             (context: target girth)

        Then structural features are appended via add_structural_features
        (cycle counts + RWPE).
        """
        nodes = sorted(self._graph.nodes())
        n = len(nodes)
        node_idx = {v: i for i, v in enumerate(nodes)}

        x = torch.zeros((n, 5), dtype=torch.float)
        for v in nodes:
            i = node_idx[v]
            deg = self._graph.degree(v)
            deficit = self._deficiency.get(v, 0)
            x[i, 0] = deg / max(self._k, 1)
            x[i, 1] = deficit / max(self._k, 1)
            x[i, 2] = 1.0 if deficit > 0 else 0.0
            x[i, 3] = self._k / 10.0
            x[i, 4] = self.g_target / 14.0

        # Build edge_index (re-indexed to 0..n-1)
        src_list: list[int] = []
        dst_list: list[int] = []
        for u, v in self._graph.edges():
            ui = node_idx[u]
            vi = node_idx[v]
            src_list.append(ui)
            dst_list.append(vi)
            src_list.append(vi)
            dst_list.append(ui)

        if src_list:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, num_nodes=n)
        # Attach node-id mapping so policy can decode actions
        data.node_ids = nodes  # pyright: ignore[reportAttributeAccessIssue]

        # Structural features (non-optional per spec)
        data = add_structural_features(
            data,
            cycle_lengths=self.cycle_lengths,
            rwpe_dim=self.rwpe_dim,
        )
        return data
