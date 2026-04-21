"""Gym-like environment for cage graph construction."""

from __future__ import annotations

import random
from collections import deque

import networkx as nx
import torch
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]

from ai.utils.device import get_preferred_device
from backend.utils.graph_utils import moore_bound

# (k, g) pairs sorted by moore_bound ascending, filtered to mb <= 60, k >= 3
PROGRESSIVE_PAIRS: list[tuple[int, int]] = [
    (3, 5),  # mb=10
    (3, 6),  # mb=14
    (4, 5),  # mb=17
    (3, 7),  # mb=22
    (4, 6),  # mb=26
    (5, 5),  # mb=26
    (3, 8),  # mb=30
    (6, 5),  # mb=37
    (5, 6),  # mb=42
    (3, 9),  # mb=46
    (7, 5),  # mb=50
    (4, 7),  # mb=53
]


class CageConstructionEnv:
    """
    RL environment for constructing (k,g)-graphs with edge-only actions.

    Action space:
      - one action for each unordered pair (u,v), u<v
      - if edge exists: try remove
      - if edge does not exist: try add
    """

    SCORE_FLOOR: float = -20.0  # do not go beneath this
    SUCCESS_REWARD: float = 50.0  # when found the correct thing
    INVALID_PENALTY: float = -0.005  # if you make an invalid action
    ADD_REWARD: float = 0.01  # add edge
    REMOVE_PENALTY: float = -0.02  # remove edge
    SATISFY_BONUS: float = 0.05  # if graph became k-regular or girth became correct

    k: int
    g: int
    default_k: int
    default_g: int
    randomize_params: bool
    max_k_norm: int
    max_g_norm: int
    mb: int
    n_min: int
    n_max: int
    max_steps: int
    num_nodes: int
    graph: nx.Graph[int]
    current_step: int
    episode_score: float
    _prev_potential: float

    # Progressive training state
    pairs: list[tuple[int, int]]
    unlocked: int
    pair_history: dict[tuple[int, int], deque[bool]]
    solve_window: int
    solve_threshold: float
    device: torch.device

    def __init__(
        self,
        k: int,
        g: int,
        n_min: int | None = None,
        n_max: int | None = None,
        max_steps: int = 1000,
        randomize_params: bool = False,
    ):
        self.default_k = k
        self.default_g = g
        self.k = k
        self.g = g
        self.randomize_params = randomize_params

        # Feature layout: degree bins + 4 scalar/context features.
        # Keep degree bins at 4 so input_dim is 4 + 4 = 8.
        self.max_k_norm = 4
        self.max_g_norm = 12
        self._update_bounds(n_min, n_max)

        self.max_steps = max_steps
        self.num_nodes = 0
        self.graph = nx.Graph()
        self.current_step = 0
        self.episode_score = 0.0
        self._prev_potential = 0.0

        self.device = torch.device(get_preferred_device())

        # Progressive training
        self.pairs = PROGRESSIVE_PAIRS
        self.unlocked = 1
        self.pair_history = {p: deque(maxlen=8) for p in self.pairs}
        self.solve_window = 8
        self.solve_threshold = 0.5

    def _update_bounds(
        self, n_min: int | None = None, n_max: int | None = None
    ) -> None:
        self.mb = moore_bound(self.k, self.g)
        self.n_min = self.mb if n_min is None else n_min
        self.n_max = max(int(self.mb * 1.5), self.mb + 10) if n_max is None else n_max

    def get_input_dim(self) -> int:
        """Feature dimension for node state."""
        # one-hot degree bins + norm degree + norm k + norm g + active flag
        return self.max_k_norm + 4

    def get_num_pair_actions(self) -> int:
        """Action count over all unordered pairs."""
        return (self.num_nodes * (self.num_nodes - 1)) // 2

    def get_action_dim(self) -> int:
        """Total action dimension."""
        return self.get_num_pair_actions()

    def _active_nodes(self) -> set[int]:
        """Active vertices are exactly those connected to the graph."""
        return {n for n in self.graph.nodes if self.graph.degree(n) > 0}

    def _active_subgraph(self) -> nx.Graph[int]:
        active = self._active_nodes()
        if not active:
            return nx.Graph()
        return self.graph.subgraph(active).copy()

    def _active_component_count(self) -> int:
        sg = self._active_subgraph()
        if sg.number_of_nodes() == 0:
            return 0
        return nx.number_connected_components(sg)

    def _is_k_regular_active(self) -> bool:
        active = self._active_nodes()
        if len(active) < (self.k + 1):
            return False
        return all(self.graph.degree(n) == self.k for n in active)

    def _is_girth_satisfied_active(self) -> bool:
        sg = self._active_subgraph()
        if sg.number_of_edges() == 0:
            return True
        cycles = nx.cycle_basis(sg)
        if not cycles:
            return True
        return min(len(c) for c in cycles) >= self.g

    def _potential(self) -> float:
        """Potential function measuring cage construction progress."""
        active = self._active_nodes()
        if not active:
            return 0.0
        target_edges = (self.k * self.mb) / 2
        num_edges = self.graph.number_of_edges()
        regularity_sum = sum(min(self.graph.degree(n), self.k) / self.k for n in active)
        regularity_score = regularity_sum / self.mb
        edge_score = min(num_edges / max(1.0, target_edges), 1.0)
        return 5.0 * (0.6 * regularity_score + 0.4 * edge_score)

    def _can_add_by_active_rule(self, u: int, v: int) -> bool:
        active = self._active_nodes()
        if not active:
            # First edge can connect any two inactive vertices.
            return True
        return (u in active) or (v in active)

    def _can_add_edge(self, u: int, v: int) -> bool:
        if self.graph.has_edge(u, v):
            return False
        if not self._can_add_by_active_rule(u, v):
            return False
        if self.graph.degree(u) >= self.k or self.graph.degree(v) >= self.k:
            return False
        try:
            path_len = int(nx.shortest_path_length(self.graph, u, v))  # pyright: ignore[reportUnknownMemberType]
            return (path_len + 1) >= self.g
        except nx.NetworkXNoPath:
            return True

    def _can_remove_edge(self, u: int, v: int) -> bool:
        if not self.graph.has_edge(u, v):
            return False
        self.graph.remove_edge(u, v)
        split = self._active_component_count() > 1
        below_floor = len(self._active_nodes()) < self.mb
        _ = self.graph.add_edge(u, v)
        return not split and not below_floor

    def report_result(self, success: bool) -> bool:
        """Record episode outcome; return True if a new stage was unlocked."""
        pair = (self.k, self.g)
        if pair in self.pair_history:
            self.pair_history[pair].append(success)
        if self.unlocked < len(self.pairs):
            newest = self.pairs[self.unlocked - 1]
            history = self.pair_history[newest]
            if (
                len(history) >= self.solve_window
                and sum(history) / len(history) >= self.solve_threshold
            ):
                self.unlocked += 1
                return True
        return False

    def reset(self, num_nodes: int | None = None) -> Data:
        """Reset environment and sample a new target when randomization is enabled."""
        if self.randomize_params:
            pool = self.pairs[: self.unlocked]
            weights = [1.0]
            for _ in range(len(pool) - 1):
                weights.append(3.0 * sum(weights))
            chosen = random.choices(pool, weights=weights, k=1)[0]
            self.k, self.g = chosen
            self._update_bounds()
        else:
            self.k = self.default_k
            self.g = self.default_g
            self._update_bounds()

        self.num_nodes = (
            random.randint(self.n_min, self.n_max) if num_nodes is None else num_nodes
        )
        self.current_step = 0
        self.episode_score = 0.0
        self._prev_potential = 0.0
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(self.num_nodes))
        return self._get_obs()

    def _get_obs(self) -> Data:
        """Convert state to PyG Data."""
        active = self._active_nodes()
        input_dim = self.get_input_dim()
        x = torch.zeros(
            (self.num_nodes, input_dim), dtype=torch.float, device=self.device
        )

        norm_k = self.k / self.max_k_norm
        norm_g = self.g / self.max_g_norm

        for i in range(self.num_nodes):
            deg = self.graph.degree(i) if i in active else 0
            idx = min(deg, self.max_k_norm - 1)
            x[i, idx] = 1.0 if i in active else 0.0
            x[i, self.max_k_norm] = (deg / max(1, self.k)) if i in active else 0.0
            x[i, self.max_k_norm + 1] = norm_k
            x[i, self.max_k_norm + 2] = norm_g
            x[i, self.max_k_norm + 3] = 1.0 if i in active else 0.0

        edges = list(self.graph.edges())
        if edges:
            edge_index = (
                torch.tensor(edges, dtype=torch.long, device=self.device)
                .t()
                .contiguous()
            )
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)

        return Data(x=x, edge_index=edge_index, num_nodes=self.num_nodes)

    def idx_to_edge(self, idx: int) -> tuple[int, int]:
        """Convert flat pair index to (u,v)."""
        count = 0
        for u in range(self.num_nodes):
            row_len = self.num_nodes - 1 - u
            if idx < count + row_len:
                v = u + 1 + (idx - count)
                return u, v
            count += row_len
        return 0, 0

    def get_valid_action_mask(self) -> torch.Tensor:
        """
        Return valid pair actions.
        - Existing edge: removable action is allowed only if it does not split active graph.
        - Missing edge: add action allowed only if _can_add_edge passes.
        """
        mask: list[bool] = []
        for u in range(self.num_nodes):
            for v in range(u + 1, self.num_nodes):
                if self.graph.has_edge(u, v):
                    mask.append(self._can_remove_edge(u, v))
                else:
                    mask.append(self._can_add_edge(u, v))
        return torch.tensor(mask, dtype=torch.bool, device=self.device)

    def step(
        self, action_idx: int
    ) -> tuple[Data, float, bool, dict[str, int | str | bool | list[int] | float]]:
        """Apply one pair action."""
        self.current_step += 1
        reward = 0.0
        done = False

        u, v = self.idx_to_edge(action_idx)
        pre_active_count = len(self._active_nodes())
        info: dict[str, int | str | bool | list[int] | float] = {
            "k": self.k,
            "g": self.g,
            "capacity_nodes": self.num_nodes,
            "active_nodes": pre_active_count,
            "step": self.current_step,
            "action_idx": action_idx,
            "edge": [u, v],
            "action_type": "unknown",
            "success": False,
        }

        valid_action = False
        if self.graph.has_edge(u, v):
            # Try remove; reject if it would split active graph.
            if not self._can_remove_edge(u, v):
                reward += self.INVALID_PENALTY
                info["action_type"] = "edge_invalid_remove"
                info["done_reason"] = "split_component"
            else:
                self.graph.remove_edge(u, v)
                valid_action = True
                reward += self.REMOVE_PENALTY
                info["action_type"] = "edge_remove"
        else:
            if self._can_add_edge(u, v):
                _ = self.graph.add_edge(u, v)
                reward += self.ADD_REWARD
                valid_action = True
                info["action_type"] = "edge_add"
            else:
                reward += self.INVALID_PENALTY
                info["action_type"] = "edge_invalid_add"
                info["done_reason"] = "add_rule_or_constraint"

        # Progress-based reward shaping (undiscounted potential difference)
        new_potential = self._potential()
        reward += new_potential - self._prev_potential
        self._prev_potential = new_potential

        girth_ok = self._is_girth_satisfied_active()
        is_k_regular = self._is_k_regular_active()
        if valid_action and (girth_ok or is_k_regular):
            reward += self.SATISFY_BONUS

        if is_k_regular and girth_ok:
            reward += self.SUCCESS_REWARD
            done = True
            info["success"] = True
            info["done_reason"] = "success"

        if (not done) and self.current_step >= self.max_steps:
            done = True
            info["success"] = False
            info["done_reason"] = "max_steps"

        self.episode_score += reward
        if (not done) and self.episode_score <= self.SCORE_FLOOR:
            done = True
            info["success"] = False
            info["done_reason"] = "score_floor"

        info["episode_score"] = self.episode_score
        info["num_edges"] = self.graph.number_of_edges()
        info["active_nodes"] = len(self._active_nodes())
        info["girth_ok"] = girth_ok
        info["is_k_regular"] = is_k_regular

        return self._get_obs(), reward, done, info
