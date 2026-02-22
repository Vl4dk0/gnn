"""
Gym-like environment for Cage Graph Construction.
"""

import random
import networkx as nx
import torch
from torch_geometric.data import Data
from typing import Any

from backend.utils.graph_utils import moore_bound, moore_hoffman_upper_bound


class CageConstructionEnv:
    """
    RL Environment for constructing (k,g)-cage graphs.

    State:
        - Graph structure (edge_index)
        - Node features: [One-hot Degree, Norm Degree, Norm K, Norm G]

    Action:
        - Add edge (u, v)

    Constraints:
        - No self-loops (handled by action space definition)
        - No multi-edges
        - Max degree k
        - Girth >= g
    """

    k: int
    g: int
    default_k: int
    default_g: int
    randomize_params: bool
    MAX_K: int
    MAX_G: int
    mb: int
    n_min: int
    n_max: int
    max_steps: int
    num_nodes: int
    graph: nx.Graph
    current_step: int
    episode_score: float

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

        # Max possible values for normalization
        self.MAX_K = 10
        self.MAX_G = 12

        # Initial bounds calculation (will update on reset if randomized)
        self._update_bounds(n_min, n_max)

        self.max_steps = max_steps
        self.num_nodes = 0
        self.graph = nx.Graph()
        self.current_step = 0
        self.episode_score = 0.0

    def _update_bounds(
        self, n_min: int | None = None, n_max: int | None = None
    ) -> None:
        self.mb = moore_bound(self.k, self.g)
        if n_min is None:
            self.n_min = self.mb
        else:
            self.n_min = n_min

        if n_max is None:
            # Heuristic upper bound
            self.n_max = max(int(self.mb * 1.5), self.mb + 10)
        else:
            self.n_max = n_max

    def reset(self, num_nodes: int | None = None) -> Data:
        """
        Reset the environment.
        """
        if self.randomize_params:
            # Curriculum: Randomize k and g with safe complexity limits
            # We want to avoid graphs that are too large for efficient training
            # Limit based on Moore Bound

            while True:
                # Candidate k, g
                k = random.randint(3, 7)
                g = random.randint(5, 10)

                try:
                    mb = moore_bound(k, g)
                    # Hard limit: Training on graphs > 60 nodes is too slow for now
                    if mb <= 60:
                        self.k = k
                        self.g = g
                        break
                except ValueError:
                    continue

            self._update_bounds()
        else:
            self.k = self.default_k
            self.g = self.default_g
            self._update_bounds()

        if num_nodes is None:
            self.num_nodes = random.randint(self.n_min, self.n_max)
        else:
            self.num_nodes = num_nodes

        self.current_step = 0
        self.episode_score = 0.0
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(self.num_nodes))

        return self._get_obs()

    def _get_obs(self) -> Data:
        """Convert current state to PyG Data object."""
        # Node features:
        # 1. One-hot encoding of degree (up to MAX_K)
        # 2. Normalized degree
        # 3. Normalized target K
        # 4. Normalized target G

        degrees = [self.graph.degree(i) for i in range(self.num_nodes)]

        # Feature dim = MAX_K + 1 (norm deg) + 1 (norm K) + 1 (norm G) = MAX_K + 3
        # e.g., if MAX_K=10, input_dim=13
        input_dim = self.MAX_K + 3
        x = torch.zeros((self.num_nodes, input_dim), dtype=torch.float)

        norm_k = self.k / self.MAX_K
        norm_g = self.g / self.MAX_G

        for i, d in enumerate(degrees):
            # One-hot degree (clamp to MAX_K-1)
            idx = min(d, self.MAX_K - 1)
            x[i, idx] = 1.0

            # Normalized features
            x[i, self.MAX_K] = d / self.k  # Norm Degree
            x[i, self.MAX_K + 1] = norm_k  # Target K
            x[i, self.MAX_K + 2] = norm_g  # Target G

        # Edge index
        if self.graph.number_of_edges() > 0:
            edge_index = (
                torch.tensor(list(self.graph.edges()), dtype=torch.long)
                .t()
                .contiguous()
            )
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index, num_nodes=self.num_nodes)

    def get_valid_action_mask(self) -> torch.Tensor:
        """
        Compute valid actions over all vertex pairs.

        Action semantics per pair (u, v):
        - If edge exists: action means REMOVE edge (always valid).
        - Else: action means ADD edge (valid only if constraints hold).
        """
        mask = []
        nodes = range(self.num_nodes)
        degrees = [self.graph.degree(i) for i in nodes]

        def check_girth(u, v):
            try:
                # Optimization: BFS with depth limit g-2
                path_len = nx.shortest_path_length(self.graph, u, v)
                return (path_len + 1) >= self.g
            except nx.NetworkXNoPath:
                return True

        for u in range(self.num_nodes):
            for v in range(u + 1, self.num_nodes):
                # Existing edge can always be removed.
                if self.graph.has_edge(u, v):
                    mask.append(True)
                    continue

                # Otherwise action means ADD edge; validate constraints.
                if degrees[u] >= self.k or degrees[v] >= self.k:
                    mask.append(False)
                    continue
                if not check_girth(u, v):
                    mask.append(False)
                    continue
                mask.append(True)

        return torch.tensor(mask, dtype=torch.bool)

    def idx_to_edge(self, idx: int) -> tuple[int, int]:
        """Convert flat index to (u, v)."""
        count = 0
        for u in range(self.num_nodes):
            row_len = self.num_nodes - 1 - u
            if idx < count + row_len:
                v = u + 1 + (idx - count)
                return u, v
            count += row_len
        return 0, 0

    def step(self, action_idx: int) -> tuple[Data, float, bool, dict[str, Any]]:
        """Execute action."""
        self.current_step += 1

        u, v = self.idx_to_edge(action_idx)
        info: dict[str, Any] = {
            "k": self.k,
            "g": self.g,
            "num_nodes": self.num_nodes,
            "step": self.current_step,
            "action_idx": action_idx,
            "edge": [u, v],
            "action_type": "unknown",
        }

        # Existing edge => REMOVE action.
        if self.graph.has_edge(u, v):
            self.graph.remove_edge(u, v)
            reward = -0.2
            done = False
            info["action_type"] = "remove"
            info["success"] = False
        else:
            # Otherwise this is an ADD action and must satisfy constraints.
            if self.graph.degree(u) >= self.k or self.graph.degree(v) >= self.k:
                reward = -1.0
                done = False
                info["action_type"] = "add"
                info["error"] = "Invalid action: degree limit"
                info["success"] = False
                info["done_reason"] = "invalid_action"
            else:
                try:
                    path_len = nx.shortest_path_length(self.graph, u, v)
                    add_ok = (path_len + 1) >= self.g
                except nx.NetworkXNoPath:
                    add_ok = True

                if not add_ok:
                    reward = -1.0
                    done = False
                    info["action_type"] = "add"
                    info["error"] = "Invalid action: girth violation"
                    info["success"] = False
                    info["done_reason"] = "invalid_action"
                else:
                    self.graph.add_edge(u, v)
                    reward = 0.1
                    done = False
                    info["action_type"] = "add"
                    info["success"] = False

        is_k_reg = all(self.graph.degree(n) == self.k for n in self.graph.nodes())

        if is_k_reg:
            # Huge reward for success
            reward = 10.0 + (self.g * 0.5)  # Bonus for harder girths
            done = True
            info["success"] = True
            info["done_reason"] = "success"

        # Dead-end: no valid add/remove actions left.
        valid_mask = self.get_valid_action_mask()
        if (not done) and (not bool(valid_mask.any())):
            reward = -2.0
            done = True
            info["success"] = False
            info["done_reason"] = "dead_end"

        if (not done) and self.current_step >= self.max_steps:
            done = True
            reward = -1.0
            info["success"] = False
            info["done_reason"] = "max_steps"

        self.episode_score += reward
        if (not done) and self.episode_score <= -10.0:
            done = True
            info["success"] = False
            info["done_reason"] = "score_floor"

        info["num_edges"] = self.graph.number_of_edges()
        info["is_k_regular"] = is_k_reg
        info["episode_score"] = self.episode_score

        return self._get_obs(), reward, done, info
