import os
import torch
import networkx as nx
from typing import Any
from torch_geometric.data import Data

from ai.cage.rl.model import ActorCritic
from ai.registry import load_model, get_best_model_id, list_trained_models
from backend.utils.graph_utils import (
    is_k_regular,
    compute_girth,
    moore_bound,
    moore_hoffman_upper_bound,
)


class RLGenerator:
    """
    RL-based Cage Graph Generator.
    """

    k: int
    g: int
    mb: int
    upper_bound: int
    MAX_K: int
    MAX_G: int
    device: torch.device
    model: ActorCritic
    num_nodes: int
    graph: nx.Graph
    step_count: int
    is_complete: bool
    success: bool
    start_time: float
    stack: list[Any]

    def __init__(
        self, k: int, g: int, model_type: str = "gin", model_path: str | None = None
    ):
        self.k = k
        self.g = g
        self.mb = moore_bound(k, g)
        self.upper_bound = moore_hoffman_upper_bound(k, g)

        # Max parameters for normalization (must match training env)
        self.MAX_K = 10
        self.MAX_G = 12

        # Load Model
        self.device = torch.device("cpu")  # Inference on CPU is fine

        if model_path:
            # Manual load from path
            print(f"Loading RL model from specific path: {model_path}")
            # We need to construct the model first?
            # Or assume we can just load the state dict into a generic shell?
            # ActorCritic needs input_dim/hidden_dim which are saved in registry config but maybe not in raw pt.
            # If using manual path, we assume default config or contained in path?
            # For simplicity, if model_path is given, we assume it's a weights file and we use default config.
            input_dim = self.MAX_K + 3
            self.model = ActorCritic(
                model_type=model_type, input_dim=input_dim, hidden_dim=128
            )
            try:
                self.model.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
            except Exception as e:
                print(
                    f"Failed to load model from {model_path}: {e}. Using random weights."
                )

        else:
            # Use Registry
            model_id = None

            # 1. Try to find the best model of the requested type
            models = list_trained_models("cage")
            for m in models:
                if m.get("training", {}).get("model_type") == model_type:
                    model_id = m["model_id"]
                    print(
                        f"Found best {model_type} model: {model_id} (Avg Reward: {m.get('metrics', {}).get('avg_reward', 'N/A')})"
                    )
                    break

            # 2. If no model of that type, try ANY best model
            if model_id is None:
                model_id = get_best_model_id("cage")
                if model_id:
                    print(
                        f"No {model_type} model found. Using best available: {model_id}"
                    )

            if model_id:
                try:
                    loaded_model = load_model("cage", model_id, device="cpu")
                    if isinstance(loaded_model, ActorCritic):
                        self.model = loaded_model
                    else:
                        print(
                            f"Model {model_id} is not an ActorCritic. Using random initialization."
                        )
                        input_dim = self.MAX_K + 3
                        self.model = ActorCritic(
                            model_type=model_type, input_dim=input_dim, hidden_dim=128
                        )
                    print(f"Successfully loaded model {model_id} from registry.")
                except Exception as e:
                    print(
                        f"Error loading from registry: {e}. Using random initialization."
                    )
                    input_dim = self.MAX_K + 3
                    self.model = ActorCritic(
                        model_type=model_type, input_dim=input_dim, hidden_dim=128
                    )
            else:
                print(
                    "No trained models found in registry. Using random initialization."
                )
                input_dim = self.MAX_K + 3
                self.model = ActorCritic(
                    model_type=model_type, input_dim=input_dim, hidden_dim=128
                )

        self.model.eval()

        # Initialization
        # Start with random number of nodes between MB and UB?
        # Or fixed? Let's pick Moore Bound to start, or slightly more.
        self.num_nodes = self.mb + 2  # Start slightly larger

        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(self.num_nodes))

        self.step_count = 0
        self.is_complete = False
        self.success = False
        self.start_time = 0

        # Backtracking stack
        # List of (graph_copy, forbidden_edges)
        self.stack = []

    def elapsed_time(self) -> float:
        import time

        if self.start_time == 0:
            return 0
        return time.time() - self.start_time

    def is_regular(self) -> bool:
        return is_k_regular(self.graph, self.k)

    def _get_obs(self) -> Data:
        """Construct PyG data from current graph."""
        degrees = [self.graph.degree(i) for i in range(self.num_nodes)]

        input_dim = self.MAX_K + 3
        x = torch.zeros((self.num_nodes, input_dim), dtype=torch.float)

        norm_k = self.k / self.MAX_K
        norm_g = self.g / self.MAX_G

        for i, d in enumerate(degrees):
            # One-hot degree
            idx = min(d, self.MAX_K - 1)
            x[i, idx] = 1.0

            # Normalized features
            x[i, self.MAX_K] = d / self.k
            x[i, self.MAX_K + 1] = norm_k
            x[i, self.MAX_K + 2] = norm_g

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

    def _get_valid_mask(self) -> torch.Tensor:
        # Re-implement mask logic locally or reuse Env?
        # Re-implementing is safer to avoid dependency issues with Gym
        mask = []
        degrees = [self.graph.degree(i) for i in range(self.num_nodes)]

        def check_girth(u, v):
            try:
                # Optimization: BFS with depth limit g-2
                path_len = nx.shortest_path_length(self.graph, u, v)
                return (path_len + 1) >= self.g
            except nx.NetworkXNoPath:
                return True

        for u in range(self.num_nodes):
            for v in range(u + 1, self.num_nodes):
                if degrees[u] >= self.k or degrees[v] >= self.k:
                    mask.append(False)
                    continue
                if self.graph.has_edge(u, v):
                    mask.append(False)
                    continue
                if not check_girth(u, v):
                    mask.append(False)
                    continue
                mask.append(True)

        return torch.tensor(mask, dtype=torch.bool)

    def _idx_to_edge(self, idx: int) -> tuple[int, int]:
        count = 0
        for u in range(self.num_nodes):
            row_len = self.num_nodes - 1 - u
            if idx < count + row_len:
                v = u + 1 + (idx - count)
                return u, v
            count += row_len
        return 0, 0

    def step(self) -> None:
        if self.start_time == 0:
            import time

            self.start_time = time.time()

        self.step_count += 1

        # Check success
        if is_k_regular(self.graph, self.k):
            girth = compute_girth(self.graph)
            if girth == self.g:
                self.is_complete = True
                self.success = True
                return

        # Get Action
        obs = self._get_obs()
        mask = self._get_valid_mask()

        # If no valid moves
        if not mask.any():
            # Dead end -> Backtrack
            if not self.stack:
                self.is_complete = True  # Failed
                self.success = False
                return

            # Pop last state
            # print("Dead end. Backtracking...")
            # self.graph, last_forbidden = self.stack.pop()
            return

        with torch.no_grad():
            logits, _ = self.model(obs)

        # Apply mask
        logits = logits.masked_fill(~mask, -1e9)

        # Sample or Greedy?
        # Greedy is better for generation usually, but sampling allows diversity
        # Let's do Greedy with some noise?
        # Or just softmax sample
        probs = torch.softmax(logits, dim=0)
        action_idx = torch.multinomial(probs, 1).item()

        u, v = self._idx_to_edge(action_idx)

        # Save state for backtracking (Deep copy is expensive!)
        # Only save every N steps? Or simple recursion?
        # For now, no backtracking implementation to save memory/speed.
        # Just pure RL generation.
        # If we want backtracking, we need to manage the stack.

        self.graph.add_edge(u, v)
