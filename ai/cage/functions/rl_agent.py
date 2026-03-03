import torch
import networkx as nx
from typing import Any
from torch_geometric.data import Data

from ai.cage.rl.model import ActorCritic
from ai.registry import load_model, get_best_model_id, list_trained_models
from backend.utils.graph_utils import (
    is_k_regular,
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
    max_k_norm: int
    max_g_norm: int
    device: torch.device
    model: ActorCritic
    num_nodes: int
    graph: nx.Graph[int]
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
        self.max_k_norm = 10
        self.max_g_norm = 12

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
            input_dim = self.max_k_norm + 4
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
                        input_dim = self.max_k_norm + 4
                        self.model = ActorCritic(
                            model_type=model_type, input_dim=input_dim, hidden_dim=128
                        )
                    print(f"Successfully loaded model {model_id} from registry.")
                except Exception as e:
                    print(
                        f"Error loading from registry: {e}. Using random initialization."
                    )
                    input_dim = self.max_k_norm + 4
                    self.model = ActorCritic(
                        model_type=model_type, input_dim=input_dim, hidden_dim=128
                    )
            else:
                print(
                    "No trained models found in registry. Using random initialization."
                )
                input_dim = self.max_k_norm + 4
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
        active = self._active_nodes()
        input_dim = self.max_k_norm + 4
        x = torch.zeros((self.num_nodes, input_dim), dtype=torch.float)

        norm_k = self.k / self.max_k_norm
        norm_g = self.g / self.max_g_norm

        for i in range(self.num_nodes):
            d = self.graph.degree(i) if i in active else 0
            # One-hot degree
            idx = min(d, self.max_k_norm - 1)
            x[i, idx] = 1.0 if i in active else 0.0

            # Normalized features
            x[i, self.max_k_norm] = (d / max(1, self.k)) if i in active else 0.0
            x[i, self.max_k_norm + 1] = norm_k
            x[i, self.max_k_norm + 2] = norm_g
            x[i, self.max_k_norm + 3] = 1.0 if i in active else 0.0

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

    def _active_nodes(self) -> set[int]:
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

    def _can_add_by_active_rule(self, u: int, v: int) -> bool:
        active = self._active_nodes()
        if not active:
            # First edge can activate any two vertices.
            return True
        # Only allow active->active or active->inactive.
        return (u in active) or (v in active)

    def _can_add_edge(self, u: int, v: int) -> bool:
        if self.graph.has_edge(u, v):
            return False
        if not self._can_add_by_active_rule(u, v):
            return False
        if self.graph.degree(u) >= self.k or self.graph.degree(v) >= self.k:
            return False
        try:
            path_len = nx.shortest_path_length(self.graph, u, v)
            return (path_len + 1) >= self.g
        except nx.NetworkXNoPath:
            return True

    def _can_remove_edge(self, u: int, v: int) -> bool:
        if not self.graph.has_edge(u, v):
            return False
        self.graph.remove_edge(u, v)
        split = self._active_component_count() > 1
        self.graph.add_edge(u, v)
        return not split

    def _get_valid_mask(self) -> torch.Tensor:
        mask: list[bool] = []

        for u in range(self.num_nodes):
            for v in range(u + 1, self.num_nodes):
                if self.graph.has_edge(u, v):
                    mask.append(self._can_remove_edge(u, v))
                else:
                    mask.append(self._can_add_edge(u, v))

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
        if self._is_k_regular_active() and self._is_girth_satisfied_active():
            self.is_complete = True
            self.success = True
            return

        # Get Action
        obs = self._get_obs()
        mask = self._get_valid_mask()

        # If no valid moves
        if not mask.any():
            # Keep the session alive; user can stop manually.
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
        action_idx = int(torch.multinomial(probs, 1).item())

        u, v = self._idx_to_edge(action_idx)

        if self.graph.has_edge(u, v):
            self.graph.remove_edge(u, v)
        elif self._can_add_edge(u, v):
            self.graph.add_edge(u, v)
