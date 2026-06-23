import time
from typing import cast

import networkx as nx
import torch
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]

from ai.cage.registry.types import CageStepEvent
from ai.cage.rl.env import CageConstructionEnv
from ai.cage.rl.model import ActorCritic
from ai.registry import get_best_model_id, list_trained_models
from ai.utils.device import configure_torch_device
from backend.utils.graph_utils import (
    compute_girth,
    is_k_regular,
    moore_bound,
    moore_hoffman_upper_bound,
)


class RLGenerator:
    """RL-based Cage Graph Generator using shared training environment transitions."""

    k: int
    g: int
    mb: int
    upper_bound: int
    device: torch.device
    model: ActorCritic
    num_nodes: int
    graph: nx.Graph[int]
    step_count: int
    candidates_evaluated: int
    is_complete: bool
    success: bool
    start_time: float
    env: CageConstructionEnv
    obs: Data
    last_event: CageStepEvent | None

    def __init__(
        self,
        k: int,
        g: int,
        model_type: str = "gin",
        model_path: str | None = None,
        model_id: str | None = None,
    ):
        self.k = k
        self.g = g
        self.mb = moore_bound(k, g)
        self.upper_bound = moore_hoffman_upper_bound(k, g)
        self.device = torch.device(configure_torch_device())

        # Keep node capacity fixed for one generation run.
        self.num_nodes = self.mb + 2
        self.env = CageConstructionEnv(
            k=self.k,
            g=self.g,
            n_min=self.num_nodes,
            n_max=self.num_nodes,
            max_steps=1_000_000_000,
            randomize_params=False,
        )
        # Runtime generation should not terminate on score floor.
        self.env.SCORE_FLOOR = -1_000_000_000.0

        self.obs = self.env.reset(num_nodes=self.num_nodes)
        self.graph = self.env.graph

        if self.obs.x is None:
            raise RuntimeError("RL generator observation is missing node features")
        input_dim = int(self.obs.x.size(1))

        self.model = self._load_actor_critic(
            model_type=model_type,
            model_path=model_path,
            input_dim=input_dim,
            model_id=model_id,
        )
        _ = self.model.eval()

        self.step_count = 0
        self.candidates_evaluated = 0
        self.is_complete = False
        self.success = False
        self.start_time = 0.0
        self.last_event = None

    def _load_actor_critic(
        self,
        model_type: str,
        model_path: str | None,
        input_dim: int,
        model_id: str | None,
    ) -> ActorCritic:
        if model_path:
            print(f"Loading RL model from specific path: {model_path}")
            model = ActorCritic(
                model_type=model_type, input_dim=input_dim, hidden_dim=128
            )
            try:
                loaded_weights = torch.load(model_path, map_location=self.device)  # pyright: ignore[reportAny]
                _ = model.load_state_dict(loaded_weights)  # pyright: ignore[reportAny]
            except Exception as e:
                print(
                    f"Failed to load model from {model_path}: {e}. Using random weights."
                )
            return model

        selected_model_id = model_id
        models = list_trained_models("cage")
        selected_model_info = None

        if selected_model_id is not None:
            selected_model_info = next(
                (
                    info
                    for info in models
                    if str(info.get("model_id", "")) == selected_model_id
                ),
                None,
            )
            if selected_model_info is None:
                print(
                    f"Requested RL model '{selected_model_id}' was not found. Falling back."
                )

        if selected_model_info is None:
            for model_info in models:
                if model_info.get("training", {}).get("model_type") == model_type:
                    selected_model_info = model_info
                    selected_model_id = str(model_info["model_id"])
                    avg_reward = model_info.get("metrics", {}).get("avg_reward", "N/A")
                    print(
                        f"Found best {model_type} model: {selected_model_id} (Avg Reward: {avg_reward})"
                    )
                    break

        if selected_model_info is None:
            best_model_id = get_best_model_id("cage")
            if best_model_id:
                selected_model_id = best_model_id
                selected_model_info = next(
                    (
                        info
                        for info in models
                        if str(info.get("model_id", "")) == best_model_id
                    ),
                    None,
                )
                print(
                    f"No {model_type} model found. Using best available: {best_model_id}"
                )

        if selected_model_info is not None and selected_model_id is not None:
            try:
                training = selected_model_info.get("training", {})
                resolved_model_type = str(training.get("model_type", model_type))
                hidden_dim = int(training.get("hidden_dim", 128))
                num_layers = int(training.get("num_layers", 3))
                dropout = float(training.get("dropout", 0.0))
                r = int(training.get("r", 3))
                shared = bool(training.get("shared", False))
                conv_type = str(training.get("conv_type", "gin"))
                heads = int(training.get("heads", 4))
                weights_path = str(selected_model_info["weights_path"])

                model = ActorCritic(
                    model_type=resolved_model_type,
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    r=r,
                    shared=shared,
                    conv_type=conv_type,
                    heads=heads,
                )
                loaded_weights = torch.load(weights_path, map_location=self.device)  # pyright: ignore[reportAny]
                _ = model.load_state_dict(loaded_weights)  # pyright: ignore[reportAny]
                print(f"Successfully loaded RL model {selected_model_id}.")
                return model
            except Exception as e:
                print(
                    f"Error loading RL model {selected_model_id}: {e}. Using random initialization."
                )
        else:
            print("No trained models found in registry. Using random initialization.")

        return ActorCritic(model_type=model_type, input_dim=input_dim, hidden_dim=128)

    def elapsed_time(self) -> float:
        if self.start_time == 0:
            return 0.0
        return time.time() - self.start_time

    def is_regular(self) -> bool:
        return is_k_regular(self.graph, self.k)

    def _pick_action(self, mask: torch.Tensor) -> int:
        with torch.no_grad():
            logits, _ = cast(tuple[torch.Tensor, torch.Tensor], self.model(self.obs))

        masked_logits = logits.masked_fill(~mask, -1e9)
        probs = torch.softmax(masked_logits, dim=0)
        return int(torch.multinomial(probs, 1).item())

    def _active_result_graph(self) -> nx.Graph[int]:
        """Return the connected non-isolated subgraph that the RL env optimizes."""
        active_nodes = [
            node for node in self.env.graph.nodes if self.env.graph.degree(node) > 0
        ]
        node_map = {node: idx for idx, node in enumerate(active_nodes)}
        result: nx.Graph[int] = nx.Graph()
        result.add_nodes_from(range(len(active_nodes)))
        for u, v in self.env.graph.edges(active_nodes):
            _ = result.add_edge(node_map[u], node_map[v])
        return result

    def _is_valid_result(self, graph: nx.Graph[int]) -> bool:
        if graph.number_of_nodes() == 0:
            return False
        if not nx.is_connected(graph):
            return False
        if not is_k_regular(graph, self.k):
            return False
        girth = compute_girth(graph)
        return girth == self.g

    def step(self) -> None:
        if self.is_complete:
            return

        if self.start_time == 0:
            self.start_time = time.time()

        mask = self.env.get_valid_action_mask()
        if not bool(mask.any()):
            # Keep session alive for manual stop; action space can recover after future edits.
            self.step_count += 1
            self.last_event = {
                "step": self.step_count,
                "action": "noop",
                "u": None,
                "v": None,
                "accepted": False,
                "reward": 0.0,
                "done": False,
                "done_reason": "no_valid_action",
            }
            return

        action_idx = self._pick_action(mask)
        next_obs, reward, done, info = self.env.step(action_idx)
        self.candidates_evaluated += 1

        self.obs = next_obs
        self.graph = self.env.graph
        self.step_count = self.env.current_step
        edge_value = info.get("edge", [])
        edge = edge_value if isinstance(edge_value, list) else []
        u = int(edge[0]) if len(edge) == 2 else None
        v = int(edge[1]) if len(edge) == 2 else None
        done_reason_value = info.get("done_reason")
        reward_value = info.get("reward", reward)
        self.last_event = {
            "step": self.step_count,
            "action": str(info.get("action_type", "unknown")),
            "u": u,
            "v": v,
            "accepted": bool(info.get("accepted", False)),
            "reward": float(reward_value)
            if isinstance(reward_value, int | float)
            else reward,
            "done": bool(info.get("done", done)),
            "done_reason": str(done_reason_value)
            if done_reason_value is not None
            else None,
        }

        if done:
            if bool(info.get("success", False)):
                result_graph = self._active_result_graph()
                if self._is_valid_result(result_graph):
                    self.graph = result_graph
                    self.success = True
            self.is_complete = True
