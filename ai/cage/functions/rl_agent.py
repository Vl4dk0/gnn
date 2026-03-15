import time

import networkx as nx
import torch
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]

from ai.cage.rl.env import CageConstructionEnv
from ai.cage.rl.model import ActorCritic
from ai.registry import get_best_model_id, list_trained_models, load_model
from ai.utils.device import configure_torch_device
from backend.utils.graph_utils import (
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
    is_complete: bool
    success: bool
    start_time: float
    env: CageConstructionEnv
    obs: Data

    def __init__(
        self, k: int, g: int, model_type: str = "gin", model_path: str | None = None
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
        )
        _ = self.model.eval()

        self.step_count = 0
        self.is_complete = False
        self.success = False
        self.start_time = 0.0

    def _load_actor_critic(
        self, model_type: str, model_path: str | None, input_dim: int
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

        model_id: str | None = None
        models = list_trained_models("cage")
        for model_info in models:
            if model_info.get("training", {}).get("model_type") == model_type:
                model_id = str(model_info["model_id"])
                avg_reward = model_info.get("metrics", {}).get("avg_reward", "N/A")
                print(
                    f"Found best {model_type} model: {model_id} (Avg Reward: {avg_reward})"
                )
                break

        if model_id is None:
            model_id = get_best_model_id("cage")
            if model_id:
                print(f"No {model_type} model found. Using best available: {model_id}")

        if model_id:
            try:
                loaded = load_model("cage", model_id)
                if isinstance(loaded, ActorCritic):
                    print(f"Successfully loaded model {model_id} from registry.")
                    return loaded
                print(
                    f"Model {model_id} is not an ActorCritic. Using random initialization."
                )
            except Exception as e:
                print(f"Error loading from registry: {e}. Using random initialization.")
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
        from typing import cast as _cast

        with torch.no_grad():
            logits, _ = _cast(tuple[torch.Tensor, torch.Tensor], self.model(self.obs))

        masked_logits = logits.masked_fill(~mask, -1e9)
        probs = torch.softmax(masked_logits, dim=0)
        return int(torch.multinomial(probs, 1).item())

    def step(self) -> None:
        if self.is_complete:
            return

        if self.start_time == 0:
            self.start_time = time.time()

        mask = self.env.get_valid_action_mask()
        if not bool(mask.any()):
            # Keep session alive for manual stop; action space can recover after future edits.
            self.step_count += 1
            return

        action_idx = self._pick_action(mask)
        next_obs, _, done, info = self.env.step(action_idx)

        self.obs = next_obs
        self.graph = self.env.graph
        self.step_count = self.env.current_step

        if done and bool(info.get("success", False)):
            self.is_complete = True
            self.success = True
