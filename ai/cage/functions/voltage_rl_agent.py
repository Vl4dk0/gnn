import json
import time
from pathlib import Path
from typing import TypedDict

import networkx as nx
import torch
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]

from ai.cage.voltage.lift import build_lift, verify_lift
from ai.cage.voltage.rl_env import VoltageAssignmentEnv
from ai.cage.voltage.rl_model import VoltageActorCritic
from ai.registry import get_trained_dir
from ai.utils.device import configure_torch_device
from backend.utils.graph_utils import is_k_regular, moore_bound


class CageStepEvent(TypedDict):
    step: int
    action: str
    u: int | None
    v: int | None
    accepted: bool
    reward: float
    done: bool
    done_reason: str | None


class VoltageRLGenerator:
    """RL-based cage generator using voltage graph lifts.

    Runs rapid episodes (3-15 steps each) assigning voltages to base graph edges.
    On each failed attempt, resets and tries again with a new base/group config.
    On success, builds the full lifted graph.
    """

    k: int
    g: int
    mb: int
    device: torch.device
    model: VoltageActorCritic
    env: VoltageAssignmentEnv
    obs: Data
    graph: nx.Graph[int]
    step_count: int
    episode_count: int
    is_complete: bool
    success: bool
    start_time: float
    last_event: CageStepEvent | None

    def __init__(
        self,
        k: int,
        g: int,
        model_id: str | None = None,
    ):
        self.k = k
        self.g = g
        self.mb = moore_bound(k, g)
        self.device = torch.device(configure_torch_device())

        self.env = VoltageAssignmentEnv(k=k, g_target=g, randomize=False)
        self.obs = self.env.reset()

        self.model = self._load_model(model_id)
        _ = self.model.eval()

        # Start with an empty graph; will be replaced by lift on success
        self.graph = nx.Graph()
        self.step_count = 0
        self.episode_count = 0
        self.is_complete = False
        self.success = False
        self.start_time = 0.0
        self.last_event = None

    def _load_model(self, model_id: str | None) -> VoltageActorCritic:
        """Load a VoltageActorCritic from the cage model registry."""
        cage_dir = get_trained_dir("cage")
        candidates: list[tuple[str, Path]] = []

        for model_dir in cage_dir.iterdir():
            if not model_dir.is_dir():
                continue
            info_path = model_dir / "info.json"
            weights_path = model_dir / "weights.pt"
            if not info_path.exists() or not weights_path.exists():
                continue
            with open(info_path) as f:
                info = json.load(f)
            if info.get("model_type") == "voltage_actor_critic":
                candidates.append((model_dir.name, weights_path))

        if not candidates:
            if model_id is not None:
                raise FileNotFoundError(
                    f"voltage_rl model_id={model_id!r} requested but no "
                    "voltage_actor_critic models exist in the cage registry"
                )
            print("No voltage RL models found. Using random weights.")
            return VoltageActorCritic(
                input_dim=self.env.get_input_dim(), edge_feat_dim=2
            )

        # Pick by model_id if specified, otherwise first available
        selected_path: Path | None = None
        selected_id: str = ""
        if model_id is not None:
            for cid, cpath in candidates:
                if cid == model_id:
                    selected_path = cpath
                    selected_id = cid
                    break
            if selected_path is None:
                available = ", ".join(cid for cid, _ in candidates)
                raise FileNotFoundError(
                    f"voltage_rl model_id={model_id!r} not found in cage "
                    f"registry. Available: [{available}]"
                )

        if selected_path is None:
            selected_id, selected_path = candidates[0]

        # Load training config from info.json
        info_path = selected_path.parent / "info.json"
        with open(info_path) as f:
            info = json.load(f)
        training = info.get("training", {})

        model = VoltageActorCritic(
            input_dim=int(training.get("input_dim", self.env.get_input_dim())),
            edge_feat_dim=2,
            hidden_dim=int(training.get("hidden_dim", 64)),
            num_layers=int(training.get("num_layers", 3)),
            dropout=float(training.get("dropout", 0.1)),
            max_action_dim=int(training.get("max_action_dim", 100)),
            conv_type=str(training.get("conv_type", "gine")),
            heads=int(training.get("heads", 4)),
        )

        weights = torch.load(selected_path, map_location=self.device)  # pyright: ignore[reportAny]
        _ = model.load_state_dict(weights)  # pyright: ignore[reportAny]
        print(f"Loaded voltage RL model: {selected_id}")
        return model

    def elapsed_time(self) -> float:
        if self.start_time == 0:
            return 0.0
        return time.time() - self.start_time

    def is_regular(self) -> bool:
        return is_k_regular(self.graph, self.k)

    def step(self) -> None:
        if self.is_complete:
            return

        if self.start_time == 0:
            self.start_time = time.time()

        self.step_count += 1

        with torch.no_grad():
            action, _, _ = self.model.get_action(
                self.obs, action_dim=self.env.get_action_dim(), deterministic=False
            )

        next_obs, reward, done, info = self.env.step(action)
        self.obs = next_obs

        # Record event
        edge_idx = int(info.get("edge_idx", 0))
        self.last_event = {
            "step": self.step_count,
            "action": f"voltage={action} edge={edge_idx}",
            "u": None,
            "v": None,
            "accepted": True,
            "reward": float(reward),
            "done": done,
            "done_reason": str(info.get("done_reason", "")) if done else None,
        }

        if done:
            self.episode_count += 1
            girth = int(info.get("girth", 0))

            if bool(info.get("success", False)) and girth >= self.g:
                # Build the full lifted graph
                lifted = build_lift(self.env.base, self.env.group, self.env.voltages)
                props = verify_lift(lifted, self.k, self.g)

                if props["is_valid_kg"]:
                    self.graph = lifted
                    self.is_complete = True
                    self.success = True
                    self.last_event["done_reason"] = (
                        f"success: girth={girth}, order={lifted.number_of_nodes()}, "
                        f"group={self.env.group.name}, voltages={self.env.voltages}"
                    )
                    return

            # Failed attempt — reset for next episode
            self.obs = self.env.reset()
            self.last_event["done_reason"] = (
                f"attempt {self.episode_count} failed (girth={girth}), retrying"
            )
