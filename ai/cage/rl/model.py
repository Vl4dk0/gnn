import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data
from typing import Any

from ai.models import MODEL_CLASSES


class ActorCritic(nn.Module):
    """
    Actor-Critic agent for Cage Generation.

    Uses a GNN as a shared encoder for both policy and value functions.
    """

    gnn: nn.Module
    actor_bilinear: nn.Bilinear
    value_head: nn.Sequential
    config: dict[str, Any]

    def __init__(
        self,
        model_type: str = "gin",
        input_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 3,
        **kwargs: Any,
    ):
        super().__init__()

        # 1. Shared Encoder (GNN)
        if model_type not in MODEL_CLASSES:
            raise ValueError(f"Unknown model type: {model_type}")

        gnn_class = MODEL_CLASSES[model_type]

        # We use the GNN to get node embeddings (output_dim=hidden_dim)
        # We override the output_dim of the base class to be hidden_dim
        # because we don't want a scalar prediction yet
        self.gnn = gnn_class(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,  # Node embeddings
            num_layers=num_layers,
            **kwargs,
        )

        # 2. Actor Head (Policy)
        # Computes compatibility score between two nodes to form an edge
        # We use a simple bilinear layer: score(u, v) = h_u^T * W * h_v
        self.actor_bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)

        # 3. Critic Head (Value)
        # Global pooling -> MLP -> Scalar Value
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

        # Save config for registry
        self.config = {
            "model_type": model_type,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            **kwargs,
        }

    def get_config(self) -> dict[str, Any]:
        """Return model configuration for registry."""
        return self.config

    def get_name(self) -> str:
        """Return model name."""
        return "actor_critic"

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            data: PyG Data object

        Returns:
            logits: Tensor of shape [num_possible_edges] (flat)
            value: Scalar value of the state
        """
        # Get node embeddings [num_nodes, hidden_dim]
        # We need to hack the GNN slightly because BaseGNN.forward usually does a projection at the end
        # But let's see... BaseGNN returns `out.squeeze(-1)` if output_dim=1.
        # If output_dim=hidden, it returns [num_nodes, hidden].
        h = self.gnn(data)

        # --- Critic ---
        # Global pooling (if batch is present, use it, otherwise batch is zeros)
        batch = (
            data.batch
            if hasattr(data, "batch") and data.batch is not None
            else torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        )
        graph_emb = global_mean_pool(h, batch)
        value = self.value_head(graph_emb)

        # --- Actor ---
        # We need to compute scores for all pairs (u, v) where u < v
        # Number of nodes
        num_nodes = h.size(0)

        # Create indices for all upper triangle pairs
        # We can cache this if num_nodes is constant, but it might vary
        # Generating meshgrid

        # Construct pairs efficiently using triu_indices
        # row (u), col (v)
        u_indices, v_indices = torch.triu_indices(
            num_nodes, num_nodes, offset=1, device=h.device
        )

        # Gather embeddings
        h_u = h[u_indices]
        h_v = h[v_indices]

        # Compute pair scores [num_pairs, 1]
        # Bilinear: x1 * A * x2 + b
        logits = self.actor_bilinear(h_u, h_v).squeeze(-1)
        return logits, value

    def get_action(
        self,
        data: Data,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select an action given state.

        Args:
            data: PyG Data object
            action_mask: Boolean tensor (True=valid)
            deterministic: If True, pick max probability

        Returns:
            action: Selected action index
            log_prob: Log probability of selected action
            value: Predicted value
        """
        logits, value = self(data)

        if action_mask is not None:
            # Apply mask (set invalid to -inf)
            # Ensure mask is on same device
            if action_mask.device != logits.device:
                action_mask = action_mask.to(logits.device)

            logits = logits.masked_fill(~action_mask, -1e9)

        # Create distribution
        dist = torch.distributions.Categorical(logits=logits)

        if deterministic:
            action = torch.argmax(logits)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        return int(action.item()), log_prob, value
