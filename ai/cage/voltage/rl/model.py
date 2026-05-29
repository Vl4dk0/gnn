"""Actor-Critic model for voltage assignment RL.

The actor outputs a distribution over group elements (the next voltage to
assign).  The critic estimates the value of the current partial assignment.

Unlike the edge-by-edge ActorCritic which uses bilinear scoring over all
node pairs, this model uses a simple MLP head over the global graph
embedding — because the action space is fixed-size (group_order).
"""

from __future__ import annotations

from typing import cast, override

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]
from torch_geometric.nn import GINEConv, GPSConv, global_mean_pool  # pyright: ignore[reportMissingTypeStubs]


class VoltageActorCritic(nn.Module):
    """Actor-Critic for voltage assignment.

    Encoder: GINEConv (GIN + edge features) on the base graph.
    Actor:   global_pool -> MLP -> logits over group elements.
    Critic:  global_pool -> MLP -> scalar value.
    """

    node_proj: nn.Linear
    edge_proj: nn.Linear
    convs: nn.ModuleList
    bns: nn.ModuleList
    actor_head: nn.Sequential
    value_head: nn.Sequential
    hidden_dim: int
    max_action_dim: int

    conv_type: str

    def __init__(
        self,
        input_dim: int = 6,
        edge_feat_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        max_action_dim: int = 100,
        conv_type: str = "gine",
        heads: int = 4,
    ):
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.hidden_dim = hidden_dim
        self.max_action_dim = max_action_dim
        self.conv_type = conv_type

        # Input projections
        self.node_proj = nn.Linear(input_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_feat_dim, hidden_dim)

        # Conv layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if conv_type == "gps":
                gine = GINEConv(mlp, train_eps=True)
                _ = self.convs.append(
                    GPSConv(hidden_dim, gine, heads=heads, dropout=dropout)
                )
            else:
                _ = self.convs.append(GINEConv(mlp, train_eps=True))
            _ = self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Actor head: graph embedding -> action logits
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_action_dim),
        )

        # Critic head: graph embedding -> scalar value
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _encode(self, data: Data) -> torch.Tensor:
        """Encode base graph with partial voltages into a graph embedding."""
        x = cast(torch.Tensor, data.x)
        edge_index = cast(torch.Tensor, data.edge_index)
        edge_attr = cast(torch.Tensor, data.edge_attr)

        h = F.relu(self.node_proj(x))
        edge_h = self.edge_proj(edge_attr)

        for i in range(len(self.convs)):
            if self.conv_type == "gps":
                h = cast(torch.Tensor, self.convs[i](h, edge_index, edge_attr=edge_h))
            else:
                h = cast(torch.Tensor, self.convs[i](h, edge_index, edge_h))
            h = cast(torch.Tensor, self.bns[i](h))
            h = F.relu(h)

        # Global pooling
        batch = (
            data.batch
            if hasattr(data, "batch") and data.batch is not None
            else torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        )
        graph_emb = cast(torch.Tensor, global_mean_pool(h, batch))
        return graph_emb

    @override
    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            logits: [batch_size, max_action_dim] action logits
            value:  [batch_size, 1] state value
        """
        graph_emb = self._encode(data)
        logits = cast(torch.Tensor, self.actor_head(graph_emb))
        value = cast(torch.Tensor, self.value_head(graph_emb))
        return logits, value

    def get_action(
        self,
        data: Data,
        action_dim: int | None = None,
        deterministic: bool = False,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Select an action (group element) given state.

        Args:
            data: PyG Data observation.
            action_dim: Actual group order (mask logits beyond this).
            deterministic: If True, pick argmax.

        Returns:
            (action, log_prob, value)
        """
        _out = cast(tuple[torch.Tensor, torch.Tensor], self(data))
        logits, value = _out

        # Mask invalid actions (beyond group order)
        if action_dim is not None and action_dim < self.max_action_dim:
            mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
            mask[:action_dim] = False
            logits = logits.masked_fill(mask, -1e9)

        dist = torch.distributions.Categorical(logits=logits)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()

        log_prob = cast(torch.Tensor, dist.log_prob(action))
        return int(action.item()), log_prob, value

    def get_config(self) -> dict[str, str | int | float | bool]:
        return {
            "hidden_dim": self.hidden_dim,
            "max_action_dim": self.max_action_dim,
        }

    def get_name(self) -> str:
        return "voltage_actor_critic"
