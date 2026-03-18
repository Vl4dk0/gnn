"""GPS (General, Powerful, Scalable) Graph Transformer model.

Based on Rampášek et al. (2022): "Recipe for a General, Powerful, Scalable
Graph Transformer."

Each GPS layer combines a local GNN convolution with global multi-head
self-attention, enabling long-range information flow while retaining
structural awareness from message passing.

Architecture per layer:
    local_conv(x, edges) + global_attention(x) → normalize → feedforward
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import cast, override

from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]
from torch_geometric.nn import (  # pyright: ignore[reportMissingTypeStubs]
    GCNConv,
    GINConv,
    GPSConv,
    SAGEConv,
)

from .base import BaseGNN


class GPS_GNN(BaseGNN):
    """
    GPS Graph Transformer wrapping a local GNN conv with global attention.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension (must be divisible by heads)
        output_dim: Output dimension (1 for regression)
        num_layers: Number of GPS layers
        dropout: Dropout probability
        conv_type: Inner conv type ("gcn", "sage", or "gin")
        heads: Number of attention heads
    """

    conv_type: str
    heads: int
    input_proj: nn.Linear
    input_bn: nn.BatchNorm1d
    gps_layers: nn.ModuleList
    output_proj: nn.Linear

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_layers: int = 4,
        dropout: float = 0.2,
        conv_type: str = "gin",
        heads: int = 4,
    ):
        super().__init__(input_dim, hidden_dim, output_dim, num_layers, dropout)
        assert hidden_dim % heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})"
        )
        self.conv_type = conv_type
        self.heads = heads

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)

        # GPS layers
        self.gps_layers = nn.ModuleList()
        for _ in range(num_layers):
            inner_conv = self._make_inner_conv(hidden_dim)
            layer = GPSConv(
                channels=hidden_dim,
                conv=inner_conv,
                heads=heads,
                dropout=dropout,
                act="relu",
            )
            _ = self.gps_layers.append(layer)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def _make_inner_conv(self, dim: int) -> GCNConv | SAGEConv | GINConv:
        """Create the inner local conv based on conv_type."""
        if self.conv_type == "gcn":
            return GCNConv(dim, dim)
        if self.conv_type == "sage":
            return SAGEConv(dim, dim, aggr="add")

        # default: gin
        mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        return GINConv(mlp, train_eps=True)

    @override
    def forward(self, data: Data) -> torch.Tensor:
        x = cast(torch.Tensor, data.x)
        edge_index = cast(torch.Tensor, data.edge_index)
        batch: torch.Tensor | None = data.batch

        # Input projection
        x = cast(torch.Tensor, self.input_proj(x))
        x = cast(torch.Tensor, self.input_bn(x))
        x = F.relu(x)

        # GPS layers
        for layer in self.gps_layers:
            x = cast(torch.Tensor, layer(x, edge_index, batch=batch))

        # Output projection
        x = cast(torch.Tensor, self.output_proj(x))
        return x

    @override
    def get_config(self) -> dict[str, str | int | float | bool]:
        config = super().get_config()
        config["conv_type"] = self.conv_type
        config["heads"] = self.heads
        return config
