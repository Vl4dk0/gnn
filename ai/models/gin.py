"""GIN (Graph Isomorphism Network) model.

Based on Xu et al. (2019): "How Powerful are Graph Neural Networks?"

Characteristics:
- Provably most expressive GNN within the 1-WL framework
- Uses MLP + sum aggregation with learnable epsilon
- Best choice when you need maximum structural discrimination
- Recommended for both degree and cycle tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import cast, override
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]
from torch_geometric.nn import GINConv  # pyright: ignore[reportMissingTypeStubs]

from .base import BaseGNN


class GIN_GNN(BaseGNN):
    """
    GIN-based Graph Neural Network.

    Uses GINConv layers which are provably as powerful as the
    Weisfeiler-Lehman graph isomorphism test:

    x_i' = MLP((1 + eps) * x_i + sum_j x_j)

    The MLP and learnable epsilon make this more expressive than
    simple sum aggregation.
    """

    convs: nn.ModuleList
    bns: nn.ModuleList

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_layers: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__(input_dim, hidden_dim, output_dim, num_layers, dropout)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        if num_layers == 1:
            # Single layer: input directly to output
            _ = self.convs.append(
                self._make_gin_conv(input_dim, output_dim, is_output=True)
            )
        else:
            # First layer
            _ = self.convs.append(self._make_gin_conv(input_dim, hidden_dim))
            _ = self.bns.append(nn.BatchNorm1d(hidden_dim))

            # Hidden layers
            for _ in range(num_layers - 2):
                _ = self.convs.append(self._make_gin_conv(hidden_dim, hidden_dim))
                _ = self.bns.append(nn.BatchNorm1d(hidden_dim))

            # Output layer
            _ = self.convs.append(
                self._make_gin_conv(hidden_dim, output_dim, is_output=True)
            )

    def _make_gin_conv(
        self, in_dim: int, out_dim: int, *, is_output: bool = False
    ) -> GINConv:
        """Create a GINConv layer with 2-layer MLP."""
        if is_output:
            mlp = nn.Sequential(nn.Linear(in_dim, out_dim))
        else:
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            )
        return GINConv(mlp, train_eps=True)

    @override
    def forward(self, data: Data) -> torch.Tensor:
        x = cast(torch.Tensor, data.x)
        edge_index = cast(torch.Tensor, data.edge_index)

        # Hidden layers with activation, batch norm, dropout
        for i in range(self.num_layers - 1):
            x = cast(torch.Tensor, self.convs[i](x, edge_index))
            x = cast(torch.Tensor, self.bns[i](x))
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layer (no activation, no dropout)
        x = cast(torch.Tensor, self.convs[-1](x, edge_index))

        return x
