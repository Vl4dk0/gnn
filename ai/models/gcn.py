"""GCN (Graph Convolutional Network) model.

Baseline model using GCNConv layers.
Based on Kipf & Welling (2017): "Semi-Supervised Classification with Graph Convolutional Networks"

Characteristics:
- Uses symmetric normalization (degree-weighted averaging)
- Good baseline, but normalizes away neighbor counts
- Less expressive than GIN for structural tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import cast, override
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]
from torch_geometric.nn import GCNConv  # pyright: ignore[reportMissingTypeStubs]

from .base import BaseGNN


class GCN_GNN(BaseGNN):
    """
    GCN-based Graph Neural Network.

    Uses GCNConv layers which perform normalized message passing:
    x_i' = W * sum_j (1/sqrt(deg_i * deg_j)) * x_j

    This normalization makes it less suitable for counting tasks
    but provides a good baseline for comparison.
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
            _ = self.convs.append(GCNConv(input_dim, output_dim))
        else:
            # First layer
            _ = self.convs.append(GCNConv(input_dim, hidden_dim))
            _ = self.bns.append(nn.BatchNorm1d(hidden_dim))

            # Hidden layers
            for _ in range(num_layers - 2):
                _ = self.convs.append(GCNConv(hidden_dim, hidden_dim))
                _ = self.bns.append(nn.BatchNorm1d(hidden_dim))

            # Output layer (no batch norm after this)
            _ = self.convs.append(GCNConv(hidden_dim, output_dim))

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
