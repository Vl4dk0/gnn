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
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from .base import BaseGNN


class GCN_GNN(BaseGNN):
    """
    GCN-based Graph Neural Network.

    Uses GCNConv layers which perform normalized message passing:
    x_i' = W * sum_j (1/sqrt(deg_i * deg_j)) * x_j

    This normalization makes it less suitable for counting tasks
    but provides a good baseline for comparison.
    """

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

        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Output layer (no batch norm after this)
        self.convs.append(GCNConv(hidden_dim, output_dim))

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index

        # Hidden layers with activation, batch norm, dropout
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layer (no activation, no dropout)
        x = self.convs[-1](x, edge_index)

        return x
