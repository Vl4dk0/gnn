"""GraphSAGE model with sum aggregation.

Based on Hamilton et al. (2017): "Inductive Representation Learning on Large Graphs"

Characteristics:
- Uses sum aggregation (preserves neighbor count information)
- Good for degree prediction (current best for that task)
- More expressive than GCN for counting tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from .base import BaseGNN


class SAGE_GNN(BaseGNN):
    """
    GraphSAGE-based Graph Neural Network with sum aggregation.

    Uses SAGEConv layers with 'add' aggregation which sums neighbor features:
    x_i' = W1 * x_i + W2 * sum_j x_j

    The sum aggregation preserves information about the number of neighbors,
    making it suitable for degree-related predictions.
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
        self.convs.append(SAGEConv(input_dim, hidden_dim, aggr="add"))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr="add"))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Output layer
        self.convs.append(SAGEConv(hidden_dim, output_dim, aggr="add"))

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
