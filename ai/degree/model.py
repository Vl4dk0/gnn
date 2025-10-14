"""Graph Neural Network for vertex degree prediction - Optimized."""
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class DegreeGNN(torch.nn.Module):
    """
    GNN optimized for degree prediction using SAGEConv with 'add' aggregation.

    Strategy: Use sum aggregation to count neighbors effectively.
    """

    def __init__(self, hidden_dim=128, input_dim=1):
        """
        Args:
            hidden_dim: Hidden layer size
            input_dim: Input feature size (should be 1 for constant features)
        """
        super(DegreeGNN, self).__init__()

        # Use SAGEConv with 'add' aggregation for counting
        self.conv1 = SAGEConv(input_dim, hidden_dim, aggr='add')
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr='add')
        self.conv3 = SAGEConv(hidden_dim, hidden_dim, aggr='add')
        self.conv4 = SAGEConv(hidden_dim, 1, aggr='add')

        # Batch normalization for stable training
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]

        Returns:
            Predicted degrees [num_nodes, 1]
        """
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # Layer 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # Output layer
        x = self.conv4(x, edge_index)

        return x
