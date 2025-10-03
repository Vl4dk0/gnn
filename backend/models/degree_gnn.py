"""Graph Neural Network for vertex degree prediction."""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class DegreeGNN(torch.nn.Module):
    """
    Graph Convolutional Network for predicting vertex degrees.
    
    Architecture:
    - Input: Node features (initialized as degree for self-supervised learning)
    - 2 GCN layers with ReLU activation
    - Output: Predicted degree for each node
    """
    
    def __init__(self, hidden_dim=16, input_dim=4):
        """
        Initialize the GNN model.
        
        Args:
            hidden_dim: Number of hidden units in GCN layers
            input_dim: Number of input features per node
        """
        super(DegreeGNN, self).__init__()
        
        # GCN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)  # Input: multiple features
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, 1)  # Output: 1 value (predicted degree)
        
    def forward(self, x, edge_index):
        """
        Forward pass through the network.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            
        Returns:
            Predicted degrees [num_nodes, 1]
        """
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Output layer
        x = self.conv3(x, edge_index)
        
        return x
