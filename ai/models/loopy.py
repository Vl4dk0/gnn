"""
Loopy GNN (r-ℓMPNN) implementation.
Based on "Weisfeiler and Leman Go Loopy" (NeurIPS 2024).

This model can count cycles up to length r+2, which standard GNNs cannot do.
It achieves this through modified message passing that considers r-neighborhoods
(simple cycles passing through each node).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import scatter

from .base import BaseGNN


class Loopy_GNN(BaseGNN):
    """
    Loopy GNN that can count cycles up to length r+2.

    Uses r-neighborhood message passing instead of standard 1-hop aggregation.
    The r-neighborhoods are computed on-the-fly and passed via the Data object.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension (1 for regression)
        num_layers: Number of Loopy layers
        dropout: Dropout probability
        r: Neighborhood radius (cycles up to r+2 are detected)
        shared: Whether to share weights across path lengths
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_layers: int = 4,
        dropout: float = 0.2,
        r: int = 3,
        shared: bool = True,
    ):
        super().__init__(input_dim, hidden_dim, output_dim, num_layers, dropout)
        self.r = r
        self.shared = shared

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Loopy layers
        self.loopy_layers = nn.ModuleList(
            [
                LoopyLayer(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    r=r,
                    shared=shared,
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, data: Data) -> Tensor:
        x = data.x
        if x is None:
            raise ValueError("Data object must have node features (x)")
        x = x.float()

        # Input projection
        x = self.input_proj(x)

        # Loopy layers
        for layer in self.loopy_layers:
            x = layer(x, data)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Output projection (node-level)
        out = self.output_proj(x)

        return out.squeeze(-1)

    def get_config(self) -> dict:
        config = super().get_config()
        config["r"] = self.r
        config["shared"] = self.shared
        return config


class LoopyLayer(nn.Module):
    """
    Single Loopy layer that processes r-neighborhoods.

    For each path length L (from 0 to r), it:
    1. Gets all paths of length L+2 from loopyN{L}
    2. Processes node embeddings along each path
    3. Aggregates contributions back to center nodes
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        r: int = 3,
        shared: bool = True,
    ):
        super().__init__()
        self.r = r
        self.shared = shared
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Learnable epsilon for self-contribution
        self.eps = nn.Parameter(torch.zeros(1))
        self.r_eps = nn.Parameter(torch.zeros(r + 1))

        # Path processors (one per path length, or shared)
        num_convs = 1 if shared else max(r, 1)
        self.path_convs = nn.ModuleList(
            [
                PathConv(in_channels, out_channels, max_distance=r + 1)
                for _ in range(num_convs)
            ]
        )

        # Final MLP
        self.final_mlp = MLP(in_channels, out_channels, num_layers=2)

    def forward(self, x: Tensor, data: Data) -> Tensor:
        device = x.device
        r_contribution = torch.zeros_like(x)

        for L in range(self.r + 1):
            key = f"loopyN{L}"
            if key not in data:
                continue

            paths = data[key].to(device)  # (L+2, num_paths)

            if paths.shape[1] == 0:
                continue

            atomic_type = data[f"loopyA{L}"].to(device)  # (L+2, num_paths)

            # Get embeddings of nodes on paths (excluding center at index 0)
            path_embeddings = x[paths[1:]]  # (L+1, num_paths, hidden_dim)
            path_atomic = atomic_type[1:]  # (L+1, num_paths)

            # Select conv (shared or per-length)
            if self.shared:
                conv = self.path_convs[0]
            else:
                conv_idx = min(L, len(self.path_convs) - 1)
                conv = self.path_convs[conv_idx]

            if L == 0:
                # Direct neighbors (path length 2, just one intermediate node)
                contribution = path_embeddings.squeeze(0)  # (num_paths, hidden_dim)
            else:
                # Process paths with GIN-style conv
                contribution = conv(
                    path_embeddings, path_atomic
                )  # (num_paths, hidden_dim)

            # Aggregate to center nodes
            center_nodes = paths[0]  # (num_paths,)
            node_contribution = scatter(
                contribution, center_nodes, dim=0, dim_size=x.size(0), reduce="sum"
            )

            r_contribution = r_contribution + (1 + self.r_eps[L]) * node_contribution

        # Final combination
        out = self.final_mlp((1 + self.eps) * x + r_contribution)

        return out


class PathConv(nn.Module):
    """
    GIN-style convolution along paths.

    Uses 1D convolution with kernel [1, 0, 1] to aggregate from neighbors on path.
    This allows each node on the path to receive messages from its adjacent nodes.
    """

    def __init__(self, in_channels: int, out_channels: int, max_distance: int = 4):
        super().__init__()
        self.eps = nn.Parameter(torch.ones(1))

        # Embedding for atomic type (distance from center)
        self.distance_embedding = nn.Embedding(max_distance + 1, in_channels)

        # Transform before conv
        self.pre_transform = nn.Linear(2 * in_channels, in_channels)

        # Final MLP
        self.mlp = MLP(in_channels, out_channels, num_layers=2)

    def forward(self, x: Tensor, atomic_type: Tensor) -> Tensor:
        """
        Process paths with GIN-style convolution.

        Args:
            x: (path_length, num_paths, hidden_dim) - node embeddings along paths
            atomic_type: (path_length, num_paths) - distance from center for each node

        Returns:
            (num_paths, hidden_dim) - aggregated path representation
        """
        # Concatenate with distance embedding
        dist_emb = self.distance_embedding(
            atomic_type
        )  # (path_length, num_paths, hidden_dim)
        x = self.pre_transform(torch.cat([x, dist_emb], dim=-1))

        # Propagate along path using 1D conv with kernel [1, 0, 1]
        x = self._path_propagate(x)

        # Apply MLP and sum over path
        x = self.mlp((1 + self.eps) * x)

        return x.sum(dim=0)  # (num_paths, hidden_dim)

    def _path_propagate(self, x: Tensor) -> Tensor:
        """
        Propagate messages along paths using conv with kernel [1, 0, 1].
        Each node receives messages from its neighbors on the path.

        Args:
            x: (path_length, num_paths, hidden_dim)

        Returns:
            (path_length, num_paths, hidden_dim)
        """
        path_length, num_paths, hidden_dim = x.shape

        # Simple implementation: manually gather from neighbors on path
        # For each position i, aggregate from i-1 and i+1 (with zero padding)
        x_padded = F.pad(x, (0, 0, 0, 0, 1, 1), mode="constant", value=0)
        # x_padded shape: (path_length+2, num_paths, hidden_dim)

        # Sum of left and right neighbors
        left = x_padded[:-2]  # positions 0 to path_length-1
        right = x_padded[2:]  # positions 2 to path_length+1
        aggregated = left + right  # (path_length, num_paths, hidden_dim)

        return aggregated


class MLP(nn.Module):
    """Simple MLP with batch normalization."""

    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 2):
        super().__init__()

        self.num_layers = num_layers
        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_dim = in_channels if i == 0 else out_channels
            self.linears.append(nn.Linear(in_dim, out_channels))
            if i < num_layers - 1:
                self.bns.append(nn.BatchNorm1d(out_channels))

    def reset_parameters(self):
        for linear in self.linears:
            linear.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        # x can be (num_nodes, hidden) or (path_len, num_paths, hidden)
        # BatchNorm1d expects (batch, features) or (batch, features, seq_len)
        original_shape = x.shape

        for i in range(self.num_layers):
            x = self.linears[i](x)
            if i < self.num_layers - 1:
                # Reshape for BatchNorm if needed
                if x.dim() == 3:
                    # (path_len, num_paths, hidden) -> (path_len * num_paths, hidden)
                    path_len, num_paths, hidden = x.shape
                    x = x.reshape(-1, hidden)
                    x = self.bns[i](x)
                    x = x.reshape(path_len, num_paths, hidden)
                else:
                    x = self.bns[i](x)
                x = F.relu(x)

        return x
