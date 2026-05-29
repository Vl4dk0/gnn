"""Shared GNN building blocks for cage models."""

import torch.nn as nn
from torch_geometric.nn import GINEConv  # pyright: ignore[reportMissingTypeStubs]


def build_gine_stack(
    hidden_dim: int, num_layers: int
) -> tuple[nn.ModuleList, nn.ModuleList]:
    """Build a stack of GINEConv + BatchNorm1d layers.

    Returns (convs, bns) each of length num_layers.  Every GINEConv uses a
    two-layer ReLU MLP of width hidden_dim and trains its eps parameter.
    """
    convs: nn.ModuleList = nn.ModuleList()
    bns: nn.ModuleList = nn.ModuleList()
    for _ in range(num_layers):
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        _ = convs.append(GINEConv(mlp, train_eps=True))
        _ = bns.append(nn.BatchNorm1d(hidden_dim))
    return convs, bns
