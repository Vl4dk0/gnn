"""Base GNN class with common interface for all models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional

from torch_geometric.data import Data


class BaseGNN(nn.Module, ABC):
    """
    Abstract base class for all GNN models.

    All models must implement the same forward signature:
        forward(data) -> node predictions

    This ensures any model can be used for any task (degree, min_cycle, etc.)
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_layers: int = 4,
        dropout: float = 0.2,
    ):
        """
        Args:
            input_dim: Input feature dimension (default 1 for constant features)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (default 1 for scalar prediction)
            num_layers: Number of GNN layers
            dropout: Dropout probability
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

    @abstractmethod
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass for node-level prediction.

        Args:
            data: PyTorch Geometric Data object containing:
                - x: Node features [num_nodes, input_dim]
                - edge_index: Edge connectivity [2, num_edges]
                - (optional) Additional attributes for specialized models

        Returns:
            Node predictions [num_nodes, output_dim]
        """
        pass

    @classmethod
    def get_name(cls) -> str:
        """Return the model type name for registry."""
        return cls.__name__.replace("_GNN", "").lower()

    def get_config(self) -> dict:
        """Return model configuration for saving."""
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }

    @classmethod
    def from_config(cls, config: dict) -> "BaseGNN":
        """Create model instance from configuration."""
        return cls(**config)
