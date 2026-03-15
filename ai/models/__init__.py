"""Unified GNN model definitions.

All models share the same interface and can be used interchangeably
for any node-level prediction task (degree, min_cycle, etc.).

Available models:
- GCN_GNN: Graph Convolutional Network (baseline)
- SAGE_GNN: GraphSAGE with sum aggregation
- GIN_GNN: Graph Isomorphism Network (most expressive for 1-WL)
- Loopy_GNN: r-ℓMPNN for cycle counting (can count cycles up to r+2)
"""

from .base import BaseGNN
from .gcn import GCN_GNN
from .sage import SAGE_GNN
from .gin import GIN_GNN
from .loopy import Loopy_GNN
from .gps import GPS_GNN

# Model registry mapping
MODEL_CLASSES = {
    "gcn": GCN_GNN,
    "sage": SAGE_GNN,
    "gin": GIN_GNN,
    "loopy": Loopy_GNN,
    "gps": GPS_GNN,
}

__all__ = [
    "BaseGNN",
    "GCN_GNN",
    "SAGE_GNN",
    "GIN_GNN",
    "Loopy_GNN",
    "GPS_GNN",
    "MODEL_CLASSES",
]
