"""GNN girth predictor for voltage-labeled base graphs.

Uses GINEConv (GIN with edge features) to process base graphs where
edge attributes encode voltage labels from a finite group.  Predicts
the girth of the corresponding voltage graph lift.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast, override

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]
from torch_geometric.nn import GINEConv, global_mean_pool  # pyright: ignore[reportMissingTypeStubs]


class GirthPredictor(nn.Module):
    """Predicts the girth of a voltage graph lift from the labeled base graph.

    Architecture:
        1. Node feature projection
        2. Edge voltage embedding (learned)
        3. Several GINEConv layers (GIN + edge features)
        4. Global mean pooling
        5. Concatenate with global context (k, g_target, group_order)
        6. MLP head producing girth regression + binary classification
    """

    node_proj: nn.Linear
    edge_embed: nn.Embedding
    edge_proj: nn.Linear
    convs: nn.ModuleList
    bns: nn.ModuleList
    context_proj: nn.Linear
    regress_head: nn.Sequential
    classify_head: nn.Sequential
    hidden_dim: int
    num_layers: int
    max_group_order: int

    def __init__(
        self,
        node_feat_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_group_order: int = 200,
    ):
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_group_order = max_group_order

        # Input projections
        self.node_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_embed = nn.Embedding(max_group_order, hidden_dim)
        self.edge_proj = nn.Linear(hidden_dim, hidden_dim)

        # GINEConv layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            _ = self.convs.append(GINEConv(mlp, train_eps=True))
            _ = self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Context projection (k, g_target, group_order -> hidden_dim)
        self.context_proj = nn.Linear(3, hidden_dim)

        # Regression head: predict girth as a positive real
        self.regress_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Classification head: predict P(girth >= g_target)
        self.classify_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @override
    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            data: PyG Data with x, edge_index, edge_attr (voltage indices),
                  and global attributes k, g_target, group_order.

        Returns:
            (girth_pred, girth_class_logit) — both shape [batch_size, 1].
        """
        x = cast(torch.Tensor, data.x)
        edge_index = cast(torch.Tensor, data.edge_index)
        edge_attr = cast(torch.Tensor, data.edge_attr)

        # Project node features
        h = F.relu(self.node_proj(x))

        # Embed edge voltages
        # edge_attr is [num_edges, 1] with integer voltage indices
        voltage_idx = edge_attr.squeeze(-1).long()
        # Clamp to valid range
        voltage_idx = voltage_idx.clamp(0, self.max_group_order - 1)
        edge_h = self.edge_proj(self.edge_embed(voltage_idx))

        # Message passing
        for i in range(self.num_layers):
            h = cast(torch.Tensor, self.convs[i](h, edge_index, edge_h))
            h = cast(torch.Tensor, self.bns[i](h))
            h = F.relu(h)

        # Global pooling
        batch = (
            data.batch
            if hasattr(data, "batch") and data.batch is not None
            else torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        )
        graph_emb = cast(torch.Tensor, global_mean_pool(h, batch))

        # Context features
        k_val = cast(torch.Tensor, data.k)
        g_val = cast(torch.Tensor, data.g_target)
        group_order_val = cast(torch.Tensor, data.group_order)

        # Build context tensor [batch_size, 3]
        if graph_emb.dim() == 1:
            graph_emb = graph_emb.unsqueeze(0)

        batch_size = graph_emb.size(0)

        # Handle both batched and single-graph cases
        if isinstance(k_val, int):
            ctx = torch.tensor(
                [[float(k_val), float(g_val), float(group_order_val)]],
                device=graph_emb.device,
            ).expand(batch_size, -1)
        else:
            ctx = torch.stack(
                [k_val.float(), g_val.float(), group_order_val.float()], dim=-1
            )

        ctx_emb = F.relu(self.context_proj(ctx))

        # Combine graph embedding with context
        combined = torch.cat([graph_emb, ctx_emb], dim=-1)

        girth_pred = cast(torch.Tensor, self.regress_head(combined))
        girth_logit = cast(torch.Tensor, self.classify_head(combined))

        return girth_pred, girth_logit


def load_girth_predictor(model_id: str) -> GirthPredictor:
    """Load a trained girth predictor from `ai/trained/voltage_girth/<model_id>/`.

    Reads the architecture parameters from the saved info.json and the weights
    from weights.pt. Raises FileNotFoundError if either is missing.
    """
    model_dir = (
        Path(__file__).resolve().parents[2]
        / "trained"
        / "voltage_girth"
        / str(model_id)
    )
    info_path = model_dir / "info.json"
    weights_path = model_dir / "weights.pt"
    if not info_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"Girth predictor {model_id!r} not found at {model_dir}"
        )
    with open(info_path) as f:
        info = cast(dict[str, object], json.load(f))
    training = cast(dict[str, object], info.get("training", {}))
    model = GirthPredictor(
        node_feat_dim=2,
        hidden_dim=int(cast(int, training.get("hidden_dim", 64))),
        num_layers=int(cast(int, training.get("num_layers", 4))),
        max_group_order=int(cast(int, training.get("max_group_order", 60))),
    )
    state = torch.load(weights_path, map_location="cpu")  # pyright: ignore[reportAny]
    _ = model.load_state_dict(state)  # pyright: ignore[reportAny]
    _ = model.eval()
    return model
