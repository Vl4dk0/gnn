"""GNN girth predictor for voltage-labeled base graphs.

Uses GINEConv (GIN with edge features) to process base graphs where
edge attributes encode voltage labels from a finite group.  Predicts
the girth of the corresponding voltage graph lift.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast, override

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]
from torch_geometric.nn import global_mean_pool  # pyright: ignore[reportMissingTypeStubs]

from ai.cage.gnn_utils import build_gine_stack
from ai.cage.train_utils import load_predictor_artifacts
from ai.utils.structural_features import structural_feature_dim


class GirthPredictor(nn.Module):
    """Predicts the girth of a voltage graph lift from the labeled base graph.

    Architecture:
        1. Node feature projection
        2. Edge voltage embedding (learned)
        3. Several GINEConv layers (GIN + edge features)
        4. Global mean pooling
        5. Concatenate with global context (k, g_target, group_order)
        6. MLP regression head producing girth prediction
    """

    node_proj: nn.Linear
    edge_embed: nn.Embedding
    edge_proj: nn.Linear
    convs: nn.ModuleList
    bns: nn.ModuleList
    context_proj: nn.Linear
    regress_head: nn.Sequential
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
        self.convs, self.bns = build_gine_stack(hidden_dim, num_layers)

        # Context projection (k, g_target, group_order -> hidden_dim)
        self.context_proj = nn.Linear(3, hidden_dim)

        # Regression head: predict girth as a positive real
        self.regress_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @override
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass.

        Args:
            data: PyG Data with x, edge_index, edge_attr (voltage indices),
                  and global attributes k, g_target, group_order.

        Returns:
            girth_pred — shape [batch_size, 1].
        """
        x = cast(torch.Tensor, data.x)
        edge_index = cast(torch.Tensor, data.edge_index)
        edge_attr = cast(torch.Tensor, data.edge_attr)

        # Project node features
        h = F.relu(self.node_proj(x))

        # Embed edge voltages
        # edge_attr is [num_edges, 1] with integer voltage indices
        voltage_idx = edge_attr.squeeze(-1).long()
        if voltage_idx.max().item() >= self.max_group_order:
            raise ValueError(
                f"Voltage index {voltage_idx.max().item()} exceeds max_group_order={self.max_group_order}"
            )
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

        return cast(torch.Tensor, self.regress_head(combined))


def load_girth_predictor(
    model_id: str,
    *,
    cycle_lengths: list[int] | None = None,
    rwpe_dim: int = 0,
) -> tuple["GirthPredictor", list[int] | None, int]:
    """Load a trained girth predictor from `ai/trained/voltage_girth/<model_id>/`.

    Reads the architecture parameters from the saved info.json and the weights
    from weights.pt. Raises FileNotFoundError if either is missing.

    If cycle_lengths or rwpe_dim are provided they must match the feature_config
    saved in info.json (if any). A mismatch raises ValueError so callers cannot
    silently run inference with wrong features.

    Returns (model, saved_cycle_lengths, saved_rwpe_dim). Callers must apply
    add_structural_features with the returned config before scoring any graph.
    """
    model_dir = (
        Path(__file__).resolve().parents[3]
        / "trained"
        / "voltage_girth"
        / str(model_id)
    )
    try:
        info, state = load_predictor_artifacts(model_dir)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Girth predictor {model_id!r} not found at {model_dir}"
        )
    training = cast(dict[str, object], info.get("training", {}))

    # Read saved feature_config (absent for legacy models trained without features)
    saved_cfg = cast(dict[str, object], training.get("feature_config", {}))
    saved_cycle_lengths_raw = saved_cfg.get("cycle_lengths", None)
    saved_cycle_lengths: list[int] | None = (
        [int(cast(int, v)) for v in cast(list[object], saved_cycle_lengths_raw)]
        if saved_cycle_lengths_raw is not None
        else None
    )
    saved_rwpe_dim = int(cast(int, saved_cfg.get("rwpe_dim", 0)))

    # If caller supplied explicit feature config, check it matches the saved one
    caller_supplied = cycle_lengths is not None or rwpe_dim != 0
    if caller_supplied:
        if (cycle_lengths or []) != (
            saved_cycle_lengths or []
        ) or rwpe_dim != saved_rwpe_dim:
            raise ValueError(
                f"Feature config mismatch for model {model_id!r}. "
                f"Saved: cycle_lengths={saved_cycle_lengths}, rwpe_dim={saved_rwpe_dim}. "
                f"Requested: cycle_lengths={cycle_lengths}, rwpe_dim={rwpe_dim}."
            )

    # Compute the base node_feat_dim (2 for legacy models) plus structural extras
    base_node_feat_dim = int(cast(int, training.get("node_feat_dim", 2)))
    extra_dim = structural_feature_dim(
        cycle_lengths=saved_cycle_lengths, rwpe_dim=saved_rwpe_dim
    )
    node_feat_dim = base_node_feat_dim + extra_dim

    model = GirthPredictor(
        node_feat_dim=node_feat_dim,
        hidden_dim=int(cast(int, training.get("hidden_dim", 64))),
        num_layers=int(cast(int, training.get("num_layers", 4))),
        max_group_order=int(cast(int, training.get("max_group_order", 60))),
    )
    _ = model.load_state_dict(state)
    _ = model.eval()
    return model, saved_cycle_lengths, saved_rwpe_dim
