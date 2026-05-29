"""GNN girth predictor for Cayley graphs.

Architecture mirrors ai.cage.voltage.supervised.model.GirthPredictor: GINEConv stacks
over the truncated Cayley ball, pooled, concatenated with a global context
vector (k, g_target, |G|, log|G|, #involutions_in_S), then through a
single regression head (predicted girth).

This model is the Cayley analogue of the voltage girth predictor: it
provides a cheap heuristic that ranks candidate generating-set swaps
during tabu search, with classical verification of the top-K.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast, override

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]
from torch_geometric.nn import global_mean_pool  # pyright: ignore[reportMissingTypeStubs]

from ai.cage.cayley.generators.data_gen import cayley_ball_to_pyg
from ai.cage.cayley.groups import FiniteGroup
from ai.cage.gnn_utils import build_gine_stack
from ai.cage.train_utils import load_predictor_artifacts


class CayleyGirthPredictor(nn.Module):
    """Predicts the girth of a Cayley graph from its truncated ball + context."""

    node_proj: nn.Linear
    edge_proj: nn.Linear
    convs: nn.ModuleList
    bns: nn.ModuleList
    context_proj: nn.Linear
    regress_head: nn.Sequential
    hidden_dim: int
    num_layers: int

    def __init__(
        self,
        node_feat_dim: int = 1,
        edge_feat_dim: int = 2,
        context_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.node_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_feat_dim, hidden_dim)

        self.convs, self.bns = build_gine_stack(hidden_dim, num_layers)

        self.context_proj = nn.Linear(context_dim, hidden_dim)

        self.regress_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @override
    def forward(self, data: Data) -> torch.Tensor:
        x = cast(torch.Tensor, data.x)
        edge_index = cast(torch.Tensor, data.edge_index)
        edge_attr = cast(torch.Tensor, data.edge_attr)

        h = F.relu(self.node_proj(x))
        edge_h = F.relu(self.edge_proj(edge_attr))

        for i in range(self.num_layers):
            h = cast(torch.Tensor, self.convs[i](h, edge_index, edge_h))
            h = cast(torch.Tensor, self.bns[i](h))
            h = F.relu(h)

        batch = (
            data.batch
            if hasattr(data, "batch") and data.batch is not None
            else torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        )
        graph_emb = cast(torch.Tensor, global_mean_pool(h, batch))
        if graph_emb.dim() == 1:
            graph_emb = graph_emb.unsqueeze(0)

        # Build context tensor [batch, 5]
        k_val = cast(torch.Tensor, data.k)
        g_val = cast(torch.Tensor, data.g_target)
        order_val = cast(torch.Tensor, data.group_order)
        log_order_val = cast(torch.Tensor, data.log_group_order)
        invol_val = cast(torch.Tensor, data.num_involutions)

        batch_size = graph_emb.size(0)
        if isinstance(k_val, int):
            ctx = torch.tensor(
                [
                    [
                        float(k_val),
                        float(g_val),
                        float(order_val),
                        float(log_order_val),
                        float(invol_val),
                    ]
                ],
                device=graph_emb.device,
            ).expand(batch_size, -1)
        else:
            ctx = torch.stack(
                [
                    k_val.float(),
                    g_val.float(),
                    order_val.float(),
                    log_order_val.float(),
                    invol_val.float(),
                ],
                dim=-1,
            )

        ctx_emb = F.relu(self.context_proj(ctx))
        combined = torch.cat([graph_emb, ctx_emb], dim=-1)

        girth_pred = cast(torch.Tensor, self.regress_head(combined))
        return girth_pred


def load_cayley_girth_predictor(model_id: str) -> CayleyGirthPredictor:
    """Load a trained Cayley girth predictor from ai/trained/cayley_girth/<model_id>/."""
    model_dir = (
        Path(__file__).resolve().parents[3] / "trained" / "cayley_girth" / str(model_id)
    )
    try:
        info, state = load_predictor_artifacts(model_dir)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Cayley girth predictor {model_id!r} not found at {model_dir}"
        )
    training = cast(dict[str, object], info.get("training", {}))
    model = CayleyGirthPredictor(
        hidden_dim=int(cast(int, training.get("hidden_dim", 64))),
        num_layers=int(cast(int, training.get("num_layers", 4))),
    )
    _ = model.load_state_dict(state)
    _ = model.eval()
    return model


def make_score_fn(model: CayleyGirthPredictor):
    """Wrap a trained model into a ScoreFn for tabu_search.

    Lower score = more promising. We negate the regression output so that
    candidates with higher predicted girth sort to the top.
    """
    device = next(model.parameters()).device
    _ = model.eval()

    def _score(
        group: FiniteGroup,
        generators: list[int],
        k: int,
        g_target: int,
    ) -> float:
        data = cayley_ball_to_pyg(group, generators, k, g_target, girth=0)
        data = data.to(device)
        with torch.no_grad():
            girth_pred = model(data)
        return -float(girth_pred.item())

    return _score


__all__ = [
    "CayleyGirthPredictor",
    "load_cayley_girth_predictor",
    "make_score_fn",
]
