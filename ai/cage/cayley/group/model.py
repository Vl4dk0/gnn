"""GNN group-promise predictor for Cayley-graph cage search.

Given a finite group G and a target (k, g), this model predicts the best
girth achievable by any degree-k symmetric generating set of G — without
running the inner search. It is the genuine ML heuristic for Cayley cage
construction: the training label is expensive (a full classical inner
search per group), the prediction is one cheap forward pass, and at
search time it is used to rank/prune the group catalogue.

Architecture mirrors ai.cage.cayley.generators.model.CayleyGirthPredictor: a stack of
GINEConv layers over the group's Cayley graph, mean-pooled, concatenated
with a context vector, then through a single regression head (best
achievable girth).
"""

from __future__ import annotations

from pathlib import Path
from typing import cast, override

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]
from torch_geometric.nn import GINEConv, global_mean_pool  # pyright: ignore[reportMissingTypeStubs]

from ai.cage.cayley.group.data_gen import group_to_pyg
from ai.cage.cayley.groups import FiniteGroup
from ai.cage.train_utils import load_predictor_artifacts


class GroupPromisePredictor(nn.Module):
    """Predicts the best achievable girth of a group for a (k, g) target."""

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
        node_feat_dim: int = 3,
        edge_feat_dim: int = 2,
        context_dim: int = 6,
        hidden_dim: int = 96,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.node_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_feat_dim, hidden_dim)

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
        graph_emb: torch.Tensor = global_mean_pool(h, batch)
        if graph_emb.dim() == 1:
            graph_emb = graph_emb.unsqueeze(0)

        k_val = cast(torch.Tensor, data.k)
        g_val = cast(torch.Tensor, data.g_target)
        order_val = cast(torch.Tensor, data.group_order)
        log_order_val = cast(torch.Tensor, data.log_group_order)
        invol_val = cast(torch.Tensor, data.num_involutions)
        conj_val = cast(torch.Tensor, data.num_conj)

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
                        float(conj_val),
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
                    conj_val.float(),
                ],
                dim=-1,
            )

        ctx_emb = F.relu(self.context_proj(ctx))
        combined = torch.cat([graph_emb, ctx_emb], dim=-1)

        girth_pred = cast(torch.Tensor, self.regress_head(combined))
        return girth_pred


def load_group_promise_predictor(model_id: str) -> GroupPromisePredictor:
    """Load a trained predictor from ai/trained/group_promise/<model_id>/."""
    model_dir = (
        Path(__file__).resolve().parents[3]
        / "trained"
        / "group_promise"
        / str(model_id)
    )
    try:
        info, state = load_predictor_artifacts(model_dir)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Group-promise predictor {model_id!r} not found at {model_dir}"
        )
    training = cast(dict[str, object], info.get("training", {}))
    model = GroupPromisePredictor(
        hidden_dim=int(cast(int, training.get("hidden_dim", 96))),
        num_layers=int(cast(int, training.get("num_layers", 4))),
    )
    _ = model.load_state_dict(state)
    _ = model.eval()
    return model


class GroupFilter:
    """Protocol: (group, k, g_target) -> predicted best achievable girth."""

    def __call__(self, group: FiniteGroup, k: int, g_target: int) -> float: ...


def make_group_filter(model: GroupPromisePredictor):
    """Wrap a trained model into a callable predicting best achievable girth.

    The returned callable is the search-time heuristic: meta-search scores
    every candidate group with it and only runs the (expensive) classical
    inner search on groups predicted to reach the target.
    """
    device = next(model.parameters()).device
    _ = model.eval()

    def _predict(group: FiniteGroup, k: int, g_target: int) -> float:
        data = group_to_pyg(group, k, g_target, best_girth=0)
        data = data.to(device)
        with torch.no_grad():
            girth_pred = model(data)
        return float(girth_pred.item())

    return _predict


__all__ = [
    "GroupFilter",
    "GroupPromisePredictor",
    "load_group_promise_predictor",
    "make_group_filter",
]
