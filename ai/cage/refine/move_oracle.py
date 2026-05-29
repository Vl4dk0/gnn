"""GNN move scorer (MoveOracle) for 2-switch Δcost prediction.

Architecture (GINE-style, following GirthPredictor in ai/cage/voltage/supervised/model.py):
  1. Node feature projection (structural features from add_structural_features)
  2. Several GINEConv layers with BatchNorm + ReLU
  3. Global mean pooling -> graph embedding
  4. Per-swap head: gather node embeddings for the 4 swap endpoints,
     concat with graph embedding, MLP -> scalar predicting Δcost.

The model is (k,g)-independent: k, g are never baked into weights.

Structural features (cycle counts + RWPE) are mandatory to break the
symmetry barrier on k-regular high-girth graphs (Chen et al. NeurIPS 2023).
Callers must apply add_structural_features before passing data to the oracle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, cast, override

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]
from torch_geometric.nn import GINEConv, global_mean_pool  # pyright: ignore[reportMissingTypeStubs]

from ai.cage.refine.swaps import Swap2
from ai.utils.structural_features import add_structural_features, structural_feature_dim


class MoveOracle(nn.Module):
    """Predicts Δcost for a 2-switch move on a k-regular graph.

    For each candidate swap (u, v, x, y):
      - Gather node embeddings h_u, h_v, h_x, h_y from the GNN backbone.
      - Concatenate with global graph embedding.
      - MLP head -> scalar Δcost prediction (lower = more promising).

    Parameters
    ----------
    node_feat_dim:
        Dimension of input node features (including structural features).
    hidden_dim:
        Width of hidden layers.
    num_layers:
        Number of GINEConv message-passing layers.
    """

    node_proj: nn.Linear
    edge_proj: nn.Linear
    convs: nn.ModuleList
    bns: nn.ModuleList
    swap_head: nn.Sequential
    hidden_dim: int
    num_layers: int

    def __init__(
        self,
        node_feat_dim: int = 15,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Project node features to hidden dim
        self.node_proj = nn.Linear(node_feat_dim, hidden_dim)

        # Edge features: we use a single constant (all-ones) edge feature
        # projected to hidden_dim, as the graph is unweighted.
        self.edge_proj = nn.Linear(1, hidden_dim)

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

        # Swap head: 4 node embeddings + global embedding -> scalar
        # Input size: 4 * hidden_dim + hidden_dim = 5 * hidden_dim
        self.swap_head = nn.Sequential(
            nn.Linear(5 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    @override
    def forward(
        self,
        data: Data,
        swap_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        data:
            PyG Data with x (node features including structural features),
            edge_index, and num_nodes.  edge_attr is optional (ignored;
            we use constant edge features).
        swap_indices:
            LongTensor of shape [num_swaps, 4] with columns [u, v, x, y]
            (the four node indices for each candidate 2-switch).

        Returns
        -------
        Tensor of shape [num_swaps] with predicted Δcost per swap.
        """
        x = cast(torch.Tensor, data.x)
        edge_index = cast(torch.Tensor, data.edge_index)
        num_nodes = int(cast(int, data.num_nodes))

        # Project node features
        h = F.relu(self.node_proj(x.float()))

        # Constant edge features (all ones), projected to hidden_dim
        num_edges = edge_index.size(1)
        edge_h = self.edge_proj(
            torch.ones(num_edges, 1, device=x.device, dtype=torch.float)
        )

        # Message passing
        for i in range(self.num_layers):
            h = cast(torch.Tensor, self.convs[i](h, edge_index, edge_h))
            if num_nodes > 1:
                h = cast(torch.Tensor, self.bns[i](h))
            h = F.relu(h)

        # Global pooling
        batch = (
            data.batch
            if hasattr(data, "batch") and data.batch is not None
            else torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        )
        graph_emb = cast(torch.Tensor, global_mean_pool(h, batch))
        # graph_emb: [1, hidden_dim] for single graph

        # Per-swap scoring
        # swap_indices: [num_swaps, 4]  columns: u, v, x, y
        num_swaps = swap_indices.size(0)
        if num_swaps == 0:
            return torch.zeros(0, device=x.device)

        h_u = h[swap_indices[:, 0]]  # [num_swaps, hidden_dim]
        h_v = h[swap_indices[:, 1]]
        h_x = h[swap_indices[:, 2]]
        h_y = h[swap_indices[:, 3]]

        # Expand graph embedding to match num_swaps
        graph_expanded = graph_emb.expand(num_swaps, -1)

        swap_feat = torch.cat([h_u, h_v, h_x, h_y, graph_expanded], dim=-1)
        delta_pred = self.swap_head(swap_feat).squeeze(-1)  # [num_swaps]

        return delta_pred


def graph_to_pyg_data(
    G: nx.Graph[int],
    cycle_lengths: list[int],
    rwpe_dim: int,
) -> Data:
    """Convert a NetworkX graph to a PyG Data object with structural features.

    Applies add_structural_features with the given config so that the
    MoveOracle receives properly enriched node features.
    """
    nodes = sorted(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    edges = list(G.edges())
    if edges:
        src = [node_idx[u] for u, _ in edges] + [node_idx[v] for _, v in edges]
        dst = [node_idx[v] for _, v in edges] + [node_idx[u] for u, _ in edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    data = Data(
        x=torch.ones((n, 1), dtype=torch.float),
        edge_index=edge_index,
        num_nodes=n,
    )
    data = add_structural_features(
        data,
        cycle_lengths=cycle_lengths,
        rwpe_dim=rwpe_dim,
    )
    return data


def build_score_fn(
    model: MoveOracle,
    cycle_lengths: list[int],
    rwpe_dim: int,
    device: torch.device | str = "cpu",
) -> Callable[[nx.Graph[int], list[Swap2]], torch.Tensor]:
    """Build a score_fn compatible with TabuRefiner from a trained MoveOracle.

    The returned callable takes (G, swaps) and returns a Tensor of predicted
    Δcost values (lower = more promising).
    """

    def score_fn(
        G: nx.Graph[int],
        swaps: list[Swap2],
    ) -> torch.Tensor:
        nodes = sorted(G.nodes())
        node_idx = {n: i for i, n in enumerate(nodes)}
        data = graph_to_pyg_data(G, cycle_lengths, rwpe_dim).to(device)
        swap_idx = torch.tensor(
            [
                [node_idx[s.u], node_idx[s.v], node_idx[s.x], node_idx[s.y]]
                for s in swaps
            ],
            dtype=torch.long,
            device=device,
        )
        model.eval()
        with torch.no_grad():
            return model(data, swap_idx)

    return score_fn


def save_move_oracle(
    model: MoveOracle,
    model_dir: Path,
    cycle_lengths: list[int],
    rwpe_dim: int,
    extra_info: dict[str, object] | None = None,
) -> None:
    """Save model weights and info.json to *model_dir*."""
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "weights.pt")

    info: dict[str, object] = {
        "model_type": "move_oracle",
        "training": {
            "node_feat_dim": model.node_proj.in_features,
            "hidden_dim": model.hidden_dim,
            "num_layers": model.num_layers,
            "feature_config": {
                "cycle_lengths": cycle_lengths,
                "rwpe_dim": rwpe_dim,
            },
        },
    }
    if extra_info:
        info.update(extra_info)

    with open(model_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)


def load_move_oracle(
    model_dir: Path,
) -> tuple[MoveOracle, list[int], int]:
    """Load a saved MoveOracle and its feature config.

    Returns (model, cycle_lengths, rwpe_dim).
    """
    info_path = model_dir / "info.json"
    weights_path = model_dir / "weights.pt"
    if not info_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"MoveOracle not found at {model_dir}")

    with open(info_path) as f:
        info = cast(dict[str, object], json.load(f))

    training = cast(dict[str, object], info.get("training", {}))
    feat_cfg = cast(dict[str, object], training.get("feature_config", {}))
    cycle_lengths_raw = feat_cfg.get("cycle_lengths", [3, 4, 5, 6, 7, 8])
    cycle_lengths = [int(cast(int, v)) for v in cast(list[object], cycle_lengths_raw)]
    rwpe_dim = int(cast(int, feat_cfg.get("rwpe_dim", 8)))

    base_node_feat_dim = 1  # ones column before structural features
    extra = structural_feature_dim(cycle_lengths=cycle_lengths, rwpe_dim=rwpe_dim)
    node_feat_dim = base_node_feat_dim + extra

    model = MoveOracle(
        node_feat_dim=node_feat_dim,
        hidden_dim=int(cast(int, training.get("hidden_dim", 64))),
        num_layers=int(cast(int, training.get("num_layers", 3))),
    )
    state = torch.load(weights_path, map_location="cpu")  # pyright: ignore[reportAny]
    _ = model.load_state_dict(state)  # pyright: ignore[reportAny]
    _ = model.eval()
    return model, cycle_lengths, rwpe_dim
