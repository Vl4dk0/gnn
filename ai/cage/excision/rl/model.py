"""Actor-Critic policy for graph repair after tree excision.

Architecture:
  Encoder : GINE backbone (GINEConv layers on the repair graph).
            Node features include degree, deficiency, and structural features
            (cycle counts + RWPE) that break the symmetry barrier on regular
            graphs (Chen et al. NeurIPS 2023).

  Actor   : For each legal (u, v) pair, score = MLP(concat(emb_u, emb_v)).
            Softmax over legal pairs gives the action distribution.
            The scoring is bilinear-MLP: takes the concatenation of u and v
            embeddings, maps through a two-layer MLP to a scalar.

  Critic  : global_mean_pool → MLP → scalar value.

The policy is (k,g)-independent: (k, g) are encoded as node features in the
observation (see RepairEnv._build_obs), not baked into separate weight matrices.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]
from torch_geometric.nn import global_mean_pool  # pyright: ignore[reportMissingTypeStubs]

from ai.cage.utils import build_gine_stack


class RepairActorCritic(nn.Module):
    """Actor-Critic for repair edge selection.

    Args:
        input_dim:   Node feature dimension (base features + structural).
        hidden_dim:  Hidden dimension for GINEConv layers.
        num_layers:  Number of GINEConv message-passing layers.
        dropout:     Dropout rate for actor/critic heads.
    """

    node_proj: nn.Linear
    edge_proj: nn.Linear
    convs: nn.ModuleList
    bns: nn.ModuleList
    pair_scorer: nn.Sequential
    value_head: nn.Sequential
    hidden_dim: int

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.hidden_dim = hidden_dim

        # Input projection: node features → hidden_dim
        self.node_proj = nn.Linear(input_dim, hidden_dim)

        # Edge projection: 1-dim edge feature (placeholder = 1.0) → hidden_dim
        # GINEConv requires edge features; we use a constant-1 feature.
        self.edge_proj = nn.Linear(1, hidden_dim)

        # GINEConv message-passing layers. LayerNorm (not BatchNorm) because the
        # PPO inner loop runs forward passes one sample at a time, which would
        # poison BN running stats.
        self.convs, _ = build_gine_stack(hidden_dim, num_layers)
        self.bns = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # Actor: pair-scoring MLP — takes concat(emb_u, emb_v), outputs scalar
        self.pair_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Critic: graph-level scalar value
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _encode(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode graph into per-node embeddings and a graph-level embedding.

        Returns:
            node_emb:  [n, hidden_dim] per-node embeddings
            graph_emb: [1, hidden_dim] global mean-pooled embedding
        """
        x = cast(torch.Tensor, data.x)
        edge_index = cast(torch.Tensor, data.edge_index)
        n = x.size(0)
        num_edges = edge_index.size(1) if edge_index.numel() > 0 else 0

        h = F.relu(self.node_proj(x))

        # Constant edge features (all 1s) — GINEConv needs something
        if num_edges > 0:
            edge_attr = torch.ones(num_edges, 1, dtype=torch.float, device=x.device)
        else:
            edge_attr = torch.empty(0, 1, dtype=torch.float, device=x.device)
        edge_h = self.edge_proj(edge_attr)

        for i in range(len(self.convs)):
            if num_edges > 0:
                h = cast(torch.Tensor, self.convs[i](h, edge_index, edge_h))
            else:
                # No edges: GINEConv would fail; fall back to identity pass
                mlp = cast(nn.Sequential, self.convs[i].nn)
                h = cast(torch.Tensor, mlp(h))
            h = cast(torch.Tensor, self.bns[i](h))
            h = F.relu(h)

        batch = torch.zeros(n, dtype=torch.long, device=x.device)
        graph_emb = cast(torch.Tensor, global_mean_pool(h, batch))
        return h, graph_emb

    def get_action_logits(
        self,
        data: Data,
        legal_pairs: list[tuple[int, int]],
        node_ids: list[int],
    ) -> torch.Tensor:
        """Compute unnormalised logits over legal (u, v) pairs.

        Args:
            data:        PyG observation.
            legal_pairs: List of (u, v) node-ID pairs (original graph IDs).
            node_ids:    Ordered list mapping position i → original node ID.
                         (data.node_ids from RepairEnv._build_obs)

        Returns:
            logits: [len(legal_pairs)] tensor of unnormalised scores.
        """
        node_emb, _ = self._encode(data)
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        pair_vecs: list[torch.Tensor] = []
        for u, v in legal_pairs:
            ui = id_to_idx[u]
            vi = id_to_idx[v]
            pair_vecs.append(torch.cat([node_emb[ui], node_emb[vi]], dim=-1))

        pair_mat = torch.stack(pair_vecs, dim=0)  # [num_pairs, 2*hidden]
        logits = cast(torch.Tensor, self.pair_scorer(pair_mat)).squeeze(-1)
        return logits

    def get_value(self, data: Data) -> torch.Tensor:
        """Return state value scalar [1, 1]."""
        _, graph_emb = self._encode(data)
        return cast(torch.Tensor, self.value_head(graph_emb))

    def get_action(
        self,
        data: Data,
        legal_pairs: list[tuple[int, int]],
        node_ids: list[int],
        deterministic: bool = False,
    ) -> tuple[tuple[int, int], torch.Tensor, torch.Tensor]:
        """Sample or pick an action from legal_pairs.

        Returns:
            (action, log_prob, value)
        """
        logits = self.get_action_logits(data, legal_pairs, node_ids)
        value = self.get_value(data)

        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            idx = int(torch.argmax(logits).item())
        else:
            idx = int(dist.sample().item())

        action = legal_pairs[idx]
        log_prob = cast(
            torch.Tensor, dist.log_prob(torch.tensor(idx, device=logits.device))
        )
        return action, log_prob, value
