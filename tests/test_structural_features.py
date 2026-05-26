"""Tests for ai/utils/structural_features.py.

Covers:
- Cycle count correctness on Petersen, K4, and Heawood graphs (brute-force
  vs networkx.simple_cycles ground truth).
- Permutation-equivariance of cycle counts.
- RWPE non-collapse sanity check.
- Smoke tests for add_structural_features (empty config, non-empty config).
"""

from __future__ import annotations

import random

import networkx as nx
import numpy as np
import pytest
import torch
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]
from torch_geometric.utils import from_networkx  # pyright: ignore[reportMissingTypeStubs]

from ai.utils.structural_features import (
    add_structural_features,
    structural_feature_dim,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _pyg_from_nx(g: nx.Graph[int]) -> Data:
    """Convert a NetworkX graph to a PyG Data with dummy x (ones)."""
    n = g.number_of_nodes()
    # Build edge_index manually to control dtype
    src: list[int] = []
    dst: list[int] = []
    for u, v in g.edges():
        src.extend([u, v])
        dst.extend([v, u])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x = torch.ones((n, 2), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, num_nodes=n)


def _brute_force_cycle_counts(g: nx.Graph[int], length: int) -> list[float]:
    """Ground-truth per-vertex count of simple cycles of given length via nx."""
    n = g.number_of_nodes()
    counts = [0.0] * n
    for cycle in nx.simple_cycles(g, length_bound=length):
        if len(cycle) == length:
            for v in cycle:
                counts[v] += 0.5
    return counts


# ── Reference graphs ─────────────────────────────────────────────────────────


def petersen_graph() -> nx.Graph[int]:
    """Return the Petersen graph (3-regular, girth 5, 10 vertices)."""
    return nx.petersen_graph()


def heawood_graph() -> nx.Graph[int]:
    """Return the Heawood graph (3-regular, girth 6, 14 vertices)."""
    return nx.heawood_graph()


def k4_graph() -> nx.Graph[int]:
    """Return the complete graph K4."""
    return nx.complete_graph(4)


# ── Cycle count tests ─────────────────────────────────────────────────────────


def test_cycle_counts_petersen_5cycles() -> None:
    """Each vertex of the Petersen graph is on exactly 12 directed 5-cycles,
    i.e. 6 undirected 5-cycles (each undirected cycle counted twice by nx).
    We verify that add_structural_features gives the same count as brute-force.
    """
    g = petersen_graph()
    data = _pyg_from_nx(g)

    result = add_structural_features(data, cycle_lengths=[5], rwpe_dim=0)
    # x was [n, 2]; after adding 1 cycle-length column it should be [n, 3]
    assert result.x is not None
    x_new = result.x
    assert x_new.shape == (10, 3), f"Expected (10, 3), got {x_new.shape}"

    # Column index 2 = count of 5-cycles through each vertex
    counts_pred = x_new[:, 2].tolist()

    # Ground truth via brute-force
    counts_gt = _brute_force_cycle_counts(g, 5)

    for v in range(10):
        assert abs(counts_pred[v] - counts_gt[v]) < 1e-6, (
            f"Vertex {v}: predicted {counts_pred[v]:.1f}, brute-force {counts_gt[v]:.1f}"
        )

    # All vertices should have the same count (Petersen is vertex-transitive)
    assert len(set(round(c, 5) for c in counts_pred)) == 1, (
        "Petersen is vertex-transitive; all 5-cycle counts must be equal"
    )


def test_cycle_counts_petersen_6cycles() -> None:
    """Verify 6-cycle counts on Petersen match brute-force."""
    g = petersen_graph()
    data = _pyg_from_nx(g)
    result = add_structural_features(data, cycle_lengths=[6], rwpe_dim=0)
    assert result.x is not None
    counts_pred = result.x[:, 2].tolist()
    counts_gt = _brute_force_cycle_counts(g, 6)
    for v in range(10):
        assert abs(counts_pred[v] - counts_gt[v]) < 1e-6, (
            f"6-cycle: vertex {v}: predicted {counts_pred[v]:.1f}, gt {counts_gt[v]:.1f}"
        )


def test_cycle_counts_k4_triangles() -> None:
    """K4 has exactly 4 triangles total. Each vertex is on exactly 2.

    C(4, 3) * 2 (orientations) / 3 (per cycle vertices) ... simpler:
    each of the 4 triangles passes through 3 vertices, so total vertex-triangle
    incidences = 4 * 3 = 12. Per vertex = 12 / 4 = 3 undirected triangles.
    But each triangle is counted from each of its 3 vertices, so each vertex
    is in C(3, 2) = 3 triangles (choosing 2 others from the 3 remaining).
    """
    g = k4_graph()
    data = _pyg_from_nx(g)
    result = add_structural_features(data, cycle_lengths=[3], rwpe_dim=0)
    assert result.x is not None
    counts_pred = result.x[:, 2].tolist()
    counts_gt = _brute_force_cycle_counts(g, 3)
    for v in range(4):
        assert abs(counts_pred[v] - counts_gt[v]) < 1e-6, (
            f"K4 triangle: vertex {v}: predicted {counts_pred[v]:.1f}, gt {counts_gt[v]:.1f}"
        )
    # K4 is vertex-transitive: all counts equal
    assert len(set(round(c, 5) for c in counts_pred)) == 1


def test_cycle_counts_heawood_6cycles() -> None:
    """Heawood graph has girth 6; each vertex is on some number of 6-cycles.
    We verify add_structural_features matches brute-force.
    """
    g = heawood_graph()
    data = _pyg_from_nx(g)
    result = add_structural_features(data, cycle_lengths=[6], rwpe_dim=0)
    assert result.x is not None
    counts_pred = result.x[:, 2].tolist()
    counts_gt = _brute_force_cycle_counts(g, 6)
    for v in range(14):
        assert abs(counts_pred[v] - counts_gt[v]) < 1e-6, (
            f"Heawood 6-cycle: vertex {v}: predicted {counts_pred[v]:.1f}, gt {counts_gt[v]:.1f}"
        )
    # Heawood is vertex-transitive: all counts equal
    assert len(set(round(c, 5) for c in counts_pred)) == 1


def test_cycle_counts_heawood_no_triangles_or_4cycles() -> None:
    """Heawood has girth 6, so no 3- or 4-cycles exist."""
    g = heawood_graph()
    data = _pyg_from_nx(g)
    result = add_structural_features(data, cycle_lengths=[3, 4], rwpe_dim=0)
    assert result.x is not None
    # columns 2 and 3 are the 3- and 4-cycle counts
    tri_counts = result.x[:, 2].tolist()
    quad_counts = result.x[:, 3].tolist()
    assert all(abs(c) < 1e-9 for c in tri_counts), "Heawood has no triangles"
    assert all(abs(c) < 1e-9 for c in quad_counts), "Heawood has no 4-cycles"


# ── Permutation equivariance ───────────────────────────────────────────────────


def test_cycle_counts_equivariant_under_relabelling() -> None:
    """Relabelling graph vertices should permute cycle counts in the same way."""
    rng = random.Random(0)
    g = petersen_graph()
    n = g.number_of_nodes()

    perm = list(range(n))
    rng.shuffle(perm)
    inv_perm = [0] * n
    for i, p in enumerate(perm):
        inv_perm[p] = i

    g_perm: nx.Graph[int] = nx.relabel_nodes(g, {i: perm[i] for i in range(n)})

    data_orig = _pyg_from_nx(g)
    data_perm = _pyg_from_nx(g_perm)

    res_orig = add_structural_features(data_orig, cycle_lengths=[5], rwpe_dim=0)
    res_perm = add_structural_features(data_perm, cycle_lengths=[5], rwpe_dim=0)

    assert res_orig.x is not None
    assert res_perm.x is not None

    counts_orig = res_orig.x[:, 2].tolist()
    counts_perm = res_perm.x[:, 2].tolist()

    # counts_perm[perm[v]] should equal counts_orig[v]
    for v in range(n):
        assert abs(counts_perm[perm[v]] - counts_orig[v]) < 1e-6, (
            f"Vertex {v}: orig count {counts_orig[v]:.1f} != perm count {counts_perm[perm[v]]:.1f}"
        )


# ── RWPE tests ─────────────────────────────────────────────────────────────────


def test_rwpe_non_collapse_on_petersen() -> None:
    """On the Petersen graph RWPE rows should not all be identical (the graph is
    vertex-transitive so they WILL be identical — but that is fine: RWPE alone
    does not break symmetry on vertex-transitive graphs. What we test here is
    that RWPE is computed and has the right shape / is non-trivially non-zero).
    """
    g = petersen_graph()
    data = _pyg_from_nx(g)
    result = add_structural_features(data, cycle_lengths=None, rwpe_dim=4)
    assert result.x is not None
    # x was [10, 2]; after adding rwpe_dim=4 columns it should be [10, 6]
    assert result.x.shape == (10, 6), f"Expected (10, 6), got {result.x.shape}"
    # RWPE values should be in (0, 1] since they are diagonal of transition matrix powers
    rwpe_cols = result.x[:, 2:].numpy()
    assert float(rwpe_cols.min()) >= 0.0
    assert float(rwpe_cols.max()) <= 1.0 + 1e-9
    # At least one RWPE column should be non-zero
    assert float(rwpe_cols.sum()) > 0.0


def test_rwpe_non_collapse_path_vs_cycle() -> None:
    """P4 (path of 4 nodes) and C4 (cycle of 4 nodes) are not isomorphic.
    Their RWPE row vectors should differ at some vertex position.
    """
    p4: nx.Graph[int] = nx.path_graph(4)
    c4: nx.Graph[int] = nx.cycle_graph(4)

    data_p4 = _pyg_from_nx(p4)
    data_c4 = _pyg_from_nx(c4)

    res_p4 = add_structural_features(data_p4, cycle_lengths=None, rwpe_dim=4)
    res_c4 = add_structural_features(data_c4, cycle_lengths=None, rwpe_dim=4)

    assert res_p4.x is not None
    assert res_c4.x is not None

    rwpe_p4 = res_p4.x[:, 2:].numpy()
    rwpe_c4 = res_c4.x[:, 2:].numpy()

    # Sort rows to make comparison structure-agnostic, then check they differ
    rwpe_p4_sorted = np.sort(rwpe_p4, axis=0)
    rwpe_c4_sorted = np.sort(rwpe_c4, axis=0)

    assert not np.allclose(rwpe_p4_sorted, rwpe_c4_sorted, atol=1e-6), (
        "RWPE should distinguish P4 from C4"
    )


# ── Smoke tests for add_structural_features ───────────────────────────────────


def test_empty_config_returns_data_unchanged() -> None:
    """Empty feature config (no cycles, no RWPE) should return the same object."""
    g = petersen_graph()
    data = _pyg_from_nx(g)
    result = add_structural_features(data, cycle_lengths=None, rwpe_dim=0)
    assert result is data, "Should return the same Data object when config is empty"


def test_empty_cycle_list_returns_data_unchanged() -> None:
    """Empty list for cycle_lengths should also be a no-op."""
    g = petersen_graph()
    data = _pyg_from_nx(g)
    result = add_structural_features(data, cycle_lengths=[], rwpe_dim=0)
    assert result is data


def test_feature_dim_matches_added_columns() -> None:
    """structural_feature_dim must agree with the actual column count added."""
    g = petersen_graph()
    data = _pyg_from_nx(g)
    n_orig = data.x.shape[1] if data.x is not None else 0  # pyright: ignore[reportOptionalMemberAccess]

    cycle_lengths = [3, 5, 6]
    rwpe_dim = 4
    expected_extra = structural_feature_dim(
        cycle_lengths=cycle_lengths, rwpe_dim=rwpe_dim
    )

    result = add_structural_features(
        data, cycle_lengths=cycle_lengths, rwpe_dim=rwpe_dim
    )
    assert result.x is not None
    actual_extra = result.x.shape[1] - n_orig
    assert actual_extra == expected_extra, (
        f"structural_feature_dim={expected_extra} but {actual_extra} columns added"
    )


def test_structural_feature_dim_zero_config() -> None:
    assert structural_feature_dim(cycle_lengths=None, rwpe_dim=0) == 0
    assert structural_feature_dim(cycle_lengths=[], rwpe_dim=0) == 0


def test_structural_feature_dim_cycle_only() -> None:
    assert structural_feature_dim(cycle_lengths=[3, 5, 6, 8], rwpe_dim=0) == 4


def test_structural_feature_dim_rwpe_only() -> None:
    assert structural_feature_dim(cycle_lengths=None, rwpe_dim=8) == 8


def test_structural_feature_dim_combined() -> None:
    assert structural_feature_dim(cycle_lengths=[3, 4, 5, 6, 7, 8], rwpe_dim=8) == 14


def test_add_features_no_x_creates_ones_column() -> None:
    """When data.x is None, a ones column should be prepended."""
    g = petersen_graph()
    n = g.number_of_nodes()
    src, dst = [], []
    for u, v in g.edges():
        src.extend([u, v])
        dst.extend([v, u])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    data = Data(edge_index=edge_index, num_nodes=n)
    # data.x is None

    result = add_structural_features(data, cycle_lengths=[5], rwpe_dim=0)
    assert result.x is not None
    # ones column (1) + 1 cycle column = 2 columns
    assert result.x.shape == (n, 2), f"Expected ({n}, 2), got {result.x.shape}"
    # First column should all be 1.0
    assert torch.all(result.x[:, 0] == 1.0)


def test_custom_attributes_preserved() -> None:
    """Custom Data attributes (like k, g_target) must survive add_structural_features."""
    g = petersen_graph()
    data = _pyg_from_nx(g)
    data.k = 3
    data.g_target = 5
    data.group_order = 7

    result = add_structural_features(data, cycle_lengths=[5], rwpe_dim=2)

    assert hasattr(result, "k") and result.k == 3
    assert hasattr(result, "g_target") and result.g_target == 5
    assert hasattr(result, "group_order") and result.group_order == 7
