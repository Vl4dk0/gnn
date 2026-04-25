"""Training data generation for the girth predictor.

Generates (base_graph, voltage_assignment, girth) triples as PyG Data
objects.  Each sample encodes a voltage-labeled base graph together with
the girth of the corresponding lift.
"""

from __future__ import annotations

import random

import torch
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]

from ai.cage.voltage.base_graphs import (
    BaseGraph,
    dumbbell,
    bouquet,
    prism_base,
    cubic_multigraph_4nodes,
)
from ai.cage.voltage.cycle_analysis import compute_lift_girth
from ai.cage.voltage.groups import (
    FiniteGroup,
    cyclic_group,
    dihedral_group,
    direct_product,
)
from backend.utils.graph_utils import moore_bound


def base_graph_to_pyg(
    base: BaseGraph,
    voltages: list[int],
    group: FiniteGroup,
    k: int,
    g_target: int,
    girth: int | float,
) -> Data:
    """Convert a voltage-labeled base graph to a PyG Data object.

    Node features (per base-graph node):
        [degree / max_degree, node_index / num_nodes]

    Edge features (per directed arc):
        [voltage_element / group_order]
        (We use a continuous encoding; the model.py uses an embedding layer.)

    Global attributes stored on the Data object:
        k, g_target, group_order, girth (label), girth_class (binary: girth >= g_target)
    """
    n = base.num_nodes

    # Node features: [normalized_degree, normalized_index]
    max_deg = max((base.degree(v) for v in range(n)), default=1)
    x = torch.zeros((n, 2), dtype=torch.float)
    for v in range(n):
        x[v, 0] = base.degree(v) / max(max_deg, 1)
        x[v, 1] = v / max(n - 1, 1)

    # Edge index and edge attributes from arcs
    src_list: list[int] = []
    dst_list: list[int] = []
    edge_voltage: list[int] = []

    # Build arc -> voltage mapping
    arc_volt: dict[int, int] = {}
    for edge_pos, fwd_id in enumerate(base.undirected_edge_ids):
        v = voltages[edge_pos]
        rev_id = base.arcs[fwd_id].reverse_id
        arc_volt[fwd_id] = v
        arc_volt[rev_id] = group.inv(v)

    for arc in base.arcs:
        src_list.append(arc.src)
        dst_list.append(arc.dst)
        edge_voltage.append(arc_volt[arc.arc_id])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    # Edge attr: voltage as integer (used as embedding index in the model)
    edge_attr = torch.tensor(edge_voltage, dtype=torch.long).unsqueeze(-1)

    # Label — infinite girth means degenerate lift (disconnected/not regular),
    # clamp to 0 and mark as negative example
    if isinstance(girth, float):
        girth_int = 0
        girth_class = 0
    else:
        girth_int = int(girth)
        girth_class = 1 if girth_int >= g_target else 0

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=n,
    )
    data.k = k
    data.g_target = g_target
    data.group_order = group.order
    data.girth = girth_int
    data.girth_class = girth_class

    return data


def _candidate_groups(max_order: int) -> list[FiniteGroup]:
    """Generate a list of candidate groups up to a given order."""
    groups: list[FiniteGroup] = []

    # Cyclic groups
    for n in range(2, max_order + 1):
        groups.append(cyclic_group(n))

    # Dihedral groups D_n (order 2n)
    for n in range(3, max_order // 2 + 1):
        if 2 * n <= max_order:
            groups.append(dihedral_group(n))

    # Some direct products of small cyclic groups
    for a in range(2, min(10, max_order)):
        for b in range(a, min(10, max_order)):
            if a * b <= max_order:
                groups.append(direct_product(cyclic_group(a), cyclic_group(b)))

    return groups


def _candidate_base_graphs(k: int) -> list[BaseGraph]:
    """Generate candidate base graphs for degree k."""
    bases: list[BaseGraph] = []

    if k == 3:
        # Dumbbell (2 nodes, 3 parallel edges) — most productive
        bases.append(dumbbell(3))
        # 4-node cubic multigraph
        bases.append(cubic_multigraph_4nodes())
        # Prism (6 nodes)
        bases.append(prism_base())

    # Dipoles for any k
    bases.append(dumbbell(k))

    # Bouquets (for even-degree lifts: lift of bouquet(m) is 2m-regular)
    if k % 2 == 0:
        bases.append(bouquet(k // 2))

    return bases


def generate_dataset(
    k: int,
    g_target: int,
    num_samples: int = 10000,
    max_group_order: int = 60,
) -> list[Data]:
    """Generate training data for the girth predictor.

    For each sample, randomly picks a base graph and group, samples a
    random voltage assignment, computes the exact lift girth, and returns
    the result as a PyG Data object.
    """
    bases = _candidate_base_graphs(k)
    groups = _candidate_groups(max_group_order)

    # Filter groups to those that give lifts of reasonable size
    # (between Moore bound and ~4x Moore bound)
    mb = moore_bound(k, g_target)

    dataset: list[Data] = []

    for _ in range(num_samples):
        base = random.choice(bases)
        # Filter groups by resulting lift size
        valid_groups = [
            g for g in groups if mb <= base.num_nodes * g.order <= max(4 * mb, mb + 200)
        ]
        if not valid_groups:
            # Fall back to all groups
            valid_groups = groups

        group = random.choice(valid_groups)
        n_edges = base.num_undirected_edges()

        # Random voltage assignment
        volt = [random.randint(0, group.order - 1) for _ in range(n_edges)]

        # Compute girth
        girth = compute_lift_girth(base, group, volt, max_girth=2 * g_target)

        data = base_graph_to_pyg(base, volt, group, k, g_target, girth)
        dataset.append(data)

    return dataset
