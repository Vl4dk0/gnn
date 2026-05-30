"""Deterministic, seeded input batteries for benchmarks.

All functions are pure and deterministic — same quick flag always yields
the same graphs in the same order.
"""

from __future__ import annotations

import networkx as nx


def node_battery(quick: bool) -> list[tuple[str, nx.Graph[int]]]:
    """Return a mixed battery of named graphs with integer node labels.

    Includes seeded Erdos-Renyi random graphs and a fixed roster of
    well-known structured graphs.
    """
    graphs: list[tuple[str, nx.Graph[int]]] = []

    # Seeded Erdos-Renyi graphs
    er_specs: list[tuple[int, float, int]]
    if quick:
        er_specs = [(20, 0.3, 0), (30, 0.2, 1)]
    else:
        er_specs = [
            (20, 0.3, 0),
            (30, 0.2, 1),
            (40, 0.15, 2),
            (50, 0.1, 3),
            (60, 0.08, 4),
        ]
    for n, p, seed in er_specs:
        g: nx.Graph[int] = nx.convert_node_labels_to_integers(
            nx.erdos_renyi_graph(n, p, seed=seed)
        )
        graphs.append((f"er_n{n}_p{int(p * 100)}_s{seed}", g))

    # Fixed roster of named structured graphs
    named: list[tuple[str, nx.Graph[int]]] = [
        ("petersen", nx.convert_node_labels_to_integers(nx.petersen_graph())),
        ("complete_k5", nx.convert_node_labels_to_integers(nx.complete_graph(5))),
        ("complete_k7", nx.convert_node_labels_to_integers(nx.complete_graph(7))),
        (
            "complete_bipartite_3_3",
            nx.convert_node_labels_to_integers(nx.complete_bipartite_graph(3, 3)),
        ),
        (
            "complete_bipartite_4_5",
            nx.convert_node_labels_to_integers(nx.complete_bipartite_graph(4, 5)),
        ),
        ("cycle_c6", nx.convert_node_labels_to_integers(nx.cycle_graph(6))),
        ("cycle_c10", nx.convert_node_labels_to_integers(nx.cycle_graph(10))),
        ("path_p8", nx.convert_node_labels_to_integers(nx.path_graph(8))),
        ("path_p15", nx.convert_node_labels_to_integers(nx.path_graph(15))),
        ("grid_3x3", nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3))),
        ("grid_4x4", nx.convert_node_labels_to_integers(nx.grid_2d_graph(4, 4))),
        (
            "hypercube_q3",
            nx.convert_node_labels_to_integers(nx.hypercube_graph(3)),
        ),
        (
            "hypercube_q4",
            nx.convert_node_labels_to_integers(nx.hypercube_graph(4)),
        ),
        ("heawood", nx.convert_node_labels_to_integers(nx.heawood_graph())),
        ("desargues", nx.convert_node_labels_to_integers(nx.desargues_graph())),
        (
            "mobius_kantor",
            nx.convert_node_labels_to_integers(nx.moebius_kantor_graph()),
        ),
        (
            "dodecahedral",
            nx.convert_node_labels_to_integers(nx.dodecahedral_graph()),
        ),
        ("frucht", nx.convert_node_labels_to_integers(nx.frucht_graph())),
        ("tutte", nx.convert_node_labels_to_integers(nx.tutte_graph())),
    ]

    if quick:
        # In quick mode keep only smaller / faster graphs
        keep = {
            "petersen",
            "complete_k5",
            "cycle_c6",
            "path_p8",
            "grid_3x3",
            "hypercube_q3",
            "heawood",
            "frucht",
        }
        named = [(name, g) for name, g in named if name in keep]

    graphs.extend(named)
    return graphs


def cage_targets(quick: bool) -> list[tuple[int, int]]:
    """Return (k, g) pairs to use as cage construction targets.

    The full set spans a wide range of Moore bounds (10 -> 106) so the harder
    targets actually separate the methods instead of every approach trivially
    succeeding. Listed in increasing Moore-bound order; the Moore bound is
    noted in the trailing comment.
    """
    if quick:
        return [(3, 5), (3, 6), (4, 5)]
    return [
        (3, 5),  # 10
        (3, 6),  # 14
        (4, 5),  # 17
        (3, 7),  # 22
        (4, 6),  # 26
        (5, 5),  # 26
        (3, 8),  # 30
        (6, 5),  # 37
        (5, 6),  # 42
        (7, 5),  # 43
        (3, 9),  # 46
        (4, 7),  # 53
        (3, 10),  # 62
        (6, 6),  # 62
        (4, 8),  # 80
        (5, 7),  # 106
    ]


def refine_instances(
    quick: bool,
) -> list[tuple[str, int, int, nx.Graph[int]]]:
    """Return (name, k, g_target, graph) instances for the refine benchmark.

    Each graph is a random k-regular graph that has short cycles below g_target,
    giving the refiner meaningful work to do.
    """
    # (name, k, n, g_target, seed)
    specs: list[tuple[str, int, int, int, int]]
    if quick:
        specs = [
            ("rr_k3_n14_g5_s0", 3, 14, 5, 0),
            ("rr_k4_n20_g5_s2", 4, 20, 5, 2),
        ]
    else:
        specs = [
            ("rr_k3_n14_g5_s0", 3, 14, 5, 0),
            ("rr_k3_n20_g6_s1", 3, 20, 6, 1),
            ("rr_k4_n20_g5_s2", 4, 20, 5, 2),
            ("rr_k4_n28_g6_s3", 4, 28, 6, 3),
        ]

    instances: list[tuple[str, int, int, nx.Graph[int]]] = []
    for name, k, n, g_target, seed in specs:
        # ensure k*n is even (required by random_regular_graph)
        if (k * n) % 2 != 0:
            n += 1
        g: nx.Graph[int] = nx.convert_node_labels_to_integers(
            nx.random_regular_graph(k, n, seed=seed)
        )
        instances.append((name, k, g_target, g))
    return instances


def excision_instances(
    quick: bool,
) -> list[tuple[str, int, int, nx.Graph[int]]]:
    """Return (name, k, g, source_graph) using known cubic cages.

    The source graphs are actual cages: petersen (3,5), heawood (3,6),
    mcgee (3,7), tutte-coxeter (3,8).  In quick mode only the two smallest
    are included.
    """
    instances: list[tuple[str, int, int, nx.Graph[int]]] = [
        (
            "petersen",
            3,
            5,
            nx.convert_node_labels_to_integers(nx.petersen_graph()),
        ),
        (
            "heawood",
            3,
            6,
            nx.convert_node_labels_to_integers(nx.heawood_graph()),
        ),
    ]
    if not quick:
        instances += [
            (
                "mcgee",
                3,
                7,
                nx.convert_node_labels_to_integers(nx.LCF_graph(24, [12, 7, -7], 8)),
            ),
            (
                "tutte_coxeter",
                3,
                8,
                nx.convert_node_labels_to_integers(
                    nx.LCF_graph(30, [-13, -9, 7, -7, 9, 13], 5)
                ),
            ),
        ]
    return instances
