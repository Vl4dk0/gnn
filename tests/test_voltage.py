from typing import cast

import torch

from ai.cage.utils import normalize_target
from ai.cage.voltage.base_graphs import dumbbell
from ai.cage.voltage.cycle_analysis import (
    compute_lift_girth,
    count_short_identity_walks,
    has_girth_at_least,
)
from ai.cage.voltage.groups import (
    cyclic_group,
    dihedral_group,
    direct_product,
    semidirect_product_cyclic,
)
from ai.cage.voltage.lift import build_lift, lift_order, verify_lift
from ai.cage.voltage.supervised.data_gen import generate_dataset


def test_cyclic_group_operations() -> None:
    group = cyclic_group(7)

    assert group.identity() == 0
    assert group.mult(3, 5) == 1
    assert group.inv(3) == 4
    assert list(group.elements()) == list(range(7))


def test_dihedral_group_has_identity_and_inverses() -> None:
    group = dihedral_group(4)

    for element in group.elements():
        assert group.mult(group.identity(), element) == element
        assert group.mult(element, group.inv(element)) == group.identity()


def test_direct_product_group_identity_and_inverses() -> None:
    """Z_4 x Z_5 should have order 20, identity at 0, and full inverses."""
    group = direct_product(cyclic_group(4), cyclic_group(5))

    assert group.order == 20
    assert group.identity() == 0
    assert list(group.elements()) == list(range(20))
    for elem in group.elements():
        assert group.mult(group.identity(), elem) == elem
        assert group.mult(elem, group.identity()) == elem
        assert group.mult(elem, group.inv(elem)) == group.identity()


def test_semidirect_product_group_identity_and_inverses() -> None:
    """Z_3 ⋊ Z_2 (phi=2, i.e. inversion in Z_3) should have order 6."""
    # phi=2, since 2^2 = 4 ≡ 1 (mod 3)
    group = semidirect_product_cyclic(3, 2, 2)

    assert group.order == 6
    assert group.identity() == 0
    assert list(group.elements()) == list(range(6))
    for elem in group.elements():
        assert group.mult(group.identity(), elem) == elem
        assert group.mult(elem, group.identity()) == elem
        assert group.mult(elem, group.inv(elem)) == group.identity()


def test_heawood_voltage_lift_is_valid_3_6_graph() -> None:
    base = dumbbell(3)
    group = cyclic_group(7)
    voltages = [0, 1, 3]

    graph = build_lift(base, group, voltages)
    props = verify_lift(graph, k=3, g=6)

    assert lift_order(base, group) == 14
    assert props["num_nodes"] == 14
    assert props["is_k_regular"]
    assert props["is_connected"]
    assert props["girth"] == 6
    assert props["is_valid_kg"]


def test_voltage_girth_analysis_matches_known_heawood_assignment() -> None:
    base = dumbbell(3)
    group = cyclic_group(7)
    voltages = [0, 1, 3]

    assert compute_lift_girth(base, group, voltages, max_girth=10) == 6
    assert has_girth_at_least(base, group, voltages, g_target=6)
    assert not has_girth_at_least(base, group, voltages, g_target=7)
    assert count_short_identity_walks(base, group, voltages, g_target=6) == 0


def test_voltage_girth_analysis_detects_bad_short_cycles() -> None:
    base = dumbbell(3)
    group = cyclic_group(7)
    voltages = [0, 1, 2]

    assert compute_lift_girth(base, group, voltages, max_girth=6) == 4
    assert not has_girth_at_least(base, group, voltages, g_target=5)
    assert count_short_identity_walks(base, group, voltages, g_target=5) > 0


def test_dataset_labels_girth_vs_tabu_cost() -> None:
    """The dataset label stored in `girth` must match the requested kind.

    For "girth" it is the lift's girth (>= g_target on feasible samples); for
    "tabu_cost" it is count_short_identity_walks (0 iff girth >= g_target). The
    two are stored in the same field so the model code is shared, so we verify
    they actually differ and stay consistent on a tiny deterministic run.
    """
    girth_data, _ = generate_dataset(
        targets=[(3, 5)], num_samples=80, max_group_order=30, seed=0, workers=1
    )
    tabu_data, _ = generate_dataset(
        targets=[(3, 5)],
        num_samples=80,
        max_group_order=30,
        seed=0,
        workers=1,
        kind="tabu_cost",
    )

    girth_labels = [int(cast(int, d.girth)) for d in girth_data]
    tabu_labels = [int(cast(int, d.girth)) for d in tabu_data]

    # girth labels are girths (>= 3); tabu labels are non-negative counts that
    # include both feasible zeros and infeasible positives.
    assert all(v >= 3 for v in girth_labels)
    assert all(v >= 0 for v in tabu_labels)
    assert min(tabu_labels) == 0
    assert max(tabu_labels) > 0
    # The same seed/config yields the same samples, so feasibility agrees:
    # girth >= g_target exactly when tabu cost == 0.
    assert (girth_labels[0] >= 5) == (tabu_labels[0] == 0)


def test_normalize_target_ranking_direction() -> None:
    """Normalization preserves the per-kind 'better' direction.

    girth is ranked DESCENDING (higher girth better) and tabu_cost ASCENDING
    (lower cost better). normalize_target must keep girth monotone increasing
    and tabu cost monotone increasing in the raw value, so the search can flip
    the sort direction by kind alone.
    """
    girth = normalize_target(torch.tensor([4.0, 6.0, 12.0]), "girth")
    assert bool(girth[0] < girth[1] < girth[2])

    tabu = normalize_target(torch.tensor([0.0, 3.0, 50.0]), "tabu_cost")
    assert bool(tabu[0] < tabu[1] < tabu[2])
    # cost 0 (feasible) maps to the smallest normalized value.
    assert float(tabu[0]) == 0.0
