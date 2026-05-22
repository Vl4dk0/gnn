from ai.cage.voltage.base_graphs import dumbbell
from ai.cage.voltage.cycle_analysis import (
    compute_lift_girth,
    count_short_identity_walks,
    has_girth_at_least,
)
from ai.cage.voltage.groups import cyclic_group, dihedral_group
from ai.cage.voltage.lift import build_lift, lift_order, verify_lift


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
