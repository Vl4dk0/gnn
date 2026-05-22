"""Tests for the classical (no-ML) Cayley cage baseline."""

from typing import cast

import pytest

from ai.cage.cayley.baseline import (
    KNOWN_CAGES,
    format_table,
    parse_targets,
    run_baseline,
)
from ai.cage.cayley.groups import available_groups


def test_parse_targets_accepts_valid_specs() -> None:
    assert parse_targets("3,6") == [(3, 6)]
    assert parse_targets("3,6;4,7;3,8") == [(3, 6), (4, 7), (3, 8)]
    assert parse_targets(" 3 , 6 ; 4,8 ") == [(3, 6), (4, 8)]


def test_parse_targets_rejects_bad_specs() -> None:
    with pytest.raises(ValueError):
        _ = parse_targets("")
    with pytest.raises(ValueError):
        _ = parse_targets("3")
    with pytest.raises(ValueError):
        _ = parse_targets("3,2")  # girth must be >= 3
    with pytest.raises(ValueError):
        _ = parse_targets("1,6")  # degree must be >= 2


def test_known_cages_reference_values() -> None:
    # Heawood graph is the (3,6)-cage; Petersen is the (3,5)-cage.
    assert KNOWN_CAGES[(3, 6)] == 14
    assert KNOWN_CAGES[(3, 5)] == 10


def test_available_groups_includes_semidirect_products() -> None:
    groups = available_groups(60)
    names = {g.name for g in groups}
    assert any(name.startswith("Z_") and "rtimes" in name for name in names)
    # Every catalogued group must respect the order bound.
    assert all(g.order <= 60 for g in groups)


def test_available_groups_abelian_cap_bounds_cyclic_orders() -> None:
    # Large max_order with a tight abelian cap: no huge cyclic group is built.
    groups = available_groups(5000, abelian_cap=50, semidirect_cap=50)
    cyclic_orders = [
        g.order
        for g in groups
        if g.name.startswith("Z_") and "x" not in g.name and "rtimes" not in g.name
    ]
    assert max(cyclic_orders) <= 50


def test_run_baseline_finds_small_cayley_graph() -> None:
    # (3,3): K4 is a Cayley graph of order 4 — the search must find girth 3.
    rows = run_baseline(
        [(3, 3)],
        max_group_order=16,
        num_random_trials=200,
        num_tabu_iters=100,
        num_workers=1,
        seed=1,
        verbose=False,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["k"] == 3 and row["g"] == 3
    found_order = row["found_order"]
    assert found_order is not None
    assert cast(int, found_order) >= 4
    assert cast(int, row["found_girth"]) >= 3


def test_format_table_renders_rows() -> None:
    rows: list[dict[str, object]] = [
        {
            "k": 3,
            "g": 6,
            "cage_order": 14,
            "found_order": 14,
            "found_girth": 6,
            "gap": 0,
            "group_name": "D_7",
            "generators": [1, 2, 3],
            "method": "random",
            "num_groups": 100,
            "elapsed": 1.5,
        }
    ]
    table = format_table(rows)
    assert "(3,6)" in table
    assert "D_7" in table
    assert "+0" in table
