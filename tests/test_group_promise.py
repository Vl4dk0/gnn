"""Tests for the group-promise predictor and group-guided Cayley search."""

import json
import pathlib
import random
from typing import cast

import torch

from ai.cage.cayley.group_data_gen import (
    best_achievable_girth,
    canonical_generating_set,
    group_to_pyg,
)
from ai.cage.cayley.group_model import GroupPromisePredictor, make_group_filter
from ai.cage.cayley.group_search import load_baseline_json
from ai.cage.cayley.groups import cyclic_group, dihedral_group


def test_canonical_generating_set_generates_whole_group() -> None:
    for group in (cyclic_group(8), dihedral_group(5)):
        gens = canonical_generating_set(group)
        # Closure under multiplication must recover every element.
        seen: set[int] = {0}
        frontier = [0]
        while frontier:
            v = frontier.pop()
            for s in gens:
                u = group.mult(v, s)
                if u not in seen:
                    seen.add(u)
                    frontier.append(u)
        assert len(seen) == group.order


def test_best_achievable_girth_finds_seven_cycle() -> None:
    # Z_7 with a degree-2 generating set is a 7-cycle: girth 7.
    random.seed(0)
    group = cyclic_group(7)
    girth = best_achievable_girth(
        group, k=2, g_target=5, num_random_trials=60, num_tabu_iters=20
    )
    assert girth == 7


def test_group_to_pyg_has_expected_shapes() -> None:
    group = dihedral_group(6)
    data = group_to_pyg(group, k=3, g_target=6, best_girth=6)
    assert int(cast(int, data.num_nodes)) == group.order
    x = cast(torch.Tensor, data.x)
    assert x.shape == (group.order, 3)
    edge_attr = cast(torch.Tensor, data.edge_attr)
    assert edge_attr.shape[1] == 2
    assert int(cast(int, data.k)) == 3
    assert int(cast(int, data.g_target)) == 6
    assert int(cast(int, data.girth)) >= int(cast(int, data.g_target))


def test_group_promise_predictor_forward_pass() -> None:
    group = dihedral_group(5)
    data = group_to_pyg(group, k=3, g_target=6, best_girth=4)
    model = GroupPromisePredictor(hidden_dim=16, num_layers=2)
    _ = model.eval()
    with torch.no_grad():
        girth_pred = model(data)
    assert girth_pred.shape == (1, 1)


def test_make_group_filter_returns_finite_score() -> None:
    model = GroupPromisePredictor(hidden_dim=16, num_layers=2)
    predict = make_group_filter(model)
    score = predict(dihedral_group(5), 3, 6)
    assert isinstance(score, float)


def test_load_baseline_json_roundtrip(tmp_path: pathlib.Path) -> None:
    payload = {
        "results": [
            {"k": 3, "g": 6, "found_order": 14, "group_name": "D_7", "elapsed": 12.3},
            {"k": 3, "g": 7, "found_order": 30, "group_name": "Z_5_rtimes_Z_6"},
        ]
    }
    path = tmp_path / "baseline.json"
    with open(path, "w") as f:
        json.dump(payload, f)

    loaded = load_baseline_json(str(path))
    assert (3, 6) in loaded and (3, 7) in loaded
    assert loaded[(3, 6)]["found_order"] == 14
    assert loaded[(3, 6)]["group_name"] == "D_7"


