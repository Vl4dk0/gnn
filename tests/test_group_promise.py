"""Tests for the group-promise predictor and group-guided Cayley search."""

import random
from typing import cast

import torch

from ai.cage.cayley.group_data_gen import (
    best_achievable_girth,
    canonical_generating_set,
    generate_group_dataset,
    group_to_pyg,
)
from ai.cage.cayley.group_model import GroupPromisePredictor, make_group_filter
from ai.cage.cayley.group_search import group_guided_meta_search
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
    assert int(cast(int, data.girth_class)) == 1  # best_girth 6 >= target 6


def test_generate_group_dataset_smoke() -> None:
    data, stats = generate_group_dataset(
        targets=[(3, 4), (3, 5)],
        max_group_order=16,
        num_random_trials=30,
        num_tabu_iters=15,
        num_workers=1,
        seed=1,
    )
    assert len(data) == cast(int, stats["produced"])
    assert len(data) > 0
    per_target = cast(dict[str, dict[str, float]], stats["per_target"])
    assert "3_4" in per_target and "3_5" in per_target


def test_group_promise_predictor_forward_pass() -> None:
    group = dihedral_group(5)
    data = group_to_pyg(group, k=3, g_target=6, best_girth=4)
    model = GroupPromisePredictor(hidden_dim=16, num_layers=2)
    _ = model.eval()
    with torch.no_grad():
        girth_pred, girth_logit = model(data)
    assert girth_pred.shape == (1, 1)
    assert girth_logit.shape == (1, 1)


def test_make_group_filter_returns_finite_score() -> None:
    model = GroupPromisePredictor(hidden_dim=16, num_layers=2)
    predict = make_group_filter(model)
    score = predict(dihedral_group(5), 3, 6)
    assert isinstance(score, float)


def test_group_guided_meta_search_smoke() -> None:
    # Untrained model, generous slack so nothing is wrongly pruned: the
    # search must still run and return a well-formed result dict.
    random.seed(0)
    model = GroupPromisePredictor(hidden_dim=16, num_layers=2)
    result = group_guided_meta_search(
        k=3,
        g_target=4,
        model=model,
        max_group_order=16,
        num_random_trials=120,
        num_tabu_iters=80,
        prune_slack=99.0,
        num_workers=1,
        verbose=False,
    )
    assert "groups_kept" in result
    assert "groups_pruned" in result
    assert "elapsed" in result
