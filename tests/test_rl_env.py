import pytest

from ai.cage.rl.env import CageConstructionEnv


def _edge_action_index(env: CageConstructionEnv, target: tuple[int, int]) -> int:
    normalized = tuple(sorted(target))
    for idx in range(env.get_action_dim()):
        if env.idx_to_edge(idx) == normalized:
            return idx
    raise AssertionError(f"edge {target} not found")


def test_rl_env_reset_builds_fixed_size_empty_observation() -> None:
    env = CageConstructionEnv(k=3, g=5, n_min=10, n_max=10, max_steps=5)
    obs = env.reset(num_nodes=10)

    assert env.num_nodes == 10
    assert env.graph.number_of_nodes() == 10
    assert env.graph.number_of_edges() == 0
    assert env.get_action_dim() == 45
    assert obs.x is not None
    assert tuple(obs.x.shape) == (10, env.get_input_dim())


def test_rl_env_rejects_disconnected_inactive_edge_after_first_edge() -> None:
    env = CageConstructionEnv(k=3, g=5, n_min=10, n_max=10, max_steps=5)
    _ = env.reset(num_nodes=10)

    _, _, _, first_info = env.step(_edge_action_index(env, (0, 1)))
    _, reward, done, second_info = env.step(_edge_action_index(env, (2, 3)))

    assert first_info["accepted"] is True
    assert second_info["accepted"] is False
    assert second_info["action_type"] == "edge_invalid_add"
    assert second_info["done_reason"] == "add_rule_or_constraint"
    assert reward < 0
    assert done is False


def test_rl_env_action_mask_rejects_edges_that_create_short_cycles() -> None:
    env = CageConstructionEnv(k=3, g=5, n_min=10, n_max=10, max_steps=5)
    _ = env.reset(num_nodes=10)
    _ = env.graph.add_edge(0, 1)
    _ = env.graph.add_edge(1, 2)

    mask = env.get_valid_action_mask()

    assert not bool(mask[_edge_action_index(env, (0, 2))].item())
    assert bool(mask[_edge_action_index(env, (2, 3))].item())


def test_edge_action_index_helper_fails_for_impossible_edge() -> None:
    env = CageConstructionEnv(k=3, g=5, n_min=10, n_max=10, max_steps=1)
    _ = env.reset(num_nodes=10)

    with pytest.raises(AssertionError, match="not found"):
        _ = _edge_action_index(env, (0, 99))
