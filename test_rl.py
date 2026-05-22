from ai.cage.rl.env import CageConstructionEnv


def test_rl_environment_smoke_run() -> None:
    env = CageConstructionEnv(k=3, g=5, n_min=10, n_max=10, max_steps=1)
    _ = env.reset(num_nodes=10)

    _, _, done, info = env.step(0)

    assert done is True
    assert info["done_reason"] == "max_steps"
