"""Shared PPO helpers for cage RL training scripts."""

from __future__ import annotations

import torch


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> torch.Tensor:
    """Generalized Advantage Estimation.

    `values` must be length len(rewards) + 1: the last entry is the
    bootstrap value of the final state.
    """
    advantages: list[float] = []
    gae = 0.0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * (1 - int(dones[i])) - values[i]
        gae = delta + gamma * lam * (1 - int(dones[i])) * gae
        advantages.insert(0, gae)
    return torch.tensor(advantages, dtype=torch.float)
