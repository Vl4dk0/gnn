"""PPO training for the GNN-guided tree-excision repair policy.

Usage:
    uv run python -m ai.cage.excision.rl.train --episodes 50 --g-target 5 --depth 1

The agent learns to repair degree-deficient graphs produced by BFS-excision of
a known (k,g)-graph.  No curriculum: the starting bank is fixed (Petersen +
Heawood) and excision depth is fixed per run.  This keeps the smoke-scale
training simple and honest.

Starting graph bank:
  - Petersen graph  (3-regular, girth 5, 10 vertices) — nx.petersen_graph()
  - Heawood graph   (3-regular, girth 6, 14 vertices) — nx.heawood_graph()

  Note: NetworkX does not include the Tutte-Coxeter (Levi) graph directly;
  using Petersen + Heawood gives two distinct girth targets for smoke-scale
  runs.  Tutte-Coxeter can be added later via a hardcoded edge list.

Saves to: ai/trained/excision_repair/excision_repair_policy/
    weights.pt   — best (or final) policy weights
    info.json    — architecture + feature config for later reload
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import deque
from typing import Any, cast

import networkx as nx
import numpy as np

from ai.cage.utils import compute_gae
import torch
import torch.optim as optim
from rich.console import Console
from torch.distributions import Categorical
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]

from ai.cage.excision.excise import excise_tree
from ai.cage.excision.rl.env import RepairEnv
from ai.cage.excision.rl.model import RepairActorCritic
from ai.utils.device import configure_torch_device
from ai.utils.structural_features import structural_feature_dim


# ---------------------------------------------------------------------------
# Graph bank
# ---------------------------------------------------------------------------


def _build_graph_bank(
    g_target: int, depth: int
) -> list[tuple[nx.Graph[int], list[int], dict[int, int]]]:
    """Build a list of (reduced_graph, deficient, deficiency_levels) instances.

    Tries every vertex of every bank graph as a root and returns only those
    excisions that produce at least one deficient vertex (otherwise the repair
    task is trivial).  If g_target doesn't match any bank graph, uses all.
    """
    bank_graphs: list[nx.Graph[int]] = []

    for graph in (nx.petersen_graph(), nx.heawood_graph()):
        relabeled = nx.convert_node_labels_to_integers(graph)
        bank_graphs.append(cast("nx.Graph[int]", relabeled))

    instances: list[tuple[nx.Graph[int], list[int], dict[int, int]]] = []
    for G in bank_graphs:
        for root in G.nodes():
            reduced, deficient, deficiency_levels = excise_tree(G, root, depth)
            if deficient and reduced.number_of_nodes() > 0:
                instances.append((reduced, deficient, deficiency_levels))

    return instances


# ---------------------------------------------------------------------------
# GAE computation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train_repair_ppo(
    total_episodes: int = 100,
    g_target: int = 5,
    depth: int = 1,
    seed: int = 42,
    hidden_dim: int = 64,
    num_layers: int = 3,
    dropout: float = 0.1,
    lr: float = 3e-4,
    cycle_lengths: list[int] | None = None,
    rwpe_dim: int = 8,
    update_interval: int = 32,
    entropy_coef: float = 0.05,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
) -> RepairActorCritic:
    """Train PPO agent for graph repair.

    Returns the best (or final) trained policy.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    _ = configure_torch_device()
    console = Console()

    if cycle_lengths is None:
        cycle_lengths = [3, 4, 5, 6, 7, 8]

    console.print("[bold]Excision Repair PPO Training[/]")
    console.print(f"  g_target={g_target}, depth={depth}, episodes={total_episodes}")
    console.print(
        f"  hidden_dim={hidden_dim}, cycle_lengths={cycle_lengths}, rwpe_dim={rwpe_dim}"
    )

    # Build instance bank
    instances = _build_graph_bank(g_target, depth)
    if not instances:
        raise RuntimeError(
            f"No valid excision instances for g_target={g_target}, depth={depth}. "
            "Check that the bank graphs have girth >= g_target."
        )
    console.print(f"  Bank size: {len(instances)} instances")

    # Determine input_dim from a sample observation
    sample_reduced, sample_deficient, _ = instances[0]
    sample_env = RepairEnv(
        sample_reduced,
        sample_deficient,
        g_target,
        cycle_lengths=cycle_lengths,
        rwpe_dim=rwpe_dim,
    )
    sample_obs = sample_env.reset()
    input_dim = int(cast(torch.Tensor, sample_obs.x).size(1))
    console.print(f"  input_dim={input_dim} (base 5 + structural {input_dim - 5})")

    # Instantiate policy
    agent = RepairActorCritic(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    optimizer = optim.Adam(agent.parameters(), lr=lr)

    # Training bookkeeping
    episode_rewards: deque[float] = deque(maxlen=50)
    success_flags: deque[bool] = deque(maxlen=50)
    best_success_rate = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    start_time = time.time()

    # PPO buffers
    obs_buf: list[Data] = []
    action_buf: list[int] = []
    logp_buf: list[torch.Tensor] = []
    val_buf: list[float] = []
    rew_buf: list[float] = []
    done_buf: list[bool] = []
    legal_pairs_buf: list[list[tuple[int, int]]] = []
    node_ids_buf: list[list[int]] = []

    episode_idx = 0

    while episode_idx < total_episodes:
        # Pick a random instance
        reduced, deficient, _ = random.choice(instances)
        env = RepairEnv(
            reduced,
            deficient,
            g_target,
            cycle_lengths=cycle_lengths,
            rwpe_dim=rwpe_dim,
        )
        obs = env.reset()
        ep_reward = 0.0
        ep_success = False

        _ = agent.eval()

        # Run one episode collecting transitions
        ep_obs: list[Data] = []
        ep_actions: list[int] = []
        ep_logps: list[torch.Tensor] = []
        ep_vals: list[float] = []
        ep_rews: list[float] = []
        ep_dones: list[bool] = []
        ep_legal_pairs: list[list[tuple[int, int]]] = []
        ep_node_ids: list[list[int]] = []

        done = False
        while not done:
            legal = env.legal_actions()
            if not legal:
                # No legal actions — episode terminates as a failure.
                # Record a synthetic terminal transition for GAE bootstrap.
                with torch.no_grad():
                    val_t = agent.get_value(obs)
                ep_rews.append(env.FAILURE_REWARD)
                ep_dones.append(True)
                ep_vals.append(float(val_t.item()))
                ep_obs.append(obs)
                # Dummy action index 0; excluded from PPO update (empty pairs list).
                ep_actions.append(0)
                ep_logps.append(torch.tensor(0.0))
                ep_legal_pairs.append([])
                ep_node_ids.append(list(cast(list[int], obs.node_ids)))  # pyright: ignore[reportAttributeAccessIssue]
                ep_success = False
                done = True
                break

            with torch.no_grad():
                node_ids: list[int] = list(cast(list[int], obs.node_ids))  # pyright: ignore[reportAttributeAccessIssue]
                action, log_prob, val_t = agent.get_action(obs, legal, node_ids)

            action_idx = legal.index(action)

            next_obs, reward, done, info = env.step(action)
            ep_success = bool(info.get("success", False))

            ep_obs.append(obs)
            ep_actions.append(action_idx)
            ep_logps.append(log_prob)
            ep_vals.append(float(val_t.item()))
            ep_rews.append(reward)
            ep_dones.append(done)
            ep_legal_pairs.append(legal)
            ep_node_ids.append(node_ids)

            obs = next_obs
            ep_reward += reward

        episode_rewards.append(ep_reward)
        success_flags.append(ep_success)
        episode_idx += 1

        # Extend buffers
        obs_buf.extend(ep_obs)
        action_buf.extend(ep_actions)
        logp_buf.extend(ep_logps)
        val_buf.extend(ep_vals)
        rew_buf.extend(ep_rews)
        done_buf.extend(ep_dones)
        legal_pairs_buf.extend(ep_legal_pairs)
        node_ids_buf.extend(ep_node_ids)

        # PPO update whenever buffer reaches update_interval steps
        if len(obs_buf) >= update_interval or episode_idx >= total_episodes:
            if obs_buf:
                _ = agent.train()

                with torch.no_grad():
                    next_val = float(agent.get_value(obs).item())

                values_ext = val_buf + [next_val]
                advantages = compute_gae(
                    rew_buf, values_ext, done_buf, gamma, gae_lambda
                )
                returns = advantages + torch.tensor(val_buf)
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                b_actions = torch.tensor(action_buf, dtype=torch.long)
                b_logps = torch.stack(logp_buf)
                N = len(obs_buf)

                for _ppo_epoch in range(4):
                    perm = np.random.permutation(N)
                    for start in range(0, N, 16):
                        mb_inds = perm[start : start + 16]
                        optimizer.zero_grad()
                        total_loss = torch.tensor(0.0)
                        valid = 0
                        for raw_i in mb_inds.tolist():
                            i = int(raw_i)
                            pairs = legal_pairs_buf[i]
                            nids = node_ids_buf[i]
                            if not pairs:
                                continue

                            logits = agent.get_action_logits(obs_buf[i], pairs, nids)
                            val_pred = agent.get_value(obs_buf[i])
                            dist_i = Categorical(logits=logits)

                            ai = b_actions[i].clamp(0, len(pairs) - 1)
                            new_logp = cast(torch.Tensor, dist_i.log_prob(ai))
                            entropy = cast(torch.Tensor, dist_i.entropy())

                            ratio = torch.exp(new_logp - b_logps[i])
                            surr1 = ratio * advantages[i]
                            surr2 = (
                                torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
                                * advantages[i]
                            )
                            pol_loss = -torch.min(surr1, surr2)
                            val_loss: torch.Tensor = (
                                0.5 * (returns[i] - val_pred.squeeze()) ** 2
                            )
                            step_loss: torch.Tensor = (
                                pol_loss
                                + value_coef * val_loss
                                - entropy_coef * entropy
                            )
                            total_loss = total_loss + step_loss
                            valid += 1

                        if valid > 0:
                            avg_loss = total_loss / valid
                            avg_loss.backward()  # pyright: ignore[reportUnknownMemberType]
                            _ = optimizer.step()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

                # Clear buffers
                obs_buf = []
                action_buf = []
                logp_buf = []
                val_buf = []
                rew_buf = []
                done_buf = []
                legal_pairs_buf = []
                node_ids_buf = []

        # Progress logging every 10 episodes
        if episode_idx % 10 == 0 or episode_idx == total_episodes:
            avg_r = sum(episode_rewards) / max(len(episode_rewards), 1)
            succ_rate = sum(success_flags) / max(len(success_flags), 1)
            elapsed = time.time() - start_time
            console.print(
                f"Ep {episode_idx:4d}/{total_episodes} | "
                f"avg_reward={avg_r:+6.2f} | "
                f"success_rate={succ_rate:.2%} | "
                f"elapsed={elapsed:.1f}s"
            )

        # Track best
        if len(success_flags) >= 5:
            sr = sum(success_flags) / len(success_flags)
            if sr > best_success_rate:
                best_success_rate = sr
                best_state = {
                    k: v.detach().cpu().clone() for k, v in agent.state_dict().items()
                }

    # Save
    save_dir = os.path.join(
        "ai", "trained", "excision_repair", "excision_repair_policy"
    )
    os.makedirs(save_dir, exist_ok=True)

    if best_state is not None:
        _ = agent.load_state_dict(best_state)

    weights_path = os.path.join(save_dir, "weights.pt")
    torch.save(
        {k: v.detach().cpu() for k, v in agent.state_dict().items()}, weights_path
    )

    info: dict[str, Any] = {
        "model_type": "repair_actor_critic",
        "model_id": "excision_repair_policy",
        "task": "excision_repair",
        "training": {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "g_target": g_target,
            "depth": depth,
            "total_episodes": total_episodes,
            "seed": seed,
            "lr": lr,
            "feature_config": {
                "cycle_lengths": cycle_lengths,
                "rwpe_dim": rwpe_dim,
            },
        },
        "metrics": {
            "best_success_rate": round(best_success_rate, 4),
            "final_episodes": episode_idx,
        },
    }
    with open(os.path.join(save_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    console.print(f"\n[green]Saved to {save_dir}[/]")
    console.print(f"Best success rate: {best_success_rate:.2%}")
    return agent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO repair policy for tree-excision",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument("--episodes", type=int, default=100, help="Total episodes")
    _ = parser.add_argument("--g-target", type=int, default=5, help="Target girth")
    _ = parser.add_argument("--depth", type=int, default=1, help="BFS excision depth")
    _ = parser.add_argument("--seed", type=int, default=42, help="Random seed")
    _ = parser.add_argument(
        "--hidden-dim", type=int, default=64, help="Hidden dimension"
    )
    _ = parser.add_argument(
        "--cycle-lengths",
        type=str,
        default="3,4,5,6,7,8",
        help="Comma-separated cycle lengths for structural features",
    )
    _ = parser.add_argument("--rwpe-dim", type=int, default=8, help="RWPE dimension")
    args = parser.parse_args()

    cycle_lengths_parsed: list[int] = [
        int(x) for x in cast(str, args.cycle_lengths).split(",")
    ]

    _ = train_repair_ppo(
        total_episodes=cast(int, args.episodes),
        g_target=cast(int, args.g_target),
        depth=cast(int, args.depth),
        seed=cast(int, args.seed),
        hidden_dim=cast(int, args.hidden_dim),
        cycle_lengths=cycle_lengths_parsed,
        rwpe_dim=cast(int, args.rwpe_dim),
    )
