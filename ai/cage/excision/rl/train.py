"""PPO training for the GNN-guided tree-excision repair policy.

Usage:
    uv run python -m ai.cage.excision.rl.train --episodes 50 --g-target 5 --depth 1

The agent learns to repair degree-deficient graphs produced by BFS-excision of
a known (k,g)-graph.  Excision depth is fixed per run and the bank is filtered
to graphs whose girth is at least --g-target.

Starting graph bank (the four smallest cubic cages, girths 5..8):
  - Petersen graph       (3-regular, girth 5, 10 vertices) — nx.petersen_graph()
  - Heawood graph        (3-regular, girth 6, 14 vertices) — nx.heawood_graph()
  - McGee graph          (3-regular, girth 7, 24 vertices) — LCF [12, 7, -7]^8
  - Tutte-Coxeter graph  (3-regular, girth 8, 30 vertices) — LCF [-13,-9,7,-7,9,13]^5

  Only graphs with girth >= g_target are kept: a source with girth < g_target
  can never be repaired back to a valid (k, g_target)-graph.

Saves to: ai/trained/excision/excision/
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
from typing import Any, TypeAlias, cast

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
from ai.cage.voltage.base_graphs import dumbbell
from ai.cage.voltage.groups import cyclic_group
from ai.cage.voltage.lift import build_lift
from ai.utils.device import configure_torch_device
from ai.utils.structural_features import structural_feature_dim

Instance: TypeAlias = "tuple[nx.Graph[int], list[int], dict[int, int]]"


# ---------------------------------------------------------------------------
# Graph bank
# ---------------------------------------------------------------------------


def _remove_matching_instance(
    H: nx.Graph[int], match_size: int, rng: random.Random
) -> Instance | None:
    """Remove a random partial matching from a valid (k,g)-graph H.

    Returns (reduced, deficient, deficiency_levels) where ``reduced = H - M`` for
    a matching M of up to ``match_size`` edges. Because H has girth >= g, every
    removed edge is a *legal* repair edge of reduced (its cycle in H has length
    >= g), so re-adding M is a guaranteed witness solution. Deficient vertices
    are exactly the endpoints of M, each with deficiency 1.

    Returns None if no non-empty matching could be drawn (degenerate H).
    """
    edges = list(H.edges())
    rng.shuffle(edges)
    matching: list[tuple[int, int]] = []
    used: set[int] = set()
    for u, v in edges:
        if u in used or v in used:
            continue
        matching.append((u, v))
        used.add(u)
        used.add(v)
        if len(matching) >= match_size:
            break

    if not matching:
        return None

    reduced: nx.Graph[int] = H.copy()
    reduced.remove_edges_from(matching)
    deficient = sorted(used)
    deficiency_levels = {v: 1 for v in deficient}
    return reduced, deficient, deficiency_levels


def _synthetic_base_graphs(g_target: int) -> list[nx.Graph[int]]:
    """Valid (k,g_target)-source graphs for the synthetic generator.

    Uses the known cubic cages whose girth is >= g_target. Every such graph is a
    valid (3, g_target)-graph, so removing a matching yields a solvable instance.
    """
    candidates = [
        nx.petersen_graph(),  # (3,5)-cage
        nx.heawood_graph(),  # (3,6)-cage
        nx.LCF_graph(24, [12, 7, -7], 8),  # McGee, (3,7)-cage
        nx.LCF_graph(30, [-13, -9, 7, -7, 9, 13], 5),  # Tutte-Coxeter, (3,8)-cage
    ]
    graphs: list[nx.Graph[int]] = []
    for graph in candidates:
        if nx.girth(graph) < g_target:
            continue
        graphs.append(cast("nx.Graph[int]", nx.convert_node_labels_to_integers(graph)))
    return graphs


def _find_lift_graph(g_target: int) -> nx.Graph[int] | None:
    """Search dumbbell(3) cyclic lifts for a 3-regular graph with girth >= g_target.

    Returns the first lift found (connected, 3-regular, girth >= g_target), or
    None if the bounded search exhausts. Lifts are bigger than the cages, giving
    larger, more varied solvable instances.
    """
    base = dumbbell(3)
    for n in range(g_target, 40):
        group = cyclic_group(n)
        # One tree edge can be fixed to 0; search the two free voltages.
        for a in range(1, n):
            for b in range(a + 1, n):
                lift = build_lift(base, group, [0, a, b])
                if (
                    nx.is_connected(lift)
                    and all(d == 3 for _, d in lift.degree())
                    and nx.girth(lift) >= g_target
                ):
                    return cast("nx.Graph[int]", lift)
    return None


def _build_synthetic_bank(
    g_target: int, match_size: int, num_instances: int, seed: int
) -> list[Instance]:
    """Guaranteed-solvable instances by matching-removal from valid (k,g)-graphs."""
    bases = _synthetic_base_graphs(g_target)
    if not bases:
        raise ValueError(
            f"No cubic cage has girth >= g_target={g_target} (max available girth 8)."
        )
    rng = random.Random(seed)
    instances: list[Instance] = []
    attempts = 0
    while len(instances) < num_instances and attempts < num_instances * 10:
        attempts += 1
        H = rng.choice(bases)
        inst = _remove_matching_instance(H, match_size, rng)
        if inst is not None:
            instances.append(inst)
    return instances


def _build_lift_bank(
    g_target: int, match_size: int, num_instances: int, seed: int
) -> list[Instance]:
    """Guaranteed-solvable instances by matching-removal from a voltage lift."""
    H = _find_lift_graph(g_target)
    if H is None:
        raise ValueError(
            f"No dumbbell(3) cyclic lift with girth >= g_target={g_target} found "
            "in the bounded search."
        )
    rng = random.Random(seed)
    instances: list[Instance] = []
    attempts = 0
    while len(instances) < num_instances and attempts < num_instances * 10:
        attempts += 1
        inst = _remove_matching_instance(H, match_size, rng)
        if inst is not None:
            instances.append(inst)
    return instances


def _build_graph_bank(
    g_target: int, depth: int
) -> list[tuple[nx.Graph[int], list[int], dict[int, int]]]:
    """Build a list of (reduced_graph, deficient, deficiency_levels) instances.

    Tries every vertex of every bank graph as a root and returns only those
    excisions that produce at least one deficient vertex (otherwise the repair
    task is trivial).  Only bank graphs whose girth is at least ``g_target`` are
    used: a source with girth < g_target can never be repaired back to a valid
    (k, g_target)-graph, so its excisions would be guaranteed failures that
    poison training.
    """
    # The four smallest cubic cages, one per girth 5..8.  McGee and
    # Tutte-Coxeter are built from LCF notation since NetworkX has no direct
    # constructor for them.
    candidate_graphs = [
        nx.petersen_graph(),  # (3,5)-cage, 10 vertices
        nx.heawood_graph(),  # (3,6)-cage, 14 vertices
        nx.LCF_graph(24, [12, 7, -7], 8),  # McGee, (3,7)-cage, 24 vertices
        nx.LCF_graph(30, [-13, -9, 7, -7, 9, 13], 5),  # Tutte-Coxeter, (3,8)-cage
    ]

    bank_graphs: list[nx.Graph[int]] = []
    for graph in candidate_graphs:
        if nx.girth(graph) < g_target:
            continue
        relabeled = nx.convert_node_labels_to_integers(graph)
        bank_graphs.append(cast("nx.Graph[int]", relabeled))

    if not bank_graphs:
        msg = f"No bank graph has girth >= g_target={g_target} (largest available girth is 8, from Tutte-Coxeter)."
        raise ValueError(msg)

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
    instance_source: str = "synthetic",
    match_size: int = 2,
    num_instances: int = 64,
    max_grad_norm: float = 0.5,
    entropy_coef_final: float = 0.005,
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
    console.print(f"  instance_source={instance_source}, match_size={match_size}")

    # Build instance bank.  `synthetic`/`lifts` produce guaranteed-solvable
    # matching-removal instances (a known witness exists); `cages` keeps the
    # original tree-excision bank for eval/comparison.
    if instance_source == "synthetic":
        instances = _build_synthetic_bank(g_target, match_size, num_instances, seed)
    elif instance_source == "lifts":
        instances = _build_lift_bank(g_target, match_size, num_instances, seed)
    elif instance_source == "cages":
        instances = _build_graph_bank(g_target, depth)
    else:
        raise ValueError(
            f"Unknown instance_source={instance_source!r} "
            "(expected synthetic|lifts|cages)."
        )
    if not instances:
        raise RuntimeError(
            f"No instances for source={instance_source}, g_target={g_target}, "
            f"depth={depth}."
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
    last_checkpoint_time = start_time
    save_dir = os.path.join("ai", "trained", "excision", "excision")

    def _save_checkpoint() -> None:
        """Persist the best policy seen so far (weights + info) to disk.

        Called periodically during training and once at the end so a run cut
        off by the SLURM wall-time still leaves the best model on disk.
        """
        if best_state is None:
            return
        os.makedirs(save_dir, exist_ok=True)
        torch.save(best_state, os.path.join(save_dir, "weights.pt"))
        info: dict[str, Any] = {
            "model_type": "repair_actor_critic",
            "model_id": "excision",
            "task": "excision",
            "training": {
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "g_target": g_target,
                "depth": depth,
                "instance_source": instance_source,
                "match_size": match_size,
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

                # Linearly decay the entropy coefficient from entropy_coef
                # (high, encourages exploration early) to entropy_coef_final
                # (low, sharpens the policy late in training).
                progress = min(episode_idx / max(total_episodes, 1), 1.0)
                cur_entropy_coef = (
                    entropy_coef + (entropy_coef_final - entropy_coef) * progress
                )

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
                                - cur_entropy_coef * entropy
                            )
                            total_loss = total_loss + step_loss
                            valid += 1

                        if valid > 0:
                            avg_loss = total_loss / valid
                            avg_loss.backward()  # pyright: ignore[reportUnknownMemberType]
                            _ = torch.nn.utils.clip_grad_norm_(
                                agent.parameters(), max_grad_norm
                            )
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

        # Track best.  Update from the very first completed episode so short runs
        # still save a meaningful checkpoint (the old >= 5 threshold left
        # best_success_rate stuck at the -1.0 sentinel for runs under 5 episodes).
        if success_flags:
            sr = sum(success_flags) / len(success_flags)
            if sr > best_success_rate:
                best_success_rate = sr
                best_state = {
                    k: v.detach().cpu().clone() for k, v in agent.state_dict().items()
                }
                # Flush the new best to disk, throttled, so a wall-time kill
                # still leaves the best model persisted.
                now = time.time()
                if now - last_checkpoint_time > 120:
                    _save_checkpoint()
                    last_checkpoint_time = now

    # Final save — captures the last improvement even if within the throttle
    # window, and writes the final info.json.
    _save_checkpoint()
    if best_state is not None:
        _ = agent.load_state_dict(best_state)

    console.print(f"\n[green]Saved to {save_dir}[/]")
    console.print(f"Best success rate: {best_success_rate:.2%}")
    return agent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entrypoint for the excision-repair PPO trainer."""
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
    _ = parser.add_argument(
        "--instance-source",
        type=str,
        default="synthetic",
        choices=["synthetic", "lifts", "cages"],
        help="Repair-instance source: synthetic/lifts are guaranteed-solvable "
        "matching-removal instances; cages is the original tree-excision bank.",
    )
    _ = parser.add_argument(
        "--match-size",
        type=int,
        default=2,
        help="Matching size removed to build a solvable instance (difficulty knob).",
    )
    _ = parser.add_argument(
        "--num-instances",
        type=int,
        default=64,
        help="Number of synthetic/lift instances to pre-generate for the bank.",
    )
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
        instance_source=cast(str, args.instance_source),
        match_size=cast(int, args.match_size),
        num_instances=cast(int, args.num_instances),
    )


if __name__ == "__main__":
    main()
