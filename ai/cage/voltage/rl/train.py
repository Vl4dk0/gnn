"""PPO training for voltage assignment RL.

Usage:
    python -m ai.cage.voltage.rl.train --steps 50000
    python -m ai.cage.voltage.rl.train --steps 200000 --hidden-dim 128
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import sys
import time
from collections import defaultdict, deque
from typing import cast

import numpy as np
import torch
import torch.optim as optim
from rich.console import Console
from rich.live import Live
from torch.distributions import Categorical
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from ai.cage.ppo_utils import compute_gae
from ai.cage.voltage.rl.env import VoltageAssignmentEnv
from ai.cage.voltage.rl.model import VoltageActorCritic
from ai.utils.device import configure_torch_device


def train_voltage_ppo(
    total_timesteps: int = 100000,
    hidden_dim: int = 64,
    num_layers: int = 3,
    dropout: float = 0.1,
    lr: float = 3e-4,
    update_interval: int = 512,
    randomize: bool = True,
    entropy_coef: float = 0.05,
    max_action_dim: int = 100,
    log_seconds: float = 3.0,
    name: str = "voltage_ppo",
    conv_type: str = "gine",
    heads: int = 4,
    save_every: int = 50_000,
) -> VoltageActorCritic:
    """Train PPO agent for voltage assignment."""
    _ = configure_torch_device()
    console = Console()
    is_tty = sys.stdout.isatty()

    env = VoltageAssignmentEnv(k=3, g_target=5, randomize=randomize)
    sample_obs = env.reset()
    input_dim = env.get_input_dim()

    agent = VoltageActorCritic(
        input_dim=input_dim,
        edge_feat_dim=2,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        max_action_dim=max_action_dim,
        conv_type=conv_type,
        heads=heads,
    )

    optimizer = optim.Adam(agent.parameters(), lr=lr)

    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    value_coef = 0.5

    console.print("Voltage Assignment PPO Training")
    console.print(f"  Hidden: {hidden_dim}, Layers: {num_layers}, LR: {lr}")
    console.print(f"  Steps: {total_timesteps}, Update: {update_interval}")
    console.print(f"  Entropy: {entropy_coef}, Randomize: {randomize}")
    console.print("")

    obs_buffer: list[Data] = []
    action_buffer: list[int] = []
    log_prob_buffer: list[torch.Tensor] = []
    value_buffer: list[float] = []
    reward_buffer: list[float] = []
    done_buffer: list[bool] = []
    action_dim_buffer: list[int] = []

    global_step = 0
    episode_rewards: deque[float] = deque(maxlen=50)
    episode_girths: deque[int] = deque(maxlen=50)
    kg_stats: dict[tuple[int, int], dict[str, float]] = defaultdict(
        lambda: {"episodes": 0.0, "successes": 0.0, "reward_sum": 0.0, "girth_sum": 0.0}
    )
    episode_idx = 0

    obs = sample_obs
    current_ep_reward = 0.0
    start_time = time.time()
    best_avg_reward = -float("inf")
    best_avg_reward_by_stage: dict[int, float] = {}
    best_model_state: dict[str, torch.Tensor] | None = None
    last_live_log_time = start_time
    last_periodic_step = 0
    save_history: list[dict[str, object]] = []
    canonical_written = False

    save_dir = os.path.join("ai", "trained", "cage", name)
    os.makedirs(save_dir, exist_ok=True)

    def _build_info() -> dict[str, object]:
        return {
            "model_type": "voltage_actor_critic",
            "model_id": name,
            "task": "cage",
            "training": {
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "conv_type": conv_type,
                "heads": heads,
                "max_action_dim": max_action_dim,
                "steps": total_timesteps,
                "learning_rate": lr,
                "update_interval": update_interval,
                "entropy_coef": entropy_coef,
                "randomize": randomize,
                "save_every": save_every,
            },
            "metrics": {
                "best_avg_reward": round(best_avg_reward, 2),
                "best_avg_reward_by_stage": {
                    str(s): round(r, 2) for s, r in best_avg_reward_by_stage.items()
                },
                "final_step": global_step,
                "final_episode": episode_idx,
                "final_unlocked": env.unlocked,
            },
            "save_history": save_history,
        }

    def _save_checkpoint(suffix: str, trigger: str) -> None:
        nonlocal canonical_written
        path = os.path.join(save_dir, f"weights{suffix}.pt")
        cpu_state = {k: v.detach().cpu().clone() for k, v in agent.state_dict().items()}
        torch.save(cpu_state, path)
        if suffix == "":
            canonical_written = True
        save_history.append(
            {
                "trigger": trigger,
                "suffix": suffix,
                "step": global_step,
                "episode": episode_idx,
                "stage": env.unlocked,
            }
        )
        with open(os.path.join(save_dir, "info.json"), "w") as f:
            json.dump(_build_info(), f, indent=2)
        console.print(
            f"  [cyan]checkpoint saved[/] suffix={suffix or '(best)'} trigger={trigger} "
            f"step={global_step} stage={env.unlocked}"
        )

    # Convert SIGTERM (SLURM time-limit kill) into a graceful exit so the
    # finally block runs and a final checkpoint is written.
    def _sigterm_handler(_signum: int, _frame: object) -> None:
        raise SystemExit(0)

    _ = signal.signal(signal.SIGTERM, _sigterm_handler)

    live_view = Live(
        "starting...", console=console, refresh_per_second=4, transient=True
    )
    if is_tty:
        live_view.start()

    try:
        while global_step < total_timesteps:
            _ = agent.eval()

            for _buf_step in range(update_interval):
                if global_step >= total_timesteps:
                    break

                with torch.no_grad():
                    action, log_prob, value = agent.get_action(
                        obs, action_dim=env.get_action_dim()
                    )

                next_obs, reward, done, info = env.step(action)

                obs_buffer.append(obs)
                action_buffer.append(action)
                log_prob_buffer.append(log_prob)
                value_buffer.append(value.item())
                reward_buffer.append(reward)
                done_buffer.append(done)
                action_dim_buffer.append(env.get_action_dim())

                obs = next_obs
                global_step += 1
                current_ep_reward += reward

                if done:
                    episode_rewards.append(current_ep_reward)
                    girth = int(info.get("girth", 0))
                    episode_girths.append(girth)
                    episode_idx += 1

                    kg_key = (env.k, env.g_target)
                    kg_stats[kg_key]["episodes"] += 1
                    kg_stats[kg_key]["successes"] += (
                        1.0 if bool(info.get("success", False)) else 0.0
                    )
                    kg_stats[kg_key]["reward_sum"] += current_ep_reward
                    kg_stats[kg_key]["girth_sum"] += girth

                    success = bool(info.get("success", False))
                    unlocked_new = env.report_result(success)
                    if unlocked_new:
                        console.print(
                            f"[green]Unlocked stage {env.unlocked}/{len(env.configs)} "
                            f"-> {env.configs[env.unlocked - 1]}[/]"
                        )
                        _save_checkpoint(f"_stage{env.unlocked}", "stage_unlock")

                    current_ep_reward = 0.0
                    obs = env.reset()

                # Periodic safety checkpoint, regardless of metric progress.
                if save_every > 0 and global_step - last_periodic_step >= save_every:
                    last_periodic_step = global_step
                    _save_checkpoint("_periodic", "periodic")

                # Live display
                if (
                    is_tty
                    and log_seconds > 0
                    and (time.time() - last_live_log_time) >= log_seconds
                ):
                    last_live_log_time = time.time()
                    avg_r = sum(episode_rewards) / max(len(episode_rewards), 1)
                    avg_g = sum(episode_girths) / max(len(episode_girths), 1)
                    elapsed = time.time() - start_time
                    fps = int(global_step / elapsed) if elapsed > 0 else 0
                    status = (
                        f"Step {global_step}/{total_timesteps} | "
                        f"Eps: {episode_idx} | "
                        f"Avg reward: {avg_r:.2f} | "
                        f"Avg girth: {avg_g:.1f} | "
                        f"FPS: {fps} | "
                        f"Unlocked: {env.unlocked}/{len(env.configs)}"
                    )
                    live_view.update(status, refresh=True)

            if not obs_buffer:
                break

            # PPO Update
            _ = agent.train()

            with torch.no_grad():
                _, next_value = cast(tuple[torch.Tensor, torch.Tensor], agent(obs))
                next_value_float = float(next_value.item())

            values = value_buffer + [next_value_float]
            advantages = compute_gae(
                reward_buffer, values, done_buffer, gamma, gae_lambda
            )
            returns = advantages + torch.tensor(value_buffer)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            b_actions = torch.tensor(action_buffer)
            b_log_probs = torch.stack(log_prob_buffer)
            batch_size = len(obs_buffer)
            minibatch_size = 64

            for _ppo_epoch in range(4):
                indices = np.random.permutation(batch_size)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = indices[start:end]

                    optimizer.zero_grad()
                    for _idx in cast(list[int], mb_inds.tolist()):
                        i = int(_idx)
                        data = obs_buffer[i]
                        logits, value = cast(
                            tuple[torch.Tensor, torch.Tensor], agent(data)
                        )

                        # Mask to the group order from when this sample was collected
                        action_dim = action_dim_buffer[i]
                        if action_dim < agent.max_action_dim:
                            mask = torch.ones(
                                logits.shape[-1], dtype=torch.bool, device=logits.device
                            )
                            mask[:action_dim] = False
                            logits = logits.masked_fill(mask, -1e9)

                        dist = Categorical(logits=logits)
                        new_log_prob = cast(torch.Tensor, dist.log_prob(b_actions[i]))
                        entropy = dist.entropy()

                        ratio = torch.exp(new_log_prob - b_log_probs[i])
                        surr1 = ratio * advantages[i]
                        surr2 = (
                            torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
                            * advantages[i]
                        )
                        policy_loss = -torch.min(surr1, surr2)
                        value_loss: torch.Tensor = (
                            0.5 * (returns[i] - value.squeeze()) ** 2
                        )
                        loss: torch.Tensor = (
                            policy_loss
                            + value_coef * value_loss
                            - entropy_coef * entropy
                        )
                        _ = loss.backward()  # pyright: ignore[reportUnknownMemberType]

                    for p in agent.parameters():
                        if p.grad is not None:
                            p.grad /= len(mb_inds)
                    _ = optimizer.step()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

            # Clear buffers
            obs_buffer = []
            action_buffer = []
            log_prob_buffer = []
            value_buffer = []
            reward_buffer = []
            done_buffer = []
            action_dim_buffer = []

            # Print rollout stats
            avg_rew = (
                sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0
            )
            avg_girth = (
                sum(episode_girths) / len(episode_girths) if episode_girths else 0.0
            )
            elapsed = time.time() - start_time
            fps = int(global_step / elapsed) if elapsed > 0 else 0

            console.print(
                f"Step {global_step:6d} | Eps: {episode_idx:4d} | "
                f"Avg rew: {avg_rew:+7.2f} | Avg girth: {avg_girth:4.1f} | "
                f"FPS: {fps} | Unlocked: {env.unlocked}/{len(env.configs)}"
            )

            # Per (k,g) stats
            for (k_val, g_val), stats in sorted(kg_stats.items()):
                eps = max(1.0, stats["episodes"])
                succ_rate = 100.0 * stats["successes"] / eps
                avg_g = stats["girth_sum"] / eps
                console.print(
                    f"    ({k_val},{g_val}): {int(stats['episodes']):4d} eps | "
                    f"{succ_rate:5.1f}% success | avg_girth={avg_g:.1f}"
                )

            stage = env.unlocked
            stage_best = best_avg_reward_by_stage.get(stage, -float("inf"))
            if avg_rew > stage_best and episode_idx > 5:
                best_avg_reward_by_stage[stage] = avg_rew
                if avg_rew > best_avg_reward:
                    best_avg_reward = avg_rew
                    best_model_state = cast(
                        dict[str, torch.Tensor],
                        copy.deepcopy(agent.state_dict()),
                    )
                console.print(
                    f"  [green]New best for stage {stage}! avg_reward={avg_rew:.2f}[/]"
                )
                _save_checkpoint("", "stage_best")

    finally:
        if is_tty:
            live_view.stop()
        # Always save a final snapshot of the current agent — runs on
        # SIGTERM, exception, or normal exit. The "best" weights.pt may
        # be older if the agent regressed, so keep both.
        _save_checkpoint("_final", "final")
        # Guarantee a canonical weights.pt exists for THIS run. SystemExit
        # raised from the SIGTERM handler propagates through finally and
        # skips any post-finally code, so this fallback HAS to live in
        # finally. The flag is in-run state, not a disk check, so we don't
        # accidentally treat a stale weights.pt left by a previous run that
        # reused the same --name as if it belonged to this run.
        if not canonical_written:
            _save_checkpoint("", "fallback_no_best")

    save_path = os.path.join(save_dir, "weights.pt")
    # Restore the best in-memory state (if any) and refresh weights.pt with
    # it. This only runs on normal exit; a SystemExit from SIGTERM bypasses
    # this branch, but the finally above already ensured weights.pt exists.
    if best_model_state is not None:
        _ = agent.load_state_dict(best_model_state)
        _save_checkpoint("", "best_restored")

    console.print(f"\nModel saved to {save_path}")
    console.print(f"Best avg reward: {best_avg_reward:.2f}")

    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train voltage assignment PPO agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "--steps", type=int, default=100000, help="Total training steps"
    )
    _ = parser.add_argument(
        "--hidden-dim", type=int, default=64, help="Hidden dimension"
    )
    _ = parser.add_argument("--num-layers", type=int, default=3, help="GINEConv layers")
    _ = parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    _ = parser.add_argument(
        "--update-interval", type=int, default=512, help="PPO update interval"
    )
    _ = parser.add_argument(
        "--entropy-coef", type=float, default=0.05, help="Entropy coefficient"
    )
    _ = parser.add_argument(
        "--name", type=str, default="voltage_ppo", help="Model name for saving"
    )
    _ = parser.add_argument(
        "--conv-type",
        type=str,
        default="gine",
        choices=["gine", "gps"],
        help="Convolution type (gine or gps)",
    )
    _ = parser.add_argument(
        "--heads", type=int, default=4, help="Attention heads (GPS only)"
    )
    _ = parser.add_argument(
        "--no-random", action="store_true", help="Disable curriculum"
    )
    _ = parser.add_argument(
        "--log", type=float, default=3.0, help="Live log interval (seconds)"
    )
    _ = parser.add_argument(
        "--save-every",
        type=int,
        default=50_000,
        help="Periodic checkpoint interval in env steps (0 disables)",
    )
    args = parser.parse_args()

    _ = train_voltage_ppo(
        total_timesteps=cast(int, args.steps),
        hidden_dim=cast(int, args.hidden_dim),
        num_layers=cast(int, args.num_layers),
        lr=cast(float, args.lr),
        update_interval=cast(int, args.update_interval),
        entropy_coef=cast(float, args.entropy_coef),
        randomize=not cast(bool, args.no_random),
        log_seconds=cast(float, args.log),
        name=cast(str, args.name),
        conv_type=cast(str, args.conv_type),
        heads=cast(int, args.heads),
        save_every=cast(int, args.save_every),
    )
