import argparse
import copy
import json
import os
import sys
import time
from collections import Counter
from collections import deque
from collections import defaultdict
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]

from rich.console import Console
from rich.live import Live
from rich.table import Table

from ai.cage.rl.env import CageConstructionEnv
from ai.cage.rl.model import ActorCritic
from ai.models import MODEL_CLASSES
from ai.models.base import BaseGNN
from ai.registry import get_trained_dir, list_model_types, model_exists, save_model
from ai.utils.device import configure_torch_device
from ai.utils.r_neighborhood import apply_r_neighborhood
from dotenv import load_dotenv

_ = load_dotenv()


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> torch.Tensor:
    """Compute Generalized Advantage Estimation (GAE)."""
    advantages: list[float] = []
    gae = 0.0

    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * (1 - int(dones[i])) - values[i]
        gae = delta + gamma * lam * (1 - int(dones[i])) * gae
        advantages.insert(0, gae)

    return torch.tensor(advantages, dtype=torch.float)


def train_ppo(
    model_type: str,
    model_name: str,
    total_timesteps: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    lr: float,
    update_interval: int,
    force: bool = False,
    randomize: bool = True,
    episode_steps: int = 500,
    log_seconds: float = 3.0,
    save_episodes: int = 20,
    conv_type: str = "gin",
    heads: int = 4,
    r: int = 3,
    entropy_coef: float = 0.05,
) -> None:
    """Train Generalist PPO agent for cage generation."""
    device = configure_torch_device()

    console = Console()
    is_tty = sys.stdout.isatty()

    model_id = f"{model_type}_{model_name}"
    r_for_obs: int | None = r if model_type == "loopy" else None

    env = CageConstructionEnv(
        k=3, g=5, max_steps=episode_steps, randomize_params=randomize
    )
    sample_obs = env.reset()
    if sample_obs.x is None:
        raise RuntimeError("Environment observation is missing node features.")
    input_dim = int(sample_obs.x.size(1))
    if r_for_obs is not None:
        sample_obs = apply_r_neighborhood(sample_obs, r=r_for_obs)

    if model_type not in MODEL_CLASSES:
        print(f"Error: Unknown model type '{model_type}'.")
        print(f"Available: {list_model_types()}")
        sys.exit(1)

    agent = ActorCritic(
        model_type=model_type,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        conv_type=conv_type,
        heads=heads,
        r=r,
    )

    optimizer = optim.Adam(agent.parameters(), lr=lr)
    resumed = False
    resume_path: Path | None = None

    if model_exists("cage", model_id):
        model_dir = get_trained_dir("cage") / model_id
        weights_path = model_dir / "weights.pt"
        info_path = model_dir / "info.json"
        if force:
            console.print(
                f"[yellow]Force mode:[/] existing model {model_id} will be overwritten from scratch."
            )
        else:
            try:
                _ = agent.load_state_dict(torch.load(weights_path, map_location=device))  # pyright: ignore[reportAny]
                resumed = True
                resume_path = weights_path
                console.print(
                    f"[green]Resume:[/] loaded existing checkpoint for {model_id} from {weights_path}."
                )
                if info_path.exists():
                    with open(info_path, "r") as f:
                        prev_info = cast(
                            dict[str, str | dict[str, str | int | float | bool]],
                            json.load(f),
                        )
                    prev_metrics = cast(dict[str, float], prev_info.get("metrics", {}))
                    if "avg_reward" in prev_metrics:
                        console.print(
                            f"[green]Resume metrics:[/] previous avg_reward={prev_metrics['avg_reward']}"
                        )
            except Exception as e:
                console.print(
                    f"[red]Failed to resume {model_id}:[/] {e}\n"
                    + "Run with --force to start from scratch for this name."
                )
                sys.exit(1)

    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    value_coef = 0.5

    console.print("Starting PPO Training")
    console.print(f"Model ID: {model_id}")
    cfg = Table(show_header=False)
    cfg.add_row("Model Type", model_type)
    cfg.add_row("Steps", str(total_timesteps))
    cfg.add_row("Hidden Dim", str(hidden_dim))
    cfg.add_row("Num Layers", str(num_layers))
    cfg.add_row("Dropout", str(dropout))
    cfg.add_row("Learning Rate", str(lr))
    cfg.add_row("Entropy Coef", str(entropy_coef))
    cfg.add_row("Randomize Curriculum", str(randomize))
    cfg.add_row("Episode Steps", str(episode_steps))
    cfg.add_row("Device", str(device))
    cfg.add_row("Input Dim", str(input_dim))
    if model_type == "gps":
        cfg.add_row("Conv Type", conv_type)
        cfg.add_row("Attention Heads", str(heads))
    console.print(cfg)
    console.print("")

    obs_buffer: list[Data] = []
    action_buffer: list[int] = []
    log_prob_buffer: list[torch.Tensor] = []
    value_buffer: list[float] = []
    reward_buffer: list[float] = []
    mask_buffer: list[torch.Tensor] = []
    done_buffer: list[bool] = []

    global_step = 0
    episode_rewards: deque[float] = deque(maxlen=20)
    episode_lens: deque[int] = deque(maxlen=20)
    kg_stats: dict[tuple[int, int], dict[str, float]] = defaultdict(
        lambda: {
            "episodes": 0.0,
            "successes": 0.0,
            "reward_sum": 0.0,
            "len_sum": 0.0,
        }
    )
    episode_idx = 0

    obs = sample_obs
    current_ep_k = env.k
    current_ep_g = env.g

    current_ep_reward = 0.0
    current_ep_len = 0
    current_ep_action_counts: Counter[str] = Counter()

    start_time = time.time()
    best_avg_reward = -float("inf")
    best_model_state: dict[str, torch.Tensor] | None = None
    fps = 0
    last_checkpoint_episode = 0
    last_live_log_time = start_time
    initial = "waiting for first live step"
    live_view = Live(initial, console=console, refresh_per_second=8, transient=True)
    if is_tty:
        live_view.start()

    def _save_checkpoint(partial: bool, avg_rew_value: float, step: int) -> None:
        training_info: dict[str, str | int | float | bool | None] = {
            "steps": total_timesteps,
            "learning_rate": lr,
            "update_interval": update_interval,
            "randomize": randomize,
            "resumed": resumed,
            "resume_path": str(resume_path) if resume_path is not None else None,
            "checkpoint_step": step,
            "partial": partial,
        }
        if model_type == "gps":
            training_info["conv_type"] = conv_type
            training_info["heads"] = heads
        save_path = save_model(
            model=cast(BaseGNN, cast(object, agent)),
            task="cage",
            model_id=model_id,
            metrics={
                "avg_reward": round(avg_rew_value, 2),
                "fps": fps,
            },
            training_info=training_info,
        )
        console.print(f"Checkpoint saved at step {step}: {save_path}")

    try:
        while global_step < total_timesteps:
            _ = agent.eval()

            for i in range(update_interval):
                if global_step >= total_timesteps:
                    break

                with torch.no_grad():
                    mask = env.get_valid_action_mask()
                    action, log_prob, value = agent.get_action(obs, action_mask=mask)

                next_obs, reward, done, info = env.step(action)
                if r_for_obs is not None:
                    next_obs = apply_r_neighborhood(next_obs, r=r_for_obs)

                action_type = str(info.get("action_type", "unknown"))
                edge = cast(list[int], info.get("edge", [0, 0]))
                action_desc = f"{edge[0]}-{edge[1]}"
                current_ep_action_counts[action_type] += 1

                obs_buffer.append(obs)
                action_buffer.append(action)
                log_prob_buffer.append(log_prob)
                value_buffer.append(value.item())
                reward_buffer.append(reward)
                mask_buffer.append(mask)
                done_buffer.append(done)

                obs = next_obs
                global_step += 1
                current_ep_reward += reward
                current_ep_len += 1

                if (
                    is_tty
                    and log_seconds > 0
                    and (time.time() - last_live_log_time) >= log_seconds
                ):
                    last_live_log_time = time.time()
                    done_reason = info.get("done_reason", "-")
                    counts_line = ", ".join(
                        f"{k}: {v}"
                        for (k, v) in sorted(current_ep_action_counts.items())
                    )
                    status_text = (
                        f"Episode {episode_idx + 1}, k={current_ep_k}, g={current_ep_g}, moores_bound={env.mb}:\n"
                        f"  step: {current_ep_len}\n"
                        f"  action: {action_type}:{action_desc}, reward: {reward:+.2f}\n"
                        f"  score: {float(info.get('episode_score', 0.0)):+.2f}\n"
                        f"  nodes: {info.get('active_nodes', 0)}\n"
                        f"  edges: {info.get('num_edges', 0)}\n"
                        f"  action_counts: {counts_line if counts_line else '-'}\n"
                        f"  done: {1 if done else 0}, reason: {done_reason}\n"
                    )
                    live_view.update(status_text, refresh=True)

                if done:
                    episode_rewards.append(current_ep_reward)
                    episode_lens.append(current_ep_len)

                    episode_idx += 1
                    kg_key = (current_ep_k, current_ep_g)
                    kg_stats[kg_key]["episodes"] += 1
                    kg_stats[kg_key]["successes"] += (
                        1.0 if bool(info.get("success", False)) else 0.0
                    )
                    kg_stats[kg_key]["reward_sum"] += current_ep_reward
                    kg_stats[kg_key]["len_sum"] += current_ep_len

                    if is_tty:
                        counts_line = ", ".join(
                            f"{k}: {v}"
                            for (k, v) in sorted(current_ep_action_counts.items())
                        )
                        episode_text = (
                            f"Episode {episode_idx}, k={current_ep_k}, g={current_ep_g}, moores_bound={env.mb}:\n"
                            f"  step: {current_ep_len}\n"
                            f"  action: {action_type}:{action_desc}, reward: {reward:+.2f}\n"
                            f"  score: {float(info.get('episode_score', 0.0)):+.2f}\n"
                            f"  nodes: {info.get('active_nodes', 0)}\n"
                            f"  edges: {info.get('num_edges', 0)}\n"
                            f"  action_counts: {counts_line if counts_line else '-'}\n"
                            f"  done: 1, reason: {info.get('done_reason', '-')}\n"
                        )
                        console.print(episode_text)

                    success = bool(info.get("success", False))
                    unlocked_new = env.report_result(success)
                    if unlocked_new:
                        console.print(
                            f"[green]Progressive:[/] unlocked stage {env.unlocked}/{len(env.pairs)} → {env.pairs[env.unlocked - 1]}"
                        )

                    current_ep_reward = 0.0
                    current_ep_len = 0
                    current_ep_action_counts = Counter()
                    obs = env.reset()
                    if r_for_obs is not None:
                        obs = apply_r_neighborhood(obs, r=r_for_obs)
                    current_ep_k = env.k
                    current_ep_g = env.g

            if not obs_buffer:
                break

            _ = agent.train()

            with torch.no_grad():
                _, next_value = cast(tuple[torch.Tensor, torch.Tensor], agent(obs))
                next_value_float: float = float(next_value.item())

            values = value_buffer + [next_value_float]
            advantages = compute_gae(
                reward_buffer, values, done_buffer, gamma, gae_lambda
            )
            returns = advantages + torch.tensor(value_buffer, device=device)

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            b_actions = torch.tensor(action_buffer, device=device)
            b_log_probs = torch.stack(log_prob_buffer)
            b_returns = returns
            b_advantages = advantages
            b_masks = mask_buffer

            batch_size = len(obs_buffer)
            minibatch_size = 64

            for _ in range(4):
                indices = np.random.permutation(batch_size)

                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = indices[start:end]

                    optimizer.zero_grad()

                    for _idx in cast(list[int], mb_inds.tolist()):
                        i = int(_idx)
                        data: Data = obs_buffer[i]
                        action_idx = b_actions[i]
                        old_log_prob = b_log_probs[i]
                        advantage = b_advantages[i]
                        ret = b_returns[i]
                        mask: torch.Tensor = b_masks[i]

                        logits, value = cast(
                            tuple[torch.Tensor, torch.Tensor], agent(data)
                        )

                        logits = logits.masked_fill(~mask, -1e9)
                        dist = Categorical(logits=logits)
                        new_log_prob = cast(torch.Tensor, dist.log_prob(action_idx))
                        entropy = dist.entropy()

                        ratio = torch.exp(new_log_prob - old_log_prob)
                        surr1 = ratio * advantage
                        surr2 = (
                            torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
                            * advantage
                        )
                        policy_loss = -torch.min(surr1, surr2)

                        value_loss: torch.Tensor = 0.5 * (ret - value.squeeze()) ** 2
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

            obs_buffer = []
            action_buffer = []
            log_prob_buffer = []
            value_buffer = []
            reward_buffer = []
            mask_buffer = []
            done_buffer = []

            avg_rew = (
                sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0
            )
            avg_len = sum(episode_lens) / len(episode_lens) if episode_lens else 0.0
            elapsed = time.time() - start_time
            fps = int(global_step / elapsed) if elapsed > 0 else 0

            rollout_text = "\n".join(
                [
                    "Rollout:",
                    f"  global step: {global_step}",
                    f"  avg reward (last 20 eps): {avg_rew:.2f}",
                    f"  avg length (last 20 eps): {avg_len:.1f}",
                    f"  fps: {fps}",
                    "",
                ]
            )
            console.print(rollout_text)

            if kg_stats:
                console.print("k,g stats: episodes/success/avg_rew/avg_len")
                ranked = sorted(
                    kg_stats.items(),
                    key=lambda kv: kv[1]["episodes"],
                    reverse=True,
                )
                for (k_val, g_val), stats in ranked[:8]:
                    eps = max(1.0, stats["episodes"])
                    succ_rate = 100.0 * stats["successes"] / eps
                    avg_pair_rew = stats["reward_sum"] / eps
                    avg_pair_len = stats["len_sum"] / eps
                    console.print(
                        f"    ({k_val},{g_val}): {int(stats['episodes'])} eps | {succ_rate:5.1f}% | {avg_pair_rew:+6.2f} | {avg_pair_len:6.1f}"
                    )

            if avg_rew > best_avg_reward:
                best_avg_reward = avg_rew
                best_model_state = copy.deepcopy(agent.state_dict())
                console.print(f"New best! Avg Reward: {best_avg_reward:.2f}")
                _save_checkpoint(partial=True, avg_rew_value=avg_rew, step=global_step)

            checkpoint_due_by_episode = (
                save_episodes > 0
                and (episode_idx - last_checkpoint_episode) >= save_episodes
            )
            if checkpoint_due_by_episode:
                _save_checkpoint(partial=True, avg_rew_value=avg_rew, step=global_step)
                last_checkpoint_episode = episode_idx
    finally:
        if is_tty:
            live_view.stop()

    if best_model_state is not None:
        _ = agent.load_state_dict(best_model_state)

    training_info: dict[str, str | int | float | bool | None] = {
        "steps": total_timesteps,
        "learning_rate": lr,
        "update_interval": update_interval,
        "randomize": randomize,
        "episode_steps": episode_steps,
        "resumed": resumed,
        "resume_path": str(resume_path) if resume_path is not None else None,
        "checkpoint_step": global_step,
        "partial": False,
    }
    if model_type == "gps":
        training_info["conv_type"] = conv_type
        training_info["heads"] = heads

    save_path = save_model(
        model=cast(BaseGNN, cast(object, agent)),
        task="cage",
        model_id=model_id,
        metrics={
            "avg_reward": round(best_avg_reward, 2),
            "fps": fps,
        },
        training_info=training_info,
    )

    console.print("Training Finished")
    console.print(f"Model saved via registry to: {save_path}")
    console.print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    _ = parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gin",
        choices=list(MODEL_CLASSES.keys()),
        help="Model type",
    )
    _ = parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Optional run name (default: ppo, resulting model_id: <model>_ppo)",
    )

    _ = parser.add_argument(
        "--steps", type=int, default=200000, help="Total training steps"
    )
    _ = parser.add_argument(
        "--hidden-dim",
        type=int,
        default=int(os.getenv("TRAINING_HIDDEN_DIM", 64)),
        help="Hidden dimension",
    )
    _ = parser.add_argument(
        "--num-layers", type=int, default=4, help="Number of GNN layers"
    )
    _ = parser.add_argument(
        "--dropout", type=float, default=0.2, help="Dropout probability"
    )
    _ = parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    _ = parser.add_argument(
        "--update-interval",
        type=int,
        default=2048,
        help="PPO update interval",
    )
    _ = parser.add_argument(
        "--episode-steps",
        type=int,
        default=500,
        help="Maximum steps per episode before done_reason=max_steps.",
    )
    _ = parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.05,
        help="Entropy coefficient for PPO (higher = more exploration)",
    )
    _ = parser.add_argument(
        "--r",
        type=int,
        default=3,
        help="r-neighborhood radius for Loopy GNN (detects cycles up to r+2)",
    )
    _ = parser.add_argument(
        "--conv-type",
        type=str,
        default="gin",
        choices=["gcn", "sage", "gin"],
        help="Inner conv type for GPS model",
    )
    _ = parser.add_argument(
        "--heads",
        type=int,
        default=4,
        help="Number of attention heads for GPS model",
    )
    _ = parser.add_argument(
        "--no-random",
        action="store_true",
        help="Disable curriculum randomization",
    )
    _ = parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Start from scratch even if model_id exists (do not resume)",
    )
    _ = parser.add_argument(
        "--log",
        type=float,
        default=3.0,
        help="Update live logs every N seconds (0 disables).",
    )
    _ = parser.add_argument(
        "--save",
        type=int,
        default=5,
        help="Save checkpoint every N completed episodes (0 disables episode-based checkpointing).",
    )
    args = parser.parse_args()
    model_arg = cast(str, args.model)
    conv_type_arg = cast(str, args.conv_type)
    heads_arg = cast(int, args.heads)
    r_arg = cast(int, args.r)

    if cast(str | None, args.name) is not None:
        run_name = cast(str, args.name)
    elif model_arg == "gps":
        run_name = f"{conv_type_arg}_ppo"
    elif model_arg == "loopy":
        run_name = f"r{r_arg}_ppo"
    else:
        run_name = "ppo"

    train_ppo(
        model_type=model_arg,
        model_name=run_name,
        total_timesteps=cast(int, args.steps),
        hidden_dim=cast(int, args.hidden_dim),
        num_layers=cast(int, args.num_layers),
        dropout=cast(float, args.dropout),
        lr=cast(float, args.lr),
        update_interval=cast(int, args.update_interval),
        force=cast(bool, args.force),
        randomize=not cast(bool, args.no_random),
        episode_steps=cast(int, args.episode_steps),
        log_seconds=cast(float, args.log),
        save_episodes=cast(int, args.save),
        conv_type=conv_type_arg,
        heads=heads_arg,
        r=r_arg,
        entropy_coef=cast(float, args.entropy_coef),
    )
