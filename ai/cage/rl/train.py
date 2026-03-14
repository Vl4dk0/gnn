import argparse
import copy
import json
import os
import re
import sys
import time
from collections import Counter
from collections import deque
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast, override

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
except ModuleNotFoundError:
    Console = None  # type: ignore[assignment]
    Live = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from ai.cage.rl.env import CageConstructionEnv
from ai.cage.rl.model import ActorCritic
from ai.models import MODEL_CLASSES
from ai.models.base import BaseGNN
from ai.registry import get_trained_dir, list_model_types, model_exists, save_model


class _TableProto(Protocol):
    def add_row(self, *args: str) -> None: ...


class _ConsoleProto(Protocol):
    def rule(self, title: str = "") -> None: ...
    def print(self, *args: object) -> None: ...


class _LiveProto(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def update(self, renderable: object, *, refresh: bool = False) -> None: ...


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
    episode_steps: int = 1000,
    log_seconds: float = 5.0,
    save_episodes: int = 20,
) -> None:
    """Train Generalist PPO agent for cage generation."""
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    console: _ConsoleProto
    table_factory: Callable[..., _TableProto]
    live_factory: Callable[..., _LiveProto] | None

    if Console is not None and Table is not None and Live is not None:
        console = Console()  # type: ignore[assignment]
        table_factory = Table  # type: ignore[assignment]
        live_factory = Live  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
        use_rich_live = True
    else:

        class _FallbackConsole:
            def rule(self, title: str = "") -> None:
                print(re.sub(r"\[[^\]]+\]", "", title))

            def print(self, *args: object) -> None:
                cleaned = [re.sub(r"\[[^\]]+\]", "", str(a)) for a in args]
                print(*cleaned)

        class _FallbackTable:
            def __init__(self, show_header: bool = False):
                del show_header
                self.rows: list[tuple[str, str]] = []

            def add_row(self, *args: str) -> None:
                if len(args) >= 2:
                    self.rows.append((args[0], args[1]))

            @override
            def __str__(self) -> str:
                return "\n".join(f"{k}: {v}" for (k, v) in self.rows)

        console = _FallbackConsole()  # type: ignore[assignment]
        table_factory = _FallbackTable
        live_factory = None
        use_rich_live = False

    model_id = f"{model_type}_{model_name}"

    env = CageConstructionEnv(
        k=3, g=5, max_steps=episode_steps, randomize_params=randomize
    )
    sample_obs = env.reset()
    if sample_obs.x is None:
        raise RuntimeError("Environment observation is missing node features.")
    input_dim = int(sample_obs.x.size(1))

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
    ).to(device)

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
                _ = agent.load_state_dict(torch.load(weights_path, map_location=device))  # type: ignore[arg-type]  # pyright: ignore[reportAny]
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
    entropy_coef = 0.01

    console.rule("Starting PPO Training")
    console.print(f"Model ID: {model_id}")
    cfg = table_factory(show_header=False)
    cfg.add_row("Model Type", model_type)
    cfg.add_row("Steps", str(total_timesteps))
    cfg.add_row("Hidden Dim", str(hidden_dim))
    cfg.add_row("Num Layers", str(num_layers))
    cfg.add_row("Dropout", str(dropout))
    cfg.add_row("Learning Rate", str(lr))
    cfg.add_row("Randomize Curriculum", str(randomize))
    cfg.add_row("Episode Steps", str(episode_steps))
    cfg.add_row("Device", str(device))
    cfg.add_row("Input Dim", str(input_dim))
    console.print(cfg)
    console.rule()

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

    obs = sample_obs.to(device)
    current_ep_k = env.k
    current_ep_g = env.g
    current_ep_nodes = env.num_nodes

    current_ep_reward = 0.0
    current_ep_len = 0
    current_ep_action_counts: Counter[str] = Counter()

    start_time = time.time()
    best_avg_reward = -float("inf")
    best_model_state: dict[str, torch.Tensor] | None = None
    fps = 0
    last_checkpoint_episode = 0
    last_live_log_time = start_time
    live_view: _LiveProto | None = None
    if use_rich_live and live_factory is not None:
        initial = table_factory(show_header=False)
        initial.add_row("status", "waiting for first live step")
        live_view = live_factory(
            initial, console=console, refresh_per_second=8, transient=True
        )
        live_view.start()

    def _save_checkpoint(partial: bool, avg_rew_value: float, step: int) -> None:
        training_info = {
            "steps": total_timesteps,
            "learning_rate": lr,
            "update_interval": update_interval,
            "randomize": randomize,
            "resumed": resumed,
            "resume_path": str(resume_path) if resume_path is not None else None,
            "checkpoint_step": step,
            "partial": partial,
        }
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
            rollout_start_time = time.time()

            for i in range(update_interval):
                if global_step >= total_timesteps:
                    break

                if (i + 1) % 100 == 0 and live_view is None:
                    elapsed = time.time() - rollout_start_time
                    steps_per_sec = (i + 1) / elapsed if elapsed > 0 else 0.0
                    print(
                        f"  [Rollout] Collecting step {i + 1}/{update_interval} (Global: {global_step}) | Speed: {steps_per_sec:.1f} steps/s",
                        end="\r",
                    )

                with torch.no_grad():
                    mask = env.get_valid_action_mask().to(device)
                    action, log_prob, value = agent.get_action(obs, action_mask=mask)

                next_obs, reward, done, info = env.step(action)
                next_obs = next_obs.to(device)

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
                    log_seconds > 0
                    and (time.time() - last_live_log_time) >= log_seconds
                ):
                    last_live_log_time = time.time()
                    done_reason = info.get("done_reason", "-")
                    if live_view is not None:
                        status = table_factory(show_header=False)
                        status.add_row(
                            "episode",
                            f"{episode_idx + 1}  k={current_ep_k} g={current_ep_g} cap={current_ep_nodes} step={current_ep_len}",
                        )
                        status.add_row("action", f"{action_type}:{action_desc}")
                        status.add_row(
                            "reward/score",
                            f"{reward:+.2f} / {float(info.get('episode_score', 0.0)):+.2f}",
                        )
                        status.add_row(
                            "edges/active",
                            f"{info.get('num_edges', 0)} / {info.get('active_nodes', 0)}",
                        )
                        counts_line = ", ".join(
                            f"{k}: {v}"
                            for (k, v) in sorted(current_ep_action_counts.items())
                        )
                        status.add_row(
                            "action counts", counts_line if counts_line else "-"
                        )
                        status.add_row("done/reason", f"{done} / {done_reason}")
                        live_view.update(status, refresh=True)
                    else:
                        console.print(
                            f"LIVE ep={episode_idx + 1} k={current_ep_k} g={current_ep_g} cap={current_ep_nodes} step={current_ep_len} action={action_type}:{action_desc} reward={reward:+.2f} score={float(info.get('episode_score', 0.0)):+.2f} edges={info.get('num_edges', 0)} active={info.get('active_nodes', 0)} done={done} reason={done_reason}"
                        )

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

                    status_str = (
                        "SUCCESS" if bool(info.get("success", False)) else "FAIL"
                    )
                    summary = table_factory(show_header=False)
                    summary.add_row("Episode", str(episode_idx))
                    summary.add_row("Status", status_str)
                    summary.add_row("k,g", f"{current_ep_k}, {current_ep_g}")
                    summary.add_row("Capacity Nodes", str(current_ep_nodes))
                    summary.add_row("Steps", str(current_ep_len))
                    summary.add_row("Reward", f"{current_ep_reward:+.2f}")
                    summary.add_row("Done Reason", str(info.get("done_reason", "-")))
                    counts_line = ", ".join(
                        f"{k}: {v}"
                        for (k, v) in sorted(current_ep_action_counts.items())
                    )
                    summary.add_row(
                        "Action Counts", counts_line if counts_line else "-"
                    )
                    console.print(summary)

                    current_ep_reward = 0.0
                    current_ep_len = 0
                    current_ep_action_counts = Counter()
                    obs = env.reset().to(device)
                    current_ep_k = env.k
                    current_ep_g = env.g
                    current_ep_nodes = env.num_nodes

            if not obs_buffer:
                break

            _ = agent.train()

            with torch.no_grad():
                _, next_value = cast(tuple[torch.Tensor, torch.Tensor], agent(obs))
                next_value_float: float = float(next_value.item())

            values = value_buffer + [next_value_float]
            advantages = compute_gae(
                reward_buffer, values, done_buffer, gamma, gae_lambda
            ).to(device)
            returns = advantages + torch.tensor(value_buffer, device=device)

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            b_actions = torch.tensor(action_buffer, device=device)
            b_log_probs = torch.stack(log_prob_buffer).to(device)
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

                    for _idx in mb_inds:  # pyright: ignore[reportAny]
                        i = int(_idx)  # pyright: ignore[reportAny]
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
                        loss.backward()  # pyright: ignore[reportUnknownMemberType, reportUnusedCallResult]

                    for p in agent.parameters():
                        if p.grad is not None:
                            p.grad /= len(mb_inds)

                    optimizer.step()  # pyright: ignore[reportUnknownMemberType, reportUnusedCallResult]

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

            rollout_tbl = table_factory(show_header=False)
            rollout_tbl.add_row("Global Step", str(global_step))
            rollout_tbl.add_row("Avg Reward (last 20 eps)", f"{avg_rew:.2f}")
            rollout_tbl.add_row("Avg Length (last 20 eps)", f"{avg_len:.1f}")
            rollout_tbl.add_row("FPS", str(fps))
            console.print(rollout_tbl)

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

            checkpoint_due_by_episode = (
                save_episodes > 0
                and (episode_idx - last_checkpoint_episode) >= save_episodes
            )
            if checkpoint_due_by_episode:
                _save_checkpoint(partial=True, avg_rew_value=avg_rew, step=global_step)
                last_checkpoint_episode = episode_idx
    finally:
        if live_view is not None:
            live_view.stop()

    if best_model_state is not None:
        _ = agent.load_state_dict(best_model_state)

    training_info = {
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

    console.rule("Training Finished")
    console.print(f"Model saved via registry to: {save_path}")
    console.rule()


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
        "--steps", type=int, default=100000, help="Total training steps"
    )
    _ = parser.add_argument(
        "--hidden-dim", type=int, default=128, help="Hidden dimension"
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
        default=1000,
        help="Maximum steps per episode before done_reason=max_steps.",
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
        default=5.0,
        help="Update live logs every N seconds (0 disables).",
    )
    _ = parser.add_argument(
        "--save",
        type=int,
        default=20,
        help="Save checkpoint every N completed episodes (0 disables episode-based checkpointing).",
    )
    args = parser.parse_args()
    run_name: str = cast(str, args.name) or "ppo"

    train_ppo(
        model_type=cast(str, args.model),
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
    )
