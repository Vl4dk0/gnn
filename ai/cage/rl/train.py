import argparse
import copy
import os
import sys
import time
from collections import deque
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
from torch_geometric.data import Data

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from ai.cage.rl.env import CageConstructionEnv
from ai.cage.rl.model import ActorCritic
from ai.models import MODEL_CLASSES
from ai.registry import list_model_types, model_exists, save_model


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
    print_every: int,  # Kept for CLI compatibility.
    force: bool = False,
    device: str = "cpu",
    randomize: bool = True,
    live_log_every: int = 1,
    max_logged_actions: int = 30,
) -> None:
    """Train Generalist PPO agent for cage generation."""
    del print_every

    model_id = f"{model_type}_{model_name}"

    if model_exists("cage", model_id) and not force:
        print(f"Error: Model '{model_id}' already exists for task 'cage'.")
        print("Use --force to overwrite, or choose a different --name.")
        sys.exit(1)

    env = CageConstructionEnv(k=3, g=5, randomize_params=randomize)

    # Input dim in env observation: MAX_K (10) + 3 = 13
    input_dim = 13

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

    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    value_coef = 0.5
    entropy_coef = 0.01

    print("=" * 60)
    print("Starting Generalist PPO Training")
    print(f"Model ID: {model_id}")
    print("=" * 60)
    print("Configuration:")
    print(f"  - Model Type: {model_type}")
    print(f"  - Steps: {total_timesteps}")
    print(f"  - Hidden Dim: {hidden_dim}")
    print(f"  - Num Layers: {num_layers}")
    print(f"  - Dropout: {dropout}")
    print(f"  - Learning Rate: {lr}")
    print(f"  - Randomize Curriculum: {randomize}")
    print(f"  - Device: {device}")
    print("=" * 60)

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

    obs = env.reset().to(device)
    current_ep_k = env.k
    current_ep_g = env.g
    current_ep_nodes = env.num_nodes

    current_ep_reward = 0.0
    current_ep_len = 0
    current_ep_actions: list[str] = []

    start_time = time.time()
    best_avg_reward = -float("inf")
    best_model_state: dict[str, Any] | None = None
    fps = 0

    while global_step < total_timesteps:
        agent.eval()
        rollout_start_time = time.time()

        for i in range(update_interval):
            if global_step >= total_timesteps:
                break

            if (i + 1) % 100 == 0:
                elapsed = time.time() - rollout_start_time
                steps_per_sec = (i + 1) / elapsed if elapsed > 0 else 0.0
                print(
                    f"  [Rollout] Collecting step {i + 1}/{update_interval} "
                    f"(Global: {global_step}) | Speed: {steps_per_sec:.1f} steps/s",
                    end="\r",
                )

            with torch.no_grad():
                mask = env.get_valid_action_mask().to(device)
                action, log_prob, value = agent.get_action(obs, action_mask=mask)

            u, v = env.idx_to_edge(action)
            next_obs, reward, done, info = env.step(action)
            next_obs = next_obs.to(device)

            action_type = str(info.get("action_type", "unknown"))
            action_str = f"{action_type}:{u}-{v} r={reward:+.2f}"
            current_ep_actions.append(action_str)
            if len(current_ep_actions) > max_logged_actions:
                current_ep_actions.pop(0)

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

            if live_log_every > 0 and current_ep_len % live_log_every == 0:
                done_reason = info.get("done_reason", "-")
                print(
                    f"  [Live] ep={episode_idx + 1} k={current_ep_k} g={current_ep_g} "
                    f"n={current_ep_nodes} step={current_ep_len} "
                    f"action={action_type}:{u}-{v} reward={reward:+.2f} "
                    f"score={float(info.get('episode_score', 0.0)):+.2f} "
                    f"edges={info.get('num_edges', 0)} "
                    f"done={done} reason={done_reason}",
                    end="\r",
                )

            if done:
                print()
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

                status = "SUCCESS" if bool(info.get("success", False)) else "FAIL"
                print(
                    f"[Episode {episode_idx}] {status} | "
                    f"k={current_ep_k} g={current_ep_g} n={current_ep_nodes} | "
                    f"steps={current_ep_len} reward={current_ep_reward:+.2f} "
                    f"reason={info.get('done_reason', '-')}"
                )
                if current_ep_actions:
                    print("  actions:", " | ".join(current_ep_actions))

                current_ep_reward = 0.0
                current_ep_len = 0
                current_ep_actions = []
                obs = env.reset().to(device)
                current_ep_k = env.k
                current_ep_g = env.g
                current_ep_nodes = env.num_nodes

        print()

        if not obs_buffer:
            break

        agent.train()

        with torch.no_grad():
            _, next_value = agent(obs)
            next_value_float = next_value.item()

        values = value_buffer + [next_value_float]
        advantages = compute_gae(reward_buffer, values, done_buffer, gamma, gae_lambda).to(device)
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

                for idx in mb_inds:
                    data = obs_buffer[idx]
                    action_idx = b_actions[idx]
                    old_log_prob = b_log_probs[idx]
                    advantage = b_advantages[idx]
                    ret = b_returns[idx]
                    mask = b_masks[idx]

                    logits, value = agent(data)

                    logits = logits.masked_fill(~mask, -1e9)
                    dist = Categorical(logits=logits)
                    new_log_prob = dist.log_prob(action_idx)
                    entropy = dist.entropy()

                    ratio = torch.exp(new_log_prob - old_log_prob)
                    surr1 = ratio * advantage
                    surr2 = (
                        torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
                        * advantage
                    )
                    policy_loss = -torch.min(surr1, surr2)

                    value_loss = 0.5 * (ret - value.squeeze()) ** 2
                    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                    loss.backward()

                for p in agent.parameters():
                    if p.grad is not None:
                        p.grad /= len(mb_inds)

                optimizer.step()

        obs_buffer = []
        action_buffer = []
        log_prob_buffer = []
        value_buffer = []
        reward_buffer = []
        mask_buffer = []
        done_buffer = []

        avg_rew = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0
        avg_len = sum(episode_lens) / len(episode_lens) if episode_lens else 0.0
        elapsed = time.time() - start_time
        fps = int(global_step / elapsed) if elapsed > 0 else 0

        print(
            f"Step {global_step} | Avg Reward: {avg_rew:.2f} | "
            f"Avg Len: {avg_len:.1f} | FPS: {fps}"
        )

        if kg_stats:
            print("  [k,g stats] episodes/success/avg_rew/avg_len:")
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
                print(
                    f"    ({k_val},{g_val}): "
                    f"{int(stats['episodes'])} eps | "
                    f"{succ_rate:5.1f}% | "
                    f"{avg_pair_rew:+6.2f} | "
                    f"{avg_pair_len:6.1f}"
                )

        if avg_rew > best_avg_reward:
            best_avg_reward = avg_rew
            best_model_state = copy.deepcopy(agent.state_dict())
            print(f"  -> New best! Avg Reward: {best_avg_reward:.2f}")

    if best_model_state is not None:
        agent.load_state_dict(best_model_state)

    training_info = {
        "steps": total_timesteps,
        "learning_rate": lr,
        "update_interval": update_interval,
        "randomize": randomize,
    }

    base_gnn_model: Any = agent

    save_path = save_model(
        model=base_gnn_model,
        task="cage",
        model_id=model_id,
        metrics={
            "avg_reward": round(best_avg_reward, 2),
            "fps": fps,
        },
        training_info=training_info,
    )

    print("=" * 60)
    print(f"Model saved via registry to: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        choices=list(MODEL_CLASSES.keys()),
        help="Model type",
    )
    parser.add_argument("--name", "-n", type=str, required=True, help="Model version name")

    parser.add_argument("--steps", type=int, default=100000, help="Total training steps")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of GNN layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--update-interval",
        type=int,
        default=2048,
        help="PPO update interval",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Unused in RL script but kept for compatibility",
    )

    parser.add_argument(
        "--no-random",
        action="store_true",
        help="Disable curriculum randomization",
    )
    parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing model")

    default_device = "cpu"
    if torch.backends.mps.is_available():
        default_device = "mps"
    elif torch.cuda.is_available():
        default_device = "cuda"

    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument(
        "--live-log-every",
        type=int,
        default=1,
        help="Print live per-episode step logs every N environment steps (0 disables).",
    )
    parser.add_argument(
        "--max-logged-actions",
        type=int,
        default=30,
        help="Number of latest actions to print when an episode ends.",
    )
    args = parser.parse_args()

    train_ppo(
        model_type=args.model,
        model_name=args.name,
        total_timesteps=args.steps,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        update_interval=args.update_interval,
        print_every=args.print_every,
        force=args.force,
        device=args.device,
        randomize=not args.no_random,
        live_log_every=args.live_log_every,
        max_logged_actions=args.max_logged_actions,
    )
