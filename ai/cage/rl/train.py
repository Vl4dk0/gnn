import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import time
from typing import Any
from torch_geometric.data import Data

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from ai.cage.rl.env import CageConstructionEnv
from ai.cage.rl.model import ActorCritic
from ai.registry import save_model, model_exists, list_model_types
from ai.models import MODEL_CLASSES, BaseGNN


def compute_gae(rewards: list[float], values: list[float], next_value: float, dones: list[bool], gamma: float = 0.99, lam: float = 0.95) -> torch.Tensor:
    """
    Compute Generalized Advantage Estimation (GAE).
    """
    advantages = []
    gae = 0.0
    
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i+1] * (1 - int(dones[i])) - values[i]
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
    print_every: int, # Ignored for now, using rollout logs
    force: bool = False,
    device: str = "cpu",
    randomize: bool = True,
) -> None:
    """
    Train Generalist PPO agent for Cage Generation.
    """
    model_id = f"{model_type}_{model_name}"
    
    # Check if model already exists
    if model_exists("cage", model_id) and not force:
        print(f"Error: Model '{model_id}' already exists for task 'cage'.")
        print(f"Use --force to overwrite, or choose a different --name.")
        sys.exit(1)

    # Init Env with randomization
    # k=3, g=5 are just placeholders, they will be randomized
    env = CageConstructionEnv(k=3, g=5, randomize_params=randomize)
    
    # Input Dim = MAX_K (10) + 3 = 13
    input_dim = 13 
    
    # Initialize Model
    if model_type not in MODEL_CLASSES:
        print(f"Error: Unknown model type '{model_type}'.")
        print(f"Available: {list_model_types()}")
        sys.exit(1)
        
    agent = ActorCritic(
        model_type=model_type, 
        input_dim=input_dim, 
        hidden_dim=hidden_dim, 
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    
    # Hyperparameters
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    value_coef = 0.5
    entropy_coef = 0.01
    
    print("=" * 60)
    print(f"Starting Generalist PPO Training")
    print(f"Model ID: {model_id}")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Model Type: {model_type}")
    print(f"  - Steps: {total_timesteps}")
    print(f"  - Hidden Dim: {hidden_dim}")
    print(f"  - Num Layers: {num_layers}")
    print(f"  - Dropout: {dropout}")
    print(f"  - Learning Rate: {lr}")
    print(f"  - Randomize Curriculum: {randomize}")
    print(f"  - Device: {device}")
    print("=" * 60)
    
    # Storage buffers
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
    
    obs = env.reset()
    obs = obs.to(device)
    
    current_ep_reward = 0.0
    current_ep_len = 0
    
    start_time = time.time()
    best_avg_reward = -float("inf")
    best_model_state: dict[str, Any] | None = None
    fps = 0
    
    while global_step < total_timesteps:
        
        # --- Rollout Phase ---
        agent.eval()
        rollout_start_time = time.time()
        
        for i in range(update_interval):
            # Check for completion inside rollout
            if global_step >= total_timesteps:
                break

            # Frequent heartbeat logging during rollout
            if (i + 1) % 100 == 0:
                elapsed = time.time() - rollout_start_time
                steps_per_sec = (i + 1) / elapsed
                print(f"  [Rollout] Collecting step {i+1}/{update_interval} (Global: {global_step}) | Speed: {steps_per_sec:.1f} steps/s", end="\r")

            with torch.no_grad():
                mask = env.get_valid_action_mask().to(device)
                action, log_prob, value = agent.get_action(obs, action_mask=mask)
                
            next_obs, reward, done, info = env.step(action)
            next_obs = next_obs.to(device)
            
            # Store data
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
            
            if done:
                episode_rewards.append(current_ep_reward)
                episode_lens.append(current_ep_len)
                current_ep_reward = 0.0
                current_ep_len = 0
                
                obs = env.reset()
                obs = obs.to(device)
        
        # Newline after rollout progress
        print()
        
        if len(obs_buffer) == 0:
            break
                
        # --- Update Phase ---
        agent.train()
        
        # Bootstrap value for the last step
        with torch.no_grad():
            _, next_value = agent(obs)
            next_value_float = next_value.item()
            
        # Compute GAE
        values = value_buffer + [next_value_float]
        advantages = compute_gae(reward_buffer, values, next_value_float, done_buffer, gamma, gae_lambda)
        advantages = advantages.to(device)
        returns = advantages + torch.tensor(value_buffer, device=device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Optimize policy and value function
        # Flatten buffers
        # obs_buffer is already a list of Data objects, no need to flatten or stack differently
        
        b_actions = torch.tensor(action_buffer, device=device)
        b_log_probs = torch.tensor(log_prob_buffer, device=device)
        b_returns = returns
        b_advantages = advantages
        b_masks = mask_buffer # List of tensors
        
        # Mini-batch updates
        batch_size = len(obs_buffer)
        minibatch_size = 64
        
        # PPO Epochs
        for _ in range(4): # 4 epochs per update
            indices = np.random.permutation(batch_size)
            
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = indices[start:end]
                
                # Accumulate gradients for this minibatch
                optimizer.zero_grad()
                
                loss_accum = 0.0
                
                # Process each sample in minibatch separately (variable graph sizes)
                for idx in mb_inds:
                    data = obs_buffer[idx]
                    action_idx = b_actions[idx]
                    old_log_prob = b_log_probs[idx]
                    advantage = b_advantages[idx]
                    ret = b_returns[idx]
                    mask = b_masks[idx]
                    
                    logits, value = agent(data)
                    
                    # Policy Loss
                    logits = logits.masked_fill(~mask, -1e9)
                    dist = Categorical(logits=logits)
                    new_log_prob = dist.log_prob(action_idx)
                    entropy = dist.entropy()
                    
                    ratio = torch.exp(new_log_prob - old_log_prob)
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
                    policy_loss = -torch.min(surr1, surr2)
                    
                    # Value Loss
                    value_loss = 0.5 * (ret - value.squeeze()) ** 2
                    
                    # Total Loss
                    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                    loss.backward() # Accumulate gradients
                    
                    loss_accum += loss.item()

                # Average gradients
                for p in agent.parameters():
                    if p.grad is not None:
                        p.grad /= len(mb_inds)
                        
                optimizer.step()
                
        # Clear buffers
        obs_buffer = []
        action_buffer = []
        log_prob_buffer = []
        value_buffer = []
        reward_buffer = []
        mask_buffer = []
        done_buffer = []
        
        # Logging
        avg_rew = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0
        avg_len = sum(episode_lens) / len(episode_lens) if episode_lens else 0.0
        fps = int(global_step / (time.time() - start_time))
        print(f"Step {global_step} | Avg Reward: {avg_rew:.2f} | Avg Len: {avg_len:.1f} | FPS: {fps}")
        
        # Track best model
        if avg_rew > best_avg_reward:
            best_avg_reward = avg_rew
            best_model_state = agent.state_dict().copy()
            print(f"  → New best! Avg Reward: {best_avg_reward:.2f}")

    # Restore best model
    if best_model_state is not None:
        agent.load_state_dict(best_model_state)

    # Save via Registry
    training_info = {
        "steps": total_timesteps,
        "learning_rate": lr,
        "update_interval": update_interval,
        "randomize": randomize
    }
    
    # Cast agent to BaseGNN to satisfy type checker (it has required methods)
    # The error "ActorCritic is not assignable to BaseGNN" happens because ActorCritic doesn't inherit from BaseGNN
    # However, it implements the required protocol (get_config, get_name, state_dict)
    # We can cast it to Any to bypass the strict check for now, or make ActorCritic inherit from BaseGNN
    # Making it inherit from BaseGNN is tricky because BaseGNN expects specific init signature
    # So we cast to Any
    base_gnn_model: Any = agent
    
    save_path = save_model(
        model=base_gnn_model,
        task="cage",
        model_id=model_id,
        metrics={
            "avg_reward": round(best_avg_reward, 2),
            "fps": fps
        },
        training_info=training_info
    )
    
    print("=" * 60)
    print(f"Model saved via registry to: {save_path}")
    print("=" * 60)
    print(f"Starting Generalist PPO Training")
    print(f"Model ID: {model_id}")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Model Type: {model_type}")
    print(f"  - Steps: {total_timesteps}")
    print(f"  - Hidden Dim: {hidden_dim}")
    print(f"  - Num Layers: {num_layers}")
    print(f"  - Dropout: {dropout}")
    print(f"  - Learning Rate: {lr}")
    print(f"  - Randomize Curriculum: {randomize}")
    print(f"  - Device: {device}")
    print("=" * 60)
    
    # Storage buffers
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
    
    obs = env.reset()
    obs = obs.to(device)
    
    current_ep_reward = 0.0
    current_ep_len = 0
    
    start_time = time.time()
    best_avg_reward = -float("inf")
    best_model_state: dict[str, Any] | None = None
    fps = 0
    
    while global_step < total_timesteps:
        
        # --- Rollout Phase ---
        agent.eval()
        rollout_start_time = time.time()
        
        for i in range(update_interval):
            # Check for completion inside rollout
            if global_step >= total_timesteps:
                break

            # Frequent heartbeat logging during rollout
            if (i + 1) % 100 == 0:
                elapsed = time.time() - rollout_start_time
                steps_per_sec = (i + 1) / elapsed
                print(f"  [Rollout] Collecting step {i+1}/{update_interval} (Global: {global_step}) | Speed: {steps_per_sec:.1f} steps/s", end="\r")

            with torch.no_grad():
                mask = env.get_valid_action_mask().to(device)
                action, log_prob, value = agent.get_action(obs, action_mask=mask)
                
            next_obs, reward, done, info = env.step(action)
            next_obs = next_obs.to(device)
            
            # Store data
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
            
            if done:
                episode_rewards.append(current_ep_reward)
                episode_lens.append(current_ep_len)
                current_ep_reward = 0.0
                current_ep_len = 0
                
                obs = env.reset()
                obs = obs.to(device)
        
        # Newline after rollout progress
        print()
        
        if len(obs_buffer) == 0:
            break
                
        # --- Update Phase ---
        agent.train()
        
        # Bootstrap value for the last step
        with torch.no_grad():
            _, next_value = agent(obs)
            next_value_float = next_value.item()
            
        # Compute GAE
        values = value_buffer + [next_value_float]
        advantages = compute_gae(reward_buffer, values, next_value_float, done_buffer, gamma, gae_lambda)
        advantages = advantages.to(device)
        returns = advantages + torch.tensor(value_buffer, device=device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Optimize policy and value function
        # Flatten buffers
        # obs_buffer is already a list of Data objects, no need to flatten or stack differently
        
        b_actions = torch.tensor(action_buffer, device=device)
        b_log_probs = torch.tensor(log_prob_buffer, device=device)
        b_returns = returns
        b_advantages = advantages
        b_masks = mask_buffer # List of tensors
        
        # Mini-batch updates
        batch_size = len(obs_buffer)
        minibatch_size = 64
        
        # PPO Epochs
        for _ in range(4): # 4 epochs per update
            indices = np.random.permutation(batch_size)
            
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = indices[start:end]
                
                # Accumulate gradients for this minibatch
                optimizer.zero_grad()
                
                loss_accum = 0.0
                
                # Process each sample in minibatch separately (variable graph sizes)
                for idx in mb_inds:
                    data = obs_buffer[idx]
                    action_idx = b_actions[idx]
                    old_log_prob = b_log_probs[idx]
                    advantage = b_advantages[idx]
                    ret = b_returns[idx]
                    mask = b_masks[idx]
                    
                    logits, value = agent(data)
                    
                    # Policy Loss
                    logits = logits.masked_fill(~mask, -1e9)
                    dist = Categorical(logits=logits)
                    new_log_prob = dist.log_prob(action_idx)
                    entropy = dist.entropy()
                    
                    ratio = torch.exp(new_log_prob - old_log_prob)
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
                    policy_loss = -torch.min(surr1, surr2)
                    
                    # Value Loss
                    value_loss = 0.5 * (ret - value.squeeze()) ** 2
                    
                    # Total Loss
                    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                    loss.backward() # Accumulate gradients
                    
                    loss_accum += loss.item()

                # Average gradients
                for p in agent.parameters():
                    if p.grad is not None:
                        p.grad /= len(mb_inds)
                        
                optimizer.step()
                
        # Clear buffers
        obs_buffer = []
        action_buffer = []
        log_prob_buffer = []
        value_buffer = []
        reward_buffer = []
        mask_buffer = []
        done_buffer = []
        
        # Logging
        avg_rew = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0
        avg_len = sum(episode_lens) / len(episode_lens) if episode_lens else 0.0
        fps = int(global_step / (time.time() - start_time))
        print(f"Step {global_step} | Avg Reward: {avg_rew:.2f} | Avg Len: {avg_len:.1f} | FPS: {fps}")
        
        # Track best model
        if avg_rew > best_avg_reward:
            best_avg_reward = avg_rew
            best_model_state = agent.state_dict().copy()
            print(f"  → New best! Avg Reward: {best_avg_reward:.2f}")

    # Restore best model
    if best_model_state is not None:
        agent.load_state_dict(best_model_state)

    # Save via Registry
    training_info = {
        "steps": total_timesteps,
        "learning_rate": lr,
        "update_interval": update_interval,
        "randomize": randomize
    }
    
    # Cast agent to BaseGNN to satisfy type checker (it has required methods)
    # The error "ActorCritic is not assignable to BaseGNN" happens because ActorCritic doesn't inherit from BaseGNN
    # However, it implements the required protocol (get_config, get_name, state_dict)
    # We can cast it to Any to bypass the strict check for now, or make ActorCritic inherit from BaseGNN
    # Making it inherit from BaseGNN is tricky because BaseGNN expects specific init signature
    # So we cast to Any
    base_gnn_model: Any = agent
    
    save_path = save_model(
        model=base_gnn_model,
        task="cage",
        model_id=model_id,
        metrics={
            "avg_reward": round(best_avg_reward, 2),
            "fps": fps
        },
        training_info=training_info
    )
    
    print("=" * 60)
    print(f"Model saved via registry to: {save_path}")
    print("=" * 60)
    print(f"Starting Generalist PPO Training")
    print(f"Model ID: {model_id}")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Model Type: {model_type}")
    print(f"  - Steps: {total_timesteps}")
    print(f"  - Hidden Dim: {hidden_dim}")
    print(f"  - Num Layers: {num_layers}")
    print(f"  - Dropout: {dropout}")
    print(f"  - Learning Rate: {lr}")
    print(f"  - Randomize Curriculum: {randomize}")
    print(f"  - Device: {device}")
    print("=" * 60)
    
    # Storage buffers
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
    
    obs = env.reset()
    obs = obs.to(device)
    
    current_ep_reward = 0.0
    current_ep_len = 0
    
    start_time = time.time()
    best_avg_reward = -float("inf")
    best_model_state: dict[str, Any] | None = None
    fps = 0
    
    while global_step < total_timesteps:
        
        # --- Rollout Phase ---
        agent.eval()
        rollout_start_time = time.time()
        
        for i in range(update_interval):
            # Check for completion inside rollout
            if global_step >= total_timesteps:
                break

            # Frequent heartbeat logging during rollout
            if (i + 1) % 100 == 0:
                elapsed = time.time() - rollout_start_time
                steps_per_sec = (i + 1) / elapsed
                print(f"  [Rollout] Collecting step {i+1}/{update_interval} (Global: {global_step}) | Speed: {steps_per_sec:.1f} steps/s", end="\r")

            with torch.no_grad():
                mask = env.get_valid_action_mask().to(device)
                action, log_prob, value = agent.get_action(obs, action_mask=mask)
                
            next_obs, reward, done, info = env.step(action)
            next_obs = next_obs.to(device)
            
            # Store data
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
            
            if done:
                episode_rewards.append(current_ep_reward)
                episode_lens.append(current_ep_len)
                current_ep_reward = 0.0
                current_ep_len = 0
                
                obs = env.reset()
                obs = obs.to(device)
        
        # Newline after rollout progress
        print()
        
        if len(obs_buffer) == 0:
            break
                
        # --- Update Phase ---
        agent.train()
        
        # Bootstrap value for the last step
        with torch.no_grad():
            _, next_value = agent(obs)
            next_value_float = next_value.item()
            
        # Compute GAE
        values = value_buffer + [next_value_float]
        advantages = compute_gae(reward_buffer, values, next_value_float, done_buffer, gamma, gae_lambda)
        advantages = advantages.to(device)
        returns = advantages + torch.tensor(value_buffer, device=device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Optimize policy and value function
        # Flatten buffers
        # obs_buffer is already a list of Data objects, no need to flatten or stack differently
        
        b_actions = torch.tensor(action_buffer, device=device)
        b_log_probs = torch.tensor(log_prob_buffer, device=device)
        b_returns = returns
        b_advantages = advantages
        b_masks = mask_buffer # List of tensors
        
        # Mini-batch updates
        batch_size = len(obs_buffer)
        minibatch_size = 64
        
        # PPO Epochs
        for _ in range(4): # 4 epochs per update
            indices = np.random.permutation(batch_size)
            
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = indices[start:end]
                
                # Accumulate gradients for this minibatch
                optimizer.zero_grad()
                
                loss_accum = 0.0
                
                # Process each sample in minibatch separately (variable graph sizes)
                for idx in mb_inds:
                    data = obs_buffer[idx]
                    action_idx = b_actions[idx]
                    old_log_prob = b_log_probs[idx]
                    advantage = b_advantages[idx]
                    ret = b_returns[idx]
                    mask = b_masks[idx]
                    
                    logits, value = agent(data)
                    
                    # Policy Loss
                    logits = logits.masked_fill(~mask, -1e9)
                    dist = Categorical(logits=logits)
                    new_log_prob = dist.log_prob(action_idx)
                    entropy = dist.entropy()
                    
                    ratio = torch.exp(new_log_prob - old_log_prob)
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
                    policy_loss = -torch.min(surr1, surr2)
                    
                    # Value Loss
                    value_loss = 0.5 * (ret - value.squeeze()) ** 2
                    
                    # Total Loss
                    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                    loss.backward() # Accumulate gradients
                    
                    loss_accum += loss.item()

                # Average gradients
                for p in agent.parameters():
                    if p.grad is not None:
                        p.grad /= len(mb_inds)
                        
                optimizer.step()
                
        # Clear buffers
        obs_buffer = []
        action_buffer = []
        log_prob_buffer = []
        value_buffer = []
        reward_buffer = []
        mask_buffer = []
        done_buffer = []
        
        # Logging
        avg_rew = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0
        avg_len = sum(episode_lens) / len(episode_lens) if episode_lens else 0.0
        fps = int(global_step / (time.time() - start_time))
        print(f"Step {global_step} | Avg Reward: {avg_rew:.2f} | Avg Len: {avg_len:.1f} | FPS: {fps}")
        
        # Track best model
        if avg_rew > best_avg_reward:
            best_avg_reward = avg_rew
            best_model_state = agent.state_dict().copy()
            print(f"  → New best! Avg Reward: {best_avg_reward:.2f}")

    # Restore best model
    if best_model_state is not None:
        agent.load_state_dict(best_model_state)

    # Save via Registry
    training_info = {
        "steps": total_timesteps,
        "learning_rate": lr,
        "update_interval": update_interval,
        "randomize": randomize
    }
    
    # Cast agent to BaseGNN to satisfy type checker (it has required methods)
    # The error "ActorCritic is not assignable to BaseGNN" happens because ActorCritic doesn't inherit from BaseGNN
    # However, it implements the required protocol (get_config, get_name, state_dict)
    # We can cast it to Any to bypass the strict check for now, or make ActorCritic inherit from BaseGNN
    # Making it inherit from BaseGNN is tricky because BaseGNN expects specific init signature
    # So we cast to Any
    base_gnn_model: Any = agent
    
    save_path = save_model(
        model=base_gnn_model,
        task="cage",
        model_id=model_id,
        metrics={
            "avg_reward": round(best_avg_reward, 2),
            "fps": fps
        },
        training_info=training_info
    )
    
    print("=" * 60)
    print(f"Model saved via registry to: {save_path}")
    print("=" * 60)
    print(f"Starting Generalist PPO Training")
    print(f"Model ID: {model_id}")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Model Type: {model_type}")
    print(f"  - Steps: {total_timesteps}")
    print(f"  - Hidden Dim: {hidden_dim}")
    print(f"  - Num Layers: {num_layers}")
    print(f"  - Dropout: {dropout}")
    print(f"  - Learning Rate: {lr}")
    print(f"  - Randomize Curriculum: {randomize}")
    print(f"  - Device: {device}")
    print("=" * 60)
    
    # Storage buffers
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
    
    obs = env.reset()
    obs = obs.to(device)
    
    current_ep_reward = 0.0
    current_ep_len = 0
    
    start_time = time.time()
    best_avg_reward = -float("inf")
    best_model_state: dict[str, Any] | None = None
    fps = 0
    
    while global_step < total_timesteps:
        
        # --- Rollout Phase ---
        agent.eval()
        rollout_start_time = time.time()
        
        for i in range(update_interval):
            # Check for completion inside rollout
            if global_step >= total_timesteps:
                break

            # Frequent heartbeat logging during rollout
            if (i + 1) % 100 == 0:
                elapsed = time.time() - rollout_start_time
                steps_per_sec = (i + 1) / elapsed
                print(f"  [Rollout] Collecting step {i+1}/{update_interval} (Global: {global_step}) | Speed: {steps_per_sec:.1f} steps/s", end="\r")

            with torch.no_grad():
                mask = env.get_valid_action_mask().to(device)
                action, log_prob, value = agent.get_action(obs, action_mask=mask)
                
            next_obs, reward, done, info = env.step(action)
            next_obs = next_obs.to(device)
            
            # Store data
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
            
            if done:
                episode_rewards.append(current_ep_reward)
                episode_lens.append(current_ep_len)
                current_ep_reward = 0.0
                current_ep_len = 0
                
                obs = env.reset()
                obs = obs.to(device)
        
        # Newline after rollout progress
        print()
        
        if len(obs_buffer) == 0:
            break
                
        # --- Update Phase ---
        agent.train()
        
        # Bootstrap value for the last step
        with torch.no_grad():
            _, next_value = agent(obs)
            next_value_float = next_value.item()
            
        # Compute GAE
        values = value_buffer + [next_value_float]
        advantages = compute_gae(reward_buffer, values, next_value_float, done_buffer, gamma, gae_lambda)
        advantages = advantages.to(device)
        returns = advantages + torch.tensor(value_buffer, device=device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Optimize policy and value function
        # Flatten buffers
        # obs_buffer is already a list of Data objects, no need to flatten or stack differently
        
        b_actions = torch.tensor(action_buffer, device=device)
        b_log_probs = torch.tensor(log_prob_buffer, device=device)
        b_returns = returns
        b_advantages = advantages
        b_masks = mask_buffer # List of tensors
        
        # Mini-batch updates
        batch_size = len(obs_buffer)
        minibatch_size = 64
        
        # PPO Epochs
        for _ in range(4): # 4 epochs per update
            indices = np.random.permutation(batch_size)
            
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = indices[start:end]
                
                # Accumulate gradients for this minibatch
                optimizer.zero_grad()
                
                loss_accum = 0.0
                
                # Process each sample in minibatch separately (variable graph sizes)
                for idx in mb_inds:
                    data = obs_buffer[idx]
                    action_idx = b_actions[idx]
                    old_log_prob = b_log_probs[idx]
                    advantage = b_advantages[idx]
                    ret = b_returns[idx]
                    mask = b_masks[idx]
                    
                    logits, value = agent(data)
                    
                    # Policy Loss
                    logits = logits.masked_fill(~mask, -1e9)
                    dist = Categorical(logits=logits)
                    new_log_prob = dist.log_prob(action_idx)
                    entropy = dist.entropy()
                    
                    ratio = torch.exp(new_log_prob - old_log_prob)
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
                    policy_loss = -torch.min(surr1, surr2)
                    
                    # Value Loss
                    value_loss = 0.5 * (ret - value.squeeze()) ** 2
                    
                    # Total Loss
                    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                    loss.backward() # Accumulate gradients
                    
                    loss_accum += loss.item()

                # Average gradients
                for p in agent.parameters():
                    if p.grad is not None:
                        p.grad /= len(mb_inds)
                        
                optimizer.step()
                
        # Clear buffers
        obs_buffer = []
        action_buffer = []
        log_prob_buffer = []
        value_buffer = []
        reward_buffer = []
        mask_buffer = []
        done_buffer = []
        
        # Logging
        avg_rew = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0
        avg_len = sum(episode_lens) / len(episode_lens) if episode_lens else 0.0
        fps = int(global_step / (time.time() - start_time))
        print(f"Step {global_step} | Avg Reward: {avg_rew:.2f} | Avg Len: {avg_len:.1f} | FPS: {fps}")
        
        # Track best model
        if avg_rew > best_avg_reward:
            best_avg_reward = avg_rew
            best_model_state = agent.state_dict().copy()
            print(f"  → New best! Avg Reward: {best_avg_reward:.2f}")

    # Restore best model
    if best_model_state is not None:
        agent.load_state_dict(best_model_state)

    # Save via Registry
    training_info = {
        "steps": total_timesteps,
        "learning_rate": lr,
        "update_interval": update_interval,
        "randomize": randomize
    }
    
    # Cast agent to BaseGNN to satisfy type checker (it has required methods)
    # The error "ActorCritic is not assignable to BaseGNN" happens because ActorCritic doesn't inherit from BaseGNN
    # However, it implements the required protocol (get_config, get_name, state_dict)
    # We can cast it to Any to bypass the strict check for now, or make ActorCritic inherit from BaseGNN
    # Making it inherit from BaseGNN is tricky because BaseGNN expects specific init signature
    # So we cast to Any
    base_gnn_model: Any = agent
    
    save_path = save_model(
        model=base_gnn_model,
        task="cage",
        model_id=model_id,
        metrics={
            "avg_reward": round(best_avg_reward, 2),
            "fps": fps
        },
        training_info=training_info
    )
    
    print("=" * 60)
    print(f"Model saved via registry to: {save_path}")
    print("=" * 60)
    print(f"Starting Generalist PPO Training")
    print(f"Model ID: {model_id}")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Model Type: {model_type}")
    print(f"  - Steps: {total_timesteps}")
    print(f"  - Hidden Dim: {hidden_dim}")
    print(f"  - Num Layers: {num_layers}")
    print(f"  - Dropout: {dropout}")
    print(f"  - Learning Rate: {lr}")
    print(f"  - Randomize Curriculum: {randomize}")
    print(f"  - Device: {device}")
    print("=" * 60)
    
    # Storage buffers
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
    
    obs = env.reset()
    obs = obs.to(device)
    
    current_ep_reward = 0.0
    current_ep_len = 0
    
    start_time = time.time()
    best_avg_reward = -float("inf")
    best_model_state: dict[str, Any] | None = None
    fps = 0
    
    while global_step < total_timesteps:
        
        # --- Rollout Phase ---
        agent.eval()
        rollout_start_time = time.time()
        
        for i in range(update_interval):
            # Check for completion inside rollout
            if global_step >= total_timesteps:
                break

            # Frequent heartbeat logging during rollout
            if (i + 1) % 100 == 0:
                elapsed = time.time() - rollout_start_time
                steps_per_sec = (i + 1) / elapsed
                print(f"  [Rollout] Collecting step {i+1}/{update_interval} (Global: {global_step}) | Speed: {steps_per_sec:.1f} steps/s", end="\r")

            with torch.no_grad():
                mask = env.get_valid_action_mask().to(device)
                action, log_prob, value = agent.get_action(obs, action_mask=mask)
                
            next_obs, reward, done, info = env.step(action)
            next_obs = next_obs.to(device)
            
            # Store data
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
            
            if done:
                episode_rewards.append(current_ep_reward)
                episode_lens.append(current_ep_len)
                current_ep_reward = 0.0
                current_ep_len = 0
                
                obs = env.reset()
                obs = obs.to(device)
        
        # Newline after rollout progress
        print()
        
        if len(obs_buffer) == 0:
            break
                
        # --- Update Phase ---
        agent.train()
        
        # Bootstrap value for the last step
        with torch.no_grad():
            _, next_value = agent(obs)
            next_value = next_value.item()
            
        # Compute GAE
        values = value_buffer + [next_value]
        advantages = compute_gae(reward_buffer, values, next_value, done_buffer, gamma, gae_lambda)
        advantages = advantages.to(device)
        returns = advantages + torch.tensor(value_buffer, device=device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Optimize policy and value function
        # Flatten buffers
        # obs_buffer is already a list of Data objects, no need to flatten or stack differently
        
        b_actions = torch.tensor(action_buffer, device=device)
        b_log_probs = torch.tensor(log_prob_buffer, device=device)
        b_returns = returns
        b_advantages = advantages
        b_masks = mask_buffer # List of tensors
        
        # Mini-batch updates
        batch_size = len(obs_buffer)
        minibatch_size = 64
        
        # PPO Epochs
        for _ in range(4): # 4 epochs per update
            indices = np.random.permutation(batch_size)
            
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = indices[start:end]
                
                # Accumulate gradients for this minibatch
                optimizer.zero_grad()
                
                loss_accum = 0.0
                
                # Process each sample in minibatch separately (variable graph sizes)
                for idx in mb_inds:
                    data = obs_buffer[idx]
                    action_idx = b_actions[idx]
                    old_log_prob = b_log_probs[idx]
                    advantage = b_advantages[idx]
                    ret = b_returns[idx]
                    mask = b_masks[idx]
                    
                    logits, value = agent(data)
                    
                    # Policy Loss
                    logits = logits.masked_fill(~mask, -1e9)
                    dist = Categorical(logits=logits)
                    new_log_prob = dist.log_prob(action_idx)
                    entropy = dist.entropy()
                    
                    ratio = torch.exp(new_log_prob - old_log_prob)
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
                    policy_loss = -torch.min(surr1, surr2)
                    
                    # Value Loss
                    value_loss = 0.5 * (ret - value.squeeze()) ** 2
                    
                    # Total Loss
                    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                    loss.backward() # Accumulate gradients
                    
                    loss_accum += loss.item()

                # Average gradients
                for p in agent.parameters():
                    if p.grad is not None:
                        p.grad /= len(mb_inds)
                        
                optimizer.step()
                
        # Clear buffers
        obs_buffer = []
        action_buffer = []
        log_prob_buffer = []
        value_buffer = []
        reward_buffer = []
        mask_buffer = []
        done_buffer = []
        
        # Logging
        avg_rew = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0
        avg_len = sum(episode_lens) / len(episode_lens) if episode_lens else 0.0
        fps = int(global_step / (time.time() - start_time))
        print(f"Step {global_step} | Avg Reward: {avg_rew:.2f} | Avg Len: {avg_len:.1f} | FPS: {fps}")
        
        # Track best model
        if avg_rew > best_avg_reward:
            best_avg_reward = avg_rew
            best_model_state = agent.state_dict().copy()
            print(f"  → New best! Avg Reward: {best_avg_reward:.2f}")

    # Restore best model
    if best_model_state is not None:
        agent.load_state_dict(best_model_state)

    # Save via Registry
    training_info = {
        "steps": total_timesteps,
        "learning_rate": lr,
        "update_interval": update_interval,
        "randomize": randomize
    }
    
    # Cast agent to BaseGNN to satisfy type checker (it has required methods)
    # The error "ActorCritic is not assignable to BaseGNN" happens because ActorCritic doesn't inherit from BaseGNN
    # However, it implements the required protocol (get_config, get_name, state_dict)
    # We can cast it to Any to bypass the strict check for now, or make ActorCritic inherit from BaseGNN
    # Making it inherit from BaseGNN is tricky because BaseGNN expects specific init signature
    # So we cast to Any
    base_gnn_model: Any = agent
    
    save_path = save_model(
        model=base_gnn_model,
        task="cage",
        model_id=model_id,
        metrics={
            "avg_reward": round(best_avg_reward, 2),
            "fps": fps
        },
        training_info=training_info
    )
    
    print("=" * 60)
    print(f"Model saved via registry to: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Required args to match other scripts
    parser.add_argument("--model", "-m", type=str, required=True, choices=list(MODEL_CLASSES.keys()), help="Model type")
    parser.add_argument("--name", "-n", type=str, required=True, help="Model version name")
    
    # Optional args
    parser.add_argument("--steps", type=int, default=100000, help="Total training steps")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of GNN layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--update-interval", type=int, default=2048, help="PPO update interval")
    parser.add_argument("--print-every", type=int, default=10, help="Unused in RL script but kept for compatibility")
    
    parser.add_argument("--no-random", action="store_true", help="Disable curriculum randomization")
    parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing model")
    
    # Auto-detect MPS
    default_device = "cpu"
    if torch.backends.mps.is_available():
        default_device = "mps"
    elif torch.cuda.is_available():
        default_device = "cuda"
        
    parser.add_argument("--device", type=str, default=default_device)
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
        randomize=not args.no_random
    )

