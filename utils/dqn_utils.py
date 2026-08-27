"""Reusable building blocks shared by the project's DQN notebooks."""

import random
from collections import deque

import numpy as np

import torch
import torch.nn as nn


class DQN(nn.Module):
    """Fully connected action-value network with configurable hidden layers."""

    def __init__(self, state_dim, action_dim, hidden_sizes=(128, 128)):
        super().__init__()

        layers = []
        input_size = state_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        layers.append(nn.Linear(input_size, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Return action values for a batch of states."""
        return self.network(x)


class ReplayBuffer:
    """Fixed-capacity replay memory for DQN transitions."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Append one transition to the replay memory."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device):
        """Sample a random batch and return tensors on the selected device."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.as_tensor(
            np.asarray(states), dtype=torch.float32, device=device
        )
        actions = torch.as_tensor(
            np.asarray(actions), dtype=torch.int64, device=device
        )
        rewards = torch.as_tensor(
            np.asarray(rewards), dtype=torch.float32, device=device
        )
        next_states = torch.as_tensor(
            np.asarray(next_states), dtype=torch.float32, device=device
        )
        dones = torch.as_tensor(
            np.asarray(dones), dtype=torch.float32, device=device
        )

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the number of stored transitions."""
        return len(self.buffer)


def set_seed(seed):
    """Seed Python, NumPy, PyTorch, and all available CUDA devices."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_state(state, state_low, state_high):
    """Map state components from their environment bounds to [-1, 1]."""
    state = np.asarray(state, dtype=np.float32)
    state_low = np.asarray(state_low, dtype=np.float32)
    state_high = np.asarray(state_high, dtype=np.float32)

    normalized = 2.0 * (state - state_low) / (state_high - state_low) - 1.0
    return normalized.astype(np.float32, copy=False)


def linear_epsilon(step, epsilon_start, epsilon_end, epsilon_decay_steps):
    """Linearly decay epsilon and then hold it at the final value."""
    if epsilon_decay_steps <= 0:
        return float(epsilon_end)

    progress = np.clip(step / epsilon_decay_steps, 0.0, 1.0)
    return float(epsilon_start + progress * (epsilon_end - epsilon_start))


def select_greedy_action(policy_net, state, device, preprocess_fn=None):
    """Select the highest-value action for one state."""
    if preprocess_fn is not None:
        state = preprocess_fn(state)

    state_tensor = torch.as_tensor(
        state, dtype=torch.float32, device=device
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = policy_net(state_tensor)

    return int(torch.argmax(q_values, dim=1).item())


def select_epsilon_greedy_action(
    policy_net,
    state,
    epsilon,
    action_space,
    device,
    preprocess_fn=None,
):
    """Select a random action with probability epsilon, otherwise greedily."""
    if random.random() < epsilon:
        return int(action_space.sample())

    return select_greedy_action(
        policy_net,
        state,
        device,
        preprocess_fn=preprocess_fn,
    )


def dqn_train_step(
    policy_net,
    target_net,
    optimizer,
    replay_buffer,
    batch_size,
    gamma,
    device,
    min_replay_size=0,
    gradient_clip=10.0,
):
    """Perform one standard DQN update, or return None before warm-up."""
    if len(replay_buffer) < max(batch_size, min_replay_size):
        return None

    states, actions, rewards, next_states, dones = replay_buffer.sample(
        batch_size, device
    )

    current_q_values = policy_net(states).gather(
        1, actions.unsqueeze(1)
    ).squeeze(1)

    with torch.no_grad():
        max_next_q_values = target_net(next_states).max(dim=1).values
        target_q_values = rewards + gamma * max_next_q_values * (1.0 - dones)

    loss = nn.functional.smooth_l1_loss(current_q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()

    if gradient_clip is not None:
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), gradient_clip)

    optimizer.step()
    return loss.item()


def hard_update_target(policy_net, target_net):
    """Copy all policy-network parameters into the target network."""
    target_net.load_state_dict(policy_net.state_dict())


def save_dqn_checkpoint(
    path,
    policy_net,
    target_net=None,
    optimizer=None,
    metadata=None,
):
    """Save a DQN model and optional training state and metadata."""
    checkpoint = {"model_state_dict": policy_net.state_dict()}

    if target_net is not None:
        checkpoint["target_state_dict"] = target_net.state_dict()

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if metadata is not None:
        checkpoint.update(metadata)

    torch.save(checkpoint, path)


def load_dqn_checkpoint(
    path,
    state_dim,
    action_dim,
    device,
    hidden_sizes=None,
):
    """Load a checkpoint and return an evaluation-ready DQN and its data."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if hidden_sizes is None:
        hidden_sizes = checkpoint.get("hidden_sizes", (128, 128))

    policy_net = DQN(state_dim, action_dim, hidden_sizes=hidden_sizes).to(device)
    policy_net.load_state_dict(checkpoint["model_state_dict"])
    policy_net.eval()

    return policy_net, checkpoint


__all__ = [
    "DQN",
    "ReplayBuffer",
    "set_seed",
    "normalize_state",
    "linear_epsilon",
    "select_greedy_action",
    "select_epsilon_greedy_action",
    "dqn_train_step",
    "hard_update_target",
    "save_dqn_checkpoint",
    "load_dqn_checkpoint",
]
