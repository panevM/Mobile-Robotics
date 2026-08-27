"""Dual-branch DQN components for persistent MiniGrid map tensors."""

import random
from collections import deque
from dataclasses import dataclass

import numpy as np

import torch
import torch.nn as nn

from utils.dqn_utils import save_dqn_checkpoint


EXPLORATION_ACTIONS = (0, 1, 2)


@dataclass(frozen=True)
class DualScaleState:
    """Immutable replay representation of one encoded persistent map."""

    local: np.ndarray
    global_map: np.ndarray
    global_scale_log2: float


def make_dual_scale_state(encoded_map):
    """Copy an encoder result into immutable replay-safe arrays."""
    local = np.array(encoded_map.local, dtype=np.float32, copy=True)
    global_map = np.array(encoded_map.global_map, dtype=np.float32, copy=True)
    local.setflags(write=False)
    global_map.setflags(write=False)

    return DualScaleState(
        local=local,
        global_map=global_map,
        global_scale_log2=float(np.log2(encoded_map.global_scale)),
    )


class DualScaleReplayBuffer:
    """Replay memory for local map, global map, and global-scale states."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store one immutable dual-scale transition."""
        self.buffer.append((state, action, reward, next_state, done))

    @staticmethod
    def _batch_states(states, device):
        local = torch.as_tensor(
            np.stack([state.local for state in states]),
            dtype=torch.float32,
            device=device,
        )
        global_map = torch.as_tensor(
            np.stack([state.global_map for state in states]),
            dtype=torch.float32,
            device=device,
        )
        global_scale = torch.as_tensor(
            [[state.global_scale_log2] for state in states],
            dtype=torch.float32,
            device=device,
        )
        return local, global_map, global_scale

    def sample(self, batch_size, device):
        """Sample structured state and transition tensors on a device."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        state_tensors = self._batch_states(states, device)
        next_state_tensors = self._batch_states(next_states, device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=device)

        return (
            state_tensors,
            actions,
            rewards,
            next_state_tensors,
            dones,
        )

    def __len__(self):
        """Return the number of stored transitions."""
        return len(self.buffer)


class _SpatialCNN(nn.Module):
    def __init__(self, input_channels, conv_channels=(32, 64, 64)):
        super().__init__()
        layers = []
        current_channels = input_channels

        for index, output_channels in enumerate(conv_channels):
            stride = 1 if index == 0 else 2
            layers.extend(
                [
                    nn.Conv2d(
                        current_channels,
                        output_channels,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                    ),
                    nn.ReLU(),
                ]
            )
            current_channels = output_channels

        layers.extend([nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten()])
        self.network = nn.Sequential(*layers)
        self.output_size = current_channels * 2 * 2

    def forward(self, spatial_map):
        return self.network(spatial_map)


class DualScaleMapDQN(nn.Module):
    """Fuse exact local and adaptive global map features into seven Q-values."""

    def __init__(
        self,
        action_dim=7,
        local_channels=4,
        global_channels=5,
        conv_channels=(32, 64, 64),
        fusion_hidden_size=256,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.local_channels = int(local_channels)
        self.global_channels = int(global_channels)
        self.conv_channels = tuple(conv_channels)
        self.fusion_hidden_size = int(fusion_hidden_size)

        self.local_cnn = _SpatialCNN(local_channels, self.conv_channels)
        self.global_cnn = _SpatialCNN(global_channels, self.conv_channels)
        fusion_input_size = (
            self.local_cnn.output_size + self.global_cnn.output_size + 1
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, fusion_hidden_size),
            nn.ReLU(),
        )
        self.dqn_head = nn.Linear(fusion_hidden_size, action_dim)

    def forward(self, local_map, global_map, global_scale):
        """Return Q-values for a batch of dual-scale map tensors."""
        local_features = self.local_cnn(local_map)
        global_features = self.global_cnn(global_map)
        combined = torch.cat(
            [local_features, global_features, global_scale], dim=1
        )
        return self.dqn_head(self.fusion(combined))

    def architecture_config(self):
        """Return constructor metadata required to rebuild this network."""
        return {
            "action_dim": self.action_dim,
            "local_channels": self.local_channels,
            "global_channels": self.global_channels,
            "conv_channels": self.conv_channels,
            "fusion_hidden_size": self.fusion_hidden_size,
        }


def mask_q_values(q_values, allowed_actions=EXPLORATION_ACTIONS):
    """Mask disallowed actions while preserving all seven network outputs."""
    allowed_actions = tuple(int(action) for action in allowed_actions)

    if not allowed_actions:
        raise ValueError("allowed_actions cannot be empty")

    masked_q_values = torch.full_like(q_values, -torch.inf)
    masked_q_values[..., list(allowed_actions)] = q_values[
        ..., list(allowed_actions)
    ]
    return masked_q_values


def select_masked_greedy_action(
    policy_net,
    state,
    device,
    allowed_actions=EXPLORATION_ACTIONS,
):
    """Select the best permitted action for one dual-scale state."""
    local_map, global_map, global_scale = (
        DualScaleReplayBuffer._batch_states([state], device)
    )

    with torch.no_grad():
        q_values = policy_net(local_map, global_map, global_scale)
        masked_values = mask_q_values(q_values, allowed_actions)

    return int(torch.argmax(masked_values, dim=1).item())


def select_masked_epsilon_greedy_action(
    policy_net,
    state,
    epsilon,
    device,
    allowed_actions=EXPLORATION_ACTIONS,
    random_generator=None,
):
    """Select randomly with epsilon probability, otherwise masked-greedily."""
    allowed_actions = tuple(int(action) for action in allowed_actions)

    if not allowed_actions:
        raise ValueError("allowed_actions cannot be empty")

    if random_generator is None:
        explore = random.random() < epsilon
    else:
        explore = random_generator.random() < epsilon

    if explore:
        if random_generator is None:
            return random.choice(allowed_actions)

        random_index = int(random_generator.integers(len(allowed_actions)))
        return allowed_actions[random_index]

    return select_masked_greedy_action(
        policy_net, state, device, allowed_actions=allowed_actions
    )


def dual_scale_dqn_train_step(
    policy_net,
    target_net,
    optimizer,
    replay_buffer,
    batch_size,
    gamma,
    device,
    min_replay_size=0,
    gradient_clip=10.0,
    allowed_actions=EXPLORATION_ACTIONS,
):
    """Perform one masked DQN update for structured map states."""
    if len(replay_buffer) < max(batch_size, min_replay_size):
        return None

    (
        state_tensors,
        actions,
        rewards,
        next_state_tensors,
        dones,
    ) = replay_buffer.sample(batch_size, device)

    current_q_values = policy_net(*state_tensors).gather(
        1, actions.unsqueeze(1)
    ).squeeze(1)

    with torch.no_grad():
        next_q_values = target_net(*next_state_tensors)
        max_next_q_values = mask_q_values(
            next_q_values, allowed_actions
        ).max(dim=1).values
        target_q_values = rewards + gamma * max_next_q_values * (1.0 - dones)

    loss = nn.functional.smooth_l1_loss(current_q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()

    if gradient_clip is not None:
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), gradient_clip)

    optimizer.step()
    return loss.item()


def save_dual_scale_checkpoint(
    path,
    policy_net,
    target_net=None,
    optimizer=None,
    metadata=None,
):
    """Save model state plus architecture and experiment metadata."""
    combined_metadata = {
        "model_class": "DualScaleMapDQN",
        "architecture": policy_net.architecture_config(),
    }

    if metadata is not None:
        combined_metadata.update(metadata)

    save_dqn_checkpoint(
        path,
        policy_net,
        target_net=target_net,
        optimizer=optimizer,
        metadata=combined_metadata,
    )


def load_dual_scale_checkpoint(path, device):
    """Reconstruct an evaluation-ready dual-scale DQN from a checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    policy_net = DualScaleMapDQN(**checkpoint["architecture"]).to(device)
    policy_net.load_state_dict(checkpoint["model_state_dict"])
    policy_net.eval()
    return policy_net, checkpoint


__all__ = [
    "DualScaleMapDQN",
    "DualScaleReplayBuffer",
    "DualScaleState",
    "EXPLORATION_ACTIONS",
    "dual_scale_dqn_train_step",
    "load_dual_scale_checkpoint",
    "make_dual_scale_state",
    "mask_q_values",
    "save_dual_scale_checkpoint",
    "select_masked_epsilon_greedy_action",
    "select_masked_greedy_action",
]
