"""Configurable masked action selection and episode-based exploration schedules."""

from __future__ import annotations

import numpy as np

from utils.minigrid_dual_scale_dqn import select_masked_greedy_action


def episode_linear_epsilon(episode_index, num_episodes, epsilon_start, epsilon_end):
    if num_episodes < 1 or not 0 <= episode_index < num_episodes:
        raise ValueError("episode index must be inside a positive-length run")
    if epsilon_start <= 0 or epsilon_end <= 0 or epsilon_end > epsilon_start:
        raise ValueError("epsilon requires 0 < end <= start")
    if num_episodes == 1 or episode_index == num_episodes - 1:
        return float(epsilon_end)
    if episode_index == 0:
        return float(epsilon_start)
    fraction = episode_index / (num_episodes - 1)
    return float(epsilon_start + fraction * (epsilon_end - epsilon_start))


class ExplorationActionSelector:
    def __init__(self, allowed_actions):
        self.allowed_actions = tuple(int(action) for action in allowed_actions)
        if not self.allowed_actions:
            raise ValueError("allowed_actions cannot be empty")

    def select_action(
        self,
        policy_net,
        state,
        device,
        *,
        epsilon=0.0,
        random_generator=None,
        random_controller=False,
    ):
        rng = random_generator or np.random.default_rng()
        if random_controller or rng.random() < epsilon:
            index = int(rng.integers(len(self.allowed_actions)))
            return self.allowed_actions[index], "random"
        return (
            select_masked_greedy_action(
                policy_net, state, device, allowed_actions=self.allowed_actions
            ),
            "greedy",
        )


__all__ = ["ExplorationActionSelector", "episode_linear_epsilon"]
