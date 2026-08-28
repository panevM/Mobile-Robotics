"""Reusable exploration deadlock tracking independent of the DQN."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DeadlockConfig:
    oscillation_threshold: int = 8
    stationary_threshold: int = 8
    no_progress_timeout_multiplier: float = 2.0

    def __post_init__(self):
        if self.oscillation_threshold < 2:
            raise ValueError("oscillation_threshold must be at least 2")
        if self.stationary_threshold < 1:
            raise ValueError("stationary_threshold must be positive")
        if self.no_progress_timeout_multiplier <= 0:
            raise ValueError("no_progress_timeout_multiplier must be positive")

    def no_progress_timeout(self, width, height):
        return max(1, int(math.ceil(self.no_progress_timeout_multiplier * (width + height))))


class DeadlockTracker:
    """Track Experiment 3.1-compatible oscillation, stationary, and progress failures."""

    def __init__(self, config=DeadlockConfig()):
        self.config = config
        self.reset()

    def reset(self, width=None, height=None, no_progress_timeout=None):
        if no_progress_timeout is not None:
            self.no_progress_timeout = int(no_progress_timeout)
        elif width is not None and height is not None:
            self.no_progress_timeout = self.config.no_progress_timeout(width, height)
        else:
            self.no_progress_timeout = None
        self.oscillation_counter = 0
        self.stationary_counter = 0
        self.steps_since_exploration_progress = 0

    def update(
        self,
        *,
        action,
        previous_action,
        position_changed,
        new_cells,
        frontier_distance_before,
        frontier_distance_after,
    ):
        frontier_progress = bool(
            frontier_distance_before is not None
            and frontier_distance_after is not None
            and frontier_distance_after < frontier_distance_before
        )
        exploration_progress = bool(new_cells > 0 or frontier_progress)
        productive = bool(position_changed or exploration_progress)
        reversal = bool(
            action in (0, 1)
            and previous_action in (0, 1)
            and action != previous_action
            and not position_changed
            and new_cells == 0
            and not frontier_progress
        )
        if productive:
            self.oscillation_counter = 0
        elif reversal:
            self.oscillation_counter = self.oscillation_counter + 1 if self.oscillation_counter else 2
        else:
            self.oscillation_counter = 0
        if position_changed or new_cells > 0:
            self.stationary_counter = 0
        else:
            self.stationary_counter += 1
        if exploration_progress:
            self.steps_since_exploration_progress = 0
        else:
            self.steps_since_exploration_progress += 1

        if self.oscillation_counter >= self.config.oscillation_threshold:
            reason = "oscillation_deadlock"
        elif self.stationary_counter >= self.config.stationary_threshold:
            reason = "stationary_deadlock"
        elif (
            self.no_progress_timeout is not None
            and self.steps_since_exploration_progress >= self.no_progress_timeout
        ):
            reason = "no_progress_deadlock"
        else:
            reason = None
        return {
            "oscillation_counter": self.oscillation_counter,
            "stationary_counter": self.stationary_counter,
            "frontier_progress_this_step": frontier_progress,
            "exploration_progress_this_step": exploration_progress,
            "steps_since_exploration_progress": self.steps_since_exploration_progress,
            "no_progress_timeout": self.no_progress_timeout,
            "deadlock_triggered": reason is not None,
            "deadlock_type": reason,
        }


__all__ = ["DeadlockConfig", "DeadlockTracker"]
