"""Composable reward calculation for persistent-map MiniGrid exploration."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ExplorationRewardConfig:
    novelty_beta: float = 1.0
    frontier_progress_beta: float = 0.15
    stationary_penalty: float = -0.5
    oscillation_penalty: float = -2.0
    escalate_oscillation_penalty: bool = False
    oscillation_penalty_max_multiplier: int = 4
    blocked_penalty: float = -2.0
    deadlock_penalty: float = -1.0
    adaptive_frontier_resolution_enabled: bool = False
    frontier_base_reward: float = 0.5
    frontier_map_scale: float = 2.0
    frontier_known_cell_scale: float = 25.0
    frontier_count_exponent: float = 0.75
    frontier_reward_cap: float = 5.0

    def __post_init__(self):
        if self.oscillation_penalty_max_multiplier < 1:
            raise ValueError("oscillation_penalty_max_multiplier must be positive")
        if self.frontier_base_reward < 0 or self.frontier_map_scale < 0:
            raise ValueError("frontier reward coefficients cannot be negative")
        if self.frontier_known_cell_scale <= 0:
            raise ValueError("frontier_known_cell_scale must be positive")
        if self.frontier_count_exponent < 0:
            raise ValueError("frontier_count_exponent cannot be negative")
        if self.frontier_reward_cap < 0:
            raise ValueError("frontier_reward_cap cannot be negative")


def compute_adaptive_frontier_resolution_reward(
    *,
    known_cells_before,
    reachable_frontier_count_before,
    resolved_frontier,
    config=ExplorationRewardConfig(),
):
    """Reward one-time frontier resolution using mapper-visible state only."""
    frontier_count = int(reachable_frontier_count_before)
    if (
        not config.adaptive_frontier_resolution_enabled
        or not resolved_frontier
        or frontier_count <= 0
    ):
        return 0.0

    known_cells = max(0, int(known_cells_before))
    numerator = config.frontier_base_reward + config.frontier_map_scale * math.log1p(
        known_cells / config.frontier_known_cell_scale
    )
    uncapped_reward = numerator / (frontier_count ** config.frontier_count_exponent)
    return float(min(config.frontier_reward_cap, uncapped_reward))


def compute_exploration_reward(
    *,
    new_cells,
    frontier_distance_before,
    frontier_distance_after,
    action,
    position_before,
    position_after,
    previous_action,
    consecutive_reversals=0,
    forward_was_known_blocked,
    coverage_milestone_bonus=0.0,
    known_cells_before=0,
    reachable_frontier_count_before=0,
    resolved_frontier=False,
    config=ExplorationRewardConfig(),
):
    """Return a fully decomposed reward using mapper-visible information only."""
    new_cells = int(new_cells)
    novelty = float(config.novelty_beta * new_cells)
    comparable_distance = (
        frontier_distance_before is not None and frontier_distance_after is not None
    )
    frontier = 0.0
    if new_cells == 0 and comparable_distance:
        frontier = float(
            config.frontier_progress_beta
            * (frontier_distance_before - frontier_distance_after)
        )
    stationary_action = tuple(position_before) == tuple(position_after)
    blocked_forward = bool(
        int(action) == 2 and stationary_action and forward_was_known_blocked
    )
    oscillating = bool(
        stationary_action
        and previous_action is not None
        and (int(previous_action), int(action)) in {(0, 1), (1, 0)}
    )
    stationary = float(config.stationary_penalty if stationary_action else 0.0)
    oscillation_multiplier = (
        min(max(1, int(consecutive_reversals)), config.oscillation_penalty_max_multiplier)
        if oscillating and config.escalate_oscillation_penalty
        else 1
    )
    oscillation = float(
        config.oscillation_penalty * oscillation_multiplier if oscillating else 0.0
    )
    blocked = float(config.blocked_penalty if blocked_forward else 0.0)
    coverage_milestone = float(coverage_milestone_bonus)
    frontier_resolution = compute_adaptive_frontier_resolution_reward(
        known_cells_before=known_cells_before,
        reachable_frontier_count_before=reachable_frontier_count_before,
        resolved_frontier=resolved_frontier,
        config=config,
    )
    distance_delta = (
        frontier_distance_before - frontier_distance_after
        if comparable_distance
        else None
    )
    return {
        "total": (
            novelty
            + frontier
            + frontier_resolution
            + coverage_milestone
            + stationary
            + oscillation
            + blocked
        ),
        "novelty": novelty,
        "frontier": frontier,
        "frontier_progress": frontier,
        "frontier_resolution": frontier_resolution,
        "coverage_milestone": coverage_milestone,
        "stationary": stationary,
        "oscillation": oscillation,
        "blocked": blocked,
        "deadlock": 0.0,
        "stationary_action": stationary_action,
        "oscillating": oscillating,
        "oscillation_multiplier": int(oscillation_multiplier if oscillating else 0),
        "blocked_forward": blocked_forward,
        "frontier_reduced": bool(new_cells == 0 and distance_delta is not None and distance_delta > 0),
        "frontier_increased": bool(new_cells == 0 and distance_delta is not None and distance_delta < 0),
        "no_reachable_frontier": frontier_distance_before is None,
        "frontier_resolved": bool(resolved_frontier and frontier_resolution > 0.0),
    }


__all__ = [
    "ExplorationRewardConfig",
    "compute_adaptive_frontier_resolution_reward",
    "compute_exploration_reward",
]
