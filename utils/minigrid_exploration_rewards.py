"""Composable reward calculation for persistent-map MiniGrid exploration."""

from __future__ import annotations

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

    def __post_init__(self):
        if self.oscillation_penalty_max_multiplier < 1:
            raise ValueError("oscillation_penalty_max_multiplier must be positive")


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
    distance_delta = (
        frontier_distance_before - frontier_distance_after
        if comparable_distance
        else None
    )
    return {
        "total": novelty + frontier + coverage_milestone + stationary + oscillation + blocked,
        "novelty": novelty,
        "frontier": frontier,
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
    }


__all__ = ["ExplorationRewardConfig", "compute_exploration_reward"]
