"""Reusable coverage milestones and completion criteria for MiniGrid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


DEFAULT_COVERAGE_MILESTONES = (
    (0.75, 5.0),
    (0.90, 10.0),
    (0.95, 20.0),
    (1.00, 50.0),
)


@dataclass(frozen=True)
class CoverageRewardConfig:
    """Coverage milestone rewards and a configurable completion criterion."""

    milestones: tuple[tuple[float, float], ...] | Mapping[float, float] = (
        DEFAULT_COVERAGE_MILESTONES
    )
    completion_threshold: float = 1.0
    completion_tolerance: float = 1e-12
    terminate_on_completion: bool = True

    def __post_init__(self):
        items = self.milestones.items() if isinstance(self.milestones, Mapping) else self.milestones
        normalized = tuple(sorted((float(threshold), float(bonus)) for threshold, bonus in items))
        if len({threshold for threshold, _ in normalized}) != len(normalized):
            raise ValueError("coverage milestone thresholds must be unique")
        if any(not 0.0 < threshold <= 1.0 for threshold, _ in normalized):
            raise ValueError("coverage milestone thresholds must be in (0, 1]")
        if not 0.0 < self.completion_threshold <= 1.0:
            raise ValueError("completion_threshold must be in (0, 1]")
        if self.completion_tolerance < 0:
            raise ValueError("completion_tolerance cannot be negative")
        object.__setattr__(self, "milestones", normalized)

    @property
    def milestone_map(self):
        return dict(self.milestones)


@dataclass(frozen=True)
class CoverageRewardUpdate:
    previous_coverage: float
    current_coverage: float
    crossed_milestones: tuple[float, ...]
    bonus: float
    coverage_success: bool


class CoverageMilestoneTracker:
    """Track one-time episode milestones without coupling to an environment or DQN."""

    def __init__(self, config=CoverageRewardConfig()):
        self.config = config
        self.reset()

    def reset(self):
        self.awarded_milestones = set()

    def is_complete(self, coverage):
        return bool(
            float(coverage) + self.config.completion_tolerance
            >= self.config.completion_threshold
        )

    def update(self, previous_coverage, current_coverage):
        previous_coverage = float(previous_coverage)
        current_coverage = float(current_coverage)
        crossed = tuple(
            threshold
            for threshold, _ in self.config.milestones
            if threshold not in self.awarded_milestones
            and previous_coverage < threshold <= current_coverage + self.config.completion_tolerance
        )
        self.awarded_milestones.update(crossed)
        bonus_by_threshold = self.config.milestone_map
        return CoverageRewardUpdate(
            previous_coverage=previous_coverage,
            current_coverage=current_coverage,
            crossed_milestones=crossed,
            bonus=float(sum(bonus_by_threshold[threshold] for threshold in crossed)),
            coverage_success=self.is_complete(current_coverage),
        )


def mapper_coverage(mapper, total_map_cells):
    if total_map_cells <= 0:
        raise ValueError("total_map_cells must be positive")
    return float(len(mapper.cells) / total_map_cells)


__all__ = [
    "CoverageMilestoneTracker",
    "CoverageRewardConfig",
    "CoverageRewardUpdate",
    "DEFAULT_COVERAGE_MILESTONES",
    "mapper_coverage",
]
