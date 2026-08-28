"""Public facade for configuration-driven MiniGrid exploration experiments."""

from utils.minigrid_coverage_rewards import CoverageMilestoneTracker, CoverageRewardConfig
from utils.minigrid_completion_verification import verify_literal_coverage_attainable
from utils.minigrid_deadlocks import DeadlockConfig, DeadlockTracker
from utils.minigrid_experiment_config import (
    ContinuationConfig,
    EnvironmentConfig,
    MiniGridExplorationExperimentConfig,
    NetworkConfig,
    OutputConfig,
    TensorConfig,
    TrainingConfig,
    ValidationConfig,
)
from utils.minigrid_exploration_checkpoints import CheckpointManager
from utils.minigrid_exploration_diagnostics import (
    save_comparison_artifacts,
    save_diagnostic_episode,
)
from utils.minigrid_exploration_episode import ExplorationEpisodeRunner
from utils.minigrid_exploration_evaluation import MiniGridExplorationEvaluator
from utils.minigrid_exploration_metrics import EpisodeResult, summarize_evaluations
from utils.minigrid_exploration_policy import ExplorationActionSelector, episode_linear_epsilon
from utils.minigrid_exploration_rewards import (
    ExplorationRewardConfig,
    compute_adaptive_frontier_resolution_reward,
)
from utils.minigrid_exploration_topology import (
    frontier_target_was_resolved,
    reachable_frontier_regions,
)
from utils.minigrid_exploration_trainer import MiniGridExplorationTrainer


__all__ = [
    "CheckpointManager",
    "ContinuationConfig",
    "CoverageMilestoneTracker",
    "CoverageRewardConfig",
    "DeadlockConfig",
    "DeadlockTracker",
    "EnvironmentConfig",
    "EpisodeResult",
    "ExplorationActionSelector",
    "ExplorationEpisodeRunner",
    "ExplorationRewardConfig",
    "MiniGridExplorationEvaluator",
    "MiniGridExplorationExperimentConfig",
    "MiniGridExplorationTrainer",
    "NetworkConfig",
    "OutputConfig",
    "TensorConfig",
    "TrainingConfig",
    "ValidationConfig",
    "compute_adaptive_frontier_resolution_reward",
    "episode_linear_epsilon",
    "frontier_target_was_resolved",
    "reachable_frontier_regions",
    "save_comparison_artifacts",
    "save_diagnostic_episode",
    "summarize_evaluations",
    "verify_literal_coverage_attainable",
]
