"""Nested configuration objects for reusable MiniGrid exploration experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from utils.minigrid_coverage_rewards import CoverageRewardConfig
from utils.minigrid_deadlocks import DeadlockConfig
from utils.minigrid_exploration_rewards import ExplorationRewardConfig


@dataclass(frozen=True)
class EnvironmentConfig:
    width: int = 19
    height: int = 19
    min_rooms: int = 4
    max_rooms: int = 6
    min_room_size: int = 5
    doorway_width: int = 1
    max_steps: int = 200

    def to_kwargs(self):
        return asdict(self)


@dataclass(frozen=True)
class TensorConfig:
    local_size: int = 15
    global_size: int = 15
    unknown_padding: int = 2
    hysteresis_bins: int = 1

    def to_kwargs(self):
        return asdict(self)


@dataclass(frozen=True)
class NetworkConfig:
    action_dim: int = 7
    local_channels: int = 4
    global_channels: int = 5
    conv_channels: tuple[int, ...] = (32, 64, 64)
    fusion_hidden_size: int = 256

    def to_kwargs(self):
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    num_episodes: int = 1_000
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 64
    replay_capacity: int = 100_000
    minimum_replay_size: int = 5_000
    target_update_frequency: int = 1_000
    gradient_clip: float = 10.0
    epsilon_start: float = 0.10
    epsilon_end: float = 0.02
    seed_start: int = 0
    seed_count: int = 10_000
    random_seed: int = 46


@dataclass(frozen=True)
class ValidationConfig:
    seeds: tuple[int, ...] = tuple(range(20_000, 20_100))
    checkpoint_seed_count: int = 20
    diagnostic_seeds: tuple[int, ...] = (20_000, 20_050, 20_099)
    frequency: int = 100
    large_validation_frequency: int = 500
    large_validation_episode_count: int = 100
    best_mean_coverage_tolerance: float = 1e-3
    coverage_thresholds: tuple[float, ...] = (0.50, 0.75, 0.90, 0.95, 1.00)
    ascii_interval: int = 5
    show_discovered_map_in_gif: bool = True

    def __post_init__(self):
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in self.seeds))
        object.__setattr__(
            self, "diagnostic_seeds", tuple(int(seed) for seed in self.diagnostic_seeds)
        )
        if self.frequency <= 0 or self.large_validation_frequency <= 0:
            raise ValueError("validation frequencies must be positive")
        if not 0 < self.checkpoint_seed_count <= len(self.seeds):
            raise ValueError("checkpoint_seed_count must fit within validation seeds")
        if not 0 < self.large_validation_episode_count <= len(self.seeds):
            raise ValueError("large_validation_episode_count must fit within validation seeds")
        if self.best_mean_coverage_tolerance < 0:
            raise ValueError("best_mean_coverage_tolerance cannot be negative")

    @property
    def checkpoint_seeds(self):
        return self.seeds[: self.checkpoint_seed_count]

    @property
    def large_validation_seeds(self):
        return self.seeds[: self.large_validation_episode_count]


@dataclass(frozen=True)
class ContinuationConfig:
    enabled: bool = False
    checkpoint: str | Path = "latest"
    additional_episodes: int | None = None
    keep_epsilon_at_floor: bool = True
    mode: str = "resume"

    def __post_init__(self):
        checkpoint = self.checkpoint
        if isinstance(checkpoint, str) and checkpoint.lower() in {"latest", "best"}:
            object.__setattr__(self, "checkpoint", checkpoint.lower())
        else:
            object.__setattr__(self, "checkpoint", Path(checkpoint))
        mode = str(self.mode).lower()
        if mode not in {"resume", "fine_tune"}:
            raise ValueError("continuation mode must be 'resume' or 'fine_tune'")
        object.__setattr__(self, "mode", mode)
        if self.additional_episodes is not None and self.additional_episodes <= 0:
            raise ValueError("additional_episodes must be positive when provided")


@dataclass(frozen=True)
class OutputConfig:
    directory: Path | None = None

    def __post_init__(self):
        if self.directory is not None:
            object.__setattr__(self, "directory", Path(self.directory))


@dataclass(frozen=True)
class MiniGridExplorationExperimentConfig:
    name: str
    initial_checkpoint: Path
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    tensor: TensorConfig = field(default_factory=TensorConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    reward: ExplorationRewardConfig = field(default_factory=ExplorationRewardConfig)
    coverage: CoverageRewardConfig = field(default_factory=CoverageRewardConfig)
    deadlock: DeadlockConfig = field(default_factory=DeadlockConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    continuation: ContinuationConfig = field(default_factory=ContinuationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    allowed_actions: tuple[int, ...] = (0, 1, 2)

    def __post_init__(self):
        object.__setattr__(self, "initial_checkpoint", Path(self.initial_checkpoint))
        object.__setattr__(self, "allowed_actions", tuple(int(a) for a in self.allowed_actions))
        if self.output.directory is None:
            object.__setattr__(self, "output", OutputConfig(self.name))
        if not self.allowed_actions:
            raise ValueError("allowed_actions cannot be empty")

    def to_dict(self):
        data = asdict(self)
        data["initial_checkpoint"] = str(self.initial_checkpoint)
        data["output"]["directory"] = str(self.output.directory)
        data["continuation"]["checkpoint"] = str(self.continuation.checkpoint)
        return data

    @classmethod
    def from_checkpoint(cls, *, name, checkpoint_path, output_directory=None):
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        architecture = checkpoint["architecture"]
        environment = checkpoint["training_environment_config"]
        tensor = checkpoint["tensor_config"]
        reward = checkpoint["reward_config"]
        source_deadlock = checkpoint["deadlock_config"]
        source_coverage = checkpoint.get("coverage_config")
        hyper = checkpoint.get("hyperparameters")
        if hyper is None:
            modular_config = checkpoint["experiment_config"]
            training_data = modular_config["training"]
            validation_data = modular_config["validation"]
            seed_start = training_data["seed_start"]
            seed_end = seed_start + training_data["seed_count"] - 1
            hyper = {
                **training_data,
                "training_seed_range": (seed_start, seed_end),
                "validation_frequency": validation_data["frequency"],
                "allowed_actions": modular_config["allowed_actions"],
            }
        else:
            seed_start, seed_end = hyper["training_seed_range"]
        deadlock_penalty = reward.get(
            "deadlock_penalty", source_deadlock.get("deadlock_penalty", -1.0)
        )
        return cls(
            name=name,
            initial_checkpoint=checkpoint_path,
            environment=EnvironmentConfig(**environment),
            tensor=TensorConfig(**tensor),
            network=NetworkConfig(**architecture),
            reward=ExplorationRewardConfig(
                novelty_beta=reward["novelty_beta"],
                frontier_progress_beta=reward["frontier_progress_beta"],
                stationary_penalty=reward["stationary_penalty"],
                oscillation_penalty=reward["oscillation_penalty"],
                escalate_oscillation_penalty=reward.get(
                    "escalate_oscillation_penalty", False
                ),
                oscillation_penalty_max_multiplier=reward.get(
                    "oscillation_penalty_max_multiplier", 4
                ),
                blocked_penalty=reward["blocked_penalty"],
                deadlock_penalty=deadlock_penalty,
            ),
            coverage=(
                CoverageRewardConfig(**source_coverage)
                if source_coverage is not None
                else CoverageRewardConfig(milestones=(), terminate_on_completion=False)
            ),
            deadlock=DeadlockConfig(
                oscillation_threshold=source_deadlock.get(
                    "oscillation_threshold",
                    source_deadlock.get("oscillation_deadlock_threshold"),
                ),
                stationary_threshold=source_deadlock.get(
                    "stationary_threshold",
                    source_deadlock.get("stationary_deadlock_threshold"),
                ),
                no_progress_timeout_multiplier=source_deadlock["no_progress_timeout_multiplier"],
            ),
            training=TrainingConfig(
                num_episodes=1_000,
                gamma=hyper["gamma"],
                learning_rate=hyper["learning_rate"],
                batch_size=hyper["batch_size"],
                replay_capacity=hyper["replay_capacity"],
                minimum_replay_size=hyper["minimum_replay_size"],
                target_update_frequency=hyper["target_update_frequency"],
                gradient_clip=hyper["gradient_clip"],
                epsilon_start=hyper["epsilon_start"],
                epsilon_end=hyper["epsilon_end"],
                seed_start=seed_start,
                seed_count=seed_end - seed_start + 1,
            ),
            validation=ValidationConfig(frequency=hyper["validation_frequency"]),
            output=OutputConfig(output_directory),
            allowed_actions=tuple(hyper["allowed_actions"]),
        )


__all__ = [
    "ContinuationConfig",
    "EnvironmentConfig",
    "MiniGridExplorationExperimentConfig",
    "NetworkConfig",
    "OutputConfig",
    "TensorConfig",
    "TrainingConfig",
    "ValidationConfig",
]
