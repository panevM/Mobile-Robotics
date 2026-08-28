"""Configuration-driven trainer for dual-scale MiniGrid exploration DQNs."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch
import torch.optim as optim

from utils.dqn_utils import hard_update_target, set_seed
from utils.minigrid_dual_scale_dqn import (
    DualScaleMapDQN,
    DualScaleReplayBuffer,
    dual_scale_dqn_train_step,
    mask_q_values,
)
from utils.minigrid_exploration_checkpoints import CheckpointManager
from utils.minigrid_exploration_diagnostics import save_json
from utils.minigrid_exploration_episode import ExplorationEpisodeRunner
from utils.minigrid_exploration_evaluation import MiniGridExplorationEvaluator
from utils.minigrid_exploration_policy import episode_linear_epsilon
from utils.minigrid_exploration_rewards import ExplorationRewardConfig
from utils.procedural_rooms_env import ENVIRONMENT_ID


class MiniGridExplorationTrainer:
    """Own network/replay/optimization while delegating episode mechanics."""

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.checkpoints = CheckpointManager(config.output.directory)
        self.runner = ExplorationEpisodeRunner(config, device)
        self.evaluator = MiniGridExplorationEvaluator(config, device)
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        self.replay_buffer = None
        self.source_checkpoint = None
        self.source_checkpoint_path = None
        self.training_history = []
        self.validation_history = []
        self.best_validation_metadata = None
        self.checkpoint_validation = None
        self.total_steps = 0
        self.completed_episodes = 0
        self.random_generator = None
        self.is_continuation = bool(config.continuation.enabled)

    def initialize(self):
        if self.is_continuation:
            self._initialize_continuation()
        else:
            self._initialize_finetune()
        return self

    def _new_network(self):
        return DualScaleMapDQN(**self.config.network.to_kwargs()).to(self.device)

    def _new_optimizer(self):
        return optim.Adam(
            self.policy_net.parameters(), lr=self.config.training.learning_rate
        )

    def _initialize_finetune(self):
        """Inherit policy weights while intentionally resetting learning state."""
        set_seed(self.config.training.random_seed)
        self.source_checkpoint_path = self.config.initial_checkpoint
        self.policy_net, self.source_checkpoint = self.checkpoints.load_inherited_policy(
            self.source_checkpoint_path, self.device
        )
        if self.source_checkpoint["architecture"] != self.config.network.to_kwargs():
            raise ValueError("configured network must exactly match the inherited architecture")
        self.target_net = self._new_network()
        hard_update_target(self.policy_net, self.target_net)
        self.optimizer = self._new_optimizer()
        self.replay_buffer = DualScaleReplayBuffer(self.config.training.replay_capacity)
        self.random_generator = np.random.default_rng(self.config.training.random_seed)
        if len(self.replay_buffer) != 0 or len(self.optimizer.state) != 0:
            raise RuntimeError("fine-tuning must begin with fresh replay and optimizer state")
        self.target_net.eval()
        self.policy_net.train()

    @staticmethod
    def _require_match(label, configured, checkpoint_value):
        if checkpoint_value is None:
            raise ValueError(f"continuation checkpoint is missing {label}")
        if configured != checkpoint_value:
            raise ValueError(
                f"continuation checkpoint {label} mismatch: "
                f"configured={configured!r}, checkpoint={checkpoint_value!r}"
            )

    def _validate_continuation_compatibility(self, checkpoint):
        hyperparameters = checkpoint.get("hyperparameters", {})
        experiment_config = checkpoint.get("experiment_config", {})
        checkpoint_actions = hyperparameters.get(
            "allowed_actions", experiment_config.get("allowed_actions")
        )
        self._require_match(
            "experiment name", self.config.name, checkpoint.get("experiment_name")
        )
        self._require_match(
            "network architecture", self.config.network.to_kwargs(), checkpoint.get("architecture")
        )
        self._require_match(
            "tensor dimensions", self.config.tensor.to_kwargs(), checkpoint.get("tensor_config")
        )
        self._require_match(
            "allowed actions", self.config.allowed_actions, tuple(checkpoint_actions or ())
        )
        self._require_match(
            "environment type", ENVIRONMENT_ID, checkpoint.get("environment_id", ENVIRONMENT_ID)
        )
        self._require_match(
            "environment configuration",
            self.config.environment.to_kwargs(),
            checkpoint.get("training_environment_config"),
        )
        checkpoint_reward = asdict(
            ExplorationRewardConfig(**checkpoint.get("reward_config", {}))
        )
        configured_reward = asdict(self.config.reward)
        if self.config.continuation.mode == "fine_tune":
            oscillation_fields = {
                "oscillation_penalty",
                "escalate_oscillation_penalty",
                "oscillation_penalty_max_multiplier",
            }
            for field_name in configured_reward.keys() - oscillation_fields:
                self._require_match(
                    f"reward component '{field_name}'",
                    configured_reward[field_name],
                    checkpoint_reward[field_name],
                )
        else:
            self._require_match(
                "reward configuration", configured_reward, checkpoint_reward
            )
        self._require_match(
            "coverage configuration", asdict(self.config.coverage), checkpoint.get("coverage_config")
        )
        self._require_match(
            "deadlock configuration", asdict(self.config.deadlock), checkpoint.get("deadlock_config")
        )

    def _initialize_continuation(self):
        """Restore an existing experiment's trainable state, with a fresh replay buffer."""
        set_seed(self.config.training.random_seed)
        self.source_checkpoint_path = self.checkpoints.resolve_continuation_checkpoint(
            self.config.continuation.checkpoint
        )
        checkpoint = torch.load(
            self.source_checkpoint_path, map_location=self.device, weights_only=False
        )
        self._validate_continuation_compatibility(checkpoint)
        self.source_checkpoint = checkpoint

        self.policy_net = self._new_network()
        self.policy_net.load_state_dict(checkpoint["model_state_dict"])
        self.target_net = self._new_network()
        self.optimizer = self._new_optimizer()
        if self.config.continuation.mode == "fine_tune":
            hard_update_target(self.policy_net, self.target_net)
        else:
            target_state = checkpoint.get("target_state_dict", checkpoint["model_state_dict"])
            self.target_net.load_state_dict(target_state)
        if (
            self.config.continuation.mode == "resume"
            and "optimizer_state_dict" in checkpoint
        ):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.replay_buffer = DualScaleReplayBuffer(self.config.training.replay_capacity)
        self.training_history = list(checkpoint.get("training_metrics", ()))
        self.validation_history = list(checkpoint.get("validation_history", ()))
        self.best_validation_metadata = checkpoint.get("best_validation_metadata")
        self.total_steps = int(checkpoint.get("total_steps", 0))
        self.completed_episodes = int(checkpoint.get("training_episode", 0))
        self.random_generator = np.random.default_rng(
            self.config.training.random_seed + self.completed_episodes
        )
        if checkpoint.get("numpy_rng_state") is not None:
            self.random_generator.bit_generator.state = checkpoint["numpy_rng_state"]

        self.target_net.eval()
        self.policy_net.train()

    def _ensure_initialized(self):
        if self.policy_net is None:
            self.initialize()

    def _training_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
        loss = dual_scale_dqn_train_step(
            self.policy_net,
            self.target_net,
            self.optimizer,
            self.replay_buffer,
            self.config.training.batch_size,
            self.config.training.gamma,
            self.device,
            min_replay_size=self.config.training.minimum_replay_size,
            gradient_clip=self.config.training.gradient_clip,
        )
        self.total_steps += 1
        if self.total_steps % self.config.training.target_update_frequency == 0:
            hard_update_target(self.policy_net, self.target_net)
        return loss

    def _metadata(self, episode, epsilon, fraction):
        epsilon_schedule = (
            "floor_during_continuation"
            if self.is_continuation and self.config.continuation.keep_epsilon_at_floor
            else "linear_by_training_episode"
        )
        hyperparameters = {
            **asdict(self.config.training),
            "epsilon_schedule": epsilon_schedule,
            "validation_frequency": self.config.validation.frequency,
            "large_validation_frequency": self.config.validation.large_validation_frequency,
            "large_validation_episode_count": self.config.validation.large_validation_episode_count,
            "allowed_actions": self.config.allowed_actions,
            "training_seed_range": (
                self.config.training.seed_start,
                self.config.training.seed_start + self.config.training.seed_count - 1,
            ),
        }
        return {
            "experiment_name": self.config.name,
            "experiment_stage": self.config.name.removeprefix("experiment_").replace("_", "."),
            "environment_id": ENVIRONMENT_ID,
            "initialization": (
                "checkpoint_policy_finetune"
                if self.is_continuation and self.config.continuation.mode == "fine_tune"
                else "checkpoint_resume"
                if self.is_continuation
                else "checkpoint_finetune"
            ),
            "pretrained_checkpoint": str(self.source_checkpoint_path.resolve()),
            "source_experiment_stage": self.source_checkpoint.get("experiment_stage"),
            "source_training_episode": self.source_checkpoint.get("training_episode"),
            "replay_buffer_initial_size": 0,
            "replay_buffer_restored": False,
            "optimizer_initialization": (
                "restored_from_checkpoint"
                if self.is_continuation
                and self.config.continuation.mode == "resume"
                and "optimizer_state_dict" in self.source_checkpoint
                else "fresh_adam"
            ),
            "target_initialization": (
                "restored_from_checkpoint"
                if self.is_continuation
                and self.config.continuation.mode == "resume"
                and "target_state_dict" in self.source_checkpoint
                else "hard_update_from_inherited_policy"
            ),
            "continuation_mode": (
                self.config.continuation.mode if self.is_continuation else None
            ),
            "training_episode": int(episode),
            "total_steps": int(self.total_steps),
            "epsilon": float(epsilon),
            "training_fraction": float(fraction),
            "epsilon_schedule": epsilon_schedule,
            "architecture": self.config.network.to_kwargs(),
            "reward_config": asdict(self.config.reward),
            "coverage_config": asdict(self.config.coverage),
            "deadlock_config": asdict(self.config.deadlock),
            "tensor_config": self.config.tensor.to_kwargs(),
            "training_environment_config": self.config.environment.to_kwargs(),
            "hyperparameters": hyperparameters,
            "experiment_config": self.config.to_dict(),
            "training_metrics": self.training_history,
            "validation_history": self.validation_history,
            "best_validation_metadata": self.best_validation_metadata,
            "checkpoint_validation": self.checkpoint_validation,
            "numpy_rng_state": self.random_generator.bit_generator.state,
        }

    def _predicted_q_metrics(self, state):
        local, global_map, scale = DualScaleReplayBuffer._batch_states([state], self.device)
        with torch.no_grad():
            values = mask_q_values(
                self.policy_net(local, global_map, scale),
                allowed_actions=self.config.allowed_actions,
            )[:, list(self.config.allowed_actions)]
        return {
            "mean_predicted_q": float(values.mean().item()),
            "max_predicted_q": float(values.max().item()),
        }

    @staticmethod
    def _total_deadlock_fraction(summary):
        return float(
            summary["fraction_oscillation_deadlock"]
            + summary["fraction_stationary_deadlock"]
            + summary["fraction_no_progress_deadlock"]
        )

    @classmethod
    def _is_better_validation(cls, candidate, current, tolerance):
        if current is None:
            return True
        source_priority = {"lightweight": 0, "large": 1}
        candidate_priority = source_priority[candidate["validation_type"]]
        current_priority = source_priority[current["validation_type"]]
        if candidate_priority != current_priority:
            return candidate_priority > current_priority

        candidate_metrics = candidate["metrics"]
        current_metrics = current["metrics"]
        coverage_difference = (
            candidate_metrics["mean_final_coverage"]
            - current_metrics["mean_final_coverage"]
        )
        if coverage_difference > tolerance:
            return True
        if coverage_difference < -tolerance:
            return False
        candidate_tiebreak = (
            candidate_metrics["fraction_reaching_100"],
            candidate_metrics["fraction_reaching_95"],
            -cls._total_deadlock_fraction(candidate_metrics),
        )
        current_tiebreak = (
            current_metrics["fraction_reaching_100"],
            current_metrics["fraction_reaching_95"],
            -cls._total_deadlock_fraction(current_metrics),
        )
        return candidate_tiebreak > current_tiebreak

    def _record_validation(self, episode, validation_type, seeds):
        results = self.evaluator.evaluate(self.policy_net, seeds=seeds)
        summary = self.evaluator.summarize(results)
        if summary["random_action_count"] != 0:
            raise RuntimeError("greedy validation unexpectedly used random actions")
        record = {
            "episode": int(episode),
            "validation_type": validation_type,
            "episode_count": len(seeds),
            "metrics": summary,
        }
        self.validation_history.append(record)
        return record

    def _print_validation(self, record):
        summary = record["metrics"]
        label = "Large validation" if record["validation_type"] == "large" else "Validation"
        prefix = f"{label} {record['episode']}:"
        if record["validation_type"] == "large":
            prefix += f" episodes={record['episode_count']} |"
        print(
            f"{prefix} coverage={summary['mean_final_coverage']:.1%} | "
            f"median={summary['median_final_coverage']:.1%} | "
            f"success={summary['fraction_coverage_success']:.1%} | "
            f"osc={summary['fraction_oscillation_deadlock']:.1%} | "
            f"stationary={summary['fraction_stationary_deadlock']:.1%} | "
            f"no-progress={summary['fraction_no_progress_deadlock']:.1%} | "
            f"50%={summary['fraction_reaching_50']:.1%} | "
            f"75%={summary['fraction_reaching_75']:.1%} | "
            f"90%={summary['fraction_reaching_90']:.1%} | "
            f"95%={summary['fraction_reaching_95']:.1%} | "
            f"100%={summary['fraction_reaching_100']:.1%}"
        )

    def _validate_and_checkpoint(self, episode, epsilon, fraction):
        run_lightweight = episode % self.config.validation.frequency == 0
        run_large = episode % self.config.validation.large_validation_frequency == 0
        records = {}
        if run_lightweight:
            records["lightweight"] = self._record_validation(
                episode, "lightweight", self.config.validation.checkpoint_seeds
            )
            diagnostic_directory = self.checkpoints.artifact_directory / f"episode_{episode:04d}"
            self.evaluator.evaluate(
                self.policy_net,
                seeds=self.config.validation.diagnostic_seeds,
                artifact_directory=diagnostic_directory,
                checkpoint_episode=episode,
            )
            save_json(
                diagnostic_directory / "validation_summary.json",
                records["lightweight"]["metrics"],
            )
            self._print_validation(records["lightweight"])
        if run_large:
            records["large"] = self._record_validation(
                episode, "large", self.config.validation.large_validation_seeds
            )
            self._print_validation(records["large"])

        candidate = records.get("large", records.get("lightweight"))
        is_best = self._is_better_validation(
            candidate,
            self.best_validation_metadata,
            self.config.validation.best_mean_coverage_tolerance,
        )
        if is_best:
            self.best_validation_metadata = candidate
        self.checkpoint_validation = records
        metadata = self._metadata(episode, epsilon, fraction)
        self.checkpoints.save(
            policy=self.policy_net,
            target=self.target_net,
            optimizer=self.optimizer,
            metadata=metadata,
            episode=episode,
            is_best=is_best,
        )
        save_json(self.checkpoints.metrics_directory / "training_metrics.json", self.training_history)
        save_json(self.checkpoints.metrics_directory / "validation_history.json", self.validation_history)
        save_json(
            self.checkpoints.metrics_directory / "best_validation.json",
            self.best_validation_metadata,
        )
        self.policy_net.train()

    def _episodes_for_this_run(self):
        if self.is_continuation and self.config.continuation.additional_episodes is not None:
            return self.config.continuation.additional_episodes
        return self.config.training.num_episodes

    def _epsilon_for_run_episode(self, run_index, run_episode_count):
        if self.is_continuation and self.config.continuation.keep_epsilon_at_floor:
            return self.config.training.epsilon_end
        return episode_linear_epsilon(
            run_index,
            run_episode_count,
            self.config.training.epsilon_start,
            self.config.training.epsilon_end,
        )

    def train(self):
        self._ensure_initialized()
        self.checkpoints.prepare()
        run_episode_count = self._episodes_for_this_run()
        starting_episode = self.completed_episodes
        target_episode = starting_episode + run_episode_count
        if self.is_continuation:
            print(f"Resuming {self.config.name}")
            print(f"Continuation mode: {self.config.continuation.mode}")
            print(f"Previous episodes: {starting_episode}")
            print(f"Additional episodes: {run_episode_count}")
            print(f"Target cumulative episodes: {target_episode}")
            print("Replay buffer: fresh (checkpoint replay persistence is unavailable)")

        final_epsilon = self.config.training.epsilon_end
        for run_index in range(run_episode_count):
            episode = starting_episode + run_index + 1
            fraction = 1.0 if run_episode_count == 1 else run_index / (run_episode_count - 1)
            epsilon = self._epsilon_for_run_episode(run_index, run_episode_count)
            final_epsilon = epsilon
            seed = self.config.training.seed_start + (
                (episode - 1) % self.config.training.seed_count
            )
            result = self.runner.run(
                seed=seed,
                policy_net=self.policy_net,
                epsilon=epsilon,
                random_generator=self.random_generator,
                transition_handler=self._training_transition,
            )
            metrics = result.checkpoint_dict()
            metrics.update(
                {
                    "episode": episode,
                    "epsilon": epsilon,
                    "training_fraction": fraction,
                    "epsilon_schedule": (
                        "floor_during_continuation"
                        if self.is_continuation
                        and self.config.continuation.keep_epsilon_at_floor
                        else "linear_by_training_episode"
                    ),
                }
            )
            self.training_history.append(metrics)
            self.completed_episodes = episode
            if episode % 25 == 0:
                recent = self.training_history[-25:]
                print(
                    f"Episode {episode:4d}/{target_episode} | run={fraction:.1%} | "
                    f"epsilon={epsilon:.3f} | coverage={np.mean([m['final_coverage'] for m in recent]):.1%} | "
                    f"success={np.mean([m['coverage_success'] for m in recent]):.1%} | "
                    f"milestones={np.mean([m['coverage_milestone_reward'] for m in recent]):.2f} | "
                    f"loss={np.nanmean([m['training_loss'] for m in recent]):.4f}"
                )
            if (
                episode % self.config.validation.frequency == 0
                or episode % self.config.validation.large_validation_frequency == 0
            ):
                self._validate_and_checkpoint(episode, epsilon, fraction)

        metadata = self._metadata(target_episode, final_epsilon, 1.0)
        self.checkpoints.save(
            policy=self.policy_net,
            target=self.target_net,
            optimizer=self.optimizer,
            metadata=metadata,
            final=True,
        )
        save_json(self.checkpoints.metrics_directory / "training_metrics.json", self.training_history)
        save_json(self.checkpoints.metrics_directory / "validation_history.json", self.validation_history)
        return self.policy_net, metadata

    def evaluate(
        self,
        *,
        policy_net=None,
        seeds=None,
        artifact_directory=None,
        environment_override=None,
    ):
        self._ensure_initialized()
        policy = policy_net or self.policy_net
        results = self.evaluator.evaluate(
            policy,
            seeds=seeds,
            artifact_directory=artifact_directory,
            environment_override=environment_override,
        )
        return results, self.evaluator.summarize(results)


__all__ = ["MiniGridExplorationTrainer"]
