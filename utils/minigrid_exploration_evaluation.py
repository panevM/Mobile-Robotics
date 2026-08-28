"""Reusable greedy/random evaluation over fixed or zero-shot environments."""

from __future__ import annotations

from pathlib import Path

from utils.minigrid_exploration_diagnostics import save_diagnostic_episode
from utils.minigrid_exploration_episode import ExplorationEpisodeRunner
from utils.minigrid_exploration_metrics import summarize_evaluations


class MiniGridExplorationEvaluator:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.runner = ExplorationEpisodeRunner(config, device)

    def evaluate(
        self,
        policy_net,
        *,
        seeds=None,
        random_controller=False,
        artifact_directory=None,
        checkpoint_episode=None,
        environment_override=None,
    ):
        seeds = tuple(self.config.validation.seeds if seeds is None else seeds)
        results = []
        if policy_net is None and not random_controller:
            raise ValueError("policy_net is required for greedy evaluation")
        was_training = bool(policy_net.training) if policy_net is not None else False
        if policy_net is not None:
            policy_net.eval()
        try:
            for seed in seeds:
                result = self.runner.run(
                    seed=seed,
                    policy_net=policy_net,
                    epsilon=0.0,
                    random_controller=random_controller,
                    capture_diagnostics=artifact_directory is not None,
                    validation_checkpoint_episode=checkpoint_episode,
                    environment_override=environment_override,
                )
                results.append(result)
                if artifact_directory is not None:
                    save_diagnostic_episode(result, Path(artifact_directory))
        finally:
            if policy_net is not None:
                policy_net.train(was_training)
        return results

    def summarize(self, results):
        return summarize_evaluations(results, self.config.validation.coverage_thresholds)


__all__ = ["MiniGridExplorationEvaluator"]
