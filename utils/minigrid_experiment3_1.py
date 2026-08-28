"""Experiment 3.1 fine-tuning for the dual-scale MiniGrid DQN."""

from __future__ import annotations

import csv
import math
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image
import torch
import torch.optim as optim

from utils.dqn_utils import hard_update_target, set_seed
from utils.minigrid_dual_scale_dqn import (
    DualScaleMapDQN,
    DualScaleReplayBuffer,
    EXPLORATION_ACTIONS,
    dual_scale_dqn_train_step,
    make_dual_scale_state,
    mask_q_values,
    save_dual_scale_checkpoint,
    select_masked_greedy_action,
)
from utils.minigrid_experiment3 import (
    ACTION_NAMES,
    DEFAULT_COVERAGE_THRESHOLDS,
    Experiment3RewardConfig,
    _encoder_from_config,
    _save_curve_png,
    _save_json,
    _save_multi_curve_png,
    _steps_to_thresholds,
    compute_exploration_reward,
    forward_cell_is_known_blocked,
    known_traversable_cells,
    reachable_frontier_distance,
    select_epsilon_greedy_action_with_source,
)
from utils.minigrid_mapper import PersistentMiniGridMapper
from utils.procedural_rooms_env import ENVIRONMENT_ID


@dataclass(frozen=True)
class Experiment31DeadlockConfig:
    """Experiment 3 deadlocks plus a map-size-aware no-progress timeout."""

    oscillation_deadlock_threshold: int = 8
    stationary_deadlock_threshold: int = 8
    deadlock_penalty: float = -1.0
    no_progress_timeout_multiplier: float = 2.0

    def __post_init__(self):
        if self.oscillation_deadlock_threshold < 2:
            raise ValueError("oscillation_deadlock_threshold must be at least 2")
        if self.stationary_deadlock_threshold < 1:
            raise ValueError("stationary_deadlock_threshold must be positive")
        if self.no_progress_timeout_multiplier <= 0:
            raise ValueError("no_progress_timeout_multiplier must be positive")

    def no_progress_timeout(self, width, height):
        return max(
            1,
            int(math.ceil(self.no_progress_timeout_multiplier * (width + height))),
        )


class ExplorationProgressDetector:
    """Preserve Experiment 3 deadlocks and track discovery-or-BFS progress."""

    def __init__(self, config, no_progress_timeout=None):
        self.config = config
        self.no_progress_timeout = no_progress_timeout
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
            self.oscillation_counter = (
                self.oscillation_counter + 1
                if self.oscillation_counter > 0
                else 2
            )
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

        if self.oscillation_counter >= self.config.oscillation_deadlock_threshold:
            reason = "oscillation_deadlock"
        elif self.stationary_counter >= self.config.stationary_deadlock_threshold:
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


def transition_is_done(terminated, truncated, deadlock_type):
    return bool(terminated or truncated or deadlock_type is not None)


def episode_end_reason(terminated, truncated, deadlock_type):
    if deadlock_type is not None:
        return deadlock_type
    if terminated:
        return "environment_terminated"
    if truncated:
        return "time_limit"
    return "in_progress"


def episode_linear_epsilon(
    episode_index,
    num_episodes,
    epsilon_start=0.10,
    epsilon_end=0.02,
):
    """Linearly interpolate epsilon by zero-based episode index."""
    if num_episodes < 1:
        raise ValueError("num_episodes must be positive")
    if epsilon_start <= 0 or epsilon_end <= 0:
        raise ValueError("epsilon must remain positive")
    if epsilon_end > epsilon_start:
        raise ValueError("epsilon_end cannot exceed epsilon_start")
    if not 0 <= episode_index < num_episodes:
        raise ValueError("episode_index must be within the fine-tuning run")
    if num_episodes == 1 or episode_index == num_episodes - 1:
        return float(epsilon_end)
    if episode_index == 0:
        return float(epsilon_start)
    progress = episode_index / (num_episodes - 1)
    return float(epsilon_start + progress * (epsilon_end - epsilon_start))


def training_fraction(episode_index, num_episodes):
    if num_episodes < 1:
        raise ValueError("num_episodes must be positive")
    return float(1.0 if num_episodes == 1 else episode_index / (num_episodes - 1))


def _safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def calculate_action_source_metrics(counts):
    """Retain Experiment 3 diagnostics and add both new-cell fractions."""
    random_actions = int(counts.get("random_action_count", 0))
    greedy_actions = int(counts.get("greedy_action_count", 0))
    random_discoveries = int(counts.get("random_discovery_actions", 0))
    greedy_discoveries = int(counts.get("greedy_discovery_actions", 0))
    random_new_cells = int(counts.get("random_new_cells", 0))
    greedy_new_cells = int(counts.get("greedy_new_cells", 0))
    total_discoveries = random_discoveries + greedy_discoveries
    total_new_cells = random_new_cells + greedy_new_cells
    random_new_cell_fraction = _safe_ratio(random_new_cells, total_new_cells)
    greedy_new_cell_fraction = _safe_ratio(greedy_new_cells, total_new_cells)
    return {
        "random_action_count": random_actions,
        "greedy_action_count": greedy_actions,
        "random_discovery_actions": random_discoveries,
        "greedy_discovery_actions": greedy_discoveries,
        "random_new_cells": random_new_cells,
        "greedy_new_cells": greedy_new_cells,
        "random_discovery_probability": _safe_ratio(random_discoveries, random_actions),
        "greedy_discovery_probability": _safe_ratio(greedy_discoveries, greedy_actions),
        "fraction_of_discovery_events_from_random": _safe_ratio(
            random_discoveries, total_discoveries
        ),
        "fraction_of_new_cells_from_random": random_new_cell_fraction,
        "fraction_new_cells_from_random": random_new_cell_fraction,
        "fraction_new_cells_from_greedy": greedy_new_cell_fraction,
        "random_position_change_probability": _safe_ratio(
            counts.get("random_position_changes", 0), random_actions
        ),
        "greedy_position_change_probability": _safe_ratio(
            counts.get("greedy_position_changes", 0), greedy_actions
        ),
        "random_forward_actions": int(counts.get("random_forward_actions", 0)),
        "greedy_forward_actions": int(counts.get("greedy_forward_actions", 0)),
        "random_forward_movements": int(counts.get("random_forward_movements", 0)),
        "greedy_forward_movements": int(counts.get("greedy_forward_movements", 0)),
        "random_stationary_actions": int(counts.get("random_stationary_actions", 0)),
        "greedy_stationary_actions": int(counts.get("greedy_stationary_actions", 0)),
    }


def summarize_evaluations(results, thresholds=DEFAULT_COVERAGE_THRESHOLDS):
    if not results:
        raise ValueError("results cannot be empty")
    end_reason_counts = Counter(result["episode_end_reason"] for result in results)
    episode_count = len(results)
    random_actions = sum(r["random_action_count"] for r in results)
    greedy_actions = sum(r["greedy_action_count"] for r in results)
    random_new_cells = sum(r["random_new_cells"] for r in results)
    greedy_new_cells = sum(r["greedy_new_cells"] for r in results)
    random_discoveries = sum(r["random_discovery_actions"] for r in results)
    greedy_discoveries = sum(r["greedy_discovery_actions"] for r in results)
    total_new_cells = random_new_cells + greedy_new_cells
    summary = {
        "mean_final_coverage": float(np.mean([r["final_coverage"] for r in results])),
        "median_final_coverage": float(np.median([r["final_coverage"] for r in results])),
        "mean_traversable_coverage": float(np.mean([r["traversable_coverage"] for r in results])),
        "new_cells_per_step": float(np.mean([r["new_cells_per_step"] for r in results])),
        "zero_information_ratio": float(np.mean([r["zero_information_ratio"] for r in results])),
        "stationary_action_ratio": float(np.mean([r["stationary_action_ratio"] for r in results])),
        "oscillation_ratio": float(np.mean([r["oscillation_ratio"] for r in results])),
        "blocked_forward_ratio": float(np.mean([r["blocked_forward_ratio"] for r in results])),
        "position_change_ratio": float(np.mean([r["position_change_ratio"] for r in results])),
        "mean_maximum_consecutive_stationary_steps": float(
            np.mean([r["maximum_consecutive_stationary_steps"] for r in results])
        ),
        "random_action_count": int(random_actions),
        "greedy_action_count": int(greedy_actions),
        "random_discovery_probability": _safe_ratio(random_discoveries, random_actions),
        "greedy_discovery_probability": _safe_ratio(greedy_discoveries, greedy_actions),
        "fraction_new_cells_from_random": _safe_ratio(random_new_cells, total_new_cells),
        "fraction_new_cells_from_greedy": _safe_ratio(greedy_new_cells, total_new_cells),
        "oscillation_deadlock_count": int(end_reason_counts["oscillation_deadlock"]),
        "stationary_deadlock_count": int(end_reason_counts["stationary_deadlock"]),
        "no_progress_deadlock_count": int(end_reason_counts["no_progress_deadlock"]),
        "time_limit_count": int(end_reason_counts["time_limit"]),
        "environment_terminated_count": int(end_reason_counts["environment_terminated"]),
        "fraction_oscillation_deadlock": float(
            end_reason_counts["oscillation_deadlock"] / episode_count
        ),
        "fraction_stationary_deadlock": float(
            end_reason_counts["stationary_deadlock"] / episode_count
        ),
        "fraction_no_progress_deadlock": float(
            end_reason_counts["no_progress_deadlock"] / episode_count
        ),
        "fraction_time_limit": float(end_reason_counts["time_limit"] / episode_count),
    }
    for threshold in thresholds:
        label = int(round(100 * threshold))
        values = np.asarray(
            [r["steps_to_coverage"][threshold] for r in results], dtype=float
        )
        reached = np.isfinite(values)
        summary[f"fraction_reaching_{label}"] = float(np.mean(reached))
        summary[f"mean_steps_to_{label}_if_reached"] = (
            float(np.mean(values[reached])) if np.any(reached) else np.nan
        )
    return summary


def _mean_padded_curve(results):
    histories = [np.asarray(result["coverage_history"], dtype=float) for result in results]
    max_length = max(len(history) for history in histories)
    padded = np.stack(
        [np.pad(history, (0, max_length - len(history)), mode="edge") for history in histories]
    )
    return np.mean(padded, axis=0)


def save_comparison_artifacts(output_dir, named_results, title):
    """Save matched-seed summaries and robust mean coverage curves."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {name: summarize_evaluations(results) for name, results in named_results.items()}
    curves = {name: _mean_padded_curve(results) for name, results in named_results.items()}
    _save_json(output_dir / "summary.json", summaries)
    with (output_dir / "mean_coverage_curves.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        names = list(curves)
        writer.writerow(["step", *names])
        for step in range(max(len(curve) for curve in curves.values())):
            writer.writerow(
                [step] + [curve[step] if step < len(curve) else "" for curve in curves.values()]
            )
    _save_multi_curve_png(output_dir / "mean_coverage_curves.png", curves, title)
    return summaries


def _save_diagnostic_artifacts(output_dir, result, frames, action_log, ascii_snapshots):
    output_dir.mkdir(parents=True, exist_ok=True)
    if frames:
        if result.get("episode_end_reason") in {
            "oscillation_deadlock",
            "stationary_deadlock",
            "no_progress_deadlock",
        }:
            frames.extend([frames[-1].copy(), frames[-1].copy()])
        frames[0].save(
            output_dir / f"seed_{result['seed']}.gif",
            save_all=True,
            append_images=frames[1:],
            duration=140,
            loop=0,
            optimize=False,
        )
    _save_json(output_dir / f"seed_{result['seed']}_actions.json", action_log)
    summary = {
        key: value
        for key, value in result.items()
        if key not in {"coverage_history", "traversable_coverage_history", "frontier_distance_history"}
    }
    _save_json(output_dir / f"seed_{result['seed']}_summary.json", summary)

    ascii_text = []
    for snapshot in ascii_snapshots:
        ascii_text.extend(
            [
                f"Step {snapshot['step']}",
                f"Coverage: {snapshot['coverage']:.1%}",
                f"Traversable coverage: {snapshot['traversable_coverage']:.1%}",
                f"Frontier distance: {snapshot['frontier_distance']}",
                snapshot["map"],
                "",
            ]
        )
    (output_dir / f"seed_{result['seed']}_mapper.txt").write_text(
        "\n".join(ascii_text), encoding="utf-8"
    )

    with (output_dir / f"seed_{result['seed']}_curves.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "map_coverage", "traversable_coverage", "frontier_distance"])
        writer.writerows(
            zip(
                range(len(result["coverage_history"])),
                result["coverage_history"],
                result["traversable_coverage_history"],
                result["frontier_distance_history"],
            )
        )
    _save_curve_png(
        output_dir / f"seed_{result['seed']}_coverage.png",
        result["coverage_history"],
        f"Greedy validation coverage, seed {result['seed']}",
        "Coverage",
        (28, 126, 92),
        percent=True,
    )
    _save_curve_png(
        output_dir / f"seed_{result['seed']}_traversable_coverage.png",
        result["traversable_coverage_history"],
        f"Greedy validation traversable coverage, seed {result['seed']}",
        "Traversable coverage",
        (50, 91, 168),
        percent=True,
    )
    _save_curve_png(
        output_dir / f"seed_{result['seed']}_frontier_distance.png",
        result["frontier_distance_history"],
        f"Reachable-frontier BFS distance, seed {result['seed']}",
        "Known-path distance",
        (202, 91, 42),
    )


def run_evaluation_episode(
    seed,
    environment_config,
    tensor_config,
    device,
    reward_config,
    policy_net=None,
    random_controller=False,
    artifact_dir=None,
    ascii_interval=5,
    validation_checkpoint_episode=None,
    thresholds=DEFAULT_COVERAGE_THRESHOLDS,
    deadlock_config=Experiment31DeadlockConfig(),
):
    """Run one greedy or random episode using Experiment 3.1 termination."""
    if not random_controller and policy_net is None:
        raise ValueError("policy_net is required for greedy evaluation")
    env_config = dict(environment_config)
    if artifact_dir is not None:
        env_config["render_mode"] = "rgb_array"
    evaluation_env = gym.make(ENVIRONMENT_ID, **env_config)
    observation, _ = evaluation_env.reset(seed=int(seed))
    mapper = PersistentMiniGridMapper()
    mapper.reset(observation)
    encoder = _encoder_from_config(tensor_config)
    state = make_dual_scale_state(encoder.encode(mapper))
    rng = np.random.default_rng(900_000 + int(seed))

    total_cells = evaluation_env.unwrapped.width * evaluation_env.unwrapped.height
    total_traversable = evaluation_env.unwrapped.total_traversable_cells
    coverage_history = [len(mapper.cells) / total_cells]
    traversable_history = [known_traversable_cells(mapper) / total_traversable]
    frontier_history = [reachable_frontier_distance(mapper)]
    new_cell_counts = []
    reward_totals = Counter()
    event_counts = Counter()
    action_log = []
    frames = []
    ascii_snapshots = []
    previous_action = None
    no_progress_timeout = deadlock_config.no_progress_timeout(
        evaluation_env.unwrapped.width, evaluation_env.unwrapped.height
    )
    deadlock_detector = ExplorationProgressDetector(
        deadlock_config, no_progress_timeout=no_progress_timeout
    )
    deadlock_type = None
    consecutive_stationary = 0
    maximum_consecutive_stationary = 0
    discovery_actions = 0

    if artifact_dir is not None:
        frames.append(Image.fromarray(evaluation_env.render()).convert("RGB"))
        ascii_snapshots.append(
            {
                "step": 0,
                "coverage": coverage_history[-1],
                "traversable_coverage": traversable_history[-1],
                "frontier_distance": frontier_history[-1],
                "map": mapper.render_ascii(),
            }
        )

    terminated = truncated = False
    step = 0
    while not (terminated or truncated or deadlock_type is not None):
        action = (
            int(rng.choice(EXPLORATION_ACTIONS))
            if random_controller
            else select_masked_greedy_action(policy_net, state, device)
        )
        position_before = tuple(mapper.position)
        frontier_before = reachable_frontier_distance(mapper)
        forward_blocked = forward_cell_is_known_blocked(mapper)
        mapper.predict_action(action)
        observation, _, terminated, truncated, _ = evaluation_env.step(action)
        update = mapper.observe(observation)
        frontier_after = reachable_frontier_distance(mapper)
        position_after = tuple(mapper.position)
        position_changed = position_after != position_before
        reward = compute_exploration_reward(
            mapping_update=update,
            frontier_distance_before=frontier_before,
            frontier_distance_after=frontier_after,
            action=action,
            position_before=position_before,
            position_after=position_after,
            previous_action=previous_action,
            forward_was_known_blocked=forward_blocked,
            config=reward_config,
        )
        state = make_dual_scale_state(encoder.encode(mapper))
        step += 1
        discovery_actions += int(update.new_cells > 0)
        coverage = len(mapper.cells) / total_cells
        traversable_coverage = known_traversable_cells(mapper) / total_traversable
        coverage_history.append(coverage)
        traversable_history.append(traversable_coverage)
        frontier_history.append(frontier_after)
        new_cell_counts.append(update.new_cells)
        deadlock_status = deadlock_detector.update(
            action=action,
            previous_action=previous_action,
            position_changed=position_changed,
            new_cells=update.new_cells,
            frontier_distance_before=frontier_before,
            frontier_distance_after=frontier_after,
        )
        deadlock_type = deadlock_status["deadlock_type"]
        if deadlock_type is not None:
            reward["deadlock"] = float(deadlock_config.deadlock_penalty)
            reward["total"] += reward["deadlock"]
        consecutive_stationary = 0 if position_changed else consecutive_stationary + 1
        maximum_consecutive_stationary = max(maximum_consecutive_stationary, consecutive_stationary)
        for key in ("total", "novelty", "frontier", "stationary", "oscillation", "blocked", "deadlock"):
            reward_totals[key] += reward[key]
        for key in (
            "stationary_action",
            "oscillating",
            "blocked_forward",
            "frontier_reduced",
            "frontier_increased",
            "no_reachable_frontier",
        ):
            event_counts[key] += int(reward[key])

        if artifact_dir is not None:
            frames.append(Image.fromarray(evaluation_env.render()).convert("RGB"))
            action_log.append(
                {
                    "step": step,
                    "action": ACTION_NAMES[action],
                    "action_id": action,
                    "action_source": "random_baseline" if random_controller else "greedy/evaluation",
                    "agent_position": list(map(int, mapper.position)),
                    "agent_direction": int(mapper.direction),
                    "position_changed": position_changed,
                    "frontier_distance": frontier_after,
                    "frontier_progress_this_step": deadlock_status["frontier_progress_this_step"],
                    "new_cells": int(update.new_cells),
                    "exploration_progress_this_step": deadlock_status["exploration_progress_this_step"],
                    "steps_since_exploration_progress": deadlock_status["steps_since_exploration_progress"],
                    "no_progress_timeout": deadlock_status["no_progress_timeout"],
                    "coverage": coverage,
                    "novelty_reward": reward["novelty"],
                    "frontier_reward": reward["frontier"],
                    "stationary_penalty": reward["stationary"],
                    "oscillation_penalty": reward["oscillation"],
                    "blocked_penalty": reward["blocked"],
                    "deadlock_penalty": reward["deadlock"],
                    "total_reward": reward["total"],
                    "oscillation_counter": deadlock_status["oscillation_counter"],
                    "stationary_counter": deadlock_status["stationary_counter"],
                    "deadlock_triggered": deadlock_status["deadlock_triggered"],
                    "deadlock_type": deadlock_type,
                    "event": f"EPISODE TERMINATED: {deadlock_type}" if deadlock_type else None,
                }
            )
            if update.new_cells > 0 or step % ascii_interval == 0 or terminated or truncated or deadlock_type:
                ascii_snapshots.append(
                    {
                        "step": step,
                        "coverage": coverage,
                        "traversable_coverage": traversable_coverage,
                        "frontier_distance": frontier_after,
                        "map": mapper.render_ascii(),
                    }
                )
        previous_action = action

    evaluation_env.close()
    end_reason = episode_end_reason(terminated, truncated, deadlock_type)
    new_cells = np.asarray(new_cell_counts, dtype=float)
    steps_to_coverage = _steps_to_thresholds(coverage_history, thresholds)
    total_new_cells = int(np.sum(new_cells))
    random_actions = step if random_controller else 0
    greedy_actions = 0 if random_controller else step
    random_new_cells = total_new_cells if random_controller else 0
    greedy_new_cells = 0 if random_controller else total_new_cells
    random_discoveries = discovery_actions if random_controller else 0
    greedy_discoveries = 0 if random_controller else discovery_actions
    result = {
        "seed": int(seed),
        "validation_checkpoint_episode": validation_checkpoint_episode,
        "final_coverage": float(coverage_history[-1]),
        "traversable_coverage": float(traversable_history[-1]),
        "coverage_history": np.asarray(coverage_history, dtype=float),
        "traversable_coverage_history": np.asarray(traversable_history, dtype=float),
        "frontier_distance_history": frontier_history,
        "steps_to_coverage": steps_to_coverage,
        "reached_50": bool(np.isfinite(steps_to_coverage.get(0.50, np.nan))),
        "reached_75": bool(np.isfinite(steps_to_coverage.get(0.75, np.nan))),
        "reached_90": bool(np.isfinite(steps_to_coverage.get(0.90, np.nan))),
        "new_cells_per_step": float(np.mean(new_cells)) if new_cells.size else 0.0,
        "zero_information_ratio": float(np.mean(new_cells == 0)) if new_cells.size else 1.0,
        "total_reward": float(reward_totals["total"]),
        "novelty_reward": float(reward_totals["novelty"]),
        "frontier_reward": float(reward_totals["frontier"]),
        "stationary_penalty": float(reward_totals["stationary"]),
        "oscillation_penalty": float(reward_totals["oscillation"]),
        "blocked_penalty": float(reward_totals["blocked"]),
        "deadlock_penalty": float(reward_totals["deadlock"]),
        "stationary_action_count": int(event_counts["stationary_action"]),
        "stationary_action_ratio": _safe_ratio(event_counts["stationary_action"], step),
        "oscillation_count": int(event_counts["oscillating"]),
        "oscillation_ratio": _safe_ratio(event_counts["oscillating"], step),
        "blocked_action_count": int(event_counts["blocked_forward"]),
        "blocked_forward_ratio": _safe_ratio(event_counts["blocked_forward"], step),
        "position_change_ratio": _safe_ratio(step - event_counts["stationary_action"], step),
        "maximum_consecutive_stationary_steps": maximum_consecutive_stationary,
        "random_action_count": random_actions,
        "greedy_action_count": greedy_actions,
        "random_discovery_actions": random_discoveries,
        "greedy_discovery_actions": greedy_discoveries,
        "random_new_cells": random_new_cells,
        "greedy_new_cells": greedy_new_cells,
        "random_discovery_probability": _safe_ratio(random_discoveries, random_actions),
        "greedy_discovery_probability": _safe_ratio(greedy_discoveries, greedy_actions),
        "fraction_new_cells_from_random": _safe_ratio(random_new_cells, total_new_cells),
        "fraction_new_cells_from_greedy": _safe_ratio(greedy_new_cells, total_new_cells),
        "frontier_distance_reduced_steps": int(event_counts["frontier_reduced"]),
        "frontier_distance_increased_steps": int(event_counts["frontier_increased"]),
        "no_reachable_frontier_steps": int(event_counts["no_reachable_frontier"]),
        "final_global_scale": int(2 ** state.global_scale_log2),
        "episode_length": step,
        "episode_end_reason": end_reason,
        "final_oscillation_counter": deadlock_detector.oscillation_counter,
        "final_stationary_counter": deadlock_detector.stationary_counter,
        "final_steps_since_exploration_progress": deadlock_detector.steps_since_exploration_progress,
        "no_progress_timeout": no_progress_timeout,
    }
    if artifact_dir is not None:
        _save_diagnostic_artifacts(Path(artifact_dir), result, frames, action_log, ascii_snapshots)
    return result


def evaluate_exploration_controller(
    seeds,
    environment_config,
    tensor_config,
    device,
    reward_config,
    policy_net=None,
    random_controller=False,
    artifact_dir=None,
    validation_checkpoint_episode=None,
    thresholds=DEFAULT_COVERAGE_THRESHOLDS,
    deadlock_config=Experiment31DeadlockConfig(),
):
    """Evaluate fixed seeds greedily unless an explicit random baseline is requested."""
    if policy_net is not None:
        policy_net.eval()
    return [
        run_evaluation_episode(
            seed,
            environment_config,
            tensor_config,
            device,
            reward_config,
            deadlock_config=deadlock_config,
            policy_net=policy_net,
            random_controller=random_controller,
            artifact_dir=Path(artifact_dir) if artifact_dir is not None else None,
            validation_checkpoint_episode=validation_checkpoint_episode,
            thresholds=thresholds,
        )
        for seed in seeds
    ]


def initialize_experiment_3_1(pretrained_checkpoint, device, learning_rate, replay_capacity):
    """Load only policy weights, then create fresh fine-tuning state."""
    pretrained_checkpoint = Path(pretrained_checkpoint)
    checkpoint = torch.load(pretrained_checkpoint, map_location=device, weights_only=False)
    policy_net = DualScaleMapDQN(**checkpoint["architecture"]).to(device)
    policy_net.load_state_dict(checkpoint["model_state_dict"])
    target_net = DualScaleMapDQN(**checkpoint["architecture"]).to(device)
    hard_update_target(policy_net, target_net)
    target_net.eval()
    policy_net.train()
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = DualScaleReplayBuffer(replay_capacity)
    if len(replay_buffer) != 0:
        raise RuntimeError("Experiment 3.1 replay buffer must start empty")
    return policy_net, target_net, optimizer, replay_buffer, checkpoint


def _validate_controlled_configuration(
    source_checkpoint,
    reward_config,
    deadlock_config,
    tensor_config,
    environment_config,
):
    source_reward = source_checkpoint.get("reward_config", {})
    if source_reward != asdict(reward_config):
        raise ValueError("Experiment 3.1 reward configuration must exactly match its source checkpoint")
    source_deadlock = source_checkpoint.get("deadlock_config", {})
    unchanged_deadlocks = {
        "oscillation_deadlock_threshold": deadlock_config.oscillation_deadlock_threshold,
        "stationary_deadlock_threshold": deadlock_config.stationary_deadlock_threshold,
        "deadlock_penalty": deadlock_config.deadlock_penalty,
    }
    for key, expected in unchanged_deadlocks.items():
        if source_deadlock.get(key) != expected:
            raise ValueError(f"Experiment 3.1 must preserve source {key}")
    if source_checkpoint.get("tensor_config") != dict(tensor_config):
        raise ValueError("Experiment 3.1 tensor configuration must match its source checkpoint")
    if source_checkpoint.get("training_environment_config") != dict(environment_config):
        raise ValueError("Experiment 3.1 environment configuration must match its source checkpoint")


def _checkpoint_metadata(
    *,
    episode,
    total_steps,
    epsilon,
    fraction,
    source_checkpoint_path,
    source_checkpoint,
    reward_config,
    deadlock_config,
    tensor_config,
    environment_config,
    training_config,
    training_metrics,
    validation_history,
    replay_buffer_initial_size,
):
    return {
        "experiment_stage": "3.1",
        "initialization": "experiment_3_finetune",
        "pretrained_checkpoint": str(Path(source_checkpoint_path).resolve()),
        "source_experiment_stage": source_checkpoint.get("experiment_stage"),
        "source_training_episode": source_checkpoint.get("training_episode"),
        "source_total_steps": source_checkpoint.get("total_steps"),
        "replay_buffer_initial_size": int(replay_buffer_initial_size),
        "optimizer_initialization": "fresh_adam",
        "target_initialization": "hard_update_from_inherited_policy",
        "training_episode": int(episode),
        "total_steps": int(total_steps),
        "fine_tuning_total_steps": int(total_steps),
        "epsilon": float(epsilon),
        "training_fraction": float(fraction),
        "epsilon_schedule": "linear_by_finetuning_episode",
        "reward_config": asdict(reward_config),
        "deadlock_config": asdict(deadlock_config),
        "tensor_config": dict(tensor_config),
        "training_environment_config": dict(environment_config),
        "hyperparameters": dict(training_config),
        "training_metrics": training_metrics,
        "validation_history": validation_history,
    }


def run_experiment_3_1(
    *,
    pretrained_checkpoint,
    output_dir,
    environment_config,
    tensor_config,
    validation_seeds,
    diagnostic_validation_seeds,
    device,
    reward_config,
    deadlock_config=Experiment31DeadlockConfig(),
    num_finetune_episodes=1_000,
    training_seed_start=0,
    training_seed_count=10_000,
    training_random_seed=45,
    learning_rate=1e-4,
    gamma=0.99,
    batch_size=64,
    replay_capacity=100_000,
    minimum_replay_size=5_000,
    target_update_frequency=1_000,
    gradient_clip=10.0,
    epsilon_start=0.10,
    epsilon_end=0.02,
    validation_frequency=100,
):
    """Fine-tune Experiment 3 weights with fresh optimizer and replay state."""
    if num_finetune_episodes < 1:
        raise ValueError("num_finetune_episodes must be positive")
    if training_seed_count < 1:
        raise ValueError("training_seed_count must be positive")
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    artifact_root = output_dir / "validation_artifacts"
    metrics_dir = output_dir / "metrics"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    set_seed(training_random_seed)
    rng = np.random.default_rng(training_random_seed)
    policy_net, target_net, optimizer, replay_buffer, source_checkpoint = initialize_experiment_3_1(
        pretrained_checkpoint, device, learning_rate, replay_capacity
    )
    _validate_controlled_configuration(
        source_checkpoint, reward_config, deadlock_config, tensor_config, environment_config
    )
    replay_buffer_initial_size = len(replay_buffer)
    training_metrics = []
    validation_history = []
    total_steps = 0
    training_env = gym.make(ENVIRONMENT_ID, **environment_config)
    total_cells = training_env.unwrapped.width * training_env.unwrapped.height
    thresholds = DEFAULT_COVERAGE_THRESHOLDS
    training_config = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "replay_capacity": replay_capacity,
        "minimum_replay_size": minimum_replay_size,
        "target_update_frequency": target_update_frequency,
        "gradient_clip": gradient_clip,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "epsilon_schedule": "linear_by_finetuning_episode",
        "num_finetune_episodes": num_finetune_episodes,
        "validation_frequency": validation_frequency,
        "deadlock_config": asdict(deadlock_config),
        "allowed_actions": EXPLORATION_ACTIONS,
        "training_seed_range": (training_seed_start, training_seed_start + training_seed_count - 1),
    }

    try:
        for episode_index in range(num_finetune_episodes):
            episode_number = episode_index + 1
            fraction = training_fraction(episode_index, num_finetune_episodes)
            epsilon = episode_linear_epsilon(
                episode_index, num_finetune_episodes, epsilon_start, epsilon_end
            )
            episode_seed = training_seed_start + (episode_index % training_seed_count)
            observation, _ = training_env.reset(seed=episode_seed)
            mapper = PersistentMiniGridMapper()
            mapper.reset(observation)
            encoder = _encoder_from_config(tensor_config)
            state = make_dual_scale_state(encoder.encode(mapper))
            total_traversable = training_env.unwrapped.total_traversable_cells
            rewards = Counter()
            events = Counter()
            source_counts = Counter()
            losses = []
            new_cell_counts = []
            scale_counts = Counter([int(2 ** state.global_scale_log2)])
            coverage_history = [len(mapper.cells) / total_cells]
            episode_steps = 0
            previous_action = None
            no_progress_timeout = deadlock_config.no_progress_timeout(
                training_env.unwrapped.width, training_env.unwrapped.height
            )
            deadlock_detector = ExplorationProgressDetector(
                deadlock_config, no_progress_timeout=no_progress_timeout
            )
            deadlock_type = None
            consecutive_stationary = 0
            maximum_consecutive_stationary = 0
            terminated = truncated = False

            while not (terminated or truncated or deadlock_type is not None):
                action, action_source = select_epsilon_greedy_action_with_source(
                    policy_net, state, epsilon, device, random_generator=rng
                )
                position_before = tuple(mapper.position)
                frontier_before = reachable_frontier_distance(mapper)
                forward_blocked = forward_cell_is_known_blocked(mapper)
                mapper.predict_action(action)
                observation, _, terminated, truncated, _ = training_env.step(action)
                update = mapper.observe(observation)
                frontier_after = reachable_frontier_distance(mapper)
                position_after = tuple(mapper.position)
                position_changed = position_after != position_before
                reward = compute_exploration_reward(
                    mapping_update=update,
                    frontier_distance_before=frontier_before,
                    frontier_distance_after=frontier_after,
                    action=action,
                    position_before=position_before,
                    position_after=position_after,
                    previous_action=previous_action,
                    forward_was_known_blocked=forward_blocked,
                    config=reward_config,
                )
                deadlock_status = deadlock_detector.update(
                    action=action,
                    previous_action=previous_action,
                    position_changed=position_changed,
                    new_cells=update.new_cells,
                    frontier_distance_before=frontier_before,
                    frontier_distance_after=frontier_after,
                )
                deadlock_type = deadlock_status["deadlock_type"]
                if deadlock_type is not None:
                    reward["deadlock"] = float(deadlock_config.deadlock_penalty)
                    reward["total"] += reward["deadlock"]
                next_state = make_dual_scale_state(encoder.encode(mapper))
                done = transition_is_done(terminated, truncated, deadlock_type)
                replay_buffer.push(state, action, reward["total"], next_state, done)
                loss = dual_scale_dqn_train_step(
                    policy_net,
                    target_net,
                    optimizer,
                    replay_buffer,
                    batch_size,
                    gamma,
                    device,
                    min_replay_size=minimum_replay_size,
                    gradient_clip=gradient_clip,
                )
                if loss is not None:
                    losses.append(loss)

                total_steps += 1
                episode_steps += 1
                for key in ("total", "novelty", "frontier", "stationary", "oscillation", "blocked", "deadlock"):
                    rewards[key] += reward[key]
                for key in (
                    "stationary_action",
                    "oscillating",
                    "blocked_forward",
                    "frontier_reduced",
                    "frontier_increased",
                    "no_reachable_frontier",
                ):
                    events[key] += int(reward[key])
                source_counts[f"{action_source}_action_count"] += 1
                source_counts[f"{action_source}_discovery_actions"] += int(update.new_cells > 0)
                source_counts[f"{action_source}_new_cells"] += int(update.new_cells)
                source_counts[f"{action_source}_position_changes"] += int(position_changed)
                source_counts[f"{action_source}_forward_actions"] += int(action == 2)
                source_counts[f"{action_source}_forward_movements"] += int(action == 2 and position_changed)
                source_counts[f"{action_source}_stationary_actions"] += int(not position_changed)
                consecutive_stationary = 0 if position_changed else consecutive_stationary + 1
                maximum_consecutive_stationary = max(maximum_consecutive_stationary, consecutive_stationary)
                new_cell_counts.append(update.new_cells)
                scale_counts[int(2 ** next_state.global_scale_log2)] += 1
                coverage_history.append(len(mapper.cells) / total_cells)
                state = next_state
                previous_action = action
                if total_steps % target_update_frequency == 0:
                    hard_update_target(policy_net, target_net)

            new_cells = np.asarray(new_cell_counts, dtype=float)
            local_tensor, global_tensor, scale_tensor = DualScaleReplayBuffer._batch_states([state], device)
            with torch.no_grad():
                predicted_q = mask_q_values(policy_net(local_tensor, global_tensor, scale_tensor))[:, list(EXPLORATION_ACTIONS)]
            source_metrics = calculate_action_source_metrics(source_counts)
            end_reason = episode_end_reason(terminated, truncated, deadlock_type)
            training_metrics.append(
                {
                    "episode": episode_number,
                    "training_fraction": fraction,
                    "epsilon": epsilon,
                    "epsilon_schedule": "linear_by_finetuning_episode",
                    "total_reward": float(rewards["total"]),
                    "novelty_reward": float(rewards["novelty"]),
                    "frontier_reward": float(rewards["frontier"]),
                    "stationary_penalty": float(rewards["stationary"]),
                    "oscillation_penalty": float(rewards["oscillation"]),
                    "blocked_penalty": float(rewards["blocked"]),
                    "deadlock_penalty": float(rewards["deadlock"]),
                    "stationary_action_count": int(events["stationary_action"]),
                    "stationary_action_ratio": _safe_ratio(events["stationary_action"], episode_steps),
                    "oscillation_count": int(events["oscillating"]),
                    "oscillation_ratio": _safe_ratio(events["oscillating"], episode_steps),
                    "blocked_forward_count": int(events["blocked_forward"]),
                    "blocked_forward_ratio": _safe_ratio(events["blocked_forward"], episode_steps),
                    "maximum_consecutive_stationary_steps": maximum_consecutive_stationary,
                    "position_change_ratio": _safe_ratio(episode_steps - events["stationary_action"], episode_steps),
                    "frontier_distance_reduced_steps": int(events["frontier_reduced"]),
                    "frontier_distance_increased_steps": int(events["frontier_increased"]),
                    "no_reachable_frontier_steps": int(events["no_reachable_frontier"]),
                    **source_metrics,
                    "final_coverage": float(coverage_history[-1]),
                    "traversable_coverage": float(known_traversable_cells(mapper) / total_traversable),
                    "new_cells_per_step": float(np.mean(new_cells)) if new_cells.size else 0.0,
                    "zero_information_ratio": float(np.mean(new_cells == 0)) if new_cells.size else 1.0,
                    "steps_to_coverage": _steps_to_thresholds(coverage_history, thresholds),
                    "episode_length": episode_steps,
                    "episode_end_reason": end_reason,
                    "oscillation_deadlock_count": int(end_reason == "oscillation_deadlock"),
                    "stationary_deadlock_count": int(end_reason == "stationary_deadlock"),
                    "no_progress_deadlock_count": int(end_reason == "no_progress_deadlock"),
                    "deadlock_episode_fraction": float(end_reason in {"oscillation_deadlock", "stationary_deadlock", "no_progress_deadlock"}),
                    "final_steps_since_exploration_progress": deadlock_detector.steps_since_exploration_progress,
                    "no_progress_timeout": no_progress_timeout,
                    "training_loss": float(np.mean(losses)) if losses else np.nan,
                    "mean_predicted_q": float(predicted_q.mean().item()),
                    "max_predicted_q": float(predicted_q.max().item()),
                    "global_scale_distribution": dict(scale_counts),
                }
            )

            if episode_number % 25 == 0:
                recent = training_metrics[-25:]
                finite_losses = [m["training_loss"] for m in recent if np.isfinite(m["training_loss"])]
                mean_loss = float(np.mean(finite_losses)) if finite_losses else np.nan
                print(
                    f"Episode {episode_number:4d}/{num_finetune_episodes} | "
                    f"training={fraction:.1%} | epsilon={epsilon:.3f} | "
                    f"coverage={np.mean([m['final_coverage'] for m in recent]):.1%} | "
                    f"reward={np.mean([m['total_reward'] for m in recent]):.2f} | "
                    f"loss={mean_loss:.4f} | move={np.mean([m['position_change_ratio'] for m in recent]):.1%} | "
                    f"deadlock={np.mean([m['deadlock_episode_fraction'] for m in recent]):.1%} | "
                    f"Pdisc(greedy)={np.mean([m['greedy_discovery_probability'] for m in recent]):.1%} | "
                    f"Pdisc(random)={np.mean([m['random_discovery_probability'] for m in recent]):.1%} | "
                    f"new-cells(greedy)={np.mean([m['fraction_new_cells_from_greedy'] for m in recent]):.1%} | "
                    f"new-cells(random)={np.mean([m['fraction_new_cells_from_random'] for m in recent]):.1%}"
                )

            if episode_number % validation_frequency == 0:
                validation_results = evaluate_exploration_controller(
                    validation_seeds,
                    environment_config,
                    tensor_config,
                    device,
                    reward_config,
                    deadlock_config=deadlock_config,
                    policy_net=policy_net,
                )
                validation_summary = summarize_evaluations(validation_results)
                if validation_summary["random_action_count"] != 0:
                    raise RuntimeError("Greedy validation unexpectedly used random actions")
                validation_history.append({"episode": episode_number, **validation_summary})
                metadata = _checkpoint_metadata(
                    episode=episode_number,
                    total_steps=total_steps,
                    epsilon=epsilon,
                    fraction=fraction,
                    source_checkpoint_path=pretrained_checkpoint,
                    source_checkpoint=source_checkpoint,
                    reward_config=reward_config,
                    deadlock_config=deadlock_config,
                    tensor_config=tensor_config,
                    environment_config=environment_config,
                    training_config=training_config,
                    training_metrics=training_metrics,
                    validation_history=validation_history,
                    replay_buffer_initial_size=replay_buffer_initial_size,
                )
                checkpoint_path = checkpoint_dir / f"checkpoint_ep_{episode_number:04d}.pth"
                save_dual_scale_checkpoint(checkpoint_path, policy_net, target_net=target_net, optimizer=optimizer, metadata=metadata)
                shutil.copy2(checkpoint_path, output_dir / "latest_checkpoint.pth")
                _save_json(metrics_dir / "training_metrics.json", training_metrics)
                _save_json(metrics_dir / "validation_history.json", validation_history)
                diagnostic_dir = artifact_root / f"episode_{episode_number:04d}"
                evaluate_exploration_controller(
                    diagnostic_validation_seeds,
                    environment_config,
                    tensor_config,
                    device,
                    reward_config,
                    deadlock_config=deadlock_config,
                    policy_net=policy_net,
                    artifact_dir=diagnostic_dir,
                    validation_checkpoint_episode=episode_number,
                )
                _save_json(diagnostic_dir / "validation_summary.json", validation_summary)
                policy_net.train()
                print(
                    f"Validation {episode_number}: coverage={validation_summary['mean_final_coverage']:.1%} | "
                    f"osc-deadlock={validation_summary['fraction_oscillation_deadlock']:.1%} | "
                    f"stationary-deadlock={validation_summary['fraction_stationary_deadlock']:.1%} | "
                    f"no-progress={validation_summary['fraction_no_progress_deadlock']:.1%} | "
                    f"time-limit={validation_summary['fraction_time_limit']:.1%} | "
                    f"new-cells(greedy)={validation_summary['fraction_new_cells_from_greedy']:.1%} | "
                    f"new-cells(random)={validation_summary['fraction_new_cells_from_random']:.1%} | "
                    f"50%={validation_summary['fraction_reaching_50']:.1%} | "
                    f"75%={validation_summary['fraction_reaching_75']:.1%} | "
                    f"90%={validation_summary['fraction_reaching_90']:.1%}"
                )
    finally:
        training_env.close()

    final_fraction = training_fraction(num_finetune_episodes - 1, num_finetune_episodes)
    final_epsilon = episode_linear_epsilon(
        num_finetune_episodes - 1, num_finetune_episodes, epsilon_start, epsilon_end
    )
    final_metadata = _checkpoint_metadata(
        episode=len(training_metrics),
        total_steps=total_steps,
        epsilon=final_epsilon,
        fraction=final_fraction,
        source_checkpoint_path=pretrained_checkpoint,
        source_checkpoint=source_checkpoint,
        reward_config=reward_config,
        deadlock_config=deadlock_config,
        tensor_config=tensor_config,
        environment_config=environment_config,
        training_config=training_config,
        training_metrics=training_metrics,
        validation_history=validation_history,
        replay_buffer_initial_size=replay_buffer_initial_size,
    )
    final_path = output_dir / "final_model.pth"
    save_dual_scale_checkpoint(final_path, policy_net, target_net=target_net, optimizer=optimizer, metadata=final_metadata)
    shutil.copy2(final_path, output_dir / "latest_checkpoint.pth")
    _save_json(metrics_dir / "training_metrics.json", training_metrics)
    _save_json(metrics_dir / "validation_history.json", validation_history)
    return policy_net, final_metadata


def load_experiment31_policy(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    policy_net = DualScaleMapDQN(**checkpoint["architecture"]).to(device)
    policy_net.load_state_dict(checkpoint["model_state_dict"])
    policy_net.eval()
    return policy_net, checkpoint


__all__ = [
    "DEFAULT_COVERAGE_THRESHOLDS",
    "Experiment31DeadlockConfig",
    "Experiment3RewardConfig",
    "ExplorationProgressDetector",
    "calculate_action_source_metrics",
    "episode_end_reason",
    "episode_linear_epsilon",
    "evaluate_exploration_controller",
    "initialize_experiment_3_1",
    "load_experiment31_policy",
    "run_evaluation_episode",
    "run_experiment_3_1",
    "save_comparison_artifacts",
    "summarize_evaluations",
    "training_fraction",
    "transition_is_done",
]
