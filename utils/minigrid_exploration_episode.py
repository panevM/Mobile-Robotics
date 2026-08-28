"""One shared environment interaction loop for training and evaluation."""

from __future__ import annotations

from collections import Counter

import gymnasium as gym
import numpy as np
from PIL import Image

from utils.minigrid_coverage_rewards import CoverageMilestoneTracker, mapper_coverage
from utils.minigrid_deadlocks import DeadlockTracker
from utils.minigrid_dual_scale_dqn import make_dual_scale_state
from utils.minigrid_exploration_metrics import (
    EpisodeResult,
    action_source_metrics,
    safe_ratio,
    steps_to_thresholds,
)
from utils.minigrid_exploration_diagnostics import compose_exploration_frame
from utils.minigrid_exploration_policy import ExplorationActionSelector
from utils.minigrid_exploration_rewards import compute_exploration_reward
from utils.minigrid_exploration_topology import (
    frontier_target_was_resolved,
    forward_cell_is_known_blocked,
    known_traversable_cells,
    reachable_frontier_distance,
    reachable_frontier_regions,
)
from utils.minigrid_map_encoder import PersistentMapTensorEncoder
from utils.minigrid_mapper import PersistentMiniGridMapper
from utils.procedural_rooms_env import ENVIRONMENT_ID


ACTION_NAMES = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}


def transition_is_done(terminated, truncated, deadlock_type, coverage_success):
    return bool(terminated or truncated or deadlock_type is not None or coverage_success)


def episode_end_reason(terminated, truncated, deadlock_type, coverage_success):
    if coverage_success:
        return "coverage_success"
    if deadlock_type is not None:
        return deadlock_type
    if terminated:
        return "environment_terminated"
    if truncated:
        return "time_limit"
    return "in_progress"


class ExplorationEpisodeRunner:
    """Execute one episode with pluggable policy and transition handling."""

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.action_selector = ExplorationActionSelector(config.allowed_actions)

    def run(
        self,
        *,
        seed,
        policy_net,
        epsilon=0.0,
        random_controller=False,
        random_generator=None,
        transition_handler=None,
        capture_diagnostics=False,
        validation_checkpoint_episode=None,
        environment_override=None,
    ):
        environment_config = environment_override or self.config.environment
        env_kwargs = (
            environment_config.to_kwargs()
            if hasattr(environment_config, "to_kwargs")
            else dict(environment_config)
        )
        if capture_diagnostics:
            env_kwargs["render_mode"] = "rgb_array"
        env = gym.make(ENVIRONMENT_ID, **env_kwargs)
        observation, _ = env.reset(seed=int(seed))
        diagnostic_map_origin = tuple(map(int, env.unwrapped.agent_pos))
        mapper = PersistentMiniGridMapper()
        mapper.reset(observation)
        encoder = PersistentMapTensorEncoder(**self.config.tensor.to_kwargs())
        encoder.reset()
        state = make_dual_scale_state(encoder.encode(mapper))
        rng = random_generator or np.random.default_rng(900_000 + int(seed))
        tracker = CoverageMilestoneTracker(self.config.coverage)
        tracker.reset()
        deadlocks = DeadlockTracker(self.config.deadlock)
        deadlocks.reset(width=env.unwrapped.width, height=env.unwrapped.height)

        total_cells = env.unwrapped.width * env.unwrapped.height
        total_traversable = env.unwrapped.total_traversable_cells
        coverage_history = [mapper_coverage(mapper, total_cells)]
        traversable_history = [known_traversable_cells(mapper) / total_traversable]
        frontier_history = [reachable_frontier_distance(mapper)]
        reward_totals = Counter()
        event_counts = Counter()
        source_counts = Counter()
        scale_counts = Counter([int(2 ** state.global_scale_log2)])
        new_cell_counts = []
        frontier_resolution_rewards = []
        known_cells_at_frontier_resolution = []
        reachable_frontiers_at_resolution = []
        losses = []
        action_log = []
        ascii_snapshots = []
        frames = []
        previous_action = None
        deadlock_type = None
        coverage_success = False
        terminated = truncated = False
        consecutive_stationary = 0
        maximum_consecutive_stationary = 0
        consecutive_reversals = 0
        maximum_consecutive_reversals = 0
        step = 0

        if capture_diagnostics:
            environment_frame = Image.fromarray(env.render()).convert("RGB")
            frames.append(
                compose_exploration_frame(
                    environment_frame,
                    mapper,
                    environment_width=env.unwrapped.width,
                    environment_height=env.unwrapped.height,
                    coverage=coverage_history[-1],
                    step=0,
                    newly_discovered=mapper.cells.keys(),
                    map_origin=diagnostic_map_origin,
                )
                if self.config.validation.show_discovered_map_in_gif
                else environment_frame
            )
            ascii_snapshots.append(
                {
                    "step": 0,
                    "coverage": coverage_history[-1],
                    "traversable_coverage": traversable_history[-1],
                    "frontier_distance": frontier_history[-1],
                    "map": mapper.render_ascii(),
                }
            )

        while not transition_is_done(terminated, truncated, deadlock_type, coverage_success):
            action, action_source = self.action_selector.select_action(
                policy_net,
                state,
                self.device,
                epsilon=epsilon,
                random_generator=rng,
                random_controller=random_controller,
            )
            position_before = tuple(mapper.position)
            frontier_before = reachable_frontier_distance(mapper)
            known_cells_before = len(mapper.cells)
            frontier_regions_before = reachable_frontier_regions(mapper)
            reachable_frontier_count_before = len(frontier_regions_before)
            forward_blocked = forward_cell_is_known_blocked(mapper)
            previous_coverage = coverage_history[-1]
            known_positions_before = set(mapper.cells)

            mapper.predict_action(action)
            observation, _, terminated, truncated, _ = env.step(action)
            mapping_update = mapper.observe(observation)
            newly_discovered = set(mapper.cells) - known_positions_before
            frontier_after = reachable_frontier_distance(mapper)
            resolved_frontier = frontier_target_was_resolved(
                frontier_regions_before,
                mapper.frontier_cells(),
                mapping_update.new_cells,
            )
            position_after = tuple(mapper.position)
            position_changed = position_after != position_before
            current_coverage = mapper_coverage(mapper, total_cells)
            coverage_update = tracker.update(previous_coverage, current_coverage)
            coverage_success = bool(
                self.config.coverage.terminate_on_completion
                and coverage_update.coverage_success
            )
            is_reversal = bool(
                not position_changed
                and previous_action is not None
                and (int(previous_action), int(action)) in {(0, 1), (1, 0)}
            )
            consecutive_reversals = consecutive_reversals + 1 if is_reversal else 0
            maximum_consecutive_reversals = max(
                maximum_consecutive_reversals, consecutive_reversals
            )
            reward = compute_exploration_reward(
                new_cells=mapping_update.new_cells,
                frontier_distance_before=frontier_before,
                frontier_distance_after=frontier_after,
                action=action,
                position_before=position_before,
                position_after=position_after,
                previous_action=previous_action,
                consecutive_reversals=consecutive_reversals,
                forward_was_known_blocked=forward_blocked,
                coverage_milestone_bonus=coverage_update.bonus,
                known_cells_before=known_cells_before,
                reachable_frontier_count_before=reachable_frontier_count_before,
                resolved_frontier=resolved_frontier,
                config=self.config.reward,
            )
            deadlock_status = deadlocks.update(
                action=action,
                previous_action=previous_action,
                position_changed=position_changed,
                new_cells=mapping_update.new_cells,
                frontier_distance_before=frontier_before,
                frontier_distance_after=frontier_after,
            )
            deadlock_type = None if coverage_success else deadlock_status["deadlock_type"]
            if deadlock_type is not None:
                reward["deadlock"] = float(self.config.reward.deadlock_penalty)
                reward["total"] += reward["deadlock"]
            next_state = make_dual_scale_state(encoder.encode(mapper))
            done = transition_is_done(terminated, truncated, deadlock_type, coverage_success)
            if transition_handler is not None:
                loss = transition_handler(state, action, reward["total"], next_state, done)
                if loss is not None:
                    losses.append(float(loss))

            step += 1
            coverage_history.append(current_coverage)
            traversable_history.append(known_traversable_cells(mapper) / total_traversable)
            frontier_history.append(frontier_after)
            new_cell_counts.append(int(mapping_update.new_cells))
            scale_counts[int(2 ** next_state.global_scale_log2)] += 1
            for key in (
                "total", "novelty", "frontier", "frontier_resolution",
                "coverage_milestone", "stationary", "oscillation", "blocked",
                "deadlock",
            ):
                reward_totals[key] += reward[key]
            for key in (
                "stationary_action", "oscillating", "blocked_forward", "frontier_reduced",
                "frontier_increased", "no_reachable_frontier",
            ):
                event_counts[key] += int(reward[key])
            event_counts["coverage_milestones_crossed"] += len(coverage_update.crossed_milestones)
            event_counts["frontier_resolution"] += int(reward["frontier_resolved"])
            if reward["frontier_resolved"]:
                frontier_resolution_rewards.append(reward["frontier_resolution"])
                known_cells_at_frontier_resolution.append(known_cells_before)
                reachable_frontiers_at_resolution.append(
                    reachable_frontier_count_before
                )
            source_counts[f"{action_source}_action_count"] += 1
            source_counts[f"{action_source}_discovery_actions"] += int(mapping_update.new_cells > 0)
            source_counts[f"{action_source}_new_cells"] += int(mapping_update.new_cells)
            source_counts[f"{action_source}_position_changes"] += int(position_changed)
            source_counts[f"{action_source}_forward_actions"] += int(action == 2)
            source_counts[f"{action_source}_forward_movements"] += int(action == 2 and position_changed)
            source_counts[f"{action_source}_stationary_actions"] += int(not position_changed)
            consecutive_stationary = 0 if position_changed else consecutive_stationary + 1
            maximum_consecutive_stationary = max(maximum_consecutive_stationary, consecutive_stationary)

            if capture_diagnostics:
                milestone_events = [
                    f"crossed {threshold:.0%}, bonus +{self.config.coverage.milestone_map[threshold]:g}"
                    for threshold in coverage_update.crossed_milestones
                ]
                if coverage_success:
                    milestone_events.append("reached complete coverage, SUCCESS")
                frontier_resolution_event = None
                if reward["frontier_resolved"]:
                    frontier_resolution_event = (
                        f"step {step}: known_cells={known_cells_before}, "
                        f"reachable_frontiers={reachable_frontier_count_before}, "
                        "resolved_frontier=True, "
                        "frontier_resolution_reward="
                        f"{reward['frontier_resolution']:.2f}"
                    )
                end_reason = episode_end_reason(terminated, truncated, deadlock_type, coverage_success)
                action_log.append(
                    {
                        "step": step,
                        "action": ACTION_NAMES.get(action, f"action_{action}"),
                        "action_id": int(action),
                        "action_source": action_source,
                        "agent_position": list(map(int, mapper.position)),
                        "agent_direction": int(mapper.direction),
                        "position_changed": position_changed,
                        "new_cells": int(mapping_update.new_cells),
                        "coverage": current_coverage,
                        "frontier_distance": frontier_after,
                        "frontier_progress_this_step": deadlock_status["frontier_progress_this_step"],
                        "exploration_progress_this_step": deadlock_status["exploration_progress_this_step"],
                        "steps_since_exploration_progress": deadlock_status["steps_since_exploration_progress"],
                        "no_progress_timeout": deadlock_status["no_progress_timeout"],
                        "crossed_coverage_milestones": list(coverage_update.crossed_milestones),
                        "coverage_milestone_bonus": reward["coverage_milestone"],
                        "coverage_success": coverage_success,
                        "novelty_reward": reward["novelty"],
                        "frontier_reward": reward["frontier"],
                        "frontier_progress_reward": reward["frontier_progress"],
                        "frontier_resolution_reward": reward[
                            "frontier_resolution"
                        ],
                        "frontier_resolved": reward["frontier_resolved"],
                        "known_cells_before": known_cells_before,
                        "reachable_frontiers_before": (
                            reachable_frontier_count_before
                        ),
                        "frontier_resolution_event": frontier_resolution_event,
                        "stationary_penalty": reward["stationary"],
                        "oscillation_penalty": reward["oscillation"],
                        "consecutive_reversals": consecutive_reversals,
                        "oscillation_penalty_multiplier": reward["oscillation_multiplier"],
                        "blocked_penalty": reward["blocked"],
                        "deadlock_penalty": reward["deadlock"],
                        "total_reward": reward["total"],
                        "oscillation_counter": deadlock_status["oscillation_counter"],
                        "stationary_counter": deadlock_status["stationary_counter"],
                        "deadlock_type": deadlock_type,
                        "milestone_events": milestone_events,
                        "event": (
                            f"EPISODE TERMINATED: {end_reason}" if done else None
                        ),
                    }
                )
                environment_frame = Image.fromarray(env.render()).convert("RGB")
                frames.append(
                    compose_exploration_frame(
                        environment_frame,
                        mapper,
                        environment_width=env.unwrapped.width,
                        environment_height=env.unwrapped.height,
                        coverage=current_coverage,
                        step=step,
                        newly_discovered=newly_discovered,
                        milestone_events=milestone_events,
                        map_origin=diagnostic_map_origin,
                    )
                    if self.config.validation.show_discovered_map_in_gif
                    else environment_frame
                )
                if (
                    mapping_update.new_cells > 0
                    or reward["frontier_resolved"]
                    or coverage_update.crossed_milestones
                    or step % self.config.validation.ascii_interval == 0
                    or done
                ):
                    ascii_snapshots.append(
                        {
                            "step": step,
                            "coverage": current_coverage,
                            "traversable_coverage": traversable_history[-1],
                            "frontier_distance": frontier_after,
                            "milestone_events": milestone_events,
                            "map": mapper.render_ascii(),
                        }
                    )
            state = next_state
            previous_action = action

        env.close()
        end_reason = episode_end_reason(terminated, truncated, deadlock_type, coverage_success)
        new_cells_array = np.asarray(new_cell_counts, dtype=float)
        threshold_steps = steps_to_thresholds(
            coverage_history, self.config.validation.coverage_thresholds
        )
        metrics = {
            "seed": int(seed),
            "validation_checkpoint_episode": validation_checkpoint_episode,
            "final_coverage": float(coverage_history[-1]),
            "traversable_coverage": float(traversable_history[-1]),
            "steps_to_coverage": threshold_steps,
            **{
                f"reached_{int(round(100 * threshold))}": bool(np.isfinite(threshold_steps[float(threshold)]))
                for threshold in self.config.validation.coverage_thresholds
            },
            "new_cells_per_step": float(np.mean(new_cells_array)) if new_cells_array.size else 0.0,
            "zero_information_ratio": float(np.mean(new_cells_array == 0)) if new_cells_array.size else 1.0,
            "total_reward": float(reward_totals["total"]),
            "novelty_reward": float(reward_totals["novelty"]),
            "frontier_reward": float(reward_totals["frontier"]),
            "frontier_progress_reward": float(reward_totals["frontier"]),
            "frontier_resolution_count": int(
                event_counts["frontier_resolution"]
            ),
            "total_frontier_resolution_reward": float(
                reward_totals["frontier_resolution"]
            ),
            "mean_frontier_resolution_reward": (
                float(np.mean(frontier_resolution_rewards))
                if frontier_resolution_rewards
                else 0.0
            ),
            "maximum_frontier_resolution_reward": (
                float(np.max(frontier_resolution_rewards))
                if frontier_resolution_rewards
                else 0.0
            ),
            "mean_known_cells_at_frontier_resolution": (
                float(np.mean(known_cells_at_frontier_resolution))
                if known_cells_at_frontier_resolution
                else np.nan
            ),
            "mean_reachable_frontiers_at_resolution": (
                float(np.mean(reachable_frontiers_at_resolution))
                if reachable_frontiers_at_resolution
                else np.nan
            ),
            "coverage_milestone_reward": float(reward_totals["coverage_milestone"]),
            "stationary_penalty": float(reward_totals["stationary"]),
            "oscillation_penalty": float(reward_totals["oscillation"]),
            "blocked_penalty": float(reward_totals["blocked"]),
            "deadlock_penalty": float(reward_totals["deadlock"]),
            "coverage_milestones_crossed": int(event_counts["coverage_milestones_crossed"]),
            "stationary_action_count": int(event_counts["stationary_action"]),
            "stationary_action_ratio": safe_ratio(event_counts["stationary_action"], step),
            "oscillation_count": int(event_counts["oscillating"]),
            "oscillation_ratio": safe_ratio(event_counts["oscillating"], step),
            "maximum_consecutive_reversals": int(maximum_consecutive_reversals),
            "blocked_action_count": int(event_counts["blocked_forward"]),
            "blocked_forward_ratio": safe_ratio(event_counts["blocked_forward"], step),
            "position_change_ratio": safe_ratio(step - event_counts["stationary_action"], step),
            "maximum_consecutive_stationary_steps": maximum_consecutive_stationary,
            "frontier_distance_reduced_steps": int(event_counts["frontier_reduced"]),
            "frontier_distance_increased_steps": int(event_counts["frontier_increased"]),
            "no_reachable_frontier_steps": int(event_counts["no_reachable_frontier"]),
            **action_source_metrics(source_counts),
            "final_global_scale": int(2 ** state.global_scale_log2),
            "episode_length": step,
            "episode_end_reason": end_reason,
            "coverage_success": coverage_success,
            "coverage_success_count": int(coverage_success),
            "oscillation_deadlock_count": int(end_reason == "oscillation_deadlock"),
            "stationary_deadlock_count": int(end_reason == "stationary_deadlock"),
            "no_progress_deadlock_count": int(end_reason == "no_progress_deadlock"),
            "final_oscillation_counter": deadlocks.oscillation_counter,
            "final_stationary_counter": deadlocks.stationary_counter,
            "final_steps_since_exploration_progress": deadlocks.steps_since_exploration_progress,
            "no_progress_timeout": deadlocks.no_progress_timeout,
            "training_loss": float(np.mean(losses)) if losses else np.nan,
            "global_scale_distribution": dict(scale_counts),
        }
        return EpisodeResult(
            metrics=metrics,
            coverage_history=coverage_history,
            traversable_coverage_history=traversable_history,
            frontier_distance_history=frontier_history,
            action_log=action_log,
            ascii_snapshots=ascii_snapshots,
            frames=frames,
        )


__all__ = [
    "ACTION_NAMES",
    "ExplorationEpisodeRunner",
    "episode_end_reason",
    "transition_is_done",
]
