"""Structured episode results and aggregate exploration metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EpisodeResult:
    metrics: dict
    coverage_history: list[float]
    traversable_coverage_history: list[float]
    frontier_distance_history: list[int | None]
    action_log: list[dict] = field(default_factory=list)
    ascii_snapshots: list[dict] = field(default_factory=list)
    frames: list = field(default_factory=list, repr=False)

    def checkpoint_dict(self):
        return dict(self.metrics)


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def action_source_metrics(counts):
    random_actions = int(counts.get("random_action_count", 0))
    greedy_actions = int(counts.get("greedy_action_count", 0))
    random_discoveries = int(counts.get("random_discovery_actions", 0))
    greedy_discoveries = int(counts.get("greedy_discovery_actions", 0))
    random_new_cells = int(counts.get("random_new_cells", 0))
    greedy_new_cells = int(counts.get("greedy_new_cells", 0))
    total_discoveries = random_discoveries + greedy_discoveries
    total_new_cells = random_new_cells + greedy_new_cells
    return {
        "random_action_count": random_actions,
        "greedy_action_count": greedy_actions,
        "random_discovery_actions": random_discoveries,
        "greedy_discovery_actions": greedy_discoveries,
        "random_new_cells": random_new_cells,
        "greedy_new_cells": greedy_new_cells,
        "random_discovery_probability": safe_ratio(random_discoveries, random_actions),
        "greedy_discovery_probability": safe_ratio(greedy_discoveries, greedy_actions),
        "fraction_discovery_events_from_random": safe_ratio(random_discoveries, total_discoveries),
        "fraction_new_cells_from_random": safe_ratio(random_new_cells, total_new_cells),
        "fraction_new_cells_from_greedy": safe_ratio(greedy_new_cells, total_new_cells),
        "random_position_change_probability": safe_ratio(counts.get("random_position_changes", 0), random_actions),
        "greedy_position_change_probability": safe_ratio(counts.get("greedy_position_changes", 0), greedy_actions),
        "random_forward_actions": int(counts.get("random_forward_actions", 0)),
        "greedy_forward_actions": int(counts.get("greedy_forward_actions", 0)),
        "random_forward_movements": int(counts.get("random_forward_movements", 0)),
        "greedy_forward_movements": int(counts.get("greedy_forward_movements", 0)),
        "random_stationary_actions": int(counts.get("random_stationary_actions", 0)),
        "greedy_stationary_actions": int(counts.get("greedy_stationary_actions", 0)),
    }


def steps_to_thresholds(coverage_history, thresholds):
    coverage = np.asarray(coverage_history, dtype=float)
    result = {}
    for threshold in thresholds:
        reached = np.flatnonzero(coverage + 1e-12 >= threshold)
        result[float(threshold)] = float(reached[0]) if reached.size else np.nan
    return result


def safe_correlation(x_values, y_values):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if np.count_nonzero(finite) < 2:
        return np.nan
    x_values = x_values[finite]
    y_values = y_values[finite]
    if np.std(x_values) == 0.0 or np.std(y_values) == 0.0:
        return np.nan
    return float(np.corrcoef(x_values, y_values)[0, 1])


def finite_mean_or_nan(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else np.nan


def summarize_evaluations(results, thresholds=(0.50, 0.75, 0.90, 0.95, 1.00)):
    if not results:
        raise ValueError("results cannot be empty")
    metrics = [result.metrics if isinstance(result, EpisodeResult) else result for result in results]
    end_reasons = Counter(item["episode_end_reason"] for item in metrics)
    count = len(metrics)
    source_counts = Counter()
    for item in metrics:
        for key in (
            "random_action_count", "greedy_action_count", "random_discovery_actions",
            "greedy_discovery_actions", "random_new_cells", "greedy_new_cells",
            "random_position_changes", "greedy_position_changes", "random_forward_actions",
            "greedy_forward_actions", "random_forward_movements", "greedy_forward_movements",
            "random_stationary_actions", "greedy_stationary_actions",
        ):
            source_counts[key] += item.get(key, 0)
    summary = {
        "mean_final_coverage": float(np.mean([m["final_coverage"] for m in metrics])),
        "median_final_coverage": float(np.median([m["final_coverage"] for m in metrics])),
        "mean_traversable_coverage": float(np.mean([m["traversable_coverage"] for m in metrics])),
        "new_cells_per_step": float(np.mean([m["new_cells_per_step"] for m in metrics])),
        "zero_information_ratio": float(np.mean([m["zero_information_ratio"] for m in metrics])),
        "stationary_action_ratio": float(np.mean([m["stationary_action_ratio"] for m in metrics])),
        "oscillation_ratio": float(np.mean([m["oscillation_ratio"] for m in metrics])),
        "blocked_forward_ratio": float(np.mean([m["blocked_forward_ratio"] for m in metrics])),
        "position_change_ratio": float(np.mean([m["position_change_ratio"] for m in metrics])),
        "mean_maximum_consecutive_stationary_steps": float(
            np.mean([m["maximum_consecutive_stationary_steps"] for m in metrics])
        ),
        "mean_frontier_resolution_count": float(
            np.mean([m.get("frontier_resolution_count", 0) for m in metrics])
        ),
        "mean_total_frontier_resolution_reward": float(
            np.mean(
                [m.get("total_frontier_resolution_reward", 0.0) for m in metrics]
            )
        ),
        "mean_frontier_resolution_reward": float(
            np.mean(
                [m.get("mean_frontier_resolution_reward", 0.0) for m in metrics]
            )
        ),
        "maximum_frontier_resolution_reward": float(
            np.max(
                [m.get("maximum_frontier_resolution_reward", 0.0) for m in metrics]
            )
        ),
        "mean_known_cells_at_frontier_resolution": finite_mean_or_nan(
            [m.get("mean_known_cells_at_frontier_resolution", np.nan) for m in metrics]
        ),
        "mean_reachable_frontiers_at_resolution": finite_mean_or_nan(
            [m.get("mean_reachable_frontiers_at_resolution", np.nan) for m in metrics]
        ),
        **action_source_metrics(source_counts),
        "coverage_success_count": int(end_reasons["coverage_success"]),
        "oscillation_deadlock_count": int(end_reasons["oscillation_deadlock"]),
        "stationary_deadlock_count": int(end_reasons["stationary_deadlock"]),
        "no_progress_deadlock_count": int(end_reasons["no_progress_deadlock"]),
        "time_limit_count": int(end_reasons["time_limit"]),
        "environment_terminated_count": int(end_reasons["environment_terminated"]),
        "fraction_coverage_success": float(end_reasons["coverage_success"] / count),
        "fraction_oscillation_deadlock": float(end_reasons["oscillation_deadlock"] / count),
        "fraction_stationary_deadlock": float(end_reasons["stationary_deadlock"] / count),
        "fraction_no_progress_deadlock": float(end_reasons["no_progress_deadlock"] / count),
        "fraction_time_limit": float(end_reasons["time_limit"] / count),
    }
    final_coverages = [m["final_coverage"] for m in metrics]
    summary["frontier_resolution_count_coverage_correlation"] = safe_correlation(
        [m.get("frontier_resolution_count", 0) for m in metrics],
        final_coverages,
    )
    summary["frontier_resolution_reward_coverage_correlation"] = safe_correlation(
        [m.get("total_frontier_resolution_reward", 0.0) for m in metrics],
        final_coverages,
    )
    for threshold in thresholds:
        label = int(round(100 * threshold))
        values = np.asarray([m["steps_to_coverage"][float(threshold)] for m in metrics], dtype=float)
        reached = np.isfinite(values)
        summary[f"fraction_reaching_{label}"] = float(np.mean(reached))
        summary[f"mean_steps_to_{label}_if_reached"] = (
            float(np.mean(values[reached])) if np.any(reached) else np.nan
        )
    return summary


__all__ = [
    "EpisodeResult",
    "action_source_metrics",
    "safe_ratio",
    "steps_to_thresholds",
    "summarize_evaluations",
]
