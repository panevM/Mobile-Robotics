"""Scratch-trained Experiment 3 for mobile dual-scale MiniGrid DQN policies."""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.optim as optim

from minigrid.core.constants import STATE_TO_IDX

from utils.dqn_utils import hard_update_target, linear_epsilon, set_seed
from utils.minigrid_dual_scale_dqn import (
    DualScaleMapDQN,
    DualScaleReplayBuffer,
    EXPLORATION_ACTIONS,
    dual_scale_dqn_train_step,
    load_dual_scale_checkpoint,
    make_dual_scale_state,
    mask_q_values,
    save_dual_scale_checkpoint,
    select_masked_greedy_action,
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
DEFAULT_COVERAGE_THRESHOLDS = (0.50, 0.75, 0.90)


@dataclass(frozen=True)
class Experiment3RewardConfig:
    """Novelty-dominant reward with explicit anti-stagnation penalties."""

    novelty_beta: float = 1.0
    frontier_progress_beta: float = 0.15
    stationary_penalty: float = -0.01
    oscillation_penalty: float = -0.04
    blocked_penalty: float = -0.02


def is_known_traversable(cell):
    """Return whether a mapper cell is traversable using known state only."""
    if cell.object_name in {"empty", "floor", "goal", "lava"}:
        return True
    return cell.object_name == "door" and cell.state == STATE_TO_IDX["open"]


def known_traversable_cells(mapper):
    """Count currently mapped cells the mapper believes are traversable."""
    return sum(is_known_traversable(cell) for cell in mapper.cells.values())


def reachable_frontier_distance(mapper):
    """Find the nearest frontier by BFS through known traversable cells.

    Unknown cells are never entered. Unreachable frontier cells therefore do
    not influence the returned distance.
    """
    start = tuple(int(value) for value in mapper.position)
    frontiers = set(mapper.frontier_cells())
    if not frontiers:
        return None

    traversable = {
        position
        for position, cell in mapper.cells.items()
        if is_known_traversable(cell)
    }
    if start not in traversable:
        return None

    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        position, distance = queue.popleft()
        if position in frontiers:
            return distance

        x, y = position
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbour = (x + dx, y + dy)
            if neighbour in traversable and neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, distance + 1))

    return None


def forward_cell_is_known_blocked(mapper):
    """Check the pre-action cell ahead without consulting MiniGrid's true map."""
    cell = mapper.cells.get(mapper.front_position())
    return cell is not None and not is_known_traversable(cell)


def compute_exploration_reward(
    *,
    mapping_update,
    frontier_distance_before,
    frontier_distance_after,
    action,
    position_before,
    position_after,
    previous_action,
    forward_was_known_blocked,
    config,
):
    """Compute Experiment 3 reward components from mapper-visible state."""
    new_cells = int(mapping_update.new_cells)
    novelty = float(config.novelty_beta * new_cells)

    comparable_frontier_distance = (
        frontier_distance_before is not None
        and frontier_distance_after is not None
    )
    frontier_progress = 0.0
    if new_cells == 0 and comparable_frontier_distance:
        frontier_progress = float(
            config.frontier_progress_beta
            * (frontier_distance_before - frontier_distance_after)
        )

    blocked_forward = bool(
        int(action) == 2
        and tuple(position_before) == tuple(position_after)
        and forward_was_known_blocked
    )
    stationary_action = tuple(position_before) == tuple(position_after)
    stationary = float(config.stationary_penalty if stationary_action else 0.0)
    oscillating = bool(
        stationary_action
        and (int(previous_action), int(action)) in {(0, 1), (1, 0)}
    ) if previous_action is not None else False
    oscillation = float(config.oscillation_penalty if oscillating else 0.0)
    blocked = float(config.blocked_penalty if blocked_forward else 0.0)
    distance_delta = None
    if comparable_frontier_distance:
        distance_delta = frontier_distance_before - frontier_distance_after

    return {
        "total": novelty + frontier_progress + stationary + oscillation + blocked,
        "novelty": novelty,
        "frontier": frontier_progress,
        "stationary": stationary,
        "oscillation": oscillation,
        "blocked": blocked,
        "stationary_action": stationary_action,
        "oscillating": oscillating,
        "blocked_forward": blocked_forward,
        "frontier_reduced": bool(new_cells == 0 and distance_delta is not None and distance_delta > 0),
        "frontier_increased": bool(new_cells == 0 and distance_delta is not None and distance_delta < 0),
        "no_reachable_frontier": frontier_distance_before is None,
    }


def select_epsilon_greedy_action_with_source(
    policy_net,
    state,
    epsilon,
    device,
    allowed_actions=EXPLORATION_ACTIONS,
    random_generator=None,
):
    """Select an action and identify the epsilon branch that produced it."""
    allowed_actions = tuple(int(action) for action in allowed_actions)
    if not allowed_actions:
        raise ValueError("allowed_actions cannot be empty")

    if random_generator is None:
        random_generator = np.random.default_rng()
    if random_generator.random() < epsilon:
        action = allowed_actions[int(random_generator.integers(len(allowed_actions)))]
        return action, "random"
    return select_masked_greedy_action(
        policy_net, state, device, allowed_actions=allowed_actions
    ), "greedy"


def _encoder_from_config(tensor_config):
    encoder = PersistentMapTensorEncoder(**tensor_config)
    encoder.reset()
    return encoder


def _steps_to_thresholds(coverage_history, thresholds):
    result = {}
    coverage = np.asarray(coverage_history, dtype=float)
    for threshold in thresholds:
        reached = np.flatnonzero(coverage >= threshold)
        result[threshold] = float(reached[0]) if reached.size else np.nan
    return result


def summarize_evaluations(results, thresholds=DEFAULT_COVERAGE_THRESHOLDS):
    """Summarize coverage and report threshold success with conditional time."""
    if not results:
        raise ValueError("results cannot be empty")

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
        "random_action_count": int(sum(r["random_action_count"] for r in results)),
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


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(data), indent=2), encoding="utf-8"
    )


def _save_curve_png(path, series, title, y_label, color, percent=False):
    """Draw a kernel-safe line chart with PIL instead of Matplotlib."""
    width, height = 900, 420
    left, top, right, bottom = 78, 44, 24, 58
    image = Image.new("RGB", (width, height), (248, 246, 240))
    draw = ImageDraw.Draw(image)
    plot_right = width - right
    plot_bottom = height - bottom
    draw.line((left, top, left, plot_bottom), fill=(45, 45, 45), width=2)
    draw.line((left, plot_bottom, plot_right, plot_bottom), fill=(45, 45, 45), width=2)

    values = np.asarray(
        [np.nan if value is None else value for value in series], dtype=float
    )
    finite = values[np.isfinite(values)]
    y_min = 0.0 if percent else (float(finite.min()) if finite.size else 0.0)
    y_max = 1.0 if percent else (float(finite.max()) if finite.size else 1.0)
    if y_max <= y_min:
        y_max = y_min + 1.0

    for tick in range(6):
        fraction = tick / 5
        y = plot_bottom - fraction * (plot_bottom - top)
        value = y_min + fraction * (y_max - y_min)
        label = f"{100 * value:.0f}%" if percent else f"{value:.1f}"
        draw.line((left, y, plot_right, y), fill=(215, 212, 205), width=1)
        draw.text((8, y - 7), label, fill=(55, 55, 55))

    points = []
    denominator = max(1, len(values) - 1)
    for index, value in enumerate(values):
        if not np.isfinite(value):
            if len(points) >= 2:
                draw.line(points, fill=color, width=3)
            points = []
            continue
        x = left + index / denominator * (plot_right - left)
        y = plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - top)
        points.append((x, y))
    if len(points) >= 2:
        draw.line(points, fill=color, width=3)

    draw.text((left, 14), title, fill=(30, 30, 30))
    draw.text((width // 2 - 30, height - 28), "Step", fill=(55, 55, 55))
    draw.text((8, top - 24), y_label, fill=(55, 55, 55))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _save_multi_curve_png(path, named_series, title):
    width, height = 960, 460
    left, top, right, bottom = 78, 48, 30, 62
    image = Image.new("RGB", (width, height), (248, 246, 240))
    draw = ImageDraw.Draw(image)
    plot_right = width - right
    plot_bottom = height - bottom
    colors = ((28, 126, 92), (202, 91, 42), (50, 91, 168), (135, 75, 145))
    draw.line((left, top, left, plot_bottom), fill=(45, 45, 45), width=2)
    draw.line((left, plot_bottom, plot_right, plot_bottom), fill=(45, 45, 45), width=2)
    for tick in range(6):
        fraction = tick / 5
        y = plot_bottom - fraction * (plot_bottom - top)
        draw.line((left, y, plot_right, y), fill=(215, 212, 205), width=1)
        draw.text((12, y - 7), f"{100 * fraction:.0f}%", fill=(55, 55, 55))

    longest = max(len(values) for values in named_series.values())
    for series_index, (name, values) in enumerate(named_series.items()):
        color = colors[series_index % len(colors)]
        values = np.asarray(values, dtype=float)
        points = []
        denominator = max(1, len(values) - 1)
        for index, value in enumerate(values):
            x = left + index / denominator * (plot_right - left)
            y = plot_bottom - np.clip(value, 0.0, 1.0) * (plot_bottom - top)
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        legend_x = left + 160 * series_index
        draw.line((legend_x, height - 28, legend_x + 24, height - 28), fill=color, width=4)
        draw.text((legend_x + 30, height - 35), name, fill=(45, 45, 45))

    draw.text((left, 16), title, fill=(30, 30, 30))
    draw.text((width // 2 - 30, plot_bottom + 22), f"Step (0-{longest - 1})", fill=(55, 55, 55))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def save_comparison_artifacts(output_dir, named_results, title):
    """Save matched-seed summaries and mean coverage curves for controllers."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {
        name: summarize_evaluations(results)
        for name, results in named_results.items()
    }
    curves = {
        name: np.mean(
            np.stack([result["coverage_history"] for result in results]), axis=0
        )
        for name, results in named_results.items()
    }
    _save_json(output_dir / "summary.json", summaries)

    with (output_dir / "mean_coverage_curves.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        names = list(curves)
        writer.writerow(["step", *names])
        for step in range(max(len(curve) for curve in curves.values())):
            writer.writerow(
                [step]
                + [curve[step] if step < len(curve) else "" for curve in curves.values()]
            )
    _save_multi_curve_png(output_dir / "mean_coverage_curves.png", curves, title)
    return summaries


def _save_diagnostic_artifacts(output_dir, result, frames, action_log, ascii_snapshots):
    output_dir.mkdir(parents=True, exist_ok=True)
    if frames:
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
        if key
        not in {
            "coverage_history",
            "traversable_coverage_history",
            "frontier_distance_history",
        }
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

    csv_path = output_dir / f"seed_{result['seed']}_curves.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "map_coverage", "traversable_coverage", "frontier_distance"])
        for row in zip(
            range(len(result["coverage_history"])),
            result["coverage_history"],
            result["traversable_coverage_history"],
            result["frontier_distance_history"],
        ):
            writer.writerow(row)

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
):
    """Run one greedy or random episode and optionally save full diagnostics."""
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
    consecutive_stationary = 0
    maximum_consecutive_stationary = 0

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
    while not (terminated or truncated):
        # Learned validation is explicitly greedy: no epsilon branch is called.
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
        reward = compute_exploration_reward(
            mapping_update=update,
            frontier_distance_before=frontier_before,
            frontier_distance_after=frontier_after,
            action=action,
            position_before=position_before,
            position_after=tuple(mapper.position),
            previous_action=previous_action,
            forward_was_known_blocked=forward_blocked,
            config=reward_config,
        )
        state = make_dual_scale_state(encoder.encode(mapper))
        step += 1

        coverage = len(mapper.cells) / total_cells
        traversable_coverage = known_traversable_cells(mapper) / total_traversable
        coverage_history.append(coverage)
        traversable_history.append(traversable_coverage)
        frontier_history.append(frontier_after)
        new_cell_counts.append(update.new_cells)
        position_changed = tuple(mapper.position) != position_before
        consecutive_stationary = 0 if position_changed else consecutive_stationary + 1
        maximum_consecutive_stationary = max(
            maximum_consecutive_stationary, consecutive_stationary
        )
        for key in ("total", "novelty", "frontier", "stationary", "oscillation", "blocked"):
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
                    "action_source": (
                        "random_baseline" if random_controller else "greedy/evaluation"
                    ),
                    "agent_position": list(map(int, mapper.position)),
                    "agent_direction": int(mapper.direction),
                    "position_changed": position_changed,
                    "new_cells": int(update.new_cells),
                    "frontier_distance_before": frontier_before,
                    "frontier_distance_after": frontier_after,
                    "coverage": coverage,
                    "novelty_reward": reward["novelty"],
                    "frontier_reward": reward["frontier"],
                    "stationary_penalty": reward["stationary"],
                    "oscillation_penalty": reward["oscillation"],
                    "blocked_penalty": reward["blocked"],
                    "total_reward": reward["total"],
                }
            )
            if update.new_cells > 0 or step % ascii_interval == 0 or terminated or truncated:
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
    new_cells = np.asarray(new_cell_counts, dtype=float)
    steps_to_coverage = _steps_to_thresholds(coverage_history, thresholds)
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
        "stationary_action_count": int(event_counts["stationary_action"]),
        "stationary_action_ratio": float(event_counts["stationary_action"] / step),
        "oscillation_count": int(event_counts["oscillating"]),
        "oscillation_ratio": float(event_counts["oscillating"] / step),
        "blocked_action_count": int(event_counts["blocked_forward"]),
        "blocked_forward_ratio": float(event_counts["blocked_forward"] / step),
        "position_change_ratio": float(1.0 - event_counts["stationary_action"] / step),
        "maximum_consecutive_stationary_steps": maximum_consecutive_stationary,
        "random_action_count": step if random_controller else 0,
        "greedy_action_count": 0 if random_controller else step,
        "frontier_distance_reduced_steps": int(event_counts["frontier_reduced"]),
        "frontier_distance_increased_steps": int(event_counts["frontier_increased"]),
        "no_reachable_frontier_steps": int(event_counts["no_reachable_frontier"]),
        "final_global_scale": int(2 ** state.global_scale_log2),
        "episode_length": step,
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
):
    """Evaluate multiple seeds with deterministic greedy learned actions."""
    if policy_net is not None:
        policy_net.eval()
    results = []
    for seed in seeds:
        seed_artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        results.append(
            run_evaluation_episode(
                seed,
                environment_config,
                tensor_config,
                device,
                reward_config,
                policy_net=policy_net,
                random_controller=random_controller,
                artifact_dir=seed_artifact_dir,
                validation_checkpoint_episode=validation_checkpoint_episode,
                thresholds=thresholds,
            )
        )
    return results


def _safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def calculate_action_source_metrics(counts):
    """Derive random-vs-greedy discovery and movement probabilities."""
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
        "random_discovery_probability": _safe_ratio(random_discoveries, random_actions),
        "greedy_discovery_probability": _safe_ratio(greedy_discoveries, greedy_actions),
        "fraction_of_discovery_events_from_random": _safe_ratio(
            random_discoveries, total_discoveries
        ),
        "fraction_of_new_cells_from_random": _safe_ratio(random_new_cells, total_new_cells),
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


def _checkpoint_metadata(
    *,
    episode,
    total_steps,
    epsilon,
    reward_config,
    tensor_config,
    environment_config,
    training_config,
    training_metrics,
    validation_history,
    replay_buffer_initial_size,
):
    return {
        "experiment_stage": 3,
        "initialization": "random",
        "pretrained_checkpoint": None,
        "replay_buffer_initial_size": int(replay_buffer_initial_size),
        "training_episode": int(episode),
        "total_steps": int(total_steps),
        "epsilon": float(epsilon),
        "reward_config": asdict(reward_config),
        "tensor_config": dict(tensor_config),
        "training_environment_config": dict(environment_config),
        "hyperparameters": dict(training_config),
        "training_metrics": training_metrics,
        "validation_history": validation_history,
    }


def run_experiment_3(
    *,
    output_dir,
    environment_config,
    tensor_config,
    validation_seeds,
    diagnostic_validation_seeds,
    device,
    reward_config=Experiment3RewardConfig(),
    num_training_episodes=2_000,
    training_seed_start=0,
    training_seed_count=10_000,
    training_random_seed=44,
    conv_channels=(32, 64, 64),
    fusion_hidden_size=256,
    learning_rate=1e-4,
    gamma=0.99,
    batch_size=64,
    replay_capacity=100_000,
    minimum_replay_size=5_000,
    target_update_frequency=1_000,
    gradient_clip=10.0,
    epsilon_start=0.10,
    epsilon_end=0.02,
    epsilon_decay_steps=300_000,
    validation_frequency=100,
):
    """Train a randomly initialized Experiment 3 policy from scratch."""
    if len(diagnostic_validation_seeds) != 3:
        raise ValueError("diagnostic_validation_seeds must contain exactly three seeds")
    if epsilon_start <= 0 or epsilon_end <= 0:
        raise ValueError("training epsilon must remain above zero")
    if epsilon_end > epsilon_start:
        raise ValueError("epsilon_end cannot exceed epsilon_start")

    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    artifact_root = output_dir / "validation_artifacts"
    metrics_dir = output_dir / "metrics"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    set_seed(training_random_seed)
    rng = np.random.default_rng(training_random_seed)
    policy_net = DualScaleMapDQN(
        action_dim=7,
        conv_channels=conv_channels,
        fusion_hidden_size=fusion_hidden_size,
    ).to(device)
    target_net = DualScaleMapDQN(
        action_dim=7,
        conv_channels=conv_channels,
        fusion_hidden_size=fusion_hidden_size,
    ).to(device)
    hard_update_target(policy_net, target_net)
    target_net.eval()
    policy_net.train()
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = DualScaleReplayBuffer(replay_capacity)
    replay_buffer_initial_size = len(replay_buffer)
    if replay_buffer_initial_size != 0:
        raise RuntimeError("Experiment 3 replay buffer must start empty")

    training_metrics = []
    validation_history = []
    total_steps = 0
    epsilon = float(epsilon_start)
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
        "epsilon_decay_steps": epsilon_decay_steps,
        "validation_frequency": validation_frequency,
        "allowed_actions": EXPLORATION_ACTIONS,
        "training_seed_range": (
            training_seed_start,
            training_seed_start + training_seed_count - 1,
        ),
    }

    try:
        for episode_index in range(num_training_episodes):
            episode_number = episode_index + 1
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
            consecutive_stationary = 0
            maximum_consecutive_stationary = 0
            terminated = truncated = False

            while not (terminated or truncated):
                epsilon = linear_epsilon(
                    total_steps, epsilon_start, epsilon_end, epsilon_decay_steps
                )
                action, action_source = select_epsilon_greedy_action_with_source(
                    policy_net,
                    state,
                    epsilon,
                    device,
                    random_generator=rng,
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
                next_state = make_dual_scale_state(encoder.encode(mapper))
                done = terminated or truncated
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
                for key in ("total", "novelty", "frontier", "stationary", "oscillation", "blocked"):
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
                source_counts[f"{action_source}_forward_movements"] += int(
                    action == 2 and position_changed
                )
                source_counts[f"{action_source}_stationary_actions"] += int(not position_changed)

                consecutive_stationary = 0 if position_changed else consecutive_stationary + 1
                maximum_consecutive_stationary = max(
                    maximum_consecutive_stationary, consecutive_stationary
                )
                new_cell_counts.append(update.new_cells)
                scale_counts[int(2 ** next_state.global_scale_log2)] += 1
                coverage_history.append(len(mapper.cells) / total_cells)
                state = next_state
                previous_action = action

                if total_steps % target_update_frequency == 0:
                    hard_update_target(policy_net, target_net)

            new_cells = np.asarray(new_cell_counts, dtype=float)
            local_tensor, global_tensor, scale_tensor = DualScaleReplayBuffer._batch_states(
                [state], device
            )
            with torch.no_grad():
                predicted_q = mask_q_values(
                    policy_net(local_tensor, global_tensor, scale_tensor)
                )[:, list(EXPLORATION_ACTIONS)]

            source_metrics = calculate_action_source_metrics(source_counts)
            training_metrics.append(
                {
                    "episode": episode_number,
                    "total_reward": float(rewards["total"]),
                    "novelty_reward": float(rewards["novelty"]),
                    "frontier_reward": float(rewards["frontier"]),
                    "stationary_penalty": float(rewards["stationary"]),
                    "oscillation_penalty": float(rewards["oscillation"]),
                    "blocked_penalty": float(rewards["blocked"]),
                    "stationary_action_count": int(events["stationary_action"]),
                    "stationary_action_ratio": _safe_ratio(events["stationary_action"], episode_steps),
                    "oscillation_count": int(events["oscillating"]),
                    "oscillation_ratio": _safe_ratio(events["oscillating"], episode_steps),
                    "blocked_forward_count": int(events["blocked_forward"]),
                    "blocked_forward_ratio": _safe_ratio(events["blocked_forward"], episode_steps),
                    "maximum_consecutive_stationary_steps": maximum_consecutive_stationary,
                    "position_change_ratio": _safe_ratio(
                        episode_steps - events["stationary_action"], episode_steps
                    ),
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
                    "epsilon": epsilon,
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
                    f"Episode {episode_number:4d}/{num_training_episodes} | "
                    f"coverage={np.mean([m['final_coverage'] for m in recent]):.1%} | "
                    f"reward={np.mean([m['total_reward'] for m in recent]):.2f} | "
                    f"novelty={np.mean([m['novelty_reward'] for m in recent]):.2f} | "
                    f"frontier={np.mean([m['frontier_reward'] for m in recent]):+.2f} | "
                    f"epsilon={epsilon:.3f} | loss={mean_loss:.4f} | "
                    f"move={np.mean([m['position_change_ratio'] for m in recent]):.1%} | "
                    f"osc={np.mean([m['oscillation_ratio'] for m in recent]):.1%} | "
                    f"Pdisc(greedy)={np.mean([m['greedy_discovery_probability'] for m in recent]):.1%} | "
                    f"Pdisc(random)={np.mean([m['random_discovery_probability'] for m in recent]):.1%}"
                )

            if episode_number % validation_frequency == 0:
                validation_results = evaluate_exploration_controller(
                    validation_seeds,
                    environment_config,
                    tensor_config,
                    device,
                    reward_config,
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
                    reward_config=reward_config,
                    tensor_config=tensor_config,
                    environment_config=environment_config,
                    training_config=training_config,
                    training_metrics=training_metrics,
                    validation_history=validation_history,
                    replay_buffer_initial_size=replay_buffer_initial_size,
                )
                checkpoint_path = checkpoint_dir / f"checkpoint_ep_{episode_number:04d}.pth"
                save_dual_scale_checkpoint(
                    checkpoint_path,
                    policy_net,
                    target_net=target_net,
                    optimizer=optimizer,
                    metadata=metadata,
                )
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
                    policy_net=policy_net,
                    artifact_dir=diagnostic_dir,
                    validation_checkpoint_episode=episode_number,
                )
                _save_json(diagnostic_dir / "validation_summary.json", validation_summary)
                policy_net.train()
                print(
                    f"Validation {episode_number}: coverage={validation_summary['mean_final_coverage']:.1%} | "
                    f"median={validation_summary['median_final_coverage']:.1%} | "
                    f"zero-info={validation_summary['zero_information_ratio']:.1%} | "
                    f"movement={validation_summary['position_change_ratio']:.1%} | "
                    f"stationary={validation_summary['stationary_action_ratio']:.1%} | "
                    f"oscillation={validation_summary['oscillation_ratio']:.1%} | "
                    f"50%={validation_summary['fraction_reaching_50']:.1%} | "
                    f"75%={validation_summary['fraction_reaching_75']:.1%} | "
                    f"90%={validation_summary['fraction_reaching_90']:.1%}"
                )
    finally:
        training_env.close()

    final_metadata = _checkpoint_metadata(
        episode=len(training_metrics),
        total_steps=total_steps,
        epsilon=epsilon,
        reward_config=reward_config,
        tensor_config=tensor_config,
        environment_config=environment_config,
        training_config=training_config,
        training_metrics=training_metrics,
        validation_history=validation_history,
        replay_buffer_initial_size=replay_buffer_initial_size,
    )
    final_path = output_dir / "final_model.pth"
    save_dual_scale_checkpoint(
        final_path,
        policy_net,
        target_net=target_net,
        optimizer=optimizer,
        metadata=final_metadata,
    )
    shutil.copy2(final_path, output_dir / "latest_checkpoint.pth")
    _save_json(metrics_dir / "training_metrics.json", training_metrics)
    _save_json(metrics_dir / "validation_history.json", validation_history)
    return policy_net, final_metadata


def load_experiment3_policy(path, device):
    """Load an Experiment 3 checkpoint for greedy evaluation."""
    return load_dual_scale_checkpoint(path, device)


__all__ = [
    "ACTION_NAMES",
    "DEFAULT_COVERAGE_THRESHOLDS",
    "Experiment3RewardConfig",
    "calculate_action_source_metrics",
    "compute_exploration_reward",
    "evaluate_exploration_controller",
    "forward_cell_is_known_blocked",
    "known_traversable_cells",
    "load_experiment3_policy",
    "reachable_frontier_distance",
    "run_evaluation_episode",
    "run_experiment_3",
    "save_comparison_artifacts",
    "select_epsilon_greedy_action_with_source",
    "summarize_evaluations",
]
