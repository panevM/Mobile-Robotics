"""Frontier-shaped fine-tuning and diagnostics for dual-scale MiniGrid DQN."""

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
    select_masked_epsilon_greedy_action,
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
class ExplorationRewardConfig:
    """Coefficients for novelty-dominant Experiment 2 reward shaping."""

    exploration_beta: float = 1.0
    frontier_progress_beta: float = 0.15
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
    forward_was_known_blocked,
    config,
):
    """Compute novelty, reachable-frontier, and blocked-motion rewards."""
    new_cells = int(mapping_update.new_cells)
    novelty = float(config.exploration_beta * new_cells)

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
    blocked = float(config.blocked_penalty if blocked_forward else 0.0)
    distance_delta = None
    if comparable_frontier_distance:
        distance_delta = frontier_distance_before - frontier_distance_after

    return {
        "total": novelty + frontier_progress + blocked,
        "novelty": novelty,
        "frontier_progress": frontier_progress,
        "blocked": blocked,
        "blocked_forward": blocked_forward,
        "frontier_reduced": bool(new_cells == 0 and distance_delta is not None and distance_delta > 0),
        "frontier_increased": bool(new_cells == 0 and distance_delta is not None and distance_delta < 0),
        "no_reachable_frontier": frontier_distance_before is None,
    }


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
        for key in ("total", "novelty", "frontier_progress", "blocked"):
            reward_totals[key] += reward[key]
        for key in (
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
                    "agent_position": list(map(int, mapper.position)),
                    "agent_direction": int(mapper.direction),
                    "new_cells": int(update.new_cells),
                    "frontier_distance_before": frontier_before,
                    "frontier_distance_after": frontier_after,
                    "coverage": coverage,
                    **{key: reward[key] for key in ("novelty", "frontier_progress", "blocked", "total")},
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
        "frontier_reward": float(reward_totals["frontier_progress"]),
        "blocked_penalty": float(reward_totals["blocked"]),
        "blocked_action_count": int(event_counts["blocked_forward"]),
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


def _checkpoint_metadata(
    *,
    source_checkpoint,
    episode,
    total_steps,
    epsilon,
    reward_config,
    tensor_config,
    environment_config,
    training_config,
    training_metrics,
    validation_history,
):
    return {
        "experiment_stage": 2,
        "source_experiment_1_checkpoint": str(Path(source_checkpoint).resolve()),
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


def run_experiment_2(
    *,
    pretrained_checkpoint,
    output_dir,
    environment_config,
    tensor_config,
    validation_seeds,
    diagnostic_validation_seeds,
    device,
    reward_config=ExplorationRewardConfig(),
    num_training_episodes=2_000,
    training_seed_start=0,
    training_seed_count=10_000,
    training_random_seed=43,
    learning_rate=5e-5,
    gamma=0.99,
    batch_size=64,
    replay_capacity=100_000,
    minimum_replay_size=5_000,
    target_update_frequency=1_000,
    gradient_clip=10.0,
    epsilon_start=0.25,
    epsilon_end=0.05,
    epsilon_decay_steps=200_000,
    validation_frequency=100,
):
    """Fine-tune Experiment 1 weights with a fresh buffer and shaped reward."""
    pretrained_checkpoint = Path(pretrained_checkpoint)
    if not pretrained_checkpoint.exists():
        raise FileNotFoundError(
            f"Experiment 1 checkpoint not found: {pretrained_checkpoint.resolve()}"
        )
    if len(diagnostic_validation_seeds) != 3:
        raise ValueError("diagnostic_validation_seeds must contain exactly three seeds")

    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    artifact_root = output_dir / "validation_artifacts"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    set_seed(training_random_seed)
    rng = np.random.default_rng(training_random_seed)
    source = torch.load(pretrained_checkpoint, map_location=device, weights_only=False)
    architecture = source["architecture"]
    source_tensor_config = source.get("tensor_config")
    if source_tensor_config is not None:
        mismatches = {
            key: (source_tensor_config[key], tensor_config.get(key))
            for key in source_tensor_config
            if tensor_config.get(key) != source_tensor_config[key]
        }
        if mismatches:
            raise ValueError(
                "Experiment 2 tensor configuration must match the pretrained "
                f"checkpoint; mismatches: {mismatches}"
            )
    policy_net = DualScaleMapDQN(**architecture).to(device)
    policy_net.load_state_dict(source["model_state_dict"])
    target_net = DualScaleMapDQN(**architecture).to(device)
    hard_update_target(policy_net, target_net)
    target_net.eval()
    policy_net.train()

    # Reward semantics changed, so neither replay nor optimizer state is reused.
    replay_buffer = DualScaleReplayBuffer(replay_capacity)
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
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
            losses = []
            new_cell_counts = []
            scale_counts = Counter([int(2 ** state.global_scale_log2)])
            coverage_history = [len(mapper.cells) / total_cells]
            episode_steps = 0
            terminated = truncated = False

            while not (terminated or truncated):
                epsilon = linear_epsilon(
                    total_steps, epsilon_start, epsilon_end, epsilon_decay_steps
                )
                action = select_masked_epsilon_greedy_action(
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
                reward = compute_exploration_reward(
                    mapping_update=update,
                    frontier_distance_before=frontier_before,
                    frontier_distance_after=frontier_after,
                    action=action,
                    position_before=position_before,
                    position_after=tuple(mapper.position),
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
                for key in ("total", "novelty", "frontier_progress", "blocked"):
                    rewards[key] += reward[key]
                for key in (
                    "blocked_forward",
                    "frontier_reduced",
                    "frontier_increased",
                    "no_reachable_frontier",
                ):
                    events[key] += int(reward[key])
                new_cell_counts.append(update.new_cells)
                scale_counts[int(2 ** next_state.global_scale_log2)] += 1
                coverage_history.append(len(mapper.cells) / total_cells)
                state = next_state

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

            training_metrics.append(
                {
                    "episode": episode_number,
                    "total_reward": float(rewards["total"]),
                    "novelty_reward": float(rewards["novelty"]),
                    "frontier_progress_reward": float(rewards["frontier_progress"]),
                    "blocked_motion_penalty": float(rewards["blocked"]),
                    "blocked_forward_actions": int(events["blocked_forward"]),
                    "frontier_distance_reduced_steps": int(events["frontier_reduced"]),
                    "frontier_distance_increased_steps": int(events["frontier_increased"]),
                    "no_reachable_frontier_steps": int(events["no_reachable_frontier"]),
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
                    f"frontier={np.mean([m['frontier_progress_reward'] for m in recent]):+.2f} | "
                    f"epsilon={epsilon:.3f} | loss={mean_loss:.4f}"
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
                validation_history.append({"episode": episode_number, **validation_summary})
                diagnostic_dir = artifact_root / f"episode_{episode_number:04d}"
                metadata = _checkpoint_metadata(
                    source_checkpoint=pretrained_checkpoint,
                    episode=episode_number,
                    total_steps=total_steps,
                    epsilon=epsilon,
                    reward_config=reward_config,
                    tensor_config=tensor_config,
                    environment_config=environment_config,
                    training_config=training_config,
                    training_metrics=training_metrics,
                    validation_history=validation_history,
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
                    f"Validation {episode_number}: coverage="
                    f"{validation_summary['mean_final_coverage']:.1%}, "
                    f"zero-info={validation_summary['zero_information_ratio']:.1%}; "
                    f"saved {checkpoint_path}"
                )
    finally:
        training_env.close()

    final_metadata = _checkpoint_metadata(
        source_checkpoint=pretrained_checkpoint,
        episode=len(training_metrics),
        total_steps=total_steps,
        epsilon=epsilon,
        reward_config=reward_config,
        tensor_config=tensor_config,
        environment_config=environment_config,
        training_config=training_config,
        training_metrics=training_metrics,
        validation_history=validation_history,
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
    return policy_net, final_metadata


def load_experiment_policy(path, device):
    """Load either Experiment 1 or Experiment 2 in greedy evaluation mode."""
    return load_dual_scale_checkpoint(path, device)


__all__ = [
    "ACTION_NAMES",
    "DEFAULT_COVERAGE_THRESHOLDS",
    "ExplorationRewardConfig",
    "compute_exploration_reward",
    "evaluate_exploration_controller",
    "forward_cell_is_known_blocked",
    "known_traversable_cells",
    "load_experiment_policy",
    "reachable_frontier_distance",
    "run_evaluation_episode",
    "run_experiment_2",
    "save_comparison_artifacts",
    "summarize_evaluations",
]
