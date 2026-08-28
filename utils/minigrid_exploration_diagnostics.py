"""Artifact and comparison output for reusable exploration episodes."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from utils.minigrid_exploration_metrics import EpisodeResult, summarize_evaluations


_CELL_COLORS = {
    "empty": (232, 226, 207),
    "floor": (232, 226, 207),
    "wall": (74, 88, 101),
    "door": (151, 105, 56),
    "goal": (63, 166, 100),
    "lava": (225, 91, 45),
    "key": (239, 196, 61),
    "ball": (73, 145, 214),
    "box": (170, 123, 79),
}


def compose_exploration_frame(
    environment_frame,
    mapper,
    *,
    environment_width,
    environment_height,
    coverage,
    step,
    newly_discovered=(),
    milestone_events=(),
    map_origin=None,
):
    """Place a persistent, mapper-only discovered map beside the environment."""
    environment_image = environment_frame.convert("RGB")
    panel_size = environment_image.height
    header_height = 38
    footer_height = 28
    panel = Image.new("RGB", (panel_size, panel_size), (14, 20, 27))
    draw = ImageDraw.Draw(panel)

    if map_origin is None:
        coordinate_width = 2 * int(environment_width) - 1
        coordinate_height = 2 * int(environment_height) - 1
        origin_x = int(environment_width) - 1
        origin_y = int(environment_height) - 1
    else:
        coordinate_width = int(environment_width)
        coordinate_height = int(environment_height)
        origin_x, origin_y = map(int, map_origin)
    margin = 14
    tile_size = max(
        2,
        min(
            (panel_size - 2 * margin) // coordinate_width,
            (panel_size - 2 * margin) // coordinate_height,
        ),
    )
    grid_width = tile_size * coordinate_width
    grid_height = tile_size * coordinate_height
    offset_x = (panel_size - grid_width) // 2
    offset_y = (panel_size - grid_height) // 2
    frontiers = set(mapper.frontier_cells())
    newly_discovered = set(newly_discovered)

    for (x, y), cell in mapper.cells.items():
        grid_x = x + origin_x
        grid_y = y + origin_y
        if not (0 <= grid_x < coordinate_width and 0 <= grid_y < coordinate_height):
            continue
        left = offset_x + grid_x * tile_size
        top = offset_y + grid_y * tile_size
        right = left + tile_size - 1
        bottom = top + tile_size - 1
        fill = _CELL_COLORS.get(cell.object_name, (155, 109, 181))
        outline = (35, 43, 50)
        width = 1
        if (x, y) in frontiers:
            outline = (245, 177, 66)
            width = max(1, tile_size // 6)
        if (x, y) in newly_discovered:
            outline = (53, 221, 184)
            width = max(2, tile_size // 5)
        draw.rectangle((left, top, right, bottom), fill=fill, outline=outline, width=width)
        if (x, y) in mapper.visited and tile_size >= 7:
            radius = max(1, tile_size // 7)
            center_x = (left + right) // 2
            center_y = (top + bottom) // 2
            draw.ellipse(
                (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
                fill=(42, 145, 129),
            )

    agent_x, agent_y = map(int, mapper.position)
    grid_x = agent_x + origin_x
    grid_y = agent_y + origin_y
    if 0 <= grid_x < coordinate_width and 0 <= grid_y < coordinate_height:
        left = offset_x + grid_x * tile_size
        top = offset_y + grid_y * tile_size
        right = left + tile_size - 1
        bottom = top + tile_size - 1
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        inset = max(1, tile_size * 0.18)
        direction_points = {
            0: ((right - inset, center_y), (left + inset, top + inset), (left + inset, bottom - inset)),
            1: ((center_x, bottom - inset), (left + inset, top + inset), (right - inset, top + inset)),
            2: ((left + inset, center_y), (right - inset, top + inset), (right - inset, bottom - inset)),
            3: ((center_x, top + inset), (left + inset, bottom - inset), (right - inset, bottom - inset)),
        }
        draw.polygon(direction_points[int(mapper.direction)], fill=(244, 77, 75))

    canvas_width = environment_image.width + panel_size
    canvas = Image.new(
        "RGB",
        (canvas_width, header_height + panel_size + footer_height),
        (22, 28, 34),
    )
    canvas.paste(environment_image, (0, header_height))
    canvas.paste(panel, (environment_image.width, header_height))
    canvas_draw = ImageDraw.Draw(canvas)
    canvas_draw.text((12, 12), "Environment", fill=(238, 238, 232))
    canvas_draw.text(
        (environment_image.width + 12, 12),
        f"Persistent discovered map | step {step} | coverage {coverage:.1%}",
        fill=(238, 238, 232),
    )
    legend = "known floor  |  wall  |  amber: frontier  |  teal border: newly discovered"
    canvas_draw.text((12, header_height + panel_size + 8), legend, fill=(190, 199, 205))
    if milestone_events:
        event_text = " | ".join(str(event) for event in milestone_events)
        text_width = canvas_draw.textlength(event_text)
        canvas_draw.rectangle(
            (
                max(0, canvas_width - text_width - 20),
                header_height + panel_size + 3,
                canvas_width,
                header_height + panel_size + footer_height,
            ),
            fill=(40, 92, 74),
        )
        canvas_draw.text(
            (max(8, canvas_width - text_width - 10), header_height + panel_size + 8),
            event_text,
            fill=(241, 250, 245),
        )
    return canvas


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(data), indent=2), encoding="utf-8")


def _save_curve(path, values, title, percent=False, color=(28, 126, 92)):
    width, height = 900, 420
    left, top, right, bottom = 78, 44, 24, 58
    image = Image.new("RGB", (width, height), (248, 246, 240))
    draw = ImageDraw.Draw(image)
    plot_right, plot_bottom = width - right, height - bottom
    draw.line((left, top, left, plot_bottom), fill=(45, 45, 45), width=2)
    draw.line((left, plot_bottom, plot_right, plot_bottom), fill=(45, 45, 45), width=2)
    numeric = np.asarray([np.nan if value is None else value for value in values], dtype=float)
    finite = numeric[np.isfinite(numeric)]
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
    for index, value in enumerate(numeric):
        if not np.isfinite(value):
            if len(points) >= 2:
                draw.line(points, fill=color, width=3)
            points = []
            continue
        x = left + index / max(1, len(numeric) - 1) * (plot_right - left)
        y = plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - top)
        points.append((x, y))
    if len(points) >= 2:
        draw.line(points, fill=color, width=3)
    draw.text((left, 14), title, fill=(30, 30, 30))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def save_diagnostic_episode(result: EpisodeResult, output_dir):
    """Save GIF, logs, mapper snapshots, trajectories, and curves."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = result.metrics["seed"]
    frames = list(result.frames)
    if frames:
        if result.metrics["episode_end_reason"] != "time_limit":
            frames.extend([frames[-1].copy(), frames[-1].copy()])
        frames[0].save(
            output_dir / f"seed_{seed}.gif",
            save_all=True,
            append_images=frames[1:],
            duration=140,
            loop=0,
            optimize=False,
        )
    save_json(output_dir / f"seed_{seed}_actions.json", result.action_log)
    save_json(output_dir / f"seed_{seed}_summary.json", result.metrics)
    lines = []
    for snapshot in result.ascii_snapshots:
        lines.extend(
            [
                f"Step {snapshot['step']}",
                f"Coverage: {snapshot['coverage']:.1%}",
                f"Traversable coverage: {snapshot['traversable_coverage']:.1%}",
                f"Frontier distance: {snapshot['frontier_distance']}",
                *[f"EVENT: {event}" for event in snapshot.get("milestone_events", [])],
                snapshot["map"],
                "",
            ]
        )
    (output_dir / f"seed_{seed}_mapper.txt").write_text("\n".join(lines), encoding="utf-8")
    with (output_dir / f"seed_{seed}_curves.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "map_coverage", "traversable_coverage", "frontier_distance"])
        writer.writerows(
            zip(
                range(len(result.coverage_history)),
                result.coverage_history,
                result.traversable_coverage_history,
                result.frontier_distance_history,
            )
        )
    _save_curve(output_dir / f"seed_{seed}_coverage.png", result.coverage_history, f"Coverage, seed {seed}", percent=True)
    _save_curve(
        output_dir / f"seed_{seed}_traversable_coverage.png",
        result.traversable_coverage_history,
        f"Traversable coverage, seed {seed}",
        percent=True,
        color=(50, 91, 168),
    )
    _save_curve(
        output_dir / f"seed_{seed}_frontier_distance.png",
        result.frontier_distance_history,
        f"Reachable-frontier BFS distance, seed {seed}",
        color=(202, 91, 42),
    )


def _mean_padded_curve(results):
    histories = [np.asarray(result.coverage_history, dtype=float) for result in results]
    length = max(len(history) for history in histories)
    return np.mean(np.stack([np.pad(h, (0, length - len(h)), mode="edge") for h in histories]), axis=0)


def save_comparison_artifacts(output_dir, named_results, thresholds=(0.50, 0.75, 0.90, 0.95, 1.00)):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {name: summarize_evaluations(results, thresholds) for name, results in named_results.items()}
    curves = {name: _mean_padded_curve(results) for name, results in named_results.items()}
    save_json(output_dir / "summary.json", summaries)
    with (output_dir / "mean_coverage_curves.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        names = list(curves)
        writer.writerow(["step", *names])
        for step in range(max(map(len, curves.values()))):
            writer.writerow([step] + [curve[step] if step < len(curve) else "" for curve in curves.values()])
    for name, curve in curves.items():
        _save_curve(output_dir / f"{name.lower().replace(' ', '_')}_coverage.png", curve, f"Mean coverage: {name}", percent=True)
    return summaries


__all__ = [
    "compose_exploration_frame",
    "json_ready",
    "save_comparison_artifacts",
    "save_diagnostic_episode",
    "save_json",
]
