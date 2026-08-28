"""Ground-truth oracle used only to verify mapper completion attainability."""

from __future__ import annotations

import gymnasium as gym

from utils.minigrid_mapper import PersistentMiniGridMapper
from utils.procedural_rooms_env import ENVIRONMENT_ID


_DIRECTION_BY_DELTA = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}
_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def verify_literal_coverage_attainable(environment_config, seeds=(0,)):
    """Exhaustively visit true traversable cells and report observation-only coverage.

    Ground-truth geometry is used only to construct the verification route. The
    persistent mapper still receives ordinary partial observations, exactly as it
    does during training.
    """
    env_kwargs = (
        environment_config.to_kwargs()
        if hasattr(environment_config, "to_kwargs")
        else dict(environment_config)
    )
    env_kwargs["max_steps"] = max(100_000, int(env_kwargs["width"] * env_kwargs["height"] * 20))
    reports = []
    for seed in seeds:
        env = gym.make(ENVIRONMENT_ID, **env_kwargs)
        observation, _ = env.reset(seed=int(seed))
        mapper = PersistentMiniGridMapper()
        mapper.reset(observation)
        traversable = set(env.unwrapped._traversable_positions())
        start = tuple(map(int, env.unwrapped.agent_pos))
        visited = {start}
        route = []
        stack = [(start, iter(_NEIGHBOURS))]
        while stack:
            position, neighbours = stack[-1]
            try:
                dx, dy = next(neighbours)
            except StopIteration:
                stack.pop()
                if stack:
                    route.append(stack[-1][0])
                continue
            neighbour = (position[0] + dx, position[1] + dy)
            if neighbour in traversable and neighbour not in visited:
                visited.add(neighbour)
                route.append(neighbour)
                stack.append((neighbour, iter(_NEIGHBOURS)))
        if visited != traversable:
            raise AssertionError("procedural traversable space was unexpectedly disconnected")

        absolute_position = start
        for target in route:
            delta = (target[0] - absolute_position[0], target[1] - absolute_position[1])
            desired_direction = _DIRECTION_BY_DELTA[delta]
            while int(env.unwrapped.agent_dir) != desired_direction:
                current_direction = int(env.unwrapped.agent_dir)
                action = 1 if (desired_direction - current_direction) % 4 in (1, 2) else 0
                mapper.predict_action(action)
                observation, _, _, truncated, _ = env.step(action)
                mapper.observe(observation)
                if truncated:
                    raise AssertionError("oracle verification exceeded its enlarged step limit")
            mapper.predict_action(2)
            observation, _, _, truncated, _ = env.step(2)
            mapper.observe(observation)
            if truncated:
                raise AssertionError("oracle verification exceeded its enlarged step limit")
            absolute_position = target

        total_cells = env.unwrapped.width * env.unwrapped.height
        report = {
            "seed": int(seed),
            "width": int(env.unwrapped.width),
            "height": int(env.unwrapped.height),
            "visited_traversable_cells": len(visited),
            "mapped_cells": len(mapper.cells),
            "total_map_cells": total_cells,
            "coverage": float(len(mapper.cells) / total_cells),
            "steps": int(env.unwrapped.step_count),
            "literal_completion_attained": len(mapper.cells) == total_cells,
        }
        reports.append(report)
        env.close()
    return reports


__all__ = ["verify_literal_coverage_attainable"]
