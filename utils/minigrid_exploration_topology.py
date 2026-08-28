"""Mapper-visible topology helpers shared by exploration experiments."""

from __future__ import annotations

from collections import deque

from minigrid.core.constants import STATE_TO_IDX


def is_known_traversable(cell):
    if cell.object_name in {"empty", "floor", "goal", "lava"}:
        return True
    return cell.object_name == "door" and cell.state == STATE_TO_IDX["open"]


def known_traversable_cells(mapper):
    return sum(is_known_traversable(cell) for cell in mapper.cells.values())


def reachable_frontier_distance(mapper):
    """BFS to the nearest frontier through known traversable cells only."""
    start = tuple(int(value) for value in mapper.position)
    frontiers = set(mapper.frontier_cells())
    if not frontiers:
        return None
    traversable = {
        position for position, cell in mapper.cells.items() if is_known_traversable(cell)
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
    cell = mapper.cells.get(mapper.front_position())
    return cell is not None and not is_known_traversable(cell)


__all__ = [
    "forward_cell_is_known_blocked",
    "is_known_traversable",
    "known_traversable_cells",
    "reachable_frontier_distance",
]
