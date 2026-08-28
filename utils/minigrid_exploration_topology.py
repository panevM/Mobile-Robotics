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


def reachable_known_positions(mapper):
    """Return mapper cells reachable from the agent through known free space."""
    start = tuple(int(value) for value in mapper.position)
    traversable = {
        position for position, cell in mapper.cells.items() if is_known_traversable(cell)
    }
    if start not in traversable:
        return set()

    queue = deque([start])
    visited = {start}
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbour = (x + dx, y + dy)
            if neighbour in traversable and neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    return visited


def reachable_frontier_regions(mapper):
    """Group reachable frontier cells into deterministic 4-connected targets."""
    reachable_frontiers = set(mapper.frontier_cells()) & reachable_known_positions(
        mapper
    )
    regions = []

    while reachable_frontiers:
        start = min(reachable_frontiers)
        reachable_frontiers.remove(start)
        region = {start}
        queue = deque([start])

        while queue:
            x, y = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbour = (x + dx, y + dy)
                if neighbour in reachable_frontiers:
                    reachable_frontiers.remove(neighbour)
                    region.add(neighbour)
                    queue.append(neighbour)

        regions.append(frozenset(region))

    return tuple(sorted(regions, key=lambda region: min(region)))


def frontier_target_was_resolved(
    frontier_regions_before,
    frontier_cells_after,
    new_cells,
):
    """Detect one-time resolution of any pre-transition frontier region."""
    if int(new_cells) <= 0 or not frontier_regions_before:
        return False

    frontier_cells_after = set(frontier_cells_after)
    return any(region.isdisjoint(frontier_cells_after) for region in frontier_regions_before)


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
    "frontier_target_was_resolved",
    "forward_cell_is_known_blocked",
    "is_known_traversable",
    "known_traversable_cells",
    "reachable_frontier_distance",
    "reachable_frontier_regions",
    "reachable_known_positions",
]
