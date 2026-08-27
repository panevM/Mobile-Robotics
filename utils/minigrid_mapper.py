from dataclasses import dataclass, field
import numpy as np

from minigrid.core.constants import (
    OBJECT_TO_IDX,
    IDX_TO_OBJECT,
    IDX_TO_COLOR,
    STATE_TO_IDX,
    DIR_TO_VEC,
)


IDX_TO_STATE = {
    value: key
    for key, value in STATE_TO_IDX.items()
}


@dataclass(frozen=True)
class MapCell:
    object_idx: int
    color_idx: int
    state: int

    @property
    def object_name(self):
        return IDX_TO_OBJECT[self.object_idx]

    @property
    def color_name(self):
        return IDX_TO_COLOR.get(self.color_idx, None)

    @property
    def state_name(self):
        if self.object_name == "door":
            return IDX_TO_STATE.get(self.state, None)

        return None


@dataclass
class MappingUpdate:
    new_cells: int = 0
    changed_cells: int = 0

    # First time an object/color combination is ever encountered
    new_semantics: set = field(default_factory=set)

    # Newly discovered objects at particular map locations
    new_object_cells: list = field(default_factory=list)


class PersistentMiniGridMapper:

    def __init__(self):
        self.cells = {}

        self.position = np.array([0, 0], dtype=int)
        self.direction = 0

        self.visited = set()

        # Keeps track of object categories already encountered
        self.seen_semantics = set()

        # Object currently carried by the agent
        self.carrying = None

    def reset(self, observation):

        self.cells = {}

        # Our own coordinate system always starts here
        self.position = np.array([0, 0], dtype=int)

        self.direction = int(observation["direction"])

        self.visited = {(0, 0)}
        self.seen_semantics = set()

        self.carrying = None

        # We know the cell we are standing on is traversable
        self.cells[(0, 0)] = MapCell(
            object_idx=OBJECT_TO_IDX["empty"],
            color_idx=0,
            state=0,
        )

        return self.observe(observation)

    def _forward_vector(self):

        return np.asarray(
            DIR_TO_VEC[self.direction],
            dtype=int
        )

    def _right_vector(self):

        dx, dy = self._forward_vector()

        return np.array(
            [-dy, dx],
            dtype=int
        )

    def _view_to_world(self, vx, vy, width, height):

        agent_vx = width // 2
        agent_vy = height - 1

        forward_distance = agent_vy - vy
        right_distance = vx - agent_vx

        world_position = (
            self.position
            + forward_distance * self._forward_vector()
            + right_distance * self._right_vector()
        )

        return tuple(world_position)

    def _is_traversable(self, cell):

        name = cell.object_name

        if name in {
            "empty",
            "floor",
            "goal",
            "lava",
        }:
            return True

        if name == "door":
            return cell.state == STATE_TO_IDX["open"]

        return False

    def front_position(self):

        return tuple(
            self.position + self._forward_vector()
        )

    def predict_action(self, action):

        # Turn left
        if action == 0:

            self.direction = (
                self.direction - 1
            ) % 4

        # Turn right
        elif action == 1:

            self.direction = (
                self.direction + 1
            ) % 4

        # Move forward
        elif action == 2:

            target = self.front_position()

            target_cell = self.cells.get(target)

            # The immediately-forward cell should normally
            # already have been observed.
            if target_cell is None:
                return

            if self._is_traversable(target_cell):

                self.position = np.array(
                    target,
                    dtype=int
                )

                self.visited.add(target)

    def observe(self, observation):

        update = MappingUpdate()

        image = observation["image"]

        width, height, _ = image.shape

        # MiniGrid gives us orientation as part
        # of the observation, so use it as a compass correction.
        self.direction = int(
            observation["direction"]
        )

        agent_vx = width // 2
        agent_vy = height - 1

        for vx in range(width):
            for vy in range(height):

                object_idx = int(
                    image[vx, vy, 0]
                )

                color_idx = int(
                    image[vx, vy, 1]
                )

                state = int(
                    image[vx, vy, 2]
                )

                # MiniGrid uses object index 0 for unseen cells.
                if object_idx == OBJECT_TO_IDX["unseen"]:
                    continue

                cell = MapCell(
                    object_idx=object_idx,
                    color_idx=color_idx,
                    state=state,
                )

                # MiniGrid puts the carried object at the
                # agent's own location in its partial view.
                #
                # Therefore we interpret this tile as inventory,
                # not as a world-map cell.
                if (
                    vx == agent_vx
                    and vy == agent_vy
                ):

                    if object_idx == OBJECT_TO_IDX["empty"]:
                        self.carrying = None
                    else:
                        self.carrying = cell

                    continue

                world_position = self._view_to_world(
                    vx,
                    vy,
                    width,
                    height,
                )

                old_cell = self.cells.get(
                    world_position
                )

                # Completely new coordinate
                if old_cell is None:

                    update.new_cells += 1

                    object_name = cell.object_name

                    if object_name not in {
                        "empty",
                        "floor",
                        "wall",
                    }:

                        update.new_object_cells.append(
                            (
                                world_position,
                                cell,
                            )
                        )

                # Known coordinate whose contents changed
                elif old_cell != cell:

                    update.changed_cells += 1

                self.cells[world_position] = cell

                # Semantic discovery
                if cell.object_name not in {
                    "empty",
                    "floor",
                    "wall",
                }:

                    semantic = (
                        cell.object_name,
                        cell.color_name,
                    )

                    if semantic not in self.seen_semantics:

                        self.seen_semantics.add(
                            semantic
                        )

                        update.new_semantics.add(
                            semantic
                        )

        return update

    def frontier_cells(self):

        frontiers = set()

        neighbours = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
        ]

        for position, cell in self.cells.items():

            if not self._is_traversable(cell):
                continue

            # Don't use terminal hazards as exploration frontiers
            if cell.object_name in {
                "goal",
                "lava",
            }:
                continue

            x, y = position

            for dx, dy in neighbours:

                neighbour = (
                    x + dx,
                    y + dy,
                )

                if neighbour not in self.cells:

                    frontiers.add(position)
                    break

        return frontiers

    def known_bounds(self, padding=1):

        coordinates = (
            list(self.cells.keys())
            + [tuple(self.position)]
        )

        xs = [
            position[0]
            for position in coordinates
        ]

        ys = [
            position[1]
            for position in coordinates
        ]

        return (
            min(xs) - padding,
            max(xs) + padding,
            min(ys) - padding,
            max(ys) + padding,
        )

    def render_ascii(self, padding=1):

        min_x, max_x, min_y, max_y = (
            self.known_bounds(padding)
        )

        frontiers = self.frontier_cells()

        agent_symbols = {
            0: ">",
            1: "v",
            2: "<",
            3: "^",
        }

        object_symbols = {
            "empty": ".",
            "floor": ".",
            "wall": "#",
            "key": "K",
            "ball": "O",
            "box": "B",
            "goal": "G",
            "lava": "~",
        }

        rows = []

        for y in range(min_y, max_y + 1):

            row = ""

            for x in range(min_x, max_x + 1):

                position = (x, y)

                if position == tuple(self.position):

                    row += agent_symbols[
                        self.direction
                    ]

                    continue

                cell = self.cells.get(position)

                if cell is None:

                    row += "?"
                    continue

                if cell.object_name == "door":

                    if (
                        cell.state
                        == STATE_TO_IDX["open"]
                    ):
                        row += "/"

                    elif (
                        cell.state
                        == STATE_TO_IDX["locked"]
                    ):
                        row += "L"

                    else:
                        row += "D"

                    continue

                if (
                    position in frontiers
                    and cell.object_name
                    in {"empty", "floor"}
                ):
                    row += "F"

                    continue

                row += object_symbols.get(
                    cell.object_name,
                    "?",
                )

            rows.append(row)

        return "\n".join(rows)