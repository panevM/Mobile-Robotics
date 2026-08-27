"""Procedural MiniGrid rooms for intrinsic-exploration experiments."""

from collections import deque
from dataclasses import dataclass

from gymnasium.envs.registration import register, registry
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Wall
from minigrid.minigrid_env import MiniGridEnv


@dataclass(frozen=True)
class RoomRect:
    """Inclusive-free rectangle describing one open BSP region."""

    x: int
    y: int
    width: int
    height: int

    @property
    def bounds(self):
        """Return `(left, top, right, bottom)` with exclusive far edges."""
        return (
            self.x,
            self.y,
            self.x + self.width,
            self.y + self.height,
        )


class ProceduralRoomsEnv(MiniGridEnv):
    """Large connected open regions separated by walls with narrow gaps."""

    def __init__(
        self,
        width=19,
        height=19,
        min_rooms=4,
        max_rooms=6,
        min_room_size=5,
        max_steps=None,
        doorway_width=1,
        max_generation_attempts=100,
        **kwargs,
    ):
        if width < 2 * min_room_size + 3:
            raise ValueError("width is too small for the requested room size")
        if height < 2 * min_room_size + 3:
            raise ValueError("height is too small for the requested room size")
        if not 2 <= min_rooms <= max_rooms:
            raise ValueError("require 2 <= min_rooms <= max_rooms")
        if min_room_size < 2:
            raise ValueError("min_room_size must be at least 2")
        if doorway_width < 1:
            raise ValueError("doorway_width must be at least 1")
        if doorway_width > min_room_size:
            raise ValueError("doorway_width cannot exceed min_room_size")

        self.min_rooms = int(min_rooms)
        self.max_rooms = int(max_rooms)
        self.min_room_size = int(min_room_size)
        self.doorway_width = int(doorway_width)
        self.max_generation_attempts = int(max_generation_attempts)

        self.number_of_rooms = 0
        self.room_rectangles = ()
        self.doorway_positions = ()
        self.total_traversable_cells = 0

        mission_space = MissionSpace(
            mission_func=lambda: "explore the environment"
        )

        if max_steps is None:
            max_steps = 4 * width * height

        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            max_steps=max_steps,
            **kwargs,
        )

    def _possible_split_orientations(self, room):
        orientations = []

        if room.width >= 2 * self.min_room_size + 1:
            orientations.append("vertical")

        if room.height >= 2 * self.min_room_size + 1:
            orientations.append("horizontal")

        return orientations

    def _choose_orientation(self, room, orientations):
        if len(orientations) == 1:
            return orientations[0]

        if room.width > 1.25 * room.height:
            return "vertical"

        if room.height > 1.25 * room.width:
            return "horizontal"

        return orientations[int(self._rand_int(0, len(orientations)))]

    def _split_room(self, room, orientation):
        doorway_positions = []

        if orientation == "vertical":
            minimum_wall_x = room.x + self.min_room_size
            maximum_wall_x = room.x + room.width - self.min_room_size - 1
            wall_x = int(
                self._rand_int(minimum_wall_x, maximum_wall_x + 1)
            )

            left_room = RoomRect(
                room.x,
                room.y,
                wall_x - room.x,
                room.height,
            )
            right_room = RoomRect(
                wall_x + 1,
                room.y,
                room.x + room.width - wall_x - 1,
                room.height,
            )

            for y in range(room.y, room.y + room.height):
                self.grid.set(wall_x, y, Wall())

            maximum_opening_y = (
                room.y + room.height - self.doorway_width
            )
            opening_y = int(self._rand_int(room.y, maximum_opening_y + 1))

            for offset in range(self.doorway_width):
                position = (wall_x, opening_y + offset)
                self.grid.set(*position, None)
                doorway_positions.append(position)

            children = (left_room, right_room)

        else:
            minimum_wall_y = room.y + self.min_room_size
            maximum_wall_y = room.y + room.height - self.min_room_size - 1
            wall_y = int(
                self._rand_int(minimum_wall_y, maximum_wall_y + 1)
            )

            top_room = RoomRect(
                room.x,
                room.y,
                room.width,
                wall_y - room.y,
            )
            bottom_room = RoomRect(
                room.x,
                wall_y + 1,
                room.width,
                room.y + room.height - wall_y - 1,
            )

            for x in range(room.x, room.x + room.width):
                self.grid.set(x, wall_y, Wall())

            maximum_opening_x = room.x + room.width - self.doorway_width
            opening_x = int(self._rand_int(room.x, maximum_opening_x + 1))

            for offset in range(self.doorway_width):
                position = (opening_x + offset, wall_y)
                self.grid.set(*position, None)
                doorway_positions.append(position)

            children = (top_room, bottom_room)

        return children, doorway_positions

    def _generate_candidate_layout(self, target_rooms):
        self.grid = Grid(self.width, self.height)
        self.grid.wall_rect(0, 0, self.width, self.height)

        rooms = [RoomRect(1, 1, self.width - 2, self.height - 2)]
        doorway_positions = []

        while len(rooms) < target_rooms:
            candidates = [
                (index, self._possible_split_orientations(room))
                for index, room in enumerate(rooms)
                if self._possible_split_orientations(room)
            ]

            if not candidates:
                return None

            candidate_index = int(self._rand_int(0, len(candidates)))
            room_index, orientations = candidates[candidate_index]
            room = rooms.pop(room_index)
            orientation = self._choose_orientation(room, orientations)
            children, new_doorways = self._split_room(room, orientation)
            rooms.extend(children)
            doorway_positions.extend(new_doorways)

        return rooms, doorway_positions

    def _traversable_positions(self):
        return {
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if self.grid.get(x, y) is None
        }

    def validate_connectivity(self):
        """Return True when every traversable cell is mutually reachable."""
        traversable = self._traversable_positions()

        if not traversable:
            return False

        start = next(iter(traversable))
        visited = {start}
        queue = deque([start])

        while queue:
            x, y = queue.popleft()

            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbour = (x + dx, y + dy)

                if neighbour in traversable and neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)

        return visited == traversable

    def _gen_grid(self, width, height):
        for _ in range(self.max_generation_attempts):
            target_rooms = int(
                self._rand_int(self.min_rooms, self.max_rooms + 1)
            )
            generated = self._generate_candidate_layout(target_rooms)

            if generated is None or not self.validate_connectivity():
                continue

            rooms, doorway_positions = generated
            self.room_rectangles = tuple(rooms)
            self.doorway_positions = tuple(doorway_positions)
            self.number_of_rooms = len(rooms)
            self.total_traversable_cells = len(self._traversable_positions())
            self.place_agent(rand_dir=True)
            self.mission = "explore the environment"
            return

        raise RuntimeError(
            "Could not generate a connected procedural room layout with "
            "the requested parameters"
        )

    def step(self, action):
        """Apply a MiniGrid action and return zero external task reward."""
        observation, _, _, truncated, info = super().step(action)
        return observation, 0.0, False, truncated, info


ENVIRONMENT_ID = "MiniGrid-ProceduralRooms-v0"

if ENVIRONMENT_ID not in registry:
    register(
        id=ENVIRONMENT_ID,
        entry_point="utils.procedural_rooms_env:ProceduralRoomsEnv",
    )


__all__ = ["ENVIRONMENT_ID", "ProceduralRoomsEnv", "RoomRect"]
