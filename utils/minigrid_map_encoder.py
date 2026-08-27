"""Dual-scale neural tensor encoding for persistent MiniGrid maps."""

from dataclasses import dataclass

import numpy as np

from minigrid.core.constants import STATE_TO_IDX


@dataclass(frozen=True)
class EncodedPersistentMap:
    """Immutable container for one dual-scale persistent-map observation."""

    local: np.ndarray
    global_map: np.ndarray
    global_scale: int


class PersistentMapTensorEncoder:
    """Encode a sparse mapper state as local and adaptive global tensors."""

    LOCAL_CHANNELS = ("free", "wall", "unknown", "frontier")
    GLOBAL_CHANNELS = (
        "known_density",
        "free_density",
        "wall_density",
        "frontier_density",
        "agent",
    )

    def __init__(
        self,
        local_size=15,
        global_size=15,
        unknown_padding=2,
        hysteresis_bins=1,
        max_global_scale=64,
    ):
        if local_size % 2 == 0 or global_size % 2 == 0:
            raise ValueError("local_size and global_size must be odd")
        if local_size < 3 or global_size < 3:
            raise ValueError("tensor sizes must be at least 3")
        if unknown_padding < 0 or hysteresis_bins < 0:
            raise ValueError("padding and hysteresis must be non-negative")
        if max_global_scale < 1 or max_global_scale & (max_global_scale - 1):
            raise ValueError("max_global_scale must be a positive power of two")

        self.local_size = int(local_size)
        self.global_size = int(global_size)
        self.unknown_padding = int(unknown_padding)
        self.hysteresis_bins = int(hysteresis_bins)
        self.max_global_scale = int(max_global_scale)
        self.global_scale = 1

    def reset(self):
        """Reset monotonic episode-scale state to exact resolution."""
        self.global_scale = 1

    @staticmethod
    def _is_free(cell):
        if cell.object_name in {"empty", "floor", "goal", "lava"}:
            return True

        return (
            cell.object_name == "door"
            and cell.state == STATE_TO_IDX["open"]
        )

    @staticmethod
    def _agent_relative_coordinates(position, mapper):
        """Return physical tensor-row and tensor-column offsets."""
        position = np.asarray(position, dtype=int)
        agent_position = np.asarray(mapper.position, dtype=int)
        delta_x, delta_y = position - agent_position

        # MiniGrid directions: east, south, west, north.
        forward_vectors = (
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1),
        )
        forward_x, forward_y = forward_vectors[int(mapper.direction)]
        right_x, right_y = -forward_y, forward_x

        forward_distance = delta_x * forward_x + delta_y * forward_y
        right_distance = delta_x * right_x + delta_y * right_y

        # Negative row is tensor-up, which is always agent-forward.
        return -int(forward_distance), int(right_distance)

    def _required_global_radius(self, mapper):
        if not mapper.cells:
            return self.unknown_padding

        maximum_offset = max(
            max(abs(row_offset), abs(column_offset))
            for position in mapper.cells
            for row_offset, column_offset in [
                self._agent_relative_coordinates(position, mapper)
            ]
        )
        return maximum_offset + self.unknown_padding

    def _update_global_scale(self, mapper):
        required_radius = self._required_global_radius(mapper)
        center = self.global_size // 2

        while self.global_scale < self.max_global_scale:
            nominal_radius = center * self.global_scale
            hysteresis_margin = self.hysteresis_bins * self.global_scale

            if required_radius <= nominal_radius + hysteresis_margin:
                break

            self.global_scale *= 2

        return self.global_scale

    def _encode_local(self, mapper, frontiers):
        local = np.zeros(
            (len(self.LOCAL_CHANNELS), self.local_size, self.local_size),
            dtype=np.float32,
        )
        local[2, :, :] = 1.0
        center = self.local_size // 2

        for position, cell in mapper.cells.items():
            row_offset, column_offset = self._agent_relative_coordinates(
                position, mapper
            )
            row = center + row_offset
            column = center + column_offset

            if not (0 <= row < self.local_size and 0 <= column < self.local_size):
                continue

            local[2, row, column] = 0.0

            if self._is_free(cell):
                local[0, row, column] = 1.0
            elif cell.object_name == "wall":
                local[1, row, column] = 1.0

            if position in frontiers:
                local[3, row, column] = 1.0

        return local

    def _global_bin(self, offset, scale, center):
        # Center the physical block containing offset zero in the middle bin.
        return center + int(np.floor((offset + scale // 2) / scale))

    def _encode_global(self, mapper, frontiers, scale):
        global_map = np.zeros(
            (
                len(self.GLOBAL_CHANNELS),
                self.global_size,
                self.global_size,
            ),
            dtype=np.float32,
        )
        center = self.global_size // 2
        physical_cells_per_bin = float(scale * scale)

        for position, cell in mapper.cells.items():
            row_offset, column_offset = self._agent_relative_coordinates(
                position, mapper
            )
            row = self._global_bin(row_offset, scale, center)
            column = self._global_bin(column_offset, scale, center)

            if not (0 <= row < self.global_size and 0 <= column < self.global_size):
                continue

            global_map[0, row, column] += 1.0 / physical_cells_per_bin

            if self._is_free(cell):
                global_map[1, row, column] += 1.0 / physical_cells_per_bin
            elif cell.object_name == "wall":
                global_map[2, row, column] += 1.0 / physical_cells_per_bin

            if position in frontiers:
                global_map[3, row, column] += 1.0 / physical_cells_per_bin

        global_map[:4] = np.clip(global_map[:4], 0.0, 1.0)
        global_map[4, center, center] = 1.0
        return global_map

    def encode(self, mapper):
        """Return local/global tensors and the current physical global scale."""
        frontiers = mapper.frontier_cells()
        scale = self._update_global_scale(mapper)
        local = self._encode_local(mapper, frontiers)
        global_map = self._encode_global(mapper, frontiers, scale)

        local.setflags(write=False)
        global_map.setflags(write=False)

        return EncodedPersistentMap(
            local=local,
            global_map=global_map,
            global_scale=scale,
        )


__all__ = ["EncodedPersistentMap", "PersistentMapTensorEncoder"]
