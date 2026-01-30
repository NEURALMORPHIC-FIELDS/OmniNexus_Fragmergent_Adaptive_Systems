"""
Avatar - Digital agent with energy management and smart navigation.

Represents an autonomous agent that moves through the world, consumes energy,
and collects resources. Tracks trajectory and interaction history.

Author: Lucian Coman
Date: January 2025
Version: 10.0.0
"""

import numpy as np
from typing import List, Tuple
from collections import deque


# Constants for energy dynamics
MOVEMENT_COST = 0.1
RICH_TERRAIN_THRESHOLD = 0.5
RICH_TERRAIN_GAIN = 2.0
POOR_TERRAIN_LOSS = 0.5
MAX_ENERGY = 100.0
MIN_ENERGY = 0.0

# Maximum trajectory length to store (for memory optimization)
MAX_TRAJECTORY_LENGTH = 10000


class Avatar:
    """
    Energy-constrained agent navigating procedural world.

    The avatar moves through a 2D grid world, consuming energy with each
    action and gaining energy by interacting with resource-rich regions.

    Attributes:
        pos (List[int]): Current position [x, y]
        energy (float): Current energy level [0, 100]
        trajectory (deque): Recent movement history (limited size)
        interactions (List[float]): History of terrain richness encountered

    Example:
        >>> avatar = Avatar(world_size=(128, 128))
        >>> avatar.move('up', world_size=(128, 128))
        >>> richness = avatar.interact(terrain_region)
    """

    def __init__(self, world_size: Tuple[int, int]):
        """
        Initialize avatar at world center with full energy.

        Args:
            world_size: Dimensions of world (width, height)
        """
        self.pos = [world_size[0] // 2, world_size[1] // 2]
        self.energy = MAX_ENERGY
        # Use deque with maxlen for memory-efficient trajectory storage
        self.trajectory: deque = deque([tuple(self.pos)], maxlen=MAX_TRAJECTORY_LENGTH)
        self.interactions: List[float] = []
        self._unique_cells: set = {(self.pos[0], self.pos[1])}

    def move(
        self,
        direction: str,
        world_size: Tuple[int, int]
    ) -> bool:
        """
        Move avatar in specified direction.

        Movement costs energy per step. Position is clamped to
        world boundaries.

        Args:
            direction: One of 'up', 'down', 'left', 'right'
            world_size: World dimensions for boundary checking

        Returns:
            bool: True if move was successful

        Raises:
            ValueError: If direction is invalid
        """
        if direction not in ['up', 'down', 'left', 'right']:
            raise ValueError(f"Invalid direction: {direction}")

        # Update position based on direction
        if direction == 'up':
            self.pos[0] = max(0, self.pos[0] - 1)
        elif direction == 'down':
            self.pos[0] = min(world_size[0] - 1, self.pos[0] + 1)
        elif direction == 'left':
            self.pos[1] = max(0, self.pos[1] - 1)
        elif direction == 'right':
            self.pos[1] = min(world_size[1] - 1, self.pos[1] + 1)

        # Energy cost for movement
        self.energy = max(MIN_ENERGY, self.energy - MOVEMENT_COST)

        # Record trajectory
        current_pos = tuple(self.pos)
        self.trajectory.append(current_pos)
        self._unique_cells.add(current_pos)

        return True

    def interact(self, region: np.ndarray) -> float:
        """
        Interact with terrain region and update energy.

        Energy dynamics:
        - Rich terrain (mean > threshold): gain energy
        - Poor terrain (mean ≤ threshold): lose energy
        Energy is clamped to [0, 100].

        Args:
            region: Terrain values from world.sample_region()

        Returns:
            float: Mean richness of region [0, 1]
        """
        richness = float(np.mean(region))
        self.interactions.append(richness)

        # Energy dynamics based on terrain richness
        if richness > RICH_TERRAIN_THRESHOLD:
            self.energy = min(MAX_ENERGY, self.energy + RICH_TERRAIN_GAIN)
        else:
            self.energy = max(MIN_ENERGY, self.energy - POOR_TERRAIN_LOSS)

        return richness

    def get_trajectory_length(self) -> float:
        """
        Calculate total Euclidean distance traveled.

        Returns:
            float: Total path length
        """
        if len(self.trajectory) < 2:
            return 0.0

        total_distance = 0.0
        trajectory_list = list(self.trajectory)
        for i in range(1, len(trajectory_list)):
            prev = np.array(trajectory_list[i-1])
            curr = np.array(trajectory_list[i])
            total_distance += np.linalg.norm(curr - prev)

        return total_distance

    def get_exploration_coverage(
        self,
        world_size: Tuple[int, int],
        grid_size: int = 10
    ) -> float:
        """
        Calculate what fraction of world has been visited.

        Divides world into grid and counts unique cells visited.

        Args:
            world_size: World dimensions
            grid_size: Size of grid cells for coverage calculation

        Returns:
            float: Coverage fraction [0, 1]
        """
        if not self.trajectory:
            return 0.0

        # Convert positions to grid cells
        visited_cells = set()
        for x, y in self.trajectory:
            cell_x = x // grid_size
            cell_y = y // grid_size
            visited_cells.add((cell_x, cell_y))

        # Calculate total possible cells
        total_cells_x = (world_size[0] + grid_size - 1) // grid_size
        total_cells_y = (world_size[1] + grid_size - 1) // grid_size
        total_cells = total_cells_x * total_cells_y

        return len(visited_cells) / total_cells

    def reset(self, world_size: Tuple[int, int]) -> None:
        """
        Reset avatar to initial state.

        Args:
            world_size: World dimensions for centering
        """
        self.pos = [world_size[0] // 2, world_size[1] // 2]
        self.energy = MAX_ENERGY
        self.trajectory = deque([tuple(self.pos)], maxlen=MAX_TRAJECTORY_LENGTH)
        self.interactions.clear()
        self._unique_cells = {(self.pos[0], self.pos[1])}

    def __repr__(self) -> str:
        return (f"Avatar(pos={self.pos}, energy={self.energy:.1f}, "
                f"steps={len(self.trajectory)})")
