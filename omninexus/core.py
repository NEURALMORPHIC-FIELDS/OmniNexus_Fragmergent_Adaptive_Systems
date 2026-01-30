"""
OmniNexus - Core orchestrator integrating all fragmergent system components.

The main system that coordinates oscillator, world, avatar, and RL agent
to create an adaptive hybrid architecture with intelligent mode switching.

Author: Vasile Lucian Borbeleac
Date: January 2025
Version: 10.0.0
"""

import numpy as np
from typing import Dict, Tuple, List

from omninexus.components.oscillator import FragmergentOscillator
from omninexus.components.optical_world import OpticalWorld
from omninexus.components.avatar import Avatar
from omninexus.components.rl_agent import RLAgent


# Constants for reward calculation
ENERGY_WEIGHT = 0.5
MODE_WEIGHT = 0.3
RICHNESS_WEIGHT = 0.2

# Mode switching constants
PERIODIC_REGEN_INTERVAL = 20
SMART_NAV_PROBABILITY = 0.7


class OmniNexus:
    """
    Fragmergent adaptive system orchestrator.

    Integrates all components into a cohesive system that exhibits
    emergent adaptive behavior through the interaction of:
    - Oscillator-driven phase dynamics
    - Procedural world generation
    - Energy-constrained navigation
    - Reinforcement learning

    Attributes:
        oscillator (FragmergentOscillator): Phase generator
        world (OpticalWorld): Procedural environment
        avatar (Avatar): Agent navigating world
        agent (RLAgent): Learning controller
        world_size (Tuple[int, int]): World dimensions
        steps (int): Total simulation steps
        mode_history (List[str]): History of mode selections

    Example:
        >>> nexus = OmniNexus(world_size=(128, 128))
        >>> for _ in range(100):
        ...     state = nexus.run_cycle(smart_navigation=True)
        >>> stats = nexus.get_statistics()
    """

    def __init__(
        self,
        world_size: Tuple[int, int] = (128, 128),
        base_freq: float = 0.1,
        harmonic_layers: int = 3,
        noise: float = 0.05,
        learning_rate: float = 0.015,
        world_complexity: float = 0.001
    ):
        """
        Initialize OmniNexus system with all components.

        Args:
            world_size: Dimensions of procedural world
            base_freq: Oscillator base frequency
            harmonic_layers: Number of harmonic overtones
            noise: Oscillator noise magnitude
            learning_rate: RL agent learning rate
            world_complexity: World generation smoothness parameter
        """
        self.world_size = world_size

        # Initialize components
        self.oscillator = FragmergentOscillator(
            base_freq=base_freq,
            harmonic_layers=harmonic_layers,
            noise=noise
        )

        self.world = OpticalWorld(
            size=world_size,
            complexity=world_complexity
        )

        self.avatar = Avatar(world_size=world_size)

        self.agent = RLAgent(learning_rate=learning_rate)

        # Generate initial world
        self.world.generate()

        # Tracking
        self.steps = 0
        self.mode_history: List[str] = []

    def run_cycle(
        self,
        smart_navigation: bool = True
    ) -> Dict:
        """
        Execute one complete system cycle.

        Cycle phases:
        1. Oscillator generates phase φ and calculates coherence
        2. RL agent decides on mode (optical vs digital)
        3. If optical: regenerate world completely
        4. Avatar moves (smart or random navigation)
        5. Avatar interacts with terrain
        6. Calculate reward and update RL policy

        Args:
            smart_navigation: If True, avatar seeks rich terrain

        Returns:
            dict: Complete system state with keys:
                - step: current step number
                - mode: 'optical' or 'digital'
                - phi: oscillator phase
                - coherence: phase coherence
                - policy: RL policy value
                - energy: avatar energy
                - richness: terrain richness at avatar position
                - reward: calculated reward signal
                - pos: avatar position (x, y)
        """
        # Phase 1: Generate oscillator phase
        phi = self.oscillator.step()
        coherence = self.oscillator.get_phase_coherence()

        # Phase 2: RL decides on mode switching
        recalc = self.agent.decide(phi, coherence)

        # Phase 3: Mode determination and world update
        if recalc or self.steps % PERIODIC_REGEN_INTERVAL == 0:
            self.world.generate()
            mode = 'optical'
        else:
            mode = 'digital'

        self.mode_history.append(mode)

        # Phase 4: Avatar navigation
        if smart_navigation and np.random.rand() < SMART_NAV_PROBABILITY:
            direction = self._smart_move()
        else:
            direction = np.random.choice(['up', 'down', 'left', 'right'])

        self.avatar.move(direction, self.world_size)

        # Phase 5: Avatar-world interaction
        region = self.world.sample_region(self.avatar.pos)
        richness = self.avatar.interact(region)

        # Phase 6: Reward calculation and learning
        reward = self._calculate_reward(mode, richness)
        self.agent.update(reward, phi)

        self.steps += 1

        return {
            'step': self.steps,
            'mode': mode,
            'phi': round(phi, 3),
            'coherence': round(coherence, 3),
            'policy': round(self.agent.policy, 3),
            'energy': round(self.avatar.energy, 2),
            'richness': round(richness, 3),
            'reward': round(reward, 3),
            'pos': tuple(self.avatar.pos)
        }

    def _smart_move(self) -> str:
        """
        Calculate best movement direction based on terrain richness.

        Samples all adjacent cells and moves toward richest region.

        Returns:
            str: Best direction ('up', 'down', 'left', 'right')
        """
        x, y = self.avatar.pos

        # Define potential moves
        moves = {
            'up': (max(0, x - 1), y),
            'down': (min(self.world_size[0] - 1, x + 1), y),
            'left': (x, max(0, y - 1)),
            'right': (x, min(self.world_size[1] - 1, y + 1))
        }

        # Evaluate richness in each direction
        richness = {}
        for direction, pos in moves.items():
            region = self.world.sample_region(pos)
            richness[direction] = np.mean(region)

        # Return direction with highest richness
        return max(richness, key=richness.get)

    def _calculate_reward(
        self,
        mode: str,
        richness: float
    ) -> float:
        """
        Calculate multi-component reward signal.

        Reward components:
        - 50%: Energy level (survival)
        - 30%: Mode bonus (optical exploration rewarded)
        - 20%: Terrain richness (resource quality)

        Args:
            mode: Current mode ('optical' or 'digital')
            richness: Terrain richness [0, 1]

        Returns:
            float: Reward signal [0, 1]
        """
        energy_component = ENERGY_WEIGHT * (self.avatar.energy / 100.0)
        mode_component = MODE_WEIGHT * (1.0 if mode == 'optical' else 0.0)
        richness_component = RICHNESS_WEIGHT * richness

        return energy_component + mode_component + richness_component

    def get_statistics(self) -> Dict:
        """
        Get comprehensive system statistics.

        Returns:
            dict: System-wide metrics including component statistics
        """
        mode_counts = {
            'optical': self.mode_history.count('optical'),
            'digital': self.mode_history.count('digital')
        }

        return {
            'steps': self.steps,
            'mode_distribution': mode_counts,
            'optical_frequency': mode_counts['optical'] / max(1, self.steps),
            'oscillator': self.oscillator.get_statistics(),
            'agent': self.agent.get_performance_metrics(),
            'avatar_energy': self.avatar.energy,
            'world_generations': self.world.generation_count,
            'trajectory_length': self.avatar.get_trajectory_length(),
            'exploration_coverage': self.avatar.get_exploration_coverage(
                self.world_size
            )
        }

    def reset(self) -> None:
        """Reset all components to initial state."""
        self.oscillator.reset()
        self.world.reset()
        self.world.generate()
        self.avatar.reset(self.world_size)
        self.agent.reset()
        self.steps = 0
        self.mode_history.clear()

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"OmniNexus(steps={self.steps}, "
                f"optical_freq={stats['optical_frequency']:.2f}, "
                f"energy={self.avatar.energy:.1f})")
