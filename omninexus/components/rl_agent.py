"""
RL Agent - Phase-coupled reinforcement learning with adaptive policy.

Implements policy gradient learning where update magnitude is modulated
by the oscillator phase, creating biologically-inspired rhythmic learning.

Author: Vasile Lucian Borbeleac
Date: January 2025
Version: 10.0.0
"""

import numpy as np
from typing import List, Dict
from collections import deque


# Constants
REWARD_BASELINE = 0.5
PHASE_BONUS_SCALE = 0.3
POLICY_MIN = 0.0
POLICY_MAX = 1.0
REWARD_WINDOW = 20


class RLAgent:
    """
    Reinforcement learning agent with phase-coupled policy updates.

    Learns optimal switching policy between optical and digital modes
    by correlating rewards with oscillator phase. Update timing matters -
    learning is synchronized with phase dynamics.

    Attributes:
        policy (float): Current policy threshold [0, 1]
        learning_rate (float): Base learning rate
        exploration (float): Exploration bonus (reserved for future use)
        reward_history (deque): Recent reward history (windowed)

    Example:
        >>> agent = RLAgent(learning_rate=0.015)
        >>> should_switch = agent.decide(phi=0.5, coherence=0.7)
        >>> agent.update(reward=0.8, phi=0.5)
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        exploration: float = 0.1
    ):
        """
        Initialize RL agent with policy at neutral value.

        Args:
            learning_rate: Base learning rate α. Typical: 0.001-0.05
            exploration: Exploration parameter (reserved for future use)
        """
        self.policy = 0.5  # Start at neutral
        self.learning_rate = learning_rate
        self.exploration = exploration

        # Use deque for efficient windowed statistics
        self._recent_rewards: deque = deque(maxlen=REWARD_WINDOW)
        self.reward_history: List[float] = []

        # Running statistics for adaptive learning rate
        self._reward_sum = 0.0
        self._reward_sq_sum = 0.0
        self._reward_count = 0

    def decide(
        self,
        phi: float,
        coherence: float = 0.5
    ) -> bool:
        """
        Decide whether to trigger optical mode regeneration.

        Decision is probabilistic based on:
        - Current policy (learned threshold)
        - Oscillator phase (higher φ → higher probability)
        - Coherence (scales the phase influence)

        Formula: P(optical) = policy + 0.3·max(0, φ)·coherence

        Args:
            phi: Current oscillator phase
            coherence: Phase coherence metric [0, 1]

        Returns:
            bool: True = switch to optical mode, False = stay digital
        """
        # Calculate decision threshold
        phase_bonus = PHASE_BONUS_SCALE * max(0.0, phi) * coherence
        threshold = self.policy + phase_bonus

        # Stochastic decision
        return np.random.rand() < threshold

    def update(
        self,
        reward: float,
        phi: float
    ) -> None:
        """
        Update policy using phase-coupled gradient.

        The key innovation: learning magnitude is modulated by φ(t).
        This creates "windows" of high learning when phase is large,
        mimicking biological learning rhythms.

        Formula: Δpolicy = α·(reward - baseline)·φ

        Args:
            reward: Observed reward signal [0, 1]
            phi: Oscillator phase at decision time
        """
        # Update reward history
        self.reward_history.append(reward)
        self._recent_rewards.append(reward)

        # Update running statistics
        self._reward_sum += reward
        self._reward_sq_sum += reward * reward
        self._reward_count += 1

        # Adaptive learning rate based on recent reward variance
        if len(self._recent_rewards) >= REWARD_WINDOW:
            recent_arr = np.array(self._recent_rewards)
            recent_std = np.std(recent_arr)
            adaptive_lr = self.learning_rate * (1 + recent_std)
        else:
            adaptive_lr = self.learning_rate

        # Phase-coupled policy gradient
        gradient = (reward - REWARD_BASELINE) * phi
        self.policy += adaptive_lr * gradient

        # Clamp policy to valid range
        self.policy = np.clip(self.policy, POLICY_MIN, POLICY_MAX)

    def get_performance_metrics(self) -> Dict:
        """
        Calculate performance statistics.

        Returns:
            dict: Metrics including mean reward, cumulative reward, etc.
        """
        if not self.reward_history:
            return {
                'mean_reward': 0.0,
                'total_reward': 0.0,
                'reward_std': 0.0,
                'steps': 0
            }

        # Use running statistics for efficiency
        n = self._reward_count
        mean_reward = self._reward_sum / n
        variance = (self._reward_sq_sum / n) - (mean_reward ** 2)
        std_reward = np.sqrt(max(0, variance))

        return {
            'mean_reward': float(mean_reward),
            'total_reward': float(self._reward_sum),
            'reward_std': float(std_reward),
            'steps': len(self.reward_history)
        }

    def reset(self) -> None:
        """Reset agent to initial state."""
        self.policy = 0.5
        self.reward_history.clear()
        self._recent_rewards.clear()
        self._reward_sum = 0.0
        self._reward_sq_sum = 0.0
        self._reward_count = 0

    def __repr__(self) -> str:
        perf = self.get_performance_metrics()
        return (f"RLAgent(policy={self.policy:.3f}, "
                f"lr={self.learning_rate}, "
                f"mean_reward={perf['mean_reward']:.3f})")
