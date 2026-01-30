"""
RL Agent - Phase-coupled reinforcement learning with adaptive policy.

Implements policy gradient learning where update magnitude is modulated
by the oscillator phase, creating biologically-inspired rhythmic learning.

Author: Vasile Lucian Borbeleac
Date: January 2025
Version: 10.0.0
"""

import numpy as np
from typing import List, Optional


class RLAgent:
    """
    Reinforcement learning agent with phase-coupled policy updates.
    
    Learns optimal switching policy between optical and digital modes
    by correlating rewards with oscillator phase. Update timing matters -
    learning is synchronized with phase dynamics.
    
    Attributes:
        policy (float): Current policy threshold [0, 1]
        learning_rate (float): Base learning rate
        exploration (float): Exploration bonus (not currently used)
        reward_history (List[float]): Complete reward history
        
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
        self.reward_history: List[float] = []
        
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
        phase_bonus = 0.3 * max(0.0, phi) * coherence
        threshold = self.policy + phase_bonus
        
        # Stochastic decision
        return np.random.rand() < threshold
    
    def update(
        self,
        reward: float,
        phi: float
    ):
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
        self.reward_history.append(reward)
        
        # Adaptive learning rate based on recent reward variance
        if len(self.reward_history) > 20:
            recent_std = np.std(self.reward_history[-20:])
            adaptive_lr = self.learning_rate * (1 + recent_std)
        else:
            adaptive_lr = self.learning_rate
        
        # Phase-coupled policy gradient
        # Baseline = 0.5 (neutral reward expectation)
        gradient = (reward - 0.5) * phi
        self.policy += adaptive_lr * gradient
        
        # Clamp policy to valid range
        self.policy = np.clip(self.policy, 0.0, 1.0)
    
    def get_performance_metrics(self) -> dict:
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
        
        arr = np.array(self.reward_history)
        return {
            'mean_reward': float(np.mean(arr)),
            'total_reward': float(np.sum(arr)),
            'reward_std': float(np.std(arr)),
            'steps': len(self.reward_history)
        }
    
    def reset(self):
        """Reset agent to initial state."""
        self.policy = 0.5
        self.reward_history.clear()
    
    def __repr__(self) -> str:
        perf = self.get_performance_metrics()
        return (f"RLAgent(policy={self.policy:.3f}, "
                f"lr={self.learning_rate}, "
                f"mean_reward={perf['mean_reward']:.3f})")