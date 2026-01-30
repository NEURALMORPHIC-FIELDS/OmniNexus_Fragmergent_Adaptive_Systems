"""
Fragmergent Oscillator - Multi-harmonic phase generator with coherence tracking.

This module implements the core oscillator that drives the fragmergent system.
The oscillator generates a phase signal φ(t) using multiple harmonics and
tracks temporal coherence to inform system decisions.

Author: Vasile Lucian Borbeleac
Date: January 2025
Version: 10.0.0
"""

import numpy as np
from typing import List, Optional


class FragmergentOscillator:
    """
    Multi-harmonic oscillator with coherence tracking.
    
    The oscillator combines multiple harmonic frequencies with additive noise
    to create a rich phase signal that exhibits both periodic and stochastic
    behavior - the fragmergent regime.
    
    Attributes:
        base_freq (float): Base oscillation frequency
        noise (float): Magnitude of additive Gaussian noise
        harmonic_layers (int): Number of harmonic overtones (1-7)
        t (int): Current time step
        history (List[float]): Complete phase history
        
    Example:
        >>> osc = FragmergentOscillator(base_freq=0.1, harmonic_layers=3)
        >>> phi = osc.step()
        >>> coherence = osc.get_phase_coherence(window=20)
    """
    
    def __init__(
        self,
        base_freq: float = 1.0,
        noise: float = 0.05,
        harmonic_layers: int = 3
    ):
        """
        Initialize the fragmergent oscillator.
        
        Args:
            base_freq: Fundamental frequency (Hz). Typical range: 0.01-0.5
            noise: Gaussian noise standard deviation. Typical range: 0.0-0.2
            harmonic_layers: Number of harmonics to include (1-7)
            
        Raises:
            ValueError: If harmonic_layers not in range [1, 7]
        """
        if not 1 <= harmonic_layers <= 7:
            raise ValueError("harmonic_layers must be between 1 and 7")
            
        self.base_freq = base_freq
        self.noise = noise
        self.harmonic_layers = harmonic_layers
        self.t = 0
        self.history: List[float] = []
        
    def step(self) -> float:
        """
        Advance oscillator by one time step and return phase.
        
        Computes φ(t) = Σ(h=1 to n) [sin(t·f·h) / h] + ε
        where f is base frequency, h is harmonic number, and ε is noise.
        
        Returns:
            float: Current phase value φ(t)
        """
        self.t += 1
        
        # Multi-harmonic synthesis
        phi = 0.0
        for h in range(1, self.harmonic_layers + 1):
            phi += np.sin(self.t * self.base_freq * h) / h
        
        # Add Gaussian noise for fragmergent behavior
        phi += np.random.normal(0, self.noise)
        
        self.history.append(phi)
        return phi
    
    def get_phase_coherence(self, window: int = 20) -> float:
        """
        Calculate temporal coherence of recent phase history.
        
        Coherence measures the predictability/stability of the oscillator.
        High coherence (→1.0) indicates regular, predictable behavior.
        Low coherence (→0.0) indicates chaotic, unpredictable behavior.
        
        Computed as: coherence = 1 - min(std(φ_recent), 1.0)
        
        Args:
            window: Number of recent steps to analyze
            
        Returns:
            float: Coherence metric in range [0, 1]
        """
        if len(self.history) < window:
            return 0.5  # Neutral coherence during warmup
        
        recent = self.history[-window:]
        std = np.std(recent)
        coherence = 1.0 - min(std, 1.0)
        
        return max(0.0, min(1.0, coherence))
    
    def reset(self):
        """Reset oscillator to initial state."""
        self.t = 0
        self.history.clear()
        
    def get_statistics(self) -> dict:
        """
        Get comprehensive statistics about oscillator behavior.
        
        Returns:
            dict: Statistics including mean, std, range, coherence
        """
        if not self.history:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'range': 0.0,
                'coherence': 0.5,
                'steps': 0
            }
        
        arr = np.array(self.history)
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'range': float(np.ptp(arr)),
            'coherence': self.get_phase_coherence(),
            'steps': len(self.history)
        }
    
    def __repr__(self) -> str:
        return (f"FragmergentOscillator(base_freq={self.base_freq}, "
                f"harmonic_layers={self.harmonic_layers}, "
                f"noise={self.noise}, t={self.t})")