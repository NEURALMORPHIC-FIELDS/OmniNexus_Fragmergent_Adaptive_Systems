"""
Optical World - FFT-based procedural world generation with adaptive complexity.

Generates smooth, continuous terrain using Fast Fourier Transform with
exponential filtering. Supports dynamic regeneration based on agent decisions.

Author: Lucian Coman
Date: January 2025
Version: 10.0.0
"""

import numpy as np
import scipy.fft as fft
from typing import Tuple, Optional


class OpticalWorld:
    """
    Procedural world generator using FFT-based synthesis.
    
    Creates smooth, natural-looking terrain by filtering random noise
    in frequency domain. The complexity parameter controls the rate
    of frequency decay, affecting terrain smoothness.
    
    Attributes:
        size (Tuple[int, int]): World dimensions (width, height)
        complexity (float): Frequency filter decay rate
        state (np.ndarray): Current world state [0, 1]
        generation_count (int): Number of regenerations performed
        
    Example:
        >>> world = OpticalWorld(size=(128, 128), complexity=0.001)
        >>> world.generate(seed=42)
        >>> region = world.sample_region(pos=(64, 64), size=5)
    """
    
    def __init__(
        self,
        size: Tuple[int, int] = (128, 128),
        complexity: float = 0.001
    ):
        """
        Initialize optical world generator.
        
        Args:
            size: World dimensions (width, height)
            complexity: Frequency decay rate. Lower = smoother terrain
                       Typical range: 0.0001 (very smooth) to 0.01 (rough)
        """
        self.size = size
        self.complexity = complexity
        self.state = np.zeros(size, dtype=np.float32)
        self.generation_count = 0
        
    def generate(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate new world state using FFT-based synthesis.
        
        Process:
        1. Generate random noise in spatial domain
        2. Transform to frequency domain (FFT)
        3. Apply exponential decay filter
        4. Transform back to spatial domain (IFFT)
        5. Normalize to [0, 1]
        
        Args:
            seed: Random seed for reproducibility (optional)
            
        Returns:
            np.ndarray: Generated world state, shape=size, values in [0, 1]
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random noise
        noise = np.random.rand(*self.size).astype(np.float32)
        
        # Transform to frequency domain
        world_fft = fft.fft2(noise)
        
        # Create exponential decay filter
        x = np.arange(self.size[0])[:, None]
        y = np.arange(self.size[1])[None, :]
        dist_sq = x**2 + y**2
        filter_kernel = np.exp(-self.complexity * dist_sq)
        
        # Apply filter in frequency domain
        filtered_fft = world_fft * filter_kernel
        
        # Transform back to spatial domain
        world = np.real(fft.ifft2(filtered_fft))
        
        # Normalize to [0, 1]
        world_min = world.min()
        world_max = world.max()
        if world_max > world_min:
            self.state = (world - world_min) / (world_max - world_min)
        else:
            self.state = np.zeros_like(world)
        
        self.generation_count += 1
        return self.state
    
    def sample_region(
        self,
        pos: Tuple[int, int],
        size: int = 5
    ) -> np.ndarray:
        """
        Extract local region around position.
        
        Args:
            pos: Center position (x, y)
            size: Half-width of region to extract
            
        Returns:
            np.ndarray: Region of world, shape varies based on boundaries
        """
        x, y = pos
        
        # Clamp to valid boundaries
        x0 = max(0, x - size)
        y0 = max(0, y - size)
        x1 = min(self.size[0], x + size + 1)
        y1 = min(self.size[1], y + size + 1)
        
        return self.state[x0:x1, y0:y1]
    
    def get_terrain_entropy(self) -> float:
        """
        Calculate Shannon entropy of terrain distribution.
        
        Higher entropy indicates more complex, varied terrain.
        Lower entropy indicates uniform, simple terrain.
        
        Returns:
            float: Entropy value (nats)
        """
        # Create histogram
        hist, _ = np.histogram(self.state.flatten(), bins=50, density=True)
        
        # Normalize to probability distribution
        hist = hist / (hist.sum() + 1e-10)
        
        # Calculate entropy (filter out zeros)
        nonzero = hist[hist > 0]
        entropy = -np.sum(nonzero * np.log(nonzero + 1e-10))
        
        return float(entropy)
    
    def get_richness_at(self, pos: Tuple[int, int]) -> float:
        """
        Get terrain richness at specific position.
        
        Args:
            pos: Position (x, y)
            
        Returns:
            float: Richness value [0, 1]
        """
        x, y = pos
        if 0 <= x < self.size[0] and 0 <= y < self.size[1]:
            return float(self.state[x, y])
        return 0.0
    
    def reset(self):
        """Reset world to empty state."""
        self.state = np.zeros(self.size, dtype=np.float32)
        self.generation_count = 0
    
    def __repr__(self) -> str:
        return (f"OpticalWorld(size={self.size}, "
                f"complexity={self.complexity}, "
                f"generations={self.generation_count})")