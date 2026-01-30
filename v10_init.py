"""
OmniNexus v10 - Research-Grade Fragmergent System

A comprehensive platform for exploring oscillator-driven hybrid architectures
with coherence-based mode switching and phase-coupled reinforcement learning.

Modules:
    - components.oscillator: Multi-harmonic phase generator
    - components.optical_world: FFT-based procedural generation
    - components.avatar: Energy-constrained navigation agent
    - components.rl_agent: Phase-coupled policy learning
    - omninexus: Core system orchestrator
    - analysis: Statistical and spectral analysis tools

Example:
    >>> from omninexus import OmniNexus
    >>> nexus = OmniNexus(world_size=(128, 128))
    >>> for _ in range(100):
    ...     state = nexus.run_cycle()
    >>> stats = nexus.get_statistics()

Author: Lucian Coman
Date: January 2025
Version: 10.0.0
License: MIT
"""

__version__ = '10.0.0'
__author__ = 'Lucian Coman'

from .omninexus import OmniNexus
from .components.oscillator import FragmergentOscillator
from .components.optical_world import OpticalWorld
from .components.avatar import Avatar
from .components.rl_agent import RLAgent
from .analysis import (
    analyze_run,
    analyze_phase_space,
    compute_fourier_spectrum,
    find_dominant_frequencies,
    compare_agents
)

__all__ = [
    'OmniNexus',
    'FragmergentOscillator',
    'OpticalWorld',
    'Avatar',
    'RLAgent',
    'analyze_run',
    'analyze_phase_space',
    'compute_fourier_spectrum',
    'find_dominant_frequencies',
    'compare_agents'
]