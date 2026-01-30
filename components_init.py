"""
OmniNexus Components Package

Individual system components that can be used independently or
integrated through the OmniNexus orchestrator.

Components:
    - FragmergentOscillator: Multi-harmonic phase generator
    - OpticalWorld: FFT-based procedural world
    - Avatar: Energy-constrained agent
    - RLAgent: Phase-coupled learning

Author: Vasile Lucian Borbeleac
Version: 10.0.0
"""

from .oscillator import FragmergentOscillator
from .optical_world import OpticalWorld
from .avatar import Avatar
from .rl_agent import RLAgent

__all__ = [
    'FragmergentOscillator',
    'OpticalWorld',
    'Avatar',
    'RLAgent'
]