"""
OmniNexus Components - Core building blocks of the fragmergent system.

Components:
    - FragmergentOscillator: Multi-harmonic phase generator
    - OpticalWorld: FFT-based procedural terrain
    - Avatar: Energy-constrained navigation agent
    - RLAgent: Phase-coupled reinforcement learning
"""

from omninexus.components.oscillator import FragmergentOscillator
from omninexus.components.optical_world import OpticalWorld
from omninexus.components.avatar import Avatar
from omninexus.components.rl_agent import RLAgent

__all__ = [
    "FragmergentOscillator",
    "OpticalWorld",
    "Avatar",
    "RLAgent",
]
