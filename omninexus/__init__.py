"""
OmniNexus - Fragmergent Adaptive Systems

A research platform for exploring edge-of-chaos computation through
phase-coupled learning and intelligent mode switching.

Author: Vasile Lucian Borbeleac
Version: 10.0.0
License: Apache 2.0
"""

from omninexus.core import OmniNexus
from omninexus.components.oscillator import FragmergentOscillator
from omninexus.components.optical_world import OpticalWorld
from omninexus.components.avatar import Avatar
from omninexus.components.rl_agent import RLAgent

__version__ = "10.0.0"
__author__ = "Vasile Lucian Borbeleac"
__all__ = [
    "OmniNexus",
    "FragmergentOscillator",
    "OpticalWorld",
    "Avatar",
    "RLAgent",
]
