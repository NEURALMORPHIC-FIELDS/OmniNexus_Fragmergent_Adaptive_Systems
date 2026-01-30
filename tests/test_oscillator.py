"""
Tests for FragmergentOscillator component.
"""

import pytest
import numpy as np
from omninexus.components.oscillator import FragmergentOscillator


class TestFragmergentOscillator:
    """Test suite for FragmergentOscillator."""

    def test_initialization_default(self):
        """Test default initialization."""
        osc = FragmergentOscillator()
        assert osc.base_freq == 1.0
        assert osc.noise == 0.05
        assert osc.harmonic_layers == 3
        assert osc.t == 0
        assert len(osc.history) == 0

    def test_initialization_custom(self):
        """Test custom initialization."""
        osc = FragmergentOscillator(
            base_freq=0.1,
            noise=0.02,
            harmonic_layers=5
        )
        assert osc.base_freq == 0.1
        assert osc.noise == 0.02
        assert osc.harmonic_layers == 5

    def test_harmonic_layers_validation(self):
        """Test that invalid harmonic_layers raises ValueError."""
        with pytest.raises(ValueError):
            FragmergentOscillator(harmonic_layers=0)

        with pytest.raises(ValueError):
            FragmergentOscillator(harmonic_layers=8)

    def test_step_returns_float(self):
        """Test that step() returns a float."""
        osc = FragmergentOscillator()
        phi = osc.step()
        assert isinstance(phi, float)

    def test_step_increments_time(self):
        """Test that step() increments time."""
        osc = FragmergentOscillator()
        assert osc.t == 0
        osc.step()
        assert osc.t == 1
        osc.step()
        assert osc.t == 2

    def test_step_updates_history(self):
        """Test that step() updates history."""
        osc = FragmergentOscillator()
        assert len(osc.history) == 0

        phi1 = osc.step()
        assert len(osc.history) == 1
        assert osc.history[0] == phi1

        phi2 = osc.step()
        assert len(osc.history) == 2
        assert osc.history[1] == phi2

    def test_coherence_range(self):
        """Test that coherence is always in [0, 1]."""
        osc = FragmergentOscillator(noise=0.1)
        for _ in range(100):
            osc.step()

        coherence = osc.get_phase_coherence()
        assert 0.0 <= coherence <= 1.0

    def test_coherence_warmup(self):
        """Test coherence during warmup period."""
        osc = FragmergentOscillator()
        osc.step()
        coherence = osc.get_phase_coherence(window=20)
        assert coherence == 0.5  # Neutral during warmup

    def test_reset(self):
        """Test reset functionality."""
        osc = FragmergentOscillator()
        for _ in range(50):
            osc.step()

        assert osc.t > 0
        assert len(osc.history) > 0

        osc.reset()
        assert osc.t == 0
        assert len(osc.history) == 0

    def test_statistics_empty(self):
        """Test statistics with empty history."""
        osc = FragmergentOscillator()
        stats = osc.get_statistics()
        assert stats['steps'] == 0
        assert stats['coherence'] == 0.5

    def test_statistics_populated(self):
        """Test statistics with data."""
        osc = FragmergentOscillator(noise=0.01)
        for _ in range(50):
            osc.step()

        stats = osc.get_statistics()
        assert stats['steps'] == 50
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'range' in stats
        assert 'coherence' in stats

    def test_deterministic_without_noise(self):
        """Test that oscillator is deterministic without noise."""
        np.random.seed(42)
        osc1 = FragmergentOscillator(noise=0.0)
        values1 = [osc1.step() for _ in range(10)]

        np.random.seed(42)
        osc2 = FragmergentOscillator(noise=0.0)
        values2 = [osc2.step() for _ in range(10)]

        np.testing.assert_array_almost_equal(values1, values2)

    def test_repr(self):
        """Test string representation."""
        osc = FragmergentOscillator(base_freq=0.2, harmonic_layers=4, noise=0.1)
        repr_str = repr(osc)
        assert 'FragmergentOscillator' in repr_str
        assert 'base_freq=0.2' in repr_str
        assert 'harmonic_layers=4' in repr_str
