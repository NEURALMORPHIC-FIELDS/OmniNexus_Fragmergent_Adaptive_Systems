"""
Analysis Tools - Comprehensive analytics for fragmergent systems.

Provides functions for phase space analysis, Fourier spectrum computation,
attractor detection, and comparative statistics.

Author: Lucian Coman
Date: January 2025
Version: 10.0.0
"""

import numpy as np
from typing import List, Dict, Tuple


def analyze_run(history: List[Dict]) -> Dict:
    """
    Comprehensive analysis of simulation run.

    Args:
        history: List of state dictionaries from run_cycle()

    Returns:
        dict: Statistical analysis with keys:
            - steps, mode counts, frequencies
            - oscillator statistics (mean, std)
            - policy statistics
            - energy statistics
            - correlations between variables
    """
    if not history:
        return {'error': 'Empty history'}

    # Extract time series
    modes = [h['mode'] for h in history]
    phis = np.array([h['phi'] for h in history])
    coherences = np.array([h.get('coherence', 0.5) for h in history])
    policies = np.array([h['policy'] for h in history])
    energies = np.array([h['energy'] for h in history])
    rewards = np.array([h.get('reward', 0.0) for h in history])

    total = len(history)
    optical = sum(1 for m in modes if m == 'optical')
    digital = total - optical

    stats = {
        'steps': total,
        'optical_count': optical,
        'digital_count': digital,
        'optical_freq': optical / total if total else 0.0,

        # Oscillator metrics
        'avg_phi': float(np.mean(phis)),
        'std_phi': float(np.std(phis)),
        'min_phi': float(np.min(phis)),
        'max_phi': float(np.max(phis)),

        # Coherence metrics
        'avg_coherence': float(np.mean(coherences)),
        'std_coherence': float(np.std(coherences)),

        # Policy metrics
        'avg_policy': float(np.mean(policies)),
        'std_policy': float(np.std(policies)),
        'final_policy': float(policies[-1]),
        'policy_range': float(np.ptp(policies)),

        # Energy metrics
        'avg_energy': float(np.mean(energies)),
        'min_energy': float(np.min(energies)),
        'max_energy': float(np.max(energies)),

        # Reward metrics
        'avg_reward': float(np.mean(rewards)),
        'total_reward': float(np.sum(rewards)),
        'std_reward': float(np.std(rewards))
    }

    # Correlations
    mode_num = np.array([1 if m == 'optical' else 0 for m in modes])
    stats['corr_phi_policy'] = safe_corr(phis, policies)
    stats['corr_phi_energy'] = safe_corr(phis, energies)
    stats['corr_coherence_reward'] = safe_corr(coherences, rewards)
    stats['corr_policy_mode'] = safe_corr(policies, mode_num)

    return stats


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate correlation with NaN protection.

    Args:
        a, b: Arrays to correlate

    Returns:
        float: Pearson correlation coefficient
    """
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    corr = np.corrcoef(a, b)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def analyze_phase_space(
    phi_history: List[float],
    policy_history: List[float],
    window: int = 50
) -> Dict:
    """
    Analyze phase space trajectory for attractors and limit cycles.

    Args:
        phi_history: Oscillator phase values
        policy_history: Policy values
        window: Window size for convergence check

    Returns:
        dict: Analysis results with keys:
            - convergence_detected: bool
            - attractor_type: 'fixed_point', 'limit_cycle', or 'chaotic'
            - final_state: (phi_mean, policy_mean)
            - stability: measure of trajectory stability
    """
    if len(phi_history) < window:
        return {'error': 'Insufficient data'}

    # Check convergence in last window
    recent_phi = np.array(phi_history[-window:])
    recent_policy = np.array(policy_history[-window:])

    phi_std = np.std(recent_phi)
    policy_std = np.std(recent_policy)

    # Determine attractor type
    if policy_std < 0.01:
        attractor_type = 'fixed_point'
        convergence_detected = True
    elif policy_std < 0.05:
        attractor_type = 'limit_cycle'
        convergence_detected = True
    else:
        attractor_type = 'chaotic'
        convergence_detected = False

    return {
        'convergence_detected': convergence_detected,
        'attractor_type': attractor_type,
        'final_state': (float(np.mean(recent_phi)), float(np.mean(recent_policy))),
        'phi_stability': float(phi_std),
        'policy_stability': float(policy_std),
        'total_points': len(phi_history)
    }


def compute_fourier_spectrum(
    signal: List[float],
    sample_rate: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectrum using FFT.

    Args:
        signal: Time series data
        sample_rate: Sampling frequency (Hz)

    Returns:
        Tuple of (frequencies, magnitudes)
    """
    signal_arr = np.array(signal)
    N = len(signal_arr)

    # Compute FFT
    fft_vals = np.fft.fft(signal_arr)

    # Compute power spectrum (magnitude)
    magnitudes = np.abs(fft_vals[:N//2])

    # Frequency bins
    frequencies = np.fft.fftfreq(N, d=1.0/sample_rate)[:N//2]

    return frequencies, magnitudes


def find_dominant_frequencies(
    signal: List[float],
    n_peaks: int = 5,
    sample_rate: float = 1.0
) -> List[Tuple[float, float]]:
    """
    Identify dominant frequency components.

    Args:
        signal: Time series data
        n_peaks: Number of peaks to return
        sample_rate: Sampling frequency

    Returns:
        List of (frequency, magnitude) tuples
    """
    freqs, mags = compute_fourier_spectrum(signal, sample_rate)

    # Find peaks
    peaks = []
    for i in range(1, len(mags) - 1):
        if mags[i] > mags[i-1] and mags[i] > mags[i+1]:
            peaks.append((float(freqs[i]), float(mags[i])))

    # Sort by magnitude and return top n
    peaks.sort(key=lambda x: x[1], reverse=True)
    return peaks[:n_peaks]


def compare_agents(
    agents_history: List[List[Dict]]
) -> Dict:
    """
    Compare performance of multiple agents.

    Args:
        agents_history: List of history lists, one per agent

    Returns:
        dict: Comparative statistics
    """
    results = []

    for i, history in enumerate(agents_history):
        stats = analyze_run(history)
        stats['agent_id'] = i
        results.append(stats)

    # Sort by total reward
    results.sort(key=lambda x: x['total_reward'], reverse=True)

    return {
        'rankings': results,
        'winner': results[0]['agent_id'] if results else None,
        'n_agents': len(agents_history)
    }


def calculate_entropy(
    signal: List[float],
    bins: int = 50
) -> float:
    """
    Calculate Shannon entropy of signal distribution.

    Args:
        signal: Time series data
        bins: Number of histogram bins

    Returns:
        float: Entropy in nats
    """
    hist, _ = np.histogram(signal, bins=bins, density=True)
    hist = hist / (hist.sum() + 1e-10)

    nonzero = hist[hist > 0]
    entropy = -np.sum(nonzero * np.log(nonzero + 1e-10))

    return float(entropy)


__all__ = [
    'analyze_run',
    'safe_corr',
    'analyze_phase_space',
    'compute_fourier_spectrum',
    'find_dominant_frequencies',
    'compare_agents',
    'calculate_entropy',
]
