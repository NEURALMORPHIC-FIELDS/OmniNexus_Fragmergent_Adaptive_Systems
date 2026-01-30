#!/usr/bin/env python3
"""
Custom Configuration Example - Advanced parameter tuning.

This example demonstrates:
- Creating custom parameter configurations
- Experimenting with oscillator settings
- Analyzing the effects of different parameters

Author: Vasile Lucian Borbeleac
Date: January 2025
"""

import numpy as np
from omninexus import OmniNexus
from omninexus.components import FragmergentOscillator, OpticalWorld
from analysis import analyze_run, analyze_phase_space


def experiment_base_frequency():
    """Experiment with different base frequencies."""
    print("\n" + "=" * 60)
    print("Experiment 1: Base Frequency Effects")
    print("=" * 60)

    frequencies = [0.05, 0.1, 0.2, 0.3, 0.4]
    results = []

    for freq in frequencies:
        nexus = OmniNexus(
            base_freq=freq,
            harmonic_layers=3,
            noise=0.05
        )

        history = [nexus.run_cycle() for _ in range(200)]
        stats = analyze_run(history)

        results.append({
            'freq': freq,
            'avg_phi': stats['avg_phi'],
            'std_phi': stats['std_phi'],
            'coherence': stats['avg_coherence'],
            'optical_freq': stats['optical_freq']
        })

        print(f"   freq={freq:.2f}: phi={stats['avg_phi']:6.3f} +/- {stats['std_phi']:.3f}, "
              f"coherence={stats['avg_coherence']:.3f}")

    return results


def experiment_harmonic_layers():
    """Experiment with different harmonic layer counts."""
    print("\n" + "=" * 60)
    print("Experiment 2: Harmonic Layers Effects")
    print("=" * 60)

    layers_list = [1, 2, 3, 5, 7]
    results = []

    for layers in layers_list:
        nexus = OmniNexus(
            base_freq=0.15,
            harmonic_layers=layers,
            noise=0.05
        )

        history = [nexus.run_cycle() for _ in range(200)]
        stats = analyze_run(history)

        # Analyze phase space
        phi_hist = [h['phi'] for h in history]
        policy_hist = [h['policy'] for h in history]
        phase_analysis = analyze_phase_space(phi_hist, policy_hist)

        results.append({
            'layers': layers,
            'phi_range': stats['max_phi'] - stats['min_phi'],
            'attractor': phase_analysis['attractor_type'],
            'stability': phase_analysis['policy_stability']
        })

        print(f"   layers={layers}: phi_range={results[-1]['phi_range']:.3f}, "
              f"attractor={phase_analysis['attractor_type']}, "
              f"stability={phase_analysis['policy_stability']:.4f}")

    return results


def experiment_noise_levels():
    """Experiment with different noise levels."""
    print("\n" + "=" * 60)
    print("Experiment 3: Noise Level Effects")
    print("=" * 60)

    noise_levels = [0.0, 0.02, 0.05, 0.1, 0.2]
    results = []

    for noise in noise_levels:
        nexus = OmniNexus(
            base_freq=0.15,
            harmonic_layers=3,
            noise=noise
        )

        history = [nexus.run_cycle() for _ in range(200)]
        stats = analyze_run(history)

        results.append({
            'noise': noise,
            'coherence': stats['avg_coherence'],
            'reward_std': stats['std_reward'],
            'optical_freq': stats['optical_freq']
        })

        print(f"   noise={noise:.2f}: coherence={stats['avg_coherence']:.3f}, "
              f"reward_std={stats['std_reward']:.3f}, "
              f"optical={stats['optical_freq']:.1%}")

    return results


def experiment_learning_rates():
    """Experiment with different learning rates."""
    print("\n" + "=" * 60)
    print("Experiment 4: Learning Rate Effects")
    print("=" * 60)

    learning_rates = [0.001, 0.005, 0.01, 0.02, 0.05]
    results = []

    for lr in learning_rates:
        nexus = OmniNexus(
            base_freq=0.15,
            harmonic_layers=3,
            noise=0.05,
            learning_rate=lr
        )

        history = [nexus.run_cycle() for _ in range(200)]
        stats = analyze_run(history)

        results.append({
            'lr': lr,
            'final_policy': stats['final_policy'],
            'policy_range': stats['policy_range'],
            'total_reward': stats['total_reward']
        })

        print(f"   lr={lr:.3f}: final_policy={stats['final_policy']:.3f}, "
              f"policy_range={stats['policy_range']:.3f}, "
              f"reward={stats['total_reward']:.2f}")

    return results


def explore_components():
    """Explore individual components."""
    print("\n" + "=" * 60)
    print("Component Exploration")
    print("=" * 60)

    # Oscillator exploration
    print("\n--- Oscillator ---")
    osc = FragmergentOscillator(base_freq=0.1, harmonic_layers=4, noise=0.02)

    for i in range(10):
        phi = osc.step()
        if i < 5 or i >= 8:
            print(f"   t={osc.t}: phi={phi:.4f}")

    print(f"   ...\n   Statistics: {osc.get_statistics()}")

    # World exploration
    print("\n--- Optical World ---")
    world = OpticalWorld(size=(64, 64), complexity=0.001)

    world.generate(seed=42)
    entropy = world.get_terrain_entropy()
    center_richness = world.get_richness_at((32, 32))

    print(f"   Size: {world.size}")
    print(f"   Terrain entropy: {entropy:.3f}")
    print(f"   Center richness: {center_richness:.3f}")

    # Sample region
    region = world.sample_region((32, 32), size=3)
    print(f"   Sample region shape: {region.shape}")
    print(f"   Sample region mean: {np.mean(region):.3f}")


def main():
    print("=" * 60)
    print("OmniNexus Custom Configuration Examples")
    print("=" * 60)

    # Run all experiments
    experiment_base_frequency()
    experiment_harmonic_layers()
    experiment_noise_levels()
    experiment_learning_rates()
    explore_components()

    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
