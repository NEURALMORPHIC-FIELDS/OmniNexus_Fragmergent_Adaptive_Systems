#!/usr/bin/env python3
"""
Multi-Agent Competition Example - Comparing different configurations.

This example demonstrates:
- Running multiple OmniNexus instances
- Comparing different preset configurations
- Analyzing competition results

Author: Vasile Lucian Borbeleac
Date: January 2025
"""

from omninexus import OmniNexus
from analysis import analyze_run, compare_agents


# Preset configurations
PRESETS = {
    'stable': {
        'base_freq': 0.05,
        'harmonic_layers': 2,
        'noise': 0.01,
        'learning_rate': 0.005
    },
    'chaotic': {
        'base_freq': 0.35,
        'harmonic_layers': 5,
        'noise': 0.15,
        'learning_rate': 0.03
    },
    'resonant': {
        'base_freq': 0.15,
        'harmonic_layers': 3,
        'noise': 0.05,
        'learning_rate': 0.015
    },
    'exploration': {
        'base_freq': 0.20,
        'harmonic_layers': 4,
        'noise': 0.10,
        'learning_rate': 0.04
    }
}


def main():
    print("=" * 60)
    print("OmniNexus Multi-Agent Competition")
    print("=" * 60)

    # Configuration
    n_steps = 200
    preset_names = list(PRESETS.keys())

    print(f"\nCompeting presets: {', '.join(preset_names)}")
    print(f"Steps per simulation: {n_steps}")

    # Create agents
    print("\n1. Initializing agents...")
    agents = {}
    histories = {}

    for name in preset_names:
        config = PRESETS[name]
        agents[name] = OmniNexus(world_size=(128, 128), **config)
        histories[name] = []
        print(f"   Created '{name}' agent")

    # Run simulation
    print(f"\n2. Running simulation...")
    for step in range(n_steps):
        for name, nexus in agents.items():
            state = nexus.run_cycle(smart_navigation=True)
            histories[name].append(state)

        # Progress indicator
        if (step + 1) % 50 == 0:
            print(f"   Completed step {step + 1}/{n_steps}")

    # Analyze individual results
    print("\n3. Individual Results:")
    print("-" * 60)

    results = {}
    for name, history in histories.items():
        stats = analyze_run(history)
        results[name] = stats

        print(f"\n   {name.upper()}")
        print(f"   Total Reward: {stats['total_reward']:.2f}")
        print(f"   Avg Reward: {stats['avg_reward']:.3f}")
        print(f"   Optical Freq: {stats['optical_freq']:.1%}")
        print(f"   Avg Energy: {stats['avg_energy']:.1f}")
        print(f"   Final Policy: {stats['final_policy']:.3f}")

    # Compare agents
    print("\n4. Competition Rankings:")
    print("-" * 60)

    # Sort by total reward
    rankings = sorted(results.items(), key=lambda x: x[1]['total_reward'], reverse=True)

    medals = ['1st', '2nd', '3rd', '4th']
    for rank, (name, stats) in enumerate(rankings):
        print(f"   {medals[rank]}: {name} (reward: {stats['total_reward']:.2f})")

    # Winner analysis
    winner_name = rankings[0][0]
    winner_stats = rankings[0][1]

    print(f"\n5. Winner Analysis: {winner_name.upper()}")
    print("-" * 60)
    print(f"   Why did '{winner_name}' win?")

    # Analyze key metrics
    metrics_comparison = []
    for name, stats in results.items():
        metrics_comparison.append({
            'name': name,
            'avg_reward': stats['avg_reward'],
            'optical_freq': stats['optical_freq'],
            'avg_energy': stats['avg_energy'],
            'coherence': stats['avg_coherence']
        })

    print(f"\n   Comparison table:")
    print(f"   {'Preset':<12} {'Reward':>8} {'Optical%':>9} {'Energy':>8} {'Coherence':>10}")
    print(f"   {'-'*12} {'-'*8} {'-'*9} {'-'*8} {'-'*10}")

    for m in metrics_comparison:
        winner_marker = " *" if m['name'] == winner_name else ""
        print(f"   {m['name']:<12} {m['avg_reward']:>8.3f} {m['optical_freq']*100:>8.1f}% "
              f"{m['avg_energy']:>8.1f} {m['coherence']:>10.3f}{winner_marker}")

    print("\n" + "=" * 60)
    print("Competition completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
