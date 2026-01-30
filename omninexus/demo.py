#!/usr/bin/env python3
"""
OmniNexus v10 Demo - Complete demonstration of fragmergent system.

Runs single-agent and multi-agent simulations with visualization and analysis.

Usage:
    python -m omninexus.demo                 # Basic demo
    python -m omninexus.demo --visualize     # With matplotlib plots
    python -m omninexus.demo --steps 500     # Custom step count
    python -m omninexus.demo --multi-agent   # Multi-agent competition

Author: Lucian Coman
Date: January 2025
Version: 10.0.0
"""

import argparse
import sys
from typing import List, Dict

from omninexus.core import OmniNexus
from analysis import (
    analyze_run,
    analyze_phase_space,
    find_dominant_frequencies,
    compare_agents
)

# Optional: Visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


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
    },
    'minimal': {
        'base_freq': 0.08,
        'harmonic_layers': 1,
        'noise': 0.0,
        'learning_rate': 0.01
    },
    'quantum': {
        'base_freq': 0.12,
        'harmonic_layers': 6,
        'noise': 0.08,
        'learning_rate': 0.02
    }
}


def run_single_agent(
    steps: int = 300,
    preset: str = 'resonant',
    verbose: bool = True
) -> tuple:
    """
    Run single agent simulation.

    Args:
        steps: Number of simulation steps
        preset: Preset configuration name
        verbose: Print progress

    Returns:
        Tuple of (nexus, history)
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}")

    config = PRESETS[preset]

    if verbose:
        print(f"\n{'='*70}")
        print(f"OmniNexus v10 - Single Agent Demo")
        print(f"{'='*70}")
        print(f"Preset: {preset}")
        print(f"Configuration: {config}")
        print(f"Steps: {steps}\n")

    # Initialize system
    nexus = OmniNexus(
        world_size=(128, 128),
        **config
    )

    # Run simulation
    history = []
    for i in range(steps):
        state = nexus.run_cycle(smart_navigation=True)
        history.append(state)

        # Print progress
        if verbose and (i < 5 or i >= steps - 5 or i % 50 == 0):
            print(f"[{state['step']:3d}] {state['mode']:7s} | "
                  f"phi={state['phi']:6.3f} "
                  f"Coh={state['coherence']:.3f} "
                  f"Pol={state['policy']:.3f} "
                  f"E={state['energy']:5.1f} "
                  f"R={state['reward']:.3f}")

    if verbose:
        print(f"\n{'='*70}")
        print_statistics(analyze_run(history))

    return nexus, history


def run_multi_agent(
    steps: int = 200,
    presets: List[str] = None,
    verbose: bool = True
) -> Dict:
    """
    Run multi-agent competition.

    Args:
        steps: Number of simulation steps
        presets: List of preset names (default: first 3)
        verbose: Print progress

    Returns:
        dict: Competition results
    """
    if presets is None:
        presets = ['stable', 'chaotic', 'resonant']

    if verbose:
        print(f"\n{'='*70}")
        print(f"Multi-Agent Competition")
        print(f"{'='*70}")
        print(f"Agents: {len(presets)}")
        print(f"Presets: {', '.join(presets)}")
        print(f"Steps: {steps}\n")

    # Initialize agents
    agents = []
    histories = []

    for i, preset in enumerate(presets):
        config = PRESETS[preset]
        nexus = OmniNexus(world_size=(128, 128), **config)
        agents.append(nexus)
        histories.append([])

    # Run parallel simulation
    for step in range(steps):
        for i, nexus in enumerate(agents):
            state = nexus.run_cycle(smart_navigation=True)
            histories[i].append(state)

    # Analyze results
    comparison = compare_agents(histories)

    if verbose:
        print(f"\n{'='*70}")
        print("COMPETITION RESULTS")
        print(f"{'='*70}\n")

        medals = ['1st', '2nd', '3rd']
        for rank, result in enumerate(comparison['rankings'], 1):
            agent_id = result['agent_id']
            preset_name = presets[agent_id]
            medal = medals[rank-1] if rank <= 3 else f'{rank}th'

            print(f"{medal} Place: {preset_name}")
            print(f"   Total Reward: {result['total_reward']:.2f}")
            print(f"   Avg Reward: {result['avg_reward']:.3f}")
            print(f"   Final Policy: {result['final_policy']:.3f}")
            print(f"   Avg Energy: {result['avg_energy']:.1f}")
            print(f"   Optical Freq: {result['optical_freq']*100:.1f}%\n")

    return comparison


def print_statistics(stats: Dict) -> None:
    """Print formatted statistics."""
    print("FINAL STATISTICS")
    print(f"{'='*70}\n")

    print(f"Mode Distribution:")
    print(f"   Optical: {stats['optical_count']} ({stats['optical_freq']*100:.1f}%)")
    print(f"   Digital: {stats['digital_count']} ({(1-stats['optical_freq'])*100:.1f}%)\n")

    print(f"Oscillator Metrics:")
    print(f"   Mean phi: {stats['avg_phi']:.3f} +/- {stats['std_phi']:.3f}")
    print(f"   Range: [{stats['min_phi']:.3f}, {stats['max_phi']:.3f}]")
    print(f"   Coherence: {stats['avg_coherence']:.3f}\n")

    print(f"Learning Metrics:")
    print(f"   Final Policy: {stats['final_policy']:.3f}")
    print(f"   Policy Range: {stats['policy_range']:.3f}")
    print(f"   phi <-> Policy correlation: {stats['corr_phi_policy']:.3f}\n")

    print(f"Performance Metrics:")
    print(f"   Energy: avg={stats['avg_energy']:.1f}, "
          f"min={stats['min_energy']:.1f}, max={stats['max_energy']:.1f}")
    print(f"   Total Reward: {stats['total_reward']:.2f}")
    print(f"   Average Reward: {stats['avg_reward']:.3f}\n")

    print(f"Key Correlations:")
    print(f"   Coherence <-> Reward: {stats['corr_coherence_reward']:.3f}")
    print(f"   phi <-> Energy: {stats['corr_phi_energy']:.3f}\n")


def visualize_results(nexus: OmniNexus, history: List[Dict]) -> None:
    """Create visualization plots."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available for visualization")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract data
    steps = [h['step'] for h in history]
    phis = [h['phi'] for h in history]
    policies = [h['policy'] for h in history]
    energies = [h['energy'] for h in history]
    modes = [h['mode'] for h in history]

    # 1. Phase over time
    ax = axes[0, 0]
    mode_colors = ['red' if m == 'optical' else 'cyan' for m in modes]
    ax.scatter(steps, phis, c=mode_colors, s=10, alpha=0.5)
    ax.plot(steps, phis, 'gray', linewidth=0.5, alpha=0.3)
    ax.set_title('Oscillator Phase (phi)', fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('phi')
    ax.grid(True, alpha=0.3)

    # 2. Phase space
    ax = axes[0, 1]
    ax.plot(phis, policies, 'purple', linewidth=1, alpha=0.5)
    ax.scatter(phis, policies, c=range(len(phis)), cmap='viridis', s=5)
    ax.set_title('Phase Space (phi vs Policy)', fontweight='bold')
    ax.set_xlabel('phi')
    ax.set_ylabel('Policy')
    ax.grid(True, alpha=0.3)

    # 3. Policy evolution
    ax = axes[1, 0]
    ax.plot(steps, policies, 'green', linewidth=2)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Policy Evolution', fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Policy')
    ax.grid(True, alpha=0.3)

    # 4. Energy dynamics
    ax = axes[1, 1]
    ax.plot(steps, energies, 'orange', linewidth=2)
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Avatar Energy', fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Energy')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('omninexus_results.png', dpi=150)
    print("\nVisualization saved to: omninexus_results.png")
    plt.show()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='OmniNexus v10 Fragmergent System Demo'
    )
    parser.add_argument(
        '--steps', type=int, default=300,
        help='Number of simulation steps'
    )
    parser.add_argument(
        '--preset', type=str, default='resonant',
        choices=list(PRESETS.keys()),
        help='Preset configuration'
    )
    parser.add_argument(
        '--multi-agent', action='store_true',
        help='Run multi-agent competition'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()
    verbose = not args.quiet

    try:
        if args.multi_agent:
            run_multi_agent(steps=args.steps, verbose=verbose)
        else:
            nexus, history = run_single_agent(
                steps=args.steps,
                preset=args.preset,
                verbose=verbose
            )

            if args.visualize:
                visualize_results(nexus, history)

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
