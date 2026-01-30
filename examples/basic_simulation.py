#!/usr/bin/env python3
"""
Basic Simulation Example - Getting started with OmniNexus.

This example demonstrates:
- Creating an OmniNexus system
- Running a basic simulation
- Accessing state information
- Getting statistics

Author: Lucian Coman
Date: January 2025
"""

from omninexus import OmniNexus


def main():
    print("=" * 60)
    print("OmniNexus Basic Simulation Example")
    print("=" * 60)

    # Create the system with default parameters
    print("\n1. Creating OmniNexus system...")
    nexus = OmniNexus(
        world_size=(128, 128),
        base_freq=0.15,
        harmonic_layers=3,
        noise=0.05,
        learning_rate=0.015,
        world_complexity=0.001
    )
    print(f"   Created: {nexus}")

    # Run simulation for 100 steps
    print("\n2. Running simulation for 100 steps...")
    history = []

    for i in range(100):
        state = nexus.run_cycle(smart_navigation=True)
        history.append(state)

        # Print every 20th step
        if i % 20 == 0:
            print(f"   Step {state['step']:3d}: "
                  f"mode={state['mode']:7s}, "
                  f"phi={state['phi']:6.3f}, "
                  f"energy={state['energy']:5.1f}")

    # Get final statistics
    print("\n3. Final Statistics:")
    stats = nexus.get_statistics()
    print(f"   Total steps: {stats['steps']}")
    print(f"   Optical mode frequency: {stats['optical_frequency']:.1%}")
    print(f"   World regenerations: {stats['world_generations']}")
    print(f"   Final energy: {stats['avatar_energy']:.1f}")
    print(f"   Trajectory length: {stats['trajectory_length']:.1f}")
    print(f"   Exploration coverage: {stats['exploration_coverage']:.1%}")

    # Oscillator stats
    osc_stats = stats['oscillator']
    print(f"\n   Oscillator:")
    print(f"      Mean phi: {osc_stats['mean']:.3f}")
    print(f"      Std phi: {osc_stats['std']:.3f}")
    print(f"      Coherence: {osc_stats['coherence']:.3f}")

    # Agent performance
    agent_stats = stats['agent']
    print(f"\n   RL Agent:")
    print(f"      Mean reward: {agent_stats['mean_reward']:.3f}")
    print(f"      Total reward: {agent_stats['total_reward']:.2f}")

    # Reset and run again
    print("\n4. Resetting system...")
    nexus.reset()
    print(f"   After reset: {nexus}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
