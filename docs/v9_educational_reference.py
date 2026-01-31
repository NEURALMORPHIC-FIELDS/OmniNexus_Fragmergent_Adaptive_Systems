#!/usr/bin/env python3
"""
OmniNexus v9 - Proof of Concept Implementation

Simplified, educational version demonstrating core fragmergent principles.
This version is preserved for learning and comparison purposes.

Differences from v10:
- Single-harmonic oscillator (simpler)
- No coherence tracking
- Random navigation only
- Fixed learning rate
- Minimal analysis tools
- ~200 lines vs v10's modular architecture

Use this version to understand the basic concepts before moving to v10.

Author: Vasile Lucian Borbeleac
Date: January 2025
Version: 9.0.0 (Educational)
"""

import numpy as np
import scipy.fft as fft


class FragmergentOscillator:
    """Simple single-harmonic oscillator."""
    
    def __init__(self, base_freq=1.0, noise=0.05):
        self.base_freq = base_freq
        self.noise = noise
        self.t = 0

    def step(self):
        self.t += 1
        phi = np.sin(self.t * self.base_freq)
        return phi + np.random.normal(0, self.noise)


class OpticalWorld:
    """FFT-based procedural world generator."""
    
    def __init__(self, size=(128, 128)):
        self.size = size
        self.state = np.zeros(size)

    def generate(self):
        noise = np.random.rand(*self.size)
        world_fft = fft.fft2(noise)
        filt = np.exp(-0.001 * (np.arange(self.size[0])[:, None]**2 + 
                                 np.arange(self.size[1])[None, :]**2))
        world = np.real(fft.ifft2(world_fft * filt))
        self.state = (world - world.min()) / (world.max() - world.min())
        return self.state

    def sample_region(self, pos, size=5):
        x, y = pos
        x0, y0 = max(0, x-size), max(0, y-size)
        x1, y1 = min(self.size[0], x+size), min(self.size[1], y+size)
        return self.state[x0:x1, y0:y1]


class Avatar:
    """Simple navigation agent with energy."""
    
    def __init__(self, world_size):
        self.pos = [world_size[0]//2, world_size[1]//2]
        self.energy = 100.0

    def move(self, direction):
        if direction == 'up':
            self.pos[0] = max(0, self.pos[0]-1)
        elif direction == 'down':
            self.pos[0] += 1
        elif direction == 'left':
            self.pos[1] = max(0, self.pos[1]-1)
        elif direction == 'right':
            self.pos[1] += 1
        self.energy -= 0.1

    def interact(self, region):
        richness = np.mean(region)
        if richness > 0.5:
            self.energy += 1.0
        else:
            self.energy -= 0.5
        self.energy = max(0, min(self.energy, 100))
        return richness


class RLAgent:
    """Simple RL agent with policy learning."""
    
    def __init__(self):
        self.policy = 0.5

    def decide(self, phi):
        return np.random.rand() < (self.policy + 0.3*max(0, phi))

    def update(self, reward, phi):
        self.policy += 0.01 * (reward - 0.5) * phi
        self.policy = np.clip(self.policy, 0.0, 1.0)


class OmniNexus:
    """Core v9 orchestrator."""
    
    def __init__(self, world_size=(128, 128)):
        self.oscillator = FragmergentOscillator()
        self.world = OpticalWorld(world_size)
        self.avatar = Avatar(world_size)
        self.agent = RLAgent()
        self.steps = 0

    def run_cycle(self):
        phi = self.oscillator.step()
        recalc = self.agent.decide(phi)

        if recalc or self.steps % 20 == 0:
            self.world.generate()
            mode = 'optical'
        else:
            mode = 'digital'

        direction = np.random.choice(['up','down','left','right'])
        self.avatar.move(direction)
        region = self.world.sample_region(self.avatar.pos)
        richness = self.avatar.interact(region)

        reward = (self.avatar.energy/100.0)*0.5 + (1.0 if mode=='optical' else 0.0)*0.5
        self.agent.update(reward, phi)

        self.steps += 1
        return {
            'step': self.steps,
            'mode': mode,
            'phi': round(phi, 2),
            'policy': round(self.agent.policy, 2),
            'energy': round(self.avatar.energy, 2),
            'pos': tuple(self.avatar.pos)
        }


def analyze_run(history):
    """Basic statistics for v9."""
    modes = [h['mode'] for h in history]
    phis = np.array([h['phi'] for h in history])
    policies = np.array([h['policy'] for h in history])
    energies = np.array([h['energy'] for h in history])

    total = len(history)
    optical = sum(1 for m in modes if m == 'optical')

    return {
        'steps': total,
        'optical_count': optical,
        'digital_count': total - optical,
        'optical_freq': optical/total if total else 0.0,
        'avg_phi': float(np.mean(phis)),
        'std_phi': float(np.std(phis)),
        'avg_policy': float(np.mean(policies)),
        'avg_energy': float(np.mean(energies)),
        'min_energy': float(np.min(energies)),
        'max_energy': float(np.max(energies)),
    }


if __name__ == "__main__":
    print("🌀 OmniNexus v9 — Proof of Concept\n")
    print("="*70)
    
    nexus = OmniNexus()
    history = []
    
    for i in range(200):
        state = nexus.run_cycle()
        history.append(state)
        
        if i < 10 or i >= 190 or i % 25 == 0:
            print(f"[Step {state['step']:3d}] Mode={state['mode']:7s}, "
                  f"Phi={state['phi']:.2f}, Policy={state['policy']:.2f}, "
                  f"Energy={state['energy']:.1f}, Pos={state['pos']}")

    stats = analyze_run(history)
    print("\n" + "="*70)
    print("📊 v9 Statistics")
    print("="*70)
    print(f"Total steps: {stats['steps']}")
    print(f"Optical: {stats['optical_count']} ({stats['optical_freq']*100:.1f}%)")
    print(f"Digital: {stats['digital_count']} ({(1-stats['optical_freq'])*100:.1f}%)")
    print(f"Avg φ: {stats['avg_phi']:.3f} ± {stats['std_phi']:.3f}")
    print(f"Avg policy: {stats['avg_policy']:.3f}")
    print(f"Energy: mean {stats['avg_energy']:.2f}, "
          f"min {stats['min_energy']:.2f}, max {stats['max_energy']:.2f}")
    print("\n✅ v9 Demo Complete!")