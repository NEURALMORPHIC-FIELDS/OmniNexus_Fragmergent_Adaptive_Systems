# Applications

## Overview

OmniNexus provides a flexible platform for exploring adaptive systems across multiple domains. This document outlines practical applications and use cases.

## Research Applications

### 1. Computational Neuroscience

**Phase-Coupled Learning Studies**

OmniNexus models how neural oscillations modulate learning:

```python
from omninexus import OmniNexus

# Simulate theta-rhythm learning (4-8 Hz equivalent)
nexus = OmniNexus(base_freq=0.06, harmonic_layers=2, noise=0.02)

# Track policy changes during different phase windows
high_phase_updates = []
low_phase_updates = []

for _ in range(1000):
    state = nexus.run_cycle()
    if state['phi'] > 0.5:
        high_phase_updates.append(state['policy'])
    else:
        low_phase_updates.append(state['policy'])

# Compare learning rates between phase windows
```

**Research Questions**:
- How does learning timing affect skill acquisition?
- What are optimal phase-reward alignments?
- How does noise affect learning robustness?

### 2. Robotics and Autonomous Systems

**Mode-Switching Controllers**

OmniNexus demonstrates intelligent mode switching between computational strategies:

| Robotic Application | Optical Mode | Digital Mode |
|---------------------|--------------|--------------|
| Navigation | Full SLAM update | Odometry only |
| Vision | Object detection | Motion tracking |
| Planning | Global replanning | Local adjustment |
| Control | Model-predictive | PID |

**Implementation Example**:

```python
class RobotController:
    def __init__(self):
        self.nexus = OmniNexus(
            world_size=(64, 64),
            learning_rate=0.02
        )

    def get_control_mode(self, sensor_coherence):
        """Decide which control mode to use."""
        state = self.nexus.run_cycle()

        if state['mode'] == 'optical':
            return 'full_computation'
        else:
            return 'incremental_update'
```

### 3. Resource Management

**Energy-Aware Computing**

Model computational workloads with energy constraints:

```python
# Configure for energy-sensitive system
nexus = OmniNexus(
    world_size=(256, 256),  # Larger = more resources
    world_complexity=0.005   # Variable resource distribution
)

# Track energy efficiency
total_computation = 0
total_energy_spent = 0

for _ in range(500):
    state = nexus.run_cycle()

    if state['mode'] == 'optical':
        total_computation += 10  # Heavy computation
        total_energy_spent += 5
    else:
        total_computation += 1   # Light computation
        total_energy_spent += 1

efficiency = total_computation / total_energy_spent
```

**Applications**:
- Mobile device power management
- Data center workload scheduling
- IoT device optimization

### 4. Financial Modeling

**Adaptive Trading Strategies**

The exploration-exploitation balance maps to trading:

| OmniNexus Concept | Trading Analog |
|-------------------|----------------|
| Optical mode | Active trading, position changes |
| Digital mode | Hold current positions |
| Coherence | Market stability |
| Phase | Market cycle position |
| Energy | Portfolio capital |

**Example Strategy**:

```python
class AdaptiveTrader:
    def __init__(self, capital):
        self.nexus = OmniNexus(learning_rate=0.01)
        self.capital = capital

    def decide_action(self, market_volatility):
        # Map volatility to coherence
        state = self.nexus.run_cycle()

        if state['mode'] == 'optical' and self.capital > 50:
            return 'trade'  # Active position
        return 'hold'       # Maintain position
```

## Educational Applications

### 1. Complex Systems Courses

OmniNexus provides hands-on examples for teaching:

**Lab 1: Emergence from Simple Rules**
```python
# Show how complex behavior emerges
nexus = OmniNexus()
for preset in ['minimal', 'resonant', 'chaotic']:
    # Compare emergent behaviors
    pass
```

**Lab 2: Phase Space Dynamics**
```python
from analysis import analyze_phase_space

# Visualize attractor formation
history = run_simulation()
phase_analysis = analyze_phase_space(
    phi_history=[h['phi'] for h in history],
    policy_history=[h['policy'] for h in history]
)
```

**Lab 3: Fourier Analysis**
```python
from analysis import find_dominant_frequencies

# Analyze periodicity in system behavior
freqs = find_dominant_frequencies(phi_history)
```

### 2. Reinforcement Learning Tutorials

**Concept Demonstrations**:

1. **Exploration vs Exploitation**
   - `chaotic` preset emphasizes exploration
   - `stable` preset emphasizes exploitation

2. **Reward Shaping**
   - Multi-component reward function
   - Trade-offs between objectives

3. **Adaptive Learning Rates**
   - Variance-based adaptation
   - Phase-coupled modulation

## Simulation Experiments

### Experiment 1: Preset Comparison

Compare all presets over identical conditions:

```python
from omninexus import OmniNexus
from omninexus.demo import PRESETS
from analysis import analyze_run

results = {}
for preset_name, config in PRESETS.items():
    nexus = OmniNexus(**config)
    history = [nexus.run_cycle() for _ in range(500)]
    results[preset_name] = analyze_run(history)

# Find best performer
best = max(results.items(), key=lambda x: x[1]['total_reward'])
print(f"Best preset: {best[0]} with reward {best[1]['total_reward']:.2f}")
```

### Experiment 2: Parameter Sensitivity

Sweep key parameters to understand sensitivity:

```python
import numpy as np

# Sweep base frequency
freq_results = {}
for freq in np.linspace(0.01, 0.4, 20):
    nexus = OmniNexus(base_freq=freq)
    history = [nexus.run_cycle() for _ in range(300)]
    freq_results[freq] = analyze_run(history)['total_reward']

# Plot freq vs reward
```

### Experiment 3: Multi-Agent Competition

Run agents with different strategies:

```python
from omninexus.demo import run_multi_agent

# Compare presets in competition
results = run_multi_agent(
    steps=500,
    presets=['stable', 'chaotic', 'resonant', 'exploration']
)

for rank in results['rankings']:
    print(f"Agent {rank['agent_id']}: {rank['total_reward']:.2f}")
```

### Experiment 4: Long-Term Stability

Test convergence over extended runs:

```python
nexus = OmniNexus()
policy_history = []

for _ in range(5000):
    state = nexus.run_cycle()
    policy_history.append(state['policy'])

# Check for convergence
final_std = np.std(policy_history[-500:])
converged = final_std < 0.02
print(f"Converged: {converged}, Final std: {final_std:.4f}")
```

## Practical Tips

### Choosing Parameters

| Goal | Recommended Settings |
|------|---------------------|
| Fast convergence | Low noise, moderate learning rate |
| Robust exploration | High noise, multiple harmonics |
| Energy efficiency | Lower world complexity |
| Rich dynamics | High harmonics, moderate noise |

### Performance Optimization

1. **Reduce world size** for rapid prototyping
2. **Disable visualization** during batch experiments
3. **Use numpy** vectorization for post-processing
4. **Profile** with different preset configurations

### Common Pitfalls

1. **Energy death**: If energy reaches 0, avatar becomes trapped
   - Solution: Reduce world complexity or movement cost

2. **Policy collapse**: Policy converges to 0 or 1
   - Solution: Reduce learning rate or increase noise

3. **No convergence**: System remains chaotic
   - Solution: Increase number of steps or adjust parameters

## Integration Examples

### With NumPy/SciPy

```python
import numpy as np
from scipy import signal

# Extract phi signal
phi_signal = np.array([h['phi'] for h in history])

# Apply Butterworth filter
b, a = signal.butter(4, 0.1)
filtered_phi = signal.filtfilt(b, a, phi_signal)
```

### With Pandas

```python
import pandas as pd

# Convert history to DataFrame
df = pd.DataFrame(history)

# Analyze with pandas
print(df.groupby('mode').agg({
    'reward': 'mean',
    'energy': 'mean'
}))
```

### With Matplotlib

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Phase vs time
axes[0, 0].plot([h['phi'] for h in history])
axes[0, 0].set_title('Phase Evolution')

# Policy vs time
axes[0, 1].plot([h['policy'] for h in history])
axes[0, 1].set_title('Policy Evolution')

# Phase space
axes[1, 0].scatter(
    [h['phi'] for h in history],
    [h['policy'] for h in history],
    c=range(len(history)), cmap='viridis', s=1
)
axes[1, 0].set_title('Phase Space')

# Energy
axes[1, 1].plot([h['energy'] for h in history])
axes[1, 1].set_title('Energy Dynamics')

plt.tight_layout()
plt.savefig('analysis.png')
```

---

*Next: [Research Analysis](05_PATENT_ANALYSIS.md)*
