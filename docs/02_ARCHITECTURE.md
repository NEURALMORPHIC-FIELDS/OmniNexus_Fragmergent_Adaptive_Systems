# Architecture Guide

## System Overview

OmniNexus follows a modular, component-based architecture where each module has a single responsibility and communicates through well-defined interfaces.

```
┌──────────────────────────────────────────────────────────────────────┐
│                         OmniNexus Orchestrator                       │
│                                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │  Oscillator │  │ OpticalWorld│  │   Avatar    │  │   RLAgent   │ │
│  │             │  │             │  │             │  │             │ │
│  │ φ(t), coh   │  │  terrain    │  │ pos, energy │  │   policy    │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │
│         │                │                │                │        │
│         └────────────────┴────────────────┴────────────────┘        │
│                              ↓                                       │
│                         run_cycle()                                  │
│                              ↓                                       │
│                      State Dictionary                                │
└──────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. FragmergentOscillator

**Location**: `omninexus/components/oscillator.py`

**Purpose**: Generate quasi-periodic phase signal with configurable dynamics.

**Interface**:
```python
class FragmergentOscillator:
    def __init__(self, base_freq: float, harmonic_layers: int, noise: float)
    def step(self) -> float                    # Returns φ(t)
    def get_phase_coherence(window: int) -> float
    def get_statistics() -> dict
    def reset()
```

**Internal State**:
- `t`: Current time step
- `history`: Complete phase history (List[float])

**Key Algorithm**:
```python
def step(self):
    phi = 0.0
    for h in range(1, self.harmonic_layers + 1):
        phi += np.sin(self.t * self.base_freq * h) / h
    phi += np.random.normal(0, self.noise)
    return phi
```

### 2. OpticalWorld

**Location**: `omninexus/components/optical_world.py`

**Purpose**: Generate procedural terrain using FFT-based synthesis.

**Interface**:
```python
class OpticalWorld:
    def __init__(self, size: Tuple[int, int], complexity: float)
    def generate(seed: int = None) -> np.ndarray
    def sample_region(pos: Tuple[int, int], size: int) -> np.ndarray
    def get_terrain_entropy() -> float
    def get_richness_at(pos: Tuple[int, int]) -> float
    def reset()
```

**Internal State**:
- `state`: 2D terrain array (np.ndarray, dtype=float32)
- `generation_count`: Number of regenerations

**Key Algorithm** (FFT Generation):
```
1. Generate random noise
2. Transform to frequency domain (FFT2)
3. Apply exponential decay filter
4. Transform back (IFFT2)
5. Normalize to [0, 1]
```

### 3. Avatar

**Location**: `omninexus/components/avatar.py`

**Purpose**: Represent an autonomous agent with energy management.

**Interface**:
```python
class Avatar:
    def __init__(self, world_size: Tuple[int, int])
    def move(direction: str, world_size: Tuple[int, int]) -> bool
    def interact(region: np.ndarray) -> float
    def get_trajectory_length() -> float
    def get_exploration_coverage(world_size, grid_size) -> float
    def reset(world_size: Tuple[int, int])
```

**Internal State**:
- `pos`: Current position [x, y]
- `energy`: Energy level [0, 100]
- `trajectory`: Movement history
- `interactions`: Terrain richness history

**Energy Dynamics**:
```
Movement: -0.1 energy per step
Rich terrain (>0.5): +2.0 energy
Poor terrain (≤0.5): -0.5 energy
```

### 4. RLAgent

**Location**: `omninexus/components/rl_agent.py`

**Purpose**: Learn optimal mode-switching policy through phase-coupled gradients.

**Interface**:
```python
class RLAgent:
    def __init__(self, learning_rate: float, exploration: float)
    def decide(phi: float, coherence: float) -> bool
    def update(reward: float, phi: float)
    def get_performance_metrics() -> dict
    def reset()
```

**Internal State**:
- `policy`: Current threshold [0, 1]
- `reward_history`: All observed rewards

**Key Algorithms**:

Decision:
```python
def decide(self, phi, coherence):
    threshold = self.policy + 0.3 * max(0, phi) * coherence
    return np.random.rand() < threshold
```

Update (Phase-Coupled Gradient):
```python
def update(self, reward, phi):
    gradient = (reward - 0.5) * phi
    self.policy += self.learning_rate * gradient
    self.policy = np.clip(self.policy, 0.0, 1.0)
```

### 5. OmniNexus (Orchestrator)

**Location**: `omninexus/core.py`

**Purpose**: Coordinate all components and execute simulation cycles.

**Interface**:
```python
class OmniNexus:
    def __init__(self, world_size, base_freq, harmonic_layers,
                 noise, learning_rate, world_complexity)
    def run_cycle(smart_navigation: bool) -> dict
    def get_statistics() -> dict
    def reset()
```

**State Dictionary** (returned by `run_cycle`):
```python
{
    'step': int,           # Current step number
    'mode': str,           # 'optical' or 'digital'
    'phi': float,          # Oscillator phase
    'coherence': float,    # Phase coherence
    'policy': float,       # RL policy value
    'energy': float,       # Avatar energy
    'richness': float,     # Terrain richness at position
    'reward': float,       # Calculated reward
    'pos': Tuple[int, int] # Avatar position
}
```

## Data Flow

```
                    ┌──────────────────────────────────────────┐
                    │              INITIALIZATION              │
                    │                                          │
                    │  OmniNexus creates and connects all      │
                    │  components with configured parameters   │
                    └──────────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│                           CYCLE LOOP                                 │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ 1. OSCILLATOR                                                │    │
│  │    phi = oscillator.step()                                   │    │
│  │    coherence = oscillator.get_phase_coherence()              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ 2. RL DECISION                                               │    │
│  │    recalc = agent.decide(phi, coherence)                     │    │
│  │    mode = 'optical' if recalc else 'digital'                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ 3. WORLD UPDATE                                              │    │
│  │    if mode == 'optical':                                     │    │
│  │        world.generate()  # Expensive FFT operation           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ 4. AVATAR ACTION                                             │    │
│  │    direction = _smart_move() or random_choice()              │    │
│  │    avatar.move(direction)                                    │    │
│  │    region = world.sample_region(avatar.pos)                  │    │
│  │    richness = avatar.interact(region)                        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ 5. LEARNING                                                  │    │
│  │    reward = _calculate_reward(mode, richness)                │    │
│  │    agent.update(reward, phi)                                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│                      Return state dict                               │
└──────────────────────────────────────────────────────────────────────┘
```

## Configuration Parameters

### Oscillator Parameters

| Parameter | Type | Range | Default | Effect |
|-----------|------|-------|---------|--------|
| `base_freq` | float | 0.01-0.5 | 0.1 | Phase oscillation speed |
| `harmonic_layers` | int | 1-7 | 3 | Signal complexity |
| `noise` | float | 0.0-0.2 | 0.05 | Stochasticity |

### World Parameters

| Parameter | Type | Range | Default | Effect |
|-----------|------|-------|---------|--------|
| `world_size` | tuple | any | (128, 128) | Environment dimensions |
| `complexity` | float | 0.0001-0.01 | 0.001 | Terrain smoothness |

### Agent Parameters

| Parameter | Type | Range | Default | Effect |
|-----------|------|-------|---------|--------|
| `learning_rate` | float | 0.001-0.05 | 0.015 | Policy adaptation speed |

## Extension Points

### Custom Oscillators
Implement custom phase generators by subclassing:
```python
class CustomOscillator(FragmergentOscillator):
    def step(self) -> float:
        # Custom phase generation logic
        pass
```

### Custom Worlds
Implement alternative terrain generators:
```python
class PerlinWorld(OpticalWorld):
    def generate(self, seed=None) -> np.ndarray:
        # Perlin noise generation
        pass
```

### Custom Reward Functions
Override the reward calculation:
```python
class CustomNexus(OmniNexus):
    def _calculate_reward(self, mode, richness) -> float:
        # Custom reward logic
        pass
```

## Thread Safety

The current implementation is **not thread-safe**. For parallel simulations:
- Create separate OmniNexus instances per thread
- Use multiprocessing for true parallelism
- Synchronize access to shared analysis structures

## Performance Considerations

### Memory Usage
- Trajectory history grows linearly with steps
- Phase history grows linearly with steps
- For long simulations (>10,000 steps), consider periodic cleanup

### CPU Bottlenecks
1. FFT world generation (optical mode)
2. Smart navigation (4 region samples per step)
3. Coherence calculation (window-based)

### Optimization Tips
- Reduce `world_size` for faster iteration
- Lower `harmonic_layers` for simpler dynamics
- Disable smart navigation for benchmarking

---

*Next: [Mathematical Framework](03_MATHEMATICS.md)*
