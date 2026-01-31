# OmniNexus Fragmergent Adaptive Systems

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-10.0.0-green.svg)](https://github.com/NEURALMORPHIC-FIELDS/OmniNexus_Fragmergent_Adaptive_Systems)

**OmniNexus** is a research platform for exploring fragmergent adaptive systems - hybrid architectures that operate at the edge of chaos, balancing order and adaptability through phase-coupled learning and intelligent mode switching.

## Overview

The **fragmergent** paradigm (fragile + emergent) creates systems that maintain dynamic equilibrium between:
- **Stability**: Predictable, efficient operation
- **Adaptability**: Responsive to environmental changes
- **Emergence**: Complex behaviors arising from simple rules

```
┌─────────────────────────────────────────────────────────┐
│            OmniNexus (Main Orchestrator)                │
│                      run_cycle()                        │
└────────┬──────────┬──────────┬──────────┬───────────────┘
         │          │          │          │
    ┌────▼──┐  ┌────▼──────┐  ┌▼─────┐  ┌─▼────┐
    │Oscil- │  │Optical    │  │Avatar│  │RL    │
    │lator  │  │World      │  │Agent │  │Agent │
    └───────┘  └───────────┘  └──────┘  └──────┘
```

## Features

- **Multi-Harmonic Oscillator**: Phase generator with configurable harmonics and coherence tracking
- **FFT-Based World Generation**: Procedural terrain using Fast Fourier Transform
- **Phase-Coupled Reinforcement Learning**: Learning synchronized with oscillator dynamics
- **Intelligent Mode Switching**: Adaptive optical/digital mode selection
- **Real-Time Analysis**: Phase space analysis, Fourier spectrum, attractor detection
- **Multi-Agent Competition**: Compare different configurations

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/NEURALMORPHIC-FIELDS/OmniNexus_Fragmergent_Adaptive_Systems.git
cd OmniNexus_Fragmergent_Adaptive_Systems

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Basic Usage

```python
from omninexus import OmniNexus

# Initialize system with default parameters
nexus = OmniNexus(world_size=(128, 128))

# Run simulation
for _ in range(300):
    state = nexus.run_cycle(smart_navigation=True)
    print(f"Step {state['step']}: mode={state['mode']}, energy={state['energy']:.1f}")

# Get statistics
stats = nexus.get_statistics()
print(f"Optical frequency: {stats['optical_frequency']:.2%}")
```

### Command Line Demo

```bash
# Basic demo
python -m omninexus.demo

# With visualization
python -m omninexus.demo --visualize

# Custom configuration
python -m omninexus.demo --steps 500 --preset chaotic

# Multi-agent competition
python -m omninexus.demo --multi-agent
```

## Presets

| Preset | Base Freq | Harmonics | Noise | Learning Rate | Behavior |
|--------|-----------|-----------|-------|---------------|----------|
| `stable` | 0.05 | 2 | 0.01 | 0.005 | Predictable, efficient |
| `chaotic` | 0.35 | 5 | 0.15 | 0.03 | High exploration |
| `resonant` | 0.15 | 3 | 0.05 | 0.015 | Balanced (default) |
| `exploration` | 0.20 | 4 | 0.10 | 0.04 | Discovery-focused |
| `minimal` | 0.08 | 1 | 0.0 | 0.01 | Deterministic baseline |
| `quantum` | 0.12 | 6 | 0.08 | 0.02 | Complex dynamics |

## Architecture

### Core Components

1. **FragmergentOscillator** (`omninexus/components/oscillator.py`)
   - Generates phase signal φ(t) using multi-harmonic synthesis
   - Tracks temporal coherence for decision-making
   - Formula: φ(t) = Σ(h=1 to n) [sin(t·f·h) / h] + ε

2. **OpticalWorld** (`omninexus/components/optical_world.py`)
   - FFT-based procedural terrain generation
   - Exponential frequency filtering for smooth terrain
   - Supports dynamic regeneration with filter kernel caching

3. **Avatar** (`omninexus/components/avatar.py`)
   - Energy-constrained agent navigation
   - Memory-efficient trajectory tracking with deque
   - Resource interaction mechanics

4. **RLAgent** (`omninexus/components/rl_agent.py`)
   - Phase-coupled policy gradient learning
   - Adaptive learning rate based on reward variance
   - Efficient running statistics with Welford's algorithm

5. **OmniNexus** (`omninexus/core.py`)
   - Main orchestrator integrating all components
   - Cycle-based execution with reward calculation
   - Statistical tracking and analysis

### Analysis Tools (`analysis/__init__.py`)

- `analyze_run()`: Comprehensive simulation statistics
- `analyze_phase_space()`: Attractor detection and stability
- `compute_fourier_spectrum()`: Frequency analysis
- `find_dominant_frequencies()`: Peak detection
- `compare_agents()`: Multi-agent benchmarking

## Documentation

- [Conceptual Overview](docs/01_CONCEPT.md) - Core fragmergent principles
- [Architecture Guide](docs/02_ARCHITECTURE.md) - System design and components
- [Mathematical Framework](docs/03_MATHEMATICS.md) - Formulas and theory
- [Applications](docs/04_APPLICATIONS.md) - Use cases and examples
- [Research Analysis](docs/05_PATENT_ANALYSIS.md) - Related work and innovation

## Project Structure

```
OmniNexus_Fragmergent_Adaptive_Systems/
├── omninexus/                    # Main package
│   ├── __init__.py               # Package initialization
│   ├── core.py                   # OmniNexus orchestrator
│   ├── demo.py                   # CLI demo application
│   └── components/               # Core components
│       ├── __init__.py
│       ├── oscillator.py         # FragmergentOscillator
│       ├── optical_world.py      # OpticalWorld
│       ├── avatar.py             # Avatar agent
│       └── rl_agent.py           # RLAgent
├── analysis/                     # Analysis tools
│   └── __init__.py               # Analysis functions
├── docs/                         # Documentation
│   ├── 01_CONCEPT.md
│   ├── 02_ARCHITECTURE.md
│   ├── 03_MATHEMATICS.md
│   ├── 04_APPLICATIONS.md
│   ├── 05_PATENT_ANALYSIS.md
│   └── v9_educational_reference.py  # Educational v9 implementation
├── examples/                     # Usage examples
│   ├── basic_simulation.py
│   ├── multi_agent_competition.py
│   └── custom_configuration.py
├── tests/                        # Test suite
│   └── ...
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Package configuration
├── CONTRIBUTING.md               # Contribution guidelines
├── CHANGELOG.md                  # Version history
├── LICENSE                       # Apache 2.0
└── README.md                     # This file
```

## Requirements

- Python 3.8+
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0 (optional, for visualization)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use OmniNexus in your research, please cite:

```bibtex
@software{omninexus2025,
  author = {Borbeleac, Vasile Lucian},
  title = {OmniNexus: Fragmergent Adaptive Systems},
  year = {2025},
  version = {10.0.0},
  url = {https://github.com/NEURALMORPHIC-FIELDS/OmniNexus_Fragmergent_Adaptive_Systems}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Author

**Vasile Lucian Borbeleac** - [NEURALMORPHIC-FIELDS](https://github.com/NEURALMORPHIC-FIELDS)

Contact: v.l.borbel@gmail.com

---

*OmniNexus: Exploring the edge of chaos, one phase at a time.*
