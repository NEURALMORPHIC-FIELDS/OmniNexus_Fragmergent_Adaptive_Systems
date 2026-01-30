# Mathematical Framework

## Overview

This document provides the mathematical foundations for the OmniNexus fragmergent system. We cover phase dynamics, coherence metrics, world generation, and the phase-coupled learning algorithm.

## 1. Oscillator Dynamics

### Multi-Harmonic Phase Signal

The core oscillator generates a phase signal using Fourier synthesis:

$$\phi(t) = \sum_{h=1}^{H} \frac{\sin(t \cdot f \cdot h)}{h} + \varepsilon(t)$$

Where:
- $t$ = discrete time step
- $f$ = base frequency (Hz)
- $H$ = number of harmonic layers (1 ≤ H ≤ 7)
- $h$ = harmonic index
- $\varepsilon(t) \sim \mathcal{N}(0, \sigma^2)$ = Gaussian noise

### Harmonic Amplitude Decay

The $1/h$ amplitude decay follows the pattern of a sawtooth wave's Fourier series:

| Harmonic | Amplitude | Relative Power |
|----------|-----------|----------------|
| 1 | 1.000 | 100% |
| 2 | 0.500 | 25% |
| 3 | 0.333 | 11% |
| 4 | 0.250 | 6% |
| 5 | 0.200 | 4% |

### Phase Range

For H harmonics, the theoretical phase range is:

$$\phi_{max} = \sum_{h=1}^{H} \frac{1}{h} = \mathcal{H}_H$$

Where $\mathcal{H}_H$ is the H-th harmonic number. For H=3: $\mathcal{H}_3 ≈ 1.833$

## 2. Coherence Metric

### Definition

Coherence quantifies the predictability of recent phase history:

$$C = 1 - \min(\sigma_{\phi}^{recent}, 1.0)$$

Where $\sigma_{\phi}^{recent}$ is the standard deviation of the last W samples:

$$\sigma_{\phi}^{recent} = \sqrt{\frac{1}{W}\sum_{i=t-W+1}^{t}(\phi_i - \bar{\phi})^2}$$

### Interpretation

| Coherence Range | Interpretation | System State |
|-----------------|----------------|--------------|
| 0.9 - 1.0 | Very High | Near-periodic, predictable |
| 0.7 - 0.9 | High | Quasi-periodic |
| 0.5 - 0.7 | Medium | Edge of chaos |
| 0.3 - 0.5 | Low | Chaotic tendencies |
| 0.0 - 0.3 | Very Low | Highly chaotic |

## 3. World Generation (FFT Synthesis)

### Algorithm

The procedural world is generated via frequency-domain filtering:

1. **Random Noise**: $N(x,y) \sim U(0,1)$ for all pixels

2. **FFT Transform**: $\hat{N}(u,v) = \mathcal{F}\{N(x,y)\}$

3. **Exponential Filter**:
$$K(u,v) = \exp(-\kappa(u^2 + v^2))$$

Where $\kappa$ is the complexity parameter.

4. **Apply Filter**: $\hat{W}(u,v) = \hat{N}(u,v) \cdot K(u,v)$

5. **Inverse FFT**: $W'(x,y) = \Re\{\mathcal{F}^{-1}\{\hat{W}(u,v)\}\}$

6. **Normalization**:
$$W(x,y) = \frac{W'(x,y) - W'_{min}}{W'_{max} - W'_{min}}$$

### Complexity Parameter Effects

| κ Value | Frequency Cutoff | Terrain Character |
|---------|------------------|-------------------|
| 0.0001 | Very Low | Ultra-smooth, large features |
| 0.001 | Low | Smooth, gentle hills |
| 0.005 | Medium | Varied, natural-looking |
| 0.01 | High | Rough, many small features |

### Terrain Entropy

Shannon entropy of terrain distribution:

$$H(W) = -\sum_{i=1}^{B} p_i \log(p_i)$$

Where $p_i$ is the probability of terrain value falling in bin $i$.

## 4. Energy Dynamics

### Avatar Energy Model

Energy $E$ evolves according to:

$$E_{t+1} = \text{clip}\Big(E_t + \Delta E_{move} + \Delta E_{interact}, 0, 100\Big)$$

Where:
- $\Delta E_{move} = -0.1$ (movement cost)
- $\Delta E_{interact} = \begin{cases} +2.0 & \text{if } \bar{r} > 0.5 \\ -0.5 & \text{otherwise} \end{cases}$

And $\bar{r}$ is the mean terrain richness in the sampled region.

### Expected Energy Change

For random terrain with $P(\bar{r} > 0.5) = 0.5$:

$$\mathbb{E}[\Delta E] = -0.1 + 0.5(2.0) + 0.5(-0.5) = +0.65$$

The avatar gains energy on average with uniform random terrain.

## 5. Reinforcement Learning

### Policy Representation

The policy is a scalar threshold $\pi \in [0, 1]$.

### Decision Function

Probability of choosing optical mode:

$$P(\text{optical}) = \pi + 0.3 \cdot \max(0, \phi) \cdot C$$

Where:
- $\pi$ = current policy
- $\phi$ = oscillator phase
- $C$ = coherence

### Reward Function

Multi-component reward signal:

$$R = 0.5 \cdot \frac{E}{100} + 0.3 \cdot \mathbf{1}_{optical} + 0.2 \cdot \bar{r}$$

Where:
- $E/100$ = normalized energy
- $\mathbf{1}_{optical}$ = indicator for optical mode
- $\bar{r}$ = terrain richness

### Phase-Coupled Policy Gradient

Policy update rule:

$$\pi_{t+1} = \text{clip}\Big(\pi_t + \alpha_t \cdot (R_t - 0.5) \cdot \phi_t, 0, 1\Big)$$

Where:
- $\alpha_t$ = adaptive learning rate
- $R_t - 0.5$ = advantage (reward relative to baseline)
- $\phi_t$ = phase modulation

### Adaptive Learning Rate

The learning rate adapts based on recent reward variance:

$$\alpha_t = \alpha_0 \cdot (1 + \sigma_R^{recent})$$

Where $\sigma_R^{recent}$ is the standard deviation of the last 20 rewards.

## 6. Analysis Metrics

### Correlation Coefficients

Pearson correlation between signals:

$$\rho_{XY} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y} = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i (x_i - \bar{x})^2 \sum_i (y_i - \bar{y})^2}}$$

Key correlations tracked:
- φ ↔ Policy
- φ ↔ Energy
- Coherence ↔ Reward

### Phase Space Analysis

For trajectory $(φ_t, π_t)$, we detect attractors by examining the final window:

**Fixed Point**: $\sigma_{\pi}^{recent} < 0.01$

**Limit Cycle**: $0.01 \leq \sigma_{\pi}^{recent} < 0.05$

**Chaotic**: $\sigma_{\pi}^{recent} \geq 0.05$

### Fourier Spectrum Analysis

Power spectrum of signal $x(t)$:

$$P(f) = |\hat{x}(f)|^2$$

Where $\hat{x}(f) = \mathcal{F}\{x(t)\}$ is the discrete Fourier transform.

Dominant frequencies are identified as local maxima in $P(f)$.

## 7. Exploration Coverage

### Grid-Based Coverage

Coverage metric:

$$\text{Coverage} = \frac{|\text{Visited Cells}|}{|\text{Total Cells}|}$$

Where cells are defined by dividing the world into a grid of size $G \times G$:

$$\text{cell}(x, y) = \Big(\lfloor x/G \rfloor, \lfloor y/G \rfloor\Big)$$

### Trajectory Length

Total Euclidean distance traveled:

$$L = \sum_{i=1}^{T-1} ||\mathbf{p}_{i+1} - \mathbf{p}_i||_2$$

## 8. Stability Analysis

### Lyapunov Stability

For the phase-coupled policy gradient, stability around equilibrium $\pi^*$ requires:

$$\frac{\partial}{\partial \pi}\Big[\mathbb{E}[\alpha(R-0.5)\phi]\Big]_{\pi=\pi^*} < 0$$

The system is generally stable when:
- Mean reward approximates the baseline (0.5)
- Phase signal has zero mean over long periods

### Basin of Attraction

The policy converges to different attractors based on:
1. Initial policy value
2. Oscillator configuration
3. World complexity

## Summary

| Component | Key Equation | Parameters |
|-----------|--------------|------------|
| Phase | $\phi(t) = \sum_h \sin(tfh)/h + \varepsilon$ | f, H, σ |
| Coherence | $C = 1 - \min(\sigma_\phi, 1)$ | W (window) |
| World | FFT filter: $K = \exp(-\kappa d^2)$ | κ |
| Energy | $E' = \text{clip}(E + \Delta E)$ | - |
| Reward | $R = 0.5E' + 0.3M + 0.2r$ | - |
| Policy | $\pi' = \text{clip}(\pi + \alpha(R-0.5)\phi)$ | α |

---

*Next: [Applications](04_APPLICATIONS.md)*
