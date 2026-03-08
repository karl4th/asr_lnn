# DREAM Library Documentation

**Version:** 0.1.3  
**Last Updated:** March 2026  
**Author:** Manifestro Team

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Architecture](#architecture)
5. [Core API](#core-api)
6. [Training Guide](#training-guide)
7. [Advanced Usage](#advanced-usage)
8. [Benchmarks](#benchmarks)
9. [Optimization](#optimization)
10. [Examples](#examples)
11. [FAQ](#faq)
12. [Citation](#citation)

---

## Introduction

### What is DREAM?

**DREAM** (Dynamic Recall and Elastic Adaptive Memory) is a novel neural network architecture for sequence processing that enables **online adaptation during inference** without gradient updates.

Unlike traditional models (LSTM, Transformer, Mamba) that remain static after training, DREAM integrates **synaptic plasticity** directly into the inference cycle through:

- **Fast Weights** — Weights that adapt on every timestep via Hebbian learning
- **Surprise Gate** — Plasticity modulation based on prediction error novelty
- **Liquid Time-Constants (LTC)** — Adaptive integration speeds
- **Sleep Consolidation** — Memory stabilization mechanism

### Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Online Adaptation** | Adapts to new patterns without gradient descent |
| ⚡ **Fast Inference** | 3-5x speedup with optimizations |
| 💾 **Memory Efficient** | 10× fewer parameters than LSTM/Transformer |
| 🎯 **Surprise-Driven** | Learns only when something is novel |
| 🔁 **Stateful Processing** | Preserves memory across sequences |
| 📊 **Batch Support** | Efficient processing with proper state management |

### Why DREAM?

**Problem with Current Models:**

| Model | Static After Training | Online Adaptation | Parameters |
|-------|----------------------|-------------------|------------|
| LSTM | ✅ Yes | ❌ No | 893K |
| Transformer | ✅ Yes | ❌ No | 551K |
| **DREAM** | ❌ **No** | ✅ **Yes** | **82K** |

**DREAM Advantages:**
- 10× smaller than baselines
- 99.9% improvement on audio tasks (vs 93-94%)
- Instant adaptation to new speakers/patterns
- Stable under noise (1.09× ratio at 10dB SNR)

### Use Cases

- **Speech Recognition** — Speaker adaptation, noise robustness
- **Time Series** — Online anomaly detection, forecasting
- **Streaming Audio** — Real-time processing with context
- **Personalization** — User-specific adaptation without retraining
- **Any Sequence Task** — Where patterns change over time

---

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy 1.24+

### Basic Installation

```bash
pip install dreamnn
```

### Development Installation

```bash
git clone https://github.com/karl4th/dream-nn.git
cd dream-nn
pip install -e ".[dev]"
```

### With Audio Support

```bash
pip install dreamnn[audio]
```

### Verify Installation

```python
import torch
from dream import DREAM

model = DREAM(input_dim=64, hidden_dim=128)
x = torch.randn(4, 50, 64)
output, state = model(x)
print(f"✅ DREAM installed! Output shape: {output.shape}")
```

---

## Quick Start

### 5-Minute Example

```python
import torch
from dream import DREAM, DREAMConfig, DREAMCell

# ============================================================
# Option 1: High-level API (like nn.LSTM)
# ============================================================
model = DREAM(
    input_dim=64,      # Input features
    hidden_dim=128,    # Hidden size
    rank=8,            # Fast weights rank
)

# Process sequence
x = torch.randn(4, 50, 64)  # (batch, time, features)
output, state = model(x)
print(f"Output: {output.shape}")  # (4, 50, 128)

# ============================================================
# Option 2: Low-level API (DREAMCell)
# ============================================================
config = DREAMConfig(input_dim=64, hidden_dim=128, rank=8)
cell = DREAMCell(config)
state = cell.init_state(batch_size=4)

# Single timestep
x_t = torch.randn(4, 64)
h_new, state = cell(x_t, state)

# Full sequence
x_seq = torch.randn(4, 50, 64)
output, final_state = cell.forward_sequence(x_seq, return_all=True)
```

### Training Mode vs Inference Mode

```python
# Training: Fast weights FROZEN (stable base training)
model.train()
output, _ = model(x)
loss.backward()
optimizer.step()

# Inference: Fast weights ACTIVE (online adaptation)
model.eval()
with torch.no_grad():
    output, state = model(x)
    # Fast weights adapt during inference!
```

### Stateful Processing

```python
# Initialize state ONCE
state = model.init_state(batch_size=4)

# Process multiple sequences (state preserved!)
for seq in sequences:
    output, state = model(seq, state=state)
    # Model adapts and remembers between sequences!
```

---

## Architecture

### Overview

DREAM cell combines four key mechanisms for adaptive sequence processing:

```
┌─────────────────────────────────────────────────────────────┐
│                    DREAM Cell                               │
│  Input: x_t (batch, input_dim)                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐                                        │
│  │ 1. Predictive   │  x̂ = tanh(C^T @ h)                    │
│  │    Coding       │  e = x - x̂                            │
│  └─────────────────┘                                        │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────┐                                        │
│  │ 2. Surprise     │  S = σ((r - τ) / (2γ))                │
│  │    Gate         │  r = ||e|| / ||μ_e||                  │
│  └─────────────────┘                                        │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────┐                                        │
│  │ 3. Fast Weights │  dU = -λ(U-U_tgt) + η·S·(h⊗e)@V       │
│  │    (STDP)       │  U ← U + dU·dt                        │
│  └─────────────────┘                                        │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────┐                                        │
│  │ 4. LTC          │  τ = τ_sys / (1 + S·scale)            │
│  │    Update       │  h_new = (1-α)h + α·tanh(u_eff)       │
│  └─────────────────┘                                        │
│                                                             │
│  Output: h_new (batch, hidden_dim)                          │
└─────────────────────────────────────────────────────────────┘
```

### Component 1: Predictive Coding

**Purpose:** Generate predictions and compute errors.

**Formulas:**
```
Prediction:  x̂_t = tanh(C^T @ h_{t-1})
Error:       e_t = x_t - x̂_t
```

**Implementation:**
```python
self.C = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
self.W = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)
self.B_base = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)

# Forward
x_pred = torch.tanh(state.h @ self.C)
error = x - x_pred
```

**Why:**
- `C` decodes hidden state to input space
- `W` projects error back to hidden space
- `B` processes new input
- Error drives learning and surprise computation

---

### Component 2: Surprise Gate

**Purpose:** Detect novelty and gate plasticity.

**Formulas:**
```
Entropy:     H = 0.5 · log(2πe · var)
Threshold:   τ = 1.0 + α · H
Relative:    r = ||e|| / (||μ_e|| + ε)
Surprise:    S = σ((r - τ) / (2γ))
```

**Implementation:**
```python
def compute_surprise(self, error, state):
    # Entropy from variance
    variance = state.error_var.mean(dim=-1)
    entropy = 0.5 * torch.log(2 * torch.pi * torch.e * (variance + eps))
    
    # Adaptive threshold
    tau = 1.0 + self.alpha * entropy
    
    # Relative error (key innovation)
    baseline = state.error_mean.norm(dim=-1) + eps
    relative_error = error.norm(dim=-1) / baseline
    
    # Surprise
    surprise = torch.sigmoid((relative_error - tau) / (2 * self.gamma))
    return surprise
```

**Why Relative Error:**
| Scenario | Absolute Error | Relative Error |
|----------|---------------|----------------|
| Small baseline | Misses subtle changes | Normalized, detects changes |
| Large baseline | False alarms | Normalized, stable |

---

### Component 3: Fast Weights (STDP)

**Purpose:** Online learning without gradient descent.

**Formulas:**
```
Fast Weights:  W_fast = U @ V^T
STDP Update:   dU = -λ(U - U_target) + (η · S) · ((h ⊗ e) @ V)
Euler Step:    U ← U + dU · dt
```

**Low-Rank Decomposition:**
```
Full Matrix:     hidden × input = 256 × 80 = 20,480 params
Low-Rank:        (hidden × rank) + (input × rank)
                 = 256×16 + 80×16 = 5,376 params
Savings:         4× reduction!
```

**Implementation:**
```python
# Initialize V (fixed orthogonal)
V_init = torch.randn(input_dim, rank)
Q, _ = torch.linalg.qr(V_init)
self.register_buffer('V', Q)

# Update U (per-batch, learnable)
def update_fast_weights(self, h_prev, error, surprise, state):
    if self.freeze_fast_weights:
        return  # Skip during static base training
    
    # Hebbian term
    eV = error @ self.V
    hebbian = h_prev[:, :, None] * eV[:, None, :]
    
    # Plasticity modulation
    plasticity = self.eta * surprise[:, None, None]
    
    # Forgetting term
    forgetting = -self.forgetting_rate * (state.U - state.U_target)
    
    # Full update
    dU = forgetting + plasticity * hebbian
    state.U = state.U + dU * self.dt
```

---

### Component 4: Liquid Time-Constants (LTC)

**Purpose:** Adaptive integration speeds.

**Formulas:**
```
Dynamic τ:   τ = τ_sys / (1 + S · scale)
Integration: dh/dt = (-h + tanh(u_eff)) / τ
Euler:       h_new = (1 - dt/τ) · h_prev + (dt/τ) · h_target
```

**Behavior:**
| Surprise Level | τ | Behavior |
|---------------|---|----------|
| High (novel) | Small | Fast updates, rapid adaptation |
| Low (predictable) | Large | Slow integration, smooth memory |

**Implementation:**
```python
def compute_ltc_update(self, h_prev, u_eff, surprise):
    # Dynamic tau
    tau = self.tau_sys / (1.0 + surprise * self.tau_surprise_scale)
    tau = torch.clamp(tau, 0.01, 50.0)
    
    # Euler integration
    h_target = torch.tanh(u_eff)
    dt_over_tau = self.dt / (tau.unsqueeze(1) + self.dt)
    dt_over_tau = torch.clamp(dt_over_tau, 0.01, 0.5)
    
    h_new = (1 - dt_over_tau) * h_prev + dt_over_tau * h_target
    return h_new
```

---

### Sleep Consolidation

**Purpose:** Stabilize fast changes into long-term memory.

**Formulas:**
```
If S̄ > S_min:
    dU_target = ζ_sleep · S̄ · (U - U_target)
    U_target ← U_target + dU_target
```

**Implementation:**
```python
avg_surprise = state.avg_surprise.mean()

if avg_surprise > self.S_min:
    dU_target = self.sleep_rate * avg_surprise * (state.U - state.U_target)
    state.U_target = state.U_target + dU_target
```

---

## Core API

### DREAMConfig

Configuration container for DREAM cell.

```python
from dream import DREAMConfig

config = DREAMConfig(
    # Dimensions
    input_dim=39,        # Input features (39 for MFCC, 80 for mel)
    hidden_dim=256,      # Hidden state size
    rank=16,             # Fast weights rank
    
    # Time
    time_step=0.1,       # Integration step (dt)
    
    # Plasticity
    forgetting_rate=0.005,    # Decay rate (λ)
    base_plasticity=0.5,      # Learning rate (η)
    
    # Surprise
    base_threshold=0.3,       # Threshold (τ₀)
    entropy_influence=0.1,    # Entropy effect (α)
    surprise_temperature=0.05, # Sensitivity (γ)
    
    # Smoothing
    error_smoothing=0.05,
    surprise_smoothing=0.05,
    
    # Homeostasis
    target_norm=2.0,
    kappa=0.5,
    
    # LTC
    ltc_enabled=True,
    ltc_tau_sys=5.0,
    ltc_surprise_scale=5.0,
    
    # Sleep
    sleep_rate=0.005,
    min_surprise_for_sleep=0.2,
)
```

### DREAMCell

Low-level DREAM cell implementation.

```python
from dream import DREAMCell, DREAMState

cell = DREAMCell(config, freeze_fast_weights=False)

# Initialize state
state = cell.init_state(batch_size=4, device='cuda')

# Single timestep
x_t = torch.randn(4, 80)
h_new, state = cell(x_t, state)

# Full sequence
x_seq = torch.randn(4, 50, 80)
output, final_state = cell.forward_sequence(x_seq, return_all=True)
```

### DREAM (High-level)

LSTM-like interface for DREAM.

```python
from dream import DREAM

model = DREAM(
    input_dim=80,
    hidden_dim=256,
    rank=16,
    freeze_fast_weights=False,  # Training mode
)

# Training
model.train()
output, state = model(x)

# Inference
model.eval()
with torch.no_grad():
    output, state = model(x)
```

### DREAMStack

Multi-layer DREAM.

```python
from dream import DREAMStack

model = DREAMStack(
    input_dim=80,
    hidden_dims=[256, 256, 128],  # 3 layers
    rank=16,
    dropout=0.1,
)

output, states = model(x)
```

### DREAMState

State container.

```python
@dataclass
class DREAMState:
    h: torch.Tensor              # (batch, hidden_dim)
    U: torch.Tensor              # (batch, hidden_dim, rank)
    U_target: torch.Tensor       # (batch, hidden_dim, rank)
    adaptive_tau: torch.Tensor   # (batch,)
    error_mean: torch.Tensor     # (batch, input_dim)
    error_var: torch.Tensor      # (batch, input_dim)
    avg_surprise: torch.Tensor   # (batch,)
    
    def detach(self) -> "DREAMState":
        """Detach for truncated BPTT."""
```

---

## Training Guide

### Two-Phase Training

DREAM uses a unique two-phase training approach:

**Phase 1: Static Base Training**
- Fast weights FROZEN
- Only slow weights (C, W, B, η) learn via backprop
- Stable convergence

**Phase 2: Adaptation/Inference**
- Fast weights ACTIVE
- Online adaptation via STDP
- No gradients needed

### Basic Training Loop

```python
import torch
import torch.nn as nn
from dream import DREAM

model = DREAM(input_dim=80, hidden_dim=256, rank=16)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    model.train()  # Freeze fast weights
    optimizer.zero_grad()
    
    output, _ = model(x)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

### Truncated BPTT (Long Sequences)

```python
from dream import DREAMCell

model = DREAMCell(config)
state = model.init_state(batch_size=4)

segment_size = 100

for start in range(0, seq_len, segment_size):
    segment = x[:, start:start+segment_size, :]
    
    output, state = model.forward_sequence(segment, state=state)
    loss = criterion(output, target[:, start:start+segment_size, :])
    
    loss.backward()
    optimizer.step()
    
    state = state.detach()  # Truncate BPTT!
```

### Multi-Epoch Training with State Persistence

```python
model = DREAMCell(config)
state = model.init_state(batch_size=4)

for epoch in range(100):
    optimizer.zero_grad()
    
    # State preserved across epochs!
    output, state = model.forward_sequence(x, state=state)
    loss = criterion(output, target)
    
    loss.backward()
    optimizer.step()
    
    state = state.detach()
```

---

## Advanced Usage

### Custom State Management

```python
from dream import DREAMState

# Initialize with specific device
state = cell.init_state(batch_size, device='cuda')

# Access state components
print(state.h.shape)        # (batch, hidden_dim)
print(state.U.shape)        # (batch, hidden_dim, rank)
print(state.adaptive_tau)   # (batch,)

# Detach state for truncated BPTT
state = state.detach()

# Manual state modification
state.U *= 0.5  # Scale fast weights
```

### Sequence Classification

```python
import torch
from dream import DREAMConfig, DREAMCell

config = DREAMConfig(input_dim=64, hidden_dim=128)
cell = DREAMCell(config)
classifier = torch.nn.Linear(128, 10)  # 10 classes

# Process sequence
batch_size = 32
seq_len = 50
x = torch.randn(batch_size, seq_len, 64)

state = cell.init_state(batch_size)
output, final_state = cell.forward_sequence(x)

# Classify using final hidden state
logits = classifier(final_state.h)
predictions = logits.argmax(dim=-1)
```

### Memory Retention Test

```python
# Test if model remembers across multiple presentations
state = cell.init_state(1)

for pass_idx in range(5):
    output, state = cell.forward_sequence(same_sequence, state)
    # Surprise should decrease as model adapts!
    print(f"Pass {pass_idx}: Surprise = {state.avg_surprise.mean().item():.4f}")
```

### Bidirectional Processing

```python
# Forward
forward_output, _ = cell.forward_sequence(x)

# Backward (reverse time)
backward_output, _ = cell.forward_sequence(x.flip(dims=[1]))

# Combine
combined = torch.cat([
    forward_output,
    backward_output.flip(dims=[1])
], dim=-1)
```

---

## Benchmarks

### Test 1: Basic ASR Reconstruction

**Task:** Reconstruct mel spectrograms from 9 audio files.

| Model | Parameters | Initial Loss | Final Loss | Improvement | Time |
|-------|------------|--------------|------------|-------------|------|
| **DREAM** | 82K | 0.9298 | **0.0010** | **99.9%** | 502s |
| LSTM | 893K | 0.7889 | 0.0478 | 93.9% | 9s |
| Transformer | 551K | 0.9416 | 0.0696 | 92.6% | 11s |

**Conclusion:** DREAM achieves best quality but slower (price of online adaptation).

### Test 2: Speaker Adaptation

**Task:** Adapt to speaker change mid-sequence.

| Model | Baseline | Max Post-Switch | Adapt Steps | Surprise Spike |
|-------|----------|-----------------|-------------|----------------|
| **DREAM** | 1.2078 | 1.9657 | **0** | 0.119 |
| LSTM | 1.0435 | 1.5807 | 0 | N/A |
| Transformer | 1.1963 | 1.6963 | 0 | N/A |

**Conclusion:** All adapt instantly, but only DREAM detects change via surprise.

### Test 3: Noise Robustness

**Task:** Reconstruction with additive white noise.

| Model | Clean (20dB) | 10dB Loss | Ratio | Surprise Response |
|-------|--------------|-----------|-------|-------------------|
| **DREAM** | 1.2308 | 1.3390 | 1.09× | ✅ Yes |
| LSTM | 1.0163 | 1.1052 | 1.09× | N/A |
| Transformer | 1.2867 | 1.3757 | 1.07× | N/A |

**Conclusion:** DREAM stable under noise (1.09×), surprise increases with noise.

### Running Benchmarks

```bash
# All benchmarks
uv run python tests/benchmarks/run_all.py

# Individual tests
uv run python tests/benchmarks/test_01_basic_asr.py
uv run python tests/benchmarks/test_02_speaker_adaptation.py
uv run python tests/benchmarks/test_03_noise_robustness.py
```

---

## Optimization

### Optimized Cell

```python
from dream.cell_optimized import DREAMCellOptimized

model = DREAMCellOptimized(
    config,
    freeze_fast_weights=True,
    use_amp=True,  # Mixed precision
).cuda()
```

**Speedup:** 1.5-2.5x faster than standard cell

### Mixed Precision (AMP)

```python
from torch.cuda.amp import autocast, GradScaler

model = DREAMCellOptimized(config).cuda()
scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        output, _ = model(x)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Speedup:** 2-3x on Tensor Core GPUs (T4, V100, A100)

### TF32 (Ampere GPUs)

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

**Speedup:** 1.5-2x on A100/RTX 30xx

### Expected Performance

| GPU | Configuration | Latency | Throughput |
|-----|---------------|---------|------------|
| T4 | Standard | 15ms | 67 it/s |
| T4 | Optimized + AMP | 5ms | 200 it/s |
| A100 | Standard | 5ms | 200 it/s |
| A100 | Optimized + AMP + TF32 | 1.5ms | 667 it/s |

---

## Examples

### Example 1: Basic Usage

```python
from dream import DREAM

model = DREAM(input_dim=64, hidden_dim=128, rank=8)
x = torch.randn(4, 50, 64)
output, state = model(x)
```

### Example 2: Stateful Processing

```python
state = model.init_state(batch_size=4)

for seq in sequences:
    output, state = model(seq, state=state)
    # State preserved!
```

### Example 3: Online Adaptation

```python
model.eval()  # Unfreeze fast weights
state = model.init_state(batch_size=1)

for speaker_data in speakers:
    output, state = model(speaker_data, state=state)
    # Adapts to each speaker!
```

### Example 4: Long Sequences

```python
segment_size = 100
state = model.init_state(batch_size=4)

for start in range(0, 1500, segment_size):
    segment = x[:, start:start+segment_size, :]
    output, state = model.forward_sequence(segment, state=state)
    state = state.detach()
```

---

## FAQ

### How is DREAM different from LSTM?

**LSTM:** Fixed weights after training, no online adaptation.

**DREAM:** Fast weights update every timestep via Hebbian learning, enabling online adaptation without gradient descent.

### When should I use DREAM vs Transformer?

**Use DREAM when:**
- You need online adaptation (new speakers, patterns)
- Data is non-stationary
- You want fewer parameters (10× smaller)
- Real-time streaming is required

**Use Transformer when:**
- Data is stationary
- You need massive parallelization
- Training speed is critical
- You have abundant compute

### Does DREAM work on GPU?

Yes! DREAM is fully GPU-compatible with optimizations:
- Mixed precision (AMP)
- TF32 support
- CUDA graphs
- Optimized kernels

### How much memory does DREAM use?

For batch=4, hidden=256, rank=16:
- Model: ~82K parameters (~330KB)
- State: ~16K per batch element
- Total: ~400MB for 1000 timesteps (with truncated BPTT)

### Can I use DREAM for classification?

Yes! Use the final hidden state:

```python
output, final_state = cell.forward_sequence(x)
logits = classifier(final_state.h)
predictions = logits.argmax(dim=-1)
```

### How do I choose hidden_dim and rank?

**Start with defaults:** hidden_dim=256, rank=16

**Then tune:**
- Increase `hidden_dim` for more capacity
- Increase `rank` for more expressive fast weights
- Decrease both for faster training

### What does freeze_fast_weights do?

```python
# Training: Fast weights FROZEN
model.train()  # or model.set_fast_weights_mode(freeze=True)
# Only slow weights (C, W, B, eta) learn via backprop

# Inference: Fast weights ACTIVE
model.eval()  # or model.set_fast_weights_mode(freeze=False)
# Fast weights adapt online via STDP
```

---

## Citation

```bibtex
@software{dream2026,
  title = {DREAM: Dynamic Recall and Elastic Adaptive Memory},
  author = {Manifestro Team},
  year = {2026},
  url = {https://github.com/karl4th/dream-nn},
  version = "0.1.3"
}
```

---

## Support

- **GitHub Issues:** https://github.com/karl4th/dream-nn/issues
- **Documentation:** https://github.com/karl4th/dream-nn
- **PyPI:** https://pypi.org/project/dreamnn/

---

**Happy Coding! 🚀**
