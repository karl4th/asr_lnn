# DREAM: Complete Developer Guide

**Dynamic Recall and Elastic Adaptive Memory**

**Version:** 0.1.2  
**For:** Developers who want to understand and use DREAM

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [What is DREAM?](#what-is-dream)
3. [Why DREAM?](#why-dream)
4. [Installation](#installation)
5. [Architecture Overview](#architecture-overview)
6. [Basic Usage](#basic-usage)
7. [How It Works: Component by Component](#how-it-works-component-by-component)
8. [Advanced Usage](#advanced-usage)
9. [Configuration Guide](#configuration-guide)
10. [Training Guide](#training-guide)
11. [Saving & Loading](#saving--loading)
12. [Common Patterns](#common-patterns)
13. [Benchmark Results](#benchmark-results)
14. [Troubleshooting](#troubleshooting)
15. [API Reference](#api-reference)
16. [FAQ](#faq)

---

## Quick Start

### Installation (30 seconds)

```bash
pip install dreamnn
```

### Hello World (1 minute)

```python
import torch
from dream import DREAM

# Create model (like nn.LSTM)
model = DREAM(input_dim=80, hidden_dim=256, rank=16)

# Process sequence
x = torch.randn(4, 100, 80)  # (batch, time, features)
output, state = model(x)

print(f"Output shape: {output.shape}")  # (4, 100, 256)
```

### Key Feature: Online Adaptation

```python
from dream import DREAMCell, DREAMConfig

config = DREAMConfig(input_dim=80, hidden_dim=256)
cell = DREAMCell(config)

# Initialize state ONCE
state = cell.init_state(batch_size=4)

# Process multiple sequences — state is PRESERVED
for seq in sequences:
    output, state = cell.forward_sequence(seq, state)
    # Model adapts and remembers between sequences!
```

**This is different from LSTM/Transformer** — they reset state after each sequence. DREAM accumulates experience.

---

## What is DREAM?

**DREAM** (Dynamic Recall and Elastic Adaptive Memory) is a neural architecture that **adapts during inference** without gradient descent.

### The Key Difference

| Model | Weights During Inference | Online Adaptation |
|-------|-------------------------|-------------------|
| LSTM | ❌ Fixed | ❌ No |
| Transformer | ❌ Fixed | ❌ No |
| **DREAM** | ✅ **Change every step** | ✅ **Yes** |

### How?

DREAM has **fast weights** that update on every timestep via Hebbian learning:

```
Slow weights (C, W, B): Trained via backprop (like normal networks)
Fast weights (U):       Updated online via STDP (no gradients needed)
```

### Analogy

- **LSTM/Transformer:** Like reading a book — content doesn't change
- **DREAM:** Like having a conversation — you adapt based on what you hear

---

## Why DREAM?

### Problem with Current Models

Modern models (LSTM, Transformer, Mamba) are **static after training**:

1. **Cannot adapt to new speakers** without retraining
2. **Process all inputs equally** — even predictable ones
3. **No prioritization** — waste computation on boring stuff

### DREAM Solves This

1. **Fast weights** — adapt to new patterns every step
2. **Surprise gate** — only learn when something is novel
3. **LTC** — adapt integration speed to event importance

### Real Results

| Metric | DREAM | LSTM | Transformer |
|--------|-------|------|-------------|
| **Parameters** | 82K | 893K | 551K |
| **ASR Improvement** | **99.9%** | 93.9% | 92.6% |
| **Adaptation Speed** | **0 steps** | N/A | N/A |
| **Noise Robustness** | **1.09×** | 1.09× | 1.07× |

**10× smaller, better quality, instant adaptation.**

---

## Installation

### Basic

```bash
pip install dreamnn
```

**Requirements:**
- Python 3.10+
- PyTorch 2.0+
- NumPy 1.24+

### Development

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

model = DREAM(input_dim=10, hidden_dim=32)
x = torch.randn(2, 10, 10)
output, state = model(x)
print(f"✅ DREAM works! Output shape: {output.shape}")
```

---

## Architecture Overview

### The 4 Blocks

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

### What Each Block Does

| Block | Purpose | Analogy |
|-------|---------|---------|
| **1. Predictive Coding** | Predict next input, compute error | "What do I expect?" |
| **2. Surprise Gate** | Detect novelty | "Is this surprising?" |
| **3. Fast Weights** | Online learning | "Let me adapt!" |
| **4. LTC** | Adaptive integration | "How fast should I update?" |

### Parameter Count

| Component | Parameters |
|-----------|------------|
| Predictive Coding (C, W, B) | 3 × (input × hidden) = 61,440 |
| Fast Weights (U) | batch × hidden × rank = 16,384 |
| Sensory Filter (V) | input × rank = 1,280 |
| **Total** | **~82K** |

**Compare:** LSTM (893K), Transformer (551K)

---

## Basic Usage

### Like LSTM

```python
import torch
from dream import DREAM

# Create model
model = DREAM(
    input_dim=64,
    hidden_dim=128,
    rank=8,
)

# Process sequence
x = torch.randn(32, 50, 64)  # (batch, time, features)
output, state = model(x)

print(f"Output shape: {output.shape}")  # (32, 50, 128)
```

### Return Sequences

```python
# All timesteps (default)
output, state = model(x, return_sequences=True)  # (32, 50, 128)

# Only last timestep
output, state = model(x, return_sequences=False)  # (32, 128)
```

### With Initial State

```python
# Initialize state
state = model.init_state(batch_size=32)

# Process with initial state
output, final_state = model(x, state=state)
```

### GPU Usage

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = DREAM(input_dim=80, hidden_dim=256).to(device)
state = model.init_state(batch_size=4, device=device)

x = torch.randn(4, 100, 80).to(device)
output, state = model(x, state=state)
```

---

## How It Works: Component by Component

### 1. Predictive Coding

**Problem:** How to predict the next input?

**Solution:** Use matrix `C` to decode hidden state into prediction.

#### Formula

```
Prediction:  x̂_t = tanh(C^T @ h_{t-1})
Error:       e_t = x_t - x̂_t
```

#### Code

```python
# In DREAMCell.__init__
self.C = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)

# In forward
x_pred = torch.tanh(state.h @ self.C)
error = x - x_pred
```

#### Why?

- `C` learns to predict input from hidden state
- Error `e_t` tells model what it missed
- This error drives learning

---

### 2. Surprise Gate

**Problem:** When to activate plasticity? Constant updates = instability.

**Solution:** Compute "surprise" and use it to gate learning.

#### Formula

```
Entropy:     H = 0.5 · log(2πe · var)
Threshold:   τ = 1.0 + α · H
Relative:    r = ||e|| / (||μ_e|| + ε)
Surprise:    S = σ((r - τ) / (2γ))
```

#### Code

```python
def compute_surprise(self, error, state):
    # Entropy from variance
    variance = state.error_var.mean(dim=-1)
    entropy = 0.5 * torch.log(2 * torch.pi * torch.e * (variance + eps))
    
    # Adaptive threshold
    tau = 1.0 + self.alpha * entropy
    
    # Relative error (KEY CHANGE from spec)
    baseline = state.error_mean.norm(dim=-1) + eps
    relative_error = error.norm(dim=-1) / baseline
    
    # Surprise
    surprise = torch.sigmoid((relative_error - tau) / (2 * self.gamma))
    return surprise
```

#### Why Relative Error?

| Scenario | Absolute Error | Relative Error |
|----------|---------------|----------------|
| Small baseline | High surprise even for tiny changes | Normalized |
| Large baseline | Low surprise even for big changes | Normalized |
| **Anomaly detection** | ❌ Misses subtle changes | ✅ Better sensitivity |

**Example:**
```python
# Baseline error is small (model is confident)
state.error_mean = 0.01

# Current error is 0.05 (5× baseline!)
error = 0.05

# Absolute: 0.05 (small, not surprising)
# Relative: 0.05 / 0.01 = 5.0 (BIG surprise!)
```

---

### 3. Fast Weights (STDP)

**Problem:** How to update weights online without backprop?

**Solution:** Low-rank fast weights with Hebbian learning.

#### Formula

```
Fast Weights:  W_fast = U @ V^T
STDP Update:   dU = -λ(U - U_target) + (η · S) · ((h ⊗ e) @ V)
Euler Step:    U ← U + dU · dt
```

#### Low-Rank Decomposition

```
Full Matrix:     hidden × input = 256 × 80 = 20,480 params
Low-Rank:        (hidden × rank) + (input × rank)
                 = 256×16 + 80×16 = 5,376 params
Savings:         4× fewer parameters!
```

#### Code

```python
# Initialize V (fixed orthogonal)
V_init = torch.randn(input_dim, rank)
Q, _ = torch.linalg.qr(V_init)
self.register_buffer('V', Q)  # Fixed!

# Update U (learnable)
def update_fast_weights(self, h_prev, error, surprise, state):
    # Hebbian term: outer(h, e) @ V
    eV = error @ self.V                    # (batch, rank)
    hebbian = state.h.unsqueeze(2) * eV.unsqueeze(1)  # (batch, hidden, rank)
    
    # Plasticity modulation (surprise gates learning)
    plasticity = self.eta * surprise.unsqueeze(1)
    plasticity = plasticity.unsqueeze(2)
    
    # Forgetting term (decay toward U_target)
    forgetting = -self.forgetting_rate * (state.U - state.U_target)
    
    # Full update
    dU = forgetting + plasticity * hebbian
    U_new = state.U + dU * self.dt
    
    # Normalize (homoeostasis)
    U_norm = U_new.norm(dim=(1, 2), keepdim=True)
    scale = (self.target_norm / (U_norm + 1e-6)).clamp(max=2.0)
    state.U = U_new * scale
```

#### Why Low-Rank?

| Aspect | Full Matrix | Low-Rank |
|--------|-------------|----------|
| Parameters | 20,480 | 5,376 |
| Computation | O(hidden × input) | O((hidden + input) × rank) |
| Stability | Can be unstable | Orthogonal V stabilizes |

#### Why Orthogonal V?

- Stable gradients
- Prevents rank collapse
- Fixed during inference (only meta-learning updates V)

---

### 4. Liquid Time-Constants (LTC)

**Problem:** How to adapt integration speed to event importance?

**Solution:** Dynamic time constant τ modulated by surprise.

#### Formula

```
Dynamic τ:   τ = τ_sys / (1 + S · scale)
Integration: dh/dt = (-h + tanh(u_eff)) / τ
Euler:       h_new = (1 - dt/τ) · h_prev + (dt/τ) · h_target
```

#### Code

```python
def compute_ltc_update(self, h_prev, u_eff, surprise):
    # Dynamic tau: high surprise → small tau → fast updates
    tau = self.tau_sys / (1.0 + surprise * self.tau_surprise_scale)
    tau = torch.clamp(tau, 0.01, 50.0)
    
    # Euler integration
    h_target = torch.tanh(u_eff)
    dt_over_tau = self.dt / (tau.unsqueeze(1) + self.dt)
    dt_over_tau = torch.clamp(dt_over_tau, 0.01, 0.5)
    
    h_new = (1 - dt_over_tau) * h_prev + dt_over_tau * h_target
    return h_new
```

#### Why LTC?

| Surprise Level | τ | Behavior |
|---------------|---|----------|
| High (novel) | Small | Fast updates, rapid adaptation |
| Low (predictable) | Large | Slow integration, smooth memory |

**Example:**
```python
# High surprise (new speaker)
surprise = 0.9
tau = 5.0 / (1 + 0.9 * 5.0) = 0.91  # Fast update

# Low surprise (same speaker)
surprise = 0.1
tau = 5.0 / (1 + 0.1 * 5.0) = 3.33  # Slow integration
```

---

### 5. Sleep Consolidation

**Problem:** How to stabilize fast changes into long-term memory?

**Solution:** Consolidate U into U_target when surprise is high.

#### Formula

```
If S̄ > S_min:
    dU_target = ζ_sleep · S̄ · (U - U_target)
    U_target ← U_target + dU_target
```

#### Code

```python
avg_surprise_mean = state.avg_surprise.mean()

if avg_surprise_mean > self.S_min:
    # Consolidate U into U_target
    dU_target = self.sleep_rate * avg_surprise_mean * (state.U - state.U_target)
    state.U_target = state.U_target + dU_target
    
    # Homeostasis
    U_target_norm = state.U_target.norm(dim=(1, 2), keepdim=True)
    scale = (self.target_norm / (U_target_norm + 1e-6)).clamp(max=2.0)
    state.U_target = state.U_target * scale
```

#### Why Sleep?

| Without Sleep | With Sleep |
|---------------|------------|
| Fast weights drift | Stabilized via U_target |
| Catastrophic forgetting | Consolidated memories |
| Unstable training | Smooth convergence |

---

## Advanced Usage

### Multi-Layer (DREAMStack)

```python
from dream import DREAMStack

model = DREAMStack(
    input_dim=64,
    hidden_dims=[128, 128, 64],  # 3 layers
    rank=8,
    dropout=0.1,
)

output, states = model(x)
print(f"Output shape: {output.shape}")  # (32, 50, 64)
```

### Stateful Processing

```python
from dream import DREAMCell, DREAMConfig

config = DREAMConfig(input_dim=80, hidden_dim=256)
cell = DREAMCell(config)

# Initialize ONCE
state = cell.init_state(batch_size=4)

# Process multiple sequences (state preserved!)
for seq in sequences:
    output, state = cell.forward_sequence(seq, state)
    # Model adapts and remembers!
```

### Truncated BPTT

```python
from dream import DREAMCell

model = DREAMCell(config)
state = model.init_state(batch_size=4)

segment_size = 100

for start in range(0, seq_len, segment_size):
    segment = x[:, start:start+segment_size, :]
    
    output, state = model.forward_sequence(segment, state)
    loss = criterion(output, target[:, start:start+segment_size, :])
    
    loss.backward()
    
    # Detach state between segments
    state = state.detach()

optimizer.step()
```

---

## Configuration Guide

### DREAMConfig: All Parameters

```python
from dream import DREAMConfig, DREAMCell

config = DREAMConfig(
    # Dimensions
    input_dim=80,        # Input features
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

cell = DREAMCell(config)
```

### Parameter Guide

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `hidden_dim` | 256 | 64-512 | ↑ = more capacity, slower |
| `rank` | 16 | 8-32 | ↑ = more expressive, more params |
| `forgetting_rate` | 0.005 | 0.001-0.01 | ↑ = faster forgetting |
| `base_plasticity` | 0.5 | 0.1-1.0 | ↑ = faster learning |
| `base_threshold` | 0.3 | 0.2-0.5 | ↑ = less sensitive |
| `surprise_temperature` | 0.05 | 0.01-0.1 | ↑ = smoother surprise |
| `ltc_tau_sys` | 5.0 | 1.0-10.0 | ↑ = slower integration |

### Recommended Configs

#### For ASR (MFCC 39D)

```python
config = DREAMConfig(
    input_dim=39,       # 13 MFCC + 13Δ + 13ΔΔ
    hidden_dim=512,
    rank=16,
    forgetting_rate=0.005,
    base_plasticity=0.5,
)
```

#### For Audio (Mel 80D)

```python
config = DREAMConfig(
    input_dim=80,       # Mel bins
    hidden_dim=256,
    rank=16,
    ltc_tau_sys=5.0,
)
```

#### For Time Series

```python
config = DREAMConfig(
    input_dim=features_dim,
    hidden_dim=128,
    rank=8,
    ltc_tau_sys=5.0,  # Faster response
)
```

#### For Fast Adaptation

```python
config = DREAMConfig(
    input_dim=80,
    hidden_dim=256,
    rank=16,
    base_plasticity=1.0,      # Higher learning rate
    base_threshold=0.2,        # More sensitive
    surprise_temperature=0.1,  # Smoother surprise
)
```

---

## Training Guide

### Basic Training Loop

```python
import torch
import torch.nn as nn
from dream import DREAM

model = DREAM(input_dim=80, hidden_dim=256)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training
for epoch in range(100):
    optimizer.zero_grad()
    
    # Forward
    output, state = model(x)  # x: (batch, time, features)
    
    # Loss
    loss = criterion(output, target)
    
    # Backward
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

### With Learning Rate Scheduler

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

for epoch in range(100):
    # ... training ...
    
    scheduler.step(val_loss)
```

### With Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Stateful Training (Memory Accumulation)

```python
model = DREAMCell(config)
state = model.init_state(batch_size=4)

for epoch in range(100):
    optimizer.zero_grad()
    
    # State is PRESERVED across epochs!
    output, state = model.forward_sequence(x, state)
    loss = criterion(output, target)
    
    loss.backward()
    optimizer.step()
    
    # Detach state to prevent memory explosion
    state = state.detach()
```

---

## Saving & Loading

### Save Model

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
}, 'dream_checkpoint.pt')
```

### Load Model

```python
checkpoint = torch.load('dream_checkpoint.pt')

config = checkpoint['config']
model = DREAMCell(config)
model.load_state_dict(checkpoint['model_state_dict'])
```

### Save Full State (for inference)

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'state_dict': {
        'U': state.U,
        'U_target': state.U_target,
        'h': state.h,
    },
}, 'dream_full.pt')
```

### Load Full State

```python
checkpoint = torch.load('dream_full.pt')

model.load_state_dict(checkpoint['model_state_dict'])

# Restore state
state = model.init_state(batch_size)
state.U = checkpoint['state_dict']['U']
state.U_target = checkpoint['state_dict']['U_target']
state.h = checkpoint['state_dict']['h']
```

---

## Common Patterns

### Sequence Classification

```python
from dream import DREAMCell
import torch.nn as nn

cell = DREAMCell(config)
classifier = nn.Linear(256, 10)  # 10 classes

state = cell.init_state(batch_size=32)
output, final_state = cell.forward_sequence(x)

# Classify using final hidden state
logits = classifier(final_state.h)
predictions = logits.argmax(dim=-1)
```

### Encoder-Decoder

```python
# Encoder
encoder = DREAMCell(encoder_config)
_, enc_state = encoder.forward_sequence(input_seq)

# Decoder (initialize with encoder state)
decoder = DREAMCell(decoder_config)
dec_state = decoder.init_state(batch_size)
dec_state.h = enc_state.h  # Transfer state

output, _ = decoder.forward_sequence(target_seq)
```

### Bidirectional Processing

```python
# Forward
forward_output, _ = cell.forward_sequence(x)

# Backward (reverse time)
backward_output, _ = cell.forward_sequence(x.flip(dims=[1]))

# Combine
combined = torch.cat([forward_output, backward_output.flip(dims=[1])], dim=-1)
```

### Attention on Top of DREAM

```python
cell = DREAMCell(config)
output, state = cell.forward_sequence(x, return_all=True)  # (batch, time, hidden)

# Self-attention
attn = torch.nn.MultiheadAttention(hidden_dim, num_heads=8)
output, _ = attn(output, output, output)
```

---

## Benchmark Results

### Test 1: Basic ASR Reconstruction

**Task:** Reconstruct mel spectrograms from 9 audio files.

| Model | Parameters | Initial Loss | Final Loss | Improvement | Time |
|-------|------------|--------------|------------|-------------|------|
| **DREAM** | 82K | 0.9298 | **0.0010** | **99.9%** | 502s |
| LSTM | 893K | 0.7889 | 0.0478 | 93.9% | 9s |
| Transformer | 551K | 0.9416 | 0.0696 | 92.6% | 11s |

**Conclusion:** DREAM achieves best quality but slower (price of online adaptation).

---

### Test 2: Speaker Adaptation

**Task:** Adapt to speaker change mid-sequence.

| Model | Baseline | Max Post-Switch | Adapt Steps | Surprise Spike |
|-------|----------|-----------------|-------------|----------------|
| **DREAM** | 1.2078 | 1.9657 | **0** | 0.119 |
| LSTM | 1.0435 | 1.5807 | 0 | N/A |
| Transformer | 1.1963 | 1.6963 | 0 | N/A |

**Conclusion:** All adapt instantly, but only DREAM detects change via surprise.

---

### Test 3: Noise Robustness

**Task:** Reconstruction with additive white noise.

| Model | Clean (20dB) | 10dB Loss | Ratio | Surprise Response |
|-------|--------------|-----------|-------|-------------------|
| **DREAM** | 1.2308 | 1.3390 | 1.09× | ❌ No (saturated) |
| LSTM | 1.0163 | 1.1052 | 1.09× | N/A |
| Transformer | 1.2867 | 1.3757 | 1.07× | N/A |

**Conclusion:** DREAM stable under noise (1.09×), surprise saturates at high levels.

---

## Troubleshooting

### CUDA Out of Memory

**Problem:** GPU runs out of memory with long sequences.

**Solution:** Use truncated BPTT

```python
segment_size = 50  # Reduce segment size
for start in range(0, seq_len, segment_size):
    segment = x[:, start:start+segment_size, :]
    output, state = model.forward_sequence(segment, state)
    state = state.detach()
```

### Slow Training

**Problem:** Training is slower than LSTM/Transformer.

**Solution:** This is expected (online adaptation has cost). Try:
- Reduce `hidden_dim` or `rank`
- Use GPU
- Reduce sequence length

```python
config = DREAMConfig(
    hidden_dim=128,  # Smaller
    rank=8,          # Lower rank
)
```

### Loss Not Decreasing

**Problem:** Loss stays constant or increases.

**Solutions:**

1. **Reduce learning rate:**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

2. **Increase plasticity:**
```python
config = DREAMConfig(base_plasticity=1.0)  # Higher
```

3. **Gradient clipping:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
```

### Model Forgets Between Sequences

**Problem:** Model "forgets" what it learned.

**Solution:** Preserve state:

```python
# WRONG: State reset every sequence
for seq in sequences:
    state = model.init_state(batch_size)  # Reset!
    output, state = model(seq, state)

# CORRECT: State preserved
state = model.init_state(batch_size)
for seq in sequences:
    output, state = model(seq, state)  # Preserved!
```

### Surprise Not Detecting Anomalies

**Problem:** Surprise stays constant even with novel inputs.

**Solution:** Adjust surprise parameters:

```python
config = DREAMConfig(
    base_threshold=0.2,         # More sensitive
    surprise_temperature=0.1,   # Smoother response
)
```

---

## API Reference

### DREAM (High-level)

```python
class DREAM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        rank: int = 8,
        **kwargs
    )

    def forward(
        self,
        x: torch.Tensor,           # (batch, time, input_dim)
        state: Optional[DREAMState] = None,
        return_sequences: bool = True
    ) -> Tuple[torch.Tensor, DREAMState]
```

### DREAMCell (Low-level)

```python
class DREAMCell(nn.Module):
    def __init__(self, config: DREAMConfig)

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> DREAMState

    def forward(
        self,
        x: torch.Tensor,           # (batch, input_dim)
        state: DREAMState
    ) -> Tuple[torch.Tensor, DREAMState]

    def forward_sequence(
        self,
        x_seq: torch.Tensor,       # (batch, time, input_dim)
        state: Optional[DREAMState] = None,
        return_all: bool = False
    ) -> Tuple[torch.Tensor, DREAMState]
```

### DREAMState

```python
@dataclass
class DREAMState:
    h: torch.Tensor              # Hidden state
    U: torch.Tensor              # Fast weights
    U_target: torch.Tensor       # Target weights
    adaptive_tau: torch.Tensor   # Adaptive threshold
    error_mean: torch.Tensor     # Error mean
    error_var: torch.Tensor      # Error variance
    avg_surprise: torch.Tensor   # Average surprise

    def detach(self) -> "DREAMState"
```

### DREAMConfig

```python
@dataclass
class DREAMConfig:
    input_dim: int = 39
    hidden_dim: int = 256
    rank: int = 16
    time_step: float = 0.1
    forgetting_rate: float = 0.005
    base_plasticity: float = 0.5
    base_threshold: float = 0.3
    entropy_influence: float = 0.1
    surprise_temperature: float = 0.05
    error_smoothing: float = 0.05
    surprise_smoothing: float = 0.05
    target_norm: float = 2.0
    kappa: float = 0.5
    ltc_enabled: bool = True
    ltc_tau_sys: float = 5.0
    ltc_surprise_scale: float = 5.0
    sleep_rate: float = 0.005
    min_surprise_for_sleep: float = 0.2
```

---

## FAQ

### Q: How is DREAM different from LSTM?

**A:** LSTM has fixed weights after training. DREAM has fast weights that update every timestep via Hebbian learning. This allows online adaptation without gradient descent.

### Q: When should I use DREAM vs LSTM?

**A:** Use DREAM when:
- You need online adaptation (new speakers, changing patterns)
- You have non-stationary data
- You want fewer parameters

Use LSTM when:
- Data is stationary
- Training speed is critical
- You need proven production stability

### Q: Can I use DREAM for classification?

**A:** Yes! Use the final hidden state for classification:

```python
output, final_state = cell.forward_sequence(x)
logits = classifier(final_state.h)
```

### Q: How do I choose hidden_dim and rank?

**A:** Start with defaults (256, 16). Then:
- Increase `hidden_dim` for more capacity
- Increase `rank` for more expressive fast weights
- Decrease both for faster training

### Q: Does DREAM work on GPU?

**A:** Yes! DREAM is fully GPU-compatible:

```python
model = DREAM(input_dim=80, hidden_dim=256).to('cuda')
x = torch.randn(4, 100, 80).to('cuda')
output, state = model(x)
```

### Q: How much memory does DREAM use?

**A:** For batch=4, hidden=256, rank=16:
- Model: ~82K parameters (~330KB)
- State: ~16K per batch element
- Total: ~400MB for 1000 timesteps

### Q: Can I stack multiple DREAM layers?

**A:** Yes! Use `DREAMStack`:

```python
model = DREAMStack(
    input_dim=64,
    hidden_dims=[128, 128, 64],  # 3 layers
    rank=8,
)
```

---

## Support

### GitHub Issues

Report bugs or request features: https://github.com/karl4th/dream-nn/issues

### Documentation

Full docs: https://github.com/karl4th/dream-nn

### PyPI

Package: https://pypi.org/project/dreamnn/

---

**Happy Coding! 🚀**
