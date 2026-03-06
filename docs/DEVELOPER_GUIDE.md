# DREAM Developer Documentation

**Dynamic Recall and Elastic Adaptive Memory**

**Version:** 0.1.2  
**For:** Developers integrating DREAM into their projects

---

## Quick Start

### Installation

```bash
pip install dreamnn
```

**Requirements:**
- Python 3.10+
- PyTorch 2.0+
- NumPy 1.24+

### 5-Minute Example

```python
import torch
from dream import DREAM

# Create model (like nn.LSTM)
model = DREAM(
    input_dim=80,      # Input features
    hidden_dim=256,    # Hidden size
    rank=16,           # Fast weights rank
)

# Process sequence
x = torch.randn(4, 100, 80)  # (batch, time, features)
output, state = model(x)

print(f"Output shape: {output.shape}")  # (4, 100, 256)
```

---

## Table of Contents

1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Advanced Usage](#advanced-usage)
4. [Configuration](#configuration)
5. [Training](#training)
6. [Saving & Loading](#saving--loading)
7. [Common Patterns](#common-patterns)
8. [Troubleshooting](#troubleshooting)
9. [API Reference](#api-reference)

---

## 1. Installation

### Basic

```bash
pip install dreamnn
```

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

---

## 2. Basic Usage

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

---

## 3. Advanced Usage

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

Preserve memory across multiple sequences:

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

### GPU Acceleration

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = DREAM(input_dim=80, hidden_dim=256).to(device)
state = model.init_state(batch_size=4, device=device)

x = torch.randn(4, 100, 80).to(device)
output, state = model(x, state=state)
```

---

## 4. Configuration

### DREAMConfig

```python
from dream import DREAMConfig, DREAMCell

config = DREAMConfig(
    # Dimensions
    input_dim=80,        # Input features
    hidden_dim=256,      # Hidden state size
    rank=16,             # Fast weights rank
    
    # Plasticity
    forgetting_rate=0.005,    # Decay rate (λ)
    base_plasticity=0.5,      # Learning rate (η)
    
    # Surprise
    base_threshold=0.3,       # Threshold (τ₀)
    entropy_influence=0.1,    # Entropy effect (α)
    surprise_temperature=0.05, # Sensitivity (γ)
    
    # LTC
    ltc_enabled=True,
    ltc_tau_sys=5.0,
    ltc_surprise_scale=5.0,
    
    # Smoothing
    error_smoothing=0.05,
    surprise_smoothing=0.05,
)

cell = DREAMCell(config)
```

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

---

## 5. Training

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

### Truncated BPTT (Long Sequences)

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

### With Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 6. Saving & Loading

### Save Model

```python
# Save state dict
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
}, 'dream_checkpoint.pt')
```

### Load Model

```python
checkpoint = torch.load('dream_checkpoint.pt')

# Recreate model
config = checkpoint['config']
model = DREAMCell(config)
model.load_state_dict(checkpoint['model_state_dict'])
```

### Save Full State (for inference)

```python
# Save model + fast weights
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
state.U = checkpoint['state_dict']['U']
state.U_target = checkpoint['state_dict']['U_target']
state.h = checkpoint['state_dict']['h']
```

---

## 7. Common Patterns

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
attn_output = torch.nn.MultiheadAttention(hidden_dim, num_heads=8)
output, _ = attn_output(output, output, output)
```

---

## 8. Troubleshooting

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

### State Management Issues

**Problem:** Model "forgets" between sequences.

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

---

## 9. API Reference

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

## Support

### GitHub Issues

Report bugs or request features: https://github.com/karl4th/dream-nn/issues

### Documentation

Full documentation: https://github.com/karl4th/dream-nn/blob/main/README.md

### PyPI

Package page: https://pypi.org/project/dreamnn/

---

**Happy Coding! 🚀**
