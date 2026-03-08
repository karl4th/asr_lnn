# DREAM Examples

Practical examples showing how to use DREAM (Dynamic Recall and Elastic Adaptive Memory).

## Quick Start

```bash
# Run any example
python examples/01_basic_usage.py
python examples/02_stateful_processing.py
python examples/03_online_adaptation.py
python examples/04_training.py
python examples/05_long_sequences.py
```

## Examples Overview

### 1. Basic Usage (`01_basic_usage.py`)

Get started with DREAM in 5 minutes.

**Covers:**
- High-level API (`DREAM` class)
- Low-level API (`DREAMCell`)
- Static base training vs inference
- Fast weights freezing/unfreezing

**Run:**
```bash
python examples/01_basic_usage.py
```

---

### 2. Stateful Processing (`02_stateful_processing.py`)

Preserve memory across multiple sequences.

**Covers:**
- State initialization and preservation
- Stateful vs stateless processing
- Long document chunking
- Multi-turn conversations

**Run:**
```bash
python examples/02_stateful_processing.py
```

---

### 3. Online Adaptation (`03_online_adaptation.py`)

Adapt during inference without gradient updates.

**Covers:**
- Speaker/pattern adaptation
- DREAM vs LSTM comparison
- Real-time personalization
- Fast weights dynamics

**Run:**
```bash
python examples/03_online_adaptation.py
```

---

### 4. Training (`04_training.py`)

Complete training guide with two-phase approach.

**Covers:**
- Static base training (fast weights frozen)
- Adaptation phase (fast weights active)
- Manual mode switching
- Multi-epoch training with state persistence

**Run:**
```bash
python examples/04_training.py
```

---

### 5. Long Sequences (`05_long_sequences.py`)

Process sequences longer than memory capacity.

**Covers:**
- Truncated BPTT
- Variable length sequences
- Streaming processing
- Memory efficiency

**Run:**
```bash
python examples/05_long_sequences.py
```

---

## Key Concepts

### Static Base Training vs Online Adaptation

```python
# Training: Fast weights FROZEN
model.train()  # or model.set_fast_weights_mode(freeze=True)
# Only slow weights (C, W, B, eta) learn via backprop

# Inference: Fast weights ACTIVE
model.eval()  # or model.set_fast_weights_mode(freeze=False)
# Fast weights adapt online via STDP
```

### Stateful Processing

```python
# Initialize state ONCE
state = model.init_state(batch_size=4)

# Preserve across sequences
for seq in sequences:
    output, state = model(seq, state=state)  # State preserved!
```

### Truncated BPTT

```python
state = model.init_state(batch_size)

for start in range(0, seq_len, segment_size):
    segment = x[:, start:start+segment_size, :]
    output, state = model.forward_sequence(segment, state=state)
    loss.backward()
    state = state.detach()  # Truncate BPTT!
```

---

## Common Patterns

### Pattern 1: Basic Sequence Processing

```python
from dream import DREAM

model = DREAM(input_dim=64, hidden_dim=128, rank=8)
x = torch.randn(4, 50, 64)
output, state = model(x)
```

### Pattern 2: Multi-Epoch Training

```python
from dream import DREAMCell

model = DREAMCell(config)
state = model.init_state(batch_size)

for epoch in range(100):
    output, state = model.forward_sequence(x, state=state)
    loss.backward()
    optimizer.step()
    state = state.detach()
```

### Pattern 3: Online Adaptation

```python
model.eval()  # Unfreeze fast weights
state = model.init_state(batch_size)

for seq in sequences:
    output, state = model(seq, state=state)
    # Fast weights adapt to each sequence!
```

---

## Troubleshooting

### CUDA Out of Memory

Use truncated BPTT with smaller segments:

```python
segment_size = 50  # Reduce from 100
```

### Slow Training

Ensure fast weights are frozen during training:

```python
model.train()  # Auto-freezes fast weights
# or
model.set_fast_weights_mode(freeze=True)
```

### Model Forgets Between Sequences

Preserve state:

```python
state = model.init_state(batch_size)
for seq in sequences:
    output, state = model(seq, state=state)  # Pass state!
```

---

## Next Steps

After running these examples:

1. **Read Documentation**: `docs/COMPLETE_GUIDE.md`
2. **Try ASR Training**: `train_asr.py` (for speech recognition)
3. **Run Benchmarks**: `tests/benchmarks/run_all.py`

---

## Citation

```bibtex
@software{dream2026,
  title = {DREAM: Dynamic Recall and Elastic Adaptive Memory},
  author = {Manifestro Team},
  year = {2026},
  url = {https://github.com/karl4th/dream-nn},
  version = "0.1.2"
}
```
