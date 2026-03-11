# Coordinated DREAMStack

**Hierarchical Coordination with Top-Down Modulation**

---

## Overview

Standard DREAMStack has independent layers — each adapts separately. Coordinated DREAMStack adds:

1. **Bottom-up prediction errors** — Lower layers send errors upward
2. **Top-down modulation** — Upper layers modulate lower layer sensitivity
3. **Two-pass processing** — Feedforward + backward modulation per timestep
4. **Hierarchical sleep** — Consolidation across all layers together

---

## Architecture

```
                    Timestep t
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌───────────────────────────────────────────────┐
│  Layer 3 (Top)                                │
│  - Integrates long-term context               │
│  - Generates modulation for Layer 2           │
│  - Predicts Layer 2 activity                  │
└───────────────────┬───────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
  Modulation (↓)          Prediction (↓)
        │                       │
        ▼                       ▼
┌───────────────────────────────────────────────┐
│  Layer 2 (Middle)                             │
│  - Integrates medium-term patterns            │
│  - Receives modulation from Layer 3           │
│  - Compares prediction with Layer 1 output    │
└───────────────────┬───────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
  Modulation (↓)          Prediction (↓)
        │                       │
        ▼                       ▼
┌───────────────────────────────────────────────┐
│  Layer 1 (Bottom)                             │
│  - Processes fast patterns                    │
│  - Receives modulation from Layer 2           │
│  - Sends prediction error upward              │
└───────────────────────────────────────────────┘
```

---

## Usage

### Basic

```python
from dream import CoordinatedDREAMStack

model = CoordinatedDREAMStack(
    input_dim=80,
    hidden_dims=[128, 128, 128, 128],  # 4 layers
    rank=16,
    dropout=0.1,
)

# Forward pass
output, states, coord_info = model(x, return_all=True)

# Access coordination info
modulations = coord_info['modulations']  # Top-down modulations
inter_errors = coord_info['inter_layer_errors']  # Prediction errors
```

### With Global Sleep

```python
output, states, global_surprise = model.forward_with_global_sleep(x)

# All layers consolidate together if global_surprise is high
if global_surprise > 0.5:
    print("Hierarchical consolidation triggered!")
```

### Comparison: Coordinated vs Uncoordinated

```python
from dream import CoordinatedDREAMStack, UncoordinatedDREAMStack

# Uncoordinated (baseline)
model_uncoord = UncoordinatedDREAMStack(
    input_dim=80,
    hidden_dims=[128, 128, 128],
)

# Coordinated (with hierarchy)
model_coord = CoordinatedDREAMStack(
    input_dim=80,
    hidden_dims=[128, 128, 128],
)
```

---

## Key Features

### 1. Top-Down Modulation

Upper layers modulate lower layer sensitivity:

```python
# In DREAMCell with coordination
modulation = layer.generate_modulation(h)  # (batch, hidden_dim)

# Applied to Surprise Gate threshold
threshold = base_threshold - kappa * modulation.mean()
# Higher modulation → lower threshold → more sensitive
```

### 2. Inter-Layer Prediction

Upper layers predict lower layer activity:

```python
# Upper layer predicts lower layer
prediction = layer.predict_lower_activity(h)

# Compare with actual
error = layer.compute_inter_layer_error(prediction, actual)

# Error used for learning
```

### 3. Hierarchical Sleep

All layers consolidate together:

```python
output, states, global_surprise = model.forward_with_global_sleep(x)

# If global_surprise > threshold:
#   All layers consolidate U → U_target together
```

---

## Benchmarks

### Test 4: Stack Coordination

Compares coordinated vs uncoordinated:

```bash
uv run python tests/benchmarks/test_04_stack_coordination.py
```

**Metrics:**
- Convergence speed
- Final reconstruction loss
- Training stability

**Expected:**
- Coordinated converges faster
- Coordinated more stable with depth

### Test 5: Hierarchical Processing

Tests temporal hierarchy:

```bash
uv run python tests/benchmarks/test_05_hierarchy.py
```

**Metrics:**
- Adaptation speed per layer
- LTC tau per layer
- Hierarchy emergence

**Expected:**
- Lower layers: Fast adaptation (short τ)
- Upper layers: Slow integration (long τ)

---

## When to Use

### Use Coordinated DREAMStack When:
- Deep stacks (4+ layers)
- Need hierarchical temporal processing
- Want better stability
- Can afford ~10% more parameters

### Use Uncoordinated DREAMStack When:
- Shallow stacks (2-3 layers)
- Need minimal overhead
- Simplicity preferred

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_dim` | 80 | Input dimension |
| `hidden_dims` | [128, 128, 128, 128] | Hidden dimensions per layer |
| `rank` | 16 | Fast weights rank |
| `dropout` | 0.1 | Dropout between layers |

---

## Example: Deep Stack

```python
# 6-layer coordinated stack
model = CoordinatedDREAMStack(
    input_dim=80,
    hidden_dims=[256, 256, 256, 256, 128, 128],
    rank=16,
    dropout=0.1,
)

# Process long sequence
x = torch.randn(4, 1500, 80)  # 30 seconds at 50fps
output, states, coord_info = model(x)

print(f"Output shape: {output.shape}")
print(f"Modulations: {len(coord_info['modulations'])} layers")
```

---

## Citation

```bibtex
@software{dream-coord2026,
  title = {Coordinated DREAMStack: Hierarchical Coordination},
  author = {Manifestro Team},
  year = {2026},
  url = {https://github.com/karl4th/dream-nn},
  version = "0.1.3"
}
```
