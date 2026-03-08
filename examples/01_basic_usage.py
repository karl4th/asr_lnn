"""
Example 1: Basic DREAM Usage

Get started with DREAM in 5 minutes.
Shows basic sequence processing with static base training.
"""

import torch
from dream import DREAM, DREAMConfig, DREAMCell

print("=" * 60)
print("DREAM Example 1: Basic Usage")
print("=" * 60)

# ============================================================
# Option 1: High-level API (like nn.LSTM)
# ============================================================
print("\n1. High-level API (DREAM class)")
print("-" * 60)

model = DREAM(
    input_dim=64,      # Input features
    hidden_dim=128,    # Hidden size
    rank=8,            # Fast weights rank
)

print(f"Model created: {model.count_parameters() if hasattr(model, 'count_parameters') else 'N/A'} parameters")

# Process sequence
x = torch.randn(4, 50, 64)  # (batch, time, features)
output, state = model(x)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}")  # (batch, time, hidden_dim)


# ============================================================
# Option 2: Low-level API (DREAMCell)
# ============================================================
print("\n2. Low-level API (DREAMCell)")
print("-" * 60)

config = DREAMConfig(
    input_dim=64,
    hidden_dim=128,
    rank=8,
    forgetting_rate=0.005,
    base_plasticity=0.5,
)

cell = DREAMCell(config)
state = cell.init_state(batch_size=4)

# Process single timestep
x_t = torch.randn(4, 64)
h_new, state = cell(x_t, state)

print(f"Timestep input:  {x_t.shape}")
print(f"Hidden state:    {h_new.shape}")


# Process full sequence
x_seq = torch.randn(4, 50, 64)
output, final_state = cell.forward_sequence(x_seq, state, return_all=True)

print(f"Sequence input:  {x_seq.shape}")
print(f"Sequence output: {output.shape}")


# ============================================================
# Static Base Training (fast weights frozen)
# ============================================================
print("\n3. Static Base Training")
print("-" * 60)

# Create model with frozen fast weights for training
model = DREAM(
    input_dim=64,
    hidden_dim=128,
    rank=8,
    freeze_fast_weights=True,  # Freeze during training
)

# Training mode = fast weights frozen
model.train()
print(f"Training mode: fast weights frozen = {model.freeze_fast_weights}")

# Simulate training
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

x = torch.randn(4, 50, 64)
target = torch.randn(4, 50, 128)

model.train()
output, _ = model(x)
loss = criterion(output, target)
loss.backward()
optimizer.step()

print(f"Training loss: {loss.item():.4f}")


# ============================================================
# Inference with Active Fast Weights
# ============================================================
print("\n4. Inference with Active Fast Weights")
print("-" * 60)

# Switch to inference mode = fast weights active
model.eval()
print(f"Inference mode: fast weights active = {not model.freeze_fast_weights}")

# Now fast weights adapt during inference
with torch.no_grad():
    output, state = model(x)
    print(f"Inference output: {output.shape}")


# ============================================================
# Key Takeaway
# ============================================================
print("\n" + "=" * 60)
print("Key Takeaway:")
print("=" * 60)
print("""
During Training (model.train()):
  - Fast weights FROZEN
  - Only slow weights (C, W, B) learn via backprop
  - Stable base training

During Inference (model.eval()):
  - Fast weights ACTIVE
  - Model adapts online via STDP
  - Captures sequence-specific patterns
""")
