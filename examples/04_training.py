"""
Example 4: Training DREAM

Complete training example with static base training phase.
Shows two-phase training: static base → adaptation.
"""

import torch
import torch.nn as nn
from dream import DREAM, DREAMConfig, DREAMCell

print("=" * 60)
print("DREAM Example 4: Training")
print("=" * 60)

# ============================================================
# Setup
# ============================================================
print("\n1. Setup")
print("-" * 60)

# Create model
model = DREAM(
    input_dim=64,
    hidden_dim=128,
    rank=8,
    freeze_fast_weights=True,  # Start with frozen fast weights
)

print(f"Model: DREAM(input_dim=64, hidden_dim=128, rank=8)")
print(f"Fast weights frozen: {model.freeze_fast_weights}")


# ============================================================
# Phase 1: Static Base Training
# ============================================================
print("\n2. Phase 1: Static Base Training")
print("-" * 60)
print("Fast weights FROZEN - training slow weights only")

# Training setup
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Simulate training data
train_data = torch.randn(32, 50, 64)
train_target = torch.randn(32, 50, 128)

# Training loop
model.train()  # This automatically freezes fast weights

for epoch in range(10):
    optimizer.zero_grad()
    
    # Forward pass
    output, _ = model(train_data)
    
    # Compute loss
    loss = criterion(output, train_target)
    
    # Backward pass
    loss.backward()
    
    # Update slow weights only (fast weights frozen)
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1}/10: Loss = {loss.item():.4f}")

print("✓ Static base training complete!")


# ============================================================
# Phase 2: Adaptation/Inference
# ============================================================
print("\n3. Phase 2: Adaptation/Inference")
print("-" * 60)
print("Fast weights ACTIVE - model adapts online")

model.eval()  # This unfreezes fast weights

# Test on new sequence
test_data = torch.randn(4, 50, 64)

with torch.no_grad():
    # Initialize state
    state = model.init_state(batch_size=4)
    
    # Process sequence (fast weights adapt!)
    output, final_state = model(test_data, state=state)
    
    print(f"  Input shape:  {test_data.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Check fast weights adaptation
    if hasattr(final_state, 'U'):
        u_norm = final_state.U.norm().item()
        print(f"  Fast weights norm: {u_norm:.4f} (adapted during inference!)")


# ============================================================
# Manual Mode Switching
# ============================================================
print("\n4. Manual Mode Switching")
print("-" * 60)

model = DREAM(input_dim=64, hidden_dim=128, rank=8)

# Default: fast weights active
print(f"Default mode: freeze_fast_weights = {model.freeze_fast_weights}")

# Manually freeze for training
model.set_fast_weights_mode(freeze=True)
print(f"After set_fast_weights_mode(True): {model.freeze_fast_weights}")

# Manually unfreeze for inference
model.set_fast_weights_mode(freeze=False)
print(f"After set_fast_weights_mode(False): {model.freeze_fast_weights}")


# ============================================================
# Multi-Epoch Training with State Persistence
# ============================================================
print("\n5. Multi-Epoch Training with State Persistence")
print("-" * 60)

model = DREAMCell(config=DREAMConfig(input_dim=64, hidden_dim=128, rank=8))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Initialize state ONCE (persists across epochs)
state = model.init_state(batch_size=4)

print("Training with persistent state across epochs...")

for epoch in range(5):
    optimizer.zero_grad()
    
    # Process sequence
    output, state = model.forward_sequence(
        torch.randn(4, 50, 64),
        state=state,
        return_all=True
    )
    
    # Compute loss
    loss = criterion(output, torch.randn(4, 50, 128))
    
    # Backward
    loss.backward()
    optimizer.step()
    
    # Detach state (truncated BPTT)
    state = state.detach()
    
    print(f"  Epoch {epoch+1}/5: Loss = {loss.item():.4f}")


# ============================================================
# Key Takeaway
# ============================================================
print("\n" + "=" * 60)
print("Key Takeaway:")
print("=" * 60)
print("""
Two-Phase Training:

Phase 1: Static Base Training (model.train())
  - Fast weights FROZEN
  - Train slow weights (C, W, B, eta) via backprop
  - Learns general patterns
  - Stable convergence

Phase 2: Adaptation/Inference (model.eval())
  - Fast weights ACTIVE
  - Online adaptation via STDP
  - Captures sequence-specific patterns
  - No gradients needed

Best Practices:
  • Use model.train() for training (auto-freezes fast weights)
  • Use model.eval() for inference (auto-unfreezes)
  • For multi-epoch training, preserve state between epochs
  • Detach state between segments (truncated BPTT)
""")
