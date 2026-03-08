"""
Example 5: Long Sequence Processing

Processing long sequences with truncated BPTT.
Handles sequences longer than memory capacity.
"""

import torch
import torch.nn as nn
from dream import DREAM, DREAMCell, DREAMConfig

print("=" * 60)
print("DREAM Example 5: Long Sequence Processing")
print("=" * 60)

# ============================================================
# Scenario: Processing 1000+ timestep sequences
# ============================================================
print("\nScenario: 30-second audio at 50fps = 1500 timesteps")
print("-" * 60)

# Can't process all at once (memory constraints)
long_sequence = torch.randn(2, 1500, 80)

print(f"Full sequence shape: {long_sequence.shape}")
print("Memory required: ~100MB+ for full backprop")


# ============================================================
# Solution: Truncated BPTT
# ============================================================
print("\n1. Truncated BPTT (Segmented Processing)")
print("-" * 60)

model = DREAMCell(config=DREAMConfig(input_dim=80, hidden_dim=256, rank=16))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Process in segments
segment_size = 100
seq_len = long_sequence.shape[1]

# Initialize state ONCE
state = model.init_state(batch_size=2)

print(f"Processing in segments of {segment_size} timesteps...")

total_loss = 0
n_segments = 0

for start in range(0, seq_len, segment_size):
    end = min(start + segment_size, seq_len)
    segment = long_sequence[:, start:end, :]
    
    optimizer.zero_grad()
    
    # Forward through segment
    output, state = model.forward_sequence(segment, state=state, return_all=True)
    
    # Compute loss
    loss = criterion(output, torch.randn_like(output))
    loss.backward()
    
    optimizer.step()
    
    # Detach state between segments (truncate BPTT)
    state = state.detach()
    
    total_loss += loss.item()
    n_segments += 1
    
    print(f"  Segment {n_segments}: Loss = {loss.item():.4f}")

print(f"\nAverage loss: {total_loss / n_segments:.4f}")


# ============================================================
# Variable Length Sequences
# ============================================================
print("\n2. Variable Length Sequences")
print("-" * 60)

# Batch with different lengths
sequences = [
    torch.randn(1, 500, 80),   # Short
    torch.randn(1, 1000, 80),  # Medium
    torch.randn(1, 1500, 80),  # Long
]

print("Processing sequences of different lengths:")

for i, seq in enumerate(sequences, 1):
    state = model.init_state(batch_size=1)
    
    # Process in chunks
    chunk_size = 100
    for start in range(0, seq.shape[1], chunk_size):
        end = min(start + chunk_size, seq.shape[1])
        chunk = seq[:, start:end, :]
        output, state = model.forward_sequence(chunk, state=state, return_all=True)
        state = state.detach()
    
    print(f"  Sequence {i}: {seq.shape[1]} timesteps → processed!")


# ============================================================
# Streaming Processing
# ============================================================
print("\n3. Streaming Processing (Real-time)")
print("-" * 60)

# Simulate real-time streaming
model.eval()
state = model.init_state(batch_size=1)

print("Simulating real-time audio stream...")

# Process chunk by chunk as they arrive
for chunk_id in range(10):
    # New chunk arrives
    chunk = torch.randn(1, 50, 80)
    
    # Process with preserved state
    output, state = model.forward_sequence(chunk, state=state, return_all=False)
    
    print(f"  Chunk {chunk_id + 1}: Processed (state preserved)")

print("✓ Streaming complete with continuous context!")


# ============================================================
# Memory Efficiency Comparison
# ============================================================
print("\n4. Memory Efficiency")
print("-" * 60)

# Full sequence (bad)
print("Full sequence backprop:")
try:
    # This would use lots of memory
    print("  ✗ Would require ~500MB+ for 1500 timesteps")
except:
    print("  ✗ OOM (Out of Memory)")

# Truncated BPTT (good)
print("\nTruncated BPTT:")
print("  ✓ Only ~10MB per segment")
print("  ✓ Constant memory usage")
print("  ✓ Can process arbitrary length")


# ============================================================
# Key Takeaway
# ============================================================
print("\n" + "=" * 60)
print("Key Takeaway:")
print("=" * 60)
print("""
Truncated BPTT Pattern:

```python
state = model.init_state(batch_size)

for start in range(0, seq_len, segment_size):
    segment = x[:, start:start+segment_size, :]
    
    output, state = model.forward_sequence(segment, state=state)
    loss = criterion(output, target)
    
    loss.backward()
    optimizer.step()
    
    state = state.detach()  # Truncate BPTT!
```

Benefits:
  ✓ Constant memory usage
  ✓ Process arbitrary length sequences
  ✓ Suitable for streaming
  ✓ Enables long-context modeling

Use Cases:
  • Long audio processing (30+ seconds)
  • Document processing (1000+ tokens)
  • Time series analysis
  • Real-time streaming
""")
