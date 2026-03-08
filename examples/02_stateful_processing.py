"""
Example 2: Stateful Processing

Demonstrates memory retention across multiple sequences.
DREAM preserves state between sequences for context-aware processing.
"""

import torch
from dream import DREAM, DREAMCell, DREAMConfig

print("=" * 60)
print("DREAM Example 2: Stateful Processing")
print("=" * 60)

# ============================================================
# Scenario: Processing multiple related sequences
# ============================================================
print("\nScenario: Conversation with memory retention")
print("-" * 60)

model = DREAM(
    input_dim=64,
    hidden_dim=128,
    rank=8,
    freeze_fast_weights=False,  # Active fast weights for adaptation
)

# Initialize state ONCE
state = model.init_state(batch_size=1)

print("State initialized once, preserved across all sequences\n")

# Simulate conversation turns
sequences = [
    torch.randn(1, 20, 64),  # Turn 1
    torch.randn(1, 25, 64),  # Turn 2
    torch.randn(1, 22, 64),  # Turn 3
]

for i, seq in enumerate(sequences, 1):
    output, state = model(seq, state=state)
    print(f"Turn {i}: Input {seq.shape} → Output {output.shape}")
    # State (including fast weights U) is PRESERVED!


# ============================================================
# Comparison: Stateful vs Stateless
# ============================================================
print("\n" + "=" * 60)
print("Stateful vs Stateless Processing")
print("=" * 60)

# Stateless (reset state every time)
print("\nStateless (reset every sequence):")
model_stateless = DREAM(input_dim=64, hidden_dim=128, rank=8)

for i, seq in enumerate(sequences, 1):
    state = model_stateless.init_state(batch_size=1)  # Reset!
    output, state = model_stateless(seq, state=state)
    print(f"  Sequence {i}: No memory from previous")

# Stateful (preserve state)
print("\nStateful (preserve state):")
model_stateful = DREAM(input_dim=64, hidden_dim=128, rank=8)
state = model_stateful.init_state(batch_size=1)

for i, seq in enumerate(sequences, 1):
    output, state = model_stateful(seq, state=state)  # Preserved!
    print(f"  Sequence {i}: Remembers context from previous")


# ============================================================
# Use Case: Document Processing
# ============================================================
print("\n" + "=" * 60)
print("Use Case: Processing Long Document in Chunks")
print("=" * 60)

# Split long document into chunks
doc_length = 500
chunk_size = 100
chunks = torch.randn(1, doc_length, 64).split(chunk_size, dim=1)

print(f"Document: {doc_length} timesteps")
print(f"Chunks: {len(chunks)} × {chunk_size} timesteps\n")

# Process with state preservation
state = model.init_state(batch_size=1)
all_outputs = []

for i, chunk in enumerate(chunks, 1):
    output, state = model(chunk, state=state)
    all_outputs.append(output)
    print(f"Chunk {i}: Processed with context from previous chunks")

# Combine outputs
final_output = torch.cat(all_outputs, dim=1)
print(f"\nFinal output shape: {final_output.shape}")


# ============================================================
# Key Takeaway
# ============================================================
print("\n" + "=" * 60)
print("Key Takeaway:")
print("=" * 60)
print("""
Stateful Processing:
  ✓ Initialize state ONCE
  ✓ Pass state between sequences
  ✓ Fast weights accumulate experience
  ✓ Model adapts to context

Use Cases:
  • Multi-turn conversations
  • Long documents (chunked processing)
  • Streaming audio/video
  • Time series with episodes
""")
