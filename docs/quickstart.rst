Quick Start
===========

This guide will show you the basics of using DREAM in 5 minutes.

Basic Usage
-----------

Creating a Model
~~~~~~~~~~~~~~~~

.. code-block:: python

   from dream import DREAM, DREAMConfig
   import torch
   
   # Configuration
   config = DREAMConfig(
       input_dim=39,      # Input dimension (e.g., MFCC 39D)
       hidden_dim=256,    # Hidden state dimension
       rank=16,           # Fast weights rank
       ltc_enabled=True,  # Enable Liquid Time-Constants
   )
   
   # Model
   model = DREAM(
       input_dim=config.input_dim,
       hidden_dim=config.hidden_dim,
       rank=config.rank,
       ltc_enabled=config.ltc_enabled,
   )
   
   print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

Processing Sequences
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Input data
   batch_size = 4
   seq_len = 100
   input_dim = 39
   
   x = torch.randn(batch_size, seq_len, input_dim)
   
   # Forward pass
   output, state = model(x)
   
   print(f"Output shape: {output.shape}")  # (batch, seq_len, hidden_dim)
   print(f"Hidden state shape: {state.h.shape}")  # (batch, hidden_dim)

Training a Model
----------------

Simple Training Loop
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dream import DREAM
   import torch
   
   # Data
   x = torch.randn(4, 100, 39)
   y = torch.randn(4, 100, 256)  # Target hidden states
   
   # Model
   model = DREAM(input_dim=39, hidden_dim=256, rank=16)
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
   
   # Training
   model.train()
   for epoch in range(10):
       optimizer.zero_grad()
       
       output, state = model(x)
       loss = torch.nn.functional.mse_loss(output, y)
       
       loss.backward()
       optimizer.step()
       
       print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

State Management
~~~~~~~~~~~~~~~~

Important: State is preserved between calls for long-term memory:

.. code-block:: python

   # Initialize state
   state = model.init_state(batch_size=4)
   
   # Sequential processing with memory preservation
   for segment in segments:
       output, state = model(segment, state=state)
       # State is preserved between segments!
   
   # Reset memory (if needed)
   state = model.init_state(batch_size=4)

Advanced Features
-----------------

DREAMStack (Multi-Layer Model)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dream import DREAMStack
   
   stack = DREAMStack(
       input_dim=39,
       hidden_dims=[128, 256, 128],  # 3 layers
       rank=16,
       dropout=0.1,
   )
   
   x = torch.randn(4, 100, 39)
   output, states = stack(x)
   
   print(f"Output shape: {output.shape}")
   print(f"Number of layer states: {len(states)}")

Masked Processing (Padding)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Variable length data
   x = torch.randn(4, 100, 39)
   lengths = torch.tensor([80, 95, 100, 65])  # Actual lengths
   
   # Processing with mask
   output, state = model.forward_with_mask(x, lengths)
   
   # Output for padding will be zero
   print(f"Output shape: {output.shape}")

Model Configuration
-------------------

DREAMConfig Parameters
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dream import DREAMConfig
   
   config = DREAMConfig(
       # Dimensions
       input_dim=39,        # Input features
       hidden_dim=256,      # Hidden state
       rank=16,             # Fast weights rank
       
       # Time
       time_step=0.1,       # Integration step
       
       # Plasticity
       forgetting_rate=0.01,    # λ (forgetting)
       base_plasticity=2.0,     # η (learning)
       
       # Surprise
       base_threshold=0.5,      # τ₀ (threshold)
       entropy_influence=0.3,   # α (entropy)
       surprise_temperature=0.2, # γ (temperature)
       
       # LTC
       ltc_enabled=True,        # Enable LTC
       ltc_tau_sys=10.0,        # Base τ
       ltc_surprise_scale=10.0, # Modulation
   )

Best Practices
--------------

For ASR (Speech Recognition)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   config = DREAMConfig(
       input_dim=39,        # MFCC 39D
       hidden_dim=512,      # Larger capacity
       rank=16,
       forgetting_rate=0.01,
       base_plasticity=2.0,
       ltc_enabled=True,
   )

For Time Series
~~~~~~~~~~~~~~~

.. code-block:: python

   config = DREAMConfig(
       input_dim=features_dim,
       hidden_dim=128,
       rank=8,
       ltc_enabled=True,
       ltc_tau_sys=5.0,  # Faster response
   )

For Online Learning
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   config = DREAMConfig(
       input_dim=input_dim,
       hidden_dim=256,
       rank=16,
       forgetting_rate=0.0,   # No forgetting
       base_plasticity=3.0,   # High plasticity
       ltc_enabled=True,
   )

Next Steps
----------

* :doc:`api` — Full API documentation
* :doc:`architecture` — Architecture details
* :doc:`examples` — Usage examples
