Quick Start
===========

Basic Usage
-----------

.. code-block:: python

   from dream import DREAM, DREAMConfig
   import torch
   
   # Create model
   model = DREAM(input_dim=39, hidden_dim=256, rank=16)
   
   # Process sequence
   x = torch.randn(4, 100, 39)
   output, state = model(x)
   
   print(f"Output shape: {output.shape}")

Training Example
----------------

.. code-block:: python

   from dream import DREAM
   import torch
   import torch.nn as nn
   
   # Data
   x = torch.randn(4, 100, 39)
   y = torch.randn(4, 100, 10)  # targets
   
   # Model
   model = DREAM(input_dim=39, hidden_dim=256, rank=16)
   head = nn.Linear(256, 10)
   optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)
   
   # Training
   for epoch in range(10):
       optimizer.zero_grad()
       output, state = model(x)
       loss = nn.functional.mse_loss(head(output), y)
       loss.backward()
       optimizer.step()
       print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

State Management
----------------

State persists between calls for long-term memory:

.. code-block:: python

   # Initialize state
   state = model.init_state(batch_size=4)
   
   # Process with memory
   for segment in segments:
       output, state = model(segment, state=state)
       # State preserved!

Configuration
-------------

.. code-block:: python

   from dream import DREAMConfig
   
   config = DREAMConfig(
       input_dim=39,        # Input features
       hidden_dim=256,      # Hidden size
       rank=16,             # Fast weights rank
       ltc_enabled=True,    # Enable LTC
       forgetting_rate=0.01,
   )
   
   model = DREAM(**config.__dict__)

Next Steps
----------

* :doc:`examples` — More usage examples
* :doc:`api` — API reference
* :doc:`architecture` — How DREAM works
